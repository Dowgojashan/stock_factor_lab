# -*- coding: utf-8 -*-
"""F2 · 稽核紀錄（應用層開發追蹤.md §3-F，Phase A）

沿用 `research.freeze` manifest 的精神（每次動作都留軌跡），格式簡化成
一行一筆 JSON（jsonl），不用完整的 DD-08 凍結鏈機制——這裡記的是「執行歷史」，
不是「凍結產物」，兩者性質不同，不用共用同一套機制。

🔴 **2026-09-09 review（AL-04）**：風控違規＋人工覆核結果是必記項目，不能只留
正常執行的軌跡——這是整個系統裡最需要留痕的一種事件。

🔴 **2026-09-10（應用層 §6 落差①）**：原本只存 `n_members`（數字），沒存
`holdings.members`（實際策略清單），導致系統完全無狀態——沒辦法回答「跟上次
相比，這次換了什麼」。這正是老師 9-8 §2.3「情境比對」要的基礎，也是清華永續
基金投審會六段式匯報的第②段（哪些判斷符合預期、哪裡有落差）跟 Agentic
Architecture 論文 board memo 固定段落「changes since last review」對應的資料。
現在補存 `members`，並在寫入前用 `find_previous()` 找同一個 RunConfig 身份
（market/group/ratio/allocation/k_mode）的上一筆紀錄，把 diff 結果一併存進
這筆紀錄——診斷資訊在寫入當下就固定下來，之後讀歷史不用重新計算。
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

from research import paths
from .calibration import CalibrationResult
from .config import RunConfig
from .engine import Holdings
from .risk import RiskReport

RUNS_DIR = paths.ROOT / "code" / "app" / "_runs"
LOG_PATH = RUNS_DIR / "audit_log.jsonl"

# 判定「同一條 RunConfig 系列」的身份欄位——不含 window_no/scheme，因為使用者
# 換窗次通常代表「同一個政策、下一期」，是我們要比較的對象；scheme 不同代表
# 換了窗口方案本身，也視為同系列（H-26 已證實不同方案的結論一致，比較有意義）。
_IDENTITY_KEYS = ("market", "group", "ratio", "allocation", "k_mode")


def _read_all_entries() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    entries = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def find_previous(config: RunConfig) -> dict | None:
    """找同一個 RunConfig 身份最近一筆歷史紀錄（依 recorded_at 排序取最新）。

    找不到（第一次跑這個身份）回傳 None——呼叫端要處理這個情況，不能假設
    一定有上一筆。
    """
    matches = [e for e in _read_all_entries()
              if all(e["config"].get(k) == getattr(config, k) for k in _IDENTITY_KEYS)]
    if not matches:
        return None
    return max(matches, key=lambda e: e["recorded_at"])


def diff_holdings(current_members: list[str], previous_entry: dict | None) -> dict:
    """算出跟上一筆紀錄相比，策略清單新增/剔除了哪些。

    `previous_entry` 是舊格式（沒有 `members` 欄位）時視同沒有上一筆——
    誠實反映「補存這個欄位之前的紀錄沒辦法拿來比」，不假裝算得出來。
    """
    if previous_entry is None or "members" not in previous_entry:
        return {"has_previous": False}
    prev = set(previous_entry["members"])
    curr = set(current_members)
    return {
        "has_previous": True,
        "previous_recorded_at": previous_entry["recorded_at"],
        "previous_window_info": previous_entry.get("window_info"),
        "added": sorted(curr - prev),
        "removed": sorted(prev - curr),
        "n_added": len(curr - prev),
        "n_removed": len(prev - curr),
        "n_unchanged": len(curr & prev),
    }


def record(holdings: Holdings, risk: RiskReport, *,
          calibration: CalibrationResult | None = None,
          override_reason: str | None = None) -> dict:
    """寫一筆稽核紀錄，回傳寫進去的內容（方便 cli.py 印出來確認）。

    `override_reason` 非 None 代表「風控違規、但人工覆核放行」——這種情況
    一定要記，不能只記「執行完成」。
    """
    if risk.violations and override_reason is None:
        raise RuntimeError(
            "有風控違規但沒有 override_reason，不可以記成正常執行——"
            "呼叫端應該先攔下、要求人工確認（C4），不是直接呼叫這裡")

    previous_entry = find_previous(holdings.config)
    diff = diff_holdings(holdings.members, previous_entry)

    entry = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "config": dataclasses.asdict(holdings.config),
        "window_info": holdings.window_info,
        "n_members": holdings.n_members,
        "members": holdings.members,
        "diff_from_previous": diff,
        "performance": holdings.performance,
        "risk": {
            "portfolio_mdd": risk.portfolio_mdd,
            "portfolio_ann_vol": risk.portfolio_ann_vol,
            "max_single_weight": risk.max_single_weight,
            "max_cluster_share": risk.max_cluster_share,
            "market_share": risk.market_share,
            "factor_exposure_f1": risk.factor_exposure_f1,
            "regime_avg_ret": risk.regime_avg_ret,
            "violations": [dataclasses.asdict(v) for v in risk.violations],
        },
        "calibration": dataclasses.asdict(calibration) if calibration else None,
        "override_reason": override_reason,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
