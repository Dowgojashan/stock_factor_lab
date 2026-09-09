# -*- coding: utf-8 -*-
"""F2 · 稽核紀錄（應用層開發追蹤.md §3-F，Phase A）

沿用 `research.freeze` manifest 的精神（每次動作都留軌跡），格式簡化成
一行一筆 JSON（jsonl），不用完整的 DD-08 凍結鏈機制——這裡記的是「執行歷史」，
不是「凍結產物」，兩者性質不同，不用共用同一套機制。

🔴 **2026-09-09 review（AL-04）**：風控違規＋人工覆核結果是必記項目，不能只留
正常執行的軌跡——這是整個系統裡最需要留痕的一種事件。
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

from research import paths
from .calibration import CalibrationResult
from .engine import Holdings
from .risk import RiskReport

RUNS_DIR = paths.ROOT / "code" / "app" / "_runs"
LOG_PATH = RUNS_DIR / "audit_log.jsonl"


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

    entry = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "config": dataclasses.asdict(holdings.config),
        "window_info": holdings.window_info,
        "n_members": holdings.n_members,
        "performance": holdings.performance,
        "risk": {
            "portfolio_mdd": risk.portfolio_mdd,
            "portfolio_ann_vol": risk.portfolio_ann_vol,
            "max_single_weight": risk.max_single_weight,
            "max_cluster_share": risk.max_cluster_share,
            "violations": [dataclasses.asdict(v) for v in risk.violations],
        },
        "calibration": dataclasses.asdict(calibration) if calibration else None,
        "override_reason": override_reason,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
