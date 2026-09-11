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

import pandas as pd

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
#
# 🔴 2026-09-10（§8-R11）補上 `mode`：原本沒有它，等 live 模式開始寫紀錄之後，
# `find_previous()` 會把一筆 **replay 展示紀錄**當成 live 執行的「上一期」拿去 diff，
# 算出的「新增/剔除幾檔」變成拿正式運作結果比一個歷史展示案例。趁還沒發作先修。
_IDENTITY_KEYS = ("mode", "market", "group", "ratio", "allocation", "k_mode")


def _window_sort_key(entry: dict) -> tuple | None:
    """從一筆紀錄取出「這一窗在時間軸上的位置」，用來判斷 diff 是不是逆序。

    正式模式沒有 window_no，用 is_end 當位置；replay 用 (is_end, window_no)。
    取不到就回 None（呼叫端視為無法判斷，不硬猜）。
    """
    wi = entry.get("window_info") or {}
    is_end = wi.get("is_end")
    if not is_end:
        return None
    return (str(is_end), int(wi.get("window_no") or 0))


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


def diff_holdings(current_members: list[str], previous_entry: dict | None,
                  current_window_info: dict | None = None) -> dict:
    """算出跟上一筆紀錄相比，策略清單新增/剔除了哪些。

    `previous_entry` 是舊格式（沒有 `members` 欄位）時視同沒有上一筆——
    誠實反映「補存這個欄位之前的紀錄沒辦法拿來比」，不假裝算得出來。

    🔴 2026-09-10（§8-R6）新增 `is_chronological`：`find_previous()` 是依
    **執行時間**（`recorded_at`）排序，不是依窗次時間。使用者先跑 window 6 再跑
    window 5，這裡會算出一份「新增 X／剔除 Y」，但那其實是**拿過去的窗比未來的窗**。
    memo 的 `change_note` 會把它敘述成「本期異動」，在治理文件裡是誤導。
    ⇒ 加一個旗標讓呼叫端能標示「這是回溯比較，不是本期異動」。
    """
    if previous_entry is None or "members" not in previous_entry:
        return {"has_previous": False}
    prev = set(previous_entry["members"])
    curr = set(current_members)

    prev_key = _window_sort_key(previous_entry)
    curr_key = _window_sort_key({"window_info": current_window_info or {}})
    if prev_key is None or curr_key is None:
        chrono = None          # 資訊不足，不硬猜
    else:
        chrono = curr_key >= prev_key

    return {
        "has_previous": True,
        "previous_recorded_at": previous_entry["recorded_at"],
        "previous_window_info": previous_entry.get("window_info"),
        "added": sorted(curr - prev),
        "removed": sorted(prev - curr),
        "n_added": len(curr - prev),
        "n_removed": len(prev - curr),
        "n_unchanged": len(curr & prev),
        "is_chronological": chrono,
        "direction_note": (
            "" if chrono is not False else
            "⚠️ 上一筆紀錄的窗次晚於本次——這是**回溯比較**，不是本期異動。"
            "（find_previous 依執行時間排序，不是窗次時間，見 §8-R6）"),
    }


#: 🔴 2026-09-11（§9.7 I-8）：`app/cluster_kb.py`（S2）與 `app/explain.py`（S3）
#: 新讀的群知識庫檔案，之前 `_input_versions()` 完全沒記——違反剛補上的 R4
#: 可重現性（同一份 RunConfig，若這些檔案換版，解釋 agent 的輸出會跟著變，
#: 但稽核紀錄看不出來）。⚠️ **只列實際會被讀到的檔案**——`cluster_quarterly_returns`
#: 雖然存在於 `_frozen/stage3/`，但 `cluster_kb.build_footprint()` 沒有讀它
#: （§9.2 資源盤點仍列為「❌ 沒用」），不放進來，避免稽核紀錄宣稱讀了其實沒讀的檔。
#: 同理 `weighting_decomposition.csv`（H7 基準數字的來源）也不在這裡——那組數字
#: 是靜態寫進 `memo.py`/`explain.py` 的已驗證事實，不是這次執行時讀檔算出來的。
_CLUSTER_KB_FILES = (
    "cluster_identity", "cluster_profile_quant", "cluster_annual_returns",
    "co_fail_regimes", "cluster_story",
    "cluster_corr_matrix_TW_normal", "cluster_corr_matrix_US_normal", "cluster_corr_matrix_XM_normal",
    "cluster_corr_matrix_TW_crisis", "cluster_corr_matrix_US_crisis", "cluster_corr_matrix_XM_crisis",
)


def _input_versions() -> dict:
    """🔴 2026-09-10（§8-R4）：記下這次讀了哪一版輸入資料。

    舊版完全沒記，理由是「這裡記的是執行歷史不是凍結產物」——那個理由在只讀
    凍結檔時成立，但**快時鐘讀的是每天在變的資料庫**：同一份 RunConfig 隔一週跑
    結果會不同，而紀錄無法區分。這違反 `pitfalls.md` §8 可重現性與 DD-08 精神。

    只記 size+mtime 而不是完整 sha256——凍結檔最大 35MB，每次執行都做雜湊會拖慢
    互動流程；size+mtime 足以偵測「檔案換過了」，真要追版本再去查 DD-08 manifest。
    """
    out = {}
    files = [
        ("candidate_index", paths.STAGE0 / "candidate_index.parquet"),
        ("returns_monthly", paths.STAGE1 / "returns_monthly.parquet"),
        ("cluster_assign", paths.STAGE3 / "cluster_assign.parquet"),
        ("walkforward_detail", paths.ROOT / "_analysis_outputs_robustness"
                               / "walkforward_matrix_detail.csv"),
    ]
    files += [(name, paths.STAGE3 / f"{name}.parquet") for name in _CLUSTER_KB_FILES]
    for name, p in files:
        if p.exists():
            st = p.stat()
            out[name] = {"bytes": st.st_size,
                        "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}
        else:
            out[name] = None
    return out


def record(holdings: Holdings, risk: RiskReport, *,
          calibration: CalibrationResult | None = None,
          override_reason: str | None = None,
          stock_holdings: "pd.DataFrame | None" = None,
          stock_as_of: str | None = None,
          stock_concentration=None,
          explanation: dict | None = None) -> dict:
    """寫一筆稽核紀錄，回傳寫進去的內容（方便 cli.py 印出來確認）。

    `override_reason` 非 None 代表「風控違規、但人工覆核放行」——這種情況
    一定要記，不能只記「執行完成」。

    🔴 `stock_holdings`／`stock_as_of`（2026-09-10，§8-R4）：**股票層級持股**。
    對一個宣稱做 fund house 治理的系統，「那天實際持有什麼」是最該留痕的一筆，
    但舊版只記策略 uid，`resolve_strategy_holdings.py` 算出的實際股票從來沒進紀錄
    ⇒ IPS §6 合規聲明的第一題根本回答不了。有解析就傳進來，沒有就留 None
    （誠實記成「本次未解析」，不假裝有）。

    🔴 `explanation`（2026-09-11，§9.7 S4，為 §9.8 前瞻驗證準備）：`app/explain.py`
    的完整輸出（`{"prompt", "facts", "explanation", "dry_run"}`）。§9.8 的執行紀律
    明講「三次前瞻驗證的解釋要先記進 audit_log.jsonl 並 git commit，才能算績效」
    ——這是造出「先預測、後驗證」可稽核軌跡的關鍵一步，不能只顯示在 UI 上就算數。
    dry-run 產生的內容也記（`dry_run: True` 欄位會如實反映），只是不能拿來當
    §9.8 正式驗證用。
    """
    if risk.violations and override_reason is None:
        raise RuntimeError(
            "有風控違規但沒有 override_reason，不可以記成正常執行——"
            "呼叫端應該先攔下、要求人工確認（C4），不是直接呼叫這裡")

    previous_entry = find_previous(holdings.config)
    diff = diff_holdings(holdings.members, previous_entry, holdings.window_info)

    if stock_holdings is not None and len(stock_holdings):
        col = "stock_id" if "stock_id" in stock_holdings.columns else stock_holdings.columns[0]
        uniq = sorted(stock_holdings[col].astype(str).unique())
        stocks = {"resolved": True, "as_of": stock_as_of,
                  "n_rows": int(len(stock_holdings)), "n_unique_stocks": len(uniq),
                  "stocks": uniq}
        # §8-R5：股票層級集中度是**風控判決**，必須留痕（不只是資訊）
        if stock_concentration is not None:
            sc = stock_concentration
            stocks["concentration"] = {
                "max_stock_weight": sc.max_stock_weight,
                "max_stock_symbol": sc.max_stock_symbol,
                "max_stock_name": sc.names.get(sc.max_stock_symbol, ""),
                "cap": holdings.config.single_stock_cap,
                "n_violations": len(sc.violations),
                "violations": [dataclasses.asdict(v) for v in sc.violations],
                "top15": sc.top(15),
            }
    else:
        stocks = {"resolved": False, "as_of": None,
                  "note": "本次未解析股票層級持股（UI 的『解析持股』是選用步驟）"}

    entry = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "config": dataclasses.asdict(holdings.config),
        "window_info": holdings.window_info,
        "tree_info": holdings.tree_info,
        "validation": holdings.validation,
        "input_versions": _input_versions(),       # §8-R4
        "stock_holdings": stocks,                  # §8-R4
        "n_members": holdings.n_members,
        "members": holdings.members,
        "diff_from_previous": diff,
        "performance": holdings.performance,
        "has_oos": holdings.has_oos,
        "reference_oos": holdings.reference_oos,
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
        "explanation": explanation,   # §9.7 S4：app/explain.py 的完整輸出，§9.8 用
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        # `default=str`：§9.7 S4 新增的 `explanation["facts"]`（`app/explain.py`）
        # 有些欄位直接來自 parquet（numpy 純量型別），標準 json 模組不吃這些型別
        # ——跟 `explain.build_prompt()`／`ui.py` 顯示時同一個處理方式，不是新問題。
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    return entry
