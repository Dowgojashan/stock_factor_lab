# -*- coding: utf-8 -*-
"""§8待辦item11 步驟1（2026-09-22）：把設計文件 §8「受限工具集」的四個固定
工具（`get_attribution`／`get_historical_distribution`／`find_similar_quarters`／
`get_action_reference`）整理成獨立、可單獨測試的函式。

🔴 範圍界線（跟使用者確認過）：**這次只做步驟1**——把工具包成乾淨的函式，
不改變現有 LLM 呼叫模式（`simulate.py`／`facts_lean.py` 依然是「程式先把
facts 算好塞進 prompt，一次呼叫」，不是讓 agent 自己決定要不要呼叫）。真正
接上 OpenAI tool-calling 迴圈（`memo._call_llm()` 支援 `tools=` 參數、agent
自己決定查不查、可能要來回好幾輪）是更大的架構改動，會動到全部 agent 角色
共用的呼叫核心，這次刻意不做，留給真的有需求（例如發現 prompt 太肥）時
再評估。

四個工具現況：
  - `get_action_reference`：已存在於 `actions.py`（含 §7.7 硬性規定的前視
    防護：只回傳 oos_end < as_of 的窗次），這裡直接 re-export，不重寫一份。
  - `get_attribution`：包一層 `diagnose.baseline_chain_decomposition()`
    （§8待辦item9剛做的），從 `outcome` dict 直接組出M4/M7/M1-R三段貢獻。
  - `get_historical_distribution`：統一介面，內部依 mechanism 分派到
    `diagnose.py` 原本散落各處的 `_m3_history()`/`_m0_history()`/
    `_m7_history()`/`_m8_history()`，回傳該機制實際會用到的分位數
    （M3 用 p10+p90、M0/M7 只用 p10、M8 只用 p90，跟各自
    `diagnose_mX()` 函式的既有判準完全一致，不是另外發明新門檻）。
  - `find_similar_quarters`：**全新實作**（設計文件原本沒有具體演算法）。
    定義「相似」＝ windows1-3（36季真實歷史）裡 `excess_vs_ball` 跟查詢值
    最接近的N季——這是本專案目前唯一有完整歷史序列、且跟M0/M4診斷同一個
    流量變數的欄位，用它當相似度基準最站得住腳，不是隨便挑一個指標。
"""
from __future__ import annotations

import pandas as pd

from app import diagnose
from app.actions import get_action_reference  # noqa: F401  re-export，四個工具統一從這裡 import

_SIMILAR_QUARTERS_PATH = diagnose._M0_HISTORY_PATH  # windows1-3修正版，跟M0/M7同一份資料源


def get_attribution(outcome: dict) -> dict | None:
    """§7.4 基準鏈拆解（M4/M7/M1-R三段貢獻），包一層
    `diagnose.baseline_chain_decomposition()`，直接吃 `monitor.outcome_layer()`
    的輸出，呼叫端不用自己拆欄位。回顧區專用，不可驅動動作（§7.0）。

    🔴🔴 code review 抓到的真bug（2026-09-22，跟 `simulate.py`／`facts_lean.py`
    同一個根因）：`outcome["ball_benchmark_return"]` 退回等權大盤替身時數值
    會跟 equal_weight_benchmark_return 完全相等（不是None）——只有
    `ball_return_is_real=True` 才能當真值傳下去，否則會讓 M7/M4 拆解算出
    誤導性的假數字（m7_gap恆為0）而非誠實的「無法拆解」。"""
    real_ball = outcome.get("ball_benchmark_return") if outcome.get("ball_return_is_real") else None
    return diagnose.baseline_chain_decomposition(
        a_hrp_return=outcome["portfolio_realized_return"],
        ball_return=real_ball,
        equal_weight_return=outcome.get("equal_weight_benchmark_return"),
        cap_weight_return=outcome.get("cap_weight_benchmark_return"))


_HISTORY_FUNCS = {
    "M3": (diagnose._m3_history, ("p10", "p90")),
    "M0": (diagnose._m0_history, ("p10",)),
    "M7": (diagnose._m7_history, ("p10",)),
    "M8": (diagnose._m8_history, ("p90",)),
}


def get_historical_distribution(mechanism: str) -> dict:
    """給定機制代號（M3/M0/M7/M8），回傳該機制**實際會用到**的歷史分位數
    （跟各自 `diagnose_mX()` 判準完全一致的門檻，不是另外算一組）。

    🔴 目前的歷史資料源固定是 windows1-3（2015-2023，皆早於window4），
    沒有動態依 `as_of` 再篩選——設計文件原簽名帶 `as_of` 參數是為了未來
    有更多窗次時可以動態排除當期之後的資料，現階段資料源本身就已經是
    「window4之前」，這個限制留在這裡誠實記錄，不是漏做。
    """
    if mechanism not in _HISTORY_FUNCS:
        raise ValueError(f"不支援的機制代號 {mechanism!r}，只有 {sorted(_HISTORY_FUNCS)}")
    hist_fn, percentiles = _HISTORY_FUNCS[mechanism]
    hist = hist_fn()
    out = {"mechanism": mechanism, "n": int(len(hist))}
    for p in percentiles:
        q = float(p[1:]) / 100.0
        out[p] = float(hist.quantile(q))
    return out


def find_similar_quarters(current_excess_vs_ball: float, n: int = 3) -> list[dict]:
    """找歷史上（windows1-3，36季真實資料）`excess_vs_ball` 最接近查詢值的
    N 季，供 agent 找「以前遇過類似情況嗎」的精確前例。相似度定義＝
    `excess_vs_ball` 差的絕對值最小——這是本專案目前唯一有完整逐季歷史
    序列、且跟M0/M4診斷同一個流量變數的欄位，不是隨便挑的相似度指標。

    回傳依相似度排序（最相似在前），每筆含 window_no/as_of/end/
    excess_vs_ball/abs_diff，讓 agent 自己判斷這些前例夠不夠像、要不要引用。
    """
    df = pd.read_csv(_SIMILAR_QUARTERS_PATH)
    df = df.dropna(subset=["excess_vs_ball"]).copy()
    df["abs_diff"] = (df["excess_vs_ball"] - current_excess_vs_ball).abs()
    top = df.sort_values("abs_diff").head(n)
    return [
        {"window_no": int(r.window_no), "as_of": r.as_of, "end": r.end,
         "excess_vs_ball": round(float(r.excess_vs_ball), 4),
         "abs_diff": round(float(r.abs_diff), 4)}
        for r in top.itertuples()
    ]
