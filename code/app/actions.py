# -*- coding: utf-8 -*-
"""§3 動作空間（實戰監控 Agent 系統）。目前只實作 §7.7 要求的 A4 洩題防線：
`get_action_reference()`——動作參照**只能看到 `oos_end < as_of` 的窗次**，
物理上在資料載入器過濾掉，不靠 prompt 約束（設計文件 §7.7）。

🔴 window 4（scheme E，oos 2024-01~2025-12）的 `oos_end`＝實驗最後一天，在
整個實驗期間（2024-Q1~2025-Q4）任何 `as_of` 都不會小於它 ⇒ 用 `oos_end < as_of`
這個一般化的時序過濾規則，window 4 自然、永遠被排除，不需要另外寫「排除
window 4」的特判——這樣同一套邏輯往後也適用於任何新窗次，不必每次改規則。

W1~W4（市值傾斜等新維度）尚未補進矩陣（§7.6，須先在 window 1-3 前置驗證），
本檔目前只涵蓋矩陣既有的 `ratio`／`allocation` 維度。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_MATRIX_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "_analysis_outputs_robustness" / "walkforward_matrix_detail.csv"
)


def _oos_end_month_end(oos_end_col: pd.Series) -> pd.Series:
    """矩陣的 `oos_end` 是「YYYY-MM」月份格式，`pd.to_datetime` 會解析成
    **該月第一天**，不是月底。若拿它直接跟day-精度的 `as_of` 比較，同一個月
    裡會出現「窗次的 oos_end（月初）< as_of（月底）」的假象，實際上該窗
    OOS 期間要到月底才真正結束——這裡統一轉成月底日期再比較，
    否則會在月份邊界洩題（已用 as_of=2025-12-31 對 window4 實測抓到過一次）。
    """
    ts = pd.PeriodIndex(oos_end_col, freq="M").to_timestamp(how="end")
    return pd.Series(ts, index=oos_end_col.index)


def get_action_reference(
    ratio: str,
    allocation: str,
    as_of: str,
    *,
    tree_key: str = "TW",
    scheme: str = "E",
    group: str = "A_hrp",
    k_mode: str = "silhouette_is",
    matrix_path: Path = DEFAULT_MATRIX_PATH,
) -> list[dict]:
    """回傳某個動作（`ratio`/`allocation` 組合）在**嚴格早於 `as_of`** 的窗次
    裡的歷史表現，供 agent 評估「這個動作歷史上表現如何」時參照。

    🔴 物理過濾：`oos_end < as_of`（字串可比較的 YYYY-MM 或 YYYY-MM-DD 皆可，
    先轉 `pd.Timestamp`），不是靠 prompt 告訴 agent「不要看某一窗」——
    矩陣裡本來就沒有該窗的列，agent 拿到的 list 物理上不含它。
    """
    df = pd.read_csv(matrix_path)
    df = df[(df.tree_key == tree_key) & (df.scheme == scheme)
            & (df.group == group) & (df.k_mode == k_mode)
            & (df.ratio == ratio) & (df.allocation == allocation)]

    as_of_ts = pd.Timestamp(as_of)
    df = df[_oos_end_month_end(df["oos_end"]).lt(as_of_ts)]

    cols = ["window_no", "is_end", "oos_start", "oos_end",
            "oos_cagr", "oos_sharpe", "oos_mdd"]
    return df[cols].sort_values("window_no").to_dict("records")


def list_available_actions(
    as_of: str,
    *,
    tree_key: str = "TW",
    scheme: str = "E",
    group: str = "A_hrp",
    k_mode: str = "silhouette_is",
    matrix_path: Path = DEFAULT_MATRIX_PATH,
) -> list[dict]:
    """列出 `as_of` 當下每個 (ratio, allocation) 動作**可參照的窗次數**，
    方便 agent／人核對「這次能看到幾窗歷史」，避免誤以為看得到 window 4。
    """
    df = pd.read_csv(matrix_path)
    df = df[(df.tree_key == tree_key) & (df.scheme == scheme)
            & (df.group == group) & (df.k_mode == k_mode)]
    as_of_ts = pd.Timestamp(as_of)
    df = df[_oos_end_month_end(df["oos_end"]).lt(as_of_ts)]

    out = []
    for (r, a), gg in df.groupby(["ratio", "allocation"], observed=True):
        out.append({
            "ratio": r, "allocation": a,
            "n_windows_visible": len(gg),
            "window_nos_visible": sorted(gg["window_no"].tolist()),
            "mean_oos_cagr": float(gg["oos_cagr"].mean()) if len(gg) else None,
        })
    return out
