# -*- coding: utf-8 -*-
"""§3 動作空間（實戰監控 Agent 系統）。目前只實作 §7.7 要求的 A4 洩題防線：
`get_action_reference()`——動作參照**只能看到 `oos_end < as_of` 的窗次**，
物理上在資料載入器過濾掉，不靠 prompt 約束（設計文件 §7.7）。

🔴 window 4（scheme E，oos 2024-01~2025-12）的 `oos_end`＝實驗最後一天，在
整個實驗期間（2024-Q1~2025-Q4）任何 `as_of` 都不會小於它 ⇒ 用 `oos_end < as_of`
這個一般化的時序過濾規則，window 4 自然、永遠被排除，不需要另外寫「排除
window 4」的特判——這樣同一套邏輯往後也適用於任何新窗次，不必每次改規則。

W1／W3（市值傾斜等其餘維度）驗證失敗未啟用（見 D29）；**W2c（條件式市值
傾斜）已於 2026-09-18 前置驗證通過**（開發追蹤 D50），`w2c_reference()`
提供這個動作**唯一的一次性驗證結果**（不是像 `get_action_reference()` 那樣
隨 `as_of` 變化的即時查詢——W2c 的驗證是固定的歷史校準/前瞻結果，見
D50），供決策 agent 引用時遵守 §7.6「不得自行計算預期效果」的規則。
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


def w2c_reference() -> dict:
    """W2c（條件式市值傾斜）的驗證結果（開發追蹤 D50，2026-09-18 凍結）。
    跟 `get_action_reference()`／`list_available_actions()` 不同：這不是隨
    `as_of` 變化的即時窗次查詢，是**一次性**的校準＋前瞻驗證結果，決策
    agent 引用時只能照抄這個 dict 的數字，不可自行外推到未驗證過的情境
    （§7.6）。`caveats` 欄位須完整帶給 agent，不可只挑正面數字。

    🔴🔴 2026-09-18（D52 抓到）：`caveats` 原本有一條「目前只到決策記錄
    層級，選這個動作不會真的改變後續季度的持股計算」——這是 D50 當時（執行
    層還沒做）的真實狀況，但 D51 做完執行層後這句話已經**過期且錯誤**，
    忘記回頭修正。正式重跑時 agent 在連續 5 個觸發季度全部選了保守的 A5、
    一次都沒選 W2c——很可能就是因為看到這句話，以為選 W2c 等於沒選。已拿掉
    這條過期 caveat，不是為了引導 agent 選 W2c，是修正一個真的過期的事實
    描述（選 W2c 現在真的會透過 `simulate.ActiveConfig` 改變下一季的持股）。"""
    return {
        "mechanism": "W2c：M1-D 觸發時啟動市值傾斜（α=0）＋ 7% 目標上限、"
                     "超過部分按比例重分配（cap-and-redistribute）",
        "calibration_period": "2024-Q1~2025-Q4（8季，用於校準傾斜強度與上限）",
        "calibration_net_improvement_pp": 23.48,
        "forward_validation_period": "2026-Q1~2026-Q2（真前瞻，未參與任何校準）",
        "forward_net_improvement_pp": 21.98,
        "net_of_real_transaction_cost": True,
        "caveats": [
            "校準期與前瞻期本質上是同一個規模集中事件的不同階段，不是兩個獨立事件樣本",
            "前瞻驗證只有 2 季，統計證據仍薄弱",
            "季度制重新平衡無法 100% 保證任何時刻都不超過 8% 硬上限——"
            "7% 目標上限下仍有 1/6 觸發季度季底微幅超過（8.68%）",
        ],
    }
