# -*- coding: utf-8 -*-
"""F（開發追蹤v2§5.3）：監控頻率先做——測「如果M1-D每月檢查一次，會不會比現行
每季檢查更早抓到2024-2025台股大型股集中度異常」。跟「調整頻率」是分開的兩件事
（`投組監控與調整_領域筆記.md`§8引用清大書「每天追蹤，不每天調整」）：這裡只
測偵測/監控本身的頻率，不涉及真的改變持股或呼叫任何agent，純資料計算、零LLM
成本。

方法：`triggers.evaluate_quarter()`本身跟`monitor.environment_layer()`都是
「給任意as_of日期就能算」的函式，不是寫死季底——`run_quarterly_series()`這個
函式名字裡有"quarterly"，但實際簽名只吃`quarter_ends: list[str]`，傳月底清單
一樣能跑，不用另外寫一套月頻邏輯，沿用完全相同的p75/p90凍結門檔、完全相同的
雙門檻遲滯規則。

測兩件事：
1. **正式8季期間（2024-2025）**：月頻vs季頻，第一次TRIGGERED差幾個月
2. **陰性對照期間（window2/window3，已驗證quarterly下零誤報）**：月頻會不會
   多抓出季頻沒有的假警報——沿用`_negative_control.get_window_members()`
   同一組真實登記日期/OOS範圍，不是另外編的

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._design_test_monitoring_frequency
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from database import Database  # noqa: E402

from app import monitor, triggers  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _negative_control import get_window_members  # noqa: E402

REGISTRATION_DATE = "2023-12-31"   # 跟simulate.py正式production同一個登記日
QUARTER_ENDS = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
                "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]


def month_ends(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="M")]


def week_ends(start: str, end: str) -> list[str]:
    """每週五（'W-FRI'）——`environment_layer`／`asof_row`本身是as-of查詢
    （取<=as_of的最後一個交易日），週五非交易日也不影響正確性。"""
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="W-FRI")]


def first_triggered(series_df: pd.DataFrame) -> str | None:
    hit = series_df[series_df["state"] == "TRIGGERED"]
    return hit.iloc[0]["as_of"] if len(hit) else None


def summarize(label: str, series_df: pd.DataFrame) -> None:
    n_triggered = (series_df["state"] == "TRIGGERED").sum()
    n_observing = (series_df["state"] == "OBSERVING").sum()
    first = first_triggered(series_df)
    print(f"  [{label}] {len(series_df)}個檢查點｜TRIGGERED={n_triggered}｜OBSERVING={n_observing}｜"
         f"首次TRIGGERED={first if first else '（無）'}")


def main():
    print(">> 連線資料庫，抓取市值寬表...")
    db = Database("TW")
    conn = db.create_connection()
    mcap_wide = monitor.fetch_mcap_wide(conn, start="2007-01-01")

    print("\n=== 測試1：正式8季期間（2024-2025），週頻vs月頻vs季頻，第一次TRIGGERED差多久 ===")
    cond_prod = triggers.register_m1d(mcap_wide, REGISTRATION_DATE)
    q_series = triggers.run_quarterly_series(mcap_wide, cond_prod, QUARTER_ENDS)
    m_series = triggers.run_quarterly_series(mcap_wide, cond_prod, month_ends("2024-01-31", "2025-12-31"))
    w_series = triggers.run_quarterly_series(mcap_wide, cond_prod, week_ends("2024-01-05", "2025-12-31"))
    summarize("季頻（現行）", q_series)
    summarize("月頻", m_series)
    summarize("週頻", w_series)
    q_first, m_first, w_first = first_triggered(q_series), first_triggered(m_series), first_triggered(w_series)
    if q_first and w_first:
        gap_days = (pd.Timestamp(w_first) - pd.Timestamp(q_first)).days
        print(f"  週頻比季頻早偵測到TRIGGERED：{-gap_days}天"
             f"（負值代表週頻更早，正值代表週頻反而較晚，理論上週頻不該比季頻晚）")
    if m_first and w_first:
        gap_days_mw = (pd.Timestamp(w_first) - pd.Timestamp(m_first)).days
        print(f"  週頻比月頻早偵測到TRIGGERED：{-gap_days_mw}天")
    q_series.to_csv(Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
                    / "monitoring_freq_prod_quarterly.csv", index=False, encoding="utf-8-sig")
    m_series.to_csv(Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
                    / "monitoring_freq_prod_monthly.csv", index=False, encoding="utf-8-sig")
    w_series.to_csv(Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
                    / "monitoring_freq_prod_weekly.csv", index=False, encoding="utf-8-sig")

    print("\n=== 測試2：陰性對照期間，週頻/月頻會不會比季頻多抓出假警報 ===")
    for label, scheme, window_no in [("陰性A（window3，2021-2023）", "E", 3),
                                     ("陰性B（window2，2015-2016）", "A", 2)]:
        _, registration_date, oos_start, oos_end = get_window_members(scheme, window_no)
        cond = triggers.register_m1d(mcap_wide, registration_date)
        q_ends = [d.strftime("%Y-%m-%d") for d in pd.date_range(oos_start, oos_end, freq="Q")]
        m_ends = month_ends(oos_start, oos_end)
        w_ends = week_ends(oos_start, oos_end)
        q_s = triggers.run_quarterly_series(mcap_wide, cond, q_ends)
        m_s = triggers.run_quarterly_series(mcap_wide, cond, m_ends)
        w_s = triggers.run_quarterly_series(mcap_wide, cond, w_ends)
        print(f"\n  {label}（登記={registration_date}）")
        summarize("季頻", q_s)
        summarize("月頻", m_s)
        summarize("週頻", w_s)


if __name__ == "__main__":
    main()
