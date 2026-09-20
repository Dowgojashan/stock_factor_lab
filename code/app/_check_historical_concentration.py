# -*- coding: utf-8 -*-
"""路徑A查證（2026-09-18，使用者要求）：查台股 2000-2023 歷史上有沒有類似
2024-2025 的極端市值集中期，作為驗證「M1-D觸發時才傾斜」這個條件式W2機制的
獨立樣本，避免只能用2024-2025本身（測試期）驗證。

方法：沿用 monitor.environment_layer() 同一套定義（Q1=市值最大20%那組佔全市場
市值比重），對2000-2026整段歷史逐月計算，找出Q1集中度本身的時間序列，
以及仿照triggers.py的「8季後累計變動」計算方式，看歷史上有沒有出現過
可比擬2024-2025（+4.34pp累計變動，第100百分位）的段落。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from app import monitor  # noqa: E402
from database import Database  # noqa: E402


def main():
    db = Database("TW")
    conn = db.create_connection()
    print(">> 抓取 2000-2026 全市場市值寬表 ...")
    mcap_wide = monitor.fetch_mcap_wide(conn, start="2000-01-01")
    print(f"   覆蓋 {mcap_wide.index.min().date()} ~ {mcap_wide.index.max().date()}，"
         f"{len(mcap_wide.columns)} 檔股票")

    # 逐月算 Q1 集中度（用月底最後一個交易日）
    monthly_dates = mcap_wide.resample("M").last().index
    rows = []
    for d in monthly_dates:
        row = monitor.asof_row(mcap_wide, d.strftime("%Y-%m-%d")).dropna()
        if len(row) == 0:
            continue
        total = row.sum()
        ranks = row.rank(pct=True)
        q1 = row[ranks >= 0.8]
        q1_weight = q1.sum() / total
        rows.append({"date": d, "q1_weight": q1_weight, "n_universe": len(row)})
    series = pd.DataFrame(rows).set_index("date")["q1_weight"]
    series.to_csv("../_analysis_outputs_applayer/hist_q1_weight_2000_2026_monthly.csv")
    print(f"   逐月序列已存檔，共 {len(series)} 筆")

    print("\n=== Q1 集中度歷年走勢（每年12月底）===")
    yearend = series[series.index.month == 12]
    print(yearend.to_string())

    print("\n=== Q1 集中度歷史最高/最低點 ===")
    print("最高：", series.idxmax().date(), f"{series.max():.4f}")
    print("最低：", series.idxmin().date(), f"{series.min():.4f}")

    print("\n=== 仿照 M1-D：任意起點8季（24個月）後的累計變動，找歷史最大值 ===")
    # 用季底日期序列（跟 triggers.py 的季度節奏一致）近似
    q_dates = series.index[series.index.month.isin([3, 6, 9, 12])]
    q_series = series.loc[q_dates]
    changes = []
    for i in range(len(q_series) - 8):
        start_date, end_date = q_series.index[i], q_series.index[i + 8]
        chg = q_series.iloc[i + 8] - q_series.iloc[i]
        changes.append({"start": start_date, "end": end_date, "chg_8q": chg})
    chg_df = pd.DataFrame(changes).sort_values("chg_8q", ascending=False)
    print(chg_df.head(15).to_string(index=False))
    print("\n...（2007年以後才是M1-D校準用的訓練樣本，這裡刻意含2000-2006一併檢查）")

    print("\n=== 2000年以來，有沒有任何8季累計變動 >= 2024-2025的 +4.34pp ===")
    hit = chg_df[chg_df["chg_8q"] >= 0.0434]
    print(f"符合的段落數：{len(hit)}")
    if len(hit):
        print(hit.to_string(index=False))
    else:
        print("（沒有找到——2024-2025 在 2000 年以來的資料裡都是獨一無二的極端值）")


if __name__ == "__main__":
    main()
