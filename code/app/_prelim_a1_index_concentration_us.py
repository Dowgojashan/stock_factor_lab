# -*- coding: utf-8 -*-
"""2026-10-18：美股版的M1-D門檻校準，跟台股`_prelim_a1_index_concentration.py`
同一個方法，但這是美股第一次真正做這件事——之前美股M1-D的X=1.86pp是**借用
台股的校準結果**（`triggers.py`的`X_P90`是單一全域常數，沒有分市場），原因
是當時美股`outstanding_shares`在2022年以前完全是空的，市值資料覆蓋率只有
200~650家，不夠拿來做15年以上的歷史分布校準。

2026-10-17 collector完成`USSharesOutstandingBackfiller`回補（`--from-date
2009-01-01`）後，美股市值覆蓋率從2009年開始大幅改善（2009年550家、2023年
6,443家，見`monitor.fetch_mcap_wide()`逐年統計），終於有條件做美股自己的
校準——但只能用**2009-2023（15年）**，不是台股的2007-2023（17年）：
2006-2008這三年collector沒有回補，覆蓋率仍停留在舊的~200家，不足採用。

直接用`monitor.fetch_mcap_wide()`（含`SANITY_MAX_CLOSE`離群值防線），不自己
重寫SQL——跟production用同一套資料清洗邏輯，避免兩邊各算一次又產生新的
不一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from database import Database  # noqa: E402
from app import monitor  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
TRAIN_START = "2009-01-01"  # collector回補的實際起點，2006-2008覆蓋率不足不採用
TRAIN_END = "2023-12-31"    # §11.7：只用訓練期資料，不可碰實驗期（2024-2025）


def main():
    db = Database("US")
    conn = db.create_connection()
    print(f"讀取美股市值資料（{TRAIN_START} 起，含SANITY_MAX_CLOSE離群值防線）…")
    wide = monitor.fetch_mcap_wide(conn, market="US", start=TRAIN_START)

    def q1_weight_at(d: str) -> float:
        row = monitor.asof_row(wide, d).dropna()
        ranks = row.rank(pct=True)
        return float(row[ranks >= 0.8].sum() / row.sum())

    qe = pd.date_range("2009-03-31", "2025-12-31", freq="Q")
    rows = []
    for d in qe:
        rows.append({"q": f"{d.year}Q{d.quarter}", "date": d,
                    "idx_q1_weight": q1_weight_at(d.strftime("%Y-%m-%d"))})
    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "a1_index_concentration_series_US.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"寫入 {out_path}（{len(df)} 季）")

    train = df[df.date <= TRAIN_END].reset_index(drop=True)
    HORIZON = 8   # 跟window4同長度（8季）
    train["chg_8q"] = train["idx_q1_weight"].shift(-HORIZON) - train["idx_q1_weight"]
    dist = train["chg_8q"].dropna()

    print(f"\n=== 美股指數端Q1集中度「8季後累計變動」歷史分布"
         f"（{TRAIN_START[:4]}-{TRAIN_END[:4]}，n={len(dist)}）===")
    print(f"  平均: {dist.mean()*100:+.2f}pp   標準差: {dist.std()*100:.2f}pp")
    print(f"  p10={dist.quantile(.10)*100:+.2f}pp  中位={dist.median()*100:+.2f}pp"
         f"  p75={dist.quantile(.75)*100:+.2f}pp  p90={dist.quantile(.90)*100:+.2f}pp")
    print(f"  最大: {dist.max()*100:+.2f}pp   最小: {dist.min()*100:+.2f}pp")

    print(f"\n對照：台股既有門檻（2007-2023校準）p75=1.20pp／p90=1.86pp（D27）")

    v_2023 = df.loc[df.date == "2023-12-31", "idx_q1_weight"].iloc[0]
    v_2025 = df.loc[df.date == "2025-12-31", "idx_q1_weight"].iloc[0]
    actual_chg = v_2025 - v_2023
    pctile = (dist < actual_chg).mean()
    print(f"\n=== 反查：2023-12-31({v_2023:.2%}) -> 2025-12-31({v_2025:.2%})，"
         f"實際變動 {actual_chg*100:+.2f}pp ===")
    print(f"  落在美股自己的2009-2023歷史分布第 {pctile*100:.1f} 百分位")
    p90_new, p75_new = dist.quantile(.90), dist.quantile(.75)
    print(f"  若用美股自己的X=p90（{p90_new*100:.2f}pp）："
         f"{'✅ 會觸發' if actual_chg > p90_new else '❌ 不會觸發'}")

    print("\n=== 逐季相對2023-12-31的累計變動（用美股自己的新門檻判斷）===")
    for _, r in df[(df.date >= "2024-01-01") & (df.date <= "2025-12-31")].iterrows():
        dev = r["idx_q1_weight"] - v_2023
        flag = "🔴 超過新p90" if dev > p90_new else ("🔶 超過新p75" if dev > p75_new else "")
        print(f"  {r['q']}: {r['idx_q1_weight']:.2%}  累計變動 {dev*100:+.2f}pp  {flag}")


if __name__ == "__main__":
    main()
