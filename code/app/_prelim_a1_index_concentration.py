# -*- coding: utf-8 -*-
"""M1-D 改版（使用者 2026-09-16 定案：只用指數端集中度，不用投組端敞口偏離）。

指數端集中度（Q1＝市值最大 20% 那組佔全市場市值的比重）完全不依賴投組持股，
是 §3.2 已驗證過的乾淨、單調量測。這裡建立它 2007-2023（符合 §11.7，不用
實驗期資料）的逐季序列，算出「N 季後累計變動」的歷史分布，拿來訂 M1-D 的 X，
再反過來檢驗 2023-12-31→2025-12-31（8 季，即實驗期）是否真的會落在門檻外。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from database import Database  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"


def main():
    db = Database("TW")
    conn = db.create_connection()
    print("讀取市值資料（2007-01 起）…")
    mcap_raw = pd.read_sql(
        f"SELECT s.date, c.company_symbol, s.market_capital FROM stock s "
        f"JOIN company c ON s.company_id=c.id WHERE {db._exchange_in_clause()} "
        f"AND s.market_capital IS NOT NULL AND s.date >= '2007-01-01' ORDER BY s.date",
        conn)
    mcap_raw["date"] = pd.to_datetime(mcap_raw["date"])
    mcap = mcap_raw.pivot(index="date", columns="company_symbol", values="market_capital")

    def asof(d: str) -> pd.Series:
        ts = pd.Timestamp(d)
        avail = mcap.index[mcap.index <= ts]
        return mcap.loc[avail.max()]

    def q1_weight(row: pd.Series) -> float:
        m = row.dropna()
        ranks = m.rank(pct=True)
        return m[ranks >= 0.8].sum() / m.sum()

    qe = pd.date_range("2007-03-31", "2025-12-31", freq="Q")
    rows = []
    for d in qe:
        rows.append({"q": f"{d.year}Q{d.quarter}", "date": d, "idx_q1_weight": q1_weight(asof(d.strftime("%Y-%m-%d")))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "a1_index_concentration_series.csv", index=False, encoding="utf-8-sig")
    print(f"寫入 {OUT_DIR / 'a1_index_concentration_series.csv'}（{len(df)} 季）")

    # 只用 2007-2023（訓練期，符合 §11.7），算「8 季後累計變動」的歷史分布
    train = df[df.date <= "2023-12-31"].reset_index(drop=True)
    HORIZON = 8   # 跟實驗期一樣長（2023-12-31 -> 2025-12-31 共 8 季）
    train["chg_8q"] = train["idx_q1_weight"].shift(-HORIZON) - train["idx_q1_weight"]
    dist = train["chg_8q"].dropna()

    print(f"\n=== 指數端 Q1 集中度「8 季後累計變動」的歷史分布（2007-2023，n={len(dist)}）===")
    print(f"  平均: {dist.mean()*100:+.2f}pp   標準差: {dist.std()*100:.2f}pp")
    print(f"  p10={dist.quantile(.10)*100:+.2f}pp  中位={dist.median()*100:+.2f}pp  p90={dist.quantile(.90)*100:+.2f}pp")
    print(f"  最大: {dist.max()*100:+.2f}pp   最小: {dist.min()*100:+.2f}pp")

    # 反過來檢驗：2023-12-31 -> 2025-12-31 實際變動是多少，落在歷史分布第幾百分位
    v_2023 = df.loc[df.date == "2023-12-31", "idx_q1_weight"].iloc[0]
    v_2025 = df.loc[df.date == "2025-12-31", "idx_q1_weight"].iloc[0]
    actual_chg = v_2025 - v_2023
    pctile = (dist < actual_chg).mean()
    print(f"\n=== 反查：2023-12-31({v_2023:.2%}) -> 2025-12-31({v_2025:.2%})，實際變動 {actual_chg*100:+.2f}pp ===")
    print(f"  落在 2007-2023 歷史分布的第 {pctile*100:.1f} 百分位")
    print(f"  若 X = 歷史 p90（{dist.quantile(.90)*100:.2f}pp）：{'✅ 會觸發' if actual_chg > dist.quantile(.90) else '❌ 不會觸發'}")
    print(f"  若 X = 歷史 p95：{'✅ 會觸發' if actual_chg > dist.quantile(.95) else '❌ 不會觸發'}")
    print(f"  若 X = 歷史最大值：{'✅ 會觸發' if actual_chg > dist.max() else '❌ 不會觸發'}")

    # 也印出逐季 vs 登記時點(2023-12-31) 的累計變動，供之後 M1-D 逐季判斷用
    print("\n=== 逐季相對登記時點(2023-12-31)的累計變動 ===")
    for _, r in df[(df.date >= "2024-01-01") & (df.date <= "2025-12-31")].iterrows():
        dev = r["idx_q1_weight"] - v_2023
        flag = "🔴 超過p90" if dev > dist.quantile(.90) else ("🔶 超過p75" if dev > dist.quantile(.75) else "")
        print(f"  {r['q']}: {r['idx_q1_weight']:.2%}  累計變動 {dev*100:+.2f}pp  {flag}")


if __name__ == "__main__":
    main()
