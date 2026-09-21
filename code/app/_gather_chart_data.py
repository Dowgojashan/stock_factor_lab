# -*- coding: utf-8 -*-
"""為9/22報告準備圖表用的底層資料（2026-09-22，使用者要求16張圖）。

這支腳本只負責**查證、彙整、存檔**成乾淨的CSV，不畫圖（畫圖另外一支腳本，
不要混在一起，方便個別除錯）。大部分資料已經在真實checkpoint／既有CSV裡，
只有兩件事需要新查資料庫：
  ①大盤（全市場市值加權）逐季前十大集中度
  ②大盤（全市場市值加權）逐季產業曝險，供跟投組自身產業曝險比較

用真實8季checkpoint（control0臂，因為它就是「現況/不調整」那條線，
L2臂的決策全部是A0/A5、投組跟control0完全相同，兩者本來就該一樣）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from database import Database  # noqa: E402

from app import industry as industry_mod  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUARTER_ENDS = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
               "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]


def load_checkpoints():
    with open("app/_runs/simulate_formal_8q_control0_L2_execlayer_v2.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    c0 = sorted([r for r in rows if r["arm"] == "control0"], key=lambda r: r["quarter_end"])
    return c0


def main():
    checkpoints = load_checkpoints()
    assert [c["quarter_end"] for c in checkpoints] == QUARTER_ENDS, "季度順序/數量不符預期"

    # ---- 1. M1-D 逐季偏離值（已經在checkpoint裡，直接撈） ----
    m1d_rows = []
    for c in checkpoints:
        m = c["m1d"]
        m1d_rows.append({"quarter_end": c["quarter_end"], "q1_weight": m["q1_weight"],
                         "registration_q1_weight": m["registration_q1_weight"],
                         "cumulative_deviation": m["cumulative_deviation"],
                         "p75": m["p75"], "p90": m["p90"], "state": m["state"]})
    pd.DataFrame(m1d_rows).to_csv(OUT_DIR / "m1d_deviation.csv", index=False, encoding="utf-8-sig")
    print("寫入 m1d_deviation.csv")

    # ---- 2. 投組自身的前十大持股集中度（從weights_end直接算，不需要DB） ----
    port_top10_rows = []
    for c in checkpoints:
        w = c["weights_end"]
        top10 = sum(sorted(w.values(), reverse=True)[:10])
        port_top10_rows.append({"quarter_end": c["quarter_end"], "portfolio_top10_weight": top10})
    print("投組自身前十大集中度（不需DB）：", port_top10_rows)

    # ---- 3. 投組自身的產業曝險（從weights_end直接算，不需要DB） ----
    industry_map = industry_mod.load_industry_map("TW")
    port_industry_rows = []
    for c in checkpoints:
        exposure = industry_mod.industry_exposure(c["weights_end"], industry_map)
        top_industry = max(exposure.items(), key=lambda kv: kv[1])
        port_industry_rows.append({"quarter_end": c["quarter_end"],
                                   "top_industry": top_industry[0], "top_industry_weight": top_industry[1],
                                   "full_exposure": exposure})

    # ---- 4. 需要DB：大盤逐季前十大集中度 + 大盤產業曝險 ----
    db = Database("TW")
    conn = db.create_connection()

    bench_top10_rows = []
    bench_industry_rows = []
    for end in QUARTER_ENDS:
        q = f"""SELECT c.company_symbol, s.market_capital
                FROM stock s JOIN company c ON s.company_id=c.id
                WHERE c.exchange_name IN ('TWSE') AND s.market_capital IS NOT NULL
                  AND s.date = (SELECT MAX(date) FROM stock WHERE date <= '{end}')"""
        d = pd.read_sql(q, conn)
        d = d.dropna(subset=["market_capital"])
        d = d[d["market_capital"] > 0]
        total = d["market_capital"].sum()
        d["weight"] = d["market_capital"] / total
        top10 = d.sort_values("weight", ascending=False).head(10)["weight"].sum()
        bench_top10_rows.append({"quarter_end": end, "benchmark_top10_weight": float(top10)})

        bench_weights = dict(zip(d["company_symbol"], d["weight"]))
        bexposure = industry_mod.industry_exposure(bench_weights, industry_map)
        top_ind = max(bexposure.items(), key=lambda kv: kv[1])
        bench_industry_rows.append({"quarter_end": end,
                                    "top_industry": top_ind[0], "top_industry_weight": top_ind[1],
                                    "full_exposure": bexposure})
        print(f"  {end}：大盤前十大集中度={top10:.2%}，大盤最大產業={top_ind[0]}({top_ind[1]:.2%})")

    # 合併輸出
    top10_df = pd.DataFrame(port_top10_rows).merge(pd.DataFrame(bench_top10_rows), on="quarter_end")
    top10_df.to_csv(OUT_DIR / "top10_concentration.csv", index=False, encoding="utf-8-sig")
    print("寫入 top10_concentration.csv")

    # 產業曝險：存兩份，一份簡表（最大產業），一份完整曝險明細（給主動權重圖用）
    ind_simple = pd.DataFrame([
        {"quarter_end": r["quarter_end"], "port_top_industry": r["top_industry"],
         "port_top_industry_weight": r["top_industry_weight"]}
        for r in port_industry_rows
    ]).merge(pd.DataFrame([
        {"quarter_end": r["quarter_end"], "bench_top_industry": r["top_industry"],
         "bench_top_industry_weight": r["top_industry_weight"]}
        for r in bench_industry_rows
    ]), on="quarter_end")
    ind_simple.to_csv(OUT_DIR / "industry_top.csv", index=False, encoding="utf-8-sig")
    print("寫入 industry_top.csv")

    # 完整明細（用最後一季，2025-12-31，做主動權重長條圖）
    last_port = port_industry_rows[-1]["full_exposure"]
    last_bench = bench_industry_rows[-1]["full_exposure"]
    all_industries = sorted(set(last_port) | set(last_bench))
    active_rows = [{"industry": ind, "portfolio_weight": last_port.get(ind, 0.0),
                    "benchmark_weight": last_bench.get(ind, 0.0),
                    "active_weight": last_port.get(ind, 0.0) - last_bench.get(ind, 0.0)}
                   for ind in all_industries]
    pd.DataFrame(active_rows).sort_values("active_weight").to_csv(
        OUT_DIR / "industry_active_weight_2025Q4.csv", index=False, encoding="utf-8-sig")
    print("寫入 industry_active_weight_2025Q4.csv")

    print("\n全部資料整理完成。")


if __name__ == "__main__":
    main()
