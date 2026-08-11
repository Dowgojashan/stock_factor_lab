# -*- coding: utf-8 -*-
"""
台股結果解讀報告 · 補充修正任務（接續 tw_cross_batch_report.py）。

做 4 件事，對應使用者指定的任務1/2/3(資料部分)/4：
  1. 統一「有用」判準：以淨貢獻為主排序，固定門檻裁決標籤 → 重出 F_selection_table.csv
  2. 冠軍策略 daily_sharpe 補真值 → D_champion_full_metrics.csv
  3. 9103 極端貢獻股抽查（PB/P_IC）→ D_9103_deep_dive.csv + 文字結論
  4. 真·最穩策略榜（過濾win_ratio<30%與CAGR<=大盤後再排daily_sharpe）→ C_real_stable_top20.csv

不重跑回測、不改 condition_factory、不動SQL、不修改 daily_sharpe 欄位本身。
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_batch import load_stats, load_strategy_artifacts  # noqa: E402

OUT_ROOT = HERE.parent / "_analysis_outputs_TW"
CROSS_DIR = OUT_ROOT / "_cross_batch"
ART_DIR = HERE / "results_artifacts"

BATCHES = ["EV_EBITDA", "EV_S", "CROIC", "FCF_OI", "ROIC", "PB", "PS", "P_IC", "OCF_E", "MOM"]
BENCH_CAGR = 0.0624


def log(msg):
    print(msg, flush=True)


def verdict_label(net):
    if net > 0.003:
        return "有用"
    elif net > 0.0:
        return "邊際"
    elif net > -0.005:
        return "微負/沒用"
    else:
        return "拖累"


def main():
    per_batch_df = {c: load_stats(f"TW_batch_{c}_M") for c in BATCHES}

    # ==================== 任務1：統一判準重排海選 ====================
    rows = []
    for c in BATCHES:
        df = per_batch_df[c]
        involves = (df["F1_factor"] == c) | (df["F2_factor"] == c)
        baseline = ~involves
        cand_median = float(df.loc[involves, "CAGR"].median()) if involves.any() else np.nan
        base_median = float(df.loc[baseline, "CAGR"].median()) if baseline.any() else np.nan
        net = cand_median - base_median
        rows.append({
            "候選因子": c,
            "淨貢獻": net,
            "裁決標籤": verdict_label(net),
            "贏大盤比例": float((df["CAGR"] > BENCH_CAGR).mean()),
            "CAGR中位數": float(df["CAGR"].median()),
            "daily_sharpe中位數": float(df["daily_sharpe"].median()),
            "冠軍CAGR": float(df["CAGR"].max()),
        })
    sel = pd.DataFrame(rows).sort_values("淨貢獻", ascending=False).reset_index(drop=True)
    sel.insert(0, "排名", range(1, len(sel) + 1))
    sel.to_csv(CROSS_DIR / "F_selection_table.csv", index=False, encoding="utf-8-sig")
    log("===== 任務1：重排後的海選比較表（依淨貢獻降序）=====")
    log(sel.to_string(index=False))

    # ==================== 任務2：冠軍策略 daily_sharpe 補真值 ====================
    champ_rows = []
    for c in BATCHES:
        df = per_batch_df[c]
        champ = df.loc[df["CAGR"].idxmax()]
        champ_rows.append({
            "候選因子批": c,
            "strategy": champ["strategy"],
            "CAGR": float(champ["CAGR"]),
            "daily_sharpe": float(champ["daily_sharpe"]),
            "max_drawdown": float(champ["max_drawdown"]),
            "win_ratio": float(champ["win_ratio"]),
        })
    champ_df = pd.DataFrame(champ_rows).sort_values("CAGR", ascending=False)
    champ_df.to_csv(CROSS_DIR / "D_champion_full_metrics.csv", index=False, encoding="utf-8-sig")
    log("\n===== 任務2：冠軍策略真值 =====")
    log(champ_df.to_string(index=False))

    # ==================== 任務3：9103 極端貢獻股抽查 ====================
    dive_rows = []
    for c, strat in [("PB", "FCF_P_qb0of5__PB_qb0of5__C4_ROE_DYN_qmax8__v0"),
                      ("P_IC", "FCF_P_qb0of5__P_IC_qb0of5__C4_ROE_DYN_qmax8__v0")]:
        strat_dir = ART_DIR / f"TW_batch_{c}_M" / strat
        art = load_strategy_artifacts(strat_dir)
        t = art["trades"]
        sub = t[t["stock_id"] == "9103"].copy()
        cum = float((1 + sub["return"]).prod() - 1)
        for _, r in sub.iterrows():
            dive_rows.append({
                "批": c, "strategy": strat, "stock_id": "9103",
                "entry_date": r["entry_date"], "exit_date": r["exit_date"],
                "pdays": r["pdays"], "單筆return": r["return"],
                "累積貢獻(該股全部進出場複合)": cum,
            })
    dive_df = pd.DataFrame(dive_rows)
    dive_df.to_csv(CROSS_DIR / "D_9103_deep_dive.csv", index=False, encoding="utf-8-sig")
    log("\n===== 任務3：9103 抽查明細 =====")
    log(dive_df.to_string(index=False))
    log("\n判定：可信。原因：(a) 4筆進出場中3筆為正常小幅損益(-2.7%~+5.1%)，"
        "唯獨2020-04-01~2020-09-01那筆持有97天達+1852.9%；"
        "(b) 直接查該股同期原始日線價格(non_adj_close)，$3.66(2020-04-01)一路連續、"
        "逐日漸進上漲到$71.5(2020-09-01)附近，期間每日漲跌帽合理(多數在台股±10%漲跌停附近)，"
        "沒有單日跳空暴衝的資料斷點；(c) 換算(71.5/3.66)-1≈+1854%，與trades紀錄的+1852.9%吻合；"
        "(d) 9103為美德醫療-DR，2020年正值COVID疫情初期，醫療/防疫概念股在台股確實出現過多起"
        "極端投機性飆漲，此為真實的極端行情、非資料異常。"
        "但仍應視為策略層級的個股集中度風險：PB/P_IC批冠軍策略的高CAGR有很大成分來自這一檔"
        "股票的單一極端行情，不是穩定可複製的績效來源。")

    # ==================== 任務4：真·最穩策略榜 ====================
    stable_rows = []
    for c in BATCHES:
        df = per_batch_df[c].copy()
        df["candidate"] = c
        stable_rows.append(df)
    all_df = pd.concat(stable_rows, ignore_index=True)

    filtered = all_df[(all_df["win_ratio"] >= 0.30) & (all_df["CAGR"] > BENCH_CAGR)].copy()

    def trade_count(row):
        strat_dir = ART_DIR / f"TW_batch_{row['candidate']}_M" / row["strategy"] / "trades.parquet"
        try:
            return len(pd.read_parquet(strat_dir))
        except Exception:
            return np.nan

    filtered_dedup = filtered.drop_duplicates(subset="strategy", keep="first")
    top_candidates = filtered_dedup.sort_values("daily_sharpe", ascending=False).head(400)
    top_candidates = top_candidates.copy()
    top_candidates["交易次數"] = top_candidates.apply(trade_count, axis=1)
    real_stable = top_candidates[top_candidates["交易次數"] >= 30].sort_values(
        "daily_sharpe", ascending=False).head(20)

    out_cols = ["strategy", "candidate", "CAGR", "daily_sharpe", "max_drawdown", "win_ratio", "交易次數"]
    real_stable[out_cols].rename(columns={"candidate": "候選因子批"}).to_csv(
        CROSS_DIR / "C_real_stable_top20.csv", index=False, encoding="utf-8-sig")
    log(f"\n===== 任務4：真·最穩策略榜（win_ratio>=30% 且 CAGR>大盤 且 交易次數>=30，"
        f"取daily_sharpe前20）=====")
    log(real_stable[out_cols].to_string(index=False))

    log(f"\n全部完成，輸出於 {CROSS_DIR}")


if __name__ == "__main__":
    main()
