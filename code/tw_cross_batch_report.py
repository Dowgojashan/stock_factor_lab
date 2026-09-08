# -*- coding: utf-8 -*-
"""
台股結果解讀 · 第二層：批間比較（F候選因子海選 / G堪用池估算 / D-extra最賺股票）。

對應交接文件 `台股結果解讀_交接ClaudeCode.md` §3 第二層。10批擺一起比，回答「哪個候選因子有用」
與「匯集堪用池夠不夠AI挑」。讀 _catalog/master_index.parquet（跨批去重用 is_cross_dup）+
各批 stats.parquet（淨貢獻計算用）。不重跑回測。

daily_sharpe 原樣使用，不重算、不修正。
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_batch import load_stats, load_strategy_artifacts, stock_cum_contrib  # noqa: E402

OUT_ROOT = HERE.parent / "_analysis_outputs_TW"
CROSS_DIR = OUT_ROOT / "_cross_batch"
CATALOG = HERE / "_catalog" / "master_index.parquet"
ART_DIR = HERE / "results_artifacts"

BATCHES = ["EV_EBITDA", "EV_S", "CROIC", "FCF_OI", "ROIC", "PB", "PS", "P_IC", "OCF_E", "MOM"]
BENCH_CAGR = 0.0624

OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
})


def log(msg):
    print(msg, flush=True)


def main():
    CROSS_DIR.mkdir(parents=True, exist_ok=True)

    # ==================== F. 候選因子海選比較表 ====================
    rows = []
    per_batch_df = {}
    for c in BATCHES:
        label = f"TW_batch_{c}_M"
        df = load_stats(label)
        per_batch_df[c] = df

        involves = (df["F1_factor"] == c) | (df["F2_factor"] == c)
        baseline = ~involves  # 不含候選因子＝純固定三因子(ROE/EPS/FCF_P)基準
        cand_median = float(df.loc[involves, "CAGR"].median()) if involves.any() else np.nan
        base_median = float(df.loc[baseline, "CAGR"].median()) if baseline.any() else np.nan
        net_contrib = cand_median - base_median if pd.notna(cand_median) and pd.notna(base_median) else np.nan

        usable_pos = int((df["CAGR"] > 0).sum())
        usable_beat = int((df["CAGR"] > BENCH_CAGR).sum())

        rows.append({
            "candidate": c,
            "n_strategies": len(df),
            "CAGR_median": float(df["CAGR"].median()),
            "pct_CAGR_positive": float((df["CAGR"] > 0).mean()),
            "pct_CAGR_beat_bench": float((df["CAGR"] > BENCH_CAGR).mean()),
            "daily_sharpe_median": float(df["daily_sharpe"].median()),
            "champion_CAGR": float(df["CAGR"].max()),
            "usable_CAGR_positive_n": usable_pos,
            "usable_CAGR_beat_bench_n": usable_beat,
            "candidate_involving_median_CAGR": cand_median,
            "fixed3_baseline_median_CAGR": base_median,
            "net_contribution": net_contrib,
        })
    sel = pd.DataFrame(rows).sort_values("pct_CAGR_beat_bench", ascending=False).reset_index(drop=True)
    sel.insert(0, "rank", range(1, len(sel) + 1))
    sel.to_csv(CROSS_DIR / "F_selection_table.csv", index=False, encoding="utf-8-sig")
    log("F. 候選因子海選比較表（依贏大盤比例排序）：")
    log(sel[["rank", "candidate", "n_strategies", "CAGR_median", "pct_CAGR_beat_bench",
             "net_contribution"]].to_string(index=False))

    net = sel[["candidate", "net_contribution"]].sort_values("net_contribution", ascending=False)
    net.to_csv(CROSS_DIR / "F_net_contribution_ranking.csv", index=False, encoding="utf-8-sig")

    # 視覺化：boxplot + 堪用數長條圖
    fig, ax = plt.subplots(figsize=(11, 5.5))
    order = sel["candidate"].tolist()
    data = [per_batch_df[c]["CAGR"].dropna().values for c in order]
    bp = ax.boxplot(data, labels=order, showmeans=True, patch_artist=True,
                     medianprops=dict(color="black", linewidth=1.3))
    for box in bp["boxes"]:
        box.set(facecolor=OI[0], alpha=0.35)
    ax.axhline(BENCH_CAGR, color=OI[1], linewidth=1.2, linestyle="--", label=f"大盤基準 {BENCH_CAGR:.1%}")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.legend(frameon=False)
    ax.set_ylabel("CAGR")
    ax.set_title("F 10批 CAGR 分布並排比較（依贏大盤比例排序）")
    fig.tight_layout()
    fig.savefig(CROSS_DIR / "F_boxplot_10batches.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(sel))
    ax.bar(x - 0.2, sel["usable_CAGR_positive_n"], width=0.4, label="CAGR>0", color=OI[0])
    ax.bar(x + 0.2, sel["usable_CAGR_beat_bench_n"], width=0.4, label="CAGR>大盤", color=OI[1])
    ax.set_xticks(x)
    ax.set_xticklabels(sel["candidate"], rotation=45)
    ax.legend(frameon=False)
    ax.set_ylabel("堪用策略數")
    ax.set_title("F 10批堪用數長條圖")
    fig.tight_layout()
    fig.savefig(CROSS_DIR / "F_usable_bar_10batches.png", bbox_inches="tight")
    plt.close(fig)

    # ==================== G. 堪用池估算（乙案：跨批匯集） ====================
    cat = pd.read_parquet(CATALOG)
    cat_tw = cat[cat["market"] == "TW"].copy()
    cat_tw["CAGR"] = pd.to_numeric(cat_tw["CAGR"], errors="coerce")

    total_rows = len(cat_tw)
    cross_dup_n = int(cat_tw["is_cross_dup"].sum())
    uniq = cat_tw[~cat_tw["is_cross_dup"]]  # 去重後獨立策略（跨批同名只留第一個）

    pool_pos_raw = int((cat_tw["CAGR"] > 0).sum())
    pool_beat_raw = int((cat_tw["CAGR"] > BENCH_CAGR).sum())
    pool_pos_uniq = int((uniq["CAGR"] > 0).sum())
    pool_beat_uniq = int((uniq["CAGR"] > BENCH_CAGR).sum())

    g_summary = pd.Series({
        "total_rows_10batches_TW": total_rows,
        "cross_dup_rows": cross_dup_n,
        "unique_strategies": len(uniq),
        "pool_CAGR_positive_raw(含跨批重複)": pool_pos_raw,
        "pool_CAGR_beat_bench_raw(含跨批重複)": pool_beat_raw,
        "pool_CAGR_positive_dedup(去重後)": pool_pos_uniq,
        "pool_CAGR_beat_bench_dedup(去重後)": pool_beat_uniq,
        "bench_CAGR_used": BENCH_CAGR,
    })
    g_summary.to_csv(CROSS_DIR / "G_usable_pool_estimate.csv", encoding="utf-8-sig")
    log("\nG. 堪用池估算（乙案，10批匯集、去重後）：")
    log(g_summary.to_string())

    # 逐月報酬齊備性抽查：每批抽5個策略檢查 return_table.parquet
    completeness_rows = []
    rng = np.random.default_rng(42)
    for c in BATCHES:
        df = per_batch_df[c]
        sample_names = df["strategy"].sample(min(5, len(df)), random_state=42).tolist()
        for name in sample_names:
            strat_dir = ART_DIR / f"TW_batch_{c}_M" / name
            rt_path = strat_dir / "return_table.parquet"
            ok, n_rows = False, 0
            if rt_path.exists():
                try:
                    rt = pd.read_parquet(rt_path)
                    n_rows = len(rt)
                    ok = n_rows > 0
                except Exception:
                    ok = False
            completeness_rows.append({"candidate": c, "strategy": name, "return_table_exists": ok,
                                       "n_rows": n_rows})
    comp_df = pd.DataFrame(completeness_rows)
    comp_df.to_csv(CROSS_DIR / "G_return_table_completeness_sample.csv", index=False, encoding="utf-8-sig")
    ok_rate = comp_df["return_table_exists"].mean()
    log(f"\nG. return_table 齊備性抽查（50檔樣本）：完整率 {ok_rate:.1%}")

    # ==================== D-extra. 跨批最賺股票（冠軍策略明細歸因） ====================
    champ_rows = []
    for c in BATCHES:
        df = per_batch_df[c]
        champ = df.loc[df["CAGR"].idxmax()]
        strat_dir = ART_DIR / f"TW_batch_{c}_M" / champ["strategy"]
        art = load_strategy_artifacts(strat_dir)
        t = art.get("trades")
        top_stock, top_contrib = None, np.nan
        if t is not None and len(t) and "stock_id" in t.columns:
            contrib = stock_cum_contrib(t)
            pos = contrib[contrib > 0]
            if len(pos):
                top_stock = str(pos.idxmax())
                top_contrib = float(pos.max())
        champ_rows.append({
            "candidate": c, "strategy": champ["strategy"], "CAGR": float(champ["CAGR"]),
            "daily_sharpe": float(champ["daily_sharpe"]), "max_drawdown": float(champ["max_drawdown"]),
            "top1_stock_id": top_stock, "top1_cum_contrib": top_contrib,
        })
    champ_df = pd.DataFrame(champ_rows).sort_values("CAGR", ascending=False)
    champ_df.to_csv(CROSS_DIR / "D_extra_champion_strategies_and_top_stock.csv", index=False,
                     encoding="utf-8-sig")
    log("\nD-extra. 跨批冠軍策略與各自Top1貢獻股：")
    log(champ_df.to_string(index=False))

    log(f"\n完成，批間比較輸出於 {CROSS_DIR}")


if __name__ == "__main__":
    main()
