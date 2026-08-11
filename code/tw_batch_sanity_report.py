# -*- coding: utf-8 -*-
"""
台股結果解讀 · 第一層：批內逐批分析（A健全性 / B分布 / C賺錢與冠軍 / E有趣模式）。

對應交接文件 `台股結果解讀_交接ClaudeCode.md` §3 第一層。10批各自獨立跑，絕不混算。
只讀既有 results_artifacts/{batch}/stats.parquet，不重跑回測、不改 condition_factory。

daily_sharpe 原樣使用，不重算、不修正（見交接文件 §1.3）。

輸出：_analysis_outputs_TW/{batch}/ 下的 CSV（數字表）+ PNG（圖）。
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
from analyze_batch import load_stats, parse_bucket, factor_list  # noqa: E402

OUT_ROOT = HERE.parent / "_analysis_outputs_TW"

BATCHES = ["EV_EBITDA", "EV_S", "CROIC", "FCF_OI", "ROIC", "PB", "PS", "P_IC", "OCF_E", "MOM"]

OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
})

BENCH_CAGR = 0.0624  # TAIEX 全期間(2000-01-04~2026-07-24)年化，見對話中確認


def log(msg):
    print(msg, flush=True)


def analyze_one_batch(candidate: str, out_dir: Path) -> dict:
    label = f"TW_batch_{candidate}_M"
    df = load_stats(label)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(df)

    # ---------------- A. 健全性檢查 ----------------
    nan_rates = {c: float(df[c].isna().mean()) for c in
                 ["CAGR", "daily_sharpe", "max_drawdown", "win_ratio"] if c in df.columns}

    extreme_hi = df.sort_values("CAGR", ascending=False).head(10)[
        ["strategy", "CAGR", "daily_sharpe", "max_drawdown", "win_ratio"]]
    extreme_lo = df.sort_values("CAGR", ascending=True).head(10)[
        ["strategy", "CAGR", "daily_sharpe", "max_drawdown", "win_ratio"]]
    extreme_dd = df.sort_values("max_drawdown", ascending=True).head(10)[
        ["strategy", "CAGR", "daily_sharpe", "max_drawdown", "win_ratio"]]
    extreme_hi.to_csv(out_dir / "A_extreme_top10_CAGR.csv", index=False, encoding="utf-8-sig")
    extreme_lo.to_csv(out_dir / "A_extreme_bottom10_CAGR.csv", index=False, encoding="utf-8-sig")
    extreme_dd.to_csv(out_dir / "A_extreme_deepest10_MDD.csv", index=False, encoding="utf-8-sig")

    # 疑似異常標記（不修正，只標記）：CAGR極端(>1.0=100%年化)、MDD=0、win_ratio=1.0
    suspicious = df[(df["CAGR"] > 1.0) | (df["max_drawdown"] >= -1e-9) | (df["win_ratio"] >= 0.999)]
    suspicious[["strategy", "CAGR", "daily_sharpe", "max_drawdown", "win_ratio"]].to_csv(
        out_dir / "A_suspicious_flagged.csv", index=False, encoding="utf-8-sig")

    # ---------------- B. 母體分布全貌 ----------------
    dist_summary = {}
    for col in ["CAGR", "max_drawdown", "daily_sharpe", "win_ratio"]:
        s = df[col].dropna()
        dist_summary[col] = {
            "min": float(s.min()), "p25": float(s.quantile(.25)), "median": float(s.median()),
            "p75": float(s.quantile(.75)), "max": float(s.max()),
            "mean": float(s.mean()), "std": float(s.std()),
        }
    pd.DataFrame(dist_summary).T.to_csv(out_dir / "B_distribution_summary.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col, color in zip(axes.flat, ["CAGR", "max_drawdown", "daily_sharpe", "win_ratio"], OI):
        s = df[col].dropna()
        ax.hist(s, bins=60, color=color, edgecolor="white", linewidth=0.3)
        ax.axvline(s.median(), color="black", linewidth=1.2, linestyle="--", label=f"中位數 {s.median():.3f}")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(col)
    fig.suptitle(f"B 母體分布 — {label}（n={n:,}）", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "B_distribution_hist.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["max_drawdown"].abs(), df["CAGR"], s=4, alpha=0.25, color=OI[0])
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.axhline(BENCH_CAGR, color=OI[1], linewidth=1.2, linestyle="--", label=f"大盤基準 {BENCH_CAGR:.1%}")
    ax.set_xlabel("|max_drawdown|")
    ax.set_ylabel("CAGR")
    ax.legend(frameon=False)
    ax.set_title(f"B 風險-報酬散布圖 — {label}")
    fig.tight_layout()
    fig.savefig(out_dir / "B_risk_return_scatter.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------- C. 有沒有賺錢與冠軍 ----------------
    pct_pos = float((df["CAGR"] > 0).mean())
    pct_neg = float((df["CAGR"] <= 0).mean())
    pct_beat_bench = float((df["CAGR"] > BENCH_CAGR).mean())
    if pct_pos > 0.65:
        verdict = "偏賺（多數策略CAGR為正）"
    elif pct_pos < 0.35:
        verdict = "偏賠（多數策略CAGR為負）"
    else:
        verdict = "兩極（賺賠分布接近對半）"

    top20_cagr = df.sort_values("CAGR", ascending=False).head(20)[
        ["strategy", "F1", "F2", "C", "V", "CAGR", "daily_sharpe", "max_drawdown", "win_ratio"]]
    top20_cagr.to_csv(out_dir / "C_top20_CAGR.csv", index=False, encoding="utf-8-sig")

    top20_sharpe = df.sort_values("daily_sharpe", ascending=False).head(20)[
        ["strategy", "F1", "F2", "C", "V", "CAGR", "daily_sharpe", "max_drawdown", "win_ratio"]]
    top20_sharpe.to_csv(out_dir / "C_top20_daily_sharpe.csv", index=False, encoding="utf-8-sig")
    top_sharpe_row = top20_sharpe.iloc[0]  # 全批daily_sharpe最高者，未必=CAGR冠軍（見健全性備註）

    top20_mdd = df.sort_values("max_drawdown", ascending=False).head(20)[
        ["strategy", "F1", "F2", "C", "V", "CAGR", "daily_sharpe", "max_drawdown", "win_ratio"]]
    top20_mdd.to_csv(out_dir / "C_top20_shallowest_MDD.csv", index=False, encoding="utf-8-sig")

    # ---------------- E. 有趣模式 ----------------
    single = df[~df["is_pair"]]
    f1_group = single.groupby("F1_factor")["CAGR"].agg(["mean", "median", "std", "count"]).sort_values(
        "median", ascending=False)
    f1_group.to_csv(out_dir / "E_F1_factor_group.csv", encoding="utf-8-sig")

    pair_stats = df.groupby("is_pair")["CAGR"].agg(["mean", "median", "std", "count"])
    pair_stats.index = pair_stats.index.map({False: "單因子(F2=None)", True: "雙因子(F2≠None)"})
    pair_stats.to_csv(out_dir / "E_single_vs_double.csv", encoding="utf-8-sig")

    c_stats = df.groupby("C_kind")["CAGR"].agg(["mean", "median", "std", "count"]).sort_values(
        "median", ascending=False)
    c_stats.to_csv(out_dir / "E_C_condition_effect.csv", encoding="utf-8-sig")
    none_median = c_stats.loc["None", "median"] if "None" in c_stats.index else np.nan
    better_c = c_stats[c_stats["median"] > none_median].index.tolist() if pd.notna(none_median) else []
    worse_c = c_stats[c_stats["median"] <= none_median].index.tolist() if pd.notna(none_median) else []

    cand_group_row = f1_group.loc[candidate] if candidate in f1_group.index else None

    summary = {
        "candidate": candidate,
        "label": label,
        "n_strategies": n,
        "nan_rate_CAGR": nan_rates.get("CAGR"),
        "nan_rate_daily_sharpe": nan_rates.get("daily_sharpe"),
        "n_suspicious_flagged": int(len(suspicious)),
        "CAGR_median": dist_summary["CAGR"]["median"],
        "CAGR_mean": dist_summary["CAGR"]["mean"],
        "pct_CAGR_positive": pct_pos,
        "pct_CAGR_negative": pct_neg,
        "pct_CAGR_beat_bench": pct_beat_bench,
        "verdict": verdict,
        "champion_CAGR": float(df["CAGR"].max()),
        "champion_strategy": df.loc[df["CAGR"].idxmax(), "strategy"],
        "champion_strategy_daily_sharpe": float(df.loc[df["CAGR"].idxmax(), "daily_sharpe"]),
        "highest_daily_sharpe_value": float(top_sharpe_row["daily_sharpe"]),
        "highest_daily_sharpe_strategy": top_sharpe_row["strategy"],
        "highest_daily_sharpe_strategy_CAGR": float(top_sharpe_row["CAGR"]),
        "highest_daily_sharpe_strategy_win_ratio": float(top_sharpe_row["win_ratio"]),
        "candidate_factor_group_median_CAGR": float(cand_group_row["median"]) if cand_group_row is not None else None,
        "candidate_factor_group_n": int(cand_group_row["count"]) if cand_group_row is not None else None,
        "better_than_none_C": ",".join(better_c),
        "worse_or_equal_C": ",".join(worse_c),
    }
    pd.Series(summary).to_csv(out_dir / "SUMMARY.csv", encoding="utf-8-sig")
    return summary


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in BATCHES:
        log(f">> 分析 TW_batch_{c}_M ...")
        out_dir = OUT_ROOT / f"TW_batch_{c}_M"
        s = analyze_one_batch(c, out_dir)
        rows.append(s)
        log(f"   n={s['n_strategies']:,}  CAGR中位={s['CAGR_median']:.3f}  "
            f"賺錢比例={s['pct_CAGR_positive']:.1%}  贏大盤比例={s['pct_CAGR_beat_bench']:.1%}  "
            f"結論={s['verdict']}")
    pd.DataFrame(rows).to_csv(OUT_ROOT / "_all_batches_summary.csv", index=False, encoding="utf-8-sig")
    log(f"\n完成，總覽存於 {OUT_ROOT / '_all_batches_summary.csv'}")


if __name__ == "__main__":
    main()
