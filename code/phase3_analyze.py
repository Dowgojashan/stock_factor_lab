# -*- coding: utf-8 -*-
"""
Phase 3 分析：C（動態條件）到底有沒有加分？產出 Candidate Strategy 池。

老師的要求（2026-08-05 meeting）：
  「[F1×F2] OK 了以後，我就直接去檢查 C」
  兩個**互相獨立**的約束（「他的限制是不一樣的」）：
    策略數不能太少   → 否則「後面就沒有 LLM 的必要」
    每策略持股數不能太少 → 否則「策略不[穩定]」
  「最後有過關的這些 F/C 的組合…這個就是 Candidate Strategy」

核心作法：每個 F 組合都有一個自己的 **None 基準**（同 F、不加 C），
C 的價值 = 該 C 相對**同一個 F 組合**的 None 基準的 CAGR 增益。
這樣才是「控制 F 之後看 C」，不是把所有 C 混在一起平均（那正是論文 4-15 的反例）。

只讀既有結果，不重跑回測、不改 daily_sharpe。

用法：python phase3_analyze.py [--market TW]
"""
import re
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from universe_benchmark import get_bench             # noqa: E402
from phase2_analyze import MIN_HOLDINGS, avg_holdings  # noqa: E402
from sweep_config import MARKET_START, date_range_suffix  # noqa: E402
from phase1_linearity import IN_SAMPLE_END           # noqa: E402

ART = HERE / "results_artifacts"
OUT = HERE.parent / "_analysis_outputs_phase3"

OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "axes.grid": True,
    "grid.color": "#DDDDDD", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def log(m):
    print(m, flush=True)


def parse(s):
    """'PB_qb2of3__P_IC_qb0of3__C4_ROE_DYN_qmax8__v0' → (F組合, C名, C編號, C來源因子)"""
    p = s.split("__")
    fcombo = f"{p[0]}__{p[1]}"
    cname = p[2]
    if cname == "None":
        return fcombo, "None", 0, "None"
    m = re.match(r"C(\d+)_(.+?)_DYN_", cname)
    return fcombo, cname, (int(m.group(1)) if m else -1), (m.group(2) if m else "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="TW", choices=["TW", "US"])
    ap.add_argument("--variant", default="strict")
    ap.add_argument("--bench", default="universe", choices=["universe", "index"])
    ap.add_argument("--start", default=None,
                    help="自訂起始日期 YYYY-MM-DD（需與 phase3_conditions.py 執行時相同）")
    ap.add_argument("--end", default=None,
                    help="自訂結束日期 YYYY-MM-DD（需與 phase3_conditions.py 執行時相同）")
    args = ap.parse_args()
    mkt = args.market
    start = args.start or MARKET_START[mkt]
    end = args.end or IN_SAMPLE_END
    rsfx = date_range_suffix(start, end, MARKET_START[mkt], IN_SAMPLE_END)
    # 每個變體有自己的一份 Phase 3 回測（白名單不同 → 展開的 F 組合不同）
    label = (f"{mkt}_L3_M{rsfx}" if args.variant == "strict"
             else f"{mkt}_L3_{args.variant}_M{rsfx}")
    sfx = f"_{args.variant}" if args.variant != "strict" else ""
    sfx += "_idxbench" if args.bench == "index" else ""
    sfx += rsfx
    OUT.mkdir(parents=True, exist_ok=True)

    bench, _ = get_bench(mkt, args.bench, start=start, end=end)
    log(f"基準 = {bench:.2%}｜變體 = {args.variant}｜期間 {start}~{end}｜持股數門檻 ≥ {MIN_HOLDINGS}\n")

    df = pd.read_parquet(ART / label / "stats.parquet")
    df[["F組合", "C", "C編號", "C來源"]] = df["strategy"].apply(lambda s: pd.Series(parse(s)))
    log(f"讀入 {len(df)} 個策略｜F 組合 {df['F組合'].nunique()} 個｜C 狀態 {df['C'].nunique()} 種")

    # Phase 3 的回測是用「舊基準下的 203 個白名單」跑的；換基準/變體後合格的 F 組合
    # 是它的子集，故不需重跑，直接在這裡篩即可。
    p2 = OUT.parent / "_analysis_outputs_phase2" / f"{mkt}_L2{sfx}_體質檢查表.csv"
    if p2.exists():
        ok_f = {f"{s.split('__')[0]}__{s.split('__')[1]}"
                for s in pd.read_csv(p2, encoding="utf-8-sig")["strategy"]}
        n0 = df["F組合"].nunique()
        df = df[df["F組合"].isin(ok_f)].copy()
        log(f"依 Phase 2（{args.variant}／{args.bench}）體質檢查表篩選："
            f"F 組合 {n0} → {df['F組合'].nunique()}，策略 {len(df)} 個")
    else:
        log(f"⚠️ 找不到 {p2.name}，未做 F 組合篩選（用全部 203 個）")

    # ---------- 每個 F 組合的 None 基準 ----------
    base = df[df["C"] == "None"].set_index("F組合")["CAGR"].to_dict()
    df["None基準CAGR"] = df["F組合"].map(base)
    df["C增益"] = df["CAGR"] - df["None基準CAGR"]

    # ---------- 持股數（老師的第二個約束） ----------
    log("計算平均每月持股數 …")
    df["平均持股數"] = [avg_holdings(label, s) for s in df["strategy"]]
    df["持股數足夠"] = df["平均持股數"] >= MIN_HOLDINGS

    # ---------- A. C 的價值（控制 F 之後） ----------
    cdf = df[df["C"] != "None"]
    g = cdf.groupby(["C編號", "C", "C來源"]).agg(
        增益中位數=("C增益", "median"),
        增益平均=("C增益", "mean"),
        增益為正比例=("C增益", lambda s: float((s > 0).mean())),
        CAGR中位數=("CAGR", "median"),
        持股數中位數=("平均持股數", "median"),
        n=("C增益", "size"),
    ).reset_index().sort_values("增益中位數", ascending=False)
    g.to_csv(OUT / f"{mkt}_L3{sfx}_C_ranking.csv", index=False, encoding="utf-8-sig")

    log(f"\n===== A. 各 C 的價值（控制 F 後，相對同 F 的 None 基準）=====")
    show = g.copy()
    for c in ["增益中位數", "增益平均", "增益為正比例", "CAGR中位數"]:
        show[c] = show[c].map(lambda v: f"{v:+.2%}" if "增益" in c and "比例" not in c
                              else (f"{v:.1%}" if "比例" in c else f"{v:.2%}"))
    show["持股數中位數"] = show["持股數中位數"].round(1)
    log(show.to_string(index=False))

    src = cdf.groupby("C來源")["C增益"].agg(["median", "mean", "size"]).sort_values(
        "median", ascending=False)
    log(f"\n----- 依 C 的來源因子彙總 -----")
    log(src.assign(median=src["median"].map("{:+.2%}".format),
                   mean=src["mean"].map("{:+.2%}".format)).to_string())

    # ---------- B. 兩個獨立約束 ----------
    log(f"\n===== B. 老師的兩個獨立約束 =====")
    n_hold_fail = int((~df["持股數足夠"]).sum())
    log(f"  約束一（策略數）    ：共 {len(df)} 個策略")
    log(f"  約束二（每策略持股數）：{n_hold_fail} 個 < {MIN_HOLDINGS} 檔被剔除"
        f"（{n_hold_fail/len(df):.1%}）")
    log(f"  持股數分布：中位數 {df['平均持股數'].median():.1f}｜"
        f"p10 {df['平均持股數'].quantile(.1):.1f}｜p90 {df['平均持股數'].quantile(.9):.1f}")

    # ---------- C. Candidate Strategy 池 ----------
    cand = df[df["持股數足夠"] & (df["CAGR"] > bench)].copy()
    cand = cand.sort_values("CAGR", ascending=False)
    keep = ["strategy", "F組合", "C", "CAGR", "None基準CAGR", "C增益",
            "max_drawdown", "win_ratio", "daily_sharpe", "平均持股數"]
    cand[keep].to_csv(OUT / f"{mkt}_L3{sfx}_candidate_strategies.csv", index=False, encoding="utf-8-sig")
    log(f"\n===== C. Candidate Strategy 池 =====")
    log(f"  條件：持股數 ≥ {MIN_HOLDINGS} 且 CAGR > 大盤 {bench:.2%}")
    log(f"  → {len(cand)} 個（佔全部 {len(cand)/len(df):.1%}）")
    log(f"\n  Top 15：")
    log(cand[["F組合", "C", "CAGR", "C增益", "max_drawdown", "平均持股數"]].head(15).to_string(index=False))

    # ---------- 圖 ----------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    order = g.sort_values("增益中位數", ascending=False)
    colors = {"ROE": OI[0], "EPS": OI[2], "FCF_P": OI[1]}
    data = [cdf.loc[cdf["C"] == c, "C增益"].values for c in order["C"]]
    bp = axes[0].boxplot(data, labels=[f"C{int(n)}" for n in order["C編號"]],
                         showmeans=True, patch_artist=True,
                         medianprops=dict(color="black", linewidth=1.2))
    for box, s in zip(bp["boxes"], order["C來源"]):
        box.set(facecolor=colors.get(s, OI[7]), alpha=0.45)
    axes[0].axhline(0, color=OI[1], linestyle="--", linewidth=1.3, label="None 基準")
    axes[0].legend(frameon=False)
    axes[0].tick_params(axis="x", labelrotation=60, labelsize=8)
    axes[0].set_ylabel("C 增益（相對同 F 的 None）")
    axes[0].set_title("各 C 的增益分布（依中位數排序；顏色＝來源因子）")

    for s, col in colors.items():
        sub = cdf.loc[cdf["C來源"] == s, "C增益"].dropna()
        if len(sub):
            axes[1].hist(sub, bins=45, alpha=0.5, label=f"{s} 衍生（中位 {sub.median():+.2%}）",
                         color=col)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1.2)
    axes[1].legend(frameon=False)
    axes[1].set_xlabel("C 增益")
    axes[1].set_title("依 C 的來源因子分組")
    fig.suptitle(f"Phase 3  C 有沒有加分？— {mkt}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / f"{mkt}_L3{sfx}_C_gain.png", bbox_inches="tight")
    plt.close(fig)

    log(f"\n輸出於 {OUT}")


if __name__ == "__main__":
    main()
