# -*- coding: utf-8 -*-
"""
Phase 4 分析：V（估值濾網）到底該不該加？對應論文 4-20 ~ 4-24。

資料來源：Phase 3 的 {mkt}_L3_M（v0）+ Phase 4 的 {mkt}_L4_M（v1），
依「F組合__C」配對比較。同一個 F×C 的 v0/v1 是同一條策略的兩個版本，
故可直接相減得到 ΔCAGR / ΔMDD。

V 的定義（fcv_core.MarketData.get_v_mask，沿用學姊論文）：
  v1 = PE 低於近 4 季均值、但高於近 4 季最低點
     ＝「相對自己歷史便宜、但不是最谷底」——買回檔，不接墜落的刀

產出對應論文：
  4-20  ΔCAGR 分布
  4-21  ΔCAGR × ΔMDD 四象限（V1 有沒有真的用報酬換到風險下降？）
  4-22  V0/V1 依 F1 桶分組
  4-23  V0/V1 依 C 分組

只讀既有結果，不重跑回測、不改 daily_sharpe。

用法：python phase4_analyze.py [--market TW] [--variant strict] [--bench universe]
"""
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
from universe_benchmark import get_bench                 # noqa: E402
from phase2_analyze import MIN_HOLDINGS, avg_holdings    # noqa: E402
from phase3_analyze import parse                         # noqa: E402
from sweep_config import MARKET_START, date_range_suffix  # noqa: E402
from phase1_linearity import IN_SAMPLE_END               # noqa: E402

ART = HERE / "results_artifacts"
OUT = HERE.parent / "_analysis_outputs_phase4"

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


def load(mkt, lab, variant="strict", rsfx=""):
    """每個變體有自己的一份 Phase 3/4 回測（白名單不同）。"""
    name = (f"{mkt}_{lab}_M{rsfx}" if variant == "strict"
            else f"{mkt}_{lab}_{variant}_M{rsfx}")
    df = pd.read_parquet(ART / name / "stats.parquet")
    df[["F組合", "C", "C編號", "C來源"]] = df["strategy"].apply(lambda s: pd.Series(parse(s)))
    df["key"] = df["F組合"] + "__" + df["C"]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="TW", choices=["TW", "US"])
    ap.add_argument("--variant", default="strict")
    ap.add_argument("--bench", default="universe", choices=["universe", "index"])
    ap.add_argument("--start", default=None,
                    help="自訂起始日期 YYYY-MM-DD（需與 phase4_valuation.py 執行時相同）")
    ap.add_argument("--end", default=None,
                    help="自訂結束日期 YYYY-MM-DD（需與 phase4_valuation.py 執行時相同）")
    args = ap.parse_args()
    mkt = args.market
    start = args.start or MARKET_START[mkt]
    end = args.end or IN_SAMPLE_END
    rsfx = date_range_suffix(start, end, MARKET_START[mkt], IN_SAMPLE_END)
    sfx = f"_{args.variant}" if args.variant != "strict" else ""
    sfx += "_idxbench" if args.bench == "index" else ""
    sfx += rsfx
    OUT.mkdir(parents=True, exist_ok=True)

    bench, _ = get_bench(mkt, args.bench, start=start, end=end)
    log(f"基準 = {bench:.2%}｜變體 = {args.variant}｜期間 {start}~{end}\n")

    v0 = load(mkt, "L3", args.variant, rsfx)
    v1 = load(mkt, "L4", args.variant, rsfx)
    log(f"v0（Phase 3）{len(v0)} 個｜v1（Phase 4）{len(v1)} 個")

    # 依 Phase 2 體質檢查表篩 F 組合（換基準/變體後合格者是原 203 的子集）
    p2 = OUT.parent / "_analysis_outputs_phase2" / f"{mkt}_L2{sfx}_體質檢查表.csv"
    if p2.exists():
        ok_f = {"__".join(s.split("__")[:2])
                for s in pd.read_csv(p2, encoding="utf-8-sig")["strategy"]}
        v0 = v0[v0["F組合"].isin(ok_f)]
        v1 = v1[v1["F組合"].isin(ok_f)]
        log(f"依 Phase 2（{args.variant}／{args.bench}）篩後：F 組合 {v0['F組合'].nunique()} 個")

    m = v0.merge(v1, on="key", suffixes=("_v0", "_v1"))
    log(f"成功配對 {len(m)} 組 v0/v1\n")

    m["dCAGR"] = m["CAGR_v1"] - m["CAGR_v0"]
    # ΔMDD 定義為「風險下降幅度」：|MDD_v0| − |MDD_v1| > 0 代表 v1 回撤較淺
    m["dMDD"] = m["max_drawdown_v0"].abs() - m["max_drawdown_v1"].abs()

    log("計算 v1 的平均持股數 …")
    l3 = f"{mkt}_L3_M{rsfx}" if args.variant == "strict" else f"{mkt}_L3_{args.variant}_M{rsfx}"
    l4 = f"{mkt}_L4_M{rsfx}" if args.variant == "strict" else f"{mkt}_L4_{args.variant}_M{rsfx}"
    m["持股數_v1"] = [avg_holdings(l4, s) for s in m["strategy_v1"]]
    m["持股數_v0"] = [avg_holdings(l3, s) for s in m["strategy_v0"]]

    # ---------- A. ΔCAGR / ΔMDD 總覽 ----------
    log("===== A. 加 V1 的整體效果 =====")
    log(f"  ΔCAGR 中位數 {m['dCAGR'].median():+.2%}｜平均 {m['dCAGR'].mean():+.2%}"
        f"｜為正比例 {float((m['dCAGR'] > 0).mean()):.1%}")
    log(f"  ΔMDD  中位數 {m['dMDD'].median():+.2%}｜為正(風險下降)比例 "
        f"{float((m['dMDD'] > 0).mean()):.1%}")
    log(f"  持股數：v0 中位 {m['持股數_v0'].median():.1f} → v1 中位 {m['持股數_v1'].median():.1f}")

    # ---------- B. 四象限（論文 4-21） ----------
    q = pd.Series(np.select(
        [(m["dCAGR"] > 0) & (m["dMDD"] > 0), (m["dCAGR"] > 0) & (m["dMDD"] <= 0),
         (m["dCAGR"] <= 0) & (m["dMDD"] > 0)],
        ["① 報酬升 風險降（雙贏）", "② 報酬升 風險升", "③ 報酬降 風險降（換取穩定）"],
        default="④ 報酬降 風險升（雙輸）"))
    log("\n===== B. ΔCAGR × ΔMDD 四象限 =====")
    vc = q.value_counts().reindex(
        ["① 報酬升 風險降（雙贏）", "② 報酬升 風險升",
         "③ 報酬降 風險降（換取穩定）", "④ 報酬降 風險升（雙輸）"]).fillna(0).astype(int)
    for k, v in vc.items():
        log(f"  {k:24s} {v:5d}  {v/len(m):6.1%}")
    m["象限"] = q.values

    # ---------- C. 依 C 分組（論文 4-23） ----------
    byc = m.groupby("C_v0").agg(dCAGR中位=("dCAGR", "median"),
                                 dMDD中位=("dMDD", "median"),
                                 n=("dCAGR", "size")).sort_values("dCAGR中位", ascending=False)
    log("\n===== C. 依 C 分組：加 V1 的 ΔCAGR（前5／後5）=====")
    log(byc.head(5).assign(dCAGR中位=byc.head(5)["dCAGR中位"].map("{:+.2%}".format),
                           dMDD中位=byc.head(5)["dMDD中位"].map("{:+.2%}".format)).to_string())
    log("  …")
    log(byc.tail(5).assign(dCAGR中位=byc.tail(5)["dCAGR中位"].map("{:+.2%}".format),
                           dMDD中位=byc.tail(5)["dMDD中位"].map("{:+.2%}".format)).to_string())

    # ---------- D. 依 F1 因子分組（論文 4-22） ----------
    m["F1因子"] = m["F組合_v0"].str.split("_qb").str[0]
    byf = m.groupby("F1因子").agg(dCAGR中位=("dCAGR", "median"),
                                   dMDD中位=("dMDD", "median"),
                                   n=("dCAGR", "size")).sort_values("dCAGR中位", ascending=False)
    log("\n===== D. 依 F1 因子分組 =====")
    log(byf.assign(dCAGR中位=byf["dCAGR中位"].map("{:+.2%}".format),
                   dMDD中位=byf["dMDD中位"].map("{:+.2%}".format)).to_string())

    # ---------- E. 最終 Candidate（v0/v1 擇優） ----------
    best = pd.concat([
        m[["strategy_v0", "F組合_v0", "C_v0", "CAGR_v0", "max_drawdown_v0",
           "win_ratio_v0", "持股數_v0"]].rename(columns=lambda c: c.replace("_v0", "")).assign(V="v0"),
        m[["strategy_v1", "F組合_v1", "C_v1", "CAGR_v1", "max_drawdown_v1",
           "win_ratio_v1", "持股數_v1"]].rename(columns=lambda c: c.replace("_v1", "")).assign(V="v1"),
    ], ignore_index=True)
    best = best[(best["持股數"] >= MIN_HOLDINGS) & (best["CAGR"] > bench)]
    best = best.sort_values("CAGR", ascending=False)
    best.to_csv(OUT / f"{mkt}_L4{sfx}_final_candidates.csv", index=False, encoding="utf-8-sig")
    log(f"\n===== E. 最終 Candidate Strategy（v0+v1 合併，持股≥{MIN_HOLDINGS} 且 CAGR>基準）=====")
    log(f"  {len(best)} 個（v0 {int((best['V']=='v0').sum())}／v1 {int((best['V']=='v1').sum())}）")
    log("\n  Top 10：")
    log(best[["F組合", "C", "V", "CAGR", "max_drawdown", "持股數"]].head(10).to_string(index=False))

    m.to_csv(OUT / f"{mkt}_L4{sfx}_v0v1_pairs.csv", index=False, encoding="utf-8-sig")

    # ---------- 圖 ----------
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    a = axes[0]
    a.hist(m["dCAGR"], bins=60, color=OI[0], edgecolor="white", linewidth=0.3)
    a.axvline(0, color=OI[1], linestyle="--", linewidth=1.5,
              label=f"0（中位 {m['dCAGR'].median():+.2%}）")
    a.legend(frameon=False); a.set_xlabel("ΔCAGR (v1 − v0)")
    a.set_title("4-20  加 V1 的 ΔCAGR 分布")

    a = axes[1]
    cols = {"① 報酬升 風險降（雙贏）": OI[2], "② 報酬升 風險升": OI[0],
            "③ 報酬降 風險降（換取穩定）": OI[4], "④ 報酬降 風險升（雙輸）": OI[1]}
    for k, col in cols.items():
        s = m[m["象限"] == k]
        if len(s):
            a.scatter(s["dMDD"], s["dCAGR"], s=10, alpha=0.45, color=col,
                      label=f"{k} {len(s)/len(m):.0%}")
    a.axhline(0, color="black", linewidth=1); a.axvline(0, color="black", linewidth=1)
    a.legend(frameon=False, fontsize=8, loc="best")
    a.set_xlabel("ΔMDD（>0 ＝ v1 回撤較淺）"); a.set_ylabel("ΔCAGR")
    a.set_title("4-21  ΔCAGR × ΔMDD 四象限")

    a = axes[2]
    o = byc.sort_values("dCAGR中位", ascending=False)
    data = [m.loc[m["C_v0"] == c, "dCAGR"].values for c in o.index]
    bp = a.boxplot(data, labels=[c.split("_DYN_")[0] if c != "None" else "None" for c in o.index],
                   showmeans=True, patch_artist=True,
                   medianprops=dict(color="black", linewidth=1.2))
    for box in bp["boxes"]:
        box.set(facecolor=OI[5], alpha=0.4)
    a.axhline(0, color=OI[1], linestyle="--", linewidth=1.2)
    a.tick_params(axis="x", labelrotation=70, labelsize=7)
    a.set_ylabel("ΔCAGR"); a.set_title("4-23  依 C 分組的 ΔCAGR")

    fig.suptitle(f"Phase 4  V 估值濾網該不該加？— {mkt}／{args.variant}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / f"{mkt}_L4{sfx}_V_effect.png", bbox_inches="tight")
    plt.close(fig)

    log(f"\n輸出於 {OUT}")


if __name__ == "__main__":
    main()
