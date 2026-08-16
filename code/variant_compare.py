# -*- coding: utf-8 -*-
"""
三方變體對照：strict / relaxed / all，回答「Phase 1 的線性檢定篩選有沒有用？」

  strict  ：只有 Phase 1「✅ 過關」的 5 個因子可當 primary
  relaxed ：再加上「⚠️ 邊際」與「⚠️ 只取極端桶」→ 12 個
  all     ：19 個因子全下 ＝ **完全不做 Phase 1 篩選**（真正的對照組）

三種可能結果的意義：
  被淘汰的因子組不出好策略        → ✅ 篩選有效
  能組出跟現在差不多的            → ⚠️ 篩選沒壞事但也沒加值
  能組出更好的                    → ❌ 篩選砍掉了好東西，判準要檢討

只讀既有分析結果，不重跑任何回測。

用法：python variant_compare.py [--market TW]
"""
import sys
import argparse
import warnings
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_variants as pv                    # noqa: E402
from universe_benchmark import get_bench       # noqa: E402

P2 = HERE.parent / "_analysis_outputs_phase2"
P4 = HERE.parent / "_analysis_outputs_phase4"
OUT = HERE.parent / "_analysis_outputs_variants"

OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "axes.grid": True,
    "grid.color": "#DDDDDD", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
VARIANTS = ["strict", "openSec", "relaxed", "all"]


def log(m):
    print(m, flush=True)


def sfx(v):
    return "" if v == "strict" else f"_{v}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="TW", choices=["TW", "US"])
    args = ap.parse_args()
    mkt = args.market
    OUT.mkdir(parents=True, exist_ok=True)
    bench, bdesc = get_bench(mkt, "universe")
    log(f"基準 = {bench:.2%}（{bdesc}）\n")

    # pv.ELIMINATED 這個模組層常數在「改為依市場讀取 Phase 1 判定」的重構中移除了，
    # 必須改用 pv.groups(市場)；同理 pv.get() 也要帶 market，否則美股會拿台股的判定。
    elim = set(pv.groups(mkt)["ELIMINATED"])
    rows, finals = [], {}
    for v in VARIANTS:
        V = pv.get(v, mkt)
        solo = pd.read_csv(P2 / f"{mkt}_L2{sfx(v)}_solo_buckets.csv", encoding="utf-8-sig")
        comb = pd.read_csv(P2 / f"{mkt}_L2{sfx(v)}_all_combos.csv", encoding="utf-8-sig")
        fin_p = P4 / f"{mkt}_L4{sfx(v)}_final_candidates.csv"
        if not fin_p.exists():
            log(f"⚠️ {v} 的 Phase 4 結果尚未產生（{fin_p.name}），略過")
            continue
        fin = pd.read_csv(fin_p, encoding="utf-8-sig")
        finals[v] = fin
        # 這個組合裡有沒有用到 Phase 1 被淘汰的因子
        fin["含淘汰因子"] = fin["F組合"].map(lambda s: any(e + "_qb" in s for e in elim))
        rows.append({
            "變體": v,
            "因子數": len(V["factors"]),
            "primary池": len(V["primary"]),
            "可當primary的桶": int(solo["可當primary"].sum()),
            "Phase2晉升": int(comb["晉升"].sum()),
            "最終候選": len(fin),
            "CAGR中位": round(float(fin["CAGR"].median()), 4),
            "CAGR_p90": round(float(fin["CAGR"].quantile(.9)), 4),
            "CAGR最高": round(float(fin["CAGR"].max()), 4),
            "MDD中位": round(float(fin["max_drawdown"].median()), 4),
            "含淘汰因子的候選": int(fin["含淘汰因子"].sum()),
        })

    t = pd.DataFrame(rows)
    t.to_csv(OUT / f"{mkt}_variant_compare.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    log("===== 三方對照總表 =====")
    log(t.to_string(index=False))

    if "strict" in finals:
        base = finals["strict"]["CAGR"]
        log("\n===== 相對 strict 的變化 =====")
        for v in VARIANTS:
            if v not in finals or v == "strict":
                continue
            f = finals[v]["CAGR"]
            # 注意：f-string 沒有 "pp" 這個格式規格，百分點要自己乘 100 再標單位
            log(f"  {v:8s} 候選數 {len(f):5d}（{len(f)/len(base)-1:+.0%}）｜"
                f"中位 {f.median():.2%}（{(f.median()-base.median())*100:+.2f}pp）｜"
                f"p90 {f.quantile(.9):.2%}（{(f.quantile(.9)-base.quantile(.9))*100:+.2f}pp）｜"
                f"最高 {f.max():.2%}（{(f.max()-base.max())*100:+.2f}pp）")

    # 被淘汰的因子到底行不行（只有 all 有這個資訊）
    if "all" in finals:
        f = finals["all"]
        g = f.groupby("含淘汰因子")["CAGR"].agg(["count", "median", "max"]).round(4)
        g.index = g.index.map({False: "不含淘汰因子", True: "含淘汰因子"})
        log("\n===== all 變體：含/不含 Phase 1 被淘汰因子的候選比較 =====")
        log(g.to_string())
        used = set()
        for s in f.loc[f["含淘汰因子"], "F組合"]:
            for e in elim:
                if e + "_qb" in s:
                    used.add(e)
        log(f"\n  7 個被淘汰的因子中，實際進入最終候選的：{sorted(used) or '（無）'}")
        log(f"  完全沒進入的：{sorted(elim - used)}")

    # 圖
    if len(finals) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        a = axes[0]
        data = [finals[v]["CAGR"].values for v in VARIANTS if v in finals]
        labs = [f"{v}\n(n={len(finals[v]):,})" for v in VARIANTS if v in finals]
        bp = a.boxplot(data, labels=labs, showmeans=True, patch_artist=True,
                       medianprops=dict(color="black", linewidth=1.4))
        for box, c in zip(bp["boxes"], [OI[2], OI[3], OI[0], OI[1]]):
            box.set(facecolor=c, alpha=0.4)
        a.axhline(bench, color=OI[1], linestyle="--", linewidth=1.2, label=f"基準 {bench:.2%}")
        a.legend(frameon=False); a.set_ylabel("CAGR")
        a.set_title("最終候選策略的 CAGR 分布")

        a = axes[1]
        for v, c in zip(VARIANTS, [OI[2], OI[3], OI[0], OI[1]]):
            if v in finals:
                s = finals[v]["CAGR"].sort_values(ascending=False).reset_index(drop=True)
                a.plot(range(1, min(300, len(s)) + 1), s.head(300).values,
                       color=c, linewidth=1.8, label=f"{v}（最高 {s.iloc[0]:.1%}）")
        a.legend(frameon=False); a.set_xlabel("排名（前 300 名）"); a.set_ylabel("CAGR")
        a.set_title("排名前 300 的策略：放寬有沒有找到更好的？")

        fig.suptitle(f"變體對照：Phase 1 篩選有沒有用？— {mkt}（基準 {bench:.2%}）", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(OUT / f"{mkt}_variant_compare.png", bbox_inches="tight")
        plt.close(fig)

    log(f"\n輸出於 {OUT}")


if __name__ == "__main__":
    main()
