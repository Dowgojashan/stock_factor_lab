# -*- coding: utf-8 -*-
"""
規模控制分析：把「規模效應」從「因子效應」裡切出來。

為什麼需要（2026-08-11 美股 Phase 3/4 跑完後發現）：
  美股最終候選池排行榜的頭部被單一組合包辦——
    前 10 名   含 REVENUE_qb0（最小營收桶） 100%
    前 50 名                               52%
    全池 3,384                              3.4%
  也就是說整體池子裡它只佔 3.4%，卻幾乎壟斷了排行榜頂端。
  Top 12 全部是同一個 F 組合 `REVENUE_qb0 × PS_qb0` 只換不同的 C。

  這正是「小型股溢酬」的樣子。在跟老師報 57.95% 之前必須先回答：
  **這些估值因子的效果，是真的估值，還是只是買小公司？**

做法（不需重跑任何回測，全部用 Phase 2 既有的 F1×F2 結果）：
  REVENUE 的 3 個桶就是天然的規模控制組——
    REVENUE_qb0 = 最小、qb1 = 中型、qb2 = 最大
  對每個因子桶 X，比較它在三個規模層裡的表現：
    X alone            ← 沒控制規模
    X × REVENUE_qb0    ← 只在小型股裡
    X × REVENUE_qb1    ← 只在中型股裡
    X × REVENUE_qb2    ← 只在大型股裡

  判讀：
    只有 qb0 有效、qb1/qb2 掉光  → **規模混淆**，這個因子其實是在買小公司
    三層都贏基準、斜率平緩        → **真的因子效應**，撐得住規模控制
    qb1/qb2 也有效但 qb0 最強     → 兩者都有，強度隨規模遞減（正常，但要說明）

用法：
  python size_control_analysis.py --market US [--variant all]
"""
import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 沿用各 phaseN_analyze.py 的中文字型設定，否則圖上中文全變方框
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE.parent / "_analysis_outputs_sizecontrol"
SIZE_FACTOR = "REVENUE"          # 規模代理：營收（資料庫沒有市值欄位可直接當因子）
SIZE_LABEL = {f"{SIZE_FACTOR}_qb0": "小型", f"{SIZE_FACTOR}_qb1": "中型",
              f"{SIZE_FACTOR}_qb2": "大型"}


def log(m):
    print(m, flush=True)


def load(market, variant):
    """讀 Phase 2 的完整組合表（含未晉升者——控制組本來就可能不晉升）。"""
    sfx = "" if variant == "strict" else f"_{variant}"
    p = OUT.parent / "_analysis_outputs_phase2" / f"{market}_L2{sfx}_all_combos.csv"
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}；請先跑 phase2_analyze.py --market {market} "
                                f"--variant {variant}")
    d = pd.read_csv(p, encoding="utf-8-sig")
    # strategy 形如 `A_qb0of3__B_qb1of3__None__v0`，取前兩段還原成無序的因子桶對
    parts = d["strategy"].str.split("__", expand=True)
    d["f1"] = parts[0].str.replace(r"of3$", "", regex=True)
    d["f2"] = parts[1].str.replace(r"of3$", "", regex=True)
    return d


def build(market, variant, bench):
    d = load(market, variant)
    solo = d[d["f2"] == "None"].set_index("f1")["CAGR"].to_dict()
    hold = d[d["f2"] == "None"].set_index("f1")["平均持股數"].to_dict()

    # 每個「因子桶 × 規模桶」的表現：配對是無序的，兩個方向都要找
    # 中文欄名不是合法識別字，itertuples 會改成 _N 且位置隨版本變動，故直接 zip 欄位
    pair, phold = {}, {}
    for f1, f2, cagr, h in zip(d["f1"], d["f2"], d["CAGR"], d["平均持股數"]):
        if f2 == "None":
            continue
        pair[(f1, f2)] = pair[(f2, f1)] = cagr
        phold[(f1, f2)] = phold[(f2, f1)] = h

    rows = []
    for x in sorted(solo):
        if x.startswith(SIZE_FACTOR + "_"):
            continue                                  # 規模因子自己不當被檢驗對象
        rec = {"因子桶": x, "單獨": solo[x], "單獨持股": hold.get(x, np.nan)}
        for sb, lab in SIZE_LABEL.items():
            rec[lab] = pair.get((x, sb), np.nan)
            rec[lab + "持股"] = phold.get((x, sb), np.nan)
        rows.append(rec)
    t = pd.DataFrame(rows)

    # 判讀指標
    t["小型超額"] = t["小型"] - bench
    t["中大型最佳"] = t[["中型", "大型"]].max(axis=1)
    t["中大型超額"] = t["中大型最佳"] - bench
    # 規模依賴度：小型比中大型好多少。越大＝越可能只是規模效應
    t["規模落差"] = t["小型"] - t["中大型最佳"]

    def verdict(r):
        if not np.isfinite(r["小型"]) or not np.isfinite(r["中大型最佳"]):
            return "資料不足"
        if r["中大型超額"] > 0.02:
            return "✅ 撐得住規模控制"
        if r["中大型超額"] > 0:
            return "⚠️ 勉強（中大型僅小贏基準）"
        if r["小型超額"] > 0.02:
            return "🔴 規模混淆（只在小型股有效）"
        return "❌ 兩層都沒效"

    t["判定"] = t.apply(verdict, axis=1)
    return t.sort_values("中大型超額", ascending=False)


def plot(t, market, bench, path):
    d = t[np.isfinite(t["小型"]) & np.isfinite(t["中大型最佳"])].copy()
    d = d.sort_values("中大型超額", ascending=True).tail(24)
    fig, ax = plt.subplots(figsize=(11, max(5, 0.34 * len(d))))
    y = np.arange(len(d))
    for col, c, m, lab in [("小型", "#d62728", "o", "× 小型（REVENUE_qb0）"),
                           ("中型", "#ff7f0e", "s", "× 中型（qb1）"),
                           ("大型", "#1f77b4", "^", "× 大型（qb2）"),
                           ("單獨", "#555555", "x", "不控制規模")]:
        ax.scatter(d[col] * 100, y, s=46, c=c, marker=m, label=lab, zorder=3, alpha=.9)
    for i, r in enumerate(d.itertuples()):
        lo = np.nanmin([r.小型, r.中型, r.大型])
        hi = np.nanmax([r.小型, r.中型, r.大型])
        ax.plot([lo * 100, hi * 100], [i, i], color="#cccccc", lw=1.6, zorder=1)
    ax.axvline(bench * 100, color="k", ls="--", lw=1.2,
               label=f"基準 {bench:.2%}")
    ax.set_yticks(y)
    ax.set_yticklabels(d["因子桶"], fontsize=9)
    ax.set_xlabel("CAGR (%)")
    ax.set_title(f"{market}｜規模控制：同一個因子桶在小／中／大型股裡的表現\n"
                 f"紅點遠在右、藍橘點掉到基準線左邊 → 該因子其實是在買小公司", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="規模控制分析（不需重跑回測）")
    ap.add_argument("--market", default="US", choices=["TW", "US"])
    ap.add_argument("--variant", default="all",
                    help="用哪個變體的 Phase 2 組合表；需含 REVENUE，故預設 all")
    ap.add_argument("--bench", default="universe", choices=["universe", "index", "index_tr"])
    args = ap.parse_args()

    from universe_benchmark import get_bench
    bench, bdesc = get_bench(args.market, args.bench)
    OUT.mkdir(parents=True, exist_ok=True)

    log(f"=== 規模控制分析｜{args.market}｜變體 {args.variant} ===")
    log(f"基準 = {bench:.2%}（{bdesc}）｜規模代理 = {SIZE_FACTOR} 三分位\n")

    t = build(args.market, args.variant, bench)

    show = t[["因子桶", "單獨", "小型", "中型", "大型", "中大型超額", "規模落差", "判定"]].copy()
    for c in ["單獨", "小型", "中型", "大型", "中大型超額", "規模落差"]:
        show[c] = show[c].map(lambda v: f"{v:+.2%}" if np.isfinite(v) else "—")
    log("===== 每個因子桶在三個規模層的 CAGR =====")
    log(show.to_string(index=False))

    log("\n===== 判定彙總 =====")
    log(t["判定"].value_counts().to_string())

    csv = OUT / f"{args.market}_{args.variant}_size_control.csv"
    png = OUT / f"{args.market}_{args.variant}_size_control.png"
    t.to_csv(csv, index=False, encoding="utf-8-sig")
    plot(t, args.market, bench, png)
    log(f"\n輸出：{csv}\n      {png}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
