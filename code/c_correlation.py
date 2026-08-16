# -*- coding: utf-8 -*-
"""
20 個 C（動態條件）到底是不是同一個東西？

老師在 2026-08-05 meeting 直接問過：「這些 C **是不是太像了**」。
本檔用兩種互補的方式回答，**都不需要重跑回測**：

  (A) 選股重疊度：兩個 C 在同一個 F 組合下，實際選到的股票有多少是一樣的。
      這是最直接的「像不像」——遮罩層面的重疊，與報酬無關。
      用 Jaccard（交集/聯集），對持股數差異很大的配對比較公允。

  (B) 績效相關：兩個 C 套在同樣一批 F 組合上，得到的 CAGR 增益是否同向。
      這回答的是「換一個 C 會不會有實質差別」。

兩者要一起看：
  重疊高 + 績效相關高 → 真的是同一個東西，可以合併/刪減
  重疊高 + 績效相關低 → 選股像但時機不同（少數股票的差異造成大結果差異）
  重疊低 + 績效相關高 → 選到不同股票卻有同樣效果（可能只是市場 beta）

用法：python c_correlation.py [--market TW] [--variant openSec]
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
from phase3_analyze import parse                       # noqa: E402

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 9,
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

ART = HERE / "results_artifacts"
OUT = HERE.parent / "_analysis_outputs_ccorr"


def log(m):
    print(m, flush=True)


def short(c):
    """C4_ROE_DYN_qmax8 → C4:ROE_qmax8（圖上要放得下）"""
    if c == "None":
        return "None"
    p = c.split("_DYN_")
    if len(p) != 2:
        return c
    head = p[0].split("_", 1)
    return f"{head[0]}:{head[1] if len(head) > 1 else ''}_{p[1]}"


def jaccard_matrix(label, fcombo, cnames):
    """同一個 F 組合下，各 C 的持股遮罩兩兩 Jaccard 重疊。"""
    masks = {}
    for c in cnames:
        p = ART / label / f"{fcombo}__{c}__v0" / "position.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        masks[c] = (d.fillna(0) != 0)
    if len(masks) < 2:
        return None, {}
    ks = list(masks)
    # 對齊到同一組 index/columns，缺的補 False
    idx = sorted(set().union(*[set(m.index) for m in masks.values()]))
    cols = sorted(set().union(*[set(m.columns) for m in masks.values()]))
    A = {k: masks[k].reindex(index=idx, columns=cols, fill_value=False).values for k in ks}
    n = len(ks)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = A[ks[i]], A[ks[j]]
            inter = np.logical_and(a, b).sum()
            union = np.logical_or(a, b).sum()
            M[i, j] = M[j, i] = inter / union if union else np.nan
    return pd.DataFrame(M, index=ks, columns=ks), {k: int(A[k].sum()) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="TW", choices=["TW", "US"])
    ap.add_argument("--variant", default="openSec")
    args = ap.parse_args()
    mkt = args.market
    label = f"{mkt}_L3_M" if args.variant == "strict" else f"{mkt}_L3_{args.variant}_M"
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ART / label / "stats.parquet")
    df[["F組合", "C", "C編號", "C來源"]] = df["strategy"].apply(lambda s: pd.Series(parse(s)))
    base = df[df["C"] == "None"].set_index("F組合")["CAGR"].to_dict()
    df["C增益"] = df["CAGR"] - df["F組合"].map(base)
    log(f"=== C 相關性分析｜{mkt}／{args.variant} ===")
    log(f"讀入 {len(df)} 個策略｜F 組合 {df['F組合'].nunique()}｜C 狀態 {df['C'].nunique()}\n")

    # ---------- (B) 績效相關：C × F組合 的增益矩陣 ----------
    piv = df[df["C"] != "None"].pivot_table(index="F組合", columns="C", values="C增益")
    corr = piv.corr(method="spearman")
    order = df[df["C"] != "None"].groupby("C")["C增益"].median().sort_values(ascending=False).index
    corr = corr.reindex(index=order, columns=order)

    log("===== (B) 績效相關：各 C 的增益在不同 F 組合上是否同向（Spearman）=====")
    off = corr.where(~np.eye(len(corr), dtype=bool))
    log(f"  非對角相關係數：中位 {off.stack().median():.3f}｜"
        f"p90 {off.stack().quantile(.9):.3f}｜最高 {off.stack().max():.3f}")
    hi = off.stack().sort_values(ascending=False)
    hi = hi[hi.index.get_level_values(0) < hi.index.get_level_values(1)]
    log("\n  相關最高的 8 對：")
    for (a, b), v in hi.head(8).items():
        log(f"    {v:+.3f}  {short(a):26s} ↔ {short(b)}")

    # ---------- (A) 選股重疊：挑持股數中位數最接近全體中位的 F 組合當代表 ----------
    hold = df.groupby("F組合")["CAGR"].size()
    rep = sorted(base, key=lambda f: abs(hash(f)))[0] if not len(hold) else None
    # 用「C 覆蓋最完整」的 F 組合當代表，避免挑到缺很多 C 的
    cnt = df[df["C"] != "None"].groupby("F組合")["C"].nunique()
    rep = cnt.sort_values(ascending=False).index[0]
    cnames = sorted(df.loc[df["F組合"] == rep, "C"].unique())
    log(f"\n===== (A) 選股重疊：以 F 組合 `{rep}` 為代表（含 {len(cnames)} 種 C 狀態）=====")
    J, sizes = jaccard_matrix(label, rep, cnames)
    if J is not None:
        J = J.reindex(index=[c for c in order if c in J.index],
                      columns=[c for c in order if c in J.columns])
        offj = J.where(~np.eye(len(J), dtype=bool))
        log(f"  Jaccard 重疊：中位 {offj.stack().median():.3f}｜"
            f"p90 {offj.stack().quantile(.9):.3f}｜最高 {offj.stack().max():.3f}")
        hj = offj.stack().sort_values(ascending=False)
        hj = hj[hj.index.get_level_values(0) < hj.index.get_level_values(1)]
        log("\n  重疊最高的 8 對：")
        for (a, b), v in hj.head(8).items():
            log(f"    {v:.3f}  {short(a):26s} ↔ {short(b)}")
        J.to_csv(OUT / f"{mkt}_{args.variant}_C_jaccard.csv", encoding="utf-8-sig")

    corr.to_csv(OUT / f"{mkt}_{args.variant}_C_corr.csv", encoding="utf-8-sig")

    # ---------- 圖：兩張熱力圖並排 ----------
    mats = [(corr, "(B) 績效相關（Spearman，各 C 的增益）", -1, 1, "RdBu_r")]
    if J is not None:
        mats.append((J, f"(A) 選股重疊（Jaccard，F＝{rep[:28]}…）", 0, 1, "viridis"))
    fig, axes = plt.subplots(1, len(mats), figsize=(9.5 * len(mats), 8.6))
    for ax, (M, title, lo, hi_, cm) in zip(np.atleast_1d(axes), mats):
        im = ax.imshow(M.values, vmin=lo, vmax=hi_, cmap=cm)
        lab = [short(c) for c in M.index]
        ax.set_xticks(range(len(M))); ax.set_xticklabels(lab, rotation=90, fontsize=7)
        ax.set_yticks(range(len(M))); ax.set_yticklabels(lab, fontsize=7)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"20 個 C 是不是太像了？— {mkt}／{args.variant}"
                 f"（依增益中位數由高到低排序）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / f"{mkt}_{args.variant}_C_similarity.png", bbox_inches="tight")
    plt.close(fig)
    log(f"\n輸出於 {OUT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
