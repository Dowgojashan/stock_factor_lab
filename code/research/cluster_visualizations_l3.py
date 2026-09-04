# -*- coding: utf-8 -*-
"""H-25c · L3 細粒度群的視覺化（2026-09-04）

背景：H-25 查出「L1 沒有任何高互補配對」是**聚合效應**不是策略沒有互補性——
把粒度降到 L3（成員中位 20~26 檔），XM 樹的跨市場配對有 **80.5%** 達到高互補，
同市場只有 1.1%。這支腳本把那個發現畫出來。

跟 H-07 的 `cluster_visualizations.py`（L1，每棵樹 6/7/3 群）分開的理由：
**群數差兩個數量級**（L3 是 218/235/453 群），L1 那套「每格標數字、每條線標圖例」
的畫法在這裡完全不可讀，必須改成矩陣影像 + 市場分塊的呈現方式。

三張圖 × 三棵 normal 樹：

  1. 群×年度報酬熱力圖  —— 看各群逐年賺賠，以及台美兩塊會不會在不同年份亮
  2. 群累積報酬疊圖      —— 依市場上色 + 各市場中位數粗線，看兩族軌跡是否分開
  3. 群間相關熱力圖      —— 🔴 **本組的核心**：依市場排序後，跨市場區塊會明顯偏冷

⚠️ **只畫成員數 >= MIN_MEMBERS 的群**：L3 最小的群只有 2 檔，相關係數估計雜訊大。
H-25 的穩健性掃描已證實結論不靠小群撐著（成員>=20 時跨市場高互補仍有 62.6%、
同市場只有 0.3%），故這裡直接採用同一個門檻，圖與統計數字口徑一致。

資料來源一律走 `stage3_hrp.rebuild_tree_returns()`（單一事實來源），三張圖用
同一份報酬矩陣算出來，彼此必然一致。

用法：
    cd code
    python -m research.cluster_visualizations_l3
    python -m research.cluster_visualizations_l3 --trees XM_normal
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import contracts as C
from . import freeze, paths
from . import stage3_hrp as S3
from .complementarity_granularity import SHORTLIST_MIN_MEMBERS as MIN_MEMBERS

OUT = paths.ROOT / "_analysis_outputs_hrp_clusters"
DEFAULT_TREES = ("TW_normal", "US_normal", "XM_normal")
LEVEL = "L3"
MKT_COLOR = {"TW": "#0072B2", "US": "#D55E00"}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "axes.grid": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def cluster_monthly(tree_id: str, log=print) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """L3 群代表的月報酬（群×月）、每群市場、每群成員數。

    群代表＝成員報酬簡單平均，跟 `stage3_hrp._cluster_meta_and_corr`／H-06／H-25
    同一套口徑；報酬矩陣走 `rebuild_tree_returns()` 單一事實來源。
    """
    wide = S3.rebuild_tree_returns(tree_id, log=lambda *a, **k: None)
    assign = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")
    assign = assign[assign.tree_id == tree_id].set_index(C.PK)[f"cluster_{LEVEL}"]
    lab = assign.reindex(wide.index)
    wide = wide[lab.notna().to_numpy()]
    lab = lab.dropna().astype(int)

    sizes = lab.value_counts().sort_index()
    keep = sizes[sizes >= MIN_MEMBERS].index
    mkt = (pd.Series(wide.index, index=wide.index).str.split("::").str[0]
           .groupby(lab.to_numpy()).agg(lambda s: s.mode().iloc[0]))

    reps = wide.groupby(lab.to_numpy()).mean().loc[keep]
    log(f"[{tree_id}/{LEVEL}] {len(sizes)} 群中取成員>={MIN_MEMBERS} 的 {len(keep)} 群"
        f"｜{reps.shape[1]} 個月｜市場組成 {mkt.loc[keep].value_counts().to_dict()}")
    return reps, mkt.loc[keep], sizes.loc[keep]


def _order_by_market(mkt: pd.Series, reps: pd.DataFrame) -> list[int]:
    """排序：先分市場，市場內部依「跟自己市場其他群的平均相關」由高到低。

    這樣同市場的高度重複群會聚在一起，跨市場區塊的對比最明顯——
    是為了讓「分散來源是市場邊界」這件事在圖上一眼看得出來，不是為了美觀。
    """
    corr = reps.T.corr()
    order: list[int] = []
    for m in ("TW", "US"):
        ids = [c for c in reps.index if mkt[c] == m]
        if not ids:
            continue
        within = corr.loc[ids, ids].mean(axis=1).sort_values(ascending=False)
        order += within.index.tolist()
    return order


def fig_annual_heatmap(tree_id: str, reps: pd.DataFrame, mkt: pd.Series,
                       order: list[int], log=print) -> None:
    """群×年度報酬。年報酬用月報酬**複利**（同 H-06 口徑），非簡單加總。"""
    yr = reps.T.copy()
    yr.index = pd.PeriodIndex(yr.index, freq="M").year
    ann = (1 + yr).groupby(level=0).prod() - 1          # 逐年複利
    piv = ann.T.loc[order]

    fig, ax = plt.subplots(figsize=(0.42 * piv.shape[1] + 3, max(4, 0.030 * len(piv) + 2)))
    vmax = float(np.nanpercentile(np.abs(piv.to_numpy()), 98))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=60, fontsize=8)
    ax.set_yticks([])

    # 市場分隔線與標籤（群太多無法逐一標，改標市場區塊）
    n_tw = sum(1 for c in order if mkt[c] == "TW")
    if 0 < n_tw < len(order):
        ax.axhline(n_tw - 0.5, color="black", lw=1.8)
    for m, lo, hi in (("TW", 0, n_tw), ("US", n_tw, len(order))):
        if hi > lo:
            ax.text(-0.8, (lo + hi) / 2 - 0.5, m, ha="right", va="center",
                    fontsize=12, fontweight="bold", color=MKT_COLOR[m])
    ax.set_title(f"{tree_id} · L3 群 × 年度報酬（{len(order)} 群，成員>={MIN_MEMBERS}）\n"
                 f"每列一個 L3 群，依市場分塊；綠=賺 紅=賠", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="年度報酬")
    _save(fig, tree_id, "L3_1_年度報酬熱力圖.png", log)


def fig_cumulative(tree_id: str, reps: pd.DataFrame, mkt: pd.Series, log=print) -> None:
    """累積報酬。群太多無法逐條標圖例，改用市場上色 + 各市場中位數粗線。"""
    cum = (1 + reps.T).cumprod()
    x = cum.index.to_timestamp() if hasattr(cum.index, "to_timestamp") else cum.index

    fig, ax = plt.subplots(figsize=(11, 6))
    for m in ("TW", "US"):
        ids = [c for c in reps.index if mkt[c] == m]
        if not ids:
            continue
        for c in ids:
            ax.plot(x, cum[c], color=MKT_COLOR[m], lw=0.4, alpha=0.10)
        ax.plot(x, cum[ids].median(axis=1), color=MKT_COLOR[m], lw=2.6,
                label=f"{m} 中位數（{len(ids)} 群）")
    ax.set_yscale("log")
    ax.set_ylabel("累積淨值（對數刻度，起始=1）")
    ax.grid(True, which="both", color="#DDDDDD", lw=0.5)
    ax.legend(frameon=False)
    ax.set_title(f"{tree_id} · L3 群累積報酬（{len(reps)} 群，成員>={MIN_MEMBERS}）\n"
                 f"細線=個別群，粗線=各市場中位數", fontsize=11)
    _save(fig, tree_id, "L3_2_累積報酬.png", log)


def fig_corr_heatmap(tree_id: str, reps: pd.DataFrame, mkt: pd.Series,
                     order: list[int], log=print) -> None:
    """🔴 核心圖：群間相關矩陣，依市場排序後看區塊結構。

    高互補門檻（`COMPLEMENTARITY_CUTS["高"]`=0.5）畫成 colormap 的分界，
    低於門檻的格子偏藍——跨市場區塊若整片偏藍，就是「免費午餐只在跨市場」
    最直接的視覺證據。
    """
    corr = reps.T.corr().loc[order, order]
    cut = C.COMPLEMENTARITY_CUTS["高"]

    fig, ax = plt.subplots(figsize=(9, 7.6))
    im = ax.imshow(corr.to_numpy(), cmap="RdYlBu_r", vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])

    n_tw = sum(1 for c in order if mkt[c] == "TW")
    if 0 < n_tw < len(order):
        ax.axhline(n_tw - 0.5, color="black", lw=2.0)
        ax.axvline(n_tw - 0.5, color="black", lw=2.0)
        # 只標左側與底部，不標上方——上方會壓到標題
        for m, lo, hi in (("TW", 0, n_tw), ("US", n_tw, len(order))):
            ax.text(-len(order) * 0.04, (lo + hi) / 2, m, ha="right", va="center",
                    fontsize=13, fontweight="bold", color=MKT_COLOR[m])
            ax.text((lo + hi) / 2, len(order) * 1.02, m, ha="center", va="top",
                    fontsize=13, fontweight="bold", color=MKT_COLOR[m])

    # 實際數字寫進標題，避免圖與統計脫節
    iu = np.triu_indices(len(order), 1)
    v = corr.to_numpy()[iu]
    same = np.array([mkt[order[i]] == mkt[order[j]] for i, j in zip(*iu)])
    txt = f"同市場 {len(v[same]):,} 對 高互補 {(v[same] < cut).mean():.1%}"
    if (~same).any():
        txt += f"｜跨市場 {len(v[~same]):,} 對 高互補 {(v[~same] < cut).mean():.1%}"
    ax.set_title(f"{tree_id} · L3 群間相關矩陣（{len(order)} 群，依市場排序）\n{txt}",
                 fontsize=11, pad=14)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    # labelpad 要推到門檻註記文字之外，否則兩者會疊在一起（單一市場樹尤其明顯，
    # 因為整條 colorbar 都是紅的、註記剛好落在標籤位置）
    cb.set_label("相關係數", labelpad=86)
    cb.ax.axhline(cut, color="black", lw=1.8)
    cb.ax.annotate(f"高互補門檻 {cut}", xy=(1.0, cut), xycoords=("axes fraction", "data"),
                   xytext=(6, 0), textcoords="offset points",
                   va="center", ha="left", fontsize=8, fontweight="bold")
    _save(fig, tree_id, "L3_3_群間相關熱力圖.png", log)


def _save(fig, tree_id: str, name: str, log=print) -> None:
    d = OUT / f"{tree_id}_L3"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    log(f"  → {p}")


def build(trees=DEFAULT_TREES, log=print) -> None:
    # ⚠️ 2026-09-04 code review 補上：圖是要進論文的下游產物，同樣適用 DD-08
    # 「消費前先驗上游」。這些圖的標題直接寫著統計數字（如「跨市場高互補 62.6%」），
    # 若上游被改動而圖沿用舊資料，圖與統計會靜默不一致。
    # （H-07 的 `cluster_visualizations.py`（L1 版）也缺同樣的驗證，屬既存缺口。）
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE1 / "_marks")
    freeze.verify_inputs(paths.STAGE3)
    for tree_id in trees:
        reps, mkt, sizes = cluster_monthly(tree_id, log)
        order = _order_by_market(mkt, reps)
        fig_annual_heatmap(tree_id, reps, mkt, order, log)
        fig_cumulative(tree_id, reps, mkt, log)
        fig_corr_heatmap(tree_id, reps, mkt, order, log)
        log("")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.cluster_visualizations_l3")
    ap.add_argument("--trees", nargs="+", default=list(DEFAULT_TREES))
    a = ap.parse_args(argv)
    build(tuple(a.trees))
    return 0


if __name__ == "__main__":
    sys.exit(main())
