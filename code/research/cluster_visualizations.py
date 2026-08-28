# -*- coding: utf-8 -*-
"""H-07 · 群的視覺化（老師 2026-08-26 意見，見開發待辦追蹤.md）

老師原話：「你可以看有什麼樣的visualization的方式去表達，8群的話很容易看得出來」
「要去用比較定量的方式去描述說，為什麼他們是不同的群」。

純程式畫圖，**沒有LLM**。四張圖，對 TW_normal／US_normal／XM_normal 各畫一套：

  1. 群×年度報酬熱力圖   （H-06的 cluster_annual_returns.parquet）
  2. 群累積報酬疊圖      （即時從 returns_monthly 重算月度代表序列，複利畫線）
  3. 群間相關熱力圖      （stage3既有的 cluster_corr_matrix_{tree_id}.parquet）
  4. 群因子組成堆疊圖    （strategy_map 的 F1_factor 分布，top4+其他）

用法：
    cd code
    python -m research.cluster_visualizations
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
from . import paths
from .cluster_temporal_profile import DEFAULT_TREES, _member_wide_returns

OUT = paths.ROOT / "_analysis_outputs_hrp_clusters"

OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "axes.grid": True,
    "grid.color": "#DDDDDD", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def fig_annual_heatmap(tree_id: str, log=print) -> None:
    ann = pd.read_parquet(paths.STAGE3 / "cluster_annual_returns.parquet")
    sub = ann[ann.tree_id == tree_id]
    piv = sub.pivot(index="cluster_id", columns="year", values="ret").sort_index()

    fig, ax = plt.subplots(figsize=(0.55 * piv.shape[1] + 2, 0.5 * piv.shape[0] + 1.5))
    vmax = float(np.abs(piv.to_numpy()).max())
    # 台灣股市慣例：紅漲綠跌（跟RdYlGn的西方預設方向相反），故用反轉色階
    im = ax.imshow(piv.to_numpy(), cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=60, fontsize=8)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([f"群{c}" for c in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.0%}", ha="center", va="center", fontsize=6,
                       color="white" if abs(v) > vmax * 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="年度報酬")
    ax.set_title(f"群×年度報酬 — {tree_id}（老師：「某幾年賺、某幾年賠」）")
    fig.tight_layout()
    p = OUT / tree_id / "1_annual_heatmap.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    log(f"  → {p}")


def fig_cumulative_returns(tree_id: str, months_long: pd.DataFrame,
                          assign: pd.DataFrame, log=print) -> None:
    wide = _member_wide_returns(tree_id, months_long, assign)
    a = assign[assign.tree_id == tree_id][[C.PK, "cluster_L1"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    clusters = sorted(a["cluster_L1"].unique())
    for i, cid in enumerate(clusters):
        members = [u for u in a.loc[a.cluster_L1 == cid, C.PK] if u in wide.index]
        rep = wide.loc[members].mean(axis=0).dropna()
        cum = (1 + rep).cumprod() - 1
        ax.plot(cum.index.to_timestamp(), cum.to_numpy(),
               color=OI[i % len(OI)], linewidth=1.6, label=f"群{cid}（{len(members)}檔）")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("累積報酬")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title(f"群累積報酬疊圖 — {tree_id}")
    fig.tight_layout()
    p = OUT / tree_id / "2_cumulative_returns.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    log(f"  → {p}")


def fig_corr_heatmap(tree_id: str, log=print) -> None:
    corr = pd.read_parquet(paths.STAGE3 / f"cluster_corr_matrix_{tree_id}.parquet")
    corr.columns = corr.columns.astype(int)
    corr.index = corr.index.astype(int)
    corr = corr.sort_index()[sorted(corr.columns)]

    fig, ax = plt.subplots(figsize=(0.7 * len(corr) + 2, 0.7 * len(corr) + 1.5))
    im = ax.imshow(corr.to_numpy(), cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr))); ax.set_xticklabels([f"群{c}" for c in corr.columns])
    ax.set_yticks(range(len(corr))); ax.set_yticklabels([f"群{c}" for c in corr.index])
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                   color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="群間相關係數")
    ax.set_title(f"群間相關熱力圖 — {tree_id}（低＝互補程度高，見H-15降級後的定義）")
    fig.tight_layout()
    p = OUT / tree_id / "3_corr_heatmap.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    log(f"  → {p}")


def fig_factor_composition(tree_id: str, assign: pd.DataFrame, log=print) -> None:
    # ⚠️ strategy_map 自己也有 cluster_L1 欄位，但那只投影「策略自己市場的常態樹」
    # （見stage4_strategy_map.py docstring）——XM_normal的群編號完全不同，必須用
    # assign 這份、依 tree_id="XM_normal" 篩出來的才對，不能信 sm 自帶的那欄。
    # 故只取 sm 的 F1_factor，避免跟 a 的 cluster_L1 撞名被 pandas 自動加 _x/_y 尾碼。
    sm = pd.read_parquet(paths.STAGE4 / "strategy_map.parquet")[[C.PK, "F1_factor"]]
    a = assign[assign.tree_id == tree_id][[C.PK, "cluster_L1"]]
    df = sm.merge(a, on=C.PK)

    top_f1 = df["F1_factor"].value_counts().head(4).index.tolist()
    ct = pd.crosstab(df["cluster_L1"], df["F1_factor"].where(
        df["F1_factor"].isin(top_f1), "其他"))
    ct = ct.reindex(columns=top_f1 + ["其他"], fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(0.9 * len(ct) + 3, 5.5))
    bottom = np.zeros(len(ct_pct))
    for i, col in enumerate(ct_pct.columns):
        ax.bar([f"群{c}" for c in ct_pct.index], ct_pct[col].to_numpy(),
              bottom=bottom, color=OI[i % len(OI)], label=col, width=0.6)
        bottom += ct_pct[col].to_numpy()
    ax.set_ylabel("F1因子組成佔比")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(f"群因子組成堆疊圖 — {tree_id}（F1，前4大+其他）")
    fig.tight_layout()
    p = OUT / tree_id / "4_factor_composition.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    log(f"  → {p}")


def build(trees=DEFAULT_TREES, log=print) -> None:
    months_long = pd.read_parquet(paths.STAGE1 / "returns_monthly.parquet")
    assign = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")
    for tree_id in trees:
        log(f"[{tree_id}] 畫圖 …")
        fig_annual_heatmap(tree_id, log)
        fig_cumulative_returns(tree_id, months_long, assign, log)
        fig_corr_heatmap(tree_id, log)
        fig_factor_composition(tree_id, assign, log)
    log(f"\n全部輸出於 {OUT}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.cluster_visualizations")
    ap.add_argument("--trees", nargs="+", default=list(DEFAULT_TREES))
    a = ap.parse_args(argv)
    build(trees=a.trees)
    return 0


if __name__ == "__main__":
    sys.exit(main())
