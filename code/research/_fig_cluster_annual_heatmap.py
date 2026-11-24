# -*- coding: utf-8 -*-
"""各群逐年報酬熱圖（2026-11-22，使用者授權，經寫作 session 轉達；論文 4-3 用）。

讀 `_frozen/stage3/cluster_annual_returns.parquet`（L1），台股、美股各畫一張，
輸出到 `_analysis_outputs_robustness/{tree}_cluster_annual_returns_heatmap.png`。
格子內標報酬（%），配色以 0 為中心的紅藍雙色。同時印出「所有群同號」的年份統計，
供論文核對。純讀凍結檔、不建樹、不呼叫 LLM。

用法（cwd=code/）：python -m research._fig_cluster_annual_heatmap
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from . import paths

OUT = paths.ROOT / "_analysis_outputs_robustness"
LABEL = {"TW_normal": "台股", "US_normal": "美股"}

plt.rcParams.update({"font.sans-serif": ["Microsoft JhengHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "savefig.dpi": 160})


def main() -> int:
    df = pd.read_parquet(paths.STAGE3 / "cluster_annual_returns.parquet")
    df = df[df.level == "L1"]
    for tree, lab in LABEL.items():
        w = df[df.tree_id == tree].pivot(index="cluster_id", columns="year", values="ret").sort_index()
        if w.isna().any().any():
            raise ValueError(f"{tree} 有缺值")
        all_neg = [y for y in w.columns if (w[y] < 0).all()]
        all_pos = [y for y in w.columns if (w[y] > 0).all()]
        n_same = len(all_neg) + len(all_pos)
        print(f"[{lab}] {w.shape[0]} 群 × {w.shape[1]} 年（{w.columns.min()}–{w.columns.max()}）"
              f"｜所有群同號 {n_same}/{w.shape[1]} 年｜全負：{all_neg}｜全正 {len(all_pos)} 年")
        mixed = [y for y in w.columns if y not in all_neg + all_pos]
        print(f"      不同號的年份：{mixed}")

        v = float(np.abs(w.values).max())
        fig, ax = plt.subplots(figsize=(0.62 * w.shape[1] + 1.8, 0.55 * w.shape[0] + 1.6))
        im = ax.imshow(w.values, cmap="RdBu_r", norm=TwoSlopeNorm(0, -v, v), aspect="auto")
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                x = w.values[i, j]
                ax.text(j, i, f"{x*100:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(x) > 0.55 * v else "black")
        ax.set_xticks(range(w.shape[1]), w.columns, rotation=60, fontsize=8)
        ax.set_yticks(range(w.shape[0]), [f"群{c}" for c in w.index], fontsize=9)
        for j, y in enumerate(w.columns):
            if y in all_neg:
                ax.add_patch(plt.Rectangle((j - .5, -.5), 1, w.shape[0], fill=False, lw=1.6, ec="black"))
        ax.set_title(f"{lab} L1 各群逐年報酬（%）｜所有群同號 {n_same}/{w.shape[1]} 年；"
                     f"黑框＝所有群皆虧損的年份", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, format=lambda x, _: f"{x*100:.0f}%")
        p = OUT / f"{tree}_cluster_annual_returns_heatmap.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print(f"   → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
