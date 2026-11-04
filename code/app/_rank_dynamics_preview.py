# -*- coding: utf-8 -*-
"""老師 11/3 意見 A：規則排名隨時間變化圖（試畫一張給使用者確認格式）。

讀步驟1 每季報酬 `elite_grid_1027/quarterly_{市場}_{池}.parquet`，只取資金比例 30% 的 208 組（13 類×16 種取幾名×分法），
每季季末用「過去 4 季累積報酬（扣成本）」在這 208 組裡排名（1＝最好），畫一類的 16 條線＋該類中位數；control0 當參考線。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from app import rule_selector as RS  # noqa: E402

SRC = RS.ROOT / "_analysis_outputs_applayer" / "elite_grid_1027"
OUT = RS.ROOT / "_analysis_outputs_applayer" / "rank_dynamics"
plt.rcParams.update({"font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False})
SCORE_LAB = {"S6_mcap": "市值"}
SCHEME_LAB = {"equal": "平均", "top1x2": "第一名2倍", "top2x2": "前兩名2倍", "proportional": "照分數比例", "tier": "階梯"}
N_COLOR = {1: "#d62728", 3: "#1f77b4", 5: "#2ca02c", 10: "#9467bd"}


def trailing_rank(market: str, pool: str, sleeve: float = 0.3, look: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = pd.read_parquet(SRC / f"quarterly_{market}_{pool}.parquet")
    c = pd.read_csv(SRC / f"combos_{market}_{pool}.csv")
    keep = c[(c.sleeve == sleeve) | (c.combo == "REF_control0")]
    q = q[q.combo.isin(keep.combo)]
    net = q.assign(r=1 + q.ret_gross - q.cost).pivot_table(index="end", columns="combo", values="r").sort_index()
    trail = net.rolling(look).apply(lambda x: x.prod(), raw=True).dropna(how="all") - 1
    grid = [x for x in trail.columns if x != "REF_control0"]
    rank = trail[grid].rank(axis=1, ascending=False, method="min")
    # control0 放進同一把尺：比它好的組數＋1
    rank["REF_control0"] = (trail[grid].gt(trail["REF_control0"], axis=0)).sum(axis=1) + 1
    return rank, keep.set_index("combo")


def plot_family(market: str, pool: str, score: str, look: int = 4) -> Path:
    rank, meta = trailing_rank(market, pool, look=look)
    n_grid = rank.shape[1] - 1
    fam = [x for x in rank.columns if x.startswith(score + "|")]
    x = pd.to_datetime(rank.index)
    fig, ax = plt.subplots(figsize=(12, 6))
    for cb in fam:
        n = int(meta.loc[cb, "n"])
        ax.plot(x, rank[cb], color=N_COLOR[n], lw=0.9, alpha=0.45)
    ax.plot(x, rank[fam].median(axis=1), color="black", lw=2.8, label=f"「{SCORE_LAB[score]}」16 組的中位數名次")
    ax.plot(x, rank["REF_control0"], color="gray", lw=1.6, ls="--", label="control0（不調整）")
    for n, col in N_COLOR.items():
        ax.plot([], [], color=col, lw=1.5, label=f"取前{n}名（各種分法）")
    ax.axvline(pd.Timestamp("2023-12-31"), color="gray", lw=1, ls=":")
    ax.text(pd.Timestamp("2024-01-31"), n_grid * 0.97, "2024–25\n（實戰評估期）", fontsize=9, color="gray", va="bottom")
    ax.set_ylim(n_grid + 5, -5)
    ax.set_ylabel(f"名次（共 {n_grid} 組，1＝最好）")
    ax.set_xlabel(f"排名時點（季末；依過去 {look} 季累積報酬、扣成本）")
    lab = {"exclude_v1": "排除V1", "baseline": "含V1"}[pool]
    mk = {"TW": "台股", "US": "美股"}[market]
    ax.set_title(f"{mk}・{lab}：「{SCORE_LAB[score]}」排名依據的名次變化（優等生資金 30%）")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"preview_{market}_{pool}_{score}_L{look}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    print(p)
    return p


if __name__ == "__main__":
    plot_family("TW", "exclude_v1", "S6_mcap")
