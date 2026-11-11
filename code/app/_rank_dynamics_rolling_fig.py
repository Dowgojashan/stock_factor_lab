# -*- coding: utf-8 -*-
"""老師 11/3 意見 (B) 的圖：每 3／6 個月滾動重挑規則（2026-11-09 補畫；計算沿用 `_rank_dynamics.py` 寫死的規則）。

每組（市場×池）一張圖，畫「各類代表、回看 2 季（半年）」的兩種重挑頻率：
  上：累積報酬（未扣成本）——滾動重挑（每 3 個月／每 6 個月）vs control0 vs 大盤 vs 每季隨機挑（1,000 次的 5～95 百分位帶）
  下：每一季實際持有的是哪一類規則（色帶），兩條分別對應兩種重挑頻率
輸出：`_analysis_outputs_applayer/rank_dynamics/{市場}_{池}/20_滾動重挑_回看2季.png`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from app import _rank_dynamics as RD  # noqa: E402

L = 2
TYPE_OF = {"S1": "票數", "S2": "因子分數", "S3": "追強", "S4": "押回彈", "S5": "混合", "S6": "市值", "REF": "control0（不調整）"}
TYPE_COLOR = {"票數": "#E69F00", "因子分數": "#56B4E9", "追強": "#D55E00", "押回彈": "#009E73",
              "混合": "#CC79A7", "市值": "#0072B2", "control0（不調整）": "#999999"}


def type_of(combo: str) -> str:
    return TYPE_OF[combo.split("|")[0].split("_")[0]]


def make(market: str, pool: str, suf: str) -> Path:
    g, n, meta = RD.load(market, pool, suf)
    ends = pd.to_datetime(g.index[RD.EVAL_START:])
    G = g.iloc[RD.EVAL_START:]
    bench = RD.bench_quarterly(market, g.index).iloc[RD.EVAL_START:]
    arms = [f + RD.REP for f in RD.FAM if f != "REF_control0"] + ["REF_control0"]
    col = {c: i for i, c in enumerate(G.columns)}
    Gv = G.values
    rng = np.random.default_rng(20261103)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), height_ratios=[3, 1.1], sharex=True)
    # 隨機帶（每 3 個月重挑）
    ai = np.array([col[a] for a in arms])
    rp = RD.random_paths(arms, 1, rng)
    rc = np.cumprod(1 + Gv[np.arange(40)[None, :], ai[rp]], axis=1) - 1
    ax.fill_between(ends, np.quantile(rc, .05, axis=0) * 100, np.quantile(rc, .95, axis=0) * 100,
                    color="#BBBBBB", alpha=0.35, lw=0, label="每季隨機挑一類（1,000 次的 5～95 百分位）")
    held = {}
    for R, ls, lab in ((1, "-", "滾動重挑：每 3 個月"), (2, "--", "滾動重挑：每 6 個月")):
        h = RD.rolling(n, arms, L, R)
        held[R] = h
        r = Gv[np.arange(40), [col[x] for x in h]]
        ax.plot(ends, (np.cumprod(1 + r) - 1) * 100, color="black", lw=2.2, ls=ls, label=f"{lab}（回看 {L} 季）")
    ax.plot(ends, (np.cumprod(1 + G["REF_control0"].values) - 1) * 100, color="#0072B2", lw=1.6, label="control0（不調整）")
    ax.plot(ends, (np.cumprod(1 + bench.values) - 1) * 100, color="#D55E00", lw=1.6, label="大盤（含息指數）")
    ax.axvline(pd.Timestamp("2023-12-31"), color="gray", lw=1, ls=":")
    ax.set_ylabel("累積報酬（%，未扣成本）")
    ax.set_title(f"{RD.GLAB[market]}・{RD.GLAB[pool]}：每 3／6 個月用過去半年表現重挑規則（13 類各取「前3名・平均分・資金30%」＋control0）")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)

    # 下：每季持有的規則類型
    w = pd.Timedelta(days=88)
    for y, R in ((1, 1), (0, 2)):
        for e, h in zip(ends, held[R]):
            ax2.barh(y, w, left=e - w, height=0.7, color=TYPE_COLOR[type_of(h)], edgecolor="white", linewidth=0.6)
    ax2.set_yticks([1, 0], ["每 3 個月", "每 6 個月"])
    ax2.set_title("每一季實際持有的規則類型", fontsize=10)
    ax2.axvline(pd.Timestamp("2023-12-31"), color="gray", lw=1, ls=":")
    ax2.legend(handles=[Patch(color=c, label=k) for k, c in TYPE_COLOR.items()], ncol=7, fontsize=8,
               loc="upper center", bbox_to_anchor=(0.5, -0.35), frameon=False)
    fig.tight_layout()
    out = RD.OUT / f"{market}_{pool}" / f"20_滾動重挑_回看{L}季.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(out)
    return out


if __name__ == "__main__":
    for grp in RD.GROUPS:
        make(*grp)
