# -*- coding: utf-8 -*-
"""greedy vs oracle 的兩張圖（2026-11-14）：讀 `_greedy_vs_oracle.py` 的輸出，不重算。

21_名次捕獲率.png：各路徑的名次捕獲率（排除V1、39 選項、全期 2016–25），灰帶＝每季隨機選 1,000 次的 5～95 百分位
22_每季名次.png  ：greedy（回看4季）每一季選中選項的當季百分位，與四個分段的平均
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

from app import _rank_dynamics as RD  # noqa: E402,F401  （沿用中文字型設定）

D = RD.OUT / "greedy_vs_oracle"
SET = "39選項（13類×前3/5/10・平均分）"
GRPS = ["台股・排除V1", "美股・排除V1"]
ORDER = [("global maxima（上帝視角）", "上帝視角：每季都選中當季最好"),
         ("事後最佳固定選項（上帝視角）", "上帝視角：事後挑一個固定到底"),
         ("local greedy（回看4季）", "貪婪法：回看 4 季（主結果）"),
         ("local greedy（回看2季）", "貪婪法：回看 2 季"),
         ("local greedy（回看1季）", "貪婪法：回看 1 季"),
         ("擴張窗 greedy（全部過去）", "貪婪法：用全部過去")]
C_GOD, C_GREEDY, C_MAIN = "#999999", "#56B4E9", "#0072B2"


def fig1(d: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, grp in zip(axes, GRPS):
        x = d[(d.組別 == grp) & (d.選項集合 == SET) & (d.期間 == "全期 2016–25")].set_index("路徑")
        lo, hi = x.loc["local greedy（回看4季）", ["隨機_名次捕獲率_p05", "隨機_名次捕獲率_p95"]]
        ax.axvspan(lo, hi, color="#DDDDDD", zorder=0, label="隨機")
        ax.axvline(0.5, color="#666666", lw=1, ls=":", zorder=1)
        ax.axvline(0.7, color="#D55E00", lw=1.5, ls="--", zorder=1, label="老師的標準：七成")
        for i, (k, lab) in enumerate(ORDER):
            v = x.loc[k, "名次捕獲率(c)"]
            col = C_GOD if "上帝" in k else (C_MAIN if "回看4季" in k else C_GREEDY)
            ax.barh(i, v, color=col, height=0.62, zorder=2)
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=10)
        ax.set_yticks(range(len(ORDER)), [lab for _, lab in ORDER])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.08)
        ax.set_title(grp, fontsize=12)
        ax.set_xlabel("名次捕獲率（1＝每季都選到最好，0＝每季都選到最差）")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("每季換成「近期表現最好」的規則，能達到上帝視角的幾成？（39 個選項、2016–25 共 40 季、扣成本）", fontsize=12)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, ["隨機挑 1,000 次的 5～95% 範圍", "老師的標準：七成"], loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.03))
    fig.text(0.01, 0.005, "資料版本：台股 2026-10-30 舊名單（股價修正、白名單更新前）；美股 2026-10-28 _liq 版", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    out = D / "21_名次捕獲率.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig2(p: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, grp in zip(axes, GRPS):
        x = p[(p.組別 == grp) & (p.選項集合 == SET) & (p.路徑 == "local greedy（回看4季）")].copy()
        x["t"] = pd.to_datetime(x.持有季末)
        cols = ["#D55E00" if v < 0.5 else "#0072B2" for v in x.當季百分位]
        ax.bar(x.t, x.當季百分位 - 0.5, bottom=0.5, width=70, color=cols)
        for lab, a, b in RD.PERIODS[:4]:
            m = x.持有季末.str[:4].between(a, b)
            v = x.loc[m, "當季百分位"].mean()
            t0, t1 = x.loc[m, "t"].min() - pd.Timedelta(days=80), x.loc[m, "t"].max() + pd.Timedelta(days=10)
            ax.plot([t0, t1], [v, v], color="black", lw=2)
            ax.text(t0 + (t1 - t0) / 2, 1.04, f"{lab} 平均 {v:.2f}", ha="center", fontsize=9)
        ax.axhline(0.5, color="#666666", lw=1)
        ax.axhline(0.7, color="#D55E00", lw=1.2, ls="--")
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("當季百分位")
        ax.set_title(f"{grp}：貪婪法（回看 4 季）每一季選中的規則，在當季 39 個選項中排第幾（藍＝前半、橘＝後半；黑線＝分段平均）",
                     fontsize=10.5, loc="left")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = D / "22_每季名次.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    d = pd.read_csv(D / "greedy_vs_oracle.csv")
    p = pd.read_csv(D / "greedy_vs_oracle_picks.csv")
    print(fig1(d))
    print(fig2(p))
