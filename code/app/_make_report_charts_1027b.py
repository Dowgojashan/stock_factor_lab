# -*- coding: utf-8 -*-
"""2026-10-27（第二批）：W2c單股上限敏感度／RepSwap-CoverageTilt強制驗證／
Active Share三項分析的圖表，資料來源為當天已產出的CSV，不重新計算，純
視覺化。風格跟`_make_report_charts_1027.py`一致。

用法：cd code; PYTHONIOENCODING=utf-8 python -m app._make_report_charts_1027b
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 11,
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

APPLAYER = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
OUT_DIR = APPLAYER / "report_charts_1027"


def chart_cap_sweep():
    df = pd.read_csv(APPLAYER / "cap_sweep_window4_2026-10-27.csv")
    bench = {"TW": 70.56, "US": 47.38}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)
    for ax, market in zip(axes, ["TW", "US"]):
        sub = df[df.market == market]
        caps = sub["cap"].tolist()
        cum = sub["cum_pct"].tolist()
        bars = ax.bar(caps, cum, color="#2980b9")
        for bar, v in zip(bars, cum):
            if v >= bench[market]:
                bar.set_color("#c0652b")
        ax.axhline(bench[market], color="black", linestyle="--", linewidth=1.2,
                  label=f"{'TAIEX' if market=='TW' else 'S&P500'}（{bench[market]:+.2f}%）")
        for x, v in zip(caps, cum):
            ax.annotate(f"{v:+.1f}%", (x, v), textcoords="offset points",
                       xytext=(0, 4), ha="center", fontsize=8.5)
        ax.set_title(f"{'台股' if market=='TW' else '美股'}：單股上限敏感度（window4，強制套用W2c）")
        ax.set_xlabel("單股上限")
        ax.set_ylabel("8季累積報酬（%）")
        ax.legend(loc="upper left", fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_cap_sweep_both_markets.png")
    plt.close(fig)


def chart_repswap_coveragetilt():
    df = pd.read_csv(APPLAYER / "coveragetilt_repswap_forced_both_markets_2026-10-27.csv")
    labels = ["control0\n（不調整）", "CoverageTilt\n強制", "RepSwap\n強制"]
    variants = ["control0", "coverage_tilt", "rep_swap"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, market in zip(axes, ["TW", "US"]):
        sub = df[df.market == market].set_index("variant").loc[variants]
        colors = ["#95a5a6", "#27ae60", "#c0392b"]
        bars = ax.bar(labels, sub["cum_pct"], color=colors)
        base = sub.loc["control0", "cum_pct"]
        for i, (bar, v) in enumerate(zip(bars, sub["cum_pct"])):
            delta = v - base
            label = f"{v:+.2f}%" if i == 0 else f"{v:+.2f}%\n（{delta:+.2f}pp）"
            ax.annotate(label, (bar.get_x() + bar.get_width() / 2, v),
                       textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8.5)
        ax.set_title(f"{'台股' if market=='TW' else '美股'}：Coverage Tilt／RepSwap強制套用（window4）")
        ax.set_ylabel("8季累積報酬（%）")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_repswap_coveragetilt_both_markets.png")
    plt.close(fig)


def chart_active_share():
    df = pd.read_csv(APPLAYER / "active_share_both_markets_2026-10-27.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, market in zip(axes, ["TW", "US"]):
        sub = df[df.market == market].sort_values("quarter_end")
        x = range(len(sub))
        labels = [q[2:7].replace("-", "Q") for q in sub["quarter_end"]]
        ax.plot(x, sub["active_share_control0"] * 100, marker="o", color="#95a5a6",
               linestyle="--", label="control0（等權）")
        ax.plot(x, sub["active_share_l2"] * 100, marker="s", color="#2980b9",
               label="真實L2持股")
        for xi, applied in zip(x, sub["w2c_applied"]):
            if applied:
                ax.axvspan(xi - 0.3, xi + 0.3, color="#f39c12", alpha=0.15)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)
        ax.set_title(f"{'台股' if market=='TW' else '美股'}：Active Share（橘底＝真的套用W2c那幾季）")
        ax.set_ylabel("Active Share（%）")
        ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_active_share_both_markets.png")
    plt.close(fig)


def main():
    chart_cap_sweep()
    chart_repswap_coveragetilt()
    chart_active_share()
    print(f"3張圖表已寫入 {OUT_DIR}")


if __name__ == "__main__":
    main()
