# -*- coding: utf-8 -*-
"""2026-10-27（第四批）：W2c單股上限敏感度分析的逐季版圖表——讀
`cap_sweep_window4_quarterly_2026-10-27.csv`，畫出台美股各一張雙panel圖
（左：逐季報酬率，右：逐季回撤軌跡），6個cap等級用漸層色、大盤用黑色虛線。

用法：cd code; PYTHONIOENCODING=utf-8 python -m app._make_report_charts_1027d
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 11,
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

APPLAYER = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
OUT_DIR = APPLAYER / "report_charts_1027"
CAPS = ["7%", "10%", "15%", "20%", "30%", "40%"]
Q_LABELS = ["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4"]


def chart_market(market: str, title_market: str, out_name: str):
    df = pd.read_csv(APPLAYER / "cap_sweep_window4_quarterly_2026-10-27.csv")
    sub = df[df.market == market]

    cmap = cm.get_cmap("YlOrRd")
    cap_colors = {cap: cmap(0.25 + 0.65 * i / (len(CAPS) - 1)) for i, cap in enumerate(CAPS)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for cap in CAPS:
        s = sub[sub.series == cap].set_index("quarter_label")
        rets = [s.loc[q, "ret_pct"] for q in Q_LABELS]
        ax1.plot(range(len(Q_LABELS)), rets, marker="o", markersize=4,
                 linewidth=1.6, color=cap_colors[cap], label=f"cap={cap}")
    bench_ret = sub[sub.series == "benchmark"].set_index("quarter_label")
    bench_vals = [bench_ret.loc[q, "ret_pct"] for q in Q_LABELS]
    ax1.plot(range(len(Q_LABELS)), bench_vals, marker="^", markersize=6,
             linewidth=2, linestyle="--", color="black", label="大盤")
    ax1.set_xticks(range(len(Q_LABELS)))
    ax1.set_xticklabels(Q_LABELS)
    ax1.axhline(0, color="gray", linewidth=0.6)
    ax1.set_ylabel("季報酬率（%）")
    ax1.set_title(f"{title_market}：逐季報酬率（不同cap）")
    ax1.legend(loc="best", fontsize=8)

    labels9 = ["登記"] + Q_LABELS
    for cap in CAPS:
        s = sub[sub.series == cap].set_index("quarter_label")
        dd = [s.loc[q, "dd_pct"] for q in labels9]
        ax2.plot(range(len(labels9)), dd, marker="o", markersize=4,
                 linewidth=1.6, color=cap_colors[cap], label=f"cap={cap}")
    bench_dd_s = sub[sub.series == "benchmark"].set_index("quarter_label")
    bench_dd = [bench_dd_s.loc[q, "dd_pct"] for q in labels9]
    ax2.plot(range(len(labels9)), bench_dd, marker="^", markersize=6,
             linewidth=2, linestyle="--", color="black", label="大盤")
    ax2.set_xticks(range(len(labels9)))
    ax2.set_xticklabels(labels9)
    ax2.axhline(0, color="gray", linewidth=0.6)
    ax2.set_ylabel("相對歷史高點的回撤（%）")
    ax2.set_title(f"{title_market}：逐季回撤軌跡（不同cap）")
    ax2.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name)
    plt.close(fig)
    print(f"寫入 {OUT_DIR / out_name}")


def main():
    chart_market("TW", "台股", "09_tw_cap_sweep_quarterly.png")
    chart_market("US", "美股", "10_us_cap_sweep_quarterly.png")


if __name__ == "__main__":
    main()
