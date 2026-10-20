# -*- coding: utf-8 -*-
"""2026-10-20：台美股4組L2正式重跑（含M2+Hot Segment校準+完整5條W2c
caveat，30%上限）定案後的應用層報告圖表。取代`report_charts_1007/`（原始
產生腳本已不在repo，資料也停在10/5、不含這次修正），資料全部來自
`production_runs_mdd_window4.csv`／`_benchmark.csv`（2026-10-20重新產出，
見`_compute_mdd_production_runs.py`），不重新推導任何結論，純視覺化既有
數字。輸出到`_analysis_outputs_applayer/report_charts_1020/`。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._make_report_charts_1020
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
OUT_DIR = APPLAYER / "report_charts_1020"
OUT_DIR.mkdir(exist_ok=True)


def chart1_control0_vs_l2():
    df = pd.read_csv(APPLAYER / "production_runs_mdd_window4.csv")
    d30 = df[df.cap == "30%"].copy()
    labels = ["TW\nbaseline", "TW\nexclude_v1", "US\nbaseline", "US\nexclude_v1"]
    keys = [("TW", "baseline"), ("TW", "exclude_v1"), ("US", "baseline"), ("US", "exclude_v1")]
    control0 = [float(d30[(d30.market == m) & (d30.variant == v) & (d30.arm == "control0")].cum_pct.iloc[0])
               for m, v in keys]
    l2 = [float(d30[(d30.market == m) & (d30.variant == v) & (d30.arm == "L2")].cum_pct.iloc[0])
         for m, v in keys]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    width = 0.35
    x = range(len(labels))
    ax.bar([i - width / 2 for i in x], control0, width, label="control0（不調整）", color="#95a5a6")
    ax.bar([i + width / 2 for i in x], l2, width, label="L2（agent真實決策，30%上限）", color="#2980b9")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("8季累積報酬（%）")
    ax.set_title("台美股4組正式重跑：control0 vs L2（2026-10-20定案，含M2+完整caveat）")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.8)
    for i, (c0, c2) in enumerate(zip(control0, l2)):
        ax.annotate(f"{c0:+.1f}%", (i - width / 2, c0), textcoords="offset points",
                   xytext=(0, 4), ha="center", fontsize=8)
        ax.annotate(f"{c2:+.1f}%", (i + width / 2, c2), textcoords="offset points",
                   xytext=(0, 4), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_control0_vs_l2_final.png")
    plt.close(fig)


def chart2_mdd_vs_benchmark():
    df = pd.read_csv(APPLAYER / "production_runs_mdd_window4.csv")
    d30 = df[(df.cap == "30%") & (df.arm == "L2")].copy()
    labels = ["TW\nbaseline", "TW\nexclude_v1", "US\nbaseline", "US\nexclude_v1"]
    keys = [("TW", "baseline"), ("TW", "exclude_v1"), ("US", "baseline"), ("US", "exclude_v1")]
    mdd = [float(d30[(d30.market == m) & (d30.variant == v)].mdd_pct.iloc[0]) for m, v in keys]
    bench = [-9.89, -9.89, -4.27, -4.27]   # TAIEX/S&P500季頻MDD，同9個日期取樣

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    width = 0.35
    x = range(len(labels))
    ax.bar([i - width / 2 for i in x], mdd, width, label="L2策略MDD（30%上限）", color="#c0392b")
    ax.bar([i + width / 2 for i in x], bench, width, label="大盤季頻MDD（同9個日期取樣）", color="#7f8c8d")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("最大回撤 MDD（%）")
    ax.set_title("L2策略 vs 大盤 最大回撤對照（季頻同口徑，2026-10-20定案）", pad=14)
    ax.set_ylim(min(mdd + bench) * 1.28, 1.5)
    ax.legend(loc="lower right", fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.8)
    for i, (m, b) in enumerate(zip(mdd, bench)):
        ax.annotate(f"{m:+.1f}%", (i - width / 2, m), textcoords="offset points",
                   xytext=(0, -13), ha="center", fontsize=8)
        ax.annotate(f"{b:+.1f}%", (i + width / 2, b), textcoords="offset points",
                   xytext=(0, -13), ha="center", fontsize=8)
    fig.text(0.5, 0.955,
             "只有TW baseline一組MDD優於大盤，其餘3組皆劣於大盤"
             "（MDD單獨看會失真，須同時參考圖01的累積報酬）",
             fontsize=8, color="#444444", ha="center")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "02_mdd_vs_benchmark_final.png")
    plt.close(fig)


def main():
    chart1_control0_vs_l2()
    chart2_mdd_vs_benchmark()
    print(f"2張圖表已寫入 {OUT_DIR}")


if __name__ == "__main__":
    main()
