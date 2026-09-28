# -*- coding: utf-8 -*-
"""給10/7報告用的6張圖表，資料全部來自本session已經驗證過的CSV／報告數字，
不重新推導任何結論，純視覺化既有結果。字型設定沿用`phase1_analyze.py`已有的
中文字型慣例（Microsoft JhengHei等），輸出到`_analysis_outputs_applayer/report_charts/`。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._make_report_charts
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
OUT_DIR = APPLAYER / "report_charts"
OUT_DIR.mkdir(exist_ok=True)


def chart1_coverage_tilt_beta():
    df = pd.read_csv(APPLAYER / "coverage_tilt_window4_beta_extended.csv")
    df = df.sort_values("beta")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["beta"], df["ann_excess"] * 100, marker="o", color="#c0392b", linewidth=2)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xscale("symlog")
    ax.set_xlabel("β（傾斜幅度，對數座標）")
    ax.set_ylabel("年化超額報酬 vs 等權大盤（%）")
    ax.set_title("Coverage Tilt：β越大表現越好，但天花板很低（window4真實危機期）")
    for x, y in zip(df["beta"], df["ann_excess"] * 100):
        if x in (0, 10, 500):
            ax.annotate(f"{y:+.2f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.axvspan(0.5, 10, color="#f1c40f", alpha=0.15)
    ax.text(2.2, -0.15, "老師講的\n「小幅有界」範圍", fontsize=8, color="#8a6d00", ha="left", va="top")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_coverage_tilt_beta.png")
    plt.close(fig)


def chart2_anchored_vs_rolling_largecap():
    anc = pd.read_csv(APPLAYER / "anchored_schemeA_composition_TW.csv")
    roll = pd.read_csv(APPLAYER / "rolling_is72oos24_TW.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    width = 0.35
    x = anc["window_no"]
    ax.bar(x - width / 2, anc["frac_top10pct_mktcap"] * 100, width, label="anchored", color="#2980b9")
    ax.bar(x + width / 2, roll["frac_top10pct_mktcap"] * 100, width, label="rolling", color="#e67e22")
    ax.set_xlabel("窗次")
    ax.set_ylabel("候選池市值前10%大型股佔比（%）")
    ax.set_title("anchored vs rolling：大型股佔比逐窗對比（5戰5敗，方向一致）")
    ax.set_xticks(list(x))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_anchored_vs_rolling_largecap.png")
    plt.close(fig)


def chart3_v1_ratio_by_window():
    anc_v1 = [0.30, 0.2667, 0.3333, 0.3667, 0.3667, 0.3333]   # fixed k_mode，scheme A，跟chart2同一套方法論
    rolling_v1 = [0.30, 0.3333, 0.30, 0.4333, 0.4333, 0.4333]  # fixed k_mode，rolling_6_2，2026-09-28驗證
    windows = list(range(1, 7))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(windows, [v * 100 for v in anc_v1], marker="o", label="anchored", color="#2980b9", linewidth=2)
    ax.plot(windows, [v * 100 for v in rolling_v1], marker="s", label="rolling", color="#e67e22", linewidth=2)
    ax.axhline(37.6, color="gray", linestyle="--", linewidth=0.8, label="候選池母體基準（37.6%）")
    ax.set_xlabel("窗次")
    ax.set_ylabel("代表策略V1（估值濾網）佔比（%）")
    ax.set_title("rolling固定用6年短訓練窗，6窗中4窗V1佔比高於anchored", fontsize=10.5)
    ax.set_xticks(windows)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_v1_ratio_by_window.png")
    plt.close(fig)


def chart4_exclude_v1_matrix():
    rows = [
        ("anchored\n保留V1", 7.22, 24.70),
        ("anchored\n排除V1", 10.78, 32.94),
        ("rolling\n保留V1", 5.13, 20.19),
        ("rolling\n排除V1", 5.64, 25.72),
    ]
    labels = [r[0] for r in rows]
    control0 = [r[1] for r in rows]
    l2 = [r[2] for r in rows]
    x = range(len(rows))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors0 = ["#95a5a6", "#27ae60", "#95a5a6", "#27ae60"]
    colors2 = ["#7f8c8d", "#1e8449", "#7f8c8d", "#1e8449"]
    ax.bar([i - width / 2 for i in x], control0, width, label="control0（不調整）", color=colors0)
    ax.bar([i + width / 2 for i in x], l2, width, label="L2（agent真實決策）", color=colors2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("8季累積報酬（%）")
    ax.set_title("排除V1：真實production管線驗證，四組數字全部同方向改善")
    ax.legend()
    for i, (c0, c2) in enumerate(zip(control0, l2)):
        ax.annotate(f"{c0:+.2f}%", (i - width / 2, c0), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
        ax.annotate(f"{c2:+.2f}%", (i + width / 2, c2), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_exclude_v1_matrix.png")
    plt.close(fig)


def chart5_monitoring_frequency():
    w = pd.read_csv(APPLAYER / "monitoring_freq_prod_weekly.csv", parse_dates=["as_of"])
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(w["as_of"], w["cumulative_deviation"] * 100, color="#34495e", linewidth=1.5, label="週頻實際偏離值")
    ax.axhline(1.86, color="#c0392b", linestyle="--", linewidth=1, label="p90門檻（TRIGGERED）＝1.86pp")
    ax.axhline(1.2, color="#e67e22", linestyle="--", linewidth=1, label="p75門檻（OBSERVING）＝1.2pp")

    for label, ts, color in [("季頻／月頻首次觸發", "2024-12-31", "#2980b9"),
                             ("週頻首次觸發（早18天）", "2024-12-13", "#27ae60")]:
        ax.axvline(pd.Timestamp(ts), color=color, linestyle=":", linewidth=1.5)
        ax.annotate(label, (pd.Timestamp(ts), ax.get_ylim()[1] * 0.9), rotation=90,
                   fontsize=8, color=color, ha="right", va="top")

    ax.set_ylabel("累計偏離值（pp）")
    ax.set_title("M1-D監控頻率：週頻比季/月頻早18天抓到2024-2025集中度異常")
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_monitoring_frequency_m1d.png")
    plt.close(fig)


def chart6_hotsample_frequency():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    labels = ["anchored\n(2024-2025真OOS)", "rolling\n(2013-2025全歷史)"]
    vals = [75.0, 48.1]
    ax.bar(labels, vals, color=["#c0392b", "#2980b9"])
    ax.set_ylabel("hot segment贏過我們投組的頻率（%）")
    ax.set_title("① hot segment贏過我們的頻率")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.1f}%", (i, v), textcoords="offset points", xytext=(0, 4), ha="center")

    ax = axes[1]
    x = ["anchored", "rolling"]
    ret_level = [50.0, 71.75]   # rolling取69.6~73.9平均
    hold_level = [37.5, 34.8]
    width = 0.35
    xi = range(len(x))
    ax.bar([i - width / 2 for i in xi], ret_level, width, label="①報酬層級：落後幅度縮小", color="#27ae60")
    ax.bar([i + width / 2 for i in xi], hold_level, width, label="②持股層級：真的換成熱門股", color="#8e44ad")
    ax.set_xticks(list(xi))
    ax.set_xticklabels(x)
    ax.set_ylabel("2季後追蹤結果（%）")
    ax.set_title("② 自我修正：報酬vs持股層級")
    ax.legend(fontsize=8)
    for i, (a, b) in enumerate(zip(ret_level, hold_level)):
        ax.annotate(f"{a:.1f}%", (i - width / 2, a), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
        ax.annotate(f"{b:.1f}%", (i + width / 2, b), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)

    fig.suptitle("hotsample贏過insample的頻率＋自我修正檢查")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_hotsample_frequency.png")
    plt.close(fig)


def main():
    chart1_coverage_tilt_beta()
    chart2_anchored_vs_rolling_largecap()
    chart3_v1_ratio_by_window()
    chart4_exclude_v1_matrix()
    chart5_monitoring_frequency()
    chart6_hotsample_frequency()
    print(f"6張圖表已寫入 {OUT_DIR}")


if __name__ == "__main__":
    main()
