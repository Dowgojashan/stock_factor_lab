# -*- coding: utf-8 -*-
"""應用層驗證模式報告用圖（2026-09-15，老師 9-15 會議 §2 用）。

資料來源：scheme E／window 4／A_hrp／legacy／equal 三市場驗證模式實測
（`文件/應用層開發追蹤.md` §10.10／§10.11 記錄的那組），數字全部來自真實
`holdings.performance`／`reference_oos`／`stock_concentration`／群間相關矩陣，
不是重新編的示意數字。

    圖 1  三市場 OOS 績效對比（CAGR／MDD／Sharpe）
    圖 2  快時鐘季度持股演變（不重複股票數／最大單檔權重，8 個季度）
    圖 3  IS vs OOS 落差對比（驗證「IS 非預期報酬」這條系統提示鐵則）
    圖 4  策略層相關 vs 股票層集中度（驗證「相關高不等於集中」這條核心論點）
    圖 5  樣本外結果落在歷史參照分布的哪個百分位（校準監控視覺化）

用法：
    cd code
    python -m app.figures_applayer
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 130,
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

OUT_DIR = None  # 主程式設定，避免 import 時就觸碰檔案系統

MARKETS = ["TW", "US", "XM"]
LABEL = {"TW": "台股", "US": "美股", "XM": "跨市場"}
COLOR = {"TW": "#c0392b", "US": "#2471a3", "XM": "#1e8449"}

# ---- 真實數字（scheme E／window 4，2026-09-15 驗證模式實測）----
PERF = {
    "TW": {"is_cagr": 0.2745285072725003, "oos_cagr": 0.0024100475054662,
          "oos_mdd": -0.1362210192630262, "oos_sharpe": 0.0832291016315055},
    "US": {"is_cagr": 0.3692048494831836, "oos_cagr": 0.4894668185532882,
          "oos_mdd": -0.1488382434036866, "oos_sharpe": 1.8401323787504376},
    "XM": {"is_cagr": 0.3711929001555683, "oos_cagr": 0.5016316174076878,
          "oos_mdd": -0.0989181307655935, "oos_sharpe": 2.2353881274455225},
}

REF = {
    "TW": {"p10": 0.1054172485150983, "median": 0.1765644659462375, "p90": 0.3384674512273133},
    "US": {"p10": 0.1115001668460922, "median": 0.2733763489843619, "p90": 0.435003827940629},
    "XM": {"p10": 0.1529171518031913, "median": 0.2561890580987948, "p90": 0.4223598591956998},
}

AVG_CORR_NORMAL = {"TW": 0.7733981908000881, "US": 0.9142163078650611, "XM": 0.693791371284808}
MAX_STOCK_WEIGHT_OOS_END = {"TW": 0.0117, "US": 0.0138, "XM": 0.0098}

QUARTERS = ["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4"]
QUARTERLY_N_STOCKS = {
    "TW": [445, 421, 474, 396, 466, 387, 315, 502],
    "US": [603, 578, 623, 689, 644, 677, 712, 774],
    "XM": [330, 290, 345, 333, 364, 334, 333, 430],
}
QUARTERLY_MAX_WEIGHT = {
    "TW": [0.0171, 0.0177, 0.0151, 0.0146, 0.0167, 0.0238, 0.0147, 0.0117],
    "US": [0.0169, 0.0231, 0.0162, 0.0117, 0.0174, 0.0112, 0.0135, 0.0156],
    "XM": [0.0147, 0.0197, 0.0118, 0.0134, 0.0141, 0.0119, 0.0130, 0.0094],
}

SUBTITLE = "資料來源：驗證模式實測（scheme E／window 4／hrp／legacy／equal，2026-09-15）"


def fig1_three_market_performance():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    metrics = [("oos_cagr", "樣本外年化報酬 (OOS CAGR)", "{:.1%}"),
              ("oos_mdd", "樣本外最大回撤 (OOS MDD)", "{:.1%}"),
              ("oos_sharpe", "樣本外 Sharpe", "{:.2f}")]
    for ax, (key, title, fmt) in zip(axes, metrics):
        vals = [PERF[m][key] for m in MARKETS]
        bars = ax.bar([LABEL[m] for m in MARKETS], vals, color=[COLOR[m] for m in MARKETS])
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.01 if v >= 0 else 0.006),
                   fmt.format(v), ha="center", va="bottom", fontsize=9)
    fig.suptitle("三市場樣本外績效對比（scheme E／window 4）", fontsize=12)
    fig.text(0.5, 0.01, SUBTITLE, ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUT_DIR / "1_三市場OOS績效對比.png")
    plt.close(fig)


def fig2_quarterly_evolution():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    x = np.arange(len(QUARTERS))
    for m in MARKETS:
        ax1.plot(x, QUARTERLY_N_STOCKS[m], marker="o", label=LABEL[m], color=COLOR[m])
        ax2.plot(x, [v * 100 for v in QUARTERLY_MAX_WEIGHT[m]], marker="o",
                label=LABEL[m], color=COLOR[m])
    ax1.set_ylabel("不重複股票數")
    ax1.set_title("快時鐘：每季重新解析持股（策略清單不變，實際股票隨季度更新）", fontsize=10)
    ax1.legend(fontsize=8)
    ax2.set_ylabel("最大單檔權重 (%)")
    ax2.axhline(8.0, color="gray", linestyle="--", linewidth=0.8)
    ax2.text(len(QUARTERS) - 1, 8.2, "單股上限 8%（TW/US 預設）", fontsize=7,
             ha="right", color="gray")
    ax2.set_xticks(x, QUARTERS, fontsize=8)
    fig.suptitle("快時鐘季度持股演變（2024Q1~2025Q4）", fontsize=12)
    fig.text(0.5, 0.01, SUBTITLE, ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT_DIR / "2_快時鐘季度持股演變.png")
    plt.close(fig)


def fig3_is_vs_oos_gap():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(MARKETS))
    w = 0.32
    is_vals = [PERF[m]["is_cagr"] for m in MARKETS]
    oos_vals = [PERF[m]["oos_cagr"] for m in MARKETS]
    b1 = ax.bar(x - w / 2, is_vals, w, label="建模期 IS CAGR（樣本內配適值）", color="#95a5a6")
    b2 = ax.bar(x + w / 2, oos_vals, w, label="樣本外 OOS CAGR（真實後續表現）", color="#34495e")
    for b, v in zip(list(b1) + list(b2), is_vals + oos_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}", ha="center",
               va="bottom", fontsize=8.5)
    ax.set_xticks(x, [LABEL[m] for m in MARKETS])
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("CAGR")
    ax.legend(fontsize=8.5)
    ax.set_title("IS 配適值 vs OOS 真實表現——IS 不能當預期報酬", fontsize=11)
    fig.text(0.5, 0.01, SUBTITLE, ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUT_DIR / "3_IS與OOS落差對比.png")
    plt.close(fig)


def fig4_corr_vs_concentration():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    corr_vals = [AVG_CORR_NORMAL[m] for m in MARKETS]
    weight_vals = [MAX_STOCK_WEIGHT_OOS_END[m] * 100 for m in MARKETS]
    b1 = ax1.bar([LABEL[m] for m in MARKETS], corr_vals, color=[COLOR[m] for m in MARKETS])
    ax1.set_title("策略層：常態期平均配對相關", fontsize=10)
    ax1.set_ylim(0, 1.0)
    for b, v in zip(b1, corr_vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    b2 = ax2.bar([LABEL[m] for m in MARKETS], weight_vals, color=[COLOR[m] for m in MARKETS])
    ax2.set_title("股票層：最大單檔持股權重 (%)", fontsize=10)
    ax2.set_ylim(0, 8.5)
    ax2.axhline(8.0, color="gray", linestyle="--", linewidth=0.8)
    for b, v in zip(b2, weight_vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}%", ha="center", fontsize=9)
    fig.suptitle("「策略層相關高」不等於「股票層集中」——三市場都成立", fontsize=12)
    fig.text(0.5, 0.01, SUBTITLE, ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(OUT_DIR / "4_策略相關vs股票集中度.png")
    plt.close(fig)


def fig5_calibration_position():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y_pos = np.arange(len(MARKETS))
    for i, m in enumerate(MARKETS):
        r = REF[m]
        ax.plot([r["p10"], r["p90"]], [i, i], color="#bbbbbb", linewidth=6, solid_capstyle="round",
               zorder=1)
        ax.plot([r["median"]], [i], marker="|", color="black", markersize=18, zorder=2)
        actual = PERF[m]["oos_cagr"]
        ax.scatter([actual], [i], color=COLOR[m], s=140, zorder=3, marker="*",
                  edgecolor="black", linewidth=0.5)
        flag = ("低於 p10，校準警示" if actual < r["p10"]
               else "高於 p90" if actual > r["p90"] else "落在 p10~p90 區間")
        ax.text(max(r["p90"], actual) + 0.02, i, f"{LABEL[m]}：本次 {actual:.1%}（{flag}）",
               va="center", fontsize=8.5)
    ax.set_yticks(y_pos, [LABEL[m] for m in MARKETS])
    ax.set_ylim(-0.6, len(MARKETS) - 1 + 0.6)
    ax.set_xlabel("OOS CAGR")
    ax.set_title("本次結果落在歷史參照分布的哪個百分位（校準監控）", fontsize=11)
    ax.set_xlim(-0.05, 0.72)
    fig.text(0.5, 0.065,
           "灰色橫條＝歷史 p10~p90 區間，黑色短線＝歷史中位數，★＝本次真實 OOS CAGR",
           ha="center", fontsize=8, color="#555")
    fig.text(0.5, 0.01, SUBTITLE, ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(OUT_DIR / "5_樣本外結果校準定位.png")
    plt.close(fig)


def main():
    global OUT_DIR
    from pathlib import Path
    OUT_DIR = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "figures"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig1_three_market_performance()
    fig2_quarterly_evolution()
    fig3_is_vs_oos_gap()
    fig4_corr_vs_concentration()
    fig5_calibration_position()
    print(f"5 張圖已存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
