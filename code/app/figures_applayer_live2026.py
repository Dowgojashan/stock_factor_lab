# -*- coding: utf-8 -*-
"""應用層正式模式（live）2026 年度報告用圖（2026-09-15，老師 9-15 會議 §2.5 用）。

資料來源：`code/app/_runs/audit_log.jsonl` 第 15~23 筆（§9.8 前瞻驗證正式版，
11 欄位新 schema）——3 個日期（2025-12-31/2026-01-01「年初」、2026-03-31、
2026-05-15）× TW/US/XM 三市場，真實 LLM 呼叫，D2 洩漏掃描 9/9 全部乾淨。
數字全部來自 `entry["performance"]`／`entry["calibration"]`／
`entry["explanation"]["facts"]`，不是重新編的示意數字。

跟 `figures_applayer.py`（scheme E 驗證模式回測）的關鍵差異：**正式模式沒有
OOS**——IS/建模期數字不是預期報酬，所以圖 1／圖 3／圖 5 沒辦法直接套用
「OOS 績效」「IS vs OOS」「樣本外結果定位」這幾個標題，改成对應的正式模式
版本：IS 績效＋校準門檻對照、IS 相對校準門檻、以及「門檻已立、尚無實現數字」
的校準狀態視覺化。圖 2／圖 4（快慢時鐘、相關vs集中度）概念不變，直接套用。

    圖 1  三市場建模期(IS)績效對比（CAGR／MDD／Sharpe）
    圖 2  快時鐘 2026 年三個檢查點持股演變（不重複股票數／最大單檔權重）
    圖 3  IS CAGR vs 歷史校準門檻(p10)——IS 依然不是預期報酬
    圖 4  策略層相關 vs 股票層集中度（2026 年初檢查點）
    圖 5  校準監控狀態：正式 12 個月門檻尚未到期
    圖 6  §9.13b 已實現績效對比（真實股價，含跨市場，投組 vs 大盤）

用法：
    cd code
    python -m app.figures_applayer_live2026
"""
from __future__ import annotations

from pathlib import Path

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

OUT_DIR = None

MARKETS = ["TW", "US", "XM"]
LABEL = {"TW": "台股", "US": "美股", "XM": "跨市場"}
COLOR = {"TW": "#c0392b", "US": "#2471a3", "XM": "#1e8449"}

# ---- 真實數字（audit_log.jsonl 第 15~23 筆，2026-09-15 讀取）----
# 策略層（IS 績效／相關／校準門檻）三個檢查點內不變，只列一份。
PERF = {
    "TW": {"is_cagr": 0.23995989220437508, "is_mdd": -0.3350764352128839, "is_sharpe": 1.1583600836240207},
    "US": {"is_cagr": 0.380033853285874, "is_mdd": -0.32504787556521975, "is_sharpe": 1.5327725889379777},
    "XM": {"is_cagr": 0.38539962036635633, "is_mdd": -0.2675964653122368, "is_sharpe": 1.4765463783083044},
}
CALIB_P10 = {"TW": 0.1054172485150983, "US": 0.1115001668460922, "XM": 0.1529171518031913}
REF = {
    "TW": {"p10": 0.1054172485150983, "median": 0.1765644659462375, "p90": 0.3384674512273133},
    "US": {"p10": 0.1115001668460922, "median": 0.2733763489843619, "p90": 0.435003827940629},
    "XM": {"p10": 0.1529171518031913, "median": 0.2561890580987948, "p90": 0.4223598591956998},
}
AVG_CORR_NORMAL = {"TW": 0.779, "US": 0.9066, "XM": 0.6737}

# 快時鐘：三個檢查點（年初／Q1 末／5 月中），股票層隨 as_of 重新解析。
CHECKPOINTS = ["2026 年初", "2026-03-31", "2026-05-15"]
CP_N_STOCKS = {"TW": [575, 506, 516], "US": [795, 784, 745], "XM": [462, 411, 373]}
CP_MAX_WEIGHT = {"TW": [0.0108, 0.0115, 0.0176],
                "US": [0.0139, 0.0151, 0.0115],
                "XM": [0.0113, 0.0126, 0.0150]}
# 圖 4 用「年初」檢查點的股票層集中度，跟策略層相關對齊同一個時間點。
MAX_WEIGHT_AT_START = {m: CP_MAX_WEIGHT[m][0] for m in MARKETS}

SUBTITLE = "資料來源：正式模式 2026 年真實呼叫（audit_log.jsonl 第 15~23 筆，2026-09-15）"


def fig1_is_performance():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    metrics = [("is_cagr", "建模期年化報酬 (IS CAGR)", "{:.1%}", False),
              ("is_mdd", "建模期最大回撤 (IS MDD)", "{:.1%}", True),
              ("is_sharpe", "建模期 Sharpe (IS)", "{:.2f}", False)]
    for ax, (key, title, fmt, is_mdd) in zip(axes, metrics):
        raw_vals = [PERF[m][key] for m in MARKETS]
        plot_vals = [abs(v) for v in raw_vals] if is_mdd else raw_vals
        bars = ax.bar([LABEL[m] for m in MARKETS], plot_vals, color=[COLOR[m] for m in MARKETS])
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="black", linewidth=0.6)
        top = max(plot_vals)
        ax.set_ylim(0, top * 1.18)
        for b, v, raw in zip(bars, plot_vals, raw_vals):
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.03,
                   fmt.format(raw), ha="center", va="bottom", fontsize=9)
    fig.suptitle("三市場建模期(IS)績效對比——正式模式，尚無 OOS 可比", fontsize=12)
    fig.text(0.5, 0.01, SUBTITLE, ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUT_DIR / "1_三市場IS績效對比.png")
    plt.close(fig)


def fig2_quarterly_evolution():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    x = np.arange(len(CHECKPOINTS))
    for m in MARKETS:
        ax1.plot(x, CP_N_STOCKS[m], marker="o", label=LABEL[m], color=COLOR[m])
        ax2.plot(x, [v * 100 for v in CP_MAX_WEIGHT[m]], marker="o",
                label=LABEL[m], color=COLOR[m])
    ax1.set_ylabel("不重複股票數")
    ax1.set_title("快時鐘：2026 年三個檢查點重新解析持股（策略清單不變）", fontsize=10)
    ax1.legend(fontsize=8)
    ax2.set_ylabel("最大單檔權重 (%)")
    ax2.set_ylim(0, 8.5)
    ax2.axhline(8.0, color="gray", linestyle="--", linewidth=0.8)
    ax2.text(len(CHECKPOINTS) - 1, 8.2, "單股上限 8%（TW/US 預設）", fontsize=7,
             ha="right", color="gray")
    ax2.set_xticks(x, CHECKPOINTS, fontsize=8.5)
    fig.suptitle("快時鐘：正式模式 2026 年逐檢查點持股演變（真實資料）", fontsize=12)
    fig.text(0.5, 0.01, SUBTITLE, ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT_DIR / "2_快時鐘2026檢查點持股演變.png")
    plt.close(fig)


def fig3_is_vs_calib_threshold():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(MARKETS))
    w = 0.32
    is_vals = [PERF[m]["is_cagr"] for m in MARKETS]
    p10_vals = [CALIB_P10[m] for m in MARKETS]
    b1 = ax.bar(x - w / 2, is_vals, w, label="本次 IS CAGR（樣本內配適值）", color="#34495e")
    b2 = ax.bar(x + w / 2, p10_vals, w, label="歷史校準門檻 p10（日後示警線）", color="#95a5a6")
    for b, v in zip(list(b1) + list(b2), is_vals + p10_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}", ha="center",
               va="bottom", fontsize=8.5)
    ax.set_xticks(x, [LABEL[m] for m in MARKETS])
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("CAGR")
    ax.legend(fontsize=8.5)
    ax.set_title("IS 配適值遠高於日後校準門檻——IS 依然不能當預期報酬", fontsize=11)
    fig.text(0.5, 0.01, SUBTITLE, ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUT_DIR / "3_IS與校準門檻對比.png")
    plt.close(fig)


def fig4_corr_vs_concentration():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    corr_vals = [AVG_CORR_NORMAL[m] for m in MARKETS]
    weight_vals = [MAX_WEIGHT_AT_START[m] * 100 for m in MARKETS]
    b1 = ax1.bar([LABEL[m] for m in MARKETS], corr_vals, color=[COLOR[m] for m in MARKETS])
    ax1.set_title("策略層：常態期平均配對相關", fontsize=10)
    ax1.set_ylim(0, 1.0)
    for b, v in zip(b1, corr_vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    b2 = ax2.bar([LABEL[m] for m in MARKETS], weight_vals, color=[COLOR[m] for m in MARKETS])
    ax2.set_title("股票層：最大單檔持股權重 (%)，2026 年初", fontsize=10)
    ax2.set_ylim(0, 8.5)
    ax2.axhline(8.0, color="gray", linestyle="--", linewidth=0.8)
    for b, v in zip(b2, weight_vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}%", ha="center", fontsize=9)
    fig.suptitle("「策略層相關高」不等於「股票層集中」——2026 正式模式同樣成立", fontsize=12)
    fig.text(0.5, 0.01, SUBTITLE, ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(OUT_DIR / "4_策略相關vs股票集中度.png")
    plt.close(fig)


def fig5_calibration_pending():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y_pos = np.arange(len(MARKETS))
    for i, m in enumerate(MARKETS):
        r = REF[m]
        ax.plot([r["p10"], r["p90"]], [i, i], color="#bbbbbb", linewidth=6, solid_capstyle="round",
               zorder=1)
        ax.plot([r["median"]], [i], marker="|", color="black", markersize=18, zorder=2)
        ax.plot([r["p10"]], [i], marker="v", color="#c0392b", markersize=10, zorder=3)
        ax.text(r["p90"] + 0.02, i, f"{LABEL[m]}：正式 12 個月校準門檻 p10={r['p10']:.1%}，"
               "完整觀測窗尚未到期", va="center", fontsize=8.5)
    ax.set_yticks(y_pos, [LABEL[m] for m in MARKETS])
    ax.set_ylim(-0.6, len(MARKETS) - 1 + 0.6)
    ax.set_xlabel("OOS CAGR（歷史參照分布）")
    ax.set_title("校準監控狀態：12 個月門檻尚未到期（部分真實已實現數據見圖 6）", fontsize=11)
    ax.set_xlim(-0.05, 0.85)
    fig.text(0.5, 0.065,
           "灰色橫條＝歷史 p10~p90 區間，黑色短線＝歷史中位數，▼＝示警門檻 p10"
           "（現有最長窗僅 1.5~7 個月，未滿 12 個月不硬套判定）",
           ha="center", fontsize=8, color="#555")
    fig.text(0.5, 0.01, SUBTITLE, ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(OUT_DIR / "5_校準監控狀態.png")
    plt.close(fig)


# ---- 圖 6：§9.13b 已實現績效（真實股價，分段窗口對齊快時鐘）----
# 大盤＝市值加權大盤（TAIEX／S&P，真正的市場基準；database.get_taiex_data()）。
# XM 沒有單一「大盤」可比——H7 已定案 XM 的 TW/US 大盤數字不合成、並列呈現
# （XM 用的是 1/3 台股＋2/3 美股的策略清單，不是任兩個獨立市場數字的線性組合）。
WINDOWS = ["01-01→03-31", "03-31→05-15", "05-15→cutoff"]
REALIZED = {
    "TW": {"portfolio": [0.0031, 0.0874, 0.0646], "market": [0.0953, 0.2979, 0.0603]},
    "US": {"portfolio": [0.0550, 0.1491, 0.0297], "market": [-0.0463, 0.1348, 0.0101]},
    "XM": {"portfolio": [0.0545, 0.1352, 0.0236]},
}


def _bar_labels(ax, bars):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + (0.006 if v >= 0 else -0.012),
               f"{v:.1%}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)


def fig6_realized_performance():
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharey=False)
    x = np.arange(len(WINDOWS))
    w = 0.32
    for ax, m in zip(axes, ["TW", "US"]):
        d = REALIZED[m]
        b1 = ax.bar(x - w / 2, d["portfolio"], w, label="投組已實現", color=COLOR[m])
        b2 = ax.bar(x + w / 2, d["market"], w, label="大盤", color="#2c2c2c")
        _bar_labels(ax, b1)
        _bar_labels(ax, b2)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x, WINDOWS, fontsize=8.5)
        ax.set_title(f"{LABEL[m]}：投組 vs 大盤（真實股價）", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper left")
        all_vals = d["portfolio"] + d["market"]
        ax.set_ylim(min(all_vals) * 1.25 if min(all_vals) < 0 else 0, max(all_vals) * 1.18)

    # XM 面板：沒有單一「大盤」可合成，改成投組 vs 各自的 TW／US 大盤並列參照。
    ax = axes[2]
    w3 = 0.26
    d = REALIZED["XM"]
    b1 = ax.bar(x - w3, d["portfolio"], w3, label="XM 投組已實現", color=COLOR["XM"])
    b2 = ax.bar(x, REALIZED["TW"]["market"], w3, label="對照：台股大盤", color="#c0392b", alpha=0.55)
    b3 = ax.bar(x + w3, REALIZED["US"]["market"], w3, label="對照：美股大盤", color="#2471a3", alpha=0.55)
    for bars in (b1, b2, b3):
        _bar_labels(ax, bars)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x, WINDOWS, fontsize=8.5)
    ax.set_title("跨市場：XM 投組 vs 台美各自大盤（不合成單一大盤）", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    all_vals = d["portfolio"] + REALIZED["TW"]["market"] + REALIZED["US"]["market"]
    ax.set_ylim(min(all_vals) * 1.25 if min(all_vals) < 0 else 0, max(all_vals) * 1.18)

    fig.suptitle("§9.13b 已實現績效：台股同期大漲、投組因設計低配台積電而落後", fontsize=12)
    fig.text(0.5, 0.01, SUBTITLE + "；價格資料至 TW 2026-07-24／US 2026-08-19",
           ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(OUT_DIR / "6_已實現績效對比.png")
    plt.close(fig)


def main():
    global OUT_DIR
    OUT_DIR = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "figures_live2026"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig1_is_performance()
    fig2_quarterly_evolution()
    fig3_is_vs_calib_threshold()
    fig4_corr_vs_concentration()
    fig5_calibration_pending()
    fig6_realized_performance()
    print(f"6 張圖已存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
