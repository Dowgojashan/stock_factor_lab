# -*- coding: utf-8 -*-
"""9/22報告用的16張圖（2026-09-22，使用者要求全部做）。純畫圖，不算新資料——
所有底層數字都來自已經查證過的真實資料（`_gather_chart_data.py`的輸出、
真實8季checkpoint、D63-D78一路驗證過的分析結果）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402

from app import diagnose  # noqa: E402

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

CHART_DIR = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

QUARTERS = ["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4"]

# ---- 已查證過的逐季報酬（見開發追蹤 D54/D67/研究進度報告） ----
BASELINE = [-0.0034, 0.1164, 0.0063, -0.0484, -0.0244, -0.0200, 0.0540, -0.0013]
W2C = [-0.0034, 0.1164, 0.0063, -0.0484, -0.0768, 0.0503, 0.1144, 0.0830]
MARKET = [0.1350, 0.1411, -0.0203, 0.0381, -0.0989, 0.0856, 0.1782, 0.1234]

TSMC_SELECT_RATE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0995]  # legacy/30檔近似；用全池7128檔比例
TSMC_INDEX_WEIGHT = [0.2478, 0.2893, 0.3168, 0.3239, 0.3530, 0.3324, 0.3618, 0.3838]


def savefig(fig, name):
    path = CHART_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已存 {path.name}")


def cum_path(returns):
    v = 1.0
    path = [1.0]
    for r in returns:
        v *= (1 + r)
        path.append(v)
    return path


# ============================================================ 1. 累積報酬走勢圖
def chart_1():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(9)
    labels = ["登記"] + QUARTERS
    for series, name, style in [(BASELINE, "不調整（現況）", "o-"), (W2C, "W2c情境（反事實）", "s-"),
                                (MARKET, "大盤（市值加權）", "^-")]:
        path = cum_path(series)
        ax.plot(x, [(p - 1) * 100 for p in path], style, label=name, linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("累積報酬 (%)")
    ax.set_title("圖1｜累積報酬走勢：不調整 vs W2c情境 vs 大盤")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend()
    ax.grid(alpha=0.3)
    savefig(fig, "01_cumulative_return")


# ============================================================ 2. 主動報酬走勢圖
def chart_2():
    fig, ax = plt.subplots(figsize=(9, 5))
    active_baseline = [(1 + b) / (1 + m) - 1 for b, m in zip(BASELINE, MARKET)]
    active_w2c = [(1 + w) / (1 + m) - 1 for w, m in zip(W2C, MARKET)]
    x = np.arange(len(QUARTERS))
    width = 0.35
    ax.bar(x - width / 2, [a * 100 for a in active_baseline], width, label="不調整 - 大盤")
    ax.bar(x + width / 2, [a * 100 for a in active_w2c], width, label="W2c情境 - 大盤")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(QUARTERS)
    ax.set_ylabel("主動報酬 (%)")
    ax.set_title("圖2｜逐季主動報酬（投組 - 大盤）")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    savefig(fig, "02_active_return")


# ============================================================ 3. 歸因瀑布圖（M1-R/M7/M4）
def chart_3():
    ball_df = pd.read_csv(CHART_DIR.parent / "ball_stock_level_recompute.csv")
    ball_map = dict(zip(ball_df["as_of"], ball_df["stock_level_return"]))

    with open("app/_runs/simulate_formal_8q_control0_L2_execlayer_v2.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    c0 = sorted([r for r in rows if r["arm"] == "control0"], key=lambda r: r["quarter_end"])

    def cum(vals):
        v = 1.0
        for x in vals:
            v *= (1 + x)
        return v - 1

    a_hrp_list, ball_list, ew_list, cw_list = [], [], [], []
    for c in c0:
        o = c["outcome"]
        a_hrp_list.append(o["portfolio_realized_return"])
        ball_list.append(ball_map[o["as_of"]])
        ew_list.append(o["equal_weight_benchmark_return"])
        cw_list.append(o["cap_weight_benchmark_return"])

    a_hrp, ball, ew, cw = cum(a_hrp_list), cum(ball_list), cum(ew_list), cum(cw_list)
    m4_gap, m7_gap, m1r_gap = a_hrp - ball, ball - ew, ew - cw
    total = a_hrp - cw

    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = ["大盤\n(市值加權)", "M1-R\n規模曝險", "M7\n候選池失效", "M4\n選股相對失效", "A_hrp\n(實際投組)"]
    values = [cw * 100, m1r_gap * 100, m7_gap * 100, m4_gap * 100, a_hrp * 100]
    starts = [0, cw * 100, (cw + m1r_gap) * 100, (cw + m1r_gap + m7_gap) * 100, 0]
    colors = ["#4C72B0", "#DD8452", "#DD8452", "#DD8452", "#4C72B0"]
    bar_heights = [cw * 100, m1r_gap * 100, m7_gap * 100, m4_gap * 100, a_hrp * 100]
    bottoms = [0, min(cw, cw + m1r_gap) * 100, min(cw + m1r_gap, cw + m1r_gap + m7_gap) * 100,
              min(cw + m1r_gap + m7_gap, a_hrp) * 100, 0]
    heights = [cw * 100, abs(m1r_gap) * 100, abs(m7_gap) * 100, abs(m4_gap) * 100, a_hrp * 100]
    ax.bar(labels, heights, bottom=bottoms, color=colors)
    for i, (lab, h, b) in enumerate(zip(labels, heights, bottoms)):
        ax.text(i, b + h / 2, f"{[cw,m1r_gap,m7_gap,m4_gap,a_hrp][i]*100:+.1f}%", ha="center", va="center",
               fontsize=9, color="white", fontweight="bold")
    ax.set_ylabel("累積報酬 (%)")
    ax.set_title(f"圖3｜歸因瀑布圖：8季累積落差拆解（總落差 {total*100:+.1f}pp）")
    ax.grid(alpha=0.3, axis="y")
    savefig(fig, "03_attribution_waterfall")
    print(f"    (查核用) a_hrp={a_hrp:+.4f} ball={ball:+.4f} ew={ew:+.4f} cw={cw:+.4f}")


# ============================================================ 4. 台積電權重vs選中率雙軸圖
def chart_4():
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(QUARTERS))
    ax1.bar(x, [r * 100 for r in TSMC_SELECT_RATE], color="#4C72B0", alpha=0.7, label="策略選中比例（代表池30檔，legacy）")
    ax1.set_ylabel("策略選中比例 (%)", color="#4C72B0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(QUARTERS)
    ax1.tick_params(axis="y", labelcolor="#4C72B0")

    ax2 = ax1.twinx()
    ax2.plot(x, [w * 100 for w in TSMC_INDEX_WEIGHT], "o-", color="#C44E52", linewidth=2.5, label="佔大盤權重")
    ax2.set_ylabel("佔大盤權重 (%)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")

    fig.suptitle("圖4｜台積電：佔大盤權重 vs 被選中的策略比例")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    savefig(fig, "04_tsmc_weight_vs_selection")


# ============================================================ 5. M1-D偏離走勢圖
def chart_5():
    df = pd.read_csv(CHART_DIR / "m1d_deviation.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    colors = ["#55A868" if s == "NONE" else "#C44E52" for s in df["state"]]
    ax.bar(x, df["cumulative_deviation"] * 100, color=colors)
    ax.axhline(df["p75"].iloc[0] * 100, color="orange", linestyle="--", label=f"p75門檻 ({df['p75'].iloc[0]*100:.1f}pp)")
    ax.axhline(df["p90"].iloc[0] * 100, color="red", linestyle="--", label=f"p90門檻 ({df['p90'].iloc[0]*100:.1f}pp)")
    ax.set_xticks(x)
    ax.set_xticklabels(QUARTERS)
    ax.set_ylabel("累計偏離 (百分點)")
    ax.set_title("圖5｜M1-D 集中度偏離走勢（綠=正常，紅=已觸發）")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    savefig(fig, "05_m1d_deviation_trend")


# ============================================================ 6. 水下回撤圖
def chart_6():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for series, name in [(BASELINE, "不調整（現況）"), (W2C, "W2c情境（反事實）"), (MARKET, "大盤（市值加權）")]:
        path = np.array(cum_path(series))
        running_max = np.maximum.accumulate(path)
        dd = (path - running_max) / running_max
        ax.plot(range(len(dd)), dd * 100, "-o", label=name, linewidth=2)
    ax.set_xticks(range(9))
    ax.set_xticklabels(["登記"] + QUARTERS)
    ax.set_ylabel("回撤 (%)")
    ax.set_title("圖6｜水下回撤圖（距歷史高點的下跌幅度）")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend()
    ax.grid(alpha=0.3)
    savefig(fig, "06_underwater_drawdown")


# ============================================================ 7. 前十大持股集中度對照
def chart_7():
    df = pd.read_csv(CHART_DIR / "top10_concentration.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["portfolio_top10_weight"] * 100, width, label="投組前十大集中度")
    ax.bar(x + width / 2, df["benchmark_top10_weight"] * 100, width, label="大盤前十大集中度")
    ax.set_xticks(x)
    ax.set_xticklabels(QUARTERS)
    ax.set_ylabel("前十大持股佔比 (%)")
    ax.set_title("圖7｜前十大持股集中度：投組 vs 大盤")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    savefig(fig, "07_top10_concentration")


# ============================================================ 8. 三檔股票對照長條圖
def chart_8():
    stocks = ["台積電\n(2330)", "鴻海\n(2317)", "聯發科\n(2454)"]
    select_rate = [12.91, 36.95, 9.68]
    weight_range = [(24.78, 38.38), (2.33, 3.42), (2.39, 3.14)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.bar(stocks, select_rate, color=["#C44E52", "#55A868", "#4C72B0"])
    ax1.set_ylabel("8季曾選中比例 (%)")
    ax1.set_title("選中比例")
    ax1.grid(alpha=0.3, axis="y")
    for i, v in enumerate(select_rate):
        ax1.text(i, v + 0.5, f"{v}%", ha="center")

    mids = [(lo + hi) / 2 for lo, hi in weight_range]
    errs = [(hi - lo) / 2 for lo, hi in weight_range]
    ax2.bar(stocks, mids, yerr=errs, capsize=8, color=["#C44E52", "#55A868", "#4C72B0"])
    ax2.set_ylabel("佔大盤權重區間 (%)")
    ax2.set_title("指數權重（8季範圍）")
    ax2.grid(alpha=0.3, axis="y")
    fig.suptitle("圖8｜台積電 vs 鴻海 vs 聯發科：選中比例 vs 指數權重")
    savefig(fig, "08_three_stocks_comparison")


# ============================================================ 9. 換手率走勢圖
def chart_9():
    df = pd.read_csv(CHART_DIR.parent / "conditional_w2_netcost.csv")
    df = df.iloc[:8]  # 只取本次8季範圍
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["base_turnover"] * 100, width, label="不調整（現況）")
    ax.bar(x + width / 2, df["cond_turnover"] * 100, width, label="W2c情境")
    ax.set_xticks(x)
    ax.set_xticklabels(QUARTERS)
    ax.set_ylabel("換手率 (%)")
    ax.set_title("圖9｜逐季換手率：不調整 vs W2c情境")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    savefig(fig, "09_turnover")


# ============================================================ 10. 逐季反事實改善幅度
def chart_10():
    diff = [(w - b) * 100 for w, b in zip(W2C, BASELINE)]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#55A868" if d >= 0 else "#C44E52" for d in diff]
    ax.bar(QUARTERS, diff, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("W2c情境 - 不調整 (百分點)")
    ax.set_title("圖10｜逐季反事實改善幅度（正=調整有幫助，負=調整反而更差）")
    ax.grid(alpha=0.3, axis="y")
    savefig(fig, "10_counterfactual_improvement")


# ============================================================ 11. 四個歷史窗次選中比例
def chart_11():
    windows = ["Window1\n(2015-2017)", "Window2\n(2018-2020)", "Window3\n(2021-2023)", "Window4\n(2024-2025)"]
    rates = [12.02, 12.89, 10.87, 12.91]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(windows, rates, color=["#4C72B0", "#4C72B0", "#4C72B0", "#C44E52"])
    ax.set_ylabel("8-12季累計曾選中比例 (%)")
    ax.set_title("圖11｜四個歷史窗次「曾選中台積電」比例（結構性穩定，非本次特有）")
    for i, v in enumerate(rates):
        ax.text(i, v + 0.3, f"{v}%", ha="center")
    ax.grid(alpha=0.3, axis="y")
    savefig(fig, "11_four_windows_comparison")


# ============================================================ 12. 配額覆蓋率對照圖
def chart_12():
    clusters = ["群1", "群2", "群3\n(最大)", "群4", "群5", "群6"]
    sizes = [240, 2258, 2336, 75, 273, 1497]
    tsmc_pct = [0, 2.5, 22.6, 0, 26.7, 15.1]
    reps_selected = [0, 0, 0, 0, 2, 1]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(clusters))
    ax1.bar(x, sizes, color="#4C72B0", alpha=0.6, label="群大小（策略數）")
    ax1.set_ylabel("群大小（策略數）", color="#4C72B0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(clusters)
    ax1.tick_params(axis="y", labelcolor="#4C72B0")

    ax2 = ax1.twinx()
    ax2.plot(x, tsmc_pct, "o-", color="#DD8452", linewidth=2, label="台積電相關策略佔比(%)")
    for i, r in enumerate(reps_selected):
        ax2.annotate(f"選中{r}個", (x[i], tsmc_pct[i]), textcoords="offset points", xytext=(0, 12),
                    ha="center", color="#C44E52", fontweight="bold")
    ax2.set_ylabel("台積電相關策略佔比 (%)", color="#DD8452")
    ax2.tick_params(axis="y", labelcolor="#DD8452")

    fig.suptitle("圖12｜配額覆蓋率：群大小 vs 台積電相關策略密度（固定5個名額，不論群多大）")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    savefig(fig, "12_quota_coverage")


# ============================================================ 13. 保底名額修正前後對照
def chart_13():
    categories = ["前十大整體\n覆蓋率(檔數/30或36)", "台積電", "聯發科"]
    before = [6, 0, 0]
    after = [12, 0, 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width / 2, before, width, label="現行規則(30檔)")
    ax.bar(x + width / 2, after, width, label="+保底名額(36檔)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("被選中檔數")
    ax.set_title("圖13｜保底名額機制修正前後對照（整體有效，但救不了台積電/聯發科）")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    savefig(fig, "13_guaranteed_slot_before_after")


# ============================================================ 14. 風險報酬散佈圖
def chart_14():
    points = [("不調整", 3.55, 9.02), ("W2c情境", 11.66, 12.15), ("大盤", 30.59, 9.89)]
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, ret, mdd in points:
        ax.scatter(mdd, ret, s=150)
        ax.annotate(name, (mdd, ret), textcoords="offset points", xytext=(8, 8))
    ax.set_xlabel("最大回撤 MDD (%)")
    ax.set_ylabel("年化報酬 CAGR (%)")
    ax.set_title("圖14｜風險報酬散佈圖")
    ax.grid(alpha=0.3)
    savefig(fig, "14_risk_return_scatter")


# ============================================================ 15. 產業主動權重圖
def chart_15():
    df = pd.read_csv(CHART_DIR / "industry_active_weight_2025Q4.csv")
    df = df.sort_values("active_weight")
    top = pd.concat([df.head(6), df.tail(6)])
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#C44E52" if v < 0 else "#55A868" for v in top["active_weight"]]
    ax.barh(top["industry"], top["active_weight"] * 100, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("主動權重 = 投組權重 - 大盤權重 (百分點)")
    ax.set_title("圖15｜產業主動權重（2025Q4，紅=低配，綠=超配）")
    ax.grid(alpha=0.3, axis="x")
    savefig(fig, "15_industry_active_weight")


# ============================================================ 16. 診斷機制觸發總覽熱力圖
def chart_16():
    # 🔴 2026-09-23（code review 抓到）：checkpoint 裡存的 `diagnosis` 是 M7/M8
    # 接線「前」當時算出來的，M7 恆記錄在 `not_available`、M8 恆不在
    # `region_b_state_warning`——不是真的沒資料，是那次跑的時候還沒接。
    # M3/M4/M6/M0 是決策當時的真實記錄，不動；M7/M8 現在用已重算的真實
    # B_all（`ball_stock_level_recompute.csv`）＋`weights_end`回顧性補算
    # （純算術、不燒LLM，且M7/M8本來就是回顧層、不驅動動作，補算不影響
    # 任何已經做出的決策——§7.0）。
    with open("app/_runs/simulate_formal_8q_control0_L2_execlayer_v2.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    c0 = sorted([r for r in rows if r["arm"] == "control0"], key=lambda r: r["quarter_end"])

    ball_df = pd.read_csv(CHART_DIR.parent / "ball_stock_level_recompute.csv")
    ball_map = dict(zip(ball_df["as_of"], ball_df["stock_level_return"]))

    mechanisms = ["M3", "M4", "M6/M0", "M7", "M8"]
    grid = np.zeros((len(mechanisms), len(c0)))
    for j, c in enumerate(c0):
        d = c["diagnosis"]
        o = c["outcome"]
        grid[0, j] = 1 if d["region_b_state_warning"]["M3"]["triggered"] else 0
        grid[1, j] = 1 if d["region_a_attributable"]["M4"]["triggered"] else 0
        grid[2, j] = 1 if d["fallback"].get("mechanism") == "M0" else 0.5 if d["fallback"].get("mechanism") == "M6" else 0

        m7 = diagnose.diagnose_m7(ball_map.get(o["as_of"]), o.get("equal_weight_benchmark_return"))
        grid[3, j] = np.nan if not m7.get("available") else (1 if m7["triggered"] else 0)
        m8 = diagnose.diagnose_m8(c["weights_end"])
        grid[4, j] = 1 if m8["triggered"] else 0

    fig, ax = plt.subplots(figsize=(9, 4.5))
    masked = np.ma.masked_invalid(grid)
    cmap = plt.cm.RdYlGn_r
    cmap.set_bad(color="lightgray")
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(QUARTERS)))
    ax.set_xticklabels(QUARTERS)
    ax.set_yticks(range(len(mechanisms)))
    ax.set_yticklabels(mechanisms)
    ax.set_title("圖16｜診斷機制觸發總覽（M3/M4/M6/M0=決策當時原始記錄；"
                 "M7/M8=以重算後真實B_all回顧性補算，不影響原決策；灰=無法判定；"
                 "M6/M0：0=正常 0.5=M6 1=M0）", fontsize=10)
    for i in range(len(mechanisms)):
        for j in range(len(c0)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i,j]:.1f}" if mechanisms[i] == "M6/M0" else ("V" if grid[i, j] else "-"),
                       ha="center", va="center", fontsize=9)
    savefig(fig, "16_diagnosis_heatmap")


def main():
    for i, fn in enumerate([chart_1, chart_2, chart_3, chart_4, chart_5, chart_6, chart_7, chart_8,
                            chart_9, chart_10, chart_11, chart_12, chart_13, chart_14, chart_15, chart_16], 1):
        print(f"圖{i}：{fn.__name__}")
        fn()
    print(f"\n全部完成，存在 {CHART_DIR}")


if __name__ == "__main__":
    main()
