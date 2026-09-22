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

# ---- 已查證過的逐季報酬（見開發追蹤 D54/D67/D80/研究進度報告） ----
# 🔴🔴 2026-09-22（D80）：W2C 這組數字原本是 D54 用 `_l2_human_approved_w2c.py`
# 算出的「假設人核准」情境；正式8季重跑修完prompt bug後，L2臂5個觸發季
# 真的全部選了W2c，真實checkpoint（`formal_8q_control0_L2_execlayer_v3`）
# 算出來的數字跟這裡分毫不差——已交叉驗證過，不是巧合（機制本身是決定性
# 的）。這裡先不改成動態讀取v3 checkpoint（數字本來就對得上，重寫的效益
# 不高），但語意上這組數字現在代表「L2真實決策結果」，不是假設情境。
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
    for series, name, style in [(BASELINE, "不調整（現況）", "o-"), (W2C, "W2c情境（L2實際決策）", "s-"),
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
    # 🔴 2026-09-21（使用者抓到）：M7/M4這兩段落差很小，柱子本身太薄，白字置中
    # 會跟柱子邊界重疊、看不清楚——柱子矮於一定高度時，改把文字放到柱子正上方
    # （黑字），不要硬塞在裡面。
    y_range = max(cw, a_hrp) * 100
    label_min_height = y_range * 0.06
    for i, (lab, h, b) in enumerate(zip(labels, heights, bottoms)):
        val = [cw, m1r_gap, m7_gap, m4_gap, a_hrp][i]
        if h < label_min_height:
            ax.text(i, b + h + y_range * 0.015, f"{val*100:+.1f}%", ha="center", va="bottom",
                   fontsize=9, color="black", fontweight="bold")
        else:
            ax.text(i, b + h / 2, f"{val*100:+.1f}%", ha="center", va="center",
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
    for series, name in [(BASELINE, "不調整（現況）"), (W2C, "W2c情境（L2實際決策）"), (MARKET, "大盤（市值加權）")]:
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


# ============================================================ 10. 逐季改善幅度（L2實際決策 vs 不調整）
def chart_10():
    diff = [(w - b) * 100 for w, b in zip(W2C, BASELINE)]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#55A868" if d >= 0 else "#C44E52" for d in diff]
    ax.bar(QUARTERS, diff, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("W2c情境 - 不調整 (百分點)")
    ax.set_title("圖10｜逐季改善幅度（L2實際決策 vs 不調整）（正=調整有幫助，負=調整反而更差）")
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
    # 🔴 2026-09-21（使用者code review二輪抓到）：checkpoint 裡存的 `diagnosis`
    # 是 M7/M8 接線「前」當時算出來的，M7 恆記錄在 `not_available`、M8 恆不在
    # `region_b_state_warning`——不是真的沒資料，是那次跑的時候還沒接。
    # M3/M4 是決策當時的真實記錄，不動；M7/M8 現在用已重算的真實 B_all
    # （`ball_stock_level_recompute.csv`）＋`weights_end`回顧性補算（純算術、
    # 不燒LLM，且M7/M8本來就是回顧層、不驅動動作，補算不影響任何已經做出的
    # 決策——§7.0）。
    #
    # 🔴🔴 第一輪修正遺漏的問題：`diagnose_m6_m0()` 定義上跟任何機制觸發互斥
    # （見 diagnose.py），但原本這裡的 M6/M0 是直接沿用 checkpoint 裡「只看
    # M3/M4」算出來的舊結果，沒有把新補算的 M7/M8 併進「是否已有機制觸發」
    # 這個判斷——導致 2024Q1（M7觸發）同時顯示 M0、2025Q2（M7觸發）同時顯示
    # M6，自相矛盾。修正：M6/M0 這裡用 M3/M4（原始記錄）∪ M7/M8（新補算）
    # 一起重新判斷「是否已有機制觸發」；`performance_below_p10` 沿用
    # checkpoint 原始記錄的值，不動（那是另一個獨立的、這次沒有要求查核的
    # 問題，見 `is_performance_below_p10()`）。只要任一機制觸發，M6/M0 就
    # 標記「不適用」（灰色），不再借用 0（跟真正的 M6 混在一起分不清）。
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
        m3_triggered = d["region_b_state_warning"]["M3"]["triggered"]
        m4_triggered = d["region_a_attributable"]["M4"]["triggered"]
        grid[0, j] = 1 if m3_triggered else 0
        grid[1, j] = 1 if m4_triggered else 0

        m7 = diagnose.diagnose_m7(ball_map.get(o["as_of"]), o.get("equal_weight_benchmark_return"))
        m7_triggered = bool(m7.get("available") and m7["triggered"])
        grid[3, j] = np.nan if not m7.get("available") else (1 if m7_triggered else 0)
        m8 = diagnose.diagnose_m8(c["weights_end"])
        grid[4, j] = 1 if m8["triggered"] else 0

        any_triggered = m3_triggered or m4_triggered or m7_triggered or m8["triggered"]
        perf = d["fallback"].get("performance_check")
        if any_triggered:
            grid[2, j] = np.nan  # 已有其他機制觸發，M6/M0 定義上不適用
        elif perf is None:
            grid[2, j] = np.nan  # 沒有 performance_check 可用，誠實留灰
        else:
            grid[2, j] = 1 if perf["below_p10"] else 0  # 1=M0（異常但無法歸因）0=M6（正常）

    fig, ax = plt.subplots(figsize=(9, 4.8))
    masked = np.ma.masked_invalid(grid)
    cmap = plt.cm.RdYlGn_r
    cmap.set_bad(color="lightgray")
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(QUARTERS)))
    ax.set_xticklabels(QUARTERS)
    ax.set_yticks(range(len(mechanisms)))
    ax.set_yticklabels(mechanisms)
    ax.set_title("圖16｜診斷機制觸發總覽", fontsize=15)
    ax.text(0.5, -0.16,
           "M3/M4=決策當時原始記錄｜M7/M8=以重算後真實B_all回顧性補算（不影響原決策）\n"
           "M6/M0：0=M6正常 1=M0無法歸因｜灰=另有機制觸發時M6/M0不適用，或M7本次無法判定",
           transform=ax.transAxes, ha="center", va="top", fontsize=9)
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
