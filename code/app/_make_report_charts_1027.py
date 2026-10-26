# -*- coding: utf-8 -*-
"""2026-10-27：台美股L2逐季報酬率、逐季回撤軌跡，分別跟大盤比較的折線圖，
給10/27進度報告用。資料來源：

①逐季報酬——直接讀`code/app/_runs/`的4組最終production checkpoint
（TW baseline/exclude_v1、US baseline/exclude_v1，皆為L2臂，30%上限、
含M2+Hot Segment+修正後M1-D門檻的最終版），`outcome.portfolio_realized_
return`逐季取出，不重新推導任何結論。
②大盤逐季報酬——直接查`taiex_tr`/`sp500_tr`表，用跟策略checkpoint同一組
9個日期（2023-12-31登記+8季末）取asof收盤價算季報酬，跟策略同一把尺。
③逐季回撤——從①②的逐季報酬複利重建NAV序列（起點=1.0），算每個時點
相對歷史高點的drawdown%，不是只看單一MDD數字，是整條軌跡。

用法（cwd必須是code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._make_report_charts_1027
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from database import Database  # noqa: E402
from app import simulate  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 11,
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

APPLAYER = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
OUT_DIR = APPLAYER / "report_charts_1027"
OUT_DIR.mkdir(exist_ok=True)

SAMPLE_DATES = [simulate.REGISTRATION_DATE] + list(simulate.QUARTER_ENDS)
Q_LABELS = ["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4"]


def quarterly_returns_from_checkpoint(path: str) -> list[float]:
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            cp = json.loads(line)
            if cp.get("arm") not in (None, "L2"):
                continue
            rows[cp["quarter_end"]] = cp["outcome"]["portfolio_realized_return"]
    return [rows[q] for q in simulate.QUARTER_ENDS]


def benchmark_quarterly_returns(table: str) -> list[float]:
    db = Database("TW" if "taiex" in table else "US")
    conn = db.create_connection()
    df = pd.read_sql(f"SELECT date, close FROM {table} ORDER BY date", conn)
    df["date"] = pd.to_datetime(df["date"])

    def asof_close(d: str) -> float:
        ts = pd.Timestamp(d)
        prior = df[df.date <= ts]
        return float(prior.iloc[-1]["close"])

    vals = [asof_close(d) for d in SAMPLE_DATES]
    return [vals[i] / vals[i - 1] - 1 for i in range(1, len(vals))]


def nav_series(rets: list[float]) -> list[float]:
    nav = [1.0]
    for r in rets:
        nav.append(nav[-1] * (1 + r))
    return nav


def drawdown_series(nav: list[float]) -> list[float]:
    peak = nav[0]
    dd = []
    for v in nav:
        peak = max(peak, v)
        dd.append((v / peak - 1) * 100)
    return dd


def chart_quarterly_returns(market: str, series: dict[str, list[float]], out_name: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(Q_LABELS))
    styles = {
        "baseline L2": dict(marker="o", color="#2980b9"),
        "exclude_v1 L2": dict(marker="s", color="#27ae60"),
        "大盤": dict(marker="^", color="#7f8c8d", linestyle="--"),
    }
    label_dy = {"baseline L2": 14, "exclude_v1 L2": -18, "大盤": 14}
    for label, rets in series.items():
        pct = [r * 100 for r in rets]
        ax.plot(x, pct, label=label, linewidth=2, markersize=6, **styles[label])
        dy = label_dy[label]
        for xi, yi in zip(x, pct):
            ax.annotate(f"{yi:+.1f}", (xi, yi), textcoords="offset points",
                       xytext=(0, dy), ha="center", fontsize=7.5,
                       color=styles[label]["color"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(Q_LABELS)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("季報酬率（%）")
    ax.set_title(f"{market}：L2策略逐季報酬率 vs 大盤（2024Q1~2025Q4）")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name)
    plt.close(fig)


def chart_drawdown(market: str, series: dict[str, list[float]], out_name: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    labels9 = ["登記"] + Q_LABELS
    x = range(len(labels9))
    styles = {
        "baseline L2": dict(marker="o", color="#2980b9"),
        "exclude_v1 L2": dict(marker="s", color="#27ae60"),
        "大盤": dict(marker="^", color="#7f8c8d", linestyle="--"),
    }
    for label, dd in series.items():
        ax.plot(x, dd, label=label, linewidth=2, markersize=6, **styles[label])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels9)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("相對歷史高點的回撤（%）")
    ax.set_title(f"{market}：L2策略逐季回撤軌跡 vs 大盤（2023Q4登記~2025Q4）")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name)
    plt.close(fig)


def main():
    tw_base = quarterly_returns_from_checkpoint(
        "app/_runs/simulate_formal_8q_control0_L2_openSec_boost_v4_cap30_m2_v2.jsonl")
    tw_ex = quarterly_returns_from_checkpoint(
        "app/_runs/simulate_anchored_w4_openSec_boost_exclude_v1_cap30_hotseg_L2.jsonl")
    us_base = quarterly_returns_from_checkpoint(
        "app/_runs/simulate_anchored_w4_openSec_balanced_baseline_cap30_hotseg_m1dfix_L2.jsonl")
    us_ex = quarterly_returns_from_checkpoint(
        "app/_runs/simulate_anchored_w4_openSec_balanced_exclude_v1_cap30_hotseg_m1dfix_L2.jsonl")

    taiex = benchmark_quarterly_returns("taiex_tr")
    sp500 = benchmark_quarterly_returns("sp500_tr")

    chart_quarterly_returns("台股", {
        "baseline L2": tw_base, "exclude_v1 L2": tw_ex, "大盤": taiex,
    }, "01_tw_quarterly_returns.png")
    chart_quarterly_returns("美股", {
        "baseline L2": us_base, "exclude_v1 L2": us_ex, "大盤": sp500,
    }, "02_us_quarterly_returns.png")

    chart_drawdown("台股", {
        "baseline L2": drawdown_series(nav_series(tw_base)),
        "exclude_v1 L2": drawdown_series(nav_series(tw_ex)),
        "大盤": drawdown_series(nav_series(taiex)),
    }, "03_tw_drawdown.png")
    chart_drawdown("美股", {
        "baseline L2": drawdown_series(nav_series(us_base)),
        "exclude_v1 L2": drawdown_series(nav_series(us_ex)),
        "大盤": drawdown_series(nav_series(sp500)),
    }, "04_us_drawdown.png")

    print(f"4張圖表已寫入 {OUT_DIR}")


if __name__ == "__main__":
    main()
