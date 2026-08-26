# -*- coding: utf-8 -*-
"""
Phase 3 的 4-16 圖：控制 F 之後，各 C 的 CAGR 折線（論文第四章 4-16 的對應版本）。

原始 4-16 的定義（見 analyze_batch.py）：
  X 軸 = 全部 F1 單層條件（跨因子混排、依中位數排序）
  每條彩色線 = 一個 C；黑色粗虛線 = None 基準
  直看＝在這個 F 區間下哪個 C 最好；橫看＝這個 C 是不是穩定有效

只取 12 個**單因子**組合（F2=None）→ 嚴格對應原始 4-16，X 軸標名稱。

用法：python phase3_fig416.py [--market TW]
"""
import re
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase3_analyze import parse                 # noqa: E402
from sweep_config import MARKET_START, date_range_suffix  # noqa: E402
from phase1_linearity import IN_SAMPLE_END        # noqa: E402

ART = HERE / "results_artifacts"
OUT = HERE.parent / "_analysis_outputs_phase3"

INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

# 每個 C 各給一個明顯不同的顏色：在色相環上均勻取樣（不用同色系分深淺，避免相近色難以區分）
QUAL_COLORS = [plt.cm.gist_rainbow(i / 20) for i in range(20)]


def pretty_f(name):
    """'PB_qb0of3__None' → 'PB 低'；'MOM_qb2of3__REVENUE_qb1of3' → 'MOM 高×REVENUE 中'"""
    lab = {0: "低", 1: "中", 2: "高"}
    def one(x):
        m = re.match(r"(.+)_qb(\d+)of(\d+)$", x)
        return f"{m.group(1)} {lab.get(int(m.group(2)), m.group(2))}" if m else x
    a, b = name.split("__")
    return one(a) if b == "None" else f"{one(a)}×{one(b)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="TW", choices=["TW", "US"])
    ap.add_argument("--start", default=None,
                    help="自訂起始日期 YYYY-MM-DD（需與 phase3_conditions.py 執行時相同）")
    ap.add_argument("--end", default=None,
                    help="自訂結束日期 YYYY-MM-DD（需與 phase3_conditions.py 執行時相同）")
    args = ap.parse_args()
    mkt = args.market
    start = args.start or MARKET_START[mkt]
    end = args.end or IN_SAMPLE_END
    rsfx = date_range_suffix(start, end, MARKET_START[mkt], IN_SAMPLE_END)
    label = f"{mkt}_L3_M{rsfx}"

    df = pd.read_parquet(ART / label / "stats.parquet")
    df[["F組合", "C", "C編號", "C來源"]] = df["strategy"].apply(lambda s: pd.Series(parse(s)))

    # 增益中位數由高到低排序 → 也決定畫線順序與顏色指派順序
    gain = (df[df["C"] != "None"]
            .merge(df[df["C"] == "None"][["F組合", "CAGR"]].rename(columns={"CAGR": "base"}),
                   on="F組合")
            .assign(g=lambda d: d["CAGR"] - d["base"])
            .groupby("C")["g"].median().sort_values(ascending=False))
    hi3, lo3 = list(gain.index[:3]), list(gain.index[-3:])
    color = {c: QUAL_COLORS[i % len(QUAL_COLORS)] for i, c in enumerate(gain.index)}

    sub = df[df["F組合"].str.endswith("__None")]
    base = (sub[sub["C"] == "None"].set_index("F組合")["CAGR"].sort_values(ascending=False))
    order = list(base.index)
    xs = np.arange(len(order))
    piv = sub.pivot_table(index="C", columns="F組合", values="CAGR").reindex(columns=order)

    fig, ax = plt.subplots(figsize=(14, 8))
    for c in gain.index:
        if c not in piv.index:
            continue
        mark = "★" if c in hi3 else ("▽" if c in lo3 else "　")
        emph = c in hi3 or c in lo3
        ax.plot(xs, piv.loc[c].values, color=color[c],
                linewidth=2.4 if emph else 1.3,
                alpha=1.0 if emph else 0.85, zorder=3 if emph else 2,
                label=f"{mark}{c.split('_DYN_')[0]} {c.split('_DYN_')[1]}"
                      f"（{gain[c]:+.1%}）")
    ax.plot(xs, base.values, color="black", linewidth=2.8, linestyle="--",
            label="None（不加C）基準", zorder=4)

    ax.set_ylabel("CAGR")
    ax.set_xticks(xs)
    ax.set_xticklabels([pretty_f(f) for f in order], rotation=70, fontsize=8)
    ax.set_xlabel("F 單層條件（依 None 基準 CAGR 排序）")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left",
              bbox_to_anchor=(1.005, 1.0), borderaxespad=0)

    fig.suptitle(f"4-16  控制 F 之後各動態因子 C 的 CAGR 折線 — Phase 3 / {mkt}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = OUT / f"{mkt}_L3_fig4-16{rsfx}_controlled_C_lines.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    print(f"輸出：{p}")
    print("\n增益中位數 前3：", ", ".join(hi3))
    print("增益中位數 後3：", ", ".join(lo3))


if __name__ == "__main__":
    main()
