# -*- coding: utf-8 -*-
"""
Phase 3 的 4-16 圖：控制 F 之後，各 C 的 CAGR 折線（論文第四章 4-16 的對應版本）。

原始 4-16 的定義（見 analyze_batch.py）：
  X 軸 = 全部 F1 單層條件（跨因子混排、依中位數排序）
  每條彩色線 = 一個 C；黑色粗虛線 = None 基準
  直看＝在這個 F 區間下哪個 C 最好；橫看＝這個 C 是不是穩定有效

Phase 3 的 F 構面是 203 個 F1×F2 組合，故畫兩個面板：
  左：只取 12 個**單因子**組合（F2=None）→ 嚴格對應原始 4-16，X 軸可標名稱
  右：全部 203 個 F 組合 → 驗證同樣型態在有 F2 的情況下是否成立

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
from phase1_linearity import IN_SAMPLE_END      # noqa: E402
from phase3_analyze import parse                 # noqa: E402

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

# C 依來源因子上色：ROE 藍系、EPS 綠系、FCF_P 橘紅系（一眼看出三群的分層）
SRC_CMAP = {"ROE": plt.cm.Blues, "EPS": plt.cm.Greens, "FCF_P": plt.cm.Oranges}


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
    args = ap.parse_args()
    mkt, label = args.market, f"{args.market}_L3_M"

    df = pd.read_parquet(ART / label / "stats.parquet")
    df[["F組合", "C", "C編號", "C來源"]] = df["strategy"].apply(lambda s: pd.Series(parse(s)))

    # 每個 C 一個顏色：同來源用同色系、由深到淺依編號排
    cs = df[df["C"] != "None"][["C編號", "C", "C來源"]].drop_duplicates().sort_values("C編號")
    color = {}
    for src, grp in cs.groupby("C來源"):
        cm = SRC_CMAP.get(src, plt.cm.Greys)
        for i, c in enumerate(grp["C"]):
            color[c] = cm(0.45 + 0.5 * i / max(1, len(grp) - 1))

    # 增益中位數最高／最低的各 3 個 C → 加粗標示
    gain = (df[df["C"] != "None"]
            .merge(df[df["C"] == "None"][["F組合", "CAGR"]].rename(columns={"CAGR": "base"}),
                   on="F組合")
            .assign(g=lambda d: d["CAGR"] - d["base"])
            .groupby("C")["g"].median().sort_values(ascending=False))
    hi3, lo3 = list(gain.index[:3]), list(gain.index[-3:])

    fig, axes = plt.subplots(1, 2, figsize=(26, 8.5),
                             gridspec_kw={"width_ratios": [1, 1.2]})

    for ax, only_single in zip(axes, [True, False]):
        sub = df.copy()
        if only_single:
            sub = sub[sub["F組合"].str.endswith("__None")]
        # X 軸：F 組合依「None 基準 CAGR」由高到低排序
        base = (sub[sub["C"] == "None"].set_index("F組合")["CAGR"]
                .sort_values(ascending=False))
        order = list(base.index)
        xs = np.arange(len(order))

        piv = sub.pivot_table(index="C", columns="F組合", values="CAGR").reindex(columns=order)

        # 20 條 C 全部標名進圖例（依增益排序，圖例順序＝強弱順序）；
        # 前3/後3 再用粗線與 ★/▽ 記號額外突顯。
        for c in gain.index:
            if c not in piv.index:
                continue
            mark = "★" if c in hi3 else ("▽" if c in lo3 else "　")
            emph = c in hi3 or c in lo3
            ax.plot(xs, piv.loc[c].values, color=color.get(c, MUTED),
                    linewidth=2.4 if emph else 1.1,
                    alpha=1.0 if emph else 0.75, zorder=3 if emph else 2,
                    label=f"{mark}{c.split('_DYN_')[0]} {c.split('_DYN_')[1]}"
                          f"（{gain[c]:+.1%}）")
        ax.plot(xs, base.values, color="black", linewidth=2.8, linestyle="--",
                label="None（不加C）基準", zorder=4)

        ax.set_ylabel("CAGR")
        if only_single:
            ax.set_xticks(xs)
            ax.set_xticklabels([pretty_f(f) for f in order], rotation=70, fontsize=8)
            ax.set_xlabel("F 單層條件（依 None 基準 CAGR 排序）")
            ax.set_title(f"(左) 嚴格對應 4-16：只取 {len(order)} 個單因子 F 組合", fontsize=11)
        else:
            ax.set_xticks([])
            ax.set_xlabel(f"全部 {len(order)} 個 F1×F2 組合（依 None 基準 CAGR 由高到低）")
            ax.set_title(f"(右) 擴充版：全部 {len(order)} 個 F 組合，驗證型態是否一致", fontsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        # 圖例移到圖外右側，避免蓋住折線；21 條全列
        ax.legend(frameon=False, fontsize=7.5, loc="upper left",
                  bbox_to_anchor=(1.005, 1.0), borderaxespad=0)

    fig.suptitle(
        f"4-16  控制 F 之後各動態因子 C 的 CAGR 折線 — Phase 3 / {mkt}"
        f"（in-sample 至 {IN_SAMPLE_END}，V 關閉）\n"
        f"20 條 C 全部繪出並列入圖例（依增益中位數由高到低排序，括號內為增益）；"
        f"顏色＝來源因子（藍 ROE｜綠 EPS｜橘 FCF_P）；★＝前3、▽＝後3；黑虛線＝不加 C 的基準",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    p = OUT / f"{mkt}_L3_fig4-16_controlled_C_lines.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    print(f"輸出：{p}")
    print("\n增益中位數 前3：", ", ".join(hi3))
    print("增益中位數 後3：", ", ".join(lo3))


if __name__ == "__main__":
    main()
