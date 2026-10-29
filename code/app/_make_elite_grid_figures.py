# -*- coding: utf-8 -*-
"""優等生權重網格：論文用圖表（2026-10-28，實戰收尾追蹤.md §四步驟1）。

讀 `_elite_grid_offline.py`／`_elite_grid_analyze.py` 的輸出，產出：
  圖1 熱力圖：排名依據 × 資金比例，平均 Sharpe 減 control0（兩期 × 四組）
  圖2 散點：2015-23 Sharpe vs 2024-25 Sharpe，1,040 組（四組）
  圖3 長條：2024-25 每季報酬（大盤／control0／市值傾斜全開／晉級第1名）
  圖4 折線：2024-25 累積淨值（同上＋其餘晉級組）
  表1 四組總表、表2 晉級組與對照組、表3 各維度平均效果（CSV＋Markdown）
全部扣成本後（市值傾斜全開、control0 一樣扣）；大盤為總報酬指數、不扣成本。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import pandas as pd  # noqa: E402

import json  # noqa: E402
from functools import lru_cache  # noqa: E402

from app._elite_grid_offline import OUT_DIR, load_benchmark as _load_benchmark  # noqa: E402

load_benchmark = lru_cache(maxsize=None)(_load_benchmark)   # 同一市場只查一次 DB

FIG_DIR = OUT_DIR / "figs"
#: --liq：美股改讀加了流動性門檻的結果（檔名 `_liq`），台股不變；輸出到 figs_liq，不覆蓋 figs。
SUFFIX = {"TW": "", "US": ""}


def tagof(market, pool):
    return f"{market}_{pool}{SUFFIX[market]}"
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "font.size": 10,
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#2b2b29", "xtick.color": "#55554f",
    "ytick.color": "#55554f", "axes.grid": False,
})
INK, MUTED, GRID = "#2b2b29", "#8a8a85", "#e4e3de"

GROUPS = [("TW", "exclude_v1"), ("TW", "baseline"), ("US", "exclude_v1"), ("US", "baseline")]
GROUP_LABEL = {("TW", "exclude_v1"): "台股・排除V1", ("TW", "baseline"): "台股・含V1",
               ("US", "exclude_v1"): "美股・排除V1", ("US", "baseline"): "美股・含V1"}
BENCH_LABEL = {"TW": "加權指數(TAIEX)", "US": "S&P500"}
FAMILY = ["S1", "S2", "S3", "S4", "S5", "S6"]
FAMILY_LABEL = {"S1": "票數", "S2": "因子分數", "S3": "追強", "S4": "押回彈", "S5": "追強＋押回彈", "S6": "市值"}
FAMILY_COLOR = {"S1": "#2a78d6", "S2": "#eb6834", "S3": "#1baf7a", "S4": "#eda100", "S5": "#e87ba4", "S6": "#008300"}
FAMILY_MARKER = {"S1": "o", "S2": "s", "S3": "^", "S4": "v", "S5": "D", "S6": "P"}
SCORE_LABEL = {"S1_votes": "票數", "S2_f1": "因子F1", "S2_f1f2": "因子F1+F2", "S6_mcap": "市值"}
for L in (1, 3, 6):
    SCORE_LABEL[f"S3_greedy_L{L}"] = f"追強{L}個月"
    SCORE_LABEL[f"S4_reversal_L{L}"] = f"押回彈{L}個月"
    SCORE_LABEL[f"S5_mixed_L{L}"] = f"追強＋押回彈{L}個月"
SCHEME_LABEL = {"equal": "平均分", "top1x2": "第一名加碼", "top2x2": "前兩名加碼",
                "proportional": "照分數比例", "tier": "階梯"}
REF_LABEL = {"REF_control0": "control0（現行，不調整）", "REF_stock_equal": "每檔等權",
             "REF_w2c_full": "市值傾斜全開（上限30%）"}
SERIES = [("bench", "大盤", "#55554f"), ("REF_control0", "control0", "#2a78d6"),
          ("REF_w2c_full", "市值傾斜全開", "#008300"), ("top1", "2015–23 晉級第1名", "#eb6834")]


def combo_label(c: str, row: pd.Series) -> str:
    if c in REF_LABEL:
        return REF_LABEL[c]
    n = int(row["n"])
    parts = [SCORE_LABEL[row["score"]], f"前{n}名"]
    if n > 1:
        parts.append(SCHEME_LABEL[row["scheme"]])
    parts.append(f"資金{int(round(row['sleeve'] * 100))}%")
    return "｜".join(parts)


def load(market, pool):
    tag = tagof(market, pool)
    t = pd.read_csv(OUT_DIR / f"metrics_all_{tag}.csv", index_col=0)
    t["label"] = [combo_label(c, r) for c, r in t.iterrows()]
    return t


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("bgr", ["#b8352e", "#e6928a", "#f0efec", "#86b6ef", "#1c5cab"])


# ───────────────────────── 圖1 ─────────────────────────

def fig_heatmap(data):
    sleeves = [0.1, 0.2, 0.3, 0.5, 1.0]
    fig, axes = plt.subplots(4, 2, figsize=(9.2, 12.6))
    for gi, g in enumerate(GROUPS):
        t = data[g]
        grid = t[~t.score.str.startswith("REF_")]
        for pj, (per, per_lbl) in enumerate([("IS", "2015–2023（挑選期）"), ("OOS", "2024–2025（驗證期）")]):
            ax = axes[gi, pj]
            base = t.loc["REF_control0", f"{per}_net_sharpe"]
            piv = (grid.pivot_table(index="family", columns="sleeve", values=f"{per}_net_sharpe", aggfunc="mean")
                   .reindex(index=FAMILY, columns=sleeves) - base)
            lim = max(0.05, float(np.nanmax(np.abs(piv.to_numpy()))))
            ax.imshow(piv.to_numpy(), cmap=diverging_cmap(), vmin=-lim, vmax=lim, aspect="auto")
            for i in range(len(FAMILY)):
                for j in range(len(sleeves)):
                    v = piv.iat[i, j]
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8.5,
                            color="white" if abs(v) > 0.6 * lim else INK)
            ax.set_xticks(range(len(sleeves)), [f"{int(s*100)}%" for s in sleeves])
            ax.set_yticks(range(len(FAMILY)), [FAMILY_LABEL[f] for f in FAMILY] if pj == 0 else [""] * 6)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{GROUP_LABEL[g]}｜{per_lbl}\ncontrol0 Sharpe＝{base:.2f}", fontsize=9.5, color=INK)
            if gi == 3:
                ax.set_xlabel("優等生資金比例")
    fig.suptitle("圖1　各排名依據 × 資金比例的平均 Sharpe（減 control0，扣成本）\n藍＝優於 control0、紅＝劣於 control0；每格為該排名依據下所有「取幾名 × 分法」組合的平均",
                 fontsize=10.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=1.2)
    fig.savefig(FIG_DIR / "fig1_heatmap_sharpe_vs_control0.png")
    plt.close(fig)


# ───────────────────────── 圖2 ─────────────────────────

def fig_scatter(data, rhos):
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.6))
    for ax, g in zip(axes.flat, GROUPS):
        t = data[g]
        grid = t[~t.score.str.startswith("REF_")]
        for f in FAMILY:
            s = grid[grid.family == f]
            ax.scatter(s.IS_net_sharpe, s.OOS_net_sharpe, s=11, alpha=0.55, color=FAMILY_COLOR[f],
                       marker=FAMILY_MARKER[f], linewidths=0, label=FAMILY_LABEL[f])
        pr = t[t.promoted == True]  # noqa: E712
        ax.scatter(pr.IS_net_sharpe, pr.OOS_net_sharpe, s=46, facecolors="none", edgecolors=INK,
                   linewidths=1.1, label="晉級組", zorder=4)
        c0 = t.loc["REF_control0"]
        ax.axvline(c0.IS_net_sharpe, color=MUTED, lw=0.8, ls="--", zorder=1)
        ax.axhline(c0.OOS_net_sharpe, color=MUTED, lw=0.8, ls="--", zorder=1)
        ax.scatter([c0.IS_net_sharpe], [c0.OOS_net_sharpe], marker="*", s=150, color=INK, zorder=5,
                   label="control0", edgecolors="white", linewidths=0.8)
        w = t.loc["REF_w2c_full"]
        ax.scatter([w.IS_net_sharpe], [w.OOS_net_sharpe], marker="X", s=90, color="#b8352e", zorder=5,
                   label="市值傾斜全開", edgecolors="white", linewidths=0.8)
        ax.set_title(f"{GROUP_LABEL[g]}　Spearman ρ＝{rhos[g]:+.2f}", fontsize=10, color=INK)
        ax.set_xlabel("2015–2023 Sharpe（挑選期）")
        ax.set_ylabel("2024–2025 Sharpe（驗證期）")
        ax.grid(color=GRID, lw=0.6)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=8.5, frameon=False, markerscale=1.4)
    fig.suptitle("圖2　1,040 組規則在挑選期與驗證期的 Sharpe（扣成本）\n虛線＝control0；ρ 為兩期排名的 Spearman 相關",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(FIG_DIR / "fig2_scatter_is_vs_oos.png")
    plt.close(fig)


# ───────────────────────── 圖3／圖4 ─────────────────────────

def oos_quarterly(market, pool, t):
    tag = tagof(market, pool)
    q = pd.read_parquet(OUT_DIR / f"quarterly_{tag}.parquet")
    q = q[q.window == 4].copy()
    q["ret_net"] = (1 - q.cost) * (1 + q.ret_gross) - 1
    top1 = t[t.promote_reason.fillna("").str.contains("整體前3")].sort_values("IS_net_sharpe", ascending=False).index[0]
    piv = q.pivot(index="end", columns="combo", values="ret_net")
    b = load_benchmark(market)
    ends = list(piv.index)
    starts = [q[q.end == e].as_of.iloc[0] for e in ends]
    br = [float(b[b.index <= pd.Timestamp(e)].iloc[-1] / b[b.index <= pd.Timestamp(s)].iloc[-1] - 1)
          for s, e in zip(starts, ends)]
    out = pd.DataFrame({"bench": br, "REF_control0": piv["REF_control0"], "REF_w2c_full": piv["REF_w2c_full"],
                        "top1": piv[top1]}, index=ends)
    return out, top1


def quarter_label(e):
    ts = pd.Timestamp(e)
    return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"


def fig_bars(data, qtabs):
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), sharey="row")
    width = 0.2
    for ax, g in zip(axes.flat, GROUPS):
        qt, top1 = qtabs[g]
        x = np.arange(len(qt))
        for k, (col, lbl, colr) in enumerate(SERIES):
            lbl2 = BENCH_LABEL[g[0]] if col == "bench" else lbl
            ax.bar(x + (k - 1.5) * width, qt[col] * 100, width * 0.88, color=colr, label=lbl2)
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.set_xticks(x, [quarter_label(e) for e in qt.index], fontsize=8.5)
        ax.set_ylabel("單季報酬（%）")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"{GROUP_LABEL[g]}\n晉級第1名：{data[g].loc[top1, 'label']}", fontsize=9.5, color=INK)
        ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper left")
    fig.suptitle("圖3　2024–2025 每季報酬（扣成本；大盤為總報酬指數）", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "fig3_oos_quarterly_bars.png")
    plt.close(fig)


def fig_cum(data, qtabs):
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6))
    for ax, g in zip(axes.flat, GROUPS):
        market, pool = g
        t = data[g]
        d = pd.read_parquet(OUT_DIR / f"daily_net_{tagof(market, pool)}_w4.parquet")
        cal = json.loads((OUT_DIR / f"meta_{tagof(market, pool)}.json").read_text(encoding="utf-8"))["windows"]["4"]["calendar"]
        as_of0, end_last = pd.Timestamp(cal[0][0]), pd.Timestamp(cal[-1][1])
        b = load_benchmark(market)
        anchor = b.index[b.index <= as_of0].max()          # 起點＝登記日前最後一個交易日，淨值=1
        d = pd.concat([pd.DataFrame(1.0, index=[anchor], columns=d.columns), d])
        top1 = qtabs[g][1]
        for c in t[t.promoted == True].index:  # noqa: E712
            if c != top1:
                ax.plot(d.index, (d[c] - 1) * 100, color="#c9c8c1", lw=0.9, zorder=1)
        ax.plot([], [], color="#c9c8c1", lw=0.9, label="其他晉級組")
        bs = b[(b.index >= anchor) & (b.index <= end_last)]
        ax.plot(bs.index, (bs / bs.iloc[0] - 1) * 100, color="#55554f", lw=1.6, ls="--", label=BENCH_LABEL[market])
        for col, lbl, colr in SERIES[1:]:
            cc = top1 if col == "top1" else col
            ax.plot(d.index, (d[cc] - 1) * 100, color=colr, lw=1.8, label=lbl)
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.set_ylabel("累積報酬（%）")
        ax.grid(color=GRID, lw=0.6)
        ax.set_title(GROUP_LABEL[g], fontsize=10, color=INK)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("圖4　2024–2025 累積報酬（日頻，扣成本）", fontsize=10.5, color=INK)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "fig4_oos_cumulative.png")
    plt.close(fig)


# ───────────────────────── 表 ─────────────────────────

def to_md(df: pd.DataFrame) -> str:
    """不依賴 tabulate 的簡易 Markdown 表格。"""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def pct(x):
    return f"{x * 100:+.1f}%"


def tables(data, rhos, sames):
    rows1 = []
    for g in GROUPS:
        t = data[g]
        pr = t[t.promoted == True]  # noqa: E712
        rows1.append({"組別": GROUP_LABEL[g], "Spearman ρ": f"{rhos[g]:+.3f}", "各維度同號比例": f"{sames[g]:.0%}",
                      "control0 2024–25": pct(t.loc["REF_control0", "OOS_net_cum"]),
                      "晉級組 2024–25": f"{pct(pr.OOS_net_cum.min())} ~ {pct(pr.OOS_net_cum.max())}",
                      "市值傾斜全開 2015–23 Sharpe": f"{t.loc['REF_w2c_full', 'IS_net_sharpe']:.2f}",
                      "市值傾斜全開 2024–25": pct(t.loc["REF_w2c_full", "OOS_net_cum"])})
    t1 = pd.DataFrame(rows1)

    rows2 = []
    for g in GROUPS:
        t = data[g]
        for c in list(t[t.promoted == True].sort_values("IS_net_sharpe", ascending=False).index) + list(REF_LABEL):  # noqa: E712
            r = t.loc[c]
            rows2.append({"組別": GROUP_LABEL[g], "規則": r.label,
                          "晉級理由": (r.promote_reason or "").replace("整體前3", "整體前3名").replace(";", " ").strip()
                          if isinstance(r.promote_reason, str) else "對照組",
                          "2015–23 Sharpe": f"{r.IS_net_sharpe:.2f}", "2015–23 累積": pct(r.IS_net_cum),
                          "2024–25 Sharpe": f"{r.OOS_net_sharpe:.2f}", "2024–25 累積": pct(r.OOS_net_cum),
                          "2024–25 MDD": pct(r.OOS_net_mdd), "季均換手": f"{r.turnover_OOS * 100:.0f}%",
                          "最大單股": f"{r.max_weight_peak_OOS * 100:.1f}%"})
    t2 = pd.DataFrame(rows2)
    for fam, lbl in FAMILY_LABEL.items():
        t2["晉級理由"] = t2["晉級理由"].str.replace(f"{fam}最佳", f"{lbl}最佳")

    rows3 = []
    dim_lbl = {"family": "排名依據", "n": "取幾名", "scheme": "分法", "sleeve": "資金比例"}
    for g in GROUPS:
        m = pd.read_csv(OUT_DIR / f"marginal_{tagof(*g)}.csv")
        m = m[m.dimension.isin(dim_lbl)]
        for r in m.itertuples():
            lv = r.level
            if r.dimension == "family":
                lv = FAMILY_LABEL[lv]
            elif r.dimension == "scheme":
                lv = SCHEME_LABEL[lv]
            elif r.dimension == "n":
                lv = f"前{int(float(lv))}名"
            else:
                lv = f"{int(round(float(lv) * 100))}%"
            rows3.append({"組別": GROUP_LABEL[g], "維度": dim_lbl[r.dimension], "水準": lv,
                          "2015–23": f"{r.IS_vs_grid_mean:+.3f}", "2024–25": f"{r.OOS_vs_grid_mean:+.3f}",
                          "兩期同號": "是" if r.same_sign else "否"})
    t3 = pd.DataFrame(rows3)

    for name, df, title in [("table1_summary", t1, "表1　四組總表（扣成本）"),
                            ("table2_promoted", t2, "表2　晉級組與對照組（扣成本）"),
                            ("table3_marginal", t3, "表3　各維度平均 Sharpe 減網格平均（扣成本）")]:
        df.to_csv(FIG_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
        (FIG_DIR / f"{name}.md").write_text(f"## {title}\n\n" + to_md(df) + "\n", encoding="utf-8")


def main():
    global FIG_DIR
    if "--liq" in sys.argv:
        SUFFIX["US"] = "_liq"
        FIG_DIR = OUT_DIR / "figs_liq"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data, rhos, sames, qtabs = {}, {}, {}, {}
    for g in GROUPS:
        t = load(*g)
        data[g] = t
        grid = t[~t.score.str.startswith("REF_")]
        rhos[g] = grid.IS_net_sharpe.rank().corr(grid.OOS_net_sharpe.rank())
        sames[g] = pd.read_csv(OUT_DIR / f"marginal_{tagof(*g)}.csv").same_sign.mean()
        qtabs[g] = oos_quarterly(*g, t)
    fig_heatmap(data)
    fig_scatter(data, rhos)
    fig_bars(data, qtabs)
    fig_cum(data, qtabs)
    tables(data, rhos, sames)
    print("輸出：", sorted(p.name for p in FIG_DIR.iterdir()))


if __name__ == "__main__":
    main()
