# -*- coding: utf-8 -*-
"""
單一因子候選批次的第四章 20 張圖分析（依學姊論文原文逐節核對過的版本）。

跟 analyze_job.py 的差異（analyze_job.py 是給舊的巨池 job 用，不動它）：
- 批次規模比照論文（~7,140策略/批，單一N=5，不需要N拆分）
- 4-13/4-17：單一張「全部F1(混排4因子)×全部F2(含None)」的完整熱力圖，不是拆因子對的小圖
- 4-16：X軸是全部F1單層條件混排一條軸（依中位數排序），不是分因子子圖+桶編號軸
- 4-9 累積貢獻曲線：只累加至前20檔個股
- Top1/EffN/年度貢獻：改用「逐股累積正報酬」（(1+r).groupby(stock_id).cumprod()-1，取最後一筆），
  對應 report_grouping.py::aggregate_stock_contributions(metric="cum") 的既有方法，不用簡化的 .sum()

用法（cwd=code/）：
  ../.venv/Scripts/python.exe analyze_batch.py --label TW_batch_PE_M
  ../.venv/Scripts/python.exe analyze_batch.py --label TW_batch_PE_M --skip-credibility
"""
import re
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ==================== 風格 ====================
OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
})


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, color=INK, pad=8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load_strategy_artifacts(strat_dir: Path) -> dict:
    """直接讀 parquet，不用 io_persistence.load_strategy_artifacts（該函式會 sanitize 掉絕對路徑的磁碟代號）。"""
    out = {}
    for name in ("trades", "position", "stock_data", "return_table"):
        p = strat_dir / f"{name}.parquet"
        out[name] = pd.read_parquet(p) if p.exists() else None
    return out


# ==================== 策略名解析 ====================
def parse_bucket(label: str) -> dict:
    if label == "None":
        return {"factor": "None", "k": np.nan}
    m = re.match(r"(.+)_qb(\d+)of\d+$", label)
    if m:
        return {"factor": m.group(1), "k": int(m.group(2))}
    return {"factor": label, "k": np.nan}


def display_bucket(name: str, with_factor: bool = True) -> str:
    """把 qbKofN 轉成人類可讀的百分位區間（僅供圖表顯示，底層策略命名/邏輯不受影響）。
    例：ROE_qb4of5 -> 'ROE 80-100%(最高)'；qb0of5 -> '0-20%(最低)'。"""
    if name == "None":
        return "None"
    m = re.match(r"(.+)_qb(\d+)of(\d+)$", name)
    if not m:
        return name
    factor, k, n = m.group(1), int(m.group(2)), int(m.group(3))
    lo, hi = round(100 * k / n), round(100 * (k + 1) / n)
    tag = "(最低)" if k == 0 else "(最高)" if k == n - 1 else ""
    pct = f"{lo}-{hi}%{tag}"
    return f"{factor} {pct}" if with_factor else pct


def display_strategy_name(name: str) -> str:
    """把完整策略名（F1__F2__C__V）裡的 F1/F2 桶名轉成百分位區間，C/V 部分保留原樣。"""
    parts = str(name).split("__")
    out = [display_bucket(p) if i < 2 else p for i, p in enumerate(parts)]
    return " | ".join(out)


def parse_name(name: str) -> dict:
    parts = str(name).split("__")
    parts += ["None"] * (4 - len(parts))
    F1, F2, C, V = parts[0], parts[1], parts[2], parts[3]
    b1, b2 = parse_bucket(F1), parse_bucket(F2)
    c_kind = "None" if C == "None" else C.split("_")[0]
    return {
        "F1": F1, "F2": F2, "C": C, "V": V,
        "F1_factor": b1["factor"], "F1_k": b1["k"],
        "F2_factor": b2["factor"], "F2_k": b2["k"],
        "C_kind": c_kind,
        "is_pair": F2 != "None",
        "base": "__".join(parts[:3]),
    }


def load_stats(label: str) -> pd.DataFrame:
    p = HERE / "results_artifacts" / label / "stats.parquet"
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}，請先跑完 run_factor_batches.py")
    df = pd.read_parquet(p)
    meta = df["strategy"].apply(parse_name).apply(pd.Series)
    df = pd.concat([df, meta], axis=1)
    for c in ("CAGR", "sharpe_ann", "max_drawdown", "avg_drawdown", "win_ratio", "ytd"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _cnum(c):
    if c == "None":
        return -1
    m = re.match(r"C(\d+)", c)
    return int(m.group(1)) if m else 999


def ordered_c_kinds(df):
    return sorted(df["C_kind"].unique(), key=_cnum)


def factor_list(df):
    return sorted(df.loc[~df["is_pair"], "F1_factor"].unique())


def f1_median_order(df):
    """單層(F2=None)CAGR中位數由高至低排序的F1條件清單（供4-12/4-16共用的統一排序）。"""
    single = df[~df["is_pair"]]
    med = single.groupby("F1")["CAGR"].median().sort_values(ascending=False)
    return med.index.tolist()


# ==================== V0/V1 配對 ====================
def build_v_pairs(df: pd.DataFrame) -> pd.DataFrame:
    v0 = df[df["V"] == "v0"].set_index("base")
    v1 = df[df["V"] == "v1"].set_index("base")
    common = v0.index.intersection(v1.index)
    v0, v1 = v0.loc[common], v1.loc[common]
    out = pd.DataFrame({
        "base": common,
        "F1": v0["F1"].values, "F2": v0["F2"].values, "C": v0["C"].values,
        "F1_factor": v0["F1_factor"].values, "C_kind": v0["C_kind"].values,
        "CAGR_v0": v0["CAGR"].values, "CAGR_v1": v1["CAGR"].values,
        "MDD_v0": v0["max_drawdown"].values, "MDD_v1": v1["max_drawdown"].values,
    })
    out["dCAGR"] = out["CAGR_v1"] - out["CAGR_v0"]
    out["dMDD"] = out["MDD_v0"].abs() - out["MDD_v1"].abs()
    return out


# ==================== 逐股累積正報酬（對應 report_grouping.py::aggregate_stock_contributions(metric="cum")） ====================
def stock_cum_contrib(trades: pd.DataFrame) -> pd.Series:
    """逐股 (1+r) 連乘 - 1，取每檔股票最後一筆，得該策略內每檔股票的累積報酬。"""
    t = trades.copy()
    t["_cum"] = (1.0 + t["return"]).groupby(t["stock_id"]).cumprod() - 1.0
    return t.groupby("stock_id")["_cum"].last()


def strategy_credibility(label: str, strategy: str) -> dict:
    art = load_strategy_artifacts(HERE / "results_artifacts" / label / strategy)
    t = art.get("trades")
    if t is None or len(t) == 0 or "stock_id" not in t.columns:
        return {"top1_stock": None, "top1_share": np.nan, "effective_n": np.nan, "cum_shares20": None}
    contrib = stock_cum_contrib(t)
    pos = contrib[contrib > 0]
    if len(pos) == 0 or pos.sum() <= 0:
        return {"top1_stock": None, "top1_share": np.nan, "effective_n": np.nan, "cum_shares20": None}
    shares = (pos / pos.sum()).sort_values(ascending=False)
    hhi = float((shares.values ** 2).sum())
    cum20 = shares.cumsum().values[:20]  # 論文4-9：只累加至前20檔個股
    return {
        "top1_stock": str(shares.index[0]),
        "top1_share": float(shares.iloc[0]),
        "effective_n": (1.0 / hhi) if hhi > 0 else np.nan,
        "cum_shares20": cum20,
    }


def compute_credibility_metrics(df, label, subset_mask, tag, log=print) -> pd.DataFrame:
    sub = df.loc[subset_mask, "strategy"].tolist()
    log(f"   [credibility:{tag}] 掃 {len(sub)} 個策略的 trades.parquet …")
    t0 = time.time()
    rows = []
    for i, name in enumerate(sub, 1):
        r = strategy_credibility(label, name)
        r["strategy"] = name
        rows.append(r)
        if i % 1000 == 0:
            log(f"      {i}/{len(sub)}（{time.time()-t0:.0f}s）")
    log(f"   [credibility:{tag}] 完成，{time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


# ==================== 4-2 Top10 熱力圖 ====================
def fig_4_2_top10(df, out_dir, label):
    cols = ["CAGR", "sharpe_ann", "max_drawdown", "win_ratio"]
    top = df.sort_values("CAGR", ascending=False).drop_duplicates(subset=cols).head(10)
    M = top[cols].to_numpy(dtype=float)
    Mz = (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-12)
    names = [n[:40] + ("…" if len(n) > 40 else "") for n in top["strategy"]]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(Mz, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7.5, color=INK)
    _style(ax, f"4-2  Top10 策略績效熱力圖 — {label}\n(色階=組內z-score，數字=原始值)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "4-2_top10_heatmap.png", bbox_inches="tight")
    plt.close(fig)


# ==================== 4-3 / 4-4 代表策略解剖 ====================
def _drawdown_from_cum(cum: pd.Series) -> pd.Series:
    roll_max = cum.cummax()
    return cum / roll_max - 1.0


def fig_4_3_representative(df, out_dir, label, rep_name):
    art = load_strategy_artifacts(HERE / "results_artifacts" / label / rep_name)
    sd = art.get("stock_data")
    rt = art.get("return_table")
    if sd is None:
        return
    cum = sd["cum_returns"].dropna()
    cum = cum[cum > 0]
    dd = _drawdown_from_cum(cum)

    fig, axes = plt.subplots(4, 1, figsize=(9, 11))
    axes[0].plot(cum.index, cum.values, color=OI[0], linewidth=1.4)
    axes[0].set_yscale("log")
    _style(axes[0], "淨值曲線 (log)", "", "cum return")

    axes[1].fill_between(dd.index, dd.values, 0, color=OI[1], alpha=0.5, linewidth=0)
    _style(axes[1], "回撤曲線", "", "drawdown")

    if rt is not None:
        months = [c for c in ("1","2","3","4","5","6","7","8","9","10","11","12") if c in rt.columns]
        m = pd.Series(rt[months].to_numpy(dtype=float).flatten())
        axes[2].bar(range(len(m)), m.values * 100, color=OI[2], width=0.9)
        _style(axes[2], "月報酬 (%)", "month index", "%")
    else:
        axes[2].axis("off")

    if "company_count" in sd.columns:
        cc = sd["company_count"].dropna()
        axes[3].plot(cc.index, cc.values, color=OI[3], linewidth=1.0)
        _style(axes[3], "持股數變化", "date", "count")
    else:
        axes[3].axis("off")

    disp = display_strategy_name(rep_name)
    short = disp if len(disp) <= 90 else disp[:87] + "…"
    fig.suptitle(f"4-3  單一策略解剖 — {label}\n{short}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "4-3_representative_dissection.png", bbox_inches="tight")
    plt.close(fig)


def fig_4_4_annual_contribution(df, out_dir, label, rep_name, top_n=10):
    art = load_strategy_artifacts(HERE / "results_artifacts" / label / rep_name)
    t = art.get("trades")
    if t is None or len(t) == 0:
        return
    t = t.copy()
    t["entry_date"] = pd.to_datetime(t["entry_date"], errors="coerce")
    t["year"] = t["entry_date"].dt.year
    t["_cum"] = (1.0 + t["return"]).groupby([t["stock_id"], t["year"]]).cumprod() - 1.0
    contrib = t.groupby(["year", "stock_id"])["_cum"].last().unstack(fill_value=0.0)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    years = contrib.index.tolist()
    x = np.arange(len(years))
    for yi, year in enumerate(years):
        row = contrib.loc[year].sort_values(ascending=False)
        pos = row[row > 0]
        if len(pos) > top_n:
            pos = pd.concat([pos.head(top_n), pd.Series({"其他": pos.iloc[top_n:].sum()})])
        base = 0.0
        for k, (stock, val) in enumerate(pos.items()):
            ax.bar(x[yi], val, bottom=base, color=OI[k % len(OI)], width=0.7, edgecolor="white", linewidth=0.3)
            base += val
        neg = row[row < 0]
        base = 0.0
        for stock, val in neg.items():
            ax.bar(x[yi], val, bottom=base, color=MUTED, width=0.7, alpha=0.5)
            base += val
    ax.set_xticks(x); ax.set_xticklabels(years, rotation=45, fontsize=8)
    ax.axhline(0, color=INK, linewidth=0.8)
    disp = display_strategy_name(rep_name)
    short = disp if len(disp) <= 80 else disp[:77] + "…"
    _style(ax, f"4-4  年度個股貢獻分布（逐股累積正報酬）— {label}\n{short}", "year", "cumulative contribution")
    fig.tight_layout()
    fig.savefig(out_dir / "4-4_annual_contribution.png", bbox_inches="tight")
    plt.close(fig)


# ==================== 4-5~4-10 可信度 ====================
def fig_credibility_suite(cred25, cred50, out_dir, label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (cred, tag) in zip(axes, [(cred25, "top25%"), (cred50, "top50%")]):
        vc = cred["top1_stock"].dropna().value_counts().head(15)
        ax.bar(range(len(vc)), vc.values, color=OI[0])
        ax.set_xticks(range(len(vc))); ax.set_xticklabels(vc.index, rotation=60, fontsize=7)
        _style(ax, f"4-5/4-7  Top1貢獻個股出現頻率 — {tag}", "stock_id", "count")
    fig.suptitle(f"{label}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "4-5_4-7_top1_frequency.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (cred, tag) in zip(axes, [(cred25, "top25%"), (cred50, "top50%")]):
        vals = cred["top1_share"].dropna()
        ax.hist(vals, bins=40, color=OI[1], edgecolor="white", linewidth=0.3)
        med = vals.median()
        ax.axvline(med, color=OI[0], linewidth=1.6, label=f"median {med:.3f}")
        ax.legend(frameon=False, fontsize=8)
        _style(ax, f"4-6/4-8  Top1貢獻占比 — {tag}", "top1 share", "count")
    fig.suptitle(f"{label}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "4-6_4-8_top1_share.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for (cred, tag), color in zip([(cred25, "top25%"), (cred50, "top50%")], [OI[0], OI[1]]):
        curves = [c for c in cred["cum_shares20"] if c is not None and len(c) > 0]
        if not curves:
            continue
        padded = np.full((len(curves), 20), np.nan)
        for i, c in enumerate(curves):
            padded[i, :len(c)] = c
        med = np.nanmedian(padded, axis=0)
        q25 = np.nanpercentile(padded, 25, axis=0)
        q75 = np.nanpercentile(padded, 75, axis=0)
        xs = np.arange(1, 21)
        ax.plot(xs, med, color=color, label=f"{tag} median", linewidth=1.6)
        ax.fill_between(xs, q25, q75, color=color, alpha=0.15)
    _style(ax, f"4-9  累積獲利貢獻曲線（前20檔個股）— {label}", "個股排名(依貢獻由高至低，至多20檔)", "累積占比")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "4-9_cumulative_contribution.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    data = [cred25["effective_n"].dropna().values, cred50["effective_n"].dropna().values]
    bp = ax.boxplot(data, labels=[f"top25%\n(n={len(data[0])})", f"top50%\n(n={len(data[1])})"],
                    showmeans=True, patch_artist=True, medianprops=dict(color=INK, linewidth=1.5))
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=OI[i], alpha=0.35)
    _style(ax, f"4-10  Effective N 盒鬚圖 — {label}", "", "Effective N")
    fig.tight_layout()
    fig.savefig(out_dir / "4-10_effective_n.png", bbox_inches="tight")
    plt.close(fig)


# ==================== 4-12 單層F CAGR盒鬚圖（依中位數排序） ====================
def fig_4_12(df, out_dir, label):
    order = f1_median_order(df)
    single = df[~df["is_pair"]]
    data = [single.loc[single["F1"] == f1, "CAGR"].dropna().values for f1 in order]
    fig, ax = plt.subplots(figsize=(max(10, 0.32 * len(order)), 5))
    bp = ax.boxplot(data, labels=[display_bucket(f) for f in order], showmeans=True, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.3))
    colors = {f: OI[i % len(OI)] for i, f in enumerate(factor_list(df))}
    for box, f1 in zip(bp["boxes"], order):
        box.set(facecolor=colors[parse_bucket(f1)["factor"]], alpha=0.4)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.tick_params(axis="x", labelrotation=75, labelsize=7)
    _style(ax, f"4-12  單層體質因子 CAGR 分布盒鬚圖（依中位數排序）— {label}", "F1 區間", "CAGR")
    fig.tight_layout()
    fig.savefig(out_dir / "4-12_factor_boxplot.png", bbox_inches="tight")
    plt.close(fig)
    return order


# ==================== 4-13 F1×F2 統一熱力圖（含None欄） ====================
def _heatmap_grid(ax, M, center, fmt="{:.2f}", fontsize=6):
    lo, hi = np.nanmin(M), np.nanmax(M)
    lo = min(lo, center - 1e-9); hi = max(hi, center + 1e-9)
    norm = TwoSlopeNorm(vmin=lo, vcenter=center, vmax=hi)
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=fontsize, color=INK)
    return im


def fig_4_13(df, out_dir, label, f1_order):
    f2_order = f1_order + ["None"]
    sub = df[(df["C_kind"] == "None") & (df["V"] == "v0")]
    M = np.full((len(f1_order), len(f2_order)), np.nan)
    for i, r in enumerate(f1_order):
        for j, c in enumerate(f2_order):
            v = sub.loc[(sub["F1"] == r) & (sub["F2"] == c), "CAGR"]
            if len(v):
                M[i, j] = v.iloc[0]
    center = float(np.nanmedian(M))
    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(f2_order)), max(8, 0.32 * len(f1_order))))
    im = _heatmap_grid(ax, M, center, fontsize=5.5)
    ax.set_xticks(range(len(f2_order))); ax.set_xticklabels([display_bucket(x) for x in f2_order], rotation=75, fontsize=6)
    ax.set_yticks(range(len(f1_order))); ax.set_yticklabels([display_bucket(x) for x in f1_order], fontsize=6)
    _style(ax, f"4-13  F[1]×F[2] 雙層體質因子平均CAGR熱力圖（C=None, V=v0）— {label}\n"
               "白格＝同因子自配(展開規則排除)", "F[2] 區間（含None）", "F[1] 區間")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "4-13_F1xF2_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    return f1_order, f2_order


# ==================== 4-15 不控制F的C盒鬚圖 ====================
def fig_4_15(df, out_dir, label):
    order = ordered_c_kinds(df)
    data, labels = [], []
    for c in order:
        vals = df.loc[df["C_kind"] == c, "CAGR"].dropna().values
        if len(vals):
            data.append(vals); labels.append(f"{c}\n(n={len(vals)})")
    fig, ax = plt.subplots(figsize=(max(9, 0.5 * len(data)), 4.8))
    bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.3),
                    flierprops=dict(marker="o", markersize=2, alpha=0.25))
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=OI[i % len(OI)], alpha=0.3)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    _style(ax, f"4-15  不控制F的動態因子C盒鬚圖（反例）— {label}", "C group", "CAGR")
    ax.tick_params(axis="x", labelsize=6.5)
    fig.tight_layout()
    fig.savefig(out_dir / "4-15_uncontrolled_C.png", bbox_inches="tight")
    plt.close(fig)


# ==================== 4-16 主入口：統一F軸的C折線 ====================
def fig_4_16(df, out_dir, label, f1_order):
    single = df[~df["is_pair"]]
    c_order = ordered_c_kinds(single)
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(max(12, 0.32 * len(f1_order)), 6))
    x = np.arange(len(f1_order))
    for ci, c in enumerate(c_order):
        ys = [single.loc[(single["F1"] == f1) & (single["C_kind"] == c), "CAGR"].mean() for f1 in f1_order]
        if c == "None":
            ax.plot(x, ys, color="black", linewidth=2.4, linestyle="--", label="None(baseline)", zorder=5)
        else:
            ax.plot(x, ys, color=cmap(ci % 20), linewidth=1.1, alpha=0.85, label=c)
    ax.set_xticks(x); ax.set_xticklabels([display_bucket(f) for f in f1_order], rotation=75, fontsize=6.5)
    ax.legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    _style(ax, f"4-16  單層體質因子固定後各動態因子C平均CAGR折線（主入口）— {label}\n"
               "X軸＝全部F1單層條件(依中位數排序，跨因子混排)；直看找強F，橫看(追一條C線)找穩C",
           "F1 區間（依CAGR中位數排序）", "mean CAGR")
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(out_dir / "4-16_controlled_C_lines.png", bbox_inches="tight")
    plt.close(fig)


# ==================== 4-17 各C的F1×F2熱力圖小圖組 ====================
def fig_4_17(df, out_dir, label, f1_order):
    f2_order = f1_order + ["None"]
    c_order = ordered_c_kinds(df)  # 含 "None"（= 4-13 本身，一併放入方便對照）
    ncols = 4
    nrows = int(np.ceil(len(c_order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    for idx, c in enumerate(c_order):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[(df["C_kind"] == c) & (df["V"] == "v0")]
        M = np.full((len(f1_order), len(f2_order)), np.nan)
        for i, r in enumerate(f1_order):
            for j, c2 in enumerate(f2_order):
                v = sub.loc[(sub["F1"] == r) & (sub["F2"] == c2), "CAGR"]
                if len(v):
                    M[i, j] = v.iloc[0]
        if np.all(np.isnan(M)):
            ax.axis("off"); continue
        center = float(np.nanmedian(M))
        _heatmap_grid(ax, M, center, fmt="{:.1f}", fontsize=3.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(c, fontsize=8)
    for idx in range(len(c_order), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle(f"4-17  F[1]×F[2]雙層體質組合於各動態因子C下之平均CAGR熱力圖 — {label}\n"
                 "(各小圖同4-13座標軸：縱=F1、橫=F2含None；僅標題文字，軸標籤省略以節省空間)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "4-17_F1xF2_by_C_grid.png", bbox_inches="tight")
    plt.close(fig)


# ==================== 4-18 自動選子樹 ====================
def fig_4_18(df, out_dir, label, f1_order):
    best_bucket, worst_bucket = f1_order[0], f1_order[-1]
    single = df[~df["is_pair"]]
    order = ordered_c_kinds(single)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, bucket, tag in zip(axes, [best_bucket, worst_bucket], ["最佳F桶(中位數CAGR最高)", "最差F桶(中位數CAGR最低)"]):
        data, labels = [], []
        for c in order:
            vals = single.loc[(single["F1"] == bucket) & (single["C_kind"] == c), "CAGR"].dropna().values
            if len(vals):
                data.append(vals); labels.append(c)
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                        medianprops=dict(color=INK, linewidth=1.3))
        for i, box in enumerate(bp["boxes"]):
            box.set(facecolor=OI[i % len(OI)], alpha=0.3)
        ax.tick_params(axis="x", labelsize=6, labelrotation=75)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        _style(ax, f"{tag}\nF1={display_bucket(bucket)}", "C group", "CAGR")
    fig.suptitle(f"4-18  固定不同子樹條件之C分布對照（自動選桶）— {label}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_dir / "4-18_subtree_C_distribution.png", bbox_inches="tight")
    plt.close(fig)
    return best_bucket, worst_bucket


# ==================== 4-20/4-21/4-22/4-23/4-24 ====================
def fig_4_20_4_21(vpairs, out_dir, label):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    vals = vpairs["dCAGR"].dropna()
    ax.hist(vals, bins=60, color=OI[0], edgecolor="white", linewidth=0.3)
    med = vals.median()
    ax.axvline(med, color=OI[1], linewidth=1.8, label=f"median {med:.3f}")
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False)
    _style(ax, f"4-20  ΔCAGR (v1-v0) 分布 — {label}", "ΔCAGR", "count")
    fig.tight_layout()
    fig.savefig(out_dir / "4-20_dCAGR_distribution.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(vpairs["dMDD"], vpairs["dCAGR"], s=6, alpha=0.15, color=OI[0], edgecolor="none")
    ax.axhline(0, color=MUTED, linewidth=0.8); ax.axvline(0, color=MUTED, linewidth=0.8)
    q1 = ((vpairs["dMDD"] > 0) & (vpairs["dCAGR"] > 0)).mean()
    q4 = ((vpairs["dMDD"] > 0) & (vpairs["dCAGR"] < 0)).mean()
    q2 = ((vpairs["dMDD"] < 0) & (vpairs["dCAGR"] > 0)).mean()
    q3 = ((vpairs["dMDD"] < 0) & (vpairs["dCAGR"] < 0)).mean()
    ax.text(0.98, 0.98, f"報酬↑風險↓ {q1:.0%}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.text(0.02, 0.98, f"報酬↑風險↑ {q2:.0%}", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.text(0.98, 0.02, f"報酬↓風險↓(favorable) {q4:.0%}", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=OI[2])
    ax.text(0.02, 0.02, f"報酬↓風險↑(worst) {q3:.0%}", transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=OI[1])
    _style(ax, f"4-21  ΔCAGR vs ΔMDD 四象限散布圖 — {label}", "ΔMDD (+=風險降)", "ΔCAGR (+=報酬升)")
    fig.tight_layout()
    fig.savefig(out_dir / "4-21_dCAGR_dMDD_quadrant.png", bbox_inches="tight")
    plt.close(fig)


def fig_4_22(vpairs, out_dir, label):
    single = vpairs[vpairs["F1"].isin(vpairs["F1"].unique())]  # F2為None的已在build_v_pairs前過濾在stats層級
    factors = sorted(vpairs["F1_factor"].unique())
    fig, axes = plt.subplots(1, len(factors), figsize=(5 * len(factors), 4.6), sharey=True)
    if len(factors) == 1:
        axes = [axes]
    for ax, factor in zip(axes, factors):
        s = vpairs[vpairs["F1_factor"] == factor]
        buckets = sorted(s["F1"].unique(), key=lambda x: parse_bucket(x)["k"])
        data0 = [s.loc[s["F1"] == b, "CAGR_v0"].dropna().values for b in buckets]
        data1 = [s.loc[s["F1"] == b, "CAGR_v1"].dropna().values for b in buckets]
        x = np.arange(len(buckets))
        ax.boxplot(data0, positions=x - 0.18, widths=0.3, patch_artist=True,
                  boxprops=dict(facecolor=OI[0], alpha=0.4), medianprops=dict(color=INK))
        ax.boxplot(data1, positions=x + 0.18, widths=0.3, patch_artist=True,
                  boxprops=dict(facecolor=OI[1], alpha=0.4), medianprops=dict(color=INK))
        ax.set_xticks(x); ax.set_xticklabels([display_bucket(b, with_factor=False) for b in buckets], fontsize=7)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        _style(ax, factor, "F1 bucket", "CAGR" if factor == factors[0] else "")
    fig.suptitle(f"4-22  V0(藍) vs V1(橘) CAGR 依F1桶分組 — {label}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_dir / "4-22_V0_V1_by_F_bucket.png", bbox_inches="tight")
    plt.close(fig)


def fig_4_23(vpairs, out_dir, label):
    order = sorted(vpairs["C_kind"].unique(), key=_cnum)
    data0, data1, labels = [], [], []
    for c in order:
        d0 = vpairs.loc[vpairs["C_kind"] == c, "CAGR_v0"].dropna().values
        d1 = vpairs.loc[vpairs["C_kind"] == c, "CAGR_v1"].dropna().values
        if len(d0) and len(d1):
            data0.append(d0); data1.append(d1); labels.append(c)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(9, 0.5 * len(labels)), 4.8))
    ax.boxplot(data0, positions=x - 0.18, widths=0.32, patch_artist=True,
              boxprops=dict(facecolor=OI[0], alpha=0.4), medianprops=dict(color=INK))
    ax.boxplot(data1, positions=x + 0.18, widths=0.32, patch_artist=True,
              boxprops=dict(facecolor=OI[1], alpha=0.4), medianprops=dict(color=INK))
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, fontsize=6.5)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    _style(ax, f"4-23  V0(藍) vs V1(橘) CAGR 依動態因子C分組 — {label}", "C group", "CAGR")
    fig.tight_layout()
    fig.savefig(out_dir / "4-23_V0_V1_by_C.png", bbox_inches="tight")
    plt.close(fig)


def fig_4_24(vpairs, out_dir, label, best_bucket):
    sub_v = vpairs[vpairs["F1"] == best_bucket]
    order = sorted(sub_v["C_kind"].unique(), key=_cnum)
    m0 = [sub_v.loc[sub_v["C_kind"] == c, "CAGR_v0"].mean() for c in order]
    m1 = [sub_v.loc[sub_v["C_kind"] == c, "CAGR_v1"].mean() for c in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(max(9, 0.5 * len(order)), 4.6))
    w = 0.35
    ax.bar(x - w/2, m0, width=w, color=OI[0], label="v0")
    ax.bar(x + w/2, m1, width=w, color=OI[1], label="v1")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=60, fontsize=6.5)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False)
    _style(ax, f"4-24  指定子樹(F1={display_bucket(best_bucket)})下 V0 vs V1 逐C對照 — {label}", "C group", "mean CAGR")
    fig.tight_layout()
    fig.savefig(out_dir / "4-24_subtree_V0_V1_by_C.png", bbox_inches="tight")
    plt.close(fig)


# ==================== main ====================
def main():
    ap = argparse.ArgumentParser(description="因子候選批次的第四章圖鑑分析")
    ap.add_argument("--label", required=True, help="批次label，如 TW_batch_PE_M")
    ap.add_argument("--skip-credibility", action="store_true")
    args = ap.parse_args()

    label = args.label
    out_dir = ROOT / "_analysis_outputs" / label / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f">> [{label}] 讀 stats.parquet …")
    df = load_stats(label)
    print(f"   策略數={len(df)}｜pair={(df['is_pair']).sum()}｜single={(~df['is_pair']).sum()}｜因子={factor_list(df)}")

    print(">> 4-2 Top10 熱力圖")
    fig_4_2_top10(df, out_dir, label)

    rep_name = df.sort_values("CAGR", ascending=False).iloc[0]["strategy"]
    print(f">> 4-3/4-4 代表策略解剖（{rep_name}）")
    fig_4_3_representative(df, out_dir, label, rep_name)
    fig_4_4_annual_contribution(df, out_dir, label, rep_name)

    print(">> 4-12 單層F CAGR盒鬚圖（依中位數排序）")
    f1_order = fig_4_12(df, out_dir, label)

    print(">> 4-13 F1×F2 統一熱力圖")
    fig_4_13(df, out_dir, label, f1_order)

    print(">> 4-15 不控制F的C盒鬚圖（反例）")
    fig_4_15(df, out_dir, label)

    print(">> 4-16 主入口：統一F軸的C折線")
    fig_4_16(df, out_dir, label, f1_order)

    print(">> 4-17 各C的F1×F2熱力圖小圖組")
    fig_4_17(df, out_dir, label, f1_order)

    print(">> V0/V1 配對 → 4-20/4-21/4-22/4-23")
    vpairs = build_v_pairs(df)
    fig_4_20_4_21(vpairs, out_dir, label)
    fig_4_22(vpairs, out_dir, label)
    fig_4_23(vpairs, out_dir, label)

    print(">> 4-18/4-24 自動選子樹深挖")
    best_bucket, worst_bucket = fig_4_18(df, out_dir, label, f1_order)
    fig_4_24(vpairs, out_dir, label, best_bucket)

    if not args.skip_credibility:
        dedupCAGR = df.drop_duplicates(subset=["CAGR", "max_drawdown", "win_ratio"])
        q75 = dedupCAGR["CAGR"].quantile(0.75)
        q50 = dedupCAGR["CAGR"].quantile(0.50)
        mask25 = df["CAGR"] >= q75
        mask50 = df["CAGR"] >= q50
        print(">> 4-5~4-10 可信度（逐策略讀 trades.parquet，較耗時）")
        cred25 = compute_credibility_metrics(df, label, mask25, "top25%")
        cred50 = compute_credibility_metrics(df, label, mask50, "top50%")
        fig_credibility_suite(cred25, cred50, out_dir, label)

        cat_dir = ROOT / "_analysis_outputs" / label
        cred50_save = cred50.drop(columns=["cum_shares20"])
        cred50_save.to_parquet(cat_dir / "credibility_metrics_top50.parquet")
        print("   → credibility_metrics_top50.parquet 已存")
    else:
        print(">> --skip-credibility：跳過 4-5~4-10")

    figs = sorted(p.name for p in out_dir.glob("*.png"))
    print(f"\n>> [{label}] 完成，圖表 {len(figs)} 張，總耗時 {time.time()-t0:.0f}s → {out_dir}")
    for f in figs:
        print("   -", f)


if __name__ == "__main__":
    main()
