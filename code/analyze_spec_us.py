# -*- coding: utf-8 -*-
"""
spec_US 策略庫分析（artifacts/parquet 路線，不需 pickle）。

讀 results_artifacts/spec_US/：
  - stats.parquet          → 2310 策略 × 指標，解析策略名為 F1/F2/C/V
  - <strategy>/stock_data.parquet（cum_returns）→ 頂級策略淨值曲線
輸出圖表到 _analysis_outputs/spec_US/：
  - figures/*.png、rank_top30.csv

設計：Okabe-Ito 色盲安全配色、固定類別順序、單軸、細線條、淡格線（依 dataviz 指南）。
圖內文字用英文以避免 matplotlib 中文缺字。
用法（cwd=code/）：../.venv/Scripts/python.exe analyze_spec_us.py
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent          # .../code
ROOT = HERE.parent
LABEL = "spec_US"
ART_DIR = HERE / "results_artifacts" / LABEL
OUT_DIR = ROOT / "_analysis_outputs" / LABEL
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito 色盲安全類別色（固定順序，不循環）
OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
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


def parse_name(name: str) -> dict:
    """策略名 split('__') → [F1, F2, C, V]（與 report_analysis.ipynb 一致）。"""
    parts = str(name).split("__")
    parts += ["None"] * (4 - len(parts))
    F1, F2, C, V = parts[0], parts[1], parts[2], parts[3]
    def factor_of(tok):   # EPS_qb1of5 → EPS；ROE_<5 → ROE；None → None
        if tok in ("None", "", None):
            return "None"
        m = re.match(r"([A-Za-z_]+?)(_qb|_<|_>|_=|$)", tok)
        return (m.group(1) if m else tok).rstrip("_")
    return {"F1": F1, "F2": F2, "C": C, "V": V,
            "F1_factor": factor_of(F1), "F2_factor": factor_of(F2),
            "C_kind": ("None" if C == "None" else C.split("_")[0])}  # C1..C8 / None


def load_stats() -> pd.DataFrame:
    p = ART_DIR / "stats.parquet"
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}，請先跑完 fcv_us.py 全量回測")
    df = pd.read_parquet(p)
    meta = df["strategy"].apply(parse_name).apply(pd.Series)
    df = pd.concat([df, meta], axis=1)
    for c in ("CAGR", "daily_sharpe", "max_drawdown", "avg_drawdown", "win_ratio", "ytd"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # 對稱重複標記（EPS__ROE 與 ROE__EPS 數值相同）
    fp = ["CAGR", "daily_sharpe", "max_drawdown", "win_ratio", "ytd"]
    df["is_dup"] = df.duplicated(subset=fp, keep="first")
    return df


def _monthly_returns(name) -> pd.Series:
    """讀某策略 return_table 的逐月報酬，攤平並去掉起訖的結構性零（策略起訖前後未持倉）。"""
    p = ART_DIR / name / "return_table.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    rt = pd.read_parquet(p)
    cols = [c for c in ("1","2","3","4","5","6","7","8","9","10","11","12") if c in rt.columns]
    m = pd.Series(rt[cols].to_numpy(dtype=float).flatten()).dropna()
    nz = m[m != 0]
    return m.loc[nz.index[0]:nz.index[-1]] if len(nz) else m


def add_sharpe_ann(df) -> pd.DataFrame:
    """從 return_table 月報酬重算『正確年化 Sharpe』(mean/std×√12)，取代壞掉的 daily_sharpe。"""
    out = []
    for name in df["strategy"]:
        m = _monthly_returns(name)
        out.append(m.mean() / m.std() * np.sqrt(12) if len(m) > 2 and m.std() > 0 else np.nan)
    df["sharpe_ann"] = out
    return df


def fig_overview(df):
    cols = [("CAGR", "CAGR"), ("sharpe_ann", "Annualized Sharpe (fixed)"),
            ("max_drawdown", "Max Drawdown"), ("win_ratio", "Win Ratio")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (c, lab) in zip(axes, cols):
        ax.hist(df[c].dropna(), bins=40, color=OI[0], edgecolor="white", linewidth=0.4)
        med = df[c].median()
        ax.axvline(med, color=OI[1], linewidth=2, label=f"median {med:.2f}")
        _style(ax, f"{lab} distribution", lab, "count")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"spec_US strategy library — {len(df)} strategies (2000–2026, Russell 3000)",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "overview_distributions.png", bbox_inches="tight")
    plt.close(fig)


def _heatmap(ax, M, row_labels, col_labels, center, cmap="RdBu_r", fmt="{:.2f}",
             counts=None):
    vals = M[~np.isnan(M)]
    if len(vals) == 0:
        return None
    lo, hi = np.nanmin(M), np.nanmax(M)
    lo = min(lo, center - 1e-9); hi = max(hi, center + 1e-9)
    norm = TwoSlopeNorm(vmin=lo, vcenter=center, vmax=hi)
    cmap_obj = plt.get_cmap(cmap)
    im = ax.imshow(M, cmap=cmap_obj, norm=norm, aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            r, g, b, _ = cmap_obj(norm(M[i, j]))          # 依格色亮度決定黑/白字
            tcol = "white" if (0.299*r + 0.587*g + 0.114*b) < 0.5 else INK
            txt = fmt.format(M[i, j])
            if counts is not None and not np.isnan(counts[i, j]):
                txt += f"\nn={int(counts[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=tcol)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def fig_heatmap_f1f2(df):
    base = ["ROE", "EPS", "FCF_P"]
    rows = [f for f in base if f in set(df["F1_factor"])]              # F1 一定有主因子，無 None
    cols = [f for f in base + ["None"] if f in set(df["F2_factor"])]   # F2 可為 None（單因子）
    M = np.full((len(rows), len(cols)), np.nan)
    N = np.full((len(rows), len(cols)), np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            sub = df[(df["F1_factor"] == r) & (df["F2_factor"] == c)]["CAGR"].dropna()
            if len(sub):
                M[i, j] = sub.mean(); N[i, j] = len(sub)
    center = float(np.nanmedian(df["CAGR"]))
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    im = _heatmap(ax, M, rows, cols, center, counts=N)
    _style(ax, "Interaction F1 × F2 — mean CAGR\n(diagonal empty: the F stage never pairs a factor with itself)",
           "F2 factor (secondary, None = single-factor)", "F1 factor (primary)")
    if im is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("mean CAGR")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "heatmap_F1xF2_cagr.png", bbox_inches="tight")
    plt.close(fig)


def fig_heatmap_yearly(df):
    row = df.sort_values("CAGR", ascending=False).iloc[0]
    name = row["strategy"]
    p = ART_DIR / name / "return_table.parquet"
    if not p.exists():
        return
    rt = pd.read_parquet(p)
    months = [str(i) for i in range(1, 13)]
    M = rt[months].to_numpy(dtype=float) * 100.0     # → 百分比
    years = [str(y) for y in rt.index.tolist()]
    fig, ax = plt.subplots(figsize=(8.5, max(4, 0.28 * len(years) + 1.5)))
    im = _heatmap(ax, M, years, [f"M{m}" for m in months], center=0.0, fmt="{:.0f}")
    short = (name[:52] + "…") if len(name) > 53 else name
    _style(ax, f"Monthly returns (%) — top strategy\n{short}", "month", "year")
    if im is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("monthly return %")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "heatmap_yearly_returns_top.png", bbox_inches="tight")
    plt.close(fig)


def fig_risk_return(df):
    fig, ax = plt.subplots(figsize=(7, 5.2))
    order = ["v0", "v1"]
    for i, v in enumerate(order):
        sub = df[df["V"] == v]
        ax.scatter(-sub["max_drawdown"], sub["CAGR"], s=14, alpha=0.55,
                   color=OI[i], edgecolor="white", linewidth=0.3, label=f"V={v}")
    _style(ax, "Risk–return of the library", "Max drawdown (|.|)", "CAGR")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False, title="V (valuation filter)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "risk_return_scatter.png", bbox_inches="tight")
    plt.close(fig)


def _box_by(df, group_col, order, title, fname, xlabel):
    data, labels = [], []
    for g in order:
        vals = df.loc[df[group_col] == g, "CAGR"].dropna().values
        if len(vals):
            data.append(vals); labels.append(f"{g}\n(n={len(vals)})")
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(data) + 2), 4.4))
    bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.5),
                    meanprops=dict(marker="D", markerfacecolor=OI[1],
                                   markeredgecolor=OI[1], markersize=5),
                    flierprops=dict(marker="o", markersize=2, alpha=0.3))
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=OI[i % len(OI)], alpha=0.35, edgecolor=MUTED)
    _style(ax, title, xlabel, "CAGR")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def fig_effects(df):
    _box_by(df, "V", ["v0", "v1"], "Effect of valuation filter (V) on CAGR",
            "effect_by_V.png", "V")
    c_order = ["None"] + sorted([c for c in df["C_kind"].unique() if c != "None"])
    _box_by(df, "C_kind", c_order, "Effect of dynamic condition (C / P3) on CAGR",
            "effect_by_C.png", "C group")
    f_order = ["ROE", "EPS", "FCF_P"]
    f_order = [f for f in f_order if f in set(df["F1_factor"])]
    _box_by(df, "F1_factor", f_order, "Effect of primary factor (F1) on CAGR",
            "effect_by_F1.png", "F1 factor")


def fig_top_equity(df, topn=6):
    top = df.sort_values("CAGR", ascending=False).drop_duplicates(
        subset=["CAGR", "max_drawdown", "win_ratio"]).head(topn)
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    for i, (_, row) in enumerate(top.iterrows()):
        sd = ART_DIR / row["strategy"] / "stock_data.parquet"
        if not sd.exists():
            continue
        s = pd.read_parquet(sd)["cum_returns"].dropna()
        s = s[s > 0]                               # log 軸需正值（long-only cumprod 恆正）
        short = row["strategy"]
        short = (short[:42] + "…") if len(short) > 43 else short
        ax.plot(s.index, s.values, linewidth=1.6, color=OI[i % len(OI)],
                label=f"{short}  (CAGR {row['CAGR']:.2f})")
        plotted += 1
    ax.set_yscale("log")                           # 長期淨值曲線用 log 軸，早年才看得見
    _style(ax, f"Top {plotted} strategies by CAGR — equity curves (log scale)",
           "date", "cumulative return (log)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_equity_curves.png", bbox_inches="tight")
    plt.close(fig)


def main():
    print(f">> 讀 {ART_DIR}/stats.parquet")
    df = load_stats()
    n_uniq = int((~df["is_dup"]).sum())
    print(f"   策略={len(df)}（數值唯一={n_uniq}，對稱重複={len(df)-n_uniq}）")
    print("   重算年化 Sharpe（從 return_table 月報酬）…")
    df = add_sharpe_ann(df)
    print(f"   sharpe_ann：中位={df['sharpe_ann'].median():.2f}、"
          f"範圍 {df['sharpe_ann'].min():.2f}~{df['sharpe_ann'].max():.2f}"
          f"（原壞掉的 daily_sharpe 中位={df['daily_sharpe'].median():.1f}）")

    # 排行 CSV（去重後）；sharpe_ann=修正值、daily_sharpe=原壞值(留存對照)
    rank_cols = ["strategy", "F1", "F2", "C", "V", "CAGR", "sharpe_ann",
                 "max_drawdown", "win_ratio", "ytd", "daily_sharpe"]
    rank = df.loc[~df["is_dup"], rank_cols].sort_values("CAGR", ascending=False)
    rank.head(30).to_csv(OUT_DIR / "rank_top30.csv", index=False, encoding="utf-8-sig")
    print(f"   排行 → {OUT_DIR/'rank_top30.csv'}")

    fig_overview(df)
    fig_risk_return(df)
    fig_effects(df)
    fig_top_equity(df)
    fig_heatmap_f1f2(df)
    fig_heatmap_yearly(df)
    figs = sorted(p.name for p in FIG_DIR.glob("*.png"))
    print(f">> 圖表 {len(figs)} 張 → {FIG_DIR}")
    for f in figs:
        print("   -", f)


if __name__ == "__main__":
    main()
