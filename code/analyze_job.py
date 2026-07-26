# -*- coding: utf-8 -*-
"""
單一 sweep job 的第四章 24 張圖鑑分析（扣導覽圖＝20張要判讀的圖）。

背景（見 峰_purring plan / 對話紀錄）：
- 圖鑑定義見「第四章_24張圖_圖鑑_銜接版.md」；判讀結論的欄位對應見「研究部流程_銜接版_v5.md」。
- 目前 sweep job（如 TW_f3_N5-10_c20_M）把 N=5 / N=10 合併在同一個 job 裡（策略名內
  F1 標籤如 ROE_qb0of5 / ROE_qb0of10 本身就編碼了 N），但 v5 文件定義 job 外層軸含 N，
  「換 job → 整套圖換」。這裡在分析層把受桶寬影響的圖（F 桶/F1×F2 桶級網格）依 N 拆開，
  不受影響的圖（索引/可信度/估值配對等）維持 job 全域一次。
- 只做「程式可產出的圖」；成片vs孤立的 LLM 視覺判讀、估值條件註記屬於後續階段，不在此檔。
- 不改動 analyze_spec_us.py / report_grouping.py / 任何 notebook。

用法（cwd=code/）：
  ../.venv/Scripts/python.exe analyze_job.py --label TW_f3_N5-10_c20_M
  ../.venv/Scripts/python.exe analyze_job.py --label TW_f3_N5-10_c20_M --skip-credibility   # 先跳過耗時的逐策略掃描
"""
import re
import sys
import time
import argparse
import warnings
from pathlib import Path
from itertools import combinations as itcombos

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

# ==================== 風格（沿用 analyze_spec_us.py 的慣例，自成一份） ====================
OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
})
FACTORS = ["ROE", "EPS", "FCF_P"]
N_GRAINS = [5, 10]


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, color=INK, pad=8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ==================== 策略名解析 ====================
def load_strategy_artifacts(strat_dir: Path) -> dict:
    """讀單一策略資料夾的 trades/stock_data/return_table（不用 io_persistence.load_strategy_artifacts：
    該函式會 sanitize 路徑的每一段，遇到絕對路徑的磁碟代號（如 'D:'）會被誤判成非法字元換掉，
    導致讀到不存在的路徑、靜默回傳 None。這裡直接讀，避免此問題。"""
    out = {}
    for name in ("trades", "position", "stock_data", "return_table"):
        p = strat_dir / f"{name}.parquet"
        out[name] = pd.read_parquet(p) if p.exists() else None
    return out


def parse_bucket(label: str) -> dict:
    """'ROE_qb0of5' -> factor=ROE,k=0,n=5；'None' -> factor=None,k=n=None。"""
    if label == "None":
        return {"factor": "None", "k": np.nan, "n": np.nan}
    m = re.match(r"(.+)_qb(\d+)of(\d+)$", label)
    if m:
        return {"factor": m.group(1), "k": int(m.group(2)), "n": int(m.group(3))}
    return {"factor": label, "k": np.nan, "n": np.nan}


def parse_name(name: str) -> dict:
    parts = str(name).split("__")
    parts += ["None"] * (4 - len(parts))
    F1, F2, C, V = parts[0], parts[1], parts[2], parts[3]
    b1, b2 = parse_bucket(F1), parse_bucket(F2)
    c_kind = "None" if C == "None" else C.split("_")[0]
    return {
        "F1": F1, "F2": F2, "C": C, "V": V,
        "F1_factor": b1["factor"], "F1_k": b1["k"], "F1_n": b1["n"],
        "F2_factor": b2["factor"], "F2_k": b2["k"], "F2_n": b2["n"],
        "C_kind": c_kind,
        "is_pair": F2 != "None",
        "same_N": (F2 == "None") or (b1["n"] == b2["n"]),
        "base": "__".join(parts[:3]),   # 去掉 v0/v1 後的組合，供 V 配對用
    }


def load_stats(label: str) -> pd.DataFrame:
    p = HERE / "results_artifacts" / label / "stats.parquet"
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}，請確認 job 已跑完（sweep_driver.py）")
    df = pd.read_parquet(p)
    meta = df["strategy"].apply(parse_name).apply(pd.Series)
    df = pd.concat([df, meta], axis=1)
    for c in ("CAGR", "sharpe_ann", "max_drawdown", "avg_drawdown", "win_ratio", "ytd"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def c_kind_sort_key(c):
    return (0, "") if c == "None" else (1, int(c[1:].split("_")[0]) if c[1:].split("_")[0].isdigit() else 0, c)


def _cnum(c):
    if c == "None":
        return -1
    m = re.match(r"C(\d+)", c)
    return int(m.group(1)) if m else 999


def ordered_c_kinds(df):
    ks = [c for c in df["C_kind"].unique()]
    return sorted(ks, key=_cnum)


# ==================== V0/V1 配對（type 七 共用） ====================
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
    # ΔMDD：+ = 風險降（max_drawdown 是負值或幅度，這裡以 |v0| - |v1| 表示風險下降幅度）
    out["dMDD"] = out["MDD_v0"].abs() - out["MDD_v1"].abs()
    return out


# ==================== 逐策略可信度指標（type 四；讀 trades.parquet） ====================
def strategy_credibility(label: str, strategy: str) -> dict:
    art = load_strategy_artifacts(HERE / "results_artifacts" / label / strategy)
    t = art.get("trades")
    if t is None or len(t) == 0 or "stock_id" not in t.columns:
        return {"top1_stock": None, "top1_share": np.nan, "effective_n": np.nan, "cum_shares": None}
    prof = t.groupby("stock_id")["return"].sum()
    pos = prof[prof > 0]
    if len(pos) == 0 or pos.sum() <= 0:
        return {"top1_stock": None, "top1_share": np.nan, "effective_n": np.nan, "cum_shares": None}
    shares = (pos / pos.sum()).sort_values(ascending=False)
    hhi = float((shares.values ** 2).sum())
    return {
        "top1_stock": str(shares.index[0]),
        "top1_share": float(shares.iloc[0]),
        "effective_n": (1.0 / hhi) if hhi > 0 else np.nan,
        "cum_shares": shares.cumsum().values,   # 供 4-9 用
    }


def compute_credibility_metrics(df: pd.DataFrame, label: str, subset_mask, tag: str, log=print) -> pd.DataFrame:
    sub = df.loc[subset_mask, "strategy"].tolist()
    log(f"   [credibility:{tag}] 掃 {len(sub)} 個策略的 trades.parquet …")
    t0 = time.time()
    rows = []
    for i, name in enumerate(sub, 1):
        r = strategy_credibility(label, name)
        r["strategy"] = name
        rows.append(r)
        if i % 2000 == 0:
            log(f"      {i}/{len(sub)}（{time.time()-t0:.0f}s）")
    log(f"   [credibility:{tag}] 完成，{time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


# ==================== 類型二：4-2 Top10 熱力圖 ====================
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


# ==================== 類型三：4-3 / 4-4 代表策略解剖 ====================
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

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=False)
    axes[0].plot(cum.index, cum.values, color=OI[0], linewidth=1.4)
    axes[0].set_yscale("log")
    _style(axes[0], "淨值曲線 (log)", "", "cum return")

    axes[1].fill_between(dd.index, dd.values, 0, color=OI[1], alpha=0.5, linewidth=0)
    _style(axes[1], "回撤曲線", "", "drawdown")

    if rt is not None:
        months = [c for c in ("1","2","3","4","5","6","7","8","9","10","11","12") if c in rt.columns]
        m = pd.Series(rt[months].to_numpy(dtype=float).flatten())
        axes[2].bar(range(len(m)), m.values * 100, color=OI[2], width=0.9)
        _style(axes[2], "月報酬 (%)（依年×月攤平，起訖含結構性零）", "month index", "%")
    else:
        axes[2].axis("off")

    if "company_count" in sd.columns:
        cc = sd["company_count"].dropna()
        axes[3].plot(cc.index, cc.values, color=OI[3], linewidth=1.0)
        _style(axes[3], "持股數變化", "date", "count")
    else:
        axes[3].axis("off")

    short = rep_name if len(rep_name) <= 70 else rep_name[:67] + "…"
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
    contrib = t.groupby(["year", "stock_id"])["return"].sum().unstack(fill_value=0.0)

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
            ax.bar(x[yi], val, bottom=base, color=OI[k % len(OI)], width=0.7,
                   edgecolor="white", linewidth=0.3)
            base += val
        neg = row[row < 0]
        base = 0.0
        for stock, val in neg.items():
            ax.bar(x[yi], val, bottom=base, color=MUTED, width=0.7, alpha=0.5)
            base += val
    ax.set_xticks(x); ax.set_xticklabels(years, rotation=45, fontsize=8)
    ax.axhline(0, color=INK, linewidth=0.8)
    short = rep_name if len(rep_name) <= 60 else rep_name[:57] + "…"
    _style(ax, f"4-4  年度個股貢獻分布 — {label}\n{short}\n(顏色重複使用，僅示意集中/分散，非個股對照色)",
           "year", "contribution (sum of trade returns)")
    fig.tight_layout()
    fig.savefig(out_dir / "4-4_annual_contribution.png", bbox_inches="tight")
    plt.close(fig)


# ==================== 類型四：4-5~4-10 可信度 ====================
def fig_credibility_suite(cred25, cred50, out_dir, label):
    def top1_freq(cred, topn=15):
        vc = cred["top1_stock"].dropna().value_counts().head(topn)
        return vc

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (cred, tag) in zip(axes, [(cred25, "top25%"), (cred50, "top50%")]):
        vc = top1_freq(cred)
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
        curves = [c for c in cred["cum_shares"] if c is not None and len(c) > 0]
        if not curves:
            continue
        maxlen = max(len(c) for c in curves)
        padded = np.full((len(curves), maxlen), np.nan)
        for i, c in enumerate(curves):
            padded[i, :len(c)] = c
        med = np.nanmedian(padded, axis=0)
        q25 = np.nanpercentile(padded, 25, axis=0)
        q75 = np.nanpercentile(padded, 75, axis=0)
        xs = np.arange(1, maxlen + 1)
        ax.plot(xs, med, color=color, label=f"{tag} median", linewidth=1.6)
        ax.fill_between(xs, q25, q75, color=color, alpha=0.15)
    _style(ax, f"4-9  累積獲利貢獻曲線 — {label}", "個股排名(依貢獻由高到低)", "累積占比")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "4-9_cumulative_contribution.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    data = [cred25["effective_n"].dropna().values, cred50["effective_n"].dropna().values]
    bp = ax.boxplot(data, labels=[f"top25%\n(n={len(data[0])})", f"top50%\n(n={len(data[1])})"],
                    showmeans=True, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.5))
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=OI[i], alpha=0.35)
    _style(ax, f"4-10  Effective N 盒鬚圖 — {label}", "", "Effective N")
    fig.tight_layout()
    fig.savefig(out_dir / "4-10_effective_n.png", bbox_inches="tight")
    plt.close(fig)


# ==================== 類型五 / 六：F、C 盒鬚圖（4-12 / 4-15，含 N 拆分） ====================
def fig_4_12_by_bucket(df, out_dir, label, n_grain):
    sub = df[(~df["is_pair"]) & (df["F1_n"] == n_grain)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, factor in zip(axes, FACTORS):
        s = sub[sub["F1_factor"] == factor].sort_values("F1_k")
        order = sorted(s["F1_k"].unique())
        data = [s.loc[s["F1_k"] == k, "CAGR"].dropna().values for k in order]
        labels = [f"qb{int(k)}of{n_grain}" for k in order]
        bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True,
                        medianprops=dict(color=INK, linewidth=1.3))
        for i, box in enumerate(bp["boxes"]):
            box.set(facecolor=OI[i % len(OI)], alpha=0.35)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        _style(ax, factor, "bucket (低→高)", "CAGR" if factor == FACTORS[0] else "")
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    fig.suptitle(f"4-12  單層體質因子 CAGR 分布盒鬚圖 — N={n_grain} — {label}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / f"4-12_factor_boxplot_N{n_grain}.png", bbox_inches="tight")
    plt.close(fig)


def fig_4_15_uncontrolled_c(df, out_dir, label):
    order = ordered_c_kinds(df)
    data, labels = [], []
    for c in order:
        vals = df.loc[df["C_kind"] == c, "CAGR"].dropna().values
        if len(vals):
            data.append(vals); labels.append(f"{c}\n(n={len(vals)})")
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(data)), 4.6))
    bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.3),
                    flierprops=dict(marker="o", markersize=2, alpha=0.25))
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=OI[i % len(OI)], alpha=0.3)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    _style(ax, f"4-15  不控制F的動態因子C盒鬚圖（反例，全庫混合N5/N10）— {label}", "C group", "CAGR")
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "4-15_uncontrolled_C.png", bbox_inches="tight")
    plt.close(fig)


# ==================== 類型六：4-16 主入口（F控制後C折線，N拆分） ====================
def fig_4_16_controlled_c(df, out_dir, label, n_grain):
    sub = df[(~df["is_pair"]) & (df["F1_n"] == n_grain)]
    c_order = ordered_c_kinds(sub)
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, factor in zip(axes, FACTORS):
        s = sub[sub["F1_factor"] == factor]
        buckets = sorted(s["F1_k"].unique())
        for ci, c in enumerate(c_order):
            ys = [s.loc[(s["F1_k"] == k) & (s["C_kind"] == c), "CAGR"].mean() for k in buckets]
            if c == "None":
                ax.plot(buckets, ys, color="black", linewidth=2.2, linestyle="--", label="None(baseline)", zorder=5)
            else:
                ax.plot(buckets, ys, color=cmap(ci % 20), linewidth=1.1, alpha=0.85)
        _style(ax, factor, f"bucket k (of {n_grain})", "mean CAGR" if factor == FACTORS[0] else "")
    axes[-1].legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle(f"4-16⭐ 單層體質固定後各動態因子C平均CAGR折線（主入口）— N={n_grain} — {label}\n"
                 "直看(固定F看各C線高度)找強F；橫看(追一條C線)找配任何F都好的穩C", fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.88, 0.90])
    fig.savefig(out_dir / f"4-16_controlled_C_lines_N{n_grain}.png", bbox_inches="tight")
    plt.close(fig)
    # 回傳「整體最強C」供 4-17/4-24 使用
    agg = sub[sub["C_kind"] != "None"].groupby("C_kind")["CAGR"].mean()
    return agg


# ==================== 類型五/六：4-13 / 4-17 桶級 F1×F2 熱力圖（同N配對） ====================
def _heatmap_grid(ax, M, center, fmt="{:.2f}"):
    lo, hi = np.nanmin(M), np.nanmax(M)
    lo = min(lo, center - 1e-9); hi = max(hi, center + 1e-9)
    norm = TwoSlopeNorm(vmin=lo, vcenter=center, vmax=hi)
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6.5, color=INK)
    return im


def fig_4_13_bucket_heatmaps(df, out_dir, label):
    pairs = list(itcombos(FACTORS, 2))
    fig, axes = plt.subplots(len(N_GRAINS), len(pairs), figsize=(5 * len(pairs), 4.6 * len(N_GRAINS)))
    for ri, n in enumerate(N_GRAINS):
        for ci, (fa, fb) in enumerate(pairs):
            ax = axes[ri][ci]
            sub = df[(df["is_pair"]) & (df["C_kind"] == "None") &
                     (df["F1_factor"] == fa) & (df["F2_factor"] == fb) &
                     (df["F1_n"] == n) & (df["F2_n"] == n)]
            if sub.empty:
                ax.axis("off"); continue
            M = sub.pivot_table(index="F1_k", columns="F2_k", values="CAGR", aggfunc="mean")
            M = M.reindex(index=range(n), columns=range(n))
            center = float(np.nanmedian(M.values))
            im = _heatmap_grid(ax, M.values, center)
            _style(ax, f"{fa}×{fb}  N={n}", f"{fb} bucket", f"{fa} bucket")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"4-13  F1×F2 桶級熱力圖（同N配對，看連續成片vs孤立）— {label}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "4-13_F1xF2_bucket_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def fig_4_17_bucket_heatmaps_with_c(df, out_dir, label, strong_c):
    pairs = list(itcombos(FACTORS, 2))
    fig, axes = plt.subplots(len(N_GRAINS), len(pairs), figsize=(5 * len(pairs), 4.6 * len(N_GRAINS)))
    for ri, n in enumerate(N_GRAINS):
        for ci, (fa, fb) in enumerate(pairs):
            ax = axes[ri][ci]
            sub = df[(df["is_pair"]) & (df["C_kind"] == strong_c) &
                     (df["F1_factor"] == fa) & (df["F2_factor"] == fb) &
                     (df["F1_n"] == n) & (df["F2_n"] == n)]
            if sub.empty:
                ax.axis("off"); continue
            M = sub.pivot_table(index="F1_k", columns="F2_k", values="CAGR", aggfunc="mean")
            M = M.reindex(index=range(n), columns=range(n))
            center = float(np.nanmedian(M.values))
            im = _heatmap_grid(ax, M.values, center)
            _style(ax, f"{fa}×{fb}  N={n}", f"{fb} bucket", f"{fa} bucket")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"4-17  F1×F2×C={strong_c} 桶級熱力圖（同N配對）— {label}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "4-17_F1xF2xC_bucket_heatmap.png", bbox_inches="tight")
    plt.close(fig)


# ==================== 類型七：4-20 / 4-21 / 4-23 估值 V ====================
def fig_4_20_4_21_valuation(vpairs, out_dir, label):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    vals = vpairs["dCAGR"].dropna()
    ax.hist(vals, bins=60, color=OI[0], edgecolor="white", linewidth=0.3)
    med = vals.median()
    ax.axvline(med, color=OI[1], linewidth=1.8, label=f"median {med:.3f}")
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False)
    _style(ax, f"4-20  ΔCAGR (v1−v0) 分布 — {label}", "ΔCAGR", "count")
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


def fig_4_22_v_by_f_bucket(vpairs, out_dir, label, n_grain):
    single = vpairs[vpairs["F1"].str.contains(f"of{n_grain}$", regex=True)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, factor in zip(axes, FACTORS):
        s = single[single["F1_factor"] == factor]
        buckets = sorted(s["F1"].unique(), key=lambda x: parse_bucket(x)["k"])
        data0 = [s.loc[s["F1"] == b, "CAGR_v0"].dropna().values for b in buckets]
        data1 = [s.loc[s["F1"] == b, "CAGR_v1"].dropna().values for b in buckets]
        x = np.arange(len(buckets))
        ax.boxplot(data0, positions=x - 0.18, widths=0.3, patch_artist=True,
                  boxprops=dict(facecolor=OI[0], alpha=0.4), medianprops=dict(color=INK))
        ax.boxplot(data1, positions=x + 0.18, widths=0.3, patch_artist=True,
                  boxprops=dict(facecolor=OI[1], alpha=0.4), medianprops=dict(color=INK))
        ax.set_xticks(x); ax.set_xticklabels([f"k{int(parse_bucket(b)['k'])}" for b in buckets], fontsize=7)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        _style(ax, factor, "F1 bucket", "CAGR" if factor == FACTORS[0] else "")
    fig.suptitle(f"4-22  V0(藍) vs V1(橘) CAGR 依F1桶分組 — N={n_grain} — {label}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_dir / f"4-22_V0_V1_by_F_bucket_N{n_grain}.png", bbox_inches="tight")
    plt.close(fig)


def fig_4_23_v_by_c(vpairs, out_dir, label):
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
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, fontsize=7)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    _style(ax, f"4-23  V0(藍) vs V1(橘) CAGR 依動態因子C分組 — {label}", "C group", "CAGR")
    fig.tight_layout()
    fig.savefig(out_dir / "4-23_V0_V1_by_C.png", bbox_inches="tight")
    plt.close(fig)


# ==================== 類型三/六/七：4-18 / 4-24 自動選子樹深挖 ====================
def fig_4_18_4_24_subtree(df, vpairs, out_dir, label):
    single = df[~df["is_pair"]]
    med_by_bucket = single.groupby("F1")["CAGR"].median().dropna()
    if med_by_bucket.empty:
        return None
    best_bucket = med_by_bucket.idxmax()
    worst_bucket = med_by_bucket.idxmin()

    # 4-18：best vs worst 子樹下的 C 分布對照
    order = ordered_c_kinds(single)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
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
        ax.tick_params(axis="x", labelsize=6.5, labelrotation=60)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        _style(ax, f"{tag}\nF1={bucket}", "C group", "CAGR")
    fig.suptitle(f"4-18  固定不同子樹條件之C分布對照（自動選桶）— {label}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_dir / "4-18_subtree_C_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 4-24：best 子樹下 V0 vs V1 逐C對照（找條件反轉）
    sub_v = vpairs[vpairs["F1"] == best_bucket]
    order2 = sorted(sub_v["C_kind"].unique(), key=_cnum)
    m0 = [sub_v.loc[sub_v["C_kind"] == c, "CAGR_v0"].mean() for c in order2]
    m1 = [sub_v.loc[sub_v["C_kind"] == c, "CAGR_v1"].mean() for c in order2]
    x = np.arange(len(order2))
    fig, ax = plt.subplots(figsize=(max(9, 0.5 * len(order2)), 4.6))
    w = 0.35
    ax.bar(x - w/2, m0, width=w, color=OI[0], label="v0")
    ax.bar(x + w/2, m1, width=w, color=OI[1], label="v1")
    ax.set_xticks(x); ax.set_xticklabels(order2, rotation=60, fontsize=7)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False)
    _style(ax, f"4-24  指定子樹(F1={best_bucket})下 V0 vs V1 逐C對照（找條件反轉）— {label}", "C group", "mean CAGR")
    fig.tight_layout()
    fig.savefig(out_dir / "4-24_subtree_V0_V1_by_C.png", bbox_inches="tight")
    plt.close(fig)
    return {"best_bucket": best_bucket, "worst_bucket": worst_bucket}


# ==================== main ====================
def main():
    ap = argparse.ArgumentParser(description="單一 job 的第四章圖鑑分析")
    ap.add_argument("--label", required=True, help="job label，如 TW_f3_N5-10_c20_M")
    ap.add_argument("--skip-credibility", action="store_true", help="跳過耗時的逐策略 trades.parquet 掃描（4-5~4-10/4-4）")
    args = ap.parse_args()

    label = args.label
    out_dir = ROOT / "_analysis_outputs" / label / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f">> [{label}] 讀 stats.parquet …")
    df = load_stats(label)
    print(f"   策略數={len(df)}｜pair={(df['is_pair']).sum()}｜single={(~df['is_pair']).sum()}")

    print(">> 4-2 Top10 熱力圖")
    fig_4_2_top10(df, out_dir, label)

    rep_name = df.sort_values("CAGR", ascending=False).iloc[0]["strategy"]
    print(f">> 4-3/4-4 代表策略解剖（{rep_name}）")
    fig_4_3_representative(df, out_dir, label, rep_name)
    fig_4_4_annual_contribution(df, out_dir, label, rep_name)

    print(">> 4-12 單F桶CAGR盒鬚圖（N拆分）")
    for n in N_GRAINS:
        fig_4_12_by_bucket(df, out_dir, label, n)

    print(">> 4-15 不控制F的C盒鬚圖（反例）")
    fig_4_15_uncontrolled_c(df, out_dir, label)

    print(">> 4-16 主入口：F控制後C折線（N拆分）")
    strong_c_by_n = {}
    for n in N_GRAINS:
        strong_c_by_n[n] = fig_4_16_controlled_c(df, out_dir, label, n)
    overall_strong_c = pd.concat(strong_c_by_n.values()).groupby(level=0).mean().idxmax()
    print(f"   → 整體最強C：{overall_strong_c}")

    print(">> 4-13 F1×F2 桶級熱力圖（同N配對）")
    fig_4_13_bucket_heatmaps(df, out_dir, label)

    print(f">> 4-17 F1×F2×C({overall_strong_c}) 桶級熱力圖（同N配對）")
    fig_4_17_bucket_heatmaps_with_c(df, out_dir, label, overall_strong_c)

    print(">> V0/V1 配對 → 4-20/4-21/4-22/4-23")
    vpairs = build_v_pairs(df)
    fig_4_20_4_21_valuation(vpairs, out_dir, label)
    for n in N_GRAINS:
        fig_4_22_v_by_f_bucket(vpairs, out_dir, label, n)
    fig_4_23_v_by_c(vpairs, out_dir, label)

    print(">> 4-18/4-24 自動選子樹深挖")
    fig_4_18_4_24_subtree(df, vpairs, out_dir, label)

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
        for c in (cred25, cred50):
            c["cum_shares"] = c["cum_shares"].apply(lambda x: None if x is None else list(x))
        cred50_save = cred50.drop(columns=["cum_shares"])
        cred50_save.to_parquet(cat_dir / "credibility_metrics_top50.parquet")
        print(f"   → credibility_metrics_top50.parquet 已存（供將來 master_index 複用）")
    else:
        print(">> --skip-credibility：跳過 4-5~4-10 與 4-4 以外的逐策略讀取")

    figs = sorted(p.name for p in out_dir.glob("*.png"))
    print(f"\n>> [{label}] 完成，圖表 {len(figs)} 張，總耗時 {time.time()-t0:.0f}s → {out_dir}")
    for f in figs:
        print("   -", f)


if __name__ == "__main__":
    main()
