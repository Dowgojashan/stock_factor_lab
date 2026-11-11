# -*- coding: utf-8 -*-
"""老師 11/3 意見：(A) 規則排名隨時間變化圖 (B) 每 3／6 個月滾動重挑規則。不呼叫 LLM、不重跑回測。

資料：步驟1 每季報酬 `elite_grid_1027/quarterly_{市場}_{池}.parquet`（2015Q1～2025Q4，44 季；
每季持股來自該季所在窗的樣本外代表策略名單）。美股用加流動性門檻的 `_liq` 版本（同步驟2）。

(A) 每季季末，資金比例 30% 的 208 組（13 類×16 種取幾名×分法）依「過去 4 季累積報酬（扣成本）」排名（1＝最好）。
    每類一張：16 條細線（依取幾名上色）＋中位數粗線＋control0 虛線；另一張總覽熱圖（13 類中位數名次×季）。

(B) 滾動重挑規則（規則在看結果前寫死於此）：
    - 評估期：2016Q1～2025Q4，共 40 季（2015 年 4 季只當最長回看期的資料）
    - 每到重挑時點（季末），只用「該季末以前」的季報酬，挑過去 L 季累積報酬（扣成本）最高者，持有接下來 R 季
    - L ∈ {1, 2, 4} 季；R ∈ {1, 2} 季（每 3／6 個月重挑）
    - 候選範圍 U：「全部組合」＝1,040 組＋control0；「各類代表」＝13 類各取事先固定的「前3名・平均分・資金30%」＋control0
    - 同分取排序在前者（pandas idxmax）
    - 比較對象：control0、大盤（含息指數）、每季隨機挑（同一候選範圍、同一重挑頻率，1,000 個種子）；
      2024–25 另列步驟1「2015–23 樣本內扣成本 Sharpe 第 1 名」固定到底（這次報告的作法）
    - 報酬：未扣成本為主；扣成本為近似（用各組合自身每季成本，不含「換規則」多出的換手）
    - 分段：2016–17、2018–20、2021–23、2024–25（對應 window1～4 的樣本外期，window1 少 2015 年）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app import rule_selector as RS  # noqa: E402

SRC = RS.ROOT / "_analysis_outputs_applayer" / "elite_grid_1027"
OUT = RS.ROOT / "_analysis_outputs_applayer" / "rank_dynamics"
plt.rcParams.update({"font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False})

GROUPS = [("TW", "exclude_v1", ""), ("TW", "baseline", ""), ("US", "exclude_v1", "_liq"), ("US", "baseline", "_liq")]
GLAB = {"TW": "台股", "US": "美股", "exclude_v1": "排除V1", "baseline": "含V1"}
FAM = {"S1_votes": "票數", "S2_f1": "因子分數(F1)", "S2_f1f2": "因子分數(F1+F2)",
       "S3_greedy_L1": "追強(回看1個月)", "S3_greedy_L3": "追強(回看3個月)", "S3_greedy_L6": "追強(回看6個月)",
       "S4_reversal_L1": "押回彈(回看1個月)", "S4_reversal_L3": "押回彈(回看3個月)", "S4_reversal_L6": "押回彈(回看6個月)",
       "S5_mixed_L1": "混合(回看1個月)", "S5_mixed_L3": "混合(回看3個月)", "S5_mixed_L6": "混合(回看6個月)",
       "S6_mcap": "市值", "REF_control0": "control0（不調整）"}
N_COLOR = {1: "#d62728", 3: "#1f77b4", 5: "#2ca02c", 10: "#9467bd"}
REP = "|N3|equal|sl30"
PERIODS = [("2016–17", "2016", "2017"), ("2018–20", "2018", "2020"), ("2021–23", "2021", "2023"),
           ("2024–25", "2024", "2025"), ("全期 2016–25", "2016", "2025")]
EVAL_START = 4          # 第 5 季＝2016Q1
N_SEEDS = 1000
SLEEVES = (0.1, 0.2, 0.3, 0.5, 1.0)


def fam_of(combo: str) -> str:
    return combo.split("|")[0]


def load(market: str, pool: str, suf: str):
    q = pd.read_parquet(SRC / f"quarterly_{market}_{pool}{suf}.parquet")
    c = pd.read_csv(SRC / f"combos_{market}_{pool}{suf}.csv")
    g = q.pivot_table(index="end", columns="combo", values="ret_gross").sort_index()
    n = (q.assign(r=q.ret_gross - q.cost)).pivot_table(index="end", columns="combo", values="r").sort_index()
    assert len(g) == 44 and g.notna().all().all(), "每季報酬不完整"
    return g, n, c.set_index("combo")


def bench_quarterly(market: str, ends: pd.Index) -> pd.Series:
    from app._elite_grid_offline import load_benchmark
    b = load_benchmark(market)
    pts = [pd.Timestamp("2014-12-31")] + list(pd.to_datetime(ends))
    lv = [b.loc[:p].iloc[-1] for p in pts]
    return pd.Series(np.array(lv[1:]) / np.array(lv[:-1]) - 1, index=ends)


# ───────────────────────── (A) 排名變化圖 ─────────────────────────

def trailing_rank(net: pd.DataFrame, meta: pd.DataFrame, look: int = 4, sleeve: float = 0.3) -> pd.DataFrame:
    grid = [x for x in meta.index[meta.sleeve == sleeve]]
    trail = (1 + net).rolling(look).apply(np.prod, raw=True).dropna() - 1
    rank = trail[grid].rank(axis=1, ascending=False, method="min")
    rank["REF_control0"] = trail[grid].gt(trail["REF_control0"], axis=0).sum(axis=1) + 1
    return rank


def plot_family(rank, meta, market, pool, score, d: Path, look=4, sleeve=0.3):
    n_grid = rank.shape[1] - 1
    fam = [x for x in rank.columns if fam_of(x) == score]
    x = pd.to_datetime(rank.index)
    fig, ax = plt.subplots(figsize=(12, 6))
    for cb in fam:
        ax.plot(x, rank[cb], color=N_COLOR[int(meta.loc[cb, "n"])], lw=0.9, alpha=0.45)
    ax.plot(x, rank[fam].median(axis=1), color="black", lw=2.8, label=f"「{FAM[score]}」16 組的中位數名次")
    ax.plot(x, rank["REF_control0"], color="gray", lw=1.6, ls="--", label="control0（不調整）")
    for n, col in N_COLOR.items():
        ax.plot([], [], color=col, lw=1.5, label=f"取前{n}名（各種分法）")
    ax.axvline(pd.Timestamp("2023-12-31"), color="gray", lw=1, ls=":")
    ax.text(pd.Timestamp("2024-01-31"), n_grid * 0.97, "2024–25\n（實戰評估期）", fontsize=9, color="gray", va="bottom")
    ax.set_ylim(n_grid + 5, -5)
    ax.set_ylabel(f"名次（共 {n_grid} 組，1＝最好）")
    ax.set_xlabel(f"排名時點（季末；依過去 {look} 季累積報酬、扣成本）")
    ax.set_title(f"{GLAB[market]}・{GLAB[pool]}：「{FAM[score]}」排名依據的名次變化（優等生資金 {sleeve:.0%}）")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    fig.tight_layout()
    order = list(FAM).index(score) + 1
    fig.savefig(d / f"{order:02d}_{score}.png", dpi=130)
    plt.close(fig)


def plot_heatmap(rank, market, pool, d: Path, look=4, sleeve=0.3, basis=None, fname="00_總覽熱圖"):
    basis = basis or f"過去 {look} 季累積報酬"
    n_grid = rank.shape[1] - 1
    fams = [f for f in FAM if f != "REF_control0"]
    med = pd.DataFrame({f: rank[[x for x in rank.columns if fam_of(x) == f]].median(axis=1) for f in fams}).T
    med.loc["REF_control0"] = rank["REF_control0"]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    im = ax.imshow(med.values, aspect="auto", cmap="Blues_r", vmin=1, vmax=n_grid)
    ax.set_yticks(range(len(med)), [FAM[f] for f in med.index])
    lab = [f"{e[:4]}Q{(int(e[5:7]) - 1) // 3 + 1}" for e in med.columns]
    ax.set_xticks(range(len(lab)), lab, rotation=90, fontsize=8)
    for i in range(len(med)):
        for j in range(med.shape[1]):
            v = med.iat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v < n_grid * 0.35 else "black")
    j24 = list(med.columns).index(next(c for c in med.columns if c >= "2024-03-31"))
    ax.axvline(j24 - 0.5, color="red", lw=1.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.set_label(f"中位數名次（共 {n_grid} 組，顏色越深越好）")
    ax.set_title(f"{GLAB[market]}・{GLAB[pool]}：13 類排名依據的中位數名次（每季季末依{basis}排名；資金 {sleeve:.0%}；紅線右側＝2024–25）")
    fig.tight_layout()
    fig.savefig(d / f"{fname}.png", dpi=130)
    plt.close(fig)
    med.index = [FAM[f] for f in med.index]
    med.to_csv(d / f"{fname}_數值.csv", encoding="utf-8-sig")


def plot_n_heatmap(rank, meta, market, pool, d: Path, look=4, sleeve=0.3):
    """2026-11-10 使用者要求：取前 N 名 × 每季的中位數名次（跨 13 類合併；同一資金比例的 208 組內排名）。"""
    n_grid = rank.shape[1] - 1
    grid = [c for c in rank.columns if c != "REF_control0"]
    nn = meta.loc[grid, "n"].astype(int)
    med = pd.DataFrame({f"取前{k}名": rank[[c for c in grid if nn[c] == k]].median(axis=1) for k in (1, 3, 5, 10)}).T
    med.loc["control0（不調整）"] = rank["REF_control0"]
    fig, ax = plt.subplots(figsize=(15, 3.6))
    im = ax.imshow(med.values, aspect="auto", cmap="Blues_r", vmin=1, vmax=n_grid)
    ax.set_yticks(range(len(med)), list(med.index))
    lab = [f"{e[:4]}Q{(int(e[5:7]) - 1) // 3 + 1}" for e in med.columns]
    ax.set_xticks(range(len(lab)), lab, rotation=90, fontsize=8)
    for i in range(len(med)):
        for j in range(med.shape[1]):
            v = med.iat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v < n_grid * 0.35 else "black")
    j24 = list(med.columns).index(next(c for c in med.columns if c >= "2024-03-31"))
    ax.axvline(j24 - 0.5, color="red", lw=1.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.set_label(f"中位數名次（共 {n_grid} 組）")
    ax.set_title(f"{GLAB[market]}・{GLAB[pool]}：取前 N 名的中位數名次（跨 13 類；每季季末依過去 {look} 季累積報酬排名；資金 {sleeve:.0%}；紅線右側＝2024–25）")
    fig.tight_layout()
    fig.savefig(d / "00b_取幾名熱圖.png", dpi=130)
    plt.close(fig)
    med.to_csv(d / "00b_取幾名熱圖_數值.csv", encoding="utf-8-sig")


def load_daily(market: str, pool: str, suf: str) -> pd.DataFrame:
    """步驟1 存的逐日淨值（扣成本，`daily_net_*_w{1..4}`，每窗從 1 起算）→ 逐日報酬，四窗接起來。"""
    parts = []
    for w in (1, 2, 3, 4):
        nav = pd.read_parquet(SRC / f"daily_net_{market}_{pool}{suf}_w{w}.parquet").sort_index()
        r = nav.pct_change()
        r.iloc[0] = nav.iloc[0] - 1
        parts.append(r)
    return pd.concat(parts).sort_index()


def trailing_sharpe_rank(daily: pd.DataFrame, meta: pd.DataFrame, ends, sleeve: float = 0.3) -> pd.DataFrame:
    """2026-11-10 使用者要求的風險調整版：每季季末回看過去一年（365 天內）的日報酬，年化 Sharpe（rf=0、×√252）排名。"""
    grid = [x for x in meta.index[meta.sleeve == sleeve]]
    rows = {}
    for e in ends:
        t = pd.Timestamp(e)
        win = daily.loc[(daily.index > t - pd.Timedelta(days=365)) & (daily.index <= t)]
        rows[e] = win.mean() / win.std() * np.sqrt(252)
    sh = pd.DataFrame(rows).T
    rank = sh[grid].rank(axis=1, ascending=False, method="min")
    rank["REF_control0"] = sh[grid].gt(sh["REF_control0"], axis=0).sum(axis=1) + 1
    return rank


# ───────────────────────── (B) 滾動重挑 ─────────────────────────

def decision_points(R: int) -> list[int]:
    return list(range(EVAL_START, 44, R))


def rolling(net: pd.DataFrame, arms: list[str], L: int, R: int) -> list[str]:
    """回傳評估期每季持有的組合（長度 40）。挑選只用決策點之前的 L 季。"""
    held = []
    for i in decision_points(R):
        score = (1 + net.iloc[i - L:i][arms]).prod()
        pick = score.idxmax()
        held += [pick] * min(R, 44 - i)
    return held


def random_paths(arms: list[str], R: int, rng) -> np.ndarray:
    dps = decision_points(R)
    idx = rng.integers(len(arms), size=(N_SEEDS, len(dps)))
    reps = [min(R, 44 - i) for i in dps]
    return np.repeat(idx, reps, axis=1)          # N_SEEDS × 40，值＝arms 索引


def period_mask(ends: pd.Index, a: str, b: str) -> np.ndarray:
    y = ends.str[:4]
    return np.asarray((y >= a) & (y <= b))


def cum(r: np.ndarray) -> float:
    return float(np.prod(1 + r) - 1)


def run_group(market: str, pool: str, suf: str):
    g, n, meta = load(market, pool, suf)
    tag = f"{market}_{pool}"
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)

    # (A)
    # 2026-11-10 使用者要求：資金 10／20／30／50／100% 都要畫，各放一個子資料夾（排名都在同一資金比例的 208 組內比）
    daily = load_daily(market, pool, suf)
    for sl in SLEEVES:
        ds = d / f"資金{sl:.0%}"
        ds.mkdir(parents=True, exist_ok=True)
        rank = trailing_rank(n, meta, sleeve=sl)
        for f in FAM:
            if f != "REF_control0":
                plot_family(rank, meta, market, pool, f, ds, sleeve=sl)
        plot_heatmap(rank, market, pool, ds, sleeve=sl)
        plot_n_heatmap(rank, meta, market, pool, ds, sleeve=sl)
        rk_sh = trailing_sharpe_rank(daily, meta, rank.index, sleeve=sl)
        plot_heatmap(rk_sh, market, pool, ds, sleeve=sl, basis="過去一年日報酬的 Sharpe", fname="00c_總覽熱圖_Sharpe")

    # (B)
    ends = g.index[EVAL_START:]
    G, N = g.iloc[EVAL_START:], n.iloc[EVAL_START:]
    bench = bench_quarterly(market, g.index).iloc[EVAL_START:]
    grid_all = [c for c in meta.index if c.startswith("S")] + ["REF_control0"]
    reps = [f + REP for f in FAM if f != "REF_control0"] + ["REF_control0"]
    assert all(r in g.columns for r in reps)
    universes = {"全部組合": grid_all, "各類代表": reps}
    is_best = pd.read_csv(SRC / f"promoted_{market}_{pool}{suf}.csv").iloc[0]["combo"]

    rows, picks = [], []

    def add(name, U, L, R, rg, rn, rand=None):
        for plab, a, b in PERIODS:
            m = period_mask(ends, a, b)
            row = {"組別": f"{GLAB[market]}・{GLAB[pool]}", "方法": name, "候選範圍": U, "回看季數": L, "重挑間隔季數": R,
                   "期間": plab, "累積報酬_未扣成本": cum(rg[m]), "累積報酬_扣成本近似": cum(rn[m]) if rn is not None else np.nan}
            if rand is not None:
                rc = np.prod(1 + rand[:, m], axis=1) - 1
                row.update({"隨機挑_平均": rc.mean(), "隨機挑_p05": np.quantile(rc, .05), "隨機挑_p95": np.quantile(rc, .95),
                            "贏過隨機的比例": float((row["累積報酬_未扣成本"] > rc).mean())})
            rows.append(row)

    # 對照
    add("control0（不調整）", "—", None, None, G["REF_control0"].values, N["REF_control0"].values)
    add("大盤", "—", None, None, bench.values, None)
    m24 = period_mask(ends, "2024", "2025")
    for plab, a, b in PERIODS:
        if plab == "2024–25":
            rows.append({"組別": f"{GLAB[market]}・{GLAB[pool]}", "方法": f"步驟1 樣本內第1名固定到底（{is_best}）", "候選範圍": "—",
                         "回看季數": None, "重挑間隔季數": None, "期間": plab,
                         "累積報酬_未扣成本": cum(G[is_best].values[m24]), "累積報酬_扣成本近似": cum(N[is_best].values[m24])})
    for f in FAM:
        if f != "REF_control0":
            add(f"固定：{FAM[f]}（前3名・平均分・資金30%）", "—", None, None, G[f + REP].values, N[f + REP].values)

    rng = np.random.default_rng(20261103)
    Gv, Nv = G.values, N.values
    col = {c: i for i, c in enumerate(G.columns)}
    for U, arms in universes.items():
        ai = np.array([col[a] for a in arms])
        for R in (1, 2):
            rp = random_paths(arms, R, rng)
            rand_g = Gv[np.arange(40)[None, :], ai[rp]]          # N_SEEDS × 40 的未扣成本季報酬
            for L in (1, 2, 4):
                held = rolling(n, arms, L, R)
                hi = np.array([col[h] for h in held])
                rg, rn = Gv[np.arange(40), hi], Nv[np.arange(40), hi]
                add("滾動重挑", U, L, R, rg, rn, rand_g)
                for i, (e, h) in enumerate(zip(ends, held)):
                    if (i % R) == 0:
                        picks.append({"組別": f"{GLAB[market]}・{GLAB[pool]}", "候選範圍": U, "回看季數": L, "重挑間隔季數": R,
                                      "決策時點": g.index[EVAL_START + i - 1], "持有起始季末": e, "挑中的組合": h,
                                      "挑中的類別": FAM[fam_of(h)]})
    print(f"{tag}：圖 {len(FAM)} 張 → {d}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(picks)


def family_share(picks: pd.DataFrame) -> pd.DataFrame:
    """每個設定、每段期間，挑中各類別的季數比例（R=2 時每次決策代表 2 季）。"""
    p = picks.copy()
    p["期間"] = pd.cut(p["持有起始季末"].str[:4].astype(int), [2015, 2017, 2020, 2023, 2025],
                     labels=["2016–17", "2018–20", "2021–23", "2024–25"])
    keys = ["組別", "候選範圍", "回看季數", "重挑間隔季數", "期間"]
    s = p.groupby(keys + ["挑中的類別"], observed=True).size().rename("次數").reset_index()
    s["比例"] = s["次數"] / s.groupby(keys, observed=True)["次數"].transform("sum")
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = [run_group(*g) for g in GROUPS]
    summ = pd.concat([r[0] for r in res], ignore_index=True)
    picks = pd.concat([r[1] for r in res], ignore_index=True)
    summ.to_csv(OUT / "rolling_summary.csv", index=False, encoding="utf-8-sig")
    picks.to_csv(OUT / "rolling_picks.csv", index=False, encoding="utf-8-sig")
    family_share(picks).to_csv(OUT / "rolling_family_share.csv", index=False, encoding="utf-8-sig")
    print(f"彙總 {len(summ)} 列、決策 {len(picks)} 筆 → {OUT}")


if __name__ == "__main__":
    main()
