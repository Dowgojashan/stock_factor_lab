# -*- coding: utf-8 -*-
"""M-13b · 用殘差樹選兵：`A3_residual`（2026-09-07）

🔴 **這支腳本要回答的問題**：M-01 證明「HRP 分的是 beta 曝險結構」——
口委看到這句話的下一句**必然**是：

> **「那你有沒有試著扣掉 beta 再分群？」**

在此之前答案是「沒有」。M-01 只做到「殘差樹的 ARI 是多少」（描述性），
**從來沒有把殘差樹接上選兵管線看它實際表現如何**。

**這是唯一能把 M-01 從否定性發現轉成建設性貢獻的實驗**：
若 `A3_residual` 的 OOS 表現優於 `A_hrp`，那「先扣 beta 再分群」就是一個
**可操作的改進**，而不只是一個批評。

---------------------------------------------------------------------------
設計：只換分群依據，其餘完全沿用（同 M-03 的共同座標軸原則）
---------------------------------------------------------------------------
  分群      **殘差相關樹**（扣掉各自市場 beta 後重建）  ← 唯一的變因
  品質分數  IS 窗 Calmar（與 A_hrp 完全相同）
  配額      同一個 `allocate()`，總量釘死同一個 `target_total`
  挑選      同一個 `select_representatives`
  多樣性門檻 該群自己的 `avg_intra_corr`，**用原始相關矩陣算**
            （與 A_hrp 同一把尺；M-15 已證明門檻會跟著分群變，這裡刻意讓它
             跟 A_hrp 用同一個相關矩陣，只讓「群的成員組成」不同）

⚠️ **市場基準用該窗自己的宇宙等權**（不是主線 normal 樹的全池），
   與 `B_all` 在該窗的定義一致。跨市場樹按策略所屬市場分開算。

---------------------------------------------------------------------------
為什麼比 M-03b 便宜很多
---------------------------------------------------------------------------
`A_hrp` 的結果**已經在 `walkforward_matrix_detail.csv` 裡**，不必重算；
本模組只需要建 **43 棵殘差樹**，而且**不需要抽樣**（殘差樹是確定性的）。
M-03b 貴在每格 30 次隨機抽樣，本模組沒有這個成本。

用法：
    cd code
    python -m research.residual_tree_selection --trees TW --schemes A   # 計時
    python -m research.residual_tree_selection                          # 全跑
"""
from __future__ import annotations

import argparse
import gc
import sys
import time

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, hrp, paths
from . import stage3_hrp as S3
from .four_group_control import (_cagr, _cagr_matrix, _mdd, _mdd_matrix,
                                 _portfolio_series, _sharpe, _small_enb)
from .partition_control import _avg_intra_corr
from .walkforward_matrix import (ALLOCATIONS, K_MODES, LEVEL, RATIO_GRID, TREES,
                                 _load_inputs, _load_k_table, allocate,
                                 build_schemes, oos_months, target_total,
                                 window_dates)
from .walkforward_partition import ENB_MAX_MEMBERS, pick_truncated

RANDOM_SEED = 42
METRICS = ("oos_cagr", "oos_mdd", "oos_sharpe", "oos_enb")


def market_benchmarks(wide_is: pd.DataFrame) -> dict[str, pd.Series]:
    """該窗**自己的宇宙**按市場分開的等權基準（＝該窗的 `B_all`，逐市場）。

    ⚠️ 刻意不用主線 normal 樹的全池等權——那是全窗的，會把 OOS 期間的成分
    帶進 IS 窗的基準。用該窗自己的宇宙才是乾淨的。
    """
    mk = pd.Series(wide_is.index, index=wide_is.index).str.split("::").str[0]
    return {m: wide_is.loc[mk[mk == m].index].mean(axis=0) for m in mk.unique()}


def residual_matrix(wide_is: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    """扣掉各自市場 beta 後的殘差矩陣（向量化，同 `beta_baseline._residuals` 的算法）。

    回傳 (殘差 ndarray, 逐策略 R²)。
    """
    mk = pd.Series(wide_is.index, index=wide_is.index).str.split("::").str[0]
    bench = market_benchmarks(wide_is)
    resid = np.empty(wide_is.shape, dtype=np.float64)
    r2 = pd.Series(index=wide_is.index, dtype=float)
    pos = {u: i for i, u in enumerate(wide_is.index)}
    for m, b in bench.items():
        idx = mk[mk == m].index
        rows = [pos[u] for u in idx]
        x = b.to_numpy(dtype=np.float64)
        xc = x - x.mean()
        var_x = float((xc ** 2).mean())
        y = wide_is.loc[idx].to_numpy(dtype=np.float64)
        yc = y - y.mean(axis=1, keepdims=True)
        beta = (yc @ xc) / (len(x) * var_x)
        alpha = y.mean(axis=1) - beta * x.mean()
        e = y - (alpha[:, None] + beta[:, None] * x[None, :])
        resid[rows] = e
        sse = (e ** 2).sum(axis=1)
        sst = (yc ** 2).sum(axis=1)
        r2.loc[idx] = 1.0 - np.divide(sse, sst, out=np.full_like(sse, np.nan), where=sst > 0)
    return resid, r2


def build_residual_tree(resid: np.ndarray, n: int, k_max: int, log=print) -> np.ndarray:
    """對殘差矩陣建 linkage。linkage 方法沿用主線的選法（最大群佔比較低者）。

    ⚠️ 必須排除零變異數列——殘差理論上不會全零，但浮點下可能出現，
    `np.corrcoef` 會給整列 NaN 讓後續靜默失效。
    """
    sd = resid.std(axis=1)
    if (sd <= 0).any():
        raise ValueError(f"殘差有 {(sd <= 0).sum()} 列零變異數，無法算相關")
    corr = np.corrcoef(resid)
    ok, min_eig = hrp.check_psd(corr)
    if not ok:
        raise AssertionError(f"殘差相關矩陣非 PSD（min_eig={min_eig:.3e}）")
    dist = hrp.corr_to_distance(corr)
    # 🔴 XM 的 corr/dist 各約 1.8GB。距離矩陣算完後相關矩陣就沒用了，
    # 立刻釋放；否則加上呼叫端稍後要算的 corr_raw，峰值會到 5.4GB。
    del corr
    gc.collect()
    cand = {}
    for method in ("single", "ward"):
        lk = hrp.build_linkage(dist, method=method)
        sizes = pd.Series(hrp.cut_clusters(lk, k_max)).value_counts()
        cand[method] = {"link": lk, "max_share": float(sizes.max() / n)}
    del dist
    gc.collect()
    m = min(cand, key=lambda x: cand[x]["max_share"])
    log(f"    殘差樹 linkage={m}（max_share {cand[m]['max_share']:.3f}）")
    return cand[m]["link"]


def _pick(cluster_map: pd.Series, quality: pd.Series, corr_raw: np.ndarray,
          pos: pd.Series, quota: pd.Series, rng) -> tuple[list[str], int]:
    """逐群用 H-10 貪婪規則挑代表。

    🔴 **門檻與相關矩陣都用「原始」的**——與 `A_hrp` 同一把尺，
    只讓「群的成員組成」不同。這是共同座標軸原則的實作。
    """
    picked, n_bf = [], 0
    for cid, g in cluster_map.groupby(cluster_map):
        m = int(quota.get(int(cid), 0))
        if m <= 0:
            continue
        members = g.index.tolist()
        idx = [pos[u] for u in members if u in pos.index]
        thr = _avg_intra_corr(corr_raw, idx, rng)
        p, bf, _ = pick_truncated(members, quality, corr_raw, pos, m, thr)
        picked += p
        n_bf += bf
    return picked, n_bf


def run_one_window(tree_key: str, srow: pd.Series, months_long, meta,
                   k_table, log=print) -> list[dict]:
    is_start, is_end, oos_start, oos_end = window_dates(srow, tree_key)
    uids = S3._tree_universe(tree_key, is_start, meta)
    wide_is = S3._pivot_window(months_long, uids, is_start, is_end)
    w = months_long[months_long.strategy_uid.isin(set(uids))]
    w = w[(w.month >= pd.Period(oos_start, "M")) & (w.month <= pd.Period(oos_end, "M"))]
    wide_oos = w.pivot(index="strategy_uid", columns="month", values="ret")
    if int(wide_oos.isna().sum().sum()) or (set(uids) - set(wide_oos.index)):
        raise ValueError(f"[{tree_key}] OOS 窗 {oos_start}~{oos_end} 資料不完整")

    quality = _cagr_matrix(wide_is) / _mdd_matrix(wide_is).abs().replace(0, np.nan)
    pos = pd.Series(range(len(wide_is.index)), index=wide_is.index)
    rng = np.random.default_rng(RANDOM_SEED)
    n_uni = len(uids)

    # 🔴 記憶體順序很要緊：XM 的相關矩陣是 15,040²≈1.8GB，建殘差樹時還會再產生
    # 相關矩陣與距離矩陣（各 1.8GB）。若此時同時持有 `corr_raw`，峰值會到 5.4GB。
    # 故**先建殘差樹並釋放中間結果，之後才算 corr_raw**。
    k_fixed = S3.L1_TARGET[tree_key]
    k_is = k_table.get((tree_key, is_start, is_end))
    if k_is is None:
        raise KeyError(f"k_stability 缺 ({tree_key}, {is_start}, {is_end})")
    resid, r2 = residual_matrix(wide_is)
    link_r = build_residual_tree(resid, n_uni, max(k_fixed, k_is), log)
    del resid
    gc.collect()
    corr_raw = np.corrcoef(wide_is.to_numpy(dtype=np.float64))

    rows = []
    for k_mode in K_MODES:
        k = k_fixed if k_mode == "fixed" else k_is
        labels = hrp.cut_clusters(link_r, k)
        rmap = pd.Series(labels, index=wide_is.index)
        sizes_s = pd.Series(labels).value_counts().sort_index()
        for ratio in RATIO_GRID:
            tot = target_total(ratio, n_uni, k)
            for how in ALLOCATIONS:
                quota, n_capped = allocate(sizes_s, tot, how)
                mem, bf = _pick(rmap, quality, corr_raw, pos, quota, rng)
                p_oos = _portfolio_series(wide_oos, mem)
                small = len(mem) <= ENB_MAX_MEMBERS
                rows.append({
                    "tree_key": tree_key, "scheme": srow.scheme, "k_mode": k_mode,
                    "ratio": str(ratio), "allocation": how,
                    "window_no": int(srow.window_no),
                    "is_start": is_start, "is_end": is_end,
                    "oos_start": oos_start, "oos_end": oos_end,
                    "n_oos_months": oos_months(oos_start, oos_end),
                    "n_clusters": int(len(sizes_s)), "target_total": tot,
                    "n_members": len(mem), "n_backfilled": bf,
                    "n_capped_clusters": n_capped,
                    "r2_median": float(r2.median()),
                    "max_cluster_share": float(sizes_s.max() / n_uni),
                    "enb_computed": bool(small),
                    "oos_cagr": _cagr(p_oos), "oos_mdd": _mdd(p_oos),
                    "oos_sharpe": _sharpe(p_oos),
                    "oos_enb": _small_enb(wide_oos, mem) if small else float("nan"),
                })
    return rows


def build(trees=TREES, schemes_filter=None, log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE0)
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE1 / "_marks")
    k_table = _load_k_table(log)
    schemes = build_schemes()
    if schemes_filter:
        schemes = schemes[schemes.scheme.isin(schemes_filter)]
    months_long, meta, _ = _load_inputs()
    log(f"窗口方案 {schemes.scheme.nunique()} 個｜總窗次 {len(schemes)}／樹")

    rows, t0 = [], time.time()
    for tree_key in trees:
        # 每個 (IS 起訖) 只建一次殘差樹——同 walkforward_matrix 的快取策略
        seen: dict[tuple[str, str], list[dict]] = {}
        for _, srow in schemes.iterrows():
            s, e, os_, oe = window_dates(srow, tree_key)
            tt = time.time()
            rows += run_one_window(tree_key, srow, months_long, meta, k_table, log)
            log(f"  [{tree_key}/{srow.scheme}/w{int(srow.window_no)}] "
                f"{time.time()-tt:.0f}s｜累計 {time.time()-t0:.0f}s")
        log("")

    df = pd.DataFrame(rows)
    for c in ("tree_key", "scheme", "k_mode", "ratio", "allocation"):
        df[c] = df[c].astype("category")
    return df[C.RESIDUAL_TREE_SELECTION.names]


def compare(res: pd.DataFrame, log=print) -> pd.DataFrame:
    """`A3_residual` vs `A_hrp` 的逐格對照（含 ratio 維度與證據分層，同 M-16 定案）。"""
    d = paths.ROOT / "_analysis_outputs_robustness"
    freeze.verify_inputs(d / "_walkforward_matrix_manifest")
    det = pd.read_csv(d / "walkforward_matrix_detail.csv")
    a = det[det.group == "A_hrp"].copy()
    a = a[a.tree_key.isin(set(res.tree_key.astype(str)))
          & a.scheme.isin(set(res.scheme.astype(str)))].copy()
    r = res.copy()
    key = ["tree_key", "scheme", "k_mode", "ratio", "allocation", "window_no"]
    for x in (a, r):
        for k in key:
            x[k] = x[k].astype(str)
    m = a.merge(r, on=key, suffixes=("_A", "_R"))
    if len(m) != len(a):
        raise AssertionError(f"只接上 {len(m)}/{len(a)} 格——格子鍵對不上")
    rows = []
    for metric in METRICS:
        diff = m[f"{metric}_R"] - m[f"{metric}_A"]       # 殘差樹 − 主線
        ok = diff.notna()
        sub = m[ok].copy()
        sub["_d"] = diff[ok]
        for ratio in ["ALL"] + sorted(sub.ratio.unique()):
            rs = sub if ratio == "ALL" else sub[sub.ratio == ratio]
            for tree, g in rs.groupby("tree_key", observed=True):
                if g.empty:
                    continue
                rows.append({
                    "tree_key": tree, "metric": metric, "ratio": ratio,
                    "n_cells": int(len(g)),
                    "residual_wins": float((g._d > 0).mean()),
                    "diff_mean": float(g._d.mean()),
                    "diff_median": float(g._d.median()),
                    "se_binomial_nominal": float(np.sqrt(0.25 / len(g))),
                    "evidence_type": ("confirmatory" if ratio == "ALL" else "exploratory"),
                })
    out = pd.DataFrame(rows)
    for c in ("tree_key", "metric", "ratio", "evidence_type"):
        out[c] = out[c].astype("category")
    return out


def _persist(df: pd.DataFrame, cmp_df: pd.DataFrame, log=print) -> None:
    """寫出兩張表**並重寫 manifest**（DD-08：改寫產物必須同時重寫 manifest）。"""
    d = paths.ROOT / "_analysis_outputs_robustness"
    p1, p2 = d / "residual_tree_selection.csv", d / "residual_tree_compare.csv"
    df.to_csv(p1, index=False, encoding="utf-8-sig")
    cmp_df.to_csv(p2, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "residual_tree_selection", d / "_residual_tree_selection_manifest",
        inputs=[paths.STAGE0 / "candidate_index.parquet",
                paths.STAGE1 / "returns_monthly.parquet",
                paths.STAGE1 / "returns_meta.parquet",
                paths.STAGE1 / "strategy_marks.parquet",
                d / "walkforward_matrix_detail.csv",
                d / "k_stability.csv"],
        outputs=[p1, p2],
        params={"seed": RANDOM_SEED, "enb_max_members": ENB_MAX_MEMBERS,
                "benchmark": "該窗自己的宇宙按市場等權（不是主線全池）",
                "threshold_and_corr": "原始相關矩陣（與 A_hrp 同一把尺）"},
        notes="M-13b：用殘差相關樹選兵。只換分群依據，品質分數／配額／挑選規則／"
              "門檻所用的相關矩陣全部沿用 A_hrp。回答口委必問的「有沒有試過扣掉 "
              "beta 再分群」。",
    )
    log(f"→ {p1.name} / {p2.name}")


def run(trees=TREES, schemes_filter=None, log=print):
    df = build(trees=trees, schemes_filter=schemes_filter, log=log)
    C.validate(df, C.RESIDUAL_TREE_SELECTION, strict_columns=True)
    log(f"✓ residual_tree_selection 契約通過（{len(df):,} 格）")
    cmp_df = compare(df, log)
    C.validate(cmp_df, C.RESIDUAL_TREE_COMPARE, strict_columns=True)
    _persist(df, cmp_df, log)
    return df, cmp_df


def _report(res: pd.DataFrame, cmp_df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 100)
    log("M-13b · 用殘差樹選兵——「有沒有試過扣掉 beta 再分群」")
    log("=" * 100)
    log(f"  格數 {len(res):,}｜殘差 R² 中位 {res.r2_median.median():.4f}"
        f"｜殘差樹最大群佔比中位 {res.max_cluster_share.median():.3f}")
    order = {"ALL": 0, "legacy": 1, "0.01": 2, "0.03": 3, "0.05": 4, "0.1": 5}
    c = cmp_df.copy()
    c["_o"] = c.ratio.astype(str).map(order).fillna(9)
    log(f"\n  {'樹':<5}{'指標':<12}{'比例':<9}{'格數':>6}{'殘差樹勝率':>11}"
        f"{'效果量(殘差−主線)':>18}{'±SE':>7}  證據")
    for (met, tree), g in c.groupby(["metric", "tree_key"], observed=True):
        for r in g.sort_values("_o").itertuples():
            tag = "驗證性(聚合)" if str(r.ratio) == "ALL" else "探索性"
            log(f"  {r.tree_key:<5}{r.metric:<12}{str(r.ratio):<9}{r.n_cells:>6,}"
                f"{r.residual_wins:>11.1%}{r.diff_mean:>18.4f}"
                f"{r.se_binomial_nominal:>7.3f}  {tag}")
        log("")
    log("判讀：")
    log("  · 殘差樹勝率 > 50% ⇒ **「先扣 beta 再分群」是可操作的改進**，")
    log("    M-01 從否定性發現轉成建設性貢獻。")
    log("  · 勝率 ≈ 50% ⇒ 扣不扣 beta 分群都一樣，M-01 的批評成立但沒有解法，")
    log("    論文要誠實寫「已試過，沒有改善」。")
    log("  ⚠️ 逐比例列一律探索性；±SE 是名目下限（格子不獨立）。")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.residual_tree_selection")
    ap.add_argument("--trees", nargs="+", default=list(TREES))
    ap.add_argument("--schemes", nargs="+")
    a = ap.parse_args(argv)
    _report(*run(trees=tuple(a.trees), schemes_filter=a.schemes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
