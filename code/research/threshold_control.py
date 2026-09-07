# -*- coding: utf-8 -*-
"""M-15 · 多樣性門檻是 M-03/M-03b 的第二個變因（2026-09-07）

🔴 **這支腳本要處理的問題**：`M03_partition_control.md` §0 寫
「其餘全部釘死，**唯一變因是分群依據**」——**嚴格說不成立**。

`select_representatives` 的 `max_pairwise_corr` = **該群自己的 `avg_intra_corr`**，
所以門檻會**跟著分群一起變**：

  · **HRP 群**：群內相關**高於**全宇宙平均（這正是分群做到的事）→ 門檻**寬鬆**
  · **隨機群**：是全宇宙的代表性樣本 → 群內相關 ≈ 全宇宙平均 → 門檻**嚴格**

⚠️ 這個設計是**刻意的**（`partition_control.py` 已寫明：對照組必須用它自己那個
分群算出來的 `avg_intra_corr`，沿用 HRP 的等於用 HRP 的資訊去幫對照組）。
**問題不在設計，在措辭**——不能宣稱「唯一變因」。

---------------------------------------------------------------------------
兩個部分
---------------------------------------------------------------------------
**A. backfill 診斷（免費）**
   `WALKFORWARD_PARTITION` 早就有 `n_backfilled_mean` 欄位，但**沒有任何分析用到它**。
   把它跟矩陣的 `n_backfilled`（A_hrp 的）並排、按比例分層。
   backfill = 多樣性門檻擋不住、退回純品質排序的檔數，
   **backfill 率越高 = 該組越退化成 `E_top_calmar`**。

**B. 固定門檻變體（貴，但這才真的釘死變因）**
   兩邊都改用**該窗全宇宙的平均相關**當 `max_pairwise_corr`，
   跑 2×2：{A_hrp, A2_random} × {群相依門檻, 固定門檻}。
   若「A 相對隨機的差距」在固定門檻下消失或反轉，
   代表 M-03b 量到的有一部分其實是門檻效應，不是分群效應。

   ⚠️ 成本考量只跑**代表性子集**：方案 E/H × 比例 legacy/3% × `k_mode=fixed`
   × 兩種分配。理由：3% 正是 M-03b 中 ENB 反轉最強、且隨機組 backfill
   暴增（9.3%→24.2%）的比例，legacy 則是 A 唯一勝出的比例——
   **兩端都取到，才看得出門檻是不是成因**。

用法：
    cd code
    python -m research.threshold_control --diagnose-only   # 只做 A 部分（秒級）
    python -m research.threshold_control                   # A + B（約 1 小時）
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, paths
from . import stage3_hrp as S3
from .four_group_control import (_cagr, _cagr_matrix, _mdd, _mdd_matrix,
                                 _portfolio_series, _sharpe, _small_enb)
from .partition_control import _avg_intra_corr, random_partition
from .walkforward_matrix import (ALLOCATIONS, LEVEL, TREES, _load_inputs, allocate,
                                 build_schemes, build_tree_for_window,
                                 target_total, window_dates)
from .walkforward_partition import ENB_MAX_MEMBERS, pick_truncated

#: 🔴 代表性子集。3% 是 ENB 反轉最強、隨機組 backfill 暴增的比例；
#: legacy 是 A 唯一勝出的比例。兩端都取才看得出門檻是不是成因。
SUBSET_SCHEMES = ("E", "H")
SUBSET_RATIOS = ("legacy", 0.03)
K_MODE = "fixed"            # 只跑 fixed，避免再乘一個維度
N_DRAWS = 30
RANDOM_SEED = 42
METRICS = ("oos_cagr", "oos_mdd", "oos_sharpe", "oos_enb")


# ============================================================================
# A. backfill 診斷
# ============================================================================

def backfill_diagnosis(log=print) -> pd.DataFrame:
    """A_hrp 與隨機分群的 backfill 率並排，按 (樹 × 比例 × 分配) 分層。

    🔴 backfill = 多樣性門檻擋不住、退回純品質排序的檔數。
    **backfill 率越高，該組越退化成 `E_top_calmar`**（純品質排序、無多樣性限制）。
    兩組的 backfill 率若差很多，「唯一變因是分群依據」就不成立——
    差的那部分是**門檻嚴格程度**。
    """
    d = paths.ROOT / "_analysis_outputs_robustness"
    freeze.verify_inputs(d / "_walkforward_matrix_manifest")
    freeze.verify_inputs(d / "_walkforward_partition_manifest")
    det = pd.read_csv(d / "walkforward_matrix_detail.csv")
    a = det[det.group == "A_hrp"].copy()
    r = pd.read_csv(d / "walkforward_partition.csv")
    key = ["tree_key", "scheme", "k_mode", "ratio", "allocation", "window_no"]
    for x in (a, r):
        for k in key:
            x[k] = x[k].astype(str)
    m = a.merge(r, on=key, suffixes=("_A", "_R"))
    if len(m) != len(a):
        raise AssertionError(f"接上隨機分群後只剩 {len(m)}/{len(a)} 格——兩表不同步")
    m["bf_a"] = m.n_backfilled / m.n_members
    m["bf_r"] = m.n_backfilled_mean / m.n_members_mean

    rows = []
    for (t, ratio, how), g in m.groupby(["tree_key", "ratio", "allocation"],
                                        observed=True):
        rows.append({"tree_key": t, "ratio": ratio, "allocation": how,
                     "n_cells": int(len(g)),
                     "backfill_rate_a": float(g.bf_a.mean()),
                     "backfill_rate_random": float(g.bf_r.mean()),
                     "backfill_diff": float(g.bf_r.mean() - g.bf_a.mean()),
                     "n_members_mean": float(g.n_members.mean())})
    out = pd.DataFrame(rows)
    log(f"  backfill 診斷：{len(out)} 列（樹 × 比例 × 分配）")
    return out


# ============================================================================
# B. 固定門檻變體
# ============================================================================

def _global_threshold(corr_full: np.ndarray, n: int, rng) -> float:
    """該窗**全宇宙**的平均兩兩相關——固定門檻用的值。

    用與 `_avg_intra_corr` 完全相同的抽樣估計器（超過 20 萬對即抽樣），
    確保兩種門檻模式的估計方式一致，差異純粹來自「用誰的母體」。
    """
    return _avg_intra_corr(corr_full, list(range(n)), rng)


def _pick_hrp(cluster_map: pd.Series, quality: pd.Series, corr_full, pos,
              quota: pd.Series, thr_mode: str, thr_global: float,
              rng) -> tuple[list[str], int]:
    picked, n_bf = [], 0
    for cid, g in cluster_map.groupby(cluster_map):
        m = int(quota.get(int(cid), 0))
        if m <= 0:
            continue
        members = g.index.tolist()
        idx = [pos[u] for u in members if u in pos.index]
        thr = thr_global if thr_mode == "global" else _avg_intra_corr(corr_full, idx, rng)
        p, bf, _ = pick_truncated(members, quality, corr_full, pos, m, thr)
        picked += p
        n_bf += bf
    return picked, n_bf


def _evaluate(members, wide_is, wide_oos) -> dict:
    p_oos = _portfolio_series(wide_oos, members)
    small = len(members) <= ENB_MAX_MEMBERS
    return {"oos_cagr": _cagr(p_oos), "oos_mdd": _mdd(p_oos),
            "oos_sharpe": _sharpe(p_oos),
            "oos_enb": _small_enb(wide_oos, members) if small else float("nan"),
            "n_members": len(members)}


def run_one_window(tree_key: str, srow: pd.Series, tree: dict, months_long,
                   n_draws: int, log=print) -> list[dict]:
    is_start, is_end, oos_start, oos_end = window_dates(srow, tree_key)
    uids = pd.Index(tree["assign"][C.PK])
    wide_is = S3._pivot_window(months_long, uids, is_start, is_end)
    w = months_long[months_long.strategy_uid.isin(set(uids))]
    w = w[(w.month >= pd.Period(oos_start, "M")) & (w.month <= pd.Period(oos_end, "M"))]
    wide_oos = w.pivot(index="strategy_uid", columns="month", values="ret")
    if int(wide_oos.isna().sum().sum()) or (set(uids) - set(wide_oos.index)):
        raise ValueError(f"[{tree_key}] OOS 窗 {oos_start}~{oos_end} 資料不完整")

    quality = _cagr_matrix(wide_is) / _mdd_matrix(wide_is).abs().replace(0, np.nan)
    corr_full = np.corrcoef(wide_is.to_numpy(dtype=np.float64))
    pos = pd.Series(range(len(wide_is.index)), index=wide_is.index)
    rng = np.random.default_rng(RANDOM_SEED)
    thr_global = _global_threshold(corr_full, len(wide_is), rng)

    labels = tree["assign"][f"cluster_{LEVEL}"].to_numpy()
    n_leaf = len(tree["link"]) + 1
    if n_leaf != len(wide_is):
        raise AssertionError(f"[{tree_key}] linkage 葉節點數 {n_leaf} != {len(wide_is)}")
    hrp_map = pd.Series(labels, index=uids)
    sizes_s = pd.Series(labels).value_counts().sort_index()
    k = len(sizes_s)
    sizes = sizes_s.tolist()

    common = {"tree_key": tree_key, "scheme": srow.scheme,
              "window_no": int(srow.window_no), "n_clusters": k,
              "is_start": is_start, "is_end": is_end,
              "oos_start": oos_start, "oos_end": oos_end,
              "threshold_global": float(thr_global)}

    rows = []
    for ratio in SUBSET_RATIOS:
        tot = target_total(ratio, len(uids), k)
        for how in ALLOCATIONS:
            quota, _ = allocate(sizes_s, tot, how)
            quota_r = pd.Series(quota.to_numpy(), index=range(1, k + 1))
            # 🔴 **兩種門檻模式共用同一批隨機分群**——這是配對比較的關鍵。
            # 若各自重抽，兩邊的差異會混進抽樣噪音；共用之後，
            # 「group_relative vs global」的差就**純粹來自門檻**。
            draws = [random_partition(uids, sizes, rng) for _ in range(n_draws)]
            for thr_mode in ("group_relative", "global"):
                base = {**common, "ratio": str(ratio), "allocation": how,
                        "threshold_mode": thr_mode, "target_total": tot}
                # ---- A_hrp ----
                mem, bf = _pick_hrp(hrp_map, quality, corr_full, pos, quota,
                                    thr_mode, thr_global, rng)
                ev = _evaluate(mem, wide_is, wide_oos)
                rows.append({**base, "group": "A_hrp", "n_draws": 1,
                             "n_backfilled_mean": float(bf),
                             **{k2: v for k2, v in ev.items()},
                             **{f"{m2}_std": np.nan for m2 in METRICS}})
                # ---- A2_random ----
                acc = {m2: [] for m2 in METRICS}
                bfs, nm = [], []
                for rmap in draws:
                    m2, bf2 = _pick_hrp(rmap, quality, corr_full, pos, quota_r,
                                        thr_mode, thr_global, rng)
                    e2 = _evaluate(m2, wide_is, wide_oos)
                    for kk in METRICS:
                        acc[kk].append(e2[kk])
                    bfs.append(bf2)
                    nm.append(e2["n_members"])
                rec = {**base, "group": "A2_random", "n_draws": n_draws,
                       "n_backfilled_mean": float(np.mean(bfs)),
                       "n_members": int(round(float(np.mean(nm))))}
                for kk in METRICS:
                    v = np.asarray(acc[kk], dtype=np.float64)
                    ok = np.isfinite(v)
                    rec[kk] = float(v[ok].mean()) if ok.any() else np.nan
                    rec[f"{kk}_std"] = float(v[ok].std(ddof=1)) if ok.sum() > 1 else np.nan
                rows.append(rec)
    return rows


def build_variant(trees=TREES, n_draws=N_DRAWS, log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE0)
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE1 / "_marks")
    schemes = build_schemes()
    schemes = schemes[schemes.scheme.isin(SUBSET_SCHEMES)]
    months_long, meta, f_combo_map = _load_inputs()
    log(f"固定門檻變體：方案 {list(SUBSET_SCHEMES)} × 比例 {list(SUBSET_RATIOS)}"
        f" × k_mode={K_MODE}｜總窗次 {len(schemes)}／樹")

    rows = []
    t0 = time.time()
    for tree_key in trees:
        wanted: dict[tuple[str, str], None] = {}
        for _, srow in schemes.iterrows():
            s, e, _, _ = window_dates(srow, tree_key)
            wanted[(s, e)] = None
        log(f"  [{tree_key}] 需建樹 {len(wanted)} 棵")
        cache = {}
        for is_start, is_end in wanted:
            tt = time.time()
            cache[(is_start, is_end)] = build_tree_for_window(
                tree_key, is_start, is_end, months_long, meta, f_combo_map, log)
            log(f"  [{tree_key}] 建樹 {is_start}~{is_end}  {time.time()-tt:.0f}s")
        for _, srow in schemes.iterrows():
            s, e, _, _ = window_dates(srow, tree_key)
            rows += run_one_window(tree_key, srow, cache[(s, e)], months_long,
                                   n_draws, log)
        log(f"  [{tree_key}] 完成｜累計 {time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    for c in ("tree_key", "scheme", "ratio", "allocation", "group", "threshold_mode"):
        df[c] = df[c].astype("category")
    return df[C.THRESHOLD_CONTROL.names]


def run(trees=TREES, n_draws=N_DRAWS, diagnose_only=False, log=print):
    out_dir = paths.ROOT / "_analysis_outputs_robustness"
    diag = backfill_diagnosis(log)
    C.validate(diag, C.BACKFILL_DIAGNOSIS, strict_columns=True)
    p1 = out_dir / "backfill_diagnosis.csv"
    diag.to_csv(p1, index=False, encoding="utf-8-sig")
    outputs = [p1]
    var = None
    if not diagnose_only:
        var = build_variant(trees=trees, n_draws=n_draws, log=log)
        C.validate(var, C.THRESHOLD_CONTROL, strict_columns=True)
        p2 = out_dir / "threshold_control.csv"
        var.to_csv(p2, index=False, encoding="utf-8-sig")
        outputs.append(p2)
    freeze.write_manifest(
        "threshold_control", out_dir / "_threshold_control_manifest",
        inputs=[paths.STAGE0 / "candidate_index.parquet",
                paths.STAGE1 / "returns_monthly.parquet",
                out_dir / "walkforward_matrix_detail.csv",
                out_dir / "walkforward_partition.csv"],
        outputs=outputs,
        params={"subset_schemes": list(SUBSET_SCHEMES),
                "subset_ratios": [str(r) for r in SUBSET_RATIOS],
                "k_mode": K_MODE, "n_draws": n_draws, "seed": RANDOM_SEED,
                "diagnose_only": diagnose_only},
        notes="M-15：多樣性門檻是 M-03/M-03b 的第二個變因。A=backfill 診斷（免費）；"
              "B=固定門檻變體（代表性子集），把門檻真的釘死看結論是否改變。",
    )
    log(f"→ {' / '.join(p.name for p in outputs)}")
    return diag, var


def _report(diag: pd.DataFrame, var: pd.DataFrame | None, log=print) -> None:
    log("\n" + "=" * 96)
    log("M-15 · 多樣性門檻是第二個變因")
    log("=" * 96)

    log("\n【A】backfill 率診斷（backfill 越高 = 越退化成純品質排序 E_top_calmar）")
    log(f"  {'樹':<5}{'比例':<9}{'分配':<14}{'A_hrp':>9}{'隨機分群':>11}{'差':>9}")
    o = {"legacy": 0, "0.01": 1, "0.03": 2, "0.05": 3, "0.1": 4}
    d = diag.copy()
    d["_o"] = d.ratio.astype(str).map(o)
    for r in d.sort_values(["tree_key", "_o", "allocation"]).itertuples():
        log(f"  {r.tree_key:<5}{str(r.ratio):<9}{r.allocation:<14}"
            f"{r.backfill_rate_a:>9.1%}{r.backfill_rate_random:>11.1%}"
            f"{r.backfill_diff:>+9.1%}")
    log("\n  ⇒ 差為正 = 隨機組被門檻擋得更凶、更退化成純品質排序，")
    log("    該格的「A vs 隨機」有一部分其實是**門檻效應**而非分群效應。")

    if var is None:
        log("\n【B】固定門檻變體：未執行（--diagnose-only）")
        return

    log("\n【B】固定門檻變體：把門檻釘死後，A 相對隨機的差距還在嗎")
    log(f"  {'樹':<5}{'比例':<9}{'門檻模式':<16}{'A CAGR':>9}{'隨機 CAGR':>11}"
        f"{'差(pp)':>9}{'A ENB':>8}{'隨機 ENB':>10}{'差':>8}")
    for (t, ratio), g in var.groupby(["tree_key", "ratio"], observed=True):
        for mode in ("group_relative", "global"):
            gg = g[g.threshold_mode == mode]
            a = gg[gg.group == "A_hrp"]
            r = gg[gg.group == "A2_random"]
            if a.empty or r.empty:
                continue
            log(f"  {t:<5}{str(ratio):<9}{mode:<16}"
                f"{a.oos_cagr.mean():>9.2%}{r.oos_cagr.mean():>11.2%}"
                f"{(a.oos_cagr.mean()-r.oos_cagr.mean())*100:>+9.2f}"
                f"{a.oos_enb.mean():>8.2f}{r.oos_enb.mean():>10.2f}"
                f"{a.oos_enb.mean()-r.oos_enb.mean():>+8.2f}")
        log("")
    log("  判讀：若「差」在 global 模式下明顯縮小或反向，代表 M-03b 量到的")
    log("        有一部分是門檻效應而非分群效應，M-03 的措辭要再收斂一次。")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.threshold_control")
    ap.add_argument("--trees", nargs="+", default=list(TREES))
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--diagnose-only", action="store_true",
                    help="只做 backfill 診斷（秒級），不跑固定門檻變體")
    a = ap.parse_args(argv)
    _report(*run(trees=tuple(a.trees), n_draws=a.draws,
                 diagnose_only=a.diagnose_only))
    return 0


if __name__ == "__main__":
    sys.exit(main())
