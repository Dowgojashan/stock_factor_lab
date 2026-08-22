# -*- coding: utf-8 -*-
"""階段 3 · HRP 階層聚類（真實資料管線，W-03/W-04）

輸入 ← `_frozen/stage1/returns_monthly.parquet`（HRP 原料）
        `_frozen/stage1/returns_meta.parquet`（hist_start，DD-03 共同窗判定用）
        `_frozen/stage0/candidate_index.parquet`（market/f_combo，ARI 驗證用）
輸出 → `_frozen/stage3/cluster_assign.parquet`
        `_frozen/stage3/cluster_meta.parquet`
        `_frozen/stage3/cluster_corr_matrix_{tree}.parquet`（L1 層級，k×k）
        `_frozen/stage3/linkage_{tree}.npy`（凍結的 linkage 矩陣）

⚠️ **本階段只建 normal 樹**。crisis 樹需要 regime 窗（階段2a），尚未開發——
   這是研究部 v9 定案的依賴順序（階段3 兩個輸入之一是 2c 的 regime 窗，
   常態樹不需要、危機樹才需要），不是遺漏。

DD-03（共同窗）已用**修復後、新候選池**的 hist_start 分布重新驗證，數字不變：
   TW 2007-01（保留 7,125/7,128）／US 2002-01（保留 8,682/8,682，XM 用 TW 窗
   時仍 100%）／XM 2007-01（TW 7,125 + US 8,682 = 15,807）。

用法：
    cd code
    python -m research.stage3_hrp                 # 全部三棵 normal 樹
    python -m research.stage3_hrp --tree TW        # 只跑一棵（除錯/測試用）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, hrp, paths

#: L1/L2 目標群數是固定的粗細層級（給 LLM 讀 / 給快篩配額用的中間層級）。
#: L3 錨定在該市場的獨立 F 組合數（DD-06：見 EXPECTED_F_COMBOS，快篩多樣性
#: 假象的根源，也是「真正不同的策略」數量級的客觀估計）。
L1_TARGET = 8
L2_TARGET = 40
L3_TARGET = {"TW": C.EXPECTED_F_COMBOS["TW"], "US": C.EXPECTED_F_COMBOS["US"],
            "XM": C.EXPECTED_F_COMBOS["TW"] + C.EXPECTED_F_COMBOS["US"]}


def _tree_universe(market_filter: str, window_start: str,
                   meta: pd.DataFrame) -> pd.Index:
    """依市場 + DD-03 共同窗篩出這棵樹要納入的策略。"""
    sub = meta if market_filter == "XM" else meta[meta.market == market_filter]
    sub = sub[sub.hist_start <= window_start]
    return pd.Index(sub.strategy_uid)


def _pivot_window(months_long: pd.DataFrame, uids: pd.Index,
                  window_start: str, window_end: str) -> pd.DataFrame:
    """long 格式報酬 → wide（策略 × 月），裁到共同窗。**不得有 NaN**（DD-03 保證）。"""
    w = months_long[months_long.strategy_uid.isin(uids)]
    w = w[(w.month >= pd.Period(window_start, "M")) & (w.month <= pd.Period(window_end, "M"))]
    wide = w.pivot(index="strategy_uid", columns="month", values="ret")
    wide = wide.loc[list(uids)]            # 固定順序，之後索引用得到
    n_nan = int(wide.isna().sum().sum())
    if n_nan:
        raise ValueError(
            f"共同窗內出現 {n_nan} 個 NaN——DD-03 的保證被打破了（策略池是否換過？"
            f"共同窗需要重新用 returns_meta 的 hist_start 分布驗證）")
    return wide


def _cluster_meta_and_corr(wide: pd.DataFrame, corr: np.ndarray, labels: np.ndarray,
                           level: str, tree_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """算每群的成員數/群內平均相關/代表策略（medoid），以及群間 k×k 相關矩陣。"""
    uids = wide.index.to_numpy()
    uniq = np.unique(labels)
    rows = []
    group_mean_series = {}   # 群代表序列（成員報酬的平均），供群間相關矩陣用
    for cid in uniq:
        idx = np.where(labels == cid)[0]
        n = len(idx)
        if n == 1:
            avg_corr, rep = np.nan, uids[idx[0]]
        else:
            sub = corr[np.ix_(idx, idx)]
            off_diag = sub[~np.eye(n, dtype=bool)]
            avg_corr = float(off_diag.mean())
            # medoid：與群內其他成員平均相關最高者，最能代表這個群
            rep = uids[idx[np.argmax(sub.mean(axis=1))]]
        rows.append({"tree_id": tree_id, "level": level, "cluster_id": int(cid),
                     "n_members": int(n), "avg_intra_corr": avg_corr,
                     "representative_uid": str(rep)})
        group_mean_series[int(cid)] = wide.iloc[idx].mean(axis=0)

    meta = pd.DataFrame(rows)

    # 群間相關矩陣：只在群數不太多時才有意義（給 LLM 讀的是 L1 這種粗層級）
    gids = sorted(group_mean_series)
    mat = pd.DataFrame(index=gids, columns=gids, dtype=float)
    means = pd.DataFrame({g: group_mean_series[g] for g in gids})
    gcorr = means.corr()
    return meta, gcorr


def run_tree(tree_key: str, months_long: pd.DataFrame, meta: pd.DataFrame,
            f_combo_map: pd.Series, log=print) -> dict:
    """跑一棵 normal 樹。`tree_key` ∈ {'TW','US','XM'}。"""
    t0 = time.time()
    window_start, window_end = C.HRP_WINDOWS[tree_key]
    tree_id = f"{tree_key}_normal"

    uids = _tree_universe(tree_key, window_start, meta)
    log(f"[{tree_id}] 窗 {window_start}~{window_end}｜策略數 {len(uids):,}")

    wide = _pivot_window(months_long, uids, window_start, window_end)
    log(f"  報酬矩陣 {wide.shape}（策略×月）｜pivot {time.time()-t0:.0f}s")

    returns = wide.to_numpy(dtype=np.float64)
    corr = np.corrcoef(returns)
    psd_ok, min_eig = hrp.check_psd(corr)
    log(f"  相關矩陣 {corr.shape}｜PSD={'✓' if psd_ok else '✗ 違反！'}（min_eig={min_eig:.3e}）"
        f"｜{time.time()-t0:.0f}s")
    if not psd_ok:
        raise AssertionError(
            f"[{tree_id}] 相關矩陣非 PSD（min_eig={min_eig:.3e}）——"
            f"DD-03 共同窗設計失敗，此樹不可信，中止")

    dist = hrp.corr_to_distance(corr)
    tri_ok, violations, max_excess = hrp.check_triangle_inequality(dist, n_samples=3000)
    log(f"  三角不等式抽檢 3000 組｜違規 {violations}"
        f"{f'（最大超出 {max_excess:.2e}）' if violations else ''}")

    cov = np.cov(returns)

    # DD-06：single vs ward。
    # ⚠️ **不能只看 cophenetic correlation**——實測發現 single linkage 在真實資料上
    # 出現典型的「鏈狀效應」（chaining）：95.5% 的策略被塞進同一個巨群、另外
    # 163/218 個「群」是單一成員，L3 vs F組合的 ARI 只有 0.0001（等於分群跟任何
    # 經濟意義上的分組完全無關）——但它的 cophenetic（0.761）卻比 ward（0.500）
    # 還高。cophenetic 衡量的是「整棵樹保真度」，不代表切出來的群有沒有用；
    # single linkage 很容易在保真度上作弊（一路鏈接，局部距離都保真，全局結構卻爛掉）。
    # 故改用「切到 L3 目標群數時，分群夠不夠平衡」當選擇標準：
    # 最大群佔比越低、singleton 群數越少，代表分群越有鑑別力。
    n_total = len(uids)
    l3_target_probe = L3_TARGET[tree_key]
    results = {}
    for method in ("single", "ward"):
        link = hrp.build_linkage(dist, method=method)
        coph = hrp.cophenetic_correlation(link, dist)
        probe_labels = hrp.cut_clusters(link, l3_target_probe)
        sizes = pd.Series(probe_labels).value_counts()
        max_share = float(sizes.max() / n_total)
        n_singleton = int((sizes == 1).sum())
        results[method] = {"link": link, "cophenetic": coph,
                           "max_share": max_share, "n_singleton": n_singleton}
        log(f"  linkage={method:<6} cophenetic={coph:.4f}  "
            f"L3探測：最大群佔比={max_share:.1%}  singleton群數={n_singleton}/{l3_target_probe}")

    method = min(results, key=lambda m: results[m]["max_share"])
    link = results[method]["link"]
    coph = results[method]["cophenetic"]
    if results[method]["max_share"] > 0.5:
        log(f"  ⚠️ 連分群較平衡的 {method} 最大群佔比也超過 50%"
            f"（{results[method]['max_share']:.1%}）——此樹的分群結果可能仍不可靠，建議人工複查")
    log(f"  → 採用 {method}（最大群佔比較低，分群較平衡；非以 cophenetic 決定，見上方註解）"
        f"｜{time.time()-t0:.0f}s")

    leaf_order = hrp.quasi_diagonal_order(link)
    weights = hrp.recursive_bisection_weights(cov, leaf_order)
    log(f"  遞迴二分權重完成（全樹持有時的權重，非任意子集）｜加總={weights.sum():.6f}")

    # 分群層級（連同平衡度診斷，用最終選定的 method 重切，非上面探測用的那次）
    l3_target = L3_TARGET[tree_key]
    targets = {"L1": L1_TARGET, "L2": L2_TARGET, "L3": l3_target}
    labels = {}
    level_diag = {}
    for lvl, target in targets.items():
        lab = hrp.cut_clusters(link, target)
        labels[lvl] = lab
        sizes = pd.Series(lab).value_counts()
        max_share = float(sizes.max() / n_total)
        n_singleton = int((sizes == 1).sum())
        level_diag[lvl] = {"n_clusters": int(len(sizes)), "max_share": max_share,
                           "n_singleton": n_singleton}
        log(f"  {lvl}：目標群數 {target}，實際群數 {len(sizes)}"
            f"｜最大群佔比 {max_share:.1%}｜singleton {n_singleton}")

    # DD-06 驗收：L3 vs 獨立 F 組合的 ARI（XM 用 market::f_combo 避免台美碰撞）
    fc = f_combo_map.reindex(wide.index)
    ari = hrp.adjusted_rand_index(pd.Series(labels["L3"]), pd.Series(fc.to_numpy()))
    log(f"  L3 vs F組合 ARI = {ari:.4f}（DD-06 分群層級的客觀選擇依據）")

    assign = pd.DataFrame({
        C.PK: wide.index, "tree_id": tree_id,
        "cluster_L1": labels["L1"], "cluster_L2": labels["L2"], "cluster_L3": labels["L3"],
    })

    meta_rows, corr_mats = [], {}
    for lvl in ("L1", "L2", "L3"):
        m, gcorr = _cluster_meta_and_corr(wide, corr, labels[lvl], lvl, tree_id)
        meta_rows.append(m)
        corr_mats[lvl] = gcorr
    cluster_meta = pd.concat(meta_rows, ignore_index=True)

    dt = time.time() - t0
    log(f"[{tree_id}] 完成，{dt:.0f}s\n")
    return {
        "tree_id": tree_id, "assign": assign, "cluster_meta": cluster_meta,
        "corr_l1": corr_mats["L1"], "link": link, "method": method,
        "cophenetic": coph, "psd_min_eig": min_eig, "ari_l3": ari,
        "n_strategies": len(uids), "n_months": returns.shape[1], "seconds": dt,
        "method_comparison": {m: {k: v for k, v in r.items() if k != "link"}
                              for m, r in results.items()},
        "level_diag": level_diag,
    }


def run(trees: list[str] | None = None, log=print) -> dict:
    freeze.verify_inputs(paths.STAGE1)
    trees = trees or ["TW", "US", "XM"]

    log("載入 returns_monthly / returns_meta / candidate_index …")
    months_long = pd.read_parquet(paths.STAGE1 / "returns_monthly.parquet")
    meta = pd.read_parquet(paths.STAGE1 / "returns_meta.parquet")
    idx = pd.read_parquet(paths.STAGE0 / "candidate_index.parquet")
    # XM 樹台美策略字串可能碰撞（DD-10），F組合比較必須連市場一起比
    f_combo_map = (idx.market.astype(str) + "::" + idx.f_combo.astype(str))
    f_combo_map.index = idx.strategy_uid
    log(f"  {len(months_long):,} 筆月報酬｜{len(meta):,} 個策略\n")

    paths.STAGE3.mkdir(parents=True, exist_ok=True)
    results = {}
    for t in trees:
        results[t] = run_tree(t, months_long, meta, f_combo_map, log)

    all_assign = pd.concat([r["assign"] for r in results.values()], ignore_index=True)
    all_assign["tree_id"] = all_assign["tree_id"].astype("category")
    all_meta = pd.concat([r["cluster_meta"] for r in results.values()], ignore_index=True)
    all_meta["tree_id"] = all_meta["tree_id"].astype("category")
    all_meta["level"] = all_meta["level"].astype("category")

    C.validate(all_assign, C.CLUSTER_ASSIGN)
    C.validate(all_meta, C.CLUSTER_META)
    log("✓ cluster_assign / cluster_meta 契約通過")

    outs = []
    p = paths.STAGE3 / "cluster_assign.parquet"
    all_assign.to_parquet(p, compression="zstd", index=False); outs.append(p)
    p = paths.STAGE3 / "cluster_meta.parquet"
    all_meta.to_parquet(p, compression="zstd", index=False); outs.append(p)

    for t, r in results.items():
        p = paths.STAGE3 / f"cluster_corr_matrix_{r['tree_id']}.parquet"
        r["corr_l1"].to_parquet(p, compression="zstd"); outs.append(p)
        p = paths.STAGE3 / f"linkage_{r['tree_id']}.npy"
        np.save(p, r["link"]); outs.append(p)

    freeze.write_manifest(
        "stage3_hrp", paths.STAGE3,
        inputs=[paths.STAGE1 / "returns_monthly.parquet",
               paths.STAGE1 / "returns_meta.parquet",
               paths.STAGE0 / "candidate_index.parquet"],
        outputs=outs,
        params={"trees_built": list(results),
               "L1_target": L1_TARGET, "L2_target": L2_TARGET, "L3_target": L3_TARGET,
               "linkage_method_chosen": {t: r["method"] for t, r in results.items()},
               "linkage_selection_criterion": "L3切割時最大群佔比最低者勝出，非cophenetic"
                                              "（見程式註解：single linkage 實測出現鏈狀效應)",
               "method_comparison": {t: r["method_comparison"] for t, r in results.items()},
               "level_diag": {t: r["level_diag"] for t, r in results.items()},
               "cophenetic": {t: r["cophenetic"] for t, r in results.items()},
               "ari_l3_vs_fcombo": {t: r["ari_l3"] for t, r in results.items()},
               "windows": {k: list(v) for k, v in C.HRP_WINDOWS.items()}},
        notes="⚠️ 只有 normal 樹；crisis 樹待階段2a regime 窗完成後補建",
    )
    return results


def _report(results: dict, log=print) -> None:
    log("=" * 66)
    log("階段3 HRP · 驗收摘要")
    log("=" * 66)
    for t, r in results.items():
        log(f"[{r['tree_id']}] {r['n_strategies']:,} 策略 × {r['n_months']} 月"
            f"｜method={r['method']}（cophenetic={r['cophenetic']:.3f}）"
            f"｜PSD min_eig={r['psd_min_eig']:.2e}｜ARI(L3 vs F組合)={r['ari_l3']:.3f}"
            f"｜{r['seconds']:.0f}s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.stage3_hrp")
    ap.add_argument("--tree", choices=["TW", "US", "XM"], action="append",
                    help="只跑指定的樹（可重複給多次）；預設全部三棵")
    a = ap.parse_args(argv)
    t0 = time.time()
    results = run(trees=a.tree)
    print(f"\n總耗時 {time.time()-t0:.0f}s")
    _report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
