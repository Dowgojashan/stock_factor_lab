# -*- coding: utf-8 -*-
"""H-26b · 群數 k 的前視偏誤診斷（2026-09-04）

🔴 **這支腳本要回答的問題**：`stage3_hrp.L1_TARGET = {"TW":6,"US":7,"XM":3}` 是
H-03 用**完整窗**（台股 2007-2025、美股 2002-2025）的輪廓係數掃描選出來的，
但 `walkforward_matrix.py` 的第一個窗 IS 只到 2012-12——**等於用「看過 2013-2025
才選出來的 k」去切一棵只該知道 2012 年以前的樹**。

這跟本專案已經抓到並修掉的兩個錯誤是同一類：
  - H-18② 總經 z-score 用全樣本凍結 → 已改 5 年滾動窗
  - H-11 原提案用「OOS 要涵蓋 COVID」選窗 → 已撤銷

⚠️ **但先別急著重跑整個矩陣。** 更便宜的問法是：**只用 IS 資料選 k，選出來的
會不會就是同一個數字？** 若各窗選出的 k 都等於既有的 6/7/3，這個前視偏誤就是
**形式上存在、實質上無影響**，附上本表即可交代，不需要重建 8,370 格的矩陣。
若不同，才需要把 k_mode 納入矩陣當第四個維度。

方法（完全比照 H-03，只是把掃描的資料窗換成各個 IS 窗）：
  1. 用該 IS 窗的報酬矩陣算相關 → 距離
  2. 跑 single／ward 兩種 linkage，用 `stage3_hrp._build_tree` 同一條規則選
     （最大群佔比較低者，非 cophenetic）
  3. 對選中的 linkage 掃描 k=3..20 的輪廓係數
  4. 用 `cluster_count_selection.recommend_k` 同一條規則挑 k
     （排除「最大群佔比 > 50%」的退化解後取最高分）

⚠️ 只做 L1（k 的爭議只在 L1；L3 是固定的因子組合數錨點，不是輪廓係數選的）。

用法：
    cd code
    python -m research.k_stability                    # 全部 IS 窗
    python -m research.k_stability --trees TW US      # 只跑指定樹
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, hrp, paths
from . import stage3_hrp as S3
from .cluster_count_selection import K_MAX, K_MIN, recommend_k
from .walkforward_matrix import build_schemes, unique_is_ends, window_dates, _load_inputs

TREES = ("TW", "US", "XM")


def _linkage_and_dist(wide: pd.DataFrame, tree_key: str) -> tuple[np.ndarray, np.ndarray, str]:
    """比照 `stage3_hrp._build_tree` 的前半段：corr → dist → 兩種 linkage → 選一個。

    選法必須跟主線完全一致（最大群佔比較低者），否則掃出來的 k 不能拿來跟
    L1_TARGET 比較——那會變成「換了 linkage 又換了窗」的混合比較。
    """
    corr = np.corrcoef(wide.to_numpy(dtype=np.float64))
    psd_ok, min_eig = hrp.check_psd(corr)
    if not psd_ok:
        raise AssertionError(f"相關矩陣非 PSD（min_eig={min_eig:.3e}）")
    dist = hrp.corr_to_distance(corr)

    n_total = len(wide)
    l3_probe = S3.L3_TARGET[tree_key]   # 探測用的 L3 目標，跟主線同一個（依樹取值）
    results = {}
    for method in ("single", "ward"):
        link = hrp.build_linkage(dist, method=method)
        labels = hrp.cut_clusters(link, l3_probe)
        sizes = pd.Series(labels).value_counts()
        results[method] = {"link": link, "max_share": float(sizes.max() / n_total)}
    method = min(results, key=lambda m: results[m]["max_share"])
    return results[method]["link"], dist, method


def scan_window(tree_key: str, is_start: str, is_end: str, months_long, meta,
                log=print) -> dict:
    """單一 IS 窗：只用該窗資料選 k，跟寫死的 L1_TARGET 比較。"""
    t0 = time.time()
    uids = S3._tree_universe(tree_key, is_start, meta)
    wide = S3._pivot_window(months_long, uids, is_start, is_end)
    link, dist, method = _linkage_and_dist(wide, tree_key)

    scan = hrp.silhouette_scan(link, dist, range(K_MIN, K_MAX + 1))
    rec = recommend_k(scan, log=lambda *a, **k: None)
    fixed = S3.L1_TARGET[tree_key]
    row_fixed = scan[scan.k == fixed]

    out = {
        "tree_key": tree_key, "is_start": is_start, "is_end": is_end,
        "n_is_months": wide.shape[1], "n_strategies": len(wide),
        "linkage_method": method,
        "k_fixed": fixed, "k_is_selected": int(rec["k"]),
        "same_as_fixed": bool(int(rec["k"]) == fixed),
        "sil_at_selected": float(rec["silhouette"]),
        "sil_at_fixed": (float(row_fixed["silhouette"].iloc[0]) if len(row_fixed)
                         else float("nan")),
        "max_share_at_fixed": (float(row_fixed["max_share"].iloc[0]) if len(row_fixed)
                               else float("nan")),
        "degenerate_fallback": bool(rec["degenerate_fallback"]),
    }
    out["sil_gap"] = out["sil_at_selected"] - out["sil_at_fixed"]
    log(f"  [{tree_key}] IS {is_start}~{is_end}｜linkage={method}｜"
        f"IS選k={out['k_is_selected']} vs 寫死k={fixed}"
        f"{'  ✓一致' if out['same_as_fixed'] else '  ⚠️不同'}"
        f"｜輪廓 {out['sil_at_selected']:.4f} vs {out['sil_at_fixed']:.4f}"
        f"｜{time.time()-t0:.0f}s")
    return out


def run(trees=TREES, log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE0)
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE1 / "_marks")

    schemes = build_schemes()
    need = unique_is_ends(schemes)
    months_long, meta, _ = _load_inputs()
    log(f"需掃描的 IS 窗：{len(need)} 個 × {len(trees)} 棵樹 = {len(need)*len(trees)}\n")

    rows = []
    for tree_key in trees:
        # ⚠️ 必須依 (is_start, is_end) 去重：rolling 的第一窗 IS 起點被夾到錨點後，
        # 跟 anchored 的某一窗**完全是同一個訓練窗**（例如 TW 都是 2007-01~2014-12）。
        # 不去重會掃兩次相同的窗，浪費時間又觸發主鍵重複的契約違規。
        seen: set[tuple[str, str]] = set()
        for _, nr in need.iterrows():
            probe = schemes[(schemes["mode"] == nr["mode"])
                           & (schemes.is_end_offset == nr.is_end_offset)]
            if nr["mode"] == "rolling":
                probe = probe[probe.min_is_months == nr.min_is_months]
            is_start, is_end, _, _ = window_dates(probe.iloc[0], tree_key)
            if (is_start, is_end) in seen:
                continue
            seen.add((is_start, is_end))
            rows.append(scan_window(tree_key, is_start, is_end, months_long, meta, log))
        log("")

    df = pd.DataFrame(rows)
    df["tree_key"] = df["tree_key"].astype("category")
    C.validate(df, C.K_STABILITY, strict_columns=True)
    log(f"✓ k_stability 契約通過（{len(df)} 個 IS 窗）")

    out_dir = paths.ROOT / "_analysis_outputs_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "k_stability.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "k_stability", out_dir / "_k_stability_manifest",
        inputs=[paths.STAGE1 / "returns_monthly.parquet",
               paths.STAGE1 / "returns_meta.parquet",
               paths.STAGE1 / "strategy_marks.parquet"],
        outputs=[p],
        params={"k_min": K_MIN, "k_max": K_MAX, "l1_target_fixed": S3.L1_TARGET},
        notes="H-26b：診斷 L1_TARGET（用全窗選出的 6/7/3）在各 IS 窗下是否仍是"
              "輪廓係數的最佳解。若全數一致，walk-forward 的 k 前視偏誤即為"
              "形式上存在、實質無影響。",
    )
    log(f"→ {p}")
    return df


def _report(df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 78)
    log("H-26b · 群數 k 的前視偏誤診斷")
    log("=" * 78)
    n_same = int(df.same_as_fixed.sum())
    log(f"總計 {len(df)} 個 IS 窗，其中 **{n_same} 個**（{n_same/len(df):.1%}）"
        f"用 IS 資料選出的 k 跟寫死的 L1_TARGET 一致")
    log("")
    for t, g in df.groupby("tree_key", observed=True):
        fixed = int(g.k_fixed.iloc[0])
        vc = g.k_is_selected.value_counts().sort_index()
        log(f"[{t}] 寫死 k={fixed}｜IS 選出的 k 分布：{vc.to_dict()}"
            f"｜一致 {int(g.same_as_fixed.sum())}/{len(g)}")
        diff = g[~g.same_as_fixed]
        if len(diff):
            log(f"   ⚠️ 不一致的窗（輪廓係數差距 sil_gap 愈小代表影響愈輕微）：")
            log(diff[["is_start", "is_end", "k_is_selected", "sil_at_selected",
                     "sil_at_fixed", "sil_gap"]].round(4).to_string(index=False))
    log("")
    log("→ 若全數一致：前視偏誤形式上存在、實質無影響，附本表即可交代")
    log("→ 若有不一致：須把 k_mode 納入 walkforward_matrix 當第四個維度重跑")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.k_stability")
    ap.add_argument("--trees", nargs="+", default=list(TREES))
    a = ap.parse_args(argv)
    df = run(trees=tuple(a.trees))
    _report(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
