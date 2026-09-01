# -*- coding: utf-8 -*-
"""H-03 · L1 群數選擇：用輪廓係數（silhouette）取代寫死的 `L1_TARGET=8`。

背景（開發待辦追蹤.md H-03）：`stage3_hrp.py` 原本 `L1_TARGET=8` 是老師在會議上舉的
資金試算例子（「假設了 ok 我臺股分8群 美股分3群」），不是他真的指定的群數。
使用者要求改用文獻既有的群數選擇準則，讓群數由資料決定，不是拍腦袋或沿用舉例數字。

方法：對 TW_normal / US_normal（XM_normal 一併跑供參考）兩棵樹，用**已凍結的
linkage**（不重新計算相關矩陣/PSD/linkage，那是`stage3_hrp.py`已經做過且驗證過的
昂貴步驟）在 k=3..20 範圍逐一切割、算輪廓係數，選平均輪廓係數最高、且分群平衡
（不是退化成一堆singleton+一個巨群）的 k。

⚠️ 只重新計算距離矩陣（corr_to_distance），不重新計算 linkage 本身——
   linkage 是樹狀結構，跟「切幾群」無關，沿用凍結版本就是同一棵樹只是切法不同，
   不會因為這次分析而讓 stage3 的既有产物產生任何差異。

用法：
    cd code
    python -m research.cluster_count_selection
    python -m research.cluster_count_selection --tree TW_normal --k-max 25
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import hrp, paths
from . import stage3_hrp as S3   # 資料準備一律走 S3.rebuild_tree_returns()，見該函式docstring

DEFAULT_TREES = ("TW_normal", "US_normal", "XM_normal")
K_MIN, K_MAX = 3, 20


def _rebuild_dist_matrix(tree_id: str, log=print) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """重算某棵樹的距離矩陣（corr_to_distance），linkage 直接讀凍結檔，不重算。

    資料準備（usable過濾／共同窗／排除零變異數）一律走
    `stage3_hrp.rebuild_tree_returns()` 這個單一事實來源，故 corr 與建樹當下逐位元
    一致——linkage 是拿同一個 corr 建的，不會對不起來。2026-08-30 code review 前
    這裡有一份逐行複製品，跟 `effective_bets._tree_corr` 各自維護，見該函式docstring。
    """
    wide = S3.rebuild_tree_returns(tree_id, log)
    corr = np.corrcoef(wide.to_numpy(dtype=np.float64))
    dist = hrp.corr_to_distance(corr)
    link = np.load(paths.STAGE3 / f"linkage_{tree_id}.npy")
    log(f"[{tree_id}] 重建距離矩陣 {dist.shape}，讀凍結linkage {link.shape}")
    return dist, link, wide.index


def scan_tree(tree_id: str, k_min: int, k_max: int, log=print) -> pd.DataFrame:
    t0 = time.time()
    dist, link, uids = _rebuild_dist_matrix(tree_id, log)
    scan = hrp.silhouette_scan(link, dist, range(k_min, k_max + 1))
    scan.insert(0, "tree_id", tree_id)
    log(f"[{tree_id}] silhouette掃描 k={k_min}..{k_max} 完成，{time.time()-t0:.0f}s")
    return scan


def recommend_k(scan: pd.DataFrame, max_share_cap: float = 0.5, log=print) -> dict:
    """選輪廓係數最高、且不退化（最大群佔比 <= max_share_cap）的 k。

    若最高輪廓係數的 k 剛好是退化解（見 hrp.silhouette_scan 說明），退而求其次選
    「不退化的解裡輪廓係數最高」的 k，並在回傳裡誠實記錄有沒有發生這種退讓。
    """
    valid = scan[scan["max_share"] <= max_share_cap]
    if valid.empty:
        best = scan.loc[scan["silhouette"].idxmax()]
        return {"k": int(best["k"]), "silhouette": float(best["silhouette"]),
               "degenerate_fallback": True,
               "note": f"所有k的最大群佔比都超過{max_share_cap:.0%}，退而選輪廓係數最高的k"}
    best_valid = valid.loc[valid["silhouette"].idxmax()]
    best_overall = scan.loc[scan["silhouette"].idxmax()]
    degenerate = int(best_overall["k"]) != int(best_valid["k"])
    return {"k": int(best_valid["k"]), "silhouette": float(best_valid["silhouette"]),
           "degenerate_fallback": degenerate,
           "note": (f"全域輪廓係數最高的k={int(best_overall['k'])}（{best_overall['silhouette']:.4f}）"
                    f"是退化解（最大群佔比{best_overall['max_share']:.1%}），已排除"
                    if degenerate else "非退化解裡的最高點，就是全域最高點")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.cluster_count_selection")
    ap.add_argument("--tree", action="append", help="只跑指定樹（可重複給），預設三棵normal樹")
    ap.add_argument("--k-min", type=int, default=K_MIN)
    ap.add_argument("--k-max", type=int, default=K_MAX)
    ap.add_argument("--max-share-cap", type=float, default=0.5,
                    help="判定退化解的最大群佔比門檻，預設50%")
    a = ap.parse_args(argv)
    trees = a.tree or list(DEFAULT_TREES)

    log = print
    log("=" * 70)
    log(f"H-03 群數選擇：{trees}，k={a.k_min}..{a.k_max}")
    log("=" * 70)

    all_scans = []
    recommendations = {}
    for tid in trees:
        scan = scan_tree(tid, a.k_min, a.k_max, log)
        all_scans.append(scan)
        rec = recommend_k(scan, a.max_share_cap, log)
        recommendations[tid] = rec
        log(f"\n[{tid}] 掃描結果：")
        log(scan.to_string(index=False))
        log(f"\n[{tid}] 建議 k = {rec['k']}（silhouette={rec['silhouette']:.4f}）")
        log(f"  {rec['note']}")
        log(f"  （對照：目前寫死的 L1_TARGET=8）\n")

    out = pd.concat(all_scans, ignore_index=True)
    out_dir = paths.ROOT / "_analysis_outputs_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "cluster_count_silhouette_scan.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    log(f"\n輸出：{p}")

    log("\n" + "=" * 70)
    log("彙總建議")
    log("=" * 70)
    for tid, rec in recommendations.items():
        log(f"  {tid:12s} → k={rec['k']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
