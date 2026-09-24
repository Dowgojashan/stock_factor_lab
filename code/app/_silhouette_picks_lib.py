# -*- coding: utf-8 -*-
"""共用函式庫：給一個IS窗（anchored或rolling皆可）＋一個變體（baseline／
exclude_v1），算出k_mode="silhouette_is"版本的代表策略名單（equal／proportional
兩種allocation）。

抽成共用模組的原因：`_check_rolling_window6_silhouette_picks.py`（§3.12）已經
驗證過一次「建樹→現場選k→重切→per-variant quota挑代表」這套流程只適用rolling
window6、且只有baseline（保留V1）一種變體。這次要擴大成anchored window4＋
rolling window6 × baseline＋exclude_v1 共4種組合，直接複製貼上4次容易讓
quota修正（§3.8的code review教訓）或silhouette重切邏輯（§3.12的教訓：uids
必須用`tree["assign"]`，不能用`S3._tree_universe`原始名單）只改到一份、
另一份悄悄過期，所以抽成單一函式讓4個組合共用同一套邏輯。

沿用`_design_test_quality_metric_variants.py::eligible_uids()`做V1篩選，
沿用`walkforward_matrix.py::run_wtc()`第487~513行的silhouette_is重切邏輯
（見`_check_rolling_window6_silhouette_picks.py`docstring的詳細說明）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import hrp  # noqa: E402
from research import k_stability  # noqa: E402
from research import stage3_hrp as S3  # noqa: E402
from research import walkforward_matrix as WF  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _design_test_quality_metric_variants import eligible_uids  # noqa: E402


def build_silhouette_picks(tree_key: str, is_start: str, is_end: str, variant: str,
                           months_long, meta_pool, f_combo_map, idx: pd.DataFrame,
                           ratio: str = "legacy", log=print) -> dict:
    """回傳 {"equal": [...], "proportional": [...], "_meta": {...}}。

    `variant`："baseline"（保留V1，全部候選人）或"exclude_v1"（沿用
    `eligible_uids()`篩選邏輯）。
    """
    log(f">> 建樹（fixed k_mode，取得可重切的linkage）IS {is_start}~{is_end}...")
    tree = WF.build_tree_for_window(tree_key, is_start, is_end, months_long, meta_pool, f_combo_map, log)

    log(">> 只用這個IS窗的資料現場選k（k_stability.scan_window，不看未來）...")
    scan = k_stability.scan_window(tree_key, is_start, is_end, months_long, meta_pool, log=log)
    k_is = scan["k_is_selected"]
    log(f"   IS選k={k_is}（寫死版k={scan['k_fixed']}，{'一致' if scan['same_as_fixed'] else '不同'}）")

    # 🔴 uids一定要用tree["assign"]（`_build_tree`內部`_drop_zero_variance`後的
    # 名單），不能用`S3._tree_universe`的原始名單——見§3.12/`_check_rolling_
    # window6_silhouette_picks.py`docstring記過的同一個教訓。
    uids = pd.Index(tree["assign"][WF.C.PK])
    wide_is = S3._pivot_window(months_long, uids, is_start, is_end)
    corr_full = np.corrcoef(wide_is.to_numpy(dtype=np.float64))

    n_leaf = len(tree["link"]) + 1
    if n_leaf != len(wide_is):
        raise AssertionError(f"linkage葉節點數{n_leaf} != wide_is列數{len(wide_is)}，"
                             f"重切的群標籤會對錯策略，中止")
    labels = hrp.cut_clusters(tree["link"], k_is)
    cmeta, _ = S3._cluster_meta_and_corr(wide_is, corr_full, labels, WF.LEVEL, tree["tree_id"])
    assign = pd.DataFrame({WF.C.PK: wide_is.index.to_numpy(), f"cluster_{WF.LEVEL}": labels})

    cagr_is = WF._cagr_matrix(wide_is)
    mdd_is = WF._mdd_matrix(wide_is)
    quality_is_base = cagr_is / mdd_is.abs().replace(0, np.nan)
    pos = pd.Series(range(len(wide_is.index)), index=wide_is.index)

    sizes_full = assign.groupby(f"cluster_{WF.LEVEL}").size()
    k = len(sizes_full)
    n_uni = len(uids)
    tot = WF.target_total(ratio, n_uni, k)
    log(f"   n_universe={n_uni}｜k={k}（silhouette選出）｜目標代表數={tot}")

    bench_cagr = WF._cagr(WF._portfolio_series(wide_is, list(uids)))
    elig = eligible_uids(variant, assign, idx, cagr_is, bench_cagr)
    assign_v = assign[assign[WF.C.PK].isin(elig)]
    if assign_v.empty:
        raise ValueError(f"[{variant}] 沒有任何策略通過篩選")
    quality_v = quality_is_base.reindex(list(elig))

    out = {}
    for allocation in ("equal", "proportional"):
        # 🔴 quota per-variant重算（§3.8的code review教訓）：用篩選後的群大小
        # 重新呼叫`WF.allocate()`，tot維持用篩選前的全宇宙算出的同一個目標值。
        sizes_v = assign_v.groupby(f"cluster_{WF.LEVEL}").size()
        quota, n_capped = WF.allocate(sizes_v, tot, allocation)
        a_members, n_bf = WF._pick_a(assign_v, cmeta, wide_is, quality_v, quota, corr_full, pos)
        log(f"   [{variant}/{allocation}] 選出 {len(a_members)} 檔（目標{tot}，backfill={n_bf}）")
        out[allocation] = a_members
    out["_meta"] = {"k_is_selected": k_is, "k_fixed": scan["k_fixed"], "n_universe": n_uni,
                    "n_eligible": len(elig), "target_total": tot,
                    "is_start": is_start, "is_end": is_end, "variant": variant}
    return out
