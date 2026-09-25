# -*- coding: utf-8 -*-
"""接續§3.19：使用者問「這個發現是在k_mode='fixed'（k=6固定）下驗證的，如果讓
k隨每個窗次現場算（silhouette_is，像§3.12查到window6會變成k=10），這個『純估值
大群＋固定配額』的模式還站得住嗎」——這裡把§3.19的4個窗次（1/3/5/6）換成
silhouette現場選k版本重跑一次cluster拆解。

沿用`_silhouette_picks_lib.py`同一套「建樹→現場選k→重切」邏輯（不重複貼上，
直接複製該函式的前半段到現場選k完成為止，因為這裡需要中間的assign/cmeta/quota
明細去印逐群分布，`_silhouette_picks_lib.build_silhouette_picks()`只回傳最終
picks，沒有暴露這些中間值）。只測baseline（保留V1），跟§3.19原本的範圍一致。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._check_cluster_stability_silhouette_k
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402

from research import hrp  # noqa: E402
from research import k_stability  # noqa: E402
from research import stage3_hrp as S3  # noqa: E402
from research import walkforward_matrix as WF  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _design_test_rolling_is6oos2 import (IS_MONTHS, OOS_LEN, RATIO, ALLOCATION,  # noqa: E402
                                          TREE_KEY, window_dates_rolling)

FACTOR_CATEGORY = {
    "ROE": "資本報酬/獲利能力", "ROIC": "資本報酬/獲利能力", "CROIC": "資本報酬/獲利能力",
    "EPS": "獲利", "FCF_P": "現金流品質", "FCF_OI": "現金流品質", "OCF_E": "現金流品質",
    "EV_EBITDA": "估值倍數", "EV_S": "估值倍數", "PB": "估值倍數", "PS": "估值倍數", "P_IC": "估值倍數",
    "MOM": "動量",
}
CHECK_WINDOWS = [1, 3, 5, 6]  # 跟§3.19同一批，方便逐一比對fixed vs silhouette


def categorize(row: pd.Series) -> str:
    return FACTOR_CATEGORY.get(row["F1_factor"], f"未分類({row['F1_factor']})")


def main():
    print(">> 載入資料...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    blocks = WF._blocks(WF.SCHEME_TOTAL_MONTHS, IS_MONTHS, OOS_LEN)
    for window_no in CHECK_WINDOWS:
        off, L = blocks[window_no - 1]
        is_start, is_end, oos_start, oos_end = window_dates_rolling(off, L, TREE_KEY)
        print(f"\n{'='*70}\nwindow{window_no}（silhouette_is）：IS {is_start}~{is_end}")

        tree = WF.build_tree_for_window(TREE_KEY, is_start, is_end, months_long, meta_pool, f_combo_map, print)
        scan = k_stability.scan_window(TREE_KEY, is_start, is_end, months_long, meta_pool, log=print)
        k_is = scan["k_is_selected"]
        print(f"   IS選k={k_is}（寫死版k={scan['k_fixed']}，{'一致' if scan['same_as_fixed'] else '不同'}）")

        uids = pd.Index(tree["assign"][WF.C.PK])
        wide_is = S3._pivot_window(months_long, uids, is_start, is_end)
        corr_full = np.corrcoef(wide_is.to_numpy(dtype=np.float64))

        n_leaf = len(tree["link"]) + 1
        if n_leaf != len(wide_is):
            raise AssertionError(f"linkage葉節點數{n_leaf} != wide_is列數{len(wide_is)}，中止")
        labels = hrp.cut_clusters(tree["link"], k_is)
        cmeta, _ = S3._cluster_meta_and_corr(wide_is, corr_full, labels, WF.LEVEL, tree["tree_id"])
        assign = pd.DataFrame({WF.C.PK: wide_is.index.to_numpy(), f"cluster_{WF.LEVEL}": labels})

        cagr_is = WF._cagr_matrix(wide_is)
        mdd_is = WF._mdd_matrix(wide_is)
        quality_is = cagr_is / mdd_is.abs().replace(0, np.nan)
        pos = pd.Series(range(len(wide_is.index)), index=wide_is.index)

        sizes = assign.groupby(f"cluster_{WF.LEVEL}").size()
        k = len(sizes)
        n_uni = len(uids)
        tot = WF.target_total(RATIO, n_uni, k)
        quota, n_capped = WF.allocate(sizes, tot, ALLOCATION)
        a_members, n_bf = WF._pick_a(assign, cmeta, wide_is, quality_is, quota, corr_full, pos)

        meta_all = idx.reindex(list(uids)).copy()
        meta_all["category"] = meta_all.apply(categorize, axis=1)
        cluster_map = assign.set_index(WF.C.PK)[f"cluster_{WF.LEVEL}"]
        meta_all["cluster"] = cluster_map.reindex(meta_all.index)

        picked_meta = idx.reindex(a_members).copy()
        picked_meta["category"] = picked_meta.apply(categorize, axis=1)
        picked_meta["cluster"] = cluster_map.reindex(picked_meta.index)

        v_pick_counts = picked_meta["category"].value_counts()
        n_val = int(v_pick_counts.get("估值倍數", 0))
        print(f"目標代表數={tot}｜實選={len(a_members)}｜k={k}群｜quota逐群={dict(quota)}")
        print(f"估值倍數總計：{n_val}/{len(a_members)}（{n_val/len(a_members):.1%}）")
        for cid in sorted(sizes.index):
            csub = meta_all[meta_all.cluster == cid]
            cpicked = picked_meta[picked_meta.cluster == cid]
            pool_dist = csub["category"].value_counts(normalize=True).round(2).to_dict()
            pick_dist = cpicked["category"].value_counts().to_dict()
            print(f"  群{cid}｜候選{len(csub)}檔（池內比例{pool_dist}）｜配額{int(quota.get(cid,0))}"
                 f"｜實選{pick_dist}")


if __name__ == "__main__":
    main()
