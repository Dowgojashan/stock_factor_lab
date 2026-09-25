# -*- coding: utf-8 -*-
"""使用者質疑§3.18「6窗平均71.65%幾乎精確等於候選池72.7%」太巧合，只有window4
不一樣——查cluster層級：是不是每個HRP群的類別組成跨窗次很穩定，配額規則
（legacy=5檔/群）又是固定的，兩者疊加自然產生高度重複的總比例，不是純粹隨機。

對window1/3/5/6（4個都是73.3%的窗次）逐一印出6個群各自：①群大小②候選池內
估值/動量/現金流品質比例③實際被選中的5檔代表各是什麼類別，看模式是否重複。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._check_cluster_level_category_stability
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402

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
CHECK_WINDOWS = [1, 3, 5, 6]  # 這4窗都是73.3%


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
        print(f"\n{'='*70}\nwindow{window_no}：IS {is_start}~{is_end}")
        tree = WF.build_tree_for_window(TREE_KEY, is_start, is_end, months_long, meta_pool, f_combo_map, print)
        assign = tree["assign"]
        assign = assign[assign.tree_id == tree["tree_id"]] if "tree_id" in assign.columns else assign
        cmeta = tree["cluster_meta"]
        cmeta = cmeta[cmeta.level == WF.LEVEL] if "level" in cmeta.columns else cmeta

        uids = pd.Index(assign[WF.C.PK])
        wide_is = WF.S3._pivot_window(months_long, uids, is_start, is_end)
        cagr_is = WF._cagr_matrix(wide_is)
        mdd_is = WF._mdd_matrix(wide_is)
        quality_is = cagr_is / mdd_is.abs().replace(0, np.nan)

        corr_full = np.corrcoef(wide_is.to_numpy(dtype=np.float64))
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

        print(f"目標代表數={tot}｜quota逐群={dict(quota)}")
        for cid in sorted(sizes.index):
            csub = meta_all[meta_all.cluster == cid]
            cpicked = picked_meta[picked_meta.cluster == cid]
            pool_dist = csub["category"].value_counts(normalize=True).round(2).to_dict()
            pick_dist = cpicked["category"].value_counts().to_dict()
            print(f"  群{cid}｜候選{len(csub)}檔（池內比例{pool_dist}）｜配額{int(quota.get(cid,0))}"
                 f"｜實選{pick_dist}")


if __name__ == "__main__":
    main()
