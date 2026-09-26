# -*- coding: utf-8 -*-
"""接續上一步：配額重分配幅度兩窗差不多，但OOS結果差很多——查「換進/換出」的
具體策略，個別OOS表現如何，才能回答為什麼anchored的换入策略比rolling的更好。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._check_holdings_width_swap_detail
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
from _design_test_quota_rules import allocate_by_weight  # noqa: E402
from _design_test_quota_holdings_width import cluster_avg_holdings  # noqa: E402
from _design_test_rolling_is6oos2 import (IS_MONTHS, OOS_LEN, RATIO,  # noqa: E402
                                          TREE_KEY, window_dates_rolling)

FACTOR_CATEGORY = {
    "ROE": "資本報酬/獲利能力", "ROIC": "資本報酬/獲利能力", "CROIC": "資本報酬/獲利能力",
    "EPS": "獲利", "FCF_P": "現金流品質", "FCF_OI": "現金流品質", "OCF_E": "現金流品質",
    "EV_EBITDA": "估值倍數", "EV_S": "估值倍數", "PB": "估值倍數", "PS": "估值倍數", "P_IC": "估值倍數",
    "MOM": "動量",
}
ANCHORED_IS_START, ANCHORED_IS_END = "2007-01", "2023-12"
ANCHORED_OOS_START, ANCHORED_OOS_END = "2024-01", "2025-12"
ROLLING_WINDOW_NO = 6


def categorize(row: pd.Series) -> str:
    return FACTOR_CATEGORY.get(row["F1_factor"], f"未分類({row['F1_factor']})")


def run_window(label: str, is_start: str, is_end: str, oos_start: str, oos_end: str,
              months_long, meta_pool, f_combo_map, idx: pd.DataFrame):
    print(f"\n{'='*70}\n{label}：IS {is_start}~{is_end}｜OOS {oos_start}~{oos_end}")
    tree = WF.build_tree_for_window(TREE_KEY, is_start, is_end, months_long, meta_pool, f_combo_map, print)
    scan = k_stability.scan_window(TREE_KEY, is_start, is_end, months_long, meta_pool, log=print)
    k_is = scan["k_is_selected"]

    uids = pd.Index(tree["assign"][WF.C.PK])
    wide_is = S3._pivot_window(months_long, uids, is_start, is_end)
    w = months_long[months_long.strategy_uid.isin(set(uids))]
    w = w[(w.month >= pd.Period(oos_start, "M")) & (w.month <= pd.Period(oos_end, "M"))]
    wide_oos = w.pivot(index="strategy_uid", columns="month", values="ret")

    corr_full = np.corrcoef(wide_is.to_numpy(dtype=np.float64))
    labels = hrp.cut_clusters(tree["link"], k_is)
    cmeta, _ = S3._cluster_meta_and_corr(wide_is, corr_full, labels, WF.LEVEL, tree["tree_id"])
    assign = pd.DataFrame({WF.C.PK: wide_is.index.to_numpy(), f"cluster_{WF.LEVEL}": labels})

    cagr_is = WF._cagr_matrix(wide_is)
    mdd_is = WF._mdd_matrix(wide_is)
    quality_is = cagr_is / mdd_is.abs().replace(0, np.nan)
    pos = pd.Series(range(len(wide_is.index)), index=wide_is.index)

    meta = idx.reindex(list(uids)).copy()
    meta["category"] = meta.apply(categorize, axis=1)

    sizes = assign.groupby(f"cluster_{WF.LEVEL}").size()
    k = len(sizes)
    n_uni = len(uids)
    tot = WF.target_total(RATIO, n_uni, k)

    quota_base, _ = WF.allocate(sizes, tot, "equal")
    a_base, _ = WF._pick_a(assign, cmeta, wide_is, quality_is, quota_base, corr_full, pos)

    avg_hold_by_cluster = cluster_avg_holdings(assign, meta)
    quota_hw, _ = allocate_by_weight(sizes, avg_hold_by_cluster, tot)
    a_hw, _ = WF._pick_a(assign, cmeta, wide_is, quality_is, quota_hw, corr_full, pos)

    added = sorted(set(a_hw) - set(a_base))
    removed = sorted(set(a_base) - set(a_hw))
    print(f"換進{len(added)}檔／換出{len(removed)}檔（共同保留{len(set(a_base)&set(a_hw))}檔）")

    def strategy_oos_cagr(uid: str) -> float:
        if uid not in wide_oos.index:
            return float("nan")
        ret = wide_oos.loc[uid].dropna()
        if len(ret) == 0:
            return float("nan")
        return float((1 + ret).prod() ** (12 / len(ret)) - 1)

    def summarize(uids_list: list[str], label: str):
        if not uids_list:
            print(f"  {label}：（空）")
            return
        rows = []
        for uid in uids_list:
            row = idx.loc[uid]
            rows.append({
                "uid": uid[:60], "category": categorize(row), "V": row["V"],
                "avg_holdings": row["avg_holdings"], "is_calmar": float(quality_is.get(uid, np.nan)),
                "oos_cagr": strategy_oos_cagr(uid),
            })
        df = pd.DataFrame(rows)
        print(f"  {label}（n={len(df)}）：平均持股={df['avg_holdings'].mean():.1f}｜"
             f"V1比例={float((df['V']=='v1').mean()):.1%}｜平均IS_Calmar={df['is_calmar'].mean():.3f}｜"
             f"平均OOS_CAGR={df['oos_cagr'].mean():+.2%}")
        print(df.to_string(index=False))

    summarize(added, "換進（holdings_width多選的）")
    summarize(removed, "換出（baseline原本選的，這次被排除）")


def main():
    print(">> 載入資料...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    run_window("anchored_w4", ANCHORED_IS_START, ANCHORED_IS_END, ANCHORED_OOS_START, ANCHORED_OOS_END,
              months_long, meta_pool, f_combo_map, idx)

    blocks = WF._blocks(WF.SCHEME_TOTAL_MONTHS, IS_MONTHS, OOS_LEN)
    off, L = blocks[ROLLING_WINDOW_NO - 1]
    r_is_start, r_is_end, r_oos_start, r_oos_end = window_dates_rolling(off, L, TREE_KEY)
    run_window("rolling_w6", r_is_start, r_is_end, r_oos_start, r_oos_end,
              months_long, meta_pool, f_combo_map, idx)


if __name__ == "__main__":
    main()
