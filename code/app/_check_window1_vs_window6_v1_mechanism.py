# -*- coding: utf-8 -*-
"""使用者問「為什麼window6（silhouette k=10, 80%估值）比window1（k=11, 65.5%）
估值佔比更高」——查cluster明細發現關鍵差異：window1有2個群（群4/群10）在動量
佔多數時，Calmar排序也真的選出動量為主（甚至0估值）；window6的動量佔多數的
群（群6/群8）Calmar排序卻還是讓估值type贏，沒有讓動量拿下多數配額。

假設：這是§3.6/3.7已經查出的「窄持股→短窗MDD估計雜訊→Calmar虛高」機制在
window6這個IS期間特別強（V1策略比例/窄持股比例特別高），直接查證V1_frac跟
avg_holdings，不要用推論回答。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._check_window1_vs_window6_v1_mechanism
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
from _design_test_quality_metric_variants import eligible_uids  # noqa: E402
from _design_test_rolling_is6oos2 import (IS_MONTHS, OOS_LEN, RATIO, ALLOCATION,  # noqa: E402
                                          TREE_KEY, window_dates_rolling)

FACTOR_CATEGORY = {
    "ROE": "資本報酬/獲利能力", "ROIC": "資本報酬/獲利能力", "CROIC": "資本報酬/獲利能力",
    "EPS": "獲利", "FCF_P": "現金流品質", "FCF_OI": "現金流品質", "OCF_E": "現金流品質",
    "EV_EBITDA": "估值倍數", "EV_S": "估值倍數", "PB": "估值倍數", "PS": "估值倍數", "P_IC": "估值倍數",
    "MOM": "動量",
}
CHECK_WINDOWS = [1, 6]
VARIANTS = ["baseline", "exclude_v1"]


def categorize(row: pd.Series) -> str:
    return FACTOR_CATEGORY.get(row["F1_factor"], f"未分類({row['F1_factor']})")


def run_window(window_no: int, variant: str, months_long, meta_pool, f_combo_map, idx: pd.DataFrame):
    blocks = WF._blocks(WF.SCHEME_TOTAL_MONTHS, IS_MONTHS, OOS_LEN)
    off, L = blocks[window_no - 1]
    is_start, is_end, oos_start, oos_end = window_dates_rolling(off, L, TREE_KEY)
    print(f"\n{'='*60}\nwindow{window_no}／{variant}：IS {is_start}~{is_end}")

    tree = WF.build_tree_for_window(TREE_KEY, is_start, is_end, months_long, meta_pool, f_combo_map, print)
    scan = k_stability.scan_window(TREE_KEY, is_start, is_end, months_long, meta_pool, log=print)
    k_is = scan["k_is_selected"]

    uids = pd.Index(tree["assign"][WF.C.PK])
    wide_is = S3._pivot_window(months_long, uids, is_start, is_end)
    corr_full = np.corrcoef(wide_is.to_numpy(dtype=np.float64))
    labels = hrp.cut_clusters(tree["link"], k_is)
    cmeta, _ = S3._cluster_meta_and_corr(wide_is, corr_full, labels, WF.LEVEL, tree["tree_id"])
    assign = pd.DataFrame({WF.C.PK: wide_is.index.to_numpy(), f"cluster_{WF.LEVEL}": labels})

    cagr_is = WF._cagr_matrix(wide_is)
    mdd_is = WF._mdd_matrix(wide_is)
    quality_is = cagr_is / mdd_is.abs().replace(0, np.nan)
    pos = pd.Series(range(len(wide_is.index)), index=wide_is.index)

    sizes_full = assign.groupby(f"cluster_{WF.LEVEL}").size()
    k = len(sizes_full)
    n_uni = len(uids)
    tot = WF.target_total(RATIO, n_uni, k)

    bench_cagr = WF._cagr(WF._portfolio_series(wide_is, list(uids)))
    elig = eligible_uids(variant, assign, idx, cagr_is, bench_cagr)
    assign_v = assign[assign[WF.C.PK].isin(elig)]
    quality_v = quality_is.reindex(list(elig))
    # 🔴 quota per-variant重算（§3.8教訓）
    sizes_v = assign_v.groupby(f"cluster_{WF.LEVEL}").size()
    quota, n_capped = WF.allocate(sizes_v, tot, ALLOCATION)
    a_members, n_bf = WF._pick_a(assign_v, cmeta, wide_is, quality_v, quota, corr_full, pos)

    sub = idx.reindex(a_members).copy()
    sub["category"] = sub.apply(categorize, axis=1)
    sub["mdd_is"] = mdd_is.reindex(a_members)
    sub["cagr_is"] = cagr_is.reindex(a_members)
    sub["calmar_is"] = quality_is.reindex(a_members)

    v1_frac = float((sub["V"] == "v1").mean())
    avg_hold = float(sub["avg_holdings"].mean())
    print(f"k={k}｜目標{tot}檔｜實選{len(a_members)}檔（backfill={n_bf}）｜V1比例={v1_frac:.1%}｜平均持股={avg_hold:.1f}檔")
    print(f"估值佔比={float((sub['category']=='估值倍數').mean()):.1%}")

    print("分類層級：平均持股／V1比例／平均IS MDD／平均Calmar")
    grp = sub.groupby("category").agg(
        n=("category", "size"), avg_holdings=("avg_holdings", "mean"),
        v1_frac=("V", lambda s: (s == "v1").mean()),
        mean_mdd_is=("mdd_is", "mean"), mean_calmar_is=("calmar_is", "mean"))
    print(grp.to_string())
    return sub


def main():
    print(">> 載入資料...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    for w in CHECK_WINDOWS:
        for variant in VARIANTS:
            run_window(w, variant, months_long, meta_pool, f_combo_map, idx)


if __name__ == "__main__":
    main()
