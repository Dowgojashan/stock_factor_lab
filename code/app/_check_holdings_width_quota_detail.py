# -*- coding: utf-8 -*-
"""使用者問「為什麼持股寬度版配額規則，anchored改善明顯、rolling改善有限」——
直接查兩窗底下逐群的平均持股數跟配額變化幅度，看是不是anchored的配額調整
本來就比rolling劇烈（權重差異更大→重分配更激進），不用推論回答。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._check_holdings_width_quota_detail
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
from _design_test_rolling_is6oos2 import RATIO, TREE_KEY, window_dates_rolling  # noqa: E402
from _design_test_rolling_is6oos2 import IS_MONTHS, OOS_LEN  # noqa: E402

ANCHORED_IS_START, ANCHORED_IS_END = "2007-01", "2023-12"
ROLLING_WINDOW_NO = 6


def run_window(label: str, is_start: str, is_end: str, months_long, meta_pool, f_combo_map, idx: pd.DataFrame):
    print(f"\n{'='*60}\n{label}：IS {is_start}~{is_end}")
    tree = WF.build_tree_for_window(TREE_KEY, is_start, is_end, months_long, meta_pool, f_combo_map, print)
    scan = k_stability.scan_window(TREE_KEY, is_start, is_end, months_long, meta_pool, log=print)
    k_is = scan["k_is_selected"]

    uids = pd.Index(tree["assign"][WF.C.PK])
    wide_is = S3._pivot_window(months_long, uids, is_start, is_end)
    corr_full = np.corrcoef(wide_is.to_numpy(dtype=np.float64))
    labels = hrp.cut_clusters(tree["link"], k_is)
    assign = pd.DataFrame({WF.C.PK: wide_is.index.to_numpy(), f"cluster_{WF.LEVEL}": labels})

    meta = idx.reindex(list(uids)).copy()
    sizes = assign.groupby(f"cluster_{WF.LEVEL}").size()
    k = len(sizes)
    n_uni = len(uids)
    tot = WF.target_total(RATIO, n_uni, k)

    quota_base, _ = WF.allocate(sizes, tot, "equal")
    avg_hold_by_cluster = cluster_avg_holdings(assign, meta)
    quota_hw, _ = allocate_by_weight(sizes, avg_hold_by_cluster, tot)

    print(f"k={k}｜目標{tot}檔｜群平均持股範圍：{avg_hold_by_cluster.min():.1f}~{avg_hold_by_cluster.max():.1f}檔"
         f"（變異係數CV={avg_hold_by_cluster.std()/avg_hold_by_cluster.mean():.2f}）")
    df = pd.DataFrame({"cluster_size": sizes, "avg_holdings": avg_hold_by_cluster,
                       "quota_baseline": quota_base, "quota_holdings_width": quota_hw})
    df["quota變化"] = df["quota_holdings_width"] - df["quota_baseline"]
    print(df.sort_values("avg_holdings").to_string())

    total_shift = df["quota變化"].abs().sum() / 2  # 除以2因為增減會互相抵銷成對計算
    print(f"配額重分配總幅度（|變化|加總/2）：{total_shift:.1f}檔，佔目標{tot}檔的{total_shift/tot:.1%}")
    return df


def main():
    print(">> 載入資料...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    run_window("anchored_w4", ANCHORED_IS_START, ANCHORED_IS_END, months_long, meta_pool, f_combo_map, idx)

    blocks = WF._blocks(WF.SCHEME_TOTAL_MONTHS, IS_MONTHS, OOS_LEN)
    off, L = blocks[ROLLING_WINDOW_NO - 1]
    r_is_start, r_is_end, _, _ = window_dates_rolling(off, L, TREE_KEY)
    run_window("rolling_w6", r_is_start, r_is_end, months_long, meta_pool, f_combo_map, idx)


if __name__ == "__main__":
    main()
