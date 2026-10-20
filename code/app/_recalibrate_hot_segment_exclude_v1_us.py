# -*- coding: utf-8 -*-
"""2026-10-19：美股版——用exclude_v1（openSec_balanced）代表策略名單，獨立
校準Hot Segment的P25/P10門檻，跟`_recalibrate_hot_segment_exclude_v1.py`（TW）
平行；跟同一批這次新寫的`_recalibrate_hot_segment_baseline_us.py`成對，
一個baseline一個exclude_v1，不共用門檻。

背景與用法同`_recalibrate_hot_segment_baseline_us.py`的docstring，這裡
只換代表策略名單為`anchored_w4_openSec_balanced_exclude_v1.parquet`
（跟`_run_exclude_v1_openSec_balanced_mcapfix_cap30.py`實際production run
用的同一批名單來源）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _design_test_hot_segment import (CALIB_ENDS, CHECK_ENDS, QUARTER_ENDS,  # noqa: E402
                                      compute_coverage, hot_segment, monthly_close,
                                      _quarterly_returns)
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402

N = 6       # production現行值（monitor.HOT_SEGMENT_N["US"]），不重新搜尋
X = 0.10    # production現行值（monitor.HOT_SEGMENT_X），跨市場共用，不重新搜尋

EXCLUDE_V1_PICKS_PATH = (Path(__file__).resolve().parent.parent.parent
                        / "_analysis_outputs_applayer" / "anchored_w4_openSec_balanced_exclude_v1.parquet")


def main():
    print(">> 載入 US MarketData（2010起，涵蓋N=6個月的最早回顧期）...")
    md = MarketData("US", start="2010-01-01")

    idx_full = pd.read_parquet(CANDIDATE_INDEX_PATH)
    idx = idx_full[idx_full["market"] == "US"].set_index("strategy_uid")

    m = pd.read_parquet(EXCLUDE_V1_PICKS_PATH)
    sub = m[(m.tree_key == "US") & (m.scheme == "E") & (m.window_no == 4)
           & (m.k_mode == "silhouette_is") & (m.ratio == "legacy")
           & (m.allocation == "equal") & (m.group == "A_hrp")]
    assert len(sub) == 1, f"{EXCLUDE_V1_PICKS_PATH.name} 篩出{len(sub)}列，預期1列"
    rep_uids = list(sub.iloc[0]["members"])
    print(f"代表策略數：{len(rep_uids)}（exclude_v1，跟`_run_exclude_v1_openSec_"
         f"balanced_mcapfix_cap30.py`同一批名單）")

    mktcap = md.get_field("report:mktcap")

    print(">> 算月頻收盤價...")
    monthly = monthly_close(md)
    print(">> 算動能報酬...")
    ret_df = _quarterly_returns(monthly, N)

    rows = []
    for q in ret_df.index:
        hot = hot_segment(ret_df.loc[q], X)
        if not hot:
            continue
        as_of = q.date().isoformat()
        cov_count, cov_mktcap = compute_coverage(md, idx, rep_uids, mktcap, hot, as_of)
        period = "calib_2015_2023" if q in CALIB_ENDS else "check_2024_2025"
        rows.append(dict(quarter=as_of, period=period, n_hot=len(hot),
                         coverage_count=cov_count, coverage_mktcap=cov_mktcap))
        mk_str = f"{cov_mktcap:.1%}" if pd.notna(cov_mktcap) else "N/A"
        print(f"  {as_of}（{period}）：hot={len(hot)}檔｜coverage_count={cov_count:.1%}｜"
             f"coverage_mktcap={mk_str}")

    out = pd.DataFrame(rows)
    out_path = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
               / "hot_segment_coverage_test_US_exclude_v1.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}")

    calib = out[out.period == "calib_2015_2023"]["coverage_count"]
    print(f"\n=== 2015-2023校準期（{len(calib)}個季底）coverage_count分布 ===")
    print(calib.describe())
    p25_new = calib.quantile(0.25)
    p10_new = calib.quantile(0.10)
    print(f"\n舊門檻（美股目前baseline/exclude_v1共用的舊值）：P25=0.3039／P10=0.2595")
    print(f"這次exclude_v1獨立重新校準：P25={p25_new:.4f}／P10={p10_new:.4f}")

    check = out[out.period == "check_2024_2025"]
    n_below_p25_old = (check.coverage_count < 0.3039).sum()
    n_below_p25_new = (check.coverage_count < p25_new).sum()
    n_below_p10_old = (check.coverage_count < 0.2595).sum()
    n_below_p10_new = (check.coverage_count < p10_new).sum()
    print(f"\n2024-2025反查（{len(check)}季）：")
    print(f"  舊門檻下 <P25次數={n_below_p25_old}／<P10次數={n_below_p10_old}")
    print(f"  這次新門檻下 <P25次數={n_below_p25_new}／<P10次數={n_below_p10_new}")


if __name__ == "__main__":
    main()
