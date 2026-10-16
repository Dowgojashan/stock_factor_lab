# -*- coding: utf-8 -*-
"""2026-10-14：用「排除V1」這個目前實際production使用的方法論，重新校準
Hot Segment的P25/P10門檻。

背景：window2／window4兩次真實8/12季LLM模擬都發現，我們實際用的exclude_v1
代表策略名單，coverage_count從未跌破現行門檻（P25=0.3363/P10=0.308）——
查`_design_test_hot_segment.py`才發現原因：**現行門檻是用`walkforward_
members.parquet`的baseline（含V1）名單校準的，不是exclude_v1**。已經證實
exclude_v1本身就會把Hot Segment覆蓋率從40.1%推高到53.6%（見4.2.3），用
baseline名單校準出來的門檻，對coverage天生較高的exclude_v1名單來說自然
形同虛設。

這支腳本完全沿用`_design_test_hot_segment.py`已驗證過的計算邏輯
（`compute_coverage`／`hot_segment`／`_quarterly_returns`，直接import重用，
不複製貼上），只把代表策略名單換成`anchored_w4_openSec_boost_exclude_v1.
parquet`，並固定用現行production已經拍板的N=3/X=10%（不重跑N/X網格搜尋，
那一步已經做過、沒有理由因為換名單重新選一次）。
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

N = 3       # production現行值（monitor.HOT_SEGMENT_N["TW"]），不重新搜尋
X = 0.10    # production現行值（monitor.HOT_SEGMENT_X），不重新搜尋

EXCLUDE_V1_PICKS_PATH = (Path(__file__).resolve().parent.parent.parent
                        / "_analysis_outputs_applayer" / "anchored_w4_openSec_boost_exclude_v1.parquet")


def main():
    print(">> 載入 TW MarketData（2010起，涵蓋N=3個月的最早回顧期）...")
    md = MarketData("TW", start="2010-01-01")

    idx_full = pd.read_parquet(CANDIDATE_INDEX_PATH)
    idx = idx_full[idx_full["market"] == "TW"].set_index("strategy_uid")

    m = pd.read_parquet(EXCLUDE_V1_PICKS_PATH)
    sub = m[(m.tree_key == "TW") & (m.allocation == "equal")]
    assert len(sub) == 1, f"{EXCLUDE_V1_PICKS_PATH.name} 缺 equal 這一列"
    rep_uids = list(sub.iloc[0]["members"])
    print(f"代表策略數：{len(rep_uids)}（exclude_v1，跟window4/window2真實模擬用的同一批名單）")

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
               / "hot_segment_coverage_test_TW_exclude_v1.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}")

    calib = out[out.period == "calib_2015_2023"]["coverage_count"]
    print(f"\n=== 2015-2023校準期（{len(calib)}個季底）coverage_count分布 ===")
    print(calib.describe())
    p25_new = calib.quantile(0.25)
    p10_new = calib.quantile(0.10)
    print(f"\n舊門檻（baseline名單校準）：P25=0.3363／P10=0.3080")
    print(f"新門檻（exclude_v1名單校準）：P25={p25_new:.4f}／P10={p10_new:.4f}")

    check = out[out.period == "check_2024_2025"]
    n_below_p25_old = (check.coverage_count < 0.3363).sum()
    n_below_p25_new = (check.coverage_count < p25_new).sum()
    n_below_p10_old = (check.coverage_count < 0.3080).sum()
    n_below_p10_new = (check.coverage_count < p10_new).sum()
    print(f"\n2024-2025反查（{len(check)}季）：")
    print(f"  舊門檻下 <P25次數={n_below_p25_old}／<P10次數={n_below_p10_old}")
    print(f"  新門檻下 <P25次數={n_below_p25_new}／<P10次數={n_below_p10_new}")


if __name__ == "__main__":
    main()
