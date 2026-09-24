# -*- coding: utf-8 -*-
"""接續§3.12：把Hot Segment覆蓋率計算換成rolling window6候選池（跟§3.12真實跑的
是同一批50檔代表策略，silhouette_is，equal allocation），不要再借用anchored
window4的既有結果（§1.1）湊時間軸對照——這樣兩邊口徑才完全一致。

完全重用`_design_test_hot_segment.py`已驗證過的核心函式（`monthly_close`／
`_quarterly_returns`／`hot_segment`／`compute_coverage`），只換`rep_uids`來源，
N=3、X=10%（已在§1.1用anchored候選池校準確定，這裡沿用同一組參數，不重新校準）。

CALIB期（2015-2023，訂歷史分布）跟CHECK期（2024-2025，反查驗證）都用rolling
window6這批50檔策略的選股規則回算——`selects_symbol()`只是檢查策略的F1/F2/C/V
條件在某個歷史時點會不會選中某檔股票，不要求策略在該時點「正在使用」，所以
往回算2015-2023在方法論上是合理的（跟`_design_test_hot_segment.py`原本的做法
精神一致，只是換了要檢查的策略清單）。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._check_hot_segment_rolling_w6
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _design_test_hot_segment import (CALIB_ENDS,  # noqa: E402
                                      compute_coverage, hot_segment, monthly_close,
                                      _quarterly_returns)

N = 3
X = 0.10
SILHOUETTE_PICKS_PATH = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
                         / "rolling_window6_silhouette_members.parquet")


def main():
    print(">> 載入 rolling window6 代表策略清單（silhouette_is, equal）...")
    m = pd.read_parquet(SILHOUETTE_PICKS_PATH)
    sub = m[(m.tree_key == "TW") & (m.scheme == "rolling_6_2") & (m.window_no == 6)
           & (m.k_mode == "silhouette_is") & (m.allocation == "equal") & (m.group == "A_hrp")]
    assert len(sub) == 1, "rolling window6 silhouette picks(equal)缺這一列"
    rep_uids = list(sub.iloc[0]["members"])
    print(f"代表策略數：{len(rep_uids)}（跟§3.12真實模擬同一批）")

    print(">> 載入 TW MarketData（2010起，涵蓋N=6個月的最早回顧期）...")
    md = MarketData("TW", start="2010-01-01")
    idx_full = pd.read_parquet(CANDIDATE_INDEX_PATH)
    idx = idx_full[idx_full["market"] == "TW"].set_index("strategy_uid")
    mktcap = md.get_field("report:mktcap")

    print(">> 算月頻收盤價、動能報酬（N=3）...")
    monthly = monthly_close(md)
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
        print(f"  {as_of}（{period}）：hot={len(hot)}檔｜coverage_count={cov_count:.1%}｜coverage_mktcap={mk_str}")

    out = pd.DataFrame(rows)
    out_path = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
               / "hot_segment_coverage_rolling_w6_TW.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}（{len(out)} 列）")

    calib = out[out["period"] == "calib_2015_2023"]
    check = out[out["period"] == "check_2024_2025"]
    p25 = calib["coverage_count"].quantile(0.25)
    p10 = calib["coverage_count"].quantile(0.10)
    print(f"\n=== rolling window6候選池：coverage_count歷史分布（2015-2023, n={len(calib)}）===")
    print(f"均值={calib['coverage_count'].mean():.1%}｜標準差={calib['coverage_count'].std():.1%}｜"
         f"p25={p25:.1%}｜p10={p10:.1%}")

    print("\n=== 反查驗證：2024-2025每季coverage，落在歷史分布第幾百分位 ===")
    for _, r in check.sort_values("quarter").iterrows():
        v = r["coverage_count"]
        pct = (calib["coverage_count"] < v).mean()
        flag = "🔴低於p10" if v < p10 else ("🔶低於p25" if v < p25 else "正常")
        print(f"  {r['quarter']}：{v:.1%}（第{pct:.0%}百分位）{flag}")


if __name__ == "__main__":
    main()
