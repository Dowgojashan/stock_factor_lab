# -*- coding: utf-8 -*-
"""2026-10-17：用baseline（含V1）代表策略名單，獨立重新校準Hot Segment的
P25/P10門檻——跟`_recalibrate_hot_segment_exclude_v1.py`（exclude_v1專用）
平行，不共用同一個門檻。

背景：2026-10-14已經發現並修正「exclude_v1名單套用baseline校準出來的
門檻會形同虛設」這個問題，重新校準出exclude_v1專用的P25=0.5743/P10=0.5085。
但`triggers.HOT_SEGMENT_P25["TW"]`/`P10["TW"]`是單一市場層級常數，沒有
分名單——這代表baseline（含V1）名單如果直接套用exclude_v1校準出來的新
門檻，等於拿「別人的」門檐校準結果硬套，可能造成過度頻繁觸發（baseline
coverage天生比exclude_v1低，更容易跌破一個「為exclude_v1訂」的門檐）。

這支腳本直接沿用`_recalibrate_hot_segment_exclude_v1.py`已驗證過的計算
邏輯（`compute_coverage`／`hot_segment`／`_quarterly_returns`，同樣直接
import重用），只把代表策略名單換成baseline（從`walkforward_members.
parquet`讀scheme=E/window_no=4/k_mode=silhouette_is/ratio=legacy/
allocation=equal/group=A_hrp這一列，跟main production run `formal_8q_
control0_L2_openSec_boost_v4`實際使用的名單來源完全一致，2026-10-17已
查證跟該run的真實checkpoint持股逐一比對一致），N=3/X=10%同樣沿用
production現行值不重新搜尋。

✅ 2026-10-19：這支腳本算出的0.4269/0.4127本來一直沒有地方存（第8行說的
「單一市場層級常數，沒有分名單」問題），論文session核對機制盤點表時
連帶抓到並確認。已把`triggers.HOT_SEGMENT_P25`/`P10`改成`{market: {pool:
值}}`兩層dict，這兩個數字正式存進`["TW"]["baseline"]`，`register_hot_
segment(market, pool)`現在強制要求明確指定pool，不再可能悄悄套用錯的
候選池門檻。
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

WALKFORWARD_MEMBERS_PATH = (Path(__file__).resolve().parent.parent.parent
                           / "_analysis_outputs_robustness" / "walkforward_members.parquet")


def main():
    print(">> 載入 TW MarketData（2010起，涵蓋N=3個月的最早回顧期）...")
    md = MarketData("TW", start="2010-01-01")

    idx_full = pd.read_parquet(CANDIDATE_INDEX_PATH)
    idx = idx_full[idx_full["market"] == "TW"].set_index("strategy_uid")

    m = pd.read_parquet(WALKFORWARD_MEMBERS_PATH)
    sub = m[(m.tree_key == "TW") & (m.scheme == "E") & (m.window_no == 4)
           & (m.k_mode == "silhouette_is") & (m.ratio == "legacy")
           & (m.allocation == "equal") & (m.group == "A_hrp")]
    assert len(sub) == 1, f"walkforward_members.parquet 篩出{len(sub)}列，預期1列"
    rep_uids = list(sub.iloc[0]["members"])
    print(f"代表策略數：{len(rep_uids)}（baseline/含V1，跟main production run"
         f"`formal_8q_control0_L2_openSec_boost_v4`同一批名單，2026-10-17已逐檔比對驗證）")

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
               / "hot_segment_coverage_test_TW_baseline.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}")

    calib = out[out.period == "calib_2015_2023"]["coverage_count"]
    print(f"\n=== 2015-2023校準期（{len(calib)}個季底）coverage_count分布 ===")
    print(calib.describe())
    p25_new = calib.quantile(0.25)
    p10_new = calib.quantile(0.10)
    print(f"\n舊門檻（最初baseline名單校準，2026-09-29前）：P25=0.3363／P10=0.3080")
    print(f"exclude_v1專用門檻（2026-10-14校準，供對照不可混用）：P25=0.5743／P10=0.5085")
    print(f"這次baseline獨立重新校準：P25={p25_new:.4f}／P10={p10_new:.4f}")

    check = out[out.period == "check_2024_2025"]
    n_below_p25_old = (check.coverage_count < 0.3363).sum()
    n_below_p25_new = (check.coverage_count < p25_new).sum()
    n_below_p10_old = (check.coverage_count < 0.3080).sum()
    n_below_p10_new = (check.coverage_count < p10_new).sum()
    print(f"\n2024-2025反查（{len(check)}季）：")
    print(f"  舊門檻下 <P25次數={n_below_p25_old}／<P10次數={n_below_p10_old}")
    print(f"  這次新門檻下 <P25次數={n_below_p25_new}／<P10次數={n_below_p10_new}")


if __name__ == "__main__":
    main()
