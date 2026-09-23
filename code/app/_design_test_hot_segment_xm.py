# -*- coding: utf-8 -*-
"""D④第一步（XM）：Hot Segment coverage 對 XM（台美混合投組）市場的校準。

XM 跟 TW/US 不一樣，不能直接套同一支`_design_test_hot_segment.py`：
  - XM window4 legacy/equal/A_hrp 的 15 個代表策略是**混合**的（5 檔 TW + 10 檔 US，
    已用`walkforward_members.parquet`實測確認，不是猜的）
  - 一檔 TW 策略的選股條件不可能選中 US 股票（反之亦然）——Hot Segment 若跨市場合併
    排名報酬，會有貨幣/尺度不可比的問題，也會讓 TW 策略對「US Hot Segment」的覆蓋率
    恆為 0、失去意義

設計決定（2026-09-22）：**TW 側跟 US 側分開算、分開報告，不強迫合成一個數字**——
沿用本專案既有的「不排優先順序，兩者都報告」原則（§7.4）。TW 側用 N=3（已在
`_design_test_hot_segment.py --market TW`確認的最佳值），US 側用 N=6（已在
`--market US`確認的最佳值），X 統一 10%，都是沿用已校準好的參數，不重新掃網格。

用法（cwd 必須是 code/）：
    python -m app._design_test_hot_segment_xm
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _design_test_hot_segment import (CALIB_ENDS, compute_coverage, hot_segment,  # noqa: E402
                                      monthly_close, _quarterly_returns)
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402

X = 0.10
N_BY_MARKET = {"TW": 3, "US": 6}   # 沿用各市場已校準出的最佳 N，不重新掃網格


def _one_side(market: str, rep_uids: list[str]) -> pd.DataFrame:
    """跟`_design_test_hot_segment.py::main()`共用`compute_coverage()`等核心邏輯，
    不再各自維護一份（2026-09-22 code review 抓到的複製貼上問題，見該檔案的說明）。"""
    n = N_BY_MARKET[market]
    print(f">> [{market}] 載入 MarketData（N={n}）...")
    md = MarketData(market, start="2010-01-01")
    idx_full = pd.read_parquet(CANDIDATE_INDEX_PATH)
    idx = idx_full[idx_full["market"] == market].set_index("strategy_uid")
    mktcap = md.get_field("report:mktcap")

    ret_df = _quarterly_returns(monthly_close(md), n)
    rows = []
    for q in ret_df.index:
        hot = hot_segment(ret_df.loc[q], X)
        if not hot:
            continue
        as_of = q.date().isoformat()
        cov_count, cov_mktcap = compute_coverage(md, idx, rep_uids, mktcap, hot, as_of)
        period = "calib_2015_2023" if q in CALIB_ENDS else "check_2024_2025"
        rows.append(dict(market=market, N=n, quarter=as_of, period=period, n_hot=len(hot),
                         n_rep=len(rep_uids), coverage_count=cov_count, coverage_mktcap=cov_mktcap))
        print(f"  [{market}] {as_of}（{period}）：hot={len(hot)}檔｜coverage_count={cov_count:.1%}")
    return pd.DataFrame(rows)


def main():
    members_path = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"
    m = pd.read_parquet(members_path)
    key = dict(tree_key="XM", scheme="E", window_no=4, k_mode="silhouette_is",
              ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    xm_members = list(sub.iloc[0]["members"])

    idx_full = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    market_of = idx_full.loc[xm_members, "market"]
    tw_uids = [u for u in xm_members if market_of[u] == "TW"]
    us_uids = [u for u in xm_members if market_of[u] == "US"]
    print(f"XM代表策略共{len(xm_members)}檔：TW側{len(tw_uids)}檔、US側{len(us_uids)}檔")

    tw_out = _one_side("TW", tw_uids)
    us_out = _one_side("US", us_uids)
    out = pd.concat([tw_out, us_out], ignore_index=True)

    out_path = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "hot_segment_coverage_test_XM.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}（{len(out)} 列）")

    for market in ["TW", "US"]:
        sub_out = out[out["market"] == market]
        calib = sub_out[sub_out["period"] == "calib_2015_2023"]
        check = sub_out[sub_out["period"] == "check_2024_2025"]
        ac = sub_out.sort_values("quarter")["coverage_count"].autocorr(lag=1)
        print(f"\n=== XM {market}側（{len(calib)}季校準, N={N_BY_MARKET[market]}, "
             f"代表策略{calib['n_rep'].iloc[0] if len(calib) else '?'}檔）===")
        print(f"autocorr(lag1)={ac:.3f}")
        if calib.empty:
            continue
        p25, p10 = calib["coverage_count"].quantile(0.25), calib["coverage_count"].quantile(0.10)
        print(f"歷史p25={p25:.1%}／p10={p10:.1%}")
        n_below_p25 = (check["coverage_count"] < p25).sum()
        n_below_p10 = (check["coverage_count"] < p10).sum()
        print(f"反查2024-2025（n={len(check)}）：{n_below_p25}季低於p25、{n_below_p10}季低於p10")
        for _, r in check.iterrows():
            pct = (calib["coverage_count"] < r["coverage_count"]).mean()
            flag = "低p10" if r["coverage_count"] < p10 else ("低p25" if r["coverage_count"] < p25 else "")
            print(f"  {r['quarter']}: {r['coverage_count']:.1%}（第{pct:.0%}百分位）{flag}")


if __name__ == "__main__":
    main()
