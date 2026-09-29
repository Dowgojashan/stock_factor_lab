# -*- coding: utf-8 -*-
"""`_coverage_tilt_window0_extended.py`拆成TW/US兩支獨立process的其中一支——
原本合在一個process裡同時跑US建樹＋US MarketData＋TW MarketData＋跨市場
`months_long`，疊加起來記憶體壓力太大，實測CPU幾乎停滯、記憶體衝到10GB
（2026-09-29真的發生過，不是預防性猜測）。拆開後這支**只**載入TW自己的
資料，不需要`WF._load_inputs()`（候選池直接重用既有的parquet，不必重建樹）。

背景／方法說明見`_coverage_tilt_window0_extended.py`（已改名保留當歷史記錄，
不再是入口）。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._coverage_tilt_window0_tw
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402

from app.performance import measure  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _design_test_hot_segment import hot_segment, monthly_close  # noqa: E402
from _design_test_hotsample_frequency import load_variant_pools, momentum_asof  # noqa: E402
from _prelim_coverage_tilt_prevalidation import (quarter_ends,  # noqa: E402
                                                 resolve_holdings_and_coverage, tilt_weights)

BETAS = [0.0, 1.0, 2.0, 5.0, 10.0]
N_TW = 3
X_HOT = 0.10
WINDOW0 = {"is_end": "2012-12-31", "oos_start": "2013-01-01", "oos_end": "2014-12-31"}
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "coverage_tilt_window0_tw.csv"


def main():
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    pools = load_variant_pools()
    uids = pools["rolling_v1"][1]
    print(f"TW window0候選池：直接重用rolling window1（{len(uids)}檔），已確認IS對齊（2007-01~2012-12）")

    md = MarketData("TW")
    md_map = {"TW": md}
    monthly = monthly_close(md)
    checkpoints = [WINDOW0["is_end"]] + quarter_ends(WINDOW0["oos_start"], WINDOW0["oos_end"])

    per_beta_rets: dict[float, list[float]] = {b: [] for b in BETAS}
    ew_rets = []
    avg_frac_log = []
    for i in range(len(checkpoints) - 1):
        as_of, end = checkpoints[i], checkpoints[i + 1]
        try:
            mom = momentum_asof(monthly, pd.Timestamp(as_of), N_TW)
        except ValueError as e:
            print(f"  {as_of}：{e}，跳過")
            continue
        hot = set(hot_segment(mom, X_HOT))
        holdings, covers, avg_frac = resolve_holdings_and_coverage(md, idx, uids, as_of, hot)

        for b in BETAS:
            wt = tilt_weights(holdings, covers, uids, b)
            res = measure(md_map, wt, as_of, end)
            per_beta_rets[b].append(res["portfolio_realized_return"])
            if b == BETAS[0]:
                ew = res["by_market_benchmark"]["TW"]["equal_weight_benchmark_return"]
                if ew is None:
                    raise ValueError(f"TW window0 {as_of}~{end}：equal_weight_benchmark_return是None")
                ew_rets.append(ew)
                avg_frac_log.append(avg_frac)
        print(f"  checkpoint {as_of}~{end} 完成")

    n_years = len(ew_rets) / 4.0
    cum_ew = pd.Series([1 + r for r in ew_rets]).prod() - 1
    ann_ew = (1 + cum_ew) ** (1 / n_years) - 1
    avg_cov = sum(avg_frac_log) / len(avg_frac_log) if avg_frac_log else float("nan")

    rows = []
    for b in BETAS:
        rets = per_beta_rets[b]
        cum = pd.Series([1 + r for r in rets]).prod() - 1
        ann = (1 + cum) ** (1 / n_years) - 1
        print(f"  β={b:.1f}: 年化={ann:+.2%}（等權大盤={ann_ew:+.2%}｜超額={ann-ann_ew:+.2%}）"
             f"策略平均覆蓋比例={avg_cov:.1%}")
        rows.append({"market": "TW", "window_no": 0, "beta": b, "ann_return": ann,
                    "ann_ew_benchmark": ann_ew, "ann_excess": ann - ann_ew,
                    "avg_coverage_frac": avg_cov, "n_rep": len(uids)})

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {OUT_PATH}")


if __name__ == "__main__":
    main()
