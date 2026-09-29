# -*- coding: utf-8 -*-
"""診斷用：`_coverage_tilt_window0_tw.py`疑似卡住，這裡只跑第一個checkpoint、
每一步都print並強制flush，找出真正卡在哪一行。用完即可刪除，不是正式產出。
"""
from __future__ import annotations

import sys
import time
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
from _prelim_coverage_tilt_prevalidation import resolve_holdings_and_coverage, tilt_weights  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("start")
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    pools = load_variant_pools()
    uids = pools["rolling_v1"][1]
    log(f"pools loaded, {len(uids)} uids")

    md = MarketData("TW")
    log("MarketData loaded")
    monthly = monthly_close(md)
    log("monthly_close computed")

    as_of = "2012-12-31"
    end = "2013-03-31"
    mom = momentum_asof(monthly, pd.Timestamp(as_of), 3)
    log("momentum computed")
    hot = set(hot_segment(mom, 0.10))
    log(f"hot segment computed, {len(hot)} symbols")

    log("starting resolve_holdings_and_coverage...")
    t0 = time.time()
    holdings, covers, avg_frac = resolve_holdings_and_coverage(md, idx, uids, as_of, hot)
    log(f"resolve_holdings_and_coverage done in {time.time()-t0:.1f}s, avg_frac={avg_frac:.3f}")

    md_map = {"TW": md}
    for b in [0.0, 1.0]:
        log(f"beta={b}: tilting weights...")
        wt = tilt_weights(holdings, covers, uids, b)
        log(f"beta={b}: weights tilted, {len(wt)} symbols, calling measure()...")
        t0 = time.time()
        res = measure(md_map, wt, as_of, end)
        log(f"beta={b}: measure() done in {time.time()-t0:.1f}s, "
           f"ret={res['portfolio_realized_return']:.4f}")

    log("all done")


if __name__ == "__main__":
    main()
