# -*- coding: utf-8 -*-
"""對 _conditional_w2.py 的好結果做證偽檢查（2026-09-18）：
①α=0 有沒有撞上 C2=8% 單一股票上限（§14 item 12 早就提醒過這個風險）
②α=0 實際上把權重集中到哪幾檔股票（是不是台積電）
③各 α 的持股市值涵蓋率（有沒有因為缺市值資料大量丟股票，膨脹集中度）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from database import Database  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402

from app import monitor  # noqa: E402
from app._prelim_a5_w2_prevalidation import resolve_w0, tilt_weights  # noqa: E402
from app._conditional_w2 import get_window4_members  # noqa: E402

CHECK_QUARTERS = ["2024-12-31", "2025-06-30", "2025-12-31", "2026-06-30"]
ALPHAS = [0.75, 0.5, 0.25, 0.0]


def main():
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    md = MarketData("TW")
    uids = get_window4_members()

    db = Database("TW")
    conn = db.create_connection()
    mcap_wide = monitor.fetch_mcap_wide(conn, start="2007-01-01")

    for as_of in CHECK_QUARTERS:
        print(f"\n{'='*60}\nas_of={as_of}")
        w0 = resolve_w0(md, idx, uids, as_of)
        mcap_row = monitor.asof_row(mcap_wide, as_of)
        print(f"  w0 持股數：{len(w0)}")

        for a in ALPHAS:
            wt = tilt_weights(w0, mcap_row, a)
            n_before = len(w0)
            n_after = len(wt)
            coverage = n_after / n_before if n_before else float("nan")
            top5 = sorted(wt.items(), key=lambda x: -x[1])[:5]
            max_w = top5[0][1] if top5 else 0
            breach = "🔴 breach C2=8%" if max_w > 0.08 else ""
            print(f"  α={a:.2f}: 持股數 {n_after}/{n_before}（涵蓋率{coverage:.1%}）"
                 f"  最大單檔權重={max_w:.2%} {breach}")
            print(f"      前5大：{[(s, f'{w:.2%}') for s, w in top5]}")


if __name__ == "__main__":
    main()
