# -*- coding: utf-8 -*-
"""接續`_coverage_tilt_window4_validation.py`：β從0拉到10在window4上是單調改善、
還沒看到plateau或反轉，往上extend測更高的β（20/50/100/200/500），看真正的效果
上限在哪，還是本來就不夠救回根本問題。

沿用同一套邏輯與函式，只換BETAS網格（原本{0,1,2,5,10}是§1.6在window1-3訂的，
這次刻意用更大的網格才能看出window4的β-報酬曲線全貌，不是為了湊好看數字亂試）。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._coverage_tilt_window4_beta_extended
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
from _design_test_hot_segment import _quarterly_returns, hot_segment, monthly_close  # noqa: E402
from _prelim_coverage_tilt_prevalidation import (MEMBERS_PATH, N_MOMENTUM, X_HOT,  # noqa: E402
                                                 get_members, quarter_ends,
                                                 resolve_holdings_and_coverage, tilt_weights)

BETAS_EXTENDED = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "coverage_tilt_window4_beta_extended.csv"
WINDOW4 = {"is_end": "2023-12-31", "oos_start": "2024-01-01", "oos_end": "2025-12-31"}


def main():
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    print("載入 TW MarketData…")
    md = MarketData("TW")
    md_map = {"TW": md}
    members_df = pd.read_parquet(MEMBERS_PATH)

    ret_df = _quarterly_returns(monthly_close(md), N_MOMENTUM)
    uids = get_members(members_df, 4)
    checkpoints = [WINDOW4["is_end"]] + quarter_ends(WINDOW4["oos_start"], WINDOW4["oos_end"])

    # 持股/覆蓋比例每季只算一次（跟β無關），跨全部β網格共用——同一套效率教訓
    # （見`resolve_holdings_and_coverage`docstring），不要對每個β值重解一次持股
    quarter_data = []
    ew_rets = []
    for i in range(len(checkpoints) - 1):
        as_of, end = checkpoints[i], checkpoints[i + 1]
        q_idx = ret_df.index[ret_df.index <= pd.Timestamp(as_of)]
        if len(q_idx) == 0:
            print(f"  {as_of}：動能資料不足，跳過")
            continue
        hot = set(hot_segment(ret_df.loc[q_idx.max()], X_HOT))
        holdings, covers, avg_frac = resolve_holdings_and_coverage(md, idx, uids, as_of, hot)
        quarter_data.append((as_of, end, holdings, covers, avg_frac))
        print(f"  {as_of}：覆蓋比例={avg_frac:.1%}")

    per_beta_rets: dict[float, list[float]] = {b: [] for b in BETAS_EXTENDED}
    for as_of, end, holdings, covers, avg_frac in quarter_data:
        for b in BETAS_EXTENDED:
            wt = tilt_weights(holdings, covers, uids, b)
            res = measure(md_map, wt, as_of, end)
            per_beta_rets[b].append(res["portfolio_realized_return"])
            if b == BETAS_EXTENDED[0]:
                ew = res["by_market_benchmark"]["TW"]["equal_weight_benchmark_return"]
                if ew is None:
                    raise ValueError(f"{as_of}~{end}：TW equal_weight_benchmark_return是None")
                ew_rets.append(ew)

    n_years = len(ew_rets) / 4.0
    cum_ew = pd.Series([1 + r for r in ew_rets]).prod() - 1
    ann_ew = (1 + cum_ew) ** (1 / n_years) - 1

    rows = []
    print("\n=== window4：β延伸網格 ===")
    for b in BETAS_EXTENDED:
        rets = per_beta_rets[b]
        cum = pd.Series([1 + r for r in rets]).prod() - 1
        ann = (1 + cum) ** (1 / n_years) - 1
        print(f"β={b:>6.1f}: 年化={ann:+.2%}  (等權大盤={ann_ew:+.2%}  超額={ann-ann_ew:+.2%})")
        rows.append({"window_no": 4, "beta": b, "ann_return": ann,
                    "ann_ew_benchmark": ann_ew, "ann_excess": ann - ann_ew, "n_rep": len(uids)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {OUT_PATH}")

    best = df.loc[df.ann_excess.idxmax()]
    print(f"\n最好的β={best.beta}（超額{best.ann_excess:+.2%}），"
         f"對照β=0（超額{df.loc[df.beta==0.0,'ann_excess'].iloc[0]:+.2%}）")


if __name__ == "__main__":
    main()
