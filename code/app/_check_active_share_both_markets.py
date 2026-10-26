# -*- coding: utf-8 -*-
"""2026-10-27：Active Share（Cremers & Petajisto, 2009）市場參數化版本——
2026-10-01台股做過一次臨時分析（`文件/完整操作流程.md`§3.2.2），當時是
ad-hoc腳本、沒有固化成可重用函式、美股從未做過。這支腳本①把`active_share()`
補成`weights.py`的正式函式②補上美股版本，跟台股同一套window4/exclude_v1
候選池、control0（等權）vs L2（W2c傾斜後，用m1dfix最終版）兩種持股比較。

`w_bench`（真實市值權重）直接用`monitor.fetch_mcap_wide()`（已套Russell 3000
篩選、SANITY_MAX_CLOSE防線）正規化，不用`report:mktcap`——這是已經在美股
M1-D校準鏈驗證過的乾淨市值資料來源，比2026-10-01台股版用的`report:mktcap`
更穩，兩個市場用同一套資料來源，可以互相比較。

用法：cd code; PYTHONIOENCODING=utf-8 python -m app._check_active_share_both_markets
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from app import monitor, simulate  # noqa: E402
from app import weights as weights_mod  # noqa: E402
from app._check_rolling_production_simulation import build_rolling_inputs  # noqa: E402

APPLAYER = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"

MARKETS = {
    "TW": APPLAYER / "anchored_w4_openSec_boost_exclude_v1.parquet",
    "US": APPLAYER / "anchored_w4_openSec_balanced_exclude_v1.parquet",
}

REGISTRATION_DATE = simulate.REGISTRATION_DATE
QUARTER_ENDS = simulate.QUARTER_ENDS


def bench_weight_row(mcap_wide: pd.DataFrame, as_of: str) -> pd.Series:
    row = monitor.asof_row(mcap_wide, as_of).dropna()
    return row / row.sum()


def run_market(tree_key: str, picks_path: Path) -> pd.DataFrame:
    print(f"\n=== {tree_key}：Active Share（control0等權 vs L2的W2c傾斜後）===")
    inputs = build_rolling_inputs(model="unused", api_key="unused",
                                  picks_path=picks_path, tree_key=tree_key,
                                  scheme="E", window_no=4)
    uids = inputs.uids_by_allocation["equal"]

    rows = []
    dates = [REGISTRATION_DATE] + list(QUARTER_ENDS)
    print(f"{'as_of':>12} {'equal(control0)':>16} {'W2c傾斜後':>12}")
    for as_of in dates:
        w_equal = simulate.resolve_weights(inputs.md, inputs.idx, uids, as_of)
        mcap_row = monitor.asof_row(inputs.mcap_wide, as_of)
        w_w2c = weights_mod.apply_w2c(w_equal, mcap_row)
        bench = bench_weight_row(inputs.mcap_wide, as_of)

        as_equal = weights_mod.active_share(w_equal, bench)
        as_w2c = weights_mod.active_share(w_w2c, bench)
        print(f"{as_of:>12} {as_equal*100:>15.1f}% {as_w2c*100:>11.1f}%")
        rows.append(dict(market=tree_key, as_of=as_of, active_share_equal=as_equal,
                         active_share_w2c=as_w2c))

    df = pd.DataFrame(rows)
    print(f"  平均：等權={df['active_share_equal'].mean()*100:.1f}%"
         f"（範圍{df['active_share_equal'].min()*100:.1f}%~{df['active_share_equal'].max()*100:.1f}%）"
         f"  W2c傾斜後={df['active_share_w2c'].mean()*100:.1f}%"
         f"（範圍{df['active_share_w2c'].min()*100:.1f}%~{df['active_share_w2c'].max()*100:.1f}%）")
    return df


def main():
    all_rows = []
    for tree_key, picks_path in MARKETS.items():
        all_rows.append(run_market(tree_key, picks_path))
    out = pd.concat(all_rows, ignore_index=True)
    out_path = APPLAYER / "active_share_both_markets_2026-10-27.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}")


if __name__ == "__main__":
    main()
