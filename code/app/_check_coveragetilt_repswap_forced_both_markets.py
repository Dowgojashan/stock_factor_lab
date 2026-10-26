# -*- coding: utf-8 -*-
"""2026-10-27：`_check_coveragetilt_repswap_forced_offline.py`的市場參數化版本
——原版只測TW，這支補上US，直接回應2026-10-08當時的決定（`weights.py`
RepSwap區塊註解：「先處理程式，測試等美股重跑完、跟台股一起測」）。

跟原版同一套離線反事實手法：不透過agent決策，8季全程強制套用Coverage Tilt／
RepSwap（不管Hot Segment有沒有真的觸發），用真實歷史報酬重算，不呼叫LLM。

這是RepSwap機制**第一次**有除了TW window4之外的證據點（`實驗收尾追蹤.md`
§四原本記載「RepSwap完全沒有做過效果驗證」），US window4現在也有意義——
2024-2025這段美股M1-D真的觸發過，是跟TW對照最直接的窗口。

用法：cd code; PYTHONIOENCODING=utf-8 python -m app._check_coveragetilt_repswap_forced_both_markets
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from app import monitor, simulate, triggers  # noqa: E402
from app import weights as weights_mod  # noqa: E402
from app._check_rolling_production_simulation import build_rolling_inputs  # noqa: E402

APPLAYER = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"

MARKETS = {
    "TW": APPLAYER / "anchored_w4_openSec_boost_exclude_v1.parquet",
    "US": APPLAYER / "anchored_w4_openSec_balanced_exclude_v1.parquet",
}

REGISTRATION_DATE = simulate.REGISTRATION_DATE
QUARTER_ENDS = simulate.QUARTER_ENDS


def nav_drawdown(rets: list[float]) -> tuple[float, float]:
    nav = [1.0]
    for r in rets:
        nav.append(nav[-1] * (1 + r))
    peak = nav[0]
    mdd = 0.0
    for v in nav:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    return (nav[-1] - 1) * 100, mdd * 100


def run_market(tree_key: str, picks_path: Path) -> pd.DataFrame:
    print(f"\n=== {tree_key}：Coverage Tilt / RepSwap 8季全程強制套用（離線反事實）===")
    inputs = build_rolling_inputs(model="unused", api_key="unused",
                                  picks_path=picks_path, tree_key=tree_key,
                                  scheme="E", window_no=4, with_hot_segment=True,
                                  hot_segment_pool="exclude_v1")
    uids = inputs.uids_by_allocation["equal"]

    print(f"{'quarter':>12} {'control0':>10} {'CoverageTilt強制':>16} "
         f"{'RepSwap強制':>12}  [HotSeg狀態]")

    rets = {"control0": [], "coverage_tilt": [], "rep_swap": []}
    prev_end = REGISTRATION_DATE
    for end in QUARTER_ENDS:
        as_of = prev_end
        monthly = monitor.monthly_close(inputs.md)
        mom = monitor.momentum_asof(monthly, as_of, monitor.HOT_SEGMENT_N[tree_key])
        hot = monitor.hot_segment(mom, monitor.HOT_SEGMENT_X)

        hs_layer = monitor.hot_segment_layer(inputs.md, inputs.idx, uids,
                                             monitor.asof_row(inputs.mcap_wide, end), end,
                                             market=tree_key)
        hs_eval = triggers.evaluate_hot_segment_quarter(
            inputs.hot_segment_cond, hs_layer["coverage_count"], "NONE")

        w_control0 = simulate.resolve_weights(inputs.md, inputs.idx, uids, as_of)
        w_ct = weights_mod.apply_coverage_tilt(inputs.md, inputs.idx, uids, as_of, hot)
        swap_result = weights_mod.rep_swap_members(inputs.md, inputs.idx, uids, as_of, hot,
                                                    inputs.cluster_of, inputs.members_by_cluster)
        w_rs = simulate.resolve_weights(inputs.md, inputs.idx, swap_result["members"], as_of)

        r_control0 = monitor.outcome_layer(inputs.md_map, w_control0, as_of, end)["portfolio_realized_return"]
        r_ct = monitor.outcome_layer(inputs.md_map, w_ct, as_of, end)["portfolio_realized_return"]
        r_rs = monitor.outcome_layer(inputs.md_map, w_rs, as_of, end)["portfolio_realized_return"]

        print(f"{end:>12} {r_control0:>+9.2%} {r_ct:>+15.2%} {r_rs:>+11.2%}  "
             f"[{hs_eval['state']}，swaps={len(swap_result['swaps'])}]")

        rets["control0"].append(r_control0)
        rets["coverage_tilt"].append(r_ct)
        rets["rep_swap"].append(r_rs)
        prev_end = end

    rows = []
    print(f"\n累積（8季全程強制套用，不經agent決策）：")
    for k, rs in rets.items():
        cum, mdd = nav_drawdown(rs)
        print(f"  {k:>16}: 累積{cum:+.2f}%  MDD(季頻取樣){mdd:+.2f}%")
        rows.append(dict(market=tree_key, variant=k, cum_pct=cum, mdd_pct=mdd))
    return pd.DataFrame(rows)


def main():
    all_rows = []
    for tree_key, picks_path in MARKETS.items():
        all_rows.append(run_market(tree_key, picks_path))
    out = pd.concat(all_rows, ignore_index=True)
    out_path = APPLAYER / "coveragetilt_repswap_forced_both_markets_2026-10-27.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}")


if __name__ == "__main__":
    main()
