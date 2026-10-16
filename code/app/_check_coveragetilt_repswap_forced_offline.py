# -*- coding: utf-8 -*-
"""2026-10-14：不透過agent決策，直接強制套用Coverage Tilt／RepSwap（8季全程，
不管Hot Segment有沒有真的觸發），離線（不呼叫LLM）用真實歷史報酬重算對照。

跟W2c forced-full-period離線測試同一個手法：agent在真實8季模擬裡（重新校準
Hot Segment門檻後）雖然真的遇到過Hot Segment單獨觸發的乾淨機會（2024Q1/Q2），
但基於coverage_tilt_reference/rep_swap_reference誠實揭露的「證據薄弱/無
驗證」，理性選擇不用——這支腳本跳過agent的選擇，純粹看數字上這兩個機制
會長怎樣，當成跟agent行為驗證獨立的機制效果驗證。

用法：
    cd code
    python -m app._check_coveragetilt_repswap_forced_offline
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import monitor, simulate, triggers  # noqa: E402
from app import weights as weights_mod  # noqa: E402
from app._check_rolling_production_simulation import build_rolling_inputs  # noqa: E402

PICKS_PATH = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
             / "anchored_w4_openSec_boost_exclude_v1.parquet")

REGISTRATION_DATE = simulate.REGISTRATION_DATE
QUARTER_ENDS = simulate.QUARTER_ENDS


def main():
    inputs = build_rolling_inputs(model="unused", api_key="unused",
                                  picks_path=PICKS_PATH, tree_key="TW",
                                  scheme="E", window_no=4, with_hot_segment=True)
    uids = inputs.uids_by_allocation["equal"]

    print(f"{'quarter':>12} {'control0':>10} {'CoverageTilt強制':>16} "
         f"{'RepSwap強制':>12}  [HotSeg狀態]")

    cum = {"control0": 1.0, "coverage_tilt": 1.0, "rep_swap": 1.0}

    prev_end = REGISTRATION_DATE
    for end in QUARTER_ENDS:
        as_of = prev_end
        monthly = monitor.monthly_close(inputs.md)
        mom = monitor.momentum_asof(monthly, as_of, monitor.HOT_SEGMENT_N["TW"])
        hot = monitor.hot_segment(mom, monitor.HOT_SEGMENT_X)

        hs_layer = monitor.hot_segment_layer(inputs.md, inputs.idx, uids,
                                             monitor.asof_row(inputs.mcap_wide, end), end, market="TW")
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

        cum["control0"] *= (1 + r_control0)
        cum["coverage_tilt"] *= (1 + r_ct)
        cum["rep_swap"] *= (1 + r_rs)
        prev_end = end

    print()
    print("累積（8季全程強制套用，不經agent決策）：")
    for k, v in cum.items():
        print(f"  {k:>16}: {(v - 1) * 100:+.2f}%")


if __name__ == "__main__":
    main()
