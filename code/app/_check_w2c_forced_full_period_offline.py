# -*- coding: utf-8 -*-
"""2026-10-14：不透過LLM決策，直接強制W2c（現行單檔市值傾斜＋30%上限）在
2024-2025整整8季全程執行，看離線真實報酬會長怎樣。

跟台股4.2.4「提早啟動」測試是同一類實驗（強制忽略M1-D的NONE/OBSERVING/
TRIGGERED判斷，每季都套用W2c），差別是這次用現行美股exclude_v1候選池、
純離線計算（不呼叫LLM，不花額度）——跟`_check_w2c_basket_offline.py`
同一套手法，只是這次涵蓋全部8季、不是只有真的觸發的4季，而且是跟control0
（完全不傾斜）對照，不是跟現行agent決策結果對照。

用法：
    cd code
    python -m app._check_w2c_forced_full_period_offline
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import monitor, simulate  # noqa: E402
from app import weights as weights_mod  # noqa: E402
from app._check_rolling_production_simulation import build_rolling_inputs  # noqa: E402

PICKS_PATH = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
             / "anchored_w4_openSec_balanced_exclude_v1.parquet")

REGISTRATION_DATE = simulate.REGISTRATION_DATE
QUARTER_ENDS = simulate.QUARTER_ENDS


def main():
    inputs = build_rolling_inputs(model="unused", api_key="unused",
                                  picks_path=PICKS_PATH, tree_key="US",
                                  scheme="E", window_no=4)
    uids = inputs.uids_by_allocation["equal"]

    print(f"{'quarter':>12} {'control0':>10} {'W2c強制全程':>12} {'W2c-basket K=3強制全程':>22}")

    cum = {"control0": 1.0, "w2c_forced": 1.0, "basket3_forced": 1.0}

    prev_end = REGISTRATION_DATE
    for end in QUARTER_ENDS:
        as_of = prev_end
        w0 = simulate.resolve_weights(inputs.md, inputs.idx, uids, as_of)
        mcap_row = monitor.asof_row(inputs.mcap_wide, as_of)

        w_control0 = w0
        w_w2c = weights_mod.apply_w2c(w0, mcap_row)
        w_basket3 = weights_mod.apply_w2c_basket(w0, mcap_row, basket_k=3)

        r_control0 = monitor.outcome_layer(inputs.md_map, w_control0, as_of, end)["portfolio_realized_return"]
        r_w2c = monitor.outcome_layer(inputs.md_map, w_w2c, as_of, end)["portfolio_realized_return"]
        r_basket3 = monitor.outcome_layer(inputs.md_map, w_basket3, as_of, end)["portfolio_realized_return"]

        cum["control0"] *= (1 + r_control0)
        cum["w2c_forced"] *= (1 + r_w2c)
        cum["basket3_forced"] *= (1 + r_basket3)

        print(f"{end:>12} {r_control0:>+9.2%} {r_w2c:>+11.2%} {r_basket3:>+21.2%}")
        prev_end = end

    print()
    print("累積（8季全程強制，不經LLM/agent決策）：")
    for k, v in cum.items():
        print(f"  {k:>16}: {(v - 1) * 100:+.2f}%")


if __name__ == "__main__":
    main()
