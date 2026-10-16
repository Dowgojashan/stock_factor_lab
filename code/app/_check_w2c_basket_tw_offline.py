# -*- coding: utf-8 -*-
"""2026-10-14：台股版的W2c-basket離線對照（不呼叫LLM，不花額度）。

跟`_check_w2c_basket_offline.py`（美股版）／`_check_w2c_forced_full_period_
offline.py`同一套手法，換成台股openSec_boost exclude_v1候選池，跑兩種對照：
①真實觸發的5季（2024-12-31~2025-12-31，M1-D從2024Q4起持續TRIGGERED，
跟美股的觸發時間點一致，因為M1-D是指數端訊號、跟候選池無關）
②8季全程強制套用（不管M1-D有沒有真的觸發）

用法：
    cd code
    python -m app._check_w2c_basket_tw_offline
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
             / "anchored_w4_openSec_boost_exclude_v1.parquet")

REGISTRATION_DATE = simulate.REGISTRATION_DATE
QUARTER_ENDS = simulate.QUARTER_ENDS
# 🔴 2026-10-14修正：真實套用W2c的季度跟「agent決策是W2c」的季度**差一季**——
# 決策當季(2024-12-31)的權重還是沿用上一季設定（流量變數延遲一季才生效的
# 設計），查真實checkpoint的`config_used.w2c_applied`欄位確認，真正從
# 2025-03-31才開始套用（2024-12-31那格`w2c_applied=False`）。
TRIGGERED_ENDS = {"2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"}


def main():
    inputs = build_rolling_inputs(model="unused", api_key="unused",
                                  picks_path=PICKS_PATH, tree_key="TW",
                                  scheme="E", window_no=4)
    uids = inputs.uids_by_allocation["equal"]

    print(f"{'quarter':>12} {'control0':>10} {'W2c(單檔)':>12} "
         f"{'W2c-basket K=3':>16} {'W2c-basket K=5':>16}  [真實觸發?]")

    cum_real = {"control0": 1.0, "w2c": 1.0, "basket3": 1.0, "basket5": 1.0}
    cum_forced = {"control0": 1.0, "w2c": 1.0, "basket3": 1.0, "basket5": 1.0}

    prev_end = REGISTRATION_DATE
    for end in QUARTER_ENDS:
        as_of = prev_end
        w0 = simulate.resolve_weights(inputs.md, inputs.idx, uids, as_of)
        mcap_row = monitor.asof_row(inputs.mcap_wide, as_of)

        w_control0 = w0
        w_w2c = weights_mod.apply_w2c(w0, mcap_row)
        w_basket3 = weights_mod.apply_w2c_basket(w0, mcap_row, basket_k=3)
        w_basket5 = weights_mod.apply_w2c_basket(w0, mcap_row, basket_k=5)

        r_control0 = monitor.outcome_layer(inputs.md_map, w_control0, as_of, end)["portfolio_realized_return"]
        r_w2c = monitor.outcome_layer(inputs.md_map, w_w2c, as_of, end)["portfolio_realized_return"]
        r_basket3 = monitor.outcome_layer(inputs.md_map, w_basket3, as_of, end)["portfolio_realized_return"]
        r_basket5 = monitor.outcome_layer(inputs.md_map, w_basket5, as_of, end)["portfolio_realized_return"]

        is_real = end in TRIGGERED_ENDS
        tag = "✅真實觸發" if is_real else "（未觸發）"
        print(f"{end:>12} {r_control0:>+9.2%} {r_w2c:>+11.2%} "
             f"{r_basket3:>+15.2%} {r_basket5:>+15.2%}  {tag}")

        for d, r in (("control0", r_control0), ("w2c", r_w2c),
                    ("basket3", r_basket3), ("basket5", r_basket5)):
            cum_forced[d] *= (1 + r)
            if is_real:
                cum_real[d] *= (1 + r)
            else:
                cum_real[d] *= (1 + r_control0)  # 未觸發季只套control0的報酬（跟真實agent行為一致）

        prev_end = end

    print()
    print("累積（只在真實觸發的5季套用機制，其餘3季維持control0——跟agent真實決策同邏輯）：")
    for k, v in cum_real.items():
        print(f"  {k:>10}: {(v - 1) * 100:+.2f}%")
    print()
    print("累積（8季全程強制套用，不管M1-D有沒有真的觸發）：")
    for k, v in cum_forced.items():
        print(f"  {k:>10}: {(v - 1) * 100:+.2f}%")


if __name__ == "__main__":
    main()
