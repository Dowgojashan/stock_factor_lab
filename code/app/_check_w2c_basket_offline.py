# -*- coding: utf-8 -*-
"""2026-10-14：W2c-basket變體的離線（不花LLM錢）真實報酬對照。

背景：美股window4真實8季LLM模擬（30%上限）觀察到W2c（單檔市值傾斜）拖累
表現34~48pp，查實際持股發現傾斜對象常常不是真正的當季領漲大型股（見
`weights.py`的`tilt_weights_basket()`docstring完整說明）。使用者選擇先做
不花錢的準備工作：用已經跑過的真實8季模擬裡「M1-D真的觸發、agent真的選了
W2c」那幾季，拿同一份候選池、同一份市值資料，離線重算三種權重方案的
真實報酬：
  ① control0（純等權，不傾斜）
  ② 現行W2c（單檔市值傾斜＋30%上限，`apply_w2c()`）——跟正式跑出來的
     checkpoint數字對照，當這支腳本本身邏輯正確的驗證
  ③ W2c-basket（籃子市值傾斜＋30%上限，`apply_w2c_basket()`，K=3/5兩組）

只呼叫`resolve_weights()`/`outcome_layer()`這類純數學函式，完全不呼叫
agents，不花任何LLM額度。

用法：
    cd code
    python -m app._check_w2c_basket_offline
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

# 🔴 2026-10-14修正：原版這裡誤把「agent決策是W2c」的第一季(2024-12-31)當成
# 「真的套用傾斜」的季度，但查真實checkpoint的`config_used.w2c_applied`欄位
# 發現2024-12-31那格其實是False（決策當季的權重還沿用上一季設定，流量變數
# 延遲一季才生效）——真正套用傾斜(w2c_applied=True)是從2025-03-31開始。
TRIGGERED_QUARTERS = [
    ("2024-12-31", "2025-03-31"),
    ("2025-03-31", "2025-06-30"),
    ("2025-06-30", "2025-09-30"),
    ("2025-09-30", "2025-12-31"),
]


def main():
    inputs = build_rolling_inputs(model="unused", api_key="unused",
                                  picks_path=PICKS_PATH, tree_key="US",
                                  scheme="E", window_no=4)
    uids = inputs.uids_by_allocation["equal"]

    print(f"{'quarter':>12} {'control0':>10} {'W2c(單檔)':>12} "
         f"{'W2c-basket K=3':>16} {'W2c-basket K=5':>16}")

    cum = {"control0": 1.0, "w2c": 1.0, "basket3": 1.0, "basket5": 1.0}

    for as_of, end in TRIGGERED_QUARTERS:
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

        cum["control0"] *= (1 + r_control0)
        cum["w2c"] *= (1 + r_w2c)
        cum["basket3"] *= (1 + r_basket3)
        cum["basket5"] *= (1 + r_basket5)

        print(f"{end:>12} {r_control0:>+9.2%} {r_w2c:>+11.2%} "
             f"{r_basket3:>+15.2%} {r_basket5:>+15.2%}")

    print()
    print("累積（只含這4個真實觸發季，不是完整8季）：")
    for k, v in cum.items():
        print(f"  {k:>10}: {(v - 1) * 100:+.2f}%")


if __name__ == "__main__":
    main()
