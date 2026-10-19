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

🔴🔴 2026-10-17重寫：原版把「哪幾季真的觸發」寫死成`TRIGGERED_QUARTERS`
四個區間，依據是修復前（`monitor.fetch_mcap_wide()`未濾離群值）的production
checkpoint的`m1d`狀態。查證發現美股至少12~17檔ticker的`close`欄位本身存錯
（例如ADTX某日收盤價存成$16.18億/股），`close×outstanding_shares`算出來
的市值被嚴重放大，導致M1-D誤判成「2024Q4起連續觸發」。`monitor.py`已加上
`SANITY_MAX_CLOSE`防線修正這批壞資料，用修正後資料重算（見
`_check_us_m1d_sanity_fix.py`）發現**美股原窗口8季全部是NONE，從未真正
觸發過**。

因此這裡改成**不再寫死觸發季清單，而是用修正後的`inputs.mcap_wide`現場
重跑完整8季的M1-D狀態判斷**（跟`triggers.register_m1d()`/`evaluate_quarter()`
同一套邏輯，`simulate.REGISTRATION_DATE`/`QUARTER_ENDS`同一組基準），每一
區間依照「前一個季末的狀態是否TRIGGERED」決定要不要套用傾斜——跟台股
`_check_w2c_basket_tw_offline.py`的「不論觸發與否都算完整8季」口徑一致，
這才是這次使用者要的「補8季版本」。如果修正後資料顯示真的有觸發季，
這支腳本會如實算出傾斜後報酬；如果像目前這樣全部NONE，四組權重方案
在每一季都會相同，這本身就是誠實的結論，不是程式錯誤。

用法：
    cd code
    PYTHONIOENCODING=utf-8 python -m app._check_w2c_basket_offline
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

PICKS_PATH = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
             / "anchored_w4_openSec_balanced_exclude_v1.parquet")
OUT_PATH = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
           / "w2c_basket_us_offline_8q_recheck.csv")


def main():
    inputs = build_rolling_inputs(model="unused", api_key="unused",
                                  picks_path=PICKS_PATH, tree_key="US",
                                  scheme="E", window_no=4)
    uids = inputs.uids_by_allocation["equal"]

    # 🆕 現場重算完整8季的M1-D狀態（用已經濾過離群值的inputs.mcap_wide）——
    # 不依賴任何寫死的觸發季清單。
    # 🔴 2026-10-18：market="US"明確傳入，理由同`_check_us_m1d_sanity_fix.py`。
    cond = triggers.register_m1d(inputs.mcap_wide, simulate.REGISTRATION_DATE, market="US")
    states = {}
    state = "NONE"
    for q in simulate.QUARTER_ENDS:
        r = triggers.evaluate_quarter(cond, inputs.mcap_wide, q, state)
        states[q] = r["state"]
        state = r["state"]

    periods = [(simulate.REGISTRATION_DATE, simulate.QUARTER_ENDS[0])]
    periods += list(zip(simulate.QUARTER_ENDS[:-1], simulate.QUARTER_ENDS[1:]))

    print(f"=== 美股window4完整8季M1-D狀態（修正後，registration="
         f"{simulate.REGISTRATION_DATE}）===")
    for q in simulate.QUARTER_ENDS:
        print(f"  {q}: {states[q]}")
    n_triggered = sum(1 for s in states.values() if s == "TRIGGERED")
    print(f"  8季中{n_triggered}季TRIGGERED\n")

    print(f"{'quarter':>12} {'觸發?':>6} {'control0':>10} {'W2c(單檔)':>12} "
         f"{'W2c-basket K=3':>16} {'W2c-basket K=5':>16}")

    cum = {"control0": 1.0, "w2c": 1.0, "basket3": 1.0, "basket5": 1.0}
    rows = []

    for as_of, end in periods:
        # 區間(as_of, end)要不要套用傾斜，依「as_of這個時點的狀態」決定——
        # 跟原版docstring記錄的「流量變數延遲一季才生效」同一個語意：
        # REGISTRATION_DATE本身視為NONE（期初不可能已經觸發）。
        triggered_this_period = states.get(as_of, "NONE") == "TRIGGERED"

        w0 = simulate.resolve_weights(inputs.md, inputs.idx, uids, as_of)
        mcap_row = monitor.asof_row(inputs.mcap_wide, as_of)

        if triggered_this_period:
            w_control0 = w0
            w_w2c = weights_mod.apply_w2c(w0, mcap_row)
            w_basket3 = weights_mod.apply_w2c_basket(w0, mcap_row, basket_k=3)
            w_basket5 = weights_mod.apply_w2c_basket(w0, mcap_row, basket_k=5)
        else:
            w_control0 = w_w2c = w_basket3 = w_basket5 = w0

        r_control0 = monitor.outcome_layer(inputs.md_map, w_control0, as_of, end)["portfolio_realized_return"]
        r_w2c = monitor.outcome_layer(inputs.md_map, w_w2c, as_of, end)["portfolio_realized_return"]
        r_basket3 = monitor.outcome_layer(inputs.md_map, w_basket3, as_of, end)["portfolio_realized_return"]
        r_basket5 = monitor.outcome_layer(inputs.md_map, w_basket5, as_of, end)["portfolio_realized_return"]

        cum["control0"] *= (1 + r_control0)
        cum["w2c"] *= (1 + r_w2c)
        cum["basket3"] *= (1 + r_basket3)
        cum["basket5"] *= (1 + r_basket5)

        flag = "是" if triggered_this_period else "否"
        print(f"{end:>12} {flag:>6} {r_control0:>+9.2%} {r_w2c:>+11.2%} "
             f"{r_basket3:>+15.2%} {r_basket5:>+15.2%}")
        rows.append(dict(as_of=as_of, end=end, triggered=triggered_this_period,
                         r_control0=r_control0, r_w2c=r_w2c,
                         r_basket3=r_basket3, r_basket5=r_basket5))

    print()
    print("累積（完整8季，不論有無觸發都算在內）：")
    for k, v in cum.items():
        print(f"  {k:>10}: {(v - 1) * 100:+.2f}%")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {OUT_PATH}")


if __name__ == "__main__":
    main()
