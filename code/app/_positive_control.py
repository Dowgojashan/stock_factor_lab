# -*- coding: utf-8 -*-
"""§11.2 陽性合成對照（2026-09-18/19）：人工構造跨門檻且持續N期的指數集中度
序列，測 M1-D 偵測器會不會響、Agent-A 會不會正確診斷。

設計：M1-D 是純指數端量測（`monitor.environment_layer()` 只吃 `mcap_wide`，
不依賴投組持股，見D21）——這代表可以**只合成 mcap_wide**、其餘（真實策略
持股、真實報酬）沿用window 4的真實資料，兩件事乾淨分離，不用整條管線都
造假。

合成序列（7季，涵蓋狀態機全部轉換規則）：
  Q1: q1_weight=0.850（登記水準）
  Q2: 0.850（持平，dev=0 -> NONE）
  Q3: 0.865（dev=0.015，介於p75~p90 -> 首次進OBSERVING）
  Q4: 0.870（dev=0.020，>p90 -> TRIGGERED）
  Q5: 0.864（dev=0.014，介於p75~p90但prev=TRIGGERED -> 遲滯維持TRIGGERED）
  Q6: 0.858（dev=0.008，<p75 -> 正確解除回NONE）
  Q7: 0.875（dev=0.025，>p90，從NONE直接跳TRIGGERED，不用先過OBSERVING）

合成 mcap_wide 做法：100檔股票，前20檔（rank>=0.8）equal cap=X，其餘80檔
equal cap=1。Q1_weight = 20X/(20X+80)，解 X = 4*w/(1-w) 得到任意目標w。
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

from app import monitor, simulate, triggers  # noqa: E402

REGISTRATION_DATE = "2020-12-31"  # 借用window 3的登記日期當時間軸基準（純巧合方便，跟window3內容無關）
TARGET_TRAJECTORY = {
    "2021-03-31": 0.850,
    "2021-06-30": 0.850,
    "2021-09-30": 0.865,
    "2021-12-31": 0.870,
    "2022-03-31": 0.864,
    "2022-06-30": 0.858,
    "2022-09-30": 0.875,
}
EXPECTED_STATE = {
    "2021-03-31": "NONE",
    "2021-06-30": "NONE",
    "2021-09-30": "OBSERVING",
    "2021-12-31": "TRIGGERED",
    "2022-03-31": "TRIGGERED",   # 遲滯維持
    "2022-06-30": "NONE",        # 解除
    "2022-09-30": "TRIGGERED",   # 重新觸發
}
N_STOCKS = 100
N_TOP = 20  # 前20% (rank>=0.8時100檔剛好是20檔，無並列問題)


def _solve_top_cap(target_w: float) -> float:
    """20X/(20X+80)=target_w 解 X。"""
    return 4.0 * target_w / (1.0 - target_w)


def build_synthetic_mcap_wide() -> pd.DataFrame:
    """一個日期對一個target_w，構造100檔股票的市值面板。"""
    symbols = [f"SYN{i:03d}" for i in range(N_STOCKS)]
    top_symbols = symbols[:N_TOP]
    rest_symbols = symbols[N_TOP:]

    rows = {}
    all_dates = [REGISTRATION_DATE] + list(TARGET_TRAJECTORY.keys())
    for d in all_dates:
        w = TARGET_TRAJECTORY.get(d, 0.850)  # 登記日期本身也用0.850當基準
        top_cap = _solve_top_cap(w)
        row = {s: top_cap for s in top_symbols}
        row.update({s: 1.0 for s in rest_symbols})
        rows[pd.Timestamp(d)] = row
    return pd.DataFrame(rows).T.sort_index()


def verify_trajectory():
    """純Python驗證，不連資料庫、不燒token：確認合成的mcap_wide算出來的
    q1_weight跟目標一致、觸發狀態跟預期一致。"""
    mcap_wide = build_synthetic_mcap_wide()
    cond = triggers.register_m1d(mcap_wide, REGISTRATION_DATE)
    print(f"登記 q1_weight={cond.registration_q1_weight:.4f}（目標0.8500）")

    state = "NONE"
    all_pass = True
    for d in TARGET_TRAJECTORY:
        env = monitor.environment_layer(mcap_wide, d)
        m1d = triggers.evaluate_quarter(cond, mcap_wide, d, state)
        target_w = TARGET_TRAJECTORY[d]
        expected = EXPECTED_STATE[d]
        w_ok = abs(env["q1_weight"] - target_w) < 1e-6
        state_ok = m1d["state"] == expected
        if not (w_ok and state_ok):
            all_pass = False
        print(f"  {d}: q1_weight={env['q1_weight']:.4f}（目標{target_w}，{'OK' if w_ok else '不符'}）"
             f"  dev={m1d['cumulative_deviation']:.4f}  state={m1d['state']}"
             f"（預期{expected}，{'OK' if state_ok else '不符'}）")
        state = m1d["state"]

    print(f"\n{'全部通過' if all_pass else '有不符，需查'}")
    return all_pass


if __name__ == "__main__":
    verify_trajectory()
