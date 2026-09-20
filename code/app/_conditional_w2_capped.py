# -*- coding: utf-8 -*-
"""接續 _conditional_w2_scrutiny 的發現（α=0.25 在 2025-09-30 因台積電單季暴衝
撞到 31.68%，違反 C2=8%）：兩條路一起查，不先假設哪條比較好。

路①：固定 α 的細網格搜尋——在 0.5~0.9 之間找「7 季全部不違反 8% 上限」的
     最激進（最小）α，代表「一個溫和、全程合規的固定傾斜」能做到多好。
路②：cap-and-redistribute——用更激進的 α（0.25／0），但對任何超過 8% 的
     個股強制砍回 8%，多出來的權重按比例分給其他持股。這樣可以在**單一
     違規個股**的季度也保持合規，同時在其他季度享有更強的傾斜——理論上
     應該比路①更好（只在真的需要時才犧牲，不是全面調降）。
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

from app import monitor, triggers  # noqa: E402
from app._prelim_a5_w2_prevalidation import resolve_w0, tilt_weights  # noqa: E402
from app._conditional_w2 import (ALL_QUARTERS, CALIBRATION_QUARTERS,  # noqa: E402
                                 FORWARD_QUARTERS, REGISTRATION_DATE,
                                 get_window4_members)
from app.performance import measure  # noqa: E402

C2_CAP = 0.08


def cap_and_redistribute(weights: dict[str, float], cap: float = C2_CAP,
                         max_iter: int = 50) -> dict[str, float]:
    """反覆把超過 cap 的持股砍到 cap，多出來的權重按比例分給還沒被砍過的
    持股，直到沒有人超過 cap 為止（標準的 iterative capping 演算法，
    避免一次性重分配又把別的股票推過上限）。"""
    w = dict(weights)
    for _ in range(max_iter):
        over = {s: v for s, v in w.items() if v > cap}
        if not over:
            break
        excess = sum(v - cap for v in over.values())
        for s in over:
            w[s] = cap
        under = {s: v for s, v in w.items() if v < cap}
        under_total = sum(under.values())
        if under_total <= 0:
            break
        for s in under:
            w[s] += excess * (under[s] / under_total)
    return w


def main():
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    print(">> 載入 TW MarketData ...")
    md = MarketData("TW")
    md_map = {"TW": md}
    uids = get_window4_members()

    db = Database("TW")
    conn = db.create_connection()
    mcap_wide = monitor.fetch_mcap_wide(conn, start="2007-01-01")
    cond = triggers.register_m1d(mcap_wide, REGISTRATION_DATE)

    checkpoints = [REGISTRATION_DATE] + ALL_QUARTERS
    state = "NONE"
    quarter_info = []
    for i in range(len(checkpoints) - 1):
        as_of, end = checkpoints[i], checkpoints[i + 1]
        m1d = triggers.evaluate_quarter(cond, mcap_wide, end, state)
        state = m1d["state"]
        w0 = resolve_w0(md, idx, uids, as_of)
        mcap_row = monitor.asof_row(mcap_wide, as_of)
        quarter_info.append({"as_of": as_of, "end": end, "state": state,
                             "w0": w0, "mcap_row": mcap_row})

    def realized_return(weights: dict[str, float], as_of: str, end: str) -> float:
        return measure(md_map, weights, as_of, end)["portfolio_realized_return"]

    baseline_rets = {q["end"]: realized_return(tilt_weights(q["w0"], q["mcap_row"], 1.0),
                                               q["as_of"], q["end"]) for q in quarter_info}

    def cum(rets: dict[str, float], quarters: set[str]) -> float:
        vals = [rets[q] for q in ALL_QUARTERS if q in quarters]
        return float(pd.Series([1 + r for r in vals]).prod() - 1)

    calib_base = cum(baseline_rets, CALIBRATION_QUARTERS)
    fwd_base = cum(baseline_rets, FORWARD_QUARTERS)

    print("\n=== 路①：細網格找「7 季全程守住 8% 上限」的最激進固定 α ===")
    for a in [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]:
        max_over_quarters = []
        for q in quarter_info:
            if q["state"] == "NONE":
                continue
            wt = tilt_weights(q["w0"], q["mcap_row"], a)
            m = max(wt.values()) if wt else 0
            max_over_quarters.append((q["end"], m))
        worst = max(max_over_quarters, key=lambda x: x[1])
        breach = "🔴 breach" if worst[1] > C2_CAP else "✅ 合規"
        print(f"  α={a:.2f}：7 季最大單檔權重出現在 {worst[0]}＝{worst[1]:.2%}  {breach}")

    print("\n=== 路②：cap-and-redistribute（α=0.25 與 α=0，強制 8% 上限＋按比例重分配）===")
    for a in [0.25, 0.0]:
        cond_rets = {}
        max_before_cap = {}
        for q in quarter_info:
            if q["state"] == "NONE":
                cond_rets[q["end"]] = baseline_rets[q["end"]]
                continue
            raw = tilt_weights(q["w0"], q["mcap_row"], a)
            max_before_cap[q["end"]] = max(raw.values()) if raw else 0
            capped = cap_and_redistribute(raw)
            cond_rets[q["end"]] = realized_return(capped, q["as_of"], q["end"])
            assert max(capped.values()) <= C2_CAP + 1e-9, "cap 沒生效，程式有 bug"

        calib_cond = cum(cond_rets, CALIBRATION_QUARTERS)
        fwd_cond = cum(cond_rets, FORWARD_QUARTERS)
        print(f"  α={a:.2f}（capped）：校準期改善={calib_cond-calib_base:+.2%}"
             f"  前瞻改善={fwd_cond-fwd_base:+.2%}")
        for e, m in max_before_cap.items():
            if m > C2_CAP:
                print(f"      {e} 原始最大權重 {m:.2%} 被砍回 {C2_CAP:.0%}")


if __name__ == "__main__":
    main()
