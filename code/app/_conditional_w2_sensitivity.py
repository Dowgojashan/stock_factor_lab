# -*- coding: utf-8 -*-
"""兩項敏感度測試（2026-09-18，使用者要求）：

①季中漂移查證：8% 上限只在 as_of（季初）重新平衡那一刻生效，之後到 end
（季底）之前不會再平衡（跟本專案既有方法論一致：checkpoint 間靜態持有，
見 D25）。如果被砍到 8% 的那檔股票在這一季內大漲，季底實際權重可能已經
漂回超過 8%——「全程合規」這個結論可能只在「重新平衡的瞬間」成立，不是
整季都成立。這裡直接算：用季初砍到 8% 的股數（隱含股數），套用季底市值，
反推季底的實際權重。

②重分配規則敏感度：原本用「按比例分給還沒到上限的持股」，這裡換成
「平均分給還沒到上限的持股」，看結果對這個設計選擇敏不敏感。
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
from app._conditional_w2 import (ALL_QUARTERS, CALIBRATION_QUARTERS,  # noqa: E402
                                 FORWARD_QUARTERS, REGISTRATION_DATE,
                                 get_window4_members)
from app._prelim_a5_w2_prevalidation import resolve_w0, tilt_weights  # noqa: E402
from app.performance import measure  # noqa: E402

C2_CAP = 0.08


def cap_and_redistribute_proportional(weights: dict[str, float], cap: float = C2_CAP,
                                      max_iter: int = 50) -> dict[str, float]:
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


def cap_and_redistribute_equal(weights: dict[str, float], cap: float = C2_CAP,
                               max_iter: int = 50) -> dict[str, float]:
    """跟 proportional 版唯一差異：超額權重平均分給還沒到上限的持股，
    不是按現有權重比例分——測試這個設計選擇對結果的敏感度。"""
    w = dict(weights)
    for _ in range(max_iter):
        over = {s: v for s, v in w.items() if v > cap}
        if not over:
            break
        excess = sum(v - cap for v in over.values())
        for s in over:
            w[s] = cap
        under = {s: v for s, v in w.items() if v < cap}
        if not under:
            break
        share = excess / len(under)
        for s in under:
            w[s] = min(w[s] + share, cap)  # 平均分配也可能把別人推過上限，一樣要夾住
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
        mcap_row_asof = monitor.asof_row(mcap_wide, as_of)
        mcap_row_end = monitor.asof_row(mcap_wide, end)
        quarter_info.append({"as_of": as_of, "end": end, "state": state,
                             "w0": w0, "mcap_row_asof": mcap_row_asof,
                             "mcap_row_end": mcap_row_end})

    print("\n=== ①季中漂移查證：被砍到 8% 的股票，季底實際權重漂到多少 ===")
    for q in quarter_info:
        if q["state"] == "NONE":
            continue
        raw = tilt_weights(q["w0"], q["mcap_row_asof"], 0.0)
        capped = cap_and_redistribute_proportional(raw)
        breached = {s: v for s, v in raw.items() if v > C2_CAP}
        if not breached:
            continue
        # 用季初市值換算隱含股數（假設投組總值=1單位），再用季底市值反推權重
        # value(s) ∝ capped_weight(s) / mcap_asof(s) * mcap_end(s)（股數不變，市值變動反映在權重）
        implied_shares = {s: capped[s] / q["mcap_row_asof"].get(s, float("nan"))
                          for s in capped}
        end_values = {s: implied_shares[s] * q["mcap_row_end"].get(s, float("nan"))
                     for s in capped}
        total_end_value = sum(v for v in end_values.values() if pd.notna(v))
        end_weights = {s: v / total_end_value for s, v in end_values.items() if pd.notna(v)}
        for s in breached:
            w_asof, w_end = capped.get(s, float("nan")), end_weights.get(s, float("nan"))
            flag = "🔴 季底又漂回超過8%" if pd.notna(w_end) and w_end > C2_CAP else "✅ 季底仍在8%以下"
            print(f"  {q['end']}｜{s}：季初重平衡後={w_asof:.2%}  季底（未再平衡）={w_end:.2%}  {flag}")

    print("\n=== ②重分配規則敏感度：proportional vs equal-split ===")

    def realized_return(weights: dict[str, float], as_of: str, end: str) -> float:
        return measure(md_map, weights, as_of, end)["portfolio_realized_return"]

    baseline_rets = {q["end"]: realized_return(tilt_weights(q["w0"], q["mcap_row_asof"], 1.0),
                                               q["as_of"], q["end"]) for q in quarter_info}

    def cum(rets: dict[str, float], quarters: set[str]) -> float:
        vals = [rets[q] for q in ALL_QUARTERS if q in quarters]
        return float(pd.Series([1 + r for r in vals]).prod() - 1)

    calib_base, fwd_base = cum(baseline_rets, CALIBRATION_QUARTERS), cum(baseline_rets, FORWARD_QUARTERS)

    for name, fn in [("proportional（原本用的）", cap_and_redistribute_proportional),
                     ("equal-split（敏感度測試）", cap_and_redistribute_equal)]:
        cond_rets = {}
        for q in quarter_info:
            if q["state"] == "NONE":
                cond_rets[q["end"]] = baseline_rets[q["end"]]
            else:
                raw = tilt_weights(q["w0"], q["mcap_row_asof"], 0.0)
                capped = fn(raw)
                assert max(capped.values()) <= C2_CAP + 1e-6
                cond_rets[q["end"]] = realized_return(capped, q["as_of"], q["end"])
        calib_c, fwd_c = cum(cond_rets, CALIBRATION_QUARTERS), cum(cond_rets, FORWARD_QUARTERS)
        print(f"  {name}：校準期改善={calib_c-calib_base:+.2%}  前瞻改善={fwd_c-fwd_base:+.2%}")


if __name__ == "__main__":
    main()
