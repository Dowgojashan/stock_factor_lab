# -*- coding: utf-8 -*-
"""模擬「L3 簡化版」（2026-09-18，使用者要求）：L2 這 8 季裡每次觸發 A5
（升級人工覆核），假設**人（使用者）每次都核准套用 W2c**，重新算出這個
情境下的實際持股與績效，跟真正跑出來的 L2（=control0，因為從沒選過
W2c）比較。

🔴 重要：這不是重新跑一次 agent 決策，是拿**已經真實跑完**的
`formal_8q_control0_L2_execlayer_v2` 的 L2 決策序列（A0/A0/A0/A5/A5/A5/
A5/A5）當基礎，只把「人在 A5 當下核准 W2c」這個假設性的人為介入疊上去，
不燒新的 LLM token，也不影響 L2 自主決策的誠實性（那條記錄原封不動留著，
見 D53）。這是把 A5「升級人工覆核」這個逃生艙真的走一次，不是幫 agent
換答案。

因果時序跟 `simulate.py` 的既有邏輯一致：第 N 季做的決策，決定第 N+1 季
要用什麼設定建構投組（決策當下沒辦法回頭改變已經發生的這一季）。
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
from app.performance import measure  # noqa: E402

RUN_ID = "formal_8q_control0_L2_execlayer_v2"


def main():
    l2_checkpoints = {c["quarter_end"]: c for c in simulate.load_checkpoints(RUN_ID)
                      if c["arm"] == "L2"}
    print("真實 L2 決策序列：")
    for q in simulate.QUARTER_ENDS:
        print(f"  {q}: state={l2_checkpoints[q]['m1d']['state']:<10s}"
             f" decision={l2_checkpoints[q]['decision']['decision']}")

    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    md = MarketData("TW")
    md_map = {"TW": md}
    db = Database("TW")
    conn = db.create_connection()
    mcap_wide = monitor.fetch_mcap_wide(conn, start="2007-01-01")
    tx = pd.read_sql("SELECT date, close FROM taiex_tr ORDER BY date", conn)
    tx["date"] = pd.to_datetime(tx["date"])
    cap_weight_series = tx.set_index("date")["close"].astype(float).sort_index()

    m = pd.read_parquet(Path(__file__).resolve().parent.parent.parent
                        / "_analysis_outputs_robustness" / "walkforward_members.parquet")
    uids_by_allocation = {}
    for alloc in ("equal", "proportional"):
        key = dict(tree_key="TW", scheme="E", window_no=4, k_mode="silhouette_is",
                  ratio="legacy", allocation=alloc, group="A_hrp")
        sub = m.copy()
        for k, v in key.items():
            sub = sub[sub[k] == v]
        uids_by_allocation[alloc] = list(sub.iloc[0]["members"])

    cond = triggers.register_m1d(mcap_wide, simulate.REGISTRATION_DATE)
    inputs = simulate.QuarterInputs(md=md, md_map=md_map, idx=idx, mcap_wide=mcap_wide,
                                    uids_by_allocation=uids_by_allocation, m1d_cond=cond,
                                    model="n/a", api_key="n/a",
                                    cap_weight_series=cap_weight_series)

    checkpoints = [simulate.REGISTRATION_DATE] + simulate.QUARTER_ENDS
    config = simulate.ActiveConfig()  # 期初：equal、無傾斜
    state = "NONE"

    rows = []
    for i, end in enumerate(simulate.QUARTER_ENDS):
        as_of = checkpoints[i]
        apply_w2c = config.last_decision_was_w2c and state == "TRIGGERED"
        weights_as_of = simulate.resolve_portfolio_weights(inputs, config, as_of, apply_w2c=apply_w2c)
        weights_end = simulate.resolve_portfolio_weights(inputs, config, end, apply_w2c=apply_w2c)
        outcome = monitor.outcome_layer(md_map, weights_as_of, as_of, end,
                                        cap_weight_series=cap_weight_series)

        real_decision = l2_checkpoints[end]["decision"]["decision"]
        real_state = l2_checkpoints[end]["m1d"]["state"]
        # 人為介入規則：真實決策若為 A5 且該季 M1-D 為 TRIGGERED，
        # 視為「人核准套用 W2c」，從下一季開始生效
        simulated_decision = "W2c" if (real_decision == "A5" and real_state == "TRIGGERED") else real_decision

        rows.append({"quarter_end": end, "state_used_for_config": state,
                    "apply_w2c_this_quarter": apply_w2c,
                    "real_decision": real_decision,
                    "simulated_decision(人核准後)": simulated_decision,
                    "portfolio_realized_return": outcome["portfolio_realized_return"]})

        # 更新下一季要用的 config／state（用「模擬決策」推進，不是真實決策）
        state = real_state  # M1-D 狀態是客觀量測，不受人的決策影響
        next_allocation = config.allocation
        if simulated_decision == "A2":
            next_allocation = "proportional" if config.allocation == "equal" else "equal"
        config = simulate.ActiveConfig(allocation=next_allocation,
                                       last_decision_was_w2c=(simulated_decision == "W2c"))

    df = pd.DataFrame(rows)
    print("\n=== 逐季（人核准 W2c 情境）===")
    print(df.to_string(index=False))

    cum_simulated = float(pd.Series([1 + r for r in df["portfolio_realized_return"]]).prod() - 1)
    real_l2_rets = [l2_checkpoints[q]["outcome"]["portfolio_realized_return"] for q in simulate.QUARTER_ENDS]
    cum_real_l2 = float(pd.Series([1 + r for r in real_l2_rets]).prod() - 1)

    print(f"\n真實 L2（從未套用 W2c，等於 control0）8季累計報酬：{cum_real_l2:+.2%}")
    print(f"人核准 W2c 情境 8季累計報酬：{cum_simulated:+.2%}")
    print(f"差異：{cum_simulated - cum_real_l2:+.2%}")

    out = (Path(__file__).resolve().parent.parent.parent
          / "_analysis_outputs_applayer" / "l2_human_approved_w2c.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out}")


if __name__ == "__main__":
    main()
