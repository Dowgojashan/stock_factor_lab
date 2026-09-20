# -*- coding: utf-8 -*-
"""查證第二次撞到的 D2 誤判（決策階段，「數字 4」，2025-06-30）。跟
_debug_summary_leak2.py 同一套做法：用已成功的 checkpoint 重建狀態，重打
一次真實呼叫，這次印出決策階段的原始文字，不 raise。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import actions, agents, facts_lean, monitor, simulate, triggers  # noqa: E402
from utils.config import Config  # noqa: E402

RUN_ID = "formal_8q_control0_L2_execlayer"
AS_OF, END = "2025-06-30", "2025-09-30"


def main():
    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")

    checkpoints = [c for c in simulate.load_checkpoints(RUN_ID) if c["arm"] == "L2"]
    last = max(checkpoints, key=lambda c: c["quarter_end"])
    assert last["quarter_end"] == AS_OF, f"預期上次做到 {AS_OF}，實際是 {last['quarter_end']}"
    print(f"從 checkpoint 重建狀態，上次做到 {last['quarter_end']}")
    state = last["m1d"]["state"]
    agent_a_memory = last["agent_a_memory_after"]
    config = simulate.ActiveConfig(**last["config_after"])

    import pandas as pd
    from database import Database
    from fcv_core import MarketData
    from resolve_strategy_holdings import CANDIDATE_INDEX_PATH

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
                                    model=model, api_key=api_key,
                                    cap_weight_series=cap_weight_series)

    apply_w2c = config.last_decision_was_w2c and state == "TRIGGERED"
    weights_as_of = simulate.resolve_portfolio_weights(inputs, config, AS_OF, apply_w2c=apply_w2c)
    weights_end = simulate.resolve_portfolio_weights(inputs, config, END, apply_w2c=apply_w2c)
    mcap_row_end = monitor.asof_row(mcap_wide, END)
    member_uids = uids_by_allocation[config.allocation]

    env = monitor.environment_layer(mcap_wide, END)
    proc = monitor.process_layer(weights_end, mcap_row_end, prev_weights=last["weights_end"],
                                 member_uids=member_uids, candidate_idx=idx)
    m1d = triggers.evaluate_quarter(cond, mcap_wide, END, state)

    prospective_facts = facts_lean.build_prospective_facts(env, proc, m1d)

    print("呼叫 3b ...")
    result_3b = agents.call_agent_a_3b(prospective_facts, model=model, api_key=api_key, dry_run=False)

    print("呼叫決策（手動重現，不 raise，直接印出來）...")
    decision_facts = facts_lean.build_decision_facts(
        m1d, actions.list_available_actions(END), result_3b["explanation"],
        w2c_reference=actions.w2c_reference())
    prompt = agents.build_decision_prompt(decision_facts)
    from app.memo import _call_llm
    explanation, usage = _call_llm(
        prompt, model, api_key, purpose="debug_decision_leak3",
        system_prompt=agents._AGENT_A_DECISION_SYSTEM_PROMPT,
        schema=agents._AGENT_A_DECISION_SCHEMA)

    print("\n=== 原始 decision ===")
    print(explanation["decision"])
    print("\n=== 原始 decision_detail ===")
    print(explanation["decision_detail"])
    print("\n=== 原始 reasoning ===")
    print(explanation["reasoning"])

    leakage = agents._scan_for_leakage_mechanism_aware(explanation, prompt)
    print("\n=== D2 掃描結果 ===")
    print(leakage)

    print("\n=== decision_facts（給 LLM 的資料）===")
    import json
    print(json.dumps(decision_facts, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
