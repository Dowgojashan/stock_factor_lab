# -*- coding: utf-8 -*-
"""查證 2026-09-18 正式重跑撞到的新 D2 誤判（總結階段，「數字 3」）。
用已成功落盤的前兩季 checkpoint 重建 L2 臂到 2024-09-30 的狀態，重新跑一次
3a/3b/決策/總結，但把 D2 攔截改成印出來而不是 raise，讓我們看到真正的原始
文字內容，抓根因（會再燒一次真實 token，但這是唯一能看到原文的方法）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import actions, agents, facts_lean, monitor, simulate, triggers  # noqa: E402
from utils.config import Config  # noqa: E402

RUN_ID = "formal_8q_control0_L2_execlayer"


def main():
    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")

    checkpoints = [c for c in simulate.load_checkpoints(RUN_ID) if c["arm"] == "L2"]
    last = max(checkpoints, key=lambda c: c["quarter_end"])
    print(f"從 checkpoint 重建狀態，上次做到 {last['quarter_end']}")
    state = last["m1d"]["state"]
    agent_a_memory = last["agent_a_memory_after"]
    excess_history = last["excess_history_after"]
    config = simulate.ActiveConfig(**last["config_after"])

    # 重建這一季需要的 inputs（跟 run_simulation 內部一樣的組法）
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

    as_of, end = "2024-06-30", "2024-09-30"
    apply_w2c = config.last_decision_was_w2c and state == "TRIGGERED"
    weights_as_of = simulate.resolve_portfolio_weights(inputs, config, as_of, apply_w2c=apply_w2c)
    weights_end = simulate.resolve_portfolio_weights(inputs, config, end, apply_w2c=apply_w2c)
    mcap_row_end = monitor.asof_row(mcap_wide, end)
    member_uids = uids_by_allocation[config.allocation]

    env = monitor.environment_layer(mcap_wide, end)
    proc = monitor.process_layer(weights_end, mcap_row_end, prev_weights=last["weights_end"],
                                 member_uids=member_uids, candidate_idx=idx)
    outcome = monitor.outcome_layer(md_map, weights_as_of, as_of, end,
                                    cap_weight_series=cap_weight_series)
    m1d = triggers.evaluate_quarter(cond, mcap_wide, end, state)

    from app import diagnose
    new_excess_history = excess_history + [outcome["excess_vs_ball"]]
    diag = diagnose.run_diagnosis(n_unique_stocks=proc["n_unique_stocks"],
                                  excess_vs_ball_history=new_excess_history)

    prospective_facts = facts_lean.build_prospective_facts(env, proc, m1d)
    retrospective_facts = facts_lean.build_retrospective_facts(outcome, diag, agent_a_memory)

    print("呼叫 3b ...")
    result_3b = agents.call_agent_a_3b(prospective_facts, model=model, api_key=api_key, dry_run=False)
    print("呼叫 3a ...")
    result_3a = agents.call_agent_a_3a(retrospective_facts, model=model, api_key=api_key, dry_run=False)

    print("呼叫決策 ...")
    decision_facts = facts_lean.build_decision_facts(
        m1d, actions.list_available_actions(end), result_3b["explanation"],
        w2c_reference=actions.w2c_reference())
    result_decision = agents.call_agent_a_decision(decision_facts, model=model, api_key=api_key, dry_run=False)
    print("決策=", result_decision["explanation"]["decision"])

    print("呼叫總結（這裡手動重現，不用 agents.call_agent_a_summary，改成印出來不 raise）...")
    prompt = agents.build_summary_prompt(result_3a["explanation"], result_3b["explanation"],
                                         result_decision["explanation"])
    from app.memo import _call_llm
    explanation, usage = _call_llm(
        prompt, model, api_key, purpose="debug_summary_leak",
        system_prompt=agents._AGENT_A_SUMMARY_SYSTEM_PROMPT,
        schema=agents._AGENT_A_SUMMARY_SCHEMA)

    print("\n=== 原始 retrospective_section ===")
    print(explanation["retrospective_section"])
    print("\n=== 原始 prospective_section ===")
    print(explanation["prospective_section"])

    leakage = agents._scan_for_leakage_mechanism_aware(explanation, prompt)
    print("\n=== D2 掃描結果 ===")
    print(leakage)


if __name__ == "__main__":
    main()
