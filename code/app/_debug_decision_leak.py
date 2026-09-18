# -*- coding: utf-8 -*-
"""除錯：重建 2023-12-31->2024-03-31 這一季（M1-D 應為 NONE）的真實決策
facts，直接呼叫底層 _call_llm，不管有沒有洩漏都印出原始文字。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from database import Database  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH, resolve_holdings  # noqa: E402

from app import actions, agents, facts_lean, monitor, triggers  # noqa: E402
from app.memo import _call_llm  # noqa: E402
from utils.config import Config  # noqa: E402

MEMBERS_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"


def get_members():
    m = pd.read_parquet(MEMBERS_PATH)
    key = dict(tree_key="TW", scheme="E", window_no=4, k_mode="silhouette_is",
              ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    return list(sub.iloc[0]["members"])


def resolve_weights(md, idx, uids, as_of):
    n_strat = len(uids)
    weights = {}
    for uid in uids:
        row = idx.loc[uid]
        try:
            syms, _ = resolve_holdings(md, row, as_of)
        except RuntimeError:
            continue
        if not syms:
            continue
        w = (1.0 / n_strat) / len(syms)
        for s in syms:
            weights[s] = weights.get(s, 0.0) + w
    return weights


def main():
    db = Database("TW")
    conn = db.create_connection()
    mcap_wide = monitor.fetch_mcap_wide(conn, start="2007-01-01")
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    print("載入 TW MarketData…")
    md = MarketData("TW")
    uids = get_members()

    as_of, end = "2023-12-31", "2024-03-31"
    weights_end = resolve_weights(md, idx, uids, end)
    mcap_row_end = monitor.asof_row(mcap_wide, end)
    env = monitor.environment_layer(mcap_wide, end)
    proc = monitor.process_layer(weights_end, mcap_row_end, prev_weights=None,
                                 member_uids=uids, candidate_idx=idx)

    cond = triggers.register_m1d(mcap_wide, "2023-12-31")
    m1d = triggers.evaluate_quarter(cond, mcap_wide, end, prev_state="NONE")
    print(f"M1-D 狀態：{m1d}")

    prospective_facts = facts_lean.build_prospective_facts(env, proc, m1d)
    prompt_3b = agents.build_3b_prompt(prospective_facts)

    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")

    print("\n=== 真實呼叫 3b ===")
    prospective_output, _ = _call_llm(
        prompt_3b, model, api_key, purpose="app_memo", est_tokens=len(prompt_3b) // 3,
        system_prompt=agents._AGENT_A_3B_SYSTEM_PROMPT, schema=agents._AGENT_A_3B_SCHEMA)
    print(prospective_output)

    decision_facts = facts_lean.build_decision_facts(
        m1d, actions.list_available_actions(end), prospective_output)
    print("\n=== 決策 facts ===")
    for k, v in decision_facts.items():
        print(f"--- {k} ---\n{v}")

    prompt_decision = agents.build_decision_prompt(decision_facts)
    print("\n=== 真實呼叫決策（不管有沒有洩漏都印出來）===")
    explanation, usage = _call_llm(
        prompt_decision, model, api_key, purpose="app_memo", est_tokens=len(prompt_decision) // 3,
        system_prompt=agents._AGENT_A_DECISION_SYSTEM_PROMPT, schema=agents._AGENT_A_DECISION_SCHEMA)
    leakage = agents._scan_for_leakage_mechanism_aware(explanation, prompt_decision)
    print(f"\nleakage_check={leakage}")
    print("\n決策輸出：")
    for k, v in explanation.items():
        print(f"[{k}]\n{v}\n")


if __name__ == "__main__":
    main()
