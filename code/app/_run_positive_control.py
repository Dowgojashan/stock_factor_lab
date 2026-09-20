# -*- coding: utf-8 -*-
"""跑陽性合成對照的完整agent流程（2026-09-19）。M1-D用`_positive_control.py`
合成的指數集中度序列；持股/報酬/process_layer用window 4的真實資料（真實
mcap_wide，跟合成的分開兩份，process_layer的median_mktcap_weighted等
指標不受合成資料影響，只有M1-D/environment_layer吃合成資料）。
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

from app import diagnose, facts_lean, monitor, simulate, triggers  # noqa: E402
from app._positive_control import (REGISTRATION_DATE, TARGET_TRAJECTORY,  # noqa: E402
                                   build_synthetic_mcap_wide)

RUN_ID = "pos_control_synthetic"


def get_window4_members():
    m = pd.read_parquet(Path(__file__).resolve().parent.parent.parent
                        / "_analysis_outputs_robustness" / "walkforward_members.parquet")
    key = dict(tree_key="TW", scheme="E", window_no=4, k_mode="silhouette_is",
              ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    return list(sub.iloc[0]["members"])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    dry_run = not args.real

    from utils.config import Config
    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")
    print(f"模型={model}  dry_run={dry_run}")

    if not dry_run:
        from utils import openai_quota as OQ
        print(f"今日額度用量：{OQ.today_totals()}")

    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    md = MarketData("TW")
    md_map = {"TW": md}
    uids = get_window4_members()

    db = Database("TW")
    conn = db.create_connection()
    real_mcap_wide = monitor.fetch_mcap_wide(conn, start="2007-01-01")  # process_layer用
    synthetic_mcap_wide = build_synthetic_mcap_wide()  # M1-D/environment_layer用
    tx = pd.read_sql("SELECT date, close FROM taiex_tr ORDER BY date", conn)
    tx["date"] = pd.to_datetime(tx["date"])
    cap_weight_series = tx.set_index("date")["close"].astype(float).sort_index()

    cond = triggers.register_m1d(synthetic_mcap_wide, REGISTRATION_DATE)
    quarters = list(TARGET_TRAJECTORY.keys())

    from app import agents

    for arm in ("control0", "L2"):
        state = "NONE"
        prev_weights = None
        config = simulate.ActiveConfig()
        agent_a_memory: dict = {"recent_quarter_full": None, "earlier_quarters_summary": []}
        excess_history: list[float] = []
        checkpoints = [c for c in simulate.load_checkpoints(RUN_ID) if c["arm"] == arm]
        checkpoints_end = [c["quarter_end"] for c in checkpoints]
        if checkpoints:
            last = max(checkpoints, key=lambda c: quarters.index(c["quarter_end"])
                       if c["quarter_end"] in quarters else -1)
            state = last["m1d"]["state"]
            agent_a_memory = last.get("agent_a_memory_after", agent_a_memory)
            excess_history = last.get("excess_history_after", [])
            config = simulate.ActiveConfig(**last["config_after"])
            print(f"[{arm}] 從 checkpoint 續跑，上次做到 {last['quarter_end']}")

        checkpoints_all = [REGISTRATION_DATE] + quarters
        for i, end in enumerate(quarters):
            if end in checkpoints_end:
                print(f"[{arm}] {end} 已有 checkpoint，跳過")
                continue
            as_of = checkpoints_all[i]
            print(f"[{arm}] 執行 {as_of} -> {end}（合成q1_weight目標={TARGET_TRAJECTORY[end]}）...")

            apply_w2c = config.last_decision_was_w2c and state == "TRIGGERED"
            weights_as_of = simulate.resolve_weights(md, idx, uids, as_of)
            weights_end = simulate.resolve_weights(md, idx, uids, end)
            if apply_w2c:
                from app import weights as weights_mod
                mcap_row_asof = monitor.asof_row(real_mcap_wide, as_of)
                mcap_row_end2 = monitor.asof_row(real_mcap_wide, end)
                weights_as_of = weights_mod.apply_w2c(weights_as_of, mcap_row_asof)
                weights_end = weights_mod.apply_w2c(weights_end, mcap_row_end2)

            mcap_row_end = monitor.asof_row(real_mcap_wide, end)
            env = monitor.environment_layer(synthetic_mcap_wide, end)  # 🔴 合成資料
            proc = monitor.process_layer(weights_end, mcap_row_end, prev_weights=prev_weights,
                                         member_uids=uids, candidate_idx=idx)
            outcome = monitor.outcome_layer(md_map, weights_as_of, as_of, end,
                                            cap_weight_series=cap_weight_series)
            m1d = triggers.evaluate_quarter(cond, synthetic_mcap_wide, end, state)  # 🔴 合成資料

            new_excess_history = excess_history + [outcome["excess_vs_ball"]]
            diag = diagnose.run_diagnosis(n_unique_stocks=proc["n_unique_stocks"],
                                          excess_vs_ball_history=new_excess_history,
                                          weights=weights_end)

            result = {
                "arm": arm, "quarter_end": end, "as_of": as_of,
                "env": env, "proc": proc, "outcome": outcome, "m1d": m1d, "diagnosis": diag,
                "weights_end": weights_end,
                "config_used": {"allocation": config.allocation, "w2c_applied": apply_w2c},
            }

            if arm == "control0":
                result["decision"] = {"decision": "A0", "decision_detail": "",
                                      "reasoning": "對照0臂：定義上完全不調整，不經過agent決策。"}
                result["config_after"] = {"allocation": config.allocation, "last_decision_was_w2c": False}
            else:
                prospective_facts = facts_lean.build_prospective_facts(env, proc, m1d)
                retrospective_facts = facts_lean.build_retrospective_facts(outcome, diag, agent_a_memory)

                result_3b = agents.call_agent_a_3b(prospective_facts, model=model, api_key=api_key, dry_run=dry_run)
                result_3a = agents.call_agent_a_3a(retrospective_facts, model=model, api_key=api_key, dry_run=dry_run)
                result["prospective_output"] = result_3b["explanation"]
                result["retrospective_output"] = result_3a["explanation"]

                from app import actions
                decision_facts = facts_lean.build_decision_facts(
                    m1d, actions.list_available_actions(end), result_3b["explanation"],
                    w2c_reference=actions.w2c_reference())
                result_decision = agents.call_agent_a_decision(decision_facts, model=model, api_key=api_key, dry_run=dry_run)
                result_summary = agents.call_agent_a_summary(
                    result_3a["explanation"], result_3b["explanation"], result_decision["explanation"],
                    model=model, api_key=api_key, dry_run=dry_run)

                decision_code = result_decision["explanation"]["decision"]
                next_allocation = config.allocation
                if decision_code == "A2":
                    next_allocation = "proportional" if config.allocation == "equal" else "equal"
                config_after = simulate.ActiveConfig(allocation=next_allocation,
                                                     last_decision_was_w2c=(decision_code == "W2c"))
                result["config_after"] = __import__("dataclasses").asdict(config_after)
                result["decision"] = result_decision["explanation"]
                result["summary"] = agents.assemble_quarterly_report(
                    program_facts={"env": env, "proc": proc, "outcome": outcome, "m1d": m1d},
                    summary_output=result_summary["explanation"], human_ruling=None)

            state = result["m1d"]["state"]
            prev_weights = result["weights_end"]
            config = simulate.ActiveConfig(**result["config_after"])
            excess_history = excess_history + [result["outcome"]["excess_vs_ball"]]
            result_snapshot = {k: v for k, v in result.items() if k != "weights_end"}
            agent_a_memory = agents.build_agent_a_memory_update(agent_a_memory, result_snapshot)
            result["agent_a_memory_after"] = agent_a_memory
            result["excess_history_after"] = excess_history
            simulate.save_checkpoint(RUN_ID, result)
            print(f"[{arm}] {end} 完成，狀態={state}，決策={result['decision']['decision']}")

    print("\n=== 總覽 ===")
    for c in sorted(simulate.load_checkpoints(RUN_ID), key=lambda x: (x["arm"], x["quarter_end"])):
        print(f"  {c['arm']:<9s} {c['quarter_end']}  狀態={c['m1d']['state']:<10s}"
             f" 決策={c['decision']['decision']}")


if __name__ == "__main__":
    main()
