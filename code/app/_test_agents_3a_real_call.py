# -*- coding: utf-8 -*-
"""agents.py 的 3a（回顧診斷）真實 LLM 呼叫測試，用 2024Q4 真實資料。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from database import Database  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH, resolve_holdings  # noqa: E402

from app import agents, diagnose, facts_lean, monitor  # noqa: E402
from app.memo import scan_for_leakage  # noqa: E402
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
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    print("載入 TW MarketData…")
    md = MarketData("TW")
    md_map = {"TW": md}
    uids = get_members()

    as_of, end = "2024-09-30", "2024-12-31"
    weights = resolve_weights(md, idx, uids, as_of)

    outcome = monitor.outcome_layer(md_map, weights, as_of, end)
    proc_n = len(weights)
    excess_hist = [outcome["excess_vs_equal_weight"]]  # 簡化：這裡只示範單期
    diag = diagnose.run_diagnosis(n_unique_stocks=proc_n, excess_vs_ball_history=excess_hist)

    retrospective_facts = facts_lean.build_retrospective_facts(
        outcome, diag, memory={"note": "第一次執行，無上一期資料可比"})
    print("\n=== 回顧 facts（即將送進 LLM）===")
    for k, v in retrospective_facts.items():
        print(f"--- {k} ---\n{v}")

    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")

    print(f"\n使用模型：{model}")
    print("\n=== 真實呼叫 Agent-A 3a（正式路徑 call_agent_a_3a，會燒真實 token）===")
    result = agents.call_agent_a_3a(retrospective_facts, model=model, api_key=api_key,
                                    purpose="app_memo", dry_run=False)

    print(f"\ndry_run={result['dry_run']}")
    print(f"leakage_check={result['leakage_check']}")
    print(f"usage={result.get('usage')}")
    print("\n=== LLM 輸出 ===")
    for k, v in result["explanation"].items():
        print(f"\n[{k}]\n{v}")


if __name__ == "__main__":
    main()
