# -*- coding: utf-8 -*-
"""agents.py 的第一次真實 LLM 呼叫測試（2024Q4，M1-D 首次觸發的那一季）。
🔴 這支腳本會真的打 OpenAI API、燒真實 token 額度——經使用者同意才執行，
不是可以隨便重跑的腳本。
"""
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

    as_of, end = "2024-09-30", "2024-12-31"
    # 🔴 見 _test_facts_lean.py 的說明：狀態層（env/proc/m1d）要用 end 解的
    # 持股／市場資料，不能跟 outcome（回顧）共用 as_of 那筆，否則兩邊快照
    # 時間點對不上（Agent-B 第一次真實質疑就抓到這個問題）。
    weights_end = resolve_weights(md, idx, uids, end)
    mcap_row_end = monitor.asof_row(mcap_wide, end)

    env = monitor.environment_layer(mcap_wide, end)
    proc = monitor.process_layer(weights_end, mcap_row_end, prev_weights=None,
                                 member_uids=uids, candidate_idx=idx)

    cond = triggers.register_m1d(mcap_wide, "2023-12-31")
    m1d = triggers.evaluate_quarter(cond, mcap_wide, end, prev_state="NONE")

    prospective_facts = facts_lean.build_prospective_facts(env, proc, m1d)
    print("\n=== 預測 facts（即將送進 LLM）===")
    for k, v in prospective_facts.items():
        print(f"--- {k} ---\n{v}")

    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")   # 借用已設定好的 purpose（見 memo.py 的既有慣例）
    print(f"\n使用模型：{model}")

    print("\n=== 真實呼叫 Agent-A 3b（會燒真實 token）===")
    result = agents.call_agent_a_3b(prospective_facts, model=model, api_key=api_key,
                                    purpose="app_memo", dry_run=False)

    print(f"\ndry_run={result['dry_run']}")
    print(f"leakage_check={result['leakage_check']}")
    print(f"usage={result.get('usage')}")
    print("\n=== LLM 輸出 ===")
    for k, v in result["explanation"].items():
        print(f"\n[{k}]\n{v}")


if __name__ == "__main__":
    main()
