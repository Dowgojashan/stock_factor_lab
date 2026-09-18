# -*- coding: utf-8 -*-
"""Agent-A 階段5（決策）真實呼叫測試，用今天已經拿到的 2024Q4 真實資料。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import agents, facts_lean  # noqa: E402
from utils.config import Config  # noqa: E402


def main():
    m1d = {
        "as_of": "2024-12-31", "q1_weight": 0.9016, "registration_q1_weight": 0.8827,
        "cumulative_deviation": 0.0189, "p75": 0.012, "p90": 0.0186,
        "prev_state": "NONE", "state": "TRIGGERED", "action": "A5",
    }
    actions_list = [
        {"ratio": "0.01", "allocation": "equal", "n_windows_visible": 3, "mean_oos_cagr": 0.2163},
        {"ratio": "0.01", "allocation": "proportional", "n_windows_visible": 3, "mean_oos_cagr": 0.2122},
        {"ratio": "0.03", "allocation": "equal", "n_windows_visible": 3, "mean_oos_cagr": 0.2072},
        {"ratio": "0.03", "allocation": "proportional", "n_windows_visible": 3, "mean_oos_cagr": 0.1981},
        {"ratio": "0.05", "allocation": "equal", "n_windows_visible": 3, "mean_oos_cagr": 0.2035},
        {"ratio": "0.05", "allocation": "proportional", "n_windows_visible": 3, "mean_oos_cagr": 0.1863},
        {"ratio": "0.1", "allocation": "equal", "n_windows_visible": 3, "mean_oos_cagr": 0.2009},
        {"ratio": "0.1", "allocation": "proportional", "n_windows_visible": 3, "mean_oos_cagr": 0.1906},
        {"ratio": "legacy", "allocation": "equal", "n_windows_visible": 3, "mean_oos_cagr": 0.2194},
        {"ratio": "legacy", "allocation": "proportional", "n_windows_visible": 3, "mean_oos_cagr": 0.2228},
    ]
    prospective_output = {
        "state_summary": "環境層／過程層目前觀測到：top_n_weight 0.5014；q1_weight 0.9016；n_unique_stocks 396；max_stock_weight 0.0146；median_mktcap_weighted 6602.0；continuity_vs_prev 約 0.2563。",
        "m1d_interpretation": "M1-D 監控的指標為 q1_weight：當前值 0.9016；registration 0.8827；累計偏離（deviation）0.0189；比較分位 p75=0.012、p90=0.0186；當前狀態為 TRIGGERED，偏離已超過歷史 p90 門檻。",
        "distance_to_threshold": "依 state=TRIGGERED，門檻已被正式跨越；deviation 0.0189 高於 p90 0.0186，偏離程度處於歷史高位。",
        "caveat": "本次資料僅提供單一 M1-D 指標（q1_weight）的登記值、偏離與分位，其他狀態變數未附帶門檻或分位資訊。",
    }

    decision_facts = facts_lean.build_decision_facts(m1d, actions_list, prospective_output)
    print("=== 決策 facts ===")
    for k, v in decision_facts.items():
        print(f"--- {k} ---\n{v}")

    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")

    print(f"\n使用模型：{model}")
    print("\n=== 真實呼叫 Agent-A 5（決策，會燒真實 token）===")
    result = agents.call_agent_a_decision(decision_facts, model=model, api_key=api_key,
                                          purpose="app_memo", dry_run=False)

    print(f"\ndry_run={result['dry_run']}")
    print(f"leakage_check={result['leakage_check']}")
    print(f"usage={result.get('usage')}")
    print("\n=== 決策輸出 ===")
    print(json.dumps(result["explanation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
