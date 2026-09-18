# -*- coding: utf-8 -*-
"""Agent-A 階段7（季度總結）真實呼叫測試，用今天已經拿到的真實 3a/3b/決策輸出。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import agents  # noqa: E402
from utils.config import Config  # noqa: E402


def main():
    retrospective_output = {
        "outcome_narrative": "評估區間 as_of 2024-09-30 至 end 2024-12-31，投組已實現報酬為 -0.0484；相對等權基準的超額為 -0.0058；excess_vs_cap_weight 本期資料未提供。",
        "diagnosis_interpretation": "本期機制觸發狀態如下：M4（區域A）未觸發（False），對應動作為A0。M3（區域B）未觸發（False），對應動作為A0。M0（fallback）已觸發（True），對應動作為A5。",
        "memory_consistency_note": "memory 顯示「第一次執行，無上一期資料可比」，本次無上一期資料。",
        "caveat": "excess_vs_cap_weight 欄位本期無資料。診斷資料僅包含 M4（A）、M3（B）與 M0（fallback）。本回顧僅為事後解釋之用途，不構成或暗示任何調整依據。",
    }
    prospective_output = {
        "state_summary": "環境層／過程層目前觀測到：top_n_weight 0.5014；q1_weight 0.9016；n_unique_stocks 396；max_stock_weight 0.0146；median_mktcap_weighted 6602.0；continuity_vs_prev 約 0.2563。",
        "m1d_interpretation": "M1-D 監控的指標為 q1_weight：當前值 0.9016；registration 0.8827；累計偏離（deviation）0.0189；比較分位 p75=0.012、p90=0.0186；當前狀態為 TRIGGERED。",
        "distance_to_threshold": "依 state=TRIGGERED，門檻已被正式跨越；deviation 0.0189 高於 p90 0.0186。",
        "caveat": "本次資料僅提供單一 M1-D 指標（q1_weight）的登記值、偏離與分位。",
    }
    decision_output = {
        "decision": "A5",
        "decision_detail": "升級人工覆核（因 M1-D 觸發且偏離超過歷史 p90）",
        "reasoning": "採用 m1d_baseline_action_csv 的基準建議 A5。依 prospective_assessment：state=TRIGGERED，deviation 0.0189 已高於 p90=0.0186。不採 A2，因其僅切換 allocation 並不解決規模曝險集中問題。",
    }

    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")

    print(f"使用模型：{model}")
    print("\n=== 真實呼叫 Agent-A 7（季度總結，會燒真實 token）===")
    result = agents.call_agent_a_summary(retrospective_output, prospective_output,
                                         decision_output, model=model, api_key=api_key,
                                         purpose="app_memo", dry_run=False)

    print(f"\ndry_run={result['dry_run']}")
    print(f"leakage_check={result['leakage_check']}")
    print(f"usage={result.get('usage')}")
    print("\n=== 總結輸出（②③）===")
    print(json.dumps(result["explanation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
