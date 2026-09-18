# -*- coding: utf-8 -*-
"""驗證 `agents._scan_for_leakage_mechanism_aware()` 的識別碼數字中性化修正。
三個情境都要驗證過才算數（不是隨便挑一個通過就結案）：
  1. 機制代號（M6）沒有巧合命中時，不可誤攔正確的句子
  2. 欄位名稱裡的雜散數字（q1_weight 的「1」）不可被拿來洗白真正捏造的數字
  3. 日期／金額這類多位數的真實資料數值，不可被誤判成識別碼而被吃掉
     （第一版 regex 在這裡踩過真的 bug，這裡是回歸測試）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import agents  # noqa: E402


def test_1_mechanism_code_not_false_flagged():
    explanation = {
        "caveat": "diagnosis_csv 未包含 M6 的判定，對 M6 本次無法評估。",
    }
    prompt = agents.build_3a_prompt({
        "outcome_csv": "as_of,end,portfolio_realized_return\n2024-09-30,2024-12-31,-0.0484\n",
        "diagnosis_csv": "mechanism,region,triggered,action\nM4,A,False,A0\nM3,B,False,A0\nM0,fallback,True,A5\n",
        "memory_compact": "{}",
    })
    leak = agents._scan_for_leakage_mechanism_aware(explanation, prompt)
    assert leak == [], f"情境1失敗（M6 不該被誤攔）：{leak}"
    print("情境1 通過：M6 機制代號沒有被誤攔")


def test_2_fabricated_digit_still_caught():
    explanation = {"test": "LLM 亂編的假數字約為一（1）成"}
    prompt = "metric,value\ntop_n_weight,0.4679\nq1_weight,0.8926\n"
    leak = agents._scan_for_leakage_mechanism_aware(explanation, prompt)
    assert len(leak) > 0, "情境2失敗（捏造的數字 1 應該被抓到，不該被 q1_weight 的 1 洗白）"
    print(f"情境2 通過：捏造數字被抓到 -> {leak}")


def test_4_list_markers_not_falsely_flagged():
    """回歸測試：決策 agent 用「1) ⋯；2) ⋯；3) ⋯」列點說明理由，這些數字是
    清單編號不是數據主張，不該被 D2 誤判成捏造數字（真實 simulate.py 第一次
    單季測試就抓到這個問題）。同時驗證算式裡的真實小數（如 "0.012 - 0.008"）
    不會被清單編號規則誤傷。"""
    decision_facts = {
        "m1d_baseline_action_csv": "state,baseline_action\nNONE,A0\n",
        "available_actions_csv": "ratio,allocation\nlegacy,equal\n",
        "prospective_assessment": {
            "m1d_interpretation": "累計偏離 0.008；相對歷史分位數 p75 0.012、p90 0.0186。",
            "distance_to_threshold": "距 p75 尚有 (0.012 - 0.008)，距 p90 尚有 (0.0186 - 0.008)。",
        },
    }
    prompt = agents.build_decision_prompt(decision_facts)
    explanation = {
        "reasoning": "1) 累計偏離 0.008 低於門檻；2) 餘裕為 (0.012 - 0.008) 與 "
                    "(0.0186 - 0.008)；3) 維持現狀。",
    }
    leak = agents._scan_for_leakage_mechanism_aware(explanation, prompt)
    assert leak == [], f"情境4失敗（清單編號 1)/2)/3) 不該被誤攔）：{leak}"
    print("情境4 通過：清單編號標記沒有被誤攔，算式裡的真實小數也沒被誤傷")


def test_3_real_dates_not_corrupted():
    """回歸測試：第一版 regex 在這裡把「2024」整個吃掉，因為 JSON 跳脫的
    `\\n` 緊貼在日期前面，被誤判成「n2024」這個假識別碼。"""
    explanation = {
        "outcome_narrative": "在 as_of 2024-09-30 至 end 2024-12-31 的區間內，"
                             "投組實現報酬為 -0.0484；相對等權重基準的超額為 -0.0058。",
    }
    prompt = agents.build_3a_prompt({
        "outcome_csv": "as_of,end,portfolio_realized_return,excess_vs_equal_weight,"
                       "excess_vs_cap_weight\n2024-09-30,2024-12-31,-0.0484,-0.0058,\n",
        "diagnosis_csv": "mechanism,region,triggered,action\nM4,A,False,A0\n",
        "memory_compact": "{}",
    })
    leak = agents._scan_for_leakage_mechanism_aware(explanation, prompt)
    assert leak == [], f"情境3失敗（真實日期/數字不該被吃掉導致誤攔）：{leak}"
    print("情境3 通過：日期等多位數真實數值沒有被誤判成識別碼")


if __name__ == "__main__":
    test_1_mechanism_code_not_false_flagged()
    test_2_fabricated_digit_still_caught()
    test_3_real_dates_not_corrupted()
    test_4_list_markers_not_falsely_flagged()
    print("\n全部四個情境都通過。")
