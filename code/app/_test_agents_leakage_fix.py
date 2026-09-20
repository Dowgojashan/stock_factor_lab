# -*- coding: utf-8 -*-
"""驗證 `agents._scan_for_leakage_mechanism_aware()` 的識別碼數字中性化修正。
六個情境都要驗證過才算數（不是隨便挑一個通過就結案）：
  1. 機制代號（M6）沒有巧合命中時，不可誤攔正確的句子
  2. 欄位名稱裡的雜散數字（q1_weight 的「1」）不可被拿來洗白真正捏造的數字
  3. 日期／金額這類多位數的真實資料數值，不可被誤判成識別碼而被吃掉
     （第一版 regex 在這裡踩過真的 bug，這裡是回歸測試）
  4. 括號式清單編號「1) 2) 3)」不可被誤攔，真實小數也不可被誤傷
  5. 句點式清單編號「1. 2. 3.」不可被誤攔，真實小數也不可被誤傷
     （2026-09-18 正式 8 季重跑撞到，情境4 沒涵蓋這個變體，見開發追蹤）
  6. 年份+季度代號連寫「2024Q3」「2024Q4」不可被誤攔，年份本身仍要能
     正確比對到資料（2026-09-19 多時間尺度敘事撞到，且發現更危險的漏網
     方向，見開發追蹤D61）
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


def test_5_period_style_list_markers_not_falsely_flagged():
    """回歸測試：季度總結 agent 用「1. ⋯2. ⋯3. ⋯」句點式清單編號（不是括號式），
    正式 8 季重跑第一次撞到——D2 誤攔「數字 3」（2026-09-18）。同時驗證真實
    小數（3.14、0.5）跟算式（0.012 這種前面接 0. 的數字）不會被這個新分支
    誤傷（負向前瞻 `(?!\\d)` 要求句點後不能立刻接數字，才不會跟小數搞混）。"""
    explanation = {
        "retrospective_section": "本季 M6 觸發，理由有三項：1. 第一點說明。"
                                 "2. 第二點說明。3. 第三點說明，超額約為0.5。",
    }
    prompt = "metric,value\nexcess,0.5\n"
    leak = agents._scan_for_leakage_mechanism_aware(explanation, prompt)
    assert leak == [], f"情境5失敗（句點式清單編號 1./2./3. 不該被誤攔）：{leak}"

    # 真小數不該被這個新分支誤傷
    explanation2 = {"note": "圓周率約3.14，差額為(0.012 - 0.008)。"}
    prompt2 = "metric,value\npi,3.14\ngap,0.012\nother,0.008\n"
    leak2 = agents._scan_for_leakage_mechanism_aware(explanation2, prompt2)
    assert leak2 == [], f"情境5失敗（真小數 3.14/0.012/0.008 不該被誤傷）：{leak2}"
    print("情境5 通過：句點式清單編號沒有被誤攔，真實小數也沒被誤傷")


def test_6_year_quarter_concat_not_falsely_flagged():
    """回歸測試：多時間尺度敘事 agent 把年份跟季度代號直接連寫成「2024Q3」
    「2024Q4」（無分隔符），正式重跑第一次撞到——D2誤攔「數字4」
    （2026-09-19）。更危險的是「2024Q3」的「3」當時完全沒被攔下（被別處
    真實數字3洗白），是D33講的漏網方向真實再現。"""
    explanation = {
        "semiannual_section": "本半年度彙整（含 2024Q3 與 2024Q4 的季度終點）："
                              "截至 2024-09-30 投組報酬 0.0063；截至 2024-12-31 "
                              "投組報酬 -0.0484。",
    }
    prompt = ("quarter_end,portfolio_realized_return\n"
             "2024-09-30,0.0063\n2024-12-31,-0.0484\n")
    leak = agents._scan_for_leakage_mechanism_aware(explanation, prompt)
    assert leak == [], f"情境6失敗（2024Q3/2024Q4連寫不該被誤攔）：{leak}"
    print("情境6 通過：年份+季度代號連寫沒有被誤攔，年份本身仍正確比對")


def test_7_digit_first_quarter_not_falsely_flagged():
    """回歸測試：跟情境6同一次呼叫緊接著撞到的鏡像模式——agent把季度簡寫
    反過來寫成「1Q、2Q、3Q、4Q」（數字在前字母在後），不是「Q1~Q4」
    （2026-09-19，見開發追蹤D61）。"""
    explanation = {
        "annual_section": "報酬則呈現先負（1Q、2Q）、後正（3Q）、再小幅負（4Q）的變化，"
                          "累計報酬為 -0.0013。",
    }
    prompt = "quarter_end,portfolio_realized_return\n2025-12-31,-0.0013\n"
    leak = agents._scan_for_leakage_mechanism_aware(explanation, prompt)
    assert leak == [], f"情境7失敗（1Q/2Q/3Q/4Q不該被誤攔）：{leak}"
    print("情境7 通過：數字在前的季度簡寫沒有被誤攔")


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
    test_5_period_style_list_markers_not_falsely_flagged()
    test_6_year_quarter_concat_not_falsely_flagged()
    test_7_digit_first_quarter_not_falsely_flagged()
    print("\n全部七個情境都通過。")
