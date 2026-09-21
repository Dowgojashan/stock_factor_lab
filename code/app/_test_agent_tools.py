# -*- coding: utf-8 -*-
"""`agent_tools.py`（§8待辦item11步驟1）的回歸測試——不燒token，純本地資料，
用真實formal_8q_control0_L2_execlayer_v2的outcome跟m0_performance_history_
corrected.csv驗證四個工具都能正確運作。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import agent_tools  # noqa: E402


def test_1_get_action_reference_reexport():
    refs = agent_tools.get_action_reference("legacy", "equal", "2024-01-01")
    assert isinstance(refs, list)
    assert all(pd_str < "2024-01" for pd_str in [r["oos_end"] for r in refs]), \
        f"情境1失敗（前視防護沒生效，出現oos_end>=2024-01的窗次）：{refs}"
    print(f"情境1 通過：get_action_reference 正常運作，回傳 {len(refs)} 個窗次，全部早於查詢日期")


def test_2a_get_attribution_old_checkpoint_graceful_none():
    """正式8季checkpoint（`formal_8q_control0_L2_execlayer_v2`）是今天
    （§8待辦item10）修B_all之前存的舊格式，outcome裡沒有
    ball_benchmark_return等新欄位——這裡驗證的是「缺欄位時誠實回傳None，
    不是假裝算出一個數字」，不是bug，是正確的優雅降級行為。"""
    with open("app/_runs/simulate_formal_8q_control0_L2_execlayer_v2.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["arm"] == "control0" and d["quarter_end"] == "2024-03-31":
                outcome = d["outcome"]
                break
        else:
            raise AssertionError("找不到 control0/2024-03-31 的真實 checkpoint")

    attr = agent_tools.get_attribution(outcome)
    assert attr is None, "情境2a失敗（舊格式checkpoint缺新欄位，應該優雅回傳None，不是報錯或瞎猜）"
    print("情境2a 通過：舊格式checkpoint（缺真實B_all欄位）正確優雅降級回傳None，不是崩潰或瞎猜")


def test_2b_get_attribution_new_format():
    """用D77 dry-run驗證過的真實2024Q1數字（見開發追蹤）建構新格式outcome，
    確認欄位齊全時能正確算出三段拆解。"""
    outcome = {
        "portfolio_realized_return": -0.0034,
        "ball_benchmark_return": -0.0084,
        "ball_return_is_real": True,
        "equal_weight_benchmark_return": -0.0009,
        "cap_weight_benchmark_return": 0.1350,
    }
    attr = agent_tools.get_attribution(outcome)
    assert attr is not None, "情境2b失敗（欄位齊全時不該回傳None）"
    assert set(attr) >= {"total_gap", "m4_gap", "m7_gap", "m1r_gap"}
    total = attr["m4_gap"] + attr["m7_gap"] + attr["m1r_gap"]
    assert abs(total - attr["total_gap"]) < 1e-9, \
        f"情境2b失敗（三段貢獻加總應該精確等於total_gap）：{total} vs {attr['total_gap']}"
    print(f"情境2b 通過：get_attribution 用真實2024Q1數字算出 total_gap={attr['total_gap']:+.4f}，"
         f"三段拆解加總精確吻合")


def test_2c_get_attribution_fallback_placeholder_not_mistaken_for_real():
    """🔴🔴 code review 抓到的真bug回歸測試（2026-09-22）：`ball_benchmark_return`
    退回等權大盤替身時，數值會跟 equal_weight_benchmark_return **完全相等**
    （不是None）——若 `get_attribution` 沒有檢查 `ball_return_is_real`，
    會把這個「假的B_all」當真的用，算出 m7_gap 恆為0 的誤導性拆解，而不是
    誠實的「無法拆解」。這裡直接建構「退回替身」的真實情境（兩值相等 +
    ball_return_is_real=False）驗證修正後的正確行為。"""
    outcome = {
        "portfolio_realized_return": -0.05,
        "ball_benchmark_return": -0.05,  # 退回替身時會跟equal_weight完全相等
        "ball_return_is_real": False,
        "equal_weight_benchmark_return": -0.05,
        "cap_weight_benchmark_return": 0.10,
    }
    attr = agent_tools.get_attribution(outcome)
    assert attr is None, ("情境2c失敗（退回替身的假B_all不該被當真值使用，"
                          f"應該回傳None，實際回傳：{attr}）")
    print("情境2c 通過：ball_return_is_real=False 時正確視為無法拆解，"
         "不會把退回的替身值誤當真實B_all算出誤導性的m7_gap=0")


def test_3_get_historical_distribution_all_mechanisms():
    for mech in ("M3", "M0", "M7", "M8"):
        dist = agent_tools.get_historical_distribution(mech)
        assert dist["mechanism"] == mech
        assert dist["n"] > 0, f"情境3失敗（{mech} 歷史樣本數應該>0）"
        print(f"  {mech}: n={dist['n']}，" +
             "，".join(f"{k}={v:.4f}" for k, v in dist.items() if k not in ("mechanism", "n")))
    try:
        agent_tools.get_historical_distribution("M99")
        raise AssertionError("情境3失敗（不支援的機制代號應該要 raise ValueError）")
    except ValueError:
        pass
    print("情境3 通過：四個機制的歷史分位數都能正確查詢，不支援的代號正確拒絕")


def test_4_find_similar_quarters():
    similar = agent_tools.find_similar_quarters(current_excess_vs_ball=-0.05, n=3)
    assert len(similar) == 3
    diffs = [s["abs_diff"] for s in similar]
    assert diffs == sorted(diffs), "情境4失敗（回傳結果應該依相似度排序，最相似在前）"
    for s in similar:
        assert abs(s["excess_vs_ball"] - (-0.05)) - s["abs_diff"] < 1e-9
    print(f"情境4 通過：find_similar_quarters 找到3季最相似歷史（"
         f"最接近的是 window{similar[0]['window_no']}/{similar[0]['as_of']}，"
         f"差距={similar[0]['abs_diff']:.4f}）")


if __name__ == "__main__":
    test_1_get_action_reference_reexport()
    test_2a_get_attribution_old_checkpoint_graceful_none()
    test_2b_get_attribution_new_format()
    test_2c_get_attribution_fallback_placeholder_not_mistaken_for_real()
    test_3_get_historical_distribution_all_mechanisms()
    test_4_find_similar_quarters()
    print("\n全部四個工具測試通過。")
