# -*- coding: utf-8 -*-
"""`app/l3_dialogue.py` 的迴歸測試：不燒 token（全程 dry_run），確認：
  1. `list_dialogue_candidates`／`list_run_ids` 對「同目錄下欄位不同的檔案」
     具備相容性（2026-09-19 AppTest headless 檢測到的真 bug：
     `_run_multiscale_narrative.py` 沿用同一個 `simulate_*.jsonl` 命名慣例，
     但記錄沒有 `arm`／`decision`，原本會讓 `list_dialogue_candidates` 直接
     KeyError 崩潰，`list_run_ids` 也會把這種檔案誤列成可選項）
  2. `build_decision_facts_for_quarter` 能從一筆真實 checkpoint 精確重建
     decision_facts（不需要另外存一份）
  3. `run_dialogue_turn` 的 dry-run 路徑回傳完整、schema 正確的假回覆
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import agents, l3_dialogue, simulate  # noqa: E402


def test_1_non_arm_checkpoint_file_does_not_crash():
    """建一個假的「跟 simulate.py 同名慣例但欄位不同」的檔案，確認
    `list_dialogue_candidates` 對它回傳空清單而不是 KeyError，
    `list_run_ids` 也不會把它列進可選項。"""
    fake_run_id = "_test_l3_non_arm_shape"
    p = simulate.checkpoint_path(fake_run_id)
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"quarter_end": "2099-01-01", "narrative": "不是 simulate.py 的格式"},
                           ensure_ascii=False) + "\n")
    try:
        candidates = l3_dialogue.list_dialogue_candidates(fake_run_id)
        assert candidates == [], f"情境1失敗（非本模組格式的檔案應回傳空清單）：{candidates}"
        assert fake_run_id not in l3_dialogue.list_run_ids(), \
            "情境1失敗（非本模組格式的 run_id 不該出現在 list_run_ids()）"
    finally:
        p.unlink(missing_ok=True)
    print("情境1 通過：欄位不同的檔案不會讓 list_dialogue_candidates/list_run_ids 崩潰或誤列")


def test_2_decision_facts_reconstruction_matches_real_checkpoint():
    """對真實跑完的正式實驗（若存在）驗證：從 checkpoint 反推的
    decision_facts 結構正確、可直接餵進 dialogue prompt。若這個 run_id
    在本機還沒跑過（例如全新環境），改用任何一個有 L2 臂資料的 run_id；
    都沒有的話跳過（不是這支測試該負責的事，那是 simulate.py 的範圍）。"""
    candidates = None
    used_run_id = None
    for run_id in l3_dialogue.list_run_ids():
        cands = l3_dialogue.list_dialogue_candidates(run_id)
        if cands:
            candidates, used_run_id = cands, run_id
            break
    if not candidates:
        print("情境2 跳過：本機 `_runs/` 沒有任何可用的 L2 臂 checkpoint。")
        return

    cp = candidates[0]
    facts = l3_dialogue.build_decision_facts_for_quarter(cp)
    expected_keys = {"m1d_baseline_action_csv", "available_actions_csv",
                     "prospective_assessment", "w2c_reference_json"}
    assert set(facts.keys()) == expected_keys, \
        f"情境2失敗（decision_facts 欄位不對）：{set(facts.keys())}"
    print(f"情境2 通過：從 {used_run_id}/{cp['quarter_end']} 重建 decision_facts 欄位正確")


def test_3_dry_run_turn_returns_full_schema():
    """dry-run 路徑不呼叫真實 LLM，但回傳結構要跟真實呼叫一致，UI 才能用
    同一套邏輯處理兩種情況。"""
    facts = {"m1d_baseline_action_csv": "state,baseline_action\nNONE,A0\n"}
    draft = {"decision": "A0", "decision_detail": "", "reasoning": "測試用假決策"}
    result = l3_dialogue.run_dialogue_turn(facts, draft, [], "這一季為什麼不用 W2c？",
                                           dry_run=True)
    assert result["dry_run"] is True
    assert result["leakage_check"] == []
    required = agents._AGENT_A_DIALOGUE_SCHEMA["schema"]["required"]
    assert set(result["explanation"].keys()) == set(required), \
        f"情境3失敗（dry-run 回傳欄位跟 schema 對不上）：{result['explanation'].keys()}"

    # 第二輪：確認 transcript 會被正確接進 prompt（人類與 agent 的內容都要在）
    transcript = [{"round": 1, "human_message": "這一季為什麼不用 W2c？",
                  "agent_response": result["explanation"]}]
    prompt = agents.build_dialogue_prompt(facts, draft, transcript, "那風險呢？")
    assert "這一季為什麼不用 W2c？" in prompt
    assert "那風險呢？" in prompt
    assert "human_input" in prompt
    print("情境3 通過：dry-run 回傳完整 schema，且對話紀錄正確接進下一輪 prompt")


if __name__ == "__main__":
    test_1_non_arm_checkpoint_file_does_not_crash()
    test_2_decision_facts_reconstruction_matches_real_checkpoint()
    test_3_dry_run_turn_returns_full_schema()
    print("\n全部測試通過。")
