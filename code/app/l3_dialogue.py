# -*- coding: utf-8 -*-
"""L3（人機對話）的資料層（設計文件 §10 階段 6，僅 L3 自主等級，≤5 輪）。

`simulate.py` 明確排除 L3 的自動迴圈（階段6需要真人即時對話，無法也不應該
自動化），`agents.py` 也只補了單輪呼叫。這支模組把兩邊接起來，供 `ui.py`
的互動分頁使用：

  - 對話的「階段5決策草案」重用**已經真實跑完**的 L2 決策
    （`_runs/simulate_{run_id}.jsonl`），不重新呼叫 3a/3b/5——那幾個階段已
    經被本季度完整驗證過（開發追蹤 D50~D61），對話只從決策草案開始接續，
    不重跑前面的階段，也不會反過來污染那份真實紀錄。
  - ≤5 輪（§6），每輪都走 D2 掃描（`agents.call_agent_a_dialogue_turn`）。
  - 只有**真實呼叫**（非 dry-run）的輪次落盤到獨立的稽核檔
    `_runs/l3_dialogue_log.jsonl`——跟 `simulate.py` 的 checkpoint、
    `audit.py` 的 `audit_log.jsonl` 都是不同檔案、不同物件模型，不要混用
    同一份，避免以後回頭讀檔時分不清三套紀錄。dry-run 只在 UI 當場顯示，
    不寫檔，避免測試流程把稽核檔灌水。
  - 對話紀錄裡人類訊息一律標 `source: "human_input"`（§5.3「對話側門」），
    跟程式事實分開列，不得混列成同一種來源。
"""
from __future__ import annotations

import datetime
import json

from app import actions, agents, facts_lean, simulate

MAX_ROUNDS = 5
LOG_PATH = simulate.RUNS_DIR / "l3_dialogue_log.jsonl"


def list_dialogue_candidates(run_id: str, *, arm: str = "L2") -> list[dict]:
    """列出可以拿來開對話的（真實）季度 checkpoint，最新一季在前面。L3 的
    決策權跟 L2 相同（§6），所以只從 L2 臂挑——control0 臂定義上不經過 agent
    決策（見 `simulate.run_quarter`），沒有決策草案可對話。

    🔴 2026-09-19（AppTest headless 檢測抓到）：`_runs/` 底下不是每個
    `simulate_*.jsonl` 都是 `simulate.run_simulation()` 產生的——例如
    `_run_multiscale_narrative.py` 沿用同一個 `checkpoint_path()` 命名慣例，
    但每筆記錄的欄位完全不同（`quarter_end`／`narrative`／…，沒有
    `arm`／`decision`）。用 `.get()` 而非 `[...]` 篩選，非本模組格式的檔案
    直接篩成空清單，不因為欄位缺失整支 UI 崩潰。"""
    cps = [c for c in simulate.load_checkpoints(run_id) if c.get("arm") == arm]
    return sorted(cps, key=lambda c: c["quarter_end"], reverse=True)


def list_run_ids() -> list[str]:
    """列出 `_runs/` 底下**真的是** `simulate.run_simulation()` 產生、且有
    L2 臂決策草案可對話的 run_id，依檔案修改時間新到舊排序（見上方
    `list_dialogue_candidates` 的欄位相容性說明——同目錄下混有其他腳本沿用
    同一命名慣例但欄位不同的檔案，這裡先過濾掉，不讓 UI 選到之後才發現
    是空的）。"""
    if not simulate.RUNS_DIR.exists():
        return []
    files = sorted(simulate.RUNS_DIR.glob("simulate_*.jsonl"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    run_ids = [p.stem[len("simulate_"):] for p in files]
    return [r for r in run_ids if list_dialogue_candidates(r)]


def build_decision_facts_for_quarter(checkpoint: dict) -> dict:
    """從一筆真實 checkpoint 反推階段5當時用的 decision_facts——
    `facts_lean.build_decision_facts()` 是純函式，用 checkpoint 裡已存的
    `m1d`／`prospective_output` 加上（不隨 as_of 變化的一次性）
    `actions.w2c_reference()`，即可精確重建，不需要另外存一份。"""
    return facts_lean.build_decision_facts(
        checkpoint["m1d"],
        actions.list_available_actions(checkpoint["quarter_end"]),
        checkpoint["prospective_output"],
        w2c_reference=actions.w2c_reference(),
    )


def new_session_id(run_id: str, quarter_end: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{run_id}__{quarter_end}__{ts}"


def run_dialogue_turn(decision_facts: dict, draft_decision: dict,
                      transcript: list[dict], human_message: str, *,
                      dry_run: bool = True) -> dict:
    """包一層 `Config()` 讀取，跟 `explain.generate()` 同一個慣例——呼叫端
    （`ui.py`）不用自己管 model/api_key，dry-run 時也不需要它們。"""
    model = api_key = None
    if not dry_run:
        from utils.config import Config
        cfg = Config()
        api_key = cfg.get_openai_api_key()
        model = cfg.get_openai_model("app_memo")
    return agents.call_agent_a_dialogue_turn(
        decision_facts, draft_decision, transcript, human_message,
        model=model, api_key=api_key, purpose="monitor_dialogue", dry_run=dry_run)


def append_round(session_id: str, *, run_id: str, quarter_end: str, round_no: int,
                 human_message: str, agent_result: dict) -> dict:
    """落盤一輪對話（append-only，跟 `simulate.save_checkpoint()` 同一個
    慣例：每一筆都是稽核紀錄，不覆蓋）。只在呼叫端確定是真實呼叫時才叫
    這支函式——dry-run 的輪次不落盤（見檔頭說明）。"""
    entry = {
        "session_id": session_id, "run_id": run_id, "quarter_end": quarter_end,
        "round": round_no,
        "human_message": {"text": human_message, "source": "human_input"},
        "agent_response": agent_result["explanation"],
        "leakage_check": agent_result["leakage_check"],
        "recorded_at": datetime.datetime.now().isoformat(),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    return entry
