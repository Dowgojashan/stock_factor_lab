# -*- coding: utf-8 -*-
"""正式 8 季實驗，第三版（2026-09-22，使用者授權重跑）：control0＋L2 兩臂 ×
8 季（2024-Q1~2025-Q4），真實 LLM 呼叫。

🔴 重跑原因：使用者發現 `_AGENT_A_DECISION_SYSTEM_PROMPT` 組出最終 prompt
的收尾句（`build_decision_prompt()`）漏列了 W2c（只寫「A0／A2／A4／A5」），
雖然 system prompt 完整描述過 W2c、schema 的 enum 也允許選它，但 A/B 對照
實驗證實這句收尾文字對模型決策有真實、一致的因果影響——同一份
decision_facts，用修正前的收尾句 3/3 次都選 A5，用修正後的收尾句 3/3 次
都選 W2c。這代表 `_v2` 那份正式紀錄「agent 連續5季都保守選A5」這個結果，
很可能是這個 prompt bug 造成的假象，不是 agent 真正的判斷傾向。已修正
`build_decision_prompt()`（收尾句補上 W2c）與 `facts_lean.diagnosis_csv()`
（同時查證時抓到的另一個真bug：M7解鎖後這支函式沒同步更新，3a 從沒真正
拿到過 M7 的資料）。

跟 `_run_formal_8q.py`（`_v2`）完全同一套跑法，只換 RUN_ID——舊版保留不
覆蓋，當作「bug 存在時期」的對照紀錄。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import simulate  # noqa: E402
from utils.config import Config  # noqa: E402

RUN_ID = "formal_8q_control0_L2_execlayer_v3"


def main():
    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")

    print(f"使用模型：{model}")
    print("=== 正式實驗第三版：control0 + L2，8 季真實呼叫（使用者已授權重跑）===")
    simulate.run_simulation(RUN_ID, model=model, api_key=api_key, dry_run=False)

    print("\n=== 完成，逐季狀態/決策總覽 ===")
    checkpoints = simulate.load_checkpoints(RUN_ID)
    print(f"checkpoint 數量：{len(checkpoints)}（預期 16 = 2臂 × 8季）")
    for c in sorted(checkpoints, key=lambda x: (x["arm"], x["quarter_end"])):
        print(f"  {c['arm']:<9s} {c['quarter_end']}  狀態={c['m1d']['state']:<10s}"
             f" 決策={c['decision']['decision']}")


if __name__ == "__main__":
    main()
