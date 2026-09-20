# -*- coding: utf-8 -*-
"""正式 8 季實驗，第二版（2026-09-18，使用者授權重跑）：control0＋L2 兩臂 ×
8 季（2024-Q1~2025-Q4），真實 LLM 呼叫（§11 實驗設計）。L3（人機對話）這次
不跑，另外處理（見開發追蹤）。

🔴 這是重跑，不是第一次跑：第一版（run_id=`formal_8q_control0_L2`，見開發
追蹤 D49）是在「決策只記錄、不執行」的舊架構上跑的——L2 選了什麼動作都不
會真的改變持股，那次的持股結果其實跟 control0 沒有差異。D51 補上動作執行
層（`ActiveConfig`，A2/W2c 現在真的會改變後續持股）後，用**新的 run_id**
重跑，保留舊版 checkpoint 當歷史紀錄，不覆蓋。

前置驗證：D45（Agent-B 移除後架構）、D46（control0 零呼叫／excess_vs_ball
修正）、D48（memory 排除 weights_end 後的成本修正）、D50（W2c 前置驗證）、
D51（動作執行層，dry_run＋權重比對測試通過）皆已驗證。

若中途額度用盡，`memo._call_llm` 會 raise，checkpoint 已落盤的部分不會遺失，
重跑這支腳本會自動從 checkpoint 續跑（§12.4）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import simulate  # noqa: E402
from utils.config import Config  # noqa: E402

RUN_ID = "formal_8q_control0_L2_execlayer_v2"


def main():
    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")

    print(f"使用模型：{model}")
    print("=== 正式實驗：control0 + L2，8 季真實呼叫（使用者已授權）===")
    simulate.run_simulation(RUN_ID, model=model, api_key=api_key, dry_run=False)

    print("\n=== 完成，逐季狀態/決策總覽 ===")
    checkpoints = simulate.load_checkpoints(RUN_ID)
    print(f"checkpoint 數量：{len(checkpoints)}（預期 16 = 2臂 × 8季）")
    for c in sorted(checkpoints, key=lambda x: (x["arm"], x["quarter_end"])):
        print(f"  {c['arm']:<9s} {c['quarter_end']}  狀態={c['m1d']['state']:<10s}"
             f" 決策={c['decision']['decision']}")


if __name__ == "__main__":
    main()
