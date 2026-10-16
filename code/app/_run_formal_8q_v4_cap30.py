# -*- coding: utf-8 -*-
"""2026-10-13：台股 openSec_boost baseline（含V1），重跑以套用使用者剛拍板
的W2c單股上限新值（7%→30%，見`weights.py`開頭docstring）。

跟`_run_formal_8q_v4.py`（舊版，7%上限，headline數字control0+11.16%/
L2+32.73%）完全同一套跑法，只換RUN_ID，避免撞checkpoint被誤判成「已完成」
整批跳過（`run_rolling_simulation`／`run_simulation`都是用run_id查checkpoint
決定要不要跳過）。舊版checkpoint保留不刪，當作7%/30%上限前後對照用。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import simulate  # noqa: E402
from utils.config import Config  # noqa: E402

RUN_ID = "formal_8q_control0_L2_openSec_boost_v4_cap30"


def main():
    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")

    print(f"使用模型：{model}")
    print("=== 台股openSec_boost baseline（W2c上限30%重跑）：control0 + L2，8季真實呼叫 ===")
    simulate.run_simulation(RUN_ID, model=model, api_key=api_key, dry_run=False)

    print("\n=== 完成，逐季狀態/決策總覽 ===")
    checkpoints = simulate.load_checkpoints(RUN_ID)
    print(f"checkpoint 數量：{len(checkpoints)}（預期 16 = 2臂 × 8季）")
    for c in sorted(checkpoints, key=lambda x: (x["arm"], x["quarter_end"])):
        o = c["outcome"]
        pr = o.get("portfolio_realized_return")
        pr_str = f"{pr:+.2%}" if pr is not None else "None"
        print(f"  {c['arm']:<9s} {c['quarter_end']}  狀態={c['m1d']['state']:<10s}"
             f" 決策={c['decision']['decision']}｜投組報酬={pr_str}")


if __name__ == "__main__":
    main()
