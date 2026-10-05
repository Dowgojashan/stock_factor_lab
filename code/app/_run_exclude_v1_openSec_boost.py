# -*- coding: utf-8 -*-
"""2026-10-05：anchored window4 exclude_v1，用這次全新的 openSec_boost 候選池
正式8季實驗，control0+L2兩臂×8季，真實LLM呼叫。

動機：直接對openSec_boost候選池算2024-2025個別策略報酬，發現前10%最佳策略
82.9%是v0、後10%最差策略94.1%是v1——V1在這個全新的池子裡依然是區分好壞
最明顯的維度。用`_build_exclude_v1_picks_openSec_boost.py`算出的exclude_v1
代表名單（已驗證baseline版本跟官方凍結的walkforward_members.parquet完全
一致，方法論可信），接進完整實戰管線，跟同一晚跑的baseline版本
（`_run_formal_8q_v4.py`，control0+11.16%/L2+32.73%）對照。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import simulate  # noqa: E402
from app._check_rolling_production_simulation import run_rolling_simulation  # noqa: E402
from utils.config import Config  # noqa: E402

RUN_ID_PREFIX = "anchored_w4_openSec_boost_exclude_v1"
PICKS_PATH = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
             / "anchored_w4_openSec_boost_exclude_v1.parquet")


def main():
    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")
    print(f"使用模型：{model}")
    print("=== openSec_boost候選池 exclude_v1：control0 + L2，8季真實呼叫 ===")

    run_rolling_simulation(run_id_prefix=RUN_ID_PREFIX, model=model, api_key=api_key,
                           dry_run=False, picks_path=PICKS_PATH,
                           tree_key="TW", scheme="E", window_no=4)

    print("\n--- 完成，逐季狀態/決策總覽 ---")
    for arm in simulate.ARMS:
        run_id = f"{RUN_ID_PREFIX}_{arm}"
        checkpoints = simulate.load_checkpoints(run_id)
        print(f"\n[{arm}] checkpoint 數量：{len(checkpoints)}（預期 8）")
        for c in sorted(checkpoints, key=lambda x: x["quarter_end"]):
            o = c["outcome"]
            pr = o.get("portfolio_realized_return")
            pr_str = f"{pr:+.2%}" if pr is not None else "None"
            print(f"  {c['quarter_end']}  狀態={c['m1d']['state']:<10s}"
                 f" 決策={c['decision']['decision']}｜投組報酬={pr_str}")


if __name__ == "__main__":
    main()
