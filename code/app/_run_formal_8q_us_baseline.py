# -*- coding: utf-8 -*-
"""2026-10-13：美股 openSec_balanced 候選池，anchored window4 baseline，
control0+L2兩臂×8季，真實LLM呼叫。

用途：跟台股同一套方法論，在美股候選池上建立「保留V1」的對照基準，之後跟
`_run_formal_8q_us_exclude_v1.py`（排除V1版本）比較，確認台股發現的「V1是
分水嶺」這個現象是不是在美股也站得住腳——這是論文第四章4.3「美股實戰驗證」
真正需要的數字，不是套用台股的結論。

用`_build_exclude_v1_picks_openSec_balanced.py`產生的baseline picks（已驗證
跟官方凍結的walkforward_members.parquet一致），接進`_check_rolling_
production_simulation.py`（2026-10-13已修正市場參數化bug，見該檔docstring）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import simulate  # noqa: E402
from app._check_rolling_production_simulation import run_rolling_simulation  # noqa: E402
from utils.config import Config  # noqa: E402

RUN_ID_PREFIX = "anchored_w4_openSec_balanced_baseline"
PICKS_PATH = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
             / "anchored_w4_openSec_balanced_baseline.parquet")


def main():
    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")
    print(f"使用模型：{model}")
    print("=== 美股openSec_balanced候選池 baseline：control0 + L2，8季真實呼叫 ===")

    run_rolling_simulation(run_id_prefix=RUN_ID_PREFIX, model=model, api_key=api_key,
                           dry_run=False, picks_path=PICKS_PATH,
                           tree_key="US", scheme="E", window_no=4)

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
