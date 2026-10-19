# -*- coding: utf-8 -*-
"""2026-10-14：台股openSec_boost exclude_v1，window4（2024-2025，8季），
重新校準Hot Segment門檻（用exclude_v1名單自己的歷史分布，見`_recalibrate_
hot_segment_exclude_v1.py`）之後重跑，第一次真正測Coverage Tilt／RepSwap
有沒有機會被agent選到——反查算出來新門檻下，8季裡有6季低於新P25、3季低於
新P10，window4本身就有真實觸發的機會，不需要跳到window2。

跟先前的`_run_exclude_v1_openSec_boost_cap30.py`差別只有`with_hot_segment=
True`（先前的exclude_v1重跑完全沒接Hot Segment，見對話記錄查證）；
用新run_id避免跟舊門檻時代的checkpoint撞名。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import simulate  # noqa: E402
from app._check_rolling_production_simulation import run_rolling_simulation  # noqa: E402
from utils.config import Config  # noqa: E402

RUN_ID_PREFIX = "anchored_w4_openSec_boost_exclude_v1_hotseg_recal"
PICKS_PATH = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
             / "anchored_w4_openSec_boost_exclude_v1.parquet")


def main():
    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")
    print(f"使用模型：{model}")
    print("=== 台股openSec_boost exclude_v1 window4（Hot Segment重新校準後）："
         "control0 + L2，8季真實呼叫 ===")

    run_rolling_simulation(run_id_prefix=RUN_ID_PREFIX, model=model, api_key=api_key,
                           dry_run=False, picks_path=PICKS_PATH,
                           tree_key="TW", scheme="E", window_no=4,
                           with_hot_segment=True, hot_segment_pool="exclude_v1")

    print("\n--- 完成，逐季狀態/決策總覽 ---")
    for arm in simulate.ARMS:
        run_id = f"{RUN_ID_PREFIX}_{arm}"
        checkpoints = simulate.load_checkpoints(run_id)
        print(f"\n[{arm}] checkpoint 數量：{len(checkpoints)}（預期 8）")
        for c in sorted(checkpoints, key=lambda x: x["quarter_end"]):
            o = c["outcome"]
            pr = o.get("portfolio_realized_return")
            pr_str = f"{pr:+.2%}" if pr is not None else "None"
            hs = c.get("hot_segment")
            hs_state = hs["state"] if hs else "None"
            print(f"  {c['quarter_end']}  M1-D={c['m1d']['state']:<10s} "
                 f"HotSeg={hs_state:<10s} 決策={c['decision']['decision']}｜投組報酬={pr_str}")


if __name__ == "__main__":
    main()
