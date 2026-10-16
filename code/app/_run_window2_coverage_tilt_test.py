# -*- coding: utf-8 -*-
"""2026-10-14：台股openSec_boost候選池，anchored window2（IS=2007-2017／
OOS=2018-2020，12季）exclude_v1，control0+L2兩臂，真實LLM呼叫。

動機：Coverage Tilt／RepSwap這兩個2026-09-29/10-08接線的機制，從未在任何
一次真實8季模擬裡被真正觸發過（window4的Hot Segment只到過OBSERVING，
從未TRIGGERED）。往回查Hot Segment校準歷史資料發現，2018-06-30跟
2020-06-30（COVID崩盤）這兩個真實歷史季度，覆蓋率都曾經跌破TRIGGERED
門檻——window2的OOS期（2018-01~2020-12）剛好完整涵蓋這兩次真實觸發事件，
是目前唯一已知有真實Hot Segment觸發的窗口。

用`_build_window2_picks_openSec_boost.py`已經建好的window2候選名單，接進
`_check_rolling_production_simulation.py`這次新增的`with_hot_segment=True`／
`quarter_ends`／`registration_date`參數（2026-10-14通用化，原本只支援
window4的8季），**這是第一次真的用非window4窗口＋Hot Segment機制跑完整
模擬**。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import simulate  # noqa: E402
from app._check_rolling_production_simulation import run_rolling_simulation  # noqa: E402
from utils.config import Config  # noqa: E402

RUN_ID_PREFIX = "anchored_w2_openSec_boost_exclude_v1_hotseg"
PICKS_PATH = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
             / "anchored_w2_openSec_boost_exclude_v1.parquet")

REGISTRATION_DATE = "2017-12-31"
QUARTER_ENDS = ["2018-03-31", "2018-06-30", "2018-09-30", "2018-12-31",
               "2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31",
               "2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"]


def main():
    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")
    print(f"使用模型：{model}")
    print("=== 台股openSec_boost window2 exclude_v1（Hot Segment真實觸發測試）："
         "control0 + L2，12季真實呼叫 ===")

    run_rolling_simulation(run_id_prefix=RUN_ID_PREFIX, model=model, api_key=api_key,
                           dry_run=False, picks_path=PICKS_PATH,
                           tree_key="TW", scheme="E", window_no=2,
                           quarter_ends=QUARTER_ENDS, registration_date=REGISTRATION_DATE,
                           with_hot_segment=True)

    print("\n--- 完成，逐季狀態/決策總覽 ---")
    for arm in simulate.ARMS:
        run_id = f"{RUN_ID_PREFIX}_{arm}"
        checkpoints = simulate.load_checkpoints(run_id)
        print(f"\n[{arm}] checkpoint 數量：{len(checkpoints)}（預期 12）")
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
