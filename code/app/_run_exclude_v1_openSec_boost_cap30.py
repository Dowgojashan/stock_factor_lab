# -*- coding: utf-8 -*-
"""2026-10-13：台股 openSec_boost exclude_v1，重跑以套用使用者剛拍板的W2c
單股上限新值（7%→30%，見`weights.py`開頭docstring）。

跟`_run_exclude_v1_openSec_boost.py`（舊版，7%上限，headline數字
control0+14.44%/L2+38.17%）完全同一套跑法，只換RUN_ID_PREFIX避免撞
checkpoint。舊版checkpoint保留不刪，當作7%/30%上限前後對照用。

🔴🔴 2026-10-19補：原版這裡沒接`with_hot_segment`——論文session核對機制
盤點表時連帶發現，正式cap30重跑腳本裡只有TW baseline（`simulate.
run_simulation()`走的那條，Hot Segment/M2是寫死接的）有Hot Segment/M2，
這支（exclude_v1）跟兩支美股cap30腳本都漏了，等於Hot Segment跟M2這次
正式重跑完全不會被觀察到。這次補上，門檻指定`hot_segment_pool=
"exclude_v1"`（這支讀的picks是exclude_v1名單，跟triggers.py新增的
pool維度對應）。RUN_ID_PREFIX也加`_hotseg`後綴——沒加的話resume邏輯會讀到
10/13已經跑完的舊checkpoint（8季全滿、沒有hot_segment/m2欄位），整批被
當成「已完成」跳過，不會真的用新邏輯重跑，是這個專案已經踩過的「checkpoint
撞名導致靜默跳過」同一類風險。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import simulate  # noqa: E402
from app._check_rolling_production_simulation import run_rolling_simulation  # noqa: E402
from utils.config import Config  # noqa: E402

RUN_ID_PREFIX = "anchored_w4_openSec_boost_exclude_v1_cap30_hotseg"
PICKS_PATH = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
             / "anchored_w4_openSec_boost_exclude_v1.parquet")


def main():
    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")
    print(f"使用模型：{model}")
    print("=== 台股openSec_boost exclude_v1（W2c上限30%重跑）：control0 + L2，8季真實呼叫 ===")

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
            print(f"  {c['quarter_end']}  狀態={c['m1d']['state']:<10s}"
                 f" 決策={c['decision']['decision']}｜投組報酬={pr_str}")


if __name__ == "__main__":
    main()
