# -*- coding: utf-8 -*-
"""2026-10-13（第三輪）：美股openSec_balanced baseline，重跑以套用使用者剛
拍板的W2c單股上限新值（7%→30%，見`weights.py`開頭docstring）。市值資料修復
（mcapfix）已經生效，不需要再重複那部分的說明，見`_run_formal_8q_us_
baseline_mcapfix.py`docstring。

用新的run_id_prefix（加`_cap30`後綴），避免跟mcapfix那一輪（7%上限，
control0+91.09%/L2+56.97%）撞checkpoint。舊的checkpoint保留不刪，當作
7%/30%上限前後對照用。

🔴🔴 2026-10-19補：原版沒接`with_hot_segment`——論文session核對機制盤點表
時連帶發現，美股目前沒有任何一次正式/測試跑用過Hot Segment（跟TW不同，
TW至少有window2/window4的exclude_v1測試過）。這次補上，`hot_segment_pool=
"baseline"`對應這支讀的baseline picks；美股的Hot Segment門檻目前
baseline/exclude_v1兩個pool暫時共用同一組舊值（尚未個別校準，見
`triggers.HOT_SEGMENT_P25`註解），這個限制要誠實帶進論文。RUN_ID_PREFIX
加`_hotseg`後綴避免撞已存在的舊checkpoint（原本沒有hot_segment/m2欄位）。

🔴🔴 2026-10-21第四輪：`_hotseg`checkpoint建立在已過期的「M1-D全程NONE」
假設上（`monitor.fetch_mcap_wide('US')`當時沒套Russell 3000篩選、且
2009-2012市值資料本身有誤，兩輪修正後美股window4真實在2025Q3進入
TRIGGERED，見`文件/實驗收尾追蹤.md`§七/§八）。這次用修正後的`monitor.py`
＋重新校準的`triggers.X_P90["US"]`/`X_P75["US"]`（0.0197/0.0139）重跑，
RUN_ID_PREFIX加`_m1dfix`後綴避免撞`_hotseg`那批過期checkpoint——舊版保留
不刪，當作「M1-D全程NONE假設下」的歷史對照。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import simulate  # noqa: E402
from app._check_rolling_production_simulation import run_rolling_simulation  # noqa: E402
from utils.config import Config  # noqa: E402

RUN_ID_PREFIX = "anchored_w4_openSec_balanced_baseline_cap30_hotseg_m1dfix"
PICKS_PATH = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
             / "anchored_w4_openSec_balanced_baseline.parquet")


def main():
    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")
    print(f"使用模型：{model}")
    print("=== 美股openSec_balanced baseline（W2c上限30%重跑）：control0 + L2，8季真實呼叫 ===")

    run_rolling_simulation(run_id_prefix=RUN_ID_PREFIX, model=model, api_key=api_key,
                           dry_run=False, picks_path=PICKS_PATH,
                           tree_key="US", scheme="E", window_no=4,
                           with_hot_segment=True, hot_segment_pool="baseline")

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
