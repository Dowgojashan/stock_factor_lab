# -*- coding: utf-8 -*-
"""2026-10-13（第三輪）：美股openSec_balanced exclude_v1，重跑以套用使用者
剛拍板的W2c單股上限新值（7%→30%，見`weights.py`開頭docstring）。

用新的run_id_prefix（加`_cap30`後綴），避免跟mcapfix那一輪（7%上限，
control0+105.61%/L2+57.57%）撞checkpoint。舊的checkpoint保留不刪。

🔴🔴 2026-10-19補：原版沒接`with_hot_segment`，這次補上，`hot_segment_pool=
"exclude_v1"`對應這支讀的exclude_v1 picks。RUN_ID_PREFIX加`_hotseg`後綴
避免撞已存在的舊checkpoint。
🆕 2026-10-19當天稍晚：美股Hot Segment baseline/exclude_v1已各自獨立校準
完成（`_recalibrate_hot_segment_{baseline,exclude_v1}_us.py`），上面「尚未
個別校準」是這支腳本寫完當下的狀態，`triggers.HOT_SEGMENT_P25["US"]`現在
已經是`{market:{pool:value}}`兩層dict，不再共用舊值——保留這段當歷史context
不是現況。

🔴🔴 2026-10-21第四輪：`_hotseg`checkpoint跟US baseline那組一樣，建立在
已過期的「M1-D全程NONE」假設上（`monitor.fetch_mcap_wide('US')`universe_
clause bug＋2009-2012市值資料本身有誤，兩輪修正後美股window4真實在2025Q3
進入TRIGGERED）。這次用修正後`monitor.py`＋重新校準的`triggers.X_P90["US"]`/
`X_P75["US"]`（0.0197/0.0139）重跑，RUN_ID_PREFIX加`_m1dfix`後綴避免撞
`_hotseg`那批過期checkpoint，對應US baseline已重跑確認+0.43pp改善
（control0+91.09%/L2+91.52%），見`文件/實驗收尾追蹤.md`§一。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import simulate  # noqa: E402
from app._check_rolling_production_simulation import run_rolling_simulation  # noqa: E402
from utils.config import Config  # noqa: E402

RUN_ID_PREFIX = "anchored_w4_openSec_balanced_exclude_v1_cap30_hotseg_m1dfix"
PICKS_PATH = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
             / "anchored_w4_openSec_balanced_exclude_v1.parquet")


def main():
    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")
    print(f"使用模型：{model}")
    print("=== 美股openSec_balanced exclude_v1（W2c上限30%重跑）：control0 + L2，8季真實呼叫 ===")

    run_rolling_simulation(run_id_prefix=RUN_ID_PREFIX, model=model, api_key=api_key,
                           dry_run=False, picks_path=PICKS_PATH,
                           tree_key="US", scheme="E", window_no=4,
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
