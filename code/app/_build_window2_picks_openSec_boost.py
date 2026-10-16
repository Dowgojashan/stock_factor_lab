# -*- coding: utf-8 -*-
"""2026-10-14：台股openSec_boost候選池，anchored window2（IS=2007-2017／
OOS=2018-2020）的baseline + exclude_v1代表策略名單。

動機：為了真正測試Coverage Tilt／RepSwap（目前尚無任何一次真實觸發），
往回查Hot Segment校準歷史資料（`_analysis_outputs_applayer/hot_segment_
coverage_test.csv`）發現，TW在2018-06-30跟2020-06-30（COVID崩盤）這兩個
真實歷史季度，覆蓋率都曾經跌破TRIGGERED門檻（0.308）——window2的OOS期
（2018-01~2020-12）剛好完整涵蓋這兩次真實觸發事件，是目前所有walk-forward
窗口裡唯一已知有兩次真實Hot Segment觸發的窗口，比window4（2024-2025，
Hot Segment從未真正TRIGGERED）更適合拿來測這兩個新機制。

完全沿用`_build_exclude_v1_picks_openSec_boost.py`／`build_silhouette_picks()`
同一套方法論，只換成window2自己的IS/OOS期間（不是window4）。這一步純粹是
建樹+選代表策略，不花LLM額度，是2026-10-14使用者要求先做的「不花錢準備
工作」之一（另一半是W2c-basket的離線測試，見`_check_w2c_basket_offline.py`）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402

from research import walkforward_matrix as WF  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _silhouette_picks_lib import build_silhouette_picks  # noqa: E402

RATIO = "legacy"
IS_START, IS_END = "2007-01", "2017-12"
OOS_START, OOS_END = "2018-01", "2020-12"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"


def save_picks(out_path: Path, variant: str, result: dict) -> None:
    rows = []
    for allocation in ("equal", "proportional"):
        rows.append({
            "tree_key": "TW", "scheme": "E", "window_no": 2,
            "k_mode": "silhouette_is", "ratio": RATIO, "allocation": allocation,
            "group": "A_hrp", "variant": variant,
            "is_start": IS_START, "is_end": IS_END, "oos_start": OOS_START, "oos_end": OOS_END,
            "members": result[allocation],
        })
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"寫入 {out_path}（{result['_meta']}）")


def main():
    print(">> 載入資料(openSec_boost池)...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    print("\n### 1. anchored window2／baseline ###")
    base = build_silhouette_picks("TW", IS_START, IS_END, "baseline",
                                  months_long, meta_pool, f_combo_map, idx)
    save_picks(OUT_DIR / "anchored_w2_openSec_boost_baseline.parquet", "baseline", base)

    print("\n### 2. anchored window2／exclude_v1 ###")
    ev1 = build_silhouette_picks("TW", IS_START, IS_END, "exclude_v1",
                                 months_long, meta_pool, f_combo_map, idx)
    save_picks(OUT_DIR / "anchored_w2_openSec_boost_exclude_v1.parquet", "exclude_v1", ev1)


if __name__ == "__main__":
    main()
