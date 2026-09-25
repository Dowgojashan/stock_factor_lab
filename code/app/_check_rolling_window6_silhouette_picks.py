# -*- coding: utf-8 -*-
"""§4④b第一步：幫rolling(IS6/OOS2) window6（IS 2017-01~2022-12、OOS 2023-01~2025-12，
跟實戰管線8季`QUARTER_ENDS`/`REGISTRATION_DATE=2023-12-31`完全對齊）算出
k_mode="silhouette_is"版本的代表策略名單（equal／proportional兩種allocation都要）。

背景：`simulate.py`目前只讀anchored的`walkforward_members.parquet`（scheme="E"／
window_no=4／**k_mode="silhouette_is"**），要接rolling候選池進實戰管線，第一步要
先補出rolling window6的silhouette版picks——之前§3全部的rolling測試都只用
k_mode="fixed"（§3.3明講是為了限定範圍，避免另外擴大成一個大工程）。

**方法（比照`walkforward_matrix.py::run_wtc()`第487~513行silhouette_is那個分支，
不修改任何共用/凍結檔案）**：
  1. `WF.build_tree_for_window()`建樹（跟之前rolling測試同一步，內部固定用
     k_mode="fixed"的linkage，但回傳的`tree["link"]`可以重複使用去重切）
  2. `research.k_stability.scan_window()`（H-26b既有的單窗診斷函式，完全自包含、
     不依賴凍結的`k_stability.csv`表——那張表沒有rolling窗次的IS區間）只用這個
     IS窗的資料現場選k，不看未來
  3. 用同一棵linkage在選出的k重切（`hrp.cut_clusters(tree["link"], k_is)`），
     不重建樹——這是production主線silhouette_is分支的作法，這裡完全複用同一套邏輯

🔴 安全設計：**輸出寫到`_analysis_outputs_applayer/`，不寫入`walkforward_members.parquet`
或任何`_analysis_outputs_robustness/`下的凍結檔**，不影響anchored那條線既有結果。

🔴 2026-09-24 code review重構：`build_silhouette_picks()`原本是本檔自己獨立的一份
實作，跟後來（§3.14）抽出來的`_silhouette_picks_lib.py`幾乎一模一樣，卻沒有改成
呼叫共用版——兩份實作分開放，之後任何一邊修了邏輯（例如uids要用`tree["assign"]`
這種教訓）很容易只改到一邊，變成divergence風險，正是`_silhouette_picks_lib.py`
自己docstring想避免的事。這裡改成直接呼叫共用版（`variant="baseline"`），刪掉
本檔重複的實作，行為不變（baseline＝不篩選，等同原本的邏輯）。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._check_rolling_window6_silhouette_picks
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
from _design_test_rolling_is6oos2 import (IS_MONTHS, OOS_LEN, RATIO,  # noqa: E402
                                          TREE_KEY, window_dates_rolling)
from _silhouette_picks_lib import build_silhouette_picks  # noqa: E402

WINDOW_NO = 6  # OOS 2023-01~2025-12，跟實戰管線 REGISTRATION_DATE/QUARTER_ENDS 對齊
OUT_PATH = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
           / "rolling_window6_silhouette_members.parquet")


def main():
    blocks = WF._blocks(WF.SCHEME_TOTAL_MONTHS, IS_MONTHS, OOS_LEN)
    off, L = blocks[WINDOW_NO - 1]
    is_start, is_end, oos_start, oos_end = window_dates_rolling(off, L, TREE_KEY)
    print(f"window{WINDOW_NO}（rolling, silhouette_is）：IS {is_start}~{is_end}｜OOS {oos_start}~{oos_end}")

    print(">> 載入資料...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    result = build_silhouette_picks(TREE_KEY, is_start, is_end, "baseline",
                                    months_long, meta_pool, f_combo_map, idx, ratio=RATIO)

    rows = []
    for allocation in ("equal", "proportional"):
        rows.append({
            "tree_key": TREE_KEY, "scheme": "rolling_6_2", "window_no": WINDOW_NO,
            "k_mode": "silhouette_is", "ratio": RATIO, "allocation": allocation, "group": "A_hrp",
            "is_start": is_start, "is_end": is_end, "oos_start": oos_start, "oos_end": oos_end,
            "members": result[allocation],
        })
    out_df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PATH, index=False)
    print(f"\n寫入 {OUT_PATH}（{result['_meta']}）")


if __name__ == "__main__":
    main()
