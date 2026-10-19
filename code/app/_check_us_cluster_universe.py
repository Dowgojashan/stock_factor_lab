# -*- coding: utf-8 -*-
"""2026-10-18：建美股window4的IS樹，取得完整候選池universe清單（美股版的
`tsmc_cluster_concentration.csv`）——給`_recompute_ball_windows123_us.py`／
`_recompute_ball_stock_level_us.py`用，補齊M0/M6/M7的美股歷史校準資料。

跟`_check_tsmc_cluster_concentration.py`（台股版）的差異：拿掉台積電選股
診斷的部分（那是台股特定的研究問題，美股沒有對應的單一龍頭股敘事），
純粹只是建樹＋存universe清單。

純本地運算，不需要資料庫/MarketData（`build_tree_for_window`只吃本地
parquet的報酬矩陣）。美股anchored scheme的IS起點固定在2002-01，window1-4
只是IS_END往後延伸，universe理論上一致（跟台股同一個假設，`walkforward_
members.parquet`已確認美股window1-4都是`is_start=2002-01`）——這裡只建
window4（IS到2023-12，最長），一次覆蓋windows1-4全部需要的universe。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from research import walkforward_matrix as WM  # noqa: E402

WINDOW4_IS_START, WINDOW4_IS_END = "2002-01", "2023-12"
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "us_cluster_universe.csv"


def main():
    print(">> 載入本地報酬矩陣（months_long/meta/f_combo_map）...")
    months_long, meta, f_combo_map = WM._load_inputs()

    print(f">> 重建美股 window4 的樹（IS {WINDOW4_IS_START}~{WINDOW4_IS_END}）"
         f"...（預估約20分鐘，跟其餘research主鏈建樹同一量級）")
    tree = WM.build_tree_for_window("US", WINDOW4_IS_START, WINDOW4_IS_END,
                                    months_long, meta, f_combo_map)

    assign = tree["assign"]
    print(f"\n樹內策略總數：{len(assign)}")

    assign.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {OUT_PATH}")


if __name__ == "__main__":
    main()
