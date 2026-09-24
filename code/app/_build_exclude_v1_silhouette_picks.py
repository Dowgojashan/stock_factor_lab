# -*- coding: utf-8 -*-
"""補§3.13後續：使用者發現rolling window6正式模擬（§3.12，+20.19%那組）其實
V1比例高達56%（equal allocation），既然§3.8已經證實排除V1是雙贏，這裡把
exclude_v1接進完整實戰管線——anchored、rolling都要跑，才能公平比較。

用`_silhouette_picks_lib.build_silhouette_picks()`算4組：
  1. anchored window4／baseline（保留V1）——**拿來跟官方凍結的
     `walkforward_members.parquet`（scheme=E/window4/silhouette_is/A_hrp）
     對照，驗證整套「現場建樹→現場選k→重切→per-variant quota挑代表」方法論
     算出來的名單是不是真的等於官方凍結流程的結果**（如果一致，代表§3.12
     rolling那條線用同一套方法論算出的silhouette picks可信）
  2. anchored window4／exclude_v1——新的，要接進實戰管線
  3. rolling window6／exclude_v1——新的，要接進實戰管線
     （rolling window6／baseline已經在§3.12算過，不重算）

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._build_exclude_v1_silhouette_picks
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
from _design_test_rolling_is6oos2 import IS_MONTHS, OOS_LEN, TREE_KEY, window_dates_rolling  # noqa: E402
from _silhouette_picks_lib import build_silhouette_picks  # noqa: E402

RATIO = "legacy"
ANCHORED_IS_START, ANCHORED_IS_END = "2007-01", "2023-12"
ANCHORED_OOS_START, ANCHORED_OOS_END = "2024-01", "2025-12"
WINDOW6_NO = 6
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
FROZEN_MEMBERS_PATH = (Path(__file__).resolve().parent.parent.parent
                       / "_analysis_outputs_robustness" / "walkforward_members.parquet")


def save_picks(out_path: Path, tree_key: str, scheme: str, window_no: int,
               is_start: str, is_end: str, oos_start: str, oos_end: str,
               variant: str, result: dict) -> None:
    rows = []
    for allocation in ("equal", "proportional"):
        rows.append({
            "tree_key": tree_key, "scheme": scheme, "window_no": window_no,
            "k_mode": "silhouette_is", "ratio": RATIO, "allocation": allocation,
            "group": "A_hrp", "variant": variant,
            "is_start": is_start, "is_end": is_end, "oos_start": oos_start, "oos_end": oos_end,
            "members": result[allocation],
        })
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"寫入 {out_path}（{result['_meta']}）")


def check_against_frozen(anchored_baseline_equal: list[str]) -> None:
    m = pd.read_parquet(FROZEN_MEMBERS_PATH)
    key = dict(tree_key="TW", scheme="E", window_no=4, k_mode="silhouette_is",
              ratio=RATIO, allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    assert len(sub) == 1
    frozen = set(sub.iloc[0]["members"])
    mine = set(anchored_baseline_equal)
    inter = frozen & mine
    print(f"\n=== 驗證：現場重算的anchored window4 baseline跟官方凍結名單比對 ===")
    print(f"官方凍結：{len(frozen)}檔｜這次現場算：{len(mine)}檔｜交集：{len(inter)}檔")
    if frozen == mine:
        print("✅ 完全一致——silhouette重切方法論驗證通過，rolling那條線用同一套方法可信")
    else:
        print(f"⚠️ 不完全一致（只在官方名單：{len(frozen-mine)}檔／只在這次名單：{len(mine-frozen)}檔）"
             f"——需要查原因，不能直接假設是誤差")


def main():
    print(">> 載入資料...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    print("\n### 1. anchored window4／baseline（驗證用）###")
    anc_base = build_silhouette_picks("TW", ANCHORED_IS_START, ANCHORED_IS_END, "baseline",
                                      months_long, meta_pool, f_combo_map, idx)
    check_against_frozen(anc_base["equal"])

    print("\n### 2. anchored window4／exclude_v1 ###")
    anc_ev1 = build_silhouette_picks("TW", ANCHORED_IS_START, ANCHORED_IS_END, "exclude_v1",
                                     months_long, meta_pool, f_combo_map, idx)
    save_picks(OUT_DIR / "anchored_w4_silhouette_exclude_v1.parquet", "TW", "E", 4,
              ANCHORED_IS_START, ANCHORED_IS_END, ANCHORED_OOS_START, ANCHORED_OOS_END,
              "exclude_v1", anc_ev1)

    print("\n### 3. rolling window6／exclude_v1 ###")
    blocks = WF._blocks(WF.SCHEME_TOTAL_MONTHS, IS_MONTHS, OOS_LEN)
    off, L = blocks[WINDOW6_NO - 1]
    r_is_start, r_is_end, r_oos_start, r_oos_end = window_dates_rolling(off, L, TREE_KEY)
    roll_ev1 = build_silhouette_picks("TW", r_is_start, r_is_end, "exclude_v1",
                                      months_long, meta_pool, f_combo_map, idx)
    save_picks(OUT_DIR / "rolling_w6_silhouette_exclude_v1.parquet", "TW", "rolling_6_2", WINDOW6_NO,
              r_is_start, r_is_end, r_oos_start, r_oos_end, "exclude_v1", roll_ev1)


if __name__ == "__main__":
    main()
