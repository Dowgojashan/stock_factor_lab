# -*- coding: utf-8 -*-
"""2026-10-13：用美股全新的 openSec_balanced 候選池（22,856策略，k=4），重算
anchored window4（IS=2002-01~2023-12/OOS=2024-01~2025-12）的 baseline + exclude_v1
兩組代表策略名單。

完全照 `_build_exclude_v1_picks_openSec_boost.py`（台股版）的方法論複製到美股，
唯一差異是 tree_key="US"、IS/OOS 日期改用美股自己的共同窗（2002-01起算，不是
台股的2007-01），其餘（build_silhouette_picks()、驗證邏輯）完全重用同一套函式，
不重新發明。

會先驗證 baseline 版本跟官方凍結的 walkforward_members.parquet（美股這次剛
重建過，見文件/研究進度報告_2026-10-13.md）一致，確保這次用的是同一套可信
方法論，不是臨時拼湊的近似版本。
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

TREE_KEY = "US"
RATIO = "legacy"
IS_START, IS_END = "2002-01", "2023-12"
OOS_START, OOS_END = "2024-01", "2025-12"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
FROZEN_MEMBERS_PATH = (Path(__file__).resolve().parent.parent.parent
                       / "_analysis_outputs_robustness" / "walkforward_members.parquet")


def save_picks(out_path: Path, variant: str, result: dict) -> None:
    rows = []
    for allocation in ("equal", "proportional"):
        rows.append({
            "tree_key": TREE_KEY, "scheme": "E", "window_no": 4,
            "k_mode": "silhouette_is", "ratio": RATIO, "allocation": allocation,
            "group": "A_hrp", "variant": variant,
            "is_start": IS_START, "is_end": IS_END, "oos_start": OOS_START, "oos_end": OOS_END,
            "members": result[allocation],
        })
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"寫入 {out_path}（{result['_meta']}）")


def check_against_frozen(baseline_equal: list[str]) -> None:
    m = pd.read_parquet(FROZEN_MEMBERS_PATH)
    key = dict(tree_key=TREE_KEY, scheme="E", window_no=4, k_mode="silhouette_is",
              ratio=RATIO, allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    assert len(sub) == 1, f"官方凍結名單裡找不到對應列：{key}"
    frozen = set(sub.iloc[0]["members"])
    mine = set(baseline_equal)
    inter = frozen & mine
    print(f"\n=== 驗證：現場重算的baseline跟官方凍結名單(這幾天才重建過的美股新池子)比對 ===")
    print(f"官方凍結：{len(frozen)}檔｜這次現場算：{len(mine)}檔｜交集：{len(inter)}檔")
    if frozen == mine:
        print("✅ 完全一致——方法論驗證通過，exclude_v1版本可信")
    else:
        print(f"⚠️ 不完全一致（只在官方名單：{len(frozen-mine)}檔／只在這次名單：{len(mine-frozen)}檔）"
             f"——需要查原因，不能直接假設是誤差")
        raise AssertionError("baseline跟官方凍結名單不一致，中止，不可帶著未驗證的方法論往下跑")


def main():
    print(">> 載入資料(這幾天重建的美股openSec_balanced池)...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    print("\n### 1. anchored window4／baseline（驗證用）###")
    base = build_silhouette_picks(TREE_KEY, IS_START, IS_END, "baseline",
                                  months_long, meta_pool, f_combo_map, idx)
    check_against_frozen(base["equal"])
    save_picks(OUT_DIR / "anchored_w4_openSec_balanced_baseline.parquet", "baseline", base)

    print("\n### 2. anchored window4／exclude_v1 ###")
    ev1 = build_silhouette_picks(TREE_KEY, IS_START, IS_END, "exclude_v1",
                                 months_long, meta_pool, f_combo_map, idx)
    save_picks(OUT_DIR / "anchored_w4_openSec_balanced_exclude_v1.parquet", "exclude_v1", ev1)


if __name__ == "__main__":
    main()
