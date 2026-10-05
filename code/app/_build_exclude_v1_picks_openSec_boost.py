# -*- coding: utf-8 -*-
"""2026-10-04：用這次全新的 openSec_boost 候選池（29,255策略，k=6），重算
anchored window4（IS=2007-2023/OOS=2024-2025）的 baseline + exclude_v1 兩組
代表策略名單。

動機：對這批新池直接計算2024-2025個別策略報酬發現，前10%最佳策略82.9%是
v0、後10%最差策略94.1%是v1——V1（PE估值濾網）在這個池子裡依然是區分好壞
最明顯的維度，F1因子類型（估值/動能/成長/體質）反而差異不大。使用者要求
重跑一次排除v1的版本，看在新池子+新實戰管線下改善幅度。

完全沿用 `_build_exclude_v1_silhouette_picks.py` 的方法論（build_silhouette_
picks()），只限定anchored window4（不跑rolling），且會先驗證baseline版本跟
官方凍結的walkforward_members.parquet一致，確保這次用的是同一套可信方法論。
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
IS_START, IS_END = "2007-01", "2023-12"
OOS_START, OOS_END = "2024-01", "2025-12"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
FROZEN_MEMBERS_PATH = (Path(__file__).resolve().parent.parent.parent
                       / "_analysis_outputs_robustness" / "walkforward_members.parquet")


def save_picks(out_path: Path, variant: str, result: dict) -> None:
    rows = []
    for allocation in ("equal", "proportional"):
        rows.append({
            "tree_key": "TW", "scheme": "E", "window_no": 4,
            "k_mode": "silhouette_is", "ratio": RATIO, "allocation": allocation,
            "group": "A_hrp", "variant": variant,
            "is_start": IS_START, "is_end": IS_END, "oos_start": OOS_START, "oos_end": OOS_END,
            "members": result[allocation],
        })
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"寫入 {out_path}（{result['_meta']}）")


def check_against_frozen(baseline_equal: list[str]) -> None:
    m = pd.read_parquet(FROZEN_MEMBERS_PATH)
    key = dict(tree_key="TW", scheme="E", window_no=4, k_mode="silhouette_is",
              ratio=RATIO, allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    assert len(sub) == 1
    frozen = set(sub.iloc[0]["members"])
    mine = set(baseline_equal)
    inter = frozen & mine
    print(f"\n=== 驗證：現場重算的baseline跟官方凍結名單(今晚才重建過的新池子)比對 ===")
    print(f"官方凍結：{len(frozen)}檔｜這次現場算：{len(mine)}檔｜交集：{len(inter)}檔")
    if frozen == mine:
        print("✅ 完全一致——方法論驗證通過，exclude_v1版本可信")
    else:
        print(f"⚠️ 不完全一致（只在官方名單：{len(frozen-mine)}檔／只在這次名單：{len(mine-frozen)}檔）"
             f"——需要查原因，不能直接假設是誤差")


def main():
    print(">> 載入資料(今晚重建的openSec_boost池)...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    print("\n### 1. anchored window4／baseline（驗證用）###")
    base = build_silhouette_picks("TW", IS_START, IS_END, "baseline",
                                  months_long, meta_pool, f_combo_map, idx)
    check_against_frozen(base["equal"])

    print("\n### 2. anchored window4／exclude_v1 ###")
    ev1 = build_silhouette_picks("TW", IS_START, IS_END, "exclude_v1",
                                 months_long, meta_pool, f_combo_map, idx)
    save_picks(OUT_DIR / "anchored_w4_openSec_boost_exclude_v1.parquet", "exclude_v1", ev1)


if __name__ == "__main__":
    main()
