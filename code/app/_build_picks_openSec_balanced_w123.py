# -*- coding: utf-8 -*-
"""2026-10-27（實戰收尾追蹤.md §四步驟1，資料缺口）：補建美股 openSec_balanced
anchored window1~3 的 baseline + exclude_v1 代表策略名單。

步驟1（優等生權重規則網格）要在 window1~3 選規則、window4 驗證；美股 exclude_v1
原本只有 window4 名單（`_build_exclude_v1_picks_openSec_balanced.py`），這支把
同一套方法論（`build_silhouette_picks()`）套到 window1~3，其餘完全不變。

IS/OOS 日期不自己算，直接讀官方 `walkforward_members.parquet` 對應窗次的
is_start/is_end/oos_start/oos_end（方案E），避免日期跟研究部主鏈對不上。
每個窗都先驗證 baseline 現場重算＝官方凍結名單，不一致就中止。

不呼叫 LLM。輸出：`_analysis_outputs_applayer/anchored_w{1,2,3}_openSec_balanced_{baseline,exclude_v1}.parquet`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import pandas as pd  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402

from research import walkforward_matrix as WF  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _silhouette_picks_lib import build_silhouette_picks  # noqa: E402

TREE_KEY = "US"
RATIO = "legacy"
WINDOWS = (1, 2, 3)
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
FROZEN_MEMBERS_PATH = (Path(__file__).resolve().parent.parent.parent
                       / "_analysis_outputs_robustness" / "walkforward_members.parquet")


def frozen_row(m: pd.DataFrame, window_no: int, allocation: str) -> pd.Series:
    key = dict(tree_key=TREE_KEY, scheme="E", window_no=window_no, k_mode="silhouette_is",
               ratio=RATIO, allocation=allocation, group="A_hrp")
    sub = m
    for k, v in key.items():
        sub = sub[sub[k] == v]
    if len(sub) != 1:
        raise AssertionError(f"官方凍結名單裡找不到唯一對應列：{key}（找到{len(sub)}列）")
    return sub.iloc[0]


def save_picks(out_path: Path, window_no: int, dates: tuple, variant: str, result: dict) -> None:
    is_start, is_end, oos_start, oos_end = dates
    rows = [{"tree_key": TREE_KEY, "scheme": "E", "window_no": window_no,
             "k_mode": "silhouette_is", "ratio": RATIO, "allocation": allocation,
             "group": "A_hrp", "variant": variant,
             "is_start": is_start, "is_end": is_end, "oos_start": oos_start, "oos_end": oos_end,
             "members": result[allocation]}
            for allocation in ("equal", "proportional")]
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"寫入 {out_path.name}（{result['_meta']}）")


def main():
    m = pd.read_parquet(FROZEN_MEMBERS_PATH)
    print(">> 載入研究部輸入...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    for w in WINDOWS:
        fr = frozen_row(m, w, "equal")
        dates = (fr.is_start, fr.is_end, fr.oos_start, fr.oos_end)
        print(f"\n########## window{w}：IS {dates[0]}~{dates[1]}／OOS {dates[2]}~{dates[3]} ##########")

        base = build_silhouette_picks(TREE_KEY, dates[0], dates[1], "baseline",
                                      months_long, meta_pool, f_combo_map, idx)
        for allocation in ("equal", "proportional"):
            frozen = set(frozen_row(m, w, allocation).members)
            mine = set(base[allocation])
            print(f"   驗證 w{w}/{allocation}：官方{len(frozen)}檔｜現場{len(mine)}檔｜交集{len(frozen & mine)}檔")
            if frozen != mine:
                raise AssertionError(f"w{w}/{allocation} baseline 跟官方凍結名單不一致，中止")
        print(f"   ✅ w{w} baseline 跟官方凍結名單完全一致")
        save_picks(OUT_DIR / f"anchored_w{w}_openSec_balanced_baseline.parquet", w, dates, "baseline", base)

        ev1 = build_silhouette_picks(TREE_KEY, dates[0], dates[1], "exclude_v1",
                                     months_long, meta_pool, f_combo_map, idx)
        save_picks(OUT_DIR / f"anchored_w{w}_openSec_balanced_exclude_v1.parquet", w, dates, "exclude_v1", ev1)


if __name__ == "__main__":
    main()
