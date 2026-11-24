# -*- coding: utf-8 -*-
"""2026-11-17（使用者授權）：研究部 11-09～11-14 依新台股候選池（32,726 檔、k=8）重跑後，
重建台股 openSec_boost anchored window1～4 的 exclude_v1 代表策略名單（步驟1／步驟2／歸因鏈的輸入）。

完全比照 `_build_picks_openSec_balanced_w123.py`（美股版）：
  - IS/OOS 日期直接讀官方 `walkforward_members.parquet`（方案E、silhouette_is、legacy）的對應窗次
  - 每個窗先驗證 baseline 現場重算＝官方名單（equal／proportional 都要完全相同），不一致就中止
  - exclude_v1 用同一個 `build_silhouette_picks()`
唯一差異：同一窗的 baseline 與 exclude_v1 共用同一棵樹與同一次選 k（兩者本來就相同，只是不重算一次）。
舊檔已備份到 `_analysis_outputs_applayer/_舊版備份_20261117_研究部11-14重跑後重建前/`。
不呼叫 LLM。輸出：`_analysis_outputs_applayer/anchored_w{1,2,3,4}_openSec_boost_exclude_v1.parquet`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import pandas as pd  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402

from research import k_stability  # noqa: E402
from research import walkforward_matrix as WF  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _silhouette_picks_lib as LIB  # noqa: E402

TREE_KEY = "TW"
RATIO = "legacy"
WINDOWS = (1, 2, 3, 4)
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
FROZEN_MEMBERS_PATH = (Path(__file__).resolve().parent.parent.parent
                       / "_analysis_outputs_robustness" / "walkforward_members.parquet")


def _memo(fn):
    cache = {}

    def wrap(tree_key, is_start, is_end, *a, **k):
        key = (tree_key, is_start, is_end)
        if key not in cache:
            cache[key] = fn(tree_key, is_start, is_end, *a, **k)
        return cache[key]
    return wrap


# 同一窗的兩個變體共用建樹與選 k（函式本身不變，只快取結果）
WF.build_tree_for_window = _memo(WF.build_tree_for_window)
k_stability.scan_window = _memo(k_stability.scan_window)


def frozen_row(m: pd.DataFrame, window_no: int, allocation: str) -> pd.Series:
    key = dict(tree_key=TREE_KEY, scheme="E", window_no=window_no, k_mode="silhouette_is",
               ratio=RATIO, allocation=allocation, group="A_hrp")
    sub = m
    for k, v in key.items():
        sub = sub[sub[k].astype(str) == str(v)]
    if len(sub) != 1:
        raise AssertionError(f"官方名單裡找不到唯一對應列：{key}（找到{len(sub)}列）")
    return sub.iloc[0]


def save_picks(out_path: Path, window_no: int, dates: tuple, result: dict) -> None:
    is_start, is_end, oos_start, oos_end = dates
    rows = [{"tree_key": TREE_KEY, "scheme": "E", "window_no": window_no,
             "k_mode": "silhouette_is", "ratio": RATIO, "allocation": allocation,
             "group": "A_hrp", "variant": "exclude_v1",
             "is_start": is_start, "is_end": is_end, "oos_start": oos_start, "oos_end": oos_end,
             "members": result[allocation]}
            for allocation in ("equal", "proportional")]
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"寫入 {out_path.name}（{result['_meta']}）", flush=True)


def main():
    m = pd.read_parquet(FROZEN_MEMBERS_PATH)
    print(">> 載入研究部輸入...", flush=True)
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    results = {}
    for w in WINDOWS:
        fr = frozen_row(m, w, "equal")
        dates = (str(fr.is_start), str(fr.is_end), str(fr.oos_start), str(fr.oos_end))
        print(f"\n########## window{w}：IS {dates[0]}~{dates[1]}／OOS {dates[2]}~{dates[3]} ##########", flush=True)
        base = LIB.build_silhouette_picks(TREE_KEY, dates[0], dates[1], "baseline",
                                          months_long, meta_pool, f_combo_map, idx)
        for allocation in ("equal", "proportional"):
            frozen = set(frozen_row(m, w, allocation).members)
            mine = set(base[allocation])
            print(f"   驗證 w{w}/{allocation}：官方{len(frozen)}檔｜現場{len(mine)}檔｜交集{len(frozen & mine)}檔", flush=True)
            if frozen != mine:
                raise AssertionError(f"w{w}/{allocation} baseline 跟官方名單不一致，中止（未寫任何檔）")
        print(f"   ✅ w{w} baseline 跟官方名單完全一致", flush=True)
        ev1 = LIB.build_silhouette_picks(TREE_KEY, dates[0], dates[1], "exclude_v1",
                                         months_long, meta_pool, f_combo_map, idx)
        results[w] = (dates, ev1)
    # 四個窗都驗證通過才寫檔，避免只更新一部分窗
    for w, (dates, ev1) in results.items():
        save_picks(OUT_DIR / f"anchored_w{w}_openSec_boost_exclude_v1.parquet", w, dates, ev1)


if __name__ == "__main__":
    main()
