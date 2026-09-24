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

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._check_rolling_window6_silhouette_picks
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import hrp  # noqa: E402
from research import k_stability  # noqa: E402
from research import stage3_hrp as S3  # noqa: E402
from research import walkforward_matrix as WF  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _design_test_rolling_is6oos2 import (IS_MONTHS, OOS_LEN, RATIO,  # noqa: E402
                                          TREE_KEY, window_dates_rolling)

WINDOW_NO = 6  # OOS 2023-01~2025-12，跟實戰管線 REGISTRATION_DATE/QUARTER_ENDS 對齊
OUT_PATH = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
           / "rolling_window6_silhouette_members.parquet")


def build_silhouette_picks(is_start: str, is_end: str, months_long, meta_pool,
                           f_combo_map, log=print) -> dict:
    log(f">> 建樹（fixed k_mode，取得可重切的linkage）IS {is_start}~{is_end}...")
    tree = WF.build_tree_for_window(TREE_KEY, is_start, is_end, months_long, meta_pool, f_combo_map, log)

    log(">> 只用這個IS窗的資料現場選k（k_stability.scan_window，不看未來）...")
    scan = k_stability.scan_window(TREE_KEY, is_start, is_end, months_long, meta_pool, log=log)
    k_is = scan["k_is_selected"]
    log(f"   IS選k={k_is}（寫死版k={scan['k_fixed']}，{'一致' if scan['same_as_fixed'] else '不同'}）")

    # 🔴 uids一定要用tree["assign"]（`_build_tree`內部`_drop_zero_variance`後的
    # 名單），不能用`S3._tree_universe`的原始名單——`build_tree_for_window`建樹
    # 過程可能丟掉零變異數策略，`tree["link"]`的leaf數對應的是丟棄後的名單。
    # 比照`walkforward_matrix.py:447`production silhouette_is分支同一個作法。
    uids = pd.Index(tree["assign"][WF.C.PK])
    wide_is = S3._pivot_window(months_long, uids, is_start, is_end)
    corr_full = np.corrcoef(wide_is.to_numpy(dtype=np.float64))

    n_leaf = len(tree["link"]) + 1
    if n_leaf != len(wide_is):
        raise AssertionError(f"linkage葉節點數{n_leaf} != wide_is列數{len(wide_is)}，"
                             f"重切的群標籤會對錯策略，中止")
    labels = hrp.cut_clusters(tree["link"], k_is)
    cmeta, _ = S3._cluster_meta_and_corr(wide_is, corr_full, labels, WF.LEVEL, tree["tree_id"])
    assign = pd.DataFrame({WF.C.PK: wide_is.index.to_numpy(), f"cluster_{WF.LEVEL}": labels})

    cagr_is = WF._cagr_matrix(wide_is)
    mdd_is = WF._mdd_matrix(wide_is)
    quality_is = cagr_is / mdd_is.abs().replace(0, np.nan)
    pos = pd.Series(range(len(wide_is.index)), index=wide_is.index)

    sizes = assign.groupby(f"cluster_{WF.LEVEL}").size()
    k = len(sizes)
    n_uni = len(uids)
    tot = WF.target_total(RATIO, n_uni, k)
    log(f"   n_universe={n_uni}｜k={k}（silhouette選出，非寫死6）｜目標代表數={tot}")

    out = {}
    for allocation in ("equal", "proportional"):
        quota, n_capped = WF.allocate(sizes, tot, allocation)
        a_members, n_bf = WF._pick_a(assign, cmeta, wide_is, quality_is, quota, corr_full, pos)
        log(f"   [{allocation}] 選出 {len(a_members)} 檔（目標{tot}，backfill={n_bf}）")
        out[allocation] = a_members
    out["_meta"] = {"k_is_selected": k_is, "k_fixed": scan["k_fixed"], "n_universe": n_uni,
                    "target_total": tot, "is_start": is_start, "is_end": is_end}
    return out


def main():
    blocks = WF._blocks(WF.SCHEME_TOTAL_MONTHS, IS_MONTHS, OOS_LEN)
    off, L = blocks[WINDOW_NO - 1]
    is_start, is_end, oos_start, oos_end = window_dates_rolling(off, L, TREE_KEY)
    print(f"window{WINDOW_NO}（rolling, silhouette_is）：IS {is_start}~{is_end}｜OOS {oos_start}~{oos_end}")

    print(">> 載入資料...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()

    result = build_silhouette_picks(is_start, is_end, months_long, meta_pool, f_combo_map)

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
