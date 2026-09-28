# -*- coding: utf-8 -*-
"""補§3.14/§3.25遺留的缺口：rolling候選池目前只有window6算過baseline／exclude_v1
兩個變體（`rolling_window6_silhouette_members.parquet`／`rolling_w6_silhouette_
exclude_v1.parquet`），windows1-5都還沒建過exclude_v1版本，baseline版本也沒有
存成同樣格式的獨立檔案。這裡補齊，讓E（hot segment贏過insample頻率回測）能對
anchored／rolling×保留V1／排除V1四個版本，用完整2013-2025（6個rolling窗次OOS
串接起來）做對稱比較，不是只测window6那8季。

沿用`_silhouette_picks_lib.build_silhouette_picks()`（跟`_build_exclude_v1_
silhouette_picks.py`同一個函式），只是這次跑windows1-5×baseline/exclude_v1
共10組（window6兩個變體已經存在，不重算）。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._build_rolling_all_windows_silhouette_picks
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
WINDOWS_TO_BUILD = (1, 2, 3, 4, 5)   # window6已經有現成檔案，不重算
VARIANTS = ("baseline", "exclude_v1")
OUT_PATH = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
           / "rolling_windows1to5_silhouette_picks.parquet")


def main():
    print(">> 載入資料...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    blocks = WF._blocks(WF.SCHEME_TOTAL_MONTHS, IS_MONTHS, OOS_LEN)

    rows = []
    for window_no in WINDOWS_TO_BUILD:
        off, L = blocks[window_no - 1]
        is_start, is_end, oos_start, oos_end = window_dates_rolling(off, L, TREE_KEY)
        for variant in VARIANTS:
            print(f"\n### window{window_no}／{variant}：IS {is_start}~{is_end} ###")
            result = build_silhouette_picks(TREE_KEY, is_start, is_end, variant,
                                            months_long, meta_pool, f_combo_map, idx,
                                            ratio=RATIO)
            for allocation in ("equal", "proportional"):
                rows.append({
                    # 🔴 "rolling_6_2"跟window_no無關，是IS6:OOS2這個walk-forward
                    # 方案本身的識別字串，6個窗次都用同一個值——跟`rolling_window6_
                    # silhouette_members.parquet`／`_design_test_rolling_is6oos2.py`
                    # 既有的scheme欄位命名慣例一致，不是window6專屬
                    "tree_key": TREE_KEY, "scheme": "rolling_6_2", "window_no": window_no,
                    "k_mode": "silhouette_is", "ratio": RATIO, "allocation": allocation,
                    "group": "A_hrp", "variant": variant,
                    "is_start": is_start, "is_end": is_end,
                    "oos_start": oos_start, "oos_end": oos_end,
                    "members": result[allocation],
                })
            print(f"   {result['_meta']}")

    out = pd.DataFrame(rows)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\n寫入 {OUT_PATH}（{len(out)}列＝{len(WINDOWS_TO_BUILD)}窗×{len(VARIANTS)}變體×2種allocation）")

    print("\n=== 總覽：各窗次baseline vs exclude_v1（equal allocation）檔數 ===")
    eq = out[out.allocation == "equal"]
    for window_no in WINDOWS_TO_BUILD:
        for variant in VARIANTS:
            row = eq[(eq.window_no == window_no) & (eq.variant == variant)]
            n = len(row.iloc[0]["members"]) if len(row) else 0
            print(f"  window{window_no}／{variant}：{n}檔")


if __name__ == "__main__":
    main()
