# -*- coding: utf-8 -*-
"""接續§3.11：使用者問「候選池F1因子73%是估值類，但剩下27%是動量/現金流品質，
即使到最近幾年，代表策略挑選會不會多選一點這27%」——§3.11只查過window6（最後
一窗）的因子分類比例，這裡擴大到全部6個rolling窗次，直接看比例隨窗次（越晚
越接近近期）有沒有趨勢變化，不要用推論回答。

沿用`_check_rolling_window6_composition.py`同一套建樹/挑選/分類邏輯，只是外層
包一個迴圈跑完6個窗次（window_no=1~6），比較candidate pool本身的類別比例
vs 30/50檔代表的類別比例，逐窗列出。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._check_rolling_all_windows_category_trend
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402

from research import walkforward_matrix as WF  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _design_test_rolling_is6oos2 import (IS_MONTHS, OOS_LEN, RATIO, ALLOCATION,  # noqa: E402
                                          TREE_KEY, window_dates_rolling)

# 文件/因子候選批次_F與C因子定義.md §2.1/2.2（跟§3.11同一份分類表）
FACTOR_CATEGORY = {
    "ROE": "資本報酬/獲利能力", "ROIC": "資本報酬/獲利能力", "CROIC": "資本報酬/獲利能力",
    "EPS": "獲利", "FCF_P": "現金流品質", "FCF_OI": "現金流品質", "OCF_E": "現金流品質",
    "EV_EBITDA": "估值倍數", "EV_S": "估值倍數", "PB": "估值倍數", "PS": "估值倍數", "P_IC": "估值倍數",
    "MOM": "動量",
}


def categorize(row: pd.Series) -> str:
    return FACTOR_CATEGORY.get(row["F1_factor"], f"未分類({row['F1_factor']})")


def main():
    print(">> 載入資料...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    blocks = WF._blocks(WF.SCHEME_TOTAL_MONTHS, IS_MONTHS, OOS_LEN)
    rows = []
    for window_no, (off, L) in enumerate(blocks, 1):
        is_start, is_end, oos_start, oos_end = window_dates_rolling(off, L, TREE_KEY)
        print(f"\n>> window{window_no}：IS {is_start}~{is_end}...")
        tree = WF.build_tree_for_window(TREE_KEY, is_start, is_end, months_long, meta_pool, f_combo_map, print)
        assign = tree["assign"]
        assign = assign[assign.tree_id == tree["tree_id"]] if "tree_id" in assign.columns else assign
        cmeta = tree["cluster_meta"]
        cmeta = cmeta[cmeta.level == WF.LEVEL] if "level" in cmeta.columns else cmeta

        uids = pd.Index(assign[WF.C.PK])
        wide_is = WF.S3._pivot_window(months_long, uids, is_start, is_end)
        cagr_is = WF._cagr_matrix(wide_is)
        mdd_is = WF._mdd_matrix(wide_is)
        quality_is = cagr_is / mdd_is.abs().replace(0, np.nan)

        corr_full = np.corrcoef(wide_is.to_numpy(dtype=np.float64))
        pos = pd.Series(range(len(wide_is.index)), index=wide_is.index)

        sizes = assign.groupby(f"cluster_{WF.LEVEL}").size()
        k = len(sizes)
        n_uni = len(uids)
        tot = WF.target_total(RATIO, n_uni, k)
        quota, n_capped = WF.allocate(sizes, tot, ALLOCATION)
        a_members, n_bf = WF._pick_a(assign, cmeta, wide_is, quality_is, quota, corr_full, pos)

        sub = idx.reindex(a_members).copy()
        sub["category"] = sub.apply(categorize, axis=1)
        pool_meta = idx.reindex(list(uids)).copy()
        pool_meta["category"] = pool_meta.apply(categorize, axis=1)

        pool_counts = pool_meta["category"].value_counts(normalize=True)
        pick_counts = sub["category"].value_counts(normalize=True)

        row = {"window_no": window_no, "is_start": is_start, "is_end": is_end,
              "n_universe": n_uni, "n_picks": len(a_members)}
        for cat in ["估值倍數", "動量", "現金流品質"]:
            row[f"pool_{cat}"] = float(pool_counts.get(cat, 0.0))
            row[f"pick_{cat}"] = float(pick_counts.get(cat, 0.0))
        rows.append(row)
        print(f"   候選池：估值{row['pool_估值倍數']:.1%}/動量{row['pool_動量']:.1%}/現金流{row['pool_現金流品質']:.1%}"
             f"｜代表：估值{row['pick_估值倍數']:.1%}/動量{row['pick_動量']:.1%}/現金流{row['pick_現金流品質']:.1%}")

    out = pd.DataFrame(rows)
    out_path = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "rolling_all_windows_category_trend.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}")

    print("\n=== 全部6窗總覽：代表策略的估值類佔比，隨窗次（越晚越近期）有沒有趨勢 ===")
    print(out[["window_no", "is_end", "pool_估值倍數", "pick_估值倍數", "pool_動量", "pick_動量"]].to_string(index=False))


if __name__ == "__main__":
    main()
