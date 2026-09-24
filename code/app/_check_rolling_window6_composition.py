# -*- coding: utf-8 -*-
"""使用者問「改用rolling之後，最後一個窗口（window6）產出的代表策略主要是用哪些」。

重建rolling(IS6/OOS2)最後一窗（window6，IS 2022-12結束、OOS 2023-01~2025-12），
baseline（保留V1，這是`_design_test_rolling_is6oos2.py`原本測的版本、也是「改用
rolling」字面上指的設定，不是§3.8/3.9另外測的exclude_v1變體），列出實際選出的30檔
代表策略的F1/F2/C/V定義，依`文件/因子候選批次_F與C因子定義.md`§2.2的因子分類
（估值倍數／資本報酬/獲利能力／現金流品質／動量）做歸類，比照9/22報告§4.2的
呈現方式（「估值便宜16個、動能強4個...」那張表）。

🔴 `_design_test_rolling_is6oos2.py`本身沒有把a_members（實際選出的策略uid清單）
存下來，只存了聚合統計（composition/CAGR/MDD），這裡重建同一個窗次來補這份清單，
沿用一模一樣的建樹/挑選流程（IS_MONTHS/OOS_LEN/RATIO/ALLOCATION/K_MODE不變）。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._check_rolling_window6_composition
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

WINDOW_NO = 6  # 最後一窗

# 文件/因子候選批次_F與C因子定義.md §2.1/2.2
FACTOR_CATEGORY = {
    "ROE": "資本報酬/獲利能力", "ROIC": "資本報酬/獲利能力", "CROIC": "資本報酬/獲利能力",
    "EPS": "獲利", "FCF_P": "現金流品質", "FCF_OI": "現金流品質", "OCF_E": "現金流品質",
    "EV_EBITDA": "估值倍數", "EV_S": "估值倍數", "PB": "估值倍數", "PS": "估值倍數", "P_IC": "估值倍數",
    "MOM": "動量",
}


def categorize(row: pd.Series) -> str:
    """以F1為主分類（F1是主因子，決定策略主要邏輯），F2是次要配對因子。"""
    return FACTOR_CATEGORY.get(row["F1_factor"], f"未分類({row['F1_factor']})")


def main():
    blocks = WF._blocks(WF.SCHEME_TOTAL_MONTHS, IS_MONTHS, OOS_LEN)
    off, L = blocks[WINDOW_NO - 1]
    is_start, is_end, oos_start, oos_end = window_dates_rolling(off, L, TREE_KEY)
    print(f"window{WINDOW_NO}（rolling, baseline含V1）：IS {is_start}~{is_end}｜OOS {oos_start}~{oos_end}")

    print(">> 載入資料、建樹...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

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
    print(f"實際選出 {len(a_members)} 檔代表（目標{tot}檔）\n")

    sub = idx.reindex(a_members).copy()
    sub["category"] = sub.apply(categorize, axis=1)
    sub["cagr_is"] = cagr_is.reindex(a_members)
    sub["mdd_is"] = mdd_is.reindex(a_members)
    sub["calmar_is"] = quality_is.reindex(a_members)

    # 🔴 使用者追問「明明台灣是大型股環境，為什麼挑出估值因子」——要驗證機制，
    # 不能只憑印象回答：查①這30檔裡估值類是不是持股真的比較窄（Calmar分母
    # MDD被壓低的機制）②更關鍵的，查「全部候選池」（不只是被選中的30檔）裡，
    # 估值類因子在這個IS窗（2017-2022）本身的Calmar是不是真的系統性比較高
    # ——如果是，代表不是挑選機制偏心，是這段IS期間估值類策略本身表現真的
    #比較「好看」（不論是真的好還是MDD壓縮的假象）
    print("\n=== 30檔代表：各分類平均IS CAGR/MDD/Calmar（驗證是否窄持股壓低MDD）===")
    grp = sub.groupby("category").agg(
        n=("category", "size"), avg_holdings=("avg_holdings", "mean"),
        mean_cagr_is=("cagr_is", "mean"), mean_mdd_is=("mdd_is", "mean"),
        mean_calmar_is=("calmar_is", "mean"))
    print(grp.to_string())

    print("\n=== 全部候選池（未篩選前）：各分類平均IS Calmar（驗證是否這段IS期間本身如此）===")
    all_meta = idx.reindex(list(uids)).copy()
    all_meta["category"] = all_meta.apply(categorize, axis=1)
    all_meta["calmar_is"] = quality_is.reindex(list(uids))
    all_meta["cagr_is"] = cagr_is.reindex(list(uids))
    all_meta["mdd_is"] = mdd_is.reindex(list(uids))
    grp_all = all_meta.groupby("category").agg(
        n=("category", "size"), avg_holdings=("avg_holdings", "mean"),
        mean_cagr_is=("cagr_is", "mean"), mean_mdd_is=("mdd_is", "mean"),
        mean_calmar_is=("calmar_is", "mean"))
    print(grp_all.to_string())

    print("=== 依F1主因子分類（比照9/22報告§4.2呈現方式）===")
    cat_counts = sub["category"].value_counts()
    for cat, cnt in cat_counts.items():
        print(f"  {cat}：{cnt}個（{cnt/len(sub):.0%}）")

    print(f"\n=== V欄位（估值濾網）分布 ===")
    v_counts = sub["V"].value_counts()
    for v, cnt in v_counts.items():
        print(f"  {v}：{cnt}個（{cnt/len(sub):.0%}）")

    print(f"\n=== F1因子細項分布 ===")
    f1_counts = sub["F1_factor"].value_counts()
    for f1, cnt in f1_counts.items():
        print(f"  {f1}：{cnt}個")

    print(f"\n=== 是否有F2（次要配對因子）===")
    has_f2 = (~sub["F2_empty"]).sum()
    print(f"  有F2：{has_f2}個／無F2：{len(sub)-has_f2}個")
    if has_f2:
        print("  F2因子細項分布：")
        f2_counts = sub[~sub["F2_empty"]]["F2_factor"].value_counts()
        for f2, cnt in f2_counts.items():
            print(f"    {f2}：{cnt}個")

    print(f"\n=== 是否有C（動態條件）===")
    has_c = sub["C_rule"].notna().sum()
    print(f"  有C：{has_c}個／無C：{len(sub)-has_c}個")
    if has_c:
        print("  C規則細項分布：")
        c_counts = sub[sub["C_rule"].notna()].groupby(["C_source", "C_rule"]).size().sort_values(ascending=False)
        for (src, rule), cnt in c_counts.items():
            print(f"    {src} {rule}：{cnt}個")

    print(f"\n=== 完整清單 ===")
    cols = ["F1_factor", "F1_band", "F1_nbands", "F2_factor", "C_source", "C_rule", "V", "avg_holdings", "category"]
    print(sub[cols].to_string())

    out = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "rolling_window6_members_breakdown.csv"
    sub.to_csv(out, encoding="utf-8-sig")
    print(f"\n寫入 {out}")


if __name__ == "__main__":
    main()
