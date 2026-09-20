# -*- coding: utf-8 -*-
"""D65延伸item2（2026-09-22）：重建window4的HRP樹，查「會選中台積電的策略」
是否集中在特定1-2個群，藉此解釋為什麼代表挑選（每群配額5個）幾乎完全排除它們。

純本地運算，不需要資料庫/MarketData（`build_tree_for_window`只吃本地parquet的
報酬矩陣），也完全不涉及LLM——重用`research/walkforward_matrix.py`已經驗證過的
`_load_inputs()`／`build_tree_for_window()`，跟正式45窗計算用的是同一套函式，
不是另外寫一份。

「會選台積電的策略」清單直接讀`_check_tsmc_representative_quality.py`已經存好
的`tsmc_representative_quality.csv`（含`selects_tsmc`欄位，920檔），不必重新
查詢資料庫的F1/F2/C/V遮罩。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from research import walkforward_matrix as WM  # noqa: E402

WINDOW4_IS_START, WINDOW4_IS_END = "2007-01", "2023-12"
QUALITY_CSV = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "tsmc_representative_quality.csv"


def main():
    q = pd.read_csv(QUALITY_CSV, index_col=0)
    tsmc_selectors = set(q[q["selects_tsmc"]].index)
    print(f"會選台積電的策略數：{len(tsmc_selectors)}")

    print(">> 載入本地報酬矩陣（months_long/meta/f_combo_map）...")
    months_long, meta, f_combo_map = WM._load_inputs()

    print(f">> 重建 window4 的 TW 樹（IS {WINDOW4_IS_START}~{WINDOW4_IS_END}）...")
    tree = WM.build_tree_for_window("TW", WINDOW4_IS_START, WINDOW4_IS_END,
                                    months_long, meta, f_combo_map)

    assign = tree["assign"]
    print(f"\n樹內策略總數（跟候選池7128略有差異，因usable_pool/零變異數過濾）："
         f"{len(assign)}")

    assign["selects_tsmc"] = assign["strategy_uid"].isin(tsmc_selectors)

    print("\n=== 每個L1群：整體大小 vs 會選台積電的策略數 ===")
    summary = assign.groupby("cluster_L1").agg(
        n_members=("strategy_uid", "count"),
        n_tsmc_selectors=("selects_tsmc", "sum"),
    )
    summary["pct_tsmc_selectors"] = summary["n_tsmc_selectors"] / summary["n_members"]
    summary["pct_of_all_tsmc_selectors"] = summary["n_tsmc_selectors"] / summary["n_tsmc_selectors"].sum()
    print(summary.to_string())

    # window4 legacy 實際挑出的30個代表，各自落在哪個群、該群配額用完沒
    members_path = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"
    m = pd.read_parquet(members_path)
    key = dict(tree_key="TW", scheme="E", window_no=4, k_mode="silhouette_is",
              ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    legacy_members = set(sub.iloc[0]["members"])

    assign["is_legacy_pick"] = assign["strategy_uid"].isin(legacy_members)
    print("\n=== 每個L1群：legacy代表挑選結果 vs 該群裡有多少台積電候選被跳過 ===")
    summary2 = assign.groupby("cluster_L1").agg(
        n_members=("strategy_uid", "count"),
        n_picked=("is_legacy_pick", "sum"),
        n_tsmc_selectors=("selects_tsmc", "sum"),
        n_tsmc_selectors_picked=("selects_tsmc", lambda s: (s & assign.loc[s.index, "is_legacy_pick"]).sum()),
    )
    print(summary2.to_string())

    out = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "tsmc_cluster_concentration.csv"
    assign.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out}")


if __name__ == "__main__":
    main()
