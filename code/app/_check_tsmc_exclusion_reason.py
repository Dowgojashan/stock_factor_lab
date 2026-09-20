# -*- coding: utf-8 -*-
"""為什麼台積電（2330）幾乎沒被 window 4（TW/A_hrp/scheme E/legacy/equal）
的 50 個策略選中——A7（開發追蹤）已經查出「8季裡7季配置0.00%」這個事實，
這支腳本查**為什麼**：逐一檢查每個策略的 F1/F2/C/V 條件，台積電通過/沒通過
哪些，藉此找出真正把它篩掉的是哪一類條件（規模因子？估值因子？動能因子？）。

方法：沿用 `resolve_strategy_holdings.py` 重建條件遮罩的同一套機制
（`condition_factory.build_conditions` → `MarketData.get_mask()`)，
對每個策略的 F1/F2/C/V 個別檢查台積電那天的布林值（不是只看最終合併結果），
才能定位「卡在哪一關」，不是只知道「沒通過」。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import (CANDIDATE_INDEX_PATH, _c_condition,  # noqa: E402
                                       _q_band_condition)

TSMC = "2330"
# 🔴 2026-09-22 code review 自己抓到的真問題：原本只測 3 個時點（登記日/年底×2），
# A7（`_prelim_a7_structural_gap_chained.py`）用真實8季解析（as_of=8個季度起點）
# 早就查到台積電在其中1季（as_of=2025-09-30）曾以0.21%短暫入選，剛好不在原本測的
# 3個日期裡——用「0/30」去描述整個8季會過度誇大排除的一致性。改成跟A7同一組
# 8個真實 as_of 日期全部測，才能誠實回答「是每季都排除，還是只是多數季度排除」。
CHECK_DATES = ["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
              "2025-03-31", "2025-06-30", "2025-09-30"]
MEMBERS_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"


def get_window4_members() -> list[str]:
    m = pd.read_parquet(MEMBERS_PATH)
    key = dict(tree_key="TW", scheme="E", window_no=4, k_mode="silhouette_is",
              ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    assert len(sub) == 1
    return list(sub.iloc[0]["members"])


def asof_bool(mask: pd.DataFrame, as_of: str, symbol: str) -> bool | None:
    if mask is None or symbol not in mask.columns:
        return None
    ts = pd.Timestamp(as_of)
    valid = mask.index[mask.index <= ts]
    if len(valid) == 0:
        return None
    return bool(mask.loc[valid.max(), symbol])


def main():
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    uids = get_window4_members()
    print(f"window 4 策略數：{len(uids)}")

    print(">> 載入 TW MarketData ...")
    md = MarketData("TW")

    rows = []
    for uid in uids:
        row = idx.loc[uid]
        f1_mask = md.get_mask(_q_band_condition(row["F1_factor"], row["F1_band"], row["F1_nbands"]))
        f2_mask = None
        if not row["F2_empty"]:
            f2_mask = md.get_mask(_q_band_condition(row["F2_factor"], row["F2_band"], row["F2_nbands"]))
        c_mask = None
        if pd.notna(row["C_rule"]):
            c_mask = md.get_mask(_c_condition(row["C_source"], row["C_rule"]))
        v_mask = md.get_v_mask() if row["V"] == "v1" else None

        for d in CHECK_DATES:
            f1_pass = asof_bool(f1_mask, d, TSMC)
            f2_pass = asof_bool(f2_mask, d, TSMC) if f2_mask is not None else None
            c_pass = asof_bool(c_mask, d, TSMC) if c_mask is not None else None
            v_pass = asof_bool(v_mask, d, TSMC) if v_mask is not None else None
            all_pass = all(p for p in [f1_pass, f2_pass, c_pass, v_pass] if p is not None)
            # 但要求至少 F1 有算出來，且沒有缺條件（None 表示算不出來，不能算數）
            fully_selected = (f1_pass is True
                              and (f2_pass is True or f2_mask is None)
                              and (c_pass is True or c_mask is None)
                              and (v_pass is True or v_mask is None))
            rows.append({
                "uid": uid, "as_of": d,
                "F1_factor": row["F1_factor"], "F1_band": row["F1_band"], "F1_nbands": row["F1_nbands"],
                "F1_pass": f1_pass,
                "F2_factor": row["F2_factor"] if f2_mask is not None else None,
                "F2_pass": f2_pass,
                "C_source": row["C_source"] if c_mask is not None else None,
                "C_rule": row["C_rule"] if c_mask is not None else None,
                "C_pass": c_pass,
                "V": row["V"], "V_pass": v_pass,
                "selected": fully_selected,
            })

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "tsmc_exclusion_reason.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"寫入 {out}")

    for d in CHECK_DATES:
        sub = df[df.as_of == d]
        n_sel = sub["selected"].sum()
        print(f"\n=== {d}：{n_sel}/{len(sub)} 個策略選中台積電 ===")
        print(f"  F1 通過率：{sub['F1_pass'].mean():.1%}（{sub['F1_pass'].sum()}/{len(sub)}）")
        f2_sub = sub[sub["F2_pass"].notna()]
        if len(f2_sub):
            print(f"  F2 通過率（有F2的策略中）：{f2_sub['F2_pass'].mean():.1%}（{f2_sub['F2_pass'].sum()}/{len(f2_sub)}）")
        c_sub = sub[sub["C_pass"].notna()]
        if len(c_sub):
            print(f"  C 通過率（有C的策略中）：{c_sub['C_pass'].mean():.1%}（{c_sub['C_pass'].sum()}/{len(c_sub)}）")
        v_sub = sub[sub["V_pass"].notna()]
        if len(v_sub):
            print(f"  V 通過率（有V的策略中）：{v_sub['V_pass'].mean():.1%}（{v_sub['V_pass'].sum()}/{len(v_sub)}）")

        # F1 沒過的策略，是哪些因子 + band 組合？
        f1_fail = sub[sub["F1_pass"] == False]  # noqa: E712
        if len(f1_fail):
            print(f"  F1 沒通過的因子/band 分布（前10）：")
            counts = f1_fail.groupby(["F1_factor", "F1_band", "F1_nbands"]).size().sort_values(ascending=False)
            for (factor, band, nbands), cnt in counts.head(10).items():
                print(f"    {factor}  band={band}/{nbands}  ({cnt} 個策略)")


if __name__ == "__main__":
    main()
