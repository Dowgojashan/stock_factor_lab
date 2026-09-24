# -*- coding: utf-8 -*-
"""接續開發追蹤§3.9：驗證「排除V1」是否真的能讓台積電/聯發科被選中，不只是
composition/CAGR指標改善（那是§3.8/3.9測的），而是直接回答9/22報告§四查證過的
真實問題——用同一組8個as_of日期（`_check_tsmc_exclusion_reason.py`已用過的那組，
對齊A7真實8季解析）、同一個真實production窗次（scheme E／window4／A_hrp，
IS 2007-01~2023-12、OOS 2024-01~2025-12），比較：

  baseline（現行30檔代表，含V1）  vs  exclude_v1（排除V1後重新挑出的代表）

台積電（2330）、聯發科（2454）逐季有沒有被選中的檔數，是否從掛零變成非零。

沿用`_check_tsmc_exclusion_reason.py`的F1/F2/C個別條件檢查方法（不是看最終
持股結果，是直接對每個策略的因子/動態條件/估值濾網個別算布林遮罩），跟
`_design_test_quality_metric_variants.py`的`eligible_uids("exclude_v1", ...)`
篩選＋per-variant quota重算邏輯（避免§3.8記錄過的配額混淆陷阱）。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._check_tsmc_mtk_exclude_v1_comparison
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import (CANDIDATE_INDEX_PATH, _c_condition,  # noqa: E402
                                       _q_band_condition)

from research import walkforward_matrix as WF  # noqa: E402

TSMC = "2330"
MEDIATEK = "2454"
TREE_KEY = "TW"
IS_START, IS_END = "2007-01", "2023-12"
OOS_START, OOS_END = "2024-01", "2025-12"
# 跟 `_check_tsmc_exclusion_reason.py` 同一組日期，對齊 A7 真實8季解析
CHECK_DATES = ["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
              "2025-03-31", "2025-06-30", "2025-09-30"]
MEMBERS_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"


def get_baseline_members() -> list[str]:
    m = pd.read_parquet(MEMBERS_PATH)
    key = dict(tree_key="TW", scheme="E", window_no=4, k_mode="silhouette_is",
              ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    assert len(sub) == 1
    row = sub.iloc[0]
    assert row["is_start"] == IS_START and row["is_end"] == IS_END, \
        f"窗次日期跟預期不符：{row['is_start']}~{row['is_end']}"
    return list(row["members"])


def build_exclude_v1_members(idx: pd.DataFrame) -> list[str]:
    print(">> 建樹 IS 2007-01~2023-12（跟scheme E window4完全對齊，排除V1版本）...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    tree = WF.build_tree_for_window(TREE_KEY, IS_START, IS_END, months_long, meta_pool, f_combo_map, print)
    assign = tree["assign"]
    assign = assign[assign.tree_id == tree["tree_id"]] if "tree_id" in assign.columns else assign
    cmeta = tree["cluster_meta"]
    cmeta = cmeta[cmeta.level == WF.LEVEL] if "level" in cmeta.columns else cmeta

    uids = pd.Index(assign[WF.C.PK])
    wide_is = WF.S3._pivot_window(months_long, uids, IS_START, IS_END)
    cagr_is = WF._cagr_matrix(wide_is)
    mdd_is = WF._mdd_matrix(wide_is)
    quality_is_base = cagr_is / mdd_is.abs().replace(0, np.nan)

    corr_full = np.corrcoef(wide_is.to_numpy(dtype=np.float64))
    pos = pd.Series(range(len(wide_is.index)), index=wide_is.index)

    n_uni = len(uids)
    sizes_full = assign.groupby(f"cluster_{WF.LEVEL}").size()
    k = len(sizes_full)
    tot = WF.target_total("legacy", n_uni, k)

    meta = idx.reindex(list(uids))
    elig = set(meta[meta.V != "v1"].index)
    assign_v = assign[assign[WF.C.PK].isin(elig)]
    # 🔴 quota per-variant重算（§3.8記錄過的陷阱，這裡只有一個變體但同樣要用
    # 篩選後的群大小重算，不能沿用篩選前的quota）
    sizes_v = assign_v.groupby(f"cluster_{WF.LEVEL}").size()
    quota, n_capped = WF.allocate(sizes_v, tot, "equal")
    quality_v = quality_is_base.reindex(list(elig))
    a_members, n_bf = WF._pick_a(assign_v, cmeta, wide_is, quality_v, quota, corr_full, pos)
    print(f"   排除V1後選出 {len(a_members)} 檔代表（目標{tot}檔）")
    return a_members


def asof_bool(mask, as_of: str, symbol: str) -> bool | None:
    if mask is None or symbol not in mask.columns:
        return None
    ts = pd.Timestamp(as_of)
    valid = mask.index[mask.index <= ts]
    if len(valid) == 0:
        return None
    return bool(mask.loc[valid.max(), symbol])


def check_members(members: list[str], idx: pd.DataFrame, md: MarketData, label: str) -> pd.DataFrame:
    rows = []
    for uid in members:
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
            for sym, col in [(TSMC, "tsmc"), (MEDIATEK, "mtk")]:
                f1_pass = asof_bool(f1_mask, d, sym)
                f2_pass = asof_bool(f2_mask, d, sym) if f2_mask is not None else None
                c_pass = asof_bool(c_mask, d, sym) if c_mask is not None else None
                v_pass = asof_bool(v_mask, d, sym) if v_mask is not None else None
                fully_selected = (f1_pass is True
                                  and (f2_pass is True or f2_mask is None)
                                  and (c_pass is True or c_mask is None)
                                  and (v_pass is True or v_mask is None))
                rows.append({"variant": label, "uid": uid, "as_of": d, "symbol": col,
                            "V": row["V"],
                            "F1_factor": row["F1_factor"], "F1_band": row["F1_band"],
                            "F1_nbands": row["F1_nbands"], "F1_pass": f1_pass,
                            "F2_factor": row["F2_factor"] if f2_mask is not None else None,
                            "F2_pass": f2_pass,
                            "C_source": row["C_source"] if c_mask is not None else None,
                            "C_rule": row["C_rule"] if c_mask is not None else None,
                            "C_pass": c_pass, "V_pass": v_pass,
                            "selected": fully_selected})
    return pd.DataFrame(rows)


def main():
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    baseline_members = get_baseline_members()
    print(f"baseline（現行）members: {len(baseline_members)} 檔")
    ev1_members = build_exclude_v1_members(idx)

    print(">> 載入 TW MarketData ...")
    md = MarketData(TREE_KEY)

    df_base = check_members(baseline_members, idx, md, "baseline")
    df_ev1 = check_members(ev1_members, idx, md, "exclude_v1")
    df = pd.concat([df_base, df_ev1], ignore_index=True)

    out = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "tsmc_mtk_exclude_v1_comparison.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out}")

    print("\n=== 逐季選中檔數：baseline（含V1）vs exclude_v1 ===")
    piv = df.groupby(["as_of", "symbol", "variant"])["selected"].sum().unstack("variant").reindex(
        columns=["baseline", "exclude_v1"])
    n_base = len(baseline_members)
    n_ev1 = len(ev1_members)
    print(f"（分母：baseline={n_base}檔，exclude_v1={n_ev1}檔）")
    print(piv.to_string())

    print("\n=== exclude_v1：台積電仍未通過的策略，卡在哪一關（F1/F2/C）===")
    ev1_tsmc = df[(df.variant == "exclude_v1") & (df.symbol == "tsmc")]
    fail = ev1_tsmc[~ev1_tsmc["selected"]]
    print(f"  F1 通過率：{fail['F1_pass'].mean():.1%}（{fail['F1_pass'].sum()}/{len(fail)}）")
    f2_sub = fail[fail["F2_pass"].notna()]
    if len(f2_sub):
        print(f"  F2 通過率（有F2的組合中）：{f2_sub['F2_pass'].mean():.1%}（{f2_sub['F2_pass'].sum()}/{len(f2_sub)}）")
    c_sub = fail[fail["C_pass"].notna()]
    if len(c_sub):
        print(f"  C 通過率（有C的組合中）：{c_sub['C_pass'].mean():.1%}（{c_sub['C_pass'].sum()}/{len(c_sub)}）")
    f1_fail = fail[fail["F1_pass"] == False]  # noqa: E712
    if len(f1_fail):
        print("  F1 沒通過的因子/band 分布（前10）：")
        counts = f1_fail.groupby(["F1_factor", "F1_band", "F1_nbands"]).size().sort_values(ascending=False)
        for (factor, band, nbands), cnt in counts.head(10).items():
            print(f"    {factor}  band={band}/{nbands}  ({cnt} 次)")
    c_fail = fail[fail["C_pass"] == False]  # noqa: E712
    if len(c_fail):
        print("  C 沒通過的規則分布（前10）：")
        counts = c_fail.groupby(["C_source", "C_rule"]).size().sort_values(ascending=False)
        for (src, rule), cnt in counts.head(10).items():
            print(f"    {src} {rule}  ({cnt} 次)")


if __name__ == "__main__":
    main()
