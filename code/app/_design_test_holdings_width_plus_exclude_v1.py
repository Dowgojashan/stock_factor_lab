# -*- coding: utf-8 -*-
"""接續§3.23：持股寬度版配額規則（目前idea②最佳候選）跟排除V1（§3.14已驗證
真實管線雙贏），兩者可能命中同一個底層機制（窄持股/V1造成Calmar失真）——
這裡測4個組合，搞清楚疊加後是互補（效果更好）還是重疊（沒有額外幫助）：

  1. baseline：calmar排序＋equal配額＋保留V1（現行）
  2. holdings_width：calmar排序＋持股寬度反向配額＋保留V1（§3.23）
  3. exclude_v1：calmar排序＋equal配額＋排除V1（§3.8/3.9的邏輯，重跑於這個
     silhouette_is+OOS評估框架下）
  4. combo：calmar排序＋持股寬度反向配額＋排除V1（這次要測的新組合）

anchored window4／rolling window6各測一次。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._design_test_holdings_width_plus_exclude_v1
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH, resolve_holdings  # noqa: E402

from research import hrp  # noqa: E402
from research import k_stability  # noqa: E402
from research import stage3_hrp as S3  # noqa: E402
from research import walkforward_matrix as WF  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _design_test_quality_metric_variants import eligible_uids  # noqa: E402
from _design_test_quota_holdings_width import cluster_avg_holdings  # noqa: E402
from _design_test_quota_rules import allocate_by_weight  # noqa: E402
from _design_test_rolling_is6oos2 import (IS_MONTHS, OOS_LEN, RATIO,  # noqa: E402
                                          TREE_KEY, window_dates_rolling)

FACTOR_CATEGORY = {
    "ROE": "資本報酬/獲利能力", "ROIC": "資本報酬/獲利能力", "CROIC": "資本報酬/獲利能力",
    "EPS": "獲利", "FCF_P": "現金流品質", "FCF_OI": "現金流品質", "OCF_E": "現金流品質",
    "EV_EBITDA": "估值倍數", "EV_S": "估值倍數", "PB": "估值倍數", "PS": "估值倍數", "P_IC": "估值倍數",
    "MOM": "動量",
}
ANCHORED_IS_START, ANCHORED_IS_END = "2007-01", "2023-12"
ANCHORED_OOS_START, ANCHORED_OOS_END = "2024-01", "2025-12"
ROLLING_WINDOW_NO = 6
COMBOS = ["baseline", "holdings_width", "exclude_v1", "combo"]


def categorize(row: pd.Series) -> str:
    return FACTOR_CATEGORY.get(row["F1_factor"], f"未分類({row['F1_factor']})")


def run_one(label: str, combo: str, is_start: str, is_end: str, oos_start: str, oos_end: str,
           months_long, meta_pool, f_combo_map, idx: pd.DataFrame, mktcap: pd.DataFrame, md: MarketData):
    print(f"\n{'='*60}\n{label}／{combo}：IS {is_start}~{is_end}")

    tree = WF.build_tree_for_window(TREE_KEY, is_start, is_end, months_long, meta_pool, f_combo_map, print)
    scan = k_stability.scan_window(TREE_KEY, is_start, is_end, months_long, meta_pool, log=print)
    k_is = scan["k_is_selected"]

    uids = pd.Index(tree["assign"][WF.C.PK])
    wide_is = S3._pivot_window(months_long, uids, is_start, is_end)
    w = months_long[months_long.strategy_uid.isin(set(uids))]
    w = w[(w.month >= pd.Period(oos_start, "M")) & (w.month <= pd.Period(oos_end, "M"))]
    wide_oos = w.pivot(index="strategy_uid", columns="month", values="ret")
    n_nan = int(wide_oos.isna().sum().sum())
    n_missing = len(set(uids) - set(wide_oos.index))
    if n_nan or n_missing:
        raise ValueError(f"[{label}] OOS資料不完整：{n_nan}個缺值、{n_missing}個策略無資料")

    corr_full = np.corrcoef(wide_is.to_numpy(dtype=np.float64))
    labels = hrp.cut_clusters(tree["link"], k_is)
    cmeta, _ = S3._cluster_meta_and_corr(wide_is, corr_full, labels, WF.LEVEL, tree["tree_id"])
    assign = pd.DataFrame({WF.C.PK: wide_is.index.to_numpy(), f"cluster_{WF.LEVEL}": labels})
    cluster_map = assign.set_index(WF.C.PK)[f"cluster_{WF.LEVEL}"]

    cagr_is = WF._cagr_matrix(wide_is)
    mdd_is = WF._mdd_matrix(wide_is)
    quality_is_base = cagr_is / mdd_is.abs().replace(0, np.nan)
    pos = pd.Series(range(len(wide_is.index)), index=wide_is.index)

    meta = idx.reindex(list(uids)).copy()
    meta["category"] = meta.apply(categorize, axis=1)

    sizes_full = assign.groupby(f"cluster_{WF.LEVEL}").size()
    k = len(sizes_full)
    n_uni = len(uids)
    tot = WF.target_total(RATIO, n_uni, k)

    use_v1_filter = combo in ("exclude_v1", "combo")
    use_holdings_width = combo in ("holdings_width", "combo")

    if use_v1_filter:
        bench_cagr = WF._cagr(WF._portfolio_series(wide_is, list(uids)))
        elig = eligible_uids("exclude_v1", assign, idx, cagr_is, bench_cagr)
    else:
        elig = set(uids)
    assign_v = assign[assign[WF.C.PK].isin(elig)]
    if assign_v.empty:
        raise ValueError(f"[{label}/{combo}] 排除V1後沒有任何策略通過篩選")
    quality_v = quality_is_base.reindex(list(elig))

    # 🔴 quota per-variant重算（§3.8教訓）：不管哪個combo，都用篩選後的群大小
    # 重算，tot維持用篩選前的全宇宙算出的同一個目標值
    sizes_v = assign_v.groupby(f"cluster_{WF.LEVEL}").size()
    if use_holdings_width:
        avg_hold_by_cluster = cluster_avg_holdings(assign_v, meta)
        quota, n_capped = allocate_by_weight(sizes_v, avg_hold_by_cluster, tot)
    else:
        quota, n_capped = WF.allocate(sizes_v, tot, "equal")

    a_members, n_bf = WF._pick_a(assign_v, cmeta, wide_is, quality_v, quota, corr_full, pos)
    if len(a_members) != tot:
        print(f"   ⚠️ 實選{len(a_members)}檔 != 目標{tot}檔（combo={combo}）")

    perf = WF._evaluate(a_members, wide_is, wide_oos, cluster_map)

    sub = idx.reindex(a_members).copy()
    sub["category"] = sub.apply(categorize, axis=1)
    v1_frac = float((sub["V"] == "v1").mean())
    avg_hold = float(sub["avg_holdings"].mean())
    val_frac = float((sub["category"] == "估值倍數").mean())

    all_syms: set[str] = set()
    for uid in a_members:
        try:
            syms, _ = resolve_holdings(md, idx.loc[uid], is_end)
        except (RuntimeError, KeyError, ValueError):
            continue
        all_syms |= set(syms)
    valid_mk = mktcap.index[mktcap.index <= pd.Timestamp(is_end)]
    if len(valid_mk):
        mk_row = mktcap.loc[valid_mk.max()]
        pct_rank = mk_row.reindex(md.common).dropna().rank(pct=True)
        held_pct = pct_rank.reindex(list(all_syms)).dropna()
        frac_top10 = float((held_pct >= 0.90).mean()) if len(held_pct) else float("nan")
    else:
        frac_top10 = float("nan")

    print(f"目標{tot}檔｜實選{len(a_members)}檔")
    print(f"估值佔比={val_frac:.1%}｜V1比例={v1_frac:.1%}｜平均持股={avg_hold:.1f}檔｜市值前10%佔比={frac_top10:.1%}")
    print(f"OOS CAGR={perf['oos_cagr']:+.2%} MDD={perf['oos_mdd']:.2%} Sharpe={perf['oos_sharpe']:.2f}")

    return {"label": label, "combo": combo, "n_members": len(a_members), "val_frac": val_frac,
           "v1_frac": v1_frac, "avg_holdings": avg_hold, "frac_top10pct_mktcap": frac_top10,
           "oos_cagr": perf["oos_cagr"], "oos_mdd": perf["oos_mdd"], "oos_sharpe": perf["oos_sharpe"]}


def main():
    print(">> 載入資料...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    md = MarketData(TREE_KEY, start="2000-01-01")
    mktcap = md.get_field("report:mktcap")

    blocks = WF._blocks(WF.SCHEME_TOTAL_MONTHS, IS_MONTHS, OOS_LEN)
    off, L = blocks[ROLLING_WINDOW_NO - 1]
    r_is_start, r_is_end, r_oos_start, r_oos_end = window_dates_rolling(off, L, TREE_KEY)

    rows = []
    for combo in COMBOS:
        rows.append(run_one("anchored_w4", combo, ANCHORED_IS_START, ANCHORED_IS_END,
                            ANCHORED_OOS_START, ANCHORED_OOS_END,
                            months_long, meta_pool, f_combo_map, idx, mktcap, md))
    for combo in COMBOS:
        rows.append(run_one("rolling_w6", combo, r_is_start, r_is_end, r_oos_start, r_oos_end,
                            months_long, meta_pool, f_combo_map, idx, mktcap, md))

    out = pd.DataFrame(rows)
    out_path = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "holdings_width_plus_exclude_v1.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}")

    print("\n=== 總覽：baseline / holdings_width / exclude_v1 / combo ===")
    print(out[["label", "combo", "n_members", "val_frac", "v1_frac", "avg_holdings",
              "frac_top10pct_mktcap", "oos_cagr", "oos_mdd", "oos_sharpe"]].to_string(index=False))


if __name__ == "__main__":
    main()
