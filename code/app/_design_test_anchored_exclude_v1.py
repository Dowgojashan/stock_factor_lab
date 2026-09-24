# -*- coding: utf-8 -*-
"""D③後續：補完2×2矩陣最後一格——anchored排除V1（開發追蹤§3.8之後的下一步）。

矩陣現況（見開發追蹤§3.4/3.5/3.8）：
              保留V1（現況）    排除V1
  anchored    已有(scheme A)    ← 這支腳本補這格
  rolling(6:2) 已有(baseline)   已有(exclude_v1)

只測baseline（保留V1，直接沿用既有scheme A官方資料，不重算）跟exclude_v1兩個
變體——holdings floor／cagr floor已經在rolling那邊測過沒有額外效果，不重複測。
窗次用跟rolling test完全對齊的IS結束點/OOS期間（scheme A本來就是這樣設計的，
差別只在IS起點固定在2007-01，不跟著移動）。

沿用`_design_test_quality_metric_variants.py`的`eligible_uids()`（V1篩選邏輯），
跟同一套quota修正（每個變體用篩選後的群大小重新算配額，避免不同變體選出的
總檔數不一樣，見開發追蹤§3.8的code review記錄）。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8，同系列腳本已記過的規則）：
    PYTHONIOENCODING=utf-8 python -m app._design_test_anchored_exclude_v1
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH, resolve_holdings  # noqa: E402

from research import walkforward_matrix as WF  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _design_test_rolling_is6oos2 import IS_MONTHS, OOS_LEN, RATIO, ALLOCATION  # noqa: E402
from _design_test_quality_metric_variants import eligible_uids  # noqa: E402

TREE_KEY = "TW"
VARIANTS = ["baseline", "exclude_v1"]


def anchored_window_dates(is_end_offset: int, oos_months: int, tree_key: str) -> tuple[str, str, str, str]:
    """跟scheme A同一套anchored公式：IS起點固定在該樹的錨點，只有IS終點往後移。"""
    is_end = WF.SCHEME_BASE + is_end_offset - 1
    oos_start = WF.SCHEME_BASE + is_end_offset
    oos_end = WF.SCHEME_BASE + is_end_offset + oos_months - 1
    is_start = pd.Period(WF.ANCHOR_START[tree_key], "M")
    return str(is_start), str(is_end), str(oos_start), str(oos_end)


def run_one_window(tree_key: str, is_start: str, is_end: str, oos_start: str, oos_end: str,
                   window_no: int, months_long, meta_pool, f_combo_map, mktcap, md: MarketData,
                   idx: pd.DataFrame, log=print) -> list[dict]:
    log(f">> 建樹 IS {is_start}~{is_end}（window{window_no}，anchored）...")
    t0 = time.time()
    tree = WF.build_tree_for_window(tree_key, is_start, is_end, months_long, meta_pool, f_combo_map, log)
    log(f"   建樹耗時 {time.time()-t0:.0f}s")

    assign = tree["assign"]
    assign = assign[assign.tree_id == tree["tree_id"]] if "tree_id" in assign.columns else assign
    cmeta = tree["cluster_meta"]
    cmeta = cmeta[cmeta.level == WF.LEVEL] if "level" in cmeta.columns else cmeta

    uids = pd.Index(assign[WF.C.PK])
    wide_is = WF.S3._pivot_window(months_long, uids, is_start, is_end)
    w = months_long[months_long.strategy_uid.isin(set(uids))]
    w = w[(w.month >= pd.Period(oos_start, "M")) & (w.month <= pd.Period(oos_end, "M"))]
    wide_oos = w.pivot(index="strategy_uid", columns="month", values="ret")
    n_nan = int(wide_oos.isna().sum().sum())
    n_missing = len(set(uids) - set(wide_oos.index))
    if n_nan or n_missing:
        raise ValueError(f"[{tree_key} w{window_no}] OOS 資料不完整：{n_nan}個缺值、{n_missing}個策略無資料")

    cagr_is = WF._cagr_matrix(wide_is)
    mdd_is = WF._mdd_matrix(wide_is)
    quality_is_base = cagr_is / mdd_is.abs().replace(0, np.nan)
    n_uni = len(uids)

    corr_full = np.corrcoef(wide_is.to_numpy(dtype=np.float64))
    pos = pd.Series(range(len(wide_is.index)), index=wide_is.index)

    cluster_map = assign.set_index(WF.C.PK)[f"cluster_{WF.LEVEL}"]
    sizes_full = assign.groupby(f"cluster_{WF.LEVEL}").size()
    k = len(sizes_full)
    tot = WF.target_total(RATIO, n_uni, k)
    bench_cagr = WF._cagr(WF._portfolio_series(wide_is, list(uids)))

    rows = []
    for variant in VARIANTS:
        elig = eligible_uids(variant, assign, idx, cagr_is, bench_cagr)
        assign_v = assign[assign[WF.C.PK].isin(elig)]
        if assign_v.empty:
            log(f"   [{variant}] 沒有任何策略通過篩選，跳過")
            continue
        sizes_v = assign_v.groupby(f"cluster_{WF.LEVEL}").size()
        quota, n_capped = WF.allocate(sizes_v, tot, ALLOCATION)
        quality_v = quality_is_base.reindex(list(elig))
        a_members, n_bf = WF._pick_a(assign_v, cmeta, wide_is, quality_v, quota, corr_full, pos)
        if not a_members:
            log(f"   [{variant}] 選不出任何代表，跳過")
            continue
        if len(a_members) != tot:
            log(f"   [{variant}] 目標{tot}檔，實際只選出{len(a_members)}檔")
        perf = WF._evaluate(a_members, wide_is, wide_oos, cluster_map)

        sub = idx.reindex(a_members)
        v1_frac = float((sub.V == "v1").mean())
        avg_hold = float(sub.avg_holdings.mean())

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

        rows.append({"window_no": window_no, "is_start": is_start, "is_end": is_end,
                    "oos_start": oos_start, "oos_end": oos_end, "variant": variant,
                    "target_total": tot, "n_members": len(a_members),
                    "v1_frac": v1_frac, "avg_holdings": avg_hold,
                    "frac_top10pct_mktcap": frac_top10,
                    "is_cagr": perf["is_cagr"], "is_mdd": perf["is_mdd"], "is_sharpe": perf["is_sharpe"],
                    "oos_cagr": perf["oos_cagr"], "oos_mdd": perf["oos_mdd"], "oos_sharpe": perf["oos_sharpe"]})
        log(f"   [{variant}] n={len(a_members)} V1比例={v1_frac:.0%} 平均持股={avg_hold:.1f} "
           f"市值前10%={frac_top10:.1%} IS CAGR={perf['is_cagr']:+.2%} MDD={perf['is_mdd']:.2%} "
           f"｜OOS CAGR={perf['oos_cagr']:+.2%} MDD={perf['oos_mdd']:.2%}")
    return rows


def main():
    blocks = WF._blocks(WF.SCHEME_TOTAL_MONTHS, IS_MONTHS, OOS_LEN)
    print(">> 載入資料...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    md = MarketData(TREE_KEY, start="2000-01-01")
    mktcap = md.get_field("report:mktcap")
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    all_rows = []
    for i, (off, L) in enumerate(blocks, 1):
        s, e, os_, oe = anchored_window_dates(off, L, TREE_KEY)
        rows = run_one_window(TREE_KEY, s, e, os_, oe, i, months_long, meta_pool, f_combo_map, mktcap, md, idx)
        all_rows += rows

    out = pd.DataFrame(all_rows)
    out_path = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "anchored_exclude_v1_TW.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}")

    print("\n=== 逐變體平均（6窗，anchored）===")
    summary = out.groupby("variant").agg(
        mean_n_members=("n_members", "mean"), mean_v1_frac=("v1_frac", "mean"),
        mean_avg_holdings=("avg_holdings", "mean"), mean_top10pct=("frac_top10pct_mktcap", "mean"),
        mean_is_cagr=("is_cagr", "mean"), mean_is_mdd=("is_mdd", "mean"),
        mean_oos_cagr=("oos_cagr", "mean"), mean_oos_mdd=("oos_mdd", "mean")).reset_index()
    print(summary.to_string(index=False))

    print("\n=== exclude_v1（anchored）逐窗明細 ===")
    ev1 = out[out.variant == "exclude_v1"].sort_values("window_no")
    print(ev1[["window_no", "is_end", "oos_start", "oos_end", "is_cagr", "is_mdd",
              "oos_cagr", "oos_mdd", "frac_top10pct_mktcap"]].to_string(index=False))


if __name__ == "__main__":
    main()
