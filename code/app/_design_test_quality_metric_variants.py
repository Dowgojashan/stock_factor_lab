# -*- coding: utf-8 -*-
"""D③後續：測試三個「代表策略挑選品質指標」的改善方向，看能不能改善
`_design_test_rolling_is6oos2.py`發現的「訓練窗越短，V1（估值濾網）策略因為
持股數少（~17檔 vs V0的~82檔）、Calmar估計雜訊大，被過度挑中」問題。

背景（開發追蹤§3.6/3.7）：Phase4（`_analysis_outputs_phase4/Phase4_結果分析
與變體對照.md`）早就證實V1「在兩個市場都有害」（台股ΔCAGR中位數−1.96%、美股
−5.74%，「分散度崩潰」）。使用者提出三個測試方向：

  A. 持股數下限——沿用老師在Phase2就定下的`MIN_HOLDINGS=10`
     （`phase2_analyze.py`：「老師：選出的股票不能太少」），這裡額外多測一個
     更嚴格的門檻(30，接近V0策略的典型持股數量級)看敏感度
  B. 直接排除V1——既然Phase4全樣本已經證實V1整體有害
  C. 品質指標穩健化——使用者的洞察：「如果CAGR不夠高，MDD壓得再低也沒有用，
     因為你都不會賺錢，當然也不容易賠錢」，這正是Calmar比率（CAGR/|MDD|）在
     分子分母都趨近0時會失真的已知弱點。做法：比照Phase1-4既有的「CAGR>大盤
     基準」門檻（`phase4_analyze.py`:158最終候選池篩選就是這樣做的），這裡在
     **每個IS窗自己的範圍內**重新套用同一個邏輯——只是原本的門檻只在候選池
     生成階段做過一次（用全樣本2000-2025），代表策略挑選（H-10）階段完全沒有
     重新檢查，這裡補上

沿用`_design_test_rolling_is6oos2.py`同一套安全設計（不呼叫`WF.run()`、不動
研究部凍結產物），只測TW、k_mode=fixed、ratio=legacy、allocation=equal。

用法（cwd 必須是 code/；⚠️ Windows預設cp950主控台，本檔log含⚠️符號，一律加
PYTHONIOENCODING=utf-8，同`_design_test_rolling_is6oos2.py`已經記過的規則）：
    PYTHONIOENCODING=utf-8 python -m app._design_test_quality_metric_variants
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
from _design_test_rolling_is6oos2 import window_dates_rolling, IS_MONTHS, OOS_LEN, RATIO, ALLOCATION  # noqa: E402

TREE_KEY = "TW"
MIN_HOLDINGS_LOOSE = 10   # 沿用老師Phase2定下的門檻
MIN_HOLDINGS_STRICT = 30  # 額外測一個更嚴格的，接近V0典型持股量級

VARIANTS = ["baseline", "holdings_ge10", "holdings_ge30", "exclude_v1", "cagr_gt_bench", "combo_best"]


def eligible_uids(variant: str, assign: pd.DataFrame, idx: pd.DataFrame,
                  cagr_is: pd.Series, bench_cagr: float) -> set[str]:
    """回傳這個變體允許進入代表挑選的策略uid集合（其餘視同不存在，被H-10略過）。"""
    all_uids = set(assign[WF.C.PK])
    meta = idx.reindex(list(all_uids))
    if variant == "baseline":
        return all_uids
    if variant == "holdings_ge10":
        return set(meta[meta.avg_holdings >= MIN_HOLDINGS_LOOSE].index)
    if variant == "holdings_ge30":
        return set(meta[meta.avg_holdings >= MIN_HOLDINGS_STRICT].index)
    if variant == "exclude_v1":
        return set(meta[meta.V != "v1"].index)
    if variant == "cagr_gt_bench":
        ok = cagr_is.reindex(list(all_uids)) > bench_cagr
        return set(ok[ok].index)
    if variant == "combo_best":
        ok_hold = meta[meta.avg_holdings >= MIN_HOLDINGS_STRICT].index
        ok_cagr = cagr_is.reindex(list(all_uids)) > bench_cagr
        ok_cagr = set(ok_cagr[ok_cagr].index)
        return set(ok_hold) & ok_cagr
    raise ValueError(variant)


def run_window_all_variants(tree_key: str, is_start: str, is_end: str, oos_start: str, oos_end: str,
                            window_no: int, months_long, meta_pool, f_combo_map, mktcap, md: MarketData,
                            idx: pd.DataFrame, log=print) -> list[dict]:
    log(f">> 建樹 IS {is_start}~{is_end}（window{window_no}）...")
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
    # 🔴 quota 不可以只算一次、跨變體重用——2026-09-23 code review 抓到：某個變體
    # 把某個群的候選人全部篩掉後，那個群配到的名額會被pandas groupby悄悄吃掉、
    # 不會重分配給其他群，導致不同變體選出的**總檔數不一樣**，變成拿「篩選後
    # 選了幾檔」跟「品質有沒有變好」混在一起比較，跟`allocate()`自己docstring
    # 講的「兩種分配方式配出的總量必須完全相等，否則無法歸因」是同一個陷阱。
    # 修法：sizes改用「篩選後還剩多少候選人」逐變體重算，讓allocate()自己的
    # 「小群不足配額時給滿、剩餘額度按其餘群大小比例重分配」邏輯去正確處理
    # 被篩空的群，tot維持用原始k算出的同一個目標值（這樣才是同一把尺）。

    # B_all 當這個IS窗自己的基準（給cagr_gt_bench變體用）
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
            log(f"   [{variant}] ⚠️ 目標{tot}檔，實際只選出{len(a_members)}檔"
               f"（篩選後候選人不足以填滿配額，記錄下來不隱藏）")
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
                    # 🔴 IS/OOS績效都要存，不能只留OOS——使用者要看「每一窗IS跟OOS
                    # 表現怎麼樣」，原本只存了oos_cagr/oos_mdd，_evaluate()其實已經
                    # 算好is_cagr/is_mdd/is_sharpe/oos_sharpe，只是沒存下來，這次補上
                    "is_cagr": perf["is_cagr"], "is_mdd": perf["is_mdd"], "is_sharpe": perf["is_sharpe"],
                    "oos_cagr": perf["oos_cagr"], "oos_mdd": perf["oos_mdd"], "oos_sharpe": perf["oos_sharpe"]})
        log(f"   [{variant}] n={len(a_members)} V1比例={v1_frac:.0%} 平均持股={avg_hold:.1f} "
           f"市值前10%={frac_top10:.1%} IS CAGR={perf['is_cagr']:+.2%} MDD={perf['is_mdd']:.2%} "
           f"｜OOS CAGR={perf['oos_cagr']:+.2%} MDD={perf['oos_mdd']:.2%}")
    return rows


def main():
    print(">> 建立rolling方案窗次表（沿用IS6/OOS2）...")
    blocks = WF._blocks(WF.SCHEME_TOTAL_MONTHS, IS_MONTHS, OOS_LEN)

    print(">> 載入資料...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    md = MarketData(TREE_KEY, start="2000-01-01")
    mktcap = md.get_field("report:mktcap")
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    all_rows = []
    for i, (off, L) in enumerate(blocks, 1):
        s, e, os_, oe = window_dates_rolling(off, L, TREE_KEY)
        rows = run_window_all_variants(TREE_KEY, s, e, os_, oe, i, months_long, meta_pool,
                                       f_combo_map, mktcap, md, idx)
        all_rows += rows

    out = pd.DataFrame(all_rows)
    out_path = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "quality_metric_variants_TW.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}")

    print("\n=== 逐變體平均（6窗）===")
    # 🔴 mean_n_members／mean_target_total 一定要跟其他指標一起印出來，不能只存
    # 在CSV裡不顯示——不然變體之間選出的總檔數不一樣（見上面quota修正的說明），
    # 會被誤讀成「品質指標比較好」，實際上只是選得比較少/比較集中（code review
    # 2026-09-23明確要求：這兩欄不能從摘要漏掉）。
    summary = out.groupby("variant").agg(
        mean_target_total=("target_total", "mean"), mean_n_members=("n_members", "mean"),
        mean_v1_frac=("v1_frac", "mean"), mean_avg_holdings=("avg_holdings", "mean"),
        mean_top10pct=("frac_top10pct_mktcap", "mean"),
        mean_is_cagr=("is_cagr", "mean"), mean_is_mdd=("is_mdd", "mean"),
        mean_oos_cagr=("oos_cagr", "mean"), mean_oos_mdd=("oos_mdd", "mean")).reset_index()
    print(summary.to_string(index=False))

    print("\n=== exclude_v1 逐窗 IS/OOS 明細（使用者要求）===")
    ev1 = out[out.variant == "exclude_v1"].sort_values("window_no")
    print(ev1[["window_no", "is_start", "is_end", "oos_start", "oos_end",
              "is_cagr", "is_mdd", "is_sharpe", "oos_cagr", "oos_mdd", "oos_sharpe"]].to_string(index=False))


if __name__ == "__main__":
    main()
