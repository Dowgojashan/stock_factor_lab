# -*- coding: utf-8 -*-
"""D③ Anchor→Rolling：IS6年/OOS2年的完整rolling走一遍（2007~2025），
跟既有anchored結果比較**績效**跟**候選池大小型股組成**。

背景：9/22老師點名「候選策略偏小型股，懷疑是anchor（IS固定2007-2023）造成的」，
之前H-26/H-27已經測過一個rolling對照組（IS=8年/OOS=3年，代號R），結果是rolling
整體績效明顯比anchored差（OOS Calmar勝率83.3%墊底，且是45個方案裡唯一虧損的格
子），但**沒有檢查過候選池的大小型股組成**，只看了報酬/MDD這些績效指標。這次
換一個IS:OOS比例（6年/2年=3:1，比原本8:3更貼近業界walk-forward慣例，且OOS=2年
可以讓最後一窗OOS落在2024-2025，直接對應這次的核心關切），**同時**檢查績效
與候選池組成兩件事，不要像上次一樣漏掉組成這塊。

🔴 安全設計（不動`research.walkforward_matrix`的共用狀態）：
  - **不呼叫`WF.run()`**——那會把結果寫進`_analysis_outputs_robustness/
    walkforward_matrix_detail.csv`等共用檔案，覆蓋掉H-26/H-27已經凍結、被
    這個專案其他一堆腳本依賴的45方案結果（尤其`walkforward_members.parquet`
    的window4/legacy/equal/A_hrp這把鑰匙，這次session已經被至少4支腳本讀過）。
  - 這支腳本**只匯入`walkforward_matrix`裡的純函式**（`_blocks`／
    `build_tree_for_window`／`_pick_a`／`_evaluate`／`allocate`／`target_total`），
    自己管理輸出路徑，寫到`_analysis_outputs_applayer/`，跟研究部的凍結產物
    完全分開。
  - **只用k_mode="fixed"**（不用`silhouette_is`）——後者需要`k_stability.csv`
    先涵蓋新窗次的(tree_key, is_start, is_end)組合，這次沒有重跑`k_stability.py`
    補那批窗次，避免範圍擴大成另一個大工程。跟anchored side比較時，anchored
    side也只取k_mode="fixed"的既有結果，兩邊口徑一致。
  - **只用ratio=legacy、allocation=equal、group∈{A_hrp,B_all}**——跟這個專案
    從D50一路到這次session反覆使用的主線設定（window4/legacy/equal/A_hrp那把
    鑰匙）同一套，不是意外挑的，是刻意跟既有大量分析保持可比較。

範圍限定：**只測TW**（老師這次點名的是台股候選池偏小型股，不是美股/XM），
如果之後要擴大到US/XM，是下一步，不是這次一次做完。

用法（cwd 必須是 code/；⚠️ Windows預設cp950主控台，本檔log含⚠️符號，一律加
PYTHONIOENCODING=utf-8，否則跑到早期窗次mktcap缺資料的分支會UnicodeEncodeError
中斷——CLAUDE.md已對整個研究部腳本記過這條規則，`_design_test_hot_segment.py`
已經補過同一個提醒，這裡同樣適用，2026-09-23 code review 抓到這裡漏補）：
    PYTHONIOENCODING=utf-8 python -m app._design_test_rolling_is6oos2
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
from research import paths  # noqa: E402

TREE_KEY = "TW"
IS_MONTHS = 72     # 6年
OOS_LEN = 24        # 2年
RATIO = "legacy"
ALLOCATION = "equal"
K_MODE = "fixed"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"


def window_dates_rolling(is_end_offset: int, oos_months: int, tree_key: str) -> tuple[str, str, str, str]:
    """跟`WF.window_dates()`同一套rolling公式，但不依賴 schemes DataFrame 的列結構，
    直接吃offset/oos_months——因為這次的方案不進`WF.build_schemes()`的共用表。"""
    is_end = WF.SCHEME_BASE + is_end_offset - 1
    oos_start = WF.SCHEME_BASE + is_end_offset
    oos_end = WF.SCHEME_BASE + is_end_offset + oos_months - 1
    anchor = pd.Period(WF.ANCHOR_START[tree_key], "M")
    want = is_end - IS_MONTHS + 1
    is_start = max(want, anchor)
    return str(is_start), str(is_end), str(oos_start), str(oos_end)


def run_one_rolling_window(tree_key: str, is_start: str, is_end: str, oos_start: str, oos_end: str,
                           window_no: int, months_long, meta, f_combo_map, mktcap, md: MarketData,
                           idx: pd.DataFrame, log=print) -> dict:
    log(f">> 建樹 IS {is_start}~{is_end}（window{window_no}）...")
    t0 = time.time()
    tree = WF.build_tree_for_window(tree_key, is_start, is_end, months_long, meta, f_combo_map, log)
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
        raise ValueError(f"[{tree_key} w{window_no}] OOS {oos_start}~{oos_end} 資料不完整："
                         f"{n_nan}個缺值、{n_missing}個策略完全無資料")

    cagr_is = WF._cagr_matrix(wide_is)
    mdd_is = WF._mdd_matrix(wide_is)
    quality_is = cagr_is / mdd_is.abs().replace(0, np.nan)
    n_uni = len(uids)

    corr_full = np.corrcoef(wide_is.to_numpy(dtype=np.float64))
    pos = pd.Series(range(len(wide_is.index)), index=wide_is.index)

    cluster_map = assign.set_index(WF.C.PK)[f"cluster_{WF.LEVEL}"]
    sizes = assign.groupby(f"cluster_{WF.LEVEL}").size()
    k = len(sizes)

    tot = WF.target_total(RATIO, n_uni, k)
    quota, n_capped = WF.allocate(sizes, tot, ALLOCATION)
    a_members, n_bf = WF._pick_a(assign, cmeta, wide_is, quality_is, quota, corr_full, pos)

    perf_a = WF._evaluate(a_members, wide_is, wide_oos, cluster_map)
    perf_b = WF._evaluate(list(uids), wide_is, wide_oos, cluster_map)

    # 候選池大小型股組成：把A_hrp成員解析到股票層，查每檔股票在IS窗末端的市值百分位
    print(f"   解析{len(a_members)}檔代表策略的股票層持股...")
    all_syms: set[str] = set()
    for uid in a_members:
        try:
            syms, _ = resolve_holdings(md, idx.loc[uid], is_end)
        except (RuntimeError, KeyError, ValueError):
            continue
        all_syms |= set(syms)

    valid_mk_dates = mktcap.index[mktcap.index <= pd.Timestamp(is_end)]
    if len(valid_mk_dates) == 0:
        # 🔴 這個IS窗末端還沒有任何市值資料（早期窗次可能發生）——誠實回傳
        # 全部NaN，不可以讓.max()對空index噴NaT、間接KeyError把整個多窗次run炸掉
        # （2026-09-23 code review 抓到，同一類問題之前在Hot Segment腳本修過一次）
        log(f"   ⚠️ IS end {is_end} 之前沒有任何市值資料，組成統計標記為無法計算")
        composition = {"n_stocks_held": len(all_syms), "n_stocks_with_mktcap": 0,
                       "median_mktcap_pct": float("nan"), "frac_top10pct_mktcap": float("nan")}
    else:
        mk_row = mktcap.loc[valid_mk_dates.max()]
        mk_universe = mk_row.reindex(md.common).dropna()
        pct_rank = mk_universe.rank(pct=True)  # 0~1，1=全市場最大市值
        held_pct = pct_rank.reindex(list(all_syms)).dropna()
        top10_pct = 0.90   # 市值前10%＝百分位>=0.90
        composition = {
            "n_stocks_held": len(all_syms),
            "n_stocks_with_mktcap": len(held_pct),
            "median_mktcap_pct": float(held_pct.median()) if len(held_pct) else float("nan"),
            "frac_top10pct_mktcap": float((held_pct >= top10_pct).mean()) if len(held_pct) else float("nan"),
        }

    return {"tree_key": tree_key, "mode": "rolling_6_2", "window_no": window_no,
           "is_start": is_start, "is_end": is_end, "oos_start": oos_start, "oos_end": oos_end,
           "n_universe": n_uni, "n_clusters": k, "target_total": tot,
           "n_capped_clusters": n_capped, "n_backfilled": n_bf,
           **{f"a_{k2}": v for k2, v in perf_a.items()},
           **{f"b_{k2}": v for k2, v in perf_b.items()},
           **composition}


def main():
    print(f">> 建立rolling方案窗次表（IS={IS_MONTHS}月/OOS={OOS_LEN}月）...")
    blocks = WF._blocks(WF.SCHEME_TOTAL_MONTHS, IS_MONTHS, OOS_LEN)
    print(f"共{len(blocks)}窗：")
    for i, (off, L) in enumerate(blocks, 1):
        s, e, os_, oe = window_dates_rolling(off, L, TREE_KEY)
        print(f"  window{i}：IS {s}~{e}｜OOS {os_}~{oe}")

    print("\n>> 載入資料...")
    months_long, meta, f_combo_map = WF._load_inputs()
    md = MarketData(TREE_KEY, start="2000-01-01")   # 建一次、6個窗共用，不在迴圈內重建
    mktcap = md.get_field("report:mktcap")
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")  # 同上，建一次

    rows = []
    for i, (off, L) in enumerate(blocks, 1):
        s, e, os_, oe = window_dates_rolling(off, L, TREE_KEY)
        row = run_one_rolling_window(TREE_KEY, s, e, os_, oe, i, months_long, meta, f_combo_map, mktcap, md, idx)
        rows.append(row)
        print(f"   window{i} A_hrp: OOS CAGR={row['a_oos_cagr']:+.2%} MDD={row['a_oos_mdd']:.2%} "
             f"｜B_all: OOS CAGR={row['b_oos_cagr']:+.2%}｜市值前10%佔比={row['frac_top10pct_mktcap']:.1%}"
             f"（{row['n_stocks_with_mktcap']}檔有市值資料／{row['n_stocks_held']}檔持股）")

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / f"rolling_is{IS_MONTHS}oos{OOS_LEN}_{TREE_KEY}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}")

    print("\n=== 跟既有anchored結果比較（讀`_analysis_outputs_robustness/"
         "walkforward_matrix_detail.csv`，k_mode=fixed／legacy／equal，不重算）===")
    anc = pd.read_csv(paths.ROOT / "_analysis_outputs_robustness" / "walkforward_matrix_detail.csv")
    anc = anc[(anc.tree_key == TREE_KEY) & (anc.mode == "anchored") & (anc.k_mode == K_MODE)
             & (anc.ratio == RATIO) & (anc.allocation == ALLOCATION) & (anc.group == "A_hrp")]
    print(f"anchored（12方案全部窗次）A_hrp OOS CAGR：mean={anc.oos_cagr.mean():+.2%}｜"
         f"median={anc.oos_cagr.median():+.2%}｜n={len(anc)}")
    print(f"rolling(6:2) A_hrp OOS CAGR：mean={out['a_oos_cagr'].mean():+.2%}｜"
         f"median={out['a_oos_cagr'].median():+.2%}｜n={len(out)}")

    r_win = pd.read_csv(paths.ROOT / "_analysis_outputs_robustness" / "walkforward_matrix_detail.csv")
    r_win = r_win[(r_win.tree_key == TREE_KEY) & (r_win.mode == "rolling") & (r_win.k_mode == K_MODE)
                 & (r_win.ratio == RATIO) & (r_win.allocation == ALLOCATION) & (r_win.group == "A_hrp")]
    if len(r_win):
        print(f"既有rolling(8:3, scheme R) A_hrp OOS CAGR：mean={r_win.oos_cagr.mean():+.2%}｜n={len(r_win)}")


if __name__ == "__main__":
    main()
