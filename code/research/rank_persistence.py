# -*- coding: utf-8 -*-
"""M-12 · IS→OOS 品質排序的持續性（rank IC）（2026-09-07）

🔴 **這支腳本要補的洞**：第五章所有結論——精選勝過狂灑、legacy 最好、
比例越高越差——**共同機制都是「IS 的品質排序在 OOS 還剩多少資訊」**，
但這個量**從來沒有被直接算出來過**。查證：全 `research/` 一個檔案都沒出現 `spearman`。
目前只能從組合層的勝率反推。

**這也是 H-28 目前最缺的東西**：M-01 與 M-03/M-03b 提供的全是否定性材料
（HRP 沒買到報酬、沒買到 alpha 來源、分群依據換成隨機也差不多），
rank IC 是唯一能提供**正面機制敘述**的量。

---------------------------------------------------------------------------
三個切面（都是同一份資料的 groupby，成本近零）
---------------------------------------------------------------------------
① **rank IC vs `oos_len_months`**：持續性隨評估期拉長是衰減還是穩定
② **分位版**：IS 品質前 1%/3%/5%/10% 的策略，在 OOS 的**平均分位**是多少。
   0.5 = 完全沒有資訊。這直接對應「比例越高越差」的機制——若前 1% 的 OOS 分位
   明顯高於前 10%，「精選有效」就有了**策略層**的直接證據，
   不必只靠 backfill 率旁證。
③ **rank IC vs 窗次序**：持續性是否隨時間衰減（結構性變化的證據）

---------------------------------------------------------------------------
兩種品質指標都算（原稿只要求 Calmar，這裡多做一個）
---------------------------------------------------------------------------
  `calmar`  CAGR/|MDD| —— `A_hrp` 與 `E_top_calmar` 用的排序
  `cagr`    純 CAGR    —— `D_top_cagr` 用的排序

兩者並排能直接回答「為什麼 E 贏過 D」：若 Calmar 的 rank IC 明顯高於 CAGR，
那 D→E 的差距就有了機制解釋，而不只是實測到的現象。這對 M-11 有直接價值。

---------------------------------------------------------------------------
⚠️ 判讀限制（依 M-10 的教訓，寫死在這裡）
---------------------------------------------------------------------------
🔴 **45 個窗次不是 45 個獨立樣本。** anchored 的 IS 是巢狀的（第 2 窗的 IS =
第 1 窗的 IS+OOS），M-10 實測相鄰窗選股的 Jaccard 0.372、全窗核心佔比 0.357。
故本模組**只報分布與逐方案結果，不對 45 個值做任何檢定**。

⚠️ 本模組**不需要建樹**（rank IC 與分群無關，只要報酬矩陣），
完全繞過 `build_tree_for_window`。

用法：
    cd code
    python -m research.rank_persistence
    python -m research.rank_persistence --trees TW --schemes A
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from . import contracts as C
from . import freeze, paths
from . import stage3_hrp as S3
from .four_group_control import _cagr_matrix, _mdd_matrix
from .walkforward_matrix import (TREES, _load_inputs, build_schemes, window_dates)

#: 分位版要看的「IS 前 q%」。沿用全案共用的 RATIO_GRID 數值，讓結果可以跟
#: walk-forward 矩陣的比例維度直接對照。
TOP_QS = (0.01, 0.03, 0.05, 0.10)
QUALITY_METRICS = ("calmar", "cagr")


def _quality(wide: pd.DataFrame, metric: str) -> pd.Series:
    """該窗的策略品質分數。口徑與 `walkforward_matrix.run_one_window` 完全一致。"""
    cagr = _cagr_matrix(wide)
    if metric == "cagr":
        return cagr
    mdd = _mdd_matrix(wide)
    # ⚠️ MDD=0 會讓 Calmar 變 inf；`run_one_window` 用同一條 replace(0, nan)。
    return cagr / mdd.abs().replace(0, np.nan)


def _window_frames(tree_key: str, is_start: str, is_end: str,
                   oos_start: str, oos_end: str, months_long, meta):
    """該窗的 (IS 報酬矩陣, OOS 報酬矩陣)。與 `run_one_window` 同一套宇宙規則。"""
    uids = S3._tree_universe(tree_key, is_start, meta)
    wide_is = S3._pivot_window(months_long, uids, is_start, is_end)
    w = months_long[months_long.strategy_uid.isin(set(uids))]
    w = w[(w.month >= pd.Period(oos_start, "M")) & (w.month <= pd.Period(oos_end, "M"))]
    wide_oos = w.pivot(index="strategy_uid", columns="month", values="ret")
    n_nan = int(wide_oos.isna().sum().sum())
    n_missing = len(set(uids) - set(wide_oos.index))
    if n_nan or n_missing:
        raise ValueError(
            f"[{tree_key}] OOS 窗 {oos_start}~{oos_end} 資料不完整："
            f"{n_nan} 個缺值、{n_missing} 個策略無資料——DD-03 共同窗保證被打破")
    return wide_is, wide_oos.loc[wide_is.index]


def analyse_window(tree_key: str, srow: pd.Series, months_long, meta) -> list[dict]:
    is_start, is_end, oos_start, oos_end = window_dates(srow, tree_key)
    wide_is, wide_oos = _window_frames(tree_key, is_start, is_end,
                                       oos_start, oos_end, months_long, meta)
    rows = []
    for metric in QUALITY_METRICS:
        q_is = _quality(wide_is, metric)
        q_oos = _quality(wide_oos, metric)
        both = pd.concat([q_is.rename("is"), q_oos.rename("oos")], axis=1).dropna()
        if len(both) < 10:
            raise ValueError(f"[{tree_key}/{srow.scheme}] 可用配對只有 {len(both)} 檔")

        rho, p_rho = stats.spearmanr(both["is"], both["oos"])
        r_pear = float(np.corrcoef(both["is"], both["oos"])[0, 1])

        # 🔴 分位版：把 OOS 品質換成**分位**（0~1，越大越好），再看 IS 前 q% 的人
        # 在 OOS 平均落在哪個分位。0.5 = 完全沒有資訊，1.0 = 完美預測。
        oos_pct = both["oos"].rank(pct=True)
        order = both["is"].sort_values(ascending=False).index
        rec = {"tree_key": tree_key, "scheme": srow.scheme,
               "window_no": int(srow.window_no), "quality_metric": metric,
               "mode": srow["mode"], "min_is_months": int(srow.min_is_months),
               "oos_len_months": int(srow.oos_len_months),
               "is_start": is_start, "is_end": is_end,
               "oos_start": oos_start, "oos_end": oos_end,
               "n_is_months": wide_is.shape[1], "n_oos_months": wide_oos.shape[1],
               "n_universe": len(wide_is), "n_pairs": int(len(both)),
               "spearman": float(rho), "spearman_p": float(p_rho),
               "pearson": r_pear}
        for q in TOP_QS:
            n_top = max(1, int(round(len(order) * q)))
            rec[f"oos_pct_top{int(q*100):02d}"] = float(oos_pct.loc[order[:n_top]].mean())
        rows.append(rec)
    return rows


def build(trees=TREES, schemes_filter=None, log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE0)
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE1 / "_marks")

    schemes = build_schemes()
    if schemes_filter:
        schemes = schemes[schemes.scheme.isin(schemes_filter)]
    months_long, meta, _ = _load_inputs()
    log(f"窗口方案 {schemes.scheme.nunique()} 個｜總窗次 {len(schemes)}")

    rows = []
    t0 = time.time()
    for tree_key in trees:
        for _, srow in schemes.iterrows():
            rows += analyse_window(tree_key, srow, months_long, meta)
        log(f"  [{tree_key}] 完成｜累計 {time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    for c in ("tree_key", "scheme", "quality_metric", "mode"):
        df[c] = df[c].astype("category")
    return df[C.RANK_PERSISTENCE.names]


def run(trees=TREES, schemes_filter=None, log=print) -> pd.DataFrame:
    df = build(trees=trees, schemes_filter=schemes_filter, log=log)
    C.validate(df, C.RANK_PERSISTENCE, strict_columns=True)
    log(f"✓ rank_persistence 契約通過（{len(df)} 列）")

    out_dir = paths.ROOT / "_analysis_outputs_robustness"
    p = out_dir / "rank_persistence.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "rank_persistence", out_dir / "_rank_persistence_manifest",
        inputs=[paths.STAGE0 / "candidate_index.parquet",
                paths.STAGE1 / "returns_monthly.parquet",
                paths.STAGE1 / "returns_meta.parquet",
                paths.STAGE1 / "strategy_marks.parquet"],
        outputs=[p],
        params={"top_qs": list(TOP_QS), "quality_metrics": list(QUALITY_METRICS),
                "quality_rule": "與 walkforward_matrix.run_one_window 同一口徑"},
        notes="M-12：IS→OOS 品質排序的持續性（Spearman rank IC）+ 分位版。"
              "不建樹。⚠️ 45 個窗互相重疊（巢狀 IS），不是獨立樣本，"
              "只報分布與逐方案結果，不做檢定。",
    )
    log(f"→ {p}")
    return df


def _report(df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 96)
    log("M-12 · IS→OOS 品質排序的持續性（rank IC）")
    log("=" * 96)

    log("\n【一】整體 rank IC（Spearman），逐樹 × 品質指標")
    log(f"  {'樹':<6}{'品質指標':<10}{'窗數':>6}{'中位':>9}{'平均':>9}"
        f"{'最小':>9}{'最大':>9}{'為正比例':>10}{'Pearson中位':>12}")
    for (t, m), g in df.groupby(["tree_key", "quality_metric"], observed=True):
        log(f"  {t:<6}{m:<10}{len(g):>6}{g.spearman.median():>9.4f}"
            f"{g.spearman.mean():>9.4f}{g.spearman.min():>9.4f}{g.spearman.max():>9.4f}"
            f"{(g.spearman > 0).mean():>10.1%}{g.pearson.median():>12.4f}")

    log("\n【二】rank IC vs OOS 窗長（持續性隨評估期拉長會不會衰減）")
    log(f"  {'樹':<6}{'品質指標':<10}" + "".join(f"{L:>12}" for L in (24, 36, 48, 60)))
    for (t, m), g in df.groupby(["tree_key", "quality_metric"], observed=True):
        cells = []
        for L in (24, 36, 48, 60):
            gg = g[g.n_oos_months == L]
            cells.append(f"{gg.spearman.median():>8.4f}(n{len(gg)})" if len(gg) else f"{'—':>12}")
        log(f"  {t:<6}{m:<10}" + "".join(f"{c:>12}" for c in cells))

    log("\n【三】🔴 分位版：IS 前 q% 的策略在 OOS 的平均分位（0.5 = 毫無資訊）")
    log(f"  {'樹':<6}{'品質指標':<10}{'前1%':>9}{'前3%':>9}{'前5%':>9}{'前10%':>9}"
        f"{'  單調遞減?':>12}")
    for (t, m), g in df.groupby(["tree_key", "quality_metric"], observed=True):
        v = [g[f"oos_pct_top{q:02d}"].median() for q in (1, 3, 5, 10)]
        mono = "✅" if all(v[i] >= v[i + 1] for i in range(3)) else "❌"
        log(f"  {t:<6}{m:<10}" + "".join(f"{x:>9.4f}" for x in v) + f"{mono:>12}")
    log("  判讀：前 1% 的 OOS 分位若明顯高於前 10%，「精選有效」就有了**策略層**")
    log("        的直接證據（不必只靠組合層勝率或 backfill 率旁證）。")

    log("\n【四】rank IC vs 窗次序（持續性是否隨時間衰減 = 結構性變化的證據）")
    log(f"  {'樹':<6}{'品質指標':<10}" + "".join(f"{f'w{i}':>10}" for i in range(1, 7)))
    for (t, m), g in df.groupby(["tree_key", "quality_metric"], observed=True):
        cells = []
        for i in range(1, 7):
            gg = g[g.window_no == i]
            cells.append(f"{gg.spearman.median():>10.4f}" if len(gg) else f"{'—':>10}")
        log(f"  {t:<6}{m:<10}" + "".join(cells))

    log("\n⚠️ **45 個窗次不是 45 個獨立樣本**——anchored 的 IS 是巢狀的")
    log("   （M-10 實測相鄰窗選股 Jaccard 0.372、全窗核心佔比 0.357）。")
    log("   本表只報分布與逐方案結果，**不對這些值做任何檢定**。")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.rank_persistence")
    ap.add_argument("--trees", nargs="+", default=list(TREES))
    ap.add_argument("--schemes", nargs="+")
    a = ap.parse_args(argv)
    _report(run(trees=tuple(a.trees), schemes_filter=a.schemes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
