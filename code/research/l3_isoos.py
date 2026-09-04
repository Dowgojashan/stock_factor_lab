# -*- coding: utf-8 -*-
"""H-25d · L3 細粒度互補性的 walk-forward 驗證（2026-09-04）

**補的是 H-25 留下的缺口**：H-25 查出「L1 沒有高互補配對是聚合效應」——把粒度
降到 L3（成員中位 20~26 檔），跨市場配對有 **62.6%** 達到高互補（成員>=20），
同市場只有 0.3%。**但那全部是用完整窗算的，從來沒有經過 out-of-sample 驗證。**

`H26_H27_walkforward矩陣結果.md` 的已知限制第 6 條就是這件事：
「本矩陣全部在 L1 群層級進行；H-25 發現的 L3 跨市場高互補未經 walk-forward 驗證」。

---------------------------------------------------------------------------
要回答的問題
---------------------------------------------------------------------------
老師對 H-13 的原話：「**in sample 都很低，可是 out sample 會不會還是很低**」——
那是在 L1 問的，`stage3_hrp_isoos.py` 也只做了 L1。本模組把同一個問題搬到 L3。

做法完全比照 H-11：**群定義只用 IS 窗決定並凍結，OOS 只用同一批群重算相關**，
不重新分群。然後比較同一對群在 IS 與 OOS 的互補判定是否一致。

  穩定率 = 「IS 判定為高互補」的配對中，OOS 仍為高互補的比例

⚠️ **只納入成員 >= MIN_MEMBERS 的群**（沿用 H-25 的門檻，圖表與統計口徑一致）。
⚠️ TW/US 樹只有同市場配對，是天然的對照組——若跨市場的穩定率明顯較高，
   「免費午餐藏在跨市場的細粒度」這個主張才站得住。

用法：
    cd code
    python -m research.l3_isoos                    # 全部樹與窗次
    python -m research.l3_isoos --trees XM         # 只跑跨市場（跨市場配對只有它有）
    python -m research.l3_isoos --schemes E H
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, paths
from . import stage3_hrp as S3
from .complementarity_granularity import SHORTLIST_MIN_MEMBERS as MIN_MEMBERS
from .walkforward_matrix import (_load_inputs, build_schemes, build_tree_for_window,
                                 window_dates)

TREES = ("TW", "US", "XM")
LEVEL = "L3"


def _cluster_reps(wide: pd.DataFrame, labels: pd.Series,
                  keep: pd.Index) -> tuple[pd.DataFrame, pd.Series]:
    """群代表序列（群×月）與每群市場。群代表＝成員報酬簡單平均，同 H-06/H-25 口徑。"""
    lab = labels.reindex(wide.index)
    reps = wide.groupby(lab.to_numpy()).mean().loc[keep]
    mkt = (pd.Series(wide.index, index=wide.index).str.split("::").str[0]
           .groupby(lab.to_numpy()).agg(lambda s: s.mode().iloc[0]).loc[keep])
    return reps, mkt


def _pair_frame(reps: pd.DataFrame, mkt: pd.Series) -> pd.DataFrame:
    """所有群對的相關係數 + same/cross 標記（向量化：一次算完 k×k 相關矩陣）。"""
    corr = np.corrcoef(reps.to_numpy(dtype=np.float64))
    ids = list(reps.index)
    iu = np.triu_indices(len(ids), 1)
    a = np.asarray(ids)[iu[0]]
    b = np.asarray(ids)[iu[1]]
    return pd.DataFrame({
        "a": a, "b": b, "corr": corr[iu],
        "pair_type": np.where(mkt.reindex(a).to_numpy() == mkt.reindex(b).to_numpy(),
                              "same", "cross"),
    })


def run_one_window(tree_key: str, srow: pd.Series, tree: dict,
                   months_long, log=print) -> list[dict]:
    """一個 (樹 × 窗次)：群定義由 IS 凍結，OOS 只重算同一批群的相關。"""
    is_start, is_end, oos_start, oos_end = window_dates(srow, tree_key)
    assign = tree["assign"]
    labels = assign.set_index(C.PK)[f"cluster_{LEVEL}"]
    uids = pd.Index(assign[C.PK])

    sizes = labels.value_counts()
    keep = pd.Index(sorted(sizes[sizes >= MIN_MEMBERS].index))
    if len(keep) < 2:
        raise AssertionError(f"[{tree_key}] 成員>={MIN_MEMBERS} 的 L3 群不足 2 個")

    wide_is = S3._pivot_window(months_long, uids, is_start, is_end)
    w = months_long[months_long.strategy_uid.isin(set(uids))]
    w = w[(w.month >= pd.Period(oos_start, "M")) & (w.month <= pd.Period(oos_end, "M"))]
    wide_oos = w.pivot(index="strategy_uid", columns="month", values="ret")
    if int(wide_oos.isna().sum().sum()) or len(set(uids) - set(wide_oos.index)):
        raise ValueError(f"[{tree_key}/{srow.scheme}/w{int(srow.window_no)}] "
                         f"OOS 窗資料不完整——DD-03 共同窗保證被打破")

    reps_is, mkt = _cluster_reps(wide_is, labels, keep)
    reps_oos, _ = _cluster_reps(wide_oos, labels, keep)
    pi, po = _pair_frame(reps_is, mkt), _pair_frame(reps_oos, mkt)
    assert (pi.a.to_numpy() == po.a.to_numpy()).all(), "IS/OOS 配對順序不一致"

    cut = C.COMPLEMENTARITY_CUTS["高"]
    df = pi.rename(columns={"corr": "corr_is"})
    df["corr_oos"] = po["corr"].to_numpy()
    df["hi_is"] = df.corr_is < cut
    df["hi_oos"] = df.corr_oos < cut

    rows = []
    for pt, g in df.groupby("pair_type", observed=True):
        n_hi_is = int(g.hi_is.sum())
        rows.append({
            "tree_key": tree_key, "scheme": srow.scheme, "window_no": int(srow.window_no),
            "is_start": is_start, "is_end": is_end,
            "oos_start": oos_start, "oos_end": oos_end,
            "n_clusters": len(keep), "pair_type": pt, "n_pairs": len(g),
            "is_pct_high": float(g.hi_is.mean()),
            "oos_pct_high": float(g.hi_oos.mean()),
            "corr_is_median": float(g.corr_is.median()),
            "corr_oos_median": float(g.corr_oos.median()),
            "n_is_high": n_hi_is,
            "n_is_high_stays_high": int((g.hi_is & g.hi_oos).sum()),
            "stability_rate": (float((g.hi_is & g.hi_oos).sum() / n_hi_is)
                               if n_hi_is else float("nan")),
        })
    log(f"  [{tree_key}/{srow.scheme}/w{int(srow.window_no)}] {len(keep)} 群｜"
        + "｜".join(f"{r['pair_type']} IS高{r['is_pct_high']:.1%}→OOS高"
                    f"{r['oos_pct_high']:.1%}(穩定{r['stability_rate']:.1%})"
                    for r in rows))
    return rows


def run(trees=TREES, schemes_filter=None, log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE0)
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE1 / "_marks")

    schemes = build_schemes()
    if schemes_filter:
        schemes = schemes[schemes.scheme.isin(schemes_filter)]
    months_long, meta, f_combo_map = _load_inputs()

    all_rows = []
    t0 = time.time()
    for tree_key in trees:
        # 快取 key 用實際日期（同 walkforward_matrix 的修正，避免重複建樹）
        wanted = {window_dates(r, tree_key)[:2] for _, r in schemes.iterrows()}
        log(f"[{tree_key}] 需建樹 {len(wanted)} 棵")
        cache = {}
        for is_start, is_end in wanted:
            tt = time.time()
            cache[(is_start, is_end)] = build_tree_for_window(
                tree_key, is_start, is_end, months_long, meta, f_combo_map, log)
            log(f"  建樹 IS {is_start}~{is_end}  {time.time()-tt:.0f}s")
        for _, srow in schemes.iterrows():
            s, e, _, _ = window_dates(srow, tree_key)
            all_rows += run_one_window(tree_key, srow, cache[(s, e)], months_long, log)
        log(f"[{tree_key}] 完成，累計 {time.time()-t0:.0f}s\n")

    df = pd.DataFrame(all_rows)
    for col in ("tree_key", "scheme", "pair_type"):
        df[col] = df[col].astype("category")
    C.validate(df, C.L3_ISOOS, strict_columns=True)
    log(f"✓ l3_isoos 契約通過（{len(df)} 列）")

    out_dir = paths.ROOT / "_analysis_outputs_robustness"
    p = out_dir / "l3_isoos.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "l3_isoos", out_dir / "_l3_isoos_manifest",
        inputs=[paths.STAGE1 / "returns_monthly.parquet",
               paths.STAGE1 / "returns_meta.parquet",
               paths.STAGE1 / "strategy_marks.parquet"],
        outputs=[p],
        params={"level": LEVEL, "min_members": MIN_MEMBERS,
               "complementarity_cuts": C.COMPLEMENTARITY_CUTS,
               "l3_target": S3.L3_TARGET},
        notes="H-25d：L3 細粒度互補性的 walk-forward 驗證。群定義由 IS 窗凍結，"
              "OOS 只重算同一批群的相關（同 H-11 做法）。補 H-25 只有全窗、"
              "無 OOS 驗證的缺口。",
    )
    log(f"→ {p}")
    return df


def _report(df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 88)
    log(f"H-25d · L3 細粒度互補性的 walk-forward 驗證（成員>={MIN_MEMBERS}）")
    log("=" * 88)
    g = df.groupby(["tree_key", "pair_type"], observed=True).agg(
        窗次數=("window_no", "size"), 群數=("n_clusters", "mean"),
        配對數=("n_pairs", "mean"),
        IS高互補=("is_pct_high", "mean"), OOS高互補=("oos_pct_high", "mean"),
        穩定率=("stability_rate", "mean"),
        IS相關中位=("corr_is_median", "mean"), OOS相關中位=("corr_oos_median", "mean"))
    log(g.round(4).to_string())
    log("")
    cross = df[df.pair_type == "cross"]
    same = df[df.pair_type == "same"]
    if len(cross):
        log(f"跨市場：IS 高互補 {cross.is_pct_high.mean():.1%} → "
            f"OOS 高互補 {cross.oos_pct_high.mean():.1%}"
            f"｜**穩定率 {cross.stability_rate.mean():.1%}**")
    if len(same):
        log(f"同市場：IS 高互補 {same.is_pct_high.mean():.1%} → "
            f"OOS 高互補 {same.oos_pct_high.mean():.1%}"
            f"｜穩定率 {same.stability_rate.mean():.1%}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.l3_isoos")
    ap.add_argument("--trees", nargs="+", default=list(TREES))
    ap.add_argument("--schemes", nargs="+")
    a = ap.parse_args(argv)
    df = run(trees=tuple(a.trees), schemes_filter=a.schemes)
    _report(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
