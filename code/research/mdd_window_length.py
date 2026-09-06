# -*- coding: utf-8 -*-
"""M-09 · MDD 顯著性按 OOS 窗長分層（2026-09-05）

🔴 **這支腳本要解決的問題**：H-26c 報出「Calmar 13/13 方案顯著、CAGR 13/13 顯著，
**但 MDD 只有 8/13**」，並在 `H26_H27_walkforward矩陣結果.md` §4.2 留下一句
沒有答案的判讀：

> 「I/J/K/L 是單位數太少（各 6~9）檢力不足，但 R（rolling）的 66.7%、p=0.194
>   是真的比較弱。」

**「檢力不足」與「效果真的弱」在只看 p 值時無法區分**——p 值同時混了效果量與樣本數。
本模組把兩者拆開，用三件事回答：

  ① **檢力下限**：給定該方案的單位數 n，雙尾二項檢定要達到 p<0.05
     **至少**要贏幾場？把它換算成「可偵測勝率下限」，跟實測勝率並排。
     實測勝率 ≥ 下限卻不顯著 → 不可能；實測 < 下限 → **無法分辨**（檢力不足）；
     實測遠低於下限 → **效果真的弱**。
  ② **效果量**：MDD 差距的實際大小（百分點），以及單位層級的 `win_share`
     （該單位 20 種設定中 A 勝出的比例）。效果量**不受樣本數影響**，
     是「真的弱不弱」的直接證據。
  ③ **按 OOS 窗長分層**：24／36／48 個月三組各自的勝率與效果量。
     若短窗雜訊是主因，效果量應**隨窗長變穩定**（變異數下降、均值不變）；
     若各窗長的效果量都接近 0，那就是**真的弱**。

---------------------------------------------------------------------------
⚠️ 為什麼不把同窗長的方案合併做檢定
---------------------------------------------------------------------------
oos_len=24 有 A(6窗)／D(5)／G(4)／J(3) 四個方案，看起來合併就有 18 窗 × 3 樹。
**但那 18 個窗互相重疊**——四個方案用不同的 min_is 起點去切同一條時間軸，
`2013-01~2014-12` 這段會同時出現在好幾個方案裡。合併做二項檢定等於把同一段歷史
數好幾次，正是 H-26c 建立整套「只在方案內部檢定」規則要避免的事。

故本模組**只做分層描述與檢力分析，不產生任何新的 p 值**。
唯一的檢定仍是 H-26c 那份逐方案的二項檢定，這裡只是把它按窗長重新排列。

用法：
    cd code
    python -m research.mdd_window_length
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

from . import contracts as C
from . import freeze, paths

OPPONENTS = ("B_all", "D_top_cagr", "E_top_calmar")
METRICS = ("cagr", "mdd", "calmar")
ALPHA = 0.05


def min_wins_for_significance(n: int, alpha: float = ALPHA) -> int | None:
    """雙尾二項檢定（H0: p=0.5）在 n 個單位下，**至少**要贏幾場才可能 p<alpha。

    回傳 None 表示**無論全勝都達不到顯著**（n 太小）——這正是 I/K/L 這類方案的處境，
    是「檢力不足」最直接的證據形式。
    """
    for w in range(n // 2 + 1, n + 1):
        if stats.binomtest(w, n, 0.5, alternative="two-sided").pvalue < alpha:
            return w
    return None


def _load_units() -> pd.DataFrame:
    d = paths.ROOT / "_analysis_outputs_robustness"
    p = d / "walkforward_significance_units.csv"
    if not p.exists():
        raise FileNotFoundError("請先執行 `python -m research.walkforward_significance`")
    freeze.verify_inputs(d / "_walkforward_significance_manifest")
    return pd.read_csv(p)


def _load_detail() -> pd.DataFrame:
    d = paths.ROOT / "_analysis_outputs_robustness"
    freeze.verify_inputs(d / "_walkforward_matrix_manifest")
    df = pd.read_csv(d / "walkforward_matrix_detail.csv")
    df["calmar"] = df.oos_cagr / df.oos_mdd.abs()
    return df.rename(columns={"oos_cagr": "cagr", "oos_mdd": "mdd"})


def _scheme_meta(detail: pd.DataFrame) -> pd.DataFrame:
    return (detail.groupby("scheme")
            .agg(mode=("mode", "first"), min_is_months=("min_is_months", "first"),
                 oos_len_months=("oos_len_months", "first"))
            .reset_index())


def effect_sizes(detail: pd.DataFrame) -> pd.DataFrame:
    """格子層級的配對差距：A_hrp 減去對手，按 (方案 × 對手 × 指標) 彙總。

    ⚠️ 這是**效果量**不是檢定。格子互相重疊，故只報均值／中位／標準差，不報 p 值。
    ⚠️ MDD 是負數，`A − 對手 > 0` 代表 A 的回撤**較淺**（方向與勝率判定一致）。
    """
    keys = ["tree_key", "scheme", "k_mode", "ratio", "allocation", "window_no"]
    # 🔴 B_all 不受 ratio／allocation 影響，它自己那一列的 ratio="all"、
    # allocation="unallocated"，**跟 A_hrp 的設定鍵永遠對不上**。直接照 `keys` 做
    # pivot 會讓 B_all 欄整欄是 NaN，`dropna()` 後一列不剩，B_all 就在 merge 時被
    # 靜默丟掉（2026-09-05 開發時實測踩到：78 列而非應有的 117 列）。
    # 故 B_all 改用窗鍵 (tree_key, scheme, k_mode, window_no) 廣播對齊。
    # 這與 `walkforward_matrix.summarize()` 的處理方式一致。
    wkeys = ["tree_key", "scheme", "k_mode", "window_no"]
    rows = []
    for metric in METRICS:
        piv = detail.pivot_table(index=keys, columns="group", values=metric, observed=True)
        if "A_hrp" not in piv.columns:
            continue
        piv = piv.dropna(subset=["A_hrp"])
        b_ref = detail[detail.group == "B_all"].set_index(wkeys)[metric]
        b_ref = b_ref[~b_ref.index.duplicated()]
        for opp in OPPONENTS:
            if opp == "B_all":
                other = pd.Series(
                    piv.reset_index().set_index(wkeys).index.map(b_ref).to_numpy(),
                    index=piv.index)
            elif opp in piv.columns:
                other = piv[opp]
            else:
                continue
            diff = (piv["A_hrp"] - other).dropna()
            for scheme, s in diff.groupby(level="scheme"):
                rows.append({"scheme": scheme, "opponent": opp, "metric": metric,
                             "n_cells": int(len(s)),
                             "diff_mean": float(s.mean()),
                             "diff_median": float(s.median()),
                             "diff_std": float(s.std(ddof=1)),
                             "pct_positive": float((s > 0).mean())})
    out = pd.DataFrame(rows)
    missing = set(OPPONENTS) - set(out.opponent.unique())
    if missing:
        raise AssertionError(f"效果量表缺對手 {missing}——多半是設定鍵對不上被靜默丟掉")
    return out


def build(log=print) -> pd.DataFrame:
    units = _load_units()
    detail = _load_detail()
    meta = _scheme_meta(detail)
    eff = effect_sizes(detail)

    agg = (units.groupby(["opponent", "metric", "scheme"])
           .agg(n_units=("unit_win", "size"), n_wins=("unit_win", "sum"),
                win_share_mean=("win_share", "mean"),
                win_share_std=("win_share", "std"))
           .reset_index())
    agg["win_rate"] = agg.n_wins / agg.n_units
    df = agg.merge(meta, on="scheme").merge(eff, on=["scheme", "opponent", "metric"])
    # merge 是 inner join——任何一邊缺鍵都會**靜默少列**。逐方案 × 對手 × 指標
    # 應該剛好是 13 × 3 × 3 = 117 列，不足即代表有東西被丟掉。
    n_expect = len(meta) * len(OPPONENTS) * len(METRICS)
    if len(df) != n_expect:
        raise AssertionError(
            f"合併後 {len(df)} 列 != 預期 {n_expect}（方案 {len(meta)} × 對手 "
            f"{len(OPPONENTS)} × 指標 {len(METRICS)}）——有鍵對不上被靜默丟掉")

    # 一律轉 float：`min_wins_for_significance` 可能回 None（n 太小到全勝也不顯著），
    # 有無 None 會讓 dtype 在 int64/float64 之間跳動，契約型別檢查因此不穩定。
    df["min_wins_for_sig"] = df.n_units.map(min_wins_for_significance).astype(float)
    #: 🔴 可偵測勝率下限：n 個單位下要達 p<0.05 的最低勝率。None → 全勝也不可能顯著。
    df["detectable_win_rate"] = df.min_wins_for_sig / df.n_units
    df["p_value"] = [
        float(stats.binomtest(int(w), int(n), 0.5, alternative="two-sided").pvalue)
        for w, n in zip(df.n_wins, df.n_units)]
    df["significant_05"] = df.p_value < ALPHA
    #: 還差幾場才到顯著門檻。0 = 已達門檻。這是比 p 值直觀得多的距離度量。
    df["wins_short_of_sig"] = (df.min_wins_for_sig - df.n_wins).clip(lower=0)

    #: 🔴 方向：雙尾檢定顯著**不代表 A_hrp 贏**——A 也可能顯著地輸。
    #: 實測 A_hrp vs D_top_cagr 在方案 A 的 CAGR 上只贏 3/18（p=0.0075），
    #: 那是**顯著落敗**。2026-09-05 開發時被測試 ② 抓到：初版把這種列標成「顯著」，
    #: 讀表的人會誤以為 A 贏。故 verdict 一律帶方向。
    df["direction"] = np.where(df.win_rate > 0.5, "A勝",
                               np.where(df.win_rate < 0.5, "A敗", "平手"))

    #: 🔴 診斷標籤——本模組存在的理由，把「不顯著」拆成三種**完全不同**的處境。
    #:
    #: ⚠️ 2026-09-05 開發時修正過一次：初版把「勝率低於門檻」一律標成「效果不足」，
    #:    但 n=6 的方案門檻是 **100%（要 6/6 全勝）**，5/6 被標成「效果不足」完全
    #:    誤導——那是檢力問題不是效果問題。故先用門檻高度判斷，再用差距判斷。
    #:
    #:   顯著（A勝）     p < 0.05 且 A_hrp 勝率過半
    #:   顯著（A敗）     p < 0.05 但 A_hrp **輸**——同樣是結論，方向相反
    #:   檢力不足        門檻 >= 95%（要近乎全勝才可能顯著），這種「不顯著」不是證據
    #:   差一場就顯著    只差 1 個單位，屬臨界
    #:   效果不足        門檻不高、也不只差一場 → **這才是效果真的弱**
    df["verdict"] = np.where(
        df.significant_05 & (df.win_rate > 0.5), "顯著（A勝）",
        np.where(df.significant_05, "顯著（A敗）",
                 np.where(df.detectable_win_rate >= 0.95, "檢力不足（門檻需近全勝）",
                          np.where(df.wins_short_of_sig <= 1, "差一場就顯著", "效果不足"))))
    for c in ("opponent", "metric", "scheme", "mode", "direction", "verdict"):
        df[c] = df[c].astype("category")
    return df[C.MDD_WINDOW_LENGTH.names]


def run(log=print) -> pd.DataFrame:
    df = build(log=log)
    C.validate(df, C.MDD_WINDOW_LENGTH, strict_columns=True)
    log(f"✓ mdd_window_length 契約通過（{len(df)} 列）")

    out_dir = paths.ROOT / "_analysis_outputs_robustness"
    p = out_dir / "mdd_window_length.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "mdd_window_length", out_dir / "_mdd_window_length_manifest",
        inputs=[out_dir / "walkforward_matrix_detail.csv",
                out_dir / "walkforward_significance_units.csv"],
        outputs=[p],
        params={"alpha": ALPHA, "opponents": list(OPPONENTS), "metrics": list(METRICS)},
        notes="M-09：把 H-26c 的逐方案檢定按 OOS 窗長重新排列，並拆開「檢力不足」"
              "與「效果真的弱」。**不產生新的 p 值以外的檢定**——同窗長的方案互相"
              "重疊，不可合併。",
    )
    log(f"→ {p}")
    return df


def _report(df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 100)
    log("M-09 · MDD 顯著性按 OOS 窗長分層——「檢力不足」還是「效果真的弱」？")
    log("=" * 100)

    b = df[df.opponent == "B_all"]
    log("\n【一】A_hrp vs B_all：三個指標的方案顯著數（H-26c 的 8/13 從哪來）")
    log(f"  {'指標':<8}{'顯著方案數':>10}{'平均勝率':>10}{'平均效果量':>12}{'格子層級勝率':>13}")
    for m in METRICS:
        g = b[b.metric == m]
        unit = "pp" if m != "calmar" else ""
        scale = 100 if m != "calmar" else 1
        log(f"  {m:<8}{int(g.significant_05.sum()):>7}/{len(g):<3}"
            f"{g.win_rate.mean():>10.1%}{g.diff_mean.mean()*scale:>10.3f}{unit:<2}"
            f"{g.pct_positive.mean():>13.1%}")

    log("\n【二】按 OOS 窗長分層（A_hrp vs B_all · MDD）")
    log(f"  {'窗長':<6}{'方案':<14}{'勝/總':>8}{'勝率':>8}{'顯著門檻':>14}"
        f"{'效果量(pp)':>12}{'標準差':>9}  判定")
    mdd = b[b.metric == "mdd"].sort_values(["oos_len_months", "scheme"])
    for r in mdd.itertuples():
        floor = ("—" if pd.isna(r.detectable_win_rate)
                 else f"{int(r.min_wins_for_sig)}/{int(r.n_units)}={r.detectable_win_rate:.0%}")
        log(f"  {int(r.oos_len_months):<6}{r.scheme + ('(rolling)' if r.mode == 'rolling' else ''):<14}"
            f"{str(int(r.n_wins)) + '/' + str(int(r.n_units)):>8}{r.win_rate:>8.1%}{floor:>14}"
            f"{r.diff_mean*100:>12.3f}{r.diff_std*100:>9.3f}  {r.verdict}")

    log("\n【三】三個指標的效果量隨窗長變化（A_hrp vs B_all，格子層級）")
    log(f"  {'窗長':<6}{'指標':<8}{'方案數':>7}{'效果量均值':>12}{'效果量標準差':>13}{'正向格子%':>11}")
    for L in sorted(b.oos_len_months.unique()):
        for m in METRICS:
            g = b[(b.oos_len_months == L) & (b.metric == m)]
            scale = 100 if m != "calmar" else 1
            log(f"  {int(L):<6}{m:<8}{len(g):>7}{g.diff_mean.mean()*scale:>12.3f}"
                f"{g.diff_std.mean()*scale:>13.3f}{g.pct_positive.mean():>11.1%}")

    log("\n【四】結論")
    ng = mdd[~mdd.significant_05]
    weak = ng[ng.verdict == "效果不足"]
    log(f"  MDD 的 {len(ng)} 個不顯著方案裡，只有 {len(weak)} 個是「效果真的不足」："
        f"{', '.join(sorted(weak.scheme.astype(str))) or '（無）'}")
    log(f"  其餘 {len(ng)-len(weak)} 個是檢力問題（門檻需近全勝，或只差一場）。")
    anch = mdd[mdd["mode"] == "anchored"].diff_mean.mean() * 100
    roll = mdd[mdd["mode"] == "rolling"].diff_mean.mean() * 100
    log(f"  MDD 效果量：anchored 平均 {anch:.3f}pp，rolling {roll:.3f}pp"
        f"（僅 anchored 的 {roll/anch:.0%}）")
    log("")
    log("  · 「檢力不足（門檻需近全勝）」= n 太小，這種不顯著**不構成 MDD 弱的證據**。")
    log("  · 「效果不足」= 門檻不高、也不只差一場 → **才是 MDD 真的弱**。")
    log("  · 效果量若隨窗長穩定 → 不是短窗雜訊，是效果本來就比 CAGR 小、比較難測到。")


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(prog="research.mdd_window_length").parse_args(argv)
    _report(run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
