# -*- coding: utf-8 -*-
"""M-11 · 換掉檢定主體：`E_top_calmar` / `D_top_cagr` vs `B_all`（2026-09-07）

🔴 **這支腳本要補的洞**：本專案至今**所有**檢定的主體都恆為 `A_hrp`
（`walkforward_significance` / `walkforward_evidence` / `mdd_window_length` 的
`OPPONENTS` 都是 `("B_all","D_top_cagr","E_top_calmar")`，主體寫死 A）。

**但矩陣自己的數字顯示，表現最好的方法是 `E_top_calmar`**：
A_hrp 在 Calmar 上贏 E 的格子只有 **19.6%**（TW 16.8%／US 8.0%／XM 34.0%，
出自 H-26/H-27 矩陣 2,700 格，記在 `M03_partition_control.md` §2）。

**而 E 從來沒有跟 `B_all` 比過勝率，也沒做過任何顯著性檢定。**

第五章目前的主張是「A_hrp 精選勝過狂灑（13/13 方案顯著）」。
更誠實的版本可能是「有效的是 IS Calmar 排序精選（E）本身；
HRP 的群配額是在這之上**扣分**的約束，換到的是跨市場回撤保護」。
**不跑這個數字，就不知道該寫哪一句。**

---------------------------------------------------------------------------
做法：把三支既有模組的主體參數化，不重跑矩陣
---------------------------------------------------------------------------
  `walkforward_matrix.summarize(df, subject=...)`        勝率
  `walkforward_significance.unit_wins(df, subject=...)`  逐方案二項檢定
  `walkforward_evidence.effect_distribution(df, subject=...)`  效果量與 Cohen's d

三者的 `subject` 預設都是 `A_hrp`，**既有產出逐位元不變**（已驗證）。
本模組只讀 `walkforward_matrix_detail.csv` 做彙整，**不擾動凍結的矩陣**
（沿用 M-02b／M-03b 的處理慣例）。

---------------------------------------------------------------------------
⚠️ 多重比較用「族內」基數
---------------------------------------------------------------------------
沿用 M-10 的定案：校正基數固定 (主體 × 對手 × 指標) 底下的 **13 個窗口方案**。
全域基數（把不同研究問題綁一起）過苛，只當補充。理由見 12.0c。

用法：
    cd code
    python -m research.subject_comparison
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

from . import contracts as C
from . import freeze, paths
from .walkforward_evidence import benjamini_hochberg, effect_distribution
from .walkforward_matrix import summarize
from .walkforward_significance import sign_tests, unit_wins

#: 要當主體跑一次的組別。A_hrp 一起跑是為了**自我校驗**——它的結果必須與
#: 既有的凍結產出完全一致，否則代表參數化改壞了某條路徑。
SUBJECTS = ("A_hrp", "E_top_calmar", "D_top_cagr")
ALPHA = 0.05
METRICS = ("cagr", "mdd", "calmar")


def _load_detail() -> pd.DataFrame:
    d = paths.ROOT / "_analysis_outputs_robustness"
    freeze.verify_inputs(d / "_walkforward_matrix_manifest")
    df = pd.read_csv(d / "walkforward_matrix_detail.csv")
    df["calmar"] = df.oos_cagr / df.oos_mdd.abs()
    return df.rename(columns={"oos_cagr": "cagr", "oos_mdd": "mdd"})


def build(log=print) -> pd.DataFrame:
    detail = _load_detail()
    raw = pd.read_csv(paths.ROOT / "_analysis_outputs_robustness"
                      / "walkforward_matrix_detail.csv")

    rows = []
    for subject in SUBJECTS:
        # ---- 勝率（逐樹）----
        summ = summarize(raw, subject=subject)
        wins = summ[summ.dimension == "tree_key"].set_index("value")

        # ---- 逐方案二項檢定 ----
        units = unit_wins(detail, subject=subject)
        sig = sign_tests(units)
        sig = sig[sig.tree_scope == "ALL"].copy()
        # 族內校正：固定 (對手 × 指標) 的 13 個方案
        sig["q_bh_family"] = np.nan
        sig["p_bonf_family"] = np.nan
        for _, idx in sig.groupby(["opponent", "metric"]).groups.items():
            sub = sig.loc[idx, "p_value"].to_numpy()
            sig.loc[idx, "q_bh_family"] = benjamini_hochberg(sub)
            sig.loc[idx, "p_bonf_family"] = np.minimum(sub * len(sub), 1.0)

        # ---- 效果量 ----
        eff = effect_distribution(detail, subject=subject)

        for opp in ("B_all", "D_top_cagr", "E_top_calmar"):
            if opp == subject:
                continue
            short = {"B_all": "B", "D_top_cagr": "D", "E_top_calmar": "E"}[opp]
            for metric in METRICS:
                g = sig[(sig.opponent == opp) & (sig.metric == metric)]
                e = eff[(eff.opponent == opp) & (eff.metric == metric)]
                for tree in ("TW", "US", "XM"):
                    et = e[e.tree_key == tree]
                    rows.append({
                        "subject": subject, "opponent": opp, "metric": metric,
                        "tree_key": tree,
                        "win_rate": float(wins.loc[tree, f"win_{short}_{metric}"]),
                        "n_cells": int(et.n_cells.iloc[0]) if len(et) else 0,
                        "cohens_d": float(et.cohens_d.iloc[0]) if len(et) else np.nan,
                        "diff_mean": float(et["mean"].iloc[0]) if len(et) else np.nan,
                        "diff_median": float(et.p50.iloc[0]) if len(et) else np.nan,
                        # 顯著性欄位是**跨樹**的（tree_scope="ALL"），三棵樹共用同一組值
                        "n_schemes": int(len(g)),
                        "n_sig_raw": int((g.p_value < ALPHA).sum()),
                        "n_sig_bh_family": int((g.q_bh_family < ALPHA).sum()),
                        "n_sig_bonferroni_family": int((g.p_bonf_family < ALPHA).sum()),
                        "min_p": float(g.p_value.min()) if len(g) else np.nan,
                    })
        log(f"  [{subject}] 完成")

    df = pd.DataFrame(rows)
    for c in ("subject", "opponent", "metric", "tree_key"):
        df[c] = df[c].astype("category")
    return df[C.SUBJECT_COMPARISON.names]


def _self_check(df: pd.DataFrame, log=print) -> None:
    """🔴 自我校驗：`A_hrp` 為主體時，本表必須與既有凍結產出完全一致。

    參數化改動了三支模組的核心迴圈，這是唯一能確認「沒有改壞既有路徑」的方法。
    """
    d = paths.ROOT / "_analysis_outputs_robustness"
    summ = pd.read_csv(d / "walkforward_matrix_summary.csv")
    summ = summ[summ.dimension == "tree_key"].set_index("value")
    bad = []
    for r in df[df.subject == "A_hrp"].itertuples():
        short = {"B_all": "B", "D_top_cagr": "D", "E_top_calmar": "E"}[r.opponent]
        ref = float(summ.loc[r.tree_key, f"win_{short}_{r.metric}"])
        if abs(r.win_rate - ref) > 1e-9:
            bad.append(f"  {r.tree_key}/{r.opponent}/{r.metric}: {r.win_rate} vs {ref}")
    if bad:
        raise AssertionError("A_hrp 主體的勝率與凍結的 summary 不符——"
                             "主體參數化改壞了既有路徑：\n" + "\n".join(bad))
    log("✓ 自我校驗通過：A_hrp 為主體時與凍結產出完全一致")


def run(log=print) -> pd.DataFrame:
    df = build(log=log)
    C.validate(df, C.SUBJECT_COMPARISON, strict_columns=True)
    _self_check(df, log)
    log(f"✓ subject_comparison 契約通過（{len(df)} 列）")

    out_dir = paths.ROOT / "_analysis_outputs_robustness"
    p = out_dir / "subject_comparison.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "subject_comparison", out_dir / "_subject_comparison_manifest",
        inputs=[out_dir / "walkforward_matrix_detail.csv",
                out_dir / "walkforward_matrix_summary.csv"],
        outputs=[p],
        params={"subjects": list(SUBJECTS), "alpha": ALPHA,
                "correction": "族內（固定 主體×對手×指標 的 13 個方案）"},
        notes="M-11：把檢定主體從寫死的 A_hrp 換成 E_top_calmar / D_top_cagr。"
              "只讀既有明細彙整，不重跑矩陣、不擾動凍結鏈。"
              "含自我校驗：A_hrp 主體的結果必須與凍結的 summary 完全一致。",
    )
    log(f"→ {p}")
    return df


def _report(df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 100)
    log("M-11 · 換掉檢定主體——第五章的主張句該怎麼寫")
    log("=" * 100)

    log("\n【一】各主體 vs B_all（狂灑）的勝率與顯著性")
    log(f"  {'主體':<14}{'指標':<9}{'TW':>8}{'US':>8}{'XM':>8}"
        f"{'族內BH':>9}{'族內Bonf':>10}{'Cohen d 中位':>13}")
    b = df[df.opponent == "B_all"]
    for subj in SUBJECTS:
        for metric in METRICS:
            g = b[(b.subject == subj) & (b.metric == metric)]
            if g.empty:
                continue
            w = {r.tree_key: r.win_rate for r in g.itertuples()}
            r0 = g.iloc[0]
            log(f"  {subj:<14}{metric:<9}"
                + "".join(f"{w.get(t, float('nan')):>8.1%}" for t in ("TW", "US", "XM"))
                + f"{int(r0.n_sig_bh_family):>4}/{int(r0.n_schemes):<4}"
                + f"{int(r0.n_sig_bonferroni_family):>5}/{int(r0.n_schemes):<4}"
                + f"{g.cohens_d.median():>13.3f}")
        log("")

    log("【二】主體之間的直接對決（勝率＝列的主體贏欄的對手）")
    for metric in METRICS:
        log(f"\n  · OOS {metric.upper()}")
        log(f"    {'主體':<14}" + "".join(f"{o:>16}" for o in
                                          ("vs A_hrp", "vs D_top_cagr", "vs E_top_calmar")))
        for subj in SUBJECTS:
            cells = []
            for opp in ("A_hrp", "D_top_cagr", "E_top_calmar"):
                g = df[(df.subject == subj) & (df.opponent == opp) & (df.metric == metric)]
                cells.append(f"{g.win_rate.mean():>16.1%}" if len(g) else f"{'—':>16}")
            log(f"    {subj:<14}" + "".join(cells))

    log("\n判讀：")
    log("  · 若 E vs B_all 的勝率與顯著方案數**不輸** A vs B_all，代表「精選勝過狂灑」")
    log("    這個主結論**不需要 HRP**——第五章的主張句要改成以挑選規則為主詞。")
    log("  · A 相對 E 的價值要另外舉證（M-03b 已知：跨市場回撤 59~89% 勝率）。")


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(prog="research.subject_comparison").parse_args(argv)
    _report(run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
