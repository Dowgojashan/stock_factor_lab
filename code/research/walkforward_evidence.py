# -*- coding: utf-8 -*-
"""M-10 · walk-forward 證據強度三件事（2026-09-05）

🔴 **這支腳本要補的三個洞**，都是 H-26/H-26c 報結論時沒有量化的假設：

---------------------------------------------------------------------------
① 效果量分布：92% 的勝率是「普遍小贏」還是「少數大贏」？
---------------------------------------------------------------------------
`walkforward_matrix_summary.csv` 只報**勝率**，M-09 只報**平均效果量**。
兩者都可能被少數極端格子撐起來。本節報每個 (樹 × 對手 × 指標) 的
**分位數（p10/p25/p50/p75/p90）與標準化效果量 Cohen's d**。

判讀：中位數與平均同號且量級相近 → 普遍性優勢；
      中位數接近 0 而平均明顯 → **少數極端格主導**，結論要收斂。

---------------------------------------------------------------------------
② 多重比較：117 個檢定報「13/13 顯著」是否高估
---------------------------------------------------------------------------
H-26c 對 13 方案 × 3 對手 × 3 指標各做一次二項檢定＝**117 個檢定**，
在 α=0.05 下純靠運氣就會有約 6 個假陽性。報「13/13 顯著」沒有做任何多重比較校正。

本節報未校正、**BH-FDR**（控制偽發現率）與 **Bonferroni**（控制族系錯誤率），
且**兩種校正基數都算**：

  族內（n=13）：固定 (對手 × 指標)，只對 13 個窗口方案校正。
                對應論文真正的宣稱「不管怎麼切窗，A 都贏 B_all」，
                **這是審查者預期的標準做法，論文主表用這個**。
  全域（n=117）：把三個對手 × 三個指標綁成一個大家族。那 117 個檢定其實回答
                **三個不同的研究問題**，全綁在一起校正**過苛**，只當補充。

⚠️ **BH 假設檢定之間獨立或正相關（PRDS）**。我們的檢定共用同一段歷史，
   **不滿足獨立性**，故 BH 在此偏樂觀。Bonferroni 不需要獨立性假設，任何情況都有效。

**實測結論**：A_hrp vs B_all 的「13/13 顯著」**完整通過族內 BH 校正**
（cagr 13/13、calmar 13/13、mdd 8/13，與未校正完全相同），
連最嚴格的族內 Bonferroni 都還剩 9/13、8/13、3/13。**這個主結論站得住。**

---------------------------------------------------------------------------
③ 相鄰窗次的選股重疊：窗次真的是獨立單位嗎
---------------------------------------------------------------------------
🔴 H-26c 整套檢定建立在「同方案的窗次互不重疊 ⇒ 可當獨立單位」。
**但那說的是 OOS 區間不重疊，不是選出的策略不重疊。**
anchored 第 2 窗的 IS = 第 1 窗的 IS + OOS，**訓練資料高度巢狀**，
選出的代表很可能大量重複——若相鄰窗次選的是同一批策略，
那些「獨立單位」在策略層面根本是同一個賭注重複下注，二項檢定的 n 被高估。

本節算相鄰窗次成員集合的 **Jaccard 相似度**與**重疊係數**，
並對照 anchored vs rolling（rolling 會丟掉舊資料，重疊應明顯較低）。

⚠️ 本節**不修正**任何既有 p 值——重疊率多高才該打折沒有客觀答案。
   它的作用是**把假設量化並揭露**，讓論文與口試能誠實討論。

依賴：`walkforward_matrix`（需要它輸出的 `walkforward_members.parquet`，
2026-09-05 為本模組新增）／`walkforward_significance`。

用法：
    cd code
    python -m research.walkforward_evidence
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, paths

OPPONENTS = ("B_all", "D_top_cagr", "E_top_calmar")
METRICS = ("cagr", "mdd", "calmar")
ALPHA = 0.05
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def _out_dir():
    return paths.ROOT / "_analysis_outputs_robustness"


def _load_detail() -> pd.DataFrame:
    d = _out_dir()
    freeze.verify_inputs(d / "_walkforward_matrix_manifest")
    df = pd.read_csv(d / "walkforward_matrix_detail.csv")
    df["calmar"] = df.oos_cagr / df.oos_mdd.abs()
    return df.rename(columns={"oos_cagr": "cagr", "oos_mdd": "mdd"})


# ============================================================================
# ① 效果量分布
# ============================================================================

def effect_distribution(detail: pd.DataFrame, subject: str = "A_hrp") -> pd.DataFrame:
    """每個 (樹 × 對手 × 指標) 的配對差距分布（`subject` 減對手）。

    🔴 `subject` 為 M-11（2026-09-07）新增，**預設 `A_hrp` 維持回溯相容**。

    ⚠️ B_all 的列 `ratio="all"`／`allocation="unallocated"`，設定鍵跟 A_hrp 對不上，
    必須用窗鍵廣播對齊——否則整組被靜默丟掉（M-09 開發時實測踩過）。
    ⚠️ 格子互相重疊，故**只報分布，不報 p 值**。
    """
    keys = ["tree_key", "scheme", "k_mode", "ratio", "allocation", "window_no"]
    wkeys = ["tree_key", "scheme", "k_mode", "window_no"]
    rows = []
    for metric in METRICS:
        piv = detail.pivot_table(index=keys, columns="group", values=metric,
                                 observed=True).dropna(subset=[subject])
        b_ref = detail[detail.group == "B_all"].set_index(wkeys)[metric]
        b_ref = b_ref[~b_ref.index.duplicated()]
        for opp in [o for o in OPPONENTS if o != subject]:
            if opp == "B_all":
                other = pd.Series(
                    piv.reset_index().set_index(wkeys).index.map(b_ref).to_numpy(),
                    index=piv.index)
            else:
                other = piv[opp]
            diff = (piv[subject] - other).dropna()
            for tree, s in diff.groupby(level="tree_key", observed=True):
                sd = float(s.std(ddof=1))
                rec = {"tree_key": tree, "opponent": opp, "metric": metric,
                       "n_cells": int(len(s)), "mean": float(s.mean()),
                       "std": sd,
                       #: 標準化效果量。|d|<0.2 小、0.5 中、0.8 大（Cohen 慣例）。
                       "cohens_d": float(s.mean() / sd) if sd > 0 else float("nan"),
                       "pct_positive": float((s > 0).mean())}
                for q in QUANTILES:
                    rec[f"p{int(q*100):02d}"] = float(s.quantile(q))
                #: 🔴 中位數 / 平均：接近 1 代表普遍性優勢；遠小於 1 代表少數極端格主導。
                rec["median_over_mean"] = (float(rec["p50"] / rec["mean"])
                                           if rec["mean"] != 0 else float("nan"))
                rows.append(rec)
    out = pd.DataFrame(rows)
    missing = {o for o in OPPONENTS if o != subject} - set(out.opponent)
    if missing:
        raise AssertionError(f"效果量分布缺對手 {missing}——設定鍵對不上被靜默丟掉")
    return out


# ============================================================================
# ② 多重比較校正
# ============================================================================

def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH 校正後的 q 值（單調化後上限 1）。

    ⚠️ BH 控制的是**偽發現率**，且假設檢定間獨立或正相關（PRDS）。
    本專案的檢定共用同一段歷史，**不滿足獨立性**，故 BH 在此偏樂觀。
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]      # 由大到小取累積最小值
    q = np.empty(n)
    q[order] = np.minimum(ranked, 1.0)
    return q


def multiplicity(log=print) -> pd.DataFrame:
    """對 H-26c 的 117 個逐方案二項檢定做多重比較校正。"""
    d = _out_dir()
    p = d / "walkforward_significance.csv"
    if not p.exists():
        raise FileNotFoundError("請先執行 `python -m research.walkforward_significance`")
    freeze.verify_inputs(d / "_walkforward_significance_manifest")
    sig = pd.read_csv(p)
    #: 🔴 只取 tree_scope="ALL" 的列。逐樹的列（TW/US/XM）是同一批單位的子集，
    #: 把它們一起丟進 FDR 等於同一個檢定數了四次，會把 q 值稀釋得毫無意義。
    sig = sig[sig.tree_scope == "ALL"].copy()
    n = len(sig)
    #: 🔴 **兩種校正基數都報，因為兩種都站得住，嚴苛程度差很多**：
    #:   全域（n=117）：把「vs B_all」「vs D」「vs E」× 三指標當成一個大家族。
    #:                   最保守，但那 117 個檢定其實回答三個不同的研究問題，
    #:                   全部綁在一起校正**過苛**。
    #:   族內（n=13） ：固定 (對手 × 指標)，只對 13 個窗口方案校正。
    #:                   對應論文真正的宣稱——「不管怎麼切窗，A 都贏 B_all」——
    #:                   **這是審查者預期的標準做法**。
    #: 論文主表用族內；全域那欄當「連最保守的校正都還剩幾個」的補充。
    sig["q_bh_global"] = benjamini_hochberg(sig.p_value.to_numpy())
    sig["p_bonf_global"] = np.minimum(sig.p_value * n, 1.0)
    sig["q_bh_family"] = np.nan
    sig["p_bonf_family"] = np.nan
    for _, idx in sig.groupby(["opponent", "metric"]).groups.items():
        sub = sig.loc[idx, "p_value"].to_numpy()
        sig.loc[idx, "q_bh_family"] = benjamini_hochberg(sub)
        sig.loc[idx, "p_bonf_family"] = np.minimum(sub * len(sub), 1.0)
    log(f"  多重比較：{n} 個檢定（tree_scope=ALL），族內基數 "
        f"{sig.groupby(['opponent', 'metric']).size().iloc[0]}")

    rows = []
    for (opp, metric), g in sig.groupby(["opponent", "metric"]):
        rows.append({"opponent": opp, "metric": metric, "n_tests": int(len(g)),
                     "n_sig_raw": int((g.p_value < ALPHA).sum()),
                     "n_sig_bh_family": int((g.q_bh_family < ALPHA).sum()),
                     "n_sig_bonferroni_family": int((g.p_bonf_family < ALPHA).sum()),
                     "n_sig_bh": int((g.q_bh_global < ALPHA).sum()),
                     "n_sig_bonferroni": int((g.p_bonf_global < ALPHA).sum()),
                     "min_p": float(g.p_value.min()),
                     "min_q_bh": float(g.q_bh_global.min()),
                     "min_p_bonferroni": float(g.p_bonf_global.min()),
                     "n_total_tests_corrected": int(n)})
    return pd.DataFrame(rows)


# ============================================================================
# ③ 相鄰窗次的選股重疊
# ============================================================================

def _jaccard(a: set, b: set) -> tuple[float, float]:
    inter = len(a & b)
    union = len(a | b)
    j = inter / union if union else float("nan")
    ov = inter / min(len(a), len(b)) if min(len(a), len(b)) else float("nan")
    return j, ov


def window_overlap(log=print) -> pd.DataFrame:
    """相鄰窗次選出的成員集合有多重疊。

    🔴 這是 H-26c「窗次可當獨立單位」假設的直接檢驗：OOS 區間不重疊 ≠ 選出的
    策略不重疊。anchored 的 IS 是巢狀的（第 2 窗的 IS = 第 1 窗 IS+OOS），
    若相鄰窗選的是同一批策略，那些「獨立單位」在策略層面是同一個賭注重複下注。
    """
    d = _out_dir()
    p = d / "walkforward_members.parquet"
    if not p.exists():
        raise FileNotFoundError(
            "找不到 walkforward_members.parquet——請重跑 "
            "`python -m research.walkforward_matrix`（2026-09-05 起才會輸出成員名單）")
    freeze.verify_inputs(d / "_walkforward_matrix_manifest")
    mem = pd.read_parquet(p)

    cellkeys = ["tree_key", "scheme", "k_mode", "ratio", "allocation", "group"]
    rows = []
    for cell, g in mem.groupby(cellkeys, observed=True):
        g = g.sort_values("window_no")
        if len(g) < 2:
            continue
        sets = [set(m) for m in g.members]
        js, ovs = [], []
        for a, b in zip(sets, sets[1:]):
            j, o = _jaccard(a, b)
            js.append(j); ovs.append(o)
        # 也算「所有窗次都選到」的核心集合大小——比相鄰重疊更嚴格的指標
        core = set.intersection(*sets)
        rec = dict(zip(cellkeys, cell))
        rec.update({"n_windows": int(len(g)),
                    "n_members_mean": float(np.mean([len(x) for x in sets])),
                    "jaccard_adjacent_mean": float(np.mean(js)),
                    "jaccard_adjacent_max": float(np.max(js)),
                    "overlap_coef_adjacent_mean": float(np.mean(ovs)),
                    #: 每一窗都被選中的策略數／平均組合大小——「從頭壓到尾」的比例
                    "core_size": int(len(core)),
                    "core_share": float(len(core) / np.mean([len(x) for x in sets]))})
        rows.append(rec)
    out = pd.DataFrame(rows)
    log(f"  重疊分析：{len(out):,} 個格子（每格 >= 2 個窗次）")
    return out


# ============================================================================
# 主流程
# ============================================================================

def run(log=print) -> dict[str, pd.DataFrame]:
    detail = _load_detail()
    d = _out_dir()

    eff = effect_distribution(detail)
    C.validate(eff, C.WF_EFFECT_DISTRIBUTION, strict_columns=True)
    p1 = d / "wf_effect_distribution.csv"
    eff.to_csv(p1, index=False, encoding="utf-8-sig")

    mul = multiplicity(log)
    C.validate(mul, C.WF_MULTIPLICITY, strict_columns=True)
    p2 = d / "wf_multiplicity.csv"
    mul.to_csv(p2, index=False, encoding="utf-8-sig")

    ovl = window_overlap(log)
    C.validate(ovl, C.WF_WINDOW_OVERLAP, strict_columns=True)
    p3 = d / "wf_window_overlap.csv"
    ovl.to_csv(p3, index=False, encoding="utf-8-sig")

    log(f"✓ 三張表契約通過（{len(eff)} / {len(mul)} / {len(ovl):,} 列）")
    freeze.write_manifest(
        "walkforward_evidence", d / "_walkforward_evidence_manifest",
        inputs=[d / "walkforward_matrix_detail.csv",
                d / "walkforward_members.parquet",
                d / "walkforward_significance.csv"],
        outputs=[p1, p2, p3],
        params={"alpha": ALPHA, "quantiles": list(QUANTILES),
                "fdr_scope": "tree_scope=ALL 的檢定（逐樹列是子集，納入會稀釋 q 值）"},
        notes="M-10：效果量分布、多重比較校正（BH + Bonferroni）、相鄰窗次選股重疊。"
              "重疊分析不修正任何既有 p 值，只把「窗次獨立」的假設量化揭露。",
    )
    log(f"→ {p1.name} / {p2.name} / {p3.name}")
    return {"effect": eff, "multiplicity": mul, "overlap": ovl}


def _report(res: dict[str, pd.DataFrame], log=print) -> None:
    eff, mul, ovl = res["effect"], res["multiplicity"], res["overlap"]

    log("\n" + "=" * 104)
    log("M-10 · walk-forward 證據強度")
    log("=" * 104)

    log("\n【①】效果量分布（A_hrp − 對手，格子層級；cagr/mdd 單位 pp）")
    log(f"  {'樹':<5}{'對手':<14}{'指標':<8}{'格數':>7}{'平均':>9}{'中位':>9}"
        f"{'p10':>9}{'p90':>9}{'標準差':>9}{'Cohen d':>9}{'正向%':>8}{'中位/平均':>10}")
    for r in eff.sort_values(["opponent", "metric", "tree_key"]).itertuples():
        sc = 100 if r.metric != "calmar" else 1
        log(f"  {r.tree_key:<5}{r.opponent:<14}{r.metric:<8}{r.n_cells:>7,}"
            f"{r.mean*sc:>9.3f}{r.p50*sc:>9.3f}{r.p10*sc:>9.3f}{r.p90*sc:>9.3f}"
            f"{r.std*sc:>9.3f}{r.cohens_d:>9.3f}{r.pct_positive:>8.1%}"
            f"{r.median_over_mean:>10.2f}")

    log("\n【②】多重比較校正（H-26c 的逐方案二項檢定，tree_scope=ALL）")
    n_tot = int(mul.n_total_tests_corrected.iloc[0])
    log(f"  校正基數 = {n_tot} 個檢定｜α = {ALPHA}")
    log(f"  {'對手':<14}{'指標':<8}{'檢定數':>7}{'未校正':>8}"
        f"{'族內BH':>8}{'族內Bonf':>10}{'全域BH':>8}{'全域Bonf':>10}{'最小 p':>11}")
    for r in mul.sort_values(["opponent", "metric"]).itertuples():
        log(f"  {r.opponent:<14}{r.metric:<8}{r.n_tests:>7}{r.n_sig_raw:>8}"
            f"{r.n_sig_bh_family:>8}{r.n_sig_bonferroni_family:>10}"
            f"{r.n_sig_bh:>8}{r.n_sig_bonferroni:>10}{r.min_p:>11.2e}")
    log("  ⚠️ **族內**（n=13，固定對手×指標）對應論文真正的宣稱「不管怎麼切窗都贏」，")
    log("     是審查者預期的標準做法；**全域**（n=117）把三個不同研究問題綁在一起，過苛。")
    log("  ⚠️ BH 假設檢定獨立或正相關；本專案的檢定共用歷史，**BH 偏樂觀**。")
    log("     Bonferroni 不需獨立性假設，任何情況都有效。")

    log("\n【③】相鄰窗次的選股重疊（A_hrp）")
    a = ovl[ovl.group == "A_hrp"]
    log(f"  {'模式':<10}{'格數':>7}{'平均組合大小':>13}{'相鄰 Jaccard':>14}"
        f"{'重疊係數':>11}{'全窗核心佔比':>13}")
    det = _load_detail()[["scheme", "mode"]].drop_duplicates()
    a = a.merge(det, on="scheme")
    for mode, g in a.groupby("mode", observed=True):
        log(f"  {mode:<10}{len(g):>7,}{g.n_members_mean.mean():>13.1f}"
            f"{g.jaccard_adjacent_mean.mean():>14.3f}"
            f"{g.overlap_coef_adjacent_mean.mean():>11.3f}{g.core_share.mean():>13.3f}")
    log(f"\n  {'方案':<8}{'格數':>7}{'相鄰 Jaccard':>14}{'重疊係數':>11}{'全窗核心佔比':>13}")
    for scheme, g in a.groupby("scheme", observed=True):
        log(f"  {str(scheme):<8}{len(g):>7,}{g.jaccard_adjacent_mean.mean():>14.3f}"
            f"{g.overlap_coef_adjacent_mean.mean():>11.3f}{g.core_share.mean():>13.3f}")
    log("\n  判讀：Jaccard 越高 → 相鄰窗次選的是同一批策略 → 那兩個「獨立單位」")
    log("        在策略層面是同一個賭注重複下注，二項檢定的有效 n 被高估。")
    log("        anchored 的 IS 是巢狀的，重疊本來就該高於 rolling。")


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(prog="research.walkforward_evidence").parse_args(argv)
    _report(run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
