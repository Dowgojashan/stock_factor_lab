# -*- coding: utf-8 -*-
"""RQ2 主結論的敏感度：「單位勝出」改用平均差／中位數差判定（2026-11-15，使用者授權，經寫作 session 轉達）。

現行（`walkforward_significance.unit_wins`）：單位＝(樹 × 方案 × 窗次)，底下 20 種設定
（5 比例 × 2 分配 × 2 k_mode）；A_hrp 在**過半設定**中優於對手才算該單位勝出
（理由見該檔 docstring：用多數決而非平均，避免少數極端格主導）。
之後逐方案做二項檢定（H0: p=0.5，單尾），再做族內（n=13）BH／Bonferroni（`walkforward_evidence.multiplicity`）。

本檔只換「單位勝出」的判定，其餘流程不變：
  majority：該單位 20 種設定中 A>對手 的比例 > 0.5（＝現行，用來對帳）
  mean    ：該單位 20 種設定的 (A−對手) 平均 > 0
  median  ：該單位 20 種設定的 (A−對手) 中位數 > 0

對手：B_all、D_top_cagr、E_top_calmar（與現行相同），另加 C_random。
  C_random 取 `walkforward_random.csv` 每格 200 次隨機抽取的平均值當對手值
  （現行流程沒有 C_random 的二項檢定，只有逐格 z 分數；這裡是新增的）。
  隨機表只有 CAGR、MDD 的平均，沒有逐次抽取的 Calmar，Calmar 不做。

只讀現有結果（walkforward_matrix_detail.csv、walkforward_random.csv，都是 2026-11-14 這一輪），不建樹、不呼叫 LLM。
輸出：`_analysis_outputs_robustness/wf_unit_rule_sensitivity_tests.csv`（逐檢定）、
      `wf_unit_rule_sensitivity_summary.csv`（每個 對手×指標×判定 的顯著方案數）
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from . import freeze, paths
from . import walkforward_evidence as WE
from . import walkforward_significance as WS

OUT = paths.ROOT / "_analysis_outputs_robustness"
RULES = ("majority", "mean", "median")
METRICS = ("cagr", "mdd", "calmar")
SETTING = ["tree_key", "scheme", "window_no", "k_mode", "ratio", "allocation"]
WKEY = ["tree_key", "scheme", "window_no", "k_mode"]
RKEY = ["tree_key", "is_start", "is_end", "oos_start", "oos_end", "n_members"]
ALPHA = WE.ALPHA


def cell_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """每個 (設定 × 對手 × 指標) 的 A − 對手。MDD 是負數，數值大＝較淺，所以一律 A−對手>0 為 A 較好。"""
    a = df[df.group == "A_hrp"].set_index(SETTING)
    rows = []
    for opp in WS.OPPONENTS:
        o = df[df.group == opp]
        for m in METRICS:
            if opp == "B_all":
                ser = o.set_index(WKEY)[m]
                ov = a.reset_index().set_index(WKEY).index.map(ser)
            else:
                ov = o.set_index(SETTING)[m].reindex(a.index).to_numpy()
            rows.append(_frame(a, np.asarray(ov, dtype=float), opp, m))
    # C_random：每格 200 次抽取的平均
    rnd = pd.read_csv(OUT / "walkforward_random.csv")
    ar = a.reset_index()
    mg = ar.merge(rnd[RKEY + ["oos_cagr", "oos_mdd"]], on=RKEY, how="left", validate="many_to_one")
    assert len(mg) == len(ar) and mg.oos_cagr.notna().all(), "C_random 對不上 A_hrp 的格子"
    a2 = a.copy()
    for m, col in (("cagr", "oos_cagr"), ("mdd", "oos_mdd")):
        rows.append(_frame(a2, mg[col].to_numpy(dtype=float), "C_random", m))
    return pd.concat(rows, ignore_index=True)


def _frame(a: pd.DataFrame, ov: np.ndarray, opp: str, m: str) -> pd.DataFrame:
    t = a.reset_index()[["tree_key", "scheme", "window_no"]].copy()
    av = a[m].to_numpy(dtype=float)
    t["diff"] = av - ov
    t["win"] = av > ov                       # 與現行相同：NaN 比較一律 False
    t["opponent"], t["metric"] = opp, m
    return t


def unit_table(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby(["opponent", "metric", "tree_key", "scheme", "window_no"], observed=True)
    u = g.agg(win_share=("win", "mean"), diff_mean=("diff", "mean"), diff_median=("diff", "median"),
              n_settings=("win", "size"), n_nan=("diff", lambda s: int(s.isna().sum()))).reset_index()
    u["win_majority"] = u.win_share > 0.5
    u["win_mean"] = u.diff_mean > 0
    u["win_median"] = u.diff_median > 0
    return u


def sign_tests(u: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (opp, m, sch), g in u.groupby(["opponent", "metric", "scheme"], observed=True):
        for scope in ("ALL", "TW", "US", "XM"):
            gg = g if scope == "ALL" else g[g.tree_key == scope]
            n = len(gg)
            for r in RULES:
                w = int(gg[f"win_{r}"].sum())
                rows.append({"opponent": opp, "metric": m, "scheme": sch, "tree_scope": scope, "rule": r,
                             "n_units": n, "n_wins": w, "win_rate": w / n,
                             "p_value": float(stats.binomtest(w, n, 0.5, alternative="greater").pvalue)})
    t = pd.DataFrame(rows)
    t["q_bh_family"] = np.nan
    t["p_bonf_family"] = np.nan
    al = t.tree_scope == "ALL"
    for _, idx in t[al].groupby(["opponent", "metric", "rule"]).groups.items():
        p = t.loc[idx, "p_value"].to_numpy()
        t.loc[idx, "q_bh_family"] = WE.benjamini_hochberg(p)
        t.loc[idx, "p_bonf_family"] = np.minimum(p * len(p), 1.0)
    return t


def summary(t: pd.DataFrame) -> pd.DataFrame:
    a = t[t.tree_scope == "ALL"]
    rows = []
    for (opp, m, r), g in a.groupby(["opponent", "metric", "rule"]):
        row = {"opponent": opp, "metric": m, "rule": r, "n_schemes": len(g),
               "n_sig_raw": int((g.p_value < ALPHA).sum()),
               "n_sig_bh_family": int((g.q_bh_family < ALPHA).sum()),
               "n_sig_bonferroni_family": int((g.p_bonf_family < ALPHA).sum()),
               "units_won_total": int(g.n_wins.sum()), "units_total": int(g.n_units.sum())}
        for s in ("TW", "US", "XM"):
            ts = t[(t.tree_scope == s) & (t.opponent == opp) & (t.metric == m) & (t.rule == r)]
            row[f"n_sig_raw_{s}"] = int((ts.p_value < ALPHA).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def reconcile(t: pd.DataFrame) -> None:
    """majority 必須與現行 walkforward_significance.csv／wf_multiplicity.csv 逐列相同。"""
    cur = pd.read_csv(OUT / "walkforward_significance.csv")
    mine = t[(t.rule == "majority") & (t.opponent != "C_random")]
    m = cur.merge(mine, on=["opponent", "metric", "scheme", "tree_scope"], suffixes=("_cur", "_new"))
    assert len(m) == len(cur) == len(mine), (len(m), len(cur), len(mine))
    assert (m.n_wins_cur == m.n_wins_new).all() and (m.n_units_cur == m.n_units_new).all()
    assert np.allclose(m.p_value_cur, m.p_value_new, rtol=0, atol=1e-15)
    mul = pd.read_csv(OUT / "wf_multiplicity.csv")
    s = summary(t)
    s = s[(s.rule == "majority") & (s.opponent != "C_random")]
    k = mul.merge(s, on=["opponent", "metric"])
    assert len(k) == len(mul)
    for c in ("n_sig_raw", "n_sig_bh_family", "n_sig_bonferroni_family"):
        assert (k[f"{c}_x"] == k[f"{c}_y"]).all(), c


def main(argv=None) -> int:
    freeze.verify_inputs(OUT / "_walkforward_significance_manifest")
    df = WS._load()
    long = cell_diffs(df)
    u = unit_table(long)
    nn = int(u.n_nan.sum())
    t = sign_tests(u)
    reconcile(t)
    print("✓ 多數決與現行 walkforward_significance／wf_multiplicity 逐列相同")
    s = summary(t)
    p1, p2, p3 = (OUT / "wf_unit_rule_sensitivity_tests.csv", OUT / "wf_unit_rule_sensitivity_summary.csv",
                  OUT / "wf_unit_rule_sensitivity_units.csv")
    t.to_csv(p1, index=False, encoding="utf-8-sig")
    s.to_csv(p2, index=False, encoding="utf-8-sig")
    u.to_csv(p3, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "wf_unit_rule_sensitivity", OUT / "_wf_unit_rule_sensitivity_manifest",
        inputs=[OUT / "walkforward_matrix_detail.csv", OUT / "walkforward_random.csv",
                OUT / "walkforward_significance.csv", OUT / "wf_multiplicity.csv"],
        outputs=[p1, p2, p3],
        params={"rules": list(RULES), "alpha": ALPHA, "family": "(opponent, metric, rule) × 13 schemes",
                "c_random": "每格 200 次抽取平均；Calmar 無資料不做"},
        notes=f"RQ2 敏感度：單位勝出改用平均差／中位數差。含 NaN 的設定格數 {nn}。")
    pd.set_option("display.width", 200)
    print(s[["opponent", "metric", "rule", "n_sig_raw", "n_sig_bh_family", "n_sig_bonferroni_family",
             "units_won_total", "units_total", "n_sig_raw_TW", "n_sig_raw_US", "n_sig_raw_XM"]].to_string(index=False))
    print(f"含 NaN 的設定格數：{nn}")
    print(p1, p2, p3, sep="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
