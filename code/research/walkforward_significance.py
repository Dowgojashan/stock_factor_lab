# -*- coding: utf-8 -*-
"""H-26c · walk-forward 結果的統計檢定（2026-09-04）

🔴 **這支腳本要解決的問題**：`walkforward_matrix` 報的是「A_hrp 在 2,700 個格子裡
贏了 92~94%」，但**那 2,700 格共用同一段歷史**——45 個窗次只對應 21 種不重複的
OOS 區間，而且區間之間還互相包含（`2019-01~2020-12`、`2019-01~2021-12`、
`2019-01~2022-12` 是同起點的三種長度）。

所以 **2,700 不是樣本數**，直接拿去做檢定會嚴重高估顯著性。

---------------------------------------------------------------------------
本模組的做法：只在「保證互不重疊」的單位上做檢定
---------------------------------------------------------------------------
關鍵觀察：**同一個方案內部的窗次，依建構方式必然互不重疊也無缺口**
（`_blocks()` 是把時間軸切成連續區塊，`t_walkforward_schemes_are_mechanical`
已鎖住這個性質）。跨方案才會重疊。

故檢定單位取 **(樹 × 方案 × 窗次)**，並**逐方案分開檢定**：

  1. 一個單位 = 某棵樹在某方案的某個窗次
     該單位底下有 20 種設定（5 比例 × 2 分配 × 2 k_mode）
  2. 該單位算「勝出」的條件：A_hrp 在**過半設定**中優於對手
     （用多數決而非平均，避免少數極端格主導）
  3. 對每個方案，用它自己的窗次做**二項檢定**（H0: p=0.5）

⚠️ **不把 13 個方案的 p 值合併**（Fisher 等方法要求獨立，但方案之間共用歷史）。
改為**逐方案各自報告**——「每一個方案單獨檢定都顯著」比一個合併的小 p 值誠實，
也更貼近老師要的「不管怎麼切都偏向 HRP」。

⚠️ 同一時期的三棵樹（TW/US/XM）不是完全獨立（XM 含台美兩市場的策略），
本模組把它們當獨立單位處理，這一點在輸出與論文都要註明。

用法：
    cd code
    python -m research.walkforward_significance
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

from . import contracts as C
from . import freeze, paths

#: 對手組別。B_all 是主結論（精選 vs 狂灑），D/E 是機制拆解用。
OPPONENTS = ("B_all", "D_top_cagr", "E_top_calmar")
#: 三個指標都要看——老師點名 MDD 是唯一代價，只看 CAGR 會得到錯誤結論
METRICS = ("cagr", "mdd", "calmar")


def _load() -> pd.DataFrame:
    d = paths.ROOT / "_analysis_outputs_robustness"
    p = d / "walkforward_matrix_detail.csv"
    if not p.exists():
        raise FileNotFoundError("請先執行 `python -m research.walkforward_matrix`")
    freeze.verify_inputs(d / "_walkforward_matrix_manifest")
    df = pd.read_csv(p)
    df["calmar"] = df.oos_cagr / df.oos_mdd.abs()
    return df.rename(columns={"oos_cagr": "cagr", "oos_mdd": "mdd"})


def unit_wins(df: pd.DataFrame) -> pd.DataFrame:
    """每個 (樹 × 方案 × 窗次 × 對手 × 指標) 單位，A_hrp 是否在過半設定中勝出。

    ⚠️ MDD 是負數，「較淺」＝數值較大，所以三個指標一律用 `>` 判勝，方向正確。
    """
    setting = ["tree_key", "scheme", "window_no", "k_mode", "ratio", "allocation"]
    wkey = ["tree_key", "scheme", "window_no", "k_mode"]
    a = df[df.group == "A_hrp"].set_index(setting)

    rows = []
    for opp in OPPONENTS:
        o = df[df.group == opp]
        for m in METRICS:
            if opp == "B_all":
                # B_all 不隨 ratio/allocation 變動，用窗次層級的值廣播
                ser = o.set_index(wkey)[m]
                opp_vals = a.reset_index().set_index(wkey).index.map(ser)
            else:
                opp_vals = o.set_index(setting)[m].reindex(a.index).to_numpy()
            t = a.reset_index()[["tree_key", "scheme", "window_no"]].copy()
            t["win"] = a[m].to_numpy() > np.asarray(opp_vals, dtype=float)
            t["opponent"], t["metric"] = opp, m
            rows.append(t)
    long = pd.concat(rows, ignore_index=True)

    # 單位＝(樹, 方案, 窗次)；該單位「勝出」＝ A 在過半設定中優於對手
    g = long.groupby(["opponent", "metric", "tree_key", "scheme", "window_no"],
                     observed=True)["win"]
    out = g.mean().rename("win_share").reset_index()
    out["n_settings"] = g.size().to_numpy()
    out["unit_win"] = out.win_share > 0.5
    return out


def sign_tests(units: pd.DataFrame) -> pd.DataFrame:
    """逐 (對手 × 指標 × 方案 × 樹範圍) 做二項檢定。

    `tree_scope="ALL"` 把三棵樹的單位併在一起（樹之間不完全獨立，見模組 docstring）；
    另外也逐樹分開報，讓讀者自己判斷。
    """
    rows = []
    for (opp, m, sch), g in units.groupby(["opponent", "metric", "scheme"], observed=True):
        for scope in ("ALL", "TW", "US", "XM"):
            gg = g if scope == "ALL" else g[g.tree_key == scope]
            n = len(gg)
            if n == 0:
                continue
            w = int(gg.unit_win.sum())
            rows.append({
                "opponent": opp, "metric": m, "scheme": sch, "tree_scope": scope,
                "n_units": n, "n_wins": w, "win_rate": w / n,
                "p_value": float(stats.binomtest(w, n, 0.5, alternative="greater").pvalue),
                "n_windows_per_tree": int(gg.window_no.nunique()),
            })
    df = pd.DataFrame(rows)
    df["significant_05"] = df.p_value < 0.05
    return df


def run(log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _load()
    units = unit_wins(df)
    tests = sign_tests(units)
    for col in ("opponent", "metric", "scheme", "tree_scope"):
        tests[col] = tests[col].astype("category")
    C.validate(tests, C.WALKFORWARD_SIGNIFICANCE, strict_columns=True)
    log(f"✓ walkforward_significance 契約通過（{len(tests)} 列檢定）")

    out_dir = paths.ROOT / "_analysis_outputs_robustness"
    p_t = out_dir / "walkforward_significance.csv"
    p_u = out_dir / "walkforward_significance_units.csv"
    tests.to_csv(p_t, index=False, encoding="utf-8-sig")
    units.to_csv(p_u, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "walkforward_significance", out_dir / "_walkforward_significance_manifest",
        inputs=[out_dir / "walkforward_matrix_detail.csv"],
        outputs=[p_t, p_u],
        params={"opponents": list(OPPONENTS), "metrics": list(METRICS),
               "unit": "(tree, scheme, window_no)",
               "unit_win_rule": "A_hrp 在過半設定(5比例×2分配×2k_mode)中優於對手",
               "test": "binomial, H0: p=0.5, alternative=greater"},
        notes="H-26c：只在方案內部（互不重疊）的窗次上做二項檢定，不把 2,700 格"
              "當樣本數；不合併各方案 p 值（方案間共用歷史）。",
    )
    log(f"→ {p_t}\n→ {p_u}")
    return units, tests


def _report(tests: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 84)
    log("H-26c · walk-forward 統計檢定（單位＝方案內互不重疊的窗次 × 樹）")
    log("=" * 84)
    for m in METRICS:
        t = tests[(tests.opponent == "B_all") & (tests.metric == m)
                  & (tests.tree_scope == "ALL")].sort_values("scheme")
        log(f"\n【A_hrp vs B_all · OOS {m.upper()}】")
        log(f"{'方案':<6}{'每樹窗數':>9}{'單位數':>8}{'A勝出':>7}{'勝率':>9}{'二項p值':>12}{'':>4}")
        for r in t.itertuples():
            log(f"{r.scheme:<6}{r.n_windows_per_tree:>9}{r.n_units:>8}{r.n_wins:>7}"
                f"{r.win_rate:>9.1%}{r.p_value:>12.2e}{'  ✓' if r.significant_05 else '  ✗'}")
        n_sig = int(t.significant_05.sum())
        log(f"  → {n_sig}/{len(t)} 個方案在 p<0.05 顯著")

    log("\n" + "-" * 84)
    log("最嚴格的單一方案（窗次最多＝單位最多）")
    log("-" * 84)
    b = tests[(tests.opponent == "B_all") & (tests.tree_scope == "ALL")]
    best = b.loc[b.groupby("metric", observed=True)["n_units"].idxmax()]
    for r in best.itertuples():
        log(f"  {r.metric.upper():<7} 方案{r.scheme}：{r.n_units} 單位"
            f"（3 樹 × {r.n_windows_per_tree} 個互不重疊窗次），"
            f"A 勝出 {r.n_wins}，p = {r.p_value:.2e}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.walkforward_significance")
    ap.parse_args(argv)
    _, tests = run()
    _report(tests)
    return 0


if __name__ == "__main__":
    sys.exit(main())
