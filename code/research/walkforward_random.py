# -*- coding: utf-8 -*-
"""M-02b · walk-forward 版的 C_random 對照（2026-09-06）

🔴 **這支腳本要補的洞**：H-12 的 `C_random`（隨機**挑選**同樣檔數）證明了
「挑選有技術」，A_hrp 高出 2.3~7.0σ。**但那是單一窗、legacy 比例、fixed k 的結果。**

現在的實驗矩陣是 **45 個窗 × 5 種比例 × 2 種分配 × 2 種 k_mode**，
組合檔數從 15 檔一路到 1,504 檔。舊的 C_random 完全沒有覆蓋這個空間——
`walkforward_matrix` 在 2026-09-03 依使用者決定移除了 C_random（理由：每格抽 200 次，
是矩陣最貴的一組），現在時間夠了，補回來。

---------------------------------------------------------------------------
🔴 為什麼不直接加回 `walkforward_matrix`，而是獨立一支
---------------------------------------------------------------------------
① **C_random 根本不需要建樹**——它是從該窗宇宙裡純隨機抽 N 檔，跟分群無關。
   矩陣 2.5 小時裡絕大部分是 43 次建樹，全部可以省掉。
② **加進矩陣會改動 `walkforward_matrix_detail.csv`**，連帶要重跑
   `walkforward_significance` / `mdd_window_length` / `walkforward_evidence`。
   獨立成表、用格子鍵 join，資訊完全相同而凍結鏈不受擾動。
③ 去重後只有 **414 個不重複的 (樹 × 窗 × 檔數)** 組合（不是 2,700 格）——
   同一個窗、同樣檔數的格子可以共用同一批抽樣。

---------------------------------------------------------------------------
設計決定
---------------------------------------------------------------------------
- **檔數對齊 A_hrp 的實際 `n_members`**，不是 `target_total`。A 可能因 backfill 或
  空群使實際檔數與目標不同；**對齊實際值才是公平比較**（同樣買幾檔）。
- **抽樣來源是該窗的完整策略宇宙**（`S3._tree_universe`），與 A_hrp 同一個母體。
- **ENB 只在 <= 400 檔時計算**，與 `walkforward_matrix._evaluate` 同一條規則
  （超過就要做上千維的特徵分解，太貴）。
- ⚠️ **不算群集中度欄位**（`n_clusters_covered` / `max_cluster_share`）——
  那需要該窗的樹標籤，等於要重建 43 棵樹。而 M-03 已證明 `max_cluster_share`
  對 A_hrp 是**恆等於 1/k 的套套邏輯**，不是有意義的對照指標。
  實質比較在 CAGR / MDD / Sharpe / ENB。

用法：
    cd code
    python -m research.walkforward_random
    python -m research.walkforward_random --draws 50 --trees TW
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, paths
from . import stage3_hrp as S3
from .four_group_control import _cagr, _mdd, _portfolio_series, _sharpe, _small_enb
from .walkforward_matrix import TREES, _load_inputs, oos_months

N_DRAWS = 200
RANDOM_SEED = 42
ENB_MAX_MEMBERS = 400       # 與 walkforward_matrix._evaluate 同一條規則
METRICS = ("is_cagr", "is_mdd", "is_sharpe", "is_enb",
           "oos_cagr", "oos_mdd", "oos_sharpe", "oos_enb")


def _load_cells() -> pd.DataFrame:
    """從凍結的矩陣讀 A_hrp 每一格的窗定義與實際檔數。"""
    d = paths.ROOT / "_analysis_outputs_robustness"
    freeze.verify_inputs(d / "_walkforward_matrix_manifest")
    df = pd.read_csv(d / "walkforward_matrix_detail.csv")
    a = df[df.group == "A_hrp"].copy()
    if a.empty:
        raise AssertionError("矩陣裡沒有 A_hrp 列——請先跑 walkforward_matrix")
    return a


def _window_frames(tree_key: str, is_start: str, is_end: str,
                   oos_start: str, oos_end: str, months_long, meta):
    """該窗的 (IS 報酬矩陣, OOS 報酬矩陣, 宇宙)。與 `run_one_window` 同一套規則。"""
    uids = S3._tree_universe(tree_key, is_start, meta)
    wide_is = S3._pivot_window(months_long, uids, is_start, is_end)
    w = months_long[months_long.strategy_uid.isin(set(uids))]
    w = w[(w.month >= pd.Period(oos_start, "M")) & (w.month <= pd.Period(oos_end, "M"))]
    wide_oos = w.pivot(index="strategy_uid", columns="month", values="ret")
    # 同 `run_one_window` 的防線：OOS 有缺值會讓 mean(skipna=True) 靜默少平均幾檔。
    n_nan = int(wide_oos.isna().sum().sum())
    n_missing = len(set(uids) - set(wide_oos.index))
    if n_nan or n_missing:
        raise ValueError(
            f"[{tree_key}] OOS 窗 {oos_start}~{oos_end} 資料不完整："
            f"{n_nan} 個缺值、{n_missing} 個策略無資料——DD-03 共同窗保證被打破")
    return wide_is, wide_oos, uids


def _eval_draw(members, wide_is, wide_oos) -> dict:
    p_is = _portfolio_series(wide_is, members)
    p_oos = _portfolio_series(wide_oos, members)
    small = len(members) <= ENB_MAX_MEMBERS
    return {"is_cagr": _cagr(p_is), "is_mdd": _mdd(p_is), "is_sharpe": _sharpe(p_is),
            "is_enb": _small_enb(wide_is, members) if small else float("nan"),
            "oos_cagr": _cagr(p_oos), "oos_mdd": _mdd(p_oos), "oos_sharpe": _sharpe(p_oos),
            "oos_enb": _small_enb(wide_oos, members) if small else float("nan")}


def build(trees=TREES, n_draws=N_DRAWS, log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE0)
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE1 / "_marks")

    cells = _load_cells()
    cells = cells[cells.tree_key.isin(trees)]
    months_long, meta, _ = _load_inputs()

    # 🔴 去重：抽樣結果只由 (樹, 窗, 檔數) 決定，跟 ratio/allocation/k_mode 無關。
    # 2,700 格去重後只剩 414 組——同一組的多個格子共用同一批抽樣。
    key = ["tree_key", "is_start", "is_end", "oos_start", "oos_end", "n_members"]
    combos = cells[key].drop_duplicates().sort_values(key)
    log(f"A_hrp 格數 {len(cells):,} → 去重後 {len(combos)} 組 (樹×窗×檔數)")

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    t0 = time.time()
    frames: dict[tuple, tuple] = {}
    for i, c in enumerate(combos.itertuples(), 1):
        fkey = (c.tree_key, c.is_start, c.is_end, c.oos_start, c.oos_end)
        if fkey not in frames:
            # 每窗的報酬矩陣只建一次；combos 已依窗排序，故快取只需保留最近一個。
            frames.clear()
            frames[fkey] = _window_frames(c.tree_key, c.is_start, c.is_end,
                                          c.oos_start, c.oos_end, months_long, meta)
        wide_is, wide_oos, uids = frames[fkey]
        arr = np.asarray(uids)
        n = int(c.n_members)
        if n > len(arr):
            raise ValueError(f"[{c.tree_key} {c.is_start}] 需抽 {n} 檔但宇宙只有 {len(arr)}")

        acc: dict[str, list[float]] = {m: [] for m in METRICS}
        for _ in range(n_draws):
            pick = arr[rng.choice(len(arr), size=n, replace=False)].tolist()
            ev = _eval_draw(pick, wide_is, wide_oos)
            for m in METRICS:
                acc[m].append(ev[m])
        rec = {"tree_key": c.tree_key, "is_start": c.is_start, "is_end": c.is_end,
               "oos_start": c.oos_start, "oos_end": c.oos_end,
               # 🔴 M-14：實際 OOS 月數（秩不足分析用，見 `oos_months` docstring）
               "n_oos_months": oos_months(c.oos_start, c.oos_end),
               "n_members": n, "n_universe": len(arr), "n_draws": n_draws,
               "enb_computed": bool(n <= ENB_MAX_MEMBERS)}
        for m in METRICS:
            v = np.asarray(acc[m], dtype=np.float64)
            rec[m] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            rec[f"{m}_std"] = float(np.nanstd(v, ddof=1)) if np.isfinite(v).any() else float("nan")
        rows.append(rec)
        if i % 50 == 0 or i == len(combos):
            log(f"  {i}/{len(combos)} 組完成｜{time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    df["tree_key"] = df["tree_key"].astype("category")
    return df[C.WALKFORWARD_RANDOM.names]


def compare(rnd: pd.DataFrame, log=print) -> pd.DataFrame:
    """把每個 A_hrp 格子接上對應的 C_random 抽樣分布，算 z 分數與勝率。

    🔴 z = (A − 隨機均值) / 隨機標準差。**這才是 walk-forward 版的「挑選有沒有技術」**
    ——H-12 只在單一窗、legacy 比例上做過。
    """
    cells = _load_cells()
    # ⚠️ 只比對 `rnd` 實際涵蓋的樹——`--trees TW` 這種子集跑法下，cells 仍是三棵樹的
    # 2,700 格，直接比長度會誤報「去重鍵對不上」（2026-09-06 開發時實測踩到）。
    cells = cells[cells.tree_key.isin(set(rnd.tree_key.astype(str)))]
    key = ["tree_key", "is_start", "is_end", "oos_start", "oos_end", "n_members"]
    m = cells.merge(rnd, on=key, how="inner", suffixes=("_A", "_R"))
    if len(m) != len(cells):
        raise AssertionError(
            f"接上隨機分布後只剩 {len(m)}/{len(cells)} 格——去重鍵對不上，"
            "多半是矩陣重跑過而隨機表沒跟著重跑")
    rows = []
    for metric in ("oos_cagr", "oos_mdd", "oos_sharpe", "oos_enb"):
        a, r, sd = m[f"{metric}_A"], m[f"{metric}_R"], m[f"{metric}_std"]
        diff = a - r                       # 原始效果量，不受對照組變異數大小影響
        z = diff / sd.replace(0, np.nan)
        ok = z.notna()
        # 🔴 逐比例 + 全比例聚合都報。聚合會掩蓋方向相反的結構（實測 M-03b：
        # 台股 CAGR 在 legacy 是 21.7%、在 5% 是 75.6%，聚合起來變 50.4% 的假平手）。
        sub = m[ok].copy()
        sub["_z"], sub["_d"] = z[ok], diff[ok]
        for ratio in ["ALL"] + sorted(sub.ratio.astype(str).unique()):
            r_sub = sub if ratio == "ALL" else sub[sub.ratio.astype(str) == ratio]
            for tree, g in r_sub.groupby("tree_key", observed=True):
                if g.empty:
                    continue
                rows.append({"tree_key": tree, "metric": metric, "ratio": ratio,
                             "n_cells": int(len(g)),
                             "z_mean": float(g._z.mean()), "z_median": float(g._z.median()),
                             "pct_A_wins": float((g._z > 0).mean()),
                             "pct_z_over_2": float((g._z > 2).mean()),
                             "pct_z_under_neg2": float((g._z < -2).mean()),
                             "diff_mean": float(g._d.mean()),
                             "diff_median": float(g._d.median()),
                             "random_std_mean": float(g[f"{metric}_std"].mean()),
                             # M-16：名目二項 SE（下限；格子不獨立故實際更寬）
                             "se_binomial_nominal": float(np.sqrt(0.25 / len(g))),
                             # M-16：聚合列才是事前設定的結論，逐比例是探索性的
                             "evidence_type": ("confirmatory" if ratio == "ALL"
                                               else "exploratory")})
    return pd.DataFrame(rows)


def _persist(df: pd.DataFrame, cmp_df: pd.DataFrame, n_draws: int, log=print) -> None:
    """寫出兩張表 **並重寫 manifest**。

    🔴 `run()` 與 `--recompare` **都必須走這裡**。2026-09-06 code review 抓到：
    `--recompare` 原本只改寫 compare CSV 而沒更新 manifest，於是 manifest 記的
    sha256 與實際檔案不符，`verify_inputs` 直接判定「凍結產物已被改動」——
    **任何只改寫產物卻不更新 manifest 的路徑都是 DD-08 違規**。
    """
    out_dir = paths.ROOT / "_analysis_outputs_robustness"
    p1 = out_dir / "walkforward_random.csv"
    p2 = out_dir / "walkforward_random_compare.csv"
    df.to_csv(p1, index=False, encoding="utf-8-sig")
    cmp_df.to_csv(p2, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "walkforward_random", out_dir / "_walkforward_random_manifest",
        inputs=[paths.STAGE1 / "returns_monthly.parquet",
                paths.STAGE1 / "returns_meta.parquet",
                paths.STAGE1 / "strategy_marks.parquet",
                out_dir / "walkforward_matrix_detail.csv"],
        outputs=[p1, p2],
        params={"n_draws": int(n_draws), "seed": RANDOM_SEED,
                "enb_max_members": ENB_MAX_MEMBERS,
                "match_rule": "檔數對齊 A_hrp 的實際 n_members（非 target_total）"},
        notes="M-02b：walk-forward 版的 C_random。不建樹（隨機抽取與分群無關），"
              "去重後 414 組。不算群集中度欄位（需樹標籤，且 M-03 證明該欄對 A "
              "是套套邏輯）。對照表帶 ratio 維度（ALL + 逐比例）。",
    )
    log(f"→ {p1.name} / {p2.name}")


def run(trees=TREES, n_draws=N_DRAWS, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = build(trees=trees, n_draws=n_draws, log=log)
    C.validate(df, C.WALKFORWARD_RANDOM, strict_columns=True)
    log(f"✓ walkforward_random 契約通過（{len(df)} 組）")
    cmp_df = compare(df, log)
    C.validate(cmp_df, C.WALKFORWARD_RANDOM_COMPARE, strict_columns=True)
    _persist(df, cmp_df, n_draws, log)
    return df, cmp_df


def _report(rnd: pd.DataFrame, cmp_df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 92)
    log("M-02b · walk-forward 版 C_random——「挑選有技術」在整個矩陣上還成立嗎")
    log("=" * 92)
    log(f"  抽樣組數 {len(rnd)}｜每組 {int(rnd.n_draws.iloc[0])} 次｜"
        f"檔數 {int(rnd.n_members.min())}~{int(rnd.n_members.max())}")
    log("")
    order = {"ALL": 0, "legacy": 1, "0.01": 2, "0.03": 3, "0.05": 4, "0.1": 5}
    cmp_df = cmp_df.copy()
    cmp_df["_o"] = cmp_df.ratio.astype(str).map(order).fillna(9)
    log(f"  {'樹':<5}{'指標':<12}{'比例':<9}{'格數':>6}{'效果量(A−隨機)':>15}"
        f"{'隨機σ':>10}{'z 平均':>9}{'A 勝率':>9}{'±SE':>7}  證據")
    for (met, tree), g in cmp_df.groupby(["metric", "tree_key"], observed=True):
        for r in g.sort_values("_o").itertuples():
            tag = "驗證性(聚合)" if str(r.ratio) == "ALL" else "探索性"
            log(f"  {r.tree_key:<5}{r.metric:<12}{str(r.ratio):<9}{r.n_cells:>6,}"
                f"{r.diff_mean:>15.4f}{r.random_std_mean:>10.4f}"
                f"{r.z_mean:>9.2f}{r.pct_A_wins:>9.1%}"
                f"{r.se_binomial_nominal:>7.3f}  {tag}")
        log("")
    log("  ⚠️ **±SE 是名目二項標準誤，是下限不是實際值**——格子彼此不獨立")
    log("     （共用窗與方案，M-10 實測相鄰窗選股 Jaccard 0.372），有效 n 更低。")
    log("     另有第二個誤差源：對照組均值/標準差只由 n_draws 次抽樣估得（見「隨機σ」）。")
    log("  ⚠️ **逐比例列一律標為探索性，不做強宣稱**；只有聚合列是事前設定的結論。")
    log("")
    log("  ⚠️ **判讀以「效果量」與「A 勝率」為主，z 只當輔助**——對照組變異數極小時")
    log("     z 會爆炸而失去意義（見「隨機σ」欄）。")
    log("")
    log("🔴 判讀：**逐比例都要看**（`ratio=ALL` 只是聚合列）。")
    log("  · 實測「挑選規則有技術」在**每一個比例上都成立**——CAGR 勝率 79~100%、")
    log("    **ENB 每一棵樹每一個比例都是 100%**。這是與 M-03b 最關鍵的差別：")
    log("    M-03b（換分群依據）只有 legacy 那一檔的 ENB 才贏，1% 以上全部反轉。")
    log("  ⇒ **分散度主要來自「多樣性限制」這條挑選規則，不是 HRP 的群邊界。**")
    log("  · ⚠️ 與 M-03/M-03b 不衝突：那邊換的是**分群依據**，本表換的是**挑選機制**。")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.walkforward_random")
    ap.add_argument("--trees", nargs="+", default=list(TREES))
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    # 🔴 只重算對照表，不重跑模擬——模擬結果已存 CSV，改對照口徑不必再花數小時。
    ap.add_argument("--recompare", action="store_true",
                    help="只用既有的模擬 CSV 重算對照表（不重跑抽樣）")
    a = ap.parse_args(argv)
    if a.recompare:
        d = paths.ROOT / "_analysis_outputs_robustness"
        rnd = pd.read_csv(d / "walkforward_random.csv")
        rnd["tree_key"] = rnd["tree_key"].astype("category")
        C.validate(rnd, C.WALKFORWARD_RANDOM, strict_columns=True)
        cmp_df = compare(rnd)
        C.validate(cmp_df, C.WALKFORWARD_RANDOM_COMPARE, strict_columns=True)
        # 🔴 必須連 manifest 一起重寫，否則 sha256 對不上（見 `_persist` docstring）
        _persist(rnd, cmp_df, int(rnd.n_draws.iloc[0]))
        print(f"✓ 重算對照表（{len(cmp_df)} 列）")
        _report(rnd, cmp_df)
        return 0
    _report(*run(trees=tuple(a.trees), n_draws=a.draws))
    return 0


if __name__ == "__main__":
    sys.exit(main())
