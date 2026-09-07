# -*- coding: utf-8 -*-
"""M-03b · walk-forward 版的 A2_random（隨機分群）對照（2026-09-06）

🔴 **這支腳本要補的洞**：M-03 證明「把分群依據換成隨機分群，A_hrp 在 OOS 報酬上
反而落敗 2.7~3.2σ」——**但那是單一窗、legacy 比例、fixed k**。

而 M-02b 已經把「挑選機制」的對照（`C_random`）擴到整個 2,700 格矩陣。
兩個結論的**證據強度不對等**：
  · 「挑選規則有用」  → 全矩陣證據 ✅
  · 「HRP 分群沒用」  → 單點證據 ⚠️
本模組把後者補齊。

---------------------------------------------------------------------------
🔴 為什麼這支比 M-02b 貴得多
---------------------------------------------------------------------------
`C_random` 是純隨機抽取，**跟分群無關**，所以 M-02b 可以完全繞過建樹。
`A2_random` 不行——它要：
  ① 該窗 HRP 樹的**群大小 profile**（隨機分群要對齊，見 M-03 的設計理由）
  ② 該窗的**相關矩陣**（貪婪多樣性規則要用）
  ③ 該窗的**品質分數**
所以 43 棵樹一棵都不能少。

---------------------------------------------------------------------------
🔴 精確的截斷優化（讓成本可行的關鍵）
---------------------------------------------------------------------------
瓶頸不是貪婪本身，是 `corr_full[np.ix_(idx, idx)]`——XM 的隨機群可達 6,679 檔，
一次切出 357MB 的子矩陣，乘上 k 個群 × N 次抽樣 × 2,700 格完全不可行。

觀察 `select_representatives` 的實作：主迴圈與 backfill **都是從品質最高往下掃**，
主迴圈 `len(picked) >= m` 就 break。**所以它的輸出只取決於 `order` 的前綴。**

⇒ 只要傳「品質前 Q 名」的子矩陣就好。**而且可以做到精確**：
   若截斷版**完全沒有 backfill**，代表 m 個名額都在前 Q 名內靠多樣性規則填滿，
   加入更低品質的候選不可能改變結果 → **與完整版逐位元相同**。
   若有 backfill，就把 Q 放大重試，直到無 backfill 或 Q 涵蓋整群。

`verify_truncation()` 對真實資料驗證這個等價性，也寫成測試斷言。

---------------------------------------------------------------------------
設計（其餘與 M-03 完全一致）
---------------------------------------------------------------------------
- 隨機分群的**群大小 profile 沿用該窗 HRP 樹**（保守設定：多給對照組資訊）
- 多樣性門檻用**該隨機群自己的 `avg_intra_corr`**（不可沿用 HRP 的，否則對照髒了）
- 配額用同一個 `allocate()`，總量釘死同一個 `target_total`
- 走**同一個** `select_representatives`，確保差異不可能來自挑選程式

用法：
    cd code
    python -m research.walkforward_partition --schemes A --trees TW --draws 5   # 計時
    python -m research.walkforward_partition                                    # 全跑
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, hrp, paths
from . import stage3_hrp as S3
from .cluster_representatives import select_representatives
from .four_group_control import (_cagr, _cagr_matrix, _mdd, _mdd_matrix,
                                 _portfolio_series, _sharpe, _small_enb)
from .partition_control import _avg_intra_corr, random_partition
from .walkforward_matrix import (ALLOCATIONS, K_MODES, LEVEL, RATIO_GRID, TREES,
                                 _load_inputs, _load_k_table, allocate,
                                 build_schemes, build_tree_for_window,
                                 oos_months, target_total, window_dates)

N_DRAWS = 30                 # 每格抽樣次數（2,700 格 × 30 = 81,000 次，聚合足夠）
RANDOM_SEED = 42
ENB_MAX_MEMBERS = 400        # 與 walkforward_matrix._evaluate 同一條規則
Q_INIT_FACTOR = 8            # 初始截斷 = max(Q_INIT_FACTOR × 配額, Q_MIN)
Q_MIN = 400
METRICS = ("oos_cagr", "oos_mdd", "oos_sharpe", "oos_enb")


def pick_truncated(members: list[str], quality: pd.Series, corr_full: np.ndarray,
                   pos: pd.Series, m: int, avg_intra: float,
                   q_init: int | None = None) -> tuple[list[str], int, int]:
    """在一個群內用 H-10 貪婪規則挑 m 個，**精確的品質前綴截斷**。

    回傳 (picked, n_backfilled, q_used)。等價性論證見模組 docstring：
    無 backfill ⇒ 與不截斷完全相同；有 backfill 就放大 Q 重試。
    """
    q_order = quality.reindex(members).dropna().sort_values(ascending=False)
    if q_order.empty:
        return [], 0, 0
    n = len(q_order)
    q = min(n, max(Q_MIN, (q_init or 0) or Q_INIT_FACTOR * m))
    while True:
        cand = q_order.index[:q]
        idx = [pos[u] for u in cand if u in pos.index]
        sub_index = pd.Index([u for u in cand if u in pos.index])
        sub_corr = corr_full[np.ix_(idx, idx)]
        picked, backfilled = select_representatives(
            q_order.reindex(sub_index), sub_corr, sub_index, m, avg_intra)
        # 無 backfill ⇒ m 個名額都在前 q 名內靠多樣性填滿 ⇒ 與完整版相同
        if not backfilled or q >= n:
            return picked, len(backfilled), q
        q = min(n, q * 4)


def pick_random_partition(uids: pd.Index, sizes: list[int], quality: pd.Series,
                          corr_full: np.ndarray, pos: pd.Series, quota: pd.Series,
                          rng) -> tuple[list[str], int]:
    """一次隨機分群 + 逐群挑代表。回傳 (名單, backfill 檔數)。"""
    rmap = random_partition(uids, sizes, rng)
    picked_all: list[str] = []
    n_bf = 0
    for cid, g in rmap.groupby(rmap):
        m = int(quota.get(int(cid), 0))
        if m <= 0:
            continue
        members = g.index.tolist()
        member_idx = [pos[u] for u in members if u in pos.index]
        avg_intra = _avg_intra_corr(corr_full, member_idx, rng)
        p, bf, _ = pick_truncated(members, quality, corr_full, pos, m, avg_intra)
        picked_all += p
        n_bf += bf
    return picked_all, n_bf


def verify_truncation(members, quality, corr_full, pos, m, avg_intra) -> bool:
    """對照「截斷版」與「完整版」的挑選結果是否逐位元相同（開發/測試用）。"""
    trunc, _, _ = pick_truncated(members, quality, corr_full, pos, m, avg_intra)
    q_order = quality.reindex(members).dropna().sort_values(ascending=False)
    idx = [pos[u] for u in q_order.index if u in pos.index]
    sub_index = pd.Index([u for u in q_order.index if u in pos.index])
    full, _ = select_representatives(q_order.reindex(sub_index),
                                     corr_full[np.ix_(idx, idx)], sub_index, m, avg_intra)
    return list(trunc) == list(full)


def _evaluate(members, wide_is, wide_oos) -> dict:
    p_is = _portfolio_series(wide_is, members)
    p_oos = _portfolio_series(wide_oos, members)
    small = len(members) <= ENB_MAX_MEMBERS
    return {"oos_cagr": _cagr(p_oos), "oos_mdd": _mdd(p_oos),
            "oos_sharpe": _sharpe(p_oos),
            "oos_enb": _small_enb(wide_oos, members) if small else float("nan"),
            "is_cagr": _cagr(p_is), "n_members": len(members)}


def run_one_window(tree_key: str, srow: pd.Series, tree: dict, months_long,
                   k_table, n_draws: int, log=print) -> list[dict]:
    is_start, is_end, oos_start, oos_end = window_dates(srow, tree_key)
    assign = tree["assign"]
    uids = pd.Index(assign[C.PK])
    wide_is = S3._pivot_window(months_long, uids, is_start, is_end)
    w = months_long[months_long.strategy_uid.isin(set(uids))]
    w = w[(w.month >= pd.Period(oos_start, "M")) & (w.month <= pd.Period(oos_end, "M"))]
    wide_oos = w.pivot(index="strategy_uid", columns="month", values="ret")
    if int(wide_oos.isna().sum().sum()) or (set(uids) - set(wide_oos.index)):
        raise ValueError(f"[{tree_key}] OOS 窗 {oos_start}~{oos_end} 資料不完整")

    cagr_is = _cagr_matrix(wide_is)
    mdd_is = _mdd_matrix(wide_is)
    quality_is = cagr_is / mdd_is.abs().replace(0, np.nan)
    n_uni = len(uids)
    corr_full = np.corrcoef(wide_is.to_numpy(dtype=np.float64))
    pos = pd.Series(range(len(wide_is.index)), index=wide_is.index)
    rng = np.random.default_rng(RANDOM_SEED)

    rows = []
    for k_mode in K_MODES:
        if k_mode == "fixed":
            labels = tree["assign"][f"cluster_{LEVEL}"].to_numpy()
        else:
            k_is = k_table.get((tree_key, is_start, is_end))
            if k_is is None:
                raise KeyError(f"k_stability 缺 ({tree_key}, {is_start}, {is_end})")
            # 🔴 對位不變式（與 `walkforward_matrix.run_one_window` 同一條防線）：
            # linkage 有 N−1 列合併紀錄，N 即建樹時的策略數。若它跟 wide_is 列數
            # 不符，代表兩者不是同一批策略，`cut_clusters` 的標籤會**靜默對錯**。
            # 本模組只用 labels 取「群大小」，錯了會在 `random_partition` 的總和
            # 檢查炸開——但那時的錯誤訊息看不出真正原因，故在源頭擋。
            n_leaf = len(tree["link"]) + 1
            if n_leaf != len(wide_is):
                raise AssertionError(
                    f"[{tree_key} {is_start}~{is_end}] linkage 葉節點數 {n_leaf} "
                    f"!= wide_is 列數 {len(wide_is)}——重切的群標籤會對錯策略")
            labels = hrp.cut_clusters(tree["link"], k_is)
        sizes_s = pd.Series(labels).value_counts().sort_index()
        k = len(sizes_s)
        sizes = sizes_s.tolist()

        for ratio in RATIO_GRID:
            tot = target_total(ratio, n_uni, k)
            for how in ALLOCATIONS:
                quota, _ = allocate(sizes_s, tot, how)
                # 隨機分群的群編號是 1..k（`random_partition` 的定義），
                # 而 quota 的索引是 HRP 的群編號。兩者只需一一對應，故按序重編。
                quota_r = pd.Series(quota.to_numpy(), index=range(1, k + 1))
                acc: dict[str, list[float]] = {m: [] for m in METRICS}
                bfs, nm = [], []
                for _ in range(n_draws):
                    mem, bf = pick_random_partition(uids, sizes, quality_is,
                                                    corr_full, pos, quota_r, rng)
                    ev = _evaluate(mem, wide_is, wide_oos)
                    for mkey in METRICS:
                        acc[mkey].append(ev[mkey])
                    bfs.append(bf); nm.append(ev["n_members"])
                rec = {"tree_key": tree_key, "scheme": srow.scheme,
                       "k_mode": k_mode, "ratio": str(ratio), "allocation": how,
                       "window_no": int(srow.window_no),
                       "is_start": is_start, "is_end": is_end,
                       "oos_start": oos_start, "oos_end": oos_end,
                       # 🔴 M-14：實際 OOS 月數。秩不足（N vs T）分析必須用它，
                       # 不能用方案的名目 oos_len_months（尾巴會併窗）。
                       "n_oos_months": oos_months(oos_start, oos_end),
                       "n_clusters": k, "target_total": tot, "n_draws": n_draws,
                       "n_members_mean": float(np.mean(nm)),
                       "n_backfilled_mean": float(np.mean(bfs)),
                       "enb_computed": bool(np.mean(nm) <= ENB_MAX_MEMBERS)}
                for mkey in METRICS:
                    v = np.asarray(acc[mkey], dtype=np.float64)
                    ok = np.isfinite(v)
                    rec[mkey] = float(v[ok].mean()) if ok.any() else float("nan")
                    rec[f"{mkey}_std"] = (float(v[ok].std(ddof=1))
                                          if ok.sum() > 1 else float("nan"))
                rows.append(rec)
    return rows


def build(trees=TREES, schemes_filter=None, n_draws=N_DRAWS, log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE0)
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE1 / "_marks")
    k_table = _load_k_table(log)
    schemes = build_schemes()
    if schemes_filter:
        schemes = schemes[schemes.scheme.isin(schemes_filter)]
    months_long, meta, f_combo_map = _load_inputs()
    log(f"窗口方案 {schemes.scheme.nunique()} 個｜總窗次 {len(schemes)}｜每格 {n_draws} 抽樣")

    all_rows = []
    t0 = time.time()
    for tree_key in trees:
        wanted: dict[tuple[str, str], None] = {}
        for _, srow in schemes.iterrows():
            s, e, _, _ = window_dates(srow, tree_key)
            wanted[(s, e)] = None
        log(f"  [{tree_key}] 需建樹 {len(wanted)} 棵")
        cache: dict[tuple[str, str], dict] = {}
        for is_start, is_end in wanted:
            tt = time.time()
            cache[(is_start, is_end)] = build_tree_for_window(
                tree_key, is_start, is_end, months_long, meta, f_combo_map, log)
            log(f"  [{tree_key}] 建樹 {is_start}~{is_end}  {time.time()-tt:.0f}s")
        for _, srow in schemes.iterrows():
            s, e, _, _ = window_dates(srow, tree_key)
            tt = time.time()
            all_rows += run_one_window(tree_key, srow, cache[(s, e)], months_long,
                                       k_table, n_draws, log)
            log(f"  [{tree_key}/{srow.scheme}/w{int(srow.window_no)}] "
                f"{time.time()-tt:.0f}s｜累計 {time.time()-t0:.0f}s")
        log("")

    df = pd.DataFrame(all_rows)
    for c in ("tree_key", "scheme", "k_mode", "ratio", "allocation"):
        df[c] = df[c].astype("category")
    return df[C.WALKFORWARD_PARTITION.names]


def compare(rnd: pd.DataFrame, log=print) -> pd.DataFrame:
    """每個 A_hrp 格子相對其隨機分群抽樣分布的 z 分數。

    🔴 z = (A_hrp − 隨機分群均值) / 隨機分群標準差。
    **這是 walk-forward 版的「HRP 分群本身有沒有貢獻」**——M-03 只在單一窗做過。
    """
    d = paths.ROOT / "_analysis_outputs_robustness"
    freeze.verify_inputs(d / "_walkforward_matrix_manifest")
    det = pd.read_csv(d / "walkforward_matrix_detail.csv")
    a = det[det.group == "A_hrp"].copy()
    a = a[a.tree_key.isin(set(rnd.tree_key.astype(str)))
          & a.scheme.isin(set(rnd.scheme.astype(str)))].copy()
    key = ["tree_key", "scheme", "k_mode", "ratio", "allocation", "window_no"]
    # ⚠️ 對 `rnd` 取副本再轉型——原本直接就地改呼叫端的 DataFrame，
    # `run()` 的順序是 validate → compare → persist，被改過 dtype 的 df 會流進
    # `_persist` 寫檔。CSV 輸出雖然相同，但「函式偷偷改掉參數」是不該留的副作用
    # （2026-09-06 code review）。
    rnd = rnd.copy()
    for df_ in (a, rnd):
        for c in key:
            df_[c] = df_[c].astype(str)
    m = a.merge(rnd, on=key, how="inner", suffixes=("_A", "_R"))
    if len(m) != len(a):
        raise AssertionError(f"接上隨機分群後只剩 {len(m)}/{len(a)} 格——格子鍵對不上")
    rows = []
    for metric in METRICS:
        diff = m[f"{metric}_A"] - m[f"{metric}_R"]   # 原始效果量，判讀以此為主
        z = diff / m[f"{metric}_std"].replace(0, np.nan)
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
    `--recompare` 原本只改寫 compare CSV 而沒更新 manifest，manifest 記的 sha256
    與實際檔案不符，`verify_inputs` 直接判定「凍結產物已被改動」——
    **任何只改寫產物卻不更新 manifest 的路徑都是 DD-08 違規**。
    """
    out_dir = paths.ROOT / "_analysis_outputs_robustness"
    p1 = out_dir / "walkforward_partition.csv"
    p2 = out_dir / "walkforward_partition_compare.csv"
    df.to_csv(p1, index=False, encoding="utf-8-sig")
    cmp_df.to_csv(p2, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "walkforward_partition", out_dir / "_walkforward_partition_manifest",
        inputs=[paths.STAGE0 / "candidate_index.parquet",
                paths.STAGE1 / "returns_monthly.parquet",
                paths.STAGE1 / "returns_meta.parquet",
                paths.STAGE1 / "strategy_marks.parquet",
                out_dir / "walkforward_matrix_detail.csv",
                out_dir / "k_stability.csv"],
        outputs=[p1, p2],
        params={"n_draws": int(n_draws), "seed": RANDOM_SEED,
                "enb_max_members": ENB_MAX_MEMBERS,
                "q_init_factor": Q_INIT_FACTOR, "q_min": Q_MIN,
                "truncation": "品質前綴截斷；無 backfill 即與完整版等價，否則放大重試"},
        notes="M-03b：walk-forward 版的 A2_random（隨機分群）。群大小 profile 沿用該窗 "
              "HRP 樹，多樣性門檻用隨機群自己的 avg_intra_corr，走同一個 "
              "select_representatives。對照表帶 ratio 維度（ALL + 逐比例）——"
              "聚合列會掩蓋方向相反的結構，判讀必須逐比例。",
    )
    log(f"→ {p1.name} / {p2.name}")


def run(trees=TREES, schemes_filter=None, n_draws=N_DRAWS,
        log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = build(trees=trees, schemes_filter=schemes_filter, n_draws=n_draws, log=log)
    C.validate(df, C.WALKFORWARD_PARTITION, strict_columns=True)
    log(f"✓ walkforward_partition 契約通過（{len(df):,} 格）")
    cmp_df = compare(df, log)
    C.validate(cmp_df, C.WALKFORWARD_RANDOM_COMPARE, strict_columns=True)
    _persist(df, cmp_df, n_draws, log)
    return df, cmp_df


def _report(rnd: pd.DataFrame, cmp_df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 92)
    log("M-03b · walk-forward 版隨機分群——「HRP 分群本身有沒有貢獻」")
    log("=" * 92)
    log(f"  格數 {len(rnd):,}｜每格 {int(rnd.n_draws.iloc[0])} 次抽樣")
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
    log("🔴 判讀：**必須逐比例看，聚合列會掩蓋方向相反的結構**。")
    log("  · M-03（單一窗、legacy 比例）的結論在 legacy 那一檔上成立，")
    log("    但**不能外推到其他比例**——實測台股 CAGR 在 legacy 是 A 勝率 21.7%、")
    log("    在 5% 卻是 75.6%，聚合起來變成 50.4% 的假平手。")
    log("  · ⚠️ 與 M-02b 不衝突：那張表換的是**挑選機制**，本表換的是**分群依據**。")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.walkforward_partition")
    ap.add_argument("--trees", nargs="+", default=list(TREES))
    ap.add_argument("--schemes", nargs="+")
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    # 🔴 只重算對照表，不重跑模擬——模擬結果已存 CSV，改對照口徑不必再花數小時。
    ap.add_argument("--recompare", action="store_true",
                    help="只用既有的模擬 CSV 重算對照表（不重跑抽樣）")
    a = ap.parse_args(argv)
    if a.recompare:
        d = paths.ROOT / "_analysis_outputs_robustness"
        rnd = pd.read_csv(d / "walkforward_partition.csv")
        for c in ("tree_key", "scheme", "k_mode", "ratio", "allocation"):
            rnd[c] = rnd[c].astype("category")
        C.validate(rnd, C.WALKFORWARD_PARTITION, strict_columns=True)
        cmp_df = compare(rnd)
        C.validate(cmp_df, C.WALKFORWARD_RANDOM_COMPARE, strict_columns=True)
        # 🔴 必須連 manifest 一起重寫，否則 sha256 對不上（見 `_persist` docstring）
        _persist(rnd, cmp_df, int(rnd.n_draws.iloc[0]))
        print(f"✓ 重算對照表（{len(cmp_df)} 列）")
        _report(rnd, cmp_df)
        return 0
    _report(*run(trees=tuple(a.trees), schemes_filter=a.schemes, n_draws=a.draws))
    return 0


if __name__ == "__main__":
    sys.exit(main())
