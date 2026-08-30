# -*- coding: utf-8 -*-
"""H-12 · 四組對照實驗（開發待辦追蹤.md 第四階段）

老師的驗證題：「選30多支 vs 狂灑下去會不會一樣」。四組：

  A_hrp       HRP跨群選代表——沿用H-10的貪婪多樣性規則（品質=Calmar比率、門檻=
              該群自己的群內平均相關），m=5／群，跟H-10驗證過的設定一致
  B_all       全部灑——上界，資金不現實但當對照
  C_random    隨機選同樣數量（跟A同一個N）——200次獨立抽樣，報告平均值+標準差，
              不是單一次的運氣
  D_top_cagr  純CAGR前N名——H-10警告過的陷阱，這裡故意在**投組層級**重現一次
  E_top_calmar 純Calmar前N名、**不設多樣性限制**（2026-08-30新增）——A跟D的差異
              同時混了兩件事（品質指標CAGR vs Calmar、有沒有多樣性限制），沒辦法
              乾淨拆開。E只換掉多樣性限制（跟A同樣的Calmar品質指標，但不檢查
              彼此相關），A vs E 才能單獨看出「多樣性限制」本身的貢獻；D vs E
              則能單獨看出「品質指標選哪個（CAGR vs Calmar）」的貢獻。

🔴 **原始結果的診斷**：2026-08-30初版四組跑完後發現，D組在美股/跨市場OOS大幅
贏過A組，一度看似推翻「歷史贏家過擬合」的假設——但查證D組實際選出的策略發現，
**XM的D組15檔全部集中在同一個HRP群、美股D組35檔有19檔(54%)集中在單一群**，
不是分散的組合，是一個集中賭注剛好在OOS期間表現好。E組就是為了把這個現象
獨立驗證：如果換成同樣不設多樣性限制的Calmar排序，是否也一樣集中、也一樣
呈現「OOS意外變好」的樣態——如果是，代表問題出在「不設多樣性限制」本身，
跟挑選標準是CAGR還是Calmar無關。

🔴 **四組的挑選依據只能用IS窗資訊**（品質分數、群定義全部來自H-11凍結的IS樹），
不可用任何OOS資訊，否則整個「測試泛化能力」的實驗就沒有意義。挑好成員後，四組
portfolio（一律等權，S-04定案）分別在IS窗跟OOS窗**各自獨立**評估：CAGR/MDD/Sharpe/
有效賭注數(ENB)。

**老師的假設**：D組最可能IS漂亮、OOS雪崩——若成立，直接證明「只挑歷史贏家會過擬合」，
也是這個實驗存在的意義。

依賴：H-09（`hrp.effective_number_of_bets`）／H-10（`cluster_representatives.
select_representatives`，本模組直接重用同一個函式）／H-11（IS/OOS窗定義、IS窗凍結
的群定義`_frozen/stage3_isoos/cluster_assign_IS.parquet`）。

⚠️ **B組（全部灑）的ENB沿用H-09已算好的`enb_raw`**（全時間窗2007-2025算出的），
不重新用IS/OOS各自的月份重算——B的N高達六七千至一萬五千，重算一次ENB要跑一次
N×N的特徵分解（XM單次約190秒），IS+OOS兩次對三棵樹合計要8分鐘以上，而B本身
又只是「不現實但當對照」的上界，不值得為它重複這筆昂貴運算。A/C/D子集只有
15~35檔，ENB重算很便宜，皆有做period-specific版本。

用法：
    cd code
    python -m research.four_group_control
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

TREES = ("TW", "US", "XM")
M_PER_CLUSTER = 5           # 跟H-10驗證過的設定一致（16/16群都改善）
N_RANDOM_DRAWS = 200
RANDOM_SEED = 42


# ============================================================================
# 基礎統計量（equal-weight portfolio，S-04定案）
# ============================================================================

def _cagr(monthly_ret: pd.Series) -> float:
    n = len(monthly_ret)
    if n == 0:
        return float("nan")
    cum = float((1 + monthly_ret).prod())
    return cum ** (12 / n) - 1


def _mdd(monthly_ret: pd.Series) -> float:
    if len(monthly_ret) == 0:
        return float("nan")
    cum = (1 + monthly_ret).cumprod()
    return float((cum / cum.cummax() - 1).min())


def _sharpe(monthly_ret: pd.Series) -> float:
    if len(monthly_ret) < 2:
        return float("nan")
    std = monthly_ret.std(ddof=0)
    if std == 0:
        return float("nan")
    return float(monthly_ret.mean() / std * np.sqrt(12))


def _cagr_matrix(wide: pd.DataFrame) -> pd.Series:
    """向量化版：對整張(策略×月)矩陣一次算完每列的CAGR。`.apply(axis=1)`在
    6,000~15,000列的規模下逐列呼叫Python函式太慢，這裡改用純numpy矩陣運算。
    """
    arr = wide.to_numpy(dtype=np.float64)
    n = arr.shape[1]
    cum_final = np.prod(1.0 + arr, axis=1)
    return pd.Series(cum_final ** (12.0 / n) - 1.0, index=wide.index)


def _mdd_matrix(wide: pd.DataFrame) -> pd.Series:
    """向量化版，理由同 `_cagr_matrix`。"""
    arr = wide.to_numpy(dtype=np.float64)
    cum = np.cumprod(1.0 + arr, axis=1)
    running_max = np.maximum.accumulate(cum, axis=1)
    return pd.Series((cum / running_max - 1.0).min(axis=1), index=wide.index)


def _portfolio_series(wide: pd.DataFrame, members: list[str]) -> pd.Series:
    """等權組合月報酬（S-04：權重定案等權）。"""
    avail = [m for m in members if m in wide.index]
    return wide.loc[avail].mean(axis=0)


def _small_enb(wide: pd.DataFrame, members: list[str]) -> float:
    """子集（<=數十檔）的有效獨立賭注數，直接對子集自己的相關矩陣做特徵分解——
    矩陣小(<=35x35)，不需要B組那種昂貴的全宇宙特徵分解。
    """
    avail = [m for m in members if m in wide.index]
    if len(avail) < 2:
        return 1.0
    sub = wide.loc[avail].to_numpy(dtype=np.float64)
    corr = np.corrcoef(sub)
    return hrp.effective_number_of_bets(corr)


# ============================================================================
# 資料準備
# ============================================================================

def _pivot_is(months_long: pd.DataFrame, uids: pd.Index, tree_key: str) -> pd.DataFrame:
    is_start, is_end = C.HRP_IS_WINDOWS[tree_key]
    return S3._pivot_window(months_long, uids, is_start, is_end)


def _pivot_oos(months_long: pd.DataFrame, uids: pd.Index, log=print) -> pd.DataFrame:
    oos_start, oos_end = C.HRP_OOS_WINDOW
    w = months_long[months_long.strategy_uid.isin(set(uids))]
    w = w[(w.month >= pd.Period(oos_start, "M")) & (w.month <= pd.Period(oos_end, "M"))]
    wide = w.pivot(index="strategy_uid", columns="month", values="ret")
    missing = set(uids) - set(wide.index)
    if missing:
        log(f"  ⚠️ {len(missing)}個策略OOS窗無資料，該策略在OOS評估中自動排除")
    return wide


# ============================================================================
# 四組挑選（全部只吃IS窗資訊）
# ============================================================================

def _pick_group_a(assign_tree: pd.DataFrame, meta_tree: pd.DataFrame, wide_is: pd.DataFrame,
                  quality_is: pd.Series, log=print) -> list[str]:
    """A_hrp：沿用H-10的貪婪多樣性規則，逐群挑m=5個代表。"""
    pos = pd.Series(range(len(wide_is.index)), index=wide_is.index)
    corr_full = np.corrcoef(wide_is.to_numpy(dtype=np.float64))
    picked_all: list[str] = []
    for cid, g in assign_tree.groupby("cluster_L1"):
        members = g[C.PK].tolist()
        qsub = quality_is.reindex(members).dropna()
        row = meta_tree[meta_tree.cluster_id == int(cid)]
        avg_intra = float(row["avg_intra_corr"].iloc[0]) if len(row) and pd.notna(row["avg_intra_corr"].iloc[0]) else 1.0
        member_idx = [pos[u] for u in members]
        sub_corr = corr_full[np.ix_(member_idx, member_idx)]
        sub_index = pd.Index(members)
        picked, _ = select_representatives(qsub, sub_corr, sub_index, M_PER_CLUSTER, avg_intra)
        picked_all += picked
    log(f"  A_hrp：{assign_tree.cluster_L1.nunique()}群 × 目標{M_PER_CLUSTER}／群 = "
        f"實際選出 {len(picked_all)} 檔")
    return picked_all


def _pick_group_d(cagr_is: pd.Series, n_target: int) -> list[str]:
    """D_top_cagr：純IS期間CAGR排序前N名，完全不看群結構——H-10警告過的陷阱。"""
    return cagr_is.sort_values(ascending=False).index[:n_target].tolist()


def _pick_group_e(quality_is: pd.Series, n_target: int) -> list[str]:
    """E_top_calmar：純IS期間Calmar排序前N名，**不設多樣性限制**（跟A同一個品質
    指標，但不檢查彼此相關）。跟D對照可以拆出「品質指標選哪個」的貢獻；跟A對照
    可以拆出「多樣性限制」本身的貢獻。"""
    return quality_is.dropna().sort_values(ascending=False).index[:n_target].tolist()


def _cluster_spread(members: list[str], cluster_map: pd.Series) -> tuple[int, float]:
    """這組成員橫跨幾個IS群、最大單一群佔比多少——2026-08-30查出D組在美股/跨市場
    的OOS「勝利」其實是集中在單一群的賭注後新增，把手動debug才看得到的事實
    變成每次跑都自動輸出的欄位（見 FOUR_GROUP_CONTROL schema註解）。
    """
    cl = cluster_map.reindex(members).dropna()
    if len(cl) == 0:
        return 0, float("nan")
    counts = cl.value_counts()
    return int(counts.shape[0]), float(counts.iloc[0] / len(cl))


def _pick_group_c_draws(universe: list[str], n_target: int, n_draws: int, seed: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    universe_arr = np.array(universe)
    draws = []
    for _ in range(n_draws):
        idx = rng.choice(len(universe_arr), size=n_target, replace=False)
        draws.append(universe_arr[idx].tolist())
    return draws


# ============================================================================
# 主流程
# ============================================================================

def build(trees=TREES, log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE3_ISOOS)
    assign_is = pd.read_parquet(paths.STAGE3_ISOOS / "cluster_assign_IS.parquet")
    meta_is = pd.read_parquet(paths.STAGE3_ISOOS / "cluster_meta_IS.parquet")
    meta_is_l1 = meta_is[meta_is.level == "L1"]
    months_long = pd.read_parquet(paths.STAGE1 / "returns_monthly.parquet")
    enb_ref = pd.read_csv(paths.ROOT / "_analysis_outputs_robustness" / "effective_number_of_bets.csv")

    rows = []
    for tree_key in trees:
        tree_id_is = f"{tree_key}_normal_IS"
        t0 = time.time()
        a = assign_is[assign_is.tree_id == tree_id_is][[C.PK, "cluster_L1"]]
        meta_tree = meta_is_l1[meta_is_l1.tree_id == tree_id_is]
        uids = pd.Index(a[C.PK])
        log(f"\n[{tree_key}] 策略數 {len(uids):,}｜群數 {a.cluster_L1.nunique()}")

        wide_is = _pivot_is(months_long, uids, tree_key)
        wide_oos = _pivot_oos(months_long, uids, log)

        cagr_is = _cagr_matrix(wide_is)
        mdd_is = _mdd_matrix(wide_is)
        quality_is = (cagr_is / mdd_is.abs()).replace([np.inf, -np.inf], np.nan)

        k = a.cluster_L1.nunique()
        n_target = k * M_PER_CLUSTER
        cluster_map = a.set_index(C.PK)["cluster_L1"]
        log(f"  目標組合大小 n_target = {k}群 × {M_PER_CLUSTER} = {n_target}")

        picked_a = _pick_group_a(a, meta_tree, wide_is, quality_is, log)
        picked_d = _pick_group_d(cagr_is, n_target)
        picked_e = _pick_group_e(quality_is, n_target)
        picked_b = uids.tolist()
        draws_c = _pick_group_c_draws(uids.tolist(), n_target, N_RANDOM_DRAWS, RANDOM_SEED)
        log(f"  E_top_calmar：不設多樣性限制，純Calmar前{n_target}名，實際選出 {len(picked_e)} 檔")

        # ---- A / D / E / B：單一次結果 ----
        for group, members in (("A_hrp", picked_a), ("D_top_cagr", picked_d),
                               ("E_top_calmar", picked_e), ("B_all", picked_b)):
            rep_is = _portfolio_series(wide_is, members)
            rep_oos = _portfolio_series(wide_oos, members)
            n_cl, max_share = _cluster_spread(members, cluster_map)
            row = {"tree_key": tree_key, "group": group, "n_members": len(members), "n_draws": 1,
                  "is_cagr": _cagr(rep_is), "is_mdd": _mdd(rep_is), "is_sharpe": _sharpe(rep_is),
                  "oos_cagr": _cagr(rep_oos), "oos_mdd": _mdd(rep_oos), "oos_sharpe": _sharpe(rep_oos),
                  "n_clusters_covered": n_cl, "max_cluster_share": round(max_share, 4)}
            if group == "B_all":
                ref = enb_ref[enb_ref.tree_id == f"{tree_key}_normal"]
                enb_full = float(ref["enb_raw"].iloc[0]) if len(ref) else None
                row["is_enb"] = enb_full
                row["oos_enb"] = enb_full
                row["note"] = "ENB沿用H-09全時間窗(2007-2025)的enb_raw，非IS/OOS各自重算，見模組docstring"
            else:
                row["is_enb"] = _small_enb(wide_is, members)
                row["oos_enb"] = _small_enb(wide_oos, members)
                row["note"] = None
            rows.append(row)

        # ---- C：200次抽樣，彙總 mean/std ----
        metrics = {"is_cagr": [], "is_mdd": [], "is_sharpe": [], "is_enb": [],
                  "oos_cagr": [], "oos_mdd": [], "oos_sharpe": [], "oos_enb": []}
        n_cl_draws, max_share_draws = [], []
        for draw in draws_c:
            rep_is = _portfolio_series(wide_is, draw)
            rep_oos = _portfolio_series(wide_oos, draw)
            metrics["is_cagr"].append(_cagr(rep_is)); metrics["is_mdd"].append(_mdd(rep_is))
            metrics["is_sharpe"].append(_sharpe(rep_is)); metrics["is_enb"].append(_small_enb(wide_is, draw))
            metrics["oos_cagr"].append(_cagr(rep_oos)); metrics["oos_mdd"].append(_mdd(rep_oos))
            metrics["oos_sharpe"].append(_sharpe(rep_oos)); metrics["oos_enb"].append(_small_enb(wide_oos, draw))
            n_cl, max_share = _cluster_spread(draw, cluster_map)
            n_cl_draws.append(n_cl); max_share_draws.append(max_share)
        row = {"tree_key": tree_key, "group": "C_random", "n_members": n_target, "n_draws": N_RANDOM_DRAWS,
              "n_clusters_covered": int(round(float(np.mean(n_cl_draws)))),
              "max_cluster_share": round(float(np.mean(max_share_draws)), 4),
              "note": f"{N_RANDOM_DRAWS}次獨立抽樣(seed={RANDOM_SEED})的平均值±標準差；"
                      f"n_clusters_covered/max_cluster_share為200次抽樣的平均"}
        for k_metric, vals in metrics.items():
            arr = np.array(vals, dtype=np.float64)
            row[k_metric] = float(np.nanmean(arr))
            row[f"{k_metric}_std"] = float(np.nanstd(arr))
        rows.append(row)

        log(f"[{tree_key}] 完成，{time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    for col in C.FOUR_GROUP_CONTROL.names:
        if col not in df.columns:
            df[col] = None
    df["tree_key"] = df["tree_key"].astype("category")
    df["group"] = df["group"].astype("category")
    return df[C.FOUR_GROUP_CONTROL.names]


def run(trees=TREES, log=print) -> pd.DataFrame:
    df = build(trees=trees, log=log)
    C.validate(df, C.FOUR_GROUP_CONTROL, strict_columns=True)
    log("\n✓ four_group_control 契約通過")

    out_dir = paths.ROOT / "_analysis_outputs_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "four_group_control.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "four_group_control", out_dir / "_four_group_control_manifest",
        inputs=[paths.STAGE3_ISOOS / "cluster_assign_IS.parquet",
               paths.STAGE3_ISOOS / "cluster_meta_IS.parquet",
               paths.STAGE1 / "returns_monthly.parquet",
               out_dir / "effective_number_of_bets.csv"],
        outputs=[p],
        params={"m_per_cluster": M_PER_CLUSTER, "n_random_draws": N_RANDOM_DRAWS,
               "random_seed": RANDOM_SEED},
        notes="H-12：四組對照(A_hrp/B_all/C_random/D_top_cagr)，挑選依據只用IS窗資訊，"
              "IS/OOS各自評估CAGR/MDD/Sharpe/ENB。B組ENB沿用H-09全時間窗數字，未按期間重算。",
    )
    log(f"→ {p}")
    return df


def _report(df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 90)
    log("H-12 · 四組對照實驗 驗收摘要")
    log("=" * 90)
    for tk, g in df.groupby("tree_key", observed=True):
        log(f"\n[{tk}]")
        show = g[["group", "n_members", "n_clusters_covered", "max_cluster_share",
                  "is_cagr", "oos_cagr", "is_mdd", "oos_mdd",
                  "is_sharpe", "oos_sharpe", "is_enb", "oos_enb"]].copy()
        show["max_cluster_share"] = (show["max_cluster_share"] * 100).round(1)
        for c in ("is_cagr", "oos_cagr", "is_mdd", "oos_mdd"):
            show[c] = (show[c] * 100).round(2)
        for c in ("is_sharpe", "oos_sharpe", "is_enb", "oos_enb"):
            show[c] = show[c].round(2)
        log(show.to_string(index=False))

        a_row = g[g.group == "A_hrp"].iloc[0]
        d_row = g[g.group == "D_top_cagr"].iloc[0]
        e_row = g[g.group == "E_top_calmar"].iloc[0]
        if pd.notna(d_row.is_cagr) and pd.notna(d_row.oos_cagr):
            gap_d = d_row.is_cagr - d_row.oos_cagr
            gap_a = a_row.is_cagr - a_row.oos_cagr
            gap_e = e_row.is_cagr - e_row.oos_cagr
            log(f"  IS→OOS CAGR衰退：D組(純CAGR) {gap_d:+.1%}｜E組(純Calmar,不設多樣性) {gap_e:+.1%}"
                f"｜A組(Calmar+多樣性限制) {gap_a:+.1%}")
            log(f"  集中度：D組跨{int(d_row.n_clusters_covered)}群(最大群佔比{d_row.max_cluster_share:.1%})"
                f"｜E組跨{int(e_row.n_clusters_covered)}群(最大群佔比{e_row.max_cluster_share:.1%})"
                f"｜A組跨{int(a_row.n_clusters_covered)}群(最大群佔比{a_row.max_cluster_share:.1%}，"
                f"應接近1/{int(a_row.n_clusters_covered)}＝完全均勻)")
            # D vs E：品質指標(CAGR vs Calmar)的貢獻；A vs E：多樣性限制本身的貢獻
            log(f"  拆解：D→E只換品質指標(CAGR→Calmar)，OOS CAGR變化"
                f"{(e_row.oos_cagr - d_row.oos_cagr):+.1%}；"
                f"E→A只加多樣性限制，OOS CAGR變化{(a_row.oos_cagr - e_row.oos_cagr):+.1%}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.four_group_control")
    ap.add_argument("--trees", nargs="+", default=list(TREES))
    a = ap.parse_args(argv)
    t0 = time.time()
    df = run(trees=a.trees)
    print(f"\n總耗時 {time.time()-t0:.0f}s")
    _report(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
