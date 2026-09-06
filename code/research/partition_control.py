# -*- coding: utf-8 -*-
"""M-03 · 分群依據的對照實驗（2026-09-05）

🔴 **這支腳本要回答的問題**：H-12 證明了「A_hrp 贏過隨機**挑選**（C_random）」，
但那只證明「挑選有技術」，**沒有回答「技術是不是來自 HRP 這個分群」**。
A_hrp 的流程是「分成 k 群 → 每群挑 m 個品質最好且彼此夠不像的」，
其中「分成 k 群」這一步可以換成**任何**分群依據。若換成隨機分群也一樣好，
那 HRP 就只是個花俏的分層抽樣器，論文第三章的立論要大幅收斂。

---------------------------------------------------------------------------
三個分群依據（其餘**全部**釘死）
---------------------------------------------------------------------------
  A_hrp       H-11 凍結的 IS 樹 L1 分群（k = 6/7/3）——主線做法
  A1_fcombo   **因子構成**分群：對 f_combo 的因子組成做 ward 階層分群，切同一個 k
  A2_random   **隨機**分群：把同一批策略隨機打散成 k 群，且**群大小分布與 HRP 完全相同**
              N_DRAWS 次獨立抽樣，報平均 ± 標準差

🔴 **共同座標軸**（同 H-27 的原則）：三組共用同一個 IS/OOS 窗、同一批策略宇宙、
同一個品質分數（Calmar）、同一個 m=5／群、同一套 H-10 貪婪多樣性規則，
**組合檔數一律 = k × 5**。唯一的變因是**分群依據**。

⚠️ **多樣性門檻也必須跟著換**：`select_representatives` 的門檻是「該群自己的
   `avg_intra_corr`」。對照組必須用**它自己那個分群算出來的** avg_intra_corr，
   不能沿用 HRP 的——否則變成「用 HRP 的資訊去幫對照組」，對照就髒了。

---------------------------------------------------------------------------
A1_fcombo 的設計（為什麼不是直接用 218 群）
---------------------------------------------------------------------------
稽核清單原本建議「`A'_fcombo` 用 f_combo 的 218 群」。**這個設計是錯的**：
A_hrp 用的是 L1（k=6/7/3），拿 218 群去比，等於同時換掉「分群依據」與
「分群粗細」兩個變因，違反共同座標軸原則（H-27 為了同一個理由堅持
equal/proportional 必須釘死同一總量）。

正確做法是**先把 f_combo 聚合成同一個 k**：
  每個 f_combo → 38 維二元特徵向量（19 個因子 × {primary, secondary} 兩個位置），
  ward linkage → 切 k 群 → 策略依自己的 f_combo 對應到群。

**為什麼特徵只放因子身分、不放 band**：`f_combo` 的字面標籤含分位門檻
（如 `MOM_qb2of3`），但 band 是**同一個因子的鬆緊度**，不是「構成」。
兩個只差 band 的 f_combo 會拿到相同特徵向量而被併在一起——這是**刻意的**。
（實測 TW 218 個 f_combo 收斂成 96 個相異特徵向量，切 k=6 後最大群佔比 19.3%，
不退化。）

⚠️ **特徵刻意不含市場欄位**。因此在 XM 樹上，台股與美股只要因子組合相同就會落在
   同一群——A1_fcombo 在 XM **必然是市場混合的**。這不是缺陷，正是要看的對照：
   HRP 把市場分開、因子構成不分，兩者的 OOS 差距就是「市場邊界」這件事的價值。
   `market_purity` 欄位量化這一點。

---------------------------------------------------------------------------
A2_random 的設計（為什麼要對齊群大小）
---------------------------------------------------------------------------
隨機分群若用**等大小**，就同時換掉了「分群依據」與「群大小分布」兩件事——
HRP 的群大小極不平均（XM 是 6,679/4,102/4,259 這種），等大小隨機分群會讓
每群配到的 5 檔來自差異很大的母體規模。故 A2_random **沿用 HRP 的群大小 profile**，
只把成員隨機打散。

⚠️ 這也意味著 A2 用到了「HRP 群大小」這一點資訊。這是**保守的**設定：
   給對照組更多資訊，A_hrp 若仍贏，結論更強。

依賴：H-10（`cluster_representatives.select_representatives`）／
H-11（`_frozen/stage3_isoos` 的 IS 樹）／H-12（本模組直接重用其量測函式，
確保 A_hrp 的數字與 `four_group_control.csv` **逐位元一致**，見測試 ①）。

用法：
    cd code
    python -m research.partition_control
    python -m research.partition_control --trees TW
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from . import contracts as C
from . import freeze, paths
from .cluster_representatives import select_representatives
from .four_group_control import (M_PER_CLUSTER, _cagr, _cagr_matrix, _mdd,
                                 _mdd_matrix, _pivot_is, _pivot_oos,
                                 _portfolio_series, _sharpe, _small_enb)

TREES = ("TW", "US", "XM")
N_DRAWS = 200               # A2_random 的抽樣次數（同 H-12 的 C_random）
RANDOM_SEED = 42
PARTITIONS = ("A_hrp", "A1_fcombo", "A2_random")


# ============================================================================
# 三種分群依據
# ============================================================================

def fcombo_partition(uids: pd.Index, cand: pd.DataFrame, k: int) -> pd.Series:
    """依**因子構成**把策略分成 k 群。回傳 strategy_uid → cluster_id。

    38 維二元特徵（19 因子 × primary/secondary），ward linkage 切 k 群。
    詳細設計理由見模組 docstring。
    """
    sub = cand[cand[C.PK].isin(set(uids))]
    facs = sorted(set(sub.F1_factor.dropna()) | set(sub.F2_factor.dropna()))
    fidx = {f: i for i, f in enumerate(facs)}
    combos = sub.drop_duplicates("f_combo")[["f_combo", "F1_factor", "F2_factor"]]
    if len(combos) < k:
        raise ValueError(f"f_combo 只有 {len(combos)} 種，切不出 {k} 群")

    X = np.zeros((len(combos), 2 * len(facs)), dtype=np.float64)
    for i, r in enumerate(combos.itertuples()):
        X[i, fidx[r.F1_factor]] = 1.0
        if isinstance(r.F2_factor, str):                 # F2_empty 的策略只有 primary
            X[i, len(facs) + fidx[r.F2_factor]] = 1.0
    labels = fcluster(linkage(X, method="ward"), k, criterion="maxclust")
    cmap = dict(zip(combos.f_combo, labels))
    out = sub.set_index(C.PK).f_combo.map(cmap)
    return out.reindex(uids).astype(int)


def random_partition(uids: pd.Index, sizes: list[int], rng) -> pd.Series:
    """把策略隨機打散成群大小完全等於 `sizes` 的 k 群。回傳 uid → cluster_id。"""
    if sum(sizes) != len(uids):
        raise ValueError(f"群大小總和 {sum(sizes)} != 策略數 {len(uids)}")
    order = rng.permutation(len(uids))
    lab = np.empty(len(uids), dtype=int)
    at = 0
    for cid, n in enumerate(sizes, start=1):
        lab[order[at:at + n]] = cid
        at += n
    return pd.Series(lab, index=uids)


# ============================================================================
# 選代表（三組共用；門檻用各自分群自己的 avg_intra_corr）
# ============================================================================

def _avg_intra_corr(corr_full: np.ndarray, idx: list[int], rng=None,
                    max_pairs: int = 200_000) -> float:
    """一個群的群內平均相關。

    ⚠️ HRP 的 `avg_intra_corr` 是 `stage3_hrp._cluster_meta_and_corr` 算的全配對平均；
    對照組的群動輒六千檔（XM 隨機分群），全配對是 2,200 萬對 × k 群 × 200 抽樣，
    不可行。故超過 `max_pairs` 時改用**隨機配對抽樣**估計。
    無偏（每對被抽中的機率相同），標準誤在 20 萬對下 < 0.001，對門檻判斷無影響。
    """
    n = len(idx)
    if n < 2:
        return 1.0
    if n * (n - 1) // 2 <= max_pairs:
        sub = corr_full[np.ix_(idx, idx)]
        iu = np.triu_indices(n, k=1)
        return float(np.nanmean(sub[iu]))
    r = rng if rng is not None else np.random.default_rng(0)
    a = r.integers(0, n, size=max_pairs * 2)
    b = r.integers(0, n, size=max_pairs * 2)
    keep = a != b
    a, b = a[keep][:max_pairs], b[keep][:max_pairs]
    arr = np.asarray(idx)
    return float(np.nanmean(corr_full[arr[a], arr[b]]))


def pick_by_partition(cluster_map: pd.Series, quality_is: pd.Series,
                      corr_full: np.ndarray, pos: pd.Series,
                      rng=None) -> tuple[list[str], int]:
    """對任一分群套用 H-10 貪婪多樣性規則，每群挑 M_PER_CLUSTER 個。

    回傳 (名單, backfill 檔數)。**三組走的是完全同一條路徑**——這是共同座標軸的
    實作保證：A_hrp 與對照組的差異不可能來自挑選程式的差異。
    """
    picked_all: list[str] = []
    n_backfilled = 0
    for cid, g in cluster_map.groupby(cluster_map):
        members = g.index.tolist()
        qsub = quality_is.reindex(members).dropna()
        if qsub.empty:
            continue
        member_idx = [pos[u] for u in members if u in pos.index]
        avg_intra = _avg_intra_corr(corr_full, member_idx, rng)
        sub_corr = corr_full[np.ix_(member_idx, member_idx)]
        sub_index = pd.Index([u for u in members if u in pos.index])
        picked, backfilled = select_representatives(
            qsub, sub_corr, sub_index, M_PER_CLUSTER, avg_intra)
        picked_all += picked
        n_backfilled += len(backfilled)
    return picked_all, n_backfilled


# ============================================================================
# 評估
# ============================================================================

def _market_purity(cluster_map: pd.Series) -> float:
    """各群「單一市場佔比」的加權平均。1.0 = 每群都只含單一市場。

    只有 XM 樹有意義（TW/US 樹恆為 1.0）。這個欄位量化「HRP 是不是在分市場」。
    """
    mk = pd.Series(cluster_map.index, index=cluster_map.index).str.split("::").str[0]
    tot = 0.0
    for _, g in mk.groupby(cluster_map.reindex(mk.index)):
        tot += g.value_counts().iloc[0]
    return float(tot / len(mk))


def _evaluate(members: list[str], wide_is, wide_oos, hrp_map: pd.Series) -> dict:
    """一組成員的 IS/OOS 等權組合績效。

    ⚠️ `n_clusters_covered`／`max_cluster_share` 一律用 **HRP 的群定義**當尺，
    否則三組各用自己的尺，集中度數字之間無法比較（A2_random 用自己的隨機群去算，
    必然「橫跨 k 群、每群 5 檔」，看起來完美分散，那是套套邏輯）。
    """
    rep_is = _portfolio_series(wide_is, members)
    rep_oos = _portfolio_series(wide_oos, members)
    cl = hrp_map.reindex(members).dropna()
    vc = cl.value_counts()
    # 🔴 Calmar 必須**逐次**算完再取平均／標準差，不能事後拿平均 CAGR 除以平均 MDD
    # ——比值的期望值不等於期望值的比值，那樣算出來的 σ 是錯的，而 Calmar 正是本表
    # 的決勝指標（品質分數本身就是 Calmar）。
    c_is, c_oos = _cagr(rep_is), _cagr(rep_oos)
    m_is, m_oos = _mdd(rep_is), _mdd(rep_oos)
    return {
        "n_members": len(members),
        "is_cagr": c_is, "is_mdd": m_is, "is_sharpe": _sharpe(rep_is),
        "is_calmar": c_is / abs(m_is) if m_is else float("nan"),
        "is_enb": _small_enb(wide_is, members),
        "oos_cagr": c_oos, "oos_mdd": m_oos, "oos_sharpe": _sharpe(rep_oos),
        "oos_calmar": c_oos / abs(m_oos) if m_oos else float("nan"),
        "oos_enb": _small_enb(wide_oos, members),
        "n_hrp_clusters_covered": int(vc.shape[0]),
        "max_hrp_cluster_share": float(vc.iloc[0] / len(cl)) if len(cl) else float("nan"),
    }


# ============================================================================
# 主流程
# ============================================================================

def build(trees=TREES, n_draws=N_DRAWS, log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE0)
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE3_ISOOS)

    assign_is = pd.read_parquet(paths.STAGE3_ISOOS / "cluster_assign_IS.parquet")
    months_long = pd.read_parquet(paths.STAGE1 / "returns_monthly.parquet")
    cand = pd.read_parquet(paths.STAGE0 / "candidate_index.parquet")

    rows = []
    for tree_key in trees:
        t0 = time.time()
        a = assign_is[assign_is.tree_id == f"{tree_key}_normal_IS"][[C.PK, "cluster_L1"]]
        uids = pd.Index(a[C.PK])
        hrp_map = a.set_index(C.PK)["cluster_L1"]
        k = int(hrp_map.nunique())
        sizes = hrp_map.value_counts().sort_index().tolist()
        log(f"\n[{tree_key}] 策略 {len(uids):,}｜HRP k={k}｜群大小 {sizes}")

        wide_is = _pivot_is(months_long, uids, tree_key)
        wide_oos = _pivot_oos(months_long, uids, log)
        cagr_is = _cagr_matrix(wide_is)
        mdd_is = _mdd_matrix(wide_is)
        quality_is = (cagr_is / mdd_is.abs()).replace([np.inf, -np.inf], np.nan)

        corr_full = np.corrcoef(wide_is.to_numpy(dtype=np.float64))
        pos = pd.Series(range(len(wide_is.index)), index=wide_is.index)
        rng = np.random.default_rng(RANDOM_SEED)

        common = {"tree_key": tree_key, "n_universe": len(uids), "k": k,
                  "n_target": k * M_PER_CLUSTER}

        # ---- A_hrp ----
        mem, bf = pick_by_partition(hrp_map.reindex(uids), quality_is, corr_full, pos, rng)
        rows.append({**common, "partition": "A_hrp", "n_draws": 1, "n_backfilled": bf,
                     "market_purity": _market_purity(hrp_map.reindex(uids)),
                     **_evaluate(mem, wide_is, wide_oos, hrp_map), "note": None})
        log(f"  A_hrp      選出 {len(mem)} 檔（backfill {bf}）")

        # ---- A1_fcombo ----
        fmap = fcombo_partition(uids, cand, k)
        fsizes = fmap.value_counts().sort_index().tolist()
        mem, bf = pick_by_partition(fmap, quality_is, corr_full, pos, rng)
        rows.append({**common, "partition": "A1_fcombo", "n_draws": 1, "n_backfilled": bf,
                     "market_purity": _market_purity(fmap),
                     **_evaluate(mem, wide_is, wide_oos, hrp_map),
                     "note": f"因子構成 ward 分群，群大小 {fsizes}"})
        log(f"  A1_fcombo  選出 {len(mem)} 檔（backfill {bf}）｜群大小 {fsizes}")

        # ---- A2_random：n_draws 次 ----
        acc: dict[str, list[float]] = {}
        purity, bfs = [], []
        for _ in range(n_draws):
            rmap = random_partition(uids, sizes, rng)
            m2, bf2 = pick_by_partition(rmap, quality_is, corr_full, pos, rng)
            ev = _evaluate(m2, wide_is, wide_oos, hrp_map)
            for kk, vv in ev.items():
                acc.setdefault(kk, []).append(float(vv))
            purity.append(_market_purity(rmap))
            bfs.append(bf2)
        row = {**common, "partition": "A2_random", "n_draws": n_draws,
               "n_backfilled": int(round(float(np.mean(bfs)))),
               "market_purity": float(np.mean(purity)),
               "note": f"{n_draws} 次獨立抽樣(seed={RANDOM_SEED})，群大小 profile 沿用 HRP"}
        for kk, vals in acc.items():
            arr = np.asarray(vals, dtype=np.float64)
            row[kk] = float(np.nanmean(arr))
            if kk not in ("n_members", "n_hrp_clusters_covered"):
                row[f"{kk}_std"] = float(np.nanstd(arr))
        row["n_members"] = int(round(row["n_members"]))
        row["n_hrp_clusters_covered"] = int(round(row["n_hrp_clusters_covered"]))
        rows.append(row)
        log(f"  A2_random  {n_draws} 抽樣完成｜OOS CAGR "
            f"{row['oos_cagr']:.2%} ± {row['oos_cagr_std']:.2%}")
        log(f"[{tree_key}] {time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    for col in C.PARTITION_CONTROL.names:
        if col not in df.columns:
            df[col] = None
    df["tree_key"] = df["tree_key"].astype("category")
    df["partition"] = df["partition"].astype("category")
    return df[C.PARTITION_CONTROL.names]


def run(trees=TREES, n_draws=N_DRAWS, log=print) -> pd.DataFrame:
    df = build(trees=trees, n_draws=n_draws, log=log)
    C.validate(df, C.PARTITION_CONTROL, strict_columns=True)
    log(f"\n✓ partition_control 契約通過（{len(df)} 列）")

    out_dir = paths.ROOT / "_analysis_outputs_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "partition_control.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "partition_control", out_dir / "_partition_control_manifest",
        inputs=[paths.STAGE0 / "candidate_index.parquet",
                paths.STAGE1 / "returns_monthly.parquet",
                paths.STAGE3_ISOOS / "cluster_assign_IS.parquet"],
        outputs=[p],
        params={"m_per_cluster": M_PER_CLUSTER, "n_draws": n_draws,
                "random_seed": RANDOM_SEED, "partitions": list(PARTITIONS),
                "fcombo_features": "19 因子 × {primary, secondary} 的 38 維二元向量，"
                                   "ward linkage，不含 band、不含市場"},
        notes="M-03：只換分群依據、其餘全部釘死的對照實驗。回答「A_hrp 的優勢是否"
              "來自 HRP 分群本身」。A2_random 沿用 HRP 的群大小 profile（保守設定）。",
    )
    log(f"→ {p}")
    return df


def _report(df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 92)
    log("M-03 · 分群依據對照（其餘全部釘死：同窗、同宇宙、同品質分數、同 m=5、同挑選規則）")
    log("=" * 92)
    for t, g in df.groupby("tree_key", observed=True):
        k = int(g.k.iloc[0])
        log(f"\n【{t}】k={k}｜組合檔數 {int(g.n_target.iloc[0])}｜宇宙 {int(g.n_universe.iloc[0]):,}")
        log(f"  {'分群依據':<12}{'市場純度':>9}{'IS CAGR':>10}{'OOS CAGR':>10}"
            f"{'OOS MDD':>10}{'OOS Calmar':>12}{'OOS ENB':>9}{'跨HRP群':>8}")
        for r in g.itertuples():
            log(f"  {r.partition:<12}{r.market_purity:>9.3f}{r.is_cagr:>10.2%}"
                f"{r.oos_cagr:>10.2%}{r.oos_mdd:>10.2%}{r.oos_calmar:>12.3f}"
                f"{r.oos_enb:>9.2f}{r.n_hrp_clusters_covered:>8}")
        a = g[g.partition == "A_hrp"].iloc[0]
        r2 = g[g.partition == "A2_random"].iloc[0]
        for metric in ("oos_cagr", "oos_mdd", "oos_calmar", "oos_enb"):
            sd = float(r2[f"{metric}_std"])
            if sd > 0:
                z = (float(a[metric]) - float(r2[metric])) / sd
                log(f"    A_hrp vs A2_random 的 {metric}：{float(a[metric]):+.4f} vs "
                    f"{float(r2[metric]):+.4f} ± {sd:.4f} → **{z:+.2f}σ**")
    log("")
    log("判讀：A_hrp 若在 OOS 上相對 A2_random 沒有明顯的 σ 優勢，代表 HRP 分群本身")
    log("      沒有貢獻——那個流程只是「分層抽樣 + 品質排序 + 多樣性限制」，")
    log("      **分層依據換成隨機也一樣**，論文第三章的立論必須收斂。")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.partition_control")
    ap.add_argument("--trees", nargs="+", default=list(TREES))
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    a = ap.parse_args(argv)
    df = run(trees=tuple(a.trees), n_draws=a.draws)
    _report(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
