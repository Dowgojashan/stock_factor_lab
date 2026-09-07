# -*- coding: utf-8 -*-
"""M-01 · 市場 beta 基準與殘差相關結構（2026-09-05）

🔴 **這支腳本要回答的問題**：`研究框架總覽_v10.md` §3.5② 宣稱
「分散效果的來源是市場邊界，不是因子邊界」，依據是 XM 樹跨市場配對相關 0.542、
同市場 0.78~0.94。**但從來沒有人算過「台美兩個市場基準本身的相關係數」**——
若市場基準之間就是 0.55，那 0.542 這個數字只是**國際分散的重現**，
跟策略選擇一點關係都沒有。

所有策略都是多頭選股組合，彼此相關必然由共同市場 beta 主導。
`stage3_hrp._build_tree` 直接對原始月報酬呼叫 `np.corrcoef`，沒有做任何 beta 調整。

---------------------------------------------------------------------------
兩個部分
---------------------------------------------------------------------------
**A. 市場基準相關（最小驗證）**
   用各市場全池等權（＝`B_all` 的定義）當該市場的策略基準，算兩者相關，
   跟現有的跨市場配對相關並排。

**B. 殘差相關結構（延伸版）**
   對每檔策略跑 `ret_i = α_i + β_i · mkt_i + ε_i`（OLS），
   **`mkt_i` 用該策略所屬市場的基準**（XM 樹裡台股策略用台股基準、美股用美股），
   取殘差矩陣重新建樹，比較：
     - 殘差版的同市場／跨市場配對相關（vs 原始版）
     - 殘差版與原始版的分群結構一致性（ARI，兩邊切同一個 k）

   若殘差版的跨市場相關掉到 0 附近、而同市場仍高 → **跨市場分散完全來自 beta**，
   §3.5② 必須改寫；若殘差版跨市場仍明顯低於同市場 → 策略選擇確實提供額外分散。

   ⚠️ 殘差相關有**機械下限 −1/(k−1)**（mkt 是同一批策略的等權平均），
      解讀必須跟這條線比、不能跟 0 比。見 `corr_mechanical_floor`。

**C. 純 beta 模型的證偽檢定（2026-09-05 補）**
   B 只能說「扣掉 beta 後沒東西了」，但無法回答「原始版的 0.458 到底有沒有比
   beta 該給的更低」。C 直接算純 beta 模型的預測值

       corr_pred(i, j) = ρ(m_i, m_j) × √(R²_i · R²_j)

   跟實測並排。**超額 = 實測 − 預測 ≈ 0 ⇒ 策略選擇沒有提供任何額外分散。**

   這條檢定同時解釋 H-25 的「粒度效應」：群越小 → 特異變異平均掉得越少 →
   R² 越低 → beta 驅動的相關被稀釋越多，於是「看起來」越互補。
   若 L3 的超額同樣 ≈ 0，則粒度效應是**特異變異稀釋**而非「小群比較特化」。

⚠️ **beta 代理的循環性**：`B_all` 是候選池等權，而候選池是**全期間贏過基準**的策略，
   故它不是中性的市場代理，會略微高估 beta、低估殘差。真正乾淨的做法要用市場指數
   （需先做 M-07 重建等權全市場序列）。本版先用 B_all，此限制須在文件揭露。

⚠️ 另一層循環性：`mkt = wide.mean(axis=0)`，每檔策略自己佔 1/N（N=6,679~15,040），
   對基準的影響可忽略，但嚴格說仍是內生的。

用法：
    cd code
    python -m research.beta_baseline
    python -m research.beta_baseline --trees TW_normal
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, hrp, paths
from . import stage3_hrp as S3

TREES = ("TW_normal", "US_normal", "XM_normal")
LEVELS = ("L1", "L3")


def market_returns(log=print) -> dict[str, pd.Series]:
    """各市場的策略基準＝該市場 normal 樹的全池等權（＝`B_all` 的定義）。**永遠是全窗。**"""
    out = {}
    for m in ("TW", "US"):
        w = S3.rebuild_tree_returns(f"{m}_normal", log=lambda *a, **k: None)
        out[m] = w.mean(axis=0)
        log(f"  [{m}] 基準序列 {len(out[m])} 個月（{w.shape[0]:,} 檔策略等權）")
    return out


def _fit_cols(m: str, cols) -> list:
    """該市場 IS 窗（`contracts.HRP_IS_WINDOWS`）涵蓋的月份——M-13a 的擬合區間。"""
    a, b = C.HRP_IS_WINDOWS[m]
    return [c for c in cols if pd.Period(a, "M") <= c <= pd.Period(b, "M")]


def benchmark_correlation(mkt: dict[str, pd.Series], log=print) -> float:
    """台美兩個市場基準本身的相關——M-01 的核心對照數字。"""
    j = pd.concat([mkt["TW"].rename("TW"), mkt["US"].rename("US")], axis=1).dropna()
    r = float(j.TW.corr(j.US))
    log(f"🔴 台股基準 vs 美股基準的相關係數 = {r:.4f}（{len(j)} 個共同月份）")
    return r


def _residuals(wide: pd.DataFrame, mkt: dict[str, pd.Series],
               is_only: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """逐策略對其所屬市場基準做 OLS，回傳 (殘差矩陣, beta, R²)。

    向量化實作：對同一市場的策略，市場序列相同，故 beta = cov(r_i, m)/var(m)
    可一次算完整批，不需逐檔跑迴歸。
    """
    market_of = pd.Series(wide.index, index=wide.index).str.split("::").str[0]
    resid = wide.copy().astype(float)
    betas = pd.Series(index=wide.index, dtype=float)
    r2s = pd.Series(index=wide.index, dtype=float)

    for m in ("TW", "US"):
        idx = market_of[market_of == m].index
        if len(idx) == 0:
            continue
        sub = wide.loc[idx]
        mk = mkt[m].reindex(sub.columns)
        if mk.isna().any():
            raise ValueError(f"[{m}] 基準序列在該樹的窗內有缺月，無法回歸")
        # 🔴 M-13a：`is_only` 時**係數只用 IS 窗估**，但殘差仍算全窗
        # ——這樣只換掉「beta 有沒有偷看 OOS」一個變因，殘差矩陣的維度不變，
        # 殘差樹與主版才是可比的。若連殘差也裁到 IS，就同時換掉了樣本區間。
        fc = _fit_cols(m, sub.columns) if is_only else list(sub.columns)
        if len(fc) < 24:
            raise ValueError(f"[{m}] 擬合區間只有 {len(fc)} 個月，太短")
        xf = mk.reindex(fc).to_numpy(dtype=np.float64)
        yf = sub[fc].to_numpy(dtype=np.float64)
        xfc = xf - xf.mean()
        var_x = float((xfc ** 2).mean())
        yfc = yf - yf.mean(axis=1, keepdims=True)
        beta = (yfc @ xfc) / (len(xf) * var_x)              # cov/var（IS 窗）
        alpha = yf.mean(axis=1) - beta * xf.mean()

        x = mk.to_numpy(dtype=np.float64)                   # 全窗
        y = sub.to_numpy(dtype=np.float64)
        e = y - (alpha[:, None] + beta[:, None] * x[None, :])
        resid.loc[idx] = e
        betas.loc[idx] = beta
        # R² = 1 - SSE/SST（全窗，與主版口徑一致）
        sse = (e ** 2).sum(axis=1)
        sst = ((y - y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        r2s.loc[idx] = 1.0 - np.divide(sse, sst, out=np.full_like(sse, np.nan), where=sst > 0)
    return resid, betas, r2s


def _pair_stats(corr: np.ndarray, uids: pd.Index, cluster_map: pd.Series,
                keep_ids: pd.Index, r2: pd.Series | None = None,
                mkt_corr: float = float("nan")) -> pd.DataFrame:
    """群代表層級的配對相關 + same/cross 標記 + 純 beta 模型的預測值。

    `corr` 是**群代表序列**的相關矩陣。給 `r2` 時額外算 `corr_pred`（beta 模型預測）
    與 `excess`（實測 − 預測）——`excess ≈ 0` 代表該配對沒有超越 beta 的分散效果。
    """
    ids = list(keep_ids)
    mkt = (pd.Series(uids, index=uids).str.split("::").str[0]
           .groupby(cluster_map.reindex(uids).to_numpy()).agg(lambda s: s.mode().iloc[0]))
    rows = []
    for i, j in itertools.combinations(range(len(ids)), 2):
        a, b = ids[i], ids[j]
        rec = {"a": a, "b": b, "corr": float(corr[i, j]),
              "pair_type": "same" if mkt[a] == mkt[b] else "cross"}
        if r2 is not None:
            rec["corr_pred"] = _beta_predicted_corr(r2, a, b, mkt, mkt_corr)
            rec["excess"] = rec["corr"] - rec["corr_pred"]
        rows.append(rec)
    return pd.DataFrame(rows)


def _reps_corr(wide: pd.DataFrame, labels: np.ndarray) -> tuple[np.ndarray, pd.Index]:
    """群代表序列（成員簡單平均，同 H-06/H-25 口徑）的相關矩陣。"""
    reps = wide.groupby(labels).mean()
    return np.corrcoef(reps.to_numpy(dtype=np.float64)), reps.index


def _cluster_r2(reps: pd.DataFrame, mkt_of: pd.Series,
                mkt: dict[str, pd.Series]) -> pd.Series:
    """每個群代表對**自己所屬市場**基準的 R²——beta 模型預測的關鍵輸入。

    ⚠️ `mkt_of` 是用「群內市場的眾數」決定的。若群是跨市場混合的，用單一市場基準
    算 R² 就只是近似。**2026-09-06 code review 實查：不是近似，是精確**——
    XM 樹在 L3（453 群）的市場純度**全部為 1.0000**（純度 <1 的群數 = 0），
    即 HRP 連在最細的層級都產出完全市場純淨的群。
    這同時獨立佐證了 §3.5② 的「HRP 分的是市場邊界」。
    若日後改了建樹規則導致出現混合群，本函式的 R² 會失真——屆時要改成
    對兩個市場基準做多元回歸取 R²。
    """
    out = {}
    for cid in reps.index:
        x = mkt[mkt_of[cid]].reindex(reps.columns).to_numpy(dtype=np.float64)
        y = reps.loc[cid].to_numpy(dtype=np.float64)
        out[cid] = float(np.corrcoef(x, y)[0, 1] ** 2)
    return pd.Series(out)


def _beta_predicted_corr(r2: pd.Series, a, b, mkt_of: pd.Series, mkt_corr: float) -> float:
    """🔴 純 beta 模型下的預測相關——本模組最關鍵的一條公式。

    設 r_i = β_i·m_i + ε_i（ε 與 m 無關），則兩群的相關為

        corr(r_i, r_j) = ρ(m_i, m_j) × √(R²_i) × √(R²_j)

    因為 σ_i = β_i·σ_m / √(R²_i)，特異變異會**稀釋** beta 驅動的相關。

    這條公式的用途是**證偽**：若實測相關 ≈ 預測相關，代表該配對的分散效果
    **完全來自「兩個市場指數的相關」＋「群自身特異變異的稀釋」**，
    策略選擇沒有提供任何額外貢獻。

    ⚠️ 同市場配對的 ρ(m_i, m_j) = 1（同一條基準），故預測值 = √(R²_i·R²_j)。
    """
    rho = 1.0 if mkt_of[a] == mkt_of[b] else mkt_corr
    return float(rho * np.sqrt(r2[a] * r2[b]))


def analyse_tree(tree_id: str, mkt: dict[str, pd.Series], mkt_corr: float,
                 log=print, is_only: bool = False) -> list[dict]:
    tree_key = tree_id.split("_")[0]
    wide = S3.rebuild_tree_returns(tree_id, log=lambda *a, **k: None)
    log(f"[{tree_id}] {wide.shape[0]:,} 檔 × {wide.shape[1]} 月")

    resid, betas, r2s = _residuals(wide, mkt, is_only=is_only)
    log(f"  beta 中位 {betas.median():.3f}｜R² 中位 {r2s.median():.3f}"
        f"（市場解釋掉的變異）")

    # 原始版：直接讀凍結的分群（就是主線那棵樹）
    assign = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")
    assign = assign[assign.tree_id == tree_id].set_index(C.PK)
    link_raw = np.load(paths.STAGE3 / f"linkage_{tree_id}.npy")
    if len(link_raw) + 1 != len(wide):
        raise AssertionError("主線 linkage 葉節點數與報酬矩陣不符")

    # 殘差版：同一套流程重建
    t0 = time.time()
    corr_r = np.corrcoef(resid.to_numpy(dtype=np.float64))
    ok, min_eig = hrp.check_psd(corr_r)
    if not ok:
        raise AssertionError(f"殘差相關矩陣非 PSD（min_eig={min_eig:.3e}）")
    dist_r = hrp.corr_to_distance(corr_r)
    # linkage 方法沿用主線的選法（最大群佔比較低者），確保只有「原始 vs 殘差」一個變因
    cand = {}
    for method in ("single", "ward"):
        lk = hrp.build_linkage(dist_r, method=method)
        sizes = pd.Series(hrp.cut_clusters(lk, S3.L3_TARGET[tree_key])).value_counts()
        cand[method] = {"link": lk, "max_share": float(sizes.max() / len(wide))}
    method_r = min(cand, key=lambda m: cand[m]["max_share"])
    link_r = cand[method_r]["link"]
    log(f"  殘差版建樹完成 {time.time()-t0:.0f}s｜linkage={method_r}")

    rows = []
    for level in LEVELS:
        k = S3.L1_TARGET[tree_key] if level == "L1" else S3.L3_TARGET[tree_key]
        lab_raw = hrp.cut_clusters(link_raw, k)
        lab_res = hrp.cut_clusters(link_r, k)
        ari = hrp.adjusted_rand_index(pd.Series(lab_raw), pd.Series(lab_res))

        for tag, w, lab in (("raw", wide, lab_raw), ("resid", resid, lab_res)):
            cm, ids = _reps_corr(w, lab)
            cmap = pd.Series(lab, index=wide.index)
            # 🔴 beta 模型預測只對原始版有意義——殘差版已經把 beta 拿掉了，
            # 再拿 beta 模型去預測它是同義反覆。
            if tag == "raw":
                reps = w.groupby(lab).mean()
                mkt_of = (pd.Series(wide.index, index=wide.index).str.split("::").str[0]
                          .groupby(lab).agg(lambda z: z.mode().iloc[0]))
                r2c = _cluster_r2(reps, mkt_of, mkt)
                pf = _pair_stats(cm, wide.index, cmap, ids, r2c, mkt_corr)
            else:
                pf = _pair_stats(cm, wide.index, cmap, ids)
                pf["corr_pred"] = np.nan
                pf["excess"] = np.nan
                r2c = None
            for pt, g in pf.groupby("pair_type"):
                rows.append({"tree_id": tree_id, "level": level, "version": tag,
                            "pair_type": pt, "n_clusters": len(ids), "n_pairs": len(g),
                            # 🔴 機械下限：mkt 是同一批策略的等權平均，故殘差在橫斷面
                            # 上被強制加總≈0。k 個群代表的平均兩兩相關因此被壓到約
                            # −1/(k−1)，**不是真的負相關**。解讀殘差相關必須跟這條線比，
                            # 不能跟 0 比（2026-09-05 開發時實測 TW L1 殘差 −0.133、
                            # 機械下限 −0.200，兩者同量級即「殘差已無系統性共動」）。
                            "corr_mechanical_floor": -1.0 / (len(ids) - 1),
                            "corr_median": float(g["corr"].median()),
                            "corr_mean": float(g["corr"].mean()),
                            # 純 beta 模型的預測值與超額（實測 − 預測）。
                            # **excess ≈ 0 ⇒ 該配對的分散完全來自 beta，策略選擇零貢獻。**
                            "corr_pred_median": (float(g["corr_pred"].median())
                                                 if g["corr_pred"].notna().any() else np.nan),
                            "excess_median": (float(g["excess"].median())
                                              if g["excess"].notna().any() else np.nan),
                            "pct_below_pred": (float((g["excess"] < 0).mean())
                                               if g["excess"].notna().any() else np.nan),
                            "cluster_r2_median": (float(r2c.median()) if r2c is not None else np.nan),
                            "cluster_size_median": float(pd.Series(lab).value_counts().median()),
                            "pct_high": float((g["corr"] < C.COMPLEMENTARITY_CUTS["高"]).mean()),
                            "ari_raw_vs_resid": float(ari),
                            "market_corr": mkt_corr if tree_key == "XM" else float("nan"),
                            "beta_median": float(betas.median()),
                            "r2_median": float(r2s.median())})
        log(f"  {level}(k={k})：ARI(原始, 殘差) = {ari:.4f}")
    return rows


def run(trees=TREES, log=print, is_only: bool = False) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE1 / "_marks")
    freeze.verify_inputs(paths.STAGE3)

    log("=== A. 市場基準 ===")
    mkt = market_returns(log)
    mkt_corr = benchmark_correlation(mkt, log)
    log("")

    rows = []
    for t in trees:
        rows += analyse_tree(t, mkt, mkt_corr, log, is_only=is_only)
        log("")

    df = pd.DataFrame(rows)
    for col in ("tree_id", "level", "version", "pair_type"):
        df[col] = df[col].astype("category")
    C.validate(df, C.BETA_BASELINE, strict_columns=True)
    log(f"✓ beta_baseline 契約通過（{len(df)} 列）")

    out_dir = paths.ROOT / "_analysis_outputs_robustness"
    # 🔴 M-13a 寫到**另一個檔名**：`enb_null` 會 `verify_inputs(_beta_baseline_manifest)`
    # 並讀 `market_corr`，若 `--is-only` 覆蓋主版產物會連鎖要求 M-05 重跑。
    # 分檔後兩者互不干擾（稽核清單第二輪已預先警告這條連鎖）。
    stem = "beta_baseline_isonly" if is_only else "beta_baseline"
    p = out_dir / f"{stem}.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        stem, out_dir / f"_{stem}_manifest",
        inputs=[paths.STAGE1 / "returns_monthly.parquet",
               paths.STAGE1 / "returns_meta.parquet",
               paths.STAGE1 / "strategy_marks.parquet",
               paths.STAGE3 / "cluster_assign.parquet"],
        outputs=[p],
        params={"levels": list(LEVELS), "market_proxy": "B_all（該市場全池等權）",
               "market_corr": mkt_corr,
               "complementarity_cuts": C.COMPLEMENTARITY_CUTS,
               "beta_fit_window": ("IS 窗（HRP_IS_WINDOWS），殘差與 R² 仍算全窗"
                                   if is_only else "全窗")},
        notes="M-01：市場 beta 基準與殘差相關結構。回答「跨市場分散是不是只是"
              "國際分散的重現」。beta 代理用 B_all 有循環性（候選池是全期間贏家），"
              "須在文件揭露；乾淨版需 M-07 的市場指數。",
    )
    log(f"→ {p}")
    return df


def _report(df: pd.DataFrame, log=print) -> None:
    mc = df.market_corr.dropna()
    mc = float(mc.iloc[0]) if len(mc) else float("nan")
    log("\n" + "=" * 88)
    log("M-01 · 市場 beta 基準與殘差相關結構")
    log("=" * 88)
    log(f"🔴 台股基準 vs 美股基準的相關係數 = {mc:.4f}")
    log("")
    log(f"{'樹':<12}{'層級':<6}{'版本':<8}{'類型':<7}{'群數':>5}{'配對':>8}"
        f"{'相關中位':>10}{'機械下限':>10}{'高互補%':>9}{'ARI':>8}")
    for r in df.sort_values(["tree_id", "level", "pair_type", "version"]).itertuples():
        floor = f"{r.corr_mechanical_floor:>10.4f}" if r.version == "resid" else f"{'—':>10}"
        log(f"{r.tree_id:<12}{r.level:<6}{r.version:<8}{r.pair_type:<7}"
            f"{r.n_clusters:>5}{r.n_pairs:>8,}{r.corr_median:>10.4f}{floor}"
            f"{r.pct_high:>9.1%}{r.ari_raw_vs_resid:>8.4f}")
    log("")
    log("⚠️ 殘差版的相關要跟「機械下限 −1/(k−1)」比，不能跟 0 比——mkt 是同一批策略的")
    log("   等權平均，殘差在橫斷面被強制加總≈0，會把群代表的平均相關壓到負值。")
    log("   殘差相關 ≈ 機械下限 ⇒ 扣掉市場後已無系統性共動。")
    log("")
    x = df[(df.tree_id == "XM_normal") & (df.pair_type == "cross")]
    for lvl in LEVELS:
        g = x[x.level == lvl]
        if len(g) < 2:
            continue
        raw = float(g[g.version == "raw"].corr_median.iloc[0])
        res = float(g[g.version == "resid"].corr_median.iloc[0])
        log(f"【{lvl}】跨市場相關：原始 {raw:.4f}（市場基準 {mc:.4f}）→ 殘差 {res:.4f}")
        if abs(raw - mc) < 0.05:
            log(f"   ⚠️ 原始版幾乎等於市場相關 → 該層級的跨市場分散**就是國際分散**")

    # ── 🔴 決定性對照：純 beta 模型預測 vs 實測 ──────────────────────────────
    log("")
    log("=" * 88)
    log("🔴 純 beta 模型檢定：corr_pred = ρ(market) × √(R²_i · R²_j)")
    log("=" * 88)
    log(f"{'樹':<12}{'層級':<6}{'類型':<7}{'群大小中位':>11}{'群R²中位':>10}"
        f"{'實測':>9}{'beta預測':>10}{'超額':>9}{'低於預測%':>11}")
    raw_rows = df[df.version == "raw"].sort_values(["tree_id", "level", "pair_type"])
    for r in raw_rows.itertuples():
        log(f"{r.tree_id:<12}{r.level:<6}{r.pair_type:<7}{r.cluster_size_median:>11,.0f}"
            f"{r.cluster_r2_median:>10.4f}{r.corr_median:>9.4f}{r.corr_pred_median:>10.4f}"
            f"{r.excess_median:>+9.4f}{r.pct_below_pred:>11.1%}")
    log("")
    log("判讀：")
    log("  · 超額 ≈ 0 且 低於預測% ≈ 50%  ⇒ 該層級的分散**完全由市場 beta 解釋**，")
    log("    策略選擇零貢獻（實測只是在 beta 模型預測值附近隨機擺動）。")
    log("  · 超額顯著為負                ⇒ 策略選擇提供了超越 beta 的真實分散。")
    log("  · L1→L3 的『粒度效應』若同時伴隨 群R² 下降，代表那是**特異變異稀釋**：")
    log("    群越小 → 平均掉的特異變異越少 → R² 越低 → beta 驅動的相關被稀釋越多，")
    log("    **不是『小群比較特化』**。此時 H-25 的敘事必須改寫。")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.beta_baseline")
    ap.add_argument("--trees", nargs="+", default=list(TREES))
    ap.add_argument("--is-only", action="store_true",
                    help="M-13a：beta 與 alpha 只用 IS 窗估計（殘差仍算全窗），"
                         "產出寫到 beta_baseline_isonly.csv")
    a = ap.parse_args(argv)
    df = run(trees=tuple(a.trees), is_only=a.is_only)
    _report(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
