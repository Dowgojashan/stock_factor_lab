# -*- coding: utf-8 -*-
"""M-05 · ENB 的虛無分布（2026-09-05）

🔴 **這支腳本要解決的問題**：`研究框架總覽_v10.md` §3.4 的核心數字是
「上萬個策略的有效獨立賭注數只有 **3.27（TW）／4.36（US）／5.77（XM）**」。
**但從來沒有人算過「這個數字該多大」**——沒有虛無分布，3.27 就只是一個沒有尺度的數。

---------------------------------------------------------------------------
三個虛無模型
---------------------------------------------------------------------------
**① `iid`：純雜訊上界**
   N 檔策略、T 個月，報酬完全獨立同分布。這是 ENB 的**上界參考點**——
   若實測 ENB 遠低於它，代表策略之間確實有大量共同結構。

   ⚠️ **N ≫ T 的關鍵性質**：本專案 N=6,679~15,040 而 T=228，
   相關矩陣的**秩最多只有 T−1=227**，有 N−T+1 個特徵值恆為 0。
   所以純雜訊的 ENB **不會是 N，而是 T 的數量級**。
   稽核清單原本假設可以拿 N 當上界，那是錯的。

**② `one_factor`：單因子（市場 beta）模型**
   `r_i = β_i · m + ε_i`，`R²` **依逐策略實測值校準**（直接重用 M-01 的
   `_residuals` 重算，不是拿彙總表的中位數——R² 的離散度本身會製造額外的
   獨立維度，只用中位數會低估虛無 ENB）。這個虛無問的是更尖銳的問題：

   > **如果策略之間唯一的共同結構就是市場 beta，ENB 會是多少？**

   若 `one_factor` 的 ENB ≈ 實測 ENB，代表 **ENB 這麼低完全由 beta 解釋**，
   與 M-01 的結論（相關結構幾乎完全由 beta 決定）互相印證。

**③ `two_factor`：兩個市場因子（只跑跨市場樹）**
   🔴 XM 樹裡台股策略跟台股大盤、美股策略跟美股大盤，那是**兩個**共同因子，
   兩者相關 = M-01 實測的 **0.5512**。拿單因子當 XM 的虛無會**低估虛無的維度**，
   讓實測看起來「比虛無更分散」——那是虛無設定錯，不是發現。
   （2026-09-05 初版就犯了這個錯：XM 實測 5.77 vs 單因子虛無 4.49，看似 +3.1σ；
   換成正確的雙因子虛無 6.86 後，實測其實是**低於**虛無的。）
   TW/US 單一市場的樹跑這個會退化成 one_factor，故不產生該列。

---------------------------------------------------------------------------
🔴 實測結論
---------------------------------------------------------------------------
| 樹 | 實測 ENB | 純雜訊虛無 | **市場因子虛無** | 實測/市場虛無 |
|---|---|---|---|---|
| TW | 3.27 | 223.19 | 3.98 ± 0.32 | **82%** |
| US | 4.36 | 282.13 | 5.73 ± 0.51 | **76%** |
| XM | 5.77 | 225.30 | 6.86 ± 0.59（雙因子）| **84%** |

實測只有純雜訊的 **1.5~2.6%**，但已經落在市場因子虛無的 **76~84%**
⇒ **那 3.27/4.36/5.77 幾乎完全是市場 beta 造成的**。
實測略**低於**市場因子虛無（2~3σ），代表真實資料的共同結構比單純的市場因子
**還要更集中**一點。

---------------------------------------------------------------------------
🔴 為什麼模擬跑得動：N×N 特徵分解可以換成 T×T
---------------------------------------------------------------------------
XM 的相關矩陣是 15,040×15,040（1.8GB），一次特徵分解約 190 秒，
模擬 200 次不可行。

但設 `Z` 為列標準化後的 N×T 矩陣，則 `corr = Z Zᵀ / T`（N×N），
它的**非零特徵值與 `Zᵀ Z / T`（T×T）完全相同**，其餘 N−T 個恆為 0。
而熵的定義裡 `0·ln0 = 0`，零特徵值**對 ENB 沒有任何貢獻**。

⇒ 只要分解 **228×228** 的矩陣就好，快了五個數量級。
本模組用實測資料驗證這個等價性（見 `verify_fast_enb`，也寫成測試斷言）。

用法：
    cd code
    python -m research.enb_null
    python -m research.enb_null --draws 50
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

TREES = ("TW_normal", "US_normal", "XM_normal")
NULLS = ("iid", "one_factor", "two_factor")
N_DRAWS = 200
RANDOM_SEED = 42


def fast_enb(z: np.ndarray) -> float:
    """由 N×T 的列標準化矩陣直接算 ENB，只分解 T×T 的 Gram 矩陣。

    數學等價性：`corr = Z Zᵀ / T` 與 `G = Zᵀ Z / T` 的非零特徵值相同
    （同為 `Z/√T` 的奇異值平方），其餘 N−T 個特徵值恆為 0；
    熵的 `0·ln0 = 0`，故零特徵值對 ENB 無貢獻。
    ⚠️ 這是**恆等式不是近似**，用 `verify_fast_enb` 對實測資料驗證過。
    """
    t = z.shape[1]
    g = (z.T @ z) / t
    ev = np.clip(np.linalg.eigvalsh(g), 0.0, None)
    total = ev.sum()
    if total <= 0:
        return 0.0
    p = ev / total
    p = p[p > 0]
    return float(np.exp(-float((p * np.log(p)).sum())))


def _standardise(x: np.ndarray) -> np.ndarray:
    """逐列（策略）標準化成均值 0、標準差 1；零變異數列直接剔除。

    ⚠️ 剔除而非填 0：零變異數策略在 `np.corrcoef` 下整列是 NaN，
    `stage3_hrp._build_tree` 與 `effective_bets._tree_corr` 都做同樣的排除。
    """
    keep = x.std(axis=1) > 0
    x = x[keep]
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)


def verify_fast_enb(tree_id: str, log=print) -> tuple[float, float]:
    """對實測資料驗證 `fast_enb` 與 H-09 凍結的 `enb_raw` 一致。

    回傳 (fast_enb 算出來的, effective_number_of_bets.csv 記錄的)。
    這是整個模組的正確性根基——若快速算法錯了，所有虛無分布都沒有意義。
    """
    wide = S3.rebuild_tree_returns(tree_id, log=lambda *a, **k: None)
    z = _standardise(wide.to_numpy(dtype=np.float64))
    fast = fast_enb(z)
    ref = pd.read_csv(paths.ROOT / "_analysis_outputs_robustness"
                      / "effective_number_of_bets.csv")
    row = ref[ref.tree_id == tree_id]
    known = float(row.enb_raw.iloc[0]) if len(row) else float("nan")
    log(f"  [{tree_id}] fast_enb = {fast:.4f}｜H-09 凍結的 enb_raw = {known:.4f}"
        f"｜差 {abs(fast-known):.2e}")
    return fast, known


# ============================================================================
# 兩個虛無模型
# ============================================================================

def _draw_iid(n: int, t: int, rng) -> np.ndarray:
    return _standardise(rng.standard_normal((n, t)))


def _draw_two_factor(t: int, r2: np.ndarray, is_us: np.ndarray,
                     rho: float, rng) -> np.ndarray:
    """雙因子模型：**兩個市場各一個因子，兩因子相關 = `rho`**（M-01 實測 0.5512）。

    🔴 為什麼跨市場樹必須用這個而不是 one_factor：XM 樹裡台股策略跟著台股大盤走、
    美股策略跟著美股大盤走，那是**兩個**共同因子。拿單一因子當虛無會**低估**
    虛無的維度，讓實測看起來「比虛無更分散」——那是虛無設定錯，不是發現。
    （2026-09-05 初版就是這樣：XM 實測 5.77 vs 單因子虛無 4.49，看似 +3.1σ。）

    ⚠️ 對 TW/US 單一市場的樹，本模型會退化成 one_factor，故只對跨市場樹跑。
    """
    m_a = rng.standard_normal(t)
    m_b = rho * m_a + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(t)
    m = np.where(is_us[:, None], m_b[None, :], m_a[None, :])    # 每檔跟自己的市場
    e = rng.standard_normal((len(r2), t))
    z = np.sqrt(r2)[:, None] * m + np.sqrt(1.0 - r2)[:, None] * e
    return _standardise(z)


def _draw_one_factor(n: int, t: int, r2: np.ndarray, rng) -> np.ndarray:
    """單因子模型：`r_i = β_i·m + ε_i`，各策略的 R² 由 `r2` 指定（實測分布）。

    以標準化後的尺度直接構造：令 `z_i = √R²_i · m + √(1−R²_i) · e_i`，
    其中 m 與 e 皆為標準常態，則 `corr(z_i, m)² = R²_i`，
    且 `corr(z_i, z_j) = √(R²_i · R²_j)`——正是 M-01 那條 beta 模型公式
    （同市場 ρ(m_i,m_j)=1 的情形）。
    """
    m = rng.standard_normal(t)
    e = rng.standard_normal((n, t))
    a = np.sqrt(r2)[:, None]
    z = a * m[None, :] + np.sqrt(1.0 - r2)[:, None] * e
    return _standardise(z)


def _market_corr() -> float:
    """台美市場基準的相關（M-01 實測 0.5512）——two_factor 虛無的關鍵參數。"""
    d = paths.ROOT / "_analysis_outputs_robustness"
    p = d / "beta_baseline.csv"
    if not p.exists():
        raise FileNotFoundError("找不到 beta_baseline.csv——請先執行 M-01"
                                "（`python -m research.beta_baseline`）")
    # 🔴 DD-08：消費上游產物前必須驗雜湊。2026-09-06 code review 補上——
    # 本模組把 M-01 算出的 market_corr 當 two_factor 虛無的關鍵參數，
    # 若 beta_baseline.csv 被改過而沒重跑，虛無分布會靜默建立在舊數字上。
    freeze.verify_inputs(d / "_beta_baseline_manifest")
    v = pd.read_csv(p).market_corr.dropna().unique()
    if len(v) != 1:
        raise AssertionError(f"beta_baseline 的 market_corr 不唯一：{v}")
    return float(v[0])


def observed_r2_vector(wide: pd.DataFrame) -> np.ndarray:
    """**逐策略**對其所屬市場基準的 R²，直接重算（不是讀彙總表的中位數）。

    🔴 為什麼要逐策略而不是用中位數：R² 的**離散度本身會製造額外的獨立維度**
    ——一群 R² 全是 0.88 的策略，跟一群 R² 從 0.5 散到 0.99 的策略，
    即使中位數相同，後者的 ENB 明顯較高。用單一中位數會**低估**虛無 ENB，
    等於偷偷讓虛無比較容易被實測打敗。故直接重用 M-01 的向量化實作重算。
    """
    from .beta_baseline import _residuals, market_returns
    mkt = market_returns(log=lambda *a, **k: None)
    _, _, r2 = _residuals(wide, mkt)
    v = r2.to_numpy(dtype=np.float64)
    # 極少數策略可能因 SST=0 得到 NaN（`_residuals` 已用 where 保護），用中位數補上
    # ——它們對 ENB 的影響可忽略，但 NaN 會讓整批模擬變 NaN。
    v = np.where(np.isnan(v), np.nanmedian(v), v)
    # R²=1 會讓 √(1−R²)=0，該策略變成純 beta 的複製品（無特異成分），故夾住上界。
    return np.clip(v, 0.0, 0.9999)


def simulate(tree_id: str, n: int, t: int, r2_vec: np.ndarray,
             is_us: np.ndarray, mkt_rho: float,
             n_draws: int, log=print) -> list[dict]:
    rng = np.random.default_rng(RANDOM_SEED)
    two_ok = bool(is_us.any() and (~is_us).any())
    rows = []
    for null in NULLS:
        # 單一市場的樹跑 two_factor 會退化成 one_factor（兩個因子只有一個被用到），
        # 產出重複的列，故直接跳過。
        if null == "two_factor" and not two_ok:
            continue
        t0 = time.time()
        vals = []
        for _ in range(n_draws):
            if null == "iid":
                z = _draw_iid(n, t, rng)
            elif null == "one_factor":
                z = _draw_one_factor(n, t, r2_vec, rng)
            else:
                z = _draw_two_factor(t, r2_vec, is_us, mkt_rho, rng)
            vals.append(fast_enb(z))
        arr = np.asarray(vals)
        rows.append({"tree_id": tree_id, "null_model": null,
                     "n_strategies": n, "n_months": t, "n_draws": n_draws,
                     "r2_used": (float(np.median(r2_vec))
                                 if null != "iid" else float("nan")),
                     "market_rho_used": (mkt_rho if null == "two_factor"
                                         else float("nan")),
                     "enb_null_mean": float(arr.mean()),
                     "enb_null_std": float(arr.std(ddof=1)),
                     "enb_null_p05": float(np.quantile(arr, 0.05)),
                     "enb_null_p95": float(np.quantile(arr, 0.95)),
                     "enb_null_min": float(arr.min()),
                     "enb_null_max": float(arr.max())})
        log(f"  [{tree_id}] {null:<11} ENB = {arr.mean():.3f} ± {arr.std(ddof=1):.3f}"
            f"  [{arr.min():.3f}, {arr.max():.3f}]  {time.time()-t0:.0f}s")
    return rows


def build(trees=TREES, n_draws=N_DRAWS, log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE1 / "_marks")
    freeze.verify_inputs(paths.STAGE3)

    log("=== 快速演算法驗證（fast_enb vs H-09 凍結的 enb_raw）===")
    rows = []
    for tree_id in trees:
        fast, known = verify_fast_enb(tree_id, log)
        if not np.isnan(known) and abs(fast - known) > 1e-3:
            raise AssertionError(
                f"[{tree_id}] fast_enb {fast:.6f} 與凍結的 enb_raw {known:.6f} 不符"
                "——T×T 等價性被破壞，虛無分布全部無效")
        wide = S3.rebuild_tree_returns(tree_id, log=lambda *a, **k: None)
        n, t = wide.shape
        log(f"  [{tree_id}] N={n:,} T={t}｜秩上限 = T−1 = {t-1}"
            f"（純雜訊的 ENB 不可能超過這個數，不是 N）")
        r2_vec = observed_r2_vector(wide)
        log(f"  [{tree_id}] 逐策略 R²：中位 {np.median(r2_vec):.4f}"
            f"｜p05 {np.quantile(r2_vec, .05):.4f}｜p95 {np.quantile(r2_vec, .95):.4f}"
            f"（離散度本身會製造額外維度，故不可只用中位數）")
        is_us = (pd.Series(wide.index).str.split("::").str[0] == "US").to_numpy()
        mkt_rho = _market_corr()
        for r in simulate(tree_id, n, t, r2_vec, is_us, mkt_rho, n_draws, log):
            r["enb_observed"] = fast
            #: 實測 ÷ 虛無均值。越小代表實測的獨立性離該虛無越遠。
            r["ratio_observed_over_null"] = fast / r["enb_null_mean"]
            #: 實測與虛無均值差幾個虛無標準差（單尾 z）。
            r["z_observed"] = ((fast - r["enb_null_mean"]) / r["enb_null_std"]
                               if r["enb_null_std"] > 0 else float("nan"))
            r["max_possible_enb"] = float(t - 1)
            rows.append(r)
        log("")

    df = pd.DataFrame(rows)
    for c in ("tree_id", "null_model"):
        df[c] = df[c].astype("category")
    return df[C.ENB_NULL.names]


def run(trees=TREES, n_draws=N_DRAWS, log=print) -> pd.DataFrame:
    df = build(trees=trees, n_draws=n_draws, log=log)
    C.validate(df, C.ENB_NULL, strict_columns=True)
    log(f"✓ enb_null 契約通過（{len(df)} 列）")

    out_dir = paths.ROOT / "_analysis_outputs_robustness"
    p = out_dir / "enb_null.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "enb_null", out_dir / "_enb_null_manifest",
        inputs=[paths.STAGE1 / "returns_monthly.parquet",
                paths.STAGE1 / "strategy_marks.parquet",
                out_dir / "effective_number_of_bets.csv",
                out_dir / "beta_baseline.csv"],
        outputs=[p],
        params={"nulls": list(NULLS), "n_draws": n_draws, "seed": RANDOM_SEED,
                "fast_enb": "非零特徵值由 T×T 的 ZᵀZ/T 取得，與 N×N 恆等"},
        notes="M-05：ENB 的虛無分布。iid 是純雜訊上界（受 N≫T 的秩限制，量級是 T "
              "而非 N）；one_factor/two_factor 用逐策略實測 R² 校準，回答「ENB 這麼低"
              "是不是完全由市場 beta 解釋」。跨市場樹必須看 two_factor——單因子會"
              "低估虛無維度。",
    )
    log(f"→ {p}")
    return df


def _report(df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 96)
    log("M-05 · ENB 的虛無分布——3.27 這個數字到底有多小")
    log("=" * 96)
    log(f"  {'樹':<12}{'N':>8}{'T':>5}{'秩上限':>8}{'虛無模型':<12}"
        f"{'虛無 ENB':>16}{'實測 ENB':>10}{'實測/虛無':>11}")
    for r in df.sort_values(["tree_id", "null_model"]).itertuples():
        log(f"  {r.tree_id:<12}{r.n_strategies:>8,}{r.n_months:>5}"
            f"{int(r.max_possible_enb):>8}{r.null_model:<12}"
            f"{r.enb_null_mean:>9.2f} ±{r.enb_null_std:>5.2f}"
            f"{r.enb_observed:>10.2f}{r.ratio_observed_over_null:>11.4f}")
    log("")
    for tree, g in df.groupby("tree_id", observed=True):
        obs = float(g.enb_observed.iloc[0])
        iid = g[g.null_model == "iid"]
        of = g[g.null_model == "one_factor"]
        log(f"【{tree}】實測 ENB = {obs:.2f}")
        if len(iid):
            n = float(iid.enb_null_mean.iloc[0])
            log(f"   純雜訊虛無 {n:.2f}（秩上限 {int(iid.max_possible_enb.iloc[0])}）"
                f" → 實測只有它的 **{obs/n:.2%}**")
        # 🔴 跨市場樹必須看 two_factor（兩個市場因子），單一市場樹看 one_factor。
        # 拿單因子當 XM 的虛無會低估虛無維度，讓實測看起來「比虛無更分散」。
        tf = g[g.null_model == "two_factor"]
        mkt = tf if len(tf) else of
        if len(mkt):
            r = mkt.iloc[0]
            name = ("雙因子虛無（兩市場，ρ=%.4f）" % r.market_rho_used
                    if r.null_model == "two_factor" else "單因子虛無")
            n = float(r.enb_null_mean); sd = float(r.enb_null_std)
            z = float(r.z_observed)
            log(f"   {name} {n:.2f} ± {sd:.2f}（R² 中位={float(r.r2_used):.3f}）"
                f" → 實測 {'高於' if z > 0 else '低於'} 虛無 {abs(z):.1f}σ"
                f"（實測是虛無的 {obs/n:.0%}）")
            if len(tf) and len(of):
                log(f"   （單因子虛無 {float(of.enb_null_mean.iloc[0]):.2f} "
                    f"是**錯的對照**——XM 有兩個市場因子，單因子會低估虛無維度）")
    log("")
    log("🔴 結論：實測 ENB 只有純雜訊的 1.5~2.6%，但已經落在**市場因子虛無的 76~84%**")
    log("   ——那 3.27/4.36/5.77 幾乎完全是市場 beta 造成的，不是策略選擇的結果。")
    log("   實測略**低於**市場因子虛無（2~3σ），代表真實資料的共同結構比單純的")
    log("   市場因子**還要更集中**一點。")
    log("")
    log("⚠️ N≫T：相關矩陣的秩最多 T−1，故純雜訊的 ENB 上界是 **T 的數量級，不是 N**。")
    log("⚠️ 因子虛無用**逐策略實測 R²**（不是中位數）——R² 的離散度本身會製造")
    log("   額外的獨立維度，只用中位數會低估虛無 ENB。")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.enb_null")
    ap.add_argument("--trees", nargs="+", default=list(TREES))
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    a = ap.parse_args(argv)
    _report(run(trees=tuple(a.trees), n_draws=a.draws))
    return 0


if __name__ == "__main__":
    sys.exit(main())
