# -*- coding: utf-8 -*-
"""市場因子的代表性，以及扣除市場因子後的群內／群間殘差相關（2026-11-22，論文 4-3）。

使用者授權（經寫作 session 轉達），整理自同日的兩支暫存腳本（mktfac.py、l3resid.py）。
純讀凍結檔、不建樹、不呼叫 LLM。

① 市場因子＝各市場 normal 樹全部候選策略的等權月報酬（`beta_baseline.market_returns`，
   即 beta_baseline／enb_null 用的那條）。計算它和三種外部序列的月報酬相關：
   - 自建等權市場基準 UNIVERSE_M（台股讀 twfix 版，美股讀原目錄；日頻淨值取月底）
   - 報酬指數 taiex_tr／sp500_tr、價格指數 taiex／sp500（`market_index_monthly.parquet`）
   回答口委可能的質疑：「策略的平均」是否真的代表市場。

② 策略層的群內／群間平均相關，原始報酬與殘差兩版，L1 與 L3 兩層。
   殘差＝對市場因子做 OLS 後的殘差（`beta_baseline._residuals`，全窗擬合）。
   平均相關用加總恆等式計算（每列標準化後，群內配對相關和＝‖Σz‖²−n·T），
   不必算 N×N 相關矩陣；群內以配對數 n(n−1) 加權，群間以 n_a·n_b 加權。
   L1 另存逐群的群內值。
   ⚠️ 市場因子是全部策略的平均，殘差在橫斷面加總約為 0，故群內殘差正相關時，
      群間必然略負（L1 明顯，L3 群數多而攤薄到接近 0）。

輸出（`_analysis_outputs_robustness/`）：
  market_factor_representativeness.csv
  residual_corr_within_between.csv
  _market_factor_check_manifest/

用法（cwd=code/）：python -m research.market_factor_check
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import freeze, paths
from . import stage3_hrp as S3
from .beta_baseline import _residuals, market_returns

OUT = paths.ROOT / "_analysis_outputs_robustness"
CODE = paths.ROOT / "code"
UNIVERSE = {"TW": CODE / "results_artifacts_twfix" / "TW_UNIVERSE_M",
            "US": CODE / "results_artifacts" / "US_UNIVERSE_M"}
INDEX_FILE = OUT / "market_index_monthly.parquet"
INDEX_NAMES = ("taiex_tr", "sp500_tr", "taiex", "sp500")


def _universe_monthly(m: str) -> pd.Series:
    c = pd.read_parquet(UNIVERSE[m] / "UNIVERSE_EQW" / "stock_data.parquet")["cum_returns"]
    mo = c.resample("M").last().pct_change().dropna()
    mo.index = mo.index.to_period("M")
    return mo


def representativeness(mkt: dict[str, pd.Series]) -> pd.DataFrame:
    idx = pd.read_parquet(INDEX_FILE)
    rows = []
    for m in ("TW", "US"):
        f = mkt[m].copy()
        f.index = pd.PeriodIndex(f.index, freq="M")
        others = {"UNIVERSE_M": _universe_monthly(m)}
        for nm in INDEX_NAMES:
            s = idx[idx.index_name == nm].set_index("month")["ret"]
            s.index = pd.PeriodIndex(s.index, freq="M")
            others[nm] = s
        for nm, s in others.items():
            j = pd.concat([f.rename("f"), s.rename("x")], axis=1).dropna()
            rows.append({"market": m, "series": nm, "corr": float(j.f.corr(j.x)),
                         "n_months": len(j), "start": str(j.index[0]), "end": str(j.index[-1]),
                         "factor_months": len(f)})
    return pd.DataFrame(rows)


def _std_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True)
    if (s == 0).any():
        raise ValueError("有零變異數的列（rebuild_tree_returns 應已排除）")
    return x / s


def _within_between(z: np.ndarray, lab: np.ndarray) -> tuple[float, float, pd.DataFrame]:
    """加總恆等式算群內／群間平均相關。全部用浮點數，避免 n·(n−1)·T 整數溢位。"""
    t = float(z.shape[1])
    labs = np.unique(lab)
    sums = np.vstack([z[lab == c].sum(axis=0) for c in labs])
    n = np.array([(lab == c).sum() for c in labs], dtype=float)
    win = (sums * sums).sum(axis=1) - n * t          # 各群群內配對相關和 ×T
    wpairs = n * (n - 1)
    ok = n > 1
    within = float(win[ok].sum() / (wpairs[ok].sum() * t))
    tot = sums.sum(axis=0)
    allp = float(tot @ tot - n.sum() * t)
    bpairs = n.sum() ** 2 - n.sum() - wpairs.sum()
    between = float((allp - win.sum()) / (bpairs * t))
    per = pd.DataFrame({"cluster_id": labs, "n_strategies": n.astype(int),
                        "corr_mean": np.where(ok, win / np.where(ok, wpairs, 1) / t, np.nan)})
    return within, between, per


def residual_structure(mkt: dict[str, pd.Series]) -> pd.DataFrame:
    ca = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")
    rows = []
    for m in ("TW", "US"):
        tree = f"{m}_normal"
        wide = S3.rebuild_tree_returns(tree, log=lambda *a, **k: None)
        resid, _, r2 = _residuals(wide, mkt)
        a = ca[ca.tree_id == tree].set_index("strategy_uid")
        versions = {"raw": _std_rows(wide.to_numpy(np.float64)),
                    "resid": _std_rows(resid.to_numpy(np.float64))}
        for lvl in ("L1", "L3"):
            lab = a[f"cluster_{lvl}"].reindex(wide.index)
            if lab.isna().any():
                raise ValueError(f"[{tree}] {lab.isna().sum()} 檔策略沒有 {lvl} 群標籤")
            lab = lab.to_numpy()
            for ver, z in versions.items():
                w, b, per = _within_between(z, lab)
                base = {"market": m, "level": lvl, "version": ver,
                        "n_clusters": int(len(per)), "n_strategies_total": int(len(lab)),
                        "n_months": int(z.shape[1]), "r2_median": float(np.nanmedian(r2))}
                rows.append({**base, "scope": "within_mean", "cluster_id": np.nan,
                             "n_strategies": np.nan, "corr_mean": w})
                rows.append({**base, "scope": "between_mean", "cluster_id": np.nan,
                             "n_strategies": np.nan, "corr_mean": b})
                if lvl == "L1":
                    for _, r in per.iterrows():
                        rows.append({**base, "scope": "within_cluster", "cluster_id": int(r.cluster_id),
                                     "n_strategies": int(r.n_strategies), "corr_mean": r.corr_mean})
    return pd.DataFrame(rows)


def run(log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    for d in (paths.STAGE1, paths.STAGE3):
        freeze.verify_inputs(d)
    mkt = market_returns(log)
    rep = representativeness(mkt)
    res = residual_structure(mkt)
    p1 = OUT / "market_factor_representativeness.csv"
    p2 = OUT / "residual_corr_within_between.csv"
    rep.to_csv(p1, index=False, encoding="utf-8-sig")
    res.to_csv(p2, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "market_factor_check", OUT / "_market_factor_check_manifest",
        inputs=[paths.STAGE1 / "returns_monthly.parquet", paths.STAGE1 / "returns_meta.parquet",
                paths.STAGE1 / "strategy_marks.parquet", paths.STAGE3 / "cluster_assign.parquet",
                UNIVERSE["TW"] / "UNIVERSE_EQW" / "stock_data.parquet",
                UNIVERSE["US"] / "UNIVERSE_EQW" / "stock_data.parquet", INDEX_FILE],
        outputs=[p1, p2],
        params={"market_factor": "beta_baseline.market_returns（normal 樹全池等權）",
                "residual": "beta_baseline._residuals（全窗 OLS）",
                "pair_weighting": "群內 n(n−1)、群間 n_a·n_b"},
        notes="論文 4-3：市場因子代表性＋群內／群間殘差相關。2026-11-22 由暫存腳本 mktfac.py／l3resid.py 整理。")
    log(rep.to_string(index=False))
    log(res[res.scope != "within_cluster"].to_string(index=False))
    log(res[res.scope == "within_cluster"][["market", "version", "cluster_id", "n_strategies", "corr_mean"]]
        .to_string(index=False))
    log(f"→ {p1}\n→ {p2}")
    return rep, res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run()
