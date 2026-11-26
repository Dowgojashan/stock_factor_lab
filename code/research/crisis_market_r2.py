# -*- coding: utf-8 -*-
"""危機期間與平時期間，市場因子對策略報酬的解釋程度（R²）（2026-11-24，論文 4-6；使用者授權，經寫作 session 轉達）。

要驗證的機制假說：危機期間各策略同時受市場衝擊、彼此差異縮小，所以策略層 ENB 下降。
若成立，危機月份裡市場因子的 R² 應高於同長度的平時對照，扣除市場因子後的殘差相關結構也應改變。

口徑（全部沿用既有模組，不另訂）：
  報酬矩陣——`stage3_hrp.rebuild_tree_returns`（TW／US／XM normal 樹）
  市場因子——`beta_baseline.market_returns`（各市場 normal 樹全池等權，同 `market_factor_check`）；
    每個策略對「自己所屬市場」的因子迴歸；XM 另報同時用台、美兩個因子的二元迴歸 R²
  危機月份——`crisis_diversification.crisis_months`（D2、D2_2022升息、D2_2020新冠、D1）；
    平時＝扣除 D1∪D2；月數 < MIN_T 的期間不算
  同長度對照——`crisis_diversification.null_draws`，同一個 SEED，且亂數消耗順序與
    `crisis_diversification.run_tree` 完全相同（每期先抽對照、再抽自助法），所以抽到的平時區段和
    crisis_diversification.csv 是同一批。程式用「群間平均相關的對照中位數」與該檔核對，不符即中止
  指標——
    (a) 每個策略在該期間月份內做時間序列 OLS（含截距）的 R²，報全部策略的中位數（另附四分位）
    (b) 危機值落在同長度平時對照分布的百分位（同 crisis_diversification：小於的比例＋一半等於）
    (c) 殘差（`beta_baseline._residuals`，全窗 OLS，同 market_factor_check）在該期間月份內的
        L1 群內／群間平均相關（`market_factor_check._within_between`，配對加權）
  期間內變異數為 0 的策略（該期間報酬全相同）不計入 R² 中位數，筆數另報。

輸出：`_analysis_outputs_robustness/crisis_market_r2.csv`、`_crisis_market_r2_manifest/`
純讀凍結檔、不建樹、不呼叫 LLM。

用法（cwd=code/）：python -m research.crisis_market_r2
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from . import crisis_diversification as CD
from . import freeze, paths
from . import stage3_hrp as S3
from .beta_baseline import _residuals, market_returns
from .market_factor_check import _within_between

OUT = paths.ROOT / "_analysis_outputs_robustness"
PERIODS = ("D2", "D2_2022升息", "D2_2020新冠", "D1")


def _zrows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """列標準化；回傳 (z, 有效列遮罩)。變異數為 0 的列設為 0。"""
    x = x - x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True)
    ok = (s[:, 0] > 1e-12)
    return np.where(ok[:, None], x / np.where(s > 1e-12, s, 1.0), 0.0), ok


class Tree:
    def __init__(self, tree: str, mkt: dict[str, pd.Series], log=print):
        t0 = time.time()
        self.tree = tree
        wide = S3.rebuild_tree_returns(f"{tree}_normal", log=lambda *a, **k: None)
        self.months = pd.PeriodIndex(wide.columns, freq="M")
        self.X = wide.to_numpy(np.float64)
        self.mk_of = pd.Series(wide.index).str.split("::").str[0].to_numpy()
        self.F = {m: mkt[m].reindex(wide.columns).to_numpy(np.float64) for m in ("TW", "US")}
        for m in set(self.mk_of):
            if np.isnan(self.F[m]).any():
                raise ValueError(f"[{tree}] {m} 市場因子在樹窗內有缺月")
        resid, _, _ = _residuals(wide, mkt)
        self.Rz = resid.to_numpy(np.float64)
        asg = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")
        asg = asg[asg.tree_id.astype(str) == f"{tree}_normal"].set_index("strategy_uid").cluster_L1
        lab = asg.reindex(wide.index)
        if lab.isna().any():
            raise ValueError(f"[{tree}] {lab.isna().sum()} 檔策略沒有 L1 群標籤")
        self.lab = lab.to_numpy()
        self.G = wide.groupby(lab.values).mean().sort_index().to_numpy(np.float64)
        log(f"  [{tree}] 載入 N={self.X.shape[0]:,} T={self.X.shape[1]} k={self.G.shape[0]}（{time.time() - t0:.0f}s）")

    def stats(self, cols: np.ndarray) -> dict:
        z, ok = _zrows(self.X[:, cols])
        r2 = np.full(len(z), np.nan)
        rr = {}
        for m in ("TW", "US"):
            f = self.F[m][cols]
            fz = (f - f.mean()) / f.std()
            rr[m] = z @ fz / len(cols)          # 每個策略與該市場因子的相關
            sel = (self.mk_of == m) & ok
            r2[sel] = rr[m][sel] ** 2
        out = {"R2_中位數": float(np.nanmedian(r2)), "R2_p25": float(np.nanpercentile(r2, 25)),
               "R2_p75": float(np.nanpercentile(r2, 75)), "零變異策略數": int((~ok).sum())}
        if self.tree == "XM":
            a, b = self.F["TW"][cols], self.F["US"][cols]
            r12 = float(np.corrcoef(a, b)[0, 1])
            r2b = (rr["TW"] ** 2 + rr["US"] ** 2 - 2 * rr["TW"] * rr["US"] * r12) / (1 - r12 ** 2)
            out["R2二因子_中位數"] = float(np.nanmedian(np.where(ok, r2b, np.nan)))
            out["台美因子相關"] = r12
        rz, rok = _zrows(self.Rz[:, cols])
        w, btw, _ = _within_between(rz[rok], self.lab[rok])
        out["殘差群內平均相關"], out["殘差群間平均相關"] = w, btw
        return out

    def group_corr(self, cols: np.ndarray) -> float:
        c = np.corrcoef(self.G[:, cols])
        return float(np.nanmean(c[~np.eye(c.shape[0], dtype=bool)]))


def run_tree(tree: str, mkt, ref: pd.DataFrame, log=print) -> list[dict]:
    T = Tree(tree, mkt, log)
    cm = CD.crisis_months(tree)
    pos = {m: i for i, m in enumerate(T.months)}
    all_crisis = {pos[m] for m in (cm["D1"] | cm["D2"]) if m in pos}
    normal_idx = np.array(sorted(set(range(len(T.months))) - all_crisis))
    rng = np.random.default_rng(CD.SEED)
    base = {"市場": tree, "資料版本": CD.MAIN_DATE_NOTE, "策略數N": T.X.shape[0], "群數k": T.G.shape[0]}
    rows = [{**base, "期間": "全期間", "月數T": len(T.months), **T.stats(np.arange(len(T.months)))},
            {**base, "期間": "平時（扣除D1∪D2）", "月數T": len(normal_idx), **T.stats(normal_idx)}]
    for name in PERIODS:
        cidx = np.array(sorted(pos[m] for m in cm[name] if m in pos))
        lens = CD._blocks(list(cidx))
        if len(cidx) < CD.MIN_T:
            rows.append({**base, "期間": name, "月數T": len(cidx), "區段長度": str(lens),
                         "備註": f"月數 < {CD.MIN_T}，不計算"})
            continue
        nul = CD.null_draws(normal_idx, lens, CD.N_NULL, rng)
        CD.boot_draws(cidx, CD.N_BOOT, rng)      # 只為了讓亂數消耗順序與 crisis_diversification 相同
        # 核對：抽到的平時區段與 crisis_diversification.csv 是同一批
        gc = float(np.median([T.group_corr(ix) for ix in nul]))
        want = float(ref[(ref.市場 == tree) & (ref.期間 == name)]["群間平均相關_平時同長度中位數"].iloc[0])
        if abs(gc - want) > 1e-9:
            raise AssertionError(f"[{tree} {name}] 對照抽樣與 crisis_diversification 不一致：{gc} vs {want}")
        obs = T.stats(cidx)
        nst = [T.stats(ix) for ix in nul]
        row = {**base, "期間": name, "月數T": len(cidx), "區段長度": str(lens),
               "月份": f"{T.months[cidx[0]]}～{T.months[cidx[-1]]}"}
        for key, v in obs.items():
            row[key] = v
            if key in ("零變異策略數", "台美因子相關"):
                continue
            nv = np.array([d[key] for d in nst])
            row[f"{key}_平時同長度中位數"] = float(np.median(nv))
            row[f"{key}_平時同長度p05"], row[f"{key}_平時同長度p95"] = np.quantile(nv, [.05, .95])
            row[f"{key}_落在平時分布百分位"] = float((nv < v).mean() + 0.5 * (nv == v).mean())
        rows.append(row)
        log(f"  [{tree}] {name}：T={len(cidx)} R²中位數 {obs['R2_中位數']:.3f}（平時同長度 "
            f"{row['R2_中位數_平時同長度中位數']:.3f}，百分位 {row['R2_中位數_落在平時分布百分位']:.0%}）"
            f"｜殘差群內 {obs['殘差群內平均相關']:.3f}（{row['殘差群內平均相關_落在平時分布百分位']:.0%}）"
            f"｜群間 {obs['殘差群間平均相關']:.3f}（{row['殘差群間平均相關_落在平時分布百分位']:.0%}）")
    return rows


def main() -> int:
    t0 = time.time()
    for d in (paths.STAGE1, paths.STAGE1 / "_marks", paths.STAGE3):
        freeze.verify_inputs(d)
    freeze.verify_inputs(OUT / "_crisis_diversification_manifest")
    ref = pd.read_csv(OUT / "crisis_diversification.csv")
    mkt = market_returns(print)
    rows = []
    for t in CD.TREES:
        rows += run_tree(t, mkt, ref)
    p = OUT / "crisis_market_r2.csv"
    pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "crisis_market_r2", OUT / "_crisis_market_r2_manifest",
        inputs=[paths.STAGE1 / "returns_monthly.parquet", paths.STAGE1 / "returns_meta.parquet",
                paths.STAGE1 / "strategy_marks.parquet", paths.STAGE3 / "cluster_assign.parquet",
                paths.STAGE2 / "regime" / "regime_table_TW.parquet", paths.STAGE2 / "regime" / "regime_table_US.parquet",
                OUT / "crisis_diversification.csv"],
        outputs=[p],
        params={"N_NULL": CD.N_NULL, "SEED": CD.SEED, "MIN_T": CD.MIN_T,
                "market_factor": "beta_baseline.market_returns（各市場 normal 樹全池等權）；策略對自己所屬市場的因子迴歸",
                "residual": "beta_baseline._residuals（全窗 OLS）", "cluster": "normal 樹 L1"},
        notes="論文 4-6：危機與平時的市場因子 R²、殘差群內／群間相關；同長度平時對照與 crisis_diversification 同一批抽樣（已核對）")
    print(f"→ {p}（{time.time() - t0:.0f}s）")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
