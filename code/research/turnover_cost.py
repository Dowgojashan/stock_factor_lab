# -*- coding: utf-8 -*-
"""H-27b · 周轉率與交易成本（2026-09-04）

**補的是 `H26_H27_walkforward矩陣結果.md` 已知限制的第 5 條**：
「全部為毛報酬，不同比例的實際周轉成本差異極大（27 檔 vs 1,003 檔），
實務可行性未評估」。

老師講過「我只有換股而已，我的代價就是我的 MDD」——他意識到有換股成本，
但把代價框在 MDD 上。本模組把**真正的交易成本**算出來。

---------------------------------------------------------------------------
🔴 為什麼必須算到「股票層」，不能用策略數當代理
---------------------------------------------------------------------------
直覺會以為「持有 1,003 個策略的周轉率是 30 個策略的 33 倍」——**這是錯的**。

多個策略之間持股會重疊，A 策略賣出的股票可能正好是 B 策略買進的，
在合併後的投組裡**這兩筆交易互相抵銷、不需實際成交**。策略越多、重疊越多、
抵銷越多，所以**每一元資金的周轉率反而可能下降**。

故本模組直接把選中策略的 `position.parquet`（已正規化為每月加總=1 的權重）
逐一疊加成合併投組權重，再算逐月周轉率：

    turnover_t = 0.5 × Σ_i |w_i,t − w_i,t−1|     （單邊周轉率）

⚠️ 這是「調倉到目標權重」的周轉率，未扣除價格漂移造成的自然權重變化
   （嚴格算法要先讓 t−1 的權重隨報酬漂移到 t 再比較）。**本版高估周轉率**，
   對「成本會不會翻轉結論」是**保守**的方向。

---------------------------------------------------------------------------
範圍
---------------------------------------------------------------------------
在**主線六棵樹的完整窗**上計算（不逐窗跑）——周轉率是策略型態的性質，
跨窗差異遠小於跨比例差異，用完整窗一次算清楚即可，成本也可控。
論文引用時須註明這是代表性估計，非逐窗實算。

成本率參數化（`COST_GRID`），台股實務約：手續費 0.1425%×2 + 證交稅 0.3%（賣出）
≈ 單邊平均 0.29%；美股遠低。故掃描 0.05%~0.30% 涵蓋兩個市場的合理範圍。

用法：
    cd code
    python -m research.turnover_cost
    python -m research.turnover_cost --trees TW
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, paths
from . import stage3_hrp as S3
from .four_group_control import _cagr, _cagr_matrix, _mdd_matrix, _portfolio_series
from .walkforward_matrix import ALLOCATIONS, RATIO_GRID, _pick_a, allocate, target_total

TREES = ("TW_normal", "US_normal", "XM_normal")
LEVEL = "L1"
#: 單邊交易成本率（掃描）。台股實務單邊平均約 0.29%（手續費×2＋證交稅），美股遠低。
COST_GRID = (0.0005, 0.0010, 0.0020, 0.0030)


def _portfolio_weights(uids: list[str], art: pd.Series, log=print) -> pd.DataFrame:
    """把選中策略的持股權重等權疊加成合併投組（月 × 股票）。

    每個策略的 `position.parquet` 已正規化為每月加總=1，故等權合併就是逐一相加
    再除以策略數。逐檔累加而非一次全載入——1,000 檔同時載入約需 2GB 記憶體。
    """
    acc: pd.DataFrame | None = None
    n = 0
    for u in uids:
        p = Path(art[u]) / "position.parquet"
        if not p.exists():
            continue
        w = pd.read_parquet(p)
        acc = w if acc is None else acc.add(w, fill_value=0.0)
        n += 1
    if acc is None or n == 0:
        raise FileNotFoundError("選中策略沒有任何 position.parquet")
    acc = acc / n
    log(f"      合併 {n}/{len(uids)} 檔策略的持股｜{acc.shape[0]} 月 × {acc.shape[1]} 股")
    return acc


def _turnover(weights: pd.DataFrame) -> tuple[float, float]:
    """(月均單邊周轉率, 去重後平均持股檔數)。"""
    w = weights.fillna(0.0).sort_index()
    d = w.diff().abs().sum(axis=1) * 0.5
    return float(d.iloc[1:].mean()), float((w != 0).sum(axis=1).mean())


def run(trees=TREES, log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE0)
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE1 / "_marks")
    freeze.verify_inputs(paths.STAGE3)

    idx = pd.read_parquet(paths.STAGE0 / "candidate_index.parquet")
    art = idx.set_index(C.PK)["artifacts_dir"]
    assign_all = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")
    meta_all = pd.read_parquet(paths.STAGE3 / "cluster_meta.parquet")
    months_long = pd.read_parquet(paths.STAGE1 / "returns_monthly.parquet")

    rows = []
    t0 = time.time()
    for tree_id in trees:
        tree_key = tree_id.split("_")[0]
        assign = assign_all[assign_all.tree_id == tree_id]
        cmeta = meta_all[(meta_all.tree_id == tree_id) & (meta_all.level == LEVEL)]
        wide = S3.rebuild_tree_returns(tree_id, log=lambda *a, **k: None)
        assign = assign[assign[C.PK].isin(wide.index)]

        cagr = _cagr_matrix(wide)
        quality = cagr / _mdd_matrix(wide).abs().replace(0, np.nan)
        corr_full = np.corrcoef(wide.to_numpy(dtype=np.float64))
        pos = pd.Series(range(len(wide.index)), index=wide.index)
        sizes = assign.groupby(f"cluster_{LEVEL}").size()
        k, n_uni = len(sizes), len(assign)
        log(f"[{tree_id}] 宇宙 {n_uni:,}｜{k} 群｜{wide.shape[1]} 個月")

        for ratio in RATIO_GRID:
            tot = target_total(ratio, n_uni, k)
            for how in ALLOCATIONS:
                quota, _ = allocate(sizes, tot, how)
                members, _ = _pick_a(assign, cmeta, wide, quality, quota, corr_full, pos)
                tt = time.time()
                w = _portfolio_weights(members, art, log)
                to, n_stk = _turnover(w)
                gross = _cagr(_portfolio_series(wide, members))
                rows.append({
                    "tree_key": tree_key, "ratio": str(ratio), "allocation": how,
                    "n_strategies": len(members),
                    "n_stocks_avg": n_stk, "monthly_turnover": to,
                    "annual_turnover": to * 12, "gross_cagr": gross,
                })
                log(f"    {str(ratio):<7}{how:<13} {len(members):>5} 檔策略｜"
                    f"持股 {n_stk:>6.0f} 檔｜月周轉 {to:.2%}｜"
                    f"毛CAGR {gross:.2%}｜{time.time()-tt:.0f}s")
        log(f"[{tree_id}] 完成，累計 {time.time()-t0:.0f}s\n")

    df = pd.DataFrame(rows)
    # 各成本率下的淨報酬：年化成本 ≈ 年周轉率 × 單邊成本率 × 2（買賣各一次）
    for c in COST_GRID:
        df[f"net_cagr_{int(c*10000)}bp"] = df.gross_cagr - df.annual_turnover * c * 2
    for col in ("tree_key", "ratio", "allocation"):
        df[col] = df[col].astype("category")
    C.validate(df, C.TURNOVER_COST, strict_columns=True)
    log(f"✓ turnover_cost 契約通過（{len(df)} 列）")

    out_dir = paths.ROOT / "_analysis_outputs_robustness"
    p = out_dir / "turnover_cost.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "turnover_cost", out_dir / "_turnover_cost_manifest",
        inputs=[paths.STAGE0 / "candidate_index.parquet",
               paths.STAGE1 / "returns_monthly.parquet",
               paths.STAGE3 / "cluster_assign.parquet",
               paths.STAGE3 / "cluster_meta.parquet"],
        outputs=[p],
        params={"level": LEVEL, "ratio_grid": [str(r) for r in RATIO_GRID],
               "allocations": list(ALLOCATIONS), "cost_grid_one_way": list(COST_GRID),
               "turnover_def": "0.5*sum|w_t - w_{t-1}|，未扣價格漂移（高估、保守）"},
        notes="H-27b：股票層的實際周轉率與交易成本。策略間持股重疊會讓買賣互相"
              "抵銷，故必須算到股票層，不能用策略數當代理。在主線完整窗上計算，"
              "為代表性估計非逐窗實算。",
    )
    log(f"→ {p}")
    return df


def _report(df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 92)
    log("H-27b · 周轉率與交易成本（主線完整窗，A_hrp）")
    log("=" * 92)
    for t, g in df.groupby("tree_key", observed=True):
        log(f"\n[{t}]")
        log(f"{'比例':<8}{'分配':<14}{'策略數':>7}{'持股數':>8}{'月周轉':>9}"
            f"{'年周轉':>9}{'毛CAGR':>9}{'淨@20bp':>10}{'淨@30bp':>10}")
        for r in g.itertuples():
            log(f"{r.ratio:<8}{r.allocation:<14}{r.n_strategies:>7}{r.n_stocks_avg:>8.0f}"
                f"{r.monthly_turnover:>9.1%}{r.annual_turnover:>9.1%}{r.gross_cagr:>9.2%}"
                f"{getattr(r, 'net_cagr_20bp'):>10.2%}{getattr(r, 'net_cagr_30bp'):>10.2%}")
    log("\n→ 若各比例的淨報酬排序跟毛報酬相同，代表成本不改變「挑越少越好」的結論")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.turnover_cost")
    ap.add_argument("--trees", nargs="+", default=list(TREES))
    a = ap.parse_args(argv)
    df = run(trees=tuple(a.trees))
    _report(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
