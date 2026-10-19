# -*- coding: utf-8 -*-
"""2026-10-18 一次性查核：M-01 的 beta_baseline.csv／enb_null.csv 時間戳是09-29，
早於TW候選池擴增（15,009→29,255），`enb_null._market_corr()`讀的`market_corr`
是否因此過期到足以影響論文引用的XM two_factor數字（86.2%）。

不重建任何樹（那很貴，見`beta_baseline.analyse_tree()`），只重用兩個既有的
向量化函式：`beta_baseline.market_returns()`（算新池的市場基準序列，cheap，
純讀parquet+pivot+mean）跟`enb_null.simulate()`（T×T矩陣的200次抽樣，不碰
N×N）。拿新rho跑一次two_factor，跟檔案裡已經存的舊rho結果並排。

✅ 這份CSV只是查核記錄，不是正式的M-01重跑產物——不寫freeze/manifest，
不覆蓋`beta_baseline.csv`／`enb_null.csv`。若之後CPU閒下來要做正式重跑，
仍要走完整的`research.beta_baseline` + `research.enb_null`。

用法：
    cd code
    python -m research._check_m01_rho_recheck
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from . import paths
from . import stage3_hrp as S3
from .beta_baseline import market_returns, benchmark_correlation
from .enb_null import N_DRAWS, RANDOM_SEED, observed_r2_vector, simulate, verify_fast_enb

TREE_ID = "XM_normal"
OLD_RHO = 0.551189   # beta_baseline.csv／enb_null.csv 現存的 market_corr（2026-09-29，舊池）
                     # 🔴 不是0.5507——那是TW_normal L1列的ari_raw_vs_resid，之前誤讀


def main() -> int:
    t0 = time.time()
    fast, known = verify_fast_enb(TREE_ID)
    wide = S3.rebuild_tree_returns(TREE_ID, log=lambda *a, **k: None)
    n, t = wide.shape
    r2_vec = observed_r2_vector(wide)
    is_us = (pd.Series(wide.index).str.split("::").str[0] == "US").to_numpy()

    mkt = market_returns(log=lambda *a, **k: None)
    new_rho = benchmark_correlation(mkt, log=lambda *a, **k: None)

    rows = []
    for rho, label in ((OLD_RHO, "old_pool_2026-09-29"), (new_rho, "new_pool_2026-10-18")):
        sim = simulate(TREE_ID, n, t, r2_vec, is_us, rho, n_draws=N_DRAWS,
                       log=lambda *a, **k: None)
        tf = [r for r in sim if r["null_model"] == "two_factor"][0]
        rows.append({
            "rho_label": label, "market_rho_used": rho,
            "tree_id": TREE_ID, "n_strategies": n, "n_months": t,
            "enb_observed": fast, "enb_observed_frozen_h09": known,
            "enb_null_mean": tf["enb_null_mean"], "enb_null_std": tf["enb_null_std"],
            "ratio_observed_over_null": fast / tf["enb_null_mean"],
            "n_draws": N_DRAWS, "random_seed": RANDOM_SEED,
            "script": "research._check_m01_rho_recheck",
        })
        print(f"[{label}] rho={rho:.6f}  虛無ENB={tf['enb_null_mean']:.4f}±{tf['enb_null_std']:.4f}"
             f"  實測/虛無={fast/tf['enb_null_mean']:.4%}")

    df = pd.DataFrame(rows)
    out = paths.ROOT / "_analysis_outputs_robustness" / "m01_rho_recheck_2026-10-18.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n→ {out}")
    print(f"總耗時 {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
