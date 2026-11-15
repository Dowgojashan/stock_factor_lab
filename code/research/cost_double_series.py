# -*- coding: utf-8 -*-
"""交易成本加倍敏感度・第一步：每個策略每月「再多扣一份」的成本（2026-11-15，使用者授權，經寫作 session 轉達）。

策略月報酬在回測內已逐筆扣過成本（backtest.py:120-123；台股買 0.1425%、賣 0.1425%+0.3%；美股買賣各 0.1%）。
成本加倍＝每個策略每月再扣一次同樣的成本。本檔用每個策略的 position.parquet（每月目標權重，加總=1）估：
    買進量_t = Σ max(w_t − w_{t−1}, 0)，賣出量_t = Σ max(w_{t−1} − w_t, 0)（第一期 w_{t−1}=0，即從現金全額買進）
    多扣成本_t = 買進量_t × 買方費率 + 賣出量_t × 賣方費率
    權重日 t（月底訊號）在下一個交易日成交，成本記在下一個月的報酬上。
近似：用目標權重的變化，沒算價格漂移造成的再平衡交易（低估）；也沒算成交日與月底的落差。
只讀持股檔，不重跑回測、不呼叫 LLM。
輸出：`_analysis_outputs_robustness/cost_double_strategy_cost.parquet`（策略 × 月，float32）
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import freeze, paths

OUT = paths.ROOT / "_analysis_outputs_robustness"
#: (買方費率, 賣方費率)，與 backtest.py:120-123 的預設值相同
FEES = {"TW": (1.425e-3, 1.425e-3 + 3e-3), "US": (1e-3, 1e-3)}
WORKERS = 12


def one(args) -> tuple[str, pd.Series | None]:
    uid, adir, mk = args
    p = Path(adir) / "position.parquet"
    if not p.exists():
        return uid, None
    w = pd.read_parquet(p).fillna(0.0).sort_index()
    v = w.to_numpy(dtype=np.float64)
    prev = np.vstack([np.zeros((1, v.shape[1])), v[:-1]])
    d = v - prev
    fb, fs = FEES[mk]
    cost = np.clip(d, 0, None).sum(1) * fb + np.clip(-d, 0, None).sum(1) * fs
    months = pd.PeriodIndex(pd.to_datetime(w.index), freq="M") + 1
    s = pd.Series(cost, index=months)
    return uid, s.groupby(level=0).sum()


def main(argv=None) -> int:
    freeze.verify_inputs(paths.STAGE0)
    idx = pd.read_parquet(paths.STAGE0 / "candidate_index.parquet")
    jobs = list(zip(idx.strategy_uid, idx.artifacts_dir, idx.market))
    t0, out, missing = time.time(), {}, []
    with ProcessPoolExecutor(WORKERS) as ex:
        for n, (uid, s) in enumerate(ex.map(one, jobs, chunksize=64), 1):
            if s is None:
                missing.append(uid)
            else:
                out[uid] = s
            if n % 5000 == 0:
                print(f"  {n:,}/{len(jobs):,}｜{time.time()-t0:.0f}s", flush=True)
    assert not missing, f"{len(missing)} 檔沒有 position.parquet，例：{missing[:3]}"
    wide = pd.DataFrame(out).T.sort_index(axis=1).fillna(0.0).astype("float32")
    wide.columns = wide.columns.astype(str)
    p = OUT / "cost_double_strategy_cost.parquet"
    wide.to_parquet(p, compression="zstd")          # 先存檔，健全性檢查失敗也不會白跑
    # 健全性：年化多扣成本 vs stage1 的逐策略換手率（每年交易筆數 / 持股數）× 來回成本
    scan = pd.read_parquet(paths.STAGE1 / "strategy_scan.parquet").set_index("strategy_uid")
    ann = wide.loc[:, (wide.columns >= "2007-01") & (wide.columns <= "2025-12")].mean(axis=1) * 12
    rt = idx.set_index("strategy_uid").market.astype(str).map({m: f[0] + f[1] for m, f in FEES.items()}).astype(float)
    ref = (scan.turnover_ann.astype(float) * rt.reindex(scan.index)).reindex(ann.index)
    for mk in ("TW", "US"):
        u = idx[idx.market == mk].strategy_uid
        print(f"[{mk}] 年化多扣成本（2007–2025）中位 {ann.reindex(u).median():.2%}｜"
              f"stage1 換手率估計中位 {ref.reindex(u).median():.2%}")
    freeze.write_manifest(
        "cost_double_series", OUT / "_cost_double_series_manifest",
        inputs=[paths.STAGE0 / "candidate_index.parquet", paths.STAGE1 / "strategy_scan.parquet"],
        outputs=[p], params={"fees_buy_sell": {k: list(v) for k, v in FEES.items()}},
        notes="每策略每月多扣一份成本（目標權重變化×費率，記在下一個月）；讀 position.parquet，不重跑回測。")
    print(f"→ {p}｜{wide.shape}｜{time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
