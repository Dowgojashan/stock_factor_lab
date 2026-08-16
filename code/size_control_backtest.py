# -*- coding: utf-8 -*-
"""
規模控制專用的小型回測：只跑「市值桶 × 各因子桶」的配對。

## 為什麼需要這支

`size_control_analysis.py` 原本用 `REVENUE`（營業收入）當市值的代理，因為
Phase 2 的因子池裡有 REVENUE、配對結果現成。但營收不等於市值——
低毛利高營收的產業（通路、代工）會被歸到偏大的規模桶，代理誤差不小。

collector 於 2026-08-13 新增了 `MKTCAP`（季底普通股市值 / marketCap）因子，
但它**沒有進過 Phase 2 的回測**（Phase 1 的 20 因子清單裡沒有它）。

要用真市值做規模控制，理論上得把 MKTCAP 加進因子池重跑 Phase 2——那會動到
整條線。**但其實不必**：規模控制只需要「MKTCAP 桶 × 因子桶」這一種配對，
用 `fcv_core.stream_strategy_masks` 的 `allowed_f_pairs` 白名單精準展開即可。

    規模 3 桶 × 19 因子 × 3 桶 = 171 個配對 + 3 個單獨 = 174 個策略

比重跑 Phase 2（1,596 個）省 89%，而且完全不影響既有結果。

⚠️ MKTCAP **不會**因此進入 F 構面的因子池——它只是這支分析的控制變數。

用法（cwd=code/）：
  python size_control_backtest.py --market TW
  python size_control_backtest.py --market US --dry-run
"""
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fcv_core import MarketData, run_spec, is_done, ART_DIR      # noqa: E402
from sweep_config import MARKET_START                             # noqa: E402
from condition_factory import build_conditions                    # noqa: E402
from phase1_linearity import IN_SAMPLE_END                         # noqa: E402
import phase_variants                                             # noqa: E402

SIZE_FACTOR = "MKTCAP"
N_BUCKETS = 3
V_MODES = ("v0",)


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def make_spec(market):
    """MKTCAP 放**第一個**——`dedup` 只取無序對 j>i，故配對名稱一律是
    `MKTCAP_qbi__X_qbj`，白名單才好對。"""
    factors = phase_variants.get("all", market)["factors"]      # 19 個（不含 MKTCAP）
    if SIZE_FACTOR in factors:
        raise AssertionError(f"{SIZE_FACTOR} 不應該在 F 構面因子池裡")
    ordered = [SIZE_FACTOR] + list(factors)
    return {
        "_meta": {"phase": "SIZE_CONTROL", "market": market, "size_factor": SIZE_FACTOR,
                  "n_buckets": N_BUCKETS, "in_sample_end": IN_SAMPLE_END,
                  "note": "只跑 市值桶×因子桶 的配對，供 size_control_analysis 用"},
        "P1": {f: {"field": f"report:{f.lower()}",
                   "conditions": [{"type": "q_band", "args": [k, N_BUCKETS]}
                                  for k in range(N_BUCKETS)]}
               for f in ordered},
        "P3": {},
    }, factors


def allowed_pairs(factors):
    """只要 MKTCAP 的 3 個桶 × 各因子的 3 個桶（外加 MKTCAP 自己的單因子）。"""
    a = set()
    for i in range(N_BUCKETS):
        s = f"{SIZE_FACTOR}_qb{i}of{N_BUCKETS}"
        a.add(f"{s}__None")
        for f in factors:
            for j in range(N_BUCKETS):
                a.add(f"{s}__{f}_qb{j}of{N_BUCKETS}")
    return a


def main():
    ap = argparse.ArgumentParser(description="規模控制專用回測（市值桶 × 因子桶）")
    ap.add_argument("--market", default="TW", choices=["TW", "US"])
    ap.add_argument("--batch-size", type=int, default=150)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    label = f"{args.market}_SIZECTRL_M"
    spec, factors = make_spec(args.market)
    allowed = allowed_pairs(factors)
    P1 = build_conditions(spec["P1"])

    log(f"=== 規模控制回測｜{args.market}｜代理＝{SIZE_FACTOR}｜in-sample 至 {IN_SAMPLE_END} ===")
    log(f"    因子 {len(factors)} 個 + {SIZE_FACTOR}｜P1 條件 {len(P1)}"
        f"｜白名單 {len(allowed)} 個 F 組合｜預期 {len(allowed)} 策略")
    log(f"    {'done' if is_done(label) else 'pending'}  {label}")

    if args.dry_run:
        log("dry-run：不執行。")
        return 0
    if is_done(label):
        log("已完成。")
        return 0

    md = MarketData(args.market, start=MARKET_START[args.market], end=IN_SAMPLE_END)
    t0 = time.time()
    meta = run_spec(md, spec, label, rebalance="M", batch_size=args.batch_size,
                    dedup=True, v_modes=V_MODES, allowed_f_pairs=allowed)
    md.release()
    log(f"✅ {label}｜展開 {meta['n_expanded']}｜回測 {meta['n_backtested']}"
        f"｜{(time.time()-t0)/60:.1f} 分鐘")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
