# -*- coding: utf-8 -*-
"""
D3：把 in-sample 對半切，各跑一次 Phase 1，比對兩期的因子判定是否一致。

## 為什麼需要

整條 SOP 都在同一份 in-sample（2000-01-01 ~ 2025-12-31）上做選擇：
Phase 1 決定因子去留、Phase 2 決定晉升、Phase 3/4 決定候選池，
八個變體累計測試超過 5 萬個策略，**目前沒有任何子期間穩健性檢定**。
2026 的樣本外只有半年多，撐不起這個規模的多重檢定。

指導教授的提醒：
  「你**不可以用太多資料**…你分析過去的那個模型，跟分析未來的那個模型，**要一樣**」

## 做法

不動用樣本外（2026 保持乾淨），把 in-sample 切成前後兩期各跑一次 Phase 1：

    前期 2000-01-01 ~ 2012-12-31
    後期 2013-01-01 ~ 2025-12-31

比對兩期的 ρ 與判定。**若因子判定大致穩定，這是遠比 2026 半年更有力的穩健性證據；
若不穩定，及早知道比口試被問出來好。**

## 注意

- 兩期的基準不同（各自期間的自建宇宙基準），故各自重算，門檻才對等。
- 只跑 Phase 1（單因子 9 桶、無 F2/C/V），不碰後面的階段——
  Phase 1 是整條線的守門員，它穩不穩定決定了後面全部。
- 產出的 label 加 `_P1H1` / `_P1H2` 後綴，不會覆蓋正式結果。

用法：
  python phase1_subperiod_check.py --market TW            # 跑回測 + 分析
  python phase1_subperiod_check.py --market TW --analyze-only
"""
import sys
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as st

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fcv_core import MarketData, run_spec, is_done, ART_DIR      # noqa: E402
from sweep_config import MARKET_START                             # noqa: E402
from condition_factory import build_conditions                    # noqa: E402
from phase1_linearity import FACTORS, N_BUCKETS, make_spec, LOOKAHEAD_FLAGGED  # noqa: E402
from phase1_analyze import classify, verdict                      # noqa: E402
from combinations import sim_conditions                           # noqa: E402
from fcv_core import ensure_dtindex_cols, write_report_artifacts  # noqa: E402

OUT = HERE.parent / "_analysis_outputs_robustness"
V_MODES = ("v0",)

HALVES = [("H1", "2000-01-01", "2012-12-31"),
          ("H2", "2013-01-01", "2025-12-31")]


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def bench_for(md, market, tag):
    """該子期間的自建宇宙基準（全買、同宇宙同成本）——門檻必須用同期間的。"""
    label = f"{market}_UNIV_{tag}_M"
    out = Path(ART_DIR) / label
    p = out / "stats.parquet"
    if not p.exists():
        mask = pd.DataFrame(True, index=md.price_index, columns=md.common)
        mask = ensure_dtindex_cols(mask, md.common)
        rc = sim_conditions(conditions={"UNIVERSE_EQW": mask}, resample="M", data=md.data)
        out.mkdir(parents=True, exist_ok=True)
        # write_report_artifacts 會回傳 stats 列並寫出各策略明細；沿用 universe_benchmark 的作法
        rows = write_report_artifacts(rc, out)
        pd.DataFrame(rows).to_parquet(p)
    return float(pd.read_parquet(p).iloc[0]["CAGR"])


def run_half(market, tag, start, end, batch_size=150):
    """跑該子期間的 20 個單因子 Phase 1。"""
    md = MarketData(market, start=start, end=end, verbose=False)
    log(f"  [{tag}] 宇宙 {len(md.common)} 檔｜交易日 {len(md.price_index)}"
        f"（{md.price_index.min().date()}~{md.price_index.max().date()}）")
    bench = bench_for(md, market, tag)
    log(f"  [{tag}] 自建宇宙基準 {bench:.2%}")
    for f in FACTORS:
        label = f"{market}_L1_{f}_{tag}_M"
        if is_done(label):
            continue
        try:
            run_spec(md, make_spec(f, market), label, rebalance="M",
                     batch_size=batch_size, dedup=True, v_modes=V_MODES, verbose=False)
        except Exception as e:
            log(f"  ⚠️ {label} 失敗：{e}")
    md.release()
    return bench


def analyze_half(market, tag, bench):
    rows = []
    for f in FACTORS:
        p = Path(ART_DIR) / f"{market}_L1_{f}_{tag}_M" / "stats.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df = df[df["strategy"].str.endswith("__v0")].copy()
        import re
        df["k"] = df["strategy"].map(
            lambda s: int(re.match(r".+_qb(\d+)of\d+__", s).group(1)))
        df = df.sort_values("k")
        if len(df) < N_BUCKETS:
            continue
        ks = df["k"].values.astype(float)
        ys = df["CAGR"].values.astype(float)
        rho, pv = st.spearmanr(ks, ys)
        vd = "🚫 前瞻偏誤" if f in LOOKAHEAD_FLAGGED else verdict(rho, pv, ys, bench)
        rows.append({"因子": f, "rho": round(float(rho), 3), "p值": round(float(pv), 4),
                     "形狀": classify(ks, ys, rho), "判定": vd,
                     "最佳桶": int(np.argmax(ys)), "最佳桶CAGR": round(float(ys.max()), 4)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Phase 1 子期間穩健性檢定")
    ap.add_argument("--market", default="TW", choices=["TW", "US"])
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    mkt = args.market
    OUT.mkdir(parents=True, exist_ok=True)

    log(f"=== Phase 1 子期間穩健性檢定｜{mkt} ===")
    res, benches = {}, {}
    for tag, s, e in HALVES:
        log(f"--- {tag}：{s} ~ {e}")
        if args.analyze_only:
            md = MarketData(mkt, start=s, end=e, verbose=False)
            benches[tag] = bench_for(md, mkt, tag)
            md.release()
        else:
            t0 = time.time()
            benches[tag] = run_half(mkt, tag, s, e)
            log(f"  [{tag}] 回測完成｜{(time.time()-t0)/60:.1f} 分鐘")
        res[tag] = analyze_half(mkt, tag, benches[tag])

    full = pd.read_csv(HERE.parent / "_analysis_outputs_phase1" /
                       f"{mkt}_phase1_linearity.csv", encoding="utf-8-sig")
    m = (full[["因子", "Spearman_rho", "判定"]]
         .rename(columns={"Spearman_rho": "rho_全期", "判定": "判定_全期"})
         .merge(res["H1"][["因子", "rho", "判定"]]
                .rename(columns={"rho": "rho_H1", "判定": "判定_H1"}), on="因子", how="left")
         .merge(res["H2"][["因子", "rho", "判定"]]
                .rename(columns={"rho": "rho_H2", "判定": "判定_H2"}), on="因子", how="left"))
    m["兩期判定一致"] = m["判定_H1"] == m["判定_H2"]
    m["rho_符號一致"] = np.sign(m["rho_H1"]) == np.sign(m["rho_H2"])

    log(f"\n基準：H1 {benches['H1']:.2%}｜H2 {benches['H2']:.2%}｜"
        f"全期見 _analysis_outputs_phase1")
    log("\n===== 三期對照 =====")
    pd.set_option("display.width", 220)
    log(m.to_string(index=False))

    ok = m[m["判定_H1"].notna()]
    log(f"\n  兩期判定完全一致：{int(ok['兩期判定一致'].sum())} / {len(ok)}"
        f"（{ok['兩期判定一致'].mean():.0%}）")
    log(f"  ρ 符號一致：{int(ok['rho_符號一致'].sum())} / {len(ok)}"
        f"（{ok['rho_符號一致'].mean():.0%}）")
    log(f"  ρ 的兩期相關：{ok['rho_H1'].corr(ok['rho_H2'], method='spearman'):.3f}")

    p = OUT / f"{mkt}_phase1_subperiod.csv"
    m.to_csv(p, index=False, encoding="utf-8-sig")
    log(f"\n輸出：{p}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
