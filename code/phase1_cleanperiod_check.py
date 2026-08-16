# -*- coding: utf-8 -*-
"""
穩健性檢定：台股 2000-2004 的資料空洞，會不會改變 Phase 1 的因子去留判定？

背景（2026-08-12 發現）
  台股 2000-2004 的財報因子有值的公司只有 690→355 家（逐年遞減，2005 才跳到 1,306），
  而且**其中 55~90% 的值是剛好 0.00**（PB/PS/P_IC/FCF_P 高達 90%）。
  `q_band` 用 `rank(pct=True)`（ties 取平均排名），大量並列的 0 會全部擠在中段，
  導致**低分位桶在 2000-2004 完全沒有持股**（實測 PB_qb0 到 2005 才有），
  而高分位桶（真正有值的少數）從 2002 就開始持股。

  → 各桶的「實際投資起始年」不同，9 桶 CAGR 曲線因此不可比，
    Spearman ρ（老師的主判準）也連帶受影響。

本檔的作法（不重跑回測）
  每個策略的 `return_table.parquet` 有年×月報酬與 YTD，
  直接用 2005-2025 的年報酬複利回推「乾淨期間 CAGR」，
  重算 ρ 與判定，跟原本 2000-2025 的版本對照，看**有沒有因子換邊**。

用法：python phase1_cleanperiod_check.py [--market TW] [--clean-start 2005]
"""
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase1_linearity import FACTORS, N_BUCKETS          # noqa: E402
from phase1_analyze import classify, verdict             # noqa: E402

ART = HERE / "results_artifacts"
OUT = HERE.parent / "_analysis_outputs_phase1"


def log(m):
    print(m, flush=True)


def yearly(label, strategy):
    """回傳該策略的年報酬 Series（index=年，值=該年報酬率）。"""
    p = ART / label / strategy / "return_table.parquet"
    if not p.exists():
        return None
    rt = pd.read_parquet(p)
    if "YTD" not in rt.columns:
        return None
    s = pd.to_numeric(rt["YTD"], errors="coerce").dropna()
    s.index = [int(x) for x in s.index]
    return s


def cagr_from_years(s, y0, y1):
    """由年報酬複利算 CAGR。整年都沒交易（報酬恰為 0）的年份仍計入——
    那正是「空手」的真實成本，不可略過，否則等於偷偷把起始年往後挪。"""
    w = s[(s.index >= y0) & (s.index <= y1)]
    if len(w) == 0:
        return np.nan
    return float((1 + w).prod() ** (1 / len(w)) - 1)


def first_active_year(s):
    """第一個報酬不為 0 的年份＝實際開始持股的年份。"""
    nz = s[s.abs() > 1e-12]
    return int(nz.index[0]) if len(nz) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="TW", choices=["TW", "US"])
    ap.add_argument("--clean-start", type=int, default=2005,
                    help="乾淨期間的起始年（台股財報因子 2005 才齊）")
    ap.add_argument("--end", type=int, default=2025)
    args = ap.parse_args()
    mkt, y0, y1 = args.market, args.clean_start, args.end
    OUT.mkdir(parents=True, exist_ok=True)

    # 基準也要用同一段期間重算，否則門檻不對等
    ub = yearly(f"{mkt}_UNIVERSE_M", "UNIVERSE_EQW")
    if ub is None:
        raise FileNotFoundError(f"找不到 {mkt}_UNIVERSE_M 的 return_table；請先跑 universe_benchmark.py")
    bench_full = cagr_from_years(ub, 2000, y1)
    bench_clean = cagr_from_years(ub, y0, y1)
    log(f"=== Phase 1 乾淨期間穩健性檢定｜{mkt} ===")
    log(f"自建宇宙基準：2000-{y1} {bench_full:.2%}  →  {y0}-{y1} {bench_clean:.2%}\n")

    rows = []
    for f in FACTORS:
        label = f"{mkt}_L1_{f}_M"
        ys_full, ys_clean, starts = [], [], []
        for k in range(N_BUCKETS):
            s = yearly(label, f"{f}_qb{k}of{N_BUCKETS}__None__None__v0")
            if s is None:
                ys_full, ys_clean = [], []
                break
            ys_full.append(cagr_from_years(s, 2000, y1))
            ys_clean.append(cagr_from_years(s, y0, y1))
            starts.append(first_active_year(s))
        if len(ys_full) != N_BUCKETS:
            log(f"⚠️ {f}：讀不到完整 9 桶，略過")
            continue
        ks = np.arange(N_BUCKETS, dtype=float)
        a_full, a_clean = np.array(ys_full), np.array(ys_clean)
        rho_f, p_f = st.spearmanr(ks, a_full)
        rho_c, p_c = st.spearmanr(ks, a_clean)
        v_f = verdict(rho_f, p_f, a_full, bench_full)
        v_c = verdict(rho_c, p_c, a_clean, bench_clean)
        act = [s for s in starts if s is not None]
        rows.append({
            "因子": f,
            "各桶起始年_最早": min(act) if act else None,
            "各桶起始年_最晚": max(act) if act else None,
            "起始年落差": (max(act) - min(act)) if act else None,
            "rho_2000": round(float(rho_f), 3), "rho_clean": round(float(rho_c), 3),
            "rho變動": round(float(rho_c - rho_f), 3),
            "判定_2000": v_f, "判定_clean": v_c,
            "判定改變": v_f != v_c,
            "形狀_clean": classify(ks, a_clean, rho_c),
            **{f"clean桶{i}": round(float(v), 4) for i, v in enumerate(a_clean)},
        })

    out = pd.DataFrame(rows)
    show = ["因子", "各桶起始年_最早", "各桶起始年_最晚", "起始年落差",
            "rho_2000", "rho_clean", "rho變動", "判定_2000", "判定_clean", "判定改變"]
    pd.set_option("display.width", 250)
    log("===== 全期(2000-2025) vs 乾淨期間({}-{}) =====".format(y0, y1))
    log(out[show].sort_values("rho變動", key=abs, ascending=False).to_string(index=False))

    chg = out[out["判定改變"]]
    log(f"\n===== 判定有改變的因子：{len(chg)} / {len(out)} =====")
    if len(chg):
        log(chg[["因子", "rho_2000", "rho_clean", "判定_2000", "判定_clean"]].to_string(index=False))
    else:
        log("  （無）→ Phase 1 的去留名單對資料空洞是穩健的")

    log("\n===== 各桶「實際開始持股年份」的落差（>0 代表桶之間不可比）=====")
    gap = out[out["起始年落差"].fillna(0) > 0]
    log(gap[["因子", "各桶起始年_最早", "各桶起始年_最晚", "起始年落差"]].to_string(index=False)
        if len(gap) else "  （無落差）")

    p = OUT / f"{mkt}_phase1_cleanperiod_check.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    log(f"\n輸出：{p}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
