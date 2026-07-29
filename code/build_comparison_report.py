# -*- coding: utf-8 -*-
"""
11個候選因子批次橫向比較彙整：讀各批 stats.parquet + credibility_metrics_top50.parquet，
算出每批的關鍵指標，輸出一份 JSON 供報告頁面使用。不重跑回測、不動任何既有程式碼。
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_batch import load_stats, f1_median_order, factor_list, parse_bucket  # noqa: E402

CANDIDATES = ["PE", "EV_EBITDA", "EV_S", "CROIC", "FCF_OI", "ROIC",
              "PB", "PS", "P_IC", "OCF_E", "MOM"]
COMMON = {"ROE", "EPS", "FCF_P"}


def summarize(candidate: str) -> dict:
    label = f"TW_batch_{candidate}_M"
    df = load_stats(label)
    order = f1_median_order(df)  # 全部F1單層條件，依中位數CAGR排序（跨4因子混排）

    single = df[~df["is_pair"]]
    cand_single = single[single["F1_factor"] == candidate]
    top_bucket_row = cand_single.loc[cand_single["CAGR"].idxmax()] if len(cand_single) else None

    # 各F1桶（含全部C×V組合）的中位數CAGR——穩健指標，用來比「這個候選因子最好的那一桶」
    # 是否贏過固定三因子最好的那一桶（而不是拿單一次回測的最大值來比）。
    bucket_median = single.groupby("F1")["CAGR"].median()
    cand_buckets = [b for b in bucket_median.index if parse_bucket(b)["factor"] == candidate]
    cand_best_bucket = max(cand_buckets, key=lambda b: bucket_median[b]) if cand_buckets else None
    cand_best_bucket_median = float(bucket_median[cand_best_bucket]) if cand_best_bucket else None

    best_overall = df.loc[df["CAGR"].idxmax()]
    top_factor = parse_bucket(order[0])["factor"] if order else None

    cred_path = HERE.parent / "_analysis_outputs" / label / "credibility_metrics_top50.parquet"
    cred = pd.read_parquet(cred_path) if cred_path.exists() else None

    return {
        "candidate": candidate,
        "n_strategies": int(len(df)),
        "median_CAGR_all": float(df["CAGR"].median()),
        "median_sharpe_all": float(df["sharpe_ann"].median()) if "sharpe_ann" in df else None,
        "median_mdd_all": float(df["max_drawdown"].median()) if "max_drawdown" in df else None,
        "median_winratio_all": float(df["win_ratio"].median()) if "win_ratio" in df else None,
        "candidate_solo_median_CAGR": float(cand_single["CAGR"].median()) if len(cand_single) else None,
        "candidate_best_bucket": cand_best_bucket,
        "candidate_best_bucket_median_CAGR": cand_best_bucket_median,
        "candidate_solo_best_single_strategy": top_bucket_row["F1"] if top_bucket_row is not None else None,
        "candidate_solo_best_single_CAGR": float(top_bucket_row["CAGR"]) if top_bucket_row is not None else None,
        "is_candidate_top_overall_single": bool(top_factor == candidate),
        "top_overall_single_factor": top_factor,
        "top_overall_single_bucket": order[0] if order else None,
        "top_overall_single_CAGR": float(single.loc[single["F1"] == order[0], "CAGR"].median()) if order else None,
        "best_strategy_name": str(best_overall["strategy"]),
        "best_strategy_CAGR": float(best_overall["CAGR"]),
        "best_strategy_sharpe": float(best_overall["sharpe_ann"]) if "sharpe_ann" in best_overall else None,
        "best_strategy_mdd": float(best_overall["max_drawdown"]) if "max_drawdown" in best_overall else None,
        "median_top1_share_top50": float(cred["top1_share"].median()) if cred is not None and len(cred) else None,
        "median_effective_n_top50": float(cred["effective_n"].median()) if cred is not None and len(cred) else None,
    }


def main():
    results = [summarize(c) for c in CANDIDATES]
    out = {
        "generated_from": "TW_batch_{candidate}_M x 11",
        "common_factors": sorted(COMMON),
        "candidates": results,
    }
    out_path = HERE.parent / "_analysis_outputs" / "comparison_summary.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"寫入 {out_path}")
    for r in results:
        print(f"  {r['candidate']:10s} n={r['n_strategies']:5d} "
              f"median_all={r['median_CAGR_all']:.3f} "
              f"solo_median={r['candidate_solo_median_CAGR']:.3f} "
              f"top_overall={'YES' if r['is_candidate_top_overall_single'] else '-'}")


if __name__ == "__main__":
    main()
