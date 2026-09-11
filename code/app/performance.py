# -*- coding: utf-8 -*-
"""L2+ · 績效衡量（應用層開發追蹤.md §9.8 S5，前瞻驗證專用，2026-09-12）

🔴 **只在 §9.8 前瞻驗證的「算績效」步驟使用，且必須晚於「解釋已產生並 git
commit」才能呼叫**（見 §9.8 執行紀律②④）——這是事後裁判，不是輸入，絕對
不可回頭餵進 `explain.py` 的 prompt，也不可在解釋產生之前算，否則就失去
「先預測、後驗證」這個設計的全部意義。

方法：直接用已解析的股票層級持股權重（`risk.StockConcentration.weights`，
`w(s) = Σ(1/N策略) × (1/n_i)`，跟全系統一致的等權慣例）套進 `as_of` 到
`end` 之間**實際發生過的股價變動**，算出這批持股的已實現簡單報酬——
不能用 `returns_monthly.parquet`（那份凍結到 2025-12，前瞻驗證期間本來
就在這之後，涵蓋不到）。

⚠️ 權重可能不滿 1（§8-R5：某策略當天解不出任何股票，視同持有現金）；
這裡也可能因為個股中途缺資料（下市／掛牌時間對不上）而進一步縮小可衡量的
權重——兩種情況都**不重新分配給其他持股**，直接如實回報「量到多少權重」，
不假裝有更高的涵蓋率。

基準：同一段期間、同一個市場全宇宙的**等權**報酬——不是
`weighting_decomposition.csv` 那個固定 2019-2025 窗的數字（H7 用的），這裡
期間不同，必須現算。XM 兩個市場分開列（H7 已定案：CAGR 不可跨市場線性
合成），這裡雖然算的是簡單報酬不是 CAGR，仍沿用「不硬合成一個數字」的
一致立場，避免另開一條合成規則。
"""
from __future__ import annotations

import pandas as pd


def _period_return(close: pd.DataFrame, symbols: list[str],
                   as_of: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    """每檔股票從 `as_of`（取 <= as_of 的最後交易日）到 `end`（取 <= end 的
    最後交易日）的簡單報酬。缺資料回傳 NaN，呼叫端自行決定要不要排除。
    """
    idx = close.index
    start_dates = idx[idx <= as_of]
    end_dates = idx[idx <= end]
    if len(start_dates) == 0 or len(end_dates) == 0:
        return {s: float("nan") for s in symbols}
    d0, d1 = start_dates.max(), end_dates.max()
    out = {}
    for s in symbols:
        if s not in close.columns:
            out[s] = float("nan")
            continue
        p0, p1 = close.at[d0, s], close.at[d1, s]
        out[s] = float(p1 / p0 - 1.0) if pd.notna(p0) and pd.notna(p1) and p0 != 0 else float("nan")
    return out


def measure(md_map: dict, weights: dict[str, float], as_of: str,
           end: str | None = None) -> dict:
    """§9.8 S5：算已實現報酬。

    `weights`：`stock_symbol -> 權重`（來自 `StockConcentration.weights`）。
    `end`：預設用每個市場自己價格資料的最後一天
    （`fcv_core.MarketData.price_index.max()`），也就是資料凍結當下的「今天」。
    """
    as_of_ts = pd.Timestamp(as_of)
    by_market: dict[str, dict] = {}
    portfolio_ret = 0.0
    covered_weight = 0.0

    for m, md in md_map.items():
        close = md.data.get("price:close")
        if close is None:
            continue
        end_ts = pd.Timestamp(end) if end else md.price_index.max()

        syms_here = [s for s in weights if s in close.columns]
        rets = _period_return(close, syms_here, as_of_ts, end_ts)
        for s, r in rets.items():
            if pd.isna(r):
                continue
            portfolio_ret += weights[s] * r
            covered_weight += weights[s]

        # 基準：同市場全宇宙等權，同一段期間，現算（不能沿用 H7 固定窗的數字）
        universe = list(close.columns)
        bench_rets = _period_return(close, universe, as_of_ts, end_ts)
        valid = [r for r in bench_rets.values() if pd.notna(r)]
        by_market[m] = {
            "as_of": str(as_of_ts.date()), "end": str(end_ts.date()),
            "n_universe": len(universe), "n_universe_valid": len(valid),
            "equal_weight_benchmark_return": (sum(valid) / len(valid)) if valid else None,
        }

    n_priced = sum(1 for s in weights if any(s in md.data.get("price:close").columns
                                             for md in md_map.values()))
    return {
        "as_of": str(as_of_ts.date()),
        "portfolio_realized_return": portfolio_ret,
        "portfolio_weight_measured": covered_weight,
        "portfolio_weight_total": sum(weights.values()),
        "n_stocks_in_weights": len(weights), "n_stocks_priced": n_priced,
        "by_market_benchmark": by_market,
        "caveat": ("portfolio_realized_return 是把 `portfolio_weight_measured` 那部分"
                  "權重的實際報酬加總，量不到的部分（權重缺口）視同 0% 報酬，"
                  "不重新分配給其他持股，也不假裝涵蓋率是 100%。"),
    }
