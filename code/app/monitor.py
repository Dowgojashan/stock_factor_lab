# -*- coding: utf-8 -*-
"""三層監控指標（設計文件 §9）。

🔴 §9.0 狀態變數 vs 流量變數，決定能不能進前瞻區（§7.0）：
  - 環境層／過程層＝狀態變數（某一時點可直接觀測），可用於前瞻決策
  - 結果層＝流量變數（一段期間實現的結果），**只能進回顧區，不可驅動動作**

🔴 §9「歷史分布一律用 expanding window」：判斷第 t 季時只用 t 之前的序列，
不可用含當期在內的彙總分布（`walkforward_matrix_detail.csv` 的 oos_max 皆為
2025-12，用 45 窗彙總判斷 2025 等於用到被判斷的那一期本身）。

範圍（2026-09-17 初版）：
  ✅ 環境層：前 N 大市值權重佔比（含 Q1 五分位）、市場廣度
  ✅ 結果層：已實現報酬、對等權／市值加權／B_all 超額（全部靠 performance.measure()）
  ✅ 過程層：不重複股票數、最大單檔權重、持股延續率、投組市值加權中位數、
     factor_exposure_F1 最大佔比（用 `candidate_index.parquet` 的 F1_band 直接算）
  ⬜ 過程層：`max_cluster_share`（需要 HRP 樹的分群指派，尚未接——那條路徑目前
     只有 `risk.assess()` 走完整 `Holdings` 物件才能拿到，跟本檔案採用的輕量
     `resolve_strategy_holdings` 直接算不是同一套管線，留待後續評估怎麼接）
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.performance import measure

MARKET = "TW"
TOPN = 10


# ============================================================ 資料載入（共用）

def fetch_mcap_wide(conn, market: str = MARKET, start: str = "2007-01-01") -> pd.DataFrame:
    """全市場逐日市值，寬表（index=date, columns=symbol）。"""
    exch = {"TW": "('TWSE')", "US": "('NASDAQ','NYSE','AMEX')"}[market]
    q = (f"SELECT s.date, c.company_symbol, s.market_capital FROM stock s "
        f"JOIN company c ON s.company_id=c.id WHERE c.exchange_name IN {exch} "
        f"AND s.market_capital IS NOT NULL AND s.date >= '{start}' ORDER BY s.date")
    d = pd.read_sql(q, conn)
    d["date"] = pd.to_datetime(d["date"])
    return d.pivot_table(index="date", columns="company_symbol", values="market_capital", aggfunc="last")


def asof_row(wide: pd.DataFrame, as_of: str) -> pd.Series:
    ts = pd.Timestamp(as_of)
    avail = wide.index[wide.index <= ts]
    if len(avail) == 0:
        raise ValueError(f"{as_of} 之前沒有任何資料")
    return wide.loc[avail.max()]


# ============================================================ 環境層（狀態變數，不依賴投組持股）

def environment_layer(mcap_wide: pd.DataFrame, as_of: str, topn: int = TOPN) -> dict:
    """前 N 大市值權重佔比 ＋ Q1（最大市值 20%）集中度——跟 §3.2／M1-D 同一套量測。
    純指數端計算，不需要投組持股，可以在任何 as_of 直接算，不受換股雜訊污染。
    """
    row = mcap_wide.pipe(lambda w: asof_row(w, as_of)).dropna()
    total = row.sum()
    sorted_desc = row.sort_values(ascending=False)
    top_n_weight = sorted_desc.iloc[:topn].sum() / total

    ranks = row.rank(pct=True)
    q1 = row[ranks >= 0.8]
    q1_weight = q1.sum() / total

    return {
        "as_of": as_of, "n_universe": len(row),
        "top_n": topn, "top_n_weight": float(top_n_weight),
        "q1_weight": float(q1_weight),
    }


def market_breadth(md, as_of: str, end: str, cap_weight_return: float) -> dict:
    """狀態變數：市場廣度＝跑贏「指數」的個股比例（trailing，以 as_of~end 這段
    期間的個股報酬 vs 同期**市值加權**指數報酬比較，「指數」在財務上慣例指
    市值加權，不是等權基準——呼叫端務必傳 TAIEX 這類市值加權報酬，不可誤傳
    等權基準進來，否則算出來的不是設計文件講的「市場廣度」）。以 as_of 為準
    的一段回顧窗，描述「現在市場有多窄」，不是對 as_of 之後做預測，故仍歸類為
    狀態變數（§9.0 的既有分類，market breadth 本來就列在狀態變數例子裡）。
    """
    close = md.data.get("price:close")
    idx = close.index
    d0 = idx[idx <= pd.Timestamp(as_of)].max()
    d1 = idx[idx <= pd.Timestamp(end)].max()
    p0, p1 = close.loc[d0], close.loc[d1]
    rets = (p1 / p0 - 1.0).dropna()
    if len(rets) == 0:
        return {"as_of": as_of, "end": end, "n_stocks": 0, "breadth": None}
    breadth = float((rets > cap_weight_return).mean())
    return {"as_of": as_of, "end": end, "n_stocks": len(rets), "breadth": breadth}


# ============================================================ 過程層（狀態變數，依賴投組持股）

def process_layer(weights: dict[str, float], mcap_row: pd.Series,
                  prev_weights: dict[str, float] | None,
                  member_uids: list[str], candidate_idx: pd.DataFrame) -> dict:
    """不重複股票數／最大單檔權重／持股延續率／投組市值加權中位數／
    factor_exposure_F1 最大佔比。"""
    n_unique = len(weights)
    max_w = max(weights.values()) if weights else 0.0
    max_sym = max(weights, key=weights.get) if weights else None

    # 持股延續率：跟前一期的權重交集（用 min(w_new, w_old) 加總，
    # 0=完全換過，1=完全沒換——跟 turnover 互補：continuity = 1 - turnover）
    continuity = None
    if prev_weights:
        keys = set(weights) | set(prev_weights)
        overlap = sum(min(weights.get(s, 0.0), prev_weights.get(s, 0.0)) for s in keys)
        continuity = float(overlap)  # 已經是 0~1（權重本身加總為1）

    # 投組市值加權中位數：把持股依市值排序，找累計權重達 50% 的那一檔市值
    med_mktcap = None
    rows = []
    for s, w in weights.items():
        mc = mcap_row.get(s)
        if pd.notna(mc) and mc > 0:
            rows.append((mc, w))
    if rows:
        rows.sort(key=lambda x: x[0])
        total_w = sum(w for _, w in rows)
        cum = 0.0
        for mc, w in rows:
            cum += w
            if cum >= total_w / 2:
                med_mktcap = mc
                break

    # factor_exposure_F1：目前這批成員策略在各 F1_band 的分布（給 M5 算 L1 距離用）
    f1_dist = factor_exposure_distribution(member_uids, candidate_idx)
    f1_share = None
    f1_top_band = None
    if f1_dist:
        f1_top_band = max(f1_dist, key=f1_dist.get)
        f1_share = f1_dist[f1_top_band]

    return {
        "n_unique_stocks": n_unique, "max_stock_weight": max_w, "max_stock_symbol": max_sym,
        "continuity_vs_prev": continuity,
        "median_mktcap_weighted": med_mktcap,
        "factor_exposure_f1_top_band": f1_top_band, "factor_exposure_f1_share": f1_share,
        "factor_exposure_f1_distribution": f1_dist,
    }


def factor_exposure_distribution(member_uids: list[str], candidate_idx: pd.DataFrame) -> dict[str, float]:
    """目前成員策略在各 F1_band 的佔比分布（dict: band -> 佔比，加總為 1）。
    給 M5「對登記時的 L1 距離」用——L1 距離需要完整分布向量，不能只看最大佔比
    （兩個分布最大佔比一樣，形狀可能完全不同）。"""
    if not member_uids:
        return {}
    sub = candidate_idx.loc[candidate_idx.index.isin(member_uids)]
    if len(sub) == 0 or "F1_band" not in sub.columns:
        return {}
    counts = sub["F1_band"].value_counts(normalize=True)
    return {str(k): float(v) for k, v in counts.items()}


def l1_distance(dist_a: dict[str, float], dist_b: dict[str, float]) -> float:
    """兩個分布（dict: 類別 -> 佔比）的 L1 距離＝ sum(|a_i - b_i|)。
    缺的類別視為佔比 0。回傳範圍 0（完全相同）~2（完全不重疊）。"""
    keys = set(dist_a) | set(dist_b)
    return sum(abs(dist_a.get(k, 0.0) - dist_b.get(k, 0.0)) for k in keys)


# ============================================================ 結果層（流量變數，僅回顧區可用）

def outcome_layer(md_map: dict, weights: dict[str, float], as_of: str, end: str,
                  cap_weight_series: pd.Series | None = None) -> dict:
    """已實現報酬、對等權大盤超額、對市值加權大盤超額（若提供 cap_weight_series，
    通常是 taiex_tr）、對 B_all 超額（B_all＝ equal_weight_benchmark_return，
    `walkforward_matrix.py:62` 已定義 B_all＝全宇宙等權，跟這裡同一件事，見 A9）。
    🔴 流量變數，不可用來驅動前瞻動作（§9.0／§7.0）。
    """
    res = measure(md_map, weights, as_of, end)
    market = list(md_map.keys())[0]
    port_ret = res["portfolio_realized_return"]
    ew_bench = res["by_market_benchmark"][market]["equal_weight_benchmark_return"]

    out = {
        "as_of": as_of, "end": end,
        "portfolio_realized_return": port_ret,
        "equal_weight_benchmark_return": ew_bench,
        "excess_vs_equal_weight": (port_ret - ew_bench) if ew_bench is not None else None,
        "excess_vs_ball": (port_ret - ew_bench) if ew_bench is not None else None,  # B_all＝等權大盤（A9）
    }
    if cap_weight_series is not None:
        ts = cap_weight_series.index
        d0 = ts[ts <= pd.Timestamp(as_of)].max()
        d1 = ts[ts <= pd.Timestamp(end)].max()
        cw_ret = float(cap_weight_series.loc[d1] / cap_weight_series.loc[d0] - 1.0)
        out["cap_weight_benchmark_return"] = cw_ret
        out["excess_vs_cap_weight"] = port_ret - cw_ret
    return out


# ============================================================ expanding window 分位數

def expanding_percentile(history: pd.Series, as_of_date: str, value: float) -> float:
    """只用 as_of_date 之前（不含）的歷史序列算百分位，避免用到被判斷的那一期本身
    （§9「歷史分布一律用 expanding window」）。`history` 的 index 須為日期。"""
    ts = pd.Timestamp(as_of_date)
    past = history[history.index < ts].dropna()
    if len(past) == 0:
        return float("nan")
    return float((past < value).mean())
