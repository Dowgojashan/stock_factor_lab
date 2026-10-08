# -*- coding: utf-8 -*-
"""三層監控指標（設計文件 §9）。

🔴 §9.0 狀態變數 vs 流量變數，決定能不能進預測區（§7.0）：
  - 環境層／過程層＝狀態變數（某一時點可直接觀測），可用於預測決策
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


# ============================================================ Hot Segment（環境層，狀態變數，
# 跟 M1-D 平行獨立——設計文件 §7.4b，2026-09-22 規格定案，2026-09-29 正式接線）
#
# 業界對應：Momentum 因子（Jegadeesh and Titman 1993），覆蓋率口徑對應 MSCI Momentum
# Indexes Methodology 的 coverage 概念。N/X 已用 2015-2023 真實資料校準（見
# `_design_test_hot_segment.py`）：TW 用 lag=1 自相關最強的 N=3；US 用 N=6（TW 校準出的
# N=3 在 US 上自相關偏弱，兩個市場各自照自己的結果選，不跨市場沿用同一個值）。

HOT_SEGMENT_N = {"TW": 3, "US": 6}  # 月數；XM 尚未獨立校準，見下方 hot_segment_layer 的檢查
HOT_SEGMENT_X = 0.10                 # 統一 10%（top decile），跨市場一致


def monthly_close(md) -> pd.DataFrame:
    """月底收盤價（`price:close` 月頻重取樣）——動能排名跟覆蓋率計算共用同一份，
    不要在呼叫端各自重算（沿用 `_design_test_hot_segment.py` 的既有教訓）。"""
    close = md.get_field("price:close")
    return close.resample("M").last()


def momentum_asof(monthly: pd.DataFrame, as_of: str, months: int) -> pd.Series:
    """`as_of` 當下，往前 `months` 個月的累積報酬（逐股票）。用 `price:close`
    直接算，不透過 factor pipeline（動能不是候選池因子）。資料不足時拋
    `ValueError`，呼叫端要誠實處理，不能靜默跳過。"""
    ts = pd.Timestamp(as_of)
    idx = monthly.index[monthly.index <= ts]
    if len(idx) <= months:
        raise ValueError(f"{as_of} 往前{months}個月的月頻資料不足（只有{len(idx)}筆）")
    end_px = monthly.loc[idx[-1]]
    start_px = monthly.loc[idx[-1 - months]]
    return (end_px / start_px - 1.0)


def hot_segment(ret_row: pd.Series, x_pct: float = HOT_SEGMENT_X) -> list[str]:
    """該時點的動能報酬序列 → 排名前 x_pct 的股票清單（NaN 一律排除）。"""
    valid = ret_row.dropna()
    if valid.empty:
        return []
    cutoff = valid.quantile(1 - x_pct)
    return valid[valid >= cutoff].index.tolist()


def hot_segment_layer(md, idx: pd.DataFrame, uids: list[str], mktcap_row: pd.Series,
                      as_of: str, market: str) -> dict:
    """環境層＋過程層混合指標：算出這一季的 Hot Segment 清單，以及目前代表策略
    對它的覆蓋率（市場層級聚合，用於 M1-D 旁的獨立觸發判定，見 `triggers.
    evaluate_hot_segment_quarter()`）。

    🔴 跟 Coverage Tilt 用的策略層級 `coverage_frac(r,t)`（見 `weights.py`）是
    **不同粒度**的計算，這裡只算彙總後的市場層級 coverage_count／coverage_mktcap，
    不要跟策略層級的覆蓋比例混用同一個函式（設計文件 §7.4b②／④是兩個獨立公式）。

    `mktcap_row`：`asof_row(mcap_wide, as_of)` 的輸出，跟 `environment_layer()`
    共用同一份市值資料。
    """
    from app._design_test_guaranteed_megacap_slot import selects_symbol

    if market not in HOT_SEGMENT_N:
        raise ValueError(f"Hot Segment 尚未替 {market} 校準 N 值（目前只有 TW/US），"
                         f"不可沿用其他市場的門檻，見設計文件§7.4b範圍限定")
    n_months = HOT_SEGMENT_N[market]

    monthly = monthly_close(md)
    mom = momentum_asof(monthly, as_of, n_months)
    hot = hot_segment(mom, HOT_SEGMENT_X)

    covered = set()
    for uid in uids:
        row = idx.loc[uid]
        for sym in hot:
            if sym in covered:
                continue
            if selects_symbol(md, row, as_of, sym) is True:
                covered.add(sym)
    cov_count = (len(covered) / len(hot)) if hot else float("nan")

    hot_mk = mktcap_row.reindex(hot)
    if hot_mk.isna().all() or hot_mk.sum() == 0:
        cov_mktcap = float("nan")
    else:
        covered_mk = mktcap_row.reindex(list(covered)).fillna(0.0)
        cov_mktcap = float(covered_mk.sum() / hot_mk.fillna(0.0).sum())

    return {
        "as_of": as_of, "market": market, "n_months": n_months, "x_pct": HOT_SEGMENT_X,
        "n_hot": len(hot), "hot_segment": hot,
        "coverage_count": cov_count, "coverage_mktcap": cov_mktcap,
    }


# ============================================================ 結果層（流量變數，僅回顧區可用）

#: §8待辦item10（2026-09-22）：window4 用股票層方法重算的真實 B_all（全買
#: 候選池，equal-weight-across-strategies，股票層 `performance.measure()`
#: 量測，跟 A_hrp／等權/市值加權大盤同一套量測系統）。修正前 `excess_vs_ball`
#: 直接借用 equal_weight_benchmark_return 當 B_all 的替身（A9決定，程式註解
#: 明講「數字剛好一樣是巧合正確不是設計正確」）——實測兩者其實不同
#: （B_all 8季累積 stock-level +8.24%／CAGR 4.04% vs 等權大盤替身 CAGR
#: 5.10%），差距約1pp/yr，不算巨大但不是「剛好相等」。目前只精確重算了
#: window4（TW/scheme E），其餘窗次/市場仍缺，`fetch_ball_returns()` 找不到
#: 對應區間時回傳 None，呼叫端會自動退回舊的等權大盤替身（見下）——不是新
#: bug，是誠實的資料涵蓋範圍限制。
_BALL_RECOMPUTE_PATHS = (
    Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "ball_stock_level_recompute.csv",  # window4
    Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "ball_stock_level_windows123.csv",  # windows1-3
)


def fetch_ball_returns(paths=_BALL_RECOMPUTE_PATHS) -> dict[tuple[str, str], float]:
    """讀`_recompute_ball_stock_level.py`／`_recompute_ball_windows123.py`算好、
    凍結存檔的真實B_all逐季報酬（window1~4全部涵蓋），key=(as_of, end) 字串
    二元組，找不到檔案就跳過（呼叫端會自動退回舊的等權大盤替身，不會崩潰）。"""
    out: dict[tuple[str, str], float] = {}
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        out.update({(r.as_of, r.end): float(r.stock_level_return) for r in df.itertuples()})
    return out


def outcome_layer(md_map: dict, weights: dict[str, float], as_of: str, end: str,
                  cap_weight_series: pd.Series | None = None,
                  ball_returns: dict[tuple[str, str], float] | None = None) -> dict:
    """已實現報酬、對等權大盤超額、對市值加權大盤超額（若提供 cap_weight_series，
    通常是 taiex_tr）、對 B_all 超額。

    🔴 2026-09-22（§8待辦item10）：`ball_returns` 若提供且該 (as_of,end) 有
    對應的真實股票層 B_all（見 `fetch_ball_returns()`），`excess_vs_ball` 用
    真實 B_all 算；**否則退回舊行為**（借用 equal_weight_benchmark_return 當
    替身，A9 決定，`walkforward_matrix.py` 定義的 B_all＝全宇宙等權跟這裡
    概念相同，只是舊版沒有真的用股票層方法重算過）——退回行為只是誠實的
    資料涵蓋範圍限制，不是靜默錯誤（`ball_return_is_real` 欄位標記用的是
    哪一種，供事後稽核）。
    🔴 流量變數，不可用來驅動預測動作（§9.0／§7.0）。
    """
    res = measure(md_map, weights, as_of, end)
    market = list(md_map.keys())[0]
    port_ret = res["portfolio_realized_return"]
    ew_bench = res["by_market_benchmark"][market]["equal_weight_benchmark_return"]

    real_ball = (ball_returns or {}).get((as_of, end))
    ball_ret = real_ball if real_ball is not None else ew_bench

    out = {
        "as_of": as_of, "end": end,
        "portfolio_realized_return": port_ret,
        "equal_weight_benchmark_return": ew_bench,
        "excess_vs_equal_weight": (port_ret - ew_bench) if ew_bench is not None else None,
        "ball_benchmark_return": ball_ret,
        "ball_return_is_real": real_ball is not None,
        "excess_vs_ball": (port_ret - ball_ret) if ball_ret is not None else None,
    }
    if cap_weight_series is not None:
        ts = cap_weight_series.index
        d0 = ts[ts <= pd.Timestamp(as_of)].max()
        d1 = ts[ts <= pd.Timestamp(end)].max()
        cw_ret = float(cap_weight_series.loc[d1] / cap_weight_series.loc[d0] - 1.0)
        out["cap_weight_benchmark_return"] = cw_ret
        out["excess_vs_cap_weight"] = port_ret - cw_ret
    return out


def monthly_breakdown(md_map: dict, weights: dict[str, float], as_of: str, end: str,
                      cap_weight_series: pd.Series | None = None) -> list[dict]:
    """把一季 [as_of, end] 拆成月頻子區間，各自呼叫 `outcome_layer()`——沿用
    既有已驗證機制，不是新的量測邏輯，只是切得更細（設計文件 §8 多時間尺度
    解釋「當月的解釋題」，方案B：敘事層級彙整，見開發追蹤）。🔴 流量變數，
    只能進回顧區，跟 `outcome_layer()` 同一個限制（§9.0）。持股在整季內
    視為不變（跟本系統既有的「checkpoint間靜態持有」假設一致，見D25）。
    """
    boundaries = [pd.Timestamp(as_of)] + list(
        pd.date_range(as_of, end, freq="M")) + [pd.Timestamp(end)]
    # 去重、排序（as_of/end 可能剛好落在月底，freq="ME"會重複產生同一天）
    boundaries = sorted(set(boundaries))
    rows = []
    for i in range(len(boundaries) - 1):
        m_start, m_end = boundaries[i].strftime("%Y-%m-%d"), boundaries[i + 1].strftime("%Y-%m-%d")
        if m_start == m_end:
            continue
        outcome = outcome_layer(md_map, weights, m_start, m_end, cap_weight_series)
        rows.append({"month_start": m_start, "month_end": m_end, **outcome})
    return rows


# ============================================================ expanding window 分位數

def expanding_percentile(history: pd.Series, as_of_date: str, value: float) -> float:
    """只用 as_of_date 之前（不含）的歷史序列算百分位，避免用到被判斷的那一期本身
    （§9「歷史分布一律用 expanding window」）。`history` 的 index 須為日期。"""
    ts = pd.Timestamp(as_of_date)
    past = history[history.index < ts].dropna()
    if len(past) == 0:
        return float("nan")
    return float((past < value).mean())


# ============================================================ 回撤觸發（2026-10-08新增，
# 跟 M1-D／Hot Segment平行、獨立判定，見 triggers.py 的 DrawdownCondition）
#
# 🔴 這裡算的「目前回撤水位」本身是**狀態變數**（當下相對歷史峰值的距離，跟
# `environment_layer()`算「目前前N大市值佔比」同一個性質：某一時點可直接觀測的
# 水位），即使它是從流量變數（`outcome["portfolio_realized_return"]`逐季累積）
# 算出來的——跟`triggers.py`開頭的三條硬規則表一致：流量變數不可驅動投組變更，
# 但可以驅動A0/A4/A5（不動/升級），而「目前回撤多深」正是用來驅動升級判斷，不是
# 用來驅動W2c/CoverageTilt/RepSwap這類投組變更動作，見`triggers.evaluate_
# drawdown_quarter()`docstring的完整說明。

def drawdown_state(nav_peak: float, nav_now: float) -> float:
    """`nav_now` 相對 `nav_peak`（本次模擬迄今最高點）的回撤，<=0（0＝站在新高）。"""
    if nav_peak <= 0:
        return 0.0
    return float(nav_now / nav_peak - 1.0)
