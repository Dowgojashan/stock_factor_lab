# -*- coding: utf-8 -*-
"""L1+ · 情境比對（應用層開發追蹤.md §9.4/§9.7，解釋 agent 專用資料，2026-09-11）

H3（§9.5）：情境比對三種方法＋股票層，**全部先做**，看效果再挑，不是本輪就
先驗證選一種：

    ① weighted_annual_returns()      群報酬型態最像哪一年（本檔案）
    ② engine.historical_same_setting_windows()  同設定歷史窗次 is_end→oos（見 engine.py）
    ③ regime_snapshot()              regime 標籤（牛/熊/危機/盤整）（本檔案）
    ④ stock_level_analog()           股票層：現在持股在類比年份的實際表現（本檔案）

🔴 H8：群層級的兩個方法（①③的群報酬分量）只對主線樹成立——`ClusterFootprint`
本身就是 `build_footprint()` 的產物，已經綁死主線樹，這裡不重複檢查。

🔴 R17（抗幻覺鐵則 vs 判讀的界線）：這裡每個函式只算**事實**（歷史上發生過什麼），
不做「所以應該怎麼做」的價值判斷——那是 LLM 在 `explain.py` 的工作，且僅限於
可逐句核對數字來源的「事實推論」。
"""
from __future__ import annotations

import pandas as pd

from .cluster_kb import ClusterFootprint


def weighted_annual_returns(footprint: ClusterFootprint) -> dict[int, float]:
    """①：把本次持股在主線樹各群的佔比，套進 `cluster_annual_returns`，回推
    「如果這個群佔比結構一直存在，歷史上每一年會是什麼報酬」——不是預測，是用
    已發生的群報酬回答「這個組合的性格比較像哪一年」需要的原始數字。

    某群在某年沒有資料時（樹的觀察窗起訖不是每群都對齊），該年只用「當年有資料
    的群」重新正規化權重計算，不假裝缺資料的群報酬是 0。
    """
    shares = {cid: footprint.share(cid) for cid in footprint.clusters_used}
    all_years: set[int] = set()
    for cid in footprint.clusters_used:
        df = footprint.annual_returns.get(cid)
        if df is not None:
            all_years |= set(int(y) for y in df["year"].tolist())

    out: dict[int, float] = {}
    for year in sorted(all_years):
        total_w, acc = 0.0, 0.0
        for cid, w in shares.items():
            df = footprint.annual_returns.get(cid)
            if df is None:
                continue
            row = df[df["year"] == year]
            if row.empty:
                continue
            acc += w * float(row["ret"].iloc[0])
            total_w += w
        if total_w > 0:
            out[year] = acc / total_w
    return out


def regime_snapshot(market: str, as_of_map: dict[str, str], regime_avg_ret: dict[str, float]) -> dict:
    """③：現在是牛/熊/危機/盤整哪一段（`ops.tools.t11_get_current_regime`，跟
    研究部階段 2a 同一套 zigzag 規則），對照這批策略歷史上在該 regime 標籤下的
    平均報酬（`risk.RiskReport.regime_avg_ret`，T8 已經算好，這裡不重算）。

    `as_of_map`：`{市場: 交易日字串}`（`ui._resolve_stock_holdings()` 回傳的
    `market_dates`），**不能**用 `assess_stock_level` 那種已經拼接成
    `"TW:2026-01-01"` 展示用字串的 `as_of`——那不是可解析的單一日期。
    XM 混合台美，沒有單一大盤指數可判——分別給 TW／US 兩份現況，不硬併成一個，
    各自用自己市場對到的交易日。
    """
    from ops import tools as T

    legs = ["TW", "US"] if market == "XM" else [market]
    current = {}
    for m in legs:
        try:
            current[m] = T.t11_get_current_regime(m, as_of=as_of_map.get(m))
        except Exception as e:  # noqa: BLE001 — 現況判定失敗不該讓整份解釋掛掉
            current[m] = {"error": str(e)}
    return {
        "current_regime": current,
        "historical_avg_return_by_regime": regime_avg_ret,
        "note": ("current_regime 是暫定判定（zigzag 回顧式演算法，最後一段可能尚未"
                 "被下一次反轉確認，見 T11 docstring）；historical_avg_return_by_regime "
                 "是這批策略過去在各 regime 標籤下的歷史平均報酬，不是本期預測，"
                 "兩者不可混為一談。"),
    }


def stock_level_analog(md_map: dict, symbols: list[str], year: int) -> dict:
    """④（老師 9-8 逐字稿原話③：「這些股票居然最慘，過去是不是有發生過」）：
    現在持股中，有哪些股票在類比年份 `year` 就已經存在，那一年的實際報酬多少。

    只用呼叫端已經載入的 `MarketData`（`md.data.get("price:close")`，跟
    `MarketData.__init__` 取得收盤價的方式相同），**不重新查資料庫**——這批
    price frame 本來就已經因為股票層級持股解析而載入在記憶體裡了。

    找不到資料的股票（那個市場沒有這檔、或當年還沒有價格資料）標記
    `existed_then=False`，不假裝有報酬可算。
    """
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year}-12-31")

    per_stock: dict[str, dict] = {}
    for md in md_map.values():
        close = md.data.get("price:close")
        if close is None:
            continue
        cols = [s for s in symbols if s in close.columns and s not in per_stock]
        if not cols:
            continue
        window = close.loc[(close.index >= year_start) & (close.index <= year_end), cols]
        for s in cols:
            series = window[s].dropna()
            if len(series) < 2:
                per_stock[s] = {"existed_then": False, "year_return": None}
            else:
                per_stock[s] = {"existed_then": True,
                                "year_return": float(series.iloc[-1] / series.iloc[0] - 1.0)}
    for s in symbols:
        per_stock.setdefault(s, {"existed_then": False, "year_return": None})

    existed = [v["year_return"] for v in per_stock.values() if v["existed_then"]]
    return {
        "year": year,
        "n_checked": len(symbols),
        "n_existed_then": len(existed),
        "avg_return_of_existing": (sum(existed) / len(existed)) if existed else None,
        "per_stock": per_stock,
    }
