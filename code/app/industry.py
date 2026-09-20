# -*- coding: utf-8 -*-
"""個股產業分類（供「產業集中度」診斷未來使用；設計文件 §7.3 #6、開發追蹤 D47/D49）。

🔴 資料源查證（D47）：本專案資料庫（`company`/`stock` 等 12 張表）完全沒有產業
分類欄位，跟既有的「產業β」缺口（`stage4_strategy_map.py`／`decision_layer_arms.py`／
`ops/tools.py`）同一個根因。已請 `stock_factor_collector` 專案（sibling repo，
`D:\\git\\stock_factor_collector`）用 `CollectorFactory/IndustryCollector.py`
（FinMind `TaiwanStockInfo` / yfinance）收集，2026-09-18 產出
`_analysis_outputs_applayer/industry_raw.csv`／`.parquet`，覆蓋率已查證 100%
（TW 1775/1775、US 2994/2994，見開發追蹤 D49）。

🔴 TWSE／TPEx 分類命名不一致（D49 查證）：FinMind 的 `industry_category` 對同一個
實質產業，TWSE（上市）跟 TPEx（上櫃）有時用不同名稱——已用 `type`（twse/tpex/emerging）
欄位做過交叉表查證（`tw_type_industry_crosstab.csv`），確認以下 6 組是**乾淨二分**
（同一產業只出現在其中一邊，兩邊從未混合）：其他電子業/其他電子類、居家生活/居家生活類、
數位雲端/數位雲端類、綠能環保/綠能環保類、運動休閒/運動休閒類、金融保險/金融業。
`_NORMALIZE_MAP` 把這 6 組合併成單一標籤（用 TWSE 那邊的名稱）。

**刻意不合併的（查過交叉表，判斷是真實差異或證據不足，不是命名慣例問題）**：
  - 化學生技醫療 vs 生技醫療業：TWSE 自己同時用兩個名字分裝不同公司
    （8 檔 vs 33 檔），不是單純的交易所命名差異，可能是真實子分類區別
  - 塑膠工業 vs 橡膠工業：不同物料產業，字面相似是巧合（橡膠工業其實是
    TWSE-only，不是像原始回報講的「兩邊混合」，已查證更正）
  - 電子工業（TWSE-only，201 檔，佔全部 TW 1775 檔的 11%）vs 電子零組件業／
    電子通路業等：沒有對應的 TPEx 拆分可合併，但顆粒度極粗，用這個分類算
    集中度時要留意「電子工業」這個超大類別可能掩蓋了內部更細緻的子產業集中——
    這是分類系統本身的顆粒度限制，不是本模組能解決的，須在論文限制裡誠實揭露

**已知限制（沿用，非本模組新增）**：分類是「現在最新」的靜態快照，不追蹤歷史異動；
美股範圍＝ `russell3000`，繼承既有的倖存者偏誤限制。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

INDUSTRY_RAW_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "_analysis_outputs_applayer" / "industry_raw.csv"
)

# TWSE 名稱 -> TPEx 名稱（合併後統一用 TWSE 那邊的標籤）
_NORMALIZE_MAP = {
    "其他電子類": "其他電子業",
    "居家生活類": "居家生活",
    "數位雲端類": "數位雲端",
    "綠能環保類": "綠能環保",
    "運動休閒類": "運動休閒",
    "金融業": "金融保險",
}

# 電子工業是 TWSE 專用的粗顆粒度大分類（201 檔），沒有可合併的 TPEx 對應——
# 只是記錄下來供呼叫端在解讀集中度數字時參考，不做任何自動處理
COARSE_CATEGORIES = ("電子工業",)


def load_industry_map(market: str = "TW") -> pd.Series:
    """回傳 company_symbol -> 正規化後產業分類 的對照（已合併 TWSE/TPEx 同義分類，
    見模組開頭的 `_NORMALIZE_MAP` 與其查證依據）。"""
    df = pd.read_csv(INDUSTRY_RAW_PATH, dtype={"company_symbol": str})
    sub = df[df["market"] == market].copy()
    sub["industry_normalized"] = sub["industry_name"].replace(_NORMALIZE_MAP)
    return sub.set_index("company_symbol")["industry_normalized"]


def industry_exposure(weights: dict[str, float], industry_map: pd.Series | None = None,
                      market: str = "TW") -> dict[str, float]:
    """把投組權重（company_symbol -> weight）依產業分類彙總，回傳
    {產業名稱 -> 佔投組總權重的比重}（缺分類的股票歸為 "未分類"，不悄悄
    丟掉——避免覆蓋率不足時集中度被低估）。`industry_map` 可由呼叫端預先
    `load_industry_map()` 一次、重複傳入避免每次都重讀 CSV（歷史序列計算
    時會呼叫很多次，見 `_prelim_industry_concentration_history.py`）。"""
    if industry_map is None:
        industry_map = load_industry_map(market)
    exposure: dict[str, float] = {}
    for symbol, w in weights.items():
        industry = industry_map.get(symbol, "未分類")
        exposure[industry] = exposure.get(industry, 0.0) + w
    return exposure


def max_industry_weight(weights: dict[str, float], industry_map: pd.Series | None = None,
                        market: str = "TW") -> tuple[str, float]:
    """投組裡佔比最大的單一產業，回傳 (產業名稱, 佔比)。給 M8 產業集中度
    診斷用（§7.3 #6、開發追蹤 D56）。"""
    exposure = industry_exposure(weights, industry_map, market)
    if not exposure:
        return ("未分類", 0.0)
    top = max(exposure.items(), key=lambda x: x[1])
    return top
