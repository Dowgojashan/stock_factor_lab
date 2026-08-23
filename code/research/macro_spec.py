# -*- coding: utf-8 -*-
"""階段 2b · 總經指標規格與發布滯後（W-01 產出）

為什麼需要（研究部 v9 階段2b · P6 前視防護）：
  總經指標**不是在它所描述的期間結束當下就知道的**。2026 年 3 月底做決策時，
  你不可能知道 3 月的 CPI——那要等 4 月初才公布。若直接用「3 月的 CPI」建模，
  就是用了當時不存在的資訊 = 前視偏誤。
  與財報的 `filingDate` 是同一個「時點可得原則」。

本模組是滯後量的**單一事實來源**：規格寫在這裡，`lag_spec.json` 由此匯出並凍結，
out 期沿用同一份、不重算。

⚠️ **一個做得到與做不到的界線（必須寫進論文 limitations）**：
  - ✅ **做得到「時點對齊」**：只使用「該時點已經發布」的期間資料。
  - ❌ **做不到「版本對齊（vintage）」**：拿不到「該時點當下看到的原始數值」。
    GDP 與失業率事後都會被修正，而 TEJ／FMP 存的是**修正後的最新值**。
    美國有 ALFRED 保存歷史版本，台灣沒有對應服務。
    → 我們消除了「用到未來期間」的偏誤，但殘留「用到未來修正值」的偏誤。
    後者影響遠小於前者（修正幅度通常在小數點後），但必須誠實揭露。

用法：
    from research.macro_spec import SPEC, available_period, dump_lag_spec
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class Indicator:
    axis: str                    # 概念軸：growth / inflation / rate_level / rate_direction / risk
    name: str                    # 指標名
    agency: str                  # 發布機構
    freq: Literal["M", "Q", "D"]
    publish_rule: str            # 人話描述的發布規則
    lag_months: int              # 期間結束後幾個月才可用（見 available_period）
    source_url: str
    vintage_available: bool      # 是否拿得到當時的原始值（未修正版）
    note: str = ""


# ============================================================================
# 美股（FMP）
# ============================================================================
US = {
    "growth": Indicator(
        axis="growth", name="實質 GDP 年增率（advance estimate）",
        agency="BEA（美國經濟分析局）", freq="Q",
        publish_rule="季末後約 30 天發布 advance estimate；second/third estimate 分別為 +60／+90 天",
        lag_months=1,
        source_url="https://www.bea.gov/data/gdp/gross-domestic-product",
        vintage_available=True,   # ALFRED 有歷史版本
        note="採 advance（首次發布）對應真實決策情境。若資料源只存修正後值，見模組開頭的限制說明。",
    ),
    "growth_alt": Indicator(
        axis="growth", name="失業率", agency="BLS（Employment Situation）", freq="M",
        publish_rule="次月第一個週五 08:30 ET 發布前一月資料",
        lag_months=1,
        source_url="https://www.bls.gov/news.release/empsit.nr0.htm",
        vintage_available=True,
        note="月頻、滯後短，若 GDP 的季頻粒度太粗可改用此欄。",
    ),
    "inflation": Indicator(
        axis="inflation", name="CPI 年增率", agency="BLS", freq="M",
        publish_rule="次月第二週（約 10–14 日）08:30 ET 發布前一月資料",
        lag_months=1,
        source_url="https://www.bls.gov/bls/news-release/cpi.htm",
        vintage_available=False,  # CPI 幾乎不修正（僅季節調整因子重算）
        note="CPI 原始指數基本不修正，vintage 問題最小。",
    ),
    "rate_level": Indicator(
        axis="rate_level", name="聯邦基金有效利率／10 年期公債殖利率",
        agency="Fed H.15／美國財政部", freq="D",
        publish_rule="每日盤後即時公布，無滯後",
        lag_months=0,
        source_url="https://www.federalreserve.gov/releases/h15/",
        vintage_available=True,
        note="市場價格類，發布即為最終值，不修正。",
    ),
    "rate_direction": Indicator(
        axis="rate_direction", name="殖利率曲線斜率（10Y − 2Y）",
        agency="美國財政部", freq="D",
        publish_rule="每日即時，無滯後",
        lag_months=0,
        source_url="https://home.treasury.gov/interest-rates-data-csv-archive",
        vintage_available=True,
    ),
    # risk 軸：v9 定案留白（原擬用 VIX）
}

# ============================================================================
# 台股（TEJ）
# ============================================================================
TW = {
    "growth": Indicator(
        axis="growth", name="實質 GDP 年增率",
        agency="行政院主計總處", freq="Q",
        publish_rule="季末後約 1 個月發布『概估統計』；『初步統計』約季末後 2 個月由國民所得統計評審會審議發布",
        lag_months=2,
        source_url="https://www.stat.gov.tw/Point.aspx?sid=t.1&n=3580&sms=11480",
        vintage_available=False,
        note="🔴 2026-08-22 collector 查證後採用保守值：主計總處只提供最新修正版，"
             "不區分概估／初步。collector 能實測的只有『2026Q2 在季末後第53天已存在』，"
             "介於概估(~30天)與初步統計(~60天)之間、無法判定是哪一階段。"
             "已依 collector 建議改採 lag_months=2（保守）：這個選擇不對稱——"
             "高估滯後只是少用一點資訊，低估滯後直接產生前視偏誤，故寧可保守。"
             "原值 lag_months=1 已作廢，見 開發待辦追蹤.md N-05。",
    ),
    "growth_alt": Indicator(
        axis="growth", name="工業生產指數年增率",
        agency="經濟部統計處", freq="M",
        publish_rule="次月 23 日發布前月工業生產統計（初步統計於次月 26 日前 16:00）",
        lag_months=1,
        source_url="https://www.moea.gov.tw/Mns/dos/bulletin/Bulletin.aspx?kind=6&html=1&menu_id=6725",
        vintage_available=False,
        note="月頻，粒度優於季頻 GDP。",
    ),
    "inflation": Indicator(
        axis="inflation", name="CPI 年增率", agency="行政院主計總處", freq="M",
        publish_rule="次月 5–8 日發布前一月資料（無固定日）",
        lag_months=1,
        source_url="https://www.stat.gov.tw/Point.aspx?sid=t.2&n=3581&sms=11480",
        vintage_available=False,
        note="台灣 CPI 亦幾乎不修正。",
    ),
    "rate_level": Indicator(
        axis="rate_level", name="央行重貼現率／貨幣市場利率",
        agency="中央銀行", freq="D",
        publish_rule="理監事會決議當日即時生效並公布；貨幣市場利率每日",
        lag_months=0,
        source_url="https://www.cbc.gov.tw/tw/lp-370-1.html",
        vintage_available=True,
    ),
    "rate_direction": Indicator(
        axis="rate_direction", name="景氣對策信號綜合判斷分數",
        agency="國家發展委員會", freq="M",
        publish_rule="**每月 27 日**發布前一月的景氣對策信號、分數與領先／同時指標",
        lag_months=1,
        source_url="https://index.ndc.gov.tw/",
        vintage_available=False,
        note="台股此軸用景氣訊號取代殖利率曲線（v9 定案：台美對齊『概念軸』而非同一指標）。"
             "⚠️ 領先指標會隨基期調整回溯修正，vintage 問題比 CPI 大。",
    ),
    # risk 軸：留白
}

SPEC = {"US": US, "TW": TW}

#: 台美各自實際採用哪一個指標當該軸的主欄位（⏳ W-02 建置時定案）
AXES = ("growth", "inflation", "rate_level", "rate_direction")


# ============================================================================
# 可得性計算
# ============================================================================

def available_period(ind: Indicator, decision_month: str | pd.Period) -> pd.Period | None:
    """在 `decision_month` **月底**做決策時，該指標最新可用的**資料期間**。

    例（CPI，lag_months=1）：
        decision_month = 2026-03（3 月底）
        → 3 月的 CPI 要 4 月初才發布 → 不可用
        → 2 月的 CPI 已於 3 月初發布  → 可用
        → 回傳 Period('2026-02')

    例（US GDP，季頻 lag_months=1）：
        decision_month = 2026-04（4 月底）
        → Q1 於季末（3/31）後約 30 天發布，即 4 月底 → **邊界情況，保守判定為不可用**
        → 回傳 Q4-2025
    """
    m = pd.Period(decision_month, freq="M")
    if ind.freq in ("D", "M"):
        return m - ind.lag_months

    # 季頻：找出「季末 + lag_months 個月」已完全落在決策月之前的最近一季
    q = (m - ind.lag_months).asfreq("Q", how="end")
    # 保守：若該季的可得月份 == 決策月本身，視為當月才剛發布、不採用
    if (q.asfreq("M", how="end") + ind.lag_months) >= m:
        q = q - 1
    return q


def dump_lag_spec(path) -> dict:
    """匯出成凍結的 JSON（out 期沿用同一份、不重算）。"""
    import json
    from pathlib import Path

    out = {
        "_doc": "階段2b 總經指標發布滯後規格。時點對齊用，非版本(vintage)對齊——見 macro_spec.py 開頭限制說明。",
        "_generated_by": "research.macro_spec.dump_lag_spec",
        "markets": {
            mkt: {key: asdict(ind) for key, ind in inds.items()}
            for mkt, inds in SPEC.items()
        },
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def summary() -> pd.DataFrame:
    rows = []
    for mkt, inds in SPEC.items():
        for key, i in inds.items():
            rows.append({"市場": mkt, "軸": key, "指標": i.name, "機構": i.agency,
                         "頻率": i.freq, "滯後(月)": i.lag_months,
                         "可得原始版本": "✓" if i.vintage_available else "✗",
                         "發布規則": i.publish_rule})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print(summary().to_string(index=False))
    print("\n可得性範例（決策月 = 2026-03 月底）:")
    for mkt, inds in SPEC.items():
        for key, i in inds.items():
            print(f"  [{mkt}] {key:<15} → {available_period(i, '2026-03')}")
