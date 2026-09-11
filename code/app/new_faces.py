# -*- coding: utf-8 -*-
"""L1+ · 新面孔 vs 老面孔（應用層開發追蹤.md §7 P3，2026-09-11）

起因：查證 `逐字稿 (1).docx`（`.venv` python + zipfile 解 `word/document.xml`，
非憑印象）找到老師 9-8 原話：

    「我到底應該有哪些策略，哪些因質很可能是現在這些股票最火的，還是說⋯
    以前都還不錯都蠻賺錢的，同時有取到他有沒有這一類的東西」

老師問的是：選出來的股票，是「現在最火」的，還是「一直都在」的常客——系統
有沒有辦法呈現這個區別。

設計依據（不是憑空發明，是既有兩個時鐘架構的直接延伸）：
    策略清單在慢時鐘週期內（B2，2 年）固定不變；股票持股隨快時鐘（B1，季度
    財報）更新。⇒ 拿「策略選定當下（is_end）」與「現在」兩個時間點的股票持股
    做差集，天然回答老師的問題：
        老面孔（persistent）：兩個時點都持有——策略選定時就在，現在還在，
                              對應老師說的「以前都還不錯都蠻賺錢」
        新面孔（new）      ：只在現在持有——策略選定之後，財報更新才冒出來的，
                              對應老師說的「現在這些股票最火的」
        淡出（exited）     ：只在選定當下持有——現在已經不符合條件了

⚠️ **不重新回測、不需要重跑**：跟 `resolve_strategy_holdings.py` 同一套機制
（`fcv_core.MarketData.get_mask()` 對任意日期都成立），只是對同一批策略呼叫
`resolve_holdings()` 兩次、換一個 `as_of`。而且第二次遠比第一次快——
`get_mask()` 用 `(field, name)` 當 key 跨 spec 快取整個布林矩陣（見
`fcv_core.py:207` docstring），`resolve_holdings()` 只是在裡面切一天，
換日期不用重算遮罩本身。**實測**（30 檔策略，MarketData 已載入）：
第一個日期 41.1s（冷快取，真的要算遮罩），第二個日期 2.9s（快取命中，
快 14 倍）——所以整組比較的成本約等於單一日期解析＋一點點，不是兩倍。
"""
from __future__ import annotations

import dataclasses

import pandas as pd

from resolve_strategy_holdings import resolve_holdings_multi


@dataclasses.dataclass
class FaceComparison:
    as_of_current: str
    as_of_previous: str
    # 🔴 2026-09-11（§9.7 I-1，S1）：改成 {市場: 實際對到的交易日}——台美交易日曆
    # 不同，XM 投組可能有兩個不同的「今天」（例如美股國定假日但台股正常交易），
    # 不可化簡成單一日期。TW-only／US-only 的情況下這個 dict 就只有一個 key。
    current_dates: dict[str, pd.Timestamp]
    previous_dates: dict[str, pd.Timestamp]
    old_faces: list[str]     # 兩個時點都持有
    new_faces: list[str]     # 只在現在持有
    exited: list[str]        # 只在之前持有
    n_current: int
    n_previous: int
    per_stock_strategies: dict[str, dict]   # stock -> {"current": [uid,...], "previous": [uid,...]}

    @property
    def persistence_rate(self) -> float:
        """老面孔佔目前持股的比例——越高代表越穩定，對應老師說的「以前都還不錯」。"""
        return len(self.old_faces) / self.n_current if self.n_current else float("nan")

    @property
    def turnover_rate(self) -> float:
        """新面孔佔目前持股的比例——對應老師說的「現在這些股票最火的」。"""
        return len(self.new_faces) / self.n_current if self.n_current else float("nan")


def compare_faces(md_map: dict, idx: pd.DataFrame, members: list[str],
                  as_of_current: str, as_of_previous: str) -> FaceComparison:
    """對同一組策略清單，比較兩個時間點的持股，分出新面孔／老面孔／淡出。

    `md_map`：`{"TW": MarketData(...), "US": MarketData(...)}`（呼叫端先載入好，
    這裡不重載——跟 `ui._resolve_stock_holdings` 共用同一組 `st.cache_resource`
    實例）。🔴 2026-09-11（§9.7 I-1，S1）：改成 dict 是因為 XM 投組混合台美策略，
    每個 uid 要依 `candidate_index` 的 `market` 欄位分派到對應的 `MarketData`，
    不能只用一個。單一市場（TW／US）呼叫時 `md_map` 只放一個 key 即可。
    `idx`：`candidate_index.parquet` 讀成 index=strategy_uid 的 DataFrame。

    ⚠️ `as_of_previous` 必須早於 `as_of_current`，否則「新／舊」的語意會反過來
    ——這裡不強制檢查（呼叫端可能故意要看兩個未來日期怎麼變化），但預設用法
    是 `as_of_previous=is_end`（策略選定當下）、`as_of_current=今天`。
    """
    cur_by_stock: dict[str, list[str]] = {}
    prev_by_stock: dict[str, list[str]] = {}
    cur_dates: dict[str, pd.Timestamp] = {}
    prev_dates: dict[str, pd.Timestamp] = {}
    for uid in members:
        m = idx.loc[uid, "market"]
        stocks_c, cur_dates[m] = resolve_holdings_multi(md_map, idx, uid, as_of_current)
        stocks_p, prev_dates[m] = resolve_holdings_multi(md_map, idx, uid, as_of_previous)
        for s in stocks_c:
            cur_by_stock.setdefault(s, []).append(uid)
        for s in stocks_p:
            prev_by_stock.setdefault(s, []).append(uid)

    cur_set, prev_set = set(cur_by_stock), set(prev_by_stock)
    all_stocks = cur_set | prev_set
    per_stock = {s: {"current": sorted(cur_by_stock.get(s, [])),
                     "previous": sorted(prev_by_stock.get(s, []))}
                for s in all_stocks}

    return FaceComparison(
        as_of_current=as_of_current, as_of_previous=as_of_previous,
        current_dates=cur_dates, previous_dates=prev_dates,
        old_faces=sorted(cur_set & prev_set),
        new_faces=sorted(cur_set - prev_set),
        exited=sorted(prev_set - cur_set),
        n_current=len(cur_set), n_previous=len(prev_set),
        per_stock_strategies=per_stock,
    )
