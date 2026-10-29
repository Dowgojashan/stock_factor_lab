# -*- coding: utf-8 -*-
"""掃描美股 `stock.close` 疑似未還原公司行動（分割／反向分割）或錯價的單日跳動（2026-10-28）。

背景：優等生權重網格（實戰收尾追蹤.md §四步驟1）發現美股小型股有單日 10 倍以上跳價
（IVT 2021-09-24 1.67→16.81、AREC 2018-03-01 0.05→1.10），而美股 `non_adj_close`
全部為空（31,563,163 筆皆 NULL；台股有值），本專案無法自行分辨真行情或還原漏掉，
需交給 collector 依權威公司行動資料查核。本腳本產出交付清單。

範圍：Russell 3000 宇宙（`fcv_core.MarketData('US')` 的 price:close，已套
`_universe_clause`），全期間。條件：單日收盤價比值 ≥ 2（漲≥100%）或 ≤ 0.5（跌≥50%），
或收盤價 > SANITY_MAX_CLOSE。另標記本專案網格是否曾持有（影響範圍）。
⚠️ 價格資料在 get_data.py:124 已 ffill，停牌日沿用前值；比值用 ffill 後的連續價計算。

輸出：`_analysis_outputs_applayer/us_price_jump_suspects_2026-10-28.csv`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import pandas as pd  # noqa: E402

from app import monitor  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
GRID = ROOT / "_analysis_outputs_applayer" / "elite_grid_1027"
OUT = ROOT / "_analysis_outputs_applayer" / "us_price_jump_suspects_2026-10-28.csv"
UP, DOWN = 2.0, 0.5


def main():
    from fcv_core import MarketData
    md = MarketData("US")
    close = md.data.get("price:close").astype(float)
    ratio = close.div(close.shift(1)).replace([np.inf, -np.inf], np.nan)   # 前一日收盤為0→inf，不可比
    ratio = ratio.where(ratio > 0)                                            # 當日收盤為0→比值0，不可比
    hit = (ratio >= UP) | (ratio <= DOWN)
    rows = []
    idx = close.index
    for sym in close.columns[hit.any(axis=0).to_numpy()]:
        for d in idx[hit[sym].to_numpy()]:
            i = idx.get_loc(d)
            prev5 = close[sym].iloc[max(i - 5, 0):i]
            next5 = close[sym].iloc[i:i + 5]
            rows.append({"symbol": sym, "date": d.date(), "prev_date": idx[i - 1].date(),
                         "prev_close": close[sym].iloc[i - 1], "close": close[sym].iloc[i],
                         "ratio": ratio[sym].iloc[i],
                         "prev5_flat": bool(prev5.nunique() == 1), "next5_flat": bool(next5.nunique() == 1),
                         "ratio_near_integer": bool(min(abs(ratio[sym].iloc[i] - round(ratio[sym].iloc[i])),
                                                        abs(1 / ratio[sym].iloc[i] - round(1 / ratio[sym].iloc[i]))) < 0.05)})
    bad_px = close.gt(monitor.SANITY_MAX_CLOSE)
    for sym in close.columns[bad_px.any(axis=0).to_numpy()]:
        d0 = idx[bad_px[sym].to_numpy()]
        rows.append({"symbol": sym, "date": d0.min().date(), "prev_date": None, "prev_close": np.nan,
                     "close": float(close[sym].max()), "ratio": np.nan, "prev5_flat": False, "next5_flat": False,
                     "ratio_near_integer": False,
                     "note": f"收盤價>{monitor.SANITY_MAX_CLOSE:,.0f}，{d0.min().date()}~{d0.max().date()} 共{len(d0)}天"})
    df = pd.DataFrame(rows)

    held = set()
    for pool in ("baseline", "exclude_v1"):
        p = GRID / f"w0_US_{pool}.parquet"
        if p.exists():
            held |= set(pd.read_parquet(p).symbol)
    df["held_in_grid"] = df.symbol.isin(held)
    df["direction"] = np.where(df.ratio >= UP, "漲≥100%", np.where(df.ratio <= DOWN, "跌≥50%", "異常高價"))
    df = df.sort_values(["held_in_grid", "symbol", "date"], ascending=[False, True, True])
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"事件數 {len(df)}｜股票數 {df.symbol.nunique()}｜網格曾持有的股票 {df[df.held_in_grid].symbol.nunique()} 檔、"
          f"事件 {int(df.held_in_grid.sum())} 筆")
    print(df.direction.value_counts().to_string())
    print("前後5日價格不動且比值接近整數（高度疑似分割未還原）：",
          int((df.ratio_near_integer & (df.prev5_flat | df.next5_flat)).sum()))
    print(f"輸出 {OUT}")


if __name__ == "__main__":
    main()
