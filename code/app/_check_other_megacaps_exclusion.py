# -*- coding: utf-8 -*-
"""D65延伸item3（2026-09-22，使用者要求）：換其他權值股測，看排除模式是不是
台積電特例，還是大型股的普遍現象。

選兩檔跟台積電一樣是台股前段班權值股、但市值/評價量級不同的公司：
  2317 鴻海（權值股，但估值倍數通常比台積電低很多——傳產代工，非AI題材溢價）
  2454 聯發科（同樣是半導體，AI題材有沾邊，但市值遠小於台積電）
跟同一組 7,128 檔候選池、同一套 F1/F2/C/V 機制、同一組8個真實as_of日期比較。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from database import Database  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import (CANDIDATE_INDEX_PATH, _c_condition,  # noqa: E402
                                       _q_band_condition)

STOCKS = {"2330": "台積電", "2317": "鴻海", "2454": "聯發科"}
CHECK_DATES = ["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
              "2025-03-31", "2025-06-30", "2025-09-30"]


def asof_bool(mask: pd.DataFrame, as_of: str, symbol: str) -> bool | None:
    if mask is None or symbol not in mask.columns:
        return None
    ts = pd.Timestamp(as_of)
    valid = mask.index[mask.index <= ts]
    if len(valid) == 0:
        return None
    return bool(mask.loc[valid.max(), symbol])


def strategy_selects(md: MarketData, row: pd.Series, as_of: str, symbol: str) -> bool | None:
    f1_mask = md.get_mask(_q_band_condition(row["F1_factor"], row["F1_band"], row["F1_nbands"]))
    f1 = asof_bool(f1_mask, as_of, symbol)
    if f1 is not True:
        return f1
    if not row["F2_empty"]:
        f2_mask = md.get_mask(_q_band_condition(row["F2_factor"], row["F2_band"], row["F2_nbands"]))
        f2 = asof_bool(f2_mask, as_of, symbol)
        if f2 is not True:
            return f2
    if pd.notna(row["C_rule"]):
        c_mask = md.get_mask(_c_condition(row["C_source"], row["C_rule"]))
        c = asof_bool(c_mask, as_of, symbol)
        if c is not True:
            return c
    if row["V"] == "v1":
        v_mask = md.get_v_mask()
        v = asof_bool(v_mask, as_of, symbol)
        if v is not True:
            return v
    return True


def fetch_mcap_weight_series(conn, symbol: str) -> pd.Series:
    q = f"""SELECT s.date, c.company_symbol, s.market_capital
            FROM stock s JOIN company c ON s.company_id=c.id
            WHERE c.exchange_name IN ('TWSE') AND s.market_capital IS NOT NULL
              AND s.date >= '2023-06-01'"""
    d = pd.read_sql(q, conn)
    d["date"] = pd.to_datetime(d["date"])
    wide = d.pivot_table(index="date", columns="company_symbol", values="market_capital", aggfunc="last")
    w = wide[symbol] / wide.sum(axis=1)
    return w.sort_index()


def asof_value(series: pd.Series, d: str) -> float:
    ts = pd.Timestamp(d)
    avail = series.index[series.index <= ts]
    return float(series.loc[avail.max()]) if len(avail) else float("nan")


def main():
    idx_full = pd.read_parquet(CANDIDATE_INDEX_PATH)
    idx = idx_full[idx_full["market"] == "TW"].set_index("strategy_uid")
    print(f"全部台股候選策略數：{len(idx)}")

    print(">> 載入 TW MarketData ...")
    md = MarketData("TW")

    db = Database("TW")
    conn = db.create_connection()

    rows = []
    for symbol, name in STOCKS.items():
        w_series = fetch_mcap_weight_series(conn, symbol)
        ever_selected: set[str] = set()
        print(f"\n=== {symbol}（{name}）===")
        for d in CHECK_DATES:
            n_sel = 0
            for uid, row in idx.iterrows():
                if strategy_selects(md, row, d, symbol) is True:
                    n_sel += 1
                    ever_selected.add(uid)
            w = asof_value(w_series, d)
            print(f"  {d}：{n_sel}/{len(idx)}（{n_sel/len(idx):.2%}）選中　"
                 f"｜指數權重={w:.2%}" if pd.notna(w) else f"  {d}：{n_sel}/{len(idx)}（{n_sel/len(idx):.2%}）選中")
            rows.append({"symbol": symbol, "name": name, "as_of": d,
                        "n_pool": len(idx), "n_selected": n_sel,
                        "pct_selected": n_sel / len(idx), "index_weight": w})
        print(f"  8季累計「曾經選過」的不重複策略數：{len(ever_selected)}/{len(idx)}"
             f"（{len(ever_selected)/len(idx):.2%}）")

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "other_megacaps_exclusion.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out}")


if __name__ == "__main__":
    main()
