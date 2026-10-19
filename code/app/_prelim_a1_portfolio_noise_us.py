# -*- coding: utf-8 -*-
"""2026-10-18：美股版的M3（選股廣度異常）歷史分布——跟台股
`_prelim_a1_portfolio_noise.py`同一個方法，這是美股第一次做這件事。

背景：`diagnose.py`查證發現M3/M0/M6/M7/M8的歷史校準檔案原本是單一路徑
（不分市場），但生成腳本全部寫死`MarketData("TW")`——美股的L2真實重跑
一直在用台股的「選股廣度正常範圍」當基準，不是美股自己的。這裡補上
美股版本，跟`_prelim_a1_portfolio_noise.py`差異只有兩處：
①`tree_key`/`MarketData`換成"US" ②市值資料改用`monitor.fetch_mcap_wide()`
（含`SANITY_MAX_CLOSE`離群值防線＋`COALESCE(market_capital,close*shares)`
回退路徑），不像台股版直接查`market_capital`——美股2022年以前`market_capital`
覆蓋率不足，必須用回退路徑才能拿到完整市值。

`walkforward_members.parquet`已經確認有美股windows1-3（scheme=E）的代表
策略清單（is_end分別是2014-12/2017-12/2020-12，跟台股用的完全同一組
日期），不需要重新建樹——這是純粹的既有資料重組，運算很輕。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from database import Database  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH, resolve_holdings  # noqa: E402
from app import monitor  # noqa: E402

MEMBERS_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"

WINDOWS = {
    1: {"is_end": "2014-12-31", "oos_start": "2015-01-01", "oos_end": "2017-12-31"},
    2: {"is_end": "2017-12-31", "oos_start": "2018-01-01", "oos_end": "2020-12-31"},
    3: {"is_end": "2020-12-31", "oos_start": "2021-01-01", "oos_end": "2023-12-31"},
}


def quarter_ends(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="Q")]


def get_members(window_no: int) -> list[str]:
    m = pd.read_parquet(MEMBERS_PATH)
    key = dict(tree_key="US", scheme="E", window_no=window_no, k_mode="silhouette_is",
              ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    assert len(sub) == 1, f"window {window_no}: 預期唯一一列，實際 {len(sub)} 列"
    return list(sub.iloc[0]["members"])


def resolve_portfolio_stocks(md: MarketData, idx: pd.DataFrame, uids: list[str],
                             as_of: str) -> set[str]:
    out: set[str] = set()
    for uid in uids:
        row = idx.loc[uid]
        try:
            syms, _ = resolve_holdings(md, row, as_of)
        except RuntimeError:
            continue
        out.update(syms)
    return out


def exposure_gap(stocks: set[str], mcap_row: pd.Series) -> dict:
    m_all = mcap_row.dropna()
    m_port = m_all[m_all.index.isin(stocks)]
    if len(m_port) < 5:
        return {"n_stocks": len(stocks), "n_priced": len(m_port), "q1_gap": None}
    ranks = m_all.rank(pct=True)
    q1_universe = m_all[ranks >= 0.8]
    idx_w_q1 = q1_universe.sum() / m_all.sum()
    port_w_q1 = m_port[m_port.index.isin(q1_universe.index)].shape[0] / len(m_port)
    return {"n_stocks": len(stocks), "n_priced": len(m_port),
           "idx_w_q1": idx_w_q1, "port_w_q1": port_w_q1, "q1_gap": port_w_q1 - idx_w_q1}


def main():
    idx_full = pd.read_parquet(CANDIDATE_INDEX_PATH)
    idx = idx_full[idx_full["market"] == "US"].set_index("strategy_uid")

    print("載入 US MarketData…")
    md = MarketData("US")

    db = Database("US")
    conn = db.create_connection()
    print("讀取美股市值資料（2014-01起，含SANITY_MAX_CLOSE離群值防線）…")
    wide = monitor.fetch_mcap_wide(conn, market="US", start="2014-01-01")

    def asof_mcap(d: str) -> pd.Series:
        return monitor.asof_row(wide, d)

    all_rows = []
    for wno, meta in WINDOWS.items():
        print(f"\n=== window {wno}（IS 結束 {meta['is_end']}，OOS {meta['oos_start']}~{meta['oos_end']}）===")
        uids = get_members(wno)
        print(f"  {len(uids)} 檔策略")

        reg_stocks = resolve_portfolio_stocks(md, idx, uids, meta["is_end"])
        reg_exp = exposure_gap(reg_stocks, asof_mcap(meta["is_end"]))
        base = reg_exp["q1_gap"]
        print(f"  登記時 {meta['is_end']}: n={reg_exp['n_stocks']} q1_gap={base}")
        all_rows.append({"window_no": wno, "date": meta["is_end"], "is_baseline": True,
                         **reg_exp, "dev_from_baseline": 0.0})

        for d in quarter_ends(meta["oos_start"], meta["oos_end"]):
            stocks = resolve_portfolio_stocks(md, idx, uids, d)
            exp = exposure_gap(stocks, asof_mcap(d))
            dev = (exp["q1_gap"] - base) if (exp["q1_gap"] is not None and base is not None) else None
            print(f"  {d}: n={exp['n_stocks']:>4} q1_gap={exp['q1_gap']} dev={dev}")
            all_rows.append({"window_no": wno, "date": d, "is_baseline": False,
                             **exp, "dev_from_baseline": dev})

    df = pd.DataFrame(all_rows)
    out = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "a1_portfolio_noise_windows123_US.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out}")

    noise = df[~df["is_baseline"]]["dev_from_baseline"].dropna()
    print("\n=== 美股投組端季度雜訊分布（windows 1-3 合併）===")
    print(f"  樣本數 n={len(noise)}")
    print(f"  平均絕對偏離: {noise.abs().mean()*100:.2f}pp")
    print(f"  標準差: {noise.std()*100:.2f}pp")
    print(f"  p90(絕對值): {noise.abs().quantile(0.90)*100:.2f}pp")
    print(f"  最大絕對偏離: {noise.abs().max()*100:.2f}pp")

    n_stocks = df[~df["is_baseline"]]["n_stocks"]
    print(f"\n=== M3用的n_stocks分布（供diagnose.py的_M3_HISTORY_PATH[US]）===")
    print(f"  p10={n_stocks.quantile(0.10):.1f}  p90={n_stocks.quantile(0.90):.1f}")


if __name__ == "__main__":
    main()
