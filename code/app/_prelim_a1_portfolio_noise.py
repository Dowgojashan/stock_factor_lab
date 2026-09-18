# -*- coding: utf-8 -*-
"""A1（實戰開發追蹤.md §2）延伸：投組端季度雜訊的歷史分布。

上一步（_prelim_a10_trigger_estimate.py）發現：window 4（實驗期）投組端的
季度敞口雜訊（最大單季 4.96pp）跟市場端兩年累計漂移（4.34pp）量級相當——
若不知道「正常換股本來就會有多少雜訊」，M1-D 的門檻沒辦法訂。

做法：對 scheme E window 1/2/3（2015-2017／2018-2020／2021-2023，全部在
實驗期之外，符合設計文件 §11.7 的資料範圍限制）各自解析當時 A_hrp
legacy/equal 投組的逐季持股，算出各自的季度敞口序列與相對各自登記時點
（is_end）的偏離，把三個窗次的雜訊樣本池在一起，當作 M1-D 門檻的參考分布。
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

MEMBERS_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"
DETAIL_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_matrix_detail.csv"

WINDOWS = {
    1: {"is_end": "2014-12-31", "oos_start": "2015-01-01", "oos_end": "2017-12-31"},
    2: {"is_end": "2017-12-31", "oos_start": "2018-01-01", "oos_end": "2020-12-31"},
    3: {"is_end": "2020-12-31", "oos_start": "2021-01-01", "oos_end": "2023-12-31"},
}


def quarter_ends(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="Q")]


def get_members(window_no: int) -> list[str]:
    m = pd.read_parquet(MEMBERS_PATH)
    key = dict(tree_key="TW", scheme="E", window_no=window_no, k_mode="silhouette_is",
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
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    print("載入 TW MarketData…")
    md = MarketData("TW")

    db = Database("TW")
    conn = db.create_connection()
    print("讀取市值資料（2014-01 起）…")
    mcap_raw = pd.read_sql(
        f"SELECT s.date, c.company_symbol, s.market_capital FROM stock s "
        f"JOIN company c ON s.company_id=c.id WHERE {db._exchange_in_clause()} "
        f"AND s.market_capital IS NOT NULL AND s.date >= '2014-01-01' ORDER BY s.date",
        conn)
    mcap_raw["date"] = pd.to_datetime(mcap_raw["date"])
    mcap = mcap_raw.pivot(index="date", columns="company_symbol", values="market_capital")

    def asof_mcap(d: str) -> pd.Series:
        ts = pd.Timestamp(d)
        avail = mcap.index[mcap.index <= ts]
        return mcap.loc[avail.max()]

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
    out = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "a1_portfolio_noise_windows123.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out}")

    noise = df[~df["is_baseline"]]["dev_from_baseline"].dropna()
    print("\n=== 投組端季度雜訊分布（windows 1-3 合併，全部在實驗期之外）===")
    print(f"  樣本數 n={len(noise)}")
    print(f"  平均絕對偏離: {noise.abs().mean()*100:.2f}pp")
    print(f"  標準差: {noise.std()*100:.2f}pp")
    print(f"  p90(絕對值): {noise.abs().quantile(0.90)*100:.2f}pp")
    print(f"  最大絕對偏離: {noise.abs().max()*100:.2f}pp")
    print(f"  分布: min={noise.min()*100:.2f}pp max={noise.max()*100:.2f}pp")


if __name__ == "__main__":
    main()
