# -*- coding: utf-8 -*-
"""A10（實戰開發追蹤.md §2）：估計 8 季裡有幾季會觸發 M1-D，判斷實驗有無鑑別力。

範圍：TW／scheme E／window 4／A_hrp／legacy／equal（30 檔策略，登記時點 2023-12-31）。
逐季（2024Q1~2025Q4）真的解析出這 30 檔策略當季的實際持股，算出這個投組自己的
五分位市值敞口，跟登記時點（2023-12-31）比較，看漂移幅度。

這是純程式、無 agent、不燒 token 的前置計算（設計文件 §11.6 A10）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402  sys.path bootstrap
import pandas as pd  # noqa: E402
from database import Database  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import (  # noqa: E402
    CANDIDATE_INDEX_PATH, resolve_holdings,
)

MEMBERS_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"

QUARTER_ENDS = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
                "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
REGISTER_DATE = "2023-12-31"


def get_members() -> list[str]:
    m = pd.read_parquet(MEMBERS_PATH)
    key = dict(tree_key="TW", scheme="E", window_no=4, k_mode="silhouette_is",
              ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    assert len(sub) == 1, f"預期唯一一列，實際 {len(sub)} 列"
    return list(sub.iloc[0]["members"])


def resolve_portfolio_stocks(md: MarketData, idx: pd.DataFrame, uids: list[str],
                             as_of: str) -> set[str]:
    """30 檔策略在 as_of 當天的持股聯集（不重複股票代號），跳過算不出遮罩的策略。"""
    out: set[str] = set()
    for uid in uids:
        row = idx.loc[uid]
        try:
            syms, _use_date = resolve_holdings(md, row, as_of)
        except RuntimeError:
            continue
        out.update(syms)
    return out


def exposure_gap(stocks: set[str], mcap_row: pd.Series) -> dict:
    """給一組股票代號 + 當天全市場市值，算等權投組 vs 市值加權指數的
    Q1（最大市值 20%）敞口——跟 §3.2 同一套量測，但這裡是「真實持股」不是「全市場」。
    """
    m_all = mcap_row.dropna()
    m_port = m_all[m_all.index.isin(stocks)]
    if len(m_port) < 5:
        return {"n_stocks": len(stocks), "n_priced": len(m_port), "q1_gap": None}
    # 用全市場的市值排序切五分位邊界，再看投組落在哪個分位
    ranks = m_all.rank(pct=True)
    q1_universe = m_all[ranks >= 0.8]              # 全市場最大 20%
    idx_w_q1 = q1_universe.sum() / m_all.sum()      # 指數裡 Q1 的權重
    port_w_q1 = m_port[m_port.index.isin(q1_universe.index)].shape[0] / len(m_port)  # 投組裡 Q1 的等權佔比
    return {"n_stocks": len(stocks), "n_priced": len(m_port),
           "idx_w_q1": idx_w_q1, "port_w_q1": port_w_q1,
           "q1_gap": port_w_q1 - idx_w_q1}


def main():
    print("讀取策略成員與候選池索引…")
    uids = get_members()
    print(f"  scheme E window 4 A_hrp legacy/equal: {len(uids)} 檔策略")
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    print("載入 TW MarketData（第一次約 1-2 分鐘）…")
    md = MarketData("TW")

    db = Database("TW")
    conn = db.create_connection()
    mcap_raw = pd.read_sql(
        f"SELECT s.date, c.company_symbol, s.market_capital FROM stock s "
        f"JOIN company c ON s.company_id=c.id WHERE {db._exchange_in_clause()} "
        f"AND s.market_capital IS NOT NULL AND s.date >= '2023-01-01' ORDER BY s.date",
        conn)
    mcap_raw["date"] = pd.to_datetime(mcap_raw["date"])
    mcap = mcap_raw.pivot(index="date", columns="company_symbol", values="market_capital")

    def asof_mcap(d: str) -> pd.Series:
        ts = pd.Timestamp(d)
        avail = mcap.index[mcap.index <= ts]
        return mcap.loc[avail.max()]

    print(f"\n登記時點 {REGISTER_DATE} 的持股與敞口：")
    reg_stocks = resolve_portfolio_stocks(md, idx, uids, REGISTER_DATE)
    reg_exp = exposure_gap(reg_stocks, asof_mcap(REGISTER_DATE))
    print(f"  {reg_exp}")

    rows = [{"date": REGISTER_DATE, **reg_exp}]
    for d in QUARTER_ENDS:
        stocks = resolve_portfolio_stocks(md, idx, uids, d)
        exp = exposure_gap(stocks, asof_mcap(d))
        rows.append({"date": d, **exp})
        print(f"  {d}: n_stocks={exp['n_stocks']:>4}  q1_gap={exp['q1_gap']}")

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "a10_trigger_estimate.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out}")

    print("\n=== A10 判讀 ===")
    base = reg_exp["q1_gap"]
    for r in rows[1:]:
        if r["q1_gap"] is None or base is None:
            continue
        dev = r["q1_gap"] - base
        print(f"  {r['date']}: 相對登記時偏離 {dev:+.4f}")


if __name__ == "__main__":
    main()
