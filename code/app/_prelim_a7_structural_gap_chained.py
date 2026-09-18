# -*- coding: utf-8 -*-
"""A7：§3.1 結構性缺口公式改逐季串接（真實資料，取代單期近似）。

現行 §3.1／§1.1 用**期初權重**（2023-12-31 台積電 24.78%）套用在整個
2024-2025 兩年期間，但台積電權重在期間內從 24.78% 一路漲到 40.66%——
用單一期初權重乘兩年報酬差會低估真實缺口（權重上升期間，缺口應該逐季擴大）。

做法：對 scheme E window 4 的 8 個季度，逐季算：
  w_port,t = 投組裡台積電的真實權重（resolve_holdings，跟 A2/A3 同方法）
  w_idx,t  = 台積電佔 TAIEX 的真實市值權重（stock.market_capital，跟
             weighting_decomposition.py 同方法）
  r_stock,t = 台積電當季真實報酬（stock.close，還原價已含息）
  r_index,t = TAIEX 當季真實報酬（taiex_tr，含息報酬指數）
  r_rest,t  = 代數反推：(r_index,t - w_idx,t * r_stock,t) / (1 - w_idx,t)
  gap_t     = (w_port,t - w_idx,t) * (r_stock,t - r_rest,t)
串接（加總）8 季的 gap_t，跟現行的單期近似（-29.92pp）比較。
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

TSMC = "2330"
MEMBERS_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "a7_structural_gap_chained.csv"

CHECKPOINTS = ["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
              "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]


def get_members() -> list[str]:
    m = pd.read_parquet(MEMBERS_PATH)
    key = dict(tree_key="TW", scheme="E", window_no=4, k_mode="silhouette_is",
              ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    assert len(sub) == 1
    return list(sub.iloc[0]["members"])


def resolve_weights(md: MarketData, idx: pd.DataFrame, uids: list[str], as_of: str) -> dict[str, float]:
    n_strat = len(uids)
    weights: dict[str, float] = {}
    for uid in uids:
        row = idx.loc[uid]
        try:
            syms, _ = resolve_holdings(md, row, as_of)
        except RuntimeError:
            continue
        if not syms:
            continue
        w = (1.0 / n_strat) / len(syms)
        for s in syms:
            weights[s] = weights.get(s, 0.0) + w
    return weights


def fetch_mcap_weight_series(conn) -> pd.Series:
    """台積電佔全市場市值的權重，逐日，之後取每個 checkpoint 當天(或之前)最後一筆。"""
    db_tw_clause = "c.exchange_name IN ('TWSE')"
    q = f"""SELECT s.date, c.company_symbol, s.market_capital
            FROM stock s JOIN company c ON s.company_id=c.id
            WHERE {db_tw_clause} AND s.market_capital IS NOT NULL
              AND s.date >= '2023-06-01'"""
    d = pd.read_sql(q, conn)
    d["date"] = pd.to_datetime(d["date"])
    wide = d.pivot_table(index="date", columns="company_symbol", values="market_capital", aggfunc="last")
    w = wide[TSMC] / wide.sum(axis=1)
    return w.sort_index()


def asof_value(series: pd.Series, d: str) -> float:
    ts = pd.Timestamp(d)
    avail = series.index[series.index <= ts]
    return float(series.loc[avail.max()])


def fetch_taiex_tr(conn) -> pd.Series:
    d = pd.read_sql("SELECT date, close FROM taiex_tr ORDER BY date", conn)
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")["close"].astype(float).sort_index()


def main():
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    print("載入 TW MarketData…")
    md = MarketData("TW")
    close = md.data.get("price:close")

    db = Database("TW")
    conn = db.create_connection()
    print("讀取台積電市值權重序列…")
    w_idx_series = fetch_mcap_weight_series(conn)
    print("讀取 TAIEX 報酬指數…")
    taiex_tr = fetch_taiex_tr(conn)

    uids = get_members()

    rows = []
    for i in range(len(CHECKPOINTS) - 1):
        t0, t1 = CHECKPOINTS[i], CHECKPOINTS[i + 1]

        weights = resolve_weights(md, idx, uids, t0)
        w_port = weights.get(TSMC, 0.0)
        w_idx = asof_value(w_idx_series, t0)

        # 台積電真實報酬（還原價，含息）
        idx0 = close.index[close.index <= pd.Timestamp(t0)].max()
        idx1 = close.index[close.index <= pd.Timestamp(t1)].max()
        r_stock = float(close.at[idx1, TSMC] / close.at[idx0, TSMC] - 1.0)

        # TAIEX 報酬指數真實報酬
        ti0 = taiex_tr.index[taiex_tr.index <= pd.Timestamp(t0)].max()
        ti1 = taiex_tr.index[taiex_tr.index <= pd.Timestamp(t1)].max()
        r_index = float(taiex_tr.loc[ti1] / taiex_tr.loc[ti0] - 1.0)

        r_rest = (r_index - w_idx * r_stock) / (1 - w_idx)
        gap = (w_port - w_idx) * (r_stock - r_rest)

        print(f"{t0}->{t1}: w_port={w_port:.2%} w_idx={w_idx:.2%} "
             f"r_stock={r_stock:+.2%} r_index={r_index:+.2%} r_rest={r_rest:+.2%} "
             f"gap={gap*100:+.2f}pp")
        rows.append({"q_start": t0, "q_end": t1, "w_port": w_port, "w_idx": w_idx,
                    "r_stock": r_stock, "r_index": r_index, "r_rest": r_rest,
                    "gap_pp": gap * 100})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {OUT_PATH}")

    chained_total = df["gap_pp"].sum()
    print(f"\n=== 逐季串接（加總）vs 單期近似 ===")
    print(f"逐季串接總缺口: {chained_total:+.2f}pp（8 季加總）")
    print(f"單期近似（現行 §3.1／§1.1）: -29.92pp")
    print(f"差異: {chained_total - (-29.92):+.2f}pp")


if __name__ == "__main__":
    main()
