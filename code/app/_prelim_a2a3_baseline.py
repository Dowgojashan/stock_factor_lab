# -*- coding: utf-8 -*-
"""A2／A3（實戰開發追蹤.md §2）：對照 0（完全不調整）的真實已實現績效基準線，
與同期等權／市值加權大盤對照。

做法：TW／scheme E／window 4／A_hrp／legacy／equal（30 檔策略）——策略清單全程
不變（對照 0 的定義），股票持股依快時鐘每季重解（財報驅動的正常換股，非調整）。
逐季用 `resolve_strategy_holdings` 算真實權重（比照 `risk.assess_stock_level()`
的公式 w=(1/n_strat)/len(syms)），再用 `performance.measure()`（S5，已在正式模式
驗證過的模組）串接算真實已實現報酬，跟同期真實 TAIEX 報酬指數比較。
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

from app.performance import measure  # noqa: E402

MEMBERS_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"
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
    """比照 risk.assess_stock_level() 的公式：w = (1/n_strat) / len(syms)。
    解不出遮罩或當天無持股的策略，視同持有現金（分母仍用 n_strat，不放大其餘策略）。
    """
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


def main():
    uids = get_members()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    print("載入 TW MarketData…")
    md = MarketData("TW")
    md_map = {"TW": md}

    db = Database("TW")
    conn = db.create_connection()
    tx = pd.read_sql("SELECT date, close FROM taiex_tr ORDER BY date", conn)
    tx["date"] = pd.to_datetime(tx["date"]); tx = tx.set_index("date")["close"].astype(float)

    def taiex_period_return(as_of: str, end: str) -> float:
        def asof(d):
            ts = pd.Timestamp(d); s = tx.index[tx.index <= ts]; return s.max()
        d0, d1 = asof(as_of), asof(end)
        return float(tx.loc[d1] / tx.loc[d0] - 1.0)

    rows = []
    print("\n=== 對照 0：逐季已實現報酬（真實資料，策略清單全程不變）===")
    for i in range(len(CHECKPOINTS) - 1):
        as_of, end = CHECKPOINTS[i], CHECKPOINTS[i + 1]
        weights = resolve_weights(md, idx, uids, as_of)
        n_stocks = len(weights)
        res = measure(md_map, weights, as_of, end)
        port_ret = res["portfolio_realized_return"]
        ew_bench = res["by_market_benchmark"]["TW"]["equal_weight_benchmark_return"]
        cw_bench = taiex_period_return(as_of, end)
        row = {"period": f"{as_of}->{end}", "n_stocks": n_stocks,
              "portfolio_return": port_ret, "equal_weight_bench": ew_bench,
              "cap_weight_bench_taiex": cw_bench,
              "vs_equal": (port_ret - ew_bench) if (port_ret is not None and ew_bench is not None) else None,
              "vs_cap": (port_ret - cw_bench) if port_ret is not None else None}
        rows.append(row)
        print(f"  {row['period']}: n={n_stocks:>4}  投組={port_ret:+.2%}  "
             f"等權={ew_bench:+.2%}  市值加權(TAIEX)={cw_bench:+.2%}  "
             f"對等權={row['vs_equal']:+.2%}  對市值加權={row['vs_cap']:+.2%}")

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "a2_baseline_control0.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out}")

    # 累計（複利串接）
    cum_port = (1 + df["portfolio_return"]).prod() - 1
    cum_ew = (1 + df["equal_weight_bench"]).prod() - 1
    cum_cw = (1 + df["cap_weight_bench_taiex"]).prod() - 1
    print(f"\n=== 2023-12-31 -> 2025-12-31 累計（複利串接 8 季）===")
    print(f"  投組: {cum_port:+.2%}")
    print(f"  等權大盤: {cum_ew:+.2%}   對等權超額: {cum_port-cum_ew:+.2%}")
    print(f"  市值加權大盤(TAIEX): {cum_cw:+.2%}   對市值加權超額: {cum_port-cum_cw:+.2%}")


if __name__ == "__main__":
    main()
