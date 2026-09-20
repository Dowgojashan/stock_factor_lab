# -*- coding: utf-8 -*-
"""§8待辦item10（延伸，2026-09-22）：把window4已經驗證過的股票層B_all重算方法
推廣到window1-3（2015-2023，36季），讓M0/M4/M6/M8的歷史門檻分布
（`m0_performance_history.csv`等，windows1-3, n=36）也建立在同一套量測系統上
——否則window4用新B_all、windows1-3門檻用舊placeholder，會產生新的跨系統
不一致（D24-D26已經示範過這類問題的危險）。

確認過（`_tree_universe`）：anchored scheme下 window1~4 的候選池universe完全
相同（都是6,679檔，因為IS起點固定在2007-01，只有IS_END往後延伸），不需要
為每個window分別重建universe，直接沿用window4已驗證的6,679檔清單
（`tsmc_cluster_concentration.csv`）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH, resolve_holdings  # noqa: E402

from app.performance import measure  # noqa: E402

CLUSTER_CSV = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "tsmc_cluster_concentration.csv"
WINDOW_QUARTERS = {
    1: ["2014-12-31", "2015-03-31", "2015-06-30", "2015-09-30", "2015-12-31",
       "2016-03-31", "2016-06-30", "2016-09-30", "2016-12-31",
       "2017-03-31", "2017-06-30", "2017-09-30", "2017-12-31"],
    2: ["2017-12-31", "2018-03-31", "2018-06-30", "2018-09-30", "2018-12-31",
       "2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31",
       "2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"],
    3: ["2020-12-31", "2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31",
       "2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31",
       "2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"],
}


def resolve_w0(md: MarketData, idx: pd.DataFrame, uids: list[str], as_of: str) -> tuple[dict[str, float], int]:
    n_strat = len(uids)
    weights: dict[str, float] = {}
    n_empty = 0
    for uid in uids:
        row = idx.loc[uid]
        try:
            syms, _ = resolve_holdings(md, row, as_of)
        except RuntimeError:
            continue
        if not syms:
            n_empty += 1
            continue
        w = (1.0 / n_strat) / len(syms)
        for s in syms:
            weights[s] = weights.get(s, 0.0) + w
    return weights, n_empty


def main():
    tree_uids = pd.read_csv(CLUSTER_CSV)["strategy_uid"].tolist()
    print(f"B_all 候選池（跟window4共用同一份universe，anchored scheme確認過相同）：{len(tree_uids)} 檔")

    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    idx = idx.loc[idx.index.intersection(tree_uids)]

    print(">> 載入 TW MarketData ...")
    md = MarketData("TW")
    md_map = {"TW": md}

    rows = []
    for window_no, quarters in WINDOW_QUARTERS.items():
        print(f"\n=== window{window_no}（OOS {quarters[1]}~{quarters[-1]}）===")
        for i in range(len(quarters) - 1):
            as_of, end = quarters[i], quarters[i + 1]
            weights, n_empty = resolve_w0(md, idx, tree_uids, as_of)
            n_unique = len(weights)
            res = measure(md_map, weights, as_of, end)
            ret = res["portfolio_realized_return"]
            print(f"  {as_of}->{end}：{n_unique}檔　股票層報酬={ret:+.4%}")
            rows.append({"window_no": window_no, "as_of": as_of, "end": end,
                        "n_unique_stocks": n_unique, "n_empty_strategies": n_empty,
                        "stock_level_return": ret})

    df = pd.DataFrame(rows)
    for w in WINDOW_QUARTERS:
        sub = df[df.window_no == w]
        cum = float((1 + sub["stock_level_return"]).prod() - 1)
        n_years = len(sub) / 4
        cagr = (1 + cum) ** (1 / n_years) - 1
        print(f"\nwindow{w} 累積={cum:+.2%}  年化CAGR={cagr:+.2%}")

    out = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "ball_stock_level_windows123.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out}")


if __name__ == "__main__":
    main()
