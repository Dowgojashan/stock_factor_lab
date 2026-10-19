# -*- coding: utf-8 -*-
"""2026-10-18：美股版的M0/M6/M7歷史校準資料——補`diagnose.py`查證發現的
缺口（M0/M3/M6/M7/M8的校準檔案原本不分市場，全部寫死用台股資料算）。

跟台股`m0_performance_history_corrected.csv`同一套欄位定義（`window_no`／
`as_of`／`end`／`portfolio_return`／`equal_weight_bench`／`excess_vs_ball`／
`ball_benchmark_return`／`ball_excess_vs_equal_weight`），但一次算完、不留
`excess_vs_ball_old_placeholder`這種只在原始台股校準史上才有意義的欄位。

依賴`_check_us_cluster_universe.py`先建好的美股完整候選池universe
（`us_cluster_universe.csv`）——那支腳本要先跑過，這支才能用。

兩個報酬來源：
①代表策略投組（`walkforward_members.parquet`的A_hrp legacy equal名單）
  真實報酬，透過`simulate.resolve_weights()`＋`monitor.outcome_layer()`
  算，同時拿到`portfolio_return`跟`equal_weight_bench`（後者是
  `outcome_layer()`內部`measure()`算出的全市場等權基準，跟代表投組的
  選股無關，是真正的市場基準）
②B_all（候選池整體，股票層equal weight across all strategies）——沿用
  `_recompute_ball_windows123.py`的`_fast_resolve_w0()`（已驗證過的快速
  持股解析寫法，不複製貼上原始邏輯）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402

from app import monitor, simulate  # noqa: E402
from app._recompute_ball_windows123 import _fast_resolve_w0  # noqa: E402

UNIVERSE_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "us_cluster_universe.csv"
MEMBERS_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "m0_performance_history_corrected_US.csv"

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


def get_members(window_no: int) -> list[str]:
    m = pd.read_parquet(MEMBERS_PATH)
    key = dict(tree_key="US", scheme="E", window_no=window_no, k_mode="silhouette_is",
              ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    assert len(sub) == 1, f"window {window_no}: 預期唯一一列，實際 {len(sub)} 列"
    return list(sub.iloc[0]["members"])


def main():
    if not UNIVERSE_PATH.exists():
        raise FileNotFoundError(f"找不到 {UNIVERSE_PATH}——請先跑"
                                f"`python -m app._check_us_cluster_universe`")
    pool_uids = pd.read_csv(UNIVERSE_PATH)["strategy_uid"].tolist()
    print(f"B_all候選池（美股window4完整universe）：{len(pool_uids)} 檔")

    idx_full = pd.read_parquet(CANDIDATE_INDEX_PATH)
    idx = idx_full[idx_full["market"] == "US"].set_index("strategy_uid")
    idx_pool = idx.loc[idx.index.intersection(pool_uids)]

    print(">> 載入 US MarketData ...")
    md = MarketData("US")
    md_map = {"US": md}

    rows = []
    for window_no, quarters in WINDOW_QUARTERS.items():
        print(f"\n=== window{window_no}（OOS {quarters[1]}~{quarters[-1]}）===")
        rep_uids = get_members(window_no)
        for i in range(len(quarters) - 1):
            as_of, end = quarters[i], quarters[i + 1]

            # ①代表投組：真實報酬 + 全市場等權基準（都來自outcome_layer）
            w_rep = simulate.resolve_weights(md, idx, rep_uids, as_of)
            outcome = monitor.outcome_layer(md_map, w_rep, as_of, end)
            port_ret = outcome["portfolio_realized_return"]
            ew_bench = outcome["equal_weight_benchmark_return"]

            # ②B_all：候選池整體股票層equal weight
            w_pool, n_empty = _fast_resolve_w0(md, idx_pool, pool_uids, as_of, row_cache={})
            ball_res = monitor.outcome_layer(md_map, w_pool, as_of, end)
            ball_ret = ball_res["portfolio_realized_return"]

            excess_vs_ball = port_ret - ball_ret
            ball_excess_vs_equal_weight = ball_ret - ew_bench
            print(f"  {as_of}->{end}：port={port_ret:+.4%} ew_bench={ew_bench:+.4%} "
                 f"ball={ball_ret:+.4%}（{n_empty}個策略當期無持股）")
            rows.append({"window_no": window_no, "as_of": as_of, "end": end,
                        "portfolio_return": port_ret, "equal_weight_bench": ew_bench,
                        "excess_vs_ball": excess_vs_ball, "ball_benchmark_return": ball_ret,
                        "ball_excess_vs_equal_weight": ball_excess_vs_equal_weight})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {OUT_PATH}")

    print(f"\n=== 美股 M0/M6 用的 excess_vs_ball 分布（n={len(df)}）===")
    print(f"  p10={df.excess_vs_ball.quantile(0.10):+.4%}")
    print(f"\n=== 美股 M7 用的 ball_excess_vs_equal_weight 分布（n={len(df)}）===")
    print(f"  p10={df.ball_excess_vs_equal_weight.quantile(0.10):+.4%}")


if __name__ == "__main__":
    main()
