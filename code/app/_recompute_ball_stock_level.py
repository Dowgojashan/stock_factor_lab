# -*- coding: utf-8 -*-
"""§8待辦item10（2026-09-22）：用股票層方法重算 B_all，讓 §7.4 基準鏈拆解
跟等權/市值加權大盤用同一套量測系統。

背景（設計文件 §7.4）：現行 B_all 的 1.39% 來自凍結的**策略層**報酬
（`returns_monthly.parquet`，策略自己的歷史回測執行軌跡），但等權/市值加權
大盤是**股價直接算**（`monitor.outcome_layer()`／`performance.measure()`）——
跨了兩套量測系統，是§5.3記載的既知陷阱。這裡用跟A_hrp／等權/市值加權大盤
同一套「股票層」方法重算B_all：window4完整候選池（6,679檔，跟HRP建樹時用的
universe完全一致，直接沿用`tsmc_cluster_concentration.csv`已存的清單，不重新
篩選）equal-weight-across-strategies聚合成單一股票層權重，逐季用
`performance.measure()`量測真實報酬。

B_all定義（沿用`resolve_w0()`同一套邏輯，見`_prelim_a5_w2_prevalidation.py`
docstring）：每個策略等權（1/n_strat），策略內部選中的股票再等權分配。
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
QUARTERS = ["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
           "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]


def resolve_w0(md: MarketData, idx: pd.DataFrame, uids: list[str], as_of: str) -> dict[str, float]:
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
    print(f"B_all 候選池（window4完整樹universe）：{len(tree_uids)} 檔")

    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    idx = idx.loc[idx.index.intersection(tree_uids)]

    print(">> 載入 TW MarketData ...")
    md = MarketData("TW")
    md_map = {"TW": md}

    rows = []
    for i in range(len(QUARTERS) - 1):
        as_of, end = QUARTERS[i], QUARTERS[i + 1]
        weights, n_empty = resolve_w0(md, idx, tree_uids, as_of)
        n_unique = len(weights)
        total_w = sum(weights.values())
        res = measure(md_map, weights, as_of, end)
        ret = res["portfolio_realized_return"]
        rows.append({"as_of": as_of, "end": end, "n_unique_stocks": n_unique,
                    "n_empty_strategies": n_empty, "total_weight": total_w,
                    "stock_level_return": ret})
        print(f"  {as_of}->{end}：{n_unique}檔不重複股票（{n_empty}個策略當期無持股）"
             f"　權重涵蓋率={total_w:.2%}　股票層報酬={ret:+.4%}")

    df = pd.DataFrame(rows)
    cum = float((1 + df["stock_level_return"]).prod() - 1)
    print(f"\n8季累積（股票層方法）：{cum:+.2%}")
    print("對照：舊版（策略層returns_monthly.parquet凍結報酬，window4 OOS）：需另外查證比對")

    out = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "ball_stock_level_recompute.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out}")


if __name__ == "__main__":
    main()
