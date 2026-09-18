# -*- coding: utf-8 -*-
"""驗證「交易成本不對稱」假說：凍結矩陣版 A_hrp 報酬是扣過台股手續費+證交稅的
（backtest.py fee_ratio=1.425/1000, tax_ratio=3/1000），但它的比較基準「全市場
等權」（market_benchmark.py）是純價格報酬、沒扣成本。即時系統版 performance.measure()
兩邊都沒扣成本，是公平比較，所以看起來超額特別高。

做法：對 scheme E windows 1-3（跟 _prelim_k_anchor_check.py 同一批），用真實
resolve_holdings 算出逐季持股權重，算出逐季換手率（turnover），估算「如果比照
凍結矩陣的成本假設倒扣手續費+證交稅」後的成本調整版超額，看是否收斂到 M-17
的 +4.53pp/+2.60pp 附近。
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

MEMBERS_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"

FEE_RATIO = 1.425 / 1000
TAX_RATIO = 3 / 1000
ROUNDTRIP_COST = 2 * FEE_RATIO + TAX_RATIO  # 賣舊(fee+tax) + 買新(fee) = 一單位換手的成本

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


def turnover(w_old: dict[str, float], w_new: dict[str, float]) -> float:
    """單邊換手率：sum(|新-舊|)/2。0=完全沒換，1=全部換過。"""
    keys = set(w_old) | set(w_new)
    return sum(abs(w_new.get(k, 0.0) - w_old.get(k, 0.0)) for k in keys) / 2.0


def main():
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    print("載入 TW MarketData…")
    md = MarketData("TW")
    md_map = {"TW": md}

    all_summary = []
    for wno, meta in WINDOWS.items():
        uids = get_members(wno)
        checkpoints = [meta["is_end"]] + quarter_ends(meta["oos_start"], meta["oos_end"])
        print(f"\n=== window {wno}（{meta['is_end']} ~ {meta['oos_end']}，{len(checkpoints)-1} 季）===")

        port_rets, ew_rets, turnovers = [], [], []
        prev_w = None
        for i in range(len(checkpoints) - 1):
            as_of, end = checkpoints[i], checkpoints[i + 1]
            weights = resolve_weights(md, idx, uids, as_of)
            if prev_w is not None:
                t = turnover(prev_w, weights)
                turnovers.append(t)
            prev_w = weights
            res = measure(md_map, weights, as_of, end)
            pr = res["portfolio_realized_return"]
            er = res["by_market_benchmark"]["TW"]["equal_weight_benchmark_return"]
            port_rets.append(pr); ew_rets.append(er)

        # 每季換手成本：用「進場那一季」的換手率估（第一季用 1.0，全新建倉）
        turnovers_full = [1.0] + turnovers  # 對齊到每個 checkpoint 進場動作
        cost_per_q = [t * ROUNDTRIP_COST for t in turnovers_full]
        port_rets_net = [pr - c for pr, c in zip(port_rets, cost_per_q)]

        cum_port = pd.Series([1 + r for r in port_rets]).prod() - 1
        cum_port_net = pd.Series([1 + r for r in port_rets_net]).prod() - 1
        cum_ew = pd.Series([1 + r for r in ew_rets]).prod() - 1
        n_years = len(port_rets) / 4.0
        ann_port = (1 + cum_port) ** (1 / n_years) - 1
        ann_port_net = (1 + cum_port_net) ** (1 / n_years) - 1
        ann_ew = (1 + cum_ew) ** (1 / n_years) - 1
        ann_excess_gross = ann_port - ann_ew
        ann_excess_net = ann_port_net - ann_ew
        avg_turnover = sum(turnovers) / len(turnovers) if turnovers else float("nan")

        print(f"  平均每季換手率: {avg_turnover:.1%}")
        print(f"  毛報酬年化超額（無成本）: {ann_excess_gross:+.2%}/yr")
        print(f"  淨報酬年化超額（扣手續費+證交稅）: {ann_excess_net:+.2%}/yr")
        print(f"  成本拖累: {ann_excess_gross - ann_excess_net:.2%}/yr")

        all_summary.append({"window_no": wno, "avg_turnover": avg_turnover,
                            "ann_excess_gross": ann_excess_gross, "ann_excess_net": ann_excess_net,
                            "cost_drag": ann_excess_gross - ann_excess_net})

    df = pd.DataFrame(all_summary)
    print("\n=== 總結：換手成本能不能解釋落差 ===")
    print(f"3 窗平均：毛超額 {df.ann_excess_gross.mean():+.2%}/yr -> 淨超額(扣成本) {df.ann_excess_net.mean():+.2%}/yr")
    print(f"平均成本拖累: {df.cost_drag.mean():.2%}/yr")
    print(f"M-17 凍結矩陣版（同樣是扣成本 vs 沒扣成本的基準）: +4.53%/yr（單一期間）/ +2.60%/yr（45窗平均）")
    print(f"\n若淨超額落在 M-17 區間附近，代表交易成本不對稱假說成立。")


if __name__ == "__main__":
    main()
