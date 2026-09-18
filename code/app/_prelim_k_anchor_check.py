# -*- coding: utf-8 -*-
"""§3.4 的 k 值錨用的是 M-17（凍結矩陣算出 A_hrp 相對等權市場 +4.53pp/yr）。
但 A2/A3 已發現凍結矩陣 vs 即時股票層量測系統差距達 3.7~15 倍——這個錨本身
可能需要用即時系統重新驗證，否則 k 會建立在不一致的基準上。

做法：對 scheme E window 1/2/3（跟 windows 1-3 的雜訊分析用同一批成員），
用跟 A2/A3 window4 完全一樣的方法（真實權重 + performance.measure() 串接）
算出真實已實現報酬，跟同期真實等權大盤比較，算出「即時系統版」的歷史超額，
拿來跟 M-17 的 +4.53pp/yr 對照。
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

        port_rets, ew_rets = [], []
        for i in range(len(checkpoints) - 1):
            as_of, end = checkpoints[i], checkpoints[i + 1]
            weights = resolve_weights(md, idx, uids, as_of)
            res = measure(md_map, weights, as_of, end)
            pr = res["portfolio_realized_return"]
            er = res["by_market_benchmark"]["TW"]["equal_weight_benchmark_return"]
            port_rets.append(pr); ew_rets.append(er)
            print(f"  {as_of}->{end}: 投組={pr:+.2%}  等權={er:+.2%}  超額={pr-er:+.2%}")

        cum_port = pd.Series([1 + r for r in port_rets]).prod() - 1
        cum_ew = pd.Series([1 + r for r in ew_rets]).prod() - 1
        n_years = len(port_rets) / 4.0
        ann_port = (1 + cum_port) ** (1 / n_years) - 1
        ann_ew = (1 + cum_ew) ** (1 / n_years) - 1
        ann_excess = ann_port - ann_ew
        print(f"  累計：投組={cum_port:+.2%}  等權={cum_ew:+.2%}  累計超額={cum_port-cum_ew:+.2%}")
        print(f"  年化：投組={ann_port:+.2%}/yr  等權={ann_ew:+.2%}/yr  年化超額={ann_excess:+.2%}/yr")
        all_summary.append({"window_no": wno, "cum_port": cum_port, "cum_ew": cum_ew,
                            "cum_excess": cum_port - cum_ew, "ann_excess": ann_excess})

    df = pd.DataFrame(all_summary)
    out = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "k_anchor_check.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out}")

    print("\n=== 即時系統版 vs M-17（凍結矩陣，+4.53pp/yr）===")
    print(f"windows 1-3 年化超額平均: {df.ann_excess.mean():+.2%}/yr")
    print(f"windows 1-3 各窗: {[f'{v:+.2%}' for v in df.ann_excess]}")
    print(f"M-17（凍結矩陣，45 窗全部）: +4.53%/yr（僅供對照，樣本數不同不可直接比對顯著性）")


if __name__ == "__main__":
    main()
