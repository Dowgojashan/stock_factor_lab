# -*- coding: utf-8 -*-
"""M0／M6 需要的「績效層歷史分布」：windows 1-3（2015-2023，符合 §11.7 只用
訓練期資料）逐季的「相對 B_all（＝等權大盤，見 A9）超額」，用來訂 p10 門檻——
performance_below_p10 ＝ 本季 excess_vs_ball 是否低於這個歷史 p10。

跟 M4 用同一個流量變數（excess_vs_ball），但角度不同：M4 看「連續 2 期 <0」
（持續性），這裡看「這一期本身是不是歷史級的極端值」（離群值）。

🔴 這是 `_prelim_k_anchor_check.py` 算過的同一組真實資料，但那支腳本**只存了
窗次層級的年化彙總**，沒有把逐季的原始值存成檔案（只印在終端機，沒有落盤）。
這裡重新計算並正確存檔——不憑印象引用之前終端機印出來的數字，重新算一次
才寫進正式分布。
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
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "m0_performance_history.csv"

WINDOWS = {
    1: {"is_end": "2014-12-31", "oos_start": "2015-01-01", "oos_end": "2017-12-31"},
    2: {"is_end": "2017-12-31", "oos_start": "2018-01-01", "oos_end": "2020-12-31"},
    3: {"is_end": "2020-12-31", "oos_start": "2021-01-01", "oos_end": "2023-12-31"},
}


def quarter_ends(start, end):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="Q")]


def get_members(window_no):
    m = pd.read_parquet(MEMBERS_PATH)
    key = dict(tree_key="TW", scheme="E", window_no=window_no, k_mode="silhouette_is",
              ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    assert len(sub) == 1
    return list(sub.iloc[0]["members"])


def resolve_weights(md, idx, uids, as_of):
    n_strat = len(uids)
    weights = {}
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

    rows = []
    for wno, meta in WINDOWS.items():
        uids = get_members(wno)
        checkpoints = [meta["is_end"]] + quarter_ends(meta["oos_start"], meta["oos_end"])
        for i in range(len(checkpoints) - 1):
            as_of, end = checkpoints[i], checkpoints[i + 1]
            weights = resolve_weights(md, idx, uids, as_of)
            res = measure(md_map, weights, as_of, end)
            pr = res["portfolio_realized_return"]
            er = res["by_market_benchmark"]["TW"]["equal_weight_benchmark_return"]
            excess = pr - er if er is not None else None
            rows.append({"window_no": wno, "as_of": as_of, "end": end,
                        "portfolio_return": pr, "equal_weight_bench": er,
                        "excess_vs_ball": excess})
            print(f"  window {wno} {as_of}->{end}: 投組={pr:+.2%} 等權={er:+.2%} 超額(vs B_all)={excess:+.2%}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {OUT_PATH}（n={len(df)}）")

    dist = df["excess_vs_ball"].dropna()
    p10 = dist.quantile(0.10)
    print(f"\n=== M0/M6 績效層歷史分布（windows 1-3，n={len(dist)}）===")
    print(f"  平均: {dist.mean():+.2%}  標準差: {dist.std():.2%}")
    print(f"  p10={p10:+.2%}  中位={dist.median():+.2%}  p90={dist.quantile(0.9):+.2%}")


if __name__ == "__main__":
    main()
