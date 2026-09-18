# -*- coding: utf-8 -*-
"""驗證 k 錨點的機制假設：即時系統版超額（+11.42%/yr）遠高於 M-17 凍結矩陣版
（+4.53%/yr），可能是季頻 vs 月頻重新平衡造成的系統性放大，也可能是樣本數
太小（只有 3 窗）的雜訊。這裡把 windows 1-3 的即時版重跑成**月頻**重新平衡
（跟凍結矩陣同一個頻率），若超額報酬因此大幅收斂到 +4.53%/yr 附近，代表
是頻率造成的假訊號；若仍然遠高於凍結矩陣版，代表頻率不是主因。
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


def month_ends(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="M")]


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
        checkpoints = [meta["is_end"]] + month_ends(meta["oos_start"], meta["oos_end"])
        print(f"\n=== window {wno}（{meta['is_end']} ~ {meta['oos_end']}，"
             f"{len(checkpoints)-1} 個月，月頻重新平衡）===")

        port_rets, ew_rets = [], []
        for i in range(len(checkpoints) - 1):
            as_of, end = checkpoints[i], checkpoints[i + 1]
            weights = resolve_weights(md, idx, uids, as_of)
            res = measure(md_map, weights, as_of, end)
            pr = res["portfolio_realized_return"]
            er = res["by_market_benchmark"]["TW"]["equal_weight_benchmark_return"]
            port_rets.append(pr); ew_rets.append(er)

        cum_port = pd.Series([1 + r for r in port_rets]).prod() - 1
        cum_ew = pd.Series([1 + r for r in ew_rets]).prod() - 1
        n_years = len(port_rets) / 12.0
        ann_port = (1 + cum_port) ** (1 / n_years) - 1
        ann_ew = (1 + cum_ew) ** (1 / n_years) - 1
        ann_excess = ann_port - ann_ew
        print(f"  累計：投組={cum_port:+.2%}  等權={cum_ew:+.2%}  累計超額={cum_port-cum_ew:+.2%}")
        print(f"  年化：投組={ann_port:+.2%}/yr  等權={ann_ew:+.2%}/yr  年化超額={ann_excess:+.2%}/yr")
        all_summary.append({"window_no": wno, "n_months": len(port_rets),
                            "cum_port": cum_port, "cum_ew": cum_ew,
                            "cum_excess": cum_port - cum_ew, "ann_excess": ann_excess})

    df = pd.DataFrame(all_summary)
    out = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "k_anchor_monthly.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out}")

    print("\n=== 三個版本對照 ===")
    print(f"月頻即時版（本次）：windows 1-3 平均年化超額 = {df.ann_excess.mean():+.2%}/yr")
    print(f"季頻即時版（上次）：windows 1-3 平均年化超額 = +11.42%/yr")
    print(f"凍結矩陣版（M-17，45 窗）：+4.53%/yr")
    print()
    gap_to_frozen = abs(df.ann_excess.mean() - 0.0453)
    gap_quarterly_to_frozen = abs(0.1142 - 0.0453)
    print(f"月頻版與凍結矩陣版的差距：{gap_to_frozen:.2%}")
    print(f"季頻版與凍結矩陣版的差距：{gap_quarterly_to_frozen:.2%}")
    if gap_to_frozen < gap_quarterly_to_frozen * 0.5:
        print("=> 月頻版明顯收斂到凍結矩陣版附近，支持「頻率造成放大」假設")
    elif gap_to_frozen > gap_quarterly_to_frozen * 0.8:
        print("=> 月頻版沒有明顯收斂，頻率不是主因，可能是樣本數/其他因素")
    else:
        print("=> 部分收斂，兩個因素可能都有影響，需要更多樣本才能拆解")


if __name__ == "__main__":
    main()
