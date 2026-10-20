# -*- coding: utf-8 -*-
"""美股版：Coverage Tilt（§7.4b④）前置驗證，mirror
`_prelim_coverage_tilt_prevalidation.py`（TW版）同一套邏輯跟紀律——
**只能用window 1-3，不可碰window 4**（§7.6）。

背景：2026-10-19查`actions.coverage_tilt_reference()`的caveat「尚未針對
US/XM市場跑β驗證，只有TW window 1-3這一組證據」，追查發現雖然有一組
`_coverage_tilt_window0_us.py`產出的美股β網格資料，但那是「window0」
（IS 2007-2012/OOS 2013-2014），跟caveat講的「window 1-3（2015-2023）」
不是同一個驗證窗——TW版自己的docstring（第16行）本來就寫明「US/XM的β驗證
是後續步驟，不在這次一次做完」，不是忘記做，是本來就排到後面。這支腳本
把這個後續步驟補上，讓美股也有一組跟TW完全同方法論、同窗次定義的β敏感度
證據。

跟TW版唯一的差異：市場換成US、`N_MOMENTUM`改用`monitor.HOT_SEGMENT_N["US"]
=6`（不是TW的3，動能回顧期本來就是市場各自校準的值）、benchmark欄位讀
`by_market_benchmark["US"]`。其餘（β網格[0,1,2,5,10]、連續版coverage_frac
定義、IS/OOS窗次切法）完全相同，不是另外發明一套。

用法（cwd 必須是 code/）：
    PYTHONIOENCODING=utf-8 python -m app._prelim_coverage_tilt_prevalidation_us
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _design_test_hot_segment import _quarterly_returns, hot_segment, monthly_close  # noqa: E402

MEMBERS_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "coverage_tilt_prevalidation_US.csv"

BETAS = [0.0, 1.0, 2.0, 5.0, 10.0]
N_MOMENTUM = 6   # US已校準值（monitor.HOT_SEGMENT_N["US"]），不是TW的3
X_HOT = 0.10

WINDOWS = {
    1: {"is_end": "2014-12-31", "oos_start": "2015-01-01", "oos_end": "2017-12-31"},
    2: {"is_end": "2017-12-31", "oos_start": "2018-01-01", "oos_end": "2020-12-31"},
    3: {"is_end": "2020-12-31", "oos_start": "2021-01-01", "oos_end": "2023-12-31"},
}


def quarter_ends(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="Q")]


def get_members(m: pd.DataFrame, window_no: int) -> list[str]:
    key = dict(tree_key="US", scheme="E", window_no=window_no, k_mode="silhouette_is",
              ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    assert len(sub) == 1
    return list(sub.iloc[0]["members"])


def resolve_holdings_and_coverage(md: MarketData, idx: pd.DataFrame, uids: list[str],
                                  as_of: str, hot_set: set[str]) -> tuple[dict, dict, float]:
    covers = {}
    holdings = {}
    for uid in uids:
        row = idx.loc[uid]
        try:
            syms, _ = resolve_holdings(md, row, as_of)
        except (RuntimeError, KeyError, ValueError):
            syms = []
        holdings[uid] = syms
        n_hot_in = len(set(syms) & hot_set) if syms else 0
        covers[uid] = (n_hot_in / len(hot_set)) if hot_set else 0.0
    avg_frac = sum(covers.values()) / len(covers) if covers else 0.0
    return holdings, covers, avg_frac


def tilt_weights(holdings: dict, covers: dict, uids: list[str], beta: float) -> dict[str, float]:
    raw_w = {uid: (1.0 + beta * covers[uid]) for uid in uids if holdings[uid]}
    total = sum(raw_w.values())
    strategy_w = {uid: v / total for uid, v in raw_w.items()} if total else {}

    weights: dict[str, float] = {}
    for uid, sw in strategy_w.items():
        syms = holdings[uid]
        w = sw / len(syms)
        for s in syms:
            weights[s] = weights.get(s, 0.0) + w
    return weights


def main():
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    print("載入 US MarketData…")
    md = MarketData("US")
    md_map = {"US": md}
    members_df = pd.read_parquet(MEMBERS_PATH)

    ret_df = _quarterly_returns(monthly_close(md), N_MOMENTUM)

    rows = []
    for wno, meta in WINDOWS.items():
        uids = get_members(members_df, wno)
        checkpoints = [meta["is_end"]] + quarter_ends(meta["oos_start"], meta["oos_end"])

        per_beta_rets: dict[float, list[float]] = {b: [] for b in BETAS}
        ew_rets = []
        avg_frac_log = []
        for i in range(len(checkpoints) - 1):
            as_of, end = checkpoints[i], checkpoints[i + 1]
            q_idx = ret_df.index[ret_df.index <= pd.Timestamp(as_of)]
            if len(q_idx) == 0:
                print(f"  window {wno} {as_of}：動能資料不足，跳過")
                continue
            hot = set(hot_segment(ret_df.loc[q_idx.max()], X_HOT))
            holdings, covers, avg_frac = resolve_holdings_and_coverage(md, idx, uids, as_of, hot)

            for b in BETAS:
                wt = tilt_weights(holdings, covers, uids, b)
                res = measure(md_map, wt, as_of, end)
                per_beta_rets[b].append(res["portfolio_realized_return"])
                if b == BETAS[0]:
                    ew = res["by_market_benchmark"]["US"]["equal_weight_benchmark_return"]
                    if ew is None:
                        raise ValueError(
                            f"window {wno} {as_of}~{end}：US equal_weight_benchmark_return"
                            f"是None（完全沒有可用報酬的股票），不是正常的資料缺口")
                    ew_rets.append(ew)
                    avg_frac_log.append(avg_frac)

        if not ew_rets:
            continue
        n_years = len(ew_rets) / 4.0
        cum_ew = pd.Series([1 + r for r in ew_rets]).prod() - 1
        ann_ew = (1 + cum_ew) ** (1 / n_years) - 1
        avg_cov = sum(avg_frac_log) / len(avg_frac_log) if avg_frac_log else float("nan")

        for b in BETAS:
            rets = per_beta_rets[b]
            cum = pd.Series([1 + r for r in rets]).prod() - 1
            ann = (1 + cum) ** (1 / n_years) - 1
            print(f"window {wno} β={b:.1f}: 年化={ann:+.2%}  (等權大盤={ann_ew:+.2%}  "
                 f"超額={ann-ann_ew:+.2%})  策略平均覆蓋比例={avg_cov:.1%}（連續版coverage_frac）")
            rows.append({"window_no": wno, "beta": b, "ann_return": ann,
                        "ann_ew_benchmark": ann_ew, "ann_excess": ann - ann_ew,
                        "avg_coverage_frac": avg_cov, "n_rep": len(uids)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {OUT_PATH}")

    print("\n=== 三窗平均，逐 β ===")
    summary = df.groupby("beta").agg(mean_ann_return=("ann_return", "mean"),
                                     mean_ann_excess=("ann_excess", "mean")).reset_index()
    print(summary.to_string(index=False))
    best = summary.loc[summary.mean_ann_excess.idxmax()]
    print(f"\n三窗平均表現最好的 β = {best.beta}（平均超額 {best.mean_ann_excess:+.2%}/yr）")
    b0 = summary.loc[summary.beta == BETAS[0], "mean_ann_excess"].iloc[0]
    print(f"對照 β=0（現況，等於不做Coverage Tilt）: {b0:+.2%}/yr")


if __name__ == "__main__":
    main()
