# -*- coding: utf-8 -*-
"""A5：Coverage Tilt（§7.4b④）前置驗證，比照`_prelim_a5_w2_prevalidation.py`
（W2c的前置驗證）同一套流程與紀律——**只能用window 1-3，不可碰window 4**
（否則就是用答案驗證答案的循環論證，§7.6已經明講過這條規則）。

🔴 2026-09-22改版：第一版用二元covers_hot（選到≥1檔Hot Segment股票就算True）驗證，
發現93~96%的legacy寬持股策略（平均約500檔持股）幾乎必然covers_hot=True（用超幾何
分布可以算出「一檔熱門股都沒選到」機率趨近於0），二元判準對這種策略沒有鑑別力，
β怎麼調都沒差（見開發追蹤§1.5）。**這版改成連續的覆蓋比例**：

    coverage_frac(r, t) = |holdings(r,t) ∩ HotSegment(t)| / |HotSegment(t)|
    w_r(t) ∝ (1/R) × (1 + β · coverage_frac(r, t))，再正規化

這裡驗證β ∈ {0, 1, 2, 5, 10}（coverage_frac量級通常在0~0.3左右，β要比二元版大才能
產生有意義的傾斜幅度，不是同一組數字複製過來）。只測TW市場（N=3、X=10%，已在
`_design_test_hot_segment.py`校準過）——US/XM的β驗證是後續步驟，不在這次一次做完。

用法（cwd 必須是 code/）：
    python -m app._prelim_coverage_tilt_prevalidation
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
from _design_test_guaranteed_megacap_slot import selects_symbol  # noqa: E402
from _design_test_hot_segment import _quarterly_returns, hot_segment, monthly_close  # noqa: E402

MEMBERS_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "coverage_tilt_prevalidation.csv"

BETAS = [0.0, 1.0, 2.0, 5.0, 10.0]
N_MOMENTUM = 3   # TW已校準值
X_HOT = 0.10

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


def resolve_tilted_weights(md: MarketData, idx: pd.DataFrame, uids: list[str],
                           as_of: str, hot_set: set[str], beta: float) -> tuple[dict[str, float], int]:
    """β=0 時還原純等權（跟resolve_w0()完全一樣的結果），β>0 時對「覆蓋Hot Segment
    的代表策略」整體加重，同一策略內部仍是等權持股（不動選股邏輯本身，只調策略間的
    相對權重——符合書上「只做符合原本設計的調整」的紀律，見領域筆記§6）。"""
    covers = {}
    holdings = {}
    for uid in uids:
        row = idx.loc[uid]
        try:
            syms, _ = resolve_holdings(md, row, as_of)
        except RuntimeError:
            syms = []
        holdings[uid] = syms
        n_hot_in = len(set(syms) & hot_set) if syms else 0
        covers[uid] = (n_hot_in / len(hot_set)) if hot_set else 0.0   # 🔴 連續版：覆蓋比例，不是二元
        # （2026-09-22改版原因：二元「選到≥1檔就算covers」對legacy寬持股策略(~500檔)
        #  幾乎恆真——93~96%策略covers_hot=True，β完全沒有鑑別力，見開發追蹤§1.5）

    avg_frac = sum(covers.values()) / len(covers) if covers else 0.0
    raw_w = {uid: (1.0 + beta * covers[uid]) for uid in uids if holdings[uid]}
    total = sum(raw_w.values())
    strategy_w = {uid: v / total for uid, v in raw_w.items()} if total else {}

    weights: dict[str, float] = {}
    for uid, sw in strategy_w.items():
        syms = holdings[uid]
        w = sw / len(syms)
        for s in syms:
            weights[s] = weights.get(s, 0.0) + w
    return weights, avg_frac


def main():
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    print("載入 TW MarketData…")
    md = MarketData("TW")
    md_map = {"TW": md}

    ret_df = _quarterly_returns(monthly_close(md), N_MOMENTUM)

    rows = []
    for wno, meta in WINDOWS.items():
        uids = get_members(wno)
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

            for b in BETAS:
                wt, avg_frac = resolve_tilted_weights(md, idx, uids, as_of, hot, b)
                res = measure(md_map, wt, as_of, end)
                per_beta_rets[b].append(res["portfolio_realized_return"])
                if b == BETAS[0]:
                    ew_rets.append(res["by_market_benchmark"]["TW"]["equal_weight_benchmark_return"])
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
