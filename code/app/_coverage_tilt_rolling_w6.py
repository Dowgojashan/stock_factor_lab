# -*- coding: utf-8 -*-
"""使用者2026-09-29指出：window0~4全部用的是anchored候選池（`scheme="E"`），
Coverage Tilt從來沒有真正拿一個跟anchored不同的rolling窗次測過（window0的TW側
只是巧合跟rolling window1重疊，不是真的測rolling）。這裡補上真正獨立的rolling
窗次——**rolling window6**（IS 2017-01~2022-12，OOS 2023-01~2025-12，共12季，
比其他window長，因為是這個session一路在用的、OOS一路延伸到現在的最後一個窗次）。

候選池：`rolling_window6_silhouette_members.parquet`（baseline，保留V1，
allocation=equal）——用baseline跟window0-4同一個變體，不是exclude_v1，維持
可比較性（exclude_v1是另一條調查線，跟Coverage Tilt這裡的比較無關）。

β網格沿用window0-4同一組{0,1,2,5,10}，不是看過這次結果才決定。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._coverage_tilt_rolling_w6
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402

from app.performance import measure  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _design_test_hot_segment import hot_segment, monthly_close  # noqa: E402
from _design_test_hotsample_frequency import momentum_asof  # noqa: E402
from _prelim_coverage_tilt_prevalidation import (quarter_ends,  # noqa: E402
                                                 resolve_holdings_and_coverage, tilt_weights)

BETAS = [0.0, 1.0, 2.0, 5.0, 10.0]
N_TW = 3
X_HOT = 0.10
WINDOW6 = {"is_end": "2022-12-31", "oos_start": "2023-01-01", "oos_end": "2025-12-31"}
MEMBERS_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "rolling_window6_silhouette_members.parquet"
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "coverage_tilt_rolling_w6.csv"


def get_members() -> list[str]:
    m = pd.read_parquet(MEMBERS_PATH)
    sub = m[m.allocation == "equal"]
    assert len(sub) == 1, f"預期唯一一列，實際{len(sub)}列"
    return list(sub.iloc[0]["members"])


def main():
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    uids = get_members()
    print(f"rolling window6候選池：{len(uids)}檔（baseline，IS 2017-01~2022-12，跟anchored window4的"
         f"訓練窗完全不同，這次真正測的是獨立rolling候選池，不是anchored）")

    md = MarketData("TW")
    md_map = {"TW": md}
    monthly = monthly_close(md)
    checkpoints = [WINDOW6["is_end"]] + quarter_ends(WINDOW6["oos_start"], WINDOW6["oos_end"])
    print(f"共{len(checkpoints)-1}季（OOS {WINDOW6['oos_start']}~{WINDOW6['oos_end']}）")

    per_beta_rets: dict[float, list[float]] = {b: [] for b in BETAS}
    ew_rets = []
    avg_frac_log = []
    for i in range(len(checkpoints) - 1):
        as_of, end = checkpoints[i], checkpoints[i + 1]
        try:
            mom = momentum_asof(monthly, pd.Timestamp(as_of), N_TW)
        except ValueError as e:
            print(f"  {as_of}：{e}，跳過")
            continue
        hot = set(hot_segment(mom, X_HOT))
        holdings, covers, avg_frac = resolve_holdings_and_coverage(md, idx, uids, as_of, hot)

        for b in BETAS:
            wt = tilt_weights(holdings, covers, uids, b)
            res = measure(md_map, wt, as_of, end)
            per_beta_rets[b].append(res["portfolio_realized_return"])
            if b == BETAS[0]:
                ew = res["by_market_benchmark"]["TW"]["equal_weight_benchmark_return"]
                if ew is None:
                    raise ValueError(f"rolling w6 {as_of}~{end}：equal_weight_benchmark_return是None")
                ew_rets.append(ew)
                avg_frac_log.append(avg_frac)
        print(f"  checkpoint {as_of}~{end} 完成（覆蓋比例={avg_frac:.1%}）")

    n_years = len(ew_rets) / 4.0
    cum_ew = pd.Series([1 + r for r in ew_rets]).prod() - 1
    ann_ew = (1 + cum_ew) ** (1 / n_years) - 1
    avg_cov = sum(avg_frac_log) / len(avg_frac_log) if avg_frac_log else float("nan")

    rows = []
    for b in BETAS:
        rets = per_beta_rets[b]
        cum = pd.Series([1 + r for r in rets]).prod() - 1
        ann = (1 + cum) ** (1 / n_years) - 1
        print(f"  β={b:.1f}: 年化={ann:+.2%}（等權大盤={ann_ew:+.2%}｜超額={ann-ann_ew:+.2%}）"
             f"策略平均覆蓋比例={avg_cov:.1%}")
        rows.append({"market": "TW", "window": "rolling_w6", "beta": b, "ann_return": ann,
                    "ann_ew_benchmark": ann_ew, "ann_excess": ann - ann_ew,
                    "avg_coverage_frac": avg_cov, "n_rep": len(uids)})

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {OUT_PATH}")


if __name__ == "__main__":
    main()
