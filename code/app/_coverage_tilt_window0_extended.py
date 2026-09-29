# -*- coding: utf-8 -*-
"""使用者2026-09-29提出兩個真實質疑，這裡回應：
①「window1-3沒有危機沒辦法測」這個講法不精確——coverage_frac本來就不是100%，
  任何時期都能算傾斜效果，不需要等危機；②可以拿別的時期／別的市場再測。

**範圍**：Coverage Tilt既有window1-3（TW，OOS 2015-2017/2018-2020/2021-2023）
都是「scheme E」（anchored，IS固定從市場自己的起點累積）家族，最早的window1
OOS已經是2015-2017，2013-2014這段完全沒測過。這裡補上：
  - TW window0：IS 2007-01~2012-12，OOS 2013-2014——剛好等於這個session一路在用
    的rolling window1（IS/OOS完全對齊，§3.4已確認scheme A window1跟rolling
    window1是同一個候選池），直接重用`_design_test_hotsample_frequency.
    load_variant_pools()`算好的`rolling_v1[1]`，不必重建樹
  - US window0：IS 2007-01~2012-12，OOS 2013-2014——US沒有現成的這個IS範圍
    候選池，這裡現場建一次（`_silhouette_picks_lib.build_silhouette_picks()`）

**兩個市場都用各自已校準的N值**（不是都套TW的N=3）：TW用N=3、US用N=6，
沿用`_design_test_hot_segment.py`§1.3已經驗證過的市場別參數，不能跨市場沿用
同一組（k值/X門檻同一個道理，這個專案已經踩過這個教訓）。

β網格沿用`_prelim_coverage_tilt_prevalidation.py`原本window1-3就定好的
{0,1,2,5,10}，不是看過這個新窗次的結果才決定——維持「網格先定、不回頭看
結果調整」的紀律。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._coverage_tilt_window0_extended
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
from research import walkforward_matrix as WF  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _design_test_hot_segment import hot_segment, monthly_close  # noqa: E402
from _design_test_hotsample_frequency import load_variant_pools, momentum_asof  # noqa: E402
from _prelim_coverage_tilt_prevalidation import (quarter_ends,  # noqa: E402
                                                 resolve_holdings_and_coverage, tilt_weights)
from _silhouette_picks_lib import build_silhouette_picks  # noqa: E402

BETAS = [0.0, 1.0, 2.0, 5.0, 10.0]
N_BY_MARKET = {"TW": 3, "US": 6}   # 已校準值，見§1.1/§1.3，不可跨市場沿用同一組
X_HOT = 0.10
WINDOW0 = {"is_start": "2007-01", "is_end": "2012-12-31", "oos_start": "2013-01-01", "oos_end": "2014-12-31"}


def get_us_window0_members(months_long, meta_pool, f_combo_map, idx: pd.DataFrame) -> list[str]:
    result = build_silhouette_picks("US", WINDOW0["is_start"], "2012-12", "baseline",
                                    months_long, meta_pool, f_combo_map, idx)
    return result["equal"]


def run_market(market: str, uids: list[str], idx: pd.DataFrame) -> list[dict]:
    print(f"\n=== {market} window0：IS {WINDOW0['is_start']}~2012-12｜OOS 2013-2014"
         f"（{len(uids)}檔代表策略，N={N_BY_MARKET[market]}）===")
    md = MarketData(market)
    md_map = {market: md}
    # 🔴 修正：原本沿用`_design_test_hot_segment._quarterly_returns()`，但那個函式
    # 只在模組常數`QUARTER_ENDS`（2015-03-31~2025-12-31）的固定日期算動能，
    # window0的checkpoint（2012-12-31~2014-09-30）全部落在這個範圍之前，導致
    # 每一個checkpoint都被誤判成「動能資料不足」而跳過（第一次執行時真的發生了，
    # 不是假設性風險）。改用`_design_test_hotsample_frequency.momentum_asof()`
    # ——同一個「trailing N個月動能，ending at目標日期」定義，但吃任意日期，
    # 不受限於那個固定清單。
    monthly = monthly_close(md)
    checkpoints = [WINDOW0["is_end"][:10]] + quarter_ends(WINDOW0["oos_start"], WINDOW0["oos_end"])

    per_beta_rets: dict[float, list[float]] = {b: [] for b in BETAS}
    ew_rets = []
    avg_frac_log = []
    for i in range(len(checkpoints) - 1):
        as_of, end = checkpoints[i], checkpoints[i + 1]
        try:
            mom = momentum_asof(monthly, pd.Timestamp(as_of), N_BY_MARKET[market])
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
                ew = res["by_market_benchmark"][market]["equal_weight_benchmark_return"]
                if ew is None:
                    raise ValueError(f"{market} window0 {as_of}~{end}：equal_weight_benchmark_return是None")
                ew_rets.append(ew)
                avg_frac_log.append(avg_frac)

    if not ew_rets:
        print(f"  {market}：沒有可用的checkpoint，跳過")
        return []
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
        rows.append({"market": market, "window_no": 0, "beta": b, "ann_return": ann,
                    "ann_ew_benchmark": ann_ew, "ann_excess": ann - ann_ew,
                    "avg_coverage_frac": avg_cov, "n_rep": len(uids)})
    return rows


def main():
    print(">> 載入資料...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    pools = load_variant_pools()
    tw_uids = pools["rolling_v1"][1]   # IS 2007-2012，跟window0完全對齊，不必重建樹
    print(f"TW window0候選池：直接重用rolling window1（{len(tw_uids)}檔），已確認IS對齊")

    print("\n>> 建US window0候選池（現場建樹，之前沒有這個IS範圍的現成資料）...")
    us_uids = get_us_window0_members(months_long, meta_pool, f_combo_map, idx)

    all_rows = []
    all_rows += run_market("TW", tw_uids, idx)
    all_rows += run_market("US", us_uids, idx)

    out = pd.DataFrame(all_rows)
    out_path = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "coverage_tilt_window0_extended.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}")

    print("\n=== 對照：window0 vs 既有window1-3（TW，來自coverage_tilt_prevalidation.csv）===")
    prev = pd.read_csv(Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "coverage_tilt_prevalidation.csv")
    prev_summary = prev.groupby("beta")["ann_excess"].mean().reset_index()
    tw0 = out[out.market == "TW"][["beta", "ann_excess"]].rename(columns={"ann_excess": "window0_excess"})
    cmp = tw0.merge(prev_summary.rename(columns={"ann_excess": "window1_3_mean_excess"}), on="beta")
    print(cmp.to_string(index=False))


if __name__ == "__main__":
    main()
