# -*- coding: utf-8 -*-
"""§1.6卡住之後，使用者選option②（繼續設計更好的驗證方式），不是硬湊window1-3
的雜訊找一個「看起來變好」的β。

關鍵洞察：window1-3沒有真正的「覆蓋率從正常掉到異常」情境可以驗證Coverage Tilt，
但window4（2024-2025）本身就是這個情境——只是先前照著`_prelim_coverage_tilt_
prevalidation.py`的紀律「只能用window1-3，不可碰window4」刻意沒去測。

這不是走回頭路破壞紀律：W2c當初的驗證邏輯是「門檻X完全用2007-2023資料校準，
校準完之後才放它去2024-2025真實觸發並評估效果」（不是循環論證，因為X沒有拿
2024-2025的表現回頭調整）。Coverage Tilt的β網格{0,1,2,5,10}同樣是在完全沒看過
window4資料的情況下（§1.6）就先定好的——這裡把這個**已經固定、沒有因為看到
window4結果而回頭調整**的網格拿去window4算一次真實效果，是同一套邏輯，不算
偷看答案。**不管結果好壞都如實記錄，不會因為看到window4結果就回頭改β網格**。

沿用`_prelim_coverage_tilt_prevalidation.py`的`resolve_holdings_and_coverage()`／
`tilt_weights()`／`get_members()`／`quarter_ends()`，只新增window4這一組。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._coverage_tilt_window4_validation
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
from _design_test_hot_segment import _quarterly_returns, hot_segment, monthly_close  # noqa: E402
from _prelim_coverage_tilt_prevalidation import (BETAS, MEMBERS_PATH, N_MOMENTUM,  # noqa: E402
                                                 X_HOT, get_members, quarter_ends,
                                                 resolve_holdings_and_coverage, tilt_weights)

OUT_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "coverage_tilt_window4_validation.csv"

WINDOW4 = {"is_end": "2023-12-31", "oos_start": "2024-01-01", "oos_end": "2025-12-31"}


def main():
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    print("載入 TW MarketData…")
    md = MarketData("TW")
    md_map = {"TW": md}
    members_df = pd.read_parquet(MEMBERS_PATH)

    ret_df = _quarterly_returns(monthly_close(md), N_MOMENTUM)

    uids = get_members(members_df, 4)
    print(f"window4 代表策略數：{len(uids)}")
    checkpoints = [WINDOW4["is_end"]] + quarter_ends(WINDOW4["oos_start"], WINDOW4["oos_end"])

    per_beta_rets: dict[float, list[float]] = {b: [] for b in BETAS}
    ew_rets = []
    avg_frac_log = []
    per_quarter_rows = []
    for i in range(len(checkpoints) - 1):
        as_of, end = checkpoints[i], checkpoints[i + 1]
        q_idx = ret_df.index[ret_df.index <= pd.Timestamp(as_of)]
        if len(q_idx) == 0:
            print(f"  {as_of}：動能資料不足，跳過")
            continue
        hot = set(hot_segment(ret_df.loc[q_idx.max()], X_HOT))
        holdings, covers, avg_frac = resolve_holdings_and_coverage(md, idx, uids, as_of, hot)

        quarter_row = {"as_of": as_of, "end": end, "avg_coverage_frac": avg_frac, "n_hot": len(hot)}
        for b in BETAS:
            wt = tilt_weights(holdings, covers, uids, b)
            res = measure(md_map, wt, as_of, end)
            per_beta_rets[b].append(res["portfolio_realized_return"])
            quarter_row[f"port_ret_beta{b}"] = res["portfolio_realized_return"]
            if b == BETAS[0]:
                ew = res["by_market_benchmark"]["TW"]["equal_weight_benchmark_return"]
                if ew is None:
                    raise ValueError(f"{as_of}~{end}：TW equal_weight_benchmark_return是None")
                ew_rets.append(ew)
                avg_frac_log.append(avg_frac)
                quarter_row["ew_benchmark_ret"] = ew
        per_quarter_rows.append(quarter_row)
        print(f"  {as_of}：覆蓋比例={avg_frac:.1%}｜hot={len(hot)}檔")

    pd.DataFrame(per_quarter_rows).to_csv(
        Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
        / "coverage_tilt_window4_per_quarter.csv", index=False, encoding="utf-8-sig")

    n_years = len(ew_rets) / 4.0
    cum_ew = pd.Series([1 + r for r in ew_rets]).prod() - 1
    ann_ew = (1 + cum_ew) ** (1 / n_years) - 1
    avg_cov = sum(avg_frac_log) / len(avg_frac_log) if avg_frac_log else float("nan")
    print(f"\nwindow4整體：策略平均覆蓋比例={avg_cov:.1%}（對照window1-3的3.7~4.5%）")

    rows = []
    for b in BETAS:
        rets = per_beta_rets[b]
        cum = pd.Series([1 + r for r in rets]).prod() - 1
        ann = (1 + cum) ** (1 / n_years) - 1
        print(f"window4 β={b:.1f}: 年化={ann:+.2%}  (等權大盤={ann_ew:+.2%}  超額={ann-ann_ew:+.2%})")
        rows.append({"window_no": 4, "beta": b, "ann_return": ann,
                    "ann_ew_benchmark": ann_ew, "ann_excess": ann - ann_ew,
                    "avg_coverage_frac": avg_cov, "n_rep": len(uids)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {OUT_PATH}")

    print("\n=== window1-3（§1.6既有結果，對照用）vs window4（這次新測）===")
    prev_path = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "coverage_tilt_prevalidation.csv"
    if prev_path.exists():
        prev = pd.read_csv(prev_path)
        prev_summary = prev.groupby("beta")["ann_excess"].mean().reset_index()
        prev_summary.columns = ["beta", "window1_3_mean_excess"]
        w4 = df[["beta", "ann_excess"]].rename(columns={"ann_excess": "window4_excess"})
        merged = prev_summary.merge(w4, on="beta")
        print(merged.to_string(index=False))
    else:
        print(f"找不到 {prev_path}，跳過對照")


if __name__ == "__main__":
    main()
