# -*- coding: utf-8 -*-
"""2026-10-27（補強）：W2c單股上限敏感度分析的逐季版本——`_cap_sweep_window4.py`
只存了8季累積cum_pct跟最終mdd_pct，這支腳本重跑同一組反事實（每季強制套用
W2c市值加權，cap=[7/10/15/20/30/40]%，不呼叫LLM，exclude_v1候選池），但把
**逐季報酬率**跟**逐季回撤軌跡**（9個季末點的季頻取樣NAV drawdown，跟
`_make_report_charts_1027.py`同一套方法論）都存下來，並加入大盤（TAIEX／
S&P500）同一組日期的逐季報酬＋drawdown軌跡做對照。

計算方式跟`_cap_sweep_window4.py`完全相同（同一份picks、同一個市場分流
tilt_weights/tilt_weights_basket+cap_and_redistribute呼叫），只是多存
中間序列，不是重新設計方法論。

🔴 **2026-10-27二次修正**：原本沒有依`W2C_VARIANT_BY_MARKET`分流，美股
錯用了台股的單檔傾斜，已同步`_cap_sweep_window4.py`的修正改用籃子版（K=3）。

用法（cwd必須是code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._cap_sweep_window4_quarterly
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from app import monitor, simulate  # noqa: E402
from app import weights as weights_mod  # noqa: E402
from app._check_rolling_production_simulation import build_rolling_inputs  # noqa: E402
from app._make_report_charts_1027 import Q_LABELS, benchmark_quarterly_returns  # noqa: E402

APPLAYER = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
CAPS = [0.07, 0.10, 0.15, 0.20, 0.30, 0.40]

MARKETS = {
    "TW": (APPLAYER / "anchored_w4_openSec_boost_exclude_v1.parquet", "taiex_tr"),
    "US": (APPLAYER / "anchored_w4_openSec_balanced_exclude_v1.parquet", "sp500_tr"),
}


def nav_and_drawdown(rets: list[float]) -> tuple[list[float], list[float]]:
    nav = [1.0]
    for r in rets:
        nav.append(nav[-1] * (1 + r))
    peak = nav[0]
    dd = []
    for v in nav:
        peak = max(peak, v)
        dd.append((v / peak - 1) * 100)
    return nav, dd


def series_rows(market: str, series_name: str, rets: list[float]) -> list[dict]:
    nav, dd = nav_and_drawdown(rets)
    rows = []
    for i, label in enumerate(["登記"] + Q_LABELS):
        rows.append(dict(
            market=market, series=series_name, quarter_label=label,
            ret_pct=(rets[i - 1] * 100) if i > 0 else None,
            nav=nav[i], dd_pct=dd[i],
        ))
    return rows


def sweep_market(tree_key: str, picks_path: Path, bench_table: str) -> list[dict]:
    print(f"\n=== {tree_key}：載入window4 exclude_v1候選池 ===")
    inputs = build_rolling_inputs(model="unused", api_key="unused",
                                  picks_path=picks_path, tree_key=tree_key,
                                  scheme="E", window_no=4)
    uids = inputs.uids_by_allocation["equal"]

    periods = [(simulate.REGISTRATION_DATE, simulate.QUARTER_ENDS[0])]
    periods += list(zip(simulate.QUARTER_ENDS[:-1], simulate.QUARTER_ENDS[1:]))

    all_rows = []
    for cap in CAPS:
        rets = []
        for as_of, end in periods:
            w0 = simulate.resolve_weights(inputs.md, inputs.idx, uids, as_of)
            mcap_row = monitor.asof_row(inputs.mcap_wide, as_of)
            if weights_mod.W2C_VARIANT_BY_MARKET.get(tree_key) == "basket":
                raw = weights_mod.tilt_weights_basket(w0, mcap_row, 0.0,
                                                      basket_k=weights_mod.W2C_BASKET_K)
            else:
                raw = weights_mod.tilt_weights(w0, mcap_row, 0.0)
            w_capped = weights_mod.cap_and_redistribute(raw, cap=cap)
            r = monitor.outcome_layer(inputs.md_map, w_capped, as_of, end)["portfolio_realized_return"]
            rets.append(r)
        all_rows += series_rows(tree_key, f"{cap:.0%}", rets)
        print(f"  cap={cap:.0%}  累積={ (pd.Series(rets)+1).prod()*100-100:+.2f}%")

    bench_rets = benchmark_quarterly_returns(bench_table)
    all_rows += series_rows(tree_key, "benchmark", bench_rets)
    return all_rows


def main():
    all_rows = []
    for tree_key, (picks_path, bench_table) in MARKETS.items():
        all_rows += sweep_market(tree_key, picks_path, bench_table)

    out = pd.DataFrame(all_rows)
    out_path = APPLAYER / "cap_sweep_window4_quarterly_2026-10-27.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}")


if __name__ == "__main__":
    main()
