# -*- coding: utf-8 -*-
"""2026-10-27：台美股正式production（4組L2 checkpoint：TW/US × baseline/
exclude_v1）的日頻peak-to-trough最大回撤（MDD），取代§四/§八/§九/§十二
目前用的季頻取樣（9個點：登記＋8季末）MDD——後者會把季中發生、季底前已經
回補的真實跌幅完全吃掉，查證過業界真實案例（月頻vs日頻可差5~8倍），季頻
又比月頻更稀疏，低估風險的問題更嚴重。

做法（不重跑LLM、不改變任何agent決策，純粹用已經決定好的設定算得更細）：
①從checkpoint的`config_used`欄位讀出每季真正生效的設定
  （allocation／w2c_applied／coverage_tilt_applied／rep_swap_applied）
②用`simulate.resolve_portfolio_weights()`重算每季`weights_as_of`
  （checkpoint只存了`weights_end`，沒存`weights_as_of`，但這是純決定性
  運算，餵同樣的config/apply_w2c旗標進去結果保證一致）
③持股在整季內視為不變（D25既有假設，`monitor.monthly_breakdown()`
  docstring也是同一個假設），用`fcv_core.MarketData`的日頻收盤價算這批
  固定持股逐日的NAV（買進持有，不中途調整），再把8季串接成一條連續
  日頻NAV曲線
④大盤端更簡單：`taiex_tr`/`sp500_tr`本來就是真日頻表（實測台股2024-2025
  有485筆交易日資料），現有圖表只是故意只取9個點，這裡改成全部日期都用

TW baseline用`simulate.run_simulation()`的「原生」候選池慣例
（`walkforward_members.parquet`，不是`build_rolling_inputs()`吃的picks
parquet），其餘三組（TW exclude_v1／US baseline／US exclude_v1）用
`build_rolling_inputs()`——跟各自checkpoint的production launcher一致，
混用會導致候選池/uids對不起來（Active Share分析踩過的同一個坑）。

用法（cwd必須是code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._daily_mdd_both_markets
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from app import monitor, simulate  # noqa: E402
from app._check_rolling_production_simulation import build_rolling_inputs  # noqa: E402

APPLAYER = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
OUT_DIR = APPLAYER / "report_charts_1027"

SERIES = {
    "TW_baseline": dict(
        checkpoint="app/_runs/simulate_formal_8q_control0_L2_openSec_boost_v4_cap30_m2_v2.jsonl",
        convention="native", market="TW", bench_table="taiex_tr",
    ),
    "TW_exclude_v1": dict(
        checkpoint="app/_runs/simulate_anchored_w4_openSec_boost_exclude_v1_cap30_hotseg_L2.jsonl",
        convention="rolling", market="TW", bench_table="taiex_tr",
        picks_path=APPLAYER / "anchored_w4_openSec_boost_exclude_v1.parquet",
    ),
    "US_baseline": dict(
        checkpoint="app/_runs/simulate_anchored_w4_openSec_balanced_baseline_cap30_hotseg_m1dfix_L2.jsonl",
        convention="rolling", market="US", bench_table="sp500_tr",
        picks_path=APPLAYER / "anchored_w4_openSec_balanced_baseline.parquet",
    ),
    "US_exclude_v1": dict(
        checkpoint="app/_runs/simulate_anchored_w4_openSec_balanced_exclude_v1_cap30_hotseg_m1dfix_L2.jsonl",
        convention="rolling", market="US", bench_table="sp500_tr",
        picks_path=APPLAYER / "anchored_w4_openSec_balanced_exclude_v1.parquet",
    ),
}


def build_native_tw_inputs() -> simulate.QuarterInputs:
    """複製`simulate.run_simulation()`原生（非build_rolling_inputs）的TW
    inputs建構邏輯（見simulate.py:437~504），只取resolve_portfolio_weights()
    需要的欄位，其餘optional欄位留None——這支腳本不跑agent/觸發判定，
    只重算持股權重。"""
    import pandas as pd
    from database import Database
    from fcv_core import MarketData
    from resolve_strategy_holdings import CANDIDATE_INDEX_PATH
    from app import triggers

    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    md = MarketData("TW")
    md_map = {"TW": md}
    db = Database("TW")
    conn = db.create_connection()
    mcap_wide = monitor.fetch_mcap_wide(conn, start="2007-01-01")

    m = pd.read_parquet(Path(__file__).resolve().parent.parent.parent
                        / "_analysis_outputs_robustness" / "walkforward_members.parquet")
    uids_by_allocation = {}
    for alloc in ("equal", "proportional"):
        key = dict(tree_key="TW", scheme="E", window_no=4, k_mode="silhouette_is",
                  ratio="legacy", allocation=alloc, group="A_hrp")
        sub = m.copy()
        for k, v in key.items():
            sub = sub[sub[k] == v]
        uids_by_allocation[alloc] = list(sub.iloc[0]["members"])

    cond = triggers.register_m1d(mcap_wide, simulate.REGISTRATION_DATE, market="TW")
    return simulate.QuarterInputs(md=md, md_map=md_map, idx=idx, mcap_wide=mcap_wide,
                                  uids_by_allocation=uids_by_allocation, m1d_cond=cond,
                                  model="unused", api_key="unused", market="TW")


def load_config_used(checkpoint_path: str) -> dict[str, dict]:
    out = {}
    with open(checkpoint_path, encoding="utf-8") as f:
        for line in f:
            cp = json.loads(line)
            if cp.get("arm") not in (None, "L2"):
                continue
            out[cp["quarter_end"]] = cp["config_used"]
    return out


def quarter_nav_multiplier(close: pd.DataFrame, weights: dict[str, float],
                           as_of: str, end: str) -> pd.Series:
    """這一季固定持股`weights`的逐日NAV倍數（以as_of當天=1.0起算，買進持有
    不中途調整，跟outcome_layer()算整季報酬同一個D25假設，只是算得更細）。
    缺資料用forward-fill補（常見的交易暫停等情況），不是重新分配給其他持股
    （跟`performance.measure()`「量不到就不裝作有」同一立場，只是這裡要維持
    連續日頻序列，forward-fill是處理單日缺值的標準做法，不是重新分配權重）。"""
    syms = [s for s in weights if s in close.columns]
    sub = close.loc[:end, syms].ffill()
    idx = sub.index
    start_dates = idx[idx <= pd.Timestamp(as_of)]
    if len(start_dates) == 0:
        raise ValueError(f"找不到 {as_of} 之前的交易日")
    d0 = start_dates.max()
    period = sub.loc[d0:end]
    w_vec = pd.Series({s: weights[s] for s in syms})
    # 🔴 踩到的pandas陷阱：`period / period.loc[d0]`（純用`/`運算子）對543欄
    # 全部算出NaN，`period.iloc[0].equals(period.loc[d0])`confirmed兩者內容
    # 完全相同——純粹是`/`沒有可靠預設成逐欄對齊，必須明確用`.div(...,
    # axis="columns")`，下面兩個除法都改成明確指定軸，不依賴預設行為。
    ratios = period.div(period.loc[d0], axis="columns")
    nav_mult = ratios.mul(w_vec, axis="columns").sum(axis=1)
    base = nav_mult.loc[d0]
    if base <= 0:
        raise ValueError(f"{as_of}~{end} 這季權重涵蓋度為0，無法計算")
    return nav_mult / base  # 正規化成d0=1.0（跟涵蓋率無關，只看相對變化）


def daily_series_for_series(name: str, cfg: dict) -> pd.DataFrame:
    print(f"\n=== {name} ===")
    if cfg["convention"] == "native":
        inputs = build_native_tw_inputs()
    else:
        inputs = build_rolling_inputs(model="unused", api_key="unused",
                                      picks_path=cfg["picks_path"], tree_key=cfg["market"],
                                      scheme="E", window_no=4)

    config_used = load_config_used(cfg["checkpoint"])
    close = inputs.md.data.get("price:close")

    periods = [(simulate.REGISTRATION_DATE, simulate.QUARTER_ENDS[0])]
    periods += list(zip(simulate.QUARTER_ENDS[:-1], simulate.QUARTER_ENDS[1:]))

    daily_nav = []
    running_nav = 1.0
    for as_of, end in periods:
        cu = config_used[end]
        config = simulate.ActiveConfig(allocation=cu["allocation"])
        w = simulate.resolve_portfolio_weights(
            inputs, config, as_of,
            apply_w2c=cu["w2c_applied"],
            apply_coverage_tilt=cu["coverage_tilt_applied"],
            apply_rep_swap=cu["rep_swap_applied"])
        mult = quarter_nav_multiplier(close, w, as_of, end)
        if daily_nav:
            mult = mult.iloc[1:]  # 去掉跟上一季末重複的d0那天
        q_nav = mult * running_nav
        daily_nav.append(q_nav)
        running_nav = float(q_nav.iloc[-1])
        print(f"  {as_of}~{end}  季末NAV累計={running_nav:.4f}  "
             f"w2c={cu['w2c_applied']}  涵蓋股數={len(w)}")

    nav = pd.concat(daily_nav).sort_index()
    nav = nav[~nav.index.duplicated(keep="last")]
    peak = nav.cummax()
    dd = (nav / peak - 1.0) * 100
    return pd.DataFrame({"nav": nav, "dd_pct": dd})


def benchmark_daily_series(bench_table: str, market: str) -> pd.DataFrame:
    import fcv_core  # noqa: F401
    from database import Database
    db = Database(market)
    conn = db.create_connection()
    df = pd.read_sql(f"SELECT date, close FROM {bench_table} "
                     f"WHERE date BETWEEN '{simulate.REGISTRATION_DATE}' AND '{simulate.QUARTER_ENDS[-1]}' "
                     f"ORDER BY date", conn)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")["close"].astype(float)
    nav = df / df.iloc[0]
    peak = nav.cummax()
    dd = (nav / peak - 1.0) * 100
    return pd.DataFrame({"nav": nav, "dd_pct": dd})


def main():
    rows = []
    all_series = {}
    for name, cfg in SERIES.items():
        df = daily_series_for_series(name, cfg)
        all_series[name] = df
        daily_mdd = df["dd_pct"].min()
        print(f"  >> {name} 日頻MDD = {daily_mdd:+.2f}%（{len(df)}個交易日）")
        rows.append(dict(series=name, daily_mdd_pct=daily_mdd, n_trading_days=len(df)))
        df.to_csv(OUT_DIR / f"daily_nav_{name}_2026-10-27.csv", encoding="utf-8-sig")

    bench_done = set()
    for name, cfg in SERIES.items():
        key = cfg["bench_table"]
        if key in bench_done:
            continue
        bench_done.add(key)
        bdf = benchmark_daily_series(cfg["bench_table"], cfg["market"])
        daily_mdd = bdf["dd_pct"].min()
        label = f"benchmark_{cfg['market']}"
        print(f"  >> {label}（{cfg['bench_table']}）日頻MDD = {daily_mdd:+.2f}%（{len(bdf)}個交易日）")
        rows.append(dict(series=label, daily_mdd_pct=daily_mdd, n_trading_days=len(bdf)))
        bdf.to_csv(OUT_DIR / f"daily_nav_{label}_2026-10-27.csv", encoding="utf-8-sig")

    summary = pd.DataFrame(rows)
    summary.to_csv(APPLAYER / "daily_mdd_summary_2026-10-27.csv", index=False, encoding="utf-8-sig")
    print("\n", summary.to_string(index=False))
    print(f"\n寫入 {APPLAYER / 'daily_mdd_summary_2026-10-27.csv'}")


if __name__ == "__main__":
    main()
