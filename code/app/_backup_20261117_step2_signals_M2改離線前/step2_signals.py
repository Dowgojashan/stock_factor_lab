# -*- coding: utf-8 -*-
"""步驟2 版本2 的季初訊號（實戰收尾追蹤.md「三種訊號配對」，2026-10-28 使用者確認）。

window4 每季季初（as_of）只用 as_of 以前的資料算：
  ① 集中度警示：`rule_selector.m1d_series`（美股用修正後資料＋凍結門檻）
  ② 熱門股覆蓋：`monitor.hot_segment_layer` 的市場層級覆蓋率＋`triggers.evaluate_hot_segment_quarter`
     既有門檻與遲滯（台美都用目前 DB 資料重算，不讀舊 checkpoint）
  ③ 突破廣度：持股（control0）中收盤價創近 20 個交易日新高的比例；門檻只用 window1~3 校準
     （> p90＝「高」、> p75＝「偏高」）；另給「大盤是否創 20 日新高」
  ④ 回撤（只當資訊）：control0 淨值距 2015 年以來高點的回撤，及其在 window1~3 季末回撤中的百分位
  ⑤ 策略群集中度 M2（只當資訊）：取自正式 checkpoint（只依代表策略的分群，不受價格修正影響）
結果快取 `step2/signals_{market}_{pool}.csv`。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from app import rule_selector as RS

CACHE = RS.ROOT / "_analysis_outputs_applayer" / "step2" / "signals_{market}_{pool}.csv"


def breakout_share(close: pd.DataFrame, syms: list[str], as_of: str, window: int = 20) -> float:
    """持股中創近 20 日新高的比例（分母＝期間內價格有變動的持股；與 S8 共用 `elite_rules.breakout_hits`）。"""
    from app import elite_rules as ER
    hit, n_ok = ER.breakout_hits(close, syms, as_of, window)
    return len(hit) / n_ok if n_ok else float("nan")


def index_at_high(bench: pd.Series, as_of: str, window: int = 20) -> bool:
    b = bench[bench.index <= pd.Timestamp(as_of)].iloc[-window:]
    return bool(b.iloc[-1] >= 0.999 * b.max())


def compute(market: str, pool: str) -> pd.DataFrame:
    path = __import__("pathlib").Path(str(CACHE).format(market=market, pool=pool))
    tag = RS.TAG[(market, pool)]
    inputs = [RS.GRID / f"w0_{tag}.parquet"] + [RS.GRID / f"daily_gross_{tag}_w{w}.parquet" for w in (1, 2, 3, 4)]
    # 2026-10-28 code review：快取比任何輸入檔舊就重算（避免步驟1 重跑後沿用過期訊號；DB 端的變動仍需手動刪快取）
    if path.exists() and all(path.stat().st_mtime > f.stat().st_mtime for f in inputs):
        return pd.read_csv(path, index_col=0)
    from database import Database
    from fcv_core import MarketData
    from resolve_strategy_holdings import CANDIDATE_INDEX_PATH
    from app import monitor, simulate, triggers
    from app._elite_grid_offline import WF_MEMBERS, load_benchmark, load_window

    md = MarketData(market); close = md.data.get("price:close")
    w0 = pd.read_parquet(RS.GRID / f"w0_{tag}.parquet")
    bench = load_benchmark(market)
    as_ofs = [simulate.REGISTRATION_DATE] + list(simulate.QUARTER_ENDS[:-1])

    # ③ 突破廣度：window1~3 校準
    hist = [breakout_share(close, list(d.symbol), a) for (w, a), d in w0[w0.window < 4].groupby(["window", "as_of"])]
    p75, p90 = float(np.percentile(hist, 75)), float(np.percentile(hist, 90))

    # ④ 回撤：control0 日淨值（window1~4 接起來）
    lv = []
    for w in (1, 2, 3, 4):
        s = pd.read_parquet(RS.GRID / f"daily_gross_{tag}_w{w}.parquet")["REF_control0"]
        lv.append(s * (lv[-1].iloc[-1] if lv else 1.0))
    nav = pd.concat(lv)
    dd = nav.div(nav.cummax()).sub(1.0)
    q_ends_hist = sorted(w0[w0.window < 4].as_of.unique())[1:]
    dd_hist = np.array([float(dd[dd.index <= pd.Timestamp(a)].iloc[-1]) for a in q_ends_hist])

    # ② 熱門股覆蓋
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    uids = load_window(market, pool, 4, pd.read_parquet(WF_MEMBERS))["uids"]
    mw = monitor.fetch_mcap_wide(Database(market).create_connection(), market=market, start="2022-01-01")
    cond = triggers.register_hot_segment(market, pool)

    m1d = RS.m1d_series(market, pool)
    m2 = {}
    for line in open(RS.RUNS / RS.CKPT[(market, pool)], encoding="utf-8"):
        cp = json.loads(line)
        if cp.get("arm") in (None, "L2"):
            m2[cp["quarter_end"]] = cp["m2"]

    rows, hs_prev = [], "NONE"
    for a in as_ofs:
        hs = monitor.hot_segment_layer(md, idx, uids, monitor.asof_row(mw, a), a, market)
        ev = triggers.evaluate_hot_segment_quarter(cond, hs["coverage_count"], hs_prev)
        hs_prev = ev["state"]
        syms = list(w0[(w0.window == 4) & (w0.as_of == a)].symbol)
        bs = breakout_share(close, syms, a)
        cur_dd = float(dd[dd.index <= pd.Timestamp(a)].iloc[-1])
        first = a == simulate.REGISTRATION_DATE
        rows.append({
            "as_of": a,
            "m1d_state": "NONE" if first else m1d.loc[a, "state"],
            "m1d_dev_pp": 0.0 if first else float(m1d.loc[a, "cumulative_deviation"]) * 100,
            "hot_state": ev["state"], "hot_coverage": hs["coverage_count"], "hot_p25": cond.p25, "hot_p10": cond.p10,
            "breakout_share": bs, "breakout_p75": p75, "breakout_p90": p90,
            "breakout_state": "高" if bs > p90 else ("偏高" if bs > p75 else "正常"),
            "index_at_20d_high": index_at_high(bench, a),
            "drawdown": cur_dd, "drawdown_pctile": float((dd_hist <= cur_dd).mean()),
            "m2_state": "NONE" if first else m2[a]["state"],
            "m2_ratio": np.nan if first else m2[a]["ratio_to_ideal"],
        })
    out = pd.DataFrame(rows).set_index("as_of")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, encoding="utf-8-sig")
    return out
