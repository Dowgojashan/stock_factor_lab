# -*- coding: utf-8 -*-
"""優等生權重規則離線網格（實戰收尾追蹤.md §四 步驟1，2026-10-27定案，不呼叫LLM）。

網格（每組市場×候選池 1,040 組）：
  排名依據13種 × 取幾名/怎麼分16種 × 優等生資金比例5種
  另加固定對照：control0（現行）、每檔等權、現行市值傾斜全開（30%上限，依市場分流變體）

規則實作全部在 `app.elite_rules`（正式模組，步驟2的 simulate.py 共用）；本腳本只負責
組資料、呼叫規則、算報酬。

時間慣例跟正式模擬一致（simulate.py:255 的決策/套用延遲精神）：季末 as_of 用當時已知
資料算分數與權重 → 套用到下一整季、季內買進持有。window4 的日曆＝simulate.REGISTRATION_DATE
＋QUARTER_ENDS；window1~3 依各窗 is_end/oos_end 用同一規則產生。

報酬：季內逐日 NAV＝現金 + Σ w_i × p_i,t/p_i,0（價格已 ffill，get_data.py:124；季初無價
的股票視同現金，跟 performance.measure() 一致）。交易成本：季初從前一季漂移後權重調到
新權重，買賣各自計費——台股 買0.1425%／賣0.1425%+證交稅0.3%（同 _conditional_w2_netcost.py），
美股 單邊10bps（2026-10-27使用者定案的文獻常用假設）。窗與窗之間延續前一窗的持股
（2026-10-28 裁示；只有第一個窗從現金建倉），所以 window1~4 必須依序一起跑。

壞資料防線（2026-10-27定案④）：①季初 as_of 收盤價 > SANITY_MAX_CLOSE 或近6個月報酬
> +1000% 的股票，不得被選為優等生（仍保留在 control0 的基礎權重裡），清單存檔；
美股另加近6個月報酬 < −90%（2026-10-28 裁示，台股不加）
②季內任何持股單季報酬 > +300% 或單日漲跌幅 > 50%，只標記存檔供人工覆核，不改結果
（用未來報酬剔除股票本身就是前視）。

用法（cwd=code/）：
  PYTHONIOENCODING=utf-8 ../.venv/Scripts/python.exe -m app._elite_grid_offline --market TW --pool exclude_v1
  加 --windows 4 --smoke 只跑 window4 與少數組合做對帳
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import pandas as pd  # noqa: E402

from app import elite_rules as ER  # noqa: E402
from app import monitor, simulate  # noqa: E402
from app import weights as weights_mod  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
APPLAYER = ROOT / "_analysis_outputs_applayer"
OUT_DIR = APPLAYER / "elite_grid_1027"
WF_MEMBERS = ROOT / "_analysis_outputs_robustness" / "walkforward_members.parquet"

SLEEVES = (0.10, 0.20, 0.30, 0.50, 1.00)
NS = (1, 3, 5, 10)
LOOKBACKS = (1, 3, 6)
COSTS = {"TW": (1.425e-3, 1.425e-3 + 3e-3), "US": (1e-3, 1e-3)}   # (買, 賣) 每單位成交金額
SANITY_TRAIL_MAX = 10.0      # 近6個月報酬 > +1000% 不得當優等生
#: 2026-10-28 使用者裁示：美股另加對稱下限（近6個月跌 > 90% 不得當優等生），台股不加。
#: 理由：collector 修正後美股仍有 407 筆無法判定（D 類，如停滯報價）的跳價，押回彈會優先選到它們。
SANITY_TRAIL_MIN = {"US": -0.90, "TW": None}
#: 2026-10-28 使用者裁示（跑之前寫死，見實戰收尾追蹤.md「美股流動性門檻」）：加 --liq 時，季末市值在
#: 宇宙中排名後 20% 的股票（或市值缺值）不得當優等生；只套美股。輸出檔名加 `_liq`，保留未加門檻的版本。
LIQ_MCAP_PCTILE_MIN = {"US": 0.20, "TW": None}
#: 2026-10-28 使用者裁示 (a)：季初前 30 天以上沒有成交價的股票（已下市／長期停牌），不得當優等生；
#: control0 不動。起因：台股 window4 每季平均 13 檔已下市股（如 3383、6289 最後交易日 2022-09-23）
#: 因財報與股價被沿用最後值而持續被策略選中，且票數名列前茅。判斷用 DB 原始收盤價（價格資料已 ffill，看不出停牌）。
TRADABLE_LOOKBACK_DAYS = 30


def tradable_symbols(conn, market: str, as_of: str) -> set[str]:
    """季初前 TRADABLE_LOOKBACK_DAYS 天內（含 as_of）DB 至少有一天「有收盤價且成交量>0」的股票代號。
    🔴 2026-10-28 code review：台股停牌股每天仍有收盤價、成交量為 0（例 2358／2443／4950 在 2024-06），
    只看收盤價會漏掉；美股 volume 欄位完整（NULL 0 筆），可用同一條件。"""
    from database import Database
    db = Database(market)
    end = pd.Timestamp(as_of); start = end - pd.Timedelta(days=TRADABLE_LOOKBACK_DAYS)
    q = (f"SELECT DISTINCT c.company_symbol FROM stock s JOIN company c ON s.company_id=c.id "
         f"WHERE c.{db._exchange_in_clause()}{db._universe_clause('c.company_symbol')} "
         f"AND s.close IS NOT NULL AND s.volume > 0 AND s.date > '{start.date()}' AND s.date <= '{end.date()}'")
    return set(pd.read_sql(q, conn).iloc[:, 0].astype(str))
FLAG_Q_RET = 3.0             # 季報酬 > +300% 標記
FLAG_DAY_MOVE = 0.5          # 單日漲跌 > 50% 標記

POOL_FILES = {  # (market, pool) → picks 檔名樣板；None＝讀 walkforward_members
    ("TW", "baseline"): None,
    ("TW", "exclude_v1"): "anchored_w{w}_openSec_boost_exclude_v1.parquet",
    ("US", "baseline"): None,
    ("US", "exclude_v1"): "anchored_w{w}_openSec_balanced_exclude_v1.parquet",
}


# ───────────────────────── 組合定義 ─────────────────────────

def score_keys() -> list[str]:
    keys = ["S1_votes", "S2_f1", "S2_f1f2"]
    for kind in ("S3_greedy", "S4_reversal", "S5_mixed"):
        keys += [f"{kind}_L{L}" for L in LOOKBACKS]
    keys.append("S6_mcap")
    return keys                                           # 13


#: 2026-10-28 新增（步驟2 三種訊號配對用）：用 `--scores S7_hot,S8_breakout --tag-suffix _sig` 另外跑，
#: 不改動原本 13 種排名依據的步驟1 結果
EXTRA_SCORES = ("S7_hot", "S8_breakout")


def ns_schemes() -> list[tuple[int, str]]:
    out = []
    for n in NS:
        for sc in ER.SCHEMES:
            if not ER.is_degenerate(n, sc):
                out.append((n, sc))
    return out                                            # 16


def all_combos(smoke: bool, scores: list[str] | None = None) -> list[dict]:
    combos = []
    sk, nsc, slv = (scores or score_keys()), ns_schemes(), SLEEVES
    if smoke:
        sk, nsc, slv = (scores or ["S1_votes", "S3_greedy_L3", "S6_mcap"]), [(1, "equal"), (5, "proportional")], (0.3, 1.0)
    for s, (n, sc), sl in itertools.product(sk, nsc, slv):
        combos.append({"combo": f"{s}|N{n}|{sc}|sl{int(round(sl * 100))}",
                       "score": s, "n": n, "scheme": sc, "sleeve": sl})
    for ref in ("REF_control0", "REF_stock_equal", "REF_w2c_full"):
        combos.append({"combo": ref, "score": ref, "n": 0, "scheme": "", "sleeve": 0.0})
    return combos


# ───────────────────────── 資料 ─────────────────────────

def load_window(market: str, pool: str, w: int, wf: pd.DataFrame) -> dict:
    tmpl = POOL_FILES[(market, pool)]
    if tmpl is None:
        sub = wf[(wf.tree_key == market) & (wf.scheme == "E") & (wf.k_mode == "silhouette_is")
                 & (wf.ratio == "legacy") & (wf.group == "A_hrp") & (wf.allocation == "equal")
                 & (wf.window_no == w)]
    else:
        df = pd.read_parquet(APPLAYER / tmpl.format(w=w))
        sub = df[df.allocation == "equal"]
    if len(sub) != 1:
        raise AssertionError(f"{market}/{pool}/w{w} 找不到唯一名單列（{len(sub)}）")
    r = sub.iloc[0]
    return {"uids": list(r.members), "is_end": str(r.is_end), "oos_end": str(r.oos_end)}


def quarter_calendar(is_end: str, oos_end: str) -> list[tuple[str, str]]:
    start = pd.Period(is_end, "M").to_timestamp(how="end").normalize()
    stop = pd.Period(oos_end, "M").to_timestamp(how="end").normalize()
    dates = [start]
    while dates[-1] < stop:
        dates.append((dates[-1] + pd.offsets.QuarterEnd(1)).normalize())
    if dates[-1] != stop:
        raise AssertionError(f"OOS 結束 {oos_end} 不在季末")
    ds = [d.strftime("%Y-%m-%d") for d in dates]
    return list(zip(ds[:-1], ds[1:]))


def load_benchmark(market: str) -> pd.Series:
    from database import Database
    conn = Database("TW").create_connection()      # taiex_tr／sp500_tr 都在同一個庫（同 _compute_mdd_production_runs.py）
    table = {"TW": "taiex_tr", "US": "sp500_tr"}[market]
    b = pd.read_sql(f"SELECT date, close FROM {table} ORDER BY date", conn)
    b["date"] = pd.to_datetime(b["date"])
    return b.set_index("date")["close"].astype(float)


# ───────────────────────── 每季 ─────────────────────────

def quarter_weights(md, idx, close, mcap_wide, market, uids, as_of, combos, log_rows, liq=False,
                    tradable: set[str] | None = None, monthly=None):
    """回傳 (syms, W[combo×syms], w0, elite_by_key, excluded)。"""
    from resolve_strategy_holdings import resolve_holdings
    w0 = simulate.resolve_weights(md, idx, uids, as_of)       # 與正式模擬同一支函式
    holdings, n_fail = {}, 0
    for u in uids:
        try:
            syms, _ = resolve_holdings(md, idx.loc[u], as_of)
        except RuntimeError:
            n_fail += 1
            continue
        if syms:
            holdings[u] = syms
    syms = sorted(w0)
    ts = pd.Timestamp(as_of)
    d1 = close.index[close.index <= ts].max()

    # 壞資料防線①（只用 as_of 以前的資料）
    px = close.loc[d1].reindex(syms)
    trail6 = ER.trailing_return(close, syms, ts, 6)
    bad = set(px[px > monitor.SANITY_MAX_CLOSE].index) | set(trail6[trail6 > SANITY_TRAIL_MAX].index)
    if SANITY_TRAIL_MIN.get(market) is not None:
        bad |= set(trail6[trail6 < SANITY_TRAIL_MIN[market]].index)
    reason = {s: "壞資料防線" for s in bad}
    if tradable is not None:
        share = len(set(syms) & tradable) / len(syms)
        if share < 0.8:   # 防線：查詢失敗或代號格式對不上時，會把全部持股判成不可交易、所有規則靜默退回 control0
            raise AssertionError(f"{as_of} 只有 {share:.0%} 的持股判為可交易，疑似 tradable_symbols 查詢有誤")
        for s in syms:
            if s not in reason and s not in tradable:
                reason[s] = f"停止交易（季初前{TRADABLE_LOOKBACK_DAYS}天無成交價）"
    mcap_row = monitor.asof_row(mcap_wide, as_of)
    if liq and LIQ_MCAP_PCTILE_MIN.get(market) is not None:
        mc = mcap_row[mcap_row.notna() & (mcap_row > 0)]
        pct = mc.rank(pct=True)                                   # 宇宙＝fetch_mcap_wide 的市場宇宙（美股＝Russell 3000）
        for s in syms:
            if s not in reason and (s not in pct.index or pct[s] < LIQ_MCAP_PCTILE_MIN[market]):
                reason[s] = "流動性門檻（市值後20%或缺值）"
    for s in sorted(reason):
        log_rows.append({"as_of": as_of, "symbol": s, "close": float(px.get(s, np.nan)),
                         "trail6m": float(trail6.get(s, np.nan)), "reason": reason[s]})
    elig = [s for s in syms if s not in reason]

    strat_rows = {u: idx.loc[u].to_dict() for u in holdings}
    pct_cache: dict[str, pd.Series | None] = {}

    def pct_at(factor):
        if factor not in pct_cache:
            f = md.get_field(f"report:{str(factor).lower()}")
            pct_cache[factor] = None if f is None else f.loc[d1].rank(pct=True)
        return pct_cache[factor]

    elig_hold = {u: [s for s in v if s in elig] for u, v in holdings.items()}
    rets = {L: ER.trailing_return(close, elig, ts, L) for L in LOOKBACKS}
    scores = {
        "S1_votes": ER.score_votes(elig_hold),
        "S2_f1": ER.score_factor(elig_hold, strat_rows, pct_at, 1.0),
        "S2_f1f2": ER.score_factor(elig_hold, strat_rows, pct_at, 0.5),
        "S6_mcap": ER.score_mcap(mcap_row, elig),
    }
    for L in LOOKBACKS:
        scores[f"S3_greedy_L{L}"] = ER.score_greedy(rets[L])
        scores[f"S4_reversal_L{L}"] = ER.score_reversal(rets[L])
    if any(c["score"] == "S7_hot" for c in combos):
        mom = monitor.momentum_asof(monthly, as_of, monitor.HOT_SEGMENT_N[market])
        scores["S7_hot"] = ER.score_hot(mom, monitor.hot_segment(mom, monitor.HOT_SEGMENT_X), elig)
    if any(c["score"] == "S8_breakout" for c in combos):
        scores["S8_breakout"] = ER.score_breakout(close, elig, ts)

    elite_by_key: dict[tuple, dict] = {}
    for c in combos:
        if c["score"].startswith("REF_"):
            continue
        key = (c["score"], c["n"], c["scheme"])
        if key in elite_by_key:
            continue
        if c["score"].startswith("S5_mixed"):
            L = int(c["score"].rsplit("_L", 1)[1])
            elite_by_key[key] = ER.mixed_elite_weights(scores[f"S3_greedy_L{L}"],
                                                       scores[f"S4_reversal_L{L}"], c["n"], c["scheme"])
        else:
            elite_by_key[key] = ER.elite_weights(scores[c["score"]], c["n"], c["scheme"])

    pos = {s: i for i, s in enumerate(syms)}
    W = np.zeros((len(combos), len(syms)))
    w0v = np.array([w0[s] for s in syms])
    for i, c in enumerate(combos):
        if c["score"] == "REF_control0":
            W[i] = w0v
        elif c["score"] == "REF_stock_equal":
            W[i] = w0v.sum() / len(syms)
        elif c["score"] == "REF_w2c_full":
            ww = weights_mod.apply_w2c_for_market(w0, mcap_row, market)
            for s, v in ww.items():
                W[i, pos[s]] = v * w0v.sum()        # apply_w2c 回傳總和1；保留 w0 的現金比例
        else:
            final = ER.overlay(w0, elite_by_key[(c["score"], c["n"], c["scheme"])], c["sleeve"])
            for s, v in final.items():
                W[i, pos[s]] = v
    return syms, W, w0, elite_by_key, n_fail


def price_ratios(close, syms, as_of, end):
    sub = close.reindex(columns=syms)
    idx = sub.index
    d0 = idx[idx <= pd.Timestamp(as_of)].max()
    period = sub.loc[d0:pd.Timestamp(end)]
    base = period.loc[d0]
    R = period.div(base, axis="columns")
    # 季初無價＝現金（同 performance.measure）。🔴 2026-10-27 踩到：`R.where(Series, 1.0)`
    # 會沿列(index)對齊而不是沿欄，條件全部變 False、整張表被換成 1.0，所有報酬靜默變 0%
    # ——改成明確用欄位遮罩。
    no_price = ~(base.notna() & (base > 0))
    R.loc[:, no_price.to_numpy()] = 1.0
    return R.fillna(1.0)


# ───────────────────────── 主流程 ─────────────────────────

def run(market: str, pool: str, windows: list[int], smoke: bool, liq: bool = False,
        scores: list[str] | None = None, tag_suffix: str = "") -> None:
    from database import Database
    from fcv_core import MarketData
    from resolve_strategy_holdings import CANDIDATE_INDEX_PATH

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if liq and LIQ_MCAP_PCTILE_MIN.get(market) is None:
        raise ValueError(f"{market} 沒有定義流動性門檻（只套美股）")
    tag = f"{market}_{pool}" + ("_liq" if liq else "") + tag_suffix + ("_smoke" if smoke else "")
    combos = all_combos(smoke, scores)
    cnames = [c["combo"] for c in combos]
    print(f"[{tag}] 組合數 {len(combos)}（含3個對照）｜窗次 {windows}")

    wf = pd.read_parquet(WF_MEMBERS)
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    md = MarketData(market)
    close = md.data.get("price:close")
    conn = Database(market).create_connection()
    mcap_wide = monitor.fetch_mcap_wide(conn, market=market, start="2007-01-01")
    bench = load_benchmark(market)
    buy_c, sell_c = COSTS[market]
    monthly = monitor.monthly_close(md) if (scores and "S7_hot" in scores) else None

    q_rows, w0_rows, elite_rows, excl_rows, flag_rows, daily_parts, meta = [], [], [], [], [], [], {}
    # 2026-10-28 使用者裁示：窗與窗之間延續前一窗的持股（不再每窗從現金重新建倉），
    # 第一個窗才從現金開始；成本＝從前一窗期末漂移後權重調到新窗第一季目標的換手。
    prev_end_w = None                                  # DataFrame combos×syms（前季漂移後權重）
    for w in windows:
        win = load_window(market, pool, w, wf)
        cal = quarter_calendar(win["is_end"], win["oos_end"])
        meta[w] = {"n_strategies": len(win["uids"]), "calendar": cal}
        if w == 4 and [e for _, e in cal] != list(simulate.QUARTER_ENDS):
            raise AssertionError(f"window4 日曆跟 simulate.QUARTER_ENDS 不一致：{cal}")
        print(f"\n== window{w}：{len(win['uids'])} 個代表策略，{len(cal)} 季 {cal[0][0]}→{cal[-1][1]} ==")
        level_g = np.ones(len(combos))
        level_n = np.ones(len(combos))
        for qi, (as_of, end) in enumerate(cal):
            syms, W, w0, elite_by_key, n_fail = quarter_weights(
                md, idx, close, mcap_wide, market, win["uids"], as_of, combos, excl_rows, liq=liq,
                tradable=tradable_symbols(conn, market, as_of), monthly=monthly)
            R = price_ratios(close, syms, as_of, end)
            if float((R.iloc[-1] - 1.0).abs().sum()) == 0.0:
                raise AssertionError(f"{as_of}→{end} 全部持股季報酬都是0，價格比值表疑似壞掉（靜默錯誤防線）")
            cash = 1.0 - W.sum(axis=1)
            nav = cash[None, :] + R.to_numpy() @ W.T          # days × combos，d0=1

            # 交易成本：從前季漂移後權重調到這季目標
            Wdf = pd.DataFrame(W, columns=syms)
            if prev_end_w is None:
                old = pd.DataFrame(0.0, index=range(len(combos)), columns=syms)
            else:
                old = prev_end_w.reindex(columns=sorted(set(syms) | set(prev_end_w.columns)), fill_value=0.0)
                Wdf = Wdf.reindex(columns=old.columns, fill_value=0.0)
            diff = Wdf.to_numpy() - old.to_numpy()
            buys = np.clip(diff, 0, None).sum(axis=1)
            sells = np.clip(-diff, 0, None).sum(axis=1)
            cost = buys * buy_c + sells * sell_c

            q_ret = nav[-1] - 1.0
            days = R.index[1:]
            g_path = level_g[None, :] * nav[1:]
            n_path = (level_n * (1.0 - cost))[None, :] * nav[1:]
            daily_parts.append((w, pd.DataFrame(g_path, index=days, columns=cnames),
                                pd.DataFrame(n_path, index=days, columns=cnames)))
            level_g = level_g * nav[-1]
            level_n = level_n * (1.0 - cost) * nav[-1]

            end_vals = W * R.to_numpy()[-1][None, :]
            prev_end_w = pd.DataFrame(end_vals / nav[-1][:, None], columns=syms)

            tot = W.sum(axis=1)
            maxw = W.max(axis=1) / np.where(tot > 0, tot, np.nan)
            nhold = (W > 0).sum(axis=1)
            for i, c in enumerate(combos):
                q_rows.append({"window": w, "q": qi + 1, "as_of": as_of, "end": end, "combo": c["combo"],
                               "ret_gross": q_ret[i], "cost": cost[i], "turnover": (buys[i] + sells[i]) / 2,
                               "max_weight": maxw[i], "n_hold": int(nhold[i])})
            for s, v in w0.items():
                w0_rows.append({"window": w, "as_of": as_of, "symbol": s, "w0": v})
            for (sk, n, sc), ew in elite_by_key.items():
                for s, v in ew.items():
                    elite_rows.append({"window": w, "as_of": as_of, "score": sk, "n": n, "scheme": sc,
                                       "symbol": s, "elite_weight": v})

            # 壞資料防線②：只標記
            qr = R.iloc[-1] - 1.0
            dmove = R.pct_change().abs().max()
            held_max = pd.Series(W.max(axis=0), index=syms)
            for s in syms:
                if qr[s] > FLAG_Q_RET or dmove[s] > FLAG_DAY_MOVE:
                    flag_rows.append({"window": w, "as_of": as_of, "symbol": s, "q_ret": float(qr[s]),
                                      "max_daily_move": float(dmove[s]), "max_weight_any_combo": float(held_max[s])})
            print(f"   {as_of}→{end}：持股{len(syms)}檔（解析失敗策略{n_fail}）｜control0 {q_ret[cnames.index('REF_control0')]:+.2%}"
                  f"｜耗時{time.time()-t0:.0f}s", flush=True)

        # 大盤
        bsub = bench[(bench.index >= bench.index[bench.index <= pd.Timestamp(cal[0][0])].max())
                     & (bench.index <= pd.Timestamp(cal[-1][1]))]
        meta[w]["bench_cum"] = float(bsub.iloc[-1] / bsub.iloc[0] - 1.0)

    # 存檔
    pd.DataFrame(q_rows).to_parquet(OUT_DIR / f"quarterly_{tag}.parquet", index=False)
    pd.DataFrame(w0_rows).to_parquet(OUT_DIR / f"w0_{tag}.parquet", index=False)
    pd.DataFrame(elite_rows).to_parquet(OUT_DIR / f"elite_weights_{tag}.parquet", index=False)
    pd.DataFrame(excl_rows, columns=["as_of", "symbol", "close", "trail6m", "reason"]).to_csv(
        OUT_DIR / f"sanity_excluded_{tag}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(flag_rows, columns=["window", "as_of", "symbol", "q_ret", "max_daily_move",
                                     "max_weight_any_combo"]).to_csv(
        OUT_DIR / f"sanity_flagged_{tag}.csv", index=False, encoding="utf-8-sig")
    for w in windows:
        g = pd.concat([p[1] for p in daily_parts if p[0] == w])
        n = pd.concat([p[2] for p in daily_parts if p[0] == w])
        g.to_parquet(OUT_DIR / f"daily_gross_{tag}_w{w}.parquet")
        n.to_parquet(OUT_DIR / f"daily_net_{tag}_w{w}.parquet")
    pd.DataFrame(combos).to_csv(OUT_DIR / f"combos_{tag}.csv", index=False, encoding="utf-8-sig")
    with open(OUT_DIR / f"meta_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"market": market, "pool": pool, "windows": meta, "costs": COSTS[market],
                   "filters": {"tradable_lookback_days": TRADABLE_LOOKBACK_DAYS, "liquidity": liq,
                               "sanity_trail_min": SANITY_TRAIL_MIN.get(market)},
                   "generated": pd.Timestamp.now().isoformat()}, f, ensure_ascii=False, indent=1, default=str)

    # 對帳摘要
    qdf = pd.DataFrame(q_rows)
    print("\n=== 各窗累積報酬（毛，季報酬複利）===")
    for w in windows:
        sub = qdf[qdf.window == w]
        cum = sub.groupby("combo")["ret_gross"].apply(lambda r: float(np.prod(1 + r) - 1))
        print(f"window{w}：control0 {cum['REF_control0']:+.2%}｜每檔等權 {cum['REF_stock_equal']:+.2%}"
              f"｜市值傾斜全開 {cum['REF_w2c_full']:+.2%}｜大盤 {meta[w]['bench_cum']:+.2%}")
    print(f"\n完成，總耗時 {time.time()-t0:.0f}s，輸出在 {OUT_DIR}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True, choices=["TW", "US"])
    ap.add_argument("--pool", required=True, choices=["exclude_v1", "baseline"])
    ap.add_argument("--windows", default="1,2,3,4")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--liq", action="store_true", help="套用美股流動性門檻（輸出檔名加 _liq）")
    ap.add_argument("--scores", default=None, help="只跑指定排名依據（逗號分隔），例：S7_hot,S8_breakout")
    ap.add_argument("--tag-suffix", default="", help="輸出檔名後綴，例：_sig")
    a = ap.parse_args(argv)
    run(a.market, a.pool, [int(x) for x in a.windows.split(",")], a.smoke, a.liq,
        a.scores.split(",") if a.scores else None, a.tag_suffix)


if __name__ == "__main__":
    main()
