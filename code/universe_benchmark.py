# -*- coding: utf-8 -*-
"""
自建宇宙基準：「什麼都不篩，把宇宙裡的股票全部等權買下來」的績效。

為什麼需要這個（見對話 2026-08-08）：
  現用的大盤基準 `taiex` 表經查證是**台灣加權股價指數（價格指數）**，
  年底值 2000→4,739｜2010→8,972｜2024→23,035 完全對應價格指數，**不含股利**。
  但我們的策略用 TEJ **還原收盤價**（配息已還原），**含股利**。
  → 一直在拿「含息的策略」比「不含息的大盤」，每個「贏大盤」都被高估約 3~4pp。
  （美股 sp500 表同樣是價格指數，2000→1,320｜2024→5,882，問題相同。）

自建基準把三個不對等一次消掉：
  1. 股利 —— 同樣用還原價
  2. 宇宙 —— 同樣只含「有因子資料」的公司（策略選股的母體）
  3. 交易成本 —— 同樣月頻換股、同樣手續費/證交稅

它回答的問題是「**因子篩選有沒有比不篩選更好**」，
比外部指數的「有沒有贏市場」更貼近本研究要證明的事。
兩者互補：外部指數留給對外對照（學姊論文用的也是外部指數）。

用法：python universe_benchmark.py [--market TW]
"""
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fcv_core import (MarketData, ensure_dtindex_cols, write_report_artifacts,  # noqa: E402
                      quick_stats_trades, ART_DIR)
from combinations import sim_conditions                                          # noqa: E402
from sweep_config import MARKET_START                                            # noqa: E402
from phase1_linearity import IN_SAMPLE_END                                       # noqa: E402


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


# 市場 → 報酬指數（含息）資料表。collector 於 2026-08-11 匯入（任務 C）。
# ⚠️ 不改 database.py 的 BENCHMARK_TABLE（那是共用 pipeline 程式），只在本檔查表。
TR_TABLE = {"TW": "taiex_tr", "US": "sp500_tr"}


def _index_cagr(market, table, start="2000-01-01"):
    """任一指數表的年化。回傳 (cagr, 起日, 訖日)；表不存在或空表回 (nan, None, None)。"""
    import fcv_core  # noqa: F401  bootstrap sys.path
    from database import Database
    from phase1_linearity import IN_SAMPLE_END as _END
    conn = Database(market).create_connection()
    try:
        s = pd.read_sql(f"SELECT date, close FROM `{table}`", conn)
    except Exception:
        return float("nan"), None, None
    finally:
        conn.close()
    if s.empty:
        return float("nan"), None, None
    s["date"] = pd.to_datetime(s["date"])
    s = s.set_index("date").sort_index()["close"].astype(float).dropna()
    s = s[(s.index >= start) & (s.index <= _END)]
    if len(s) < 2:
        return float("nan"), None, None
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1), s.index[0].date(), s.index[-1].date()


def get_bench(market, kind="universe"):
    """統一的基準取得介面，供 phase1/3/4_analyze 共用。

      index    ：外部**價格**指數（taiex/sp500）——不含股利
      index_tr ：外部**報酬**指數（taiex_tr/sp500_tr）——含股利
      universe ：自建宇宙基準（本檔算出來的）——含股利、同宇宙、同成本

    ⚠️ **index_tr 與另外兩者的期間不同，不可直接比大小**：
       台灣報酬指數 2003 才發布，跳過了 2000-2002 那段崩盤
       （加權指數 8,756 → 4,500），起算點差異足以造成數個百分點的假差距。
       要對照請用 bench_table() 取同期間的三方數字。

    主基準維持 universe：它問的是「因子篩選有沒有比不篩選更好」，
    才是本研究要證明的事；報酬指數留給論文做對外的傳統基準對照。

    回傳 (年化報酬, 說明字串)。
    """
    if kind == "index":
        from phase1_analyze import bench_cagr
        v, b0, b1 = bench_cagr(market)
        return v, f"外部價格指數 {b0}~{b1}（不含股利）"
    if kind == "index_tr":
        v, b0, b1 = _index_cagr(market, TR_TABLE[market])
        if b0 is None:
            raise FileNotFoundError(
                f"{TR_TABLE[market]} 不存在或為空表；請先請 collector 匯入該市場的報酬指數")
        return v, f"外部報酬指數 {b0}~{b1}（含股利）"
    p = Path(ART_DIR) / f"{market}_UNIVERSE_M" / "stats.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"找不到自建宇宙基準 {p}；請先執行：python universe_benchmark.py --market {market}")
    v = float(pd.read_parquet(p).iloc[0]["CAGR"])
    return v, "自建宇宙基準（全買、含股利、同宇宙同成本）"


def bench_table(market):
    """三方基準對照，**每一列都標明起算年**——期間不同就不能比大小。

    論文的基準章節直接用這張表：它同時說明了
    (a) 價格指數 vs 報酬指數的差＝股利貢獻
    (b) 報酬指數 vs 自建基準的差＝宇宙與權重方式（市值加權 vs 等權）
    """
    tr_start = "2003-01-01" if market == "TW" else "2000-01-01"
    rows = []
    v, d0, d1 = _index_cagr(market, __import__("database").Database.BENCHMARK_TABLE[market])
    rows.append(("外部價格指數（不含股利）", v, d0, d1))
    v, d0, d1 = _index_cagr(market, TR_TABLE[market])
    rows.append(("外部報酬指數（含股利）", v, d0, d1))
    # 價格指數同樣截到報酬指數的起點，才能單獨看出股利貢獻
    v, d0, d1 = _index_cagr(market, __import__("database").Database.BENCHMARK_TABLE[market],
                            start=tr_start)
    rows.append(("外部價格指數（同報酬指數期間）", v, d0, d1))
    try:
        v, _ = get_bench(market, "universe")
        rows.append(("自建宇宙基準（含股利、等權）", v, "2000-01-01", IN_SAMPLE_END))
    except FileNotFoundError:
        pass
    return pd.DataFrame(rows, columns=["基準", "年化", "起", "訖"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="TW", choices=["TW", "US"])
    args = ap.parse_args()
    market = args.market
    label = f"{market}_UNIVERSE_M"

    log(f"=== 自建宇宙基準｜{market}｜in-sample 至 {IN_SAMPLE_END} ===")
    md = MarketData(market, start=MARKET_START[market], end=IN_SAMPLE_END)

    # 「全選」遮罩：宇宙內所有公司、每個交易日都為 True。
    # md.common 已是「有因子資料 ∩ 有價格」的交集＝策略選股的同一個母體，
    # 故這裡不再另外篩，才能與策略公平對照。
    mask = pd.DataFrame(True, index=md.price_index, columns=md.common)
    mask = ensure_dtindex_cols(mask, md.common)
    log(f"宇宙 {len(md.common)} 檔｜交易日 {len(md.price_index)}"
        f"（{md.price_index.min().date()}~{md.price_index.max().date()}）")
    log(f"近似交易次數 {quick_stats_trades(mask)}")

    out_dir = Path(ART_DIR) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # 與所有 Phase 完全相同的回測路徑：月頻換股、引擎預設費率（台股 0.1425% + 0.3%）
    rc = sim_conditions(conditions={"UNIVERSE_EQW": mask}, resample="M", data=md.data)
    rows = write_report_artifacts(rc, out_dir)
    md.release()

    s = pd.DataFrame(rows)
    s.to_parquet(out_dir / "stats.parquet")
    (out_dir / "_DONE").write_text(datetime.now().isoformat(), encoding="utf-8")

    r = s.iloc[0]
    log(f"\n===== 結果（{time.time()-t0:.0f}s）=====")
    for k in ["CAGR", "daily_sharpe", "max_drawdown", "win_ratio"]:
        if k in r:
            log(f"  {k:14s} = {r[k]}")

    # 與外部價格指數對照
    try:
        from phase1_analyze import bench_cagr
        ext, b0, b1 = bench_cagr(market)
        log(f"\n  外部指數（價格指數，不含股利）= {ext:.2%}  [{b0}~{b1}]")
        log(f"  自建宇宙基準（含股利、同宇宙、同成本）= {float(r['CAGR']):.2%}")
        log(f"  → 差距 {float(r['CAGR'])-ext:+.2%}"
            f"（就是先前被高估的幅度）")
    except Exception as e:
        log(f"  外部指數對照失敗：{e}")

    log(f"\n輸出於 {out_dir}")


if __name__ == "__main__":
    main()
