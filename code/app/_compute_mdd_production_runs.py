# -*- coding: utf-8 -*-
"""2026-10-16：補算台美股8季真實LLM生產模擬（window4，2024-2025）的MDD，
給老師報告用。只讀既有checkpoint，不重跑、不呼叫LLM。

🔴 2026-10-17修正（外部審閱抓到的頻率不一致問題）：策略MDD是用checkpoint的
季報酬複利重建nav（只有9個點：期初+8季末），大盤原本卻是用taiex_tr/sp500_tr
的**日頻**資料算MDD——季頻會漏掉季內的回撤（例如2025年4月的關稅暴跌就發生
在季內），日頻大盤數字因此系統性大於季頻策略數字，兩者不是同一個量尺，
不能直接比較「策略MDD遠小於大盤」。

修正後本腳本同時輸出兩種大盤MDD：
①`_series_mdd()`：日頻，保留當對照／上下文參考，但明確標註不可跟策略MDD
  直接比較
②`_series_mdd_at_dates()`：**只在策略checkpoint的同一組9個日期（期初+8季末）
  取樣**算MDD，這才是跟策略MDD頻率一致、可以公平比較的版本——代價是這個
  版本本身也會低估大盤的「真實」日頻MDD（因為一樣漏掉季內波動），但至少
  策略跟大盤用的是同一把尺。

MDD算法（策略端）：從outcome.portfolio_realized_return逐季複利重建nav序列
（不依賴checkpoint裡的drawdown欄位——2026-10-08前跑的舊版checkpoint沒有
這個欄位，用drawdown.current_dd會對這些run直接KeyError）。

用法：cd code; PYTHONIOENCODING=utf-8 python -m app._compute_mdd_production_runs
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from app import simulate  # noqa: E402

QUARTER_ENDS = simulate.QUARTER_ENDS
REGISTRATION_DATE = simulate.REGISTRATION_DATE


def mdd_from_checkpoints_native(run_id: str, arm: str) -> tuple[float, float, int]:
    """native run_simulation()：單一run_id，兩臂混在一起存，用arm欄位篩。"""
    cps = [c for c in simulate.load_checkpoints(run_id) if c["arm"] == arm]
    cps = sorted(cps, key=lambda c: c["quarter_end"])
    return _mdd_and_cum(cps)


def mdd_from_checkpoints_rolling(run_id_prefix: str, arm: str) -> tuple[float, float, int]:
    """run_rolling_simulation()：run_id本身已含arm後綴。"""
    run_id = f"{run_id_prefix}_{arm}"
    cps = sorted(simulate.load_checkpoints(run_id), key=lambda c: c["quarter_end"])
    return _mdd_and_cum(cps)


def _mdd_and_cum(cps: list[dict]) -> tuple[float, float, int]:
    """從outcome.portfolio_realized_return逐季複利重建nav序列算MDD——不依賴
    checkpoint裡的drawdown欄位（舊版checkpoint，2026-10-08前跑的，沒有這欄）。"""
    if not cps:
        return float("nan"), float("nan"), 0
    nav = 1.0
    navs = [1.0]
    for c in cps:
        nav *= (1 + c["outcome"]["portfolio_realized_return"])
        navs.append(nav)
    nav_series = pd.Series(navs)
    peak = nav_series.cummax()
    mdd = float((nav_series / peak - 1).min())
    return mdd, nav - 1.0, len(cps)


def _series_mdd(dates: pd.Series, closes: pd.Series, start: str, end: str) -> float:
    """日頻MDD——保留當背景參考，不可直接跟策略的季頻MDD比較，見檔頭說明。"""
    s = closes[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    nav = s / s.iloc[0]
    peak = nav.cummax()
    return float((nav / peak - 1).min())


def _series_mdd_at_dates(dates: pd.Series, closes: pd.Series, sample_dates: list[str]) -> float:
    """🆕 2026-10-17新增：只在sample_dates（期初+8季末，共9點）取樣算MDD，
    跟策略端checkpoint的季頻nav序列用同一把尺，才是可以公平比較的版本。
    每個sample date取「當天或之前最近一個交易日」的收盤價（asof語意）。"""
    s = pd.Series(closes.values, index=pd.DatetimeIndex(dates)).sort_index()
    sampled = []
    for d in sample_dates:
        ts = pd.Timestamp(d)
        prior = s[s.index <= ts]
        if len(prior) == 0:
            raise ValueError(f"{d} 之前查無任何交易日資料")
        sampled.append(prior.iloc[-1])
    nav = pd.Series(sampled) / sampled[0]
    peak = nav.cummax()
    return float((nav / peak - 1).min())


def main():
    import fcv_core  # noqa: F401
    from database import Database

    print("=== 台美股8季真實LLM生產模擬 MDD（window4, 2024-2025）===\n")

    rows = []

    # 🔴 2026-10-20更新：四組"30%"全部改指向含M2+Hot Segment pool校準+完整
    # 5條W2c caveat的最新checkpoint（舊版不含M2，已過期，見`文件/實驗收尾
    # 追蹤.md`§一）。"7%"歷史對照組保留不動。
    # 🔴🔴 2026-10-22更新：US兩組再改指向`_m1dfix`版本——`_hotseg`checkpoint
    # 建立在過期的「M1-D全程NONE」假設上（universe_clause bug＋2009-2012
    # 市值資料本身有誤，兩輪修正後美股window4真實在2025Q3進入TRIGGERED）。
    # TW不受這個bug影響，維持`_hotseg`版本不變。

    # --- TW baseline（native convention）---
    for cap, run_id in [("7%", "formal_8q_control0_L2_openSec_boost_v4"),
                        ("30%", "formal_8q_control0_L2_openSec_boost_v4_cap30_m2_v2")]:
        for arm in simulate.ARMS:
            mdd, cum, n = mdd_from_checkpoints_native(run_id, arm)
            rows.append(dict(market="TW", variant="baseline", cap=cap, arm=arm,
                             mdd=mdd, cum=cum, n=n))

    # --- TW exclude_v1（rolling convention）---
    for cap, prefix in [("7%", "anchored_w4_openSec_boost_exclude_v1"),
                        ("30%", "anchored_w4_openSec_boost_exclude_v1_cap30_hotseg")]:
        for arm in simulate.ARMS:
            mdd, cum, n = mdd_from_checkpoints_rolling(prefix, arm)
            rows.append(dict(market="TW", variant="exclude_v1", cap=cap, arm=arm,
                             mdd=mdd, cum=cum, n=n))

    # --- US baseline（rolling convention, mcapfix）---
    for cap, prefix in [("7%(mcapfix)", "anchored_w4_openSec_balanced_baseline_mcapfix"),
                        ("30%", "anchored_w4_openSec_balanced_baseline_cap30_hotseg_m1dfix")]:
        for arm in simulate.ARMS:
            mdd, cum, n = mdd_from_checkpoints_rolling(prefix, arm)
            rows.append(dict(market="US", variant="baseline", cap=cap, arm=arm,
                             mdd=mdd, cum=cum, n=n))

    # --- US exclude_v1（rolling convention, mcapfix）---
    for cap, prefix in [("7%(mcapfix)", "anchored_w4_openSec_balanced_exclude_v1_mcapfix"),
                        ("30%", "anchored_w4_openSec_balanced_exclude_v1_cap30_hotseg_m1dfix")]:
        for arm in simulate.ARMS:
            mdd, cum, n = mdd_from_checkpoints_rolling(prefix, arm)
            rows.append(dict(market="US", variant="exclude_v1", cap=cap, arm=arm,
                             mdd=mdd, cum=cum, n=n))

    df = pd.DataFrame(rows)
    df["mdd_pct"] = df["mdd"] * 100
    df["cum_pct"] = df["cum"] * 100
    print(df.to_string(index=False))

    missing = df[df.n != 8]
    if len(missing):
        print(f"\n⚠️ 以下組合checkpoint數不是8，數字可能不完整：\n{missing}")

    # --- 大盤對照：TAIEX / S&P500 TR 同期MDD ---
    sample_dates = [REGISTRATION_DATE] + list(QUARTER_ENDS)
    print(f"\n=== 大盤對照（同一段視窗 {REGISTRATION_DATE} -> {QUARTER_ENDS[-1]}）===")
    db = Database("TW")
    conn = db.create_connection()
    tx = pd.read_sql("SELECT date, close FROM taiex_tr ORDER BY date", conn)
    tx["date"] = pd.to_datetime(tx["date"])
    taiex_mdd_daily = _series_mdd(tx["date"], tx["close"].astype(float),
                                  REGISTRATION_DATE, QUARTER_ENDS[-1])
    taiex_mdd_q = _series_mdd_at_dates(tx["date"], tx["close"].astype(float), sample_dates)
    print(f"TAIEX TR MDD（日頻，僅供參考，不可跟策略MDD直接比較）：{taiex_mdd_daily:+.2%}")
    print(f"TAIEX TR MDD（季頻，跟策略同一組9個日期取樣，可公平比較）：{taiex_mdd_q:+.2%}")

    sp500_mdd_daily = sp500_mdd_q = float("nan")
    try:
        sp = pd.read_sql("SELECT date, close FROM sp500_tr ORDER BY date", conn)
        sp["date"] = pd.to_datetime(sp["date"])
        sp500_mdd_daily = _series_mdd(sp["date"], sp["close"].astype(float),
                                      REGISTRATION_DATE, QUARTER_ENDS[-1])
        sp500_mdd_q = _series_mdd_at_dates(sp["date"], sp["close"].astype(float), sample_dates)
        print(f"S&P500 TR MDD（日頻，僅供參考）：{sp500_mdd_daily:+.2%}")
        print(f"S&P500 TR MDD（季頻，可公平比較）：{sp500_mdd_q:+.2%}")
    except Exception as e:
        print(f"S&P500 TR 查詢失敗：{e}")

    out_path = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
               / "production_runs_mdd_window4.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}")

    # 🆕 2026-10-17：大盤MDD之前只印不存，外部審閱指出無法追溯，這裡補存成獨立CSV
    bench_path = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
                 / "production_runs_mdd_benchmark.csv")
    bench_df = pd.DataFrame([
        dict(index="TAIEX_TR", market="TW", freq="daily", mdd_pct=taiex_mdd_daily * 100,
            note="日頻，不可跟策略季頻MDD直接比較"),
        dict(index="TAIEX_TR", market="TW", freq="quarterly_matched", mdd_pct=taiex_mdd_q * 100,
            note=f"跟策略同一組{len(sample_dates)}個日期取樣，可公平比較"),
        dict(index="SP500_TR", market="US", freq="daily", mdd_pct=sp500_mdd_daily * 100,
            note="日頻，不可跟策略季頻MDD直接比較"),
        dict(index="SP500_TR", market="US", freq="quarterly_matched", mdd_pct=sp500_mdd_q * 100,
            note=f"跟策略同一組{len(sample_dates)}個日期取樣，可公平比較"),
    ])
    bench_df.to_csv(bench_path, index=False, encoding="utf-8-sig")
    print(f"寫入 {bench_path}")


if __name__ == "__main__":
    main()
