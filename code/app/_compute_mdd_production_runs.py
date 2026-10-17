# -*- coding: utf-8 -*-
"""2026-10-16：補算台美股8季真實LLM生產模擬（window4，2024-2025）的MDD，
給老師報告用。只讀既有checkpoint，不重跑、不呼叫LLM。

MDD算法：checkpoint['drawdown']['current_dd']已經是「相對截至當季為止歷史
高點的回撤」（nav_after/nav_peak_after - 1），取8季裡最小值（最負）即為
這段視窗內的已實現MDD。

同時算TAIEX／S&P500 TR同一段視窗（2023-12-31→2025-12-31）的MDD當對照基準。

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
    s = closes[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    nav = s / s.iloc[0]
    peak = nav.cummax()
    return float((nav / peak - 1).min())


def main():
    import fcv_core  # noqa: F401
    from database import Database

    print("=== 台美股8季真實LLM生產模擬 MDD（window4, 2024-2025）===\n")

    rows = []

    # --- TW baseline（native convention）---
    for cap, run_id in [("7%", "formal_8q_control0_L2_openSec_boost_v4"),
                        ("30%", "formal_8q_control0_L2_openSec_boost_v4_cap30")]:
        for arm in simulate.ARMS:
            mdd, cum, n = mdd_from_checkpoints_native(run_id, arm)
            rows.append(dict(market="TW", variant="baseline", cap=cap, arm=arm,
                             mdd=mdd, cum=cum, n=n))

    # --- TW exclude_v1（rolling convention）---
    for cap, prefix in [("7%", "anchored_w4_openSec_boost_exclude_v1"),
                        ("30%", "anchored_w4_openSec_boost_exclude_v1_cap30")]:
        for arm in simulate.ARMS:
            mdd, cum, n = mdd_from_checkpoints_rolling(prefix, arm)
            rows.append(dict(market="TW", variant="exclude_v1", cap=cap, arm=arm,
                             mdd=mdd, cum=cum, n=n))

    # --- US baseline（rolling convention, mcapfix）---
    for cap, prefix in [("7%(mcapfix)", "anchored_w4_openSec_balanced_baseline_mcapfix"),
                        ("30%", "anchored_w4_openSec_balanced_baseline_cap30")]:
        for arm in simulate.ARMS:
            mdd, cum, n = mdd_from_checkpoints_rolling(prefix, arm)
            rows.append(dict(market="US", variant="baseline", cap=cap, arm=arm,
                             mdd=mdd, cum=cum, n=n))

    # --- US exclude_v1（rolling convention, mcapfix）---
    for cap, prefix in [("7%(mcapfix)", "anchored_w4_openSec_balanced_exclude_v1_mcapfix"),
                        ("30%", "anchored_w4_openSec_balanced_exclude_v1_cap30")]:
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
    print("\n=== 大盤對照（同一段視窗 2023-12-31 -> 2025-12-31）===")
    db = Database("TW")
    conn = db.create_connection()
    tx = pd.read_sql("SELECT date, close FROM taiex_tr ORDER BY date", conn)
    tx["date"] = pd.to_datetime(tx["date"])
    taiex_mdd = _series_mdd(tx["date"], tx["close"].astype(float),
                            REGISTRATION_DATE, QUARTER_ENDS[-1])
    print(f"TAIEX TR MDD：{taiex_mdd:+.2%}")

    try:
        sp = pd.read_sql("SELECT date, close FROM sp500_tr ORDER BY date", conn)
        sp["date"] = pd.to_datetime(sp["date"])
        sp500_mdd = _series_mdd(sp["date"], sp["close"].astype(float),
                                REGISTRATION_DATE, QUARTER_ENDS[-1])
        print(f"S&P500 TR MDD：{sp500_mdd:+.2%}")
    except Exception as e:
        print(f"S&P500 TR 查詢失敗：{e}")

    out_path = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
               / "production_runs_mdd_window4.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}")


if __name__ == "__main__":
    main()
