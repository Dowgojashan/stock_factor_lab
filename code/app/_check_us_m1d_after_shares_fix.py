# -*- coding: utf-8 -*-
"""collector 2026-10-28 第二輪修正（close 還原＋outstanding_shares 528檔/31,124列）後，
重算美股 M1-D（指數端前20%市值集中度）：門檻校準＋window4 逐季狀態，跟修正前比對。

不覆寫任何既有檔案：計算邏輯照抄 `_prelim_a1_index_concentration_us.py`（同一個
q1_weight 定義、同一個 2009-2023 訓練期、同一個「排除2009-2011起訖窗」的 production 版本），
逐季狀態用正式的 `triggers.register_m1d`／`run_quarterly_series`。
修正前的對照：既有輸出 `a1_index_concentration_series_US.csv`（修正前產生）、
`triggers.X_P90/X_P75["US"]`（現行凍結門檻）、US L2 正式 checkpoint 裡的 m1d 狀態。
不呼叫 LLM。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import pandas as pd  # noqa: E402
from database import Database  # noqa: E402

from app import monitor, simulate, triggers  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OLD_SERIES = ROOT / "_analysis_outputs_applayer" / "a1_index_concentration_series_US.csv"
CKPT = ROOT / "code/app/_runs/simulate_anchored_w4_openSec_balanced_exclude_v1_cap30_hotseg_m1dfix_L2.jsonl"
TRAIN_END = "2023-12-31"


def main():
    conn = Database("US").create_connection()
    wide = monitor.fetch_mcap_wide(conn, market="US", start="2009-01-01")

    def q1_weight_at(d):
        row = monitor.asof_row(wide, d).dropna()
        ranks = row.rank(pct=True)
        return float(row[ranks >= 0.8].sum() / row.sum())

    qe = pd.date_range("2009-03-31", "2025-12-31", freq="Q")
    new = pd.DataFrame({"date": qe, "q1_new": [q1_weight_at(d.strftime("%Y-%m-%d")) for d in qe]})

    # 門檻重算（同 production 版本）
    train = new[new.date <= TRAIN_END].reset_index(drop=True)
    train["chg_8q"] = train.q1_new.shift(-8) - train.q1_new
    train["ws"], train["we"] = train.date, train.date.shift(-8)
    dist = train.dropna(subset=["chg_8q"])
    dist = dist[~(dist.ws.dt.year.between(2009, 2011) | dist.we.dt.year.between(2009, 2011))]
    p90_new, p75_new = dist.chg_8q.quantile(.90), dist.chg_8q.quantile(.75)
    print(f"門檻：現行 p90={triggers.X_P90['US']*100:.4f}pp / p75={triggers.X_P75['US']*100:.4f}pp"
          f"｜新資料重算 p90={p90_new*100:.4f}pp / p75={p75_new*100:.4f}pp")

    if OLD_SERIES.exists():
        old = pd.read_csv(OLD_SERIES, parse_dates=["date"])[["date", "idx_q1_weight"]]
        cmp_ = new.merge(old, on="date", how="left")
        cmp_["diff_pp"] = (cmp_.q1_new - cmp_.idx_q1_weight) * 100
        print(f"q1_weight 新舊差（2009-2025）：最大絕對差 {cmp_.diff_pp.abs().max():.3f}pp；"
              f"2022Q2 以後最大 {cmp_[cmp_.date >= '2022-04-01'].diff_pp.abs().max():.3f}pp")

    old_states = {}
    for line in open(CKPT, encoding="utf-8"):
        cp = json.loads(line)
        old_states[cp["quarter_end"]] = cp["m1d"]

    cond_now = triggers.register_m1d(wide, simulate.REGISTRATION_DATE, market="US")
    cond_new = triggers.register_m1d(wide, simulate.REGISTRATION_DATE, market="US", p75=p75_new, p90=p90_new)
    s_now = triggers.run_quarterly_series(wide, cond_now, simulate.QUARTER_ENDS)
    s_new = triggers.run_quarterly_series(wide, cond_new, simulate.QUARTER_ENDS)
    print(f"\n登記日 q1_weight：修正前(checkpoint) {old_states[simulate.QUARTER_ENDS[0]]['registration_q1_weight']:.4%}"
          f"｜修正後 {cond_now.registration_q1_weight:.4%}")
    print("\n季別        修正前偏離/狀態(checkpoint)   修正後偏離｜現行門檻狀態｜新門檻狀態")
    for (_, a), (_, b) in zip(s_now.iterrows(), s_new.iterrows()):
        o = old_states[a.as_of]
        print(f"{a.as_of}  {o['cumulative_deviation']*100:+.3f}pp {o['state']:<10}   "
              f"{a.cumulative_deviation*100:+.3f}pp｜{a.state:<10}｜{b.state}")


if __name__ == "__main__":
    main()
