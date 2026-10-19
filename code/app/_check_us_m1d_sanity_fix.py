# -*- coding: utf-8 -*-
"""2026-10-17：確認`monitor.fetch_mcap_wide()`加上`SANITY_MAX_CLOSE`離群值
防線後，美股原窗口（window4, 2024-2025, REGISTRATION_DATE=2023-12-31）的
M1-D真實觸發模式是什麼——不呼叫LLM、不燒API成本，純粹重算觸發序列，
拿去跟現有（修復前）production checkpoint比對，確認「美股從2024Q4起連續
觸發W2c」這個既有敘事是否站得住腳。

用法：cd code; PYTHONIOENCODING=utf-8 python -m app._check_us_m1d_sanity_fix
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from app import monitor, triggers, simulate  # noqa: E402

OUT_PATH = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
           / "us_m1d_sanity_fix_recheck.csv")


def main():
    from database import Database

    db = Database("US")
    conn = db.create_connection()
    wide = monitor.fetch_mcap_wide(conn, market="US", start="2007-01-01")

    # 🔴 2026-10-18：market="US"明確傳入——觸發門檻已改成按市場分開校準
    # （美股現在有自己的p90=8.61pp，不再借用台股的1.86pp），漏傳會悄悄退回
    # 台股門檻，等於白跑這支腳本原本要驗證的事。
    cond = triggers.register_m1d(wide, simulate.REGISTRATION_DATE, market="US")
    print(f"=== 美股原窗口（修正後）registration({simulate.REGISTRATION_DATE}) "
         f"q1_weight={cond.registration_q1_weight:.4%} ===")

    rows = []
    state = "NONE"
    for q in simulate.QUARTER_ENDS:
        r = triggers.evaluate_quarter(cond, wide, q, state)
        state = r["state"]
        rows.append(dict(quarter_end=q, q1_weight=r["q1_weight"],
                         cumulative_deviation=r["cumulative_deviation"],
                         prev_state=r["prev_state"], state=r["state"], action=r["action"]))
        print(f"  {q}  q1w={r['q1_weight']:.4%}  dev={r['cumulative_deviation']*100:+.2f}pp"
             f"  {r['prev_state']}->{r['state']} ({r['action']})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {OUT_PATH}")

    n_triggered = (df["state"] == "TRIGGERED").sum()
    print(f"\n=== 結論：8季中{n_triggered}季TRIGGERED（修復前production checkpoint"
         f"顯示5季：2024Q4~2025Q4連續觸發）===")
    if n_triggered == 0:
        print("⚠️ 修正後完全沒有觸發——現有4.4/4.2.2的美股W2c敘事（TSSI/RPAY/PLTR/"
             "AAPL/AVGO/TSLA選股清單、-1.45%/+51.65%等數字）建立在離群值污染的假"
             "觸發訊號上，需要用修正後資料重跑美股4組正式production run才能取得"
             "可信數字。")


if __name__ == "__main__":
    main()
