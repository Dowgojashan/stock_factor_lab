# -*- coding: utf-8 -*-
"""triggers.py 正確性驗證：跟設計文件 §7.4a 已記錄的真實逐季觸發時程對照。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from database import Database  # noqa: E402

from app import monitor, triggers  # noqa: E402

QUARTER_ENDS = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
                "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]

EXPECTED_STATE = {
    "2024-03-31": "NONE", "2024-06-30": "NONE", "2024-09-30": "NONE",
    "2024-12-31": "TRIGGERED",   # 首次觸發
    "2025-03-31": "TRIGGERED",   # 短暫回落但維持觀察（遲滯，不解除）
    "2025-06-30": "TRIGGERED", "2025-09-30": "TRIGGERED", "2025-12-31": "TRIGGERED",
}


def main():
    db = Database("TW")
    conn = db.create_connection()
    mcap_wide = monitor.fetch_mcap_wide(conn, start="2007-01-01")

    cond = triggers.register_m1d(mcap_wide, "2023-12-31")
    print(f"登記：{cond}")

    df = triggers.run_quarterly_series(mcap_wide, cond, QUARTER_ENDS)
    print(df.to_string(index=False))

    print("\n=== 對照 §7.4a 已記錄的逐季狀態 ===")
    all_ok = True
    for _, r in df.iterrows():
        exp = EXPECTED_STATE[r["as_of"]]
        ok = "OK" if r["state"] == exp else "MISMATCH"
        if r["state"] != exp:
            all_ok = False
        print(f"  {r['as_of']}: dev={r['cumulative_deviation']*100:+.2f}pp  "
             f"狀態={r['state']:<10} 預期={exp:<10} {ok}")

    print("\n全部一致" if all_ok else "\n有不一致，需要檢查")
    assert all_ok, "跟 §7.4a 已記錄的逐季狀態對不上"


if __name__ == "__main__":
    main()
