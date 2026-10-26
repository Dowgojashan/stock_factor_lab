# -*- coding: utf-8 -*-
"""2026-10-27：Active Share（Cremers & Petajisto, 2009）市場參數化版本——
2026-10-01台股做過一次臨時分析（`文件/完整操作流程.md`§3.2.2），當時是
ad-hoc腳本、沒有固化成可重用函式、美股從未做過。這支腳本①把`active_share()`
補成`weights.py`的正式函式②補上美股版本。

🔴🔴 2026-10-27 code review修正（第二版，第一版有兩個bug）：
①第一版在每個日期都無條件呼叫`apply_w2c()`算「W2c傾斜後」這欄，等於假設
W2c**每季都套用**——但查真實production checkpoint的`config_used.w2c_
applied`欄位，TW baseline L2這8季只有3季（2025-06-30/09-30/12-31）真的
套用傾斜、US baseline L2只有1季（2025-12-31）真的套用（一季決策延遲
機制下，agent選擇W2c的決策要等下一季才生效）。第一版docstring誤寫成
「用m1dfix最終版」，暗示這是真實production路徑，但程式碼算的其實是
forced-every-quarter的假設性情境，文字跟程式碼對不上。
②修①的過程中，原本想沿用`build_rolling_inputs()`/`uids`重算control0
持股，但TW baseline走的是`simulate.py`原生慣例（讀`walkforward_members.
parquet`固定key），不是`build_rolling_inputs()`吃的picks parquet
慣例——兩套候選池/持股解析路徑不是同一組`uids`，混用會讓control0欄位
算出跟真實checkpoint持股對不起來的假持股。

**這一版改成直接讀兩個arm（control0／L2）各自真實checkpoint的`weights_
end`**，不重新呼叫`resolve_weights()`/`apply_w2c()`，徹底避開這兩個坑——
兩欄都是m1dfix/m2_v2最終版8季真正持有過的持股。

用法：cd code; PYTHONIOENCODING=utf-8 python -m app._check_active_share_both_markets
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from database import Database  # noqa: E402
from app import monitor  # noqa: E402
from app import weights as weights_mod  # noqa: E402

APPLAYER = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
RUNS_DIR = Path(__file__).resolve().parent / "_runs"

REGISTRATION_DATE = "2023-12-31"
QUARTER_ENDS = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
                "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]

MARKETS = {
    "TW": dict(
        control0=RUNS_DIR / "simulate_formal_8q_control0_L2_openSec_boost_v4_cap30_m2_v2.jsonl",
        l2=RUNS_DIR / "simulate_formal_8q_control0_L2_openSec_boost_v4_cap30_m2_v2.jsonl",
        arm_split=True,  # 單一檔案、用arm欄位篩
    ),
    "US": dict(
        control0=RUNS_DIR / "simulate_anchored_w4_openSec_balanced_baseline_cap30_hotseg_m1dfix_control0.jsonl",
        l2=RUNS_DIR / "simulate_anchored_w4_openSec_balanced_baseline_cap30_hotseg_m1dfix_L2.jsonl",
        arm_split=False,  # 兩個獨立檔案
    ),
}


def bench_weight_row(mcap_wide: pd.DataFrame, as_of: str) -> pd.Series:
    row = monitor.asof_row(mcap_wide, as_of).dropna()
    return row / row.sum()


def load_weights_by_quarter(path: Path, arm_filter: str | None) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            cp = json.loads(line)
            if arm_filter and cp.get("arm") != arm_filter:
                continue
            out[cp["quarter_end"]] = dict(weights_end=cp["weights_end"],
                                          w2c_applied=cp.get("config_used", {}).get("w2c_applied", False))
    return out


def run_market(tree_key: str, cfg: dict) -> pd.DataFrame:
    print(f"\n=== {tree_key}：Active Share（真實control0 vs 真實L2持股，m1dfix/m2_v2最終版）===")
    db = Database(tree_key)
    conn = db.create_connection()
    mcap_wide = monitor.fetch_mcap_wide(conn, market=tree_key, start="2007-01-01")

    control0_by_q = load_weights_by_quarter(cfg["control0"], "control0" if cfg["arm_split"] else None)
    l2_by_q = load_weights_by_quarter(cfg["l2"], "L2" if cfg["arm_split"] else None)
    assert len(control0_by_q) == 8 and len(l2_by_q) == 8, \
        f"{tree_key}: 預期各8季checkpoint，實際control0={len(control0_by_q)} L2={len(l2_by_q)}"

    rows = []
    print(f"{'quarter_end':>12} {'真實control0':>14} {'真實L2':>10}  [w2c_applied]")
    for q in QUARTER_ENDS:
        bench = bench_weight_row(mcap_wide, q)
        as_c0 = weights_mod.active_share(control0_by_q[q]["weights_end"], bench)
        as_l2 = weights_mod.active_share(l2_by_q[q]["weights_end"], bench)
        applied = l2_by_q[q]["w2c_applied"]
        print(f"{q:>12} {as_c0*100:>13.1f}% {as_l2*100:>9.1f}%  [{applied}]")
        rows.append(dict(market=tree_key, quarter_end=q, active_share_control0=as_c0,
                         active_share_l2=as_l2, w2c_applied=applied))

    df = pd.DataFrame(rows)
    print(f"  平均：control0={df['active_share_control0'].mean()*100:.1f}%"
         f"（範圍{df['active_share_control0'].min()*100:.1f}%~{df['active_share_control0'].max()*100:.1f}%）"
         f"  L2={df['active_share_l2'].mean()*100:.1f}%"
         f"（範圍{df['active_share_l2'].min()*100:.1f}%~{df['active_share_l2'].max()*100:.1f}%，"
         f"{int(df['w2c_applied'].sum())}/8季真的套用W2c）")
    return df


def main():
    all_rows = []
    for tree_key, cfg in MARKETS.items():
        all_rows.append(run_market(tree_key, cfg))
    out = pd.concat(all_rows, ignore_index=True)
    out_path = APPLAYER / "active_share_both_markets_2026-10-27.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}")


if __name__ == "__main__":
    main()
