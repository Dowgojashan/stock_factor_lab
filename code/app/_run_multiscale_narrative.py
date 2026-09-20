# -*- coding: utf-8 -*-
"""把多時間尺度解釋（方案B）套用到已完成的正式8季實驗（D53/D54的
`formal_8q_control0_L2_execlayer_v2`），產出真實的月/半年/年敘事。
不重跑任何決策，只用既有真實checkpoint的資料往上（半年/年）跟往下
（月）彙整。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from database import Database  # noqa: E402
from fcv_core import MarketData  # noqa: E402

from app import agents, facts_lean, monitor, simulate  # noqa: E402

SOURCE_RUN_ID = "formal_8q_control0_L2_execlayer_v2"
RUN_ID = "multiscale_narrative_l2"

# 半年/年的彙整邊界（日曆對齊：2024Q1-Q2=2024H1, Q3-Q4=2024H2，
# 2024全年=Q1-Q4；2025同理）
SEMIANNUAL_GROUPS = {
    "2024-06-30": ["2024-03-31", "2024-06-30"],           # 2024H1
    "2024-12-31": ["2024-09-30", "2024-12-31"],           # 2024H2
    "2025-06-30": ["2025-03-31", "2025-06-30"],           # 2025H1
    "2025-12-31": ["2025-09-30", "2025-12-31"],           # 2025H2
}
ANNUAL_GROUPS = {
    "2024-12-31": ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"],  # 2024全年
    "2025-12-31": ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"],  # 2025全年
}


def quarter_summary(c: dict) -> dict:
    return {
        "quarter_end": c["quarter_end"], "m1d_state": c["m1d"]["state"],
        "decision": c["decision"]["decision"],
        "portfolio_realized_return": c["outcome"]["portfolio_realized_return"],
        "excess_vs_equal_weight": c["outcome"].get("excess_vs_equal_weight"),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    dry_run = not args.real

    from utils.config import Config
    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")
    print(f"模型={model}  dry_run={dry_run}")

    if not dry_run:
        from utils import openai_quota as OQ
        print(f"今日額度用量：{OQ.today_totals()}")

    cps = {c["quarter_end"]: c for c in simulate.load_checkpoints(SOURCE_RUN_ID) if c["arm"] == "L2"}
    quarters = simulate.QUARTER_ENDS
    registration = simulate.REGISTRATION_DATE

    md = MarketData("TW")
    md_map = {"TW": md}
    db = Database("TW")
    conn = db.create_connection()
    tx = pd.read_sql("SELECT date, close FROM taiex_tr ORDER BY date", conn)
    tx["date"] = pd.to_datetime(tx["date"])
    cap_weight_series = tx.set_index("date")["close"].astype(float).sort_index()

    existing = {c["quarter_end"] for c in simulate.load_checkpoints(RUN_ID)}

    for i, end in enumerate(quarters):
        if end in existing:
            print(f"{end} 已有 checkpoint，跳過")
            continue
        as_of = registration if i == 0 else quarters[i - 1]
        weights_as_of = cps[as_of]["weights_end"] if i > 0 else None
        if weights_as_of is None:
            # 第一季用該季自己的weights_end近似（第一季的as_of=registration，
            # 沒有更早的checkpoint可用，這是唯一的邊界情況，用該季末權重
            # 近似季初——跟真正持有的股票集合一致，只是權重比例可能有微小
            # 落差，對「當月描述性分解」這種非觸發用途影響可忽略）
            weights_as_of = cps[end]["weights_end"]

        print(f"[{end}] 算月頻分解 {as_of} -> {end} ...")
        monthly_rows = monitor.monthly_breakdown(md_map, weights_as_of, as_of, end,
                                                  cap_weight_series=cap_weight_series)
        monthly_csv = facts_lean.monthly_breakdown_csv(monthly_rows)

        semiannual_csv = None
        if end in SEMIANNUAL_GROUPS:
            qs = [quarter_summary(cps[q]) for q in SEMIANNUAL_GROUPS[end]]
            semiannual_csv = facts_lean.multiscale_rollup_csv(qs)

        annual_csv = None
        if end in ANNUAL_GROUPS:
            qs = [quarter_summary(cps[q]) for q in ANNUAL_GROUPS[end]]
            annual_csv = facts_lean.multiscale_rollup_csv(qs)

        quarterly_context = {
            "decision": cps[end]["decision"]["decision"],
            "m1d_state": cps[end]["m1d"]["state"],
            "retrospective_output": cps[end].get("retrospective_output", {}),
        }

        print(f"[{end}] 呼叫多時間尺度敘事（半年資料={'有' if semiannual_csv else '無'}，"
             f"年度資料={'有' if annual_csv else '無'}）...")
        result = agents.call_agent_a_multiscale_narrative(
            monthly_csv, semiannual_csv, annual_csv, quarterly_context,
            model=model, api_key=api_key, dry_run=dry_run)

        checkpoint = {"quarter_end": end, "monthly_csv": monthly_csv,
                     "semiannual_csv": semiannual_csv, "annual_csv": annual_csv,
                     "narrative": result["explanation"]}
        simulate.save_checkpoint(RUN_ID, checkpoint)
        print(f"[{end}] 完成")

    print("\n=== 總覽 ===")
    for c in sorted(simulate.load_checkpoints(RUN_ID), key=lambda x: x["quarter_end"]):
        print(f"  {c['quarter_end']}  半年資料={'有' if c['semiannual_csv'] else '無'}"
             f"  年度資料={'有' if c['annual_csv'] else '無'}")


if __name__ == "__main__":
    main()
