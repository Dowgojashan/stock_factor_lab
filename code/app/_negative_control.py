# -*- coding: utf-8 -*-
"""§11.2 陰性對照（2026-09-18，使用者要求做完整版）：拿另一個歷史窗次
（不是window 4）跑完整的 control0+L2 agent 流程，測系統在**正常年份**
會不會誤報。

🔴 陰性A／陰性B**不是** Agent-A／Agent-B（Agent-B 已於D44移除，跟這裡無關，
使用者確認過這個命名巧合）——是兩個不同的歷史測試期間，純編號：
  陰性A：scheme E window 3（2021-01~2023-12，OOS 26.39%贏全買4.65pp，12季）
  陰性B：scheme A window 2／D window 1（2015-01~2016-12，OOS 10.6%仍贏全買
        4.0pp，8季；兩個scheme的window剛好日期相同，任選一個即可，這裡用A）

沿用 `simulate.py` 的共用元件（QuarterInputs／ActiveConfig／run_quarter／
checkpoint機制），不動 `run_simulation()` 本身（避免影響已驗證過的正式
8季實驗那條路徑）——這裡是另外寫一個參數化的跑法，跑其他窗次。

🔴 §7.7的window排除規則（`oos_end < as_of`）不需要為這裡特別處理：測試窗
自己的oos_end在自己的OOS期間內任何as_of都不會小於它，天然就會排除掉自己
（跟window 4被排除是同一條規則、不需要特判）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from database import Database  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402

from app import monitor, simulate, triggers  # noqa: E402

ARMS = ("control0", "L2")


def quarter_ends(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="Q")]


def get_window_members(scheme: str, window_no: int) -> tuple[dict[str, list[str]], str, str, str]:
    """回傳 (uids_by_allocation, registration_date, oos_start, oos_end)。"""
    m = pd.read_parquet(Path(__file__).resolve().parent.parent.parent
                        / "_analysis_outputs_robustness" / "walkforward_members.parquet")
    uids_by_allocation = {}
    is_end = oos_start = oos_end = None
    for alloc in ("equal", "proportional"):
        key = dict(tree_key="TW", scheme=scheme, window_no=window_no, k_mode="silhouette_is",
                  ratio="legacy", allocation=alloc, group="A_hrp")
        sub = m.copy()
        for k, v in key.items():
            sub = sub[sub[k] == v]
        assert len(sub) == 1, f"找不到 {key}"
        row = sub.iloc[0]
        uids_by_allocation[alloc] = list(row["members"])
        is_end = pd.Period(row["is_end"], freq="M").end_time.strftime("%Y-%m-%d")
        oos_start_m = pd.Period(row["oos_start"], freq="M")
        oos_end = pd.Period(row["oos_end"], freq="M").end_time.strftime("%Y-%m-%d")
        oos_start = oos_start_m.start_time.strftime("%Y-%m-%d")
    return uids_by_allocation, is_end, oos_start, oos_end


def run_negative_control(run_id: str, *, scheme: str, window_no: int,
                         model: str, api_key: str, dry_run: bool = True) -> None:
    uids_by_allocation, registration_date, oos_start, oos_end = get_window_members(scheme, window_no)
    quarters = quarter_ends(oos_start, oos_end)
    print(f"陰性對照：scheme={scheme} window={window_no}，登記={registration_date}，"
         f"{len(quarters)} 季（{quarters[0]} ~ {quarters[-1]}）")

    if not dry_run:
        from utils import openai_quota as OQ
        totals = OQ.today_totals()
        print(f"今日額度用量：{totals}（§12.4③ 續跑前先查額度，不足直接停）")

    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    md = MarketData("TW")
    md_map = {"TW": md}
    db = Database("TW")
    conn = db.create_connection()
    mcap_wide = monitor.fetch_mcap_wide(conn, start="2007-01-01")
    tx = pd.read_sql("SELECT date, close FROM taiex_tr ORDER BY date", conn)
    tx["date"] = pd.to_datetime(tx["date"])
    cap_weight_series = tx.set_index("date")["close"].astype(float).sort_index()

    cond = triggers.register_m1d(mcap_wide, registration_date)
    inputs = simulate.QuarterInputs(md=md, md_map=md_map, idx=idx, mcap_wide=mcap_wide,
                                    uids_by_allocation=uids_by_allocation, m1d_cond=cond,
                                    model=model, api_key=api_key,
                                    cap_weight_series=cap_weight_series)

    from app import agents

    for arm in ARMS:
        state = "NONE"
        prev_weights = None
        config = simulate.ActiveConfig()
        agent_a_memory: dict = {"recent_quarter_full": None, "earlier_quarters_summary": []}
        excess_history: list[float] = []
        checkpoints = [c for c in simulate.load_checkpoints(run_id) if c["arm"] == arm]
        if checkpoints:
            last = max(checkpoints, key=lambda c: quarters.index(c["quarter_end"])
                       if c["quarter_end"] in quarters else -1)
            state = last["m1d"]["state"]
            agent_a_memory = last.get("agent_a_memory_after", agent_a_memory)
            excess_history = last.get("excess_history_after", [])
            config = simulate.ActiveConfig(**last["config_after"])
            print(f"[{arm}] 從 checkpoint 續跑，上次做到 {last['quarter_end']}")

        checkpoints_end = [c["quarter_end"] for c in checkpoints]
        checkpoints_all = [registration_date] + quarters
        for i, end in enumerate(quarters):
            if end in checkpoints_end:
                print(f"[{arm}] {end} 已有 checkpoint，跳過")
                continue
            as_of = checkpoints_all[i]
            print(f"[{arm}] 執行 {as_of} -> {end} ...")
            result = simulate.run_quarter(inputs, arm=arm, as_of=as_of, end=end,
                                          prev_state=state, prev_weights=prev_weights,
                                          config=config, memory=agent_a_memory,
                                          excess_history=excess_history, dry_run=dry_run)
            state = result["m1d"]["state"]
            prev_weights = result["weights_end"]
            config = simulate.ActiveConfig(**result["config_after"])
            excess_history = excess_history + [result["outcome"]["excess_vs_ball"]]
            result_snapshot = {k: v for k, v in result.items() if k != "weights_end"}
            agent_a_memory = agents.build_agent_a_memory_update(agent_a_memory, result_snapshot)
            result["agent_a_memory_after"] = agent_a_memory
            result["excess_history_after"] = excess_history
            simulate.save_checkpoint(run_id, result)
            print(f"[{arm}] {end} 完成，狀態={state}，決策={result['decision']['decision']}")
