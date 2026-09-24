# -*- coding: utf-8 -*-
"""§4④b第二步：把非官方候選池（rolling window6、或anchored/rolling的exclude_v1
變體，silhouette_is）接進完整實戰監控管線（monitor→triggers→diagnose→
facts_lean→agents→simulate），重跑一次跟9/22報告同一組8季
（2024-03-31~2025-12-31）、同樣的control0/L2兩臂。

🔴 2026-09-24 通用化：原本只認`rolling_window6_silhouette_members.parquet`
一個檔案，現在`build_rolling_inputs()`／`run_rolling_simulation()`都改成吃
`picks_path`／`scheme`／`window_no`參數，才能重用同一套管線接線邏輯去跑
anchored+exclude_v1、rolling+exclude_v1（§3.14），不用複製貼上整支檔案。
官方anchored+baseline（9/22報告那組）不需要這支腳本——直接用`simulate.
run_simulation()`／`_run_formal_8q_v3.py`那條路即可，那才是唯一權威。

🔴 安全設計（沿用§3.3/§4④b拍板的做法，不改`simulate.py`等共用程式碼）：
  - **不呼叫`simulate.run_simulation()`**——那支函式內部寫死讀anchored的
    `walkforward_members.parquet`（scheme="E"／window_no=4）。這裡複製它的
    setup邏輯，只把候選來源換成呼叫端指定的picks parquet，其餘（`monitor`／
    `triggers`／`diagnose`／`facts_lean`／`agents`／`actions`／`weights`模組、
    `simulate.run_quarter`本身）**原封不動重用**。
  - checkpoint用獨立的run_id（呼叫端指定的`run_id_prefix`），跟production的
    `simulate_{run_id}.jsonl`不會撞名，不影響既有anchored那條線的稽核紀錄。

🔴 M1-D門檻不需要重新校準——查過`triggers.py`/設計文件v16→v17：M1-D只看
**指數端集中度**（`mcap_wide`，大盤市值分布），不依賴候選池持股，門檻
X=1.86pp是市場層級的訊號，跟候選池是anchored還是rolling無關，兩條線可以
直接共用同一個`triggers.register_m1d()`結果。

🔴🔴 這支腳本預設`dry_run=True`（不燒真實LLM額度）——`simulate.py`本身的
docstring就警告過「不要在沒有明確授權的情況下用dry_run=False」，這條規則
同樣適用在這裡，真的要跑完整2臂×8季的L2 agent決策前要另外跟使用者確認。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8；先跑過
`_check_rolling_window6_silhouette_picks.py`產生候選池才能跑這支）：
    PYTHONIOENCODING=utf-8 python -m app._check_rolling_production_simulation
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402

from app import monitor, simulate, triggers  # noqa: E402

DEFAULT_PICKS_PATH = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
                      / "rolling_window6_silhouette_members.parquet")


def build_rolling_inputs(model: str, api_key: str, *, picks_path: Path = DEFAULT_PICKS_PATH,
                         tree_key: str = "TW", scheme: str = "rolling_6_2",
                         window_no: int = 6) -> simulate.QuarterInputs:
    """複製`simulate.run_simulation()`的setup邏輯（見該檔291~328行），只把
    `uids_by_allocation`的來源從anchored的`walkforward_members.parquet`
    （scheme="E"/window_no=4）換成呼叫端指定的picks parquet
    （`tree_key`/`scheme`/`window_no`用來在該parquet裡篩對列，不同picks檔
    可能有不同的scheme/window_no標籤，見§3.14各檔案的存檔方式）。"""
    import fcv_core  # noqa: F401  sys.path bootstrap
    from database import Database
    from fcv_core import MarketData
    from resolve_strategy_holdings import CANDIDATE_INDEX_PATH

    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    md = MarketData(tree_key)
    md_map = {tree_key: md}
    db = Database(tree_key)
    conn = db.create_connection()
    mcap_wide = monitor.fetch_mcap_wide(conn, start="2007-01-01")

    tx = pd.read_sql("SELECT date, close FROM taiex_tr ORDER BY date", conn)
    tx["date"] = pd.to_datetime(tx["date"])
    cap_weight_series = tx.set_index("date")["close"].astype(float).sort_index()

    ball_returns = monitor.fetch_ball_returns()

    if not picks_path.exists():
        raise FileNotFoundError(f"找不到 {picks_path}——請先產生這批候選池的silhouette picks")
    m = pd.read_parquet(picks_path)
    uids_by_allocation = {}
    for alloc in ("equal", "proportional"):
        sub = m[(m.tree_key == tree_key) & (m.scheme == scheme) & (m.window_no == window_no)
               & (m.k_mode == "silhouette_is") & (m.allocation == alloc) & (m.group == "A_hrp")]
        assert len(sub) == 1, f"{picks_path.name} 缺 {alloc} 這一列（tree_key={tree_key}, scheme={scheme}, window_no={window_no}）"
        uids_by_allocation[alloc] = list(sub.iloc[0]["members"])

    # M1-D只看指數端集中度（mcap_wide），跟候選池anchored/rolling無關，
    # 直接沿用跟production同一套register_m1d，見本檔docstring說明
    cond = triggers.register_m1d(mcap_wide, simulate.REGISTRATION_DATE)

    return simulate.QuarterInputs(md=md, md_map=md_map, idx=idx, mcap_wide=mcap_wide,
                                  uids_by_allocation=uids_by_allocation, m1d_cond=cond,
                                  model=model, api_key=api_key,
                                  cap_weight_series=cap_weight_series,
                                  ball_returns=ball_returns)


def run_rolling_simulation(*, run_id_prefix: str, model: str, api_key: str, dry_run: bool = True,
                           max_quarters: int | None = None, picks_path: Path = DEFAULT_PICKS_PATH,
                           tree_key: str = "TW", scheme: str = "rolling_6_2",
                           window_no: int = 6) -> None:
    """跟`simulate.run_simulation()`同一套主迴圈骨架（見該檔271~389行），
    只換掉QuarterInputs的建構方式；`run_quarter()`本身完全重用、未修改。

    🔴 `run_id_prefix`一定要由呼叫端明確傳入、dry_run跟真實跑要用不同值——
    2026-09-23 code review抓到：若寫死同一個run_id，dry_run產生的checkpoint
    會讓後續真實跑（dry_run=False）續跑邏輯誤判成「8季全部已完成」而整批
    跳過，不會報錯，會靜默地讓使用者以為跑了一輪真的、其實零次真實LLM呼叫。
    """
    from app import agents

    if not dry_run:
        from utils import openai_quota as OQ
        totals = OQ.today_totals()
        print(f"今日額度用量：{totals}")

    inputs = build_rolling_inputs(model, api_key, picks_path=picks_path, tree_key=tree_key,
                                  scheme=scheme, window_no=window_no)
    quarters = simulate.QUARTER_ENDS[:max_quarters] if max_quarters else simulate.QUARTER_ENDS

    for arm in simulate.ARMS:
        run_id = f"{run_id_prefix}_{arm}"
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
            if "config_after" in last:
                config = simulate.ActiveConfig(**last["config_after"])
            print(f"[{arm}] 從 checkpoint 續跑，上次做到 {last['quarter_end']}")

        checkpoints_end = [c["quarter_end"] for c in checkpoints]
        for i, end in enumerate(quarters):
            if end in checkpoints_end:
                print(f"[{arm}] {end} 已有 checkpoint，跳過")
                continue
            as_of = simulate.REGISTRATION_DATE if i == 0 else quarters[i - 1]
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
            print(f"[{arm}] {end} 完成，狀態={state}，決策={result['decision']['decision']}"
                 f"｜OOS超額(vs B_all)={result['outcome']['excess_vs_ball']:+.2%}")


if __name__ == "__main__":
    # dry_run=True：只驗證管線接得起來、數字算得出來，不燒真實LLM額度。
    # run_id_prefix刻意跟真實跑分開（"rolling_w6_dryrun" vs 之後真實跑要用的
    # "rolling_w6"），避免checkpoint撞名導致真實跑被誤判成「已完成」而整批跳過。
    run_rolling_simulation(run_id_prefix="rolling_w6_dryrun",
                           model="gpt-5", api_key="dry-run-placeholder", dry_run=True)
