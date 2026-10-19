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
from app import weights as weights_mod  # noqa: E402

DEFAULT_PICKS_PATH = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
                      / "rolling_window6_silhouette_members.parquet")


#: 市值加權基準指數表，依市場對應——跟`simulate.run_simulation()`寫死抓
#: `taiex_tr`同一組資料，只是這裡要支援多市場，不能再寫死。美股用`sp500_tr`
#: （已查證資料庫確實有這張表，跟`taiex_tr`同一種「日頻total return指數」結構）。
_CAP_WEIGHT_BENCHMARK_TABLE = {"TW": "taiex_tr", "US": "sp500_tr"}
#: `monitor.fetch_mcap_wide()`的全市場逐日市值起算日——TW historically用
#: 2007-01-01（HRP共同窗起點），US共同窗較早（2002-01），但這裡只是M1-D的
#: 指數端集中度監控用，REGISTRATION_DATE固定是2023-12-31，兩個市場都只需要
#: 這之前幾年的資料建立基準，不需要抓到各自HRP窗的最早起點，維持同一個早期
#: 起算日即可（抓多不影響正確性，只是多抓幾年資料，沒有副作用）。
_MCAP_WIDE_START = "2007-01-01"


def build_rolling_inputs(model: str, api_key: str, *, picks_path: Path = DEFAULT_PICKS_PATH,
                         tree_key: str = "TW", scheme: str = "rolling_6_2",
                         window_no: int = 6, registration_date: str | None = None,
                         with_hot_segment: bool = False,
                         hot_segment_pool: str | None = None) -> simulate.QuarterInputs:
    """複製`simulate.run_simulation()`的setup邏輯（見該檔291~328行），只把
    `uids_by_allocation`的來源從anchored的`walkforward_members.parquet`
    （scheme="E"/window_no=4）換成呼叫端指定的picks parquet
    （`tree_key`/`scheme`/`window_no`用來在該parquet裡篩對列，不同picks檔
    可能有不同的scheme/window_no標籤，見§3.14各檔案的存檔方式）。

    🔴 2026-10-13 修正一個從未被踩到的bug：這支函式雖然早就支援`tree_key`
    參數，但`fetch_mcap_wide()`呼叫沒有傳入`market=tree_key`（悄悄預設用
    monitor.MARKET="TW"），且基準指數查詢寫死抓`taiex_tr`——這兩處只有在
    `tree_key="US"`時才會露出問題，先前只被TW呼叫過，從沒實測踩到。
    這次要把這支函式用在美股驗證，修正後兩處都改成依`tree_key`對應。

    🔴 2026-10-14 新增兩個參數，支援非window4的歷史窗口（例如window2，
    測Coverage Tilt/RepSwap真實觸發用）：
      - `registration_date`：M1-D的登記基準日，None時沿用
        `simulate.REGISTRATION_DATE`（window4的2023-12-31），**不是任何
        窗口通用的日期**，換窗口一定要明確傳對應的OOS起算日前一天
      - `with_hot_segment`：True時額外註冊Hot Segment門檻＋載入HRP分群
        （`weights.load_cluster_membership()`），讓`QuarterInputs`支援
        Coverage Tilt／RepSwap——預設False維持向後相容（美股原本就不需要
        這段，之前這裡完全沒接過）。分群資料固定讀主線`_frozen/stage3/
        cluster_assign.parquet`的`{tree_key}_normal`——這是全時間窗建的樹，
        不隨window_no變化（跟HRP分群本身的設計一致：樹只建一次，walk-forward
        窗口只是從同一棵樹的IS段選代表、OOS段驗證，不是每個窗口各建一棵樹）。
      - `hot_segment_pool`：🔴🔴 2026-10-19新增，`with_hot_segment=True`時
        **必填**（沒有預設值）——Hot Segment門檻現在依`(market, pool)`兩層
        校準（見`triggers.HOT_SEGMENT_P25`），`picks_path`指到baseline還是
        exclude_v1候選池，`hot_segment_pool`就要對應傳"baseline"／
        "exclude_v1"，兩者不保證一致，不接受省略去悄悄猜（之前就是因為
        TW只留了exclude_v1那組門檻、baseline校準完沒地方放，才會有「門檻
        用錯候選池」的風險，見`triggers.py`該段說明）。
    """
    import fcv_core  # noqa: F401  sys.path bootstrap
    from database import Database
    from fcv_core import MarketData
    from resolve_strategy_holdings import CANDIDATE_INDEX_PATH

    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    md = MarketData(tree_key)
    md_map = {tree_key: md}
    db = Database(tree_key)
    conn = db.create_connection()
    mcap_wide = monitor.fetch_mcap_wide(conn, market=tree_key, start=_MCAP_WIDE_START)

    benchmark_table = _CAP_WEIGHT_BENCHMARK_TABLE[tree_key]
    tx = pd.read_sql(f"SELECT date, close FROM {benchmark_table} ORDER BY date", conn)
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
    # 🔴 2026-10-18：market=tree_key明確傳入——門檻已改成按市場分開校準
    # （triggers.X_P90/X_P75），TW/US兩個市場的門檻不一樣，不能漏傳悄悄
    # 退回預設"TW"（美股若漏傳會用錯台股的門檻，跟之前Hot Segment
    # 漏傳market=tree_key同一類bug，見`build_rolling_inputs()`docstring
    # 2026-10-13的教訓）。
    reg_date = registration_date or simulate.REGISTRATION_DATE
    cond = triggers.register_m1d(mcap_wide, reg_date, market=tree_key)

    extra = {}
    if with_hot_segment:
        if hot_segment_pool is None:
            raise ValueError("with_hot_segment=True 但未指定 hot_segment_pool"
                             "（\"baseline\" 或 \"exclude_v1\"）——不接受省略，"
                             "見本函式 docstring 2026-10-19 說明")
        extra["hot_segment_cond"] = triggers.register_hot_segment(tree_key, hot_segment_pool)
        cluster_assign_path = (Path(__file__).resolve().parent.parent.parent
                               / "_frozen" / "stage3" / "cluster_assign.parquet")
        cluster_of, members_by_cluster = weights_mod.load_cluster_membership(
            cluster_assign_path, tree_id=f"{tree_key}_normal")
        extra["cluster_of"] = cluster_of
        extra["members_by_cluster"] = members_by_cluster
        # M2（策略群集中度，2026-10-17接線）：跟simulate.run_simulation()
        # 同一個k來源（stage3_hrp.L1_TARGET），依tree_key取對應市場的群數，
        # 不可寫死TW的6——這支函式TW/US都會呼叫。
        from research.stage3_hrp import L1_TARGET
        extra["m2_cond"] = triggers.register_m2(k=L1_TARGET[tree_key])

    return simulate.QuarterInputs(md=md, md_map=md_map, idx=idx, mcap_wide=mcap_wide,
                                  uids_by_allocation=uids_by_allocation, m1d_cond=cond,
                                  model=model, api_key=api_key, market=tree_key,
                                  cap_weight_series=cap_weight_series,
                                  ball_returns=ball_returns, **extra)


def run_rolling_simulation(*, run_id_prefix: str, model: str, api_key: str, dry_run: bool = True,
                           max_quarters: int | None = None, picks_path: Path = DEFAULT_PICKS_PATH,
                           tree_key: str = "TW", scheme: str = "rolling_6_2",
                           window_no: int = 6, quarter_ends: list[str] | None = None,
                           registration_date: str | None = None,
                           with_hot_segment: bool = False,
                           hot_segment_pool: str | None = None) -> None:
    """跟`simulate.run_simulation()`同一套主迴圈骨架（見該檔271~389行），
    只換掉QuarterInputs的建構方式；`run_quarter()`本身完全重用、未修改。

    🔴 `run_id_prefix`一定要由呼叫端明確傳入、dry_run跟真實跑要用不同值——
    2026-09-23 code review抓到：若寫死同一個run_id，dry_run產生的checkpoint
    會讓後續真實跑（dry_run=False）續跑邏輯誤判成「8季全部已完成」而整批
    跳過，不會報錯，會靜默地讓使用者以為跑了一輪真的、其實零次真實LLM呼叫。

    🔴 2026-10-14新增`quarter_ends`/`registration_date`/`with_hot_segment`：
    原版這裡的季度迴圈跟state tracking曾經只支援M1-D（沒有Hot Segment／
    回撤觸發的跨季state），而且季度清單／登記日都寫死沿用
    `simulate.QUARTER_ENDS`/`REGISTRATION_DATE`（window4的2024-2025）——
    測其他歷史窗口（例如window2，2018-2020，用來真正觸發Coverage Tilt／
    RepSwap）完全沒辦法用。這次補齊：①跨季state tracking對齊`simulate.
    run_simulation()`原生迴圈（hot_state/dd_state/nav/nav_peak/dd_history
    全部跟著續跑／累積）②季度清單與登記日改成參數，None時退回window4的
    既有預設值，保持向後相容。"""
    from app import agents

    if not dry_run:
        from utils import openai_quota as OQ
        totals = OQ.today_totals()
        print(f"今日額度用量：{totals}")

    inputs = build_rolling_inputs(model, api_key, picks_path=picks_path, tree_key=tree_key,
                                  scheme=scheme, window_no=window_no,
                                  registration_date=registration_date,
                                  with_hot_segment=with_hot_segment,
                                  hot_segment_pool=hot_segment_pool)
    reg_date = registration_date or simulate.REGISTRATION_DATE
    all_quarters = quarter_ends or simulate.QUARTER_ENDS
    quarters = all_quarters[:max_quarters] if max_quarters else all_quarters

    for arm in simulate.ARMS:
        run_id = f"{run_id_prefix}_{arm}"
        state = "NONE"
        hot_state = "NONE"
        dd_state = "NONE"
        m2_state = "NONE"
        nav, nav_peak = 1.0, 1.0
        dd_history = pd.Series(dtype=float)
        prev_weights = None
        config = simulate.ActiveConfig()
        agent_a_memory: dict = {"recent_quarter_full": None, "earlier_quarters_summary": []}
        excess_history: list[float] = []
        checkpoints = [c for c in simulate.load_checkpoints(run_id) if c["arm"] == arm]
        if checkpoints:
            last = max(checkpoints, key=lambda c: quarters.index(c["quarter_end"])
                       if c["quarter_end"] in quarters else -1)
            state = last["m1d"]["state"]
            if last.get("hot_segment"):
                hot_state = last["hot_segment"]["state"]
            if last.get("m2"):
                m2_state = last["m2"]["state"]
            if last.get("drawdown"):
                dd_state = last["drawdown"]["state"]
                nav = last["drawdown"]["nav_after"]
                nav_peak = last["drawdown"]["nav_peak_after"]
                dd_points = {c["quarter_end"]: c["drawdown"]["current_dd"]
                            for c in checkpoints if c.get("drawdown")}
                dd_history = pd.Series(dd_points)
                dd_history.index = pd.to_datetime(dd_history.index)
                dd_history = dd_history.sort_index()
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
            as_of = reg_date if i == 0 else quarters[i - 1]
            print(f"[{arm}] 執行 {as_of} -> {end} ...")
            result = simulate.run_quarter(inputs, arm=arm, as_of=as_of, end=end,
                                          prev_state=state, prev_weights=prev_weights,
                                          config=config, memory=agent_a_memory,
                                          excess_history=excess_history, dry_run=dry_run,
                                          prev_hot_state=hot_state, prev_nav=nav,
                                          nav_peak=nav_peak, dd_history=dd_history,
                                          prev_dd_state=dd_state, prev_m2_state=m2_state)
            state = result["m1d"]["state"]
            hot_state = result["hot_segment"]["state"] if result["hot_segment"] else "NONE"
            m2_state = result["m2"]["state"] if result["m2"] else "NONE"
            dd_state = result["drawdown"]["state"]
            nav, nav_peak = result["drawdown"]["nav_after"], result["drawdown"]["nav_peak_after"]
            dd_history[pd.Timestamp(end)] = result["drawdown"]["current_dd"]
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
