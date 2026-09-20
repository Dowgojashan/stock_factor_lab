# -*- coding: utf-8 -*-
"""模擬主迴圈（設計文件 §10 每季迴圈、§11 實驗設計、§12.4 斷點續跑）。

把今天完成的所有片段串起來：`monitor.py`（三層指標）→ `triggers.py`
（M1-D）→ `diagnose.py`（M3/M4/M6/M0）→ `facts_lean.py`（3a/3b/決策 facts）
→ `agents.py`（3a／3b／5 決策／7 總結——單一 Agent-A 架構，2026-09-18
Agent-B／4.5 已正式移除，見設計文件 v18→v19、開發追蹤 D44）。

🔴🔴 範圍與風險提醒：這支檔案能跑**完整** 3 臂 × 8 季模擬，但那需要上百次
真實 LLM 呼叫（§12.3 估計約 106 次，隨 Agent-B 移除從 138 次降低），是有
意義的真實成本與時間——
**不要在沒有明確授權的情況下呼叫 `run_simulation(..., dry_run=False)` 跑
完整實驗**，先用 `dry_run=True` 或只跑一兩季驗證管線正確，全量實驗另外
跟使用者確認。

🔴 §12.4 斷點續跑（必要功能）：
  1. 每完成一個（臂,季）即落盤 checkpoint（memory／登記條件／已採取動作）
  2. 重啟時從 checkpoint **精確還原**，不可用重新摘要的版本（telephone effect）
  3. 續跑前先查當日額度，不足直接停
  4. checkpoint 落盤路徑跟既有 `code/app/_runs/` 稽核紀錄同一個目錄，
     但**不是**透過 `audit.record()`（那支函式綁定 Holdings／RiskReport，
     是給 replay/live 主線用的；本模組走的是輕量 `resolve_strategy_holdings`
     管線，不是同一套物件，這裡用自己的 JSON Lines 格式，檔名區分開避免
     跟主線的 audit_log.jsonl 混在一起）
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from app import actions, diagnose, facts_lean, monitor, triggers, weights as weights_mod
# 🔴 `utils` 是根目錄套件，需要 root 先進 sys.path（通常靠 `import fcv_core`
# 自動加好，見 CLAUDE.md）。跟 `memo.py:355` 同一個理由，這裡故意不在模組
# 頂層 import，改在用到的地方才 import——模組頂層 import 若在 root 還沒進
# sys.path 前就被觸發（例如被其他還沒 bootstrap 的模組 import），會直接
# ModuleNotFoundError（已用 `_test_simulate_dry_run.py` 實測抓到這個問題）。

RUNS_DIR = Path(__file__).resolve().parent / "_runs"
ARMS = ("control0", "L2")   # L3 需要人機對話，不在自動迴圈範圍內，見下方說明
QUARTER_ENDS = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
                "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
REGISTRATION_DATE = "2023-12-31"

# L3（人機對話）不納入這支檔案的自動迴圈——§10 階段 6「僅 L3，≤5 輪」
# 需要真人在迴圈中對話，無法（也不應該）自動化模擬人的角色。control0／L2
# 兩臂全自動；L3 的執行要另外設計一個互動介面，不是這支批次腳本的範圍。


def checkpoint_path(run_id: str) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR / f"simulate_{run_id}.jsonl"


def load_checkpoints(run_id: str) -> list[dict]:
    """讀回已完成的 (臂,季) checkpoint，用於續跑時跳過已做過的部分。"""
    p = checkpoint_path(run_id)
    if not p.exists():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_checkpoint(run_id: str, checkpoint: dict) -> None:
    """落盤一筆 (臂,季) checkpoint（append，不覆蓋——每一筆都是稽核紀錄）。"""
    p = checkpoint_path(run_id)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(checkpoint, ensure_ascii=False, default=str) + "\n")


def already_done(run_id: str, arm: str, quarter_end: str) -> bool:
    """續跑時判斷這個 (臂,季) 是否已經有 checkpoint，避免重打 LLM。"""
    return any(c["arm"] == arm and c["quarter_end"] == quarter_end
              for c in load_checkpoints(run_id))


def resolve_weights(md, idx, uids: list[str], as_of: str) -> dict[str, float]:
    from resolve_strategy_holdings import resolve_holdings
    n_strat = len(uids)
    weights: dict[str, float] = {}
    for uid in uids:
        row = idx.loc[uid]
        try:
            syms, _ = resolve_holdings(md, row, as_of)
        except RuntimeError:
            continue
        if not syms:
            continue
        w = (1.0 / n_strat) / len(syms)
        for s in syms:
            weights[s] = weights.get(s, 0.0) + w
    return weights


@dataclasses.dataclass
class QuarterInputs:
    """跑一季需要的全部外部依賴，集中一處方便測試時替換成假資料。

    🔴 2026-09-18（開發追蹤 D51，動作執行層）：`uids` 單一清單改成
    `uids_by_allocation`——A2（切換 allocation）要真的換到 proportional
    的名單，不能只換字串標籤。"""
    md: object              # fcv_core.MarketData
    md_map: dict
    idx: object              # candidate_index dataframe
    mcap_wide: object         # monitor.fetch_mcap_wide 的輸出
    uids_by_allocation: dict[str, list[str]]   # {"equal": [...], "proportional": [...]}
    m1d_cond: object          # triggers.M1DCondition
    model: str
    api_key: str
    cap_weight_series: object = None  # taiex_tr 序列，給 outcome_layer 算
                                      # excess_vs_cap_weight 用（code review
                                      # 抓到之前沒接，這欄一直是空的）
    ball_returns: dict = None  # §8待辦item10：真實股票層B_all逐季報酬
                               # （`monitor.fetch_ball_returns()`），給
                               # outcome_layer 算 excess_vs_ball 用真的
                               # B_all，不是借用等權大盤當替身


@dataclasses.dataclass
class ActiveConfig:
    """§7.5「動作具有持續性」——A2 選了之後成為新的基準狀態，A0 維持的是
    這個狀態，不是回到最初的等權配置。

    🔴 W2c 刻意**不**做成持續性旗標（D51 設計決定，見 `weights.py` 的驗證
    範圍說明）：W2c 只在 window 4 觸發期間、逐季重新評估「這季 M1-D 有沒有
    觸發」才驗證過，從沒測過「選一次就一直傾斜到有人明確關掉」這種持續性
    語意——若讓它像 A2 一樣持續，M1-D 一旦回到 NONE，就會在完全沒驗證過的
    情境下（正常年份）繼續傾斜，正是 W2（永遠開啟版）失敗的同一個原因
    （D29）。因此 `apply_w2c` 每季重新判斷：上一季決策是否為 W2c，**且**
    這一季開始時 M1-D 仍是 TRIGGERED，兩者同時成立才套用；M1-D 一旦回到
    NONE，傾斜自動關閉，不需要 agent 額外選一個動作去「關掉」它。"""
    allocation: str = "equal"       # 只有 A2 會改變，其餘動作維持不動
    last_decision_was_w2c: bool = False  # 供下一季判斷是否套用 W2c


def resolve_portfolio_weights(inputs: QuarterInputs, config: ActiveConfig,
                              as_of: str, *, apply_w2c: bool) -> dict[str, float]:
    """依目前的 `ActiveConfig` 解出這個時點的投組權重：先用當前 allocation
    對應的策略名單算出 w0（等權彙總），若 `apply_w2c` 為真再疊加 W2c 傾斜
    （§7.5，D50/D51）。"""
    uids = inputs.uids_by_allocation[config.allocation]
    w0 = resolve_weights(inputs.md, inputs.idx, uids, as_of)
    if not apply_w2c:
        return w0
    mcap_row = monitor.asof_row(inputs.mcap_wide, as_of)
    return weights_mod.apply_w2c(w0, mcap_row)


def run_quarter(inputs: QuarterInputs, *, arm: str, as_of: str, end: str,
                prev_state: str, prev_weights: dict[str, float] | None,
                config: ActiveConfig, memory: dict, excess_history: list[float],
                dry_run: bool = True) -> dict:
    """跑完一季的完整流程（§10 階段 2~7，不含階段 6 人機對話——那是 L3 專屬，
    另外設計）。回傳這一季的完整結果，供落盤 checkpoint 與下一季串接用。

    🔴 時間點規則（見 facts_lean.py／D34）：狀態層（env／proc／m1d）用 `end`
    （季末，最新可得資料）；結果層（outcome）用 `as_of`（季初，測這筆持股
    這段期間表現如何）。呼叫端不可混用。

    🔴 D51（動作執行層）：`config` 是**上一季決策結束後**的 `ActiveConfig`
    （由呼叫端 `run_simulation()` 依上一季的 `decision` 算出），代表「這一季
    的投組要用什麼設定建構」——本函式不自己決定要不要傾斜/換 allocation，
    只負責照著傳進來的 config 建構並回報這一季的結果，決策邏輯留在
    `run_simulation()`（跟 `prev_state`／`prev_weights` 一樣，都是呼叫端算好
    才傳進來的既有慣例）。
    """
    apply_w2c = config.last_decision_was_w2c and prev_state == "TRIGGERED"
    weights_as_of = resolve_portfolio_weights(inputs, config, as_of, apply_w2c=apply_w2c)
    weights_end = resolve_portfolio_weights(inputs, config, end, apply_w2c=apply_w2c)
    mcap_row_end = monitor.asof_row(inputs.mcap_wide, end)
    member_uids = inputs.uids_by_allocation[config.allocation]

    env = monitor.environment_layer(inputs.mcap_wide, end)
    proc = monitor.process_layer(weights_end, mcap_row_end, prev_weights=prev_weights,
                                 member_uids=member_uids, candidate_idx=inputs.idx)
    outcome = monitor.outcome_layer(inputs.md_map, weights_as_of, as_of, end,
                                    cap_weight_series=inputs.cap_weight_series,
                                    ball_returns=inputs.ball_returns)
    m1d = triggers.evaluate_quarter(inputs.m1d_cond, inputs.mcap_wide, end, prev_state)

    # code review 抓到：這裡要用 excess_vs_ball（M4 定義「相對 B_all 超額」的
    # 正確語意），不是 excess_vs_equal_weight——兩者現在數字剛好一樣（A9：
    # B_all＝等權大盤），但引用錯欄位是巧合正確不是設計正確，若未來
    # excess_vs_ball 的算法改用真實 B_all 序列，這裡不改會悄悄讀到錯數字。
    new_excess_history = excess_history + [outcome["excess_vs_ball"]]
    diag = diagnose.run_diagnosis(n_unique_stocks=proc["n_unique_stocks"],
                                  excess_vs_ball_history=new_excess_history,
                                  weights=weights_end)

    quarter_result = {
        "arm": arm, "quarter_end": end, "as_of": as_of,
        "env": env, "proc": proc, "outcome": outcome, "m1d": m1d, "diagnosis": diag,
        "weights_end": weights_end,
        "config_used": {"allocation": config.allocation, "w2c_applied": apply_w2c},
    }

    # control0 臂：完全不調整，不呼叫任何 agent（§11.1「對照0：完全不調整
    # （僅季度重解持股＝現行系統正常行為）」——這是純程式基準線，真的不燒
    # token。code review 抓到：這個判斷原本放在 3a／3b 呼叫**之後**，
    # control0 其實還是燒了兩次真實呼叫，且其輸出從未被使用——已改成在
    # 呼叫任何 agent 之前就提早回傳）
    if arm == "control0":
        quarter_result["decision"] = {"decision": "A0", "decision_detail": "",
                                      "reasoning": "對照 0 臂：定義上完全不調整，不經過 agent 決策。"}
        # control0 定義上永遠維持初始設定（D51）：ActiveConfig 不隨決策演進，
        # 因為根本沒有決策——每季都回傳跟傳進來時一樣的 config，不會漂移。
        quarter_result["config_after"] = dataclasses.asdict(config)
        return quarter_result

    # L2 臂：agent 決策，無對話（§11.1）——3a→3b→5 決策→7 總結
    # 🔴 2026-09-18：Agent-B（質疑）／4.5（修訂）已正式移除（設計文件
    # v18→v19，開發追蹤 D44），3a／3b 的草稿直接進入決策階段，不再有
    # 「修訂版」這個中間產物。
    prospective_facts = facts_lean.build_prospective_facts(env, proc, m1d)
    retrospective_facts = facts_lean.build_retrospective_facts(outcome, diag, memory)

    from app import agents
    result_3b = agents.call_agent_a_3b(prospective_facts, model=inputs.model,
                                       api_key=inputs.api_key, dry_run=dry_run)
    result_3a = agents.call_agent_a_3a(retrospective_facts, model=inputs.model,
                                       api_key=inputs.api_key, dry_run=dry_run)

    quarter_result["prospective_output"] = result_3b["explanation"]
    quarter_result["retrospective_output"] = result_3a["explanation"]

    decision_facts = facts_lean.build_decision_facts(
        m1d, actions.list_available_actions(end), result_3b["explanation"],
        w2c_reference=actions.w2c_reference())
    result_decision = agents.call_agent_a_decision(
        decision_facts, model=inputs.model, api_key=inputs.api_key, dry_run=dry_run)

    result_summary = agents.call_agent_a_summary(
        result_3a["explanation"], result_3b["explanation"], result_decision["explanation"],
        model=inputs.model, api_key=inputs.api_key, dry_run=dry_run)

    # 🔴 D51：這一季的決策決定「下一季」要用什麼 config 建構投組（§7.5
    # 持續性規則——A2 換了 allocation 就一直維持到再被明確換回去；W2c
    # 不持續，見 ActiveConfig 的 docstring）。A0／A4／A5 都不改變 config。
    decision_code = result_decision["explanation"]["decision"]
    next_allocation = config.allocation
    if decision_code == "A2":
        next_allocation = "proportional" if config.allocation == "equal" else "equal"
    config_after = ActiveConfig(allocation=next_allocation,
                                last_decision_was_w2c=(decision_code == "W2c"))

    quarter_result["config_after"] = dataclasses.asdict(config_after)
    quarter_result.update({
        "decision": result_decision["explanation"],
        "summary": agents.assemble_quarterly_report(
            program_facts={"env": env, "proc": proc, "outcome": outcome, "m1d": m1d},
            summary_output=result_summary["explanation"], human_ruling=None),
    })
    return quarter_result


def run_simulation(run_id: str, *, model: str, api_key: str, dry_run: bool = True,
                   max_quarters: int | None = None) -> None:
    """3 臂（control0／L2）× 8 季主迴圈，含斷點續跑。

    🔴 `dry_run=False` 會真的燒 token——完整跑完兩臂 × 8 季估計數十到上百次
    真實呼叫（§12.3），**不要在沒有明確授權的情況下用 dry_run=False 呼叫這支
    函式跑完整迴圈**。`max_quarters` 可以限制只跑前 N 季，方便小規模驗證。
    """
    import fcv_core  # noqa: F401  sys.path bootstrap
    import pandas as pd
    from database import Database
    from fcv_core import MarketData
    from resolve_strategy_holdings import CANDIDATE_INDEX_PATH
    from utils import openai_quota as OQ

    if not dry_run:
        totals = OQ.today_totals()
        print(f"今日額度用量：{totals}（§12.4③ 續跑前先查額度，不足直接停）")

    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    md = MarketData("TW")
    md_map = {"TW": md}
    db = Database("TW")
    conn = db.create_connection()
    mcap_wide = monitor.fetch_mcap_wide(conn, start="2007-01-01")

    # code review 抓到：outcome_layer 支援算 excess_vs_cap_weight，但一直沒接
    # TAIEX 報酬指數序列進去，該欄位在模擬中恆為空——這裡補上（跟今天所有
    # _prelim_*.py 用的 taiex_tr 抓取方式一致）。
    tx = pd.read_sql("SELECT date, close FROM taiex_tr ORDER BY date", conn)
    tx["date"] = pd.to_datetime(tx["date"])
    cap_weight_series = tx.set_index("date")["close"].astype(float).sort_index()

    # §8待辦item10：真實股票層B_all（目前只精確重算了window4/TW，見
    # monitor.fetch_ball_returns docstring）——找不到對應區間時 outcome_layer
    # 會自動退回舊的等權大盤替身，不會崩潰。
    ball_returns = monitor.fetch_ball_returns()

    m = pd.read_parquet(Path(__file__).resolve().parent.parent.parent
                        / "_analysis_outputs_robustness" / "walkforward_members.parquet")
    # 🔴 D51：兩個 allocation 的名單都要抓——A2 決策要真的切換到 proportional
    # 的成員清單，不是只換字串標籤（兩者在 walkforward_members.parquet 裡
    # 本來就是分開的兩列，早就存在，只是先前沒被讀取）。
    uids_by_allocation = {}
    for alloc in ("equal", "proportional"):
        key = dict(tree_key="TW", scheme="E", window_no=4, k_mode="silhouette_is",
                  ratio="legacy", allocation=alloc, group="A_hrp")
        sub = m.copy()
        for k, v in key.items():
            sub = sub[sub[k] == v]
        uids_by_allocation[alloc] = list(sub.iloc[0]["members"])

    cond = triggers.register_m1d(mcap_wide, REGISTRATION_DATE)
    inputs = QuarterInputs(md=md, md_map=md_map, idx=idx, mcap_wide=mcap_wide,
                           uids_by_allocation=uids_by_allocation, m1d_cond=cond,
                           model=model, api_key=api_key,
                           cap_weight_series=cap_weight_series,
                           ball_returns=ball_returns)

    quarters = QUARTER_ENDS[:max_quarters] if max_quarters else QUARTER_ENDS

    from app import agents

    for arm in ARMS:
        state = "NONE"
        prev_weights = None
        config = ActiveConfig()  # D51：每個臂從基準設定重新開始（equal、無傾斜）
        # §8 記憶結構：單一 Agent-A 架構（Agent-B 移除後，見開發追蹤 D44），
        # 只剩 Agent-A 的記憶（最近一期全文＋更早期程式產生的結構化摘要），
        # 見 `agents.build_agent_a_memory_update()`。
        agent_a_memory: dict = {"recent_quarter_full": None, "earlier_quarters_summary": []}
        excess_history: list[float] = []
        checkpoints = [c for c in load_checkpoints(run_id) if c["arm"] == arm]
        if checkpoints:
            last = max(checkpoints, key=lambda c: quarters.index(c["quarter_end"])
                       if c["quarter_end"] in quarters else -1)
            state = last["m1d"]["state"]
            agent_a_memory = last.get("agent_a_memory_after", agent_a_memory)
            excess_history = last.get("excess_history_after", [])
            if "config_after" in last:  # D51：舊 checkpoint（D51 之前跑的）沒有這欄，退回基準設定
                config = ActiveConfig(**last["config_after"])
            print(f"[{arm}] 從 checkpoint 續跑，上次做到 {last['quarter_end']}"
                 f"（記憶已精確還原，非重新摘要——§12.4②；投組設定={config}）")

        checkpoints_end = [c["quarter_end"] for c in checkpoints]
        for i, end in enumerate(quarters):
            if end in checkpoints_end:
                print(f"[{arm}] {end} 已有 checkpoint，跳過")
                continue
            as_of = REGISTRATION_DATE if i == 0 else quarters[i - 1]
            print(f"[{arm}] 執行 {as_of} -> {end} ...")
            result = run_quarter(inputs, arm=arm, as_of=as_of, end=end,
                                 prev_state=state, prev_weights=prev_weights,
                                 config=config, memory=agent_a_memory,
                                 excess_history=excess_history, dry_run=dry_run)
            state = result["m1d"]["state"]
            prev_weights = result["weights_end"]
            config = ActiveConfig(**result["config_after"])
            excess_history = excess_history + [result["outcome"]["excess_vs_ball"]]
            # 🔴 用淺拷貝存進記憶，不能讓 recent_quarter_full 直接參照 result
            # 本身——下面緊接著要把 agent_a_memory_after 寫回 result，若
            # recent_quarter_full 是同一個物件，會變成 result 包含自己的
            # 記憶、記憶又包含 result 自己，形成循環參照，json.dumps 會直接
            # 炸掉（已用 dry_run 測試實測抓到這個問題）。
            #
            # 🔴🔴 code review（3季試跑實測抓到）：`weights_end` 是逐股權重
            # 字典（TW 實測 400~500 檔，JSON 序列化約 13~14k 字元），§8「最近
            # 一期全文」講的是保留 agent 自己寫的敘事（避免 telephone effect），
            # 從來不是要把這張原始數值表也塞進 LLM prompt——3a/3b 的 schema
            # 都沒有要求引用逐股權重，純粹是每季白白多燒 token。checkpoint
            # 本身（`save_checkpoint(run_id, result)`）仍完整保留 weights_end
            # 供稽核，這裡只是餵進 memory 的那份要排除它。
            result_snapshot = {k: v for k, v in result.items() if k != "weights_end"}
            agent_a_memory = agents.build_agent_a_memory_update(agent_a_memory, result_snapshot)
            result["agent_a_memory_after"] = agent_a_memory
            result["excess_history_after"] = excess_history
            save_checkpoint(run_id, result)
            print(f"[{arm}] {end} 完成，狀態={state}，決策={result['decision']['decision']}")
