# -*- coding: utf-8 -*-
"""監控專用精簡 facts 組裝（設計文件 §12.3）：表格用 CSV、巢狀用 compact JSON。

🔴🔴 最重要的規則（§7.7、§10 階段 3a/3b）：**3b（前瞻）呼叫物理上不得包含任何
流量變數／回顧資料**，不能只靠 prompt 文字告訴 agent「不要用這個」——看得到
就可能被用到。`build_prospective_facts()` 因此有一道**程式層的物理防線**：
組出 dict 之後，逐一檢查 key 名稱有沒有踩到已知的流量變數清單
（`_FLOW_VARIABLE_KEYWORDS`），踩到就直接 `raise`，不是印警告——跟
`actions.py`／`get_action_reference()` 用「資料載入器物理過濾」而非「prompt
約束」防洩題是同一個精神，這裡把它做成執行期斷言，而不是只在文件裡寫規則。

CSV 格式規則（§12.3②，實測省 72% token）：
  - 表格欄位數 ≤ 8（§12.3 的「防線」，避免 LLM 對錯欄位——CSV 的數字離表頭遠，
    這種錯誤 D2 抓不到）
  - 巢狀結構（memory）維持 JSON，但用 compact（去 indent，`separators=(",",":")`)
"""
from __future__ import annotations

import csv
import io
import json

# 流量變數關鍵字——3b 前瞻 facts 絕對不能出現這些（§9.0／§7.0）
_FLOW_VARIABLE_KEYWORDS = (
    "realized_return", "excess_vs", "portfolio_return", "cap_weight_benchmark_return",
    "equal_weight_benchmark_return", "attribution", "basis_chain", "m1_r", "m1-r",
)


def _to_csv(rows: list[dict], columns: list[str] | None = None) -> str:
    """dict 列表轉 CSV 字串。`columns` 未指定時用第一列的 key 順序。
    每 10 列重複一次表頭（§12.3②「列數多時每10列重複一次表頭」的防線）。"""
    if not rows:
        return ""
    cols = columns or list(rows[0].keys())
    if len(cols) > 8:
        raise ValueError(f"表格欄位數 {len(cols)} 超過 8，§12.3 的防線——請拆表或精簡欄位")
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for i, row in enumerate(rows):
        if i > 0 and i % 10 == 0:
            writer.writeheader()
        writer.writerow(row)
    return buf.getvalue()


def _compact_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _assert_no_flow_leakage(d: dict, context: str) -> None:
    """物理防線：遞迴檢查 dict 的 key／CSV 字串值裡有沒有流量變數關鍵字。"""
    def _walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                kl = str(k).lower()
                if any(kw in kl for kw in _FLOW_VARIABLE_KEYWORDS):
                    raise RuntimeError(
                        f"🔴 物理防線觸發（{context}）：key「{path}.{k}」疑似流量變數，"
                        f"不得出現在前瞻區 facts 裡（§7.7）。若這是誤判，請檢查關鍵字表。")
                _walk(v, f"{path}.{k}")
        elif isinstance(node, str):
            low = node.lower()
            for kw in _FLOW_VARIABLE_KEYWORDS:
                if kw in low:
                    raise RuntimeError(
                        f"🔴 物理防線觸發（{context}）：字串內容裡出現流量變數關鍵字"
                        f"「{kw}」，疑似 CSV 欄位名洩漏了流量資料，不得出現在前瞻區。")
    _walk(d)


# ============================================================ 3b：前瞻區（狀態變數，可驅動動作）

def three_tier_state_csv(env: dict, proc: dict) -> str:
    """環境層＋過程層的狀態變數，攤成一張窄表（§9 三層指標裡的狀態變數子集）。"""
    rows = [
        {"metric": "top_n_weight", "value": round(env.get("top_n_weight", float("nan")), 4)},
        {"metric": "q1_weight", "value": round(env.get("q1_weight", float("nan")), 4)},
        {"metric": "n_unique_stocks", "value": proc.get("n_unique_stocks")},
        {"metric": "max_stock_weight", "value": round(proc.get("max_stock_weight", 0.0), 4)},
        {"metric": "median_mktcap_weighted", "value": proc.get("median_mktcap_weighted")},
        {"metric": "continuity_vs_prev", "value": proc.get("continuity_vs_prev")},
    ]
    return _to_csv(rows, columns=["metric", "value"])


def m1d_state_csv(m1d: dict) -> str:
    """M1-D（triggers.py 的 evaluate_quarter 輸出）——這是前瞻區**唯一**可以
    驅動動作的判準（§7.4「M1 的三個角色」）。"""
    rows = [{
        "metric": "q1_weight", "value": round(m1d["q1_weight"], 4),
        "registration": round(m1d["registration_q1_weight"], 4),
        "deviation": round(m1d["cumulative_deviation"], 4),
        "p75": m1d["p75"], "p90": m1d["p90"],
        "state": m1d["state"], "action": m1d["action"],
    }]
    return _to_csv(rows, columns=["metric", "value", "registration", "deviation", "p75", "p90", "state"])
    # 注意：action 欄位刻意不放進 CSV——那是程式已經決定好的動作，不該讓 agent
    # 誤以為那是「建議」而照抄；action 用另一個管道單獨提供給流程控制邏輯。


def available_actions_csv(actions: list[dict]) -> str:
    """`actions.list_available_actions()` 的輸出——只含 §7.7 過濾過、
    看不到 window 4 的歷史窗次資料，本身就是物理隔離的一部分。"""
    rows = [{
        "ratio": a["ratio"], "allocation": a["allocation"],
        "n_windows_visible": a["n_windows_visible"],
        "mean_oos_cagr": round(a["mean_oos_cagr"], 4) if a["mean_oos_cagr"] is not None else None,
    } for a in actions]
    return _to_csv(rows, columns=["ratio", "allocation", "n_windows_visible", "mean_oos_cagr"])


def build_prospective_facts(env: dict, proc: dict, m1d: dict) -> dict:
    """3b：前瞻區 facts。🔴 物理上不含任何流量變數——組完後立刻跑防洩題檢查，
    不是事後補救，是這個函式的必經路徑（呼叫端無法繞過）。

    🔴 2026-09-17（Agent-B 第一次真實質疑就抓到）：原本這裡也塞了
    `available_actions_csv`，但 3b 的角色定義是「只描述現況，不選動作」
    （選動作是階段 5 的事，見設計文件 §10），給了 3b 用不到的資料只會造成
    混淆（B 質疑「3b 完全沒引用這份資料，跟前瞻評估脫節」，抓得對）——已
    移除，`available_actions_csv` 留給階段 5 決策時的 facts 建構式用（尚未
    實作）。
    """
    # 🔴 2026-09-17（同一次 Agent-B 質疑抓到的第二個真問題）：呼叫端曾經把
    # `env`（用季初 as_of 算的）跟 `m1d`（用季末 end 算的）混在同一份「現況」
    # facts 裡卻沒有註明，兩個 q1_weight 對不上（0.8926 vs 0.9016），B 正確
    # 抓到這個不一致。這裡加物理防線：`env["as_of"]` 跟 `m1d["as_of"]` 必須
    # 一致，不一致就直接 raise，不是留給人工事後發現——呼叫端應該統一用
    # 同一個時間點（通常是這一季的 end，最新可得資料）算 env／proc／m1d。
    if env.get("as_of") != m1d.get("as_of"):
        raise ValueError(
            f"env 跟 m1d 的 as_of 不一致（env={env.get('as_of')!r}，"
            f"m1d={m1d.get('as_of')!r}）——同一份前瞻 facts 裡的狀態變數必須是"
            f"同一個時間點的快照，呼叫端請統一用同一個 as_of 算 env／proc／m1d。")

    facts = {
        "three_tier_state_csv": three_tier_state_csv(env, proc),
        "m1d_state_csv": m1d_state_csv(m1d),
    }
    _assert_no_flow_leakage(facts, context="build_prospective_facts")
    return facts


# ============================================================ 3a：回顧區（流量變數，僅供解釋）

def outcome_csv(outcome: dict) -> str:
    """結果層（流量變數）——只能在 3a 回顧區使用（§9.0／§7.0），不可用於 3b。"""
    rows = [{
        "as_of": outcome["as_of"], "end": outcome["end"],
        "portfolio_realized_return": round(outcome["portfolio_realized_return"], 4),
        "excess_vs_equal_weight": round(outcome["excess_vs_equal_weight"], 4)
        if outcome.get("excess_vs_equal_weight") is not None else None,
        "excess_vs_cap_weight": round(outcome["excess_vs_cap_weight"], 4)
        if outcome.get("excess_vs_cap_weight") is not None else None,
    }]
    return _to_csv(rows, columns=["as_of", "end", "portfolio_realized_return",
                                  "excess_vs_equal_weight", "excess_vs_cap_weight"])


def diagnosis_csv(diagnosis: dict) -> str:
    """`diagnose.run_diagnosis()` 的輸出攤平成一張表（M3/M4/M6/M0 的觸發狀態）。"""
    rows = []
    m4 = diagnosis["region_a_attributable"]["M4"]
    rows.append({"mechanism": "M4", "region": "A", "triggered": m4["triggered"],
                 "action": m4.get("action")})
    m3 = diagnosis["region_b_state_warning"]["M3"]
    rows.append({"mechanism": "M3", "region": "B", "triggered": m3["triggered"],
                 "action": m3.get("action")})
    fb = diagnosis["fallback"]
    if fb.get("mechanism"):
        rows.append({"mechanism": fb["mechanism"], "region": "fallback",
                     "triggered": True, "action": fb.get("action")})
    return _to_csv(rows, columns=["mechanism", "region", "triggered", "action"])


def memory_compact_json(memory: dict) -> str:
    """跨期記憶——巢狀結構，保留 JSON 但用 compact（§12.3②）。"""
    return _compact_json(memory)


def m1d_baseline_action_csv(m1d: dict) -> str:
    """§10 階段 5：「程式：依 3b 提供對應動作」——triggers.py 的 state→action
    映射（NONE→A0／OBSERVING→A4／TRIGGERED→A5）當作**基準動作**，決策 agent
    看得到這個基準，但仍要自己判斷要不要照做、要不要疊加 A2（見 §7.5，A2
    不解決 M1，是獨立的策略互補性軸），不是照抄了事。"""
    rows = [{"state": m1d["state"], "baseline_action": m1d["action"]}]
    return _to_csv(rows, columns=["state", "baseline_action"])


# ============================================================ 階段 5：決策

def build_decision_facts(m1d: dict, actions: list[dict], prospective_output: dict) -> dict:
    """階段 5 決策用的 facts：M1-D 基準動作 ＋ 可用動作的歷史條件分布
    （§7.7 過濾過，看不到 window 4）＋ 3b 的完整輸出（決策理由只能引用 3b，
    不可引用 3a——見設計文件 §10「理由必須明寫依據 3b 的哪幾項」）。
    🔴 刻意不放 3a／outcome／diagnosis：物理上讓決策 agent 看不到回顧區資料，
    不靠 prompt 文字約束（跟 §7.7 的一貫精神一致）。"""
    facts = {
        "m1d_baseline_action_csv": m1d_baseline_action_csv(m1d),
        "available_actions_csv": available_actions_csv(actions),
        "prospective_assessment": dict(prospective_output),
    }
    _assert_no_flow_leakage(facts, context="build_decision_facts")
    return facts


def build_retrospective_facts(outcome: dict, diagnosis: dict, memory: dict) -> dict:
    """3a：回顧區 facts。可以含流量變數，但**這區的輸出不得作為動作依據**——
    那是 agent prompt 層與流程設計的約束（§7.0），這個函式只負責組資料，
    不做「防止拿去驅動動作」的執行期檢查（那件事發生在 §10 的流程分岔，
    不是 facts 組裝階段）。"""
    return {
        "outcome_csv": outcome_csv(outcome),
        "diagnosis_csv": diagnosis_csv(diagnosis),
        "memory_compact": memory_compact_json(memory),
    }
