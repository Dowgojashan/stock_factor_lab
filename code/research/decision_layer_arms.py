# -*- coding: utf-8 -*-
"""S-05 · 決策層對照組設計（開發待辦追蹤.md 方向C 第三步）

老師的要求（S-05原始記錄）：方向C 必須證明 LLM 決策層比純規則好，否則就只是加
一層成本。至少要三組對照：

  A_rule    純規則基準——「總經狀態X → 固定用群Y」的查表規則，不用LLM
  B_llm     LLM 決策——讀 S-02 的總經狀態+群素材，自己選群
  C_all     全群等權——完全不看總經狀態，永遠用全部群

**若 LLM 贏不了純規則，這是要誠實揭露的結果，不是失敗**（比照學長論文對 Critic
模組的消融實驗：他也誠實報告 Critic「尚無法被視為穩定提升績效的關鍵模組」）。

---------------------------------------------------------------------------
🔴 本模組的範圍邊界——刻意不做的事，以及為什麼
---------------------------------------------------------------------------
S-05 要求的是**對照組設計**，不是完整的無前視偏誤歷史回測：

  - `cluster_macro_conditional.parquet`（S-02）是用**全樣本**算出來的條件式績效。
    拿它去決定「某個過去月份該選哪個群」，等於用了那個月份之後才發生的資訊，
    是前視偏誤——這正是 H-17（上帝視角基準化）／H-18（無未來資訊對照組）要處理
    的問題，**目前仍是開放項目、尚未定案**，本模組不假裝解決它。
  - 決策頻率（多久重新決策一次）是 H-19（時間顆粒度實驗）要定案的事，老師的
    立場是「每月太怪、每季到半年合理」，但**尚未選定季或半年**。

**故本模組只做「靜態單一快照比較」**：給定當下（或指定）的總經狀態，看三組
各自會選出什麼——回答「這三種方法思路上有沒有差異」，不是「回測起來誰的歷史
績效比較好」。要做到後者，須先等 H-19 拍板決策頻率、H-17/H-18 拍板前視偏誤
處理方式，屆時可以直接重用這裡的三個 `*_decision()` 函式接上真正的walk-forward
迴圈，不需要重寫決策邏輯本身。

用法：
    cd code
    python -m research.decision_layer_arms --dry-run                    # 只印prompt不花錢
    python -m research.decision_layer_arms --tree TW_normal --market TW --month 2025-12
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import contracts as C
from . import paths
from .macro_decision_input import group_decision_context, macro_state_snapshot

RESERVE_RATIO = 0.05
#: 單次呼叫的預估token數。決策任務要讀全部候選群的側寫（TW6/US7群，XM3群），
#: 比cluster_identity（單群）大，先用cluster_identity實測值(3627)乘以群數估，
#: 取整數留餘裕；跑完第一批後可依實際usage校正。
EST_TOKENS_PER_CALL = 6_000

_SYSTEM_PROMPT = (
    "你是量化研究系統的總經決策層。你會拿到①目前的總經狀態（z-score與投資時鐘"
    "分類）②候選的策略群清單，每群附上客觀的結構特徵與投資時鐘四格條件式績效。"
    "你的工作是根據目前的總經狀態，從候選群裡選出你認為在這個狀態下適合配置的群。\n"
    "鐵則：\n"
    "1. 只能根據提供的數字做判斷，禁止杜撰未提供的總經事件、產業趨勢、個股消息。\n"
    "2. 你看不到具體日期，不要假裝知道現在是哪一年、發生過什麼歷史事件——只能"
    "根據給定的z-score與clock_cell分類本身做推理。\n"
    "3. 選擇時優先參考『條件式』績效（該群在目前這個clock_cell下的歷史"
    "avg_ret_median/win_ratio），不要只看群的整體結構特徵而忽略條件式數字。\n"
    "4. 至少選1群、最多選全部——rationale必須具體引用你選中/排除某群的理由"
    "（用提供的數字），不能只是空泛地說『這群比較好』。\n"
    "5. 如果所有群在目前clock_cell下的條件式表現都差不多（沒有明顯區分度），"
    "要在caveat中明講，不要硬掰出一個不存在的偏好。"
)

_DECISION_SCHEMA = {
    "name": "macro_decision",
    "schema": {
        "type": "object",
        "properties": {
            "selected_clusters": {
                "type": "array", "items": {"type": "integer"},
                "description": "選中的群id清單，必須是候選清單裡出現過的id，至少1個。"},
            "rationale": {
                "type": "string",
                "description": "為什麼選這些群——須具體引用提供的條件式績效或結構數字。"},
            "caveat": {
                "type": "string",
                "description": "此判讀的限制，例如各群條件式表現差異不大、樣本數不足等。"},
        },
        "required": ["selected_clusters", "rationale", "caveat"],
        "additionalProperties": False,
    },
    "strict": True,
}


# ============================================================================
# A_rule · 純規則基準（無LLM）
# ============================================================================

def rule_based_decision(tree_id: str, clock_cell: str, top_n: int = 1) -> list[int]:
    """固定查表規則：該clock_cell下條件式avg_ret_median最高的top_n個群。"""
    cond = pd.read_parquet(paths.STAGE3 / "cluster_macro_conditional.parquet")
    sub = cond[(cond.tree_id == tree_id) & (cond.clock_cell == clock_cell)]
    sub = sub.dropna(subset=["avg_ret_median"]).sort_values("avg_ret_median", ascending=False)
    return sub["cluster_id"].head(top_n).astype(int).tolist()


# ============================================================================
# C_all · 全群等權（無LLM，不看總經狀態）
# ============================================================================

def equal_weight_all_decision(tree_id: str) -> list[int]:
    assign = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")
    return sorted(assign[assign.tree_id == tree_id]["cluster_L1"].unique().tolist())


# ============================================================================
# B_llm · LLM 決策
# ============================================================================

def _all_cluster_ids(tree_id: str) -> list[int]:
    return equal_weight_all_decision(tree_id)


def build_prompt(tree_id: str, macro_state: dict, group_contexts: dict[int, dict]) -> str:
    return (
        f"樹：{tree_id}\n\n"
        f"【目前總經狀態】\n{json.dumps(macro_state, ensure_ascii=False, indent=2)}\n\n"
        f"【候選群清單】\n"
        f"{json.dumps(group_contexts, ensure_ascii=False, indent=2)}\n\n"
        "請依給定的 JSON schema 輸出，只能使用以上資訊。"
    )


def _call_llm(prompt: str, model: str, api_key: str, *, est_tokens: int = 0) -> tuple[dict, dict]:
    import requests
    from utils import openai_quota as OQ
    OQ.check_free_tier_budget(model, estimated_tokens=est_tokens, reserve_ratio=RESERVE_RATIO)
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model,
             "messages": [{"role": "system", "content": _SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}],
             "response_format": {"type": "json_schema", "json_schema": _DECISION_SCHEMA}},
        timeout=90,
    )
    OQ.raise_for_openai_response(resp)
    body = resp.json()
    usage = body.get("usage", {})
    OQ.log_usage(model, "decision_layer_arm_b", usage)
    return json.loads(body["choices"][0]["message"]["content"]), usage


def llm_decision(tree_id: str, market: str, month: str, *, model: str | None = None,
                 dry_run: bool = False, log=print) -> dict:
    """B_llm：讀 macro_state_snapshot + 全部候選群的 group_decision_context，
    請 LLM 選群。回傳 {"selected_clusters":[...], "rationale":..., "caveat":...}
    （dry_run 時 rationale/caveat 為 "(dry-run)"，selected_clusters 為空list）。
    """
    macro_state = macro_state_snapshot(market, month)
    cluster_ids = _all_cluster_ids(tree_id)
    group_contexts = {cid: group_decision_context(tree_id, cid) for cid in cluster_ids}
    prompt = build_prompt(tree_id, macro_state, group_contexts)

    if dry_run:
        log(f"\n{'='*70}\n{tree_id}｜市場{market}｜月份{month}（僅供查詢用，不會出現在prompt裡）"
            f"\n{'='*70}")
        log(prompt)
        return {"selected_clusters": [], "rationale": "(dry-run)", "caveat": "(dry-run)"}

    from utils.config import Config
    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = model or cfg.get_openai_model("decision_layer_arm_b")
    decision, usage = _call_llm(prompt, model, api_key, est_tokens=EST_TOKENS_PER_CALL)
    log(f"  [{tree_id}] LLM選了群{decision['selected_clusters']}"
        f"｜{usage.get('total_tokens', 0)}tok")
    return decision


# ============================================================================
# 三組對照（單一快照，見模組開頭的範圍邊界說明）
# ============================================================================

def compare_snapshot(tree_id: str, market: str, month: str, *, top_n_rule: int = 1,
                     model: str | None = None, dry_run: bool = False,
                     log=print) -> dict:
    macro_state = macro_state_snapshot(market, month)
    clock_cell = macro_state["clock_cell"]
    if clock_cell is None:
        raise ValueError(f"{market} {month} 沒有clock_cell分類（資料不足或缺值），無法比較")

    a_rule = rule_based_decision(tree_id, clock_cell, top_n=top_n_rule)
    c_all = equal_weight_all_decision(tree_id)
    b_llm = llm_decision(tree_id, market, month, model=model, dry_run=dry_run, log=log)

    result = {
        "tree_id": tree_id, "market": market, "clock_cell": clock_cell,
        "macro_state": macro_state,
        "A_rule": a_rule, "B_llm": b_llm["selected_clusters"],
        "B_llm_rationale": b_llm["rationale"], "B_llm_caveat": b_llm["caveat"],
        "C_all": c_all,
    }
    log(f"\n[{tree_id}｜{market}｜clock_cell={clock_cell}]")
    log(f"  A_rule（規則基準，top{top_n_rule}）：{a_rule}")
    log(f"  B_llm （LLM決策）：{result['B_llm']}")
    log(f"  C_all （全群等權）：{c_all}")
    overlap_ab = set(a_rule) & set(result["B_llm"])
    log(f"  A/B 重疊：{sorted(overlap_ab)}｜A⊆B：{set(a_rule) <= set(result['B_llm'])}")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.decision_layer_arms")
    ap.add_argument("--tree", default="TW_normal")
    ap.add_argument("--market", default="TW")
    ap.add_argument("--month", default="2025-12", help="查表用，不會出現在prompt裡")
    ap.add_argument("--top-n-rule", type=int, default=1)
    ap.add_argument("--model", help="覆寫模型（預設讀 config.ini 的 decision_layer_arm_b_model）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    compare_snapshot(a.tree, a.market, a.month, top_n_rule=a.top_n_rule,
                    model=a.model, dry_run=a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
