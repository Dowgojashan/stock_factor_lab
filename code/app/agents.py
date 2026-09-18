# -*- coding: utf-8 -*-
"""Agent-A（設計文件 §8）。複用 `memo._call_llm`／`memo.scan_for_leakage`
——網路/錯誤處理/額度記帳/D2 掃描不重寫一份（跟 `explain.py` 同一個決定）。

✅ 現況（2026-09-18）：**單一 Agent-A 架構**（3a 回顧診斷、3b 前瞻評估、
5 決策、7 總結，共 4 個角色）皆已實作並用真實 LLM 呼叫驗證成功。

🔴🔴 2026-09-18：**Agent-B（質疑者）與 4.5（修訂）已正式移除**（設計文件
v18→v19）。理由（使用者提出疑問、查證後雙方同意，見開發追蹤 D44）：
M1-D 是單一確定性門檻規則，不符合 Ang et al. 設下的「多 agent 需要真正
不同專業＋有意義分歧」門檻；本專案已有真實先例（舊三 Agent 線上系統廢止後，
`decision_layer_arms.py` 完全不用對抗 agent 一樣站得住腳）；今天實測
Agent-B 命中率低，唯一抓到的問題也只是測試資料組錯，不是分析歧義。
**誠實揭露的代價**：型二錯誤（訊號當雜訊放過）沒有專門的守門角色了，
只能靠 A5（升級人工覆核）。

**尚未做的**：
  - `memory`（跨期記憶）已有基本版本（`build_agent_a_memory_update()`），
    但每支呼叫函式仍是無狀態單次呼叫，記憶的組裝/傳遞由呼叫端負責
  - 「受限工具集」互動式版本（§8：`get_attribution`／
    `get_historical_distribution`／`find_similar_quarters`／
    `get_action_reference`，agent 主動查詢）——目前是「單次把 facts 塞進
    prompt」的簡化版，不是 tool-calling
  - L3（人機對話）介面——不在這個模組範圍內

🔴 鐵則（§5.1，適用 L2／L3 兩種自主等級的**強制**版本，取交集最嚴格的，
每個角色各自的 system prompt 都有具體化）：
  a. 不可引用未提供的數字——D2 掃描（`scan_for_leakage`）強制執行，不是自律
  b. 不可推翻程式的風控判決
  c. 不可生成選項——3b／回顧不選動作，那是階段 5 的事
  d. 不可做價值判斷（沒有可驗證判準的情況下）

🔴 §7.6：**不得自行計算任何預期效果**——輸出只能描述「facts 裡已經有的
數字距門檻多遠」，不能自創「如果 xxx 則預期 yyy」這種推算。
"""
from __future__ import annotations

import copy
import re

from .memo import _call_llm, scan_for_leakage

# 🔴🔴 2026-09-17（3a 第一次真實呼叫就抓到，深查後發現範圍比原以為的大）：
# D2 的 `scan_for_leakage` 用泛用數字正則（`memo._NUMBER_RE`）擷取「看起來
# 像數字」的片段，但本系統的識別碼命名慣例（M0～M7、M1-D、A0～A5、W1～W4、
# F1／F2、L1～L3、C2／C5、T8……）大量是「字母＋數字」的組合，欄位名稱本身
# 也是（`q1_weight`、`m1d_state_csv`）。這造成**兩個方向的問題**：
#   ① 誤攔（原本抓到的案例）：LLM 正確寫「diagnosis_csv 未包含 M6 的判定」，
#      但因為 diagnosis_csv 剛好沒有 M6 那一列，代號裡的「6」在 prompt 別處
#      找不到獨立的「6」可對照，被誤判成捏造數字
#   ② 🔴 更嚴重的漏網（追查①時才發現）：欄位名稱本身含的雜散數字
#      （`q1_weight`／`m1d_state_csv` 裡的「1」）會被當成「prompt 裡已提供
#      的合法數字 1」，如果 LLM 剛好捏造了一個等於「1」的數字，D2 完全抓
#      不到——這是比①更危險的方向（該擋的沒擋住，不是該放行的被誤擋）。
#
# 修法：**通用化**，不是只處理 M 開頭代號。任何「以字母開頭、且內含數字」的
# 識別碼（不論是機制代號還是欄位名稱），比對前一律把**數字部分拿掉**，只留
# 字母骨架——這樣代號本身不會貢獻任何「數字」進比對池，不管是被誤攔（案例①）
# 還是被拿來洗白捏造數字（案例②）都解決。真正以數字開頭的資料數值
# （例如 "0.9016"）完全不受影響，因為規則明確要求**以字母開頭**才觸發。
#
# 不動共用的 `scan_for_leakage`／`_NUMBER_RE` 本身（那是 explain.py／memo.py
# 共用、已測試過的核心機制）——這裡在比對前做前處理，範圍限定在本系統。
# ⚠️ **這個盲區很可能也存在於 explain.py 的正式管線**（它的 facts 也有大量
# 同類識別碼，如 F1_factor／cluster_L1），但那是已上線的既有系統，要不要動
# 需要使用者另外決定，這裡不擅自去改。
#: 🔴 第一版（`\d[A-Za-z0-9_]*`，數字長度不設限）測出一個真的會誤刪資料的
#: bug：JSON 把 CSV 內的換行跳脫成字面上的 `\n`（反斜線加字母 n），若這個
#: `n` 緊貼在日期數字前面（例如 `...weight\n2024-09-30...`），會被誤判成
#: 「字母 n 開頭、內含數字 2024 的識別碼」，把日期的「2024」整個吃掉，
#: 讓比對池少了一個真數字，衍生出全新的誤判。本系統真正的識別碼
#: （M6、A0、q1_weight、m1d_state_csv…）數字部分都只有 1~2 碼，用
#: `\d{1,2}(?!\d)` 限制digit run長度、且不可接著更多數字，真正的資料數值
#: （日期、金額這類 3 碼以上的連續數字）就不會被誤吃——用重建過的真實案例
#: 逐一驗證過三種情境才定案，見 `_test_agents_leakage_fix.py`。
_IDENTIFIER_WITH_DIGIT_RE = re.compile(r"\b[A-Za-z][A-Za-z_]*\d{1,2}(?!\d)[A-Za-z_]*\b")

# 🔴🔴 2026-09-18（真實跑 simulate.py 第一季就抓到，同一類根因的新案例）：
# 決策 agent 的 reasoning 用「1) ⋯；2) ⋯；3) ⋯」列點說明理由，這些「1」
# 「2」「3」是**清單編號**，不是數字主張，但 D2 照樣把它們當成獨立數字去比對
# prompt，對不上就攔下——跟識別碼那次是同一種問題（泛用數字正則抓到非
# 資料性質的類數字符號），但這次是編號標記，不是識別碼，`_IDENTIFIER_WITH_
# DIGIT_RE` 管不到（那個規則要求「字母開頭」，"1)" 不是字母開頭）。
#
# 用負向後顧 `(?<![\d.])` 確保比對到的數字不是某個小數/長數字的尾段
# （例如 "0.008)" 這種算式結果，"8" 前面是數字/小數點，不會被誤判成清單
# 編號「8)」而被剝掉）——已用真實案例「(0.012 - 0.008)」這種算式驗證過
# 不會被誤傷，同時 "1)"／"(1)" 這類清單標記會被正確中性化。
_LIST_MARKER_RE = re.compile(r"(?:(?<![\d.])\d{1,2}\)|\(\d{1,2}\))")


def _strip_identifier_digits(text: str) -> str:
    """把「以字母開頭、內含 1~2 碼數字」的識別碼（M6、A0、q1_weight、
    m1d_state_csv……）跟「1)／(1) 這類清單編號標記」的數字部分拿掉，避免
    這些非資料性質的類數字符號被 D2 誤判成獨立數字（不論是被誤攔還是被拿來
    洗白捏造數字，見上方說明）。真正的資料數值（以數字開頭的獨立數字、或
    算式裡的小數/長數字）不受影響。"""
    text = _IDENTIFIER_WITH_DIGIT_RE.sub(lambda m: re.sub(r"\d", "", m.group()), text)
    text = _LIST_MARKER_RE.sub(lambda m: re.sub(r"\d", "", m.group()), text)
    return text


def _scan_for_leakage_mechanism_aware(explanation: dict, prompt: str) -> list[str]:
    """`memo.scan_for_leakage()` 的域特定包裝：比對前先把兩邊的識別碼數字都
    拿掉，其餘邏輯完全不變（真正的資料數字洩漏，這個前處理不影響偵測）。"""
    clean_explanation = copy.deepcopy(explanation)

    def _walk(node):
        if isinstance(node, str):
            return _strip_identifier_digits(node)
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(v) for v in node]
        return node

    clean_explanation = _walk(clean_explanation)
    clean_prompt = _strip_identifier_digits(prompt)
    return scan_for_leakage(clean_explanation, clean_prompt)

# ============================================================ Agent-A · 3b 前瞻評估

_AGENT_A_3B_SYSTEM_PROMPT = (
    "你是一個投組監控系統的分析 agent（Agent-A），現在執行的是「前瞻評估」"
    "（設計文件 §10 階段 3b）。這是整個監控迴圈裡**唯一會驅動實際動作**的"
    "步驟，你的輸出必須完全站得住腳。\n\n"
    "鐵則（違反任何一條，這份輸出會被系統攔下）：\n"
    "1. 不可引用【客觀資料】以外的任何數字——包含你自己推算、估計、或從其他"
    "來源記得的數字都不行。系統會逐字比對你寫的每個數字有沒有出現在資料裡，"
    "對不上就會被攔下。\n"
    "2. 不可推翻或質疑資料裡已經算好的判準（例如 M1-D 的觸發狀態、門檻）——"
    "那是程式依照已凍結的規則算出來的，你的角色是**解讀**，不是重新判定。\n"
    "3. 不可建議或暗示任何具體動作（例如「應該調整 XX」「建議切換到 YY」）——"
    "選擇動作是另一個獨立的階段，且必須依據程式提供的歷史條件分布，不是你"
    "在這裡的判斷。你只需要客觀描述現況。\n"
    "4. 不可自行計算任何「如果這樣則預期那樣」的效果——你沒有被授權做這種"
    "推算，樣本不足或沒有提供時只能寫「資料未提供，無法判斷」，不可外推。\n"
    "5. 若你認為某個指標的資料不足以下判斷，明講「資料不足」，不要硬掰。\n"
    "6. 這是前瞻評估，你此刻不知道這一期之後會發生什麼——不可用任何事後"
    "諸葛的語氣（例如暗示自己已經知道結果）。"
)

_AGENT_A_3B_SCHEMA = {
    "name": "prospective_assessment",
    "schema": {
        "type": "object",
        "properties": {
            "state_summary": {
                "type": "string",
                "description": "目前三層監控指標（環境層／過程層）的狀態摘要——"
                               "只能引用 three_tier_state_csv 裡的數字，客觀描述"
                               "現況，不做價值判斷。"},
            "m1d_interpretation": {
                "type": "string",
                "description": "對 M1-D（指數端集中度偏離登記水準）目前狀態的"
                               "解讀——引用 m1d_state_csv 的累計偏離、門檻、"
                               "目前狀態（NONE／OBSERVING／TRIGGERED），說明"
                               "這代表什麼，不重新判定是否該觸發（那已經是"
                               "程式算好的）。"},
            "distance_to_threshold": {
                "type": "string",
                "description": "針對尚未觸發（或已觸發但想說明餘裕）的指標，"
                               "描述目前數值距離門檻還有多遠——只能引用資料"
                               "裡已提供的數字做加減，不可推算資料裡沒有的"
                               "百分位或門檻。"},
            "caveat": {
                "type": "string",
                "description": "這份評估的限制——例如哪些狀態變數本次沒有"
                               "提供、樣本不足之處，只能根據提供的資訊寫。"},
        },
        "required": ["state_summary", "m1d_interpretation", "distance_to_threshold", "caveat"],
        "additionalProperties": False,
    },
    "strict": True,
}


def build_3b_prompt(prospective_facts: dict) -> str:
    """3b 的 prompt——**只吃 `facts_lean.build_prospective_facts()` 的輸出**，
    這個函式本身沒有能力拿到回顧資料（呼叫端若傳錯物件，缺欄位會在組 prompt
    時就出錯，而不是靜默地被忽略）。"""
    import json
    return (
        "【客觀資料 · 由程式算出，不可推翻，數字已格式化，請直接照抄】\n"
        f"{json.dumps(prospective_facts, ensure_ascii=False, indent=2, default=str)}\n\n"
        "請依給定的 JSON schema 輸出前瞻評估。記住：這裡不選動作、不做預測、"
        "不引用資料外的數字，只客觀描述現況與距門檻多遠。"
    )


def call_agent_a_3b(prospective_facts: dict, *, model: str, api_key: str,
                    purpose: str = "monitor_3b", dry_run: bool = True) -> dict:
    """Agent-A 階段 3b（前瞻評估）。`dry_run=True`（預設）不呼叫 LLM，只回傳
    佔位輸出方便測試管線組裝是否正確；`dry_run=False` 才會真的燒 token。"""
    prompt = build_3b_prompt(prospective_facts)
    required = _AGENT_A_3B_SCHEMA["schema"]["required"]

    if dry_run:
        explanation = {k: "(dry-run，未呼叫 LLM)" for k in required}
        return {"prompt": prompt, "explanation": explanation, "dry_run": True, "leakage_check": []}

    explanation, usage = _call_llm(
        prompt, model, api_key, purpose=purpose, est_tokens=len(prompt) // 3,
        system_prompt=_AGENT_A_3B_SYSTEM_PROMPT, schema=_AGENT_A_3B_SCHEMA)

    leakage = _scan_for_leakage_mechanism_aware(explanation, prompt)
    if leakage:
        raise RuntimeError(
            "D2 洩漏掃描攔下這份 3b 前瞻評估，發現無法對應到資料的數字：\n  "
            + "\n  ".join(leakage))

    return {"prompt": prompt, "explanation": explanation, "dry_run": False,
           "leakage_check": leakage, "usage": usage}


# ============================================================ Agent-A · 3a 回顧診斷

_AGENT_A_3A_SYSTEM_PROMPT = (
    "你是一個投組監控系統的分析 agent（Agent-A），現在執行的是「回顧診斷」"
    "（設計文件 §10 階段 3a）。這一步是**事後解釋**，用的是已經實現的績效"
    "資料——但你的輸出**不可以被用來當作調整投組的依據**，這是純粹的解釋，"
    "不是決策。\n\n"
    "鐵則（違反任何一條，這份輸出會被系統攔下）：\n"
    "1. 不可引用【客觀資料】以外的任何數字——系統會逐字比對，對不上就攔下。\n"
    "2. 不可推翻或質疑資料裡 diagnosis_csv 已經判定好的機制觸發狀態（M3／M4／"
    "M6／M0）——那是程式依照已凍結的門檻算出來的，你的角色是**解讀為什麼**，"
    "不是重新判定觸不觸發。\n"
    "3. 不可建議或暗示任何具體動作——即使你覺得某個結果很糟，回顧區的結論"
    "**依規定不能拿去驅動任何投組調整**，你只需要客觀解釋已經發生的事。\n"
    "4. 不可自行計算任何未提供的比例或貢獻度——例如「這個機制佔了大約六成」"
    "這種話，除非資料裡明確給了這個數字。\n"
    "5. 若某個機制的判準資料本次沒有提供，明講「本次無法評估」，不要用其他"
    "機制的證據去湊。"
)

_AGENT_A_3A_SCHEMA = {
    "name": "retrospective_diagnosis",
    "schema": {
        "type": "object",
        "properties": {
            "outcome_narrative": {
                "type": "string",
                "description": "本期已實現報酬與超額的客觀描述——只能引用"
                               "outcome_csv 裡的數字。"},
            "diagnosis_interpretation": {
                "type": "string",
                "description": "解讀 diagnosis_csv 裡各機制（M3／M4／M6／M0）"
                               "的觸發狀態代表什麼意思，只能引用資料裡已經"
                               "判定好的結果，不可重新判定。"},
            "memory_consistency_note": {
                "type": "string",
                "description": "跟 memory 裡上一期的判定相比，這一期是維持"
                               "還是改變——若引用上一期的判定，必須說明本期"
                               "有什麼新的程式證據支持維持或推翻，不可只憑"
                               "「上次也是這樣判」當理由（循環論證）。若沒有"
                               "上一期資料可比，明講「本次無上一期資料」。"},
            "caveat": {
                "type": "string",
                "description": "這份回顧診斷的限制——例如哪些機制本次沒有"
                               "資料可評估、樣本不足之處。"},
        },
        "required": ["outcome_narrative", "diagnosis_interpretation",
                    "memory_consistency_note", "caveat"],
        "additionalProperties": False,
    },
    "strict": True,
}


def build_3a_prompt(retrospective_facts: dict) -> str:
    """3a 的 prompt——吃 `facts_lean.build_retrospective_facts()` 的輸出
    （含流量變數，這是它跟 3b 的關鍵差異）。"""
    import json
    return (
        "【客觀資料 · 由程式算出，不可推翻，數字已格式化，請直接照抄】\n"
        f"{json.dumps(retrospective_facts, ensure_ascii=False, indent=2, default=str)}\n\n"
        "請依給定的 JSON schema 輸出回顧診斷。記住：這是事後解釋，不是決策，"
        "不建議任何動作，不引用資料外的數字。"
    )


def call_agent_a_3a(retrospective_facts: dict, *, model: str, api_key: str,
                    purpose: str = "monitor_3a", dry_run: bool = True) -> dict:
    """Agent-A 階段 3a（回顧診斷）。跟 3b 是**兩次獨立呼叫**（§7.7／§10
    v14→v15 的改版），3a 的 prompt 絕不能被拿去餵給 3b，反之亦然。"""
    prompt = build_3a_prompt(retrospective_facts)
    required = _AGENT_A_3A_SCHEMA["schema"]["required"]

    if dry_run:
        explanation = {k: "(dry-run，未呼叫 LLM)" for k in required}
        return {"prompt": prompt, "explanation": explanation, "dry_run": True, "leakage_check": []}

    explanation, usage = _call_llm(
        prompt, model, api_key, purpose=purpose, est_tokens=len(prompt) // 3,
        system_prompt=_AGENT_A_3A_SYSTEM_PROMPT, schema=_AGENT_A_3A_SCHEMA)

    leakage = _scan_for_leakage_mechanism_aware(explanation, prompt)
    if leakage:
        raise RuntimeError(
            "D2 洩漏掃描攔下這份 3a 回顧診斷，發現無法對應到資料的數字：\n  "
            + "\n  ".join(leakage))

    return {"prompt": prompt, "explanation": explanation, "dry_run": False,
           "leakage_check": leakage, "usage": usage}


# ============================================================ Agent-A · 5 決策

# 🔴 §10 階段 5：程式先依 3b 給出基準動作（triggers.py 的 state→action
# 映射），agent 看基準動作＋可用動作的歷史條件分布，決定要不要照做、要不要
# 疊加 A2——不是照抄了事。理由必須明寫依據 3b 的哪幾項；引用 3a 回顧區當
# 理由視為違規。這裡用 `facts_lean.build_decision_facts()` 物理排除 3a／
# outcome／diagnosis，agent 根本看不到回顧區資料，不是只靠 prompt 約束。
# 動作空間固定為 A0／A2／A4／A5（W1~W3 已於 A5 前置驗證失敗棄用，見 D29）。

_AGENT_A_DECISION_SYSTEM_PROMPT = (
    "你是投組監控系統的分析 agent（Agent-A），現在執行的是「決策」"
    "（設計文件 §10 階段 5）。你要從固定的動作空間裡選一個，並說明理由。\n\n"
    "動作空間（只能選這四個之一）：\n"
    "- A0：維持現狀（不改變投組）\n"
    "- A2：切換 allocation（equal↔proportional），只影響持股在群間的配重，"
    "**不解決規模曝險問題**（這是已知限制，見 available_actions_csv 的歷史"
    "資料）\n"
    "- A4：進入觀察名單（不動持股，但列入後續追蹤）\n"
    "- A5：升級人工覆核\n\n"
    "鐵則：\n"
    "1. 不可引用【客觀資料】以外的任何數字。\n"
    "2. **理由必須明寫依據前瞻評估（prospective_assessment）的哪幾項**——"
    "你完全看不到回顧區（3a）的內容，這是刻意的物理隔離，不是資料遺漏，"
    "不要因為看不到就猜測或杜撰回顧區可能講了什麼。\n"
    "3. **選 A0 也必須寫理由**——『不動』是一個決定，不是預設值，要說明"
    "為什麼在目前狀態下不動是合理的。\n"
    "4. 若考慮選 A2，只能引用 available_actions_csv 裡**已經算好**的歷史"
    "條件分布數字（例如某個 ratio/allocation 組合過去幾窗的平均表現），"
    "**不可以自己計算或推算『如果選這個，預期能改善多少』**——沒有這種"
    "授權，那是無根據的推算。\n"
    "5. m1d_baseline_action_csv 是程式依規則算出的基準動作，你可以說明"
    "為什麼採用或為什麼偏離，但不能無視它、也不能假裝它建議了別的東西。"
)

_AGENT_A_DECISION_SCHEMA = {
    "name": "agent_a_decision",
    "schema": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["A0", "A2", "A4", "A5"],
                        "description": "選定的動作。"},
            "decision_detail": {
                "type": "string",
                "description": "若選 A2，說明要切換到哪個 ratio/allocation "
                               "組合（須引用 available_actions_csv 裡實際"
                               "存在的組合）；其他動作可留空字串或簡述。"},
            "reasoning": {
                "type": "string",
                "description": "決策理由，必須明寫依據 prospective_assessment "
                               "的哪幾項，以及跟 m1d_baseline_action_csv 的"
                               "關係（採用或偏離，為什麼）。"},
        },
        "required": ["decision", "decision_detail", "reasoning"],
        "additionalProperties": False,
    },
    "strict": True,
}


def build_decision_prompt(decision_facts: dict) -> str:
    import json
    return (
        "【客觀資料 · 由程式算出，不可推翻，數字已格式化，請直接照抄】\n"
        f"{json.dumps(decision_facts, ensure_ascii=False, indent=2, default=str)}\n\n"
        "請從動作空間（A0／A2／A4／A5）選一個，依給定的 JSON schema 輸出"
        "決策與理由。記住：理由只能引用 prospective_assessment 裡的內容，"
        "不可自行計算任何動作的預期效果。"
    )


def call_agent_a_decision(decision_facts: dict, *, model: str, api_key: str,
                          purpose: str = "monitor_decision", dry_run: bool = True) -> dict:
    """Agent-A 階段 5（決策）。`decision_facts` 須用
    `facts_lean.build_decision_facts()` 組出（物理排除 3a／回顧區資料）。"""
    prompt = build_decision_prompt(decision_facts)
    required = _AGENT_A_DECISION_SCHEMA["schema"]["required"]

    if dry_run:
        explanation = {k: "(dry-run，未呼叫 LLM)" if k != "decision" else "A0" for k in required}
        return {"prompt": prompt, "explanation": explanation, "dry_run": True, "leakage_check": []}

    explanation, usage = _call_llm(
        prompt, model, api_key, purpose=purpose, est_tokens=len(prompt) // 3,
        system_prompt=_AGENT_A_DECISION_SYSTEM_PROMPT, schema=_AGENT_A_DECISION_SCHEMA)

    leakage = _scan_for_leakage_mechanism_aware(explanation, prompt)
    if leakage:
        raise RuntimeError(
            "D2 洩漏掃描攔下這份決策，發現無法對應到資料的數字：\n  "
            + "\n  ".join(leakage))

    return {"prompt": prompt, "explanation": explanation, "dry_run": False,
           "leakage_check": leakage, "usage": usage}


# ============================================================ Agent-A · 7 季度總結

# 🔴 §10 階段 7（2026-09-18 更新）：Agent-B 移除後，四段式結構變成
# ①程式事實②回顧③前瞻④人的裁決與理由——原本標題寫「五段式」（含④B的
# 異議原文），現在跟本來就寫「四段式」的另外兩處（§10「每季都要有解釋」
# 表、§12.3①）一致了，之前那個五段/四段不一致的小瑕疵因此自然解決。
#
# **只有②③需要 LLM 生成**：①程式事實是客觀資料，由程式直接附上，不需要
# LLM 重寫；④人的裁決是人核准／對話後才有的內容，L2 臂沒有人對話這個
# 環節，如實記錄「無」。所以這裡的 `call_agent_a_summary()` 只負責生成
# ②③的敘事，①④由呼叫端（`simulate.py`）組裝進最終報告，不經過這支函式。

_AGENT_A_SUMMARY_SYSTEM_PROMPT = (
    "你是投組監控系統的分析 agent（Agent-A），現在執行的是「季度總結」"
    "（設計文件 §10 階段 7）。你要把這一季已經產生的回顧診斷、前瞻評估、"
    "決策整合成兩段連貫的敘事，給人看的報告——**不是重新分析，是把已經"
    "確定的內容寫成讀得順的段落**。\n\n"
    "鐵則：\n"
    "1. 不可引用【客觀資料】以外的任何數字。\n"
    "2. 不可以在整合的過程中**新增**任何這一季稍早的分析裡沒有的結論、"
    "數字或判斷——你的任務是整合與潤飾，不是重新推論。\n"
    "3. retrospective_section 只能取材自回顧診斷的內容，prospective_section "
    "只能取材自前瞻評估與決策的內容——不要把兩者的內容混在一起講。\n"
    "4. prospective_section 必須清楚交代這一季做了什麼決策、理由是什麼——"
    "這是給人看的報告，讀者要能一眼看懂『這一季判斷了什麼、決定怎麼做』。"
)

_AGENT_A_SUMMARY_SCHEMA = {
    "name": "agent_a_quarterly_summary",
    "schema": {
        "type": "object",
        "properties": {
            "retrospective_section": {
                "type": "string",
                "description": "②【回顧】整合自回顧診斷的內容——上一季為什麼"
                               "是這個結果。只能取材自回顧診斷，不驅動動作。"},
            "prospective_section": {
                "type": "string",
                "description": "③【前瞻】整合自前瞻評估與決策的內容——當下"
                               "曝險與容忍度的距離，以及這一季做了什麼決策、"
                               "為什麼。"},
        },
        "required": ["retrospective_section", "prospective_section"],
        "additionalProperties": False,
    },
    "strict": True,
}


def build_summary_prompt(retrospective_output: dict, prospective_output: dict,
                         decision_output: dict) -> str:
    import json
    return (
        "【這一季的回顧診斷（②的取材來源）】\n"
        f"{json.dumps(retrospective_output, ensure_ascii=False, indent=2, default=str)}\n\n"
        "【這一季的前瞻評估（③取材來源之一）】\n"
        f"{json.dumps(prospective_output, ensure_ascii=False, indent=2, default=str)}\n\n"
        "【這一季的決策（③取材來源之一）】\n"
        f"{json.dumps(decision_output, ensure_ascii=False, indent=2, default=str)}\n\n"
        "請把以上內容整合成兩段連貫的敘事（回顧／前瞻），依給定的 JSON schema"
        "輸出。不要新增任何以上內容沒有的結論或數字。"
    )


def call_agent_a_summary(retrospective_output: dict, prospective_output: dict,
                         decision_output: dict, *, model: str, api_key: str,
                         purpose: str = "monitor_summary", dry_run: bool = True) -> dict:
    """Agent-A 階段 7（季度總結，僅生成②③）。①④⑤由呼叫端組裝，不經過
    這支函式——見上方模組內的說明。"""
    prompt = build_summary_prompt(retrospective_output, prospective_output, decision_output)
    required = _AGENT_A_SUMMARY_SCHEMA["schema"]["required"]

    if dry_run:
        explanation = {k: "(dry-run，未呼叫 LLM)" for k in required}
        return {"prompt": prompt, "explanation": explanation, "dry_run": True, "leakage_check": []}

    explanation, usage = _call_llm(
        prompt, model, api_key, purpose=purpose, est_tokens=len(prompt) // 3,
        system_prompt=_AGENT_A_SUMMARY_SYSTEM_PROMPT, schema=_AGENT_A_SUMMARY_SCHEMA)

    leakage = _scan_for_leakage_mechanism_aware(explanation, prompt)
    if leakage:
        raise RuntimeError(
            "D2 洩漏掃描攔下這份季度總結，發現無法對應到資料的數字：\n  "
            + "\n  ".join(leakage))

    return {"prompt": prompt, "explanation": explanation, "dry_run": False,
           "leakage_check": leakage, "usage": usage}


def assemble_quarterly_report(*, program_facts: dict, summary_output: dict,
                              human_ruling: str | None) -> dict:
    """把①程式事實／②③（LLM 生成）／④人的裁決組成最終報告（四段式，
    2026-09-18 隨 Agent-B 移除從五段變回四段，見上方模組內說明）。
    `human_ruling` 在 L2 臂（無人機對話）通常是 None，如實記錄「無」，
    不可以編一個。"""
    return {
        "program_facts": program_facts,
        "retrospective_section": summary_output["retrospective_section"],
        "prospective_section": summary_output["prospective_section"],
        "human_ruling": human_ruling if human_ruling is not None else "（無——本臂無人機對話環節）",
    }


# ============================================================ memory（§8）

# 🔴 §8：「記憶：最近一期全文 ＋ 更早期由**程式產生**的結構化摘要」
# （TradingAgents 的 telephone effect 對策）。關鍵是**更早期的摘要必須是
# 程式產生、格式固定**，不能讓 LLM 每次重新摘要——LLM 重新摘要會逐次失真
# （telephone effect：一路傳話下去，內容會偷偷漂移），程式產生的固定結構
# 摘要每次都是同一個真相來源，不會漂移。
#
# 🔴🔴 2026-09-18：單一 Agent-A 架構下只剩一種記憶（Agent-B 專屬的
# 「歷次質疑／是否被採納／事後是否成立」記憶隨 B 移除一併作廢，
# `build_agent_b_memory_update()` 已刪除）。
#
# 🔴🔴「記憶中 agent 自己的話不得當成事實引用」——這條規則已經寫進 3a 的
# schema（`memory_consistency_note` 的 description），這裡的 memory 結構
# 本身也呼應這個規則：更早期只存**結構化欄位**（狀態、決策、代號），不存
# LLM 當時寫的敘事文字，agent 想引用早期內容時只能引用這些結構化事實，
# 沒有敘事文字可以當「上次已經論證過」的藉口去循環引用。

def _structured_quarter_summary(quarter_result: dict) -> dict:
    """把一季的完整結果壓成程式產生的固定結構摘要（不含 LLM 敘事文字）——
    給「更早期」的記憶用。欄位固定，每次產生規則一樣，不會隨時間漂移。"""
    return {
        "quarter_end": quarter_result["quarter_end"],
        "m1d_state": quarter_result["m1d"]["state"],
        "m1d_deviation": round(quarter_result["m1d"]["cumulative_deviation"], 4),
        "decision": quarter_result["decision"]["decision"],
        "diagnosis_triggered": {
            "M4": quarter_result["diagnosis"]["region_a_attributable"]["M4"]["triggered"],
            "M3": quarter_result["diagnosis"]["region_b_state_warning"]["M3"]["triggered"],
            "fallback": quarter_result["diagnosis"]["fallback"].get("mechanism"),
        },
    }


def build_agent_a_memory_update(prev_memory: dict, quarter_result: dict) -> dict:
    """跑完一季後更新 Agent-A 的記憶：上一次的「最近一期全文」被降級成
    結構化摘要放進 `earlier_quarters_summary`，這一季的全文變成新的
    「最近一期全文」。"""
    earlier = list(prev_memory.get("earlier_quarters_summary", []))
    if prev_memory.get("recent_quarter_full"):
        earlier.append(_structured_quarter_summary(prev_memory["recent_quarter_full"]))
    return {
        "recent_quarter_full": quarter_result,
        "earlier_quarters_summary": earlier,
    }
