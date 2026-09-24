# -*- coding: utf-8 -*-
"""Agent-A（設計文件 §8）。複用 `memo._call_llm`／`memo.scan_for_leakage`
——網路/錯誤處理/額度記帳/D2 掃描不重寫一份（跟 `explain.py` 同一個決定）。

✅ 現況（2026-09-18）：**單一 Agent-A 架構**（3a 回顧診斷、3b 預測評估、
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
  - 🔴 2026-09-19：L3（人機對話）單輪呼叫已補上（`call_agent_a_dialogue_turn`，
    §10 階段6），但迴圈控制（≤5 輪、落盤、session 管理）在 `app/l3_dialogue.py`，
    UI 在 `app/ui.py` 新分頁——這支模組仍只管單次呼叫

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
#:
#: 🔴🔴🔴 2026-09-22（第6次踩到同一個根因，這次直接修根因，不再補特例）：
#: 正式8季重跑撞到「A4皆不對當前已觸發的集中風險採取直接緩解」這句話——
#: "A4"緊貼在中文字「皆」前面（中文書寫本來就不加空格），原本用`\b`當
#: 首尾邊界，但Python的`\b`是Unicode-aware的：中文字元在`\w`的定義裡，
#: 跟英數字一樣算「單字字元」，所以"4"跟"皆"之間**沒有**單字邊界，
#: `\b`在那個位置比對失敗，導致整個"A4"沒被辨識成識別碼、沒被剝掉數字，
#: "4"被當成獨立資料數字送去比對，對不上就誤攔。下面`_DIGIT_QUARTER_RE`
#: 也有同一個根因。`_YEAR_QUARTER_RE`(D33)只用lookbehind錨定，不依賴`\b`，
#: 不受影響。這是同一個根因的第6次變形（前5次：純數字識別碼、括號清單、
#: 句點清單、年份接季度、數字接字母鏡像——每次都補一條新規則，但都沒
#: 處理「`\b`在中文語境下不可靠」這個真正的根因）。**這次改用ASCII專屬
#: 的環顧`(?<![A-Za-z0-9_])`／`(?![A-Za-z0-9_])`取代`\b`**——中文字元不在
#: `[A-Za-z0-9_]`這個字元類別裡，所以緊貼中文字時環顧會正確判定「這裡
#: 就是識別碼的邊界」，不像`\b`會被中文字元誤判成「同一個單字裡面」。
_IDENTIFIER_WITH_DIGIT_RE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z_]*\d{1,2}(?!\d)[A-Za-z_]*(?![A-Za-z0-9_])")

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
#
# 🔴🔴 2026-09-18（正式 8 季重跑撞到，同一類問題的第三個案例）：季度總結
# 階段的 agent 用「1. ⋯2. ⋯3. ⋯」句點式清單編號（不是括號式），原本的規則
# 管不到——用離線測試（不重打真實 API）直接證實「1. 」「3. 」這種句點編號
# 完全沒被中性化。新增第三個分支 `(?<![\d.])\d{1,2}\.(?!\d)`：數字前面不能
# 是數字/小數點（避免從長數字中間切出尾段）、後面接句點、句點後不能立刻
# 接數字（避免誤傷真小數，例如 "0.012" 的 "0." 後面是 "012"，被這個負向
# 前瞻擋下不會誤判成清單編號「0.」；"3.14" 同理不會被誤傷）。
_LIST_MARKER_RE = re.compile(
    r"(?:(?<![\d.])\d{1,2}\)|\(\d{1,2}\)|(?<![\d.])\d{1,2}\.(?!\d))")

# 🔴🔴 2026-09-19（多時間尺度敘事真實呼叫撞到，同一類問題的第四個案例，
# 這次是更危險的「漏網」方向，不是誤攔）：agent 把年份跟季度代號直接連寫
# 成「2024Q3」「2024Q4」（中間沒分隔符）。`_IDENTIFIER_WITH_DIGIT_RE` 要求
# `\b[A-Za-z]` 開頭——但「4Q」是數字接字母，兩者都是 \w，中間沒有單字邊界
# `\b`，所以規則完全比對不到「Q3」「Q4」這段，年份後面貼著的識別碼數字
# 沒被中性化。更嚴重的是：這次「2024Q4」的「4」被攔下了，但「2024Q3」的
# 「3」**沒有被攔下**——因為 facts 別處剛好存在一個真正的數字「3」把它洗白
# 了（跟 D33 講的「更危險的漏網方向」一模一樣，這裡是它的第四個真實案例）。
# 修法：用 lookbehind 精確定位「緊接在 4 碼年份後面」的字母+1~2碼數字，
# 不需要 `\b`（lookbehind 本身就是精確錨點），只清掉季度代號那段的數字，
# 前面的年份完全不動。
_YEAR_QUARTER_RE = re.compile(r"(?<=\d{4})[A-Za-z]\d{1,2}(?!\d)")

# 🔴🔴 2026-09-19（同一次呼叫、修好上面那條後緊接著撞到的第五個案例，
# 鏡像模式）：agent 這次把季度簡寫反過來寫成「1Q、2Q、3Q、4Q」（數字在前、
# 字母在後），不是「Q1~Q4」。`_IDENTIFIER_WITH_DIGIT_RE` 要求字母開頭比對
# 不到；`_YEAR_QUARTER_RE` 要求前面緊接4碼年份也比對不到（這裡是孤立的
# "1Q"，前面沒有年份）。跟之前每一次一樣：這次「1」「2」「4」被攔下，但
# 「3Q」的「3」又被別處真實數字洗白、沒被攔下——確認這一整類「類數字符號
# 洗白」風險是系統性的，不是單一巧合。修法：獨立比對「1~2碼數字+單一字母」
# 這個鏡像形狀，跟字母開頭的規則對稱。
# 🔴🔴🔴 2026-09-22：同一個 \b／CJK 根因（見上方 `_IDENTIFIER_WITH_DIGIT_RE`
# 的說明）——這裡的結尾 `(?![A-Za-z\d])\b` 也會在數字字母組合緊貼中文字時
# 判斷失敗（例如「3Q季」），改用純 ASCII 環顧，不再依賴 `\b`。
_DIGIT_QUARTER_RE = re.compile(r"(?<![A-Za-z0-9_])\d{1,2}[A-Za-z](?![A-Za-z0-9_])")


def _strip_identifier_digits(text: str) -> str:
    """把「以字母開頭、內含 1~2 碼數字」的識別碼（M6、A0、q1_weight、
    m1d_state_csv……）跟「1)／(1) 這類清單編號標記」的數字部分拿掉，避免
    這些非資料性質的類數字符號被 D2 誤判成獨立數字（不論是被誤攔還是被拿來
    洗白捏造數字，見上方說明）。真正的資料數值（以數字開頭的獨立數字、或
    算式裡的小數/長數字）不受影響。"""
    text = _IDENTIFIER_WITH_DIGIT_RE.sub(lambda m: re.sub(r"\d", "", m.group()), text)
    text = _LIST_MARKER_RE.sub(lambda m: re.sub(r"\d", "", m.group()), text)
    text = _YEAR_QUARTER_RE.sub(lambda m: re.sub(r"\d", "", m.group()), text)
    text = _DIGIT_QUARTER_RE.sub(lambda m: re.sub(r"\d", "", m.group()), text)
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


def _check_leakage_or_raise(explanation: dict, prompt: str, *, stage_name: str) -> list[str]:
    """4 個 `call_agent_a_*` 函式共用的 D2 檢查收尾。

    🔴🔴 2026-09-18（正式 8 季重跑連續撞到兩次不同的誤判，第二次因為 LLM
    非決定性重打同一段話沒能重現，只能作罷）：原本攔下後只把 `leakage`
    摘要塞進例外訊息，**沒有印出完整原始文字**，導致每次要查真正的誤判
    原因都得另外花一次真實 API 呼叫去重建情境、賭運氣重現。這裡把完整
    `explanation` 一併印出來（stdout，背景執行的 log 會留存），之後任何
    一種新的誤判類型，直接看這次撞到的當下留下的原始文字即可定位根因，
    不必再賭一次 LLM 會不會巧合寫出同樣的東西。"""
    leakage = _scan_for_leakage_mechanism_aware(explanation, prompt)
    if leakage:
        import json
        print(f"\n🔴🔴 D2 洩漏掃描攔下「{stage_name}」，完整原始輸出如下（供事後查證，"
             f"不用再另外花錢重跑診斷）：\n{json.dumps(explanation, ensure_ascii=False, indent=2)}\n")
        raise RuntimeError(
            f"D2 洩漏掃描攔下這份「{stage_name}」，發現無法對應到資料的數字：\n  "
            + "\n  ".join(leakage))
    return leakage

# ============================================================ Agent-A · 3b 預測評估

_AGENT_A_3B_SYSTEM_PROMPT = (
    "你是一個投組監控系統的分析 agent（Agent-A），現在執行的是「預測評估」"
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
    "6. 這是預測評估，你此刻不知道這一期之後會發生什麼——不可用任何事後"
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
        "請依給定的 JSON schema 輸出預測評估。記住：這裡不選動作、不做預測、"
        "不引用資料外的數字，只客觀描述現況與距門檻多遠。"
    )


def call_agent_a_3b(prospective_facts: dict, *, model: str, api_key: str,
                    purpose: str = "monitor_3b", dry_run: bool = True) -> dict:
    """Agent-A 階段 3b（預測評估）。`dry_run=True`（預設）不呼叫 LLM，只回傳
    佔位輸出方便測試管線組裝是否正確；`dry_run=False` 才會真的燒 token。"""
    prompt = build_3b_prompt(prospective_facts)
    required = _AGENT_A_3B_SCHEMA["schema"]["required"]

    if dry_run:
        explanation = {k: "(dry-run，未呼叫 LLM)" for k in required}
        return {"prompt": prompt, "explanation": explanation, "dry_run": True, "leakage_check": []}

    explanation, usage = _call_llm(
        prompt, model, api_key, purpose=purpose, est_tokens=len(prompt) // 3,
        system_prompt=_AGENT_A_3B_SYSTEM_PROMPT, schema=_AGENT_A_3B_SCHEMA)

    leakage = _check_leakage_or_raise(explanation, prompt, stage_name="3b 預測評估")

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
    "M6／M7／M0）——那是程式依照已凍結的門檻算出來的，你的角色是**解讀為什麼**，"
    "不是重新判定觸不觸發。\n"
    "3. 不可建議或暗示任何具體動作——即使你覺得某個結果很糟，回顧區的結論"
    "**依規定不能拿去驅動任何投組調整**，你只需要客觀解釋已經發生的事。\n"
    "4. 不可自行計算任何未提供的比例或貢獻度——baseline_chain_csv 裡的"
    "pct_of_total 欄位已經算好，可以直接照抄引用，但不可以自己另外推算"
    "資料裡沒有的百分比或倍數關係。\n"
    "5. 若某個機制的判準資料本次沒有提供，明講「本次無法評估」，不要用其他"
    "機制的證據去湊。baseline_chain_csv 的 available 欄位若為 false，"
    "代表這期無法拆解基準鏈，直接明講，不要勉強解讀。"
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
                "description": "解讀 diagnosis_csv 裡各機制（M3／M4／M6／M7／M0）"
                               "的觸發狀態代表什麼意思，只能引用資料裡已經"
                               "判定好的結果，不可重新判定。"},
            "baseline_chain_interpretation": {
                "type": "string",
                "description": "解讀 baseline_chain_csv 的基準鏈拆解（設計文件"
                               "§7.4：A_hrp→B_all→等權大盤→市值加權大盤，"
                               "對應M4/M7/M1-R三段貢獻）——只能照抄裡面已經"
                               "算好的 gap／pct_of_total 數字，不可自行推算。"
                               "若 available 為 false，明講本期無法拆解，"
                               "不要勉強解讀。這段解釋**不代表任何投組調整"
                               "建議**，M1-R是否要調整屬於期初政策（§3.4），"
                               "不是這裡能決定的事。"},
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
                    "baseline_chain_interpretation", "memory_consistency_note", "caveat"],
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

    leakage = _check_leakage_or_raise(explanation, prompt, stage_name="3a 回顧診斷")

    return {"prompt": prompt, "explanation": explanation, "dry_run": False,
           "leakage_check": leakage, "usage": usage}


# ============================================================ Agent-A · 5 決策

# 🔴 §10 階段 5：程式先依 3b 給出基準動作（triggers.py 的 state→action
# 映射），agent 看基準動作＋可用動作的歷史條件分布，決定要不要照做、要不要
# 疊加 A2——不是照抄了事。理由必須明寫依據 3b 的哪幾項；引用 3a 回顧區當
# 理由視為違規。這裡用 `facts_lean.build_decision_facts()` 物理排除 3a／
# outcome／diagnosis，agent 根本看不到回顧區資料，不是只靠 prompt 約束。
# 動作空間固定為 A0／A2／A4／A5／W2c（W1~W3 已於 A5 前置驗證失敗棄用，見
# D29；W2c 於 D50 前置驗證通過、D51 接上執行層，選它現在真的會改變後續
# 季度的持股計算，見 `simulate.ActiveConfig`）。

_AGENT_A_DECISION_SYSTEM_PROMPT = (
    "你是投組監控系統的分析 agent（Agent-A），現在執行的是「決策」"
    "（設計文件 §10 階段 5）。你要從固定的動作空間裡選一個，並說明理由。\n\n"
    "動作空間（只能選這五個之一）：\n"
    "- A0：維持現狀（不改變投組）\n"
    "- A2：切換 allocation（equal↔proportional），只影響持股在群間的配重，"
    "**不解決規模曝險問題**（這是已知限制，見 available_actions_csv 的歷史"
    "資料）\n"
    "- A4：進入觀察名單（不動持股，但列入後續追蹤）\n"
    "- A5：升級人工覆核\n"
    "- W2c：條件式市值傾斜（只在 M1-D 觸發時啟動），已前置驗證通過"
    "（見 w2c_reference_json，開發追蹤 D50），是唯一直接對治規模曝險的"
    "已驗證動作。選它**會真的改變下一季的投組建構**（開發追蹤 D51 已接上"
    "執行層，不是純紀錄的文字）——所以跟其他動作一樣，必須認真評估、不可"
    "隨便選，也不可因為它是新選項就迴避評估；w2c_reference_json 的"
    "caveats 欄位須一併考慮，不可只看正面數字，但 caveats 不是「不要選它」"
    "的理由，是「選的話要在理由裡承認這些限制」\n\n"
    "鐵則：\n"
    "1. 不可引用【客觀資料】以外的任何數字。\n"
    "2. **理由必須明寫依據預測評估（prospective_assessment）的哪幾項**——"
    "你完全看不到回顧區（3a）的內容，這是刻意的物理隔離，不是資料遺漏，"
    "不要因為看不到就猜測或杜撰回顧區可能講了什麼。\n"
    "3. **選 A0 也必須寫理由**——『不動』是一個決定，不是預設值，要說明"
    "為什麼在目前狀態下不動是合理的。\n"
    "4. 若考慮選 A2 或 W2c，只能引用 available_actions_csv／"
    "w2c_reference_json 裡**已經算好**的歷史條件分布或驗證結果，"
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
            "decision": {"type": "string", "enum": ["A0", "A2", "A4", "A5", "W2c"],
                        "description": "選定的動作。"},
            "decision_detail": {
                "type": "string",
                "description": "若選 A2，說明要切換到哪個 ratio/allocation "
                               "組合（須引用 available_actions_csv 裡實際"
                               "存在的組合）；若選 W2c，須引用 "
                               "w2c_reference_json 的驗證結果與 caveats；"
                               "其他動作可留空字串或簡述。"},
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
        "請從動作空間（A0／A2／A4／A5／W2c）選一個，依給定的 JSON schema 輸出"
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

    leakage = _check_leakage_or_raise(explanation, prompt, stage_name="決策")

    return {"prompt": prompt, "explanation": explanation, "dry_run": False,
           "leakage_check": leakage, "usage": usage}


# ============================================================ Agent-A · 6 人機對話（僅 L3）
#
# 🔴🔴 2026-09-19：這個模組原本在檔頭寫「L3（人機對話）介面——不在這個模組
# 範圍內」，那是對的——當時只有 L2（無對話）自動跑完。使用者現在明確要求
# 把 L3 接起來（扩建 `app/ui.py`），這裡只補「一輪對話怎麼呼叫 LLM」這個
# 最小單元，迴圈控制（≤5 輪、落盤、session 管理）交給新的
# `app/l3_dialogue.py`，跟 `call_agent_a_decision()` 只管單次呼叫、迴圈交給
# `simulate.py` 是同一個分工原則。
#
# 設計依據（§6／§10 階段6）：
#   - 決策權：跟 L2 一樣（從動作空間選一個），對話**不能片面推翻**已經做出
#     的階段5決策——對話的作用是讓人向 agent 提問、agent 可以在對話中修正
#     立場，但「最終決策改成什麼」仍是人核准的事，agent 只負責誠實表態
#     「根據這輪對話，我還支不支持原本的草案」，不是自己片面改決策。
#   - 清單外選項：對話中可共同提出，但**必須聲明尚未經程式回測驗證**
#     （§6：「須經程式回測驗證」），不可暗示已驗證或可直接採用。
#   - D2 照掃：`_check_leakage_or_raise` 沿用，不重寫——人類在對話中打的
#     數字會進 prompt（§5.3 講的「對話側門」），D2 天然不會攔對這些數字的
#     合法引用，這是設計預期行為，不是漏洞。

_AGENT_A_DIALOGUE_SYSTEM_PROMPT = (
    "你是投組監控系統的分析 agent（Agent-A），現在是「人機對話」"
    "（設計文件 §10 階段 6，僅 L3 自主等級，最多 5 輪）。系統已經在"
    "階段 5 做出一個決策草案，人類覆核者現在要跟你討論這個決策草案的內容。\n\n"
    "鐵則（§5.1／§6 表格 L3 欄）：\n"
    "1. 不可引用【客觀資料】【階段5決策草案】【對話紀錄】以外的任何數字——"
    "人類在對話中提供的數字可以引用（已經在對話紀錄裡），但不可以自己"
    "杜撰新數字，也不可以自行計算對話紀錄或客觀資料裡沒有的衍生數字。\n"
    "2. 不可片面推翻程式的風控判決。\n"
    "3. 不可片面生成清單外的新動作——若對話中你或人類認為需要清單外的"
    "選項，必須把 proposes_out_of_list_option 設為 true 並在 "
    "out_of_list_option_description 完整說明，且**必須明確聲明這個選項"
    "尚未經程式回測驗證、本次對話不能確認是否可行**，不可暗示它已經驗證"
    "過或可以直接採用。\n"
    "4. 不可做沒有可驗證判準的價值判斷。\n"
    "5. 對話不能片面推翻階段5的決策——你可以在 still_recommends_stage5_"
    "decision 誠實表態這輪對話後你是否還支持原本的草案，並在 "
    "revised_recommendation 說明你會改為建議什麼，但**最終要不要真的改變"
    "決策，是人類覆核者核准的事，不是你自己片面決定**。\n"
    "6. 回覆要聚焦在人類這一輪的問題或意見，不要逐輪重複前面已經講過的"
    "全部內容。"
)

_AGENT_A_DIALOGUE_SCHEMA = {
    "name": "agent_a_dialogue_turn",
    "schema": {
        "type": "object",
        "properties": {
            "response": {"type": "string", "description": "給人類覆核者的回覆內容。"},
            "still_recommends_stage5_decision": {
                "type": "boolean",
                "description": "根據目前為止的對話，是否仍然支持階段5的決策草案。"},
            "revised_recommendation": {
                "type": "string",
                "description": "若 still_recommends_stage5_decision 為 false，"
                               "說明改為建議什麼（須是動作空間內的選項，或"
                               "明確標記為清單外選項）；若為 true，留空字串。"},
            "proposes_out_of_list_option": {
                "type": "boolean",
                "description": "這一輪是否（不論你或人類）共同提出了清單外的"
                               "新選項。"},
            "out_of_list_option_description": {
                "type": "string",
                "description": "若 proposes_out_of_list_option 為 true，描述"
                               "這個選項，且必須包含『尚未經程式回測驗證』的"
                               "聲明；否則留空字串。"},
        },
        "required": ["response", "still_recommends_stage5_decision",
                     "revised_recommendation", "proposes_out_of_list_option",
                     "out_of_list_option_description"],
        "additionalProperties": False,
    },
    "strict": True,
}


def build_dialogue_prompt(decision_facts: dict, draft_decision: dict,
                          transcript: list[dict], human_message: str) -> str:
    import json
    lines = [
        "【客觀資料 · 由程式算出，不可推翻，數字已格式化，請直接照抄】",
        json.dumps(decision_facts, ensure_ascii=False, indent=2, default=str),
        "",
        "【階段5 決策草案 · 已經做出，對話目的是討論它，不是重新決策】",
        json.dumps(draft_decision, ensure_ascii=False, indent=2, default=str),
        "",
        "【對話紀錄（依序，source=human_input 的內容是人類自己提供，"
        "不是程式事實）】",
    ]
    if not transcript:
        lines.append("（尚無對話紀錄，這是第一輪）")
    for turn in transcript:
        lines.append(f"人類（第{turn['round']}輪，source=human_input）："
                     f"{turn['human_message']}")
        lines.append(f"你（第{turn['round']}輪）：{turn['agent_response']['response']}")
    lines += [
        "",
        f"【人類這一輪的訊息（source=human_input，第 {len(transcript) + 1} 輪）】",
        human_message,
        "",
        "請依 schema 回覆。",
    ]
    return "\n".join(lines)


def call_agent_a_dialogue_turn(decision_facts: dict, draft_decision: dict,
                               transcript: list[dict], human_message: str, *,
                               model: str, api_key: str,
                               purpose: str = "monitor_dialogue",
                               dry_run: bool = True) -> dict:
    """§10 階段 6，單輪。迴圈（≤5 輪）與落盤由 `app/l3_dialogue.py` 負責，
    這支函式只管一輪的 LLM 呼叫，跟 `call_agent_a_decision()` 只管單次決策
    呼叫是同一個分工。"""
    prompt = build_dialogue_prompt(decision_facts, draft_decision, transcript, human_message)
    required = _AGENT_A_DIALOGUE_SCHEMA["schema"]["required"]

    if dry_run:
        explanation = {
            "response": "(dry-run，未實際呼叫 LLM)",
            "still_recommends_stage5_decision": True,
            "revised_recommendation": "",
            "proposes_out_of_list_option": False,
            "out_of_list_option_description": "",
        }
        assert set(explanation) == set(required)
        return {"prompt": prompt, "explanation": explanation, "dry_run": True, "leakage_check": []}

    explanation, usage = _call_llm(
        prompt, model, api_key, purpose=purpose, est_tokens=len(prompt) // 3,
        system_prompt=_AGENT_A_DIALOGUE_SYSTEM_PROMPT, schema=_AGENT_A_DIALOGUE_SCHEMA)

    leakage = _check_leakage_or_raise(explanation, prompt, stage_name="人機對話")

    return {"prompt": prompt, "explanation": explanation, "dry_run": False,
           "leakage_check": leakage, "usage": usage}


# ============================================================ Agent-A · 7 季度總結

# 🔴 §10 階段 7（2026-09-18 更新）：Agent-B 移除後，四段式結構變成
# ①程式事實②回顧③預測④人的裁決與理由——原本標題寫「五段式」（含④B的
# 異議原文），現在跟本來就寫「四段式」的另外兩處（§10「每季都要有解釋」
# 表、§12.3①）一致了，之前那個五段/四段不一致的小瑕疵因此自然解決。
#
# **只有②③需要 LLM 生成**：①程式事實是客觀資料，由程式直接附上，不需要
# LLM 重寫；④人的裁決是人核准／對話後才有的內容，L2 臂沒有人對話這個
# 環節，如實記錄「無」。所以這裡的 `call_agent_a_summary()` 只負責生成
# ②③的敘事，①④由呼叫端（`simulate.py`）組裝進最終報告，不經過這支函式。

_AGENT_A_SUMMARY_SYSTEM_PROMPT = (
    "你是投組監控系統的分析 agent（Agent-A），現在執行的是「季度總結」"
    "（設計文件 §10 階段 7）。你要把這一季已經產生的回顧診斷、預測評估、"
    "決策整合成兩段連貫的敘事，給人看的報告——**不是重新分析，是把已經"
    "確定的內容寫成讀得順的段落**。\n\n"
    "鐵則：\n"
    "1. 不可引用【客觀資料】以外的任何數字。\n"
    "2. 不可以在整合的過程中**新增**任何這一季稍早的分析裡沒有的結論、"
    "數字或判斷——你的任務是整合與潤飾，不是重新推論。\n"
    "3. retrospective_section 只能取材自回顧診斷的內容，prospective_section "
    "只能取材自預測評估與決策的內容——不要把兩者的內容混在一起講。\n"
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
                "description": "③【預測】整合自預測評估與決策的內容——當下"
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
        "【這一季的預測評估（③取材來源之一）】\n"
        f"{json.dumps(prospective_output, ensure_ascii=False, indent=2, default=str)}\n\n"
        "【這一季的決策（③取材來源之一）】\n"
        f"{json.dumps(decision_output, ensure_ascii=False, indent=2, default=str)}\n\n"
        "請把以上內容整合成兩段連貫的敘事（回顧／預測），依給定的 JSON schema"
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

    leakage = _check_leakage_or_raise(explanation, prompt, stage_name="季度總結")

    return {"prompt": prompt, "explanation": explanation, "dry_run": False,
           "leakage_check": leakage, "usage": usage}


# ============================================================ Agent-A · 多時間尺度敘事（方案B）

# 🔴🔴 老師9-15「當月→當季→半年→一年」的解釋題（開發追蹤D60）。方案B：
# 底層量測/觸發維持季度不變（已驗證），這裡是**敘事層級彙整**——把已經
# 算好的月頻/半年/年度真實資料整合成連貫敘事，並明確比較「不同尺度講的
# 故事是否一致」（老師原話：「一邊觀察哪邊對、哪邊錯」）。跟階段7季度
# 總結同一個限制：純回顧性質，不驅動任何動作，不可用來支持預測決策。

_AGENT_A_MULTISCALE_SYSTEM_PROMPT = (
    "你是投組監控系統的分析 agent（Agent-A），現在執行的是「多時間尺度"
    "解釋」——老師要求的「當月的解釋題、當季的解釋題、半年的解釋題、"
    "一年的解釋題」。這是**純回顧、事後解釋**，跟階段3a/階段7同一個限制："
    "不建議任何動作、不能用來支持任何預測決策，只整合已經發生、已經算好"
    "的真實資料成連貫敘事。\n\n"
    "鐵則：\n"
    "1. 不可引用【客觀資料】以外的任何數字。\n"
    "2. **這是整合與潤飾，不是重新推論**——不可新增資料裡沒有的結論。\n"
    "3. 若某個時間尺度沒有提供資料（例如這次還不到半年或一年的整理點），"
    "在對應欄位裡明講「本次無此尺度的彙整資料」，不可編造。\n"
    "4. **核心任務**：明確比較不同時間尺度講的故事是否一致——例如某個月"
    "表現異常，是被季度平滑掉了，還是季度也確實反映了？半年/一年看整段"
    "軌跡時，跟逐季分別看時的結論會不會不一樣？**只能引用資料裡已經存在"
    "的數字做這個比較，不可自行推算新的統計量（例如自己算標準差、自己"
    "算相關係數）**——若需要這類統計量但資料沒提供，只能說「無法判斷」。\n"
    "5. 不可做價值判斷（哪個尺度「更重要」），只描述一致或不一致的事實。"
)

_AGENT_A_MULTISCALE_SCHEMA = {
    "name": "agent_a_multiscale_narrative",
    "schema": {
        "type": "object",
        "properties": {
            "monthly_section": {
                "type": "string",
                "description": "當月的解釋題：整合 monthly_breakdown_csv，逐月"
                               "描述這一季內三個月各自的報酬與超額。若未提供"
                               "月頻資料，寫「本次無月頻彙整資料」。"},
            "semiannual_section": {
                "type": "string",
                "description": "半年的解釋題：整合 semiannual_rollup_csv（若"
                               "提供），描述這半年整段軌跡的敘事。若未到半年"
                               "整理點，寫「本次無半年彙整資料」。"},
            "annual_section": {
                "type": "string",
                "description": "一年的解釋題：整合 annual_rollup_csv（若提供），"
                               "描述這一年整段軌跡的敘事。若未到一年整理點，"
                               "寫「本次無年度彙整資料」。"},
            "cross_scale_consistency_note": {
                "type": "string",
                "description": "**核心欄位**：比較月/季/半年/年講的故事是否"
                               "一致——只能引用已提供的數字做比較，不可自行"
                               "計算新的統計量；哪個尺度看起來平滑掉了什麼、"
                               "哪個尺度的結論跟另一個尺度不一樣，只能就資料"
                               "裡已有的數字描述現象，不可下價值判斷。"},
        },
        "required": ["monthly_section", "semiannual_section", "annual_section",
                    "cross_scale_consistency_note"],
        "additionalProperties": False,
    },
    "strict": True,
}


def build_multiscale_prompt(monthly_csv: str | None, semiannual_csv: str | None,
                            annual_csv: str | None, quarterly_context: dict) -> str:
    import json
    parts = [
        "【本季（當季）已有的回顧診斷／決策，供對照，不可重新推論其結論】\n"
        f"{json.dumps(quarterly_context, ensure_ascii=False, indent=2, default=str)}\n",
        "【當月的解釋題 · monthly_breakdown_csv】\n"
        f"{monthly_csv if monthly_csv else '（本次無月頻資料）'}\n",
        "【半年的解釋題 · semiannual_rollup_csv】\n"
        f"{semiannual_csv if semiannual_csv else '（本次無半年彙整資料）'}\n",
        "【一年的解釋題 · annual_rollup_csv】\n"
        f"{annual_csv if annual_csv else '（本次無年度彙整資料）'}\n",
    ]
    return "\n".join(parts) + (
        "\n請依給定的 JSON schema 輸出多時間尺度敘事。記住：純回顧整合，"
        "不建議動作，不引用資料外的數字，不自行計算新的統計量。")


def call_agent_a_multiscale_narrative(monthly_csv: str | None, semiannual_csv: str | None,
                                      annual_csv: str | None, quarterly_context: dict, *,
                                      model: str, api_key: str,
                                      purpose: str = "monitor_multiscale",
                                      dry_run: bool = True) -> dict:
    """多時間尺度解釋（方案B，開發追蹤D60）。至少要提供 monthly_csv（每季都
    該有月頻資料）；semiannual_csv／annual_csv 依報告時點可為 None。"""
    prompt = build_multiscale_prompt(monthly_csv, semiannual_csv, annual_csv, quarterly_context)
    required = _AGENT_A_MULTISCALE_SCHEMA["schema"]["required"]

    if dry_run:
        explanation = {k: "(dry-run，未呼叫 LLM)" for k in required}
        return {"prompt": prompt, "explanation": explanation, "dry_run": True, "leakage_check": []}

    explanation, usage = _call_llm(
        prompt, model, api_key, purpose=purpose, est_tokens=len(prompt) // 3,
        system_prompt=_AGENT_A_MULTISCALE_SYSTEM_PROMPT, schema=_AGENT_A_MULTISCALE_SCHEMA)

    leakage = _check_leakage_or_raise(explanation, prompt, stage_name="多時間尺度敘事")

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
