# -*- coding: utf-8 -*-
"""L3 · 解釋 Agent（應用層開發追蹤.md §9，Phase D 重新設計，2026-09-11）

跟 `memo.py`（D1 六欄位 IC memo，`replay` 用）的根本差異：使用者明確要求
「不要只會統整，要能像學長那樣獨立分析判斷、解釋為什麼這樣操作」（§9 起因）。
`memo.py` 的鐵則「程式判決、LLM 只負責轉述」對這個目標太死板——但這不代表
放棄抗幻覺鐵則，而是照 §9.7-R17 的界線重新畫：

    ✅ **事實推論**：把兩個都餵給它的數字放在一起講出關聯（例如「策略層相關
       0.78 但股票層最大權重僅 1.76%，代表『像』是因為都暴露同一個市場，不是
       買同一批股票」）——這種推論任何人拿到同樣的數字都能覆核，跟
       `cluster_story.py`／投影片本來就在做的事是同一類。
    ❌ **價值判決**：「所以這個組合不好」「應該改用哪個方案」——沒有程式準則
       可覆核，這條線 `memo.py` 的鐵則 7 已經畫過，這裡延續。

🔴 2026-09-15：`live`／`replay` 兩種模式現在都用這支 agent（原本只給 `live`
用，`replay` 走 `memo.py` 六欄位版本）。R15（群檔案全樣本前視）／H8
（walk-forward 每窗 k 不同、群 id 對不上）已透過 `assemble_facts()` 改用
`cluster_kb.build_window_footprint()`（這一窗自己現場重建的樹）解決；R16
（基準涵蓋 62.2%）查證後確認不適用（見 `assemble_facts` 內的說明）。另外
`risk_flags` 欄位的措辭依模式分成兩版（見 `_build_system_prompt`／
`_build_schema`）：`live` 是真正的盲測風險預測，`replay` 因為生成當下已經
知道這一窗真實 OOS 結果，改為誠實定位成「事後機制歸因」，避免用預測式
措辭包裝事後諸葛。

資料組裝需要股票層級持股（`resolve_strategy_holdings` 的輸出）跟群知識庫
（`cluster_kb.build_footprint`），這兩件事目前只有 `ui.py` 走過（需要連
資料庫、載入 `MarketData`），CLI 尚未接——`cli.py` 的 `--memo` 目前對 `live`
模式仍是舊版 `memo.py`（策略層級），這是已知的範圍限制，不是遺漏。
"""
from __future__ import annotations

import copy
import json

import pandas as pd

from . import scenario as SC
from .cluster_kb import ClusterFootprint, build_footprint, build_window_footprint
from .engine import Holdings, alt_group_win_rates, historical_same_setting_windows, mechanism_consistency
from .memo import (_call_llm, _fmt_alt, _fmt_diff, _fmt_perf, _fmt_reference, _fmt_violation,
                   _pct, _ratio, scan_for_leakage)
from .risk import RiskReport, StockConcentration
from .calibration import CalibrationResult
from .audit import diff_holdings, find_previous

RESERVE_RATIO = 0.2

_SYSTEM_PROMPT_HEAD = (
    "你是量化投資系統的資深分析師（不是轉譯器）。你會拿到這一期選股結果的完整"
    "客觀資料：策略層的群知識（身份、機制、歷史績效型態）、股票層的實際持股與"
    "集中度、風控/校準判決、跟情境比對用的歷史數字。\n"
    "你的角色是**做出獨立的分析與判斷**，不是把資料轉述一遍——要能回答"
    "「這批持股是什麼性格」「為什麼這樣挑會賺錢」「現在的情況跟過去哪個時期"
    "像，那時候後來賺賠如何」這類需要綜合判斷的問題，正面地解釋這套方法為什麼"
    "可行、為什麼能賺錢，不要寫成「分散是假的」這種負面診斷語氣。\n\n"
    "但獨立判斷不等於可以杜撰。抗幻覺鐵則的界線畫在這裡：\n"
    "1. **事實推論可以，價值判決不行**。事實推論＝把提供的數字放在一起講出"
    "關聯或型態（例如「策略層相關高但股票層集中度低，代表『像』來自曝險同一個"
    "市場而非買同一批股票」），這種推論任何人拿同樣的數字都能覆核。價值判決＝"
    "「這個組合不好」「應該改用哪個候選方案」，沒有程式準則可覆核，禁止。"
    "是否更換方法屬政策層決定，不在這份分析的權責內。\n"
    "2. **禁止引用未提供的數字**，禁止杜撰個股、產業、總體經濟事件或市場情緒——"
    "包括你自己訓練資料裡可能知道、但沒有出現在下方 JSON 裡的任何事實。\n"
    "3. **禁止推翻或質疑程式給的風控/校準判決**（違規就是違規、通過就是通過）。\n"
    "4. **這是歷史資料與現況的分析，不是預測**——情境比對（scenario）呈現的是"
    "「過去發生過什麼」，禁止寫成「未來會怎樣」「可望達到」等預測性語言，也"
    "禁止用「後來證明」這類暗示你知道後續發展的說法。\n"
    "5. 若有風控違規且已被人工覆核放行，必須如實寫出違規內容跟覆核原因。\n"
    "6. 數字已事先格式化好（百分比/小數位數），請直接照抄，不要自己重新換算。\n"
    "7. **is_* 是樣本內配適值，不是預期報酬**（IS CAGR 中位數歷史上比 OOS 高"
    "1.48 倍）；`live` 模式依定義沒有 OOS，禁止用 is_* 頂替。\n"
    "7b. `mode` 欄位是 `\"live\"`（正式模式，選現在該持有什麼）或 `\"replay\"`"
    "（驗證模式，重現某個歷史時間點）。`replay` 模式下，部分欄位"
    "（例如 `historical_avg_return_by_regime`、群知識庫的身份/機制敘述）會是"
    "`null` 或直接不存在——這是刻意的設計，因為那些統計如果照樣提供，會讓你"
    "在解釋一個歷史時間點時，用到那個時間點當下還沒發生的未來資訊。看到這些"
    "欄位缺席時，直接略過或在 caveat 提一句「此模式未提供」，不要當成資料"
    "遺漏去猜測或补上你自己的版本。\n"
    "8. scenario 區塊裡的情境比對**都只是 n=1 或少量歷史實現**，不是統計推論——"
    "描述型態可以，禁止寫成「所以未來也會這樣」。\n"
    "9. alternative_note 只陳述提供的替代方案數字，若替代方案數字比目前選定的好，"
    "必須把 `alternative_context` 那段定錨事實一併寫進去，禁止建議換方案、"
    "禁止寫「屬於正常波動」（研究已證實是系統性結果）。\n"
    "10. change_note 只描述新增/剔除檔數，禁止推測換股原因。\n"
)

# 🔴 2026-09-15：鐵則 11（risk_flags）改成依模式而異。`live` 模式下 LLM 真的
# 不知道未來，risk_flags 是貨真價實的盲測式風險預測（§9.8 前瞻驗證設計本意）。
# 但 `replay` 模式下 facts 裡的 performance/reference_oos_distribution 已經
# 是這一窗**真實發生過的** OOS 結果——LLM 生成 risk_flags 當下其實已經知道
# 「後來怎麼了」，此時如果沿用「盲測預測＋事後驗證判準」的措辭，內容其實是
# 事後諸葛（先知道結果，再回頭找一個聽起來像預測的說法），跟措辭宣稱的
# 「不知道未來、自訂判準」互相矛盾。拆成兩版：`live` 保留原文，`replay`
# 改為誠實地定位成「事後機制歸因」——不假裝是預測，判準改成「歸因是否成立」
# 的查核方式，而不是「風險有沒有應驗」。JSON 欄位名稱（risk_flags/risk/
# verification_criterion）兩版都不變，避免動到 audit.py／D2／ui.py 既有的
# 欄位相依。
_RISK_FLAGS_RULE_LIVE = (
    "11. **risk_flags（風險提示，供事後檢驗用）**：不限定固定清單，自由指出你"
    "認為這批持股/這套方法在**當下的具體狀況**下最值得留意的風險（可以是"
    "集中度、風格單一化、對特定因子的依賴、規模結構、跟歷史危機期的相似度等"
    "，但必須是**這一期特有的觀察**，不能是每期都能套用的空泛提醒）。"
    "**每一條都必須自帶一個具體、可日後查核的判準**：一句話講清楚「之後要"
    "比對哪個數字/哪個對照組，看到什麼結果就算這個風險應驗了」。判準要在"
    "你完全不知道之後實際表現的情況下自訂——這正是你現在的處境（沒人給你"
    "任何未來資訊），不是要你假裝不知道。\n"
    "⚠️ 這跟鐵則 4「不是預測」並不衝突：判準描述的是「條件式的可驗證關係」"
    "（例如「若大盤由權值股帶動，本組合因低配該權值股會落後——檢驗：比較"
    "市值加權大盤與等權大盤同期報酬，前者顯著較高則此風險應驗」），不是"
    "「本組合將會落後」這種無條件斷言——差別在於你描述的是機制與檢驗方式，"
    "不是替結果打包票。"
)

_RISK_FLAGS_RULE_REPLAY = (
    "11. **risk_flags（這裡不是預測，是事後機制歸因）**：你現在拿到的"
    "performance／reference_oos_distribution 已經是這一窗**真實發生過**的"
    "OOS 結果——你不是在猜未來，是在回頭指出「這批持股/這套方法的哪些特徵"
    "或機制，最可能是造成這個已經發生的結果的原因」（可以是集中度、風格"
    "單一化、對特定因子的依賴、規模結構、跟歷史危機期的相似度等，但必須是"
    "**這一窗特有的觀察**，不能是每一窗都能套用的空泛提醒；也不能倒果為因"
    "地說「因為績效好/壞所以有風險」，要講出具體是哪個機制）。"
    "**每一條都必須自帶一個具體、可查核的歸因判準**：一句話講清楚「要去比對"
    "哪個更細的數字/哪個對照組，看到什麼結果，才能佐證這個機制真的是原因、"
    "不只是巧合」。\n"
    "⚠️ 這跟鐵則 4「不是預測」的關係：這裡本來就不是預測，是對已知結果的"
    "事後解釋，措辭上不要用「會/將會/若⋯則落後」這種面向未來的條件句"
    "（那是 live 模式的寫法），改用「這一窗落後，較可能的機制是⋯」這種"
    "面向過去、解釋已發生結果的句式；判準的作用是讓這個歸因保持可覆核，"
    "不是把它包裝成一次預測。"
)


# 🔴 2026-09-15（§10.10 驗證後發現）：驗證模式九次真呼叫人工檢查後發現
# `mechanism_note` 讀起來像固定不變的行銷文案——同一套「三個機制都很棒」的
# 正面敘事，不管這一窗實際賺賠都照樣講，沒有一句話跟本窗的真實結果對話，
# 導致跟後面 `risk_flags` 解釋「這次為什麼沒賺錢」的內容讀起來像兩份互不
# 相干的文件。兩模式都會發生（不是 replay 專屬問題），故加進共用的鐵則。
_SYSTEM_PROMPT_TAIL = (
    "12. **mechanism_note 要跟本次實際結果對話**：這個欄位講的是方法/機制的"
    "長期價值來源，但若這一窗（或這次執行）的實際 `performance` 明顯偏離"
    "文中描述的型態（例如長期勝率很高，這次卻落在後段班；或反之特別亮眼），"
    "結尾必須用一句話誠實承認「這是機制較弱/較強的一次實現」，不能讓這段話"
    "讀起來跟其他欄位（例如 `risk_flags`）的風險歸因互相矛盾、脫節。"
)


def _build_system_prompt(is_live: bool) -> str:
    return (_SYSTEM_PROMPT_HEAD + (_RISK_FLAGS_RULE_LIVE if is_live else _RISK_FLAGS_RULE_REPLAY)
           + _SYSTEM_PROMPT_TAIL)

def _risk_flags_field(is_live: bool) -> dict:
    if is_live:
        description = ("§9.8 層四用：這一期特有的風險提示，每條自帶事後可"
                       "查核的判準（見系統提示鐵則 11）。1~5 條，不要湊數，"
                       "沒有值得特別提的就少列，不要為了填滿硬掰空泛提醒。")
        risk_desc = ("這一期特有的具體風險描述，只能根據"
                    "提供的資訊寫，不能引用未提供的事實。")
        crit_desc = ("之後要比對哪個數字/"
                    "哪個對照組、看到什麼結果就算這個"
                    "風險應驗——具體到可以照著執行。")
    else:
        description = ("這一窗特有的事後機制歸因（不是預測——生成當下已知這一窗"
                       "實際的 OOS 結果，見系統提示鐵則 11）。1~5 條，不要湊數，"
                       "沒有值得特別提的就少列，不要為了填滿硬掰空泛提醒。")
        risk_desc = ("這一窗特有的具體機制歸因，只能根據"
                    "提供的資訊寫，不能引用未提供的事實，不能只是複述績效"
                    "好壞而不指出機制。")
        crit_desc = ("要去比對哪個更細的數字/哪個對照組，看到什麼結果，"
                    "才能佐證這個機制真的是原因、不只是巧合——"
                    "具體到可以照著執行。")
    return {
        "type": "array",
        "description": description,
        "minItems": 1, "maxItems": 5,
        "items": {
            "type": "object",
            "properties": {
                "risk": {"type": "string", "description": risk_desc},
                "verification_criterion": {"type": "string", "description": crit_desc},
            },
            "required": ["risk", "verification_criterion"],
            "additionalProperties": False,
        },
    }


_EXPLAIN_SCHEMA = {
    "name": "portfolio_explanation",
    "schema": {
        "type": "object",
        "properties": {
            "strategy_footprint_note": {
                "type": "string",
                "description": "【一之1｜策略層】這批策略落在主線樹哪些群、各群身份"
                               "（identity_label）、機制（mechanism_note）與定量特徵"
                               "（CAGR/MDD 中位、主力因子）；只能引用提供的群知識，"
                               "不能自己重新詮釋群的意義。"},
            "stock_holdings_note": {
                "type": "string",
                "description": "【一之2｜股票層，老師 9-8 原話①】幾檔不重複股票、"
                               "最大持股權重、跨策略重疊度最高的前幾檔。"},
            "mechanism_note": {
                "type": "string",
                "description": "【二｜為什麼這樣挑會賺錢】依三機制數字（稀有群超配"
                               "倍數、對 B_all 的 CAGR 超額分布與勝率、三組候選方案"
                               "的 OOS MDD 排名）正面解釋這套規則的價值來源，"
                               "只能引用提供的數字。"},
            "character_note": {
                "type": "string",
                "description": "【三｜這批的性格，老師 9-8 原話④】從主力 F1 因子曝險"
                               "（動能 vs 品質估值）跟新舊面孔比例（若有比較資料）"
                               "判斷這批持股偏向「追火」還是「一直賺」，沒有新舊面孔"
                               "資料時明講「本次未比較，只能從因子面判斷」。"},
            "concentration_risk_note": {
                "type": "string",
                "description": "【四之1/2｜集中度風險】股票層最大權重與是否違規、"
                               "特定權值股對照（若提供）；並點出策略層相關高不等於"
                               "股票層集中（把兩個層級的數字放在一起講出這個關聯，"
                               "屬於允許的事實推論）。"},
            "structure_risk_note": {
                "type": "string",
                "description": "【四之3/4/5｜結構風險】平時 vs 危機期群間相關的變化、"
                               "多樣性機制（backfill）是否實際生效、選用的幾個群彼此"
                               "結構重疊程度（story 裡的 complementarity/co_fail）。"},
            "scenario_note": {
                "type": "string",
                "description": "【五｜情境比對，老師 9-8 原話③】綜合四種比對數字"
                               "（群報酬型態最像哪一年、同設定歷史窗次實際 OOS 結果、"
                               "現在的 regime 標籤與歷史平均表現、現有股票在極端年份"
                               "的實際表現）描述現況跟過去哪些時期相似、那些時期後續"
                               "賺賠如何——只能描述型態，禁止預測。"},
            "change_note": {
                "type": "string",
                "description": "跟上一次同一組設定的執行結果相比，新增/剔除了幾檔"
                               "策略；沒有上一筆可比就明講「這是第一次執行」。"},
            "alternative_note": {
                "type": "string",
                "description": "同一格設定下，其他候選方案的檔數與 IS 績效數字，"
                               "並附上 alternative_context 的定錨事實，不建議換方案。"},
            "caveat": {
                "type": "string",
                "description": "本份分析的限制：情境比對是少量歷史實現不是統計推論、"
                               "is_* 非預期報酬、群知識庫止於 2025 年底、正式模式無 "
                               "OOS 等，只能根據提供的資訊寫。"},
            "risk_flags": _risk_flags_field(is_live=True),
        },
        "required": ["strategy_footprint_note", "stock_holdings_note", "mechanism_note",
                    "character_note", "concentration_risk_note", "structure_risk_note",
                    "scenario_note", "change_note", "alternative_note", "caveat",
                    "risk_flags"],
        "additionalProperties": False,
    },
    "strict": True,
}


# 🔴 2026-09-15（§10.10 驗證後發現）：多檢查點驗證（同一窗次、不同 as_of
# 各呼叫一次）實測發現，11 欄位裡有 6 個根本不吃 as_of（群層/機制/風控上限
# 這些是策略層固定屬性，同一窗次不管哪個 as_of 都算出同一份 facts），逼
# LLM 對著同一份資料重講三次，讀起來高度重複；只有這裡列的 5 個欄位會
# 隨 as_of（股票層持股、regime、個股類比）真的變。拆成兩組，讓多檢查點
# 驗證可以「窗次層只呼叫一次、檢查點層才逐點呼叫」，不強迫 LLM 對沒有
# 新資訊的欄位硬掰出差異（硬要求每次都不一樣反而會誘發杜撰，違反鐵則 1）。
# 一般 UI 單次呼叫的 `generate()` 預設 `fields=None`＝全部 11 欄位一次要齊，
# 行為完全不變，這只是加一個可選的呼叫方式。
WINDOW_FIELDS = ["strategy_footprint_note", "mechanism_note", "structure_risk_note",
                 "alternative_note", "change_note", "caveat"]
CHECKPOINT_FIELDS = ["stock_holdings_note", "character_note", "concentration_risk_note",
                     "scenario_note", "risk_flags"]


def _build_schema(is_live: bool, fields: list[str] | None = None) -> dict:
    """§10.7：`risk_flags` 欄位描述依模式而異（見 `_build_system_prompt`
    同一段說明），其餘欄位兩模式共用，故用淺層複製只換掉那一個欄位，
    不整份重新定義一次。

    `fields`（§10.10）：只保留這幾個欄位（`WINDOW_FIELDS`／`CHECKPOINT_FIELDS`
    之一，或呼叫端自訂子集），`None` 就是全部 11 欄位（預設行為，`generate()`
    原本的單次呼叫用這個，不受影響）。
    """
    schema = (_EXPLAIN_SCHEMA if (is_live and fields is None)
             else copy.deepcopy(_EXPLAIN_SCHEMA))
    schema["schema"]["properties"]["risk_flags"] = _risk_flags_field(is_live)
    if fields is not None:
        all_props = schema["schema"]["properties"]
        unknown = [f for f in fields if f not in all_props]
        if unknown:
            raise ValueError(f"_build_schema 收到未知的欄位名稱：{unknown}")
        schema["schema"]["properties"] = {k: all_props[k] for k in fields}
        schema["schema"]["required"] = list(fields)
    return schema


def _cluster_facts(footprint: ClusterFootprint) -> list[dict]:
    out = []
    for cid in footprint.clusters_used:
        prof = footprint.profile.get(cid, {})
        ident = footprint.identity.get(cid, {})
        out.append({
            "cluster_id": cid,
            "n_selected": footprint.counts[cid],
            "n_in_universe": footprint.total_in_tree.get(cid),
            "share_of_this_portfolio": _pct(footprint.share(cid)),
            "identity_label": ident.get("identity_label"),
            "mechanism_note": ident.get("mechanism_note"),
            "performance_pattern": ident.get("performance_pattern"),
            "CAGR_median": _pct(prof["CAGR_median"]) if prof.get("CAGR_median") is not None else None,
            "MDD_median": _pct(prof["MDD_median"]) if prof.get("MDD_median") is not None else None,
            "top1_F1": prof.get("top1_F1"),
            "top1_F1_pct": _pct(prof["top1_F1_pct"]) if prof.get("top1_F1_pct") is not None else None,
            "top1_C_source": prof.get("top1_C_source"),
            "pct_years_positive": _pct(prof["pct_years_positive"]) if prof.get("pct_years_positive") is not None else None,
            "best_year": prof.get("best_year"), "best_year_ret": _pct(prof["best_year_ret"]) if prof.get("best_year_ret") is not None else None,
            "worst_year": prof.get("worst_year"), "worst_year_ret": _pct(prof["worst_year_ret"]) if prof.get("worst_year_ret") is not None else None,
        })
    return out


def _rarest_cluster_protection(footprint: ClusterFootprint, n_universe: int | None) -> dict | None:
    """機制一：稀有但真正不同的群，實際被保障超配了多少倍（依這次真正選中的
    members 現算，不是套用投影片上寫死的 TW 數字）。"""
    if not footprint.clusters_used or not n_universe:
        return None
    total_selected = sum(footprint.counts.values())
    rows = []
    for cid in footprint.clusters_used:
        n_univ = footprint.total_in_tree.get(cid)
        if not n_univ:
            continue
        share = n_univ / n_universe
        expected = total_selected * share
        actual = footprint.counts[cid]
        rows.append({
            "cluster_id": cid, "share_in_universe": _pct(share),
            "n_in_universe": n_univ, "expected_if_random": round(expected, 2),
            "actual_selected": actual,
            "overweight_multiple": round(actual / expected, 1) if expected > 0 else None,
        })
    if not rows:
        return None
    rarest = min(rows, key=lambda r: r["n_in_universe"])
    return {"per_cluster": rows, "rarest_cluster": rarest}


def _fmt_mechanism_consistency(m: dict | None) -> dict | None:
    """🔴 2026-09-12（§9.8 正式驗證第一次真呼叫抓到的真 bug）：`mechanism_consistency()`
    回傳的是**未格式化的原始小數**（例如 `-0.14554961969523622`），但系統提示
    鐵則 6 要求「數字已事先格式化好，請直接照抄，不要自己重新換算」——原本
    這裡沒套 `_pct()`，等於違反自己訂的前提。實測後果：同一份未格式化資料，
    LLM 有時原樣照抄小數（比對得上），有時「好心」幫忙換算成百分比字串
    （比對不上，被 D2 誤判成洩漏數字而攔下一份其實正確的解釋）。統一格式化
    後兩種情況都不會再發生。"""
    if m is None:
        return None
    out = {"n_cells": m["n_cells"]}
    for k in ("cagr_excess_vs_Ball_min", "cagr_excess_vs_Ball_max",
             "cagr_excess_vs_Ball_median", "win_rate_vs_Ball"):
        out[k] = _pct(m[k])
    for k in ("oos_mdd_median_A_hrp", "oos_mdd_median_D_top_cagr", "oos_mdd_median_B_all"):
        if k in m:
            out[k] = _pct(m[k])
    return out


def _fmt_historical_windows(rows: list[dict]) -> list[dict]:
    """同上一個函式的理由：`historical_same_setting_windows()` 也是原始小數，
    這裡統一格式化，不要指望 LLM 自己判斷該不該換算。"""
    return [
        {"scheme": r["scheme"], "window_no": r["window_no"], "k_mode": r["k_mode"],
         "is_end": r["is_end"], "oos_start": r["oos_start"], "oos_end": r["oos_end"],
         "oos_cagr": _pct(r["oos_cagr"]), "oos_mdd": _pct(r["oos_mdd"]),
         "oos_sharpe": _ratio(r["oos_sharpe"])}
        for r in rows
    ]


def _fmt_stock_analog(d: dict | None) -> dict | None:
    """同 `_fmt_mechanism_consistency` 的理由：`scenario.stock_level_analog()`
    的 `year_return`／`avg_return_of_existing` 也是原始小數，統一格式化。"""
    if d is None:
        return None
    per_stock = {s: {"existed_then": v["existed_then"],
                     "year_return": _pct(v["year_return"]) if v["year_return"] is not None else None}
                for s, v in d["per_stock"].items()}
    return {"year": d["year"], "n_checked": d["n_checked"], "n_existed_then": d["n_existed_then"],
           "avg_return_of_existing": (_pct(d["avg_return_of_existing"])
                                      if d["avg_return_of_existing"] is not None else None),
           "per_stock": per_stock}


def _fmt_alt_context(d: dict | None) -> dict | None:
    """同上：`engine.alt_group_win_rates()` 也是原始未四捨五入的小數
    （例如 `0.16777777777777778`），統一格式化——不只是為了 D2，也是因為
    這種一長串小數點直接出現在最終解釋文字裡本身就不像正式產出。

    🔴 2026-09-14 code review 修正後才發現的連帶真 bug：`alt_group_win_rates()`
    的 `baseline`／`others` 兩個新欄位是字串／清單，不是數字，原本這裡對
    `d.items()` 裡「除了 n_cells 以外全部」都呼叫 `_pct()`，會直接對字串
    `.2%` 格式化拋 `ValueError`——這支函式從沒被涵蓋進先前的修正驗證，
    真呼叫第一次就炸了。改成只格式化 `{metric}_vs_{group}` 這種數值欄位，
    `baseline`／`others` 原樣帶過去（LLM 需要知道基準是哪個方法）。
    """
    if d is None:
        return None
    out = {"n_cells": d["n_cells"], "baseline": d["baseline"], "others": d["others"]}
    for k, v in d.items():
        if k not in ("n_cells", "baseline", "others"):
            out[k] = _pct(v)
    return out


def _fmt_regime(d: dict | None) -> dict | None:
    """同上：`scenario.regime_snapshot()` 的 `pct_change_so_far`／
    `historical_avg_return_by_regime` 也是原始小數，統一格式化。

    🔴 2026-09-15：`historical_avg_return_by_regime` 在驗證模式會是 `None`
    （見 `scenario.regime_snapshot` 的說明——那項統計對驗證窗次有前視風險，
    刻意不提供），這裡要能處理，不能對 `None` 呼叫 `.items()`。
    """
    if d is None:
        return None
    current = {}
    for m, v in d["current_regime"].items():
        if "error" in v:
            current[m] = v
        else:
            current[m] = {**v, "pct_change_so_far": _pct(v["pct_change_so_far"])}
    hist = d["historical_avg_return_by_regime"]
    return {"current_regime": current,
           "historical_avg_return_by_regime": (
               {k: _pct(v) for k, v in hist.items()} if hist is not None else None),
           "note": d["note"]}


def _structure_facts(footprint: ClusterFootprint) -> dict:
    used = footprint.clusters_used
    pairs_normal, pairs_crisis = [], []
    for i, a in enumerate(used):
        for b in used[i + 1:]:
            pairs_normal.append(footprint.pair_corr(a, b, crisis=False))
            c = footprint.pair_corr(a, b, crisis=True)
            if c is not None:
                pairs_crisis.append(c)
    story_rows = footprint.story.to_dict("records") if not footprint.story.empty else []
    # 🔴 §9.8 正式驗證同一輪抓到：crisis_dest_share 是原始小數（危機期轉往目的群
    # 的成員佔比），跟 mechanism_consistency／historical_same_setting_windows
    # 同一個問題，統一格式化。
    co_fail_rows = {}
    for cid in used:
        row = footprint.co_fail.get(cid)
        if not row:
            continue
        row = dict(row)
        if row.get("crisis_dest_share") is not None:
            row["crisis_dest_share"] = _pct(row["crisis_dest_share"])
        co_fail_rows[cid] = row
    return {
        "clusters_used": used,
        "avg_pair_corr_normal": round(sum(pairs_normal) / len(pairs_normal), 4) if pairs_normal else None,
        "avg_pair_corr_crisis": round(sum(pairs_crisis) / len(pairs_crisis), 4) if pairs_crisis else None,
        "pair_story": story_rows,
        "co_fail_regimes": co_fail_rows,
    }


def assemble_facts(holdings: Holdings, risk: RiskReport, calib: CalibrationResult, *,
                   stock_detail, stock_concentration: StockConcentration,
                   market_dates: dict[str, str] | None = None,
                   footprint: ClusterFootprint | None = None,
                   md_map: dict | None = None,
                   face_comparison=None) -> dict:
    """組裝解釋 agent 要用的全部事實。

    🔴 2026-09-15：**兩種模式都支援**——原本只接受 `mode="live"`，因為 R15
    （群知識庫全樣本前視）／R16（基準涵蓋 62.2%）／H8（walk-forward 每窗 k
    不同、群 id 對不上）三個問題只存在於 `replay` 模式。現在的處理方式：
        - R15/H8：`replay` 模式改用 `build_window_footprint()`（這一窗自己
          現場重建的樹，只用這一窗 IS 期間內的資料算群統計，不讀主線樹的
          全樣本群知識庫）取代 `build_footprint()`，見 `cluster_kb.py` 該
          函式的說明。
        - R16：查證後確認**不適用**——這裡的基準比較用的是
          `reference_oos_distribution()`（依 market/group/ratio/allocation
          聚合，任何存在的設定組合都至少有自己這一格可用）跟自建宇宙基準
          `BENCHMARK_CAGR`，不是 R16 講的「等權大盤固定 2019-2025 窗」那組
          （那組目前沒有被 explain.py 使用），故沒有覆蓋率缺口的問題。
        - 實作時另外發現兩個 R15 同類、但原本 R15/R16/H8 清單沒點名的前視
          點，一併處理：`mechanism_consistency()`／`historical_same_setting_
          windows()` 都改成只納入 `oos_end`／`is_end` 不晚於這一窗自己
          `is_end` 的窗次；`risk.regime_avg_ret`（T8，讀全樣本 regime
          統計）對 `replay` 一律不提供（見 `scenario.regime_snapshot` 的
          說明）。
    """
    cfg = holdings.config
    is_live = cfg.mode == "live"
    as_of_is_end = None if is_live else holdings.window_info["is_end"]

    if footprint is None:
        if is_live:
            footprint = build_footprint(cfg.market, holdings.members)
        else:
            from . import clustering as CL
            months_long, _meta, _f_combo_map = CL.load_inputs(log=lambda *a, **k: None)
            window_tree = CL.build_window_tree(
                cfg.market, holdings.window_info["is_start"], holdings.window_info["is_end"],
                months_long, _meta, _f_combo_map, log=lambda *a, **k: None)
            from resolve_strategy_holdings import CANDIDATE_INDEX_PATH
            candidate_idx = pd.read_parquet(CANDIDATE_INDEX_PATH)
            footprint = build_window_footprint(window_tree, holdings.members, candidate_idx)

    overlap = (stock_detail.drop_duplicates(["strategy_uid", "stock_symbol"])
              .groupby(["stock_symbol", "company_name"])["strategy_uid"]
              .nunique().reset_index(name="n_strategies")
              .sort_values("n_strategies", ascending=False).head(10))

    war = SC.weighted_annual_returns(footprint)
    stock_analog = {}
    if war:
        best_year = max(war, key=war.get)
        worst_year = min(war, key=war.get)
        if md_map:
            top_symbols = [r["stock_symbol"] for r in stock_concentration.top(20)]
            stock_analog = {
                "best_year": SC.stock_level_analog(md_map, top_symbols, best_year),
                "worst_year": SC.stock_level_analog(md_map, top_symbols, worst_year),
            }

    regime = None
    if market_dates:
        # as_of 用快時鐘實際解析到的交易日（每個市場自己的），不是 IS 結束日。
        # regime_avg_ret 驗證模式一律不給（見上方說明與 scenario.py 的理由）。
        regime = SC.regime_snapshot(cfg.market, market_dates,
                                    risk.regime_avg_ret if is_live else None)

    facts = {
        "mode": cfg.mode, "market": cfg.market, "group": cfg.group,
        "ratio": cfg.ratio, "allocation": cfg.allocation,
        "window_info": holdings.window_info, "n_members": holdings.n_members,
        "validation": holdings.validation,
        "performance": _fmt_perf(holdings.performance, holdings.has_oos),
        "reference_oos_distribution": _fmt_reference(holdings.reference_oos, market=cfg.market),

        "section1_strategy_footprint": {
            "tree_id": footprint.tree_id, "clusters": _cluster_facts(footprint),
        },
        "section1_stock_holdings": {
            "n_unique_stocks": stock_concentration.n_unique_stocks,
            "n_strategies": stock_concentration.n_strategies,
            "n_empty_strategies": stock_concentration.n_empty_strategies,
            "max_stock_weight": _pct(stock_concentration.max_stock_weight),
            "max_stock_symbol": stock_concentration.max_stock_symbol,
            "top15": [{**r, "weight": _pct(r["weight"])} for r in stock_concentration.top(15)],
            "most_overlapped": overlap.to_dict("records"),
        },

        "mechanism": {
            "rarest_cluster_protection": _rarest_cluster_protection(
                footprint, holdings.tree_info.get("n_universe")),
            "consistency_and_drawdown": _fmt_mechanism_consistency(
                mechanism_consistency(cfg.market, as_of_is_end=as_of_is_end)),
        },

        "risk": {
            "max_single_weight": _pct(risk.max_single_weight),
            "single_stock_cap": _pct(cfg.single_stock_cap),
            "max_cluster_share": _pct(risk.max_cluster_share),
            "cluster_cap": _pct(cfg.cluster_cap),
            "n_clusters_covered": risk.n_clusters_covered,
            "violations": [_fmt_violation(v) for v in risk.violations],
            "stock_level_violations": [_fmt_violation(v) for v in stock_concentration.violations],
            "factor_exposure_F1": {k: _pct(v) for k, v in risk.factor_exposure_f1.items()},
            "n_backfilled": holdings.performance.get("n_backfilled"),
            "target_total": holdings.performance.get("target_total"),
        },
        "structure": _structure_facts(footprint),

        "character": {
            "factor_exposure_F1": {k: _pct(v) for k, v in risk.factor_exposure_f1.items()},
            "new_vs_old_faces": (
                {"persistence_rate": _pct(face_comparison.persistence_rate),
                 "turnover_rate": _pct(face_comparison.turnover_rate),
                 "n_old": len(face_comparison.old_faces),
                 "n_new": len(face_comparison.new_faces),
                 "n_exited": len(face_comparison.exited)}
                if face_comparison is not None else None),
        },

        "scenario": {
            "weighted_annual_returns_by_year": {str(y): _pct(v) for y, v in war.items()},
            "historical_same_setting_windows": _fmt_historical_windows(
                historical_same_setting_windows(cfg, as_of_is_end=as_of_is_end)),
            "regime": _fmt_regime(regime),
            "stock_level_analog": {k: _fmt_stock_analog(v) for k, v in stock_analog.items()},
        },

        "change_vs_previous": _fmt_diff(
            diff_holdings(holdings.members, find_previous(cfg), holdings.window_info)),
        "alternative_groups": {g: _fmt_alt(v) for g, v in holdings.alternative_groups.items()},
        # 🔴 2026-09-14：跟 memo.py/_alt_context 同一個 bug 家族——沒有傳
        # baseline=cfg.group 的話永遠算 A_hrp 的勝率，使用者選另外兩個方法
        # 當主要邏輯時會讓 LLM 讀到錯誤的比較基準。
        "alternative_context": _fmt_alt_context(alt_group_win_rates(cfg.market, baseline=cfg.group)),

        "calibration": {
            "status": calib.status,
            "oos_cagr": _pct(calib.oos_cagr) if calib.oos_cagr is not None else None,
            "oos_cagr_p10": _pct(calib.thresholds.oos_cagr_p10),
            "n_historical_cells": calib.thresholds.n_cells,
            "flagged": calib.flagged, "note": calib.note,
        },
    }
    return facts


def build_prompt(facts: dict, *, fields: list[str] | None = None,
                 window_context: dict | None = None) -> str:
    """`fields`／`window_context`（§10.10，多檢查點驗證用）：`fields` 是這次
    要 LLM 填哪幾個欄位（`None`＝全部 11 個，原本行為不變）；`window_context`
    是同一窗次先前 `generate(..., fields=WINDOW_FIELDS)` 產生的窗次層結果，
    當作背景附進去，讓檢查點層的 `risk_flags`／`scenario_note` 可以直接引用
    裡面已經講過的機制，不用自己重新推導一遍、也不會跟窗次層的敘述脫節。
    """
    n = len(fields) if fields is not None else len(_EXPLAIN_SCHEMA["schema"]["required"])
    context_block = ""
    if window_context is not None:
        context_block = (
            "【這一窗已經產生的窗次層分析，當作背景參考，不用重新展開，"
            "但你的欄位若要引用其中提到的機制，須跟它保持一致，不能矛盾】\n"
            f"{json.dumps(window_context, ensure_ascii=False, indent=2, default=str)}\n\n"
        )
    return (
        "【客觀資料 · 由程式算出，不可推翻，數字已格式化，請直接照抄】\n"
        f"{json.dumps(facts, ensure_ascii=False, indent=2, default=str)}\n\n"
        f"{context_block}"
        f"請依給定的 JSON schema 輸出{n}個欄位。每個欄位都要做出實質的分析判斷"
        "（不是把數字複誦一遍），但只能根據以上資料——事實推論可以，價值判決"
        "（例如建議換方案）不行，見系統提示的鐵則 1。"
    )


def generate(holdings: Holdings, risk: RiskReport, calib: CalibrationResult, *,
            stock_detail, stock_concentration: StockConcentration,
            market_dates: dict[str, str] | None = None,
            footprint: ClusterFootprint | None = None, md_map: dict | None = None,
            face_comparison=None, dry_run: bool = True, model: str | None = None,
            purpose: str = "app_memo",
            fields: list[str] | None = None, window_context: dict | None = None) -> dict:
    """`fields`／`window_context`：見 `build_prompt()` 的說明。UI 目前的單次
    呼叫（`ui.py` AI 解讀分頁）不傳這兩個參數，行為與 §10.9 以前完全相同；
    這兩個參數只給多檢查點驗證腳本用，`WINDOW_FIELDS`／`CHECKPOINT_FIELDS`
    是預先分好的兩組欄位名稱。
    """
    is_live = holdings.config.mode == "live"
    schema = _build_schema(is_live, fields=fields)
    facts = assemble_facts(holdings, risk, calib, stock_detail=stock_detail,
                           stock_concentration=stock_concentration, market_dates=market_dates,
                           footprint=footprint, md_map=md_map, face_comparison=face_comparison)
    prompt = build_prompt(facts, fields=fields, window_context=window_context)

    required = schema["schema"]["required"]
    if dry_run:
        explanation = {k: "(dry-run，未呼叫 LLM)" for k in required if k != "risk_flags"}
        if "risk_flags" in required:
            explanation["risk_flags"] = [{"risk": "(dry-run，未呼叫 LLM)",
                                         "verification_criterion": "(dry-run，未呼叫 LLM)"}]
    else:
        from utils.config import Config
        cfg = Config()
        api_key = cfg.get_openai_api_key()
        model = model or cfg.get_openai_model(purpose)
        explanation, _usage = _call_llm(
            prompt, model, api_key, purpose=purpose, est_tokens=len(prompt) // 3,
            system_prompt=_build_system_prompt(is_live), schema=schema)

    # 🔴 2026-09-12（§9.8 四層驗證標準·層二「數字錯誤＝0，D2 自動」）：explain.py
    # 一直沒有接 D2 掃描——`memo.scan_for_leakage()` 本來就是通用的（吃任意
    # {欄位: 文字} dict 跟 prompt 字串，不綁 memo 的六欄位 schema），這裡直接
    # 複用，不重寫一份。跟 memo.py 一樣：抓到就攔下，不寫進稽核紀錄——
    # §9.8 的前瞻驗證需要每一份進 audit_log.jsonl 的解釋都先過這一關。
    leakage: list[str] = []
    if not dry_run:
        # 🔴 2026-09-12：`risk_flags[].verification_criterion` 依鐵則 11 是 LLM
        # **自訂**的事後檢驗門檻（例如「若回撤超過 5% 則視為應驗」），門檻數字
        # 本來就不必是 facts 裡已存在的數字——這是它跟其餘欄位唯一的例外，
        # 不是漏洞。逐字比對這個子欄位會把合理的自訂門檻誤判成洩漏，故只掃
        # `risk` 本身（那句仍應只描述 facts 裡的既有型態，不能捏造）。
        scannable = {k: v for k, v in explanation.items() if k != "risk_flags"}
        if "risk_flags" in explanation:
            scannable["risk_flags"] = [{"risk": rf["risk"]} for rf in explanation["risk_flags"]]
        leakage = scan_for_leakage(scannable, prompt)
        if leakage:
            raise RuntimeError(
                "D2 洩漏掃描攔下這份解釋，發現無法對應到判決資料的數字：\n  "
                + "\n  ".join(leakage))

    return {"prompt": prompt, "facts": facts, "explanation": explanation,
           "dry_run": dry_run, "leakage_check": leakage}
