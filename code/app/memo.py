# -*- coding: utf-8 -*-
"""L3 · AI 解釋層（D1/D2，應用層開發追蹤.md §0.3/§0.4/§3-D，Phase D）

架構完全比照 `research/cluster_story.py`／`research/cluster_identity.py`：
**程式先算出判決，LLM 只負責把「為什麼是這個判決」寫成人話**，不准自己推論、
不准引用沒被餵給它的數字。這裡「判決」＝ `engine.Holdings` + `risk.RiskReport`
+ `calibration.CalibrationResult`，全部已經是算好的結構化事實。

D1（本期選股說明）：先做這個，情境比對（老師舉的「這些股票是不是剛好都是最近
最紅的」那種）第二輪再加，見應用層開發追蹤.md D1。

D2（洩漏掃描，`replay` 模式強制開啟）：
    ⚠️ 2026-09-09 review 已更正過一次期望值——不能簡單類比成 `cluster_identity.py`
    的關鍵詞掃描（那是封閉集合字串比對），洩漏偵測本質是開放式語意問題。
    這裡先做**數字掃描**當第一道防線：把 LLM 輸出裡出現的每一個百分比/數字，
    對照有沒有出現在餵給它的判決資料裡（容許四捨五入誤差）——能抓到「捏造一個
    沒被餵過的數字」這種最常見、最容易查的洩漏形式，但**抓不到不含數字的敘述性
    洩漏**（例如空泛地講「這段時間市場情緒轉熱」卻沒有具體數字）。這是已知限制，
    不是假裝做到完整的語意比對，真正的語意掃描留待有需要時再加開一次 LLM 呼叫
    去驗證。

D3：額度沿用 `utils/openai_quota.py`，`purpose="app_memo"`——
    ⚠️ **`config.ini` 要有 `[openai] app_memo_model` 這個 key 才能用非 dry-run 模式**，
    這是新增的用途，需要你自己去 `config.ini` 補這個 key（內容我不會幫你看/貼出來）。
"""
from __future__ import annotations

import dataclasses
import json
import re

from .audit import diff_holdings, find_previous
from .calibration import CalibrationResult
from .engine import Holdings, alt_group_win_rates
from .risk import RiskReport, Violation

RESERVE_RATIO = 0.2   # 跟 cluster_story 同一個保守係數，額度快用完先煞車

_SYSTEM_PROMPT = (
    "你是量化投資系統的分析結果轉譯器。你會拿到這一期選股結果的**客觀統計數字**，"
    "以及**由程式算出、不可推翻的風控/校準判決**。"
    "你的工作只是把這些數字寫成一份給投資委員會看的中文備忘錄，不是自己重新判斷、"
    "不是預測未來表現。\n"
    "鐵則：\n"
    "1. 禁止引用未提供給你的數字，禁止杜撰個股、產業、總體經濟事件或市場情緒。\n"
    "2. 禁止推翻或質疑程式給的風控判決（違規就是違規、通過就是通過）。\n"
    "3. 這是歷史資料的回顧說明，不是預測——禁止對未來表現做任何預測或保證，"
    "也禁止用「後來證明」「事後看來」這類暗示你知道後續發展的說法。\n"
    "4. 若有風控違規且已被人工覆核放行，必須如實寫出違規內容跟覆核原因，"
    "不能淡化或省略。\n"
    "5. 判決資料裡的數字已經事先格式化好（百分比/小數位數），請直接照抄，"
    "不要自己重新換算、四捨五入或改變位數。\n"
    "6. change_note：只描述提供的新增/剔除檔數，禁止推測換股的原因（例如不能說"
    "「因為市場轉向所以換股」，因為這個原因沒有被提供給你）。\n"
    "7. alternative_note：只陳述提供的替代方案數字，禁止建議「應該改用哪一個」"
    "或評論哪個方案比較好——這件事已經由研究部的統計檢定回答過，不是這份備忘錄"
    "的工作，你的角色只是把數字攤開陳述。"
    "**若本期替代方案的數字比目前選定的好，必須把 `alternative_context` 那段話"
    "一併寫進去（可改寫語氣但不可改變意思、不可省略）**——那是程式提供的定錨事實。"
    "⚠️ 禁止自行寫成「屬於正常波動，長期會回歸」這類說法，那是錯的：研究已證實"
    "這是系統性結果不是隨機變異。\n"
    "8. **is_* 與 oos_* 是兩種不可互相比較的數字**（見 performance 裡的 `_is_caveat`）："
    "is_* 是樣本內配適值，不是預期報酬。禁止把 is_cagr 說成「預期報酬」「可望達到」，"
    "禁止把 is_* 跟 oos_* 並排比較或相減。若 `oos_cagr` 是 null，代表這個模式依定義"
    "沒有樣本外，要照 `_no_oos_reason` 說明，不可留白也不可用 is_* 頂替。\n"
    "9. 若有 `reference_oos_distribution`，那才是談「這類設定歷史上表現如何」時該引用的"
    "數字，並且必須附上它的 `caveat`。談基準時要遵守 `_benchmark_note`——"
    "**禁止寫「贏過大盤」**。\n"
    "10. 若 `change_vs_previous` 裡有 `direction_warning`，代表這是回溯比較不是本期異動，"
    "change_note 必須照實說明，不可寫成「本期換了幾檔」。\n"
    "11. 若 `validation` 有 `structural_caveat`，caveat 欄位必須包含它的意思。"
)

_MEMO_SCHEMA = {
    "name": "ic_memo",
    "schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "這期選股結果的摘要：選了幾檔、哪個市場、用什麼規則，"
                               "只能引用提供的數字。"},
            "risk_note": {
                "type": "string",
                "description": "風控檢查結果的說明。有違規要明講違規內容跟覆核原因"
                               "（若有）；沒有違規就直接說通過。"},
            "calibration_note": {
                "type": "string",
                "description": "校準監控結果的說明：這期表現跟歷史常態分布相比如何，"
                               "只能引用提供的 p10 門檻數字，不能延伸解讀原因。"},
            "change_note": {
                "type": "string",
                "description": "跟上一次同一組設定的執行結果相比，這次新增/剔除了幾檔"
                               "策略；沒有上一筆紀錄可比就明講「這是第一次執行，沒有"
                               "前期可比較」，不能假裝有更早的紀錄。只能引用提供的"
                               "diff 數字，不能推測換股的原因。"},
            "alternative_note": {
                "type": "string",
                "description": "同一格設定下，若改用其他候選方案（H-26/H-27/M-03 已"
                               "驗證過的替代規則）會有什麼不同：檔數、會增減幾檔、"
                               "OOS 績效數字。只是攤開既有比較結果給決策者參考，"
                               "不能寫成「應該改用哪一個」的建議或評論優劣。"},
            "caveat": {
                "type": "string",
                "description": "這份備忘錄的限制（例如：策略層級不是股票層級、"
                               "replay 模式是歷史回顧不是即時建議等，只能根據提供的資訊寫）。"},
        },
        "required": ["summary", "risk_note", "calibration_note", "change_note",
                    "alternative_note", "caveat"],
        "additionalProperties": False,
    },
    "strict": True,
}


def _pct(x: float) -> str:
    """比例類數字（CAGR/MDD/權重/佔比）統一格式化成百分比字串，小數點後2位。"""
    return f"{x:.2%}"


def _ratio(x: float) -> str:
    """比率類數字（Sharpe/Calmar）統一格式化成小數，小數點後3位。"""
    return f"{x:.3f}"


def _fmt_violation(v: Violation) -> dict:
    d = dataclasses.asdict(v)
    d["value"] = _pct(d["value"])
    d["limit"] = _pct(d["limit"])
    return d


def _fmt_diff(diff: dict) -> dict:
    """只給 LLM 看數量，不給實際策略 uid 清單——memo 該講的是「換了幾檔」，
    不是逐一唸出策略名稱，那種細節留給 UI 的原始資料表格。"""
    if not diff.get("has_previous"):
        return {"has_previous": False}
    out = {
        "has_previous": True,
        "previous_recorded_at": diff["previous_recorded_at"],
        "n_added": diff["n_added"], "n_removed": diff["n_removed"],
        "n_unchanged": diff["n_unchanged"],
    }
    # §8-R6：逆序比較要明講，不然 change_note 會把「回溯比較」寫成「本期異動」
    if diff.get("is_chronological") is False:
        out["direction_warning"] = diff["direction_note"]
    return out


def _fmt_perf(perf: dict, has_oos: bool) -> dict:
    """績效數字。🔴 §8-R13：正式模式只有 IS，且 IS 不是預期報酬。"""
    out = {"is_cagr": _pct(perf["is_cagr"]), "is_mdd": _pct(perf["is_mdd"])}
    if "is_sharpe" in perf:
        out["is_sharpe"] = _ratio(perf["is_sharpe"])
    out["n_backfilled"] = perf.get("n_backfilled")
    if has_oos:
        out.update({"oos_cagr": _pct(perf["oos_cagr"]), "oos_mdd": _pct(perf["oos_mdd"]),
                    "oos_sharpe": _ratio(perf["oos_sharpe"])})
    else:
        out["oos_cagr"] = None
        out["_no_oos_reason"] = (
            "本模式的 IS 用掉全部可用資料，依定義沒有樣本外可留，因此沒有 OOS 數字。")
    out["_is_caveat"] = (
        "上列 is_* 是**樣本內配適值，不是預期報酬**——歷史上 IS CAGR 中位數比實際 OOS "
        "高 7.62pp（1.48 倍，74.2% 的格子皆然）。⚠️ 但方向不一致：IS MDD 與 IS Sharpe "
        "反而比 OOS 更差（IS 涵蓋 2008 金融海嘯，OOS 窗都從 2013 之後開始）。"
        "**IS 與 OOS 在任何方向上都不可比，不得並排呈現或相減。**")
    return out


def _fmt_reference(ref: dict | None) -> dict | None:
    """R13 的主要績效參照＋R8② 的基準對照。"""
    if not ref:
        return None
    return {
        "oos_cagr_p10": _pct(ref["oos_cagr_p10"]),
        "oos_cagr_median": _pct(ref["oos_cagr_median"]),
        "oos_cagr_p90": _pct(ref["oos_cagr_p90"]),
        "oos_mdd_median": _pct(ref["oos_mdd_median"]),
        "benchmark_cagr": _pct(ref["benchmark_cagr"]),   # R8②：自建宇宙基準
        "n_cells": ref["n_cells"],
        "caveat": ref["caveat"],
        "_benchmark_note": (
            "benchmark_cagr 是自建宇宙基準（contracts.BENCHMARK_CAGR）。"
            "⚠️ 台股 A_hrp 相對**市值加權**大盤是輸的（M-17：19.55% vs 20.91%），"
            "贏的是**等權市場**（15.02%）——差異來自加權方式（台積電佔指數 40.23%），"
            "不可寫成「贏大盤」。"),
    }


def _fmt_alt(alt: dict) -> dict:
    """替代方案的數字。⚠️ replay 給的是 oos_*，正式模式給的是 is_*（沒有 OOS），
    欄位名照原樣帶出去，**不改名成同一個**——讓 LLM 看得出兩者不是同一種東西。"""
    out = {"n_members": alt["n_members"],
           "n_would_add": alt["n_would_add"], "n_would_remove": alt["n_would_remove"]}
    for k in ("oos_cagr", "oos_mdd", "is_cagr", "is_mdd"):
        if k in alt:
            out[k] = _pct(alt[k])
    for k in ("oos_sharpe", "is_sharpe"):
        if k in alt:
            out[k] = _ratio(alt[k])
    return out


#: 🔴 §8-R12：`alternative_note` 常會出現「這期我們的方法輸給另一個方法」——
#: 單一窗次「看起來選錯」是**常態不是例外**（見下方 `_alt_context()` 的三市場實測）。
#: memo 規則已禁止 LLM 建議換方法，但沒有定錨句的話，非技術背景的讀者只看這一期
#: 會直覺問「那為什麼不換」。
#:
#: ⚠️ **這句話不可以寫成「屬於正常變異、再等等就會回來」**——M-03 已證明那是
#: **系統性結果不是隨機變異**，那樣寫是誤導。所以由程式給一句固定的、誠實的定錨，
#: LLM 只能照抄（抗幻覺鐵則：程式判決、LLM 寫字）。
#: 數字取**聚合列（900 格）**——依 M-16 證據分層，聚合是驗證性、逐比例是探索性。
#:
#: 🔴 2026-09-11（§9.7 S2）：原本是寫死 TW 數字（16.8%／13.2%）的靜態字串，三市場
#: 開放後**不可原樣沿用**——那會把 TW 專屬事實講成通用事實，是正確性錯誤，不只是
#: 措辭問題。改成依 `holdings.config.market` 現算（`engine.alt_group_win_rates()`）。
def _alt_context(market: str) -> str:
    r = alt_group_win_rates(market)
    if r is None:
        return ("（此市場沒有可算出定錨句的歷史對照資料，本段從缺——"
                "不可自行杜撰數字頂替。）")
    return (
        f"A_hrp 在報酬類指標上輸給 D_top_cagr／E_top_calmar 是**已知的系統性結果**，"
        f"不是本期特例——{r['n_cells']} 格{market}市場歷史逐格對照中，A_hrp 僅"
        f"{r['calmar_vs_E_top_calmar']:.1%}（Calmar）／{r['oos_cagr_vs_E_top_calmar']:.1%}"
        f"（OOS CAGR）勝過 E_top_calmar；對 D_top_cagr 則是"
        f"{r['calmar_vs_D_top_cagr']:.1%}／{r['oos_cagr_vs_D_top_cagr']:.1%}。"
        f"選用 A_hrp 的理由不是報酬優勢（M-03/M-03b 已證實 HRP 分群不提供報酬優勢），"
        f"而是分散度與跨市場回撤控制。是否更換方法屬政策層決定，"
        f"不在單期備忘錄的權責內。")


def build_prompt(holdings: Holdings, risk: RiskReport, calib: CalibrationResult) -> str:
    perf = holdings.performance
    facts = {
        "mode": holdings.config.mode,
        "market": holdings.config.market,
        "group": holdings.config.group,
        "ratio": holdings.config.ratio,
        "allocation": holdings.config.allocation,
        "window_info": holdings.window_info,
        "n_members": holdings.n_members,
        "validation": holdings.validation,          # G7 三態標籤＋結構性限制
        # 🔴 §8-R13：正式模式沒有 OOS，只有 IS。IS CAGR 是**樣本內配適值**，實測
        # 900 格台股 A_hrp 顯示 IS CAGR 中位數 23.60% vs OOS 15.98%（高 7.62pp、
        # 1.48 倍，74.2% 的格子皆然）。⚠️ 方向不一致：IS MDD/Sharpe 反而比 OOS 更差
        # （IS 涵蓋 2008 海嘯，OOS 窗都從 2013 之後開始）⇒ **兩者不可並排比較**。
        # 這裡把警告直接放進 facts，LLM 只能照抄，不會自己編一句樂觀的話。
        "performance": _fmt_perf(perf, holdings.has_oos),
        # R13 主要績效參照：凍結表同類設定的 OOS 分布（含 R8② 基準）
        "reference_oos_distribution": _fmt_reference(holdings.reference_oos),
        "risk": {
            "max_single_weight": _pct(risk.max_single_weight),
            "single_stock_cap": _pct(holdings.config.single_stock_cap),
            "max_cluster_share": _pct(risk.max_cluster_share),
            "cluster_cap": _pct(holdings.config.cluster_cap),
            "n_clusters_covered": risk.n_clusters_covered,
            "violations": [_fmt_violation(v) for v in risk.violations],
            # 2026-09-10（應用層 §6 落差②）：T8 本來就算好、先前沒接進 memo 的三組數字。
            "factor_exposure_F1": {k: _pct(v) for k, v in risk.factor_exposure_f1.items()},
            "market_share": {k: _pct(v) for k, v in risk.market_share.items()},
            "regime_avg_ret": {k: _pct(v) for k, v in risk.regime_avg_ret.items()},
        },
        "calibration": {
            # §8-R3：正式模式 status="tracking_started"，oos_* 是 None
            "status": calib.status,
            "oos_cagr": _pct(calib.oos_cagr) if calib.oos_cagr is not None else None,
            "oos_cagr_p10": _pct(calib.thresholds.oos_cagr_p10),
            "oos_calmar": _ratio(calib.oos_calmar) if calib.oos_calmar is not None else None,
            "oos_calmar_p10": _ratio(calib.thresholds.oos_calmar_p10),
            "n_historical_cells": calib.thresholds.n_cells,
            "flagged": calib.flagged,
            "note": calib.note,
        },
        # 2026-09-10（應用層 §6 落差①）：跟上一次同一組 RunConfig 身份的執行結果相比。
        "change_vs_previous": _fmt_diff(
            diff_holdings(holdings.members, find_previous(holdings.config),
                          holdings.window_info)),
        # 2026-09-10（應用層 §6 落差③）：同一格設定下的其他候選方案（H-12 四組對照）。
        "alternative_groups": {g: _fmt_alt(v) for g, v in holdings.alternative_groups.items()},
        "alternative_context": _alt_context(holdings.config.market),  # §8-R12 定錨句，照抄用
    }
    return (
        "【程式判決 · 不可推翻，數字已格式化，請照抄】\n"
        f"{json.dumps(facts, ensure_ascii=False, indent=2)}\n\n"
        "請依給定的 JSON schema 輸出，只能使用以上資訊，數字部分直接照抄不要重新換算。"
    )


def _call_llm(prompt: str, model: str, api_key: str, *, purpose: str = "app_memo",
             est_tokens: int = 0, system_prompt: str | None = None,
             schema: dict | None = None) -> tuple[dict, dict]:
    """呼叫端（cli.py／ui.py）只接 `RuntimeError` 就能顯示乾淨錯誤訊息——
    這裡把網路例外（逾時/斷線）跟回應結構異常（refusal、格式跑掉）都轉成
    `RuntimeError` 再往外拋，不要讓 `requests.RequestException`／`KeyError`
    這類非 RuntimeError 的例外裸奔出去，那樣會讓呼叫端的 `except RuntimeError`
    接不住，整個 CLI/Streamlit session 直接崩掉（2026-09-09 code review 抓到）。

    `system_prompt`／`schema` 預設用本檔案的 D1 six-欄位版本；
    2026-09-11：`explain.py`（解釋 agent，10 欄位）複用同一個網路/錯誤處理/
    額度記帳包裝，不重寫一份，改傳自己的 prompt/schema 進來。
    """
    import requests
    from utils import openai_quota as OQ
    system_prompt = system_prompt if system_prompt is not None else _SYSTEM_PROMPT
    schema = schema if schema is not None else _MEMO_SCHEMA
    OQ.check_free_tier_budget(model, estimated_tokens=est_tokens, reserve_ratio=RESERVE_RATIO)

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model,
                 "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": prompt}],
                 "response_format": {"type": "json_schema", "json_schema": schema}},
            timeout=90,
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"呼叫 OpenAI API 時網路發生問題（逾時/斷線）：{e}") from e

    OQ.raise_for_openai_response(resp)

    try:
        body = resp.json()
        message = body["choices"][0]["message"]
        content = message["content"]
        if content is None:
            raise RuntimeError(
                f"LLM 拒絕回覆這次請求（structured output refusal）："
                f"{message.get('refusal', '(無說明)')}")
        memo = json.loads(content)
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise RuntimeError(f"OpenAI 回應不是預期的結構，無法解析：{e}") from e

    usage = body.get("usage", {})
    # 2026-09-09：app_memo_model 這個 key 還沒設定，經使用者同意先借用既有
    # purpose（例如 "cluster_story"）的模型設定跟額度記帳，帳本上會顯示成
    # 那個 purpose 的用量，不是獨立的 app_memo 額度——之後要上線再拆開。
    OQ.log_usage(model, purpose, usage)
    return memo, usage


_NUMBER_RE = re.compile(r"-?\d+\.?\d*%?")


def scan_for_leakage(memo: dict, prompt: str, *, tol: float = 0.06) -> list[str]:
    """D2 第一道防線：memo 裡出現的每個數字，有沒有對得到 prompt 裡餵過的數字。

    只查數字，不查敘述——見本檔案開頭的限制說明。`tol` 是容許的相對誤差
    （四捨五入/百分比轉換造成的正常落差，不是抓漏洞用的寬鬆值）。
    """
    prompt_numbers = [float(n.rstrip("%")) for n in _NUMBER_RE.findall(prompt) if n not in ("", "-", ".")]
    suspicious = []
    for field, text in memo.items():
        if not isinstance(text, str):
            continue
        for n in _NUMBER_RE.findall(text):
            n_clean = n.rstrip("%")
            if not n_clean or n_clean in ("-", "."):
                continue
            v = float(n_clean)
            if not any(abs(v - p) <= max(tol * abs(p), tol) for p in prompt_numbers):
                suspicious.append(f"[{field}] 出現數字 {n}，在餵給 LLM 的判決資料裡找不到對應值")
    return suspicious


def generate(holdings: Holdings, risk: RiskReport, calib: CalibrationResult, *,
            dry_run: bool = True, model: str | None = None,
            purpose: str = "app_memo") -> dict:
    """`purpose` 決定去 config.ini 讀哪組 `[openai] {purpose}_model`跟額度記帳。

    2026-09-09：`app_memo_model` 這個 key 還沒設定，經使用者同意（選項 B）先借用
    既有 purpose（例如 "cluster_story"）的模型設定跟額度記帳——帳本上會顯示成
    那個 purpose 的用量，不是獨立的 app_memo 額度，之後要上線再拆開成獨立 key。
    """
    prompt = build_prompt(holdings, risk, calib)

    if dry_run:
        # 2026-09-10 code review：欄位清單直接從 schema 的 required 衍生，不要
        # 手動另外列一份——先前就是手動列的清單漏了新增的兩個欄位，UI 端
        # 用固定 key 去讀 memo 字典會直接 KeyError 崩掉，不是「顯示不完整」而已。
        memo = {k: "(dry-run，未呼叫 LLM)" for k in _MEMO_SCHEMA["schema"]["required"]}
    else:
        from utils.config import Config
        cfg = Config()
        api_key = cfg.get_openai_api_key()
        model = model or cfg.get_openai_model(purpose)
        memo, _usage = _call_llm(prompt, model, api_key, purpose=purpose,
                                 est_tokens=len(prompt) // 3)

    leakage = []
    if holdings.config.mode == "replay" and not dry_run:
        leakage = scan_for_leakage(memo, prompt)
        if leakage:
            raise RuntimeError(
                "D2 洩漏掃描攔下這份 memo，發現無法對應到判決資料的數字：\n  "
                + "\n  ".join(leakage))

    return {"prompt": prompt, "memo": memo, "leakage_check": leakage, "dry_run": dry_run}
