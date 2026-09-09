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

from .calibration import CalibrationResult
from .engine import Holdings
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
    "不要自己重新換算、四捨五入或改變位數。"
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
            "caveat": {
                "type": "string",
                "description": "這份備忘錄的限制（例如：策略層級不是股票層級、"
                               "replay 模式是歷史回顧不是即時建議等，只能根據提供的資訊寫）。"},
        },
        "required": ["summary", "risk_note", "calibration_note", "caveat"],
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
        "performance": {
            "is_cagr": _pct(perf["is_cagr"]), "is_mdd": _pct(perf["is_mdd"]),
            "oos_cagr": _pct(perf["oos_cagr"]), "oos_mdd": _pct(perf["oos_mdd"]),
            "oos_sharpe": _ratio(perf["oos_sharpe"]),
            "n_backfilled": perf["n_backfilled"],
        },
        "risk": {
            "max_single_weight": _pct(risk.max_single_weight),
            "single_stock_cap": _pct(holdings.config.single_stock_cap),
            "max_cluster_share": _pct(risk.max_cluster_share),
            "cluster_cap": _pct(holdings.config.cluster_cap),
            "n_clusters_covered": risk.n_clusters_covered,
            "violations": [_fmt_violation(v) for v in risk.violations],
        },
        "calibration": {
            "oos_cagr": _pct(calib.oos_cagr), "oos_cagr_p10": _pct(calib.thresholds.oos_cagr_p10),
            "oos_calmar": _ratio(calib.oos_calmar), "oos_calmar_p10": _ratio(calib.thresholds.oos_calmar_p10),
            "n_historical_cells": calib.thresholds.n_cells,
            "flagged": calib.flagged,
        },
    }
    return (
        "【程式判決 · 不可推翻，數字已格式化，請照抄】\n"
        f"{json.dumps(facts, ensure_ascii=False, indent=2)}\n\n"
        "請依給定的 JSON schema 輸出，只能使用以上資訊，數字部分直接照抄不要重新換算。"
    )


def _call_llm(prompt: str, model: str, api_key: str, *, purpose: str = "app_memo",
             est_tokens: int = 0) -> tuple[dict, dict]:
    """呼叫端（cli.py／ui.py）只接 `RuntimeError` 就能顯示乾淨錯誤訊息——
    這裡把網路例外（逾時/斷線）跟回應結構異常（refusal、格式跑掉）都轉成
    `RuntimeError` 再往外拋，不要讓 `requests.RequestException`／`KeyError`
    這類非 RuntimeError 的例外裸奔出去，那樣會讓呼叫端的 `except RuntimeError`
    接不住，整個 CLI/Streamlit session 直接崩掉（2026-09-09 code review 抓到）。
    """
    import requests
    from utils import openai_quota as OQ
    OQ.check_free_tier_budget(model, estimated_tokens=est_tokens, reserve_ratio=RESERVE_RATIO)

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model,
                 "messages": [{"role": "system", "content": _SYSTEM_PROMPT},
                             {"role": "user", "content": prompt}],
                 "response_format": {"type": "json_schema", "json_schema": _MEMO_SCHEMA}},
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
        memo = {
            "summary": "(dry-run，未呼叫 LLM)",
            "risk_note": "(dry-run，未呼叫 LLM)",
            "calibration_note": "(dry-run，未呼叫 LLM)",
            "caveat": "(dry-run，未呼叫 LLM)",
        }
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
