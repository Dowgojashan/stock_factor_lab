# -*- coding: utf-8 -*-
"""OpenAI 額度偵測工具（供 T10 generate_return_story_text／cluster_story 用）。

背景：這兩個功能點都要花錢呼叫 OpenAI API，使用者希望在額度用完時能被通知，
再決定要不要繼續付費或切換另一組帳號的 key。實測發現這個帳號的一般 API key
查不到「剩餘額度」（billing 端點只認瀏覽器 session key，官方 Usage API 要
Admin key 才有權限），故拆成兩種偵測方式：

  1. 反應式（reactive）：包在每次真正呼叫 LLM API 的地方，偵測 OpenAI 回傳的
     `insufficient_quota` 錯誤——這是官方穩定、文件化的錯誤代碼，用一般的
     project API key 就能觸發/偵測到，不需要額外權限。`raise_for_openai_response()`
     負責這件事。

  2. 主動式（proactive）：呼叫官方 Organization Costs API
     （`/v1/organization/costs`）查當天已花費金額，跟使用者設定的每日免費
     額度比較。**這支端點需要 Admin API key**（在 OpenAI org 設定裡另外產生、
     需要 `api.usage.read` scope），跟 config.ini 裡原本 `[openai] api_key`
     （一般 project key）不是同一種，須另外設定 `[openai] admin_key`。
     `check_usage_today()` 負責這件事。

     ⚠️ **2026-08-25：admin_key 尚未到位，本模組的回應欄位解析是照 OpenAI
     官方文件寫的（`data[].results[].amount.value`），還沒有拿真實回應核對過**。
     等 admin_key 設定好，第一件事就是打一次真的、對照欄位名稱是否正確，
     發現不符要立刻修正——不能把這裡的假設當成已驗證的事實用。
     用 `python -m utils.openai_quota --selftest` 可以用合成回應驗證解析邏輯
     本身沒寫錯（但這驗證不了欄位名稱是否跟真實API一致）。

用法：
    from utils.openai_quota import (
        check_usage_today, raise_for_openai_response,
        QuotaExhaustedError, OpenAIAPIError,
    )

    # 反應式：每次真正呼叫 LLM API 後都檢查一次
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    raise_for_openai_response(resp)   # 額度用盡會 raise QuotaExhaustedError

    # 主動式（需要 admin_key，且尚未實測校正過欄位名稱）：
    status = check_usage_today(admin_key, daily_limit_usd=5.0)
    if status.exhausted:
        ...
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

OPENAI_API_BASE = "https://api.openai.com/v1"
REQUEST_TIMEOUT = 15   # 秒。查用量是輕量的 metadata 讀取，不是模型推論，不需要長逾時


# ============================================================================
# 每日免費額度（data-sharing 方案）· 用量帳本與暫停機制
# ============================================================================
#
# 使用者的帳號有 OpenAI「與 OpenAI 共享流量」的每日免費額度，分兩個獨立的池：
#   標準池 1,000,000 tokens/日
#   小模型池 10,000,000 tokens/日
# 兩池分開計算、互不流用（用完標準池不會自動吃小模型池）。
#
# ⚠️ **名單是使用者 2026-08-29 從 OpenAI 後台複製的，不是我查來的**。OpenAI 可能
#    隨時調整涵蓋的模型與額度，這份常數僅代表當下狀態，發現對不上請直接更新這裡。
#
# ⚠️ **不在名單上的模型 = 付費**。`classify_model()` 對未知模型一律回 None
#    （代表「不在免費名單」），**不做寬鬆猜測**——猜錯的代價是靜默燒錢，
#    而漏判的代價只是多問一句，兩者不對稱。

FREE_TIER_LIMITS = {
    "standard": 1_000_000,
    "mini": 10_000_000,
}

#: 小模型池（10M）。比對時**優先於標準池**——`gpt-5-mini` 同時符合兩邊的前綴，
#: 必須先判小模型池才不會被誤歸到標準池、用錯額度上限。
_MINI_TIER_MODELS = (
    "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5-mini", "gpt-5-nano",
    "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini", "o3-mini", "o4-mini",
)

#: 標準池（1M）
_STANDARD_TIER_MODELS = (
    "gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-4.1", "gpt-4o", "o1", "o3",
)

#: 用量帳本。放 code/_catalog/（既有的「執行紀錄」慣例所在），純 token 統計、
#: 不含任何金鑰或提示詞內容，可安全進版控——論文要交代 LLM 成本時直接引用。
LEDGER_PATH = Path(__file__).resolve().parent.parent / "code" / "_catalog" / "llm_usage.jsonl"


class FreeTierExhaustedError(RuntimeError):
    """今日免費額度（該池）已用盡或即將用盡——**暫停機制**的觸發訊號。

    刻意與 `QuotaExhaustedError` 分開：後者是 OpenAI 真的回報帳戶額度用盡
    （已經花到錢或被擋），前者是**我們自己主動踩煞車**，還沒真的超額。
    呼叫端接到這個要停下來問人，不要自動改用付費模型繼續跑。
    """


def _matches_family(model: str, prefix: str) -> bool:
    """模型名是否屬於 `prefix` 這個family。

    🔴 **不能用單純的 startswith**（2026-08-29 自我測試抓到的真實bug）：
       `"gpt-5.6-terra".startswith("gpt-5")` 是 True，會把不在免費名單的 gpt-5.6
       誤判成免費的 gpt-5，然後靜默去燒付費額度——正是本模組要防的事。
       任何未來版本（5.6／5.9／…）都會中這個陷阱。

    規則：前綴後面**不可以接數字或小數點**（那代表是不同版本的 family），
    接 `-`（日期/變體後綴，如 `gpt-4o-2024-08-06`）或字串結束才算同一個 family。
    """
    if not model.startswith(prefix):
        return False
    rest = model[len(prefix):]
    return rest == "" or rest[0] not in ".0123456789"


def classify_model(model: str) -> str | None:
    """模型 → 免費額度池（'standard' / 'mini'），不在免費名單回 None。

    **先比小模型池**（見 `_MINI_TIER_MODELS` 上方說明），比對規則見 `_matches_family`。
    """
    m = (model or "").strip().lower()
    if not m:
        return None
    for prefix in _MINI_TIER_MODELS:
        if _matches_family(m, prefix):
            return "mini"
    for prefix in _STANDARD_TIER_MODELS:
        if _matches_family(m, prefix):
            return "standard"
    return None


def _day_key(now: _dt.datetime | None = None, tz_utc: bool = True) -> str:
    """帳本的「當日」定義。

    ⚠️ **OpenAI 每日免費額度實際在哪個時區重置，官方文件沒有明說，我也沒有實測過**。
    這裡預設用 UTC（API 額度最常見的作法），若之後發現實際是太平洋時間或其他時區
    導致跨日判斷差一天，改這裡即可。用 `tz_utc=False` 可改用本機時區比對。
    """
    now = now or (_dt.datetime.now(_dt.timezone.utc) if tz_utc else _dt.datetime.now())
    if tz_utc and now.tzinfo is not None:
        now = now.astimezone(_dt.timezone.utc)
    return now.date().isoformat()


def log_usage(model: str, purpose: str, usage: dict, *,
              now: _dt.datetime | None = None, path: Path | None = None) -> dict:
    """把一次 LLM 呼叫的用量追加進帳本（JSONL，一行一筆）。回傳寫入的那筆紀錄。

    `usage` 直接吃 OpenAI 回應裡的 `usage` 物件（含 prompt_tokens/completion_tokens/
    total_tokens）。缺欄位時以 0 計，不猜測、不推估。
    """
    path = path or LEDGER_PATH
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    total = int(usage.get("total_tokens", 0) or 0) or (prompt + completion)
    rec = {
        "ts": (now or _dt.datetime.now(_dt.timezone.utc)).astimezone(
            _dt.timezone.utc).isoformat(timespec="seconds"),
        "day_utc": _day_key(now),
        "model": model,
        "tier": classify_model(model),
        "purpose": purpose,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def read_ledger(path: Path | None = None) -> list[dict]:
    """讀回整本帳本。檔案不存在回空 list（第一次跑本來就沒有，不是錯誤）。"""
    path = path or LEDGER_PATH
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def today_totals(*, now: _dt.datetime | None = None,
                 path: Path | None = None) -> dict[str, int]:
    """今日（UTC）各池已用 token 數。未分類（付費）模型歸在 'unclassified'。"""
    day = _day_key(now)
    totals = {"standard": 0, "mini": 0, "unclassified": 0}
    for r in read_ledger(path):
        if r.get("day_utc") != day:
            continue
        tier = r.get("tier") or "unclassified"
        totals[tier] = totals.get(tier, 0) + int(r.get("total_tokens", 0) or 0)
    return totals


def check_free_tier_budget(model: str, *, estimated_tokens: int = 0,
                           now: _dt.datetime | None = None,
                           path: Path | None = None,
                           reserve_ratio: float = 0.0) -> dict:
    """呼叫 LLM **之前**檢查今日免費額度還夠不夠。不夠就 raise FreeTierExhaustedError。

    `estimated_tokens`：本次預估會用掉多少（不知道就傳0，那就只擋「已經超額」的情況）。
    `reserve_ratio`：預留比例，例如 0.05 代表用到 95% 就踩煞車，留一點餘裕給
                     正在進行中的請求，避免剛好卡在邊界上被 OpenAI 擋掉。

    **不在免費名單的模型直接 raise**——這是刻意的：使用者的意圖是「盡量用免費的，
    超過再決定要不要花錢」，靜默改用付費模型跑下去正好違反這個意圖。
    """
    tier = classify_model(model)
    if tier is None:
        raise FreeTierExhaustedError(
            f"模型 `{model}` 不在每日免費額度名單內，跑下去會直接計費。\n"
            f"  免費名單（標準池 {FREE_TIER_LIMITS['standard']:,} tok/日）："
            f"{', '.join(_STANDARD_TIER_MODELS)}\n"
            f"  免費名單（小模型池 {FREE_TIER_LIMITS['mini']:,} tok/日）："
            f"{', '.join(_MINI_TIER_MODELS)}\n"
            f"→ 請改用名單內的模型，或明確確認要付費後再跑")

    limit = FREE_TIER_LIMITS[tier]
    used = today_totals(now=now, path=path)[tier]
    budget = int(limit * (1.0 - reserve_ratio))
    projected = used + max(0, int(estimated_tokens))
    status = {"model": model, "tier": tier, "limit": limit, "used": used,
              "projected": projected, "budget": budget,
              "remaining": max(0, budget - used)}
    if projected > budget:
        raise FreeTierExhaustedError(
            f"今日「{tier}」池免費額度不足：已用 {used:,}／預估本次再用 "
            f"{estimated_tokens:,} → 合計 {projected:,}，超過可用上限 {budget:,}"
            f"（總額度 {limit:,}"
            f"{f'，已預留 {reserve_ratio:.0%}' if reserve_ratio else ''}）。\n"
            f"→ 已暫停，請決定：等明天額度重置／改用另一個池的小模型／確認要付費")
    return status


class QuotaExhaustedError(RuntimeError):
    """OpenAI 回報額度已用盡（error.code == 'insufficient_quota'）。"""


class OpenAIAPIError(RuntimeError):
    """OpenAI 回傳其他非額度錯誤，原樣包起來方便呼叫端分流處理
    （例如暫時性 rate_limit_exceeded 可以重試，額度用盡不該重試）。
    """

    def __init__(self, status_code: int, code: str | None, message: str):
        self.status_code = status_code
        self.code = code
        super().__init__(f"[{status_code}] {code}: {message}")


def _parse_openai_error_body(body: Any) -> tuple[str | None, str]:
    err = body.get("error", {}) if isinstance(body, dict) else {}
    if not isinstance(err, dict):
        err = {}
    return err.get("code"), err.get("message", str(body))


def raise_for_openai_response(resp: requests.Response) -> None:
    """對一次 OpenAI API 回應做錯誤分流。2xx 原樣放行不動作；
    額度用盡 → `QuotaExhaustedError`；其他非2xx → `OpenAIAPIError`。

    呼叫端只要在每次真正打 LLM API 的地方把 `requests.Response` 丟進來，
    就能統一分辨「額度用盡」跟「其他錯誤」，不必自己重寫一次錯誤解析。
    """
    if resp.ok:
        return
    try:
        body = resp.json()
    except ValueError:
        body = {}
    code, message = _parse_openai_error_body(body)
    # ⚠️ **不綁定 HTTP status 判斷額度**（2026-08-25 code review 修正）：
    # 原本寫成 `status==429 and code=="insufficient_quota"`，但額度類錯誤不保證
    # 一定是 429（帳單上限、組織額度等情境可能以 403 或其他 status 回來）。
    # 一旦 status 不合就會被歸到一般 OpenAIAPIError，使用者就收不到「額度用完、
    # 要不要換 key」這個唯一真正需要人介入的訊號——那正是本模組存在的理由。
    # 故改為只認 error code/type，不看 status。
    quota_codes = {"insufficient_quota", "billing_hard_limit_reached",
                   "account_deactivated", "quota_exceeded"}
    err_type = (body.get("error", {}) or {}).get("type") if isinstance(body, dict) else None
    if (code in quota_codes) or (err_type in quota_codes):
        raise QuotaExhaustedError(
            f"OpenAI 額度已用盡（HTTP {resp.status_code}，code={code}）：{message}"
            "（請確認要繼續付費，或切換另一組帳號的 API key）")
    raise OpenAIAPIError(resp.status_code, code, message)


@dataclass
class UsageStatus:
    date: str            # 查詢當天日期（UTC），ISO格式
    total_usd: float      # 當天累計花費（美金）
    daily_limit_usd: float
    exhausted: bool        # total_usd >= daily_limit_usd
    raw: Any               # 原始回應，供除錯或欄位核對用


def _extract_total_usd(body: dict) -> float:
    """從 `/v1/organization/costs` 的回應結構撈出累計美金花費。

    ⚠️ 照官方文件寫的欄位路徑（`data[].results[].amount.value`），
    尚未用真實回應驗證，admin_key 到位後第一件事就是核對這裡（見模組docstring）。
    """
    total = 0.0
    for bucket in body.get("data", []):
        for result in bucket.get("results", []):
            amount = result.get("amount", {})
            total += float(amount.get("value", 0.0))
    return total


def check_usage_today(admin_key: str, daily_limit_usd: float,
                      *, now: _dt.datetime | None = None) -> UsageStatus:
    """查詢今天（UTC）累計花費，跟每日免費額度比較。

    `daily_limit_usd` 沒有官方API可以查（OpenAI 不提供「額度上限」的查詢端點，
    上限是帳號owner在後台自己設的），必須由使用者告知實際數字、當參數傳入。
    """
    now = now or _dt.datetime.now(_dt.timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_ts = int(start_of_day.timestamp())

    r = requests.get(
        f"{OPENAI_API_BASE}/organization/costs",
        headers={"Authorization": f"Bearer {admin_key}"},
        params={"start_time": start_ts, "bucket_width": "1d", "limit": 1},
        timeout=REQUEST_TIMEOUT,
    )
    raise_for_openai_response(r)
    body = r.json()
    total = _extract_total_usd(body)
    return UsageStatus(
        date=start_of_day.date().isoformat(),
        total_usd=total,
        daily_limit_usd=daily_limit_usd,
        exhausted=total >= daily_limit_usd,
        raw=body,
    )


# ============================================================================
# 合成資料自我測試（無法用真實 admin_key 驗證前，先確保解析邏輯本身沒寫錯）
# ============================================================================

def _fake_response(status_code: int, json_body: dict) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status_code
    import json as _json
    resp._content = _json.dumps(json_body).encode("utf-8")
    return resp


def selftest(log=print) -> None:
    log("== T1 反應式：正常回應（2xx）不該 raise ==")
    ok_resp = _fake_response(200, {"data": []})
    raise_for_openai_response(ok_resp)   # 沒 raise 就算過
    log("  ✓ 通過")

    log("== T2 反應式：insufficient_quota 要 raise QuotaExhaustedError ==")
    quota_resp = _fake_response(429, {
        "error": {"message": "You exceeded your current quota, please check your plan and billing details.",
                  "type": "insufficient_quota", "code": "insufficient_quota"}})
    try:
        raise_for_openai_response(quota_resp)
        raise AssertionError("預期 raise QuotaExhaustedError 但沒有")
    except QuotaExhaustedError:
        log("  ✓ 通過")

    log("== T2b 反應式：額度錯誤若不是429（如403帳單上限）也要抓得到 ==")
    for status, code in ((403, "billing_hard_limit_reached"), (400, "quota_exceeded")):
        r = _fake_response(status, {"error": {"message": "billing", "type": code, "code": code}})
        try:
            raise_for_openai_response(r)
            raise AssertionError(f"預期 raise QuotaExhaustedError 但沒有（{status}/{code}）")
        except QuotaExhaustedError:
            pass
    log("  ✓ 通過（不綁定429，額度判定只認 error code/type）")

    log("== T3 反應式：一般 rate_limit_exceeded（同樣429）不該被誤判成額度用盡 ==")
    rate_resp = _fake_response(429, {
        "error": {"message": "Rate limit reached", "type": "requests",
                  "code": "rate_limit_exceeded"}})
    try:
        raise_for_openai_response(rate_resp)
        raise AssertionError("預期 raise OpenAIAPIError 但沒有")
    except QuotaExhaustedError:
        raise AssertionError("rate_limit_exceeded 不該被誤判成 QuotaExhaustedError")
    except OpenAIAPIError as e:
        assert e.code == "rate_limit_exceeded"
        log("  ✓ 通過（正確分類成 OpenAIAPIError，非額度問題）")

    log("== T4 反應式：其他4xx（如參數錯）要 raise OpenAIAPIError ==")
    bad_resp = _fake_response(400, {
        "error": {"message": "Invalid parameter", "type": "invalid_request_error",
                  "code": "invalid_parameter"}})
    try:
        raise_for_openai_response(bad_resp)
        raise AssertionError("預期 raise OpenAIAPIError 但沒有")
    except OpenAIAPIError as e:
        assert e.status_code == 400
        log("  ✓ 通過")

    log("== T5 主動式：_extract_total_usd 解析（照官方文件欄位路徑，未經真實資料驗證）==")
    fake_costs_body = {
        "object": "page",
        "data": [{
            "object": "bucket",
            "start_time": 1735084800, "end_time": 1735171200,
            "results": [
                {"object": "organization.costs.result",
                 "amount": {"value": 1.23, "currency": "usd"},
                 "line_item": None, "project_id": None},
                {"object": "organization.costs.result",
                 "amount": {"value": 0.45, "currency": "usd"},
                 "line_item": None, "project_id": None},
            ],
        }],
        "has_more": False, "next_page": None,
    }
    total = _extract_total_usd(fake_costs_body)
    assert abs(total - 1.68) < 1e-9, f"預期 1.68，實際 {total}"
    log(f"  ✓ 通過（合成資料算出 {total} 美金）")

    log("== T6 主動式：exhausted 方向不能顛倒 ==")
    under = UsageStatus(date="2026-08-25", total_usd=1.68, daily_limit_usd=5.0,
                        exhausted=1.68 >= 5.0, raw=None)
    over = UsageStatus(date="2026-08-25", total_usd=5.5, daily_limit_usd=5.0,
                       exhausted=5.5 >= 5.0, raw=None)
    assert under.exhausted is False and over.exhausted is True
    log("  ✓ 通過")

    log("== T7 免費額度：模型分池不能歸錯（mini 必須優先於 standard）==")
    assert classify_model("gpt-5-mini") == "mini", "gpt-5-mini 應歸小模型池，不是標準池"
    assert classify_model("gpt-5-nano") == "mini"
    assert classify_model("gpt-4o-mini-2024-07-18") == "mini", "帶版本後綴也要認得"
    assert classify_model("gpt-5") == "standard"
    assert classify_model("gpt-4o-2024-08-06") == "standard"
    assert classify_model("o3-mini") == "mini"
    assert classify_model("o3") == "standard"
    assert classify_model("") is None
    # 🔴 回歸測試：純 startswith 會讓 gpt-5.6 被誤判成免費的 gpt-5 而靜默燒錢，
    #    這是 2026-08-29 自我測試抓到的真實bug，見 _matches_family()
    assert classify_model("gpt-5.6-terra") is None, "gpt-5.6 不在名單，不可誤判成 gpt-5"
    assert classify_model("gpt-5.6-sol") is None
    assert classify_model("gpt-5.9") is None, "任何未來版本都不該被舊版前綴吃掉"
    assert classify_model("gpt-51") is None, "gpt-51 不是 gpt-5"
    assert classify_model("gpt-4.5") is None, "gpt-4.5 不在名單（名單只有4.1與4o）"
    # 反向：合法的版本/日期後綴仍要認得
    assert classify_model("gpt-5-2025-01-01") == "standard"
    assert classify_model("gpt-5.4") == "standard"
    assert classify_model("gpt-5.4-mini") == "mini"
    log("  ✓ 通過（含 gpt-5.6 誤判回歸測試）")

    log("== T8 免費額度：帳本寫入與當日彙總 ==")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ledger.jsonl"
        t0 = _dt.datetime(2026, 8, 29, 10, 0, tzinfo=_dt.timezone.utc)
        log_usage("gpt-5", "test_a", {"prompt_tokens": 100, "completion_tokens": 50}, now=t0, path=p)
        log_usage("gpt-5-mini", "test_b", {"prompt_tokens": 200, "completion_tokens": 80}, now=t0, path=p)
        log_usage("gpt-5.6-terra", "test_paid", {"total_tokens": 999}, now=t0, path=p)
        tot = today_totals(now=t0, path=p)
        assert tot["standard"] == 150, f"standard 應為150，實際{tot['standard']}"
        assert tot["mini"] == 280, f"mini 應為280，實際{tot['mini']}"
        assert tot["unclassified"] == 999, "不在名單的模型要單獨歸類，不能混進免費池"
        # 隔天不該累計進來
        t1 = t0 + _dt.timedelta(days=1)
        assert today_totals(now=t1, path=p)["standard"] == 0, "跨日必須重新計算"
        log("  ✓ 通過（含跨日重置）")

        log("== T9 免費額度：預算檢查與暫停機制 ==")
        # 額度還夠 → 不該 raise
        st = check_free_tier_budget("gpt-5", estimated_tokens=1000, now=t0, path=p)
        assert st["tier"] == "standard" and st["used"] == 150
        # 預估會爆掉 → 要 raise
        try:
            check_free_tier_budget("gpt-5", estimated_tokens=FREE_TIER_LIMITS["standard"],
                                   now=t0, path=p)
            raise AssertionError("預期 raise FreeTierExhaustedError 但沒有")
        except FreeTierExhaustedError:
            pass
        # 不在免費名單 → 直接 raise，不可靜默放行
        try:
            check_free_tier_budget("gpt-5.6-terra", now=t0, path=p)
            raise AssertionError("付費模型應該要被擋下並要求確認")
        except FreeTierExhaustedError:
            pass
        # reserve_ratio 要真的收緊門檻
        try:
            check_free_tier_budget("gpt-5", estimated_tokens=FREE_TIER_LIMITS["standard"] - 200,
                                   now=t0, path=p, reserve_ratio=0.5)
            raise AssertionError("reserve_ratio=0.5 應讓可用上限砍半而擋下")
        except FreeTierExhaustedError:
            pass
        log("  ✓ 通過")

    log("\n全部合成資料測試通過。"
        "⚠️ 但這只證明解析邏輯本身沒寫錯，不保證跟真實API回應欄位一致——"
        "admin_key 到位後仍須打一次真的來核對，見模組docstring。"
        "\n⚠️ 免費額度名單是使用者提供的當下狀態，OpenAI 可能調整，發現不符請更新常數。")


def cmd_usage(args) -> int:
    """看帳本：今日各池用量 + 歷史累計。"""
    recs = read_ledger()
    if not recs:
        print(f"帳本還沒有任何紀錄（{LEDGER_PATH}）")
        return 0

    tot = today_totals()
    print(f"=== 今日（{_day_key()} UTC）免費額度使用狀況 ===")
    for tier in ("standard", "mini"):
        used, limit = tot[tier], FREE_TIER_LIMITS[tier]
        pct = used / limit * 100 if limit else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {tier:9s} {bar} {used:>10,} / {limit:,} ({pct:.2f}%)")
    if tot.get("unclassified"):
        print(f"  ⚠️ 付費（不在免費名單）：{tot['unclassified']:,} tokens")

    print(f"\n=== 歷史累計（全部 {len(recs)} 次呼叫）===")
    by_purpose: dict[str, dict] = {}
    for r in recs:
        k = r.get("purpose", "?")
        d = by_purpose.setdefault(k, {"n": 0, "tokens": 0, "models": set()})
        d["n"] += 1
        d["tokens"] += int(r.get("total_tokens", 0) or 0)
        d["models"].add(r.get("model", "?"))
    for k, d in sorted(by_purpose.items(), key=lambda kv: -kv[1]["tokens"]):
        print(f"  {k:20s} {d['n']:>5} 次  {d['tokens']:>12,} tok  "
              f"模型：{', '.join(sorted(d['models']))}")
    days = sorted({r.get("day_utc", "?") for r in recs})
    print(f"\n涵蓋日期：{days[0]} ~ {days[-1]}（{len(days)} 天）")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="utils.openai_quota")
    ap.add_argument("--selftest", action="store_true", help="用合成資料驗證解析邏輯")
    ap.add_argument("--usage", action="store_true", help="查看用量帳本（今日額度 + 歷史累計）")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if a.usage:
        return cmd_usage(a)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
