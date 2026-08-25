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
import sys
from dataclasses import dataclass
from typing import Any

import requests

OPENAI_API_BASE = "https://api.openai.com/v1"
REQUEST_TIMEOUT = 15   # 秒。查用量是輕量的 metadata 讀取，不是模型推論，不需要長逾時


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

    log("\n全部合成資料測試通過。"
        "⚠️ 但這只證明解析邏輯本身沒寫錯，不保證跟真實API回應欄位一致——"
        "admin_key 到位後仍須打一次真的來核對，見模組docstring。")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="utils.openai_quota")
    ap.add_argument("--selftest", action="store_true", help="用合成資料驗證解析邏輯")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
