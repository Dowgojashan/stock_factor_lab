# -*- coding: utf-8 -*-
"""L2+ · 解釋 agent 重複執行穩定度（應用層開發追蹤.md §9.8 四層驗證標準·層三，2026-09-12）

沿用 `research/decision_repeatability.py`（S-07）的方法論精神——同一份輸入
重複呼叫真實 API N 次（每次都是獨立呼叫，不是快取重播），量化穩不穩定。

但 S-07 測的是一個離散的「選中哪些群」集合，天然可以算交集/聯集/Jaccard；
這裡的輸出是十個自由文字欄位，沒有現成的集合可比，逐字比對又太嚴格（同一件事
換個說法就會判定不一致）。改用跟 D2（`memo.scan_for_leakage`）同一套技術：
從每個欄位的文字裡抽出**引用了哪些數字**，把「這個欄位引用的數字集合」當成
可比較的離散物件——同一份 facts，五次獨立呼叫若講的是同一件事，引用的核心
數字應該高度重疊（措辭可以不同，但不能這次講 A_hrp 贏 16.8%、下次講贏 27%）。
這是「核心判斷穩不穩」的一個可查證代理指標，不是唯一標準（見 §9.8 層三：
「核心判斷 5 次一致；邊緣措辭可浮動」——本工具只能量到跟數字有關的那部分核心
判斷，純敘述性的判斷穩定度仍需人工核對，跟 D2 的已知限制同一類）。

⚠️ 每呼叫一次 `repeatability_check()` 就是 `n_repeats` 次真實 API 呼叫，
**要花錢**，額度記帳跟 `explain.generate()` 共用同一套 `openai_quota`，
呼叫前務必評估要對 9 組裡的哪幾組跑（不必全部都跑 5 次）。
"""
from __future__ import annotations

import itertools
from collections import Counter

from . import explain as EX
from .memo import _NUMBER_RE

DEFAULT_N_REPEATS = 5


def _numbers_in(text: str) -> frozenset[float]:
    out = set()
    for n in _NUMBER_RE.findall(text):
        n_clean = n.rstrip("%")
        if not n_clean or n_clean in ("-", "."):
            continue
        out.add(round(float(n_clean), 4))
    return frozenset(out)


def repeatability_check(holdings, risk, calib, *, stock_detail, stock_concentration,
                        market_dates: dict[str, str] | None = None,
                        footprint=None, md_map: dict | None = None,
                        face_comparison=None, n_repeats: int = DEFAULT_N_REPEATS,
                        model: str | None = None, purpose: str = "app_memo",
                        log=print) -> dict:
    """對同一份 facts snapshot 重複呼叫 `explain.generate()` `n_repeats` 次，
    逐欄位量化「引用數字集合」的穩定度。

    facts 只組裝一次（`assemble_facts()`），確保 `n_repeats` 次真的是同一份
    輸入重複問，不是每次都重新抓一次可能已經變動的資料庫/檔案狀態。
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats 必須 >= 1，收到 {n_repeats}")

    facts = EX.assemble_facts(holdings, risk, calib, stock_detail=stock_detail,
                              stock_concentration=stock_concentration,
                              market_dates=market_dates, footprint=footprint,
                              md_map=md_map, face_comparison=face_comparison)
    prompt = EX.build_prompt(facts)

    from utils.config import Config
    cfg = Config()
    api_key = cfg.get_openai_api_key()
    actual_model = model or cfg.get_openai_model(purpose)

    # 🔴 2026-09-12 code review：每次呼叫都是獨立的真實 API 呼叫、都要花錢——
    # 原本第 4 次網路逾時／refusal 就會讓整個函式連同前 3 次已經成功（已經
    # 花錢）的結果一起丟掉，逼使用者整組重跑再花一次錢。改成單次失敗只跳過、
    # 記錄下來，用**實際成功的次數**算後續統計，不是原本要求的 `n_repeats`。
    runs: list[dict] = []
    errors: list[str] = []
    for i in range(n_repeats):
        try:
            explanation, _usage = EX._call_llm(
                prompt, actual_model, api_key, purpose=purpose, est_tokens=len(prompt) // 3,
                system_prompt=EX._SYSTEM_PROMPT, schema=EX._EXPLAIN_SCHEMA)
        except RuntimeError as e:
            errors.append(str(e))
            log(f"  第 {i + 1}/{n_repeats} 次失敗（跳過，不影響已成功的結果）：{e}")
            continue
        runs.append(explanation)
        log(f"  第 {i + 1}/{n_repeats} 次完成")

    n_success = len(runs)
    if n_success == 0:
        raise RuntimeError(
            f"{n_repeats} 次呼叫全部失敗，沒有任何結果可統計：\n  " + "\n  ".join(errors))

    fields = list(EX._EXPLAIN_SCHEMA["schema"]["properties"])
    per_field = {}
    for field in fields:
        sets = [_numbers_in(r[field]) for r in runs]
        counts = Counter(sets)
        _, mode_count = counts.most_common(1)[0]
        core = frozenset.intersection(*sets) if sets else frozenset()
        union = frozenset.union(*sets) if sets else frozenset()
        pairs = list(itertools.combinations(sets, 2))
        jaccards = [len(a & b) / len(a | b) if (a | b) else 1.0 for a, b in pairs]
        per_field[field] = {
            "exact_match_rate": round(mode_count / n_success, 4),
            "mean_pairwise_jaccard": round(sum(jaccards) / len(jaccards), 4) if jaccards else None,
            "stable_core_numbers": sorted(core),
            "unstable_fringe_numbers": sorted(union - core),
        }

    overall_jaccard = [v["mean_pairwise_jaccard"] for v in per_field.values()
                       if v["mean_pairwise_jaccard"] is not None]
    return {
        "n_repeats": n_repeats, "n_success": n_success, "errors": errors,
        "model": actual_model, "per_field": per_field,
        "overall_mean_jaccard": (round(sum(overall_jaccard) / len(overall_jaccard), 4)
                                 if overall_jaccard else None),
        "runs": runs,
    }
