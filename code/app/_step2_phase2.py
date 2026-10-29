# -*- coding: utf-8 -*-
"""步驟2 階段2：正式跑（實戰收尾追蹤.md §四步驟2；2026-10-28 使用者授權：含V1、排除V1 都跑）。

一組（市場×候選池）× 方案 A／B／C／D × 風格（平衡／追強／逆勢）× 重複 5 次 × 8 季。模型預設 gpt-5-mini。
  - 每次呼叫立即追加寫入 `step2/phase2_{market}_{pool}_{model}.jsonl`；重跑時自動跳過已完成的 (方案,風格,重複,季)
  - 免費額度不足（FreeTierExhaustedError）就停下，不改用付費模型；隔天重跑同一指令即可接續
  - 迴圈順序是「重複 → 方案 → 風格（平行）」，中途停下時先完成的是每個方案的第 1 次重複
  - 三種風格用執行緒平行；選擇只用季初資訊（同階段1）；結算另由 `_step2_phase2_settle.py` 用實際換手成本做
`--robust`：gpt-5 穩健性檢查（A、C，平衡型，1 次）。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401

from app import rule_selector as RS  # noqa: E402
from app import rule_selector_llm as RL  # noqa: E402
from app import stock_scorer_llm as SD  # noqa: E402

OUT = RS.ROOT / "_analysis_outputs_applayer" / "step2"
LOCK = threading.Lock()


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    keys = set()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        keys.add((r["plan"], r["style"], r["rep"], r["k"]))
    return keys


def run_chain(ctx, plan, style, rep, done, path, model, api_key):
    """一條 8 季決策鏈（同一方案、風格、重複）。"""
    from app.memo import _call_llm
    from utils import openai_quota as OQ
    for k in range(1, 9):
        if (plan, style, rep, k) in done:
            continue
        try:
            if plan in ("A", "B", "C"):
                if ctx["version"] == 2:
                    facts = RL.build_facts_v2(ctx["market"], ctx["pool"], plan, k, ctx["g"], ctx["sig"], ctx["bq"])
                    r = RL.decide_v2(plan, style, facts, model=model, api_key=api_key)
                else:
                    facts = RL.build_facts(ctx["market"], ctx["pool"], plan, k, ctx["g"], ctx["states"], ctx["bq"])
                    r = RL.decide(plan, style, facts, model=model, api_key=api_key)
                rec = {"choice": r["choice"], "combo": {x["code"]: x["combo"] for x in facts["rows"]}[r["choice"]],
                       "reason": r["reason"], "leakage": r["leakage"], "usage": r["usage"]}
            else:
                as_of = ctx["as_ofs"][k - 1]
                f = SD.features(ctx["md"], ctx["close"], ctx["mw"], ctx["cand"][as_of], as_of)
                tab, codes = SD.anonymized_table(f, seed=1000 * rep + k)
                prompt = (SD.build_prompt_v2(k, ctx["sig"].loc[as_of], ctx["bq"][k - 1], tab) if ctx["version"] == 2
                          else SD.build_prompt(k, ctx["states"][as_of], ctx["dev"][as_of], ctx["bq"][k - 1], tab))
                out = None
                for _ in range(2):   # 評分不完整重問一次
                    out, usage = _call_llm(prompt, model, api_key, purpose=f"step2{'v2' if ctx['version'] == 2 else ''}_D_{style}", est_tokens=6000,
                                           system_prompt=SD.SYSTEM + (RL.STYLES_V2 if ctx["version"] == 2 else RL.STYLES)[style], schema=SD.schema(list(codes)))
                    if not SD.check_ratings(out["ratings"], codes):
                        break
                probs = SD.check_ratings(out["ratings"], codes)
                if probs:
                    raise RuntimeError(f"D 評分仍不完整：{probs}")
                rec = {"ratings": out["ratings"], "codes": codes, "reason": out["reason"], "usage": usage}
        except OQ.FreeTierExhaustedError as e:
            print(f"⛔ 免費額度不足，停止（{plan}/{style}/rep{rep}/第{k}季）：{e}", flush=True)
            return "quota"
        rec.update({"plan": plan, "style": style, "rep": rep, "k": k, "model": model})
        with LOCK:
            with open(path, "a", encoding="utf-8") as fo:
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{ctx['market']}/{ctx['pool']} {plan}/{style}/rep{rep} 第{k}季 ✓ tokens {rec['usage'].get('total_tokens')}", flush=True)
    return "ok"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True, choices=["TW", "US"])
    ap.add_argument("--pool", required=True, choices=["exclude_v1", "baseline"])
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--plans", default="A,B,C,D"); ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--styles", default="balanced,momentum,contrarian")
    ap.add_argument("--robust", action="store_true", help="gpt-5 穩健性檢查：A、C，平衡型，1 次")
    ap.add_argument("--version", type=int, default=1, choices=[1, 2],
                    help="2＝三種訊號配對＋完整監控訊號（選單 MENUS2、prompt 含五個訊號），輸出 phase2v2_*")
    a = ap.parse_args(argv)
    if a.robust:
        a.model, a.plans, a.reps, a.styles = "gpt-5", "A,C", 1, "balanced"
    from app import simulate
    from utils.config import Config
    api_key = Config().get_openai_api_key()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (f"phase2v2_{a.market}_{a.pool}_{a.model}.jsonl" if a.version == 2
                  else f"phase2_{a.market}_{a.pool}_{a.model}.jsonl")
    done = done_keys(path)
    g, _ = RS.load_returns(a.market, a.pool, with_sig=(a.version == 2))
    ctx = {"market": a.market, "pool": a.pool, "g": g, "version": a.version, "states": RS.m1d_states(a.market, a.pool),
           "dev": RL.load_deviation(a.market, a.pool), "bq": RL.bench_last_quarter(a.market),
           "as_ofs": [simulate.REGISTRATION_DATE] + list(simulate.QUARTER_ENDS[:-1])}
    if a.version == 2:
        from app import step2_signals
        ctx["sig"] = step2_signals.compute(a.market, a.pool)
    plans = a.plans.split(",")
    if "D" in plans:
        from app._step2_planD import setup
        ctx["md"], ctx["close"], ctx["mw"] = setup(a.market, a.pool)
        ctx["cand"] = SD.candidates(a.market, a.pool)
    print(f"[{a.market}/{a.pool}] 模型 {a.model}｜方案 {plans}｜風格 {a.styles}｜重複 {a.reps}｜已完成 {len(done)} 次", flush=True)
    styles = a.styles.split(",")
    # 2026-10-28：改成「重複」在外層——額度中途用完時，已完成的是「所有方案的前 n 次重複」，可直接互相比較
    for rep in range(1, a.reps + 1):
        for plan in plans:
            with ThreadPoolExecutor(max_workers=len(styles)) as ex:
                res = list(ex.map(lambda st: run_chain(ctx, plan, st, rep, done, path, a.model, api_key), styles))
            if "quota" in res:
                print("⛔ 額度不足，已停止；隔天重跑同一指令會從中斷處接續", flush=True)
                return
    print(f"[{a.market}/{a.pool}] 全部完成", flush=True)


if __name__ == "__main__":
    main()
