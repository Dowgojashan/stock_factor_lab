# -*- coding: utf-8 -*-
"""步驟2 階段1：LLM 試跑（實戰收尾追蹤.md §四步驟2「階段0、1規格」）。

台股・排除V1、平衡型、跑 1 次 8 季，方案 A／B／C 各 8 次呼叫。只用來找 bug、看 LLM 是否照指示，
不用來挑方案。`--dry-run` 只印第一季的 prompt 與長度，不呼叫 LLM。
輸出：`_analysis_outputs_applayer/step2/phase1_pilot_*.jsonl`（每次呼叫的 prompt／回覆／用量）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401

from app import rule_selector as RS  # noqa: E402
from app import rule_selector_llm as RL  # noqa: E402

OUT = RS.ROOT / "_analysis_outputs_applayer" / "step2"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="TW"); ap.add_argument("--pool", default="exclude_v1")
    ap.add_argument("--style", default="balanced", choices=list(RL.STYLES))
    ap.add_argument("--plans", default="A,B,C"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=None, help="覆蓋模型（例：gpt-5-mini）；不給則用 config 的 app_memo_model")
    ap.add_argument("--tag", default="", help="輸出檔名後綴，避免覆蓋前一次試跑")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    g, c = RS.load_returns(a.market, a.pool)
    states = RS.m1d_states(a.market, a.pool)
    RL.load_deviation(a.market, a.pool)
    bq = RL.bench_last_quarter(a.market)
    model = api_key = None
    if not a.dry_run:
        from utils.config import Config
        cfg = Config(); api_key = cfg.get_openai_api_key(); model = a.model or cfg.get_openai_model("app_memo")
        print(f"模型：{model}")
    for plan in a.plans.split(","):
        picks, log = [], []
        for k in range(1, 9):
            facts = RL.build_facts(a.market, a.pool, plan, k, g, states, bq)
            r = RL.decide(plan, a.style, facts, model=model, api_key=api_key, dry_run=a.dry_run)
            if a.dry_run:
                print(f"\n===== 方案{plan} 第1季 prompt（{len(r['system']) + len(r['prompt'])} 字）=====\n{r['system']}\n\n{r['prompt']}")
                break
            combo = {x["code"]: x["combo"] for x in facts["rows"]}[r["choice"]]
            picks.append(combo)
            log.append({"plan": plan, "style": a.style, "k": k, "as_of": facts["as_of"], "choice": r["choice"],
                        "combo": combo, "reason": r["reason"], "leakage": r["leakage"], "usage": r["usage"],
                        "prompt": r["prompt"]})
            print(f"方案{plan} 第{k}季：{r['choice']} {RL.describe(combo)}｜{r['reason'][:80]}"
                  f"｜tokens {r['usage'].get('total_tokens')}" + (f"｜⚠️洩漏 {r['leakage']}" if r["leakage"] else ""), flush=True)
        if a.dry_run:
            continue
        res = RS.settle(g, c, picks)
        print(f"==> 方案{plan}（{a.style}）8季累積 未扣成本 {res['cum_gross']:+.1%}｜近似扣成本 {res['cum_net_approx']:+.1%}")
        with open(OUT / f"phase1_pilot_{a.market}_{a.pool}_{plan}_{a.style}{a.tag}.jsonl", "w", encoding="utf-8") as f:
            for x in log:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
            f.write(json.dumps({"summary": res}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
