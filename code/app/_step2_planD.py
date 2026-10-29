# -*- coding: utf-8 -*-
"""步驟2 方案D：記憶測試（--memory-test）與試跑（--pilot）。規格見 `stock_scorer_llm.py`。

--memory-test：用實際的 D prompt（台股・排除V1 第3季）＋「強制把台積電放進候選」的最壞情境，
  問 LLM 能否辨識任何一家公司；gpt-5-mini 與 gpt-5 各問一次（共 4 次呼叫）。
--pilot：台股・排除V1、平衡型、8 季（8 次呼叫）；並先驗證「全部持平且資金0」時的報酬＝步驟1 control0。
輸出：`_analysis_outputs_applayer/step2/planD_*.jsonl`
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import pandas as pd  # noqa: E402

from app import rule_selector as RS  # noqa: E402
from app import rule_selector_llm as RL  # noqa: E402
from app import stock_scorer_llm as SD  # noqa: E402

OUT = RS.ROOT / "_analysis_outputs_applayer" / "step2"
PROBE_SYSTEM = ("你是金融研究助理。以下資料已去識別化。請誠實回答你能否從表中辨識出任何一家真實公司或市場、年份；"
                "無法辨識就直接說無法辨識，不要編造。信心用 0~100 表示。")
PROBE_SCHEMA = {"name": "memory_probe", "strict": True,
                "schema": {"type": "object", "additionalProperties": False, "required": ["guess", "confidence", "reasoning"],
                           "properties": {"guess": {"type": "string"}, "confidence": {"type": "integer"},
                                          "reasoning": {"type": "string"}}}}


def setup(market, pool):
    from database import Database
    from fcv_core import MarketData
    from app import monitor
    md = MarketData(market); close = md.data.get("price:close")
    mw = monitor.fetch_mcap_wide(Database(market).create_connection(), market=market, start="2022-01-01")
    return md, close, mw


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="TW"); ap.add_argument("--pool", default="exclude_v1")
    ap.add_argument("--style", default="balanced", choices=list(RL.STYLES))
    ap.add_argument("--memory-test", action="store_true"); ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--model", default="gpt-5-mini"); ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    from app.memo import _call_llm
    from app import simulate
    from utils.config import Config
    api_key = Config().get_openai_api_key()
    OUT.mkdir(parents=True, exist_ok=True)
    md, close, mw = setup(a.market, a.pool)
    cand = SD.candidates(a.market, a.pool)
    states = RS.m1d_states(a.market, a.pool); dev = RL.load_deviation(a.market, a.pool)
    bq = RL.bench_last_quarter(a.market)
    as_ofs = [simulate.REGISTRATION_DATE] + list(simulate.QUARTER_ENDS[:-1])
    ends = list(simulate.QUARTER_ENDS)

    if a.memory_test:
        k = 3; as_of = as_ofs[k - 1]
        tests = []
        f = SD.features(md, close, mw, cand[as_of], as_of)
        tab, _ = SD.anonymized_table(f, seed=1000 + k)
        tests.append(("D_actual", SD.build_prompt(k, states[as_of], dev[as_of], bq[k - 1], tab)))
        worst = "2330" if a.market == "TW" else "NVDA"
        f2 = SD.features(md, close, mw, sorted(set(cand[as_of]) | {worst}), as_of)
        tab2, codes2 = SD.anonymized_table(f2, seed=2000 + k)
        tests.append(("D_with_top_mcap", SD.build_prompt(k, states[as_of], dev[as_of], bq[k - 1], tab2)))
        hidden = [c for c, s in codes2.items() if s == worst][0]
        with open(OUT / f"planD_memory_test{a.tag}.jsonl", "a", encoding="utf-8") as fo:
            for model in ("gpt-5-mini", "gpt-5"):
                for name, prompt in tests:
                    q = prompt + "\n\n問題：你能從這張表辨識出任何一檔是哪一家真實公司嗎？請給出你的猜測（代號與公司）。"
                    out, usage = _call_llm(q, model, api_key, purpose="step2_planD_memory", est_tokens=6000,
                                           system_prompt=PROBE_SYSTEM, schema=PROBE_SCHEMA)
                    rec = {"model": model, "test": name, "hidden_code": hidden if name == "D_with_top_mcap" else None,
                           **out, "usage": usage, "prompt": q}
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(f"\n=== {model}／{name}" + (f"（市值最大股藏在 {hidden}）" if name == "D_with_top_mcap" else "") +
                          f" ===\n--> 猜測：{out['guess']}｜信心 {out['confidence']}｜{out['reasoning'][:220]}", flush=True)
        if name:  # 印一份 prompt 供人工檢查
            print("\n----- D prompt 範例 -----\n" + tests[0][1])

    if a.pilot:
        w0 = pd.read_parquet(RS.GRID / f"w0_{RS.TAG[(a.market, a.pool)]}.parquet")
        w0 = w0[w0.window == 4]
        q = pd.read_parquet(RS.GRID / f"quarterly_{RS.TAG[(a.market, a.pool)]}.parquet")
        c0 = q[(q.window == 4) & (q.combo == "REF_control0")].set_index("as_of").ret_gross
        log, rets = [], []
        for k, (as_of, end) in enumerate(zip(as_ofs, ends), start=1):
            w0q = dict(zip(w0[w0.as_of == as_of].symbol, w0[w0.as_of == as_of].w0))
            chk = SD.quarter_return(close, w0q, {}, as_of, end)
            if abs(chk - c0[as_of]) > 1e-9:
                raise AssertionError(f"{as_of} 對帳失敗：D 結算的 control0 {chk:.6f} ≠ 步驟1 {c0[as_of]:.6f}")
            # 對帳②：全部「持平」＝候選平均分；若這季步驟1 的票數前10名沒有邊界同分（權重都相等），
            # 應等於步驟1 的「票數・前10名・平均分・資金30%」
            el = pd.read_parquet(RS.GRID / f"elite_weights_{RS.TAG[(a.market, a.pool)]}.parquet")
            el = el[(el.window == 4) & (el.as_of == as_of) & (el.score == "S1_votes") & (el.n == 10) & (el.scheme == "equal")]
            if el.elite_weight.nunique() == 1:
                flat = SD.quarter_return(close, w0q, {s: 1 / len(cand[as_of]) for s in cand[as_of]}, as_of, end)
                ref = q[(q.window == 4) & (q.as_of == as_of) & (q.combo == "S1_votes|N10|equal|sl30")].ret_gross.iloc[0]
                if abs(flat - ref) > 1e-9:
                    raise AssertionError(f"{as_of} 對帳失敗：D 全部持平 {flat:.6f} ≠ 步驟1 票數前10名 {ref:.6f}")
            f = SD.features(md, close, mw, cand[as_of], as_of)
            tab, codes = SD.anonymized_table(f, seed=1000 + k)
            prompt = SD.build_prompt(k, states[as_of], dev[as_of], bq[k - 1], tab)
            out, usage = _call_llm(prompt, a.model, api_key, purpose=f"step2_D_{a.style}", est_tokens=6000,
                                   system_prompt=SD.SYSTEM + RL.STYLES[a.style], schema=SD.schema(list(codes)))
            probs = SD.check_ratings(out["ratings"], codes)
            if probs:   # 評分不完整：重問一次，仍不完整就停，不用預設值悄悄補
                out, usage = _call_llm(prompt, a.model, api_key, purpose=f"step2_D_{a.style}", est_tokens=6000,
                                       system_prompt=SD.SYSTEM + RL.STYLES[a.style], schema=SD.schema(list(codes)))
                probs = SD.check_ratings(out["ratings"], codes)
                if probs:
                    raise RuntimeError(f"第{k}季 LLM 評分仍不完整：{probs}")
            elite = SD.elite_from_ratings(out["ratings"], codes)
            r = SD.quarter_return(close, w0q, elite, as_of, end)
            rets.append(r)
            acts = {x["code"]: x["action"] for x in out["ratings"]}
            n_up = sum(v == "加碼" for v in acts.values()); n_dn = sum(v == "減碼" for v in acts.values())
            log.append({"k": k, "as_of": as_of, "codes": codes, "ratings": out["ratings"], "reason": out["reason"],
                        "ret": r, "control0": float(c0[as_of]), "usage": usage, "prompt": prompt})
            print(f"第{k}季：候選{len(codes)}檔｜加碼{n_up}／減碼{n_dn}｜本季 {r:+.2%}（control0 {c0[as_of]:+.2%}）"
                  f"｜漏評{len(codes) - len(acts)}｜tokens {usage.get('total_tokens')}｜{out['reason'][:70]}", flush=True)
        cum = float(pd.Series(rets).add(1).prod() - 1)
        print(f"==> 方案D（{a.style}，{a.model}）8季累積 未扣成本 {cum:+.1%}（control0 {float(c0.add(1).prod() - 1):+.1%}）")
        with open(OUT / f"planD_pilot_{a.market}_{a.pool}_{a.style}{a.tag}.jsonl", "w", encoding="utf-8") as fo:
            for x in log:
                fo.write(json.dumps(x, ensure_ascii=False) + "\n")
            fo.write(json.dumps({"summary": {"cum_gross": cum}}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
