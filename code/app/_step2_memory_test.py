# -*- coding: utf-8 -*-
"""步驟2 方案D 記憶測試（實戰收尾追蹤.md §四步驟2「階段0、1規格」）。

問題：LLM 的訓練資料含 2024–2025 行情，給它去識別化的數字，它還能不能認出「是哪個市場／哪幾年」
或「是哪家公司」？認得出來＝會用記憶作答（事後知識污染），方案D（逐檔質性評分）就不能照原設計跑。
4 次呼叫：市場層級 TW、US（8 季大盤季報酬＋集中度累計偏離），個股層級 2 檔（2024-06-30 的
台股市值第一大、美股 NVDA；只給百分位與近期報酬，不給價格、名稱、日期）。
輸出：`_analysis_outputs_applayer/step2/memory_test.jsonl`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import pandas as pd  # noqa: E402

from app import rule_selector as RS  # noqa: E402
from app import rule_selector_llm as RL  # noqa: E402

OUT = RS.ROOT / "_analysis_outputs_applayer" / "step2"
SYSTEM = ("你是金融研究助理。以下資料已去識別化。請誠實回答你能否從這些數字辨識出真實的市場、年份或公司；"
          "如果無法辨識就直接說無法辨識，不要編造。信心用 0~100 表示。")
SCHEMA = {"name": "memory_probe", "strict": True,
          "schema": {"type": "object", "additionalProperties": False,
                     "required": ["guess", "confidence", "reasoning"],
                     "properties": {"guess": {"type": "string"}, "confidence": {"type": "integer"},
                                    "reasoning": {"type": "string"}}}}


def market_prompt(market, pool):
    from app import simulate
    from app._elite_grid_offline import load_benchmark
    b = load_benchmark(market)
    ends = [simulate.REGISTRATION_DATE] + list(simulate.QUARTER_ENDS)
    rets = [float(b[b.index <= pd.Timestamp(e1)].iloc[-1] / b[b.index <= pd.Timestamp(e0)].iloc[-1] - 1)
            for e0, e1 in zip(ends[:-1], ends[1:])]
    dev = RL.load_deviation(market, pool)
    devs = [dev.get(e, None) for e in ends[1:-1]]
    lines = ["某個股票市場連續 8 季的大盤（含息）季報酬：" + "、".join(f"第{k+1}季 {r:+.1%}" for k, r in enumerate(rets)),
             "同期「市值前20%公司佔全市場市值比重」相對起點的累計變化（個百分點，季末）：" +
             "、".join(f"第{k+1}季 {d:+.2f}" for k, d in enumerate(devs)),
             "問題：你能辨識這是哪個國家的市場、哪幾年嗎？請給出你的猜測（市場與起訖年份）。"]
    return "\n".join(lines)


def stock_prompt(market, sym, as_of):
    from fcv_core import MarketData
    from app import monitor
    from database import Database
    md = MarketData(market); close = md.data.get("price:close")
    ts = pd.Timestamp(as_of); idx = close.index; d1 = idx[idx <= ts].max()
    feats = []
    for m in (1, 3, 6, 12):
        d0 = idx[idx <= ts - pd.DateOffset(months=m)].max()
        feats.append(f"近{m}個月報酬 {close.loc[d1, sym] / close.loc[d0, sym] - 1:+.0%}")
    mw = monitor.fetch_mcap_wide(Database(market).create_connection(), market=market, start="2023-01-01")
    row = monitor.asof_row(mw, as_of).dropna()
    rank = int(row.rank(ascending=False)[sym]); share = float(row[sym] / row.sum())
    pcts = []
    for f in ("pb", "roe", "mom"):
        fr = md.get_field(f"report:{f}")
        if fr is not None and sym in fr.columns:
            pcts.append(f"{f.upper()} 全市場百分位 {fr.loc[d1].rank(pct=True)[sym]:.0%}")
    return ("某市場中的一家上市公司，某季季末的特徵（已去識別化）：\n"
            f"市值排名第 {rank} 名（全市場 {len(row)} 家），約佔全市場市值 {share:.1%}；"
            + "；".join(feats) + "；" + "；".join(pcts) +
            "\n問題：你能辨識這是哪一家公司嗎？請給出你的猜測。")


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=None); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    from app.memo import _call_llm
    from utils.config import Config
    cfg = Config(); api_key = cfg.get_openai_api_key(); model = a.model or cfg.get_openai_model("app_memo")
    OUT.mkdir(parents=True, exist_ok=True)
    tests = [("market_TW", market_prompt("TW", "exclude_v1")), ("market_US", market_prompt("US", "exclude_v1")),
             ("stock_TW_top1", stock_prompt("TW", "2330", "2024-06-30")), ("stock_US_NVDA", stock_prompt("US", "NVDA", "2024-06-30"))]
    with open(OUT / f"memory_test{a.tag}.jsonl", "w", encoding="utf-8") as f:
        for name, prompt in tests:
            out, usage = _call_llm(prompt, model, api_key, purpose="step2_memory_test", est_tokens=6000,
                                   system_prompt=SYSTEM, schema=SCHEMA)
            rec = {"test": name, "prompt": prompt, **out, "usage": usage}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"\n=== {name} ===\n{prompt}\n--> 猜測：{out['guess']}｜信心 {out['confidence']}｜{out['reasoning'][:200]}", flush=True)


if __name__ == "__main__":
    main()
