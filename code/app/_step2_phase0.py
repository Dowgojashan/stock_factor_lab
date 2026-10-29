# -*- coding: utf-8 -*-
"""步驟2 階段0：不用 LLM 的換規則對照組（實戰收尾追蹤.md §四步驟2「階段0、1規格」）。

四組（台美 × 排除V1／含V1）× 三份選單（A 規則選擇器／B 參數設定／C 資金旋鈕），各算：
固定規則各自到底、上季最好續用、上季最差改用、警示觸發換市值、ε-greedy、Thompson、
隨機、事後最佳。隨機性的挑法跑 1,000 個種子，報平均與 5%／95% 區間。不呼叫 LLM。
輸出：`_analysis_outputs_applayer/step2/phase0_*.csv`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import pandas as pd  # noqa: E402

from app import rule_selector as RS  # noqa: E402

OUT = RS.ROOT / "_analysis_outputs_applayer" / "step2"
SEEDS = 1000
GROUPS = [("TW", "exclude_v1"), ("TW", "baseline"), ("US", "exclude_v1"), ("US", "baseline")]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=1, choices=[1, 2],
                    help="2＝版本2 選單（MENUS2，含熱門股／突破股）＋「訊號規則」對照，輸出 phase0v2_*")
    a = ap.parse_args(argv)
    v2 = a.version == 2
    OUT.mkdir(parents=True, exist_ok=True)
    from fcv_core import MarketData
    rows, picks_rows = [], []
    closes = {}
    for market, pool in GROUPS:
        g, c = RS.load_returns(market, pool, with_sig=v2)
        if market not in closes:
            closes[market] = MarketData(market).data.get("price:close")
        wb = RS.WeightBook(market, pool, closes[market], with_sig=v2)   # 2026-10-28：扣成本改用實際換手（換規則時的真實換手）
        states = RS.m1d_states(market, pool)
        if v2:
            from app import step2_signals
            sig = step2_signals.compute(market, pool)
        for plan, fn in (RS.MENUS2 if v2 else RS.MENUS).items():
            arms = fn(market, pool)
            missing = [a for a in arms if a not in g.columns]
            if missing:
                raise AssertionError(f"{market}/{pool}/{plan} 選單有規則不在步驟1結果裡：{missing[:3]}")
            det = {f"固定：{a}": f"fixed:{a}" for a in (arms if plan != "B" else ["REF_control0"])}
            det.update({"上季最好續用（追強）": "winner", "上季最差改用（逆勢）": "loser",
                        "警示觸發換市值": "trigger", "事後最佳（上限）": "oracle"})
            if v2:   # 不用 LLM 的「訊號規則」（優先順序事先寫死，見 rule_selector.SIGNAL_RULE）
                r = wb.settle(RS.run_signal_rule(sig, wb.as_ofs))
                rows.append({"market": market, "pool": pool, "plan": plan, "policy": "訊號規則（三訊號）",
                             "cum_gross": r["cum_gross"], "cum_net": r["cum_net"], "p05": np.nan, "p95": np.nan,
                             "p05_net": np.nan, "p95_net": np.nan})
                picks_rows.append({"market": market, "pool": pool, "plan": plan, "policy": "訊號規則（三訊號）",
                                   **{f"q{k+1}": p for k, p in enumerate(r["picks"])}})
            for name, rule in det.items():
                r = wb.settle(RS.run_deterministic(g, arms, rule, states))
                rows.append({"market": market, "pool": pool, "plan": plan, "policy": name,
                             "cum_gross": r["cum_gross"], "cum_net": r["cum_net"],
                             "p05": np.nan, "p95": np.nan, "p05_net": np.nan, "p95_net": np.nan})
                picks_rows.append({"market": market, "pool": pool, "plan": plan, "policy": name,
                                   **{f"q{k+1}": p for k, p in enumerate(r["picks"])}})
            for kind, name in (("egreedy", "ε-greedy（ε=0.1）"), ("thompson", "Thompson sampling"), ("random", "隨機挑")):
                res = [wb.settle(RS.run_bandit(g, arms, kind, s)) for s in range(SEEDS)]
                cg = np.array([x["cum_gross"] for x in res]); cn = np.array([x["cum_net"] for x in res])
                rows.append({"market": market, "pool": pool, "plan": plan, "policy": name,
                             "cum_gross": cg.mean(), "cum_net": cn.mean(),
                             "p05": np.percentile(cg, 5), "p95": np.percentile(cg, 95),
                             "p05_net": np.percentile(cn, 5), "p95_net": np.percentile(cn, 95)})
            print(f"{market}/{pool}/{plan}：選單 {len(arms)} 個規則", flush=True)
        # 方案D 的隨機基準（2026-10-28 操作定義）：候選＝票數前10名（同 stock_scorer_llm.candidates），
        # 每檔均勻隨機給加碼／持平／減碼，換算方式同方案D
        from app import elite_rules as ER
        from app import stock_scorer_llm as SD
        cand = SD.candidates(market, pool)
        rng = np.random.default_rng(0)
        resD = []
        for seed in range(SEEDS):
            ws = []
            for k, a in enumerate(wb.as_ofs):
                acts = rng.choice(list(SD.SHARE), size=len(cand[a]))
                codes = {f"C{i+1}": s_ for i, s_ in enumerate(cand[a])}
                elite = SD.elite_from_ratings([{"code": c_, "action": x} for c_, x in zip(codes, acts)], codes)
                ws.append(ER.overlay(wb.w0[a], elite, SD.SLEEVE))
            resD.append(wb.settle([(("randD", seed), k) for k in range(8)], ws))
        cg = np.array([x["cum_gross"] for x in resD]); cn = np.array([x["cum_net"] for x in resD])
        rows.append({"market": market, "pool": pool, "plan": "D", "policy": "隨機挑",
                     "cum_gross": cg.mean(), "cum_net": cn.mean(), "p05": np.percentile(cg, 5),
                     "p95": np.percentile(cg, 95), "p05_net": np.percentile(cn, 5), "p95_net": np.percentile(cn, 95)})
        print(f"{market}/{pool}/D：隨機評分 {SEEDS} 次", flush=True)
    df = pd.DataFrame(rows)
    tag = "phase0v2" if v2 else "phase0"
    df.to_csv(OUT / f"{tag}_policies.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(picks_rows).to_csv(OUT / f"{tag}_picks.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    for (m, p, plan), sub in df.groupby(["market", "pool", "plan"], sort=False):
        print(f"\n=== {m}/{p}／選單{plan} ===")
        s = sub[["policy", "cum_gross", "cum_net", "p05", "p95"]].copy()
        for col in ("cum_gross", "cum_net", "p05", "p95"):
            s[col] = s[col].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}")
        print(s.to_string(index=False))


if __name__ == "__main__":
    main()
