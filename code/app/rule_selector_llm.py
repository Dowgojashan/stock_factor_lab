# -*- coding: utf-8 -*-
"""步驟2：LLM 每季從規則選單挑一個（方案 A／B／C），三種風格（方案 E）。

規格見 實戰收尾追蹤.md §四步驟2「階段0、1規格」（2026-10-28 寫死）：
  - 資訊全部去識別化：不給公司名、日期、年份；季度用「第 k 季」；規則用中文描述＋代號
  - 只給季初（as_of）以前的資訊：上季報酬、近四季平均、2015–2023（以「歷史期」稱）平均與波動、
    集中度警示狀態與累計偏離、大盤上季報酬
  - LLM 只能回傳選單中的代號（json_schema enum 強制），結算一律用步驟1 的規則報酬表
呼叫沿用 `memo._call_llm`（含免費額度煞車與用量記帳）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from app import rule_selector as RS

SCORE_LABEL = {"S1_votes": "票數", "S2_f1": "因子F1", "S2_f1f2": "因子F1+F2", "S6_mcap": "市值",
               "S7_hot": "熱門股（全市場近期漲幅前10%）", "S8_breakout": "突破股（創近20個交易日新高）"}
for _L in (1, 3, 6):
    SCORE_LABEL[f"S3_greedy_L{_L}"] = f"追強{_L}個月"
    SCORE_LABEL[f"S4_reversal_L{_L}"] = f"押回彈{_L}個月"
    SCORE_LABEL[f"S5_mixed_L{_L}"] = f"追強＋押回彈{_L}個月"
SCHEME_LABEL = {"equal": "平均分", "top1x2": "第一名加碼", "top2x2": "前兩名加碼",
                "proportional": "照分數比例", "tier": "階梯"}
STATE_LABEL = {"NONE": "正常", "OBSERVING": "觀察", "TRIGGERED": "觸發"}

STYLES = {
    "balanced": "你的風格是「平衡型」：優先看市場監控訊號（集中度警示）判斷市況；沒有明確訊號時偏向維持不調整（control0），"
                "有明確訊號時才選擇與訊號相符的規則。",
    "momentum": "你的風格是「追強型」：相信近期有效的做法會延續，傾向選擇最近表現最好的規則。",
    "contrarian": "你的風格是「逆勢型」：相信近期失效的做法會回來、近期過熱的做法會反轉，傾向選擇最近表現落後但歷史表現不差的規則。",
}
# 版本2：平衡型改成看三種訊號（2026-10-28 dry-run 發現沿用版本1「集中度警示」字樣，會讓 LLM 只看集中度）；追強／逆勢不變
STYLES_V2 = {**STYLES, "balanced": STYLES["balanced"].replace("（集中度警示）", "（集中度警示、熱門股覆蓋、突破廣度三種訊號）")}
assert STYLES_V2["balanced"] != STYLES["balanced"]
SYSTEM_BASE = ("你是基金的投資組合經理人。研究部每季給你一份已選好的股票名單（control0：每個策略等權、策略內平分），"
               "你的工作是每季季初從「規則選單」中選一個規則，決定下一季怎麼對名單裡的優等生加碼。"
               "你只能根據提供的數字判斷，不可使用你對任何真實市場、年份或公司的記憶；"
               "提供的資料已去識別化，不要嘗試推測是哪個市場或哪一年。只能回傳選單中的代號。")


def describe(combo: str) -> str:
    if combo == "REF_control0":
        return "不調整（control0）"
    score, n, scheme, sl = combo.split("|")
    n = int(n[1:]); sl = int(sl[2:])
    parts = [f"排名依據：{SCORE_LABEL[score]}", f"取前{n}名"]
    if n > 1:
        parts.append(f"分法：{SCHEME_LABEL[scheme]}")
    parts.append(f"優等生資金{sl}%")
    return "，".join(parts)


def _series(g: pd.DataFrame, arms) -> pd.DataFrame:
    """window1~4 依時間排好的季報酬（列＝季）。"""
    return g[arms].reset_index(drop=True)


def build_facts(market, pool, plan, k: int, g, states, bench_q) -> dict:
    """第 k 季（1..8）季初可見的資訊。"""
    arms = RS.MENUS[plan](market, pool)
    allq = _series(g, arms)
    n_hist = int((g.index.get_level_values("window") < 4).sum())
    upto = allq.iloc[: n_hist + k - 1]                 # 到季初為止已實現的季
    hist = allq.iloc[:n_hist]
    as_of = g[g.index.get_level_values("window") == 4].index[k - 1][2]
    if (market, pool) not in DEV:
        raise RuntimeError("先呼叫 load_deviation(market, pool)")
    rows = []
    for i, a in enumerate(arms):
        rows.append({"code": f"R{i}", "combo": a, "desc": describe(a),
                     "last_q": float(upto[a].iloc[-1]), "avg_4q": float(upto[a].iloc[-4:].mean()),
                     "hist_mean": float(hist[a].mean()), "hist_std": float(hist[a].std(ddof=1))})
    return {"k": k, "as_of": as_of, "state": states[as_of], "deviation_pp": DEV[(market, pool)].get(as_of, 0.0),
            "bench_last_q": bench_q[k - 1], "rows": rows}


DEV: dict = {}


def load_deviation(market, pool) -> dict:
    """{as_of: 季初可見的集中度累計偏離(pp)}。來源同 `rule_selector.m1d_series`（美股用修正後資料）。"""
    from app import simulate
    s = RS.m1d_series(market, pool)
    out = {simulate.REGISTRATION_DATE: 0.0}
    for qe in simulate.QUARTER_ENDS[:-1]:
        out[qe] = float(s.loc[qe, "cumulative_deviation"]) * 100
    DEV[(market, pool)] = out
    return out


def bench_last_quarter(market) -> list[float]:
    """window4 每季季初可見的「大盤上一季報酬」。"""
    from app import simulate
    from app._elite_grid_offline import load_benchmark
    b = load_benchmark(market)
    ends = [simulate.REGISTRATION_DATE] + list(simulate.QUARTER_ENDS[:-1])
    out = []
    for e in ends:
        e = pd.Timestamp(e)
        s = e - pd.offsets.QuarterEnd(1)
        out.append(float(b[b.index <= e].iloc[-1] / b[b.index <= s].iloc[-1] - 1))
    return out


def build_prompt(plan: str, facts: dict) -> str:
    lines = [f"現在是評估期的第 {facts['k']} 季季初（共 8 季）。",
             f"市場集中度警示：{STATE_LABEL[facts['state']]}（相對評估期起點的集中度累計偏離 {facts['deviation_pp']:+.2f} 個百分點；"
             "偏離超過觀察門檻為「觀察」、超過觸發門檻為「觸發」）。",
             # 2026-10-28 使用者選 (b)：補一句「指標衡量什麼」的事實描述，不告訴 LLM 該怎麼做。
             # 試跑發現沒有這句時，LLM 把「觸發」字面解讀成「風險升高、不要加碼」。
             "（這個指標衡量「市值最大的 20% 公司佔全市場市值的比重」相對評估期起點的變化；"
             "比重上升代表這段期間大型股的漲幅相對高於其他股票。）",
             f"大盤上一季報酬：{facts['bench_last_q']:+.1%}。", "",
             "規則選單（報酬皆為未扣成本的季報酬；「歷史期」為評估期之前的 36 季）：",
             "| 代號 | 規則 | 上一季 | 近四季平均 | 歷史期平均 | 歷史期波動 |", "|---|---|---|---|---|---|"]
    for r in facts["rows"]:
        lines.append(f"| {r['code']} | {r['desc']} | {r['last_q']:+.1%} | {r['avg_4q']:+.1%} | "
                     f"{r['hist_mean']:+.1%} | {r['hist_std']:.1%} |")
    lines += ["", "請選一個代號套用到這一季，並用一兩句話說明理由（只能引用上面的數字）。"]
    return "\n".join(lines)


def schema(codes: list[str]) -> dict:
    return {"name": "rule_choice", "strict": True,
            "schema": {"type": "object", "additionalProperties": False, "required": ["choice", "reason"],
                       "properties": {"choice": {"type": "string", "enum": codes},
                                      "reason": {"type": "string", "description": "一兩句話，只能引用提供的數字"}}}}


def decide(plan, style, facts, *, model, api_key, dry_run=False) -> dict:
    from app.memo import _call_llm, scan_for_leakage
    prompt = build_prompt(plan, facts)
    codes = [r["code"] for r in facts["rows"]]
    system = SYSTEM_BASE + STYLES[style]
    if dry_run:
        return {"prompt": prompt, "system": system, "choice": "R0", "reason": "(dry-run)", "usage": {}}
    out, usage = _call_llm(prompt, model, api_key, purpose=f"step2_{plan}_{style}",
                           est_tokens=8000, system_prompt=system, schema=schema(codes))
    leak = scan_for_leakage({"reason": out["reason"]}, prompt)
    return {"prompt": prompt, "system": system, "choice": out["choice"], "reason": out["reason"],
            "usage": usage, "leakage": leak}


# ───────────────────────── 版本2（2026-10-28）：三種訊號＋兩項參考資訊 ─────────────────────────
# 只描述「指標衡量什麼」的事實，不告訴 LLM 該怎麼做（同警示說明 (b) 的原則）。

def signals_block(r) -> list[str]:
    """r：`step2_signals.compute()` 的一列（該季季初）。"""
    st = lambda x: STATE_LABEL.get(str(x), str(x))
    # 2026-10-28 code review：覆蓋率算不出來（NaN）時不要印出「nan%」
    hot_head = (f"2. 熱門股覆蓋：{st(r.hot_state)}（研究部名單覆蓋了全市場近期漲幅前 10% 股票中的 {r.hot_coverage:.0%}；"
                if r.hot_coverage == r.hot_coverage else f"2. 熱門股覆蓋：{st(r.hot_state)}（本季覆蓋率無法計算，狀態沿用上一季；")
    lines = ["市場監控訊號（季初）：",
             f"1. 集中度警示：{st(r.m1d_state)}（集中度累計偏離 {r.m1d_dev_pp:+.2f} 個百分點）。"
             "這個指標衡量市值最大的 20% 公司佔全市場市值的比重相對評估期起點的變化；比重上升代表大型股的漲幅相對高於其他股票。",
             hot_head +
             f"低於 {r.hot_p25:.0%} 為「觀察」、低於 {r.hot_p10:.0%} 為「觸發」）。覆蓋率低代表市場上漲勢最強的一群股票，名單持有得很少。",
             f"3. 突破廣度：{r.breakout_state}（名單持股中有 {r.breakout_share:.0%} 的股票收盤價創近 20 個交易日新高；"
             f"歷史期的 75／90 百分位為 {r.breakout_p75:.0%}／{r.breakout_p90:.0%}）；大盤{'有' if str(r.index_at_20d_high) == 'True' else '沒有'}創近 20 個交易日新高。",
             f"4. 回撤（參考）：研究部名單（control0）淨值距歷史高點 {r.drawdown:+.1%}，在歷史期季末回撤中屬第 {r.drawdown_pctile:.0%} 百分位（越低代表回撤越深）。"]
    if str(r.m2_state) != "nan" and r.m2_ratio == r.m2_ratio:
        lines.append(f"5. 策略群集中度（參考）：{st(r.m2_state)}（代表策略中最大一群的佔比是理想均分的 {r.m2_ratio:.1f} 倍）。")
    else:
        lines.append("5. 策略群集中度（參考）：評估期第一季尚無資料。")
    return lines


def build_facts_v2(market, pool, plan, k, g, sig, bench_q) -> dict:
    """版本2：選單用 MENUS2，訊號用 step2_signals。報酬欄位定義同版本1。"""
    arms = RS.MENUS2[plan](market, pool)
    allq = _series(g, arms)
    n_hist = int((g.index.get_level_values("window") < 4).sum())
    upto = allq.iloc[: n_hist + k - 1]
    hist = allq.iloc[:n_hist]
    as_of = g[g.index.get_level_values("window") == 4].index[k - 1][2]
    rows = [{"code": f"R{i}", "combo": a, "desc": describe(a), "last_q": float(upto[a].iloc[-1]),
             "avg_4q": float(upto[a].iloc[-4:].mean()), "hist_mean": float(hist[a].mean()),
             "hist_std": float(hist[a].std(ddof=1))} for i, a in enumerate(arms)]
    return {"k": k, "as_of": as_of, "sig": sig.loc[as_of], "bench_last_q": bench_q[k - 1], "rows": rows}


def build_prompt_v2(plan: str, facts: dict) -> str:
    lines = [f"現在是評估期的第 {facts['k']} 季季初（共 8 季）。", ""] + signals_block(facts["sig"]) + [
             f"大盤上一季報酬：{facts['bench_last_q']:+.1%}。", "",
             "規則選單（報酬皆為未扣成本的季報酬；「歷史期」為評估期之前的 36 季）：",
             "| 代號 | 規則 | 上一季 | 近四季平均 | 歷史期平均 | 歷史期波動 |", "|---|---|---|---|---|---|"]
    for r in facts["rows"]:
        lines.append(f"| {r['code']} | {r['desc']} | {r['last_q']:+.1%} | {r['avg_4q']:+.1%} | "
                     f"{r['hist_mean']:+.1%} | {r['hist_std']:.1%} |")
    lines += ["", "請選一個代號套用到這一季，並用一兩句話說明理由（只能引用上面的數字）。"]
    return "\n".join(lines)


def decide_v2(plan, style, facts, *, model, api_key, dry_run=False) -> dict:
    from app.memo import _call_llm, scan_for_leakage
    prompt = build_prompt_v2(plan, facts)
    codes = [r["code"] for r in facts["rows"]]
    system = SYSTEM_BASE + STYLES_V2[style]
    if dry_run:
        return {"prompt": prompt, "system": system, "choice": "R0", "reason": "(dry-run)", "usage": {}, "leakage": []}
    out, usage = _call_llm(prompt, model, api_key, purpose=f"step2v2_{plan}_{style}",
                           est_tokens=10000, system_prompt=system, schema=schema(codes))
    return {"prompt": prompt, "system": system, "choice": out["choice"], "reason": out["reason"],
            "usage": usage, "leakage": scan_for_leakage({"reason": out["reason"]}, prompt)}
