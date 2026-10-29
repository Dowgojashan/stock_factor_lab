# -*- coding: utf-8 -*-
"""步驟2 方案D：LLM 逐檔質性評分（實戰收尾追蹤.md §四步驟2「模型與方案D定案」，2026-10-28）。

每季季初：
  1. 候選股＝票數前10名（邊界同分全部納入）——直接取步驟1「S1_votes|N10|equal」的優等生名單
     （`elite_weights_*.parquet`），沿用步驟1的壞資料防線與（美股）流動性門檻
  2. 去識別化：只給「在候選股之內的相對排名」（1＝數值最高）——近1/3/6/12個月報酬、PB、ROE、動能、市值；
     不給市值排名／佔比／全市場家數／精確報酬（記憶測試證實這些會讓 LLM 認出台積電、NVIDIA）；
     每季用固定種子打亂順序、以 C1..Cn 代號呈現
  3. LLM 對每檔回「加碼／持平／減碼」→ 2／1／0 份，在優等生資金 30% 內依份數分配，其餘 70% 照 control0；
     全部減碼＝該季 control0
報酬：季內買進持有，價格比值與步驟1 同一支函式（`_elite_grid_offline.price_ratios`），未扣成本。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app import rule_selector as RS
from app import rule_selector_llm as RL

SLEEVE = 0.30
SHARE = {"加碼": 2, "持平": 1, "減碼": 0}
FEATS = [("ret1", "近1個月報酬"), ("ret3", "近3個月報酬"), ("ret6", "近6個月報酬"), ("ret12", "近12個月報酬"),
         ("pb", "PB"), ("roe", "ROE"), ("mom", "動能"), ("mcap", "市值")]


def candidates(market, pool) -> dict[str, list[str]]:
    el = pd.read_parquet(RS.GRID / f"elite_weights_{RS.TAG[(market, pool)]}.parquet")
    el = el[(el.window == 4) & (el.score == "S1_votes") & (el.n == 10) & (el.scheme == "equal")]
    return {a: sorted(sub.symbol) for a, sub in el.groupby("as_of")}


def features(md, close, mcap_wide, syms, as_of) -> pd.DataFrame:
    from app import monitor
    ts = pd.Timestamp(as_of); idx = close.index; d1 = idx[idx <= ts].max()
    f = pd.DataFrame(index=syms)
    from app import elite_rules as ER
    for m in (1, 3, 6, 12):   # 與步驟1 同一支函式（含 p0>0、p1>0 防線；缺值者排名顯示「—」）
        f[f"ret{m}"] = ER.trailing_return(close, syms, ts, m).reindex(syms)
    for k in ("pb", "roe", "mom"):
        fr = md.get_field(f"report:{k}")
        f[k] = fr.loc[d1].reindex(syms) if fr is not None else np.nan
    f["mcap"] = monitor.asof_row(mcap_wide, as_of).reindex(syms)
    return f


def anonymized_table(f: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    """回傳 (相對排名表, {代號: 股票})。排名 1＝候選中數值最高；缺值顯示「—」。"""
    ranks = f.rank(ascending=False, method="min")
    order = np.random.default_rng(seed).permutation(len(f))
    syms = list(f.index[order])
    codes = {f"C{i+1}": s for i, s in enumerate(syms)}
    tab = ranks.loc[syms].copy()
    tab.index = list(codes)
    return tab, codes


def build_prompt(k, state, dev, bench_q, tab: pd.DataFrame) -> str:
    n = len(tab)
    lines = [f"現在是評估期的第 {k} 季季初（共 8 季）。",
             f"市場集中度警示：{RL.STATE_LABEL[state]}（相對評估期起點的集中度累計偏離 {dev:+.2f} 個百分點；"
             "偏離超過觀察門檻為「觀察」、超過觸發門檻為「觸發」）。",
             "（這個指標衡量「市值最大的 20% 公司佔全市場市值的比重」相對評估期起點的變化；"
             "比重上升代表這段期間大型股的漲幅相對高於其他股票。）",
             f"大盤上一季報酬：{bench_q:+.1%}。", "",
             f"研究部本季共識度最高的 {n} 檔候選股（已去識別化，順序隨機）。表中數字是各項指標在這 {n} 檔之內的排名，"
             f"1 代表數值最高、{n} 代表最低：", "",
             "| 代號 | " + " | ".join(lbl for _, lbl in FEATS) + " |", "|" + "---|" * (len(FEATS) + 1)]
    for code, r in tab.iterrows():
        lines.append(f"| {code} | " + " | ".join("—" if pd.isna(r[c]) else str(int(r[c])) for c, _ in FEATS) + " |")
    lines += ["", "請對每一檔給出「加碼」「持平」或「減碼」：加碼的股票會分到較多的優等生資金，減碼的不分。"
              "並用一兩句話說明整體理由（只能引用上面的數字）。"]
    return "\n".join(lines)


def schema(codes: list[str]) -> dict:
    return {"name": "stock_ratings", "strict": True,
            "schema": {"type": "object", "additionalProperties": False, "required": ["ratings", "reason"],
                       "properties": {
                           "ratings": {"type": "array", "minItems": len(codes), "maxItems": len(codes), "items": {
                               "type": "object", "additionalProperties": False, "required": ["code", "action"],
                               "properties": {"code": {"type": "string", "enum": codes},
                                              "action": {"type": "string", "enum": list(SHARE)}}}},
                           "reason": {"type": "string"}}}}


SYSTEM = (RL.SYSTEM_BASE.replace("從「規則選單」中選一個規則，決定下一季怎麼對名單裡的優等生加碼",
                                 "對研究部給的候選股逐檔評分，決定下一季優等生資金怎麼分配")
          .replace("只能回傳選單中的代號。", "只能使用表中的代號。"))


def check_ratings(ratings: list[dict], codes: dict) -> list[str]:
    """回傳問題清單（漏評／重複），空清單＝每檔恰好評一次。"""
    seen = [r["code"] for r in ratings]
    probs = [f"漏評 {c}" for c in codes if c not in seen]
    probs += [f"重複 {c}" for c in set(seen) if seen.count(c) > 1]
    return probs


def elite_from_ratings(ratings: list[dict], codes: dict) -> dict[str, float]:
    act = {r["code"]: r["action"] for r in ratings}
    shares = {codes[c]: SHARE.get(act.get(c, "持平"), 1) for c in codes}    # 漏評的視為持平
    tot = sum(shares.values())
    return {s: v / tot for s, v in shares.items() if v > 0} if tot > 0 else {}


def quarter_return(close, w0: dict, elite: dict, as_of, end) -> float:
    from app._elite_grid_offline import price_ratios
    from app import elite_rules as ER
    final = ER.overlay(w0, elite, SLEEVE)
    syms = sorted(final)
    R = price_ratios(close, syms, as_of, end)
    w = np.array([final[s] for s in syms])
    return float((1.0 - w.sum()) + R.iloc[-1].to_numpy() @ w - 1.0)


def build_prompt_v2(k, sig_row, bench_q, tab: pd.DataFrame) -> str:
    """版本2：同 build_prompt，但市場資訊換成五個訊號（`rule_selector_llm.signals_block`）。"""
    n = len(tab)
    lines = [f"現在是評估期的第 {k} 季季初（共 8 季）。", ""] + RL.signals_block(sig_row) + [
             f"大盤上一季報酬：{bench_q:+.1%}。", "",
             f"研究部本季共識度最高的 {n} 檔候選股（已去識別化，順序隨機）。表中數字是各項指標在這 {n} 檔之內的排名，"
             f"1 代表數值最高、{n} 代表最低：", "",
             "| 代號 | " + " | ".join(lbl for _, lbl in FEATS) + " |", "|" + "---|" * (len(FEATS) + 1)]
    for code, r in tab.iterrows():
        lines.append(f"| {code} | " + " | ".join("—" if pd.isna(r[c]) else str(int(r[c])) for c, _ in FEATS) + " |")
    lines += ["", "請對每一檔給出「加碼」「持平」或「減碼」：加碼的股票會分到較多的優等生資金，減碼的不分。"
              "並用一兩句話說明整體理由（只能引用上面的數字）。"]
    return "\n".join(lines)
