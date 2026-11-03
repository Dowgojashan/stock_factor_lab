# -*- coding: utf-8 -*-
"""步驟2 AI 理由分析（2026-11-03 使用者要求；不呼叫 LLM、不花額度）。

讀 `step2/phase2{,v2}_{市場}_{池}_gpt-5-mini.jsonl` 的 choice／combo／reason（方案 D 為 ratings／reason），
用與正式執行相同的函式重建當時的 prompt（方案 A／B／C 完整重建；方案 D 只重建訊號與大盤段落，
候選股排名表需要價格資料，不重建），做四項分析：
  1. 一致性：A／B／C——理由是否提到所選規則的類型（市值→「市值／大型股」、熱門股→「熱門股」…；
     不調整→「不調整／維持／control0」）；D——理由裡「加碼／減碼／持平」子句提到的代號，與實際評分是否相同
  2. 數字正確性：理由中帶 % 或「個百分點」的數字，是否出現在當時的 prompt 裡
  3. 風格差異：各風格理由引用各項資訊的比例；選擇傾向（不調整比例、所選規則的「上一季報酬」在選單中的排名、
     「歷史期平均」排名）
  4. 代表性例子：事先固定為「版本2、第1次重複、台股・排除V1／美股・排除V1、方案A 與 D、第2季（未觸發）與第5季」
輸出：`step2/reason_analysis/`（逐筆 csv＋摘要 md）
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import pandas as pd  # noqa: E402

from app import rule_selector as RS  # noqa: E402
from app import rule_selector_llm as RL  # noqa: E402
from app import simulate  # noqa: E402

OUT = RS.ROOT / "_analysis_outputs_applayer" / "step2"
DST = OUT / "reason_analysis"
GROUPS = [("TW", "exclude_v1"), ("TW", "baseline"), ("US", "exclude_v1"), ("US", "baseline")]
GLAB = {("TW", "exclude_v1"): "台股・排除V1", ("TW", "baseline"): "台股・含V1",
        ("US", "exclude_v1"): "美股・排除V1", ("US", "baseline"): "美股・含V1"}
STYLE = {"balanced": "平衡型", "momentum": "追強型", "contrarian": "逆勢型"}
FAMILY_KW = {"REF": ("不調整", "control0", "維持"), "S1": ("票數",), "S2": ("因子",),
             "S3": ("追強",), "S4": ("押回彈", "回彈"), "S5": ("追強＋押回彈", "混合", "追強", "押回彈"),
             "S6": ("市值", "大型"), "S7": ("熱門股",), "S8": ("突破", "新高")}
FAMILY_NAME = {"REF": "不調整", "S1": "票數", "S2": "因子分數", "S3": "追強", "S4": "押回彈",
               "S5": "追強＋押回彈", "S6": "市值", "S7": "熱門股", "S8": "突破股"}
INFO_KW = {"上一季報酬": ("上一季", "上季"), "近四季平均": ("近四季",), "歷史期平均／波動": ("歷史期",),
           "集中度警示": ("集中度",), "熱門股覆蓋": ("熱門股覆蓋", "覆蓋"), "突破廣度": ("突破廣度",),
           "回撤": ("回撤",), "策略群集中度": ("策略群",), "大盤": ("大盤",)}
NUM = re.compile(r"([+\-−]?\d+(?:\.\d+)?)\s*(%|個百分點|pp)")


def nums(text: str) -> list[str]:
    return [n.replace("−", "-").lstrip("+-") for n, _ in NUM.findall(text)]


def family(combo: str) -> str:
    return "REF" if combo.startswith("REF_") else combo.split("_")[0]


def prompts_abc(v: int, m: str, p: str) -> dict:
    """{(plan, k): (prompt, facts)}，與 `_step2_phase2` 相同的函式重建。"""
    g, _ = RS.load_returns(m, p, with_sig=(v == 2))
    bq = RL.bench_last_quarter(m)
    out = {}
    if v == 2:
        from app import step2_signals
        sig = step2_signals.compute(m, p)
        for plan in "ABC":
            for k in range(1, 9):
                f = RL.build_facts_v2(m, p, plan, k, g, sig, bq); out[(plan, k)] = (RL.build_prompt_v2(plan, f), f)
    else:
        states = RS.m1d_states(m, p); RL.load_deviation(m, p)
        for plan in "ABC":
            for k in range(1, 9):
                f = RL.build_facts(m, p, plan, k, g, states, bq); out[(plan, k)] = (RL.build_prompt(plan, f), f)
    return out


def prompt_d_numbers(v: int, m: str, p: str) -> dict:
    """方案D prompt 中帶 % / 個百分點 的數字（訊號段＋大盤）；候選股表只有名次（整數），不含 %。"""
    bq = RL.bench_last_quarter(m)
    as_ofs = [simulate.REGISTRATION_DATE] + list(simulate.QUARTER_ENDS[:-1])
    out = {}
    if v == 2:
        from app import step2_signals
        sig = step2_signals.compute(m, p)
        for k in range(1, 9):
            out[k] = set(nums("\n".join(RL.signals_block(sig.loc[as_ofs[k - 1]])) + f"大盤上一季報酬：{bq[k - 1]:+.1%}"))
    else:
        dev = RL.load_deviation(m, p)
        for k in range(1, 9):
            a = as_ofs[k - 1]
            out[k] = set(nums(f"{dev[a]:+.2f} 個百分點 大盤上一季報酬：{bq[k - 1]:+.1%}"))
    return out


def d_consistency(reason: str, ratings: list[dict]) -> tuple[int, int]:
    """理由中「單一動作」子句提到的代號，跟實際評分相同的個數／檢查的個數。"""
    act = {r["code"]: r["action"] for r in ratings}
    ok = n = 0
    for clause in re.split(r"[；;。\n]", reason):
        acts = [a for a in ("加碼", "減碼", "持平") if a in clause]
        if len(acts) != 1:
            continue
        for code in set(re.findall(r"C\d+", clause)):
            if code in act:
                n += 1; ok += int(act[code] == acts[0])
    return ok, n


def main():
    DST.mkdir(parents=True, exist_ok=True)
    rows, drows, examples = [], [], []
    for v in (1, 2):
        pre = "phase2v2" if v == 2 else "phase2"
        for m, p in GROUPS:
            pr = prompts_abc(v, m, p)
            pdn = prompt_d_numbers(v, m, p)
            rec = [json.loads(l) for l in open(OUT / f"{pre}_{m}_{p}_gpt-5-mini.jsonl", encoding="utf-8")]
            for r in rec:
                reason = r["reason"]
                info = {k: int(any(w in reason for w in ws)) for k, ws in INFO_KW.items()}
                rn = nums(reason)
                base = {"版本": v, "組別": GLAB[(m, p)], "方案": r["plan"], "風格": STYLE[r["style"]],
                        "重複": r["rep"], "季": r["k"], "理由": reason, **{f"提到_{k}": x for k, x in info.items()}}
                if r["plan"] in "ABC":
                    prompt, facts = pr[(r["plan"], r["k"])]
                    pn = set(nums(prompt))
                    fam = family(r["combo"])
                    rows_m = pd.DataFrame(facts["rows"])
                    ch = rows_m[rows_m.combo == r["combo"]].iloc[0]
                    rk_last = float((rows_m.last_q > ch.last_q).sum() + 1) / len(rows_m)
                    rk_hist = float((rows_m.hist_mean > ch.hist_mean).sum() + 1) / len(rows_m)
                    rows.append({**base, "選擇": RL.describe(r["combo"]), "類型": FAMILY_NAME.get(fam, fam),
                                 "理由提到所選類型": int(any(w in reason for w in FAMILY_KW.get(fam, ()))),
                                 "數字個數": len(rn), "數字對得上": sum(x in pn for x in rn),
                                 "對不上的數字": "、".join(x for x in rn if x not in pn),
                                 "所選規則上一季報酬排名分位": rk_last, "所選規則歷史期平均排名分位": rk_hist,
                                 "選單大小": len(rows_m)})
                else:
                    ok, n = d_consistency(reason, r["ratings"])
                    acts = pd.Series([x["action"] for x in r["ratings"]]).value_counts()
                    drows.append({**base, "子句代號檢查數": n, "子句代號相符": ok,
                                  "數字個數": len(rn), "數字對得上": sum(x in pdn[r["k"]] for x in rn),
                                  "對不上的數字": "、".join(x for x in rn if x not in pdn[r["k"]]),
                                  "加碼檔數": int(acts.get("加碼", 0)), "減碼檔數": int(acts.get("減碼", 0)),
                                  "持平檔數": int(acts.get("持平", 0))})
                if v == 2 and r["rep"] == 1 and (m, p) in (("TW", "exclude_v1"), ("US", "exclude_v1")) \
                        and r["plan"] in ("A", "D") and r["k"] in (2, 5):
                    examples.append({"組別": GLAB[(m, p)], "方案": r["plan"], "風格": STYLE[r["style"]], "季": r["k"],
                                     "選擇": RL.describe(r["combo"]) if r["plan"] != "D" else
                                     f"加碼 {sum(x['action']=='加碼' for x in r['ratings'])} 檔、減碼 {sum(x['action']=='減碼' for x in r['ratings'])} 檔",
                                     "理由": reason})
            print(f"v{v} {GLAB[(m, p)]} ✓", flush=True)
    A = pd.DataFrame(rows); Dd = pd.DataFrame(drows); E = pd.DataFrame(examples)
    A.to_csv(DST / "reasons_abc.csv", index=False, encoding="utf-8-sig")
    Dd.to_csv(DST / "reasons_d.csv", index=False, encoding="utf-8-sig")
    E.to_csv(DST / "examples.csv", index=False, encoding="utf-8-sig")
    write_summary(A, Dd, E)


def pct(x):
    return f"{x:.0%}"


def write_summary(A: pd.DataFrame, D: pd.DataFrame, E: pd.DataFrame):
    L = ["# 步驟2 AI 理由分析（不呼叫 LLM；方案A／B／C 共 %d 筆、方案D %d 筆）" % (len(A), len(D)), ""]
    # 1 一致性
    L += ["## 1. 理由與選擇是否一致", ""]
    t = A.groupby("類型").agg(筆數=("理由", "size"), 一致=("理由提到所選類型", "mean")).sort_values("筆數", ascending=False)
    L += ["| AI 選的規則類型 | 筆數 | 理由有提到該類型 |", "|---|---:|---:|"]
    L += [f"| {i} | {r.筆數} | {pct(r.一致)} |" for i, r in t.iterrows()]
    L += ["", f"整體：{pct(A['理由提到所選類型'].mean())}（方案A／B／C）。"]
    dn, dok = D["子句代號檢查數"].sum(), D["子句代號相符"].sum()
    L += [f"方案D：理由中明確說「加碼／減碼／持平」的代號共 {dn} 個，與實際評分相同 {dok} 個（{pct(dok / dn) if dn else '—'}）。", ""]
    # 2 數字
    L += ["## 2. 理由中的數字是否出自 prompt", ""]
    for name, X in (("方案A／B／C", A), ("方案D（只比對訊號與大盤段落）", D)):
        n, ok = X["數字個數"].sum(), X["數字對得上"].sum()
        bad = X[X["數字個數"] > X["數字對得上"]]
        L.append(f"- {name}：理由中共 {n} 個數字，對得上 {ok} 個（{pct(ok / n) if n else '—'}）；"
                 f"含對不上數字的理由 {len(bad)} 筆（{pct(len(bad) / len(X))}）")
    bad = A[A["數字個數"] > A["數字對得上"]].head(6)
    if len(bad):
        L += ["", "對不上的例子（多半是 AI 自己算的差距或四捨五入不同）：", ""]
        L += [f"- 〔{r.組別}・v{r.版本}・{r.方案}・{r.風格}・第{r.季}季〕對不上：{r.對不上的數字}｜理由：{r.理由[:110]}" for r in bad.itertuples()]
    L.append("")
    # 3 風格
    L += ["## 3. 不同風格的差異", ""]
    cols = [c for c in A.columns if c.startswith("提到_")]
    g = A.groupby("風格")[cols].mean()
    L += ["**理由引用各項資訊的比例（方案A／B／C）**", "", "| 風格 | " + " | ".join(c[3:] for c in cols) + " |", "|---|" + "---:|" * len(cols)]
    L += [f"| {s} | " + " | ".join(pct(g.loc[s, c]) for c in cols) + " |" for s in ("平衡型", "追強型", "逆勢型")]
    b = A.assign(選不調整=(A.類型 == "不調整").astype(int)).groupby("風格").agg(
        選不調整=("選不調整", "mean"), 上一季排名=("所選規則上一季報酬排名分位", "mean"),
        歷史排名=("所選規則歷史期平均排名分位", "mean"))
    L += ["", "**選擇傾向（排名分位：越小代表在選單中越前面，0.5 約為中間）**", "",
          "| 風格 | 選「不調整」的比例 | 所選規則的上一季報酬排名分位 | 所選規則的歷史期平均排名分位 |", "|---|---:|---:|---:|"]
    L += [f"| {s} | {pct(b.loc[s, '選不調整'])} | {b.loc[s, '上一季排名']:.2f} | {b.loc[s, '歷史排名']:.2f} |" for s in ("平衡型", "追強型", "逆勢型")]
    v = A.groupby(["版本"])[["提到_集中度警示", "提到_熱門股覆蓋", "提到_突破廣度"]].mean()
    L += ["", "**版本1 vs 版本2：理由提到市場訊號的比例**", "", "| 版本 | 集中度警示 | 熱門股覆蓋 | 突破廣度 |", "|---|---:|---:|---:|"]
    L += [f"| 版本{i} | {pct(r.iloc[0])} | {pct(r.iloc[1])} | {pct(r.iloc[2])} |" for i, r in v.iterrows()]
    dd = D.groupby("風格")[["加碼檔數", "減碼檔數", "持平檔數"]].mean()
    L += ["", "**方案D：每季平均給幾檔加碼／減碼／持平**", "", "| 風格 | 加碼 | 減碼 | 持平 |", "|---|---:|---:|---:|"]
    L += [f"| {s} | {dd.loc[s, '加碼檔數']:.1f} | {dd.loc[s, '減碼檔數']:.1f} | {dd.loc[s, '持平檔數']:.1f} |" for s in ("平衡型", "追強型", "逆勢型")]
    # 4 例子
    L += ["", "## 4. 代表性例子（事先固定：版本2、第1次、方案A 與 D、第2季與第5季）", ""]
    for gname in ("台股・排除V1", "美股・排除V1"):
        L += [f"### {gname}", "", "| 方案 | 風格 | 季 | AI 的選擇 | 理由 |", "|---|---|---|---|---|"]
        for r in E[E.組別 == gname].sort_values(["方案", "季", "風格"]).itertuples():
            L.append(f"| {r.方案} | {r.風格} | 第{r.季}季 | {r.選擇} | {r.理由.replace('|', '／')} |")
        L.append("")
    (DST / "reason_analysis.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
