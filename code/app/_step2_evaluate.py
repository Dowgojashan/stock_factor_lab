# -*- coding: utf-8 -*-
"""步驟2 判定「AI 有用」（實戰收尾追蹤.md「判斷標準」與「操作定義」，看結果之前寫死）。

每一種 AI 挑法（方案×風格、混法）三條都成立才判「有效」：
  ① 5 次重複的扣成本平均 > 隨機挑（1,000 次）扣成本分布的 95 百分位
     （A/B/C 用各自選單；D 用隨機評分；風格混合、條件式切換用該方案的；平均混合、分工混合用選單 A 的）
  ② 扣成本平均 > 同一份選單下不用 LLM 的方法中最高者（ε-greedy、Thompson、上季最好續用、上季最差改用、
     警示觸發換市值）；D 與平均／分工混合沒有自己的選單，用選單 A 的（報告須註明）
  ③ 5 次重複中至少 4 次扣成本累積 > control0
另輸出 gpt-5 與 gpt-5-mini 同一份 prompt 下的選擇一致率（穩健性檢查）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from app import rule_selector as RS  # noqa: E402

OUT = RS.ROOT / "_analysis_outputs_applayer" / "step2"
GROUPS = [("TW", "exclude_v1"), ("TW", "baseline"), ("US", "exclude_v1"), ("US", "baseline")]
NONLLM = ["ε-greedy（ε=0.1）", "Thompson sampling", "上季最好續用（追強）", "上季最差改用（逆勢）", "警示觸發換市值"]
STYLE = {"balanced": "平衡型", "momentum": "追強型", "contrarian": "逆勢型"}


def main(model: str = "gpt-5-mini", version: int = 1):
    p0 = pd.read_csv(OUT / ("phase0v2_policies.csv" if version == 2 else "phase0_policies.csv"))
    nonllm_names = NONLLM + (["訊號規則（三訊號）"] if version == 2 else [])
    pre = "phase2v2" if version == 2 else "phase2"
    out = []
    for m, p in GROUPS:
        settled = pd.read_csv(OUT / f"{pre}_settled_{m}_{p}_{model}.csv")
        settled = settled[settled.kind != "不用LLM"]   # 2026-10-29：結算檔另含不用 LLM 方法的逐條結果，判定時排除
        z = p0[(p0.market == m) & (p0.pool == p)]
        c0 = float(z[z.policy == "固定：REF_control0"].cum_net.iloc[0])
        rnd = {pl: float(z[(z.plan == pl) & (z.policy == "隨機挑")].p95_net.iloc[0]) for pl in "ABCD"}
        nonllm = {pl: float(z[(z.plan == pl) & z.policy.isin(nonllm_names)].cum_net.max()) for pl in "ABC"}
        for (kind, plan, style), g in settled.groupby(["kind", "plan", "style"]):
            base = plan if plan in ("A", "B", "C", "D") else "A"
            r_bar = rnd[base]
            n_bar = nonllm[base if base in ("A", "B", "C") else "A"]
            mean = g.cum_net.mean()
            c1, c2, c3 = mean > r_bar, mean > n_bar, int((g.cum_net > c0).sum()) >= 4
            out.append({"組別": f"{m}・{p}", "類型": kind, "方案": plan, "風格": STYLE.get(style, style),
                        "重複次數": len(g), "扣成本平均": mean, "最低": g.cum_net.min(), "最高": g.cum_net.max(),
                        "control0": c0, "隨機挑95百分位": r_bar, "不用LLM最佳": n_bar,
                        "贏control0次數": int((g.cum_net > c0).sum()),
                        "日頻Sharpe平均（參考）": g.sharpe_net.mean() if "sharpe_net" in g else float("nan"),
                        "日頻MDD平均（參考）": g.mdd_net.mean() if "mdd_net" in g else float("nan"),
                        "①贏隨機": c1, "②贏不用LLM": c2, "③穩定": c3, "判定": "有效" if (c1 and c2 and c3) else "無法與隨機區分"})
    df = pd.DataFrame(out)
    df.to_csv(OUT / f"phase2_verdict_v{version}_{model}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    show = df.copy()
    show["日頻Sharpe平均（參考）"] = show["日頻Sharpe平均（參考）"].map(lambda v: f"{v:.2f}")
    for c in ["扣成本平均", "最低", "最高", "control0", "隨機挑95百分位", "不用LLM最佳", "日頻MDD平均（參考）"]:
        show[c] = show[c].map(lambda v: f"{v:+.1%}")
    for c in ["①贏隨機", "②贏不用LLM", "③穩定"]:
        show[c] = show[c].map(lambda v: "✓" if v else "✗")
    print(show.to_string(index=False))
    print("\n判定為「有效」的：", len(df[df.判定 == "有效"]), "/", len(df))

    # 穩健性：gpt-5 vs gpt-5-mini（台股・排除V1，方案A、C，平衡型）
    import json
    # 2026-10-29：原本固定讀版本1 的檔案，版本2 判定時會印出版本1 的一致率；改成依版本讀
    g5_path = OUT / f"{pre}_TW_exclude_v1_gpt-5.jsonl"
    if not g5_path.exists():
        print(f"（版本{version} 沒有 gpt-5 穩健性檢查檔，略過）"); return
    print(f"穩健性（版本{version}）：")
    g5 = [json.loads(l) for l in open(g5_path, encoding="utf-8")]
    mini = [json.loads(l) for l in open(OUT / f"{pre}_TW_exclude_v1_gpt-5-mini.jsonl", encoding="utf-8")]
    for plan in ("A", "C"):
        a = {r["k"]: r["combo"] for r in g5 if r["plan"] == plan}
        agree = [sum(r["combo"] == a[r["k"]] for r in mini if r["plan"] == plan and r["style"] == "balanced" and r["k"] == k)
                 for k in range(1, 9)]
        print(f"穩健性 方案{plan}：每季 gpt-5-mini 5 次中與 gpt-5 選擇相同的次數 {agree}（平均一致率 {sum(agree) / 40:.0%}）")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--version", type=int, default=1, choices=[1, 2])
    ap.add_argument("--model", default="gpt-5-mini")
    a = ap.parse_args(); main(a.model, a.version)
