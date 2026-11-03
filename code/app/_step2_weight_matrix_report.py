# -*- coding: utf-8 -*-
"""步驟2 權重矩陣整理（2026-11-03 使用者要求）：讀 `_step2_phase2_settle.py` 存的長表 `*_weights_*.parquet`。

(a) 寬表 CSV：`step2/weight_matrices/v{版本}_{市場}_{候選池}/{挑法}.csv`
    列＝曾被持有的股票（代號、名稱），欄＝第1～8季權重（%）；資料夾內 `_index.csv` 列出每個檔是哪種挑法
(b) 矩陣特徵：每條鏈算 8 季平均的
      最大單一權重、前10大持股合計、主動比重（½×Σ|w−control0|）、
      目標換手（½×Σ|w_k − w_{k−1}|，季與季之間的目標權重變化，不含季內價格漂移；第1季不計）、
      最常被加碼的股票（AI 權重減 control0 最多者）與被加碼的季數（多 > 1 個百分點）
    → `step2/weight_matrix_features_v{版本}.csv`（逐鏈）、`weight_matrix_summary.csv`／`.md`（5 次平均）
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app import rule_selector as RS  # noqa: E402

OUT = RS.ROOT / "_analysis_outputs_applayer" / "step2"
MODEL = "gpt-5-mini"
GROUPS = [("TW", "exclude_v1"), ("TW", "baseline"), ("US", "exclude_v1"), ("US", "baseline")]
GLAB = {("TW", "exclude_v1"): "台股・排除V1", ("TW", "baseline"): "台股・含V1",
        ("US", "exclude_v1"): "美股・排除V1", ("US", "baseline"): "美股・含V1"}
STYLE = {"balanced": "平衡型", "momentum": "追強型", "contrarian": "逆勢型"}
C0 = "不用LLM|A|固定：REF_control0|0"


def parse(chain: str) -> tuple[str, str, str, str]:
    """鏈名稱＝類型|方案|風格（或不用AI的規則名，本身可能含 |）|重複。"""
    parts = chain.split("|")
    return parts[0], parts[1], "|".join(parts[2:-1]), parts[-1]


def label(chain: str) -> str:
    kind, plan, style, rep = parse(chain)
    if kind == "不用LLM":
        return f"不用AI：{style.replace('固定：REF_control0', 'control0')}（{plan} 選單）"
    return f"{kind}・{plan}・{STYLE.get(style, style)}・第{rep}次"


def safe(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", s)


def names_of(market: str) -> pd.Series:
    from resolve_strategy_holdings import _load_company_names
    return _load_company_names(market)


def features(mat: pd.DataFrame, c0: pd.DataFrame) -> dict:
    """mat／c0：index=季(1..8)，columns=股票，值=權重（0~1）。"""
    c0 = c0.reindex(index=mat.index, columns=mat.columns, fill_value=0)
    mx = mat.max(axis=1)
    top10 = mat.apply(lambda r: r.nlargest(10).sum(), axis=1)
    active = 0.5 * (mat - c0).abs().sum(axis=1)
    turn = 0.5 * mat.diff().abs().sum(axis=1).iloc[1:]
    over = mat - c0
    boosted = (over > 0.01).sum()
    top_sym = over.max().idxmax() if over.max().max() > 0.01 else None
    return {"最大單一權重": mx.mean(), "前10大合計": top10.mean(), "主動比重": active.mean(),
            "目標換手": turn.mean(), "最常加碼股票": top_sym,
            "被加碼季數": int(boosted.get(top_sym, 0)) if top_sym else 0,
            "有加碼的季數": int((active > 0.005).sum())}


def run_version(v: int) -> pd.DataFrame:
    pre = "phase2v2" if v == 2 else "phase2"
    rows = []
    for m, p in GROUPS:
        wt = pd.read_parquet(OUT / f"{pre}_weights_{m}_{p}_{MODEL}.parquet")
        wt["chain"] = wt.chain.astype(str)
        wide = wt.pivot_table(index=["chain", "k"], columns="symbol", values="weight", fill_value=0.0)
        names = names_of(m)
        c0 = wide.loc[C0]
        d = OUT / "weight_matrices" / f"v{v}_{m}_{p}"
        d.mkdir(parents=True, exist_ok=True)
        idx = []
        for chain in wide.index.get_level_values(0).unique():
            mat = wide.loc[chain]
            mat = mat.loc[:, mat.sum() > 0]
            # (a) 寬表：列＝股票、欄＝季
            t = (mat.T * 100).round(4)
            t.columns = [f"第{k}季(%)" for k in t.columns]
            t.insert(0, "名稱", [names.get(s, "") for s in t.index])
            t = t.loc[t.drop(columns="名稱").max(axis=1).sort_values(ascending=False).index]
            t.index.name = "代號"
            fn = safe(label(chain)) + ".csv"
            t.to_csv(d / fn, encoding="utf-8-sig")
            idx.append({"檔名": fn, "挑法": label(chain), "鏈": chain, "持有過的股票數": len(t)})
            # (b) 特徵
            f = features(mat, c0)
            kind, plan, style, rep = parse(chain)
            rows.append({"版本": v, "組別": GLAB[(m, p)], "類型": kind, "方案": plan, "風格": STYLE.get(style, style),
                         "重複": int(rep), **f, "最常加碼股票名稱": names.get(f["最常加碼股票"], "") if f["最常加碼股票"] else ""})
        pd.DataFrame(idx).to_csv(d / "_index.csv", index=False, encoding="utf-8-sig")
        print(f"v{v} {GLAB[(m, p)]}：{len(idx)} 條鏈 → {d}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"weight_matrix_features_v{v}.csv", index=False, encoding="utf-8-sig")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["版本", "組別", "類型", "方案", "風格"]
    num = ["最大單一權重", "前10大合計", "主動比重", "目標換手", "有加碼的季數"]
    g = df.groupby(keys, sort=False)
    s = g[num].mean().reset_index()
    mode = g.apply(lambda x: x["最常加碼股票"].dropna().mode().iloc[0] if x["最常加碼股票"].notna().any() else "").reset_index(name="最常加碼股票（5次中最常見）")
    nm = df.dropna(subset=["最常加碼股票"]).drop_duplicates("最常加碼股票").set_index("最常加碼股票")["最常加碼股票名稱"]
    s = s.merge(mode, on=keys)
    s["最常加碼股票（5次中最常見）"] = [f"{c} {nm.get(c, '')}".strip() if c else "（沒有加碼）"
                                  for c in s["最常加碼股票（5次中最常見）"]]
    return s


def main():
    all_df = pd.concat([run_version(1), run_version(2)], ignore_index=True)
    s = summarize(all_df)
    s.to_csv(OUT / "weight_matrix_summary.csv", index=False, encoding="utf-8-sig")
    show = s.copy()
    for c in ["最大單一權重", "前10大合計", "主動比重", "目標換手"]:
        show[c] = show[c].map(lambda x: f"{x:.1%}")
    show["有加碼的季數"] = show["有加碼的季數"].map(lambda x: f"{x:.1f}")
    cols = list(show.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(x) for x in r) + " |" for r in show.itertuples(index=False)]
    (OUT / "weight_matrix_summary.md").write_text("# 步驟2 權重矩陣特徵（5 次平均；8 季平均）\n\n" + "\n".join(md) + "\n",
                                                  encoding="utf-8")
    print(f"彙總 {len(s)} 列 → {OUT / 'weight_matrix_summary.csv'}")


if __name__ == "__main__":
    main()
