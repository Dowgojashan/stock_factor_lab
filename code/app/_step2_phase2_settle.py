# -*- coding: utf-8 -*-
"""步驟2 階段2 結算：用實際換手成本（`rule_selector.WeightBook`）結算 LLM 各方案與事先寫死的混法。

混法（實戰收尾追蹤.md「混合方式定案」，正式跑前寫死）：
  ① 平均混合：同風格同重複下，A／B／C／D 四個方案的權重矩陣平均
  ② 分工混合：A 選的排名依據＋取幾名＋分法，搭配 C 選的資金比例（C 選 control0 或 A 選 control0 → control0）
  ③ 風格混合：同方案同重複下，追強型與逆勢型的權重各半
  ④ 條件式切換：季初警示為「觸發」時採追強型、否則採逆勢型；另測反方向當對照
  ⑤（版本2，2026-10-29 看過版本2結果後才補，**不是事先寫死的**，報告須註明）條件式切換・三訊號：
     集中度觸發、熱門股覆蓋觸發、突破廣度「高」任一成立即視為觸發；同樣附反向對照
權重矩陣的線性性質：同一季買進持有時，權重平均＝報酬平均；換手成本仍依實際權重計算。
輸出：`step2/phase2_settled_{market}_{pool}.csv`（每條鏈）、`phase2_summary_{market}_{pool}.csv`（彙總）
2026-10-29 加：每條鏈的日頻 Sharpe／MDD（扣成本，rf=0、年化 252，同步驟1）；
  `{pre}_daily_*.parquet`：每條鏈＋不用 LLM 的方法（階段0 的選擇）＋大盤的逐日淨值（扣成本；大盤為報酬指數）
  `{pre}_weights_*.parquet`：每條鏈每季的權重矩陣（時間×股票，長表）
  每條鏈都驗證「逐日淨值最後一天＝季複利累積」
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import pandas as pd  # noqa: E402

from app import elite_rules as ER  # noqa: E402
from app import rule_selector as RS  # noqa: E402
from app import stock_scorer_llm as SD  # noqa: E402

OUT = RS.ROOT / "_analysis_outputs_applayer" / "step2"


def avg_weights(ws: list[dict]) -> dict:
    out = defaultdict(float)
    for w in ws:
        for s, v in w.items():
            out[s] += v / len(ws)
    return dict(out)


def daily_metrics(nav: pd.Series) -> tuple[float, float]:
    """(年化 Sharpe, 日頻 MDD)；rf=0、252 日，高點含起點 1.0（同步驟1 `_elite_grid_analyze`）。"""
    r = nav.pct_change().dropna()
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std(ddof=1) > 0 else float("nan")
    mdd = float((nav / nav.cummax().clip(lower=1.0) - 1.0).min())
    return sharpe, mdd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True); ap.add_argument("--pool", required=True)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--version", type=int, default=1, choices=[1, 2])
    a = ap.parse_args(argv)
    from fcv_core import MarketData
    close = MarketData(a.market).data.get("price:close")
    v2 = a.version == 2
    pre = "phase2v2" if v2 else "phase2"
    wb = RS.WeightBook(a.market, a.pool, close, with_sig=v2)
    states = RS.m1d_states(a.market, a.pool)
    trig3 = None
    if v2:
        from app import step2_signals
        sig = step2_signals.compute(a.market, a.pool)
        sig.index = sig.index.astype(str)
        trig3 = {str(d): bool(r.m1d_state == "TRIGGERED" or r.hot_state == "TRIGGERED" or r.breakout_state == "高")
                 for d, r in sig.iterrows()}
    recs = [json.loads(l) for l in open(OUT / f"{pre}_{a.market}_{a.pool}_{a.model}.jsonl", encoding="utf-8")]
    W = {}   # (plan, style, rep) -> [8 個權重 dict]
    picks = {}
    for (plan, style, rep), grp in pd.DataFrame(recs).groupby(["plan", "style", "rep"]):
        grp = grp.sort_values("k")
        if list(grp.k) != list(range(1, 9)):
            print(f"⚠️ {plan}/{style}/rep{rep} 不完整（{list(grp.k)}），略過"); continue
        ws = []
        for k0, r in enumerate(grp.itertuples()):
            if plan in ("A", "B", "C"):
                ws.append(wb.target(k0, r.combo))
            else:
                elite = SD.elite_from_ratings(r.ratings, r.codes)
                ws.append(ER.overlay(wb.w0[wb.as_ofs[k0]], elite, SD.SLEEVE))
        W[(plan, style, rep)] = ws
        if plan in ("A", "C"):
            picks[(plan, style, rep)] = list(grp.combo)

    rows, navs, wrows = [], {}, []

    def add(kind, plan, style, rep, ws, keys=None):
        key = ("w", kind, plan, style, rep)
        keys = keys or [(key, k) for k in range(8)]
        r = wb.settle(keys, ws)
        d = wb.daily(keys, ws)
        assert abs(d.net.iloc[-1] - 1 - r["cum_net"]) < 1e-9 and abs(d.gross.iloc[-1] - 1 - r["cum_gross"]) < 1e-9, \
            (kind, plan, style, rep, d.net.iloc[-1] - 1, r["cum_net"])
        sh, mdd = daily_metrics(d.net)
        rows.append({"kind": kind, "plan": plan, "style": style, "rep": rep,
                     "cum_gross": r["cum_gross"], "cum_net": r["cum_net"], "sharpe_net": sh, "mdd_net": mdd})
        cid = f"{kind}|{plan}|{style}|{rep}"
        navs[cid] = d.net
        for k in range(8):
            w = ws[k] if ws is not None else wb.target(k, keys[k])
            wrows.extend((cid, k + 1, wb.as_ofs[k], s, v) for s, v in w.items() if v > 0)

    for (plan, style, rep), ws in W.items():
        add("方案", plan, style, rep, ws)
    styles = sorted({s for _, s, _ in W}); reps = sorted({r for _, _, r in W})
    for style in styles:
        for rep in reps:
            if all((p, style, rep) in W for p in "ABCD"):
                add("平均混合", "ABCD", style, rep, [avg_weights([W[(p, style, rep)][k] for p in "ABCD"]) for k in range(8)])
            if ("A", style, rep) in picks and ("C", style, rep) in picks:
                ws = []
                for k, (pa, pc) in enumerate(zip(picks[("A", style, rep)], picks[("C", style, rep)])):
                    if pa == "REF_control0" or pc == "REF_control0":
                        ws.append(wb.target(k, "REF_control0"))
                    else:
                        sc, n, sch, _ = pa.split("|"); sl = pc.split("|")[3]
                        ws.append(wb.target(k, f"{sc}|{n}|{sch}|{sl}"))
                add("分工混合", "A×C", style, rep, ws)
    for plan in "ABCD":
        for rep in reps:
            m, c = W.get((plan, "momentum", rep)), W.get((plan, "contrarian", rep))
            if m and c:
                add("風格混合", plan, "追強＋逆勢", rep, [avg_weights([m[k], c[k]]) for k in range(8)])
                trig = [states[wb.as_ofs[k]] == "TRIGGERED" for k in range(8)]
                add("條件式切換", plan, "觸發→追強", rep, [m[k] if trig[k] else c[k] for k in range(8)])
                add("條件式切換（反向對照）", plan, "觸發→逆勢", rep, [c[k] if trig[k] else m[k] for k in range(8)])
                if trig3 is not None:
                    t3 = [trig3[str(wb.as_ofs[k])] for k in range(8)]
                    add("條件式切換・三訊號（事後補）", plan, "觸發→追強", rep, [m[k] if t3[k] else c[k] for k in range(8)])
                    add("條件式切換・三訊號（事後補，反向對照）", plan, "觸發→逆勢", rep,
                        [c[k] if t3[k] else m[k] for k in range(8)])
    # 不用 LLM 的方法：照階段0 的逐季選擇結算（同一套權重與成本；隨機挑只有分布，不畫逐日）
    p0 = pd.read_csv(OUT / ("phase0v2_picks.csv" if v2 else "phase0_picks.csv"))
    p0 = p0[(p0.market == a.market) & (p0.pool == a.pool)]
    for r0 in p0.itertuples():
        add("不用LLM", r0.plan, r0.policy, 0, None, keys=[getattr(r0, f"q{k}") for k in range(1, 9)])
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{pre}_settled_{a.market}_{a.pool}_{a.model}.csv", index=False, encoding="utf-8-sig")
    from app._elite_grid_offline import load_benchmark
    nav = pd.DataFrame(navs)
    b = load_benchmark(a.market).reindex(nav.index).ffill()
    nav["大盤|-|-|0"] = b / b.iloc[0]
    nav.to_parquet(OUT / f"{pre}_daily_{a.market}_{a.pool}_{a.model}.parquet")
    wt = pd.DataFrame(wrows, columns=["chain", "k", "as_of", "symbol", "weight"])
    wt["chain"] = wt.chain.astype("category"); wt["weight"] = wt.weight.astype("float32")
    wt.to_parquet(OUT / f"{pre}_weights_{a.market}_{a.pool}_{a.model}.parquet", compression="zstd")
    print(f"逐日淨值 {nav.shape}、權重矩陣 {len(wt):,} 列；全部鏈的逐日終值＝季複利（已驗證）")
    df = df[df.kind != "不用LLM"]
    summ = (df.groupby(["kind", "plan", "style"])
              .agg(n=("rep", "count"), gross_mean=("cum_gross", "mean"), gross_min=("cum_gross", "min"),
                   gross_max=("cum_gross", "max"), net_mean=("cum_net", "mean"), net_min=("cum_net", "min"),
                   net_max=("cum_net", "max"), sharpe_mean=("sharpe_net", "mean"),
                   mdd_mean=("mdd_net", "mean")).reset_index())
    c0 = wb.settle(["REF_control0"] * 8)
    summ.to_csv(OUT / f"{pre}_summary_{a.market}_{a.pool}_{a.model}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    show = summ.copy()
    show["sharpe_mean"] = show.sharpe_mean.map(lambda v: f"{v:.2f}")
    for col in [c for c in show.columns if c.startswith(("gross", "net", "mdd"))]:
        show[col] = show[col].map(lambda v: f"{v:+.1%}")
    print(f"=== {a.market}/{a.pool}（{a.model}）control0：未扣 {c0['cum_gross']:+.1%}／扣成本 {c0['cum_net']:+.1%} ===")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
