# -*- coding: utf-8 -*-
"""LLM 規則選擇器 vs local greedy：2024–25 同期比較（2026-11-14，使用者同意；不呼叫 LLM，只讀步驟2 現成紀錄）。

設計（看結果前寫死）：
  - 期間：window4 共 8 季（2024Q1～2025Q4）；window1～3（2015～2023）只當回看與樣本內資料。
  - 選單：步驟2 的選單，各自分開比（`rule_selector.MENUS`／`MENUS2`）
      版本1：A（control0＋6 類各自最佳，7 個）、B（6 類×前1/3/5/10×資金10/20/30/50%＋control0，97 個）、C（control0＋市值前3名×資金10/30/50%，4 個）
      版本2：A2（9 個）、B2（129 個）、C2（10 個），多了熱門股 S7、突破股 S8
    方案 D（逐檔評分）與各種混法不是「從選單挑一個」，不列入。
  - 路徑（同一份選單內）：
      LLM：每個方案×風格（平衡／追強／逆勢）各 5 次重複，報 5 次平均，另列最低～最高
      local greedy：季初只用之前的季報酬，挑過去 L 季累積報酬（扣成本）最高者；主 L=4，L=1、2 為敏感度
      事後最佳（上帝視角）：每季選當季最好；另列「事後挑一個固定到底」
      固定樣本內最佳：選單中 2015–2023（window1～3）季報酬扣成本 Sharpe 最高者，固定到底
      control0、隨機（每季從選單均勻抽，1,000 次）
  - KPI：選單內名次百分位（主，1＝當季最好、0＝最差，同分取平均名次）、累積報酬（扣成本）、
    日頻 MDD 與 Sharpe（rf=0、×√252；把各季選中規則的逐日淨值接起來）
  - 成本：一律用各規則在步驟1 自己路徑的成本（換規則多出的換手不計，屬近似；所有路徑同一口徑）。
    LLM 另附步驟2 用實際換手結算的累積報酬（`phase2*_settled`）當參考。
資料版本：台股 elite_grid／step2 為 2026-10-30～11-02（股價已修正；策略名單為研究部重跑前的舊名單），美股 _liq 2026-10-28～29（有效）。
輸出：`_analysis_outputs_applayer/rank_dynamics/greedy_vs_oracle/llm_vs_greedy_2024_25.csv`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app import rule_selector as RS  # noqa: E402

GRID = RS.GRID
STEP2 = RS.ROOT / "_analysis_outputs_applayer" / "step2"
OUT = RS.ROOT / "_analysis_outputs_applayer" / "rank_dynamics" / "greedy_vs_oracle"
GROUPS = [("TW", "exclude_v1"), ("US", "exclude_v1"), ("TW", "baseline"), ("US", "baseline")]
VER = {"TW": "台股 2026-10-30～11-02（股價已修正；策略名單為研究部重跑前的舊名單）",
       "US": "美股 _liq 2026-10-28～29（資料未變，有效）"}
STYLE = {"balanced": "平衡型", "momentum": "追強型", "contrarian": "逆勢型"}
N_SEEDS, SEED = 1000, 20261114


def load(market, pool):
    g, c = RS.load_returns(market, pool, with_sig=True)
    n = g - c
    assert len(n) == 44 and n.index.get_level_values(0)[-8:].tolist() == [4] * 8
    tag = RS.TAG[(market, pool)]
    parts = []
    for w in (1, 2, 3, 4):
        nav = pd.concat([pd.read_parquet(GRID / f"daily_net_{tag}_w{w}.parquet"),
                         pd.read_parquet(GRID / f"daily_net_{tag}_sig_w{w}.parquet")], axis=1)
        nav = nav.loc[:, ~nav.columns.duplicated()].sort_index()
        r = nav.pct_change()
        r.iloc[0] = nav.iloc[0] - 1
        parts.append(r)
    return n, pd.concat(parts).sort_index()


def pct_rank(R: np.ndarray) -> np.ndarray:
    rk = pd.DataFrame(R).rank(axis=1, method="average").values
    return (rk - 1) / (R.shape[1] - 1)


def daily_stats(daily, starts, ends, held):
    r = np.concatenate([daily.loc[(daily.index > s) & (daily.index <= e), h].values for s, e, h in zip(starts, ends, held)])
    w = np.r_[1.0, np.cumprod(1 + r)]
    return float((w / np.maximum.accumulate(w) - 1).min()), float(r.mean() / r.std(ddof=1) * np.sqrt(252))


def llm_picks(market, pool, ver):
    f = STEP2 / f"phase2{'v2' if ver == 2 else ''}_{market}_{pool}_gpt-5-mini.jsonl"
    rows = [json.loads(x) for x in open(f, encoding="utf-8")]
    d = pd.DataFrame(rows)
    d = d[d.plan.isin(["A", "B", "C"])]
    d["k"], d["rep"] = d.k.astype(int), d.rep.astype(int)
    return d, f


def settled(market, pool, ver):
    f = STEP2 / f"phase2{'v2' if ver == 2 else ''}_settled_{market}_{pool}_gpt-5-mini.csv"
    s = pd.read_csv(f)
    return s[s.kind == "方案"]


def run_group(market, pool):
    n, daily = load(market, pool)
    idx = n.index
    E = list(range(36, 44))                                  # window4 的 8 季
    as_of = [idx[i][2] for i in E]
    ends = as_of[1:] + ["2025-12-31"]
    rows = []
    for ver, menus in ((1, RS.MENUS), (2, RS.MENUS2)):
        llm, src = llm_picks(market, pool, ver)
        st = settled(market, pool, ver)
        for plan, mf in menus.items():
            arms = mf(market, pool)
            K = len(arms)
            V = n[arms].values                               # 44×K
            R = V[E]
            P = pct_rank(R)
            col = {a: j for j, a in enumerate(arms)}
            menu_name = f"版本{ver}・選單{plan}{'2' if ver == 2 else ''}（{K} 個）"
            base = {"組別": f"{market}・{pool}", "資料版本": VER[market], "選單": menu_name, "選項數": K}

            def add(name, picks, extra=None):
                pi = np.array([col[a] for a in picks])
                r = R[np.arange(8), pi]
                mdd, sh = daily_stats(daily, as_of, ends, picks)
                rows.append({**base, "路徑": name, "名次百分位": float(P[np.arange(8), pi].mean()),
                             "累積報酬_扣成本": float(np.prod(1 + r) - 1), "MDD_日頻": mdd, "Sharpe_日頻": sh, **(extra or {})})

            for L in (4, 1, 2):
                add(f"local greedy（回看{L}季）", [arms[int(np.argmax(np.prod(1 + V[i - L:i], axis=0)))] for i in E])
            add("global maxima（上帝視角）", [arms[j] for j in R.argmax(axis=1)])
            add("事後最佳固定（上帝視角）", [arms[int(np.argmax(np.prod(1 + R, axis=0)))]] * 8)
            is_q = V[:36]
            add("固定樣本內最佳（2015–23 季 Sharpe）", [arms[int(np.nanargmax(is_q.mean(0) / is_q.std(0, ddof=1)))]] * 8)
            add("control0（不調整）", ["REF_control0"] * 8)
            rng = np.random.default_rng(SEED)
            ri = rng.integers(K, size=(N_SEEDS, 8))
            rp = P[np.arange(8)[None, :], ri].mean(axis=1)
            rc = np.prod(1 + R[np.arange(8)[None, :], ri], axis=1) - 1
            rows.append({**base, "路徑": "每季隨機選（1,000 次）", "名次百分位": float(rp.mean()),
                         "累積報酬_扣成本": float(np.median(rc)), "隨機_名次p05": float(np.quantile(rp, .05)),
                         "隨機_名次p95": float(np.quantile(rp, .95)), "隨機_累積p05": float(np.quantile(rc, .05)),
                         "隨機_累積p95": float(np.quantile(rc, .95))})
            for sty in ("balanced", "momentum", "contrarian"):
                sub = llm[(llm.plan == plan) & (llm["style"] == sty)]
                reps = []
                for rep, g in sub.groupby("rep"):
                    g = g.sort_values("k")
                    assert g.k.tolist() == list(range(1, 9)), f"{src.name} {plan}/{sty}/rep{rep} 季數不完整"
                    picks = g.combo.tolist()
                    assert all(a in col for a in picks), f"{plan}/{sty}/rep{rep} 選了選單外的規則"
                    pi = np.array([col[a] for a in picks])
                    r = R[np.arange(8), pi]
                    mdd, sh = daily_stats(daily, as_of, ends, picks)
                    reps.append({"pct": P[np.arange(8), pi].mean(), "cum": np.prod(1 + r) - 1, "mdd": mdd, "sh": sh})
                rp_ = pd.DataFrame(reps)
                ss = st[(st.plan == plan) & (st["style"] == sty)]
                rows.append({**base, "路徑": f"LLM 方案{plan}・{STYLE[sty]}（{len(rp_)} 次平均）",
                             "名次百分位": float(rp_.pct.mean()), "名次百分位_最低": float(rp_.pct.min()),
                             "名次百分位_最高": float(rp_.pct.max()), "累積報酬_扣成本": float(rp_.cum.mean()),
                             "MDD_日頻": float(rp_.mdd.mean()), "Sharpe_日頻": float(rp_.sh.mean()),
                             "贏過隨機比例_名次": float((rp_.pct.mean() > rp).mean()),
                             "參考_步驟2實際換手結算_累積": float(ss.cum_net.mean()) if len(ss) else np.nan})
    print(f"{market}・{pool} 完成", flush=True)
    return rows


def main():
    rows = []
    for grp in GROUPS:
        rows += run_group(*grp)
    p = OUT / "llm_vs_greedy_2024_25.csv"
    pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8-sig")
    print(p)


if __name__ == "__main__":
    main()
