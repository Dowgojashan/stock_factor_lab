# -*- coding: utf-8 -*-
"""老師 11/10 意見：local greedy 離 global maxima（上帝視角）有多遠（2026-11-12，設計經使用者確認）。

不呼叫 LLM、不重跑回測：直接用步驟1 每季報酬 `elite_grid_1027/quarterly_*.parquet`（與規則排名熱力圖同一份）。

設計（看結果前寫死）：
  - 選項（主）：優等生資金 100%，13 類規則 × 取前 3／5／10 名，分法＝平均分 ⇒ 39 個。
    「取幾名」是選項的一部分，所以每季也只用過去資訊決定取幾名。
    敏感度：資金 100% 的完整 208 組（多了取 1 名與 4 種分法）。
  - 評估期：2016Q1～2025Q4 共 40 季；2015 年 4 季只當回看資料。每季重挑一次。
  - Local greedy：季末只用該季以前、過去 L 季的累積報酬（扣成本）挑最高者，持有下一季；主結果 L=4，L=1、2 為敏感度；
    另列「擴張窗」＝用到目前為止全部歷史（＝不含前視的「固定樣本內最佳」）。同分取排序在前者。
  - Global maxima（oracle）：每季直接選當季實際報酬（扣成本）最高者。附加參考：事後選一個固定到底最好的選項。
  - 對照：control0、全部選項每季平均、每季隨機選（1,000 次）。
  - KPI：
      (c) 名次捕獲率（主）＝每季選中選項在當季所有選項中的百分位（1＝最好、0＝最差、平均名次處理同分）取平均；oracle＝1，隨機≈0.5
      (a) 捕獲率＝(路徑−基準)/(oracle−基準)，基準＝control0 或全部選項平均；分「複利累積」與「每季平均報酬」兩版
      (b) 報酬比＝路徑/oracle；同樣分兩版
    另附 MDD、Sharpe（rf=0；季頻 ×√4、日頻 ×√252，日頻用各季選中選項的逐日淨值接起來）。
  - 報酬一律扣成本（各選項在步驟1 自己路徑的成本）；換選項多出的換手不計，屬近似。
資料版本：台股＝2026-10-30（舊名單：台股股價修正 11-01 與白名單更新 11-09 之前），美股＝2026-10-28 `_liq`（資料未變，有效）。
輸出：`_analysis_outputs_applayer/rank_dynamics/greedy_vs_oracle/`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app import _rank_dynamics as RD  # noqa: E402

OUT = RD.OUT / "greedy_vs_oracle"
E0 = RD.EVAL_START
T = 44 - E0                     # 40 季
LOOKS = [4, 1, 2]               # 第一個是主結果
N_SEEDS = 1000
SEED = 20261112
VERSION = {"TW": "2026-10-30（舊名單：台股股價修正與白名單更新前，待研究部重跑後補新版）",
           "US": "2026-10-28 _liq（美股資料未變，有效）"}


def arm_sets(meta: pd.DataFrame) -> dict[str, list[str]]:
    m = meta[meta.sleeve == 1.0]
    main = m[(m.n.isin([3, 5, 10])) & (m.scheme == "equal")].index.tolist()
    assert len(main) == 39 and len(m) == 208, (len(main), len(m))
    return {"39選項（13類×前3/5/10・平均分）": main, "208組（敏感度）": m.index.tolist()}


def greedy(n: pd.DataFrame, arms: list[str], L: int | None) -> np.ndarray:
    """回傳評估期每季選中的 arms 索引。決策只用第 i 季以前的資料（iloc[:i]）。"""
    V = n[arms].values
    out = []
    for i in range(E0, 44):
        lo = 0 if L is None else i - L
        assert lo >= 0
        out.append(int(np.argmax(np.prod(1 + V[lo:i], axis=0))))   # argmax 同分取第一個
    return np.array(out)


def pct_rank(R: np.ndarray) -> np.ndarray:
    """R：40×K 當季報酬 → 每格的百分位（1＝當季最好），同分取平均名次。"""
    rk = pd.DataFrame(R).rank(axis=1, ascending=True, method="average").values   # 1＝最差
    return (rk - 1) / (R.shape[1] - 1)


def mdd(r: np.ndarray) -> float:
    w = np.cumprod(1 + r)
    return float((w / np.maximum.accumulate(np.r_[1.0, w])[1:] - 1).min())


def sharpe(r: np.ndarray, ann: float) -> float:
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ann)) if len(r) > 1 and r.std() > 0 else np.nan


def daily_path(daily: pd.DataFrame, ends: list[str], held: list[str]) -> np.ndarray:
    """每季持有的選項，取該季（上季末, 本季末] 的逐日淨報酬接起來。"""
    parts = []
    prev = pd.Timestamp(ends[0]) - pd.offsets.QuarterEnd(1)     # 期間第一季的上季末（期間都是連續季）
    for e, h in zip(ends, held):
        t = pd.Timestamp(e)
        parts.append(daily.loc[(daily.index > prev) & (daily.index <= t), h].values)
        prev = t
    return np.concatenate(parts)


def run_group(market: str, pool: str, suf: str):
    g, n, meta = RD.load(market, pool, suf)
    daily = RD.load_daily(market, pool, suf)
    ends = list(n.index[E0:])
    endsI = pd.Index(ends)
    grp = f"{RD.GLAB[market]}・{RD.GLAB[pool]}"
    rows, picks = [], []
    c0 = n["REF_control0"].values[E0:]

    for set_name, arms in arm_sets(meta).items():
        R = n[arms].values[E0:]                      # 40×K 當季淨報酬
        P = pct_rank(R)
        K = len(arms)
        avg = R.mean(axis=1)                          # 全部選項每季平均
        oi = R.argmax(axis=1)
        orc = R[np.arange(T), oi]
        fixed_best = int(np.argmax(np.prod(1 + R, axis=0)))

        paths = {}                                    # 名稱 → (每季索引 or None, 每季報酬)
        for L in LOOKS:
            paths[f"local greedy（回看{L}季）"] = greedy(n, arms, L)
        paths["擴張窗 greedy（全部過去）"] = greedy(n, arms, None)
        paths["global maxima（上帝視角）"] = oi
        paths["事後最佳固定選項（上帝視角）"] = np.full(T, fixed_best)

        rng = np.random.default_rng(SEED)
        rand = rng.integers(K, size=(N_SEEDS, T))
        rR, rP = R[np.arange(T)[None, :], rand], P[np.arange(T)[None, :], rand]

        for plab, a, b in RD.PERIODS:
            m = RD.period_mask(endsI, a, b)
            cum = lambda r: float(np.prod(1 + r[..., m], axis=-1) - 1) if np.ndim(r) == 1 else np.prod(1 + r[:, m], axis=1) - 1
            O_c, O_m = cum(orc), float(orc[m].mean())
            base = {"control0": (cum(c0), float(c0[m].mean())), "全選項平均": (cum(avg), float(avg[m].mean()))}
            r_cum, r_mean, r_pct = cum(rR), rR[:, m].mean(axis=1), rP[:, m].mean(axis=1)

            def kpi(name, r, pct, idx=None):
                Gc, Gm = cum(r), float(r[m].mean())
                row = {"組別": grp, "資料版本": VERSION[market], "選項集合": set_name, "選項數": K, "期間": plab,
                       "路徑": name, "名次捕獲率(c)": float(pct[m].mean()) if pct is not None else np.nan,
                       "累積報酬_扣成本": Gc, "每季平均報酬": Gm}
                for bn, (Bc, Bm) in base.items():
                    row[f"捕獲率(a)_複利_基準{bn}"] = (Gc - Bc) / (O_c - Bc)
                    row[f"捕獲率(a)_每季平均_基準{bn}"] = (Gm - Bm) / (O_m - Bm)
                row["報酬比(b)_複利"] = Gc / O_c
                row["報酬比(b)_每季平均"] = Gm / O_m
                row["MDD_季頻"] = mdd(r[m])
                row["Sharpe_季頻"] = sharpe(r[m], 4)
                if idx is not None:
                    dr = daily_path(daily, [e for e, k in zip(ends, m) if k], [arms[j] for j, k in zip(idx, m) if k])
                    row["MDD_日頻"], row["Sharpe_日頻"] = mdd(dr), sharpe(dr, 252)
                if pct is not None:
                    row["隨機_名次捕獲率_p05"], row["隨機_名次捕獲率_p50"], row["隨機_名次捕獲率_p95"] = np.quantile(r_pct, [.05, .5, .95])
                    row["贏過隨機比例_名次"] = float((row["名次捕獲率(c)"] > r_pct).mean())
                row["隨機_累積報酬_p05"], row["隨機_累積報酬_p50"], row["隨機_累積報酬_p95"] = np.quantile(r_cum, [.05, .5, .95])
                row["贏過隨機比例_累積報酬"] = float((Gc > r_cum).mean())
                rows.append(row)

            for name, idx in paths.items():
                kpi(name, R[np.arange(T), idx], P[np.arange(T), idx], idx)
            kpi("control0（不調整）", c0, None)
            kpi("全選項每季平均", avg, None)
            # 隨機本身（中位數那一條的 KPI 意義不大，只列分布摘要）
            rows.append({"組別": grp, "資料版本": VERSION[market], "選項集合": set_name, "選項數": K, "期間": plab,
                         "路徑": "每季隨機選（1,000 次中位數）", "名次捕獲率(c)": float(np.median(r_pct)),
                         "累積報酬_扣成本": float(np.median(r_cum)), "每季平均報酬": float(np.median(r_mean))})

        for name in ("local greedy（回看4季）", "global maxima（上帝視角）"):
            for i, (e, j) in enumerate(zip(ends, paths[name])):
                picks.append({"組別": grp, "選項集合": set_name, "路徑": name, "決策時點": n.index[E0 + i - 1],
                              "持有季末": e, "選中": arms[j], "類別": RD.FAM[RD.fam_of(arms[j])],
                              "當季報酬_扣成本": R[i, j], "當季百分位": P[i, j],
                              "當季最好": arms[oi[i]], "當季最好報酬": orc[i]})
    print(f"{grp} 完成", flush=True)
    return rows, picks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, picks = [], []
    for grp in RD.GROUPS:
        r, p = run_group(*grp)
        rows += r
        picks += p
    pd.DataFrame(rows).to_csv(OUT / "greedy_vs_oracle.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(picks).to_csv(OUT / "greedy_vs_oracle_picks.csv", index=False, encoding="utf-8-sig")
    print(OUT)


if __name__ == "__main__":
    main()
