# -*- coding: utf-8 -*-
"""優等生權重網格的分析（實戰收尾追蹤.md §四步驟1「晉級規則」A~D，2026-10-27寫死、跑之前已 commit）。

A. window1~3：各窗用自己的代表策略名單，接成 2015~2023 的日頻淨值；算 Sharpe（rf=0，
   日報酬×√252）、累積報酬、日頻MDD，扣成本與不扣成本各一套
B. 晉級：以**扣成本後 Sharpe** 評選——整體前3名＋6種排名依據（S1~S6）各自最佳1組，
   去重後最多9組；control0／每檔等權／現行市值傾斜全開另列固定對照
C. window4：晉級組＋對照組放正文；完整表放附錄
D. 證偽：全部網格組在 window1~3 的排名 vs window4 排名的 Spearman ρ；各維度平均效果
   在兩段期間是否同號

輸入：`_elite_grid_offline.py` 的輸出（`_analysis_outputs_applayer/elite_grid_1027/`）
用法（cwd=code/）：python -m app._elite_grid_analyze --market TW --pool exclude_v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import pandas as pd  # noqa: E402

from functools import lru_cache  # noqa: E402

from app._elite_grid_offline import OUT_DIR, load_benchmark  # noqa: E402

load_benchmark_cached = lru_cache(maxsize=None)(load_benchmark)   # 同一市場只查一次 DB

IS_WINDOWS = (1, 2, 3)
OOS_WINDOW = 4
FAMILIES = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")   # S7 熱門股、S8 突破股（2026-10-28，只出現在 _sig 網格）


def daily_returns(level: pd.DataFrame) -> pd.DataFrame:
    """窗內淨值（起點=1，不含起點那天）→ 日報酬。"""
    prev = level.shift(1)
    prev.iloc[0] = 1.0
    return level.div(prev).sub(1.0)


def metrics(rets: pd.DataFrame) -> pd.DataFrame:
    lvl = (1.0 + rets).cumprod()
    n_days = len(rets)
    cum = lvl.iloc[-1] - 1.0
    # 🔴 2026-10-28 code review：MDD 的高點要含起點 1.0，否則第一天就下跌的那段回撤不會被算進去
    peak = lvl.cummax().clip(lower=1.0)
    out = pd.DataFrame({
        "cum": cum,
        "cagr": (1.0 + cum) ** (252.0 / n_days) - 1.0,
        "vol": rets.std() * np.sqrt(252.0),
        "sharpe": rets.mean() / rets.std() * np.sqrt(252.0),
        "mdd": lvl.div(peak).sub(1.0).min(),
    })
    return out


def period_returns(tag: str, windows, kind: str) -> pd.DataFrame:
    parts = [daily_returns(pd.read_parquet(OUT_DIR / f"daily_{kind}_{tag}_w{w}.parquet")) for w in windows]
    return pd.concat(parts)


def bench_returns(market: str, windows, tag: str) -> pd.Series:
    """大盤日報酬，取跟策略同一批交易日（以窗起點前一個交易日為基準）。"""
    b = load_benchmark_cached(market)
    parts = []
    for w in windows:
        idx = pd.read_parquet(OUT_DIR / f"daily_net_{tag}_w{w}.parquet").index
        start = b.index[b.index < idx[0]].max()
        sub = b.loc[start:idx[-1]].reindex(pd.DatetimeIndex([start]).append(idx), method="ffill")
        parts.append(sub.pct_change().iloc[1:])
    return pd.concat(parts)


def run(market: str, pool: str, suffix: str = "") -> None:
    tag = f"{market}_{pool}{suffix}"   # suffix="_liq"＝美股流動性門檻版
    combos = pd.read_csv(OUT_DIR / f"combos_{tag}.csv").set_index("combo")
    combos["family"] = combos["score"].str[:2]
    grid = combos.index[~combos["score"].str.startswith("REF_")]
    refs = combos.index[combos["score"].str.startswith("REF_")]

    res = {}
    for label, wins in (("IS", IS_WINDOWS), ("OOS", (OOS_WINDOW,))):
        for kind in ("net", "gross"):
            r = period_returns(tag, wins, kind)
            m = metrics(r)
            b = bench_returns(market, wins, tag)
            bm = metrics(b.to_frame("BENCH")).loc["BENCH"]
            res[(label, kind)] = (m, bm)

    q = pd.read_parquet(OUT_DIR / f"quarterly_{tag}.parquet")
    q["period"] = np.where(q.window == OOS_WINDOW, "OOS", "IS")
    qa = q.groupby(["combo", "period"]).agg(turnover=("turnover", "mean"), max_weight=("max_weight", "mean"),
                                            max_weight_peak=("max_weight", "max")).unstack("period")
    qa.columns = [f"{a}_{b}" for a, b in qa.columns]

    table = combos[["score", "family", "n", "scheme", "sleeve"]].copy()
    for (label, kind), (m, _) in res.items():
        for col in m.columns:
            table[f"{label}_{kind}_{col}"] = m[col]
    table = table.join(qa)

    # B. 晉級（扣成本後 Sharpe）
    is_sh = table.loc[grid, "IS_net_sharpe"]
    top3 = list(is_sh.sort_values(ascending=False).index[:3])
    fam_best = [is_sh[table.loc[grid, "family"] == f].idxmax() for f in FAMILIES
                if (table.loc[grid, "family"] == f).any()]
    promoted = list(dict.fromkeys(top3 + fam_best))
    table["promoted"] = table.index.isin(promoted)
    table["promote_reason"] = ""
    for c in top3:
        table.loc[c, "promote_reason"] += "整體前3;"
    for c in fam_best:
        table.loc[c, "promote_reason"] += f"{table.loc[c, 'family']}最佳;"

    # D. 證偽
    oos_sh = table.loc[grid, "OOS_net_sharpe"]
    rho = is_sh.rank().corr(oos_sh.rank())
    rho_cum = table.loc[grid, "IS_net_cum"].rank().corr(table.loc[grid, "OOS_net_cum"].rank())
    marg_rows = []
    for dim in ("family", "score", "n", "scheme", "sleeve"):
        g = table.loc[grid].groupby(dim)
        is_m = g["IS_net_sharpe"].mean() - is_sh.mean()
        oos_m = g["OOS_net_sharpe"].mean() - oos_sh.mean()
        for lv in is_m.index:
            marg_rows.append({"dimension": dim, "level": lv, "IS_vs_grid_mean": is_m[lv],
                              "OOS_vs_grid_mean": oos_m[lv], "same_sign": bool(np.sign(is_m[lv]) == np.sign(oos_m[lv])),
                              "n_combos": int(g.size()[lv])})
    marg = pd.DataFrame(marg_rows)

    # 輸出
    table.to_csv(OUT_DIR / f"metrics_all_{tag}.csv", encoding="utf-8-sig")
    marg.to_csv(OUT_DIR / f"marginal_{tag}.csv", index=False, encoding="utf-8-sig")
    show_cols = ["promote_reason", "IS_net_sharpe", "IS_net_cum", "IS_net_mdd", "OOS_net_sharpe", "OOS_net_cum",
                 "OOS_gross_cum", "OOS_net_mdd", "turnover_OOS", "max_weight_peak_OOS"]
    main = table.loc[promoted + list(refs), show_cols]
    main.to_csv(OUT_DIR / f"promoted_{tag}.csv", encoding="utf-8-sig")

    bis, boos = res[("IS", "net")][1], res[("OOS", "net")][1]
    lines = [f"# {tag} 晉級結果（規則見腳本docstring，跑之前已寫死）",
             f"網格 {len(grid)} 組＋對照 {len(refs)} 組",
             f"大盤 IS：Sharpe {bis.sharpe:.3f}｜累積 {bis.cum:+.2%}｜MDD {bis.mdd:+.2%}",
             f"大盤 OOS：Sharpe {boos.sharpe:.3f}｜累積 {boos.cum:+.2%}｜MDD {boos.mdd:+.2%}",
             f"D. Spearman ρ（IS扣成本Sharpe排名 vs OOS扣成本Sharpe排名，{len(grid)}組）＝{rho:+.3f}",
             f"   參考：累積報酬排名的 ρ＝{rho_cum:+.3f}",
             f"   各維度平均效果兩期同號比例＝{marg.same_sign.mean():.0%}（{marg.same_sign.sum()}/{len(marg)}）",
             "", main.to_string()]
    (OUT_DIR / f"summary_{tag}.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True, choices=["TW", "US"])
    ap.add_argument("--pool", required=True, choices=["exclude_v1", "baseline"])
    ap.add_argument("--suffix", default="", help="例：_liq")
    a = ap.parse_args(argv)
    run(a.market, a.pool, a.suffix)


if __name__ == "__main__":
    main()
