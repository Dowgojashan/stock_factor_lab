# -*- coding: utf-8 -*-
"""k 錨點最終版：把 _prelim_k_anchor_full45.py 的 32 窗（17 獨立窗，凍結期內
oos_end<=2023-12-31）加上 _prelim_k_anchor_turnover_cost.py 驗證過的換手成本
調整（台股手續費 1.425/1000 + 證交稅 3/1000），算出扣過真實交易成本的即時
系統版超額報酬——這是拿來凍結 k 的最終錨點依據（跟凍結矩陣版的「扣成本組合
vs 沒扣成本基準」比較基礎一致：兩邊都用扣過成本的組合報酬）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH, resolve_holdings  # noqa: E402

from app.performance import measure  # noqa: E402

MEMBERS_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "k_anchor_full45_netcost.csv"

FREEZE_CUTOFF = pd.Timestamp("2023-12-31")
FEE_RATIO = 1.425 / 1000
TAX_RATIO = 3 / 1000
ROUNDTRIP_COST = 2 * FEE_RATIO + TAX_RATIO


def quarter_ends(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="Q")]


def month_end(ym: str) -> str:
    """把 parquet 裡「YYYY-MM」格式的月份字串轉成該月最後一天。
    🔴 修正：原版直接把 `oos_end`（如 "2017-12"）傳給 `quarter_ends()`，
    `pd.Timestamp("2017-12")` 解析成該月第一天，導致每個窗次都漏算最後一季
    （已用 `_prelim_k_anchor_check.py` 手寫完整日期版本對照抓到）。"""
    return pd.Period(ym, freq="M").end_time.strftime("%Y-%m-%d")


def load_usable_windows() -> pd.DataFrame:
    m = pd.read_parquet(MEMBERS_PATH)
    key = dict(tree_key="TW", k_mode="silhouette_is", ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    sub = sub.copy()
    sub["oos_end_dt"] = pd.to_datetime(sub["oos_end"])
    usable = sub[sub["oos_end_dt"] <= FREEZE_CUTOFF].copy()
    return usable.sort_values(["scheme", "window_no"]).reset_index(drop=True)


def resolve_weights(md: MarketData, idx: pd.DataFrame, uids: list[str], as_of: str) -> dict[str, float]:
    n_strat = len(uids)
    weights: dict[str, float] = {}
    for uid in uids:
        row = idx.loc[uid]
        try:
            syms, _ = resolve_holdings(md, row, as_of)
        except RuntimeError:
            continue
        if not syms:
            continue
        w = (1.0 / n_strat) / len(syms)
        for s in syms:
            weights[s] = weights.get(s, 0.0) + w
    return weights


def turnover(w_old: dict[str, float], w_new: dict[str, float]) -> float:
    keys = set(w_old) | set(w_new)
    return sum(abs(w_new.get(k, 0.0) - w_old.get(k, 0.0)) for k in keys) / 2.0


def main():
    windows = load_usable_windows()
    print(f"可用窗次（oos_end<=2023-12-31，排除實驗期間）：{len(windows)} / 45")

    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    print("載入 TW MarketData…")
    md = MarketData("TW")
    md_map = {"TW": md}

    t0 = time.time()
    results = []
    for i, row in windows.iterrows():
        scheme, wno = row["scheme"], row["window_no"]
        is_end, oos_start, oos_end = row["is_end"], row["oos_start"], row["oos_end"]
        uids = list(row["members"])
        checkpoints = [month_end(str(is_end))] + quarter_ends(str(oos_start), month_end(str(oos_end)))

        port_rets, ew_rets, turnovers = [], [], []
        prev_w = None
        for j in range(len(checkpoints) - 1):
            as_of, end = checkpoints[j], checkpoints[j + 1]
            weights = resolve_weights(md, idx, uids, as_of)
            if prev_w is not None:
                turnovers.append(turnover(prev_w, weights))
            prev_w = weights
            res = measure(md_map, weights, as_of, end)
            pr = res["portfolio_realized_return"]
            er = res["by_market_benchmark"]["TW"]["equal_weight_benchmark_return"]
            port_rets.append(pr)
            ew_rets.append(er)

        turnovers_full = [1.0] + turnovers  # 第一季視為全新建倉
        cost_per_q = [t * ROUNDTRIP_COST for t in turnovers_full]
        port_rets_net = [pr - c for pr, c in zip(port_rets, cost_per_q)]

        cum_port = pd.Series([1 + r for r in port_rets]).prod() - 1
        cum_port_net = pd.Series([1 + r for r in port_rets_net]).prod() - 1
        cum_ew = pd.Series([1 + r for r in ew_rets]).prod() - 1
        n_years = len(port_rets) / 4.0
        ann_port = (1 + cum_port) ** (1 / n_years) - 1
        ann_port_net = (1 + cum_port_net) ** (1 / n_years) - 1
        ann_ew = (1 + cum_ew) ** (1 / n_years) - 1
        ann_excess_gross = ann_port - ann_ew
        ann_excess_net = ann_port_net - ann_ew
        avg_turnover = sum(turnovers) / len(turnovers) if turnovers else float("nan")

        elapsed = time.time() - t0
        print(f"[{i+1}/{len(windows)}] {scheme}-{wno} ({is_end}~{oos_end}, {len(port_rets)}季)："
              f"毛超額={ann_excess_gross:+.2%}  淨超額={ann_excess_net:+.2%}  "
              f"換手={avg_turnover:.0%}  累計耗時={elapsed:.0f}s")

        results.append({
            "scheme": scheme, "window_no": wno, "is_end": is_end,
            "oos_start": oos_start, "oos_end": oos_end, "n_quarters": len(port_rets),
            "avg_turnover": avg_turnover,
            "ann_excess_gross": ann_excess_gross, "ann_excess_net": ann_excess_net,
            "cost_drag": ann_excess_gross - ann_excess_net,
        })

    df = pd.DataFrame(results)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {OUT_PATH}")

    uniq = df.drop_duplicates(subset=["is_end", "oos_start", "oos_end"])
    print(f"\n=== 32 窗（含重複）vs {len(uniq)} 獨立窗：淨超額（扣真實交易成本）===")
    print(f"32 窗平均淨超額 = {df.ann_excess_net.mean():+.2%}/yr")
    print(f"{len(uniq)} 獨立窗平均淨超額 = {uniq.ann_excess_net.mean():+.2%}/yr"
         f"（標準差 {uniq.ann_excess_net.std():.2%}，勝率 {(uniq.ann_excess_net>0).mean():.1%}）")
    print(f"平均換手成本拖累 = {uniq.cost_drag.mean():.2%}/yr")
    import scipy.stats as st
    t1, p1 = st.ttest_1samp(uniq.ann_excess_net, 0.0453)
    t2, p2 = st.ttest_1samp(uniq.ann_excess_net, 0.0260)
    print(f"對 H0=4.53%: t={t1:.2f}, p={p1:.6f}")
    print(f"對 H0=2.60%: t={t2:.2f}, p={p2:.6f}")


if __name__ == "__main__":
    main()
