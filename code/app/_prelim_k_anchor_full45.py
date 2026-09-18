# -*- coding: utf-8 -*-
"""k 錨點驗證的完整版：把 _prelim_k_anchor_check.py 的方法（真實權重 + performance.measure()
串接的即時系統版超額報酬）從 3 窗（scheme E windows 1-3）擴大到 M-17 原始統計用的全部 45 窗
（scheme A-L, R，TW，A_hrp，跟凍結矩陣同一組 tree_key/k_mode/ratio/allocation/group）。

🔴 §11.7 凍結原則：k 只能用 2007-2023 的資料決定，2024-2025 屬於實驗期間不可用來定參數。
45 窗裡有 13 窗的 oos_end 落在 2024-01 之後（會碰到實驗期間），這裡明確排除，
只用 oos_end<=2023-12-31 的 32 窗——這是本腳本跟「全部 45 窗」版本的唯一差異，
其餘方法論（真實 resolve_holdings + performance.measure()、季頻重新平衡）跟
_prelim_k_anchor_check.py 完全一致，直接可比。
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
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "k_anchor_full45.csv"

FREEZE_CUTOFF = pd.Timestamp("2023-12-31")


def quarter_ends(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="Q")]


def month_end(ym: str) -> str:
    """把 parquet 裡「YYYY-MM」格式的月份字串轉成該月最後一天。
    🔴 `pd.Timestamp("2017-12")` 會解析成該月**第一天**（2017-12-01），
    若直接拿去當 `quarter_ends()` 的 end 參數，`pd.date_range` 會判定
    2017-12-31 超過終點而漏掉最後一季——每個窗次都會少算一季（已用
    `_prelim_k_anchor_check.py` 的手寫完整日期版本對照抓到，同一窗次
    印出的季數少 1）。跟 `actions.py` 的 `_oos_end_month_end()` 同一個問題。
    """
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

        port_rets, ew_rets = [], []
        for j in range(len(checkpoints) - 1):
            as_of, end = checkpoints[j], checkpoints[j + 1]
            weights = resolve_weights(md, idx, uids, as_of)
            res = measure(md_map, weights, as_of, end)
            pr = res["portfolio_realized_return"]
            er = res["by_market_benchmark"]["TW"]["equal_weight_benchmark_return"]
            port_rets.append(pr)
            ew_rets.append(er)

        cum_port = pd.Series([1 + r for r in port_rets]).prod() - 1
        cum_ew = pd.Series([1 + r for r in ew_rets]).prod() - 1
        n_years = len(port_rets) / 4.0
        ann_port = (1 + cum_port) ** (1 / n_years) - 1
        ann_ew = (1 + cum_ew) ** (1 / n_years) - 1
        ann_excess = ann_port - ann_ew

        elapsed = time.time() - t0
        print(f"[{i+1}/{len(windows)}] {scheme}-{wno} ({is_end}~{oos_end}, {len(port_rets)}季)"
              f"：年化超額={ann_excess:+.2%}/yr  累計耗時={elapsed:.0f}s")

        results.append({
            "scheme": scheme, "window_no": wno, "is_end": is_end,
            "oos_start": oos_start, "oos_end": oos_end, "n_quarters": len(port_rets),
            "cum_port": cum_port, "cum_ew": cum_ew, "cum_excess": cum_port - cum_ew,
            "ann_port": ann_port, "ann_ew": ann_ew, "ann_excess": ann_excess,
        })

    df = pd.DataFrame(results)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {OUT_PATH}")

    print("\n=== 即時系統版（32 窗，凍結期內）vs 凍結矩陣版 M-17（45 窗，+2.60pp/yr，勝率71.1%）===")
    print(f"即時系統版 32 窗平均年化超額 = {df.ann_excess.mean():+.2%}/yr")
    print(f"即時系統版 32 窗勝率 = {(df.ann_excess > 0).mean():.1%}")
    print(f"即時系統版標準差 = {df.ann_excess.std():.2%}")
    import scipy.stats as st
    t_stat, p_val = st.ttest_1samp(df.ann_excess, 0.0453)
    print(f"對 H0: 均值=4.53% 做 t 檢定：t={t_stat:.3f}, p={p_val:.4f}")
    t_stat2, p_val2 = st.ttest_1samp(df.ann_excess, 0.0260)
    print(f"對 H0: 均值=2.60% 做 t 檢定：t={t_stat2:.3f}, p={p_val2:.4f}")


if __name__ == "__main__":
    main()
