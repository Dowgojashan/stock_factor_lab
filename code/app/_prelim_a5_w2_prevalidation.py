# -*- coding: utf-8 -*-
"""A5：W2（股票層市值傾斜）前置驗證，§7.6 規定**只能用 window 1-3**，
不可碰 window 4（否則就是用答案驗證答案的循環論證）。

W2 定義（§7.5）：`w ∝ (1/n)^α × mktcap^(1−α)`。這裡採用「對現有等權-across-
策略基準權重 w0(s) 做傾斜」的具體化版本：`w(s) ∝ w0(s)^α × mktcap(s)^(1−α)`，
正規化後加總為 1。這個具體化選擇的理由：
  - α=1 時 w(s) ∝ w0(s)，**完全還原現有 legacy/equal 基準**（不改變狀態）
  - α=0 時 w(s) ∝ mktcap(s)，變成對「目前這批被選中的股票」做市值加權
    （不是對全市場市值加權，是對投組已選中的股票子集市值加權）
  - α 介於中間時連續內插，跟現有 `ratio`／`allocation` 矩陣維度一樣是
    「同一組持股，換一種配權重方式」，可以直接放進矩陣比較

用 window 1-3（scheme E／A_hrp／legacy／equal，跟 k 錨點驗證同一批持股，
2015-2023，8 季一組 × 3 窗）逐季驗證 α ∈ {1.0, 0.75, 0.5, 0.25, 0.0} 的表現，
建立「已驗證格子」，之後才能讓 W2 進實驗的動作空間。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from database import Database  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH, resolve_holdings  # noqa: E402

from app.performance import measure  # noqa: E402

MEMBERS_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "a5_w2_prevalidation.csv"

ALPHAS = [1.0, 0.75, 0.5, 0.25, 0.0]

WINDOWS = {
    1: {"is_end": "2014-12-31", "oos_start": "2015-01-01", "oos_end": "2017-12-31"},
    2: {"is_end": "2017-12-31", "oos_start": "2018-01-01", "oos_end": "2020-12-31"},
    3: {"is_end": "2020-12-31", "oos_start": "2021-01-01", "oos_end": "2023-12-31"},
}


def quarter_ends(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="Q")]


def get_members(window_no: int) -> list[str]:
    m = pd.read_parquet(MEMBERS_PATH)
    key = dict(tree_key="TW", scheme="E", window_no=window_no, k_mode="silhouette_is",
              ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    assert len(sub) == 1
    return list(sub.iloc[0]["members"])


def resolve_w0(md: MarketData, idx: pd.DataFrame, uids: list[str], as_of: str) -> dict[str, float]:
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


def fetch_mcap_wide(conn, start="2014-01-01") -> pd.DataFrame:
    q = f"""SELECT s.date, c.company_symbol, s.market_capital
            FROM stock s JOIN company c ON s.company_id=c.id
            WHERE c.exchange_name IN ('TWSE') AND s.market_capital IS NOT NULL
              AND s.date >= '{start}'"""
    d = pd.read_sql(q, conn)
    d["date"] = pd.to_datetime(d["date"])
    return d.pivot_table(index="date", columns="company_symbol", values="market_capital", aggfunc="last").sort_index()


def mcap_asof(wide: pd.DataFrame, as_of: str) -> pd.Series:
    ts = pd.Timestamp(as_of)
    avail = wide.index[wide.index <= ts]
    return wide.loc[avail.max()]


def tilt_weights(w0: dict[str, float], mcap_row: pd.Series, alpha: float) -> dict[str, float]:
    """w(s) ∝ w0(s)^alpha * mktcap(s)^(1-alpha)。缺市值的股票在 alpha<1 時
    無法算比例，直接排除（跟現有系統「量不到就不裝作有」的一致立場）。"""
    if alpha >= 0.999:
        total = sum(w0.values())
        return {s: w / total for s, w in w0.items()} if total else {}
    raw = {}
    for s, w in w0.items():
        mc = mcap_row.get(s)
        if pd.isna(mc) or mc <= 0 or w <= 0:
            continue
        raw[s] = (w ** alpha) * (mc ** (1 - alpha))
    total = sum(raw.values())
    return {s: v / total for s, v in raw.items()} if total else {}


def main():
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    print("載入 TW MarketData…")
    md = MarketData("TW")
    md_map = {"TW": md}

    db = Database("TW")
    conn = db.create_connection()
    print("讀取市值序列（2014-01 起）…")
    mcap_wide = fetch_mcap_wide(conn)

    rows = []
    for wno, meta in WINDOWS.items():
        uids = get_members(wno)
        checkpoints = [meta["is_end"]] + quarter_ends(meta["oos_start"], meta["oos_end"])

        per_alpha_rets: dict[float, list[float]] = {a: [] for a in ALPHAS}
        ew_rets = []
        coverage = []
        for i in range(len(checkpoints) - 1):
            as_of, end = checkpoints[i], checkpoints[i + 1]
            w0 = resolve_w0(md, idx, uids, as_of)
            mcap_row = mcap_asof(mcap_wide, as_of)
            for a in ALPHAS:
                wt = tilt_weights(w0, mcap_row, a)
                if a < 0.999:
                    coverage.append(sum(wt.get(s, 0) for s in wt) and len(wt) / max(len(w0), 1))
                res = measure(md_map, wt, as_of, end)
                per_alpha_rets[a].append(res["portfolio_realized_return"])
                if a == 1.0:
                    ew_rets.append(res["by_market_benchmark"]["TW"]["equal_weight_benchmark_return"])

        n_years = (len(checkpoints) - 1) / 4.0
        cum_ew = pd.Series([1 + r for r in ew_rets]).prod() - 1
        ann_ew = (1 + cum_ew) ** (1 / n_years) - 1
        cov = sum(coverage) / len(coverage) if coverage else float("nan")

        for a in ALPHAS:
            rets = per_alpha_rets[a]
            cum = pd.Series([1 + r for r in rets]).prod() - 1
            ann = (1 + cum) ** (1 / n_years) - 1
            print(f"window {wno} α={a:.2f}: 年化={ann:+.2%}  (等權大盤={ann_ew:+.2%}  "
                 f"超額={ann-ann_ew:+.2%})  {'' if a>=0.999 else f'平均持股涵蓋率={cov:.1%}'}")
            rows.append({"window_no": wno, "alpha": a, "ann_return": ann,
                        "ann_ew_benchmark": ann_ew, "ann_excess": ann - ann_ew,
                        "avg_mcap_coverage": cov if a < 0.999 else 1.0})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {OUT_PATH}")

    print("\n=== 三窗平均，逐 α ===")
    summary = df.groupby("alpha").agg(mean_ann_return=("ann_return", "mean"),
                                       mean_ann_excess=("ann_excess", "mean")).reset_index()
    print(summary.to_string(index=False))
    best = summary.loc[summary.mean_ann_excess.idxmax()]
    print(f"\n三窗平均表現最好的 α = {best.alpha}（平均超額 {best.mean_ann_excess:+.2%}/yr）")
    a1 = summary.loc[summary.alpha == 1.0, "mean_ann_excess"].iloc[0]
    print(f"對照 α=1.0（現況，等於不做 W2）: {a1:+.2%}/yr")


if __name__ == "__main__":
    main()
