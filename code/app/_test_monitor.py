# -*- coding: utf-8 -*-
"""monitor.py 正確性驗證：跟今天已經算過、確認無誤的數字對照。
不是正式測試檔（正式測試在 research.tests／未來的 app 測試套件），
這裡只是開發時的驗證腳本，用完可以留著當範例用法。
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

from app import monitor  # noqa: E402

MEMBERS_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"


def get_members(scheme, window_no):
    m = pd.read_parquet(MEMBERS_PATH)
    key = dict(tree_key="TW", scheme=scheme, window_no=window_no, k_mode="silhouette_is",
              ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    assert len(sub) == 1
    return list(sub.iloc[0]["members"])


def resolve_weights(md, idx, uids, as_of):
    n_strat = len(uids)
    weights = {}
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
    db = Database("TW")
    conn = db.create_connection()

    print("=== 測試 1：environment_layer 對照 A1 的 Q1 集中度 ===")
    mcap_wide = monitor.fetch_mcap_wide(conn, start="2007-01-01")
    env_2023 = monitor.environment_layer(mcap_wide, "2023-12-31")
    env_2025 = monitor.environment_layer(mcap_wide, "2025-12-31")
    print(f"  2023-12-31 q1_weight={env_2023['q1_weight']:.4%}（A1 已發布：88.27%）")
    print(f"  2025-12-31 q1_weight={env_2025['q1_weight']:.4%}（A1 已發布：92.61%）")
    assert abs(env_2023["q1_weight"] - 0.8827) < 0.001, "q1_weight 2023 對不上 A1"
    assert abs(env_2025["q1_weight"] - 0.9261) < 0.001, "q1_weight 2025 對不上 A1"
    print("  通過")

    print("\n=== 測試 2：outcome_layer 對照 A2/A3 的第一季已實現報酬 ===")
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    print("載入 TW MarketData…")
    md = MarketData("TW")
    md_map = {"TW": md}
    uids = get_members("E", 4)
    weights = resolve_weights(md, idx, uids, "2023-12-31")
    out = monitor.outcome_layer(md_map, weights, "2023-12-31", "2024-03-31")
    print(f"  投組報酬={out['portfolio_realized_return']:+.4%}（A2 已發布 2023-12-31->2024-03-31：應對照 a2_baseline_control0.csv 第一列）")
    print(f"  對等權超額={out['excess_vs_equal_weight']:+.4%}")

    print("\n=== 測試 3：process_layer 基本檢查 ===")
    mcap_row = monitor.asof_row(mcap_wide, "2023-12-31")
    proc = monitor.process_layer(weights, mcap_row, prev_weights=None,
                                 member_uids=uids, candidate_idx=idx)
    print(f"  n_unique_stocks={proc['n_unique_stocks']}")
    print(f"  max_stock_weight={proc['max_stock_weight']:.4%}（symbol={proc['max_stock_symbol']}）")
    print(f"  factor_exposure_f1 最大佔比={proc['factor_exposure_f1_share']}（band={proc['factor_exposure_f1_top_band']}）")
    print(f"  median_mktcap_weighted={proc['median_mktcap_weighted']}")
    assert proc["n_unique_stocks"] > 0

    print("\n=== 測試 4：市場廣度（跟市值加權 TAIEX 比較，不是等權基準）===")
    tx = pd.read_sql("SELECT date, close FROM taiex_tr ORDER BY date", conn)
    tx["date"] = pd.to_datetime(tx["date"]); tx = tx.set_index("date")["close"].astype(float)
    d0 = tx.index[tx.index <= pd.Timestamp("2023-12-31")].max()
    d1 = tx.index[tx.index <= pd.Timestamp("2024-03-31")].max()
    cw_ret = float(tx.loc[d1] / tx.loc[d0] - 1.0)
    breadth = monitor.market_breadth(md, "2023-12-31", "2024-03-31", cw_ret)
    print(f"  TAIEX報酬指數同期報酬={cw_ret:+.4%}")
    print(f"  {breadth}")

    print("\n=== 測試 5：expanding_percentile 用真實 Q1 序列自我檢查 ===")
    df = pd.read_csv(Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "a1_index_concentration_series.csv", parse_dates=["date"])
    series = df.set_index("date")["idx_q1_weight"]
    pct = monitor.expanding_percentile(series, "2025-12-31", 0.9261)
    print(f"  2025-12-31 的 q1_weight 在此之前歷史的百分位: {pct:.1%}（應該很高，接近或等於 100%）")

    print("\n全部測試跑完。")


if __name__ == "__main__":
    main()
