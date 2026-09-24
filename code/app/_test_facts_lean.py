# -*- coding: utf-8 -*-
"""facts_lean.py 驗證：組真實資料、確認 CSV 格式正確，並且**故意**觸發一次
物理防線（塞一個流量變數進預測 facts），確認防線真的會擋下來，不是裝飾用的。
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

from app import actions, diagnose, facts_lean, monitor, triggers  # noqa: E402

MEMBERS_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"


def get_members():
    m = pd.read_parquet(MEMBERS_PATH)
    key = dict(tree_key="TW", scheme="E", window_no=4, k_mode="silhouette_is",
              ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
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
    mcap_wide = monitor.fetch_mcap_wide(conn, start="2007-01-01")

    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    print("載入 TW MarketData…")
    md = MarketData("TW")
    md_map = {"TW": md}
    uids = get_members()

    as_of, end = "2024-09-30", "2024-12-31"   # 2024Q4：M1-D 首次觸發的那一季

    # 🔴 2026-09-17（Agent-B 第一次真實質疑抓到的真問題）：狀態層（env／proc／
    # m1d，「現在長什麼樣」）跟結果層（outcome，「這段期間發生了什麼」）本來
    # 就該用不同時間點的持股快照，之前這裡誤用同一份 `weights`（as_of 解的）
    # 餵給兩邊，導致 proc 描述的其實是「季初」的持股，跟用 end 算的 m1d 對不上
    # （q1_weight 兩邊數字不一致，B 抓到）。正確做法：
    #   - outcome（回顧，測「as_of→end 這段期間，季初那筆持股表現如何」）
    #     ⇒ 用 **as_of** 解的持股
    #   - env／proc／m1d（預測，測「現在——也就是這一季结束、快時鐘重解後——
    #     長什麼樣」）⇒ 用 **end** 解的持股／市場資料，跟 as_of 那筆脫鉤
    weights_as_of = resolve_weights(md, idx, uids, as_of)
    weights_end = resolve_weights(md, idx, uids, end)
    mcap_row_end = monitor.asof_row(mcap_wide, end)

    env = monitor.environment_layer(mcap_wide, end)
    proc = monitor.process_layer(weights_end, mcap_row_end, prev_weights=weights_as_of,
                                 member_uids=uids, candidate_idx=idx)
    outcome = monitor.outcome_layer(md_map, weights_as_of, as_of, end)

    cond = triggers.register_m1d(mcap_wide, "2023-12-31")
    # 模擬跑到這一季：先跑過前面三季（都是 NONE），這裡示範直接評一次
    m1d = triggers.evaluate_quarter(cond, mcap_wide, end, prev_state="NONE")

    acts = actions.list_available_actions(end)

    excess_hist = [outcome["excess_vs_equal_weight"]]
    diag = diagnose.run_diagnosis(n_unique_stocks=proc["n_unique_stocks"],
                                  excess_vs_ball_history=excess_hist)

    print("\n=== 3b 預測區 facts（應該成功，不觸發防線）===")
    prospective = facts_lean.build_prospective_facts(env, proc, m1d)
    for k, v in prospective.items():
        print(f"--- {k} ---")
        print(v)

    print("\n=== 3a 回顧區 facts ===")
    retrospective = facts_lean.build_retrospective_facts(
        outcome, diag, memory={"last_period_summary": "測試用假資料"})
    for k, v in retrospective.items():
        print(f"--- {k} ---")
        print(v)

    print("\n=== 故意觸發物理防線：把流量變數塞進預測 facts ===")
    try:
        bad_actions = list(acts) + [{"ratio": "legacy", "allocation": "equal",
                                     "n_windows_visible": 1,
                                     "mean_oos_cagr": 0.1,
                                     "portfolio_realized_return": 0.05}]
        # 直接手動組一個違規 dict，繞過既有函式，模擬未來有人不小心塞了流量變數
        bad_facts = dict(prospective)
        bad_facts["leaked_outcome"] = {"portfolio_realized_return": 0.05}
        facts_lean._assert_no_flow_leakage(bad_facts, context="故意測試")
        print("🔴 FAIL：防線沒有擋下來！")
    except RuntimeError as e:
        print(f"通過：防線正確擋下 -> {e}")


if __name__ == "__main__":
    main()
