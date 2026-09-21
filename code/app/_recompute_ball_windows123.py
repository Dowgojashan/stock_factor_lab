# -*- coding: utf-8 -*-
"""§8待辦item10（延伸，2026-09-22）：把window4已經驗證過的股票層B_all重算方法
推廣到window1-3（2015-2023，36季），讓M0/M4/M6/M8的歷史門檻分布
（`m0_performance_history.csv`等，windows1-3, n=36）也建立在同一套量測系統上
——否則window4用新B_all、windows1-3門檻用舊placeholder，會產生新的跨系統
不一致（D24-D26已經示範過這類問題的危險）。

確認過（`_tree_universe`）：anchored scheme下 window1~4 的候選池universe完全
相同（都是6,679檔，因為IS起點固定在2007-01，只有IS_END往後延伸），不需要
為每個window分別重建universe，直接沿用window4已驗證的6,679檔清單
（`tsmc_cluster_concentration.csv`）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import (CANDIDATE_INDEX_PATH, _c_condition,  # noqa: E402
                                       _q_band_condition)

from app.performance import measure  # noqa: E402

CLUSTER_CSV = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "tsmc_cluster_concentration.csv"
WINDOW_QUARTERS = {
    1: ["2014-12-31", "2015-03-31", "2015-06-30", "2015-09-30", "2015-12-31",
       "2016-03-31", "2016-06-30", "2016-09-30", "2016-12-31",
       "2017-03-31", "2017-06-30", "2017-09-30", "2017-12-31"],
    2: ["2017-12-31", "2018-03-31", "2018-06-30", "2018-09-30", "2018-12-31",
       "2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31",
       "2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"],
    3: ["2020-12-31", "2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31",
       "2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31",
       "2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"],
}


#: 🔴🔴 2026-09-22 效能查證：`resolve_strategy_holdings.resolve_holdings()`
#: 對每個策略都是先把「整條時間序列」（~6561個交易日）的遮罩做 AND，才切一天
#: 出來——但這裡只需要單一 as_of 那一天。原始寫法對6,679檔策略跑一季量測
#: 要 11+ 分鐘（先切「那一天」的那一列再AND，只要47.8秒，快14倍）。
#: 這裡用「先切列、再對列做AND」的等價但快得多的寫法，只在這支批次腳本內
#: 用，不動`resolve_strategy_holdings.py`共用模組本身（很多其他地方在用，
#: 不在沒有明確授權下改動已驗證的共用程式）。回傳結果跟原始`resolve_holdings`
#: 語意完全相同（同一組布林遮罩、同一個日期，只是計算順序不同）。
def _fast_resolve_w0(md: MarketData, idx: pd.DataFrame, uids: list[str], as_of: str,
                     row_cache: dict) -> tuple[dict[str, float], int]:
    ts = pd.Timestamp(as_of)

    def mask_row(cond):
        key = (cond["field"], cond["name"])
        if key in row_cache:
            return row_cache[key]
        m = md.get_mask(cond)
        if m is None:
            row_cache[key] = None
            return None
        valid = m.index[m.index <= ts]
        row = m.loc[valid.max()] if len(valid) else None
        row_cache[key] = row
        return row

    n_strat = len(uids)
    weights: dict[str, float] = {}
    n_empty = 0
    for uid in uids:
        row = idx.loc[uid]
        r1 = mask_row(_q_band_condition(row.F1_factor, row.F1_band, row.F1_nbands))
        if r1 is None:
            n_empty += 1
            continue
        combined = r1
        if not row.F2_empty:
            r2 = mask_row(_q_band_condition(row.F2_factor, row.F2_band, row.F2_nbands))
            if r2 is None:
                n_empty += 1
                continue
            combined = combined & r2
        if pd.notna(row.C_rule):
            r3 = mask_row(_c_condition(row.C_source, row.C_rule))
            if r3 is None:
                n_empty += 1
                continue
            combined = combined & r3
        if row.V == "v1":
            vm = md.get_v_mask()
            vvalid = vm.index[vm.index <= ts]
            vrow = vm.loc[vvalid.max()] if len(vvalid) else None
            if vrow is None:
                n_empty += 1
                continue
            combined = combined & vrow
        syms = combined[combined].index.tolist()
        if not syms:
            n_empty += 1
            continue
        w = (1.0 / n_strat) / len(syms)
        for s in syms:
            weights[s] = weights.get(s, 0.0) + w
    return weights, n_empty


def main():
    tree_uids = pd.read_csv(CLUSTER_CSV)["strategy_uid"].tolist()
    print(f"B_all 候選池（跟window4共用同一份universe，anchored scheme確認過相同）：{len(tree_uids)} 檔")

    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    idx = idx.loc[idx.index.intersection(tree_uids)]

    print(">> 載入 TW MarketData ...")
    md = MarketData("TW")
    md_map = {"TW": md}

    rows = []
    for window_no, quarters in WINDOW_QUARTERS.items():
        print(f"\n=== window{window_no}（OOS {quarters[1]}~{quarters[-1]}）===")
        for i in range(len(quarters) - 1):
            as_of, end = quarters[i], quarters[i + 1]
            # row_cache 每季重開一個新的——它快取的是「某條件在某天的那一列」，
            # 換一天就失效；底層真正貴的部分（md.get_mask() 的完整時間序列遮罩）
            # 存在 md._mask_cache，那個才是跨季持續有效、真正省時間的快取。
            weights, n_empty = _fast_resolve_w0(md, idx, tree_uids, as_of, row_cache={})
            n_unique = len(weights)
            res = measure(md_map, weights, as_of, end)
            ret = res["portfolio_realized_return"]
            print(f"  {as_of}->{end}：{n_unique}檔　股票層報酬={ret:+.4%}")
            rows.append({"window_no": window_no, "as_of": as_of, "end": end,
                        "n_unique_stocks": n_unique, "n_empty_strategies": n_empty,
                        "stock_level_return": ret})

    df = pd.DataFrame(rows)
    for w in WINDOW_QUARTERS:
        sub = df[df.window_no == w]
        cum = float((1 + sub["stock_level_return"]).prod() - 1)
        n_years = len(sub) / 4
        cagr = (1 + cum) ** (1 / n_years) - 1
        print(f"\nwindow{w} 累積={cum:+.2%}  年化CAGR={cagr:+.2%}")

    out = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "ball_stock_level_windows123.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out}")


if __name__ == "__main__":
    main()
