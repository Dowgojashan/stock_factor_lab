# -*- coding: utf-8 -*-
"""D65延伸item4（2026-09-22，使用者要求）：換其他window（1~3）測，看台積電排除
模式是window4（2024-2025 AI行情）特有的，還是長期存在的結構特徵。

window定義（`walkforward_members.parquet`，TW／scheme E／A_hrp／legacy／equal）：
  window1 OOS 2015-01~2017-12
  window2 OOS 2018-01~2020-12
  window3 OOS 2021-01~2023-12
  window4 OOS 2024-01~2025-12（已查過，8季累計12.91%，7/8季完全掛零）

用同一套 F1/F2/C/V 機制、同一組7,128檔候選池（因子規則本身不隨window改變，
只是評估的as_of日期不同），逐季（每個OOS期間的季底）檢查台積電選中比例。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from database import Database  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import (CANDIDATE_INDEX_PATH, _c_condition,  # noqa: E402
                                       _q_band_condition)

TSMC = "2330"
WINDOW_QUARTERS = {
    1: ["2015-03-31", "2015-06-30", "2015-09-30", "2015-12-31",
       "2016-03-31", "2016-06-30", "2016-09-30", "2016-12-31",
       "2017-03-31", "2017-06-30", "2017-09-30", "2017-12-31"],
    2: ["2018-03-31", "2018-06-30", "2018-09-30", "2018-12-31",
       "2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31",
       "2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"],
    3: ["2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31",
       "2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31",
       "2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"],
}


def asof_bool(mask: pd.DataFrame, as_of: str, symbol: str) -> bool | None:
    if mask is None or symbol not in mask.columns:
        return None
    ts = pd.Timestamp(as_of)
    valid = mask.index[mask.index <= ts]
    if len(valid) == 0:
        return None
    return bool(mask.loc[valid.max(), symbol])


def strategy_selects_tsmc(md: MarketData, row: pd.Series, as_of: str) -> bool | None:
    f1_mask = md.get_mask(_q_band_condition(row["F1_factor"], row["F1_band"], row["F1_nbands"]))
    f1 = asof_bool(f1_mask, as_of, TSMC)
    if f1 is not True:
        return f1
    if not row["F2_empty"]:
        f2_mask = md.get_mask(_q_band_condition(row["F2_factor"], row["F2_band"], row["F2_nbands"]))
        f2 = asof_bool(f2_mask, as_of, TSMC)
        if f2 is not True:
            return f2
    if pd.notna(row["C_rule"]):
        c_mask = md.get_mask(_c_condition(row["C_source"], row["C_rule"]))
        c = asof_bool(c_mask, as_of, TSMC)
        if c is not True:
            return c
    if row["V"] == "v1":
        v_mask = md.get_v_mask()
        v = asof_bool(v_mask, as_of, TSMC)
        if v is not True:
            return v
    return True


def fetch_mcap_weight_series(conn) -> pd.Series:
    q = """SELECT s.date, c.company_symbol, s.market_capital
           FROM stock s JOIN company c ON s.company_id=c.id
           WHERE c.exchange_name IN ('TWSE') AND s.market_capital IS NOT NULL
             AND s.date >= '2014-06-01'"""
    d = pd.read_sql(q, conn)
    d["date"] = pd.to_datetime(d["date"])
    wide = d.pivot_table(index="date", columns="company_symbol", values="market_capital", aggfunc="last")
    w = wide[TSMC] / wide.sum(axis=1)
    return w.sort_index()


def asof_value(series: pd.Series, d: str) -> float:
    ts = pd.Timestamp(d)
    avail = series.index[series.index <= ts]
    return float(series.loc[avail.max()]) if len(avail) else float("nan")


def main():
    idx_full = pd.read_parquet(CANDIDATE_INDEX_PATH)
    idx = idx_full[idx_full["market"] == "TW"].set_index("strategy_uid")
    print(f"全部台股候選策略數：{len(idx)}")

    print(">> 載入 TW MarketData ...")
    md = MarketData("TW")
    db = Database("TW")
    conn = db.create_connection()
    w_series = fetch_mcap_weight_series(conn)

    rows = []
    for window_no, dates in WINDOW_QUARTERS.items():
        ever_selected: set[str] = set()
        print(f"\n=== window{window_no}（OOS {dates[0]}~{dates[-1]}）===")
        for d in dates:
            n_sel = 0
            for uid, row in idx.iterrows():
                if strategy_selects_tsmc(md, row, d) is True:
                    n_sel += 1
                    ever_selected.add(uid)
            w = asof_value(w_series, d)
            print(f"  {d}：{n_sel}/{len(idx)}（{n_sel/len(idx):.2%}）選中　"
                 f"｜台積電指數權重={w:.2%}" if pd.notna(w) else f"  {d}：{n_sel}/{len(idx)}（{n_sel/len(idx):.2%}）選中")
            rows.append({"window_no": window_no, "as_of": d, "n_pool": len(idx),
                        "n_selected": n_sel, "pct_selected": n_sel / len(idx),
                        "tsmc_index_weight": w})
        print(f"  window{window_no} 全期累計「曾經選過」的不重複策略數："
             f"{len(ever_selected)}/{len(idx)}（{len(ever_selected)/len(idx):.2%}）")

    # window4（已查過）對照，直接寫死已知結果供比較，不重跑
    print("\n=== 對照：window4（2024-2025，已查過）===")
    print("  8季累計曾選過比例：12.91%（920/7128），7/8季完全掛零，"
         "唯一例外2025-09-30達9.95%")

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "tsmc_other_windows.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out}")


if __name__ == "__main__":
    main()
