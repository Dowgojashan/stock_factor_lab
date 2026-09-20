# -*- coding: utf-8 -*-
"""D65延伸item5（2026-09-22，使用者要求）：換美股候選池測，看台股這個「評價溢價股
結構性排除」現象是台股極端集中度造成的特例，還是價值因子普遍會有的現象。

選 NVIDIA（NVDA）當對照——2024-2025 AI行情美股的代表性個股，跟台積電同一個
主題敘事（AI/半導體），評價同樣明顯偏貴。跟老師9/15會議記錄「美股沒有這個問題」
的既有觀察對照驗證。
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

SYMBOL = "NVDA"
CHECK_DATES = ["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
              "2025-03-31", "2025-06-30", "2025-09-30"]


def asof_bool(mask: pd.DataFrame, as_of: str, symbol: str) -> bool | None:
    if mask is None or symbol not in mask.columns:
        return None
    ts = pd.Timestamp(as_of)
    valid = mask.index[mask.index <= ts]
    if len(valid) == 0:
        return None
    return bool(mask.loc[valid.max(), symbol])


def strategy_selects(md: MarketData, row: pd.Series, as_of: str, symbol: str) -> bool | None:
    f1_mask = md.get_mask(_q_band_condition(row["F1_factor"], row["F1_band"], row["F1_nbands"]))
    f1 = asof_bool(f1_mask, as_of, symbol)
    if f1 is not True:
        return f1
    if not row["F2_empty"]:
        f2_mask = md.get_mask(_q_band_condition(row["F2_factor"], row["F2_band"], row["F2_nbands"]))
        f2 = asof_bool(f2_mask, as_of, symbol)
        if f2 is not True:
            return f2
    if pd.notna(row["C_rule"]):
        c_mask = md.get_mask(_c_condition(row["C_source"], row["C_rule"]))
        c = asof_bool(c_mask, as_of, symbol)
        if c is not True:
            return c
    if row["V"] == "v1":
        v_mask = md.get_v_mask()
        v = asof_bool(v_mask, as_of, symbol)
        if v is not True:
            return v
    return True


def fetch_mcap_weight_series(conn, symbol: str) -> pd.Series | None:
    q = f"""SELECT s.date, c.company_symbol, s.market_capital
            FROM stock s JOIN company c ON s.company_id=c.id
            WHERE c.exchange_name NOT IN ('TWSE', 'TPEx') AND s.market_capital IS NOT NULL
              AND s.date >= '2023-06-01'"""
    d = pd.read_sql(q, conn)
    if symbol not in d["company_symbol"].unique():
        return None
    d["date"] = pd.to_datetime(d["date"])
    wide = d.pivot_table(index="date", columns="company_symbol", values="market_capital", aggfunc="last")
    w = wide[symbol] / wide.sum(axis=1)
    return w.sort_index()


def asof_value(series: pd.Series | None, d: str) -> float:
    if series is None:
        return float("nan")
    ts = pd.Timestamp(d)
    avail = series.index[series.index <= ts]
    return float(series.loc[avail.max()]) if len(avail) else float("nan")


def main():
    idx_full = pd.read_parquet(CANDIDATE_INDEX_PATH)
    idx = idx_full[idx_full["market"] == "US"].set_index("strategy_uid")
    print(f"全部美股候選策略數：{len(idx)}")

    print(">> 載入 US MarketData ...")
    md = MarketData("US")
    print(f"   NVDA 是否在因子宇宙裡：{SYMBOL in (md.common if hasattr(md, 'common') else [])}")

    db = Database("US")
    conn = db.create_connection()
    w_series = fetch_mcap_weight_series(conn, SYMBOL)
    if w_series is None:
        print("⚠️ 資料庫market_capital表查不到NVDA的市值權重序列（表結構或篩選條件跟TW不同），"
             "略過權重欄位，仍照跑選中率查證。")

    ever_selected: set[str] = set()
    rows = []
    for d in CHECK_DATES:
        n_sel = 0
        for uid, row in idx.iterrows():
            sel = strategy_selects(md, row, d, SYMBOL)
            if sel is True:
                n_sel += 1
                ever_selected.add(uid)
        w = asof_value(w_series, d)
        w_str = f"{w:.2%}" if pd.notna(w) else "查不到"
        print(f"  {d}：{n_sel}/{len(idx)}（{n_sel/len(idx):.2%}）選中　｜NVDA市值權重={w_str}")
        rows.append({"symbol": SYMBOL, "as_of": d, "n_pool": len(idx),
                    "n_selected": n_sel, "pct_selected": n_sel / len(idx), "index_weight": w})

    print(f"\n8季累計「曾經選過NVDA」的不重複策略數：{len(ever_selected)}/{len(idx)}"
         f"（{len(ever_selected)/len(idx):.2%}）")
    print("\n=== 對照：台股同期（window4）===")
    print("  台積電8季累計曾選過比例：12.91%，7/8季完全掛零")

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "nvda_us_pool_check.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out}")


if __name__ == "__main__":
    main()
