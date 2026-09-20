# -*- coding: utf-8 -*-
"""切開「是HRP分群的問題，還是整個候選池本身的問題」（2026-09-22，使用者要求）。

前面幾次查證（legacy 30檔、放大到668檔）都是從 window 4 的 HRP 代表策略池抽樣，
放大比例也只是同一批 6 個群裡多挑代表——沒有真正跳出 HRP 分群的框架。這裡直接
對**全部 7,128 個台股候選策略**（`candidate_index.parquet` 裡 market=='TW' 的
全部列，完全不經過 HRP 分群/代表挑選這一層）做同一套 F1/F2/C/V 檢查，才能回答
「就算不分群、把所有策略都當候選，台積電還是選不到嗎」。
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

TSMC = "2330"
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


def main():
    idx_full = pd.read_parquet(CANDIDATE_INDEX_PATH)
    idx = idx_full[idx_full["market"] == "TW"].set_index("strategy_uid")
    print(f"全部台股候選策略數（未經HRP分群）：{len(idx)}")

    print(">> 載入 TW MarketData ...")
    md = MarketData("TW")

    ever_selected: set[str] = set()
    rows = []
    for d in CHECK_DATES:
        n_sel = 0
        selecting_uids = []
        for uid, row in idx.iterrows():
            sel = strategy_selects_tsmc(md, row, d)
            if sel is True:
                n_sel += 1
                ever_selected.add(uid)
                selecting_uids.append(uid)
        rows.append({"as_of": d, "n_pool": len(idx), "n_selected_tsmc": n_sel,
                    "pct": n_sel / len(idx)})
        print(f"  {d}：{n_sel}/{len(idx)}（{n_sel/len(idx):.2%}）選中台積電")
        if selecting_uids and n_sel <= 15:
            for u in selecting_uids:
                print(f"      {u}")

    print(f"\n8季累計「曾經選過台積電」的不重複策略數：{len(ever_selected)}/{len(idx)}"
         f"（{len(ever_selected)/len(idx):.2%}）")

    # 對照：window 4 HRP 各比例的「曾經選過」比例（前面已查過的結果，直接列出來比較）
    print("\n=== 對照：HRP window4 代表池 vs 全部候選池，「曾經選過台積電」比例 ===")
    print("  legacy(30)=10.00%  1%(67)=7.46%  3%(200)=10.50%  5%(334)=13.17%  10%(668)=13.62%")
    print(f"  全部候選池({len(idx)})={len(ever_selected)/len(idx):.2%}")

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "tsmc_full_pool_check.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out}")


if __name__ == "__main__":
    main()
