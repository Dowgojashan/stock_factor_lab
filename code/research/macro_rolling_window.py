# -*- coding: utf-8 -*-
"""H-18② · 總經訊號的滾動窗版本（開發待辦追蹤.md 軸線二·總經）

`stage2b_macro.py` 現有做法：z-score 的平均/標準差、投資時鐘四格的成長/通膨
中位數邊界，是**拿2000-2025全樣本算一次凍結**的（`fit_zscore_params`／
`fit_clock_bounds`）。程式註解自己寫「防look-ahead」，但那只防住了「IS/OOS
用不同參數導致標準不一致」，**沒防住「早期月份的分類，其實用到了後期資料」
這件事**——例如 2005 年那個月的 z-score，其實悄悄用到了 2006-2025 年的資料
去算標準化參數，這是2026-08-31討論H-17/H-18時查證出來、先前沒被點出的真實
方法論缺口。

本模組建立**寫實版本**：每個月的 z-score 跟投資時鐘四格邊界，只用「到那個月
為止過去5年（60個月）」的資料算——使用者2026-09-01定案的滾動窗長度。**嚴格
要求滿60個月才開始分類**（`min_periods=window`，不是「有多少算多少」的擴張窗
退讓），避免早期月份因樣本點太少而產生的雜訊，跟「用了未來資料」這個真正要
量測的效應混在一起分不清楚。

輸出兩張表：
  `macro_history_rolling_{market}.parquet`  滾動窗版本的完整月頻特徵表（跟
      `MACRO_HISTORY` schema同構，但獨立存放，不覆寫主線`macro_history_{market}.
      parquet`——跟H-11「獨立目錄、不共用檔名」的做法一致）
  `macro_clock_comparison.parquet`          全樣本凍結版 vs 滾動窗版的逐月
      clock_cell對照，兩者的**差距本身就是H-18要的論文內容**

用法：
    cd code
    python -m research.macro_rolling_window
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, paths
from .macro_spec import AXES
from .stage2b_macro import IN_SAMPLE_START, MACRO_DIR, SMOOTH_WINDOW, align_by_lag

#: 使用者2026-09-01定案：5年滾動窗
ROLLING_WINDOW_MONTHS = 60

OUT_DIR = paths.STAGE2 / "macro_rolling"


def apply_zscore_rolling(df: pd.DataFrame, window: int = ROLLING_WINDOW_MONTHS) -> pd.DataFrame:
    """跟 stage2b_macro.apply_zscore 對應，但 mean/std 改用滾動窗——每個月只拿
    過去window個月（含當月）的資料算，不是全樣本凍結參數。`min_periods=window`
    嚴格要求滿窗才給值，避免早期月份因樣本點太少產生額外雜訊。
    """
    df = df.copy()
    for a in AXES:
        if a not in df.columns:
            continue
        roll_mean = df[a].rolling(window, min_periods=window).mean()
        roll_std = df[a].rolling(window, min_periods=window).std()
        df[f"{a}_z"] = (df[a] - roll_mean) / roll_std
        df[f"{a}_z_s3"] = df[f"{a}_z"].rolling(SMOOTH_WINDOW, min_periods=1).mean()
    return df


def apply_clock_rolling(df: pd.DataFrame, window: int = ROLLING_WINDOW_MONTHS) -> pd.DataFrame:
    """跟 stage2b_macro.apply_clock 對應，但成長/通膨中位數邊界改用滾動窗——
    每個月的「高/低」判準是「相對於過去window個月自己的中位數」，不是全樣本
    凍結的中位數。判準邏輯（`>=`中位數為高）跟原版完全一致，只換了中位數的算法。
    """
    df = df.copy()
    roll_g_med = df["growth_z"].rolling(window, min_periods=window).median()
    roll_i_med = df["inflation_z"].rolling(window, min_periods=window).median()
    valid = (df["growth_z"].notna() & df["inflation_z"].notna()
            & roll_g_med.notna() & roll_i_med.notna())
    g_hi = df["growth_z"] >= roll_g_med
    i_hi = df["inflation_z"] >= roll_i_med
    cell = np.select(
        [valid & g_hi & ~i_hi, valid & g_hi & i_hi,
         valid & ~g_hi & i_hi, valid & ~g_hi & ~i_hi],
        ["復甦", "過熱", "停滯性通膨", "衰退"], default=None)
    df["clock_cell"] = pd.Categorical(cell, categories=list(C.CLOCK_CELLS))
    return df


def build_market_rolling(raw: pd.DataFrame, market: str, window: int = ROLLING_WINDOW_MONTHS,
                         log=print) -> pd.DataFrame:
    months = pd.period_range(IN_SAMPLE_START, paths.IN_SAMPLE_END, freq="M")
    df = align_by_lag(raw, market, months)   # 滯後對齊本身沒有look-ahead疑慮，沿用不改
    df = apply_zscore_rolling(df, window)
    df = apply_clock_rolling(df, window)

    df = df.reset_index()
    df.insert(0, "market", market)
    df["market"] = df["market"].astype("category")
    n_classified = int(df["clock_cell"].notna().sum())
    log(f"  [{market}] 滾動窗版：{len(df)}個月中{n_classified}個月有分類"
        f"（前{window-1}個月因不滿{window}個月歷史無法分類，見模組docstring）")
    return df


def compare_with_frozen(rolling: pd.DataFrame, market: str, log=print) -> pd.DataFrame:
    frozen = pd.read_parquet(MACRO_DIR / f"macro_history_{market}.parquet")[
        ["market", "month", "clock_cell"]].rename(columns={"clock_cell": "frozen_clock_cell"})
    roll = rolling[["market", "month", "clock_cell"]].rename(
        columns={"clock_cell": "rolling_clock_cell"})
    merged = frozen.merge(roll, on=["market", "month"], how="inner", validate="one_to_one")
    both_valid = merged["frozen_clock_cell"].notna() & merged["rolling_clock_cell"].notna()
    # ⚠️ `np.where(cond, bool_series, None)` 混了 True/False/None 會產生 object dtype，
    # 不符schema宣告的bool——改用pandas的nullable "boolean" dtype（大寫，非內建bool），
    # 才能同時裝布林值跟缺值（2026-09-01開發時實測踩到的真實bug）。
    raw_match = (merged["frozen_clock_cell"] == merged["rolling_clock_cell"])
    merged["match"] = raw_match.where(both_valid, other=pd.NA).astype("boolean")
    n_comparable = int(both_valid.sum())
    n_match = int((merged.loc[both_valid, "match"]).sum())
    log(f"  [{market}] 可比較{n_comparable}個月，其中分類相同{n_match}個"
        f"（{n_match/n_comparable:.1%}）｜不同{n_comparable-n_match}個"
        f"（{(n_comparable-n_match)/n_comparable:.1%}）")
    return merged


def run(window: int = ROLLING_WINDOW_MONTHS, log=print) -> dict[str, pd.DataFrame]:
    freeze.verify_inputs(paths.STAGE2 / "macro")
    raw_path = MACRO_DIR / "macro_raw.parquet"
    raw = pd.read_parquet(raw_path)
    C.validate(raw, C.MACRO_RAW)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rolling_tables, comparisons = {}, []
    for m in C.MARKETS:
        df = build_market_rolling(raw, m, window, log)
        C.validate(df, C.MACRO_HISTORY)
        p = OUT_DIR / f"macro_history_rolling_{m}.parquet"
        df.to_parquet(p, compression="zstd", index=False)
        rolling_tables[m] = df
        comparisons.append(compare_with_frozen(df, m, log))

    comp = pd.concat(comparisons, ignore_index=True)
    comp["market"] = comp["market"].astype("category")
    comp["frozen_clock_cell"] = comp["frozen_clock_cell"].astype("category")
    comp["rolling_clock_cell"] = comp["rolling_clock_cell"].astype("category")
    C.validate(comp, C.MACRO_CLOCK_COMPARISON, strict_columns=True)
    log("✓ macro_clock_comparison 契約通過")
    p_comp = OUT_DIR / "macro_clock_comparison.parquet"
    comp.to_parquet(p_comp, compression="zstd", index=False)

    freeze.write_manifest(
        "macro_rolling_window", OUT_DIR,
        inputs=[MACRO_DIR / "macro_raw.parquet"]
              + [MACRO_DIR / f"macro_history_{m}.parquet" for m in C.MARKETS],
        outputs=[OUT_DIR / f"macro_history_rolling_{m}.parquet" for m in C.MARKETS] + [p_comp],
        params={"window_months": window},
        notes="H-18②（軸線二）：總經z-score/投資時鐘四格的滾動窗（無look-ahead）版本，"
              "跟既有全樣本凍結版比較差距。獨立於_frozen/stage2/macro/，不覆寫主線。",
    )
    log(f"→ {OUT_DIR}")
    return {"rolling": rolling_tables, "comparison": comp}


def _report(result: dict, log=print) -> None:
    comp = result["comparison"]
    log("\n" + "=" * 78)
    log("H-18② · 總經滾動窗 vs 全樣本凍結版 對照摘要")
    log("=" * 78)
    for m, g in comp.groupby("market", observed=True):
        valid = g[g["match"].notna()]
        if len(valid) == 0:
            continue
        n_match = int(valid["match"].sum())
        log(f"\n[{m}] 可比較{len(valid)}個月｜相同{n_match}({n_match/len(valid):.1%})"
           f"｜不同{len(valid)-n_match}({(len(valid)-n_match)/len(valid):.1%})")
        diff = valid[~valid["match"]]
        if len(diff):
            log("  分類不同的月份範例（前5筆）：")
            log(diff[["month", "frozen_clock_cell", "rolling_clock_cell"]].head(5).to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.macro_rolling_window")
    ap.add_argument("--window", type=int, default=ROLLING_WINDOW_MONTHS)
    a = ap.parse_args(argv)
    result = run(window=a.window)
    _report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
