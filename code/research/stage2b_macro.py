# -*- coding: utf-8 -*-
"""階段 2b · 總經 → 月頻特徵表（W-02）

輸入 ← `_frozen/stage2/macro/macro_raw.parquet`（**collector 交付，見 MACRO_RAW 契約**）
輸出 → `_frozen/stage2/macro/macro_history_{market}.parquet`
        `_frozen/stage2/macro/zscore_params_{market}.json`   ← 凍結，out 期沿用
        `_frozen/stage2/macro/clock_bounds_{market}.json`    ← 凍結，out 期沿用

處理鏈（研究部 v9 階段2b 離線建置流程 1–4）：
    原始長表
      → ① 套用發布滯後（macro_spec.available_period）：把「期間」對齊到「哪個月底可得」
      → ② 攤平成月頻（季頻指標在下一筆發布前維持前值）
      → ③ z-score 標準化（**平均/標準差只用 in-sample 算並凍結**，防 look-ahead）
      → ④ 近 3 月平滑版並存（P4）
      → ⑤ 投資時鐘四格（成長 × 通膨，各以**歷史中位數**切，邊界凍結）

⚠️ **前視偏誤的兩道防線**：
  1. 滯後對齊（本模組 ①）——不使用該時點尚未發布的期間。
  2. 參數凍結（③⑤）——z-score 的 μ/σ 與四格邊界只用 2000–2025 算一次，
     out 期沿用同一組。否則「用未來資料決定標準化基準」本身就是洩漏。

用法：
    cd code
    python -m research.stage2b_macro --selftest      # 用合成資料驗證處理鏈
    python -m research.stage2b_macro                 # 需 collector 交付 macro_raw.parquet
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, paths
from .macro_spec import AXES, SPEC, available_period

#: 有效的 unit 值（selftest 用；必須是 contracts.MACRO_RAW 允許的其中一個）
_SELFTEST_UNIT = "percent_yoy"

MACRO_DIR = paths.STAGE2 / "macro"
IN_SAMPLE_START = "2000-01"
SMOOTH_WINDOW = 3


# ---------------------------------------------------------------- ① 滯後對齊

def _estimate_release_date(period_str: str, ind) -> pd.Timestamp:
    """沒有實測 release_date 時，用 macro_spec 的推估滯後量反推一個發布日估計值。

    定義：期間 P 的資料在「P 月底 + lag_months 個月」的月底前必定已發布
    （與 `available_period` 反向對應：available_period(ind, m) 回傳的是
    m 月底時最新可用的期間，故該期間的估計發布日 <= m 月底）。
    只在 release_date 缺漏、需要與其他有真實發布日的期間混合比較時使用。
    """
    per = pd.Period(period_str)
    end_month = per.asfreq("M", how="end")
    return (end_month + ind.lag_months).asfreq("M", how="end").to_timestamp(how="end")


def align_by_lag(raw: pd.DataFrame, market: str, months: pd.PeriodIndex) -> pd.DataFrame:
    """把長表依發布滯後攤平到月頻。

    對每個決策月 m，每個軸取「m 月底時最新可用的期間」對應的值。
    若原始表有 `release_date`（真 point-in-time），優先使用它；
    否則退回 macro_spec 的推估滯後量。

    ⚠️ 混合覆蓋率（同一指標裡部分期間有 release_date、部分沒有）：
       缺的那些期間**不會被排除**，改用推估滯後量補一個發布日估計值再一起比較，
       否則會被靜默排除在候選之外，讓「最新可用期間」的判斷少算了選項。
       （目前實際交付的資料是整個指標 0% 或 100%，不會混合；但程式邏輯不應該
       假設輸入永遠是這種乾淨情況。）
    """
    spec = SPEC[market]
    out = pd.DataFrame(index=months)
    src = pd.DataFrame(index=months, dtype=object)

    for axis in AXES:
        key = axis if axis in spec else f"{axis}_alt"
        if key not in spec:
            out[axis] = np.nan
            continue
        ind = spec[key]
        sub = raw[(raw.market == market) & (raw.indicator_key == key)]
        if sub.empty:
            out[axis] = np.nan
            continue

        by_period = dict(zip(sub["period"].astype(str), sub["value"].astype(float)))
        has_release = "release_date" in sub.columns and sub["release_date"].notna().any()
        if has_release:
            raw_rel = dict(zip(sub["period"].astype(str), pd.to_datetime(sub["release_date"])))
            # ⚠️ **不可無條件信任 release_date**——實測 US GDP 的 ALFRED 資料
            # 1998Q1~2014Q2（64 季、16 年）全部標記同一天 2014-09-26，那不是
            # BEA 當年真正的發布日，是 ALFRED 資料庫本身開始追蹤 vintage 的
            # 起點（vintage-tracking 起點偽裝成發布日）。若照單全收，會讓
            # 2014-09 之前的所有決策月都判定「這個指標還沒發布過任何值」，
            # 產生大片不該存在的 NaN（實測 US growth 176/312 個月缺值，
            # 且 2014-09 那個月一次冒出一堆同天「可得」的期間，`max()` 又
            # 沒有 tie-break，還會選到最舊的 1998Q1 而非最新一期）。
            # 合理性判準：與 macro_spec 推估的正常滯後相比，若實際落差超過
            # 180 天（半年緩衝，涵蓋各指標正常的季節性延遲），視為不可信、
            # 退回用推估值——這正是 vintage 追蹤起點的訊號，不是真發布延遲。
            plausible_slack = pd.Timedelta(days=180)
            rel = {}
            for p, d in raw_rel.items():
                est = _estimate_release_date(p, ind)
                if pd.isna(d) or d > est + plausible_slack:
                    rel[p] = est          # 缺值，或不合理地晚 → 退回推估
                else:
                    rel[p] = d            # 合理範圍內 → 相信實測值
        else:
            rel = None

        vals, srcs = [], []
        for m in months:
            if rel is not None:
                # point-in-time（實測 + 個別缺漏的推估補值）：取「發布日 <= 該月最後一天」中最新者。
                # ⚠️ tie-break：若多筆並列同一個最晚發布日（前述 vintage 起點問題的殘餘可能性、
                # 或資料源本來就對一批期間標同一天），優先取**期間本身較新**的那筆——
                # 「最近一次公布的是最新一期數字」在經濟上才合理，不能讓 dict 疊代順序決定答案。
                cutoff = m.to_timestamp(how="end")
                ok = [(p, d) for p, d in rel.items() if d <= cutoff]
                p = max(ok, key=lambda x: (x[1], pd.Period(x[0])))[0] if ok else None
            else:
                per = available_period(ind, m)
                p = None if per is None else str(per)
            vals.append(by_period.get(p, np.nan) if p else np.nan)
            srcs.append(p)
        out[axis] = vals
        src[axis] = srcs

    # 季頻指標在下一筆發布前維持前值（不是補未來，是「當時最新已知值」）。
    # src 也要跟著 ffill，否則被延續值的月份會顯示「無來源期間」，
    # 稽核者會誤以為那個月沒有引用任何資料——但它其實引用了更早那期的值。
    out = out.ffill()
    src = src.ffill()
    out.index.name = "month"
    out["src_periods"] = [
        "|".join(f"{a}={src[a].iloc[i]}" for a in AXES if a in src.columns)
        for i in range(len(months))
    ]
    return out


# ------------------------------------------------------- ③④ 標準化與平滑

def fit_zscore_params(df: pd.DataFrame) -> dict:
    """**只用 in-sample 算**，凍結後 out 期沿用。"""
    ins = df[df.index >= pd.Period(IN_SAMPLE_START, "M")]
    ins = ins[ins.index <= pd.Period(paths.IN_SAMPLE_END, "M")]
    return {a: {"mean": float(ins[a].mean()), "std": float(ins[a].std()),
                "n_obs": int(ins[a].notna().sum())}
            for a in AXES if a in ins.columns}


def apply_zscore(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    for a in AXES:
        p = params.get(a)
        if not p or not p["std"] or np.isnan(p["std"]):
            df[f"{a}_z"] = np.nan
        else:
            df[f"{a}_z"] = (df[a] - p["mean"]) / p["std"]
        df[f"{a}_z_s3"] = df[f"{a}_z"].rolling(SMOOTH_WINDOW, min_periods=1).mean()
    return df


# ------------------------------------------------------------ ⑤ 投資時鐘四格

def fit_clock_bounds(df: pd.DataFrame) -> dict:
    """成長、通膨各以 **in-sample 歷史中位數** 切高/低（P3），邊界凍結。"""
    ins = df[(df.index >= pd.Period(IN_SAMPLE_START, "M")) &
             (df.index <= pd.Period(paths.IN_SAMPLE_END, "M"))]
    return {"growth_median": float(ins["growth_z"].median()),
            "inflation_median": float(ins["inflation_z"].median()),
            "basis": "z-score 的 in-sample 中位數",
            "cells": {"高成長×低通膨": "復甦", "高成長×高通膨": "過熱",
                      "低成長×高通膨": "停滯性通膨", "低成長×低通膨": "衰退"}}


def apply_clock(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """成長×通膨 兩軸切四格。

    ⚠️ `>=` 對 NaN 一律回傳 False，所以 `~(nan >= x)` 會變成 True——
    NaN 月份原本該被排除，卻會被 `np.select` 的四個條件之一接住並貼上標籤
    （`default` 因此永遠用不到，是死碼）。故先算 `valid` 遮罩，缺值月份強制留白。
    """
    df = df.copy()
    valid = df["growth_z"].notna() & df["inflation_z"].notna()
    g_hi = df["growth_z"] >= bounds["growth_median"]
    i_hi = df["inflation_z"] >= bounds["inflation_median"]
    cell = np.select(
        [valid & g_hi & ~i_hi, valid & g_hi & i_hi,
         valid & ~g_hi & i_hi, valid & ~g_hi & ~i_hi],
        ["復甦", "過熱", "停滯性通膨", "衰退"], default=None)
    df["clock_cell"] = pd.Categorical(cell, categories=list(C.CLOCK_CELLS))
    return df


# ---------------------------------------------------------------- 主流程

def build_market(raw: pd.DataFrame, market: str, log=print) -> tuple[pd.DataFrame, dict, dict]:
    months = pd.period_range(IN_SAMPLE_START, paths.IN_SAMPLE_END, freq="M")
    df = align_by_lag(raw, market, months)
    log(f"  [{market}] 滯後對齊完成，{len(df)} 個月；"
        f"各軸非空率 " + ", ".join(f"{a}={df[a].notna().mean():.0%}" for a in AXES if a in df))

    zp = fit_zscore_params(df)
    df = apply_zscore(df, zp)
    cb = fit_clock_bounds(df)
    df = apply_clock(df, cb)

    df = df.reset_index()
    df.insert(0, "market", market)
    df["market"] = df["market"].astype("category")
    log(f"  [{market}] 四格分布: {df.clock_cell.value_counts().to_dict()}")
    return df, zp, cb


def run(raw_path=None, log=print) -> dict:
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = raw_path or (MACRO_DIR / "macro_raw.parquet")
    if not raw_path.exists():
        raise FileNotFoundError(
            f"找不到 {raw_path}\n"
            f"階段2b 需要 collector 交付總經原始資料（契約見 contracts.MACRO_RAW）。\n"
            f"目前資料庫中**沒有任何總經指標**（singleindicator 全為財報科目）。")

    raw = pd.read_parquet(raw_path)
    C.validate(raw, C.MACRO_RAW)
    outs = {}
    for m in C.MARKETS:
        df, zp, cb = build_market(raw, m, log)
        C.validate(df, C.MACRO_HISTORY)
        p = MACRO_DIR / f"macro_history_{m}.parquet"
        df.to_parquet(p, compression="zstd", index=False)
        (MACRO_DIR / f"zscore_params_{m}.json").write_text(
            json.dumps(zp, ensure_ascii=False, indent=2), encoding="utf-8")
        (MACRO_DIR / f"clock_bounds_{m}.json").write_text(
            json.dumps(cb, ensure_ascii=False, indent=2), encoding="utf-8")
        outs[m] = p
    freeze.write_manifest(
        "stage2b_macro", MACRO_DIR, inputs=[raw_path], outputs=list(outs.values()),
        params={"in_sample": f"{IN_SAMPLE_START}..{paths.IN_SAMPLE_END}",
                "smooth_window": SMOOTH_WINDOW, "clock_split": "in-sample 中位數"},
        notes="z-score 參數與四格邊界已凍結；out 期沿用不重算")
    return outs


# ---------------------------------------------------------------- 自我測試

def selftest(log=print) -> None:
    """用合成資料驗證整條處理鏈——在 collector 交付前就能確認邏輯正確。"""
    log("=" * 62)
    log("階段2b 處理鏈自我測試（合成資料）")
    log("=" * 62)
    rng = np.random.default_rng(0)
    rows = []
    for mkt in C.MARKETS:
        for key in ("growth", "inflation", "rate_level", "rate_direction"):
            ind = SPEC[mkt].get(key) or SPEC[mkt][f"{key}_alt"]
            if ind.freq == "Q":
                periods = pd.period_range("1999-01", "2025-12", freq="Q")
            else:
                periods = pd.period_range("1999-01", "2025-12", freq="M")
            for p in periods:
                rows.append({"market": mkt, "indicator_key": key, "period": str(p),
                             "value": float(rng.normal()), "unit": _SELFTEST_UNIT,
                             "release_date": None})
    raw = pd.DataFrame(rows)
    raw["market"] = raw["market"].astype("category")
    C.validate(raw, C.MACRO_RAW)
    log(f"合成原始表 {len(raw)} 列，通過 MACRO_RAW 契約")

    for mkt in C.MARKETS:
        df, zp, cb = build_market(raw, mkt, log)
        C.validate(df, C.MACRO_HISTORY)
        # 前視偏誤稽核：逐月檢查引用的期間都不晚於該月
        bad = 0
        for r in df.itertuples():
            for tok in str(r.src_periods).split("|"):
                if "=" not in tok:
                    continue
                _, per = tok.split("=", 1)
                if per in ("None", "nan", ""):
                    continue
                end = pd.Period(per).asfreq("M", how="end")
                if end > r.month:
                    bad += 1
        log(f"  [{mkt}] ✓ 契約通過｜前視稽核 {len(df)} 個月，違規 {bad} 筆")
        assert bad == 0, "偵測到前視偏誤：引用了決策月之後的資料期間"
        assert abs(df[df.month <= pd.Period(paths.IN_SAMPLE_END, "M")]["growth_z"].mean()) < 0.05, \
            "in-sample z-score 平均應接近 0"
    log("\n✅ 處理鏈驗證通過——等 collector 交付 macro_raw.parquet 即可正式建置")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="research.stage2b_macro")
    ap.add_argument("--selftest", action="store_true", help="用合成資料驗證處理鏈")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    outs = run()
    print(f"產出 {len(outs)} 份 macro_history")
    return 0


if __name__ == "__main__":
    sys.exit(main())
