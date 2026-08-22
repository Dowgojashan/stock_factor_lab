# -*- coding: utf-8 -*-
"""階段1 · 單趟掃描器（M2，整條線最重的工作）

輸入 ← `_frozen/stage0/candidate_index.parquet` + 每策略 3 個 parquet
        + `_frozen/stage1/mktcap_pct_{market}.parquet`
輸出 → `_frozen/stage1/strategy_scan.parquet`（逐策略原始指標）
        `_frozen/stage1/returns_monthly.parquet`（HRP 原料，long 格式）
        `_frozen/stage1/returns_meta.parquet`（hist_start/n_months，DD-03 共同窗的依據）
        `_frozen/stage1/annual_returns.parquet`（C-6，只存數字不貼牛熊標籤）
        `_frozen/stage1/scan_errors.parquet`

設計要點：
  DD-04  **跳過 position.parquet**（佔 77% 位元組但無獨佔資訊）→ 掃描量 19.8 GB → 4.7 GB
  DD-05  **一趟掃完算齊所有欄位**（I/O 是瓶頸，逐欄各掃一次會把成本乘上欄位數）
         依 job 目錄分片 checkpoint、可續跑；單策略失敗不中止全局
  斷循環  本階段**不得產出任何需要 regime 的欄位**；annual_returns 只存數字不貼標籤

用法：
    cd code
    python -m research.stage1_scan --limit 200      # 測速
    python -m research.stage1_scan                  # 全量（可續跑）
    python -m research.stage1_scan --resume
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, paths
from .stage1_mktcap import SMALLCAP_PCT_MAX

TRADING_DAYS = 252
SCAN_DIR = paths.STAGE1 / "scan"           # 分片 checkpoint
_MKTCAP: pd.DataFrame | None = None        # 每個 worker process 一份


# ============================================================ 單策略指標

def _from_return_table(rt: pd.DataFrame) -> dict:
    """月報酬矩陣 → 月序列 + 年度統計。

    起始期偵測：取**首個非零月**。已驗證與 stock_data 的日頻起點吻合
    （日頻起點 2004-03-31 → 月表首非零月 2004-04）。
    """
    mcols = [c for c in rt.columns if c != "YTD"]
    years = rt.index.astype(int).to_numpy()
    vals = rt[mcols].astype(float).to_numpy().reshape(-1)
    periods = pd.PeriodIndex(
        [f"{y}-{int(c):02d}" for y in years for c in mcols], freq="M")

    nz = np.nonzero(np.nan_to_num(vals) != 0)[0]
    if len(nz) == 0:
        return {"_empty": True}
    lo, hi = int(nz[0]), int(nz[-1])

    monthly = pd.Series(vals[lo:hi + 1], index=periods[lo:hi + 1], name="ret")

    # 年度報酬：YTD 欄已驗證 == 12 個月複利（差 0.00e+00），直接用
    ann = rt["YTD"].astype(float)
    ann.index = years
    ann = ann.loc[years[lo // 12]: years[hi // 12]]

    neg = monthly < 0
    # 最長連續虧損月數
    runs, cur = 0, 0
    for f in neg.to_numpy():
        cur = cur + 1 if f else 0
        runs = max(runs, cur)

    return {
        "_empty": False,
        "_monthly": monthly,
        "_annual": ann,
        "hist_start": str(periods[lo]),
        "hist_end": str(periods[hi]),
        "n_months": hi - lo + 1,
        "n_years": int(len(ann)),
        "worst_year": float(ann.min()),
        "neg_year_count": int((ann < 0).sum()),
        "annual_ret_std": float(ann.std()) if len(ann) > 1 else np.nan,
        "max_consec_loss_months": int(runs),
    }


def _from_stock_data(sd: pd.DataFrame, n_total_months: int) -> dict:
    """日頻淨值 + 換股日持股數 → 波動與可執行性。"""
    out: dict = {}

    nav = sd["cum_returns"].dropna()
    if len(nav) > 2:
        dret = nav.pct_change().dropna()
        out["ann_vol"] = float(dret.std() * np.sqrt(TRADING_DAYS))
        # 🆕 資料品質診斷：實測發現有策略單日 +3908%（NAV 22.58 → 905.0），
        #    30 檔持股的組合不可能有這種報酬 → 底層價格資料錯誤。
        #    留下極值供 stage1_marks 標記 data_glitch，不在此淘汰。
        out["max_daily_ret"] = float(dret.max())
        out["min_daily_ret"] = float(dret.min())
    else:
        out.update(ann_vol=np.nan, max_daily_ret=np.nan, min_daily_ret=np.nan)

    # backtest.py:344  company_count = (position != 0).sum(axis=1)
    #   → 只存在於 position 的換股日索引上，值可以是 0（＝當期空手）。
    #   ⚠️ 「某月完全沒有紀錄」語意曖昧：可能是空手，也可能是該月未重建部位。
    #      故**不猜**，拆成兩個語意明確的欄位，各自可解釋：
    #        empty_ratio     = 明確空手（count==0）的比例  ← 無歧義
    #        eval_month_ratio= 有部位紀錄的月份 / 總月份跨度 ← 換股頻率，非空手率
    cc = sd["company_count"].dropna()
    if len(cc):
        held = cc[cc > 0]
        out["holdings_median"] = float(held.median()) if len(held) else 0.0
        out["holdings_p10"] = float(held.quantile(0.10)) if len(held) else 0.0
        out["empty_ratio"] = float((cc == 0).mean())
        n_eval_months = cc.index.to_period("M").nunique()
        out["n_eval_months"] = int(n_eval_months)
        out["eval_month_ratio"] = (float(n_eval_months / n_total_months)
                                   if n_total_months else np.nan)
        out["n_holding_months"] = int(held.index.to_period("M").nunique())
    else:
        out.update(holdings_median=np.nan, holdings_p10=np.nan, empty_ratio=np.nan,
                   n_eval_months=0, eval_month_ratio=np.nan, n_holding_months=0)
    return out


def _from_trades(tr: pd.DataFrame, n_years: int, holdings_median: float | None) -> dict:
    """交易明細 → 可信度（關卡A）、換手率、規模依賴（C-5）。"""
    out = {"effective_n": np.nan, "top1_share": np.nan, "top20_cum_share": np.nan,
           "top1_stock": None, "rotation_score": np.nan, "rotation_n_years": 0,
           "turnover_ann": np.nan, "entries_per_year": np.nan, "size_tilt_pct": np.nan,
           "smallcap_share": np.nan, "size_n_obs": 0, "n_trades": int(len(tr))}
    if len(tr) == 0 or "stock_id" not in tr.columns:
        return out

    # --- 關卡A：沿用 analyze_batch 的定義，確保與第四章圖鑑數字一致 ---
    from analyze_batch import stock_cum_contrib
    contrib = stock_cum_contrib(tr)
    pos = contrib[contrib > 0]
    if len(pos) and pos.sum() > 0:
        shares = (pos / pos.sum()).sort_values(ascending=False)
        hhi = float((shares.to_numpy() ** 2).sum())
        out["effective_n"] = (1.0 / hhi) if hhi > 0 else np.nan
        out["top1_share"] = float(shares.iloc[0])
        out["top1_stock"] = str(shares.index[0])
        out["top20_cum_share"] = float(shares.cumsum().iloc[min(19, len(shares) - 1)])

    tr = tr.copy()
    tr["_year"] = pd.to_datetime(tr["entry_date"]).dt.year

    # --- rotation_score：年度前三大貢獻股在相鄰年度的 Jaccard，全期平均 ---
    # 年度貢獻用 position × return（該年對組合報酬的實際貢獻），而非跨年累積連乘。
    tr["_contrib"] = tr["position"].astype(float) * tr["return"].astype(float)
    top3 = {}
    for y, g in tr.groupby("_year"):
        s = g.groupby("stock_id")["_contrib"].sum().sort_values(ascending=False)
        s = s[s > 0]
        if len(s):
            top3[int(y)] = set(s.index[:3].astype(str))
    ys = sorted(top3)
    jac = [len(top3[a] & top3[b]) / len(top3[a] | top3[b])
           for a, b in zip(ys, ys[1:]) if b == a + 1 and (top3[a] | top3[b])]
    out["rotation_n_years"] = len(ys)
    # 有效交易年數 < 3 → NaN（避免誤殺短歷史策略，見 SDD DD-09）
    out["rotation_score"] = float(np.mean(jac)) if (jac and len(ys) >= 3) else np.nan

    # --- 年換手率 = 每年新進場檔數 / 中位持股數（倍數，非檔數）---
    if n_years > 0:
        out["entries_per_year"] = float(len(tr) / n_years)
        if holdings_median and holdings_median > 0:
            out["turnover_ann"] = float(len(tr) / n_years / holdings_median)

    # --- C-5 規模依賴：持股在當月的市值橫斷面分位 ---
    if _MKTCAP is not None:
        months = pd.to_datetime(tr["entry_date"]).dt.to_period("M")
        sids = tr["stock_id"].astype(str)
        cols = _MKTCAP.columns
        idx = _MKTCAP.index
        m_ok = months.isin(idx) & sids.isin(cols)
        if m_ok.any():
            vals = _MKTCAP.to_numpy()
            ri = idx.get_indexer(months[m_ok])
            ci = cols.get_indexer(sids[m_ok])
            got = vals[ri, ci]
            got = got[~np.isnan(got)]
            if len(got):
                out["size_tilt_pct"] = float(np.median(got))
                out["smallcap_share"] = float((got < SMALLCAP_PCT_MAX).mean())
                out["size_n_obs"] = int(len(got))
    return out


def scan_one(row: dict) -> tuple[dict | None, pd.Series | None, pd.Series | None, dict | None]:
    """掃一個策略。回傳 (指標, 月報酬, 年報酬, 錯誤)。單策略失敗不中止全局。"""
    uid, d = row["strategy_uid"], Path(row["artifacts_dir"])
    try:
        rt = _from_return_table(pd.read_parquet(d / "return_table.parquet"))
        if rt.get("_empty"):
            return None, None, None, {"strategy_uid": uid, "error": "return_table 全為零（無交易）"}

        rec = {"strategy_uid": uid, "market": row["market"]}
        rec.update({k: v for k, v in rt.items() if not k.startswith("_")})
        rec.update(_from_stock_data(pd.read_parquet(d / "stock_data.parquet"),
                                    rt["n_months"]))
        rec.update(_from_trades(pd.read_parquet(d / "trades.parquet"),
                                rt["n_years"], rec.get("holdings_median")))
        # 月頻年化波動：日頻版會被單日資料異常灌爆，留一個穩健版供對照
        mo = rt["_monthly"]
        rec["ann_vol_monthly"] = (float(mo.std() * np.sqrt(12))
                                  if len(mo) > 2 else np.nan)
        return rec, rt["_monthly"], rt["_annual"], None
    except Exception as e:
        return None, None, None, {"strategy_uid": uid, "error": f"{type(e).__name__}: {e}",
                                  "trace": traceback.format_exc(limit=3)}


# ============================================================ 分片與並行

def _init_worker(market: str):
    global _MKTCAP
    p = paths.STAGE1 / f"mktcap_pct_{market}.parquet"
    _MKTCAP = pd.read_parquet(p) if p.exists() else None


def _shard_path(job: str) -> Path:
    return SCAN_DIR / f"{job}.parquet"


def scan_job(job: str, rows: list[dict], market: str, workers: int, log) -> dict:
    """掃一個 job 目錄（分片單位）。已完成的分片直接跳過。"""
    t0 = time.time()
    recs, months, annuals, errs = [], [], [], []

    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers, initializer=_init_worker, initargs=(market,)) as pool:
            it = pool.imap_unordered(scan_one, rows, chunksize=32)
            for i, (rec, mo, an, err) in enumerate(it, 1):
                if rec: recs.append(rec); months.append((rec["strategy_uid"], mo)); annuals.append((rec["strategy_uid"], an))
                if err: errs.append(err)
                if i % 1000 == 0:
                    log(f"      {i}/{len(rows)}  ({time.time()-t0:.0f}s)")
    else:
        _init_worker(market)
        for i, r in enumerate(rows, 1):
            rec, mo, an, err = scan_one(r)
            if rec: recs.append(rec); months.append((rec["strategy_uid"], mo)); annuals.append((rec["strategy_uid"], an))
            if err: errs.append(err)
            if i % 200 == 0:
                log(f"      {i}/{len(rows)}  ({time.time()-t0:.0f}s)")

    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_parquet(_shard_path(job), compression="zstd", index=False)

    mo_df = pd.concat(
        [s.rename("ret").to_frame().assign(strategy_uid=u).reset_index().rename(columns={"index": "month"})
         for u, s in months if s is not None], ignore_index=True) if months else pd.DataFrame()
    mo_df.to_parquet(SCAN_DIR / f"{job}__months.parquet", compression="zstd", index=False)

    an_df = pd.concat(
        [s.rename("ret").to_frame().assign(strategy_uid=u).rename_axis("year").reset_index()
         for u, s in annuals if s is not None], ignore_index=True) if annuals else pd.DataFrame()
    an_df.to_parquet(SCAN_DIR / f"{job}__years.parquet", compression="zstd", index=False)

    if errs:
        pd.DataFrame(errs).to_parquet(SCAN_DIR / f"{job}__errors.parquet", index=False)

    dt = time.time() - t0
    log(f"   [{job}] {len(recs)} 成功 / {len(errs)} 失敗，{dt:.0f}s "
        f"({len(rows)/dt:.0f} 策略/秒)")
    return {"job": job, "ok": len(recs), "err": len(errs), "sec": dt}


# ============================================================ 主流程

def run(limit: int | None = None, workers: int | None = None,
        resume: bool = True, log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE0)                # 上游凍結未被改動才准跑
    idx = pd.read_parquet(paths.STAGE0 / "candidate_index.parquet")
    for m in C.MARKETS:
        if not (paths.STAGE1 / f"mktcap_pct_{m}.parquet").exists():
            raise FileNotFoundError(
                f"缺 mktcap_pct_{m}.parquet，請先跑 python -m research.stage1_mktcap")

    if limit:
        idx = idx.groupby("market", observed=True).head(limit // 2)
        log(f"⚠️ 測速模式：只掃 {len(idx)} 個策略")

    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    idx = idx.assign(_job=[Path(p).parent.name for p in idx["artifacts_dir"]])
    log(f"掃描 {len(idx)} 個策略 / {idx['_job'].nunique()} 個 job 分片 / {workers} workers\n")

    stats = []
    for market, mdf in idx.groupby("market", observed=True):
        for job, jdf in mdf.groupby("_job", observed=True):
            if resume and not limit and _shard_path(job).exists():
                log(f"   [{job}] 已完成，跳過（--no-resume 可強制重掃）")
                continue
            log(f"   [{job}] {len(jdf)} 個策略 …")
            rows = jdf[["strategy_uid", "market", "artifacts_dir"]].to_dict("records")
            stats.append(scan_job(job, rows, market, workers, log))

    return _merge(idx, limit is not None, log)


def _merge(idx: pd.DataFrame, is_probe: bool, log) -> pd.DataFrame:
    """合併分片 → 落地四份產物。"""
    def _cat(suffix: str) -> pd.DataFrame:
        fs = sorted(SCAN_DIR.glob(f"*{suffix}.parquet"))
        return (pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
                if fs else pd.DataFrame())

    shards = sorted(p for p in SCAN_DIR.glob("*.parquet")
                    if not any(t in p.name for t in ("__months", "__years", "__errors")))
    scan = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    months = _cat("__months")
    years = _cat("__years")
    errors = _cat("__errors")

    log(f"\n合併：scan {len(scan)} 列 / 月報酬 {len(months):,} 列 / "
        f"年報酬 {len(years):,} 列 / 錯誤 {len(errors)} 筆")

    if is_probe:
        log("（測速模式，不落地正式產物）")
        return scan

    months["month"] = months["month"].astype("period[M]")
    meta = (scan[["strategy_uid", "market", "hist_start", "hist_end", "n_months"]]
            .reset_index(drop=True))

    C.validate(months, C.RETURNS_MONTHLY)
    C.validate(meta, C.RETURNS_META)
    log("  ✓ returns_monthly / returns_meta 契約通過")

    outs = []
    for df, name in ((scan, "strategy_scan"), (months, "returns_monthly"),
                     (years, "annual_returns"), (meta, "returns_meta")):
        p = paths.STAGE1 / f"{name}.parquet"
        df.to_parquet(p, compression="zstd", index=False)
        outs.append(p)
        log(f"  → {name}.parquet  {len(df):,} 列, {p.stat().st_size/1024/1024:.1f} MB")
    if len(errors):
        errors.to_parquet(paths.STAGE1 / "scan_errors.parquet", index=False)

    freeze.write_manifest(
        "stage1_scan", paths.STAGE1,
        inputs=[paths.STAGE0 / "candidate_index.parquet"]
               + [paths.STAGE1 / f"mktcap_pct_{m}.parquet" for m in C.MARKETS],
        outputs=outs,
        params={"skip_position_parquet": True, "trading_days": TRADING_DAYS,
                "smallcap_pct_max": SMALLCAP_PCT_MAX},
        notes="DD-04 跳過 position.parquet；DD-05 單趟掃描；不含任何 regime 相關欄位（斷循環）",
    )
    return scan


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.stage1_scan")
    ap.add_argument("--limit", type=int, help="只掃前 N 個策略（測速用，不落地）")
    ap.add_argument("--workers", type=int, help="平行處理數（預設 CPU-1）")
    ap.add_argument("--no-resume", action="store_true", help="強制重掃所有分片")
    a = ap.parse_args(argv)

    t0 = time.time()
    scan = run(limit=a.limit, workers=a.workers, resume=not a.no_resume)
    print(f"\n總耗時 {time.time()-t0:.0f}s")
    _report(scan)
    return 0


def _report(scan: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 62)
    log("階段1 掃描 · 欄位健康度")
    log("=" * 62)
    cols = ["effective_n", "top1_share", "rotation_score", "empty_ratio",
            "eval_month_ratio", "holdings_median", "holdings_p10", "ann_vol",
            "ann_vol_monthly", "size_tilt_pct", "smallcap_share", "turnover_ann",
            "max_consec_loss_months", "worst_year", "max_daily_ret"]
    rows = []
    for c in cols:
        if c not in scan.columns:
            continue
        s = scan[c]
        rows.append({"欄位": c, "非空率": f"{s.notna().mean():.1%}",
                     "中位": f"{s.median():.3f}" if s.notna().any() else "-",
                     "min": f"{s.min():.3f}" if s.notna().any() else "-",
                     "max": f"{s.max():.3f}" if s.notna().any() else "-"})
    log(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
