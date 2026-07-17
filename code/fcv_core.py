# -*- coding: utf-8 -*-
"""
FCV 核心（市場泛化）：載入一次 → 跑多個 spec。供 sweep_driver 驅動，US/TW 共用。

設計要點（見 A4 開發計畫）：
- **載入共享**：`MarketData` 載一次 Data(market)，其 field frame 物件穩定 →
  condition_factory 的 q_band rank 快取（以 frame id 為 key）跨 spec 自動重用，每因子只排名一次。
- **宇宙篩選在讀取層**：database.py 的 market-gated `_universe_clause`（US→russell3000、TW→不篩），
  故本檔天然市場泛化；這裡只做 `univ ∩ price` 並在建遮罩前限縮財報 frame（filter-before-rank）。
- **串流分批**：不同時 materialize 全部遮罩/報告，峰值 O(1 batch)（全量會 OOM，見 A4 §10）。
- **對稱去重**：F1×F2 只產無序對（EPS__ROE 與 ROE__EPS 交集相同）→ 省 45%。
- **sharpe_ann**：落地時就從 return_table 月報酬重算正確年化 Sharpe（daily_sharpe 是壞值，見 A4 §10）。
"""
import os
import sys
import time
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# --- bootstrap：config.ini 為相對路徑 ../config.ini → cwd 必須是 code/ ---
_ROOT = str(Path(__file__).resolve().parent.parent)
_HERE = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)          # get_data / backtest / combinations 在 ROOT
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)          # condition_factory / io_persistence 在 code/
if Path.cwd() != Path(_HERE):
    os.chdir(_HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from get_data import Data
from condition_factory import build_conditions
from combinations import sim_conditions
from io_persistence import export_report_collection_artifacts

# ==================== 常數 ====================
MIN_TRADES = 5
ENTRY_THRESH = 1e-9
ART_DIR = Path("results_artifacts")

# 用來界定「有因子的宇宙」。TW 缺的欄位會自動略過。
FACTOR_FIELDS = ["report:ROE", "report:EPS", "report:FCF_P",
                 "report:DEBTRATIO", "report:REVENUE", "report:PE"]

# 各市場回測窗起點（A4 定案：台美同為 2000–2026）
MARKET_START = {"US": "2000-01-01", "TW": "2000-01-01"}


# ==================== 對齊工具 ====================
def align_to_trading_days(df, price_index):
    if df is None or getattr(df, "empty", True):
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.loc[df.index.notna()]
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df.reindex(price_index, method="ffill")


def quick_stats_trades(pos):
    """近似交易次數（有任一股票變動的天數）；numpy 版，比 .diff() 快很多。"""
    v = pos.values
    if v.shape[0] < 2:
        return 0
    return int((v[1:] != v[:-1]).any(axis=1).sum())


def ensure_dtindex_cols(df, cols):
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df.reindex(columns=cols, fill_value=False)


# ==================== 市場資料（載入一次、跨 spec 共用）====================
class MarketData:
    """載入某市場的價量/財報一次，供多個 spec 重複使用。

    宇宙 = 有因子 ∩ 有價格（US 的 russell3000 圈定已在 database.py 完成）。
    財報 frame 在 get_field() 就限縮到宇宙欄位 → q_band 排名母體正確（filter-before-rank）。
    """

    def __init__(self, market, start=None, univ_limit=0, verbose=True):
        self.market = str(market).upper()
        t0 = time.time()
        if verbose:
            print(f">> 載入 Data(market='{self.market}') ...", flush=True)
        self.data = Data(market=self.market)

        univ = set()
        for f in FACTOR_FIELDS:
            try:
                d = self.data.get(f)
            except Exception:
                continue                      # 該市場沒有這個因子（如 TW 無 DEBTRATIO）
            if d is not None and hasattr(d, "columns"):
                univ |= set(d.columns)

        close = self.data.get("price:close")
        common = [c for c in close.columns if c in univ]
        if univ_limit and univ_limit > 0:
            common = common[:univ_limit]       # 冒煙用
        self.common = common

        start = start or MARKET_START.get(self.market, "2000-01-01")
        START = pd.Timestamp(start)
        for k in list(self.data.all_price_dict.keys()):
            df = self.data.all_price_dict[k].reindex(columns=common)
            df.index = pd.to_datetime(df.index)
            self.data.all_price_dict[k] = df[df.index >= START]

        idx = self.data.get("price:close").index
        self.price_index = idx if isinstance(idx, pd.DatetimeIndex) else pd.to_datetime(idx)

        self._field_cache = {}
        self._mask_cache = {}
        self._v_mask = None
        if verbose:
            print(f"   因子宇宙={len(univ)}｜回測宇宙={len(common)}｜起點>={START.date()}｜"
                  f"交易日={len(self.price_index)} "
                  f"({self.price_index.min().date()}~{self.price_index.max().date()})｜"
                  f"耗時 {time.time()-t0:.0f}s", flush=True)

    def get_field(self, field):
        """取得對齊交易日、且（財報）已限縮到宇宙欄位的 frame。跨 spec 快取。"""
        if field in self._field_cache:
            return self._field_cache[field]
        try:
            raw = self.data.get(field)
        except Exception:
            raw = None
        if raw is None or not hasattr(raw, "columns"):
            self._field_cache[field] = None
            return None
        if field.startswith("report:"):
            f = align_to_trading_days(raw, self.price_index)
            f = f.reindex(columns=self.common)     # ★ filter-before-rank
        else:
            f = raw if raw.index.equals(self.price_index) else align_to_trading_days(raw, self.price_index)
        self._field_cache[field] = f
        return f

    def get_mask(self, cond):
        """建某條件的遮罩，跨 spec 快取（同名條件不重算）。"""
        key = (cond["field"], cond["name"])
        if key in self._mask_cache:
            return self._mask_cache[key]
        df = self.get_field(cond["field"])
        if df is None:
            return None
        try:
            m = cond["cond"](df)
        except Exception as e:
            warnings.warn(f"[get_mask] {cond['name']}: {e}")
            return None
        if not m.index.equals(self.price_index):
            m = m.reindex(self.price_index, method="ffill").fillna(False)
        self._mask_cache[key] = m
        return m

    def get_v_mask(self, pe_field="report:pe", window=4):
        """V 構面：PE 低於近 window 季均值但高於最低（估值濾網）。建一次、限縮到宇宙。"""
        if self._v_mask is not None:
            return self._v_mask
        pe_raw = self.data.get(pe_field)
        pe = pe_raw.sort_index()
        if not isinstance(pe.index, pd.DatetimeIndex):
            pe.index = pd.to_datetime(pe.index)
        pe_q = pe
        if len(pe.index) >= 10:
            gap = pe.index.to_series().diff().dropna().dt.days.median()
            if gap is not None and gap < 40:               # 日頻 → 壓成季
                try:
                    pe_q = pe.resample("QE").last()
                except Exception:
                    pe_q = pe.resample("Q").last()
        mean4 = pe_q.rolling(window, min_periods=window).mean()
        min4 = pe_q.rolling(window, min_periods=window).min()
        v = (pe_q < mean4) & (pe_q > min4)
        v = v.reindex(self.price_index, method="ffill").fillna(False)
        self._v_mask = v.reindex(columns=self.common, fill_value=False)
        return self._v_mask

    def release(self):
        self._field_cache.clear()
        self._mask_cache.clear()
        self._v_mask = None


# ==================== 串流展開（F/C/V，含對稱去重）====================
def stream_strategy_masks(P1, P3, md, dedup=True):
    """串流 yield (name, daily_mask)，不同時 materialize 全部。

    dedup=True：F1×F2 只產**無序對**（j>i）→ EPS__ROE 與 ROE__EPS 不重複，省 45%。
    """
    v_mask = md.get_v_mask()
    P1_masks = {c["name"]: md.get_mask(c) for c in P1}
    P3_masks = {c["name"]: md.get_mask(c) for c in P3}
    P3_items = [("None", None)] + [(k, v) for k, v in P3_masks.items() if v is not None]

    for i, p1 in enumerate(P1):
        m1 = P1_masks.get(p1["name"])
        if m1 is None:
            continue
        cands = [None]                                  # F2=None＝單因子
        for j, p2 in enumerate(P1):
            if p2["field"] == p1["field"]:
                continue                                # 同因子不配對
            if dedup and j < i:
                continue                                # ★ 只取無序對
            cands.append(p2)

        for p2 in cands:
            if p2 is None:
                base, p2name = m1, "None"
            else:
                m2 = P1_masks.get(p2["name"])
                if m2 is None:
                    continue
                base, p2name = (m1 & m2), p2["name"]
            for p3name, m3 in P3_items:
                cmask = base if m3 is None else (base & m3)
                key = f"{p1['name']}__{p2name}__{p3name}"
                yield f"{key}__v0", cmask
                yield f"{key}__v1", cmask & v_mask


# ==================== sharpe_ann（取代壞掉的 daily_sharpe）====================
def sharpe_ann_from_return_table(return_table):
    """從 return_table（逐年×逐月報酬）算正確年化 Sharpe：mean/std×√12。
    去掉起訖的結構性零（策略起訖前後未持倉）。daily_sharpe 是壞值，見 A4 §10。"""
    try:
        rt = pd.DataFrame(return_table).T
        rt.columns = [str(c) for c in rt.columns]
        cols = [c for c in (str(i) for i in range(1, 13)) if c in rt.columns]
        if not cols:
            return np.nan
        m = pd.Series(rt[cols].to_numpy(dtype=float).flatten()).dropna()
        nz = m[m != 0]
        if len(nz) == 0:
            return np.nan
        m = m.loc[nz.index[0]:nz.index[-1]]
        if len(m) < 3 or not np.isfinite(m.std()) or m.std() == 0:
            return np.nan
        return float(m.mean() / m.std() * np.sqrt(12))
    except Exception:
        return np.nan


# ==================== 跑一份 spec ====================
def done_marker(label, art_dir=ART_DIR):
    return Path(art_dir) / label / "_DONE"


def is_done(label, art_dir=ART_DIR):
    return done_marker(label, art_dir).exists()


def run_spec(md, spec, label, rebalance="M", batch_size=150, art_dir=ART_DIR,
             min_trades=MIN_TRADES, smoke_limit=None, dedup=True, verbose=True):
    """跑一份 spec：串流展開 → 分批回測 → 逐批落地 → 合併 stats → 寫 _DONE。

    回傳 dict(label, n_expanded, n_kept, n_backtested, seconds)。
    """
    t0 = time.time()
    P1 = build_conditions(spec.get("P1", {}))
    P3 = build_conditions(spec.get("P3", {}))
    out_dir = Path(art_dir) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f">> [{label}] P1={len(P1)} P3={len(P3)} rebalance={rebalance} "
              f"dedup={dedup} batch={batch_size}", flush=True)

    stats_rows, batch = [], {}
    n_seen = n_kept = n_bt = 0
    bi = 0

    def flush():
        nonlocal batch, bi, n_bt
        if not batch:
            return
        bi += 1
        rc = sim_conditions(conditions=batch, resample=rebalance, data=md.data)
        export_report_collection_artifacts(rc, out_dir, to_parquet=True)
        for nm, rep in rc.reports.items():
            try:
                s = rep.get_stats()
                s["strategy"] = nm
                s["sharpe_ann"] = sharpe_ann_from_return_table(getattr(rep, "return_table", None))
                stats_rows.append(s)
            except Exception as e:
                warnings.warn(f"[stats] {nm}: {e}")
        n_bt += len(batch)
        if verbose:
            print(f"   [{label} batch {bi}] +{len(batch)}（累計 {n_bt}）{time.time()-t0:.0f}s",
                  flush=True)
        batch = {}
        del rc

    for name, mask in stream_strategy_masks(P1, P3, md, dedup=dedup):
        n_seen += 1
        if quick_stats_trades(mask) < min_trades:
            continue
        batch[name] = ensure_dtindex_cols(mask, md.common)
        n_kept += 1
        if len(batch) >= batch_size:
            flush()
        if smoke_limit and n_kept >= smoke_limit:
            break
    flush()

    # 合併 stats（逐批 export 寫過 batch-local stats，這裡覆蓋成完整版）
    if stats_rows:
        sdf = pd.DataFrame(stats_rows)
        try:
            sdf.to_parquet(out_dir / "stats.parquet")
        except Exception as e:
            warnings.warn(f"[stats] parquet 失敗改 csv：{e}")
            sdf.to_csv(out_dir / "stats.csv", index=False)

    secs = time.time() - t0
    meta = dict(label=label, market=md.market, rebalance=rebalance, dedup=dedup,
                n_expanded=n_seen, n_kept=n_kept, n_backtested=n_bt,
                seconds=round(secs, 1), finished_at=datetime.now().isoformat(timespec="seconds"),
                universe=len(md.common), meta=spec.get("_meta", {}))
    done_marker(label, art_dir).write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    if verbose:
        print(f"   [{label}] 展開={n_seen} 保留={n_kept} 回測={n_bt}｜{secs:.0f}s ✓_DONE",
              flush=True)
    return meta
