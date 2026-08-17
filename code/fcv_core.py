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
from io_persistence import _to_parquet_or_csv, _sanitize, _ensure_dir

# ==================== 常數 ====================
MIN_TRADES = 5
ENTRY_THRESH = 1e-9
ART_DIR = Path("results_artifacts")
DEFAULT_START = "2000-01-01"     # MarketData 的 fallback；per-market 起點由 sweep_config.MARKET_START 傳入

# 用來界定「有因子的宇宙」。TW 缺的欄位會自動略過。
# 後段 11 個為因子候選批次（run_factor_batches.py）新增，涵蓋論文3.6.1節候選因子中
# 台股資料庫實際可得的欄位（見 fcv_backtest.ipynb 已驗證的 factor_name 清單）。
FACTOR_FIELDS = ["report:ROE", "report:EPS", "report:FCF_P",
                 "report:DEBTRATIO", "report:REVENUE", "report:PE",
                 "report:EV_EBITDA", "report:EV_S", "report:CROIC", "report:FCF_OI",
                 "report:ROIC", "report:PB", "report:PS", "report:P_IC", "report:OCF_E",
                 "report:MOM"]


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

    def __init__(self, market, start=None, end=None, univ_limit=0, verbose=True):
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

        START = pd.Timestamp(start or DEFAULT_START)
        # end=None（預設）＝跑到資料最末端，與加入本參數前的行為完全相同。
        # 給定 end 時＝in-sample 上界，2026-08 起改以此明確界定樣本內/外（見 重跑計畫_老師方法論SOP.md）。
        END = pd.Timestamp(end) if end else None
        for k in list(self.data.all_price_dict.keys()):
            df = self.data.all_price_dict[k].reindex(columns=common)
            df.index = pd.to_datetime(df.index)
            df = df[df.index >= START]
            if END is not None:
                df = df[df.index <= END]
            self.data.all_price_dict[k] = df

        idx = self.data.get("price:close").index
        self.price_index = idx if isinstance(idx, pd.DatetimeIndex) else pd.to_datetime(idx)

        self._field_cache = {}
        self._ann_cache = {}   # get_field_ann 的快取（公告點稀疏 frame）
        self._mask_cache = {}
        self._v_mask = None
        if verbose:
            rng = f"起點>={START.date()}" + (f"｜終點<={END.date()}" if END is not None else "")
            print(f"   因子宇宙={len(univ)}｜回測宇宙={len(common)}｜{rng}｜"
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

    def get_field_ann(self, field):
        """**公告點稀疏** frame：每個非 NaN 就是該股的一次財報公告，NaN＝當天沒有新財報。

        為什麼需要它（2026-08-17）：
          `get_data.Data.get()` 在回傳財報前做了一次 `fillna(ffill)`（get_data.py:136），
          讓每個交易日都有值。這對**橫斷面**條件（q_band）是必要的——每列要有完整的
          同期同儕才排得了名。但對**時序**條件是災難：ffill 之後看不出「這天有沒有新財報」，
          只能用「值有沒有變」去猜，而那個猜測在各公司公告日分散的美股完全失效
          （riseq1 通過率 0.8%、qmax4 97.5%，應各約 50% 與 25%）。

        作法：重用 `format_report_data` / `adjust_index_of_report`，只是**跳過那次 ffill**。
        不修改 get_data.py / format_data.py（共用 pipeline），把差異收斂在這裡。

        ⚠️ 回傳的 frame **不對齊交易日**（索引是財報日／公告日）。
           條件算完後由 `get_mask` 統一 reindex 回 price_index。
        """
        if field in self._ann_cache:
            return self._ann_cache[field]
        f = None
        try:
            from format_data import format_report_data, adjust_index_of_report
            item = field.split(":", 1)[1].upper().replace(" ", "")
            d = format_report_data(self.data.raw_report_data, item, market=self.market)
            if self.market != "US":
                d = adjust_index_of_report(d)      # 台股套法定期限；美股已用 filing_date
            f = next(iter(d.values()))
            f.index = pd.to_datetime(f.index)
            f = f.sort_index().reindex(columns=self.common)   # ★ 同 get_field 的 filter-before-rank
            f = f[(f.index >= self.price_index.min()) & (f.index <= self.price_index.max())]
        except Exception as e:
            # ⚠️ **不可退回密集 frame**：那會靜默地把時序條件變回出事的版本。
            #    回 None 讓 get_mask 跳過該條件——展開數會少、log 對不上，是「看得見的失敗」。
            #    本專案已經因為「靜默降級」付出過代價（面板壓縮在美股失效數週未察覺）。
            warnings.warn(f"[get_field_ann] {field} 取不到公告點 frame：{e}；"
                          f"該條件將被跳過（不退回密集 frame，避免靜默失準）")
            f = None
        self._ann_cache[field] = f
        return f

    def get_mask(self, cond):
        """建某條件的遮罩，跨 spec 快取（同名條件不重算）。

        ⚠️ 快取 key=(field, name)，跨 spec 共用的正確性**依賴一個不變式**：
           條件名必須唯一決定其語意（含 args）。`condition_factory.auto_name` 對所有型別都
           把 args 編進名字（q_band→qbKofN、rise_q→riseqN…）故成立。
           M/Q 兩 job 用同一 spec、名稱相同 → 正確重用同一 mask（這是載入共享的關鍵）。
           **若日後新增條件型別而 auto_name 未把 args 編進名字，此快取會取到錯的 mask。**
        """
        key = (cond["field"], cond["name"])
        if key in self._mask_cache:
            return self._mask_cache[key]
        # 「季」語意的時序條件要吃公告點稀疏 frame；橫斷面條件（q_band）要吃密集 frame。
        # 分派依據是 condition_factory.ANNOUNCEMENT_TYPES，由 build_conditions 標在 needs_ann。
        df = (self.get_field_ann(cond["field"]) if cond.get("needs_ann")
              else self.get_field(cond["field"]))
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
def stream_strategy_masks(P1, P3, md, dedup=True, v_modes=("v0", "v1"),
                          allowed_f_pairs=None):
    """串流 yield (name, daily_mask)，不同時 materialize 全部。

    dedup=True：F1×F2 只產**無序對**（j>i）→ EPS__ROE 與 ROE__EPS 不重複，省 45%。

    v_modes：要展開哪些 V 構面。預設 ("v0","v1") ＝加入本參數前的行為完全相同。
      傳 ("v0",) 可只跑不套估值濾網的版本——供「一次只開一個構面」的分階段實驗
      （Phase 1 單因子線性檢定）使用，策略數直接減半。

    allowed_f_pairs：F 構面白名單，一組 f"{F1名}__{F2名}" 字串（F2=None 時寫 "None"）。
      預設 None ＝不過濾，行為與加入本參數前完全相同。
      給定時只展開名單內的 F 組合——供分階段實驗把上一關**已篩選過**的組合帶進下一關
      （Phase 3 只跑 Phase 2 晉升的 203 個 F 組合，而不是全部 630 個再事後丟棄）。
      過濾發生在 C 迴圈之前，故被排除的組合完全不會產生任何遮罩運算。
    """
    allowed = set(allowed_f_pairs) if allowed_f_pairs is not None else None
    v_modes = tuple(v_modes)
    bad = [v for v in v_modes if v not in ("v0", "v1")]
    if bad:
        raise ValueError(f"v_modes 只接受 'v0'/'v1'，收到 {bad}")
    v_mask = md.get_v_mask() if "v1" in v_modes else None
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
            if allowed is not None and f"{p1['name']}__{p2name}" not in allowed:
                continue                                # 不在白名單 → 連 C 都不展開
            for p3name, m3 in P3_items:
                cmask = base if m3 is None else (base & m3)
                key = f"{p1['name']}__{p2name}__{p3name}"
                if "v0" in v_modes:
                    yield f"{key}__v0", cmask
                if "v1" in v_modes:
                    yield f"{key}__v1", cmask & v_mask


# ==================== return_table / sharpe_ann（取代壞掉的 daily_sharpe）====================
def return_table_to_df(return_table):
    """把 rep.return_table（{year:{month:val}}）穩健地轉成 年×月 DataFrame。
    ⚠️ 鍵型別混雜（int 年 + str 'YTD' 之類）會讓 pd.DataFrame 建構時排序失敗
       （'<' not supported between 'str' and 'int'），退化策略（交易稀疏）常見。
       故先把所有鍵統一成 str 再建，避免混型別比較。"""
    if not isinstance(return_table, dict) or not return_table:
        return None
    norm = {}
    for k, v in return_table.items():
        norm[str(k)] = {str(ik): iv for ik, iv in v.items()} if isinstance(v, dict) else v
    df = pd.DataFrame(norm).T
    df.columns = [str(c) for c in df.columns]
    return df


def sharpe_ann_from_return_table(return_table):
    """從 return_table（逐年×逐月報酬）算正確年化 Sharpe：mean/std×√12。
    去掉起訖的結構性零（策略起訖前後未持倉）。daily_sharpe 是壞值，見 A4 §10。"""
    try:
        rt = return_table_to_df(return_table)
        if rt is None:
            return np.nan
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


# ==================== 逐 report 落地（get_stats 只算一次）====================
def write_report_artifacts(rc, out_dir):
    """逐策略寫明細 artifacts（trades/position/stock_data/return_table）並收 stats。
    **get_stats() 每策略只算一次**（不用 export_report_collection_artifacts，避免重複 get_stats），
    順便補上正確的 sharpe_ann。回傳 stats Series 清單。"""
    out_dir = Path(out_dir)
    rows = []
    for name, rep in rc.reports.items():
        sdir = _ensure_dir(out_dir / _sanitize(name))
        if hasattr(rep, "trades"):
            _to_parquet_or_csv(rep.trades, sdir / "trades", True)
        if hasattr(rep, "position"):
            _to_parquet_or_csv(rep.position, sdir / "position", True)
        if hasattr(rep, "stock_data"):
            _to_parquet_or_csv(rep.stock_data, sdir / "stock_data", True)
        try:
            rt_df = return_table_to_df(getattr(rep, "return_table", None))
            if rt_df is not None:
                _to_parquet_or_csv(rt_df, sdir / "return_table", True)
        except Exception as e:
            warnings.warn(f"[return_table] {name}: {e}")
        try:
            s = rep.get_stats()                                   # ★ 只算一次
            s["strategy"] = name
            s["sharpe_ann"] = sharpe_ann_from_return_table(getattr(rep, "return_table", None))
            rows.append(s)
        except Exception as e:
            warnings.warn(f"[stats] {name}: {e}")
    return rows


# ==================== 跑一份 spec ====================
def done_marker(label, art_dir=ART_DIR):
    return Path(art_dir) / label / "_DONE"


def is_done(label, art_dir=ART_DIR):
    return done_marker(label, art_dir).exists()


def run_spec(md, spec, label, rebalance="M", batch_size=150, art_dir=ART_DIR,
             min_trades=MIN_TRADES, smoke_limit=None, dedup=True, verbose=True,
             v_modes=("v0", "v1"), allowed_f_pairs=None):
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
        stats_rows.extend(write_report_artifacts(rc, out_dir))   # 逐 report get_stats 只算一次
        n_bt += len(batch)
        if verbose:
            print(f"   [{label} batch {bi}] +{len(batch)}（累計 {n_bt}）{time.time()-t0:.0f}s",
                  flush=True)
        batch = {}
        del rc

    for name, mask in stream_strategy_masks(P1, P3, md, dedup=dedup, v_modes=v_modes,
                                            allowed_f_pairs=allowed_f_pairs):
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

    # #1：0 策略視為失敗，不寫 _DONE（否則會被 catalog 略過又被 is_done 永久跳過）。
    #     raise 讓 sweep_driver 的 except 寫 _FAILED + log，下次會重試。
    if n_bt == 0:
        raise RuntimeError(
            f"[{label}] 展開後 0 個策略通過 MIN_TRADES（資料可能缺失或全被濾除）；不寫 _DONE")

    # 合併 stats
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
