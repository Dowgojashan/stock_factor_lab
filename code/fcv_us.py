# -*- coding: utf-8 -*-
"""
FCV 美股版完整多因子回測（獨立執行腳本）。
忠實移植 fcv_backtest.ipynb 的 F/C/V 遮罩邏輯，改為 market="US"：
  - Data(market="US")，並把價格宇宙限縮到「有因子的 S&P500 宇宙」以控記憶體 + 欄位對齊
  - 結果存成 US 專屬 label（不覆蓋台股）
  - SMOKE_LIMIT：先小量驗證整條管線，通過再設 None 跑全量
用法：cwd=code/ 執行；.venv python。
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = r"d:/研究所/stock_factor_lab-master/stock_factor_lab"
sys.path.insert(0, ROOT)          # get_data / backtest / combinations 在 ROOT
os.chdir(ROOT + "/code")          # condition_factory / io_persistence 在 code/；../config.ini
sys.path.insert(0, ".")

from get_data import Data
from condition_factory import build_conditions
from combinations import sim_conditions
from io_persistence import save_all_for_label, export_report_collection_artifacts

# ==================== 參數 ====================
USE_CACHE = True
MARKET = "US"
JSON_FILE = "spec_US.json"                # A3 q_band 分位數 spec（宇宙＝Russell 3000）
LABEL = "spec_US"                         # US 專屬（避免覆蓋台股）
MIN_TRADES = 5
TOP_K = 10000
ENTRY_THRESH = 1e-9
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "150"))   # 分批回測：每批策略數（記憶體安全）
SMOKE_LIMIT = int(os.environ.get("SMOKE_LIMIT", "0")) or None   # 0/未設=全量；>0=只回測前 N 個
PICKLE_DIR = Path("results_pickle")
ART_DIR = Path("results_artifacts")

timings, strategy_count = {}, {}
_aligned_cache, _mask_cache = {}, {}
_v_mask_cache = None
_v_mask_cache_key = None

# ==================== 對齊工具（cell 11）====================
def align_to_trading_days(df, price_index, field_name="", warn_on_forward_fill=True):
    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        if df.index.isna().any():
            df = df.loc[df.index.notna()]
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    aligned = df.reindex(price_index, method="ffill")
    return aligned

def get_trading_day_index(data):
    price_close = data.get("price:close")
    idx = price_close.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
    return idx

# ==================== build_masks（cell 14）====================
def build_masks(conditions, field_cache, price_index=None, prefix=""):
    masks = {}
    for cond in conditions:
        field_name = cond["field"]; cond_name = cond["name"]
        cache_key = (field_name, cond_name)
        if USE_CACHE and cache_key in _mask_cache:
            masks[cond_name] = _mask_cache[cache_key]; continue
        df = field_cache.get(field_name)
        if df is None:
            print(f"{prefix}欄位 {field_name} 不存在，略過 {cond_name}"); continue
        if price_index is not None and not df.index.equals(price_index):
            df_aligned = align_to_trading_days(df, price_index, field_name=field_name)
        else:
            df_aligned = df
        try:
            mask = cond["cond"](df_aligned)
            if price_index is not None and not mask.index.equals(price_index):
                mask = mask.reindex(price_index, method="ffill").fillna(False)
            masks[cond_name] = mask
            if USE_CACHE:
                _mask_cache[cache_key] = mask
        except Exception as e:
            warnings.warn(f"[build_masks] 計算 {cond_name} 時發生錯誤: {e}"); continue
    return masks

# ==================== f_factor（cell 16）====================
def f_factor(P1_conditions, field_cache, price_index=None):
    P1_masks = build_masks(P1_conditions, field_cache, price_index=price_index, prefix="P1")
    P2_candidate_map = defaultdict(list)
    for p1_cond in P1_conditions:
        p1_name = p1_cond["name"]; p1_field = p1_cond["field"]
        P2_candidate_map[p1_name].append(None)
        for p2_cond in P1_conditions:
            if p2_cond["field"] == p1_field:
                continue
            P2_candidate_map[p1_name].append(p2_cond)
    p1p2_masks = {}
    for p1_cond in P1_conditions:
        p1_name = p1_cond["name"]; p1_mask = P1_masks.get(p1_name)
        if p1_mask is None:
            continue
        for p2_cond in P2_candidate_map[p1_name]:
            if p2_cond is None:
                final_mask = p1_mask.copy(); p2_name = "None"
            else:
                p2_name = p2_cond["name"]; p2_field = p2_cond["field"]; p2_func = p2_cond["cond"]
                p2_cache_key = (p2_field, p2_name)
                if USE_CACHE and p2_cache_key in _mask_cache:
                    p2_mask = _mask_cache[p2_cache_key]
                else:
                    df = field_cache.get(p2_field)
                    if df is None:
                        print(f"P2 欄位 {p2_field} 不存在，略過 {p2_name}"); continue
                    if price_index is not None and not df.index.equals(price_index):
                        df = align_to_trading_days(df, price_index, field_name=p2_field)
                    p2_mask = p2_func(df)
                    if price_index is not None and not p2_mask.index.equals(price_index):
                        p2_mask = p2_mask.reindex(price_index, method="ffill").fillna(False)
                    if USE_CACHE:
                        _mask_cache[p2_cache_key] = p2_mask
                final_mask = p1_mask & p2_mask
            if price_index is not None and not final_mask.index.equals(price_index):
                final_mask = final_mask.reindex(price_index, method="ffill").fillna(False)
            p1p2_masks[f"{p1_name}__{p2_name}"] = final_mask
    return p1p2_masks

# ==================== c_factor（cell 18）====================
def c_factor(p1p2_masks, P3_conditions, field_cache, price_index=None):
    P3_masks = build_masks(P3_conditions, field_cache, price_index=price_index, prefix="P3")
    final_masks = {}
    for combo_key, base_mask in p1p2_masks.items():
        if price_index is not None and not base_mask.index.equals(price_index):
            base_mask = base_mask.reindex(price_index, method="ffill").fillna(False)
        P3_cond_list = [None] + list(P3_masks.items())
        for p3 in P3_cond_list:
            if p3 is None:
                final_final_mask = base_mask.copy(); combo3_key = f"{combo_key}__None"
            else:
                p3_name, p3_mask = p3; combo3_key = f"{combo_key}__{p3_name}"
                if price_index is not None and not p3_mask.index.equals(price_index):
                    p3_mask = p3_mask.reindex(price_index, method="ffill").fillna(False)
                final_final_mask = base_mask & p3_mask
            if price_index is not None and not final_final_mask.index.equals(price_index):
                final_final_mask = final_final_mask.reindex(price_index, method="ffill").fillna(False)
            final_masks[combo3_key] = final_final_mask
    return final_masks

# ==================== V 構面（cell 19）====================
def build_v_pe_mask(data, pe_field="report:pe", window=4, price_index=None,
                    daily_to_quarter=True, use_cache=True):
    global _v_mask_cache, _v_mask_cache_key
    if price_index is None:
        price_index = data.get("price:close").index
    if not isinstance(price_index, pd.DatetimeIndex):
        price_index = pd.to_datetime(price_index)
    cache_key = (pe_field, window, id(price_index))
    if use_cache and _v_mask_cache is not None and _v_mask_cache_key == cache_key:
        return _v_mask_cache
    pe_raw = data.get(pe_field)
    pe_base = pe_raw.sort_index()
    if not isinstance(pe_base.index, pd.DatetimeIndex):
        pe_base.index = pd.to_datetime(pe_base.index)
    pe_for_rolling = pe_base
    if daily_to_quarter and len(pe_base.index) >= 10:
        median_gap = pe_base.index.to_series().diff().dropna().dt.days.median()
        if (median_gap is not None) and (median_gap < 40):
            try:
                pe_for_rolling = pe_base.resample("QE").last()
            except Exception:
                pe_for_rolling = pe_base.resample("Q").last()
    pe_mean4_q = pe_for_rolling.rolling(window, min_periods=window).mean()
    pe_min4_q = pe_for_rolling.rolling(window, min_periods=window).min()
    v1_q = (pe_for_rolling < pe_mean4_q) & (pe_for_rolling > pe_min4_q)
    v1 = v1_q.reindex(price_index, method="ffill").fillna(False)
    if use_cache:
        _v_mask_cache = v1; _v_mask_cache_key = cache_key
    return v1

def expand_with_v(final_masks, v_mask, suffix_v0="__v0", suffix_v1="__v1", price_index=None):
    expanded = {}
    if price_index is None and hasattr(v_mask, "index"):
        price_index = v_mask.index
    for name, m in final_masks.items():
        if price_index is not None and not m.index.equals(price_index):
            m = m.reindex(price_index, method="ffill").fillna(False)
        expanded[f"{name}{suffix_v0}"] = m.copy()
        v_mask_aligned = v_mask
        if price_index is not None and not v_mask.index.equals(price_index):
            v_mask_aligned = v_mask.reindex(price_index, method="ffill").fillna(False)
        expanded[f"{name}{suffix_v1}"] = (m & v_mask_aligned)
    return expanded

# ==================== 串流展開（記憶體安全版 F/C/V）====================
def stream_strategy_masks(P1, P3, field_cache, v_mask, price_index):
    """串流產生 F/C/V 展開後的每個策略遮罩，一次 yield 一個 (name, daily_mask)。
    語意等同 f_factor→c_factor→expand_with_v，但**不同時 materialize 全部 2310 個 frame**：
    逐 base 組合即時展開 C×V 後 yield，交由呼叫端分批回測+落地+釋放 → 峰值記憶體 O(1 batch)。"""
    P1_masks = build_masks(P1, field_cache, price_index=price_index, prefix="P1")
    P3_masks = build_masks(P3, field_cache, price_index=price_index, prefix="P3")
    P3_items = [("None", None)] + list(P3_masks.items())   # None＝不加 P3
    for p1_cond in P1:
        p1_name, p1_field = p1_cond["name"], p1_cond["field"]
        p1_mask = P1_masks.get(p1_name)
        if p1_mask is None:
            continue
        # P2 候選＝None + 所有「異因子」的 P1 條件（與 f_factor 一致）
        p2_candidates = [None] + [c for c in P1 if c["field"] != p1_field]
        for p2_cond in p2_candidates:
            if p2_cond is None:
                base, p2_name = p1_mask, "None"
            else:
                p2_name = p2_cond["name"]
                p2_mask = P1_masks.get(p2_name)
                if p2_mask is None:
                    continue
                base = p1_mask & p2_mask
            for p3_name, p3_mask in P3_items:
                cmask = base if p3_mask is None else (base & p3_mask)
                key = f"{p1_name}__{p2_name}__{p3_name}"
                yield f"{key}__v0", cmask
                yield f"{key}__v1", cmask & v_mask


# ==================== 預篩（cell 23）+ 淨化（cell 24）====================
def count_trades_from_position(pos, thresh=ENTRY_THRESH):
    if isinstance(pos, pd.Series):
        pos = pos.to_frame()
    held = pos.values > thresh
    held_y = np.vstack([np.zeros((1, held.shape[1]), dtype=bool), held[:-1, :]])
    return int((held & (~held_y)).sum())

def quick_stats(pos):
    # numpy 版（比 pandas .diff() 快很多）：計算「有任一股票變動」的天數
    v = pos.values
    if v.shape[0] < 2:
        return dict(trades_total=0)
    changed = (v[1:] != v[:-1]).any(axis=1).sum()
    return dict(trades_total=int(changed))

def load_russell_symbols(data):
    """讀 DB 的 russell3000 成分表，回傳 symbol 集合（宇宙圈定用）。
    成分表由 collector 端維護（見 stock_factor_collector/UPDATE_US.md §0.6），
    symbol 已正規化為破折號格式（如 BRK-B），與價格/因子欄位對齊。"""
    conn = data.db.create_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT symbol FROM russell3000")
        return {r[0] for r in cur.fetchall() if r and r[0]}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def ensure_dtindex_cols(df, cols):
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy(); df.index = pd.to_datetime(df.index, errors="coerce")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    # 對齊欄位到價格宇宙（美股：position 欄位 ⊆ price 欄位，補 False 保證對齊）
    df = df.reindex(columns=cols, fill_value=False)
    return df


def main():
    t0 = time.time()
    print(f">> 載入 Data(market='{MARKET}') ...")
    data = Data(market=MARKET)

    # --- 宇宙限縮：價格 → 有因子的宇宙（控記憶體 + 欄位對齊）---
    factor_fields = ["report:ROE", "report:EPS", "report:FCF_P",
                     "report:DEBTRATIO", "report:REVENUE", "report:PE"]
    univ = set()
    for f in factor_fields:
        d = data.get(f)
        if d is not None:
            univ |= set(d.columns)
    close = data.get("price:close")
    # --- Russell 3000 宇宙圈定：先篩宇宙，之後 q_band 才在正確母體上排名（A3 filter-before-rank）---
    russell = load_russell_symbols(data)
    common = [c for c in close.columns if c in univ and c in russell]
    print(f"   Russell 3000 成分={len(russell)}｜∩有因子∩有價格={len(common)}")
    _ulim = int(os.environ.get("US_UNIV_LIMIT", "0") or 0)   # >0：只留前 N 檔（冒煙測試用）
    if _ulim > 0:
        common = common[:_ulim]
        print(f"   [US_UNIV_LIMIT] 宇宙限縮為前 {_ulim} 檔：{common}")
    START = pd.Timestamp(os.environ.get("US_START", "2000-01-01"))   # 回測窗 2000–2026（老師定案）
    for k in list(data.all_price_dict.keys()):
        df = data.all_price_dict[k].reindex(columns=common)
        df.index = pd.to_datetime(df.index)
        data.all_price_dict[k] = df[df.index >= START]
    print(f"   因子宇宙={len(univ)}, 價格宇宙={close.shape[1]}, 回測宇宙(交集)={len(common)}, 起點>={START.date()}")

    price_index = get_trading_day_index(data)
    print(f"   交易日索引：{len(price_index)} 日，{price_index.min().date()} ~ {price_index.max().date()}")

    # --- 條件解析 + 欄位快取 ---
    with open(JSON_FILE, encoding="utf-8") as f:
        cfg = json.load(f)
    P1 = build_conditions(cfg.get("P1", {}))
    P3 = build_conditions(cfg.get("P3", {}))
    print(f"   P1 條件 {len(P1)} 個、P3 條件 {len(P3)} 個")

    field_cache = {}
    all_fields = {c["field"] for c in P1 + P3} | {"price:close"}
    for field in all_fields:
        raw = data.get(field)
        if raw is None:
            print(f"   欄位 {field} 不存在，略過"); continue
        if field.startswith("report:"):
            field_cache[field] = align_to_trading_days(raw, price_index, field_name=field)
        else:
            field_cache[field] = raw if raw.index.equals(price_index) else align_to_trading_days(raw, price_index, field_name=field)

    # --- 先篩宇宙：把財報 frame 欄位限縮到 Russell 3000∩宇宙，q_band 才在正確母體上排名 ---
    #     q_band 對「傳進來的 frame」做橫斷面 rank(axis=1)，故限縮欄位＝排名母體＝Russell 3000。
    #     不可「全 US 排名後再交集」（見 A3 計畫步驟1 filter-before-rank 警告）。
    for field in list(field_cache):
        if field.startswith("report:"):
            field_cache[field] = field_cache[field].reindex(columns=common)

    # --- V 遮罩（建一次，限縮到 Russell 宇宙）---
    v_mask = build_v_pe_mask(data, pe_field="report:pe", window=4, price_index=price_index, use_cache=USE_CACHE)
    v_mask = v_mask.reindex(columns=common, fill_value=False)

    # --- 串流展開 + 分批回測 + 逐批落地（記憶體安全）---
    #     不一次 materialize 全部遮罩/報告：每湊滿 BATCH_SIZE 就回測→寫 artifacts→釋放，
    #     峰值記憶體 ≈ 一個 batch 的遮罩+報告（O(1 batch)），與策略總數無關。
    label = LABEL + ("_SMOKE" if SMOKE_LIMIT else "")
    art_label_dir = ART_DIR / label
    print(f">> 串流展開 + 分批回測（BATCH_SIZE={BATCH_SIZE}）→ label={label}")
    t = time.time()
    stats_rows, batch = [], {}
    n_seen = n_kept = n_bt = 0
    batch_idx = 0

    def flush():
        nonlocal batch, batch_idx, n_bt
        if not batch:
            return
        batch_idx += 1
        rc = sim_conditions(conditions=batch, resample="M", data=data)
        export_report_collection_artifacts(rc, art_label_dir, to_parquet=True)  # 逐策略 artifacts
        for nm, rep in rc.reports.items():
            try:
                s = rep.get_stats(); s["strategy"] = nm; stats_rows.append(s)
            except Exception as e:
                warnings.warn(f"[stats] {nm}: {e}")
        n_bt += len(batch)
        print(f"   [batch {batch_idx}] 回測+落地 {len(batch)} 個（累計 {n_bt}）耗時 {time.time()-t:.0f}s", flush=True)
        batch = {}
        del rc

    for name, mask in stream_strategy_masks(P1, P3, field_cache, v_mask, price_index):
        n_seen += 1
        if quick_stats(mask)["trades_total"] < MIN_TRADES:
            continue
        batch[name] = ensure_dtindex_cols(mask, common)
        n_kept += 1
        if len(batch) >= BATCH_SIZE:
            flush()
        if SMOKE_LIMIT and n_kept >= SMOKE_LIMIT:
            break
    flush()
    print(f"   展開總數={n_seen}、預篩保留={n_kept}、回測={n_bt}（MIN_TRADES={MIN_TRADES}，耗時 {time.time()-t:.1f}s）")

    # --- 合併 stats 落地（逐批 export 已寫過 batch-local stats，這裡覆蓋成完整版）---
    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        try:
            stats_df.to_parquet(art_label_dir / "stats.parquet")
        except Exception as e:
            warnings.warn(f"[stats] parquet 失敗改 csv：{e}")
            stats_df.to_csv(art_label_dir / "stats.csv", index=False)
        print(f">> 完整 stats 落地 → {art_label_dir}/stats（{len(stats_df)} 策略）")
        try:
            top = stats_df.set_index("strategy").sort_values("CAGR", ascending=False)
            print("\n===== 摘要（前 5 名 by CAGR）=====")
            print(top.head(5).to_string())
        except Exception:
            pass
    print(f"\n總耗時 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
