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
from io_persistence import save_all_for_label

# ==================== 參數 ====================
USE_CACHE = True
MARKET = "US"
JSON_FILE = "fcv_experiment_spec.json"
LABEL = "fcv_experiment_spec_US"          # US 專屬（避免覆蓋台股）
MIN_TRADES = 5
TOP_K = 10000
ENTRY_THRESH = 1e-9
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
    common = [c for c in close.columns if c in univ]
    START = pd.Timestamp(os.environ.get("US_START", "2018-01-01"))   # 美股 MVP：2018+
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

    # --- F / C / V 展開 ---
    t = time.time()
    p1p2 = f_factor(P1, field_cache, price_index=price_index)
    final_masks = c_factor(p1p2, P3, field_cache, price_index=price_index)
    v_mask = build_v_pe_mask(data, pe_field="report:pe", window=4, price_index=price_index, use_cache=USE_CACHE)
    final_masks = expand_with_v(final_masks, v_mask, price_index=price_index)
    print(f"   展開後策略總數：{len(final_masks)}（耗時 {time.time()-t:.1f}s）")

    # --- 預篩 ---
    kept = {}
    for name, pos in final_masks.items():
        if quick_stats(pos)["trades_total"] >= MIN_TRADES:
            kept[name] = pos
    if TOP_K and kept:
        kept = dict(sorted(kept.items(), key=lambda kv: count_trades_from_position(kv[1]), reverse=True)[:TOP_K])
    print(f"   預篩後策略：{len(kept)} / {len(final_masks)}（MIN_TRADES={MIN_TRADES}）")

    # --- 欄位/索引淨化 + 對齊到價格宇宙 ---
    kept = {name: ensure_dtindex_cols(pos, common) for name, pos in kept.items()}

    # --- SMOKE：先小量驗證 ---
    label = LABEL
    if SMOKE_LIMIT:
        kept = dict(list(kept.items())[:SMOKE_LIMIT])
        label = LABEL + "_SMOKE"
        print(f"   [SMOKE] 只回測前 {len(kept)} 個策略，label={label}")

    # --- 回測 + 儲存 ---
    print(f">> 回測 {len(kept)} 個策略 ...")
    t = time.time()
    rc = sim_conditions(conditions=kept, resample="M", data=data)
    print(f"   回測完成（耗時 {time.time()-t:.1f}s）")
    save_all_for_label(report_collection=rc, base_pickle_dir=PICKLE_DIR,
                       base_artifacts_dir=ART_DIR, label=label, to_parquet=True)
    print(f">> 已存至 results_pickle/{label} 與 results_artifacts/{label}")

    # --- 摘要 ---
    stats = rc.get_stats()
    print("\n===== 摘要（前 5 名 by CAGR）=====")
    st = stats.T.sort_values("CAGR", ascending=False)
    print(st.head(5).to_string())
    print(f"\n總耗時 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
