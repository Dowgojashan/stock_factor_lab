# -*- coding: utf-8 -*-
"""
Catalog Builder：把各 job 的 stats 併成**百科全書索引** `master_index.parquet`。

這張表就是後續 **LLM Agent 查詢的入口**（A4 §6）：
  一列 = 一個策略，帶「設定標籤（market/N/因子/頻率）+ 結構（F1/F2/C/V）+ 風險報酬指標」，
  並指向該策略的明細 artifacts 位置。
  LLM 服務保守型 → filter(max_drawdown > -X, sharpe_ann > Y)；激進型 → 換條件。

用法：python catalog_builder.py            # 掃描所有 _DONE 的 job 併表
      python catalog_builder.py --include-smoke
"""
import re
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ART_DIR = Path("results_artifacts")
CATALOG_DIR = Path("_catalog")

# 數值指紋：用來抓「完全相同」的重複策略
FINGERPRINT = ["CAGR", "max_drawdown", "win_ratio", "ytd"]

# 舊實驗產物，預設不納入 catalog（避免與新 sweep job 重疊雙數）。
# 尤其 spec_US（N=5、月頻、未去重）是新 US_f3_N5-10_c20_M 的子集 → 會重複計數。
LEGACY_LABELS = {"spec_US", "fcv_experiment_spec", "fcv_experiment_spec_US"}

# 與 io_persistence._sanitize 對齊（此處自帶一份，讓 catalog_builder 可獨立執行、不必 import
# 帶 pickle/report 依賴的 io_persistence）。artifacts 目錄名由 export 端 sanitize，
# 故 artifacts_dir 也必須套同一規則，否則路徑對不上（#5）。
_ILLEGAL = r'[\/:*?"<>|]'


def _sanitize(s: str) -> str:
    if s is None:
        return "None"
    return re.sub(_ILLEGAL, "_", str(s).strip()).rstrip(" .")


def parse_name(name: str) -> dict:
    """策略名 split('__') → [F1, F2, C, V]（與 report_analysis / analyze_spec_us 一致）。"""
    parts = str(name).split("__")
    parts += ["None"] * (4 - len(parts))
    F1, F2, C, V = parts[0], parts[1], parts[2], parts[3]

    def factor_of(tok):
        if tok in ("None", "", None):
            return "None"
        m = re.match(r"([A-Za-z_]+?)(_qb|_<|_>|_=|$)", tok)
        return (m.group(1) if m else tok).rstrip("_")

    def band_of(tok):
        m = re.search(r"_qb(\d+)of(\d+)$", str(tok))
        return (int(m.group(1)), int(m.group(2))) if m else (np.nan, np.nan)

    k1, n1 = band_of(F1)
    k2, n2 = band_of(F2)
    return {"F1": F1, "F2": F2, "C": C, "V": V,
            "F1_factor": factor_of(F1), "F2_factor": factor_of(F2),
            "F1_band": k1, "F1_N": n1, "F2_band": k2, "F2_N": n2,
            "C_kind": ("None" if C == "None" else C.split("_")[0])}


def load_job(job_dir: Path):
    """讀一個 job 的 stats + _DONE meta。回傳 DataFrame 或 None。"""
    done = job_dir / "_DONE"
    stats_pq = job_dir / "stats.parquet"
    stats_csv = job_dir / "stats.csv"
    if not done.exists():
        return None
    if stats_pq.exists():
        df = pd.read_parquet(stats_pq)
    elif stats_csv.exists():
        df = pd.read_csv(stats_csv)
    else:
        return None
    try:
        meta = json.loads(done.read_text(encoding="utf-8"))
    except Exception:
        meta = {}
    df["job_label"] = job_dir.name
    df["market"] = meta.get("market")
    df["rebalance"] = meta.get("rebalance")
    df["universe_size"] = meta.get("universe")
    spec_meta = meta.get("meta", {}) or {}
    df["N_list"] = str(spec_meta.get("N_list"))
    df["factors"] = str(spec_meta.get("factors"))
    df["artifacts_dir"] = df["strategy"].map(lambda s: str(job_dir / _sanitize(s)))  # #5
    return df


def build(include_smoke=False, include_legacy=False):
    if not ART_DIR.exists():
        raise SystemExit(f"找不到 {ART_DIR}")
    jobs = sorted(p for p in ART_DIR.iterdir() if p.is_dir())
    if not include_smoke:
        jobs = [p for p in jobs if not p.name.endswith("_SMOKE")]
    if not include_legacy:
        jobs = [p for p in jobs if p.name not in LEGACY_LABELS]

    frames, used = [], []
    for j in jobs:
        df = load_job(j)
        if df is None:
            continue
        frames.append(df)
        used.append((j.name, len(df)))
    if not frames:
        raise SystemExit("沒有任何已完成(_DONE)且有 stats 的 job。先跑 sweep_driver.py。")

    cat = pd.concat(frames, ignore_index=True)
    meta = cat["strategy"].apply(parse_name).apply(pd.Series)
    cat = pd.concat([cat, meta], axis=1)

    for c in ("CAGR", "sharpe_ann", "daily_sharpe", "max_drawdown", "avg_drawdown",
              "win_ratio", "ytd"):
        if c in cat.columns:
            cat[c] = pd.to_numeric(cat[c], errors="coerce")

    # 去重（兩種）：
    #  is_dup       ── 同一 job 內、數值指紋完全相同（抓對稱重複；新 job 已展開去重故 ≈0，舊 spec_US 會被抓）
    #  is_cross_dup ── 同一 (market, rebalance) 下、同名策略跨 job 重複（抓 legacy/config 重疊，防雙數）
    fp = [c for c in FINGERPRINT if c in cat.columns]
    cat["is_dup"] = cat.duplicated(subset=["job_label"] + fp, keep="first")
    cat["is_cross_dup"] = cat.duplicated(subset=["market", "rebalance", "strategy"], keep="first")

    front = ["strategy", "market", "rebalance", "job_label",
             "F1", "F2", "C", "V", "F1_factor", "F2_factor",
             "F1_band", "F1_N", "F2_band", "F2_N", "C_kind",
             "CAGR", "sharpe_ann", "max_drawdown", "avg_drawdown", "win_ratio", "ytd",
             "is_dup", "is_cross_dup", "universe_size", "N_list", "factors", "artifacts_dir"]
    cols = [c for c in front if c in cat.columns] + [c for c in cat.columns if c not in front]
    cat = cat[cols]

    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    out = CATALOG_DIR / "master_index.parquet"
    try:
        cat.to_parquet(out)
    except Exception as e:
        print(f"[warn] parquet 失敗改 csv：{e}")
        out = CATALOG_DIR / "master_index.csv"
        cat.to_csv(out, index=False)

    uniq = cat.loc[~cat["is_dup"] & ~cat["is_cross_dup"]]
    reg = CATALOG_DIR / "dedup_registry.parquet"
    try:
        uniq[[c for c in ("strategy", "job_label", "market", "rebalance") if c in uniq.columns]] \
            .to_parquet(reg)
    except Exception:
        pass

    print("===== Catalog 建立完成 =====")
    for name, n in used:
        print(f"  {name:26s} {n:>7,} 策略")
    print(f"\n  總計 {len(cat):,} 列｜獨立 {len(uniq):,}｜"
          f"同 job 重複 {int(cat['is_dup'].sum()):,}｜跨 job 重疊 {int(cat['is_cross_dup'].sum()):,}")
    print(f"  → {out}")
    if "market" in cat.columns:
        print("\n  各市場/頻率分佈：")
        g = cat.groupby(["market", "rebalance"]).size().reset_index(name="n")
        print("   " + g.to_string(index=False).replace("\n", "\n   "))
    if "CAGR" in cat.columns and len(uniq):
        print("\n  風險光譜（獨立策略）：")
        print(f"    CAGR       中位 {uniq['CAGR'].median():.3f}｜"
              f"範圍 {uniq['CAGR'].min():.3f} ~ {uniq['CAGR'].max():.3f}")
        if "sharpe_ann" in uniq.columns and uniq["sharpe_ann"].notna().any():
            print(f"    sharpe_ann 中位 {uniq['sharpe_ann'].median():.2f}｜"
                  f"範圍 {uniq['sharpe_ann'].min():.2f} ~ {uniq['sharpe_ann'].max():.2f}")
    return cat


def main():
    ap = argparse.ArgumentParser(description="建立百科全書索引 master_index")
    ap.add_argument("--include-smoke", action="store_true", help="連 _SMOKE 的 job 也納入")
    ap.add_argument("--include-legacy", action="store_true",
                    help=f"連舊實驗產物也納入（{sorted(LEGACY_LABELS)}），預設排除以免重疊雙數")
    args = ap.parse_args()
    build(include_smoke=args.include_smoke, include_legacy=args.include_legacy)


if __name__ == "__main__":
    main()
