# -*- coding: utf-8 -*-
"""階段1 前置 · MKTCAP 橫斷面分位表（C-5 規模依賴欄位的原料）

輸入 ← 資料庫 `report:mktcap`（collector 於 2026-08-13 新增的季底普通股市值）
輸出 → `_frozen/stage1/mktcap_pct_{market}.parquet`（月 × 股票，值 = 當月橫斷面分位 0–100）

為什麼獨立成一支、不併進掃描器（SDD DD-05 的例外）：
  主掃描要讀 4.7 GB、跑數小時，**不應該在過程中依賴資料庫連線**——
  MySQL 曾因硬關機損壞過（見 CLAUDE.md §5），DB 打嗝不該毀掉幾小時的掃描。
  故把唯一需要 DB 的部分抽出來先跑一次、落地成 parquet，掃描器只讀檔案。

⚠️ MKTCAP 只是規模控制的控制變數，**不進入 F 構面的因子池**（沿用
   size_control_backtest.py 的立場）。

用法：
    cd code
    python -m research.stage1_mktcap [--market TW|US]
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from . import contracts as C
from . import freeze, paths

FIELD = "report:mktcap"
#: 最小市值三分之一的分位上界（smallcap_share 的判準）
SMALLCAP_PCT_MAX = 100.0 / 3.0


def build_market(market: str, log=print) -> pd.DataFrame:
    import fcv_core                                    # 掛 sys.path 已由 paths 完成

    t0 = time.time()
    log(f"[{market}] 載入 MarketData（含 DB 讀取，約 100 秒）…")
    md = fcv_core.MarketData(market, end=f"{paths.IN_SAMPLE_END}-31", verbose=True)

    raw = md.get_field(FIELD)
    if raw is None:
        raise RuntimeError(f"[{market}] 取不到 {FIELD}；請確認 collector 已補上 MKTCAP")
    log(f"[{market}] 原始 frame {raw.shape}，非空率 {raw.notna().mean().mean():.1%}")

    # 日頻 → 月頻（取月底最後一筆）。trades 的 entry_date 是換股日（月頻），
    # 之後用 entry_date.to_period('M') 對進來即可。
    monthly = raw.resample("M").last()
    monthly.index = monthly.index.to_period("M")

    # 橫斷面分位：每個月各自對當月有值的股票排名 → 0–100
    pct = monthly.rank(axis=1, pct=True, na_option="keep") * 100.0

    n_valid = int(pct.notna().sum().sum())
    log(f"[{market}] 月頻分位表 {pct.shape}，有效格 {n_valid:,}，"
        f"期間 {pct.index.min()} ~ {pct.index.max()}，{time.time()-t0:.0f}s")

    try:
        md.release()
    except Exception:
        pass
    return pct.astype("float32")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.stage1_mktcap")
    ap.add_argument("--market", choices=C.MARKETS, help="只跑單一市場（預設兩個都跑）")
    a = ap.parse_args(argv)

    paths.ensure_dirs()
    markets = [a.market] if a.market else list(C.MARKETS)
    outs = []
    for m in markets:
        pct = build_market(m)
        out = paths.STAGE1 / f"mktcap_pct_{m}.parquet"
        pct.to_parquet(out, compression="zstd")
        outs.append(out)
        print(f"  → {out.name}  {out.stat().st_size/1024/1024:.1f} MB")

    if len(markets) == len(C.MARKETS):
        freeze.write_manifest(
            "stage1_mktcap", paths.STAGE1 / "_mktcap",
            inputs=[], outputs=outs,
            params={"field": FIELD, "freq": "M", "smallcap_pct_max": SMALLCAP_PCT_MAX,
                    "in_sample_end": paths.IN_SAMPLE_END},
            notes="C-5 規模欄位的原料；抽出成獨立前置，讓主掃描不依賴 DB 連線",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
