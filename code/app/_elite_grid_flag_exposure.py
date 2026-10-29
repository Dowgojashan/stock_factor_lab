# -*- coding: utf-8 -*-
"""優等生網格：被標記股票（單季>+300% 或單日漲跌>50%）的權重曝險（2026-10-27）。

`_elite_grid_offline.py` 的壞資料防線②只標記不剔除（用未來報酬剔除＝前視）。
美股標記量遠多於台股（US baseline 333筆，其中321筆是單日>50%，像反向分割未還原
或資料錯誤），這支量化每一組規則在每一季持有多少「被標記股票」的權重，
供敏感度分析：主結果照事先寫死的規則不變，另外報「排除高曝險組合後」的結論是否改變。

權重重建：final = (1−資金比例)×w0 + 資金比例×Σw0×elite（同 elite_rules.overlay），
用 `w0_*.parquet` 與 `elite_weights_*.parquet` 還原，不重跑網格。
對照組 control0 曝險＝w0；每檔等權／市值傾斜全開不在這裡重建（另列為 NaN）。

用法（cwd=code/）：python -m app._elite_grid_flag_exposure --market US --pool baseline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from app._elite_grid_offline import OUT_DIR  # noqa: E402


def run(market: str, pool: str, suffix: str = "") -> pd.DataFrame:
    tag = f"{market}_{pool}{suffix}"   # suffix="_liq"＝美股流動性門檻版
    flags = pd.read_csv(OUT_DIR / f"sanity_flagged_{tag}.csv", dtype={"symbol": str, "as_of": str})  # 台股代號會被讀成整數，對不上 parquet 的字串
    flagged = set(zip(flags.window, flags.as_of, flags.symbol))
    w0 = pd.read_parquet(OUT_DIR / f"w0_{tag}.parquet")
    el = pd.read_parquet(OUT_DIR / f"elite_weights_{tag}.parquet")
    combos = pd.read_csv(OUT_DIR / f"combos_{tag}.csv")

    w0["flag"] = [k in flagged for k in zip(w0.window, w0.as_of, w0.symbol)]
    el["flag"] = [k in flagged for k in zip(el.window, el.as_of, el.symbol)]
    w0_tot = w0.groupby(["window", "as_of"]).w0.sum()
    w0_flag = w0[w0.flag].groupby(["window", "as_of"]).w0.sum().reindex(w0_tot.index, fill_value=0.0)
    el_flag = (el[el.flag].groupby(["window", "as_of", "score", "n", "scheme"]).elite_weight.sum())

    rows = []
    quarters = w0_tot.index
    for c in combos.itertuples():
        for (w, a) in quarters:
            if c.score == "REF_control0":
                exp = w0_flag[(w, a)]
            elif c.score.startswith("REF_"):
                exp = float("nan")
            else:
                ef = el_flag.get((w, a, c.score, c.n, c.scheme), 0.0)
                exp = (1 - c.sleeve) * w0_flag[(w, a)] + c.sleeve * w0_tot[(w, a)] * ef
            rows.append({"combo": c.combo, "window": w, "as_of": a, "flag_exposure": exp})
    df = pd.DataFrame(rows)
    summ = df.groupby("combo").flag_exposure.agg(["mean", "max"]).rename(
        columns={"mean": "flag_exp_mean", "max": "flag_exp_max"})
    summ.to_csv(OUT_DIR / f"flag_exposure_{tag}.csv", encoding="utf-8-sig")
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True, choices=["TW", "US"])
    ap.add_argument("--pool", required=True, choices=["exclude_v1", "baseline"])
    ap.add_argument("--suffix", default="", help="例：_liq")
    a = ap.parse_args(argv)
    s = run(a.market, a.pool, a.suffix)
    print(s.describe().round(4).to_string())
    for thr in (0.01, 0.05, 0.10):
        print(f"單季曝險最大值 > {thr:.0%} 的組合數：{int((s.flag_exp_max > thr).sum())}")


if __name__ == "__main__":
    main()
