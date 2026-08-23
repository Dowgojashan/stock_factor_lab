# -*- coding: utf-8 -*-
"""階段 2c · 交叉佐證（2a regime × 2b 總經四格 匯流，W-11）

輸入 ← `_frozen/stage2/regime/regime_table_{market}.parquet`（2a）
       `_frozen/stage2/macro/macro_history_{market}.parquet`（2b，只用 clock_cell 欄）
輸出 → `_frozen/stage2/consistency/regime_consistency_{market}.parquet`（給階段3/4）
       `_frozen/stage2/consistency/regime_divergence_{market}.csv`（背離時段清單，人讀用）

⚠️ **只事後核對，不改動 2a/2b 內部計算**（研究部 v9 §改動3「做法甲：先切後驗」）：
   - 2a 判「熊/危機」的段，核對是否**多落在** 2b 的「停滯性通膨／衰退」格（低成長）；
   - 2a 判「牛」的段，核對是否**多落在** 2b 的「復甦／過熱」格（高成長）；
   - **盤整不檢查**：盤整本身無方向性預期，勉強配一個「該落在哪格」的期待只是自欺；
   - 「多落在」＝該段內非空月份中，落入預期格的比例 > 50%（多數決，見 contracts.
     REGIME_EXPECTED_CELLS）。
   - **不一致（背離）不修改 regime，只標記**——這是做法甲的立場（價格為主、總經
     為輔），背離是「待解釋的現象」，必須在論文明講以答口委「發現背離為何不修 regime」。

⚠️ **輸出目錄的解讀**：研究部 v9 文件把 2c 的產物歸在「regime/（全域共用）」底下描述，
   但 `regime/` 已經是 2a 的 `write_manifest` 輸出目錄——若 2c 也把 MANIFEST.json 寫進
   同一個資料夾，會**覆蓋 2a 的 manifest**，導致 `freeze.verify_inputs(STAGE2/"regime")`
   之後只認得到 2c 的產物、認不到 2a 的（凍結鏈斷裂，違反 DD-08 的精神）。故本階段
   產物改放獨立的 `consistency/` 子目錄，語意仍是「regime 相關產物」，只是每個子階段
   各自持有自己的 manifest（跟 2a→regime/、2b→macro/ 是同一個模式）。

用法：
    cd code
    python -m research.stage2c_consistency
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import contracts as C
from . import freeze, paths

CONSISTENCY_DIR = paths.STAGE2 / "consistency"

#: 「多落在」的多數決門檻（>50%）
MAJORITY_THRESHOLD = 0.50


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.PeriodIndex:
    """段的起訖日 -> 涵蓋的月份清單（含頭尾月）。"""
    return pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")


def _majority_cell(cells: pd.Series) -> str | None:
    """眾數格。平手時取字母序最小者，確保結果可重現（不依賴 value_counts 的隱含排序）。"""
    if cells.empty:
        return None
    counts = cells.value_counts()
    top = counts[counts == counts.max()].index
    return sorted(top)[0]


def check_segments(regime: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """核對每個 regime 段是否落在預期的總經格（規則見模組開頭）。

    對每一段，把 [start, end] 展開成月份清單，去 macro_history 撈對應月份的
    clock_cell（缺值月份自然被排除，不用特別處理——2b 的月頻表本身就可能有
    NaN，reindex 後 dropna 即可）。
    """
    cell_by_month = macro.set_index("month")["clock_cell"]
    rows = []
    for seg in regime.itertuples():
        months = month_range(seg.start, seg.end)
        cells = cell_by_month.reindex(months).dropna()
        n_valid = len(cells)
        majority = _majority_cell(cells)
        expected = C.REGIME_EXPECTED_CELLS.get(seg.label)
        checked = expected is not None

        n_matched = pct = consistent = None
        if checked and n_valid:
            n_matched = int(cells.isin(expected).sum())
            pct = 100.0 * n_matched / n_valid
            consistent = pct > MAJORITY_THRESHOLD * 100

        rows.append({
            "market": seg.market, "seg_start": seg.start, "seg_end": seg.end,
            "label": seg.label, "checked": checked,
            "expected_group": "|".join(expected) if expected else None,
            "n_months_valid": n_valid,
            "n_months_matched": float(n_matched) if n_matched is not None else None,
            "pct_match": pct, "majority_cell": majority,
            "consistent": consistent,
        })
    out = pd.DataFrame(rows)
    out["consistent"] = out["consistent"].astype("boolean")   # nullable bool：None=未檢查/無資料
    return out


def run(log=print) -> dict[str, pd.DataFrame]:
    CONSISTENCY_DIR.mkdir(parents=True, exist_ok=True)
    results, outs, ins = {}, [], []
    for m in C.MARKETS:
        rp = paths.STAGE2 / "regime" / f"regime_table_{m}.parquet"
        mp = paths.STAGE2 / "macro" / f"macro_history_{m}.parquet"
        if not rp.exists():
            raise FileNotFoundError(f"找不到 {rp}，請先跑 python -m research.stage2a_regime")
        if not mp.exists():
            raise FileNotFoundError(f"找不到 {mp}，請先跑 python -m research.stage2b_macro")
        ins += [rp, mp]

        regime = pd.read_parquet(rp)
        C.validate(regime, C.REGIME_TABLE)
        macro = pd.read_parquet(mp)
        C.validate(macro, C.MACRO_HISTORY)

        out = check_segments(regime, macro)
        C.validate(out, C.REGIME_CONSISTENCY)

        checked = out[out.checked]
        n_skipped_consolidation = int((~out.checked).sum())
        judged = checked[checked["consistent"].notna()]
        n_no_data = len(checked) - len(judged)
        n_ok = int(judged["consistent"].sum())
        n_judged = len(judged)

        log(f"[{m}] regime 段共 {len(out)}：盤整不檢查 {n_skipped_consolidation}，"
            f"無總經資料可比對 {n_no_data}，可判定 {n_judged}")
        if n_judged:
            log(f"[{m}]   段數計：{n_ok}/{n_judged} 一致（{n_ok/n_judged:.0%}）")
            w_ok = judged.loc[judged["consistent"], "n_months_valid"].sum()
            w_tot = judged["n_months_valid"].sum()
            log(f"[{m}]   月數加權：{int(w_ok)}/{int(w_tot)} 個月一致"
                f"（{w_ok/w_tot:.0%}，長段不該跟短段等權，故另算一次）")
        else:
            log(f"[{m}]   無可判定的段")

        div = judged[~judged["consistent"]].sort_values("seg_start")
        log(f"[{m}]   背離段 {len(div)} 筆")
        if len(div):
            log(div[["seg_start", "seg_end", "label", "majority_cell", "pct_match"]]
                .to_string(index=False))

        p = CONSISTENCY_DIR / f"regime_consistency_{m}.parquet"
        out.to_parquet(p, compression="zstd", index=False)
        outs.append(p)
        dp = CONSISTENCY_DIR / f"regime_divergence_{m}.csv"
        div.to_csv(dp, index=False, encoding="utf-8-sig")
        outs.append(dp)
        results[m] = out

    freeze.write_manifest(
        "stage2c_consistency", CONSISTENCY_DIR,
        inputs=ins, outputs=outs,
        params={"majority_threshold": MAJORITY_THRESHOLD,
               "expected_cells": C.REGIME_EXPECTED_CELLS},
        notes="只事後核對 2a×2b，不改動兩者內部計算；背離段保留不修，僅標記（做法甲）。"
              "consistency/ 是獨立子目錄，理由見模組開頭說明（避免覆蓋 2a 的 regime/ manifest）")
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.stage2c_consistency")
    ap.parse_args(argv)
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
