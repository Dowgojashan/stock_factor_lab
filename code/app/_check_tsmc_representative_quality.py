# -*- coding: utf-8 -*-
"""D65延伸項目1（2026-09-22，使用者要求）：找出HRP代表挑選為什麼會把「候選池
對台積電的低配」放大到接近完全排除。

已從程式碼查清楚機制（`research/walkforward_matrix.py::_pick_a` 呼叫
`research/cluster_representatives.py::select_representatives`）：**每個群內部
依品質分數（Calmar = IS期CAGR / |IS期MDD|）由高到低貪婪納入代表，只有品質最高
的候選者才會被考慮，多樣性門檻只負責剔除「跟已選入者過度相關」的候選，不會
把低品質但分散的候選拉進來**——這套機制本身沒有look-ahead（quality_is只用該窗
IS資料），不是bug，是設計上的必然結果：如果會選中台積電的策略，在window4的IS
期（2024年之前）的Calmar排名本來就不是該群最頂尖的，就會被排除，不管多樣性
與否。

這裡做的是**經驗驗證**（不是重建window4那棵樹的精確版，那需要重跑整個相關矩陣，
成本高很多——見開發追蹤）：用 `candidate_index.parquet` 已經存好的
CAGR／max_drawdown（Phase1-4正式回測的全樣本績效，非window4專屬IS期，只能當
方向性代理指標，不是精確重現），比較「會選中台積電的策略」vs「其餘策略」的
Calmar分布，檢查方向是否支持「品質較低所以被排除」這個假設。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import (CANDIDATE_INDEX_PATH, _c_condition,  # noqa: E402
                                       _q_band_condition)

TSMC = "2330"
CHECK_DATES = ["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
              "2025-03-31", "2025-06-30", "2025-09-30"]


def asof_bool(mask: pd.DataFrame, as_of: str, symbol: str) -> bool | None:
    if mask is None or symbol not in mask.columns:
        return None
    ts = pd.Timestamp(as_of)
    valid = mask.index[mask.index <= ts]
    if len(valid) == 0:
        return None
    return bool(mask.loc[valid.max(), symbol])


def strategy_selects_tsmc(md: MarketData, row: pd.Series, as_of: str) -> bool | None:
    f1_mask = md.get_mask(_q_band_condition(row["F1_factor"], row["F1_band"], row["F1_nbands"]))
    f1 = asof_bool(f1_mask, as_of, TSMC)
    if f1 is not True:
        return f1
    if not row["F2_empty"]:
        f2_mask = md.get_mask(_q_band_condition(row["F2_factor"], row["F2_band"], row["F2_nbands"]))
        f2 = asof_bool(f2_mask, as_of, TSMC)
        if f2 is not True:
            return f2
    if pd.notna(row["C_rule"]):
        c_mask = md.get_mask(_c_condition(row["C_source"], row["C_rule"]))
        c = asof_bool(c_mask, as_of, TSMC)
        if c is not True:
            return c
    if row["V"] == "v1":
        v_mask = md.get_v_mask()
        v = asof_bool(v_mask, as_of, TSMC)
        if v is not True:
            return v
    return True


def main():
    idx_full = pd.read_parquet(CANDIDATE_INDEX_PATH)
    idx = idx_full[idx_full["market"] == "TW"].set_index("strategy_uid")
    print(f"全部台股候選策略數：{len(idx)}")

    print(">> 載入 TW MarketData ...")
    md = MarketData("TW")

    ever_selected: set[str] = set()
    for d in CHECK_DATES:
        for uid, row in idx.iterrows():
            if strategy_selects_tsmc(md, row, d) is True:
                ever_selected.add(uid)

    idx["calmar"] = (idx["CAGR"] / idx["max_drawdown"].abs()).replace(
        [np.inf, -np.inf], np.nan)
    idx["selects_tsmc"] = idx.index.isin(ever_selected)

    print(f"\n曾經選過台積電的策略數：{len(ever_selected)}/{len(idx)}"
         f"（{len(ever_selected)/len(idx):.2%}）")

    grp = idx.groupby("selects_tsmc")["calmar"]
    print("\n=== Calmar（CAGR/|MDD|，全樣本回測績效，非window4專屬IS期）分布對照 ===")
    for k, g in grp:
        label = "會選台積電" if k else "不會選台積電"
        print(f"  {label}（n={len(g)}）：中位數={g.median():.3f}　"
             f"平均={g.mean():.3f}　p25={g.quantile(0.25):.3f}　p75={g.quantile(0.75):.3f}")

    sel = idx[idx["selects_tsmc"]]["calmar"].dropna()
    rest = idx[~idx["selects_tsmc"]]["calmar"].dropna()
    from scipy import stats
    u_stat, p_value = stats.mannwhitneyu(sel, rest, alternative="two-sided")
    print(f"\nMann-Whitney U 檢定（Calmar分布是否有系統性差異）：p={p_value:.4f}"
         f"　{'有顯著差異' if p_value < 0.05 else '無顯著差異'}")

    out = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "tsmc_representative_quality.csv"
    idx[["market", "F1_factor", "CAGR", "max_drawdown", "calmar", "selects_tsmc"]].to_csv(
        out, encoding="utf-8-sig")
    print(f"\n寫入 {out}")


if __name__ == "__main__":
    main()
