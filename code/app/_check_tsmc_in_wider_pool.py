# -*- coding: utf-8 -*-
"""便宜的前置檢查（2026-09-22，使用者要求）：window 4 目前用 ratio="legacy"
（30個代表策略）——放大到 1%/3%/5%/10%（凍結矩陣裡本來就有算好的名單，
不需要重跑HRP建樹）之後，候選池裡有沒有策略會選到台積電？

這是決定「放寬比例」這個方向值不值得繼續做完整績效重跑的前置診斷：如果連最大
的10%（668檔）都完全沒有策略選台積電，代表HRP分群本身（依報酬序列相似度分群，
不是依因子類型分群）在這個群裡就是清一色價值型策略，放大比例也無濟於事；
如果有，才值得往下做完整的8季績效比較。

跟 `_check_tsmc_exclusion_reason.py` 同一套機制（`condition_factory`重建F1/F2/C/V
遮罩），差別只是候選名單從30檔換成各種ratio的名單，且只關心「選中與否」不需要
逐關拆解（那件事已經在legacy版本查清楚了）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import (CANDIDATE_INDEX_PATH, _c_condition,  # noqa: E402
                                       _q_band_condition)

TSMC = "2330"
CHECK_DATES = ["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
              "2025-03-31", "2025-06-30", "2025-09-30"]
MEMBERS_PATH = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"
RATIOS = ["legacy", "0.01", "0.03", "0.05", "0.1"]


def get_members(ratio: str) -> list[str]:
    m = pd.read_parquet(MEMBERS_PATH)
    key = dict(tree_key="TW", scheme="E", window_no=4, k_mode="silhouette_is",
              ratio=ratio, allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    assert len(sub) == 1, f"ratio={ratio} 找不到唯一一列：{len(sub)}"
    return list(sub.iloc[0]["members"])


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
        return f1  # False 或 None（算不出來），提早結束省算力
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
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    print(">> 載入 TW MarketData ...")
    md = MarketData("TW")

    members_by_ratio = {r: get_members(r) for r in RATIOS}
    for r in RATIOS:
        print(f"  ratio={r:<8s} n={len(members_by_ratio[r])}")

    rows = []
    for ratio in RATIOS:
        uids = members_by_ratio[ratio]
        ever_selected: set[str] = set()
        for d in CHECK_DATES:
            n_sel = 0
            for uid in uids:
                row = idx.loc[uid]
                sel = strategy_selects_tsmc(md, row, d)
                if sel is True:
                    n_sel += 1
                    ever_selected.add(uid)
            rows.append({"ratio": ratio, "n_pool": len(uids), "as_of": d, "n_selected_tsmc": n_sel})
            print(f"  ratio={ratio:<8s} {d}：{n_sel}/{len(uids)} 選中台積電")
        print(f"  ratio={ratio:<8s} 8季累計「曾經選過台積電」的不重複策略數："
             f"{len(ever_selected)}/{len(uids)}（{len(ever_selected)/len(uids):.2%}）")

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "tsmc_wider_pool_check.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out}")


if __name__ == "__main__":
    main()
