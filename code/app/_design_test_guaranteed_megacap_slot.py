# -*- coding: utf-8 -*-
"""D71後續item(b)：設計並測試一個新的配額規則——「保底名額」（guaranteed slot），
不是取代既有equal/proportional分配，是額外疊加：每個群多保留1個名額，專門給
「該群裡Calmar品質最高、且會選中前十大權值股中至少一檔」的候選人（如果該群
有這種候選人的話，沒有就不硬湊）。

設計原則（避免變成「為了塞台積電硬調」）：
  - 🔴 2026-09-22查證：業界有正當先例——CFA Level III課綱裡「passive
    portfolio construction」的標準方法之一是stratified sampling（分層
    抽樣），原文「for the largest-cap portion of an indexed portfolio,
    full replication is a sensible and desirable approach...stratified
    sampling typically ensures that the largest constituents are fully
    held」——分層抽樣對「指數前N大成分股」設保底覆蓋率下限，是業界／
    考試課綱認可的標準追蹤誤差控制手法，不是本專案發明的取巧規則（完整
    查證過程見`文件/實戰開發追蹤_v2.md`§2.7；先前這句話沒附來源，屬於
    未查證的斷言，這次補上）。
  - 目標是「前十大權值股整體覆蓋率」，不是專門為台積電量身訂做。
  - 不取代現有5個依Calmar品質排序＋多樣性篩選的名額，是額外加1個保底名額——
    現有機制的行為完全不變，只是多了一道覆蓋率保險。
  - 沒有候選人就不硬湊（跟原本select_representatives()「候選不足就不強行湊數」
    同一個設計慣例）。

用同一批window4的6,679檔樹成員（`tsmc_cluster_concentration.csv`已存的cluster
指派），重新查每個策略對前十大權值股（2023-12-31排名，2330/2454/2317/2881/
2412/2382/2308/6505/2882/2303）的選中狀態，設計新規則後比較覆蓋率變化。
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

TOP10 = ["2330", "2454", "2317", "2881", "2412", "2382", "2308", "6505", "2882", "2303"]
REF_DATE = "2023-12-31"
CLUSTER_CSV = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "tsmc_cluster_concentration.csv"


def asof_bool(mask: pd.DataFrame, as_of: str, symbol: str) -> bool | None:
    if mask is None or symbol not in mask.columns:
        return None
    ts = pd.Timestamp(as_of)
    valid = mask.index[mask.index <= ts]
    if len(valid) == 0:
        return None
    return bool(mask.loc[valid.max(), symbol])


def selects_symbol(md: MarketData, row: pd.Series, as_of: str, symbol: str) -> bool | None:
    f1_mask = md.get_mask(_q_band_condition(row["F1_factor"], row["F1_band"], row["F1_nbands"]))
    f1 = asof_bool(f1_mask, as_of, symbol)
    if f1 is not True:
        return f1
    if not row["F2_empty"]:
        f2_mask = md.get_mask(_q_band_condition(row["F2_factor"], row["F2_band"], row["F2_nbands"]))
        f2 = asof_bool(f2_mask, as_of, symbol)
        if f2 is not True:
            return f2
    if pd.notna(row["C_rule"]):
        c_mask = md.get_mask(_c_condition(row["C_source"], row["C_rule"]))
        c = asof_bool(c_mask, as_of, symbol)
        if c is not True:
            return c
    if row["V"] == "v1":
        v_mask = md.get_v_mask()
        v = asof_bool(v_mask, as_of, symbol)
        if v is not True:
            return v
    return True


def main():
    cluster_df = pd.read_csv(CLUSTER_CSV).set_index("strategy_uid")
    idx_full = pd.read_parquet(CANDIDATE_INDEX_PATH)
    idx = idx_full[idx_full["market"] == "TW"].set_index("strategy_uid")
    idx = idx.loc[idx.index.intersection(cluster_df.index)]
    print(f"樹內策略數：{len(idx)}")

    print(">> 載入 TW MarketData ...")
    md = MarketData("TW")

    for sym in TOP10:
        idx[f"sel_{sym}"] = [selects_symbol(md, row, REF_DATE, sym) is True for _, row in idx.iterrows()]
    idx["n_top10_selected"] = idx[[f"sel_{s}" for s in TOP10]].sum(axis=1)
    idx["selects_any_top10"] = idx["n_top10_selected"] > 0

    idx = idx.join(cluster_df[["cluster_L1"]])
    idx["calmar"] = pd.NA  # 用candidate_index已存的CAGR/max_drawdown當品質分數，跟H-10同口徑
    full = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    idx["CAGR"] = full.loc[idx.index, "CAGR"]
    idx["max_drawdown"] = full.loc[idx.index, "max_drawdown"]
    idx["calmar"] = idx["CAGR"] / idx["max_drawdown"].abs()

    # 讀既有legacy equal的實際挑選結果（5個/群，Calmar排序＋多樣性篩選）
    members_path = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"
    m = pd.read_parquet(members_path)
    key = dict(tree_key="TW", scheme="E", window_no=4, k_mode="silhouette_is",
              ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    existing_picks = set(sub.iloc[0]["members"])
    idx["is_existing_pick"] = idx.index.isin(existing_picks)

    print(f"\n=== 現行規則（equal，5個/群，共30檔）的前十大權值股覆蓋 ===")
    existing_df = idx[idx["is_existing_pick"]]
    print(f"  30檔代表裡，會選中前十大權值股至少一檔的：{existing_df['selects_any_top10'].sum()}/30")
    for sym in TOP10:
        n = existing_df[f"sel_{sym}"].sum()
        print(f"    {sym}：{n}/30")

    print("\n=== 新規則：每群額外加1個保底名額（Calmar最高且選中前十大權值股任一檔者）===")
    new_picks = set(existing_picks)
    added = []
    for cid, g in idx.groupby("cluster_L1"):
        candidates = g[g["selects_any_top10"] & ~g["is_existing_pick"]].dropna(subset=["calmar"])
        if candidates.empty:
            print(f"  群{cid}：沒有額外候選人（該群裡除了已選的，沒有其他選中前十大權值股的策略）")
            continue
        best = candidates.sort_values("calmar", ascending=False).iloc[0]
        new_picks.add(best.name)
        added.append((cid, best.name, best["calmar"]))
        selected_syms = [s for s in TOP10 if best[f"sel_{s}"]]
        print(f"  群{cid}：新增 {best.name}（Calmar={best['calmar']:.3f}，"
             f"選中：{selected_syms}）")

    new_df = idx.loc[list(new_picks)]
    print(f"\n=== 新規則後（{len(new_picks)}檔，原30檔+{len(added)}個保底名額）===")
    print(f"  會選中前十大權值股至少一檔的：{new_df['selects_any_top10'].sum()}/{len(new_picks)}")
    for sym in TOP10:
        n = new_df[f"sel_{sym}"].sum()
        print(f"    {sym}：{n}/{len(new_picks)}")

    out = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "guaranteed_megacap_slot_test.csv"
    idx.to_csv(out, encoding="utf-8-sig")
    print(f"\n寫入 {out}")


if __name__ == "__main__":
    main()
