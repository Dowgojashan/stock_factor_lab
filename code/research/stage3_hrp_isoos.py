# -*- coding: utf-8 -*-
"""H-11 · IS/OOS（in-sample/out-of-sample）分群結構驗證（開發待辦追蹤.md 第四階段）

老師的關鍵指引：**切在投組配置層，不是策略參數層**——策略池（F×C×V定義）完全不動，
只有「用哪一段月份去建HRP樹、算群間相關」這件事分成 IS/OOS 兩段。

⚠️ **這支完全獨立於 `stage3_hrp.py` 的主線六棵樹**（`_frozen/stage3/`）：
  - 主線六棵樹用的是**全時間窗**（`contracts.HRP_WINDOWS`，TW/XM 2007-2025、
    US 2002-2025），是 H-01~H-10 一路沿用至今的正式產物，**本模組完全不動它、
    不重跑它、不覆寫它**。
  - 本模組另外用**IS窗**（`contracts.HRP_IS_WINDOWS`，H-11定案v2）建一批**全新的、
    獨立命名空間**的樹（tree_id 加 `_IS` 後綴），輸出到**獨立目錄**
    `_frozen/stage3_isoos/`（見 paths.STAGE3_ISOOS），不寫進 `_frozen/stage3/`，
    不共用檔名、不共用MANIFEST，兩批資料不會混淆。

**做法（H-11拍板的流程）**：
  1. 只用 IS 窗月份，重跑一次三棵 normal 樹的完整 HRP 流程（相關→距離→linkage→
     準對角化→遞迴二分→切L1/L3）——群的定義（哪個策略屬於哪個群）**在這裡凍結**。
  2. **不對 OOS 月份重新分群**——群的定義原封不動沿用步驟1凍結的結果，只是換一段
     月份（2019-01~2025-12）重新計算「這批已經固定的群，彼此之間的相關係數」。
  3. 比較同一組群對，在 IS 窗跟 OOS 窗算出來的相關係數是否維持一致的（低）相關——
     這是 H-13「in sample都很低，可是out sample會不會還是很低」的核心素材。

只做 L1、只做 normal 樹（crisis 樹樣本量太小、H-14/H-16 已定案不切 IS/OOS，
跟 H-06/H-07/cluster_story 同樣的理由）。

用法：
    cd code
    python -m research.stage3_hrp_isoos
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, paths
from . import stage3_hrp as S3

LEVEL = "L1"   # 比較表只做L1，跟cluster_story/H-06/H-07同樣的理由


def _complementarity(corr: float) -> str:
    """跟 cluster_story._complementarity 同一套判定規則（程式判定，不交給LLM）。"""
    if corr < C.COMPLEMENTARITY_CUTS["高"]:
        return "高"
    if corr < C.COMPLEMENTARITY_CUTS["中"]:
        return "中"
    return "低"


def _load_usable_meta() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """跟 stage3_hrp.run() 同一段資料準備：usable_pool 過濾 + F組合對照表。"""
    months_long = pd.read_parquet(paths.STAGE1 / "returns_monthly.parquet")
    meta = pd.read_parquet(paths.STAGE1 / "returns_meta.parquet")
    idx = pd.read_parquet(paths.STAGE0 / "candidate_index.parquet")
    f_combo_map = (idx.market.astype(str) + "::" + idx.f_combo.astype(str))
    f_combo_map.index = idx.strategy_uid

    marks = pd.read_parquet(paths.STAGE1 / "strategy_marks.parquet")
    usable = set(marks.loc[marks.is_usable, C.PK])
    meta = meta[meta.strategy_uid.isin(usable)]
    return months_long, meta, f_combo_map


def build_is_tree(tree_key: str, months_long: pd.DataFrame, meta: pd.DataFrame,
                  f_combo_map: pd.Series, log=print) -> dict:
    """只用IS窗月份，跑一棵完整的normal樹——群定義在這裡凍結。"""
    is_start, is_end = C.HRP_IS_WINDOWS[tree_key]
    # 通用策略池的篩選門檻是「hist_start <= 窗起點」，IS窗起點跟HRP_WINDOWS的
    # 全時間窗起點相同（TW/XM 2007-01、US 2002-01），所以合格的策略宇宙跟主線
    # 六棵樹**完全一樣**——差別只在於算相關矩陣時用的月份少了（截到2018-12）。
    uids = S3._tree_universe(tree_key, is_start, meta)
    log(f"[{tree_key}_normal_IS] IS窗 {is_start}~{is_end}｜策略數 {len(uids):,}"
        f"（與全時間窗宇宙相同，僅月份範圍不同）")
    wide = S3._pivot_window(months_long, uids, is_start, is_end)
    return S3._build_tree(f"{tree_key}_normal_IS", tree_key, wide, f_combo_map, log)


def compute_oos_group_corr(tree_id_is: str, tree_key: str, assign_is: pd.DataFrame,
                           months_long: pd.DataFrame, log=print) -> pd.DataFrame:
    """群定義完全沿用IS窗凍結的結果，只用OOS窗月份重算「這批固定的群」彼此的相關。

    跟 `stage3_hrp._cluster_meta_and_corr` 的群代表序列定義（成員報酬簡單平均）
    口徑一致，只是這裡的月份換成OOS窗、群成員名單是外部傳入的凍結結果。
    """
    oos_start, oos_end = C.HRP_OOS_WINDOW
    a = assign_is[assign_is.tree_id == tree_id_is][[C.PK, f"cluster_{LEVEL}"]].rename(
        columns={f"cluster_{LEVEL}": "_cl"})
    uids = a[C.PK]

    w = months_long[months_long.strategy_uid.isin(set(uids))]
    w = w[(w.month >= pd.Period(oos_start, "M")) & (w.month <= pd.Period(oos_end, "M"))]
    wide_oos = w.pivot(index="strategy_uid", columns="month", values="ret")

    n_missing = len(set(uids) - set(wide_oos.index))
    if n_missing:
        log(f"  ⚠️ {n_missing} 個IS窗策略在OOS窗完全沒有報酬資料（不應發生，"
            f"DD-03共同窗保證涵蓋到2025-12），已排除於OOS相關矩陣外")

    group_mean_series = {}
    for cid, g in a.groupby("_cl"):
        members = [u for u in g[C.PK] if u in wide_oos.index]
        if not members:
            continue
        rep = wide_oos.loc[members].mean(axis=0)
        group_mean_series[int(cid)] = rep

    gids = sorted(group_mean_series)
    means = pd.DataFrame({g: group_mean_series[g] for g in gids})
    gcorr = means.corr()
    log(f"[{tree_id_is}] OOS窗({oos_start}~{oos_end})群間相關矩陣 {gcorr.shape}"
        f"｜{means.shape[0]}個共同月份")
    return gcorr


def build_comparison(tree_id_is: str, corr_is: pd.DataFrame, corr_oos: pd.DataFrame) -> pd.DataFrame:
    """長表：每對群在IS/OOS的相關係數、互補程度判定是否一致。"""
    rows = []
    ids = sorted(corr_is.columns)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            c_is = float(corr_is.loc[a, b])
            comp_is = _complementarity(c_is)
            if a in corr_oos.columns and b in corr_oos.columns:
                c_oos = float(corr_oos.loc[a, b])
                comp_oos = _complementarity(c_oos)
                delta = c_oos - c_is
                stable = (comp_is == comp_oos)
            else:
                c_oos, comp_oos, delta, stable = None, None, None, None
            rows.append({"tree_id": tree_id_is, "level": LEVEL, "cluster_a": a, "cluster_b": b,
                        "corr_is": round(c_is, 6),
                        "corr_oos": round(c_oos, 6) if c_oos is not None else None,
                        "delta": round(delta, 6) if delta is not None else None,
                        "complementarity_is": comp_is, "complementarity_oos": comp_oos,
                        "complementarity_stable": stable})
    return pd.DataFrame(rows)


def run(trees=("TW", "US", "XM"), log=print) -> dict[str, pd.DataFrame]:
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE1 / "_marks")
    months_long, meta, f_combo_map = _load_usable_meta()

    is_results, comparisons = {}, []
    for tk in trees:
        is_results[tk] = build_is_tree(tk, months_long, meta, f_combo_map, log)

    assign_is = pd.concat([r["assign"] for r in is_results.values()], ignore_index=True)
    assign_is["tree_id"] = assign_is["tree_id"].astype("category")
    meta_is = pd.concat([r["cluster_meta"] for r in is_results.values()], ignore_index=True)
    meta_is["tree_id"] = meta_is["tree_id"].astype("category")
    meta_is["level"] = meta_is["level"].astype("category")

    C.validate(assign_is, C.CLUSTER_ASSIGN_ISOOS)
    C.validate(meta_is, C.CLUSTER_META_ISOOS)
    log("✓ cluster_assign_isoos / cluster_meta_isoos 契約通過")

    for tk in trees:
        tree_id_is = f"{tk}_normal_IS"
        corr_is = is_results[tk]["corr_l1"]
        corr_oos = compute_oos_group_corr(tree_id_is, tk, assign_is, months_long, log)
        comparisons.append(build_comparison(tree_id_is, corr_is, corr_oos))

    comparison = pd.concat(comparisons, ignore_index=True)
    comparison["tree_id"] = comparison["tree_id"].astype("category")
    comparison["level"] = comparison["level"].astype("category")
    comparison["complementarity_is"] = comparison["complementarity_is"].astype("category")
    comparison["complementarity_oos"] = comparison["complementarity_oos"].astype("category")
    C.validate(comparison, C.ISOOS_CORR_COMPARISON, strict_columns=True)
    log("✓ isoos_corr_comparison 契約通過")

    paths.STAGE3_ISOOS.mkdir(parents=True, exist_ok=True)
    outs = []
    for name, df in (("cluster_assign_IS.parquet", assign_is),
                     ("cluster_meta_IS.parquet", meta_is),
                     ("isoos_corr_comparison.parquet", comparison)):
        p = paths.STAGE3_ISOOS / name
        df.to_parquet(p, compression="zstd", index=False)
        outs.append(p)
    for tk in trees:
        tree_id_is = f"{tk}_normal_IS"
        p = paths.STAGE3_ISOOS / f"cluster_corr_matrix_IS_{tree_id_is}.parquet"
        is_results[tk]["corr_l1"].to_parquet(p, compression="zstd"); outs.append(p)
        p = paths.STAGE3_ISOOS / f"linkage_IS_{tree_id_is}.npy"
        np.save(p, is_results[tk]["link"]); outs.append(p)

    freeze.write_manifest(
        "stage3_hrp_isoos", paths.STAGE3_ISOOS,
        inputs=[paths.STAGE1 / "returns_monthly.parquet",
               paths.STAGE1 / "returns_meta.parquet",
               paths.STAGE1 / "strategy_marks.parquet",
               paths.STAGE0 / "candidate_index.parquet"],
        outputs=outs,
        params={"is_windows": {k: list(v) for k, v in C.HRP_IS_WINDOWS.items()},
               "oos_window": list(C.HRP_OOS_WINDOW),
               "trees_built": [f"{t}_normal_IS" for t in trees],
               "linkage_method_chosen": {t: is_results[t]["method"] for t in trees}},
        notes="H-11 v2：IS窗建樹凍結群定義，OOS窗只重算同一批群的相關係數（不重新分群）。"
              "完全獨立於_frozen/stage3/的主線六棵樹（全時間窗），不共用檔名/MANIFEST。",
    )
    log(f"→ {paths.STAGE3_ISOOS}")
    return {"assign_is": assign_is, "meta_is": meta_is, "comparison": comparison}


def _report(tables: dict[str, pd.DataFrame], log=print) -> None:
    comparison = tables["comparison"]
    log("\n" + "=" * 78)
    log("H-11 · IS/OOS 群間相關穩定度 驗收摘要")
    log("=" * 78)
    for tid, g in comparison.groupby("tree_id", observed=True):
        valid = g[g["complementarity_oos"].notna()]
        n_stable = int(valid["complementarity_stable"].sum())
        log(f"\n[{tid}] {len(g)} 對群｜可比較(OOS有算出){len(valid)}對"
            f"｜互補程度判定IS/OOS一致：{n_stable}/{len(valid) if len(valid) else 1}"
            f"（{n_stable/len(valid):.1%})" if len(valid) else "")
        log(f"  corr_is 範圍 {g['corr_is'].min():.3f}~{g['corr_is'].max():.3f}"
            f"｜corr_oos 範圍 {valid['corr_oos'].min():.3f}~{valid['corr_oos'].max():.3f}"
            if len(valid) else "")
        unstable = valid[~valid["complementarity_stable"]]
        if len(unstable):
            log(f"  判定不一致的群對：\n{unstable[['cluster_a','cluster_b','corr_is','corr_oos','complementarity_is','complementarity_oos']].to_string(index=False)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.stage3_hrp_isoos")
    ap.add_argument("--trees", nargs="+", default=["TW", "US", "XM"])
    a = ap.parse_args(argv)
    t0 = time.time()
    tables = run(trees=a.trees)
    print(f"\n總耗時 {time.time()-t0:.0f}s")
    _report(tables)
    return 0


if __name__ == "__main__":
    sys.exit(main())
