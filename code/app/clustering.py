# -*- coding: utf-8 -*-
"""L1 · 分群與代表挑選（應用層開發追蹤.md §7 的 P1）

把「建樹 → 分群 → 挑代表」這段從 `research/walkforward_matrix.py` 抽出來，
讓應用層可以對**任意 IS 窗**或**既有凍結樹**跑同一套流程。

🔴 **本檔案不實作任何演算法，全部呼叫研究部既有函式**（`stage3_hrp`／
`walkforward_matrix`／`cluster_representatives`）。這是刻意的：

  - 重用 > 重寫。本專案最危險的錯誤是「靜默對錯位」那一類（例如
    `walkforward_matrix.py:435` 用斷言鎖住的 linkage 葉節點數不變式），
    自己重寫一份等於把那些踩過的雷全部重踩一次。
  - ⚠️ **但這也決定了一致性測試能證明什麼**（應用層 §8-R14／G4 更正）：
    `verify_against_frozen()` 驗證的是**接線正確**（參數傳對、日期切對、
    排序一致），**不是「演算法正確」**——演算法是同一份程式碼，測不出來。
    不可對外宣稱成「驗證了演算法」。

兩條路徑：

  ① `build_window_tree()` ── 對任意 (is_start, is_end) 現場建樹。
     給「驗證模式」與一致性測試用。

  ② `load_mainline_tree()` ── 直接讀 `_frozen/stage3/` 的主線樹。
     🔴 **正式模式走這條**：`contracts.HRP_WINDOWS["TW"] = ("2007-01","2025-12")`
     ＝228 個月，正是正式模式的 IS 窗，那棵樹（`TW_normal`，k=6，6,679 檔）
     早就建好且 DD-08 凍結，不需要重建（應用層 §7.4b）。

⚠️ **權重一律等權**（應用層 §8-R2）：驗證過的 8,370 格全部是等權組合
（`walkforward_matrix.py:406` 的 `_portfolio_series` ＝ `mean(axis=0)`），
HRP 只用來分群與配額，**從來沒有拿來配權重**。`ops.tools.t9` 的 HRP 權重
只能當並列對照，不得當實際權重（否則 C2 的 8% 上限也失去依據）。

⚠️ **`allocation` 決定的是「各群分到幾個代表名額」**（`walkforward_matrix.py:539`
的 `allocate()`），不是投組權重。很容易望文生義搞錯。

用法：
    cd code
    python -m app.clustering verify --n 8          # 一致性驗收（對凍結表）
    python -m app.clustering mainline --ratio legacy --allocation equal
"""
from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import pandas as pd

from research import contracts as C
from research import hrp, paths
from research import stage3_hrp as S3
from research import walkforward_matrix as WF

LEVEL = WF.LEVEL          # "L1"
_MAINLINE_KIND = "normal"  # 連續完整共同窗；crisis 樹是危機月份子集，不是我們要的


# ============================================================================
# 資料結構
# ============================================================================

@dataclasses.dataclass
class Tree:
    """一棵樹 + 它的 IS 報酬矩陣，足以拿去挑代表。"""
    tree_key: str                 # "TW" / "US" / "XM"
    tree_id: str                  # "TW_normal" 或 "TW_wf_2007-01_2022-12"
    is_start: str
    is_end: str
    assign: pd.DataFrame          # [strategy_uid, cluster_{LEVEL}]
    cluster_meta: pd.DataFrame    # [cluster_id, n_members, avg_intra_corr, ...]
    wide_is: pd.DataFrame         # 策略 × 月，無 NaN（DD-03 保證）
    link: np.ndarray | None       # 主線樹讀不到 linkage，為 None（不影響挑代表）
    source: str                   # "frozen_mainline" | "rebuilt_window"

    @property
    def n_universe(self) -> int:
        return len(self.wide_is)

    @property
    def k(self) -> int:
        return int(self.assign[f"cluster_{LEVEL}"].nunique())


@dataclasses.dataclass
class Selection:
    """一組 (ratio × allocation × group) 挑出來的成員 + IS 診斷。"""
    members: list[str]
    group: str
    ratio: str
    allocation: str
    k_mode: str
    n_members: int
    n_backfilled: int
    n_capped_clusters: int
    target_total: int
    cluster_info: dict            # max_cluster_share / n_clusters / n_clusters_covered
    performance_is: dict          # is_cagr / is_mdd / is_sharpe（⚠️ 樣本內，見 R13）


# ============================================================================
# 建樹／載樹
# ============================================================================

def load_inputs(log=print):
    """讀 returns_monthly / meta / f_combo_map。回傳給下面每個函式共用。

    直接呼叫 `walkforward_matrix._load_inputs()`——它已經處理好 is_usable 過濾。
    """
    log("載入 returns_monthly / returns_meta / strategy_marks / candidate_index …")
    return WF._load_inputs()


def build_window_tree(tree_key: str, is_start: str, is_end: str,
                      months_long, meta, f_combo_map, log=print) -> Tree:
    """對任意 IS 窗現場建樹（驗證模式／一致性測試用）。

    完全委派給 `walkforward_matrix.build_tree_for_window()`，故與研究部主線
    在同一輸入下必然得到同一棵樹。
    """
    log(f"  建樹 {tree_key} {is_start}~{is_end} …")
    t = WF.build_tree_for_window(tree_key, is_start, is_end,
                                 months_long, meta, f_combo_map, log=log)
    assign = t["assign"]
    if "tree_id" in assign.columns:
        assign = assign[assign.tree_id == t["tree_id"]]
    cmeta = t["cluster_meta"]
    if "level" in cmeta.columns:
        cmeta = cmeta[cmeta.level == LEVEL]

    uids = pd.Index(assign[C.PK])
    wide_is = S3._pivot_window(months_long, uids, is_start, is_end)
    return Tree(tree_key=tree_key, tree_id=t["tree_id"], is_start=is_start, is_end=is_end,
                assign=assign.reset_index(drop=True), cluster_meta=cmeta.reset_index(drop=True),
                wide_is=wide_is, link=t["link"], source="rebuilt_window")


def load_mainline_tree(tree_key: str, months_long, log=print) -> Tree:
    """讀 `_frozen/stage3/` 已凍結的主線 normal 樹（正式模式走這條）。

    🔴 為什麼可以直接用（應用層 §7.4b）：`contracts.HRP_WINDOWS[tree_key]` 就是
    這棵樹的建構窗（TW＝2007-01~2025-12＝228 個月），正好等於正式模式的 IS；
    k 由 H-03 用輪廓係數在**同一個窗**上選出（TW=6），與 `k_mode="silhouette_is"`
    的語意一致——不需要也不應該重算。

    ⚠️ 只取 `{tree_key}_normal`。`_crisis` 樹是危機月份子集（TW 僅 17 個月），
    策略宇宙相同但觀察窗不同，不是正式模式要的東西。
    """
    is_start, is_end = C.HRP_WINDOWS[tree_key]
    tree_id = f"{tree_key}_{_MAINLINE_KIND}"

    ca = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")
    ca = ca[ca.tree_id == tree_id]
    if ca.empty:
        raise ValueError(f"_frozen/stage3/cluster_assign.parquet 裡找不到 {tree_id}")
    assign = ca[[C.PK, f"cluster_{LEVEL}"]].reset_index(drop=True)

    cm = pd.read_parquet(paths.STAGE3 / "cluster_meta.parquet")
    cm = cm[(cm.tree_id == tree_id) & (cm.level == LEVEL)].reset_index(drop=True)
    if cm.empty:
        raise ValueError(f"cluster_meta.parquet 裡找不到 {tree_id}/{LEVEL}")

    uids = pd.Index(assign[C.PK])
    wide_is = S3._pivot_window(months_long, uids, is_start, is_end)
    log(f"  載入凍結主線樹 {tree_id}：{len(assign):,} 檔策略、"
        f"k={assign[f'cluster_{LEVEL}'].nunique()}、IS {is_start}~{is_end}"
        f"（{wide_is.shape[1]} 個月）")
    return Tree(tree_key=tree_key, tree_id=tree_id, is_start=is_start, is_end=is_end,
                assign=assign, cluster_meta=cm, wide_is=wide_is,
                link=None, source="frozen_mainline")


def recut(tree: Tree, k: int, log=print) -> Tree:
    """用不同的 k 重切**同一棵 linkage**（不重建樹）——`k_mode="silhouette_is"` 用。

    對位不變式與 `walkforward_matrix.py:435` 同一套：linkage 有 N-1 列合併紀錄，
    N 必須等於 `wide_is` 的列數，否則標籤會**靜默地對錯策略**。
    """
    if tree.link is None:
        raise ValueError(f"{tree.tree_id} 沒有 linkage（凍結主線樹讀不到），無法重切")
    n_leaf = len(tree.link) + 1
    if n_leaf != len(tree.wide_is):
        raise AssertionError(
            f"[{tree.tree_id}] linkage 葉節點數 {n_leaf} != wide_is 列數 "
            f"{len(tree.wide_is)}——重切的群標籤會對錯策略")

    corr_full = np.corrcoef(tree.wide_is.to_numpy(dtype=np.float64))
    labels = hrp.cut_clusters(tree.link, k)
    cmeta, _ = S3._cluster_meta_and_corr(tree.wide_is, corr_full, labels, LEVEL, tree.tree_id)
    assign = pd.DataFrame({C.PK: tree.wide_is.index.to_numpy(),
                           f"cluster_{LEVEL}": labels})
    return dataclasses.replace(tree, assign=assign, cluster_meta=cmeta)


# ============================================================================
# 挑代表
# ============================================================================

def _quality_and_corr(wide_is: pd.DataFrame):
    """IS 品質分數（Calmar，同 H-10 口徑）＋ 相關矩陣 ＋ 位置索引。

    ⚠️ `corr_full` 是 N×N（台股 6,679 ⇒ 約 340 MB float64），算一次就好——
    `walkforward_matrix._pick_a` 的 docstring 記錄過在函式內重算會爆記憶體。
    """
    cagr_is = WF._cagr_matrix(wide_is)
    mdd_is = WF._mdd_matrix(wide_is)
    quality_is = cagr_is / mdd_is.abs().replace(0, np.nan)
    corr_full = np.corrcoef(wide_is.to_numpy(dtype=np.float64))
    pos = pd.Series(range(len(wide_is.index)), index=wide_is.index)
    return cagr_is, quality_is, corr_full, pos


def evaluate_is_only(members: list[str], wide_is: pd.DataFrame,
                     cluster_map: pd.Series) -> tuple[dict, dict]:
    """IS-only 版的 `walkforward_matrix._evaluate()`（應用層 §8-R14）。

    原版簽名是 `_evaluate(members, wide_is, wide_oos, cluster_map)`，一次算 IS+OOS。
    正式模式沒有 OOS（IS 用掉全部資料），故必須另走這條。

    🔴 **回傳的 IS 數字不是預期報酬**（應用層 §8-R13）：實測 900 格台股 A_hrp，
    `is_cagr` 中位數 23.60% vs `oos_cagr` 15.98%——IS CAGR 平均高 7.62pp（1.48 倍），
    74.2% 的格子皆然。⚠️ 但方向不一致：IS MDD/Sharpe 反而比 OOS 更差（IS 涵蓋
    2008 海嘯，OOS 窗都從 2013 之後開始）。**IS 與 OOS 在任何方向上都不可比，
    呼叫端不得並排顯示。**
    """
    if not members:
        raise ValueError("組合成員為空，無法評估——請檢查配額分配或品質分數是否全為 NaN")
    p_is = WF._portfolio_series(wide_is, members)      # 等權，見本檔案 docstring
    cl = cluster_map.reindex(members).dropna()
    vc = cl.value_counts()
    perf = {"is_cagr": WF._cagr(p_is), "is_mdd": WF._mdd(p_is), "is_sharpe": WF._sharpe(p_is)}
    cluster_info = {
        "n_clusters_covered": int(cl.nunique()),
        "max_cluster_share": float(vc.iloc[0] / len(cl)) if len(cl) else float("nan"),
    }
    return perf, cluster_info


def select(tree: Tree, ratio, allocation: str, group: str,
           k_mode: str = "silhouette_is", *, _cache: dict | None = None) -> Selection:
    """在一棵樹上挑出一組成員。忠實複製 `run_one_window` 的挑選段落。

    `_cache` 讓同一棵樹跑多組 (ratio × allocation × group) 時不用重算
    `corr_full`／`quality_is`——那是整個流程最貴的一步。
    """
    if _cache is None:
        _cache = {}
    if "corr" not in _cache:
        _cache["cagr_is"], _cache["quality_is"], _cache["corr_full"], _cache["pos"] = \
            _quality_and_corr(tree.wide_is)
        _cache["corr"] = True
    cagr_is, quality_is = _cache["cagr_is"], _cache["quality_is"]
    corr_full, pos = _cache["corr_full"], _cache["pos"]

    col = f"cluster_{LEVEL}"
    cluster_map = tree.assign.set_index(C.PK)[col]
    sizes = tree.assign.groupby(col).size()
    k = len(sizes)
    n_uni = tree.n_universe

    tot = WF.target_total(ratio, n_uni, k)
    quota, n_capped = WF.allocate(sizes, tot, allocation)

    a_members, n_bf = WF._pick_a(tree.assign, tree.cluster_meta, tree.wide_is,
                                 quality_is, quota, corr_full, pos)
    if group == "A_hrp":
        members, n_backfilled = a_members, n_bf
    else:
        # D/E 用 A 的**實際**檔數對齊（共同座標軸，walkforward_matrix.py:543）
        n_eff = len(a_members)
        if group == "D_top_cagr":
            members = cagr_is.sort_values(ascending=False).index[:n_eff].tolist()
        elif group == "E_top_calmar":
            members = quality_is.dropna().sort_values(ascending=False).index[:n_eff].tolist()
        else:
            raise ValueError(f"不支援的 group：{group}（B_all/C_random 不走這條）")
        n_backfilled = 0

    perf, cinfo = evaluate_is_only(members, tree.wide_is, cluster_map)
    cinfo["n_clusters"] = k
    return Selection(members=members, group=group, ratio=str(ratio), allocation=allocation,
                     k_mode=k_mode, n_members=len(members), n_backfilled=n_backfilled,
                     n_capped_clusters=n_capped, target_total=tot,
                     cluster_info=cinfo, performance_is=perf)


# ============================================================================
# G4 一致性驗收
# ============================================================================

def verify_against_frozen(n_cells: int = 8, tree_key: str = "TW", log=print) -> pd.DataFrame:
    """對凍結表抽 n_cells 格，用本模組重跑，比對成員清單是否 100% 相同（G4）。

    ⚠️ **這證明的是「接線正確」不是「演算法正確」**——演算法是同一份研究部程式碼
    （見本檔案 docstring）。它能抓到的是參數傳錯、日期切錯、排序不一致、
    群標籤對錯位這類靜默錯誤，那正是本專案最常踩的一類雷。
    """
    detail = pd.read_csv(WF.paths.ROOT / "_analysis_outputs_robustness"
                         / "walkforward_matrix_detail.csv")
    members_df = pd.read_parquet(WF.paths.ROOT / "_analysis_outputs_robustness"
                                 / "walkforward_members.parquet")
    detail["ratio"] = detail["ratio"].astype(str)
    members_df["ratio"] = members_df["ratio"].astype(str)

    cand = members_df[(members_df.tree_key == tree_key)
                      & (members_df.group.isin(("A_hrp", "D_top_cagr", "E_top_calmar")))]
    # 固定亂數種子，讓驗收可重現（可重現性是本專案的硬要求，DD-08 精神）
    sample = cand.sample(n=min(n_cells, len(cand)), random_state=20260910)

    months_long, meta, f_combo_map = load_inputs(log=log)
    k_table = WF._load_k_table(log=log)

    rows = []
    tree_cache: dict[tuple, tuple[Tree, dict]] = {}
    for i, (_, r) in enumerate(sample.iterrows(), 1):
        is_start, is_end = str(r.is_start), str(r.is_end)
        log(f"[{i}/{len(sample)}] {r.scheme}/w{int(r.window_no)} {is_start}~{is_end} "
            f"k_mode={r.k_mode} ratio={r.ratio} {r.allocation} {r.group}")

        base_key = (tree_key, is_start, is_end)
        if base_key not in tree_cache:
            t = build_window_tree(tree_key, is_start, is_end,
                                  months_long, meta, f_combo_map, log=lambda *a, **k: None)
            tree_cache[base_key] = (t, {})
        base_tree, _ = tree_cache[base_key]

        if r.k_mode == "fixed":
            tree, cache_key = base_tree, base_key + ("fixed",)
        else:
            k_is = k_table.get((tree_key, is_start, is_end))
            if k_is is None:
                raise KeyError(f"k_stability 缺 ({tree_key}, {is_start}, {is_end})")
            cache_key = base_key + (f"k{k_is}",)
            if cache_key not in tree_cache:
                tree_cache[cache_key] = (recut(base_tree, k_is), {})
            tree = tree_cache[cache_key][0]
        cache = tree_cache.setdefault(cache_key, (tree, {}))[1]

        ratio = r.ratio if r.ratio == "legacy" else float(r.ratio)
        sel = select(tree, ratio, r.allocation, r.group, k_mode=r.k_mode, _cache=cache)

        expected = list(r["members"])
        same = sorted(sel.members) == sorted(expected)

        d = detail[(detail.tree_key == tree_key) & (detail.scheme == r.scheme)
                   & (detail.window_no == r.window_no) & (detail.k_mode == r.k_mode)
                   & (detail.ratio == r.ratio) & (detail.allocation == r.allocation)
                   & (detail.group == r.group)]
        exp_is_cagr = float(d.iloc[0]["is_cagr"]) if len(d) == 1 else float("nan")
        gap = abs(sel.performance_is["is_cagr"] - exp_is_cagr)

        rows.append({"scheme": r.scheme, "window_no": int(r.window_no), "k_mode": r.k_mode,
                     "ratio": r.ratio, "allocation": r.allocation, "group": r.group,
                     "n_expected": len(expected), "n_got": len(sel.members),
                     "members_identical": same, "is_cagr_gap": gap})
        log(f"      成員 {len(sel.members)}/{len(expected)} 相同={same}  "
            f"is_cagr 差={gap:.2e}")

    out = pd.DataFrame(rows)
    n_ok = int(out.members_identical.sum())
    max_gap = float(out.is_cagr_gap.max())
    log(f"\n{'='*66}")
    log(f"G4 一致性驗收：成員清單 {n_ok}/{len(out)} 完全相同｜is_cagr 最大差 {max_gap:.2e}")
    log("⚠️ 本測試證明的是『接線正確』，不是『演算法正確』——演算法是同一份研究部程式碼")
    log(f"{'='*66}")
    return out


# ============================================================================
# CLI
# ============================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="應用層分群／代表挑選（§7 P1）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="G4 一致性驗收（對凍結表）")
    v.add_argument("--n", type=int, default=8)
    v.add_argument("--market", default="TW")

    m = sub.add_parser("mainline", help="在凍結主線樹上挑代表（正式模式）")
    m.add_argument("--market", default="TW")
    m.add_argument("--ratio", default="legacy")
    m.add_argument("--allocation", default="equal", choices=list(WF.ALLOCATIONS))
    m.add_argument("--group", default="A_hrp",
                   choices=["A_hrp", "D_top_cagr", "E_top_calmar"])

    a = ap.parse_args(argv)
    if a.cmd == "verify":
        verify_against_frozen(n_cells=a.n, tree_key=a.market)
        return 0

    months_long, _, _ = load_inputs()
    tree = load_mainline_tree(a.market, months_long)
    ratio = a.ratio if a.ratio == "legacy" else float(a.ratio)
    sel = select(tree, ratio, a.allocation, a.group, k_mode="mainline_h03")
    print(f"\n樹　　　：{tree.tree_id}（{tree.source}）"
          f" IS {tree.is_start}~{tree.is_end}　k={tree.k}　宇宙={tree.n_universe:,}")
    print(f"設定　　：ratio={sel.ratio} allocation={sel.allocation} group={sel.group}")
    print(f"選出　　：{sel.n_members} 檔（目標 {sel.target_total}，"
          f"backfill {sel.n_backfilled}，capped 群 {sel.n_capped_clusters}）")
    print(f"集中度　：最大群佔比 {sel.cluster_info['max_cluster_share']:.2%}，"
          f"涵蓋 {sel.cluster_info['n_clusters_covered']}/{sel.cluster_info['n_clusters']} 群")
    print(f"IS 績效 ：CAGR {sel.performance_is['is_cagr']:.2%}　"
          f"MDD {sel.performance_is['is_mdd']:.2%}　"
          f"Sharpe {sel.performance_is['is_sharpe']:.3f}")
    print("🔴 上列 IS 數字是樣本內配適值，不是預期報酬（§8-R13：歷史上 IS CAGR "
          "平均比實際 OOS 高 7.62pp），且不可與凍結表的 OOS 數字並排比較")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
