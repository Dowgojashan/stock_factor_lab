# -*- coding: utf-8 -*-
"""idea②：三個配額/挑選規則設計，都保留HRP分群，只換「配額怎麼分」或「群內怎麼挑」。

背景：idea①（純CAGR排序取代Calmar）測出負面結果（§3.21）——問題不是排序指標，
是「每群只挑單一期表現最好的」這個規則對窄持股/極端值沒有抑制力，且§3.19查出
「近乎純估值大群」不論大小都跟其他群拿一樣的固定配額，是鎖死估值集中度的關鍵。

三個方向（跟baseline＝現行Calmar+equal配額對照）：

**方向A：配額依群的「純度」反向分配**——用`(1-purity)`當權重（purity=群內最大
類別佔比），套用跟`WF.allocate()`的proportional分支同一套largest-remainder演算法
（只是把`sizes`換成這個權重），純度越高的群配額越少、越混合的群配額越多。
設下限`min_weight`避免100%純的群配額直接歸零（仍保留基本代表性）。

**方向B：配額之外疊加「類別保底」**——先用現行equal配額正常挑，再對每個群檢查
「群內存在、但目前0檔被選中」的類別，若有且該類別在這個群裡有候選人，額外加選
1檔（該群額外多1個名額，仿9/22報告§4.5的保底名額精神，這次用在因子類別而非
市值覆蓋上）。

**方向C：群內配額依類別比例切分，各類別內部各自選品質最佳**——群的配額不再是
「品質最高的N檔，不分類別」，改成先依群內各類別的候選人數比例切分子配額（用
同一套largest-remainder法），再各類別內部依品質排序選出子配額數量的代表。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._design_test_quota_rules
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH, resolve_holdings  # noqa: E402

from research import hrp  # noqa: E402
from research import k_stability  # noqa: E402
from research import stage3_hrp as S3  # noqa: E402
from research import walkforward_matrix as WF  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _design_test_rolling_is6oos2 import (IS_MONTHS, OOS_LEN, RATIO,  # noqa: E402
                                          TREE_KEY, window_dates_rolling)

FACTOR_CATEGORY = {
    "ROE": "資本報酬/獲利能力", "ROIC": "資本報酬/獲利能力", "CROIC": "資本報酬/獲利能力",
    "EPS": "獲利", "FCF_P": "現金流品質", "FCF_OI": "現金流品質", "OCF_E": "現金流品質",
    "EV_EBITDA": "估值倍數", "EV_S": "估值倍數", "PB": "估值倍數", "PS": "估值倍數", "P_IC": "估值倍數",
    "MOM": "動量",
}
ANCHORED_IS_START, ANCHORED_IS_END = "2007-01", "2023-12"
ANCHORED_OOS_START, ANCHORED_OOS_END = "2024-01", "2025-12"
ROLLING_WINDOW_NO = 6
RULES = ["baseline", "A_purity", "B_bonus", "C_category_quota"]
MIN_WEIGHT_A = 0.15   # 方向A：100%純的群仍保留這個最低權重，不讓配額直接歸零


def categorize(row: pd.Series) -> str:
    return FACTOR_CATEGORY.get(row["F1_factor"], f"未分類({row['F1_factor']})")


# ============================================================================
# 方向A：配額依「1-純度」分配（沿用WF.allocate()的proportional largest-remainder演算法）
# ============================================================================

def allocate_by_weight(sizes: pd.Series, weight: pd.Series, total: int) -> tuple[pd.Series, int]:
    """跟`WF.allocate(..., how="proportional")`同一套largest-remainder演算法，
    只是分配依據從`sizes`換成任意`weight`（仍用`sizes`當每群配額上限，不能超過
    該群實際候選人數）。"""
    sizes = sizes.astype(int)
    cap_total = int(sizes.sum())
    total = int(min(total, cap_total))
    total_weight = float(weight.sum())
    if total_weight <= 0:
        # 退化情況：全部群權重都是0，退回equal
        return WF.allocate(sizes, total, "equal")
    raw = weight / total_weight * total
    q = np.floor(raw).astype(int).clip(lower=1, upper=sizes)
    while int(q.sum()) < total:
        room = (sizes - q)
        frac = (raw - q).where(room > 0, -np.inf)
        if not np.isfinite(frac).any():
            break
        q[frac.idxmax()] += 1
    while int(q.sum()) > total:
        over = (q - raw).where(q > 1, -np.inf)
        if not np.isfinite(over).any():
            break
        q[over.idxmax()] -= 1
    return q, int((q >= sizes).sum())


def cluster_purity(assign: pd.DataFrame, meta: pd.DataFrame) -> pd.Series:
    """每個群「群內最大類別佔比」，meta要已經有category欄位、且index=strategy_uid。"""
    cluster_map = assign.set_index(WF.C.PK)[f"cluster_{WF.LEVEL}"]
    m = meta.copy()
    m["cluster"] = cluster_map.reindex(m.index)
    purity = m.groupby("cluster")["category"].apply(lambda s: s.value_counts(normalize=True).iloc[0])
    return purity


# ============================================================================
# 方向B：baseline配額挑完後，每群額外疊加1個「類別保底」名額
# ============================================================================

def pick_with_category_bonus(assign: pd.DataFrame, cmeta: pd.DataFrame, wide_is: pd.DataFrame,
                             quality: pd.Series, quota: pd.Series, corr_full: np.ndarray,
                             pos: pd.Series, meta: pd.DataFrame) -> tuple[list[str], int]:
    a_members, n_bf = WF._pick_a(assign, cmeta, wide_is, quality, quota, corr_full, pos)
    picked = set(a_members)
    cluster_map = assign.set_index(WF.C.PK)[f"cluster_{WF.LEVEL}"]

    bonus_added = []
    for cid, g in assign.groupby(f"cluster_{WF.LEVEL}"):
        members_in_cluster = set(g[WF.C.PK])
        already_picked = members_in_cluster & picked
        if not already_picked:
            continue  # 這個群配額是0，不做保底（沒有名額基礎可以疊加）
        picked_categories = set(meta.reindex(list(already_picked))["category"])
        cluster_meta = meta.reindex(list(members_in_cluster))
        missing_categories = set(cluster_meta["category"].unique()) - picked_categories
        if not missing_categories:
            continue
        # 每個群最多疊加1個保底名額，給「該群裡目前完全沒被選中的類別」裡品質最好的一檔
        candidates = cluster_meta[cluster_meta["category"].isin(missing_categories)].index
        candidates = [c for c in candidates if c not in picked]
        if not candidates:
            continue
        best = quality.reindex(candidates).idxmax()
        if pd.isna(quality.get(best, np.nan)):
            continue
        picked.add(best)
        bonus_added.append(best)
    return list(picked), n_bf


# ============================================================================
# 方向C：群內配額依類別比例切分，各類別內部各自選品質最佳
# ============================================================================

def pick_with_category_quota(assign: pd.DataFrame, wide_is: pd.DataFrame, quality: pd.Series,
                             quota: pd.Series, meta: pd.DataFrame) -> list[str]:
    a_members: list[str] = []
    for cid, g in assign.groupby(f"cluster_{WF.LEVEL}"):
        q_total = int(quota.get(cid, 0))
        if q_total <= 0:
            continue
        members_in_cluster = list(g[WF.C.PK])
        cluster_meta = meta.reindex(members_in_cluster)
        cat_sizes = cluster_meta["category"].value_counts()
        if len(cat_sizes) == 0:
            continue
        cat_quota, _ = allocate_by_weight(cat_sizes, cat_sizes.astype(float), q_total)
        for cat, cq in cat_quota.items():
            if cq <= 0:
                continue
            cat_members = cluster_meta[cluster_meta["category"] == cat].index
            ranked = quality.reindex(cat_members).dropna().sort_values(ascending=False)
            a_members += list(ranked.head(int(cq)).index)
    return a_members


# ============================================================================
# 主流程
# ============================================================================

def run_one(label: str, rule: str, is_start: str, is_end: str, oos_start: str, oos_end: str,
           months_long, meta_pool, f_combo_map, idx: pd.DataFrame, mktcap: pd.DataFrame, md: MarketData):
    print(f"\n{'='*60}\n{label}／{rule}：IS {is_start}~{is_end}")

    tree = WF.build_tree_for_window(TREE_KEY, is_start, is_end, months_long, meta_pool, f_combo_map, print)
    scan = k_stability.scan_window(TREE_KEY, is_start, is_end, months_long, meta_pool, log=print)
    k_is = scan["k_is_selected"]

    uids = pd.Index(tree["assign"][WF.C.PK])
    wide_is = S3._pivot_window(months_long, uids, is_start, is_end)
    w = months_long[months_long.strategy_uid.isin(set(uids))]
    w = w[(w.month >= pd.Period(oos_start, "M")) & (w.month <= pd.Period(oos_end, "M"))]
    wide_oos = w.pivot(index="strategy_uid", columns="month", values="ret")
    n_nan = int(wide_oos.isna().sum().sum())
    n_missing = len(set(uids) - set(wide_oos.index))
    if n_nan or n_missing:
        raise ValueError(f"[{label}] OOS資料不完整：{n_nan}個缺值、{n_missing}個策略無資料")

    corr_full = np.corrcoef(wide_is.to_numpy(dtype=np.float64))
    labels = hrp.cut_clusters(tree["link"], k_is)
    cmeta, _ = S3._cluster_meta_and_corr(wide_is, corr_full, labels, WF.LEVEL, tree["tree_id"])
    assign = pd.DataFrame({WF.C.PK: wide_is.index.to_numpy(), f"cluster_{WF.LEVEL}": labels})
    cluster_map = assign.set_index(WF.C.PK)[f"cluster_{WF.LEVEL}"]

    cagr_is = WF._cagr_matrix(wide_is)
    mdd_is = WF._mdd_matrix(wide_is)
    quality_is = cagr_is / mdd_is.abs().replace(0, np.nan)   # 全部用現行Calmar，只換配額/挑選規則
    pos = pd.Series(range(len(wide_is.index)), index=wide_is.index)

    meta = idx.reindex(list(uids)).copy()
    meta["category"] = meta.apply(categorize, axis=1)

    sizes = assign.groupby(f"cluster_{WF.LEVEL}").size()
    k = len(sizes)
    n_uni = len(uids)
    tot = WF.target_total(RATIO, n_uni, k)

    if rule == "baseline":
        quota, n_capped = WF.allocate(sizes, tot, "equal")
        a_members, n_bf = WF._pick_a(assign, cmeta, wide_is, quality_is, quota, corr_full, pos)
    elif rule == "A_purity":
        purity = cluster_purity(assign, meta)
        weight = (1 - purity).clip(lower=MIN_WEIGHT_A)
        quota, n_capped = allocate_by_weight(sizes, weight, tot)
        a_members, n_bf = WF._pick_a(assign, cmeta, wide_is, quality_is, quota, corr_full, pos)
    elif rule == "B_bonus":
        quota, n_capped = WF.allocate(sizes, tot, "equal")
        a_members, n_bf = pick_with_category_bonus(assign, cmeta, wide_is, quality_is, quota,
                                                   corr_full, pos, meta)
    elif rule == "C_category_quota":
        quota, n_capped = WF.allocate(sizes, tot, "equal")
        a_members = pick_with_category_quota(assign, wide_is, quality_is, quota, meta)
        n_bf = 0
    else:
        raise ValueError(rule)

    # 🔴 2026-09-26 code review抓到：這個檢查原本不管超過還是不足都印同一句話，
    # 但「超過」（B_bonus刻意疊加保底名額）是設計預期的正常行為，「不足」
    # （C_category_quota若某類別候選人不夠填滿子配額、沒有backfill）才是
    # §3.8同一類陷阱要警示的真異常——兩種情況訊息不該一樣，會誤導人以為
    # B_bonus每次多出來的名額是bug
    if len(a_members) > tot:
        print(f"   ℹ️ 實選{len(a_members)}檔 > 目標{tot}檔（規則={rule}，若為B_bonus屬設計預期"
             f"——保底名額本來就會疊加在原本配額之外，不是異常；跨規則比較檔數時要留意分母不同）")
    elif len(a_members) < tot:
        print(f"   ⚠️ 實選{len(a_members)}檔 < 目標{tot}檔（規則={rule}，可能某類別候選人不足，"
             f"沒有backfill，這是真的異常，跨規則比較時這格要特別注意）")

    perf = WF._evaluate(a_members, wide_is, wide_oos, cluster_map)

    sub = idx.reindex(a_members).copy()
    sub["category"] = sub.apply(categorize, axis=1)
    v1_frac = float((sub["V"] == "v1").mean())
    avg_hold = float(sub["avg_holdings"].mean())
    val_frac = float((sub["category"] == "估值倍數").mean())

    all_syms: set[str] = set()
    for uid in a_members:
        try:
            syms, _ = resolve_holdings(md, idx.loc[uid], is_end)
        except (RuntimeError, KeyError, ValueError):
            continue
        all_syms |= set(syms)
    valid_mk = mktcap.index[mktcap.index <= pd.Timestamp(is_end)]
    if len(valid_mk):
        mk_row = mktcap.loc[valid_mk.max()]
        pct_rank = mk_row.reindex(md.common).dropna().rank(pct=True)
        held_pct = pct_rank.reindex(list(all_syms)).dropna()
        frac_top10 = float((held_pct >= 0.90).mean()) if len(held_pct) else float("nan")
    else:
        frac_top10 = float("nan")

    print(f"k={k}｜目標{tot}檔｜實選{len(a_members)}檔")
    print(f"估值佔比={val_frac:.1%}｜V1比例={v1_frac:.1%}｜平均持股={avg_hold:.1f}檔｜市值前10%佔比={frac_top10:.1%}")
    print(f"OOS CAGR={perf['oos_cagr']:+.2%} MDD={perf['oos_mdd']:.2%} Sharpe={perf['oos_sharpe']:.2f}")

    return {"label": label, "rule": rule, "k": k, "target_total": tot, "n_members": len(a_members),
           "val_frac": val_frac, "v1_frac": v1_frac, "avg_holdings": avg_hold,
           "frac_top10pct_mktcap": frac_top10, "is_cagr": perf["is_cagr"], "is_mdd": perf["is_mdd"],
           "oos_cagr": perf["oos_cagr"], "oos_mdd": perf["oos_mdd"], "oos_sharpe": perf["oos_sharpe"]}


def main():
    print(">> 載入資料...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    md = MarketData(TREE_KEY, start="2000-01-01")
    mktcap = md.get_field("report:mktcap")

    blocks = WF._blocks(WF.SCHEME_TOTAL_MONTHS, IS_MONTHS, OOS_LEN)
    off, L = blocks[ROLLING_WINDOW_NO - 1]
    r_is_start, r_is_end, r_oos_start, r_oos_end = window_dates_rolling(off, L, TREE_KEY)

    rows = []
    for rule in RULES:
        rows.append(run_one("anchored_w4", rule, ANCHORED_IS_START, ANCHORED_IS_END,
                            ANCHORED_OOS_START, ANCHORED_OOS_END,
                            months_long, meta_pool, f_combo_map, idx, mktcap, md))
    for rule in RULES:
        rows.append(run_one("rolling_w6", rule, r_is_start, r_is_end, r_oos_start, r_oos_end,
                            months_long, meta_pool, f_combo_map, idx, mktcap, md))

    out = pd.DataFrame(rows)
    out_path = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "quota_rules_comparison.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}")

    print("\n=== 總覽：baseline vs A/B/C ===")
    print(out[["label", "rule", "n_members", "val_frac", "v1_frac", "avg_holdings",
              "frac_top10pct_mktcap", "oos_cagr", "oos_mdd", "oos_sharpe"]].to_string(index=False))


if __name__ == "__main__":
    main()
