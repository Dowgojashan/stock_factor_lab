# -*- coding: utf-8 -*-
"""階段 3 · HRP 階層聚類（真實資料管線，W-03/W-04/W-12）

輸入 ← `_frozen/stage1/returns_monthly.parquet`（HRP 原料）
        `_frozen/stage1/returns_meta.parquet`（hist_start，DD-03 共同窗判定用）
        `_frozen/stage1/strategy_marks.parquet`（is_usable，v9規定階段3只對usable_pool算）
        `_frozen/stage0/candidate_index.parquet`（market/f_combo，ARI 驗證用）
        `_frozen/stage2/regime/regime_table_{market}.parquet`（危機樹的危機窗，2a）
輸出 → `_frozen/stage3/cluster_assign.parquet`（normal + crisis 六棵樹）
        `_frozen/stage3/cluster_meta.parquet`
        `_frozen/stage3/cluster_corr_matrix_{tree_id}.parquet`（L1 層級，k×k）
        `_frozen/stage3/linkage_{tree_id}.npy`（凍結的 linkage 矩陣）
        `_frozen/stage3/co_fail_regimes.parquet`（危機期共跌的操作型定義，見下）

DD-03（共同窗）已用**修復後、新候選池**的 hist_start 分布重新驗證，數字不變：
   TW 2007-01（保留 7,125/7,128）／US 2002-01（保留 8,682/8,682，XM 用 TW 窗
   時仍 100%）／XM 2007-01（TW 7,125 + US 8,682 = 15,807）。

⚠️ **crisis 樹的策略宇宙跟 normal 樹完全相同**（同一批 DD-03 共同窗篩出的策略），
   差別只在於相關矩陣改用「危機窗月份子集」而非整段共同窗——這樣常態/危機兩棵
   樹的分群結果才能直接比較（同一批策略，只是換一段觀察窗），符合 v9「比較兩棵
   樹的結構差異」的設計意圖。

⚠️ **危機窗定義（v9 只寫「危機樹三組各用對應危機窗，台↔美判兩套」，未明講怎麼
   合併，此為本階段的解讀，已用實際數字核對過）**：
   - TW/US 單市場樹：用該市場自己的 2a 危機段，落在該樹 DD-03 窗內的月份。
   - XM 合併樹：**聯集**（TW 危機月 ∪ US 危機月），不用交集。
     實測：TW∩US 交集只有 6 個月（2008-05~2008-10，GFC 高峰台美同步崩跌那段），
     聯集 26 個月。若用交集，會把「台股自己 2022 年危機、美股當時沒有」
     「美股自己 2020 COVID 崩盤、台股當時已回穩」這些台美不同步的月份全部丟掉
     ——這些正是論文要呈現的「跨市場分散有效」證據，交集會靜默抹掉最有價值的樣本。

⚠️ **crisis 樹月數遠少於 normal 樹**（TW 17 個月／US 22 個月／XM 26 個月，
   vs normal 樹 228/288/228 個月），N（策略數）遠大於 T（月數）在 normal 樹
   已是常態（已驗證 PSD 不受影響），crisis 樹只是同一現象的極端版：相關矩陣
   會嚴重秩不足（rank ≤ T），但樣本相關矩陣本身仍是 Gram 矩陣、數學上仍是
   PSD——秩不足不等於非 PSD，`check_psd` 驗的是後者。**這是預期中的資料稀疏，
   不是錯誤**；分群結果的統計把握度天然較低，須在論文誠實揭露。
   零變異數策略（危機窗內完全沒有報酬變化，corrcoef 會產生 NaN）會被偵測並
   排除出該樹，並記錄排除數量。

⚠️ **2026-08-26 老師意見後定案（開發待辦追蹤.md H-14/H-15/H-16）：crisis 樹「留但
   降級」**——保留產出（`co_fail_regimes`、危機期相關矩陣），但用途限定在論文的
   描述性揭露，不再是任何選兵決策的依據（見 `ops/tools.py` T1 已移除
   `cluster_diversify`）。**crisis 樹刻意不做 in-sample/out-of-sample 切分**：
   17-26 個月本來就不足以支撐一次穩健估計，切了只會讓兩段都不可靠，不是更嚴謹。
   之後若要做 H-11（normal 樹的 IS/OOS），這條規則不適用於 crisis 樹。

用法：
    cd code
    python -m research.stage3_hrp                 # 全部六棵樹（3 normal + 3 crisis）
    python -m research.stage3_hrp --tree TW        # 只跑一個市場（normal+crisis 各一棵）
    python -m research.stage3_hrp --normal-only    # 只跑 normal（除錯/測試用）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, hrp, paths

#: L1 目標群數（給 LLM 讀 / 給快篩配額用的粗粒度層級）。
#: 2026-08-28（H-03）改用輪廓係數（`hrp.silhouette_scan`，見
#: `research/cluster_count_selection.py`）對每棵樹的凍結linkage逐一切割選出，
#: 取代原本寫死的 8——那個 8 其實是老師在會議上舉的資金試算例子（「假設了 ok
#: 我臺股分8群美股3群」），不是他真的指定的群數，見開發待辦追蹤.md H-03。
#: 三棵normal樹的輪廓係數掃描結果：TW在k=6（0.046，扣掉k=3那個67.6%最大群佔比
#: 的退化解）、US在k=7（0.062，全域最高點非退化）、XM在k=3（0.279，大幅領先
#: 其他k，很可能是台美市場邊界本身主導了低k時的分群結構，非策略層級細緻分群，
#: 使用者2026-08-28確認照資料走）。crisis 樹沿用同市場normal樹的k（不是資料
#: 驅動的巧合，是刻意設計——常態/危機要切在同一個粒度上才能比較「有效群數是否
#: 塌縮」，見模組開頭）。
L1_TARGET = {"TW": 6, "US": 7, "XM": 3}
#: L2 層級已移除（H-04，2026-08-28）：原L2_TARGET=40從未被任何下游決策邏輯
#: 讀取（只在T3的profile欄位列表裡出現，且contracts/三份架構文件都找不到40
#: 這個數字的依據），純粹是沒有用途的技術債，順手拿掉。
L3_TARGET = {"TW": C.EXPECTED_F_COMBOS["TW"], "US": C.EXPECTED_F_COMBOS["US"],
            "XM": C.EXPECTED_F_COMBOS["TW"] + C.EXPECTED_F_COMBOS["US"]}

CO_FAIL_LEVEL = "L1"   # co_fail_regimes 只在 L1 算，跟 cluster_corr_matrix 同粒度


def _tree_universe(market_filter: str, window_start: str,
                   meta: pd.DataFrame) -> pd.Index:
    """依市場 + DD-03 共同窗篩出這棵樹要納入的策略（normal/crisis 共用同一批）。

    ⚠️ `meta` 進來前必須已經過 usable_pool 過濾（見 run() 的載入邏輯）——v9
    明講「階段2、3只對usable_pool算，不對全庫」，這裡本身不重複做is_usable
    過濾，只信任呼叫端已經篩過。
    """
    sub = meta if market_filter == "XM" else meta[meta.market == market_filter]
    sub = sub[sub.hist_start <= window_start]
    return pd.Index(sub.strategy_uid)


def _pivot_window(months_long: pd.DataFrame, uids: pd.Index,
                  window_start: str, window_end: str) -> pd.DataFrame:
    """long 格式報酬 → wide（策略 × 月），裁到共同窗。**不得有 NaN**（DD-03 保證）。"""
    w = months_long[months_long.strategy_uid.isin(uids)]
    w = w[(w.month >= pd.Period(window_start, "M")) & (w.month <= pd.Period(window_end, "M"))]
    wide = w.pivot(index="strategy_uid", columns="month", values="ret")
    wide = wide.loc[list(uids)]            # 固定順序，之後索引用得到
    n_nan = int(wide.isna().sum().sum())
    if n_nan:
        raise ValueError(
            f"共同窗內出現 {n_nan} 個 NaN——DD-03 的保證被打破了（策略池是否換過？"
            f"共同窗需要重新用 returns_meta 的 hist_start 分布驗證）")
    return wide


def _pivot_months(months_long: pd.DataFrame, uids: pd.Index,
                  months: pd.PeriodIndex) -> pd.DataFrame:
    """long 格式報酬 → wide，只取指定的（可不連續的）月份子集——crisis 樹用。

    月份子集必為 DD-03 共同窗的子集（危機窗已在載入時裁過），故同樣不得有 NaN。
    """
    w = months_long[months_long.strategy_uid.isin(uids) & months_long.month.isin(months)]
    wide = w.pivot(index="strategy_uid", columns="month", values="ret")
    wide = wide.loc[list(uids), list(months)]
    n_nan = int(wide.isna().sum().sum())
    if n_nan:
        raise ValueError(
            f"危機窗內出現 {n_nan} 個 NaN——危機月份是否超出了該樹的 DD-03 共同窗範圍？")
    return wide


def _load_crisis_months(tree_key: str) -> pd.PeriodIndex:
    """該樹危機窗的月份集合（見模組開頭「危機窗定義」的解讀說明）。"""
    window_start, window_end = C.HRP_WINDOWS[tree_key]
    ws, we = pd.Period(window_start, "M"), pd.Period(window_end, "M")

    def _one(m: str) -> set[pd.Period]:
        seg = pd.read_parquet(paths.STAGE2 / "regime" / f"regime_table_{m}.parquet")
        crisis = seg[seg.label == "危機"]
        months: set[pd.Period] = set()
        for r in crisis.itertuples():
            months |= set(pd.period_range(r.start.to_period("M"), r.end.to_period("M"), freq="M"))
        return {mm for mm in months if ws <= mm <= we}

    months = (_one("TW") | _one("US")) if tree_key == "XM" else _one(tree_key)
    return pd.PeriodIndex(sorted(months), freq="M")


def _drop_zero_variance(wide: pd.DataFrame, log=print) -> tuple[pd.DataFrame, int]:
    """排除該窗內完全沒有報酬波動的策略（會讓 corrcoef 整列產生 NaN）。

    normal 樹（228/288月）極少見；crisis 樹（17~26月）樣本少，較可能出現。
    抽成函式是為了讓 `_build_tree`（建樹當下）與 `rebuild_tree_returns`（事後重建）
    **共用同一條排除規則**——這兩處若各寫一份，日後改了其中一個門檻，重建出來的
    矩陣就跟當初建樹用的不一樣，而且不會報錯。回傳 (過濾後的wide, 被排除的檔數)。
    """
    std0 = wide.std(axis=1) == 0
    n = int(std0.sum())
    if n:
        log(f"  ⚠️ 排除 {n} 個零變異數策略（此窗內完全無報酬波動，corrcoef 會產生 NaN）")
        wide = wide.loc[~std0]
    return wide, n


def rebuild_tree_returns(tree_id: str, log=print) -> pd.DataFrame:
    """重建某棵樹**當初建樹時實際用的**報酬矩陣（策略×月，已排除零變異數列）。

    ⚠️ **這是「重建某棵樹的輸入」的單一事實來源**（2026-08-30 code review 抽出）。
    在此之前，`effective_bets._tree_corr()` 與 `cluster_count_selection.
    _rebuild_dist_matrix()` 各自維護了一份逐行相同的複製品——那種重複最危險的
    地方不是多打幾行字，而是**以後只要有人改了其中一邊的過濾規則（usable 定義、
    共同窗取法、零變異數排除），另一邊會靜默沿用舊規則，兩邊算出的相關矩陣就
    不再是同一個東西，而且不會報錯**，只會讓 H-03（群數選擇）與 H-09（ENB）
    悄悄建立在不同的資料上。

    步驟必須與 `run()` + `_build_tree()` 開頭完全一致：
      usable_pool 過濾 → DD-03 共同窗（crisis 樹改取危機月份子集）→ 排除零變異數。
    呼叫端自行決定要拿它算 corr、dist 還是別的東西。

    ⚠️ 本函式**不做** `freeze.verify_inputs`——它讀 STAGE1 的凍結產物，但驗證屬於
    「跑一支完整流程前」的職責，應由呼叫端的 `run()` 負責（各模組讀的上游集合不同，
    在這裡硬驗會驗到呼叫端根本沒用到的東西）。
    """
    tree_key, kind = tree_id.rsplit("_", 1)
    months_long = pd.read_parquet(paths.STAGE1 / "returns_monthly.parquet")
    meta = pd.read_parquet(paths.STAGE1 / "returns_meta.parquet")
    marks = pd.read_parquet(paths.STAGE1 / "strategy_marks.parquet")
    usable = set(marks.loc[marks.is_usable, C.PK])
    meta = meta[meta.strategy_uid.isin(usable)]

    window_start, window_end = C.HRP_WINDOWS[tree_key]
    uids = _tree_universe(tree_key, window_start, meta)
    if kind == "normal":
        wide = _pivot_window(months_long, uids, window_start, window_end)
    else:
        crisis_months = _load_crisis_months(tree_key)
        wide = _pivot_months(months_long, uids, crisis_months)

    wide, _ = _drop_zero_variance(wide, log)
    return wide


def _build_tree(tree_id: str, tree_key: str, wide: pd.DataFrame,
                f_combo_map: pd.Series, log=print) -> dict:
    """給定已 pivot 好的報酬矩陣（策略×月，無NaN），跑完 HRP 全流程。

    normal/crisis 共用此函式——差別只在傳入的 `wide` 是完整共同窗還是危機窗子集，
    其餘（PSD檢查、linkage選擇、分群、群間相關）完全相同，見模組開頭設計理由。
    """
    t0 = time.time()

    # 排除規則與 rebuild_tree_returns 共用同一個函式，見 _drop_zero_variance docstring
    wide, n_dropped_zero_var = _drop_zero_variance(wide, log)

    uids = wide.index
    log(f"  報酬矩陣 {wide.shape}（策略×月）")

    returns = wide.to_numpy(dtype=np.float64)
    corr = np.corrcoef(returns)
    psd_ok, min_eig = hrp.check_psd(corr)
    log(f"  相關矩陣 {corr.shape}｜PSD={'✓' if psd_ok else '✗ 違反！'}（min_eig={min_eig:.3e}）"
        f"｜{time.time()-t0:.0f}s")
    if not psd_ok:
        raise AssertionError(
            f"[{tree_id}] 相關矩陣非 PSD（min_eig={min_eig:.3e}）——"
            f"此樹不可信，中止")

    dist = hrp.corr_to_distance(corr)
    tri_ok, violations, max_excess = hrp.check_triangle_inequality(dist, n_samples=3000)
    log(f"  三角不等式抽檢 3000 組｜違規 {violations}"
        f"{f'（最大超出 {max_excess:.2e}）' if violations else ''}")

    cov = np.cov(returns)

    # DD-06：single vs ward，選擇標準見 run_tree()/run_crisis_tree() 呼叫處註解
    # （precedent 記在原 run_tree 開發時的踩雷紀錄，此處不重複）。
    n_total = len(uids)
    l3_target_probe = L3_TARGET[tree_key]
    results = {}
    for method in ("single", "ward"):
        link = hrp.build_linkage(dist, method=method)
        coph = hrp.cophenetic_correlation(link, dist)
        probe_labels = hrp.cut_clusters(link, l3_target_probe)
        sizes = pd.Series(probe_labels).value_counts()
        max_share = float(sizes.max() / n_total)
        n_singleton = int((sizes == 1).sum())
        results[method] = {"link": link, "cophenetic": coph,
                           "max_share": max_share, "n_singleton": n_singleton}
        log(f"  linkage={method:<6} cophenetic={coph:.4f}  "
            f"L3探測：最大群佔比={max_share:.1%}  singleton群數={n_singleton}/{l3_target_probe}")

    method = min(results, key=lambda m: results[m]["max_share"])
    link = results[method]["link"]
    coph = results[method]["cophenetic"]
    if results[method]["max_share"] > 0.5:
        log(f"  ⚠️ 連分群較平衡的 {method} 最大群佔比也超過 50%"
            f"（{results[method]['max_share']:.1%}）——此樹的分群結果可能仍不可靠，建議人工複查")
    log(f"  → 採用 {method}（最大群佔比較低，分群較平衡；非以 cophenetic 決定）"
        f"｜{time.time()-t0:.0f}s")

    leaf_order = hrp.quasi_diagonal_order(link)
    weights = hrp.recursive_bisection_weights(cov, leaf_order)
    log(f"  遞迴二分權重完成（全樹持有時的權重，非任意子集）｜加總={weights.sum():.6f}")

    l1_target = L1_TARGET[tree_key]
    l3_target = L3_TARGET[tree_key]
    targets = {"L1": l1_target, "L3": l3_target}
    labels = {}
    level_diag = {}
    for lvl, target in targets.items():
        lab = hrp.cut_clusters(link, target)
        labels[lvl] = lab
        sizes = pd.Series(lab).value_counts()
        max_share = float(sizes.max() / n_total)
        n_singleton = int((sizes == 1).sum())
        level_diag[lvl] = {"n_clusters": int(len(sizes)), "max_share": max_share,
                           "n_singleton": n_singleton}
        log(f"  {lvl}：目標群數 {target}，實際群數 {len(sizes)}"
            f"｜最大群佔比 {max_share:.1%}｜singleton {n_singleton}")

    fc = f_combo_map.reindex(wide.index)
    ari = hrp.adjusted_rand_index(pd.Series(labels["L3"]), pd.Series(fc.to_numpy()))
    log(f"  L3 vs F組合 ARI = {ari:.4f}")

    assign = pd.DataFrame({
        C.PK: wide.index, "tree_id": tree_id,
        "cluster_L1": labels["L1"], "cluster_L3": labels["L3"],
    })

    meta_rows, corr_mats = [], {}
    for lvl in ("L1", "L3"):
        m, gcorr = _cluster_meta_and_corr(wide, corr, labels[lvl], lvl, tree_id)
        meta_rows.append(m)
        corr_mats[lvl] = gcorr
    cluster_meta = pd.concat(meta_rows, ignore_index=True)

    dt = time.time() - t0
    log(f"[{tree_id}] 完成，{dt:.0f}s\n")
    return {
        "tree_id": tree_id, "tree_key": tree_key, "assign": assign, "cluster_meta": cluster_meta,
        "corr_l1": corr_mats["L1"], "link": link, "method": method,
        "cophenetic": coph, "psd_min_eig": min_eig, "ari_l3": ari,
        "n_strategies": len(uids), "n_dropped_zero_var": n_dropped_zero_var,
        "n_months": returns.shape[1], "seconds": dt,
        "method_comparison": {m: {k: v for k, v in r.items() if k != "link"}
                              for m, r in results.items()},
        "level_diag": level_diag,
    }


def _cluster_meta_and_corr(wide: pd.DataFrame, corr: np.ndarray, labels: np.ndarray,
                           level: str, tree_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """算每群的成員數/群內平均相關/代表策略（medoid），以及群間 k×k 相關矩陣。"""
    uids = wide.index.to_numpy()
    uniq = np.unique(labels)
    rows = []
    group_mean_series = {}   # 群代表序列（成員報酬的平均），供群間相關矩陣用
    for cid in uniq:
        idx = np.where(labels == cid)[0]
        n = len(idx)
        if n == 1:
            avg_corr, rep = np.nan, uids[idx[0]]
        else:
            sub = corr[np.ix_(idx, idx)]
            off_diag = sub[~np.eye(n, dtype=bool)]
            avg_corr = float(off_diag.mean())
            # medoid：與群內其他成員平均相關最高者，最能代表這個群
            rep = uids[idx[np.argmax(sub.mean(axis=1))]]
        rows.append({"tree_id": tree_id, "level": level, "cluster_id": int(cid),
                     "n_members": int(n), "avg_intra_corr": avg_corr,
                     "representative_uid": str(rep)})
        group_mean_series[int(cid)] = wide.iloc[idx].mean(axis=0)

    meta = pd.DataFrame(rows)

    gids = sorted(group_mean_series)
    means = pd.DataFrame({g: group_mean_series[g] for g in gids})
    gcorr = means.corr()
    return meta, gcorr


def run_tree(tree_key: str, months_long: pd.DataFrame, meta: pd.DataFrame,
            f_combo_map: pd.Series, log=print) -> dict:
    """跑一棵 normal 樹（整段 DD-03 共同窗）。`tree_key` ∈ {'TW','US','XM'}。"""
    window_start, window_end = C.HRP_WINDOWS[tree_key]
    uids = _tree_universe(tree_key, window_start, meta)
    log(f"[{tree_key}_normal] 窗 {window_start}~{window_end}｜策略數 {len(uids):,}")
    wide = _pivot_window(months_long, uids, window_start, window_end)
    return _build_tree(f"{tree_key}_normal", tree_key, wide, f_combo_map, log)


def run_crisis_tree(tree_key: str, months_long: pd.DataFrame, meta: pd.DataFrame,
                    f_combo_map: pd.Series, log=print) -> dict:
    """跑一棵 crisis 樹（同一批策略，相關只用危機窗月份算）。"""
    window_start, _ = C.HRP_WINDOWS[tree_key]
    uids = _tree_universe(tree_key, window_start, meta)
    crisis_months = _load_crisis_months(tree_key)
    log(f"[{tree_key}_crisis] 危機窗 {len(crisis_months)} 個月"
        f"（{crisis_months.min()}~{crisis_months.max()}）｜策略數 {len(uids):,}")
    if len(crisis_months) < 3:
        raise ValueError(f"{tree_key} 危機月數過少（{len(crisis_months)}），無法可靠算相關")
    wide = _pivot_months(months_long, uids, crisis_months)
    return _build_tree(f"{tree_key}_crisis", tree_key, wide, f_combo_map, log)


def build_co_fail_regimes(normal_assign: pd.DataFrame, crisis_assign: pd.DataFrame,
                          tree_key: str, level: str = CO_FAIL_LEVEL) -> pd.DataFrame:
    """危機期「共跌」的操作型定義（v9）：常態樹的兩個群，若在危機樹裡的成員多數
    被分進同一個危機群，代表它們在危機時塌在一起——即使常態時期看起來是不同的
    一群策略。

    做法：每個常態群，取其成員在危機樹裡的 cluster 眾數（危機期的「主要去向」），
    視為該群的危機期歸屬；兩個常態群若危機期歸屬相同，互為 co_fail 對象。
    """
    col = f"cluster_{level}"
    n = normal_assign[normal_assign.tree_id == f"{tree_key}_normal"][[C.PK, col]]
    cr = crisis_assign[crisis_assign.tree_id == f"{tree_key}_crisis"][[C.PK, col]]
    merged = n.merge(cr, on=C.PK, suffixes=("_normal", "_crisis"))

    rows = []
    for g, sub in merged.groupby(f"{col}_normal"):
        vc = sub[f"{col}_crisis"].value_counts()
        rows.append({"tree_key": tree_key, "level": level, "cluster_normal": int(g),
                     "n_members": int(len(sub)), "crisis_dest_cluster": int(vc.index[0]),
                     "crisis_dest_share": float(vc.iloc[0] / len(sub))})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    peers_by_dest = out.groupby("crisis_dest_cluster")["cluster_normal"].apply(list).to_dict()
    out["co_fail_peers"] = out.apply(
        lambda r: "|".join(str(x) for x in peers_by_dest[r["crisis_dest_cluster"]]
                           if x != r["cluster_normal"]),
        axis=1)
    out["n_co_fail_peers"] = out["co_fail_peers"].apply(lambda s: 0 if not s else len(s.split("|")))
    return out


def run(trees: list[str] | None = None, build_crisis: bool = True, log=print) -> dict:
    # ⚠️ 兩個獨立manifest都要驗：STAGE1根目錄=stage1_scan（returns_monthly等），
    # _marks/=stage1_marks（strategy_marks，下面is_usable過濾要用）。2026-08-25
    # code review修正前，兩個階段共用一份manifest、後寫的蓋掉先寫的，這裡曾經
    # 只驗到其中一半、另一半完全沒有雜湊保護，見 stage1_marks.py 的說明。
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE1 / "_marks")
    trees = trees or ["TW", "US", "XM"]

    log("載入 returns_monthly / returns_meta / strategy_marks / candidate_index …")
    months_long = pd.read_parquet(paths.STAGE1 / "returns_monthly.parquet")
    meta = pd.read_parquet(paths.STAGE1 / "returns_meta.parquet")
    idx = pd.read_parquet(paths.STAGE0 / "candidate_index.parquet")
    # XM 樹台美策略字串可能碰撞（DD-10），F組合比較必須連市場一起比
    f_combo_map = (idx.market.astype(str) + "::" + idx.f_combo.astype(str))
    f_combo_map.index = idx.strategy_uid

    # 🔴 修過的 bug：v9 明講「階段2、3只對usable_pool算，不對全庫」，但 meta
    # （returns_meta）是階段1_scan 的全量輸出，本身不含 is_usable 過濾——之前
    # 版本直接拿 meta 建樹，等於把階段1尾端硬篩掉的策略（低EffN不輪動/空手過長）
    # 全部漏回HRP。實測：TW污染6.26%(446)、US 3.70%(321)、XM 4.85%(767)，
    # 比例跟階段1回報的淘汰率幾乎一致——證實是「篩選形同沒生效」而非邊緣案例。
    marks = pd.read_parquet(paths.STAGE1 / "strategy_marks.parquet")
    usable = set(marks.loc[marks.is_usable, C.PK])
    n_before = len(meta)
    meta = meta[meta.strategy_uid.isin(usable)]
    log(f"  usable_pool 過濾：returns_meta {n_before:,} → {len(meta):,}"
        f"（排除 {n_before - len(meta):,} 個階段1標記為不可用的策略）")
    log(f"  {len(months_long):,} 筆月報酬｜{len(meta):,} 個可用策略\n")

    paths.STAGE3.mkdir(parents=True, exist_ok=True)
    results = {}
    for t in trees:
        results[f"{t}_normal"] = run_tree(t, months_long, meta, f_combo_map, log)
    if build_crisis:
        for t in trees:
            results[f"{t}_crisis"] = run_crisis_tree(t, months_long, meta, f_combo_map, log)

    all_assign = pd.concat([r["assign"] for r in results.values()], ignore_index=True)
    all_assign["tree_id"] = all_assign["tree_id"].astype("category")
    all_meta = pd.concat([r["cluster_meta"] for r in results.values()], ignore_index=True)
    all_meta["tree_id"] = all_meta["tree_id"].astype("category")
    all_meta["level"] = all_meta["level"].astype("category")

    C.validate(all_assign, C.CLUSTER_ASSIGN)
    C.validate(all_meta, C.CLUSTER_META)
    log("✓ cluster_assign / cluster_meta 契約通過")

    outs = []
    p = paths.STAGE3 / "cluster_assign.parquet"
    all_assign.to_parquet(p, compression="zstd", index=False); outs.append(p)
    p = paths.STAGE3 / "cluster_meta.parquet"
    all_meta.to_parquet(p, compression="zstd", index=False); outs.append(p)

    for tid, r in results.items():
        p = paths.STAGE3 / f"cluster_corr_matrix_{r['tree_id']}.parquet"
        r["corr_l1"].to_parquet(p, compression="zstd"); outs.append(p)
        p = paths.STAGE3 / f"linkage_{r['tree_id']}.npy"
        np.save(p, r["link"]); outs.append(p)

    co_fail_tables = []
    if build_crisis:
        log("=" * 66)
        log("co_fail_regimes（危機期共跌，見模組開頭定義）")
        log("=" * 66)
        for t in trees:
            cf = build_co_fail_regimes(all_assign, all_assign, t, CO_FAIL_LEVEL)
            co_fail_tables.append(cf)
            n_grouped = int((cf["n_co_fail_peers"] > 0).sum())
            log(f"[{t}] {CO_FAIL_LEVEL} 常態群 {len(cf)} 個，其中 {n_grouped} 個"
                f"在危機期與至少一個其他群塌在一起")
        all_co_fail = pd.concat(co_fail_tables, ignore_index=True)
        all_co_fail["tree_key"] = all_co_fail["tree_key"].astype("category")
        all_co_fail["level"] = all_co_fail["level"].astype("category")
        C.validate(all_co_fail, C.CO_FAIL_REGIMES)
        p = paths.STAGE3 / "co_fail_regimes.parquet"
        all_co_fail.to_parquet(p, compression="zstd", index=False); outs.append(p)
        log("✓ co_fail_regimes 契約通過")

        # 結構塌縮診斷：normal vs crisis 在 L1 的最大群佔比（v9「有效群數大幅下降」的量化）
        log("")
        for t in trees:
            dn = results[f"{t}_normal"]["level_diag"]["L1"]
            dc = results[f"{t}_crisis"]["level_diag"]["L1"]
            log(f"[{t}] L1 最大群佔比：normal {dn['max_share']:.1%} → "
                f"crisis {dc['max_share']:.1%}"
                f"（{'塌縮' if dc['max_share'] > dn['max_share'] else '未塌縮'}）")

    freeze.write_manifest(
        "stage3_hrp", paths.STAGE3,
        inputs=[paths.STAGE1 / "returns_monthly.parquet",
               paths.STAGE1 / "returns_meta.parquet",
               paths.STAGE1 / "strategy_marks.parquet",
               paths.STAGE0 / "candidate_index.parquet"]
              + ([paths.STAGE2 / "regime" / f"regime_table_{m}.parquet" for m in C.MARKETS]
                 if build_crisis else []),
        outputs=outs,
        params={"trees_built": list(results),
               "L1_target": L1_TARGET, "L3_target": L3_TARGET,
               "L1_target_method": "silhouette（H-03，見cluster_count_selection.py）",
               "linkage_method_chosen": {tid: r["method"] for tid, r in results.items()},
               "linkage_selection_criterion": "L3切割時最大群佔比最低者勝出，非cophenetic"
                                              "（見程式註解：single linkage 實測出現鏈狀效應)",
               "method_comparison": {tid: r["method_comparison"] for tid, r in results.items()},
               "level_diag": {tid: r["level_diag"] for tid, r in results.items()},
               "cophenetic": {tid: r["cophenetic"] for tid, r in results.items()},
               "ari_l3_vs_fcombo": {tid: r["ari_l3"] for tid, r in results.items()},
               "n_dropped_zero_var": {tid: r["n_dropped_zero_var"] for tid, r in results.items()},
               "windows": {k: list(v) for k, v in C.HRP_WINDOWS.items()},
               "crisis_months_n": ({t: len(_load_crisis_months(t)) for t in trees}
                                   if build_crisis else {})},
        notes="normal+crisis 六棵樹" if build_crisis else "⚠️ 只有 normal 樹（--normal-only）",
    )
    return results


def _report(results: dict, log=print) -> None:
    log("=" * 66)
    log("階段3 HRP · 驗收摘要")
    log("=" * 66)
    for tid, r in results.items():
        log(f"[{r['tree_id']}] {r['n_strategies']:,} 策略 × {r['n_months']} 月"
            f"｜method={r['method']}（cophenetic={r['cophenetic']:.3f}）"
            f"｜PSD min_eig={r['psd_min_eig']:.2e}｜ARI(L3 vs F組合)={r['ari_l3']:.3f}"
            f"｜{r['seconds']:.0f}s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.stage3_hrp")
    ap.add_argument("--tree", choices=["TW", "US", "XM"], action="append",
                    help="只跑指定的市場（可重複給多次）；預設全部三個")
    ap.add_argument("--normal-only", action="store_true",
                    help="只跑 normal 樹，不建 crisis 樹（除錯/測試用）")
    a = ap.parse_args(argv)
    t0 = time.time()
    results = run(trees=a.tree, build_crisis=not a.normal_only)
    print(f"\n總耗時 {time.time()-t0:.0f}s")
    _report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
