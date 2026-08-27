# -*- coding: utf-8 -*-
"""實戰部工具層（T1~T13，W-05）——strategy_map 與 agentic LLM 之間的橋。

依實戰部架構v8 §3.5：中粒度、原子化（各工具獨立、互不依賴）、無狀態（輸入→輸出，
不記憶、不依賴呼叫順序；模組內的 `functools.lru_cache` 只是效能快取，快取的是
唯讀凍結檔本身，不影響函式的輸入→輸出純粹性）、查詢類回傳一律附 `confidence`。

**策略ID一律是 `strategy_uid`（= market::strategy），不可傳裸 strategy**——
v8.1的明文規定，台美策略字串碰撞1,585個，傳裸字串會在跨市場場景取到錯的策略。

依骨架步驟分四組（見架構文件 §3.5 工具清單）：
    Step2快篩：T1 get_recommended_criteria／T2 filter_pool
    Step3查情報：T3 get_strategy_profile／T4 get_macro_fit／T5 get_complements／
                T6 check_correlation／T7 get_return_story_verdict／T13 get_cluster_info
    Step4配權/組合檢查：T8 compute_portfolio_risk／T9 compute_weights
    Step5解釋：T10 generate_return_story_text
    Agent0用：T11 get_current_regime／T12 query_macro_model

⚠️ **T10（2026-08-25 實作）呼叫真實LLM會花錢**：預設不主動觸發，由使用者決定
   何時呼叫。受 `utils/openai_quota` 的額度偵測保護，額度用盡會明確raise、
   不會靜默失敗或重試。

⚠️ **T5 的「群間互補解釋文字」相依 LLM點③ 的離線產物**（`research/cluster_story.py`，
   要花錢跑）。沒跑過時 `explanation` 為 None，工具仍照常回傳客觀數字（相關值/
   成員/代表策略），絕不編造文字。**互補程度是程式判定的**（`contracts.
   COMPLEMENTARITY_CUTS`），LLM 只為既定判決寫說明，見該模組 docstring。

⚠️ **T7 的「產業β」安檢回傳 None**——需要持股逐月產業分類資料，目前資料庫/
   管線完全沒有這項資料收集（見階段4 stage4_strategy_map.py docstring），
   硬做會是編造。

⚠️ **T1 的分位數門檻是本階段的具體化**（架構文件只寫「標準/更嚴/最嚴」等
   質化描述、數值本身「待資料」）：本模組在**呼叫當下**用 `market` 篩出的
   strategy_map 即時算市場內分位數，不是寫死的絕對數字——因為台美分布差異
   很大（例如 smallcap_share 中位數 TW 32.3% vs US 46.8%），寫死絕對值會讓
   同一個「嚴格」門檻在兩個市場意義完全不同。對照表見 `_MDD_PCT_LEVEL` 等常數。

用法：
    cd code
    python -c "from ops import tools; print(tools.t3_get_strategy_profile(['TW::...']))"
"""
from __future__ import annotations

import functools
import json
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd

from research import contracts as C
from research import hrp, paths

RegimeLabel = Literal["牛", "熊", "危機", "盤整"]
InvestType = Literal["保守型", "積極型", "全天候"]


# ============================================================================
# 快取載入（唯讀凍結檔；同一行程內重複呼叫不必重讀硬碟）
# ============================================================================

@functools.lru_cache(maxsize=1)
def _strategy_map() -> pd.DataFrame:
    return pd.read_parquet(paths.STAGE4 / "strategy_map.parquet")


@functools.lru_cache(maxsize=1)
def _cluster_assign() -> pd.DataFrame:
    return pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")


@functools.lru_cache(maxsize=1)
def _cluster_meta() -> pd.DataFrame:
    return pd.read_parquet(paths.STAGE3 / "cluster_meta.parquet")


@functools.lru_cache(maxsize=1)
def _co_fail() -> pd.DataFrame:
    return pd.read_parquet(paths.STAGE3 / "co_fail_regimes.parquet")


@functools.lru_cache(maxsize=1)
def _cluster_story_cached(mtime: float) -> pd.DataFrame:
    return pd.read_parquet(paths.STAGE3 / "cluster_story.parquet")


def _cluster_story() -> pd.DataFrame | None:
    """LLM點③的凍結解釋文字。**可能不存在**——這是要花錢跑的離線一次性產物
    （`python -m research.cluster_story`），沒跑過就回 None，呼叫端須自行處理，
    不可假設一定有；不存在時 T5/T13 照常回傳客觀數字，只是沒有解釋文字。

    ⚠️ 快取以檔案 mtime 當 key，不直接快取「不存在」這個結果——否則在同一個
    行程裡先查過（回None）、之後才跑出產物的話，會永遠拿到過期的 None。
    """
    p = paths.STAGE3 / "cluster_story.parquet"
    if not p.exists():
        return None
    return _cluster_story_cached(p.stat().st_mtime)


def _lookup_story(tree_id: str, level: str, a: int, b: int) -> dict | None:
    """查某一對群的解釋文字。群對在表裡以 (小,大) 正規化儲存，故查詢前先排序。"""
    df = _cluster_story()
    if df is None:
        return None
    lo, hi = sorted((int(a), int(b)))
    r = df[(df.tree_id == tree_id) & (df.level == level)
           & (df.cluster_a == lo) & (df.cluster_b == hi)]
    if r.empty:
        return None
    r = r.iloc[0]
    return {"complementarity": r.complementarity, "mechanism_note": r.mechanism_note,
            "complement_note": r.complement_note, "caveat": r.caveat, "model": r.model}


@functools.lru_cache(maxsize=1)
def _returns_monthly() -> pd.DataFrame:
    return pd.read_parquet(paths.STAGE1 / "returns_monthly.parquet")


@functools.lru_cache(maxsize=1)
def _regime_performance() -> pd.DataFrame:
    return pd.read_parquet(paths.STAGE4 / "regime_performance.parquet")


@functools.lru_cache(maxsize=1)
def _macro_performance() -> pd.DataFrame:
    return pd.read_parquet(paths.STAGE4 / "macro_performance.parquet")


@functools.lru_cache(maxsize=8)
def _cluster_corr_matrix(tree_id: str) -> pd.DataFrame:
    df = pd.read_parquet(paths.STAGE3 / f"cluster_corr_matrix_{tree_id}.parquet")
    df.columns = df.columns.astype(int)
    df.index = df.index.astype(int)
    return df


@functools.lru_cache(maxsize=8)
def _macro_static(market: str) -> tuple[dict, dict]:
    zp = json.loads((paths.STAGE2 / "macro" / f"zscore_params_{market}.json").read_text(encoding="utf-8"))
    cb = json.loads((paths.STAGE2 / "macro" / f"clock_bounds_{market}.json").read_text(encoding="utf-8"))
    return zp, cb


def _strategy_market(uid: str) -> str:
    return uid.split("::", 1)[0]


def _tree_id_for(uid: str, scope: Literal["own", "xm"] = "own",
                 variant: Literal["normal", "crisis"] = "normal") -> str:
    key = "XM" if scope == "xm" else _strategy_market(uid)
    return f"{key}_{variant}"


def _sanitize(v: Any) -> Any:
    """NaN/NaT/pd.NA → None，供 JSON 序列化與跨進程傳遞。"""
    if isinstance(v, (list, dict)):
        return v
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _confidence(n: int, high: int = 20, mid: int = 5) -> str:
    if n >= high:
        return "高"
    if n >= mid:
        return "中"
    return "低"


# ============================================================================
# T1 · get_recommended_criteria（12格矩陣，Step2快篩用）
# ============================================================================

#: 質化描述 → 市場內分位數門檻（見模組開頭「T1的分位數門檻」說明）
_MDD_PCT_LEVEL = {"標準": 50, "更嚴": 70, "最嚴": 85}       # mdd_pct 越高＝MDD越淺
_CAGR_PCT_LEVEL = {"標準": 50, "前段": 70}                  # cagr_pct
_TOP1_SHARE_Q = {"寬": 0.90, "嚴": 0.75, "最嚴": 0.50}       # top1_share 的市場內分位「上限」
_ROTATION_Q = {"寬": 0.95, "嚴": 0.90, "最嚴": 0.75}         # rotation_score 的市場內分位「上限」
_SMALLCAP_Q = {"寬": 0.90, "中": 0.75, "最嚴": 0.50}         # smallcap_share 的市場內分位「上限」
_HOLDINGS_P10_Q = 0.25                                       # holdings_p10 下限＝市場內第25百分位


def _q(market: str, col: str, q: float) -> float:
    sm = _strategy_map()
    return float(sm.loc[sm.market == market, col].quantile(q))


def _all_weather_qualifying_uids(min_labels: int = 3, min_avg_ret: float = 0.0) -> set[str]:
    """全天候用：regime_performance 裡跨至少 min_labels 個regime標籤平均報酬都 >= min_avg_ret
    （「不偏食牛市」的操作型定義，本階段解讀，見階段4 regime_fit 同一套邏輯延伸）。
    """
    rp = _regime_performance()
    ok = rp[(rp.n_months >= C.REGIME_FIT_MIN_MONTHS) & (rp.avg_ret >= min_avg_ret)]
    counts = ok.groupby("strategy_uid").size()
    return set(counts[counts >= min_labels].index)


def t1_get_recommended_criteria(invest_type: InvestType, regime: RegimeLabel, market: str) -> dict:
    """T1 · 該「類型×regime」格的預設撈取條件（丙：預設起點，LLM可有理由偏離）。

    回傳 `criteria`（T2可直接吃的 [(col,op,val),...] 條件列表）、
    `uid_whitelist`（需要額外白名單交集時才有，如全天候的跨regime穩健）、
    `method`（固定回傳"filter"；危機格門檻更嚴但**不再**用"cluster_diversify"，
    見2026-08-26老師意見：危機樹樣本量過小(17-26個月)，不可靠到不能拿來做
    決策依據，`cluster_corr_matrix`/`co_fail_regimes`只能當論文的描述性揭露，
    見開發待辦追蹤.md H-15。"cluster_diversify"仍是合法method值，保留給日後
    想做「crisis樹選法 vs 一般選法」對照實驗時使用，見`output_a._pick_cluster_diversified_crisis`）。
    """
    c: list[tuple[str, str, Any]] = []
    uid_whitelist: set[str] | None = None
    method = "filter"
    notes = []

    if invest_type == "保守型":
        c.append(("mdd_pct", ">=", _MDD_PCT_LEVEL["標準"]))
        if regime == "牛":
            pass
        elif regime == "熊":
            c[-1] = ("mdd_pct", ">=", _MDD_PCT_LEVEL["更嚴"])
            c.append(("credibility_grade", "in", ["高"]))
            c.append(("regime_fit", "contains", "熊市抗跌"))
        elif regime == "危機":
            c[-1] = ("mdd_pct", ">=", _MDD_PCT_LEVEL["最嚴"])
            c.append(("credibility_grade", "in", ["高"]))
            c.append(("regime_fit", "contains", "危機抗跌"))
            notes.append("危機格：門檻已是三格中最嚴，不再用cluster_diversify"
                         "（危機樹樣本量過小不可靠，見H-15）；如需群間分散仍可自行呼叫"
                         "T13查co_fail_regimes當參考，但不是本函式的預設路徑")
        elif regime == "盤整":
            c.append(("return_shape", "==", "穩定爬升"))
            c.append(("factor_type", "!=", "動能型"))
        # 全格共用下限（保守型最嚴）
        c.append(("smallcap_share", "<=", _q(market, "smallcap_share", _SMALLCAP_Q["最嚴"])))
        c.append(("holdings_p10", ">=", _q(market, "holdings_p10", _HOLDINGS_P10_Q)))

    elif invest_type == "積極型":
        c.append(("cagr_pct", ">=", _CAGR_PCT_LEVEL["標準"]))
        c.append(("top1_share", "<=", _q(market, "top1_share", _TOP1_SHARE_Q["寬"])))
        c.append(("credibility_grade", "in", ["中", "高"]))
        c.append(("rotation_score", "<=", _q(market, "rotation_score", _ROTATION_Q["寬"])))
        if regime == "牛":
            c[0] = ("cagr_pct", ">=", _CAGR_PCT_LEVEL["前段"])
            c.append(("stability_grade", "in", ["高原"]))
        elif regime == "熊":
            c[1] = ("top1_share", "<=", _q(market, "top1_share", _TOP1_SHARE_Q["嚴"]))
            c[2] = ("credibility_grade", "in", ["高"])
            c[3] = ("rotation_score", "<=", _q(market, "rotation_score", _ROTATION_Q["嚴"]))
            c.append(("mdd_pct", ">=", _MDD_PCT_LEVEL["標準"]))
            notes.append("regime_fit偏好「熊市仍有表現」，非硬性條件，交由Step3 LLM精挑時優先排序")
        elif regime == "危機":
            c[1] = ("top1_share", "<=", _q(market, "top1_share", _TOP1_SHARE_Q["最嚴"]))
            c[2] = ("credibility_grade", "in", ["高"])
            c[3] = ("rotation_score", "<=", _q(market, "rotation_score", _ROTATION_Q["最嚴"]))
            c.append(("stability_grade", "in", ["高原"]))
            notes.append("危機格：門檻已是三格中最嚴，不再用cluster_diversify，見保守型危機格說明")
        elif regime == "盤整":
            c.append(("stability_grade", "in", ["高原"]))
            c.append(("factor_type", "!=", "動能型"))
            c.append(("return_shape", "==", "穩定爬升"))

    else:  # 全天候
        c.append(("stability_grade", "in", ["高原"]))
        # ⚠️ 實測發現：min_labels若隨regime而異（原設計危機格要求4/4全過），對台股會
        # 完全篩空——台股regime_table實測0個盤整段（全被牛/熊/危機吸收，見階段2a），
        # 導致「盤整」這個標籤永遠沒有月份可算，min_labels=3等於變相要求「牛+熊+危機
        # 全部過」，比min_labels=4更嚴卻是意外的副作用，非刻意設計。改用**跨市場一致
        # 的min_labels=2**（至少2個regime標籤不偏食，而非規定哪幾個），較貼近「不
        # 偏食牛市」的字面意義，且台美都有實質候選（TW 30個／US 1,690個，見驗證記錄）。
        uid_whitelist = _all_weather_qualifying_uids(min_labels=2)
        if regime == "熊":
            c.append(("effective_n", ">=", _q(market, "effective_n", 0.5)))
        elif regime == "危機":
            notes.append("危機格：不再用cluster_diversify（見H-15），沿用base條件"
                         "（stability_grade高原 + 跨regime穩健白名單），不額外加碼")

    return {"invest_type": invest_type, "regime": regime, "market": market,
           "criteria": c, "uid_whitelist": sorted(uid_whitelist) if uid_whitelist else None,
           "method": method, "notes": notes}


# ============================================================================
# T2 · filter_pool（純程式縮池 + cluster配額，Step2快篩用）
# ============================================================================

_OPS = {
    ">=": lambda s, v: s >= v, "<=": lambda s, v: s <= v,
    ">": lambda s, v: s > v, "<": lambda s, v: s < v,
    "==": lambda s, v: s == v, "!=": lambda s, v: s != v,
    "in": lambda s, v: s.isin(v), "not_in": lambda s, v: ~s.isin(v),
    "contains": lambda s, v: s.fillna("").astype(str).str.contains(v, regex=False),
}


def t2_filter_pool(conditions: list[tuple[str, str, Any]], market: str | None = None,
                   uid_whitelist: list[str] | None = None,
                   cluster_quota: int | None = None, cluster_level: str = "L3",
                   usable_only: bool = True) -> dict:
    """T2 · 純程式縮池。`cluster_quota` 是研究部v9發現的「快篩多樣性假象」對策——
    usable_pool看似幾萬個，實際只來自407個獨立F組合，若不加配額，篩到的候選
    可能全部來自少數幾個F組合/群，LLM以為在挑很多種、實際只有幾種真正不同的東西。
    """
    sm = _strategy_map()
    mask = pd.Series(True, index=sm.index)
    if usable_only:
        mask &= sm.is_usable
    if market:
        mask &= sm.market == market
    if uid_whitelist is not None:
        mask &= sm.strategy_uid.isin(uid_whitelist)
    for col, op, val in conditions:
        if col not in sm.columns:
            raise KeyError(f"strategy_map 沒有欄位 {col!r}")
        mask &= _OPS[op](sm[col], val)
    pool = sm[mask]
    n_before_quota = len(pool)

    n_clusters_covered = None
    if cluster_quota is not None:
        cl_col = f"cluster_{cluster_level}"
        has_cluster = pool[cl_col].notna()
        quota_part = (pool[has_cluster].sort_values("cagr_pct", ascending=False)
                         .groupby(cl_col, group_keys=False).head(cluster_quota))
        pool = pd.concat([quota_part, pool[~has_cluster]])
        n_clusters_covered = int(pool.loc[pool[cl_col].notna(), cl_col].nunique())

    uids = pool["strategy_uid"].tolist()
    return {"strategy_uids": uids, "n_matched": n_before_quota, "n_after_quota": len(uids),
           "n_clusters_covered": n_clusters_covered, "confidence": _confidence(len(uids))}


# ============================================================================
# T3 · get_strategy_profile
# ============================================================================

_PROFILE_COLUMNS = [
    "strategy_uid", "market", "f_combo", "F1_factor", "F1_band", "F2_factor", "F2_band",
    "F2_empty", "C_id", "C_source", "C_rule", "V", "v1_beneficial",
    "factor_type", "factor_type_basis",
    "CAGR", "max_drawdown", "sharpe_ann", "cagr_pct", "mdd_pct",
    "return_shape", "risk_shape",
    "credibility_grade", "credibility_score_pct", "effective_n", "top1_share", "rotation_score",
    "stability_grade",
    "holdings_median", "holdings_p10", "empty_ratio", "smallcap_share", "size_tilt_pct",
    "is_usable", "data_glitch", "regime_fit", "macro_best_cell", "macro_best_cell_avg_ret",
    "cluster_L1", "cluster_L2", "cluster_L3", "co_fail_peers",
]


def t3_get_strategy_profile(strategy_uids: list[str]) -> list[dict]:
    """T3 · 批次查完整profile（見 _PROFILE_COLUMNS，涵蓋Gate A/B/C+regime_fit+macro_fit摘要+HRP投影）。"""
    sm = _strategy_map()
    sub = sm[sm.strategy_uid.isin(strategy_uids)][_PROFILE_COLUMNS]
    missing = set(strategy_uids) - set(sub.strategy_uid)
    if missing:
        raise KeyError(f"strategy_map 找不到這些 strategy_uid（前5個）：{sorted(missing)[:5]}")
    return [{k: _sanitize(v) for k, v in rec.items()} for rec in sub.to_dict("records")]


# ============================================================================
# T13 · get_cluster_info
# ============================================================================

def t13_get_cluster_info(strategy_uid: str | None = None, tree_id: str | None = None,
                         cluster_id: int | None = None, level: str = "L1") -> dict:
    """T13 · 群結構查詢：所屬群、群內成員數/平均相關/代表策略、該群各regime表現、co_fail_regimes。

    傳 `strategy_uid` 時自動用其**自己市場的normal樹**；也可直接指定
    `tree_id`+`cluster_id` 查任一棵樹（含XM、crisis）的任一群。
    """
    ca = _cluster_assign()
    if strategy_uid is not None:
        own_tree = _tree_id_for(strategy_uid, scope="own", variant="normal")
        row = ca[(ca.strategy_uid == strategy_uid) & (ca.tree_id == own_tree)]
        if row.empty:
            return {"error": f"{strategy_uid} 不在 {own_tree}（DD-03窗外或非usable_pool）"}
        tree_id = own_tree
        cluster_id = int(row.iloc[0][f"cluster_{level}"])
    if tree_id is None or cluster_id is None:
        raise ValueError("須提供 strategy_uid，或同時提供 tree_id + cluster_id")

    meta = _cluster_meta()
    m = meta[(meta.tree_id == tree_id) & (meta.level == level) & (meta.cluster_id == cluster_id)]
    if m.empty:
        return {"error": f"{tree_id}/{level}/群{cluster_id} 不存在"}
    m = m.iloc[0]

    members = ca[(ca.tree_id == tree_id) & (ca[f"cluster_{level}"] == cluster_id)]["strategy_uid"]
    rp = _regime_performance()
    grp = rp[rp.strategy_uid.isin(members)]
    regime_perf = (grp.groupby("label", observed=True)
                      .agg(avg_ret=("avg_ret", "mean"), n_strategies=("avg_ret", "count"))
                      .reset_index())
    regime_perf["avg_ret"] = regime_perf["avg_ret"].round(4)
    regime_perf = regime_perf.astype(object).where(regime_perf.notna(), None)

    co_fail = None
    tree_key = tree_id.rsplit("_", 1)[0]
    if tree_id.endswith("_normal"):
        cf = _co_fail()
        r = cf[(cf.tree_key == tree_key) & (cf.level == level) & (cf.cluster_normal == cluster_id)]
        if not r.empty:
            r = r.iloc[0]
            co_fail = {"crisis_dest_cluster": int(r.crisis_dest_cluster),
                      "crisis_dest_share": float(r.crisis_dest_share),
                      "co_fail_peers": [int(x) for x in r.co_fail_peers.split("|") if x]}

    return {
        "tree_id": tree_id, "level": level, "cluster_id": cluster_id,
        "n_members": int(m.n_members),
        "avg_intra_corr": _sanitize(m.avg_intra_corr),
        "representative_uid": m.representative_uid,
        "regime_performance": regime_perf.to_dict("records"),
        "co_fail_regimes": co_fail,
        "confidence": _confidence(int(m.n_members), high=10, mid=3),
    }


# ============================================================================
# T5 · get_complements
# ============================================================================

def t5_get_complements(strategy_uid: str, scope: Literal["own", "xm"] = "own",
                       level: str = "L1", k: int = 3) -> dict:
    """T5 · 回傳與本策略所屬群相關最低的K個群 + 各群代表策略 + 群間相關值。

    群間互補的「凍結解釋文字」來自 LLM點③（`research/cluster_story.py`，離線一次性
    產物）。**若尚未跑過該產物，`explanation` 會是 None**——此時仍照常回傳相關值等
    客觀數字，只是沒有文字說明，呼叫端不可假設一定有。
    """
    tree_id = _tree_id_for(strategy_uid, scope=scope, variant="normal")
    ca = _cluster_assign()
    row = ca[(ca.strategy_uid == strategy_uid) & (ca.tree_id == tree_id)]
    if row.empty:
        return {"error": f"{strategy_uid} 不在 {tree_id}（DD-03窗外或非usable_pool）"}
    my_cluster = int(row.iloc[0][f"cluster_{level}"])

    corr = _cluster_corr_matrix(tree_id)
    if my_cluster not in corr.columns:
        return {"error": f"群{my_cluster}不在{tree_id}的L1相關矩陣裡（相關矩陣只存L1粒度）"}
    others = corr[my_cluster].drop(index=my_cluster, errors="ignore").dropna().sort_values()
    lowest = others.head(k)

    meta = _cluster_meta()
    reps = {}
    for cid in lowest.index:
        m = meta[(meta.tree_id == tree_id) & (meta.level == level) & (meta.cluster_id == cid)]
        reps[int(cid)] = m.iloc[0].representative_uid if len(m) else None

    return {
        "strategy_uid": strategy_uid, "tree_id": tree_id, "level": level, "my_cluster": my_cluster,
        "lowest_corr_clusters": [
            {"cluster_id": int(cid), "corr": round(float(v), 4),
             "representative_uid": reps[int(cid)],
             # 解釋文字來自LLM點③的凍結產物；沒跑過就是 None（見docstring）
             "explanation": _lookup_story(tree_id, level, my_cluster, int(cid))}
            for cid, v in lowest.items()
        ],
    }


# ============================================================================
# T6 · check_correlation（即時算，上限約50檔）
# ============================================================================

def t6_check_correlation(strategy_uids: list[str]) -> dict:
    """T6 · 兩兩相關矩陣（常態窗 + 危機窗），即時算，不再預存N²（v8改寫）。"""
    if len(strategy_uids) > 50:
        raise ValueError(f"T6上限約50檔策略，收到{len(strategy_uids)}檔")
    if len(strategy_uids) < 2:
        raise ValueError("T6至少需要2檔策略")

    rl = _returns_monthly()
    wide = rl[rl.strategy_uid.isin(strategy_uids)].pivot(
        index="strategy_uid", columns="month", values="ret").reindex(strategy_uids)

    common = wide.dropna(axis=1, how="any")   # 常態：只用全員都有資料的共同月份（不用pairwise-complete，見DD-03精神）
    corr_normal = common.T.corr().round(4) if common.shape[1] >= 2 else None

    markets = {_strategy_market(u) for u in strategy_uids}
    crisis_key = "XM" if len(markets) > 1 else next(iter(markets))
    crisis_months = set(_crisis_months(crisis_key))
    crisis_common = wide[[m for m in wide.columns if m in crisis_months]].dropna(axis=1, how="any")
    corr_crisis = crisis_common.T.corr().round(4) if crisis_common.shape[1] >= 2 else None

    return {
        "strategy_uids": strategy_uids,
        "normal": {"n_months": int(common.shape[1]),
                  "corr": corr_normal.to_dict() if corr_normal is not None else None,
                  "confidence": _confidence(common.shape[1], high=60, mid=24)},
        "crisis": {"n_months": int(crisis_common.shape[1]),
                  "corr": corr_crisis.to_dict() if corr_crisis is not None else None,
                  "confidence": _confidence(crisis_common.shape[1], high=15, mid=6)},
    }


def _crisis_months(tree_key: str):
    """借用 stage3_hrp 的危機窗定義（同一份，不重複定義規則），延遲 import 避免循環。"""
    from research import stage3_hrp as s3
    return s3._load_crisis_months(tree_key)


# ============================================================================
# T7 · get_return_story_verdict（三道安檢，產業β資料不存在）
# ============================================================================

def t7_get_return_story_verdict(strategy_uid: str) -> dict:
    """T7 · 四道安檢判決：靠少數股/產業β/規模效應/真alpha。

    ⚠️ 「產業β」需要持股逐月產業分類資料，目前資料庫/管線完全沒有這項資料收集
    （見 stage4_strategy_map.py docstring），固定回傳 None；`真alpha` 因此只
    基於前三道，引用時須註明此限制，不可宣稱「四道安檢皆通過」。
    """
    sm = _strategy_map().set_index("strategy_uid")
    if strategy_uid not in sm.index:
        return {"error": f"{strategy_uid} 不存在"}
    r = sm.loc[strategy_uid]
    few_stock = bool(r.effective_n < 30 and r.top1_share > 0.15)
    size_driven = bool(r.smallcap_share > 0.5)
    sector_beta = None
    real_alpha = bool((not few_stock) and (not size_driven) and (r.credibility_grade != "低"))
    return {
        "strategy_uid": strategy_uid,
        "靠少數股": few_stock, "產業β": sector_beta, "規模效應": size_driven,
        "真alpha": real_alpha,
        "note": "產業β資料不存在，真alpha判決僅基於前三道，引用時須註明此限制",
    }


# ============================================================================
# T8 · compute_portfolio_risk（即時算）
# ============================================================================

def t8_compute_portfolio_risk(strategy_uids: list[str], weights: list[float] | None = None) -> dict:
    """T8 · 組合層即時計算：整體回撤/波動/因子曝險集中度/台美比重/各regime整體表現
    + cluster涵蓋群數與各群權重佔比。`weights` 預設等權；若已由T9算出HRP權重可傳入。
    """
    n = len(strategy_uids)
    w = np.array(weights) if weights is not None else np.full(n, 1.0 / n)
    if len(w) != n:
        raise ValueError("weights 長度須與 strategy_uids 一致")

    rl = _returns_monthly()
    wide = rl[rl.strategy_uid.isin(strategy_uids)].pivot(
        index="strategy_uid", columns="month", values="ret").reindex(strategy_uids)
    common = wide.dropna(axis=1, how="any").sort_index(axis=1)
    port_ret = pd.Series((common.to_numpy() * w[:, None]).sum(axis=0), index=common.columns)
    cum = (1 + port_ret).cumprod()
    mdd = float((cum / cum.cummax() - 1).min()) if len(cum) else None
    ann_vol = float(port_ret.std() * np.sqrt(12)) if len(port_ret) > 1 else None
    ann_ret = float(cum.iloc[-1] ** (12 / len(port_ret)) - 1) if len(port_ret) else None

    sm = _strategy_map().set_index("strategy_uid")
    prof = sm.loc[strategy_uids]
    market_share = {"TW": float((prof.market == "TW").mean()), "US": float((prof.market == "US").mean())}
    factor_exposure_f1 = prof["F1_factor"].value_counts(normalize=True).round(3).to_dict()

    ca = _cluster_assign()
    cluster_coverage = {}
    for lvl in ("L1", "L2", "L3"):
        cids = []
        for uid, wt in zip(strategy_uids, w):
            r = ca[(ca.strategy_uid == uid) & (ca.tree_id == _tree_id_for(uid, "own", "normal"))]
            cids.append((int(r.iloc[0][f"cluster_{lvl}"]) if len(r) else None, wt))
        s = pd.Series({i: wt for i, (c, wt) in enumerate(cids) if c is not None})
        c_of = {i: c for i, (c, wt) in enumerate(cids) if c is not None}
        weight_by_cluster = pd.Series(dtype=float)
        if len(s):
            weight_by_cluster = s.groupby(pd.Series(c_of)).sum()
        cluster_coverage[lvl] = {"n_clusters": int(weight_by_cluster.shape[0]),
                                 "weight_share": {int(k): round(float(v), 4)
                                                  for k, v in weight_by_cluster.items()}}

    rp = _regime_performance()
    sub = rp[rp.strategy_uid.isin(strategy_uids)]
    regime_avg_ret = {}
    for lbl, g in sub.groupby("label", observed=True):
        gg = g.set_index("strategy_uid")["avg_ret"].reindex(strategy_uids)
        regime_avg_ret[lbl] = round(float((gg.fillna(0).to_numpy() * w).sum()), 4)

    return {
        "n_strategies": n, "weights": [round(float(x), 6) for x in w],
        "n_common_months": int(common.shape[1]),
        "portfolio_mdd": mdd, "portfolio_ann_vol": ann_vol, "portfolio_ann_ret": ann_ret,
        "market_share": market_share, "factor_exposure_F1": factor_exposure_f1,
        "cluster_coverage": cluster_coverage, "regime_avg_ret": regime_avg_ret,
        "confidence": _confidence(int(common.shape[1]), high=60, mid=24),
    }


# ============================================================================
# T9 · compute_weights（HRP遞迴二分 + 等權baseline）
# ============================================================================

def t9_compute_weights(strategy_uids: list[str]) -> dict:
    """T9 · HRP遞迴二分權重（+等權baseline對照）。LLM不碰數字，純程式即時算。

    對任意子集重新跑一次完整HRP（相關→距離→linkage→準對角化→遞迴二分）——
    這跟研究部階段3的六棵樹是「同一套演算法、不同輸入」，不是查表，因為
    使用者選的子集通常不等於整棵樹的全部成員。
    """
    n = len(strategy_uids)
    eq_w = {u: round(1 / n, 6) for u in strategy_uids}
    if n < 2:
        return {"error": "至少需要2檔策略才能算相關/權重", "equal_weight": eq_w}

    rl = _returns_monthly()
    wide = rl[rl.strategy_uid.isin(strategy_uids)].pivot(
        index="strategy_uid", columns="month", values="ret").reindex(strategy_uids)
    common = wide.dropna(axis=1, how="any")
    if common.shape[1] < 6:
        return {"error": f"共同月數過少({common.shape[1]})，HRP權重不可靠，改用等權",
               "equal_weight": eq_w, "n_common_months": int(common.shape[1])}

    # cophenetic 在 n=2 時分母可能是0（單一配對談不上「保真度」）；build_tree 內建算它
    # 但 T9 用不到這個值，這裡只是靜音掉這個無害警告，不影響權重本身的正確性。
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.cluster.hierarchy")
        result = hrp.build_tree(common.to_numpy(dtype=np.float64), method="ward")
    hrp_w = {u: round(float(x), 6) for u, x in zip(strategy_uids, result.weights)}
    return {"hrp_weight": hrp_w, "equal_weight": eq_w,
           "n_common_months": int(common.shape[1]), "psd_ok": bool(result.psd_ok),
           "confidence": _confidence(common.shape[1], high=60, mid=24)}


# ============================================================================
# T10 · generate_return_story_text（Step5解釋，LLM點④，2026-08-25 實作）
# ============================================================================

#: 結構化輸出schema——逼LLM對T7的四道安檢**各自**只能寫對應那一項的話，不能
#: 自由發揮成一整段夾雜因果推論的敘事。這是研究部v9「暫掛」清單裡「return_story
#: 殘餘幻覺」的具體對策（原文：「LLM『串』時可能偷渡程式沒給的因果連接...
#: 方向：結構化條列輸出，禁止添加程式未提供的因果」）。
#: ⚠️ 這個 response_format 的wrapper shape（type=json_schema）尚未用真實API
#: 呼叫驗證過，見 t10_generate_return_story_text docstring 的驗證待辦。
_STORY_JSON_SCHEMA = {
    "name": "return_story",
    "schema": {
        "type": "object",
        "properties": {
            "few_stock_note": {
                "type": "string",
                "description": "只針對「是否靠少數股撐報酬」這一項寫一句話，"
                               "禁止提及其他三道安檢、禁止引入未提供的數字"},
            "sector_beta_note": {
                "type": "string",
                "description": "產業β資料不存在，只能說明「此項無法判定」，禁止臆測"},
            "size_driven_note": {
                "type": "string",
                "description": "只針對「是否靠規模效應（重倉小型股）」這一項寫一句話"},
            "real_alpha_note": {
                "type": "string",
                "description": "只針對「真alpha」判決寫一句話，且必須註明"
                               "此判決僅基於前三道（產業β缺失）"},
            "summary": {
                "type": "string",
                "description": "把以上四句整合成2-3句總結，不得新增以上四句之外的任何主張"},
        },
        "required": ["few_stock_note", "sector_beta_note", "size_driven_note",
                    "real_alpha_note", "summary"],
        "additionalProperties": False,
    },
    "strict": True,
}

_STORY_SYSTEM_PROMPT = (
    "你是量化回測系統的判決轉譯器。你只能把使用者提供的『程式判決』與『支持數字』"
    "轉成通順的中文句子，禁止做任何新的判斷、禁止引用使用者沒有給你的數字、"
    "禁止提出使用者沒有給你的因果解釋。若某一項資料缺失，只能說明缺失，不能臆測。"
)


def _build_return_story_user_prompt(verdict: dict, evidence: dict) -> str:
    """組 user prompt。只把T7判決+其依據數字餵給LLM，不給其他任何策略資訊
    （不給策略名稱、不給因子邏輯、不給市場）——結構上就沒有材料可以拿去
    「發揮」出程式沒給的因果，這是抗幻覺鐵則(1)（LLM不碰篩選/判斷）的延伸。
    """
    # 數字先四捨五入再進prompt：實測（2026-08-25）不修的話，float原始精度
    # （如 effective_n=28.417404303353255）會被模型一字不漏抄進文字裡，既難讀
    # 又多燒token。四捨五入是**呈現層**的處理，判決本身仍以未捨入的原值計算，
    # 故不影響任何判定結果。
    def _n(key, digits=1, pct=False):
        v = evidence.get(key)
        if v is None:
            return "無資料"
        return f"{v:.1%}" if pct else f"{round(float(v), digits)}"

    return (
        "策略的程式判決（已凍結，不可質疑或推翻）：\n"
        f"- 靠少數股撐報酬：{verdict['靠少數股']}\n"
        f"- 產業β：{verdict['產業β']}（None＝資料不存在，非False）\n"
        f"- 規模效應（重倉小型股）：{verdict['規模效應']}\n"
        f"- 真alpha：{verdict['真alpha']}（註：{verdict['note']}）\n\n"
        "支持這些判決的客觀數字：\n"
        f"- effective_n（有效持股分散度，1/HHI）：{_n('effective_n')}\n"
        f"- top1_share（最大單股累積貢獻占比）：{_n('top1_share', pct=True)}\n"
        f"- smallcap_share（持股落在最小市值三分之一的比例）：{_n('smallcap_share', pct=True)}\n"
        f"- credibility_grade（可信度等級）：{evidence.get('credibility_grade')}\n\n"
        "請用給定的 JSON schema 輸出，且只能使用以上提供的資訊。"
        "引用數字時直接沿用上面給的寫法，不要自行改寫成更多小數位。"
    )


def t10_generate_return_story_text(strategy_uid: str, *, model: str | None = None,
                                   temperature: float | None = None) -> dict:
    """T10 · Step5解釋（LLM點④）：把T7的凍結判決串成可讀文字（按需生成，非預寫）。

    **抗幻覺設計（研究部v9定案，三道鐵則的具體實作）**：
      1. LLM不下判斷、不推翻數字——四道安檢的True/False/None全部是T7算好凍結的，
         本函式只餵給LLM「判決+支持數字」，不給策略名稱/因子邏輯等其他資訊，
         結構上讓LLM沒有材料可以「發揮」出程式沒給的因果。
      2. 結構化條列輸出（JSON schema，見`_STORY_JSON_SCHEMA`）：逼LLM對「這一項」
         只能寫「這一項」的話，不能把四道安檢混著講、不能生出schema以外的欄位。
      3. 回傳`raw_verdict`/`raw_evidence` 供事後核對文字有沒有跟凍結數字矛盾
         （「可數字反驗」鐵則——不一致就該觸發人工複查，本函式不自動做這層
         語意比對，留給呼叫端或人工審閱）。

    🔴 **`temperature` 預設不送（2026-08-25 實測修正）**：原本寫死 `temperature=0`
    以滿足 v9 方法論「溫度/seed固定」的可複現要求，但實測 gpt-5.6 系列會回
    `400 unsupported_value: 'temperature' does not support 0`，這類新型推理模型
    只接受預設值。故改為預設不送此參數；只有在確認該模型支援時才由呼叫端明確傳入。
    **連帶影響**：對不支援調溫的模型，「固定溫度」這條可複現手段不可用，可複現性
    只剩「凍結輸入（判決+數字皆為程式算好）+ 結構化schema限制輸出形狀」兩道。
    這是模型端的限制、不是實作疏漏，論文方法論若要宣稱可複現須誠實註明此點。

    ⚠️ **呼叫這個函式會真的花錢**（OpenAI Chat Completions API），受
    `utils.openai_quota` 保護——額度用盡會 raise `QuotaExhaustedError`，
    不會靜默失敗或重試；其他API錯誤raise `OpenAIAPIError`。

    ⚠️ **驗證待辦（2026-08-25，admin_key尚未到位前寫的）**：`response_format`
    用的 `{"type": "json_schema", "json_schema": {...}}` wrapper shape、
    以及 `model` 參數的合法命名，都還沒有拿真實帳號打過一次驗證——邏輯本身
    可用合成回應跑通（見 `research/tests.py` 的 `t_ops_t10_*` 系列），但欄位
    名稱/shape 是否跟目前的真實API一致，須等第一次真實呼叫才能確認。
    """
    verdict = t7_get_return_story_verdict(strategy_uid)
    if "error" in verdict:
        return verdict

    profile = t3_get_strategy_profile([strategy_uid])[0]
    evidence = {k: profile.get(k) for k in
               ("effective_n", "top1_share", "smallcap_share", "credibility_grade")}

    import requests
    from utils.config import Config
    from utils import openai_quota as OQ

    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = model or cfg.get_openai_model("t10")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _STORY_SYSTEM_PROMPT},
            {"role": "user", "content": _build_return_story_user_prompt(verdict, evidence)},
        ],
        "response_format": {"type": "json_schema", "json_schema": _STORY_JSON_SCHEMA},
    }
    if temperature is not None:      # 見 docstring：新型推理模型不接受非預設溫度
        payload["temperature"] = temperature

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=60,
    )
    OQ.raise_for_openai_response(resp)
    body = resp.json()
    story = json.loads(body["choices"][0]["message"]["content"])

    return {
        "strategy_uid": strategy_uid,
        "story": story,
        "raw_verdict": verdict,
        "raw_evidence": evidence,
        "model": model,
        "usage": body.get("usage"),
    }


# ============================================================================
# T11 · get_current_regime（Agent0用，跑當前資料）
# ============================================================================

def t11_get_current_regime(market: str, as_of: str | None = None) -> dict:
    """T11 · 用研究部階段2a**同一套**規則（同一份程式碼，非重寫）跑當前資料。

    ⚠️ zigzag本質是回顧式演算法：最後一段可能尚未被下一次反轉「確認」，
    是暫定判定（provisional），措辭須讓Agent2誠實傳達這個不確定性。
    """
    from database import Database
    from research import stage2a_regime as r2a

    db = Database(market)
    raw = db.get_taiex_data()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").drop_duplicates("date")
    if as_of:
        raw = raw[raw["date"] <= pd.Timestamp(as_of)]
    if raw.empty:
        return {"error": f"{market} 在 as_of={as_of} 之前沒有資料"}
    price = raw.set_index("date")["close"].astype(float)

    pivots = r2a.zigzag_pivots(price, r2a.DEFAULT_PARAMS["bear_thresh"], r2a.DEFAULT_PARAMS["bull_thresh"])
    segs = r2a.classify_segments(pivots, r2a.DEFAULT_PARAMS)
    cur = segs.iloc[-1]
    return {
        "market": market, "as_of": str(price.index.max().date()),
        "label": cur.label, "since": str(cur.start.date()),
        # ⚠️ 不能用 cur.pct_change（屬性存取）——pandas Series 本身有同名的
        # .pct_change() 方法，屬性存取會拿到方法物件而非該欄的值，必須用 [] 取值。
        "pct_change_so_far": round(float(cur["pct_change"]), 4),
        "days_so_far": int(cur.days),
        "provisional": True,
        "note": "最後一段尚未被下一次反轉確認，屬暫定判定（zigzag回顧式演算法的固有限制）",
    }


# ============================================================================
# T12 · query_macro_model ／ T4 · get_macro_fit（Agent0/Step3查情報用）
# ============================================================================

def _zscore_current(market: str, raw: dict[str, float | None]) -> dict[str, float | None]:
    zp, _ = _macro_static(market)
    out = {}
    for axis in ("growth", "inflation", "rate_level", "rate_direction"):
        v = raw.get(axis)
        p = zp.get(axis)
        out[f"{axis}_z"] = None if (v is None or p is None or not p.get("std")) else (v - p["mean"]) / p["std"]
    return out


def _clock_cell_for(market: str, z: dict[str, float | None]) -> str | None:
    _, cb = _macro_static(market)
    g, i = z.get("growth_z"), z.get("inflation_z")
    if g is None or i is None:
        return None
    g_hi, i_hi = g >= cb["growth_median"], i >= cb["inflation_median"]
    if g_hi and not i_hi:
        return "復甦"
    if g_hi and i_hi:
        return "過熱"
    if not g_hi and i_hi:
        return "停滯性通膨"
    return "衰退"


def _knn_analog_months(market: str, z: dict[str, float | None], k: int) -> pd.DataFrame:
    """方法一：歐氏距離找當前總經最像的歷史K個月（研究部v9 P2定案：距離公式=歐氏距離）。"""
    zcols = ["growth_z", "inflation_z", "rate_level_z", "rate_direction_z"]
    if any(z.get(c) is None for c in zcols):
        return pd.DataFrame(columns=["month", "clock_cell", "dist"])
    mh = pd.read_parquet(paths.STAGE2 / "macro" / f"macro_history_{market}.parquet")
    valid = mh.dropna(subset=zcols)
    cur = np.array([z[c] for c in zcols])
    dist = np.sqrt(((valid[zcols].to_numpy() - cur) ** 2).sum(axis=1))
    out = valid.assign(dist=dist).sort_values("dist").head(k)
    return out[["month", "clock_cell", "dist"]]


def t12_query_macro_model(market: str, current_raw: dict[str, float | None], k: int = 12) -> dict:
    """T12 · Agent0用：抓當前總經→查2b凍結模型→類比月+所在格+macro_fit。"""
    z = _zscore_current(market, current_raw)
    cell = _clock_cell_for(market, z)
    analogs = _knn_analog_months(market, z, k)
    return {
        "market": market, "clock_cell": cell, "z": {k2: _sanitize(v2) for k2, v2 in z.items()},
        "analog_months": [str(m) for m in analogs["month"]],
        "analog_detail": [{"month": str(r.month), "clock_cell": r.clock_cell, "dist": round(float(r.dist), 4)}
                          for r in analogs.itertuples()],
        "confidence": _confidence(len(analogs), high=k, mid=max(1, k // 3)),
    }


def t4_get_macro_fit(strategy_uids: list[str], current_raw: dict[str, float | None], k: int = 12) -> list[dict]:
    """T4 · Step3查情報用：方法一(類比月報酬/勝率) + 方法二(所在格表現)，兩法各自值不混。

    `current_raw` 依策略自己市場各查一次（台美總經不同套）。
    """
    mp = _macro_performance()
    rl = _returns_monthly()
    cache: dict[str, dict] = {}
    out = []
    for uid in strategy_uids:
        m = _strategy_market(uid)
        if m not in cache:
            cache[m] = t12_query_macro_model(m, current_raw, k=k)
        q = cache[m]
        analog_months = {pd.Period(x) for x in q["analog_months"]}
        r = rl[(rl.strategy_uid == uid) & (rl.month.isin(analog_months))]
        method1 = {"n_months": int(len(r)),
                  "avg_ret": _sanitize(r.ret.mean()) if len(r) else None,
                  "win_ratio": _sanitize((r.ret > 0).mean()) if len(r) else None}
        cell = q["clock_cell"]
        m2 = mp[(mp.strategy_uid == uid) & (mp.clock_cell == cell)] if cell else pd.DataFrame()
        method2 = ({"n_months": int(m2.iloc[0].n_months), "avg_ret": _sanitize(m2.iloc[0].avg_ret),
                   "win_ratio": _sanitize(m2.iloc[0].win_ratio), "confidence": m2.iloc[0].confidence}
                  if len(m2) else {"n_months": 0, "avg_ret": None, "win_ratio": None, "confidence": None})
        out.append({"strategy_uid": uid, "clock_cell": cell,
                   "method1_analog": method1, "method2_cell": method2})
    return out
