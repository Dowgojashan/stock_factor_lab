# -*- coding: utf-8 -*-
"""產出A · 20年情境×策略表現對照回顧表（研究部v9第二個產出物）

> **老師逐字稿要的東西**：「最後整理20年對照 + 明顯的模式趨勢（2022熊、2015/2018/2020
> 下殺段…），可做成表格」。這是**回顧型呈現物**，屬研究部——素材（regime_table、
> 各策略×四格表現、HRP互補結構）全部已在研究部產出裡，不需等實戰部agentic LLM。

**定位（產出A vs 產出B，v9明講的界線）**：
  - 產出A（本模組）＝離線回顧：把in-sample 20年切成各regime情境，對照各情境下
    「若照12格矩陣的預設條件選一組兵」的組合表現。
  - 產出B（實戰部，未做）＝當前情境即時應用：agentic LLM 現場挑兵、現場解釋。
  - A不是B的副產品——A用**純量化規則**代替B的agentic LLM判斷（見下方選兵方法），
    對應研究部v9第八部分「baseline①純量化（無LLM）」那一階，不是簡化版的B。

⚠️ **選兵方法是本階段的具體化，不是真正的Agent1**：Agent1的Step3（agentic LLM
   精挑）尚未建置（見W-05交接：只做到Step2快篩的T1/T2工具）。本模組用**純量化
   規則**代替：T1+T2篩出候選 → 依`credibility_score_pct`排序、每個cluster_L1
   最多取1檔，取到`GROUP_SIZE`檔為止（跨群分散，呼應「選互補一組而非最強N個」）。
   這個規則跟未來真正的agentic Agent1會選出**不同**的組合，是預期中的事——
   目的是先驗證「照矩陣機械選一組會長怎樣」，agentic版本上線後可以對照差異。

⚠️ **2026-08-26老師意見後修正（H-15）：危機格不再用群間分散選法**。原本危機格
   走`_pick_cluster_diversified_crisis()`（用T5/T13找危機樹低相關群），但危機樹
   樣本量過小（17-26個月），老師意見是「留但降級」——`cluster_corr_matrix`/
   `co_fail_regimes`只當論文的描述性揭露，不能拿來做選兵決策。現在危機格跟其他
   regime一樣走`_pick_diversified()`（一般跨cluster_L1分散＋更嚴格的T1門檻），
   `_pick_cluster_diversified_crisis()`函式保留但不再是預設路徑，供日後想對照
   「危機樹選法 vs 一般選法」時使用（見開發待辦追蹤.md H-15）。

**權重**：🆕 2026-08-29 改為**等權為主，HRP降為對照組**（S-04定案）。理由：讀學長
   郭鎧菘論文（結合量化交易與多代理人決策之金融交易框架初探）發現其實證結論——
   「多代理的主要貢獻偏向標的篩選，而非資金權重配置；平均權重配置在多數情境下
   能取得相對穩健的績效」，直接回答了老師會議問的「這每一群裡面到底要用績效
   最好的，還是績效去除以MVD」——答案是選誰才是重點，權重不需要另外最佳化。
   `beats_market`（贏大盤判定）以等權組合為準；HRP權重（T9遞迴二分，用該組策略
   完整共同月份算）仍照算，結果記在`*_hrp`欄位供對照，不影響任何判定或篩選。
   HRP權重反映的是策略間長期相關結構、非情境窗內擬合，作為「等權vs風險平價
   差多少」的對照組留在論文裡。

**「贏大盤」的基準**：直接用regime_table自己的`pct_change`欄（該段大盤基準漲跌幅），
   不重算——同一份資料源、同一套定義，不會有兩套基準打架的問題。

**「免費午餐」量化**：組合在情境窗內的MDD，對比「若把同一組策略各自單獨持有」的
   平均MDD——差距即為分散效果（正值＝組合確實比平均個股抗跌）。

⚠️ **月頻粒度限制**：regime_table的段是逐日切的（zigzag用日頻價格），但策略報酬
   只有月頻。段內覆蓋到的月份，其月報酬代表「整個月」而非段內實際天數，短段
   （數週）可能覆蓋到的月份大半不在段內——這是資料粒度先天限制，沿用階段2c/3
   已用過的月份展開慣例（`month_range`），不是本模組獨有的簡化。

用法：
    cd code
    python -m research.output_a
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from ops import tools as T
from . import contracts as C
from . import freeze, paths

OUT_DIR = paths.FROZEN / "output_a"

GROUP_SIZE = 5             # 一般選法：跨群取到這麼多檔為止（危機格自H-15起也走這條路徑）
K_CRISIS_GROUPS = 4        # 僅供 _pick_cluster_diversified_crisis()（H-15後非預設路徑）使用
MIN_SEGMENT_MONTHS = 2     # 段內至少要有幾個月資料才評估（MDD至少要2點才有意義）


# ============================================================================
# 選兵（純量化規則，代替尚未建置的Agent1 Step3，見模組開頭說明）
# ============================================================================

def _pick_diversified(pool: pd.DataFrame, size: int) -> list[str]:
    """跨cluster_L1分散：依credibility_score_pct排序，每群最多取1檔，取到size檔為止。"""
    pool = pool.sort_values("credibility_score_pct", ascending=False)
    seen_clusters: set[float] = set()
    picked: list[str] = []
    # 先第一輪：每群只取最好的1檔
    for r in pool.itertuples():
        if len(picked) >= size:
            break
        cl = getattr(r, "cluster_L1")
        key = cl if pd.notna(cl) else f"_nan_{r.strategy_uid}"   # 沒有群id的各自獨立看待
        if key in seen_clusters:
            continue
        seen_clusters.add(key)
        picked.append(r.strategy_uid)
    # 群數不夠size時，允許同群補第二檔（次佳者），不強行湊數但也不留空
    if len(picked) < size:
        for r in pool.itertuples():
            if len(picked) >= size or r.strategy_uid in picked:
                continue
            picked.append(r.strategy_uid)
    return picked


def _pick_cluster_diversified_crisis(pool: pd.DataFrame, market: str) -> tuple[list[str], str]:
    """危機格：種子群 + T5找的K-1個低相關群（排除co_fail重疊），每群取1檔。"""
    if pool.empty:
        return [], ""
    seed_row = pool.sort_values("credibility_score_pct", ascending=False).iloc[0]
    seed_uid, seed_cluster = seed_row.strategy_uid, seed_row.cluster_L1
    if pd.isna(seed_cluster):
        # 種子沒有群id（DD-03窗外），退化成普通分散選法
        return _pick_diversified(pool, GROUP_SIZE), "種子策略無群id，退化為普通分散選法"

    seed_info = T.t13_get_cluster_info(strategy_uid=seed_uid, level="L1")
    excluded = {int(seed_cluster)}
    if seed_info.get("co_fail_regimes"):
        excluded |= set(seed_info["co_fail_regimes"]["co_fail_peers"])

    comp = T.t5_get_complements(seed_uid, scope="own", level="L1", k=8)
    chosen_clusters = [int(seed_cluster)]
    notes = []
    if "error" not in comp:
        for c in comp["lowest_corr_clusters"]:
            if len(chosen_clusters) >= K_CRISIS_GROUPS:
                break
            cid = c["cluster_id"]
            if cid in excluded or cid in chosen_clusters:
                notes.append(f"群{cid}與已選群co_fail或重複，跳過")
                continue
            chosen_clusters.append(cid)
    picked = [seed_uid]
    for cid in chosen_clusters[1:]:
        cand = pool[pool.cluster_L1 == cid].sort_values("credibility_score_pct", ascending=False)
        if len(cand):
            picked.append(cand.iloc[0].strategy_uid)
        else:
            notes.append(f"群{cid}在候選池中無合格策略，該群未納入")
    if len(picked) < 2:
        notes.append("找不到足夠的低相關群候選，改用普通分散選法補足")
        picked = list(dict.fromkeys(picked + _pick_diversified(pool, GROUP_SIZE)))[:GROUP_SIZE]
    return picked, "；".join(notes)


# ============================================================================
# 情境窗內的組合表現（segment-restricted，非全歷史）
# ============================================================================

def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.PeriodIndex:
    return pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")


def _segment_stats(wide_returns: pd.DataFrame, uids: list[str], months: pd.PeriodIndex,
                   weights: dict[str, float]) -> dict:
    """給定情境窗月份與權重，算組合報酬/MDD + 個股平均MDD（免費午餐對照組）。"""
    cols = [m for m in months if m in wide_returns.columns]
    sub = wide_returns.loc[uids, cols].dropna(axis=1, how="any")
    if sub.shape[1] < MIN_SEGMENT_MONTHS:
        return {"n_months": int(sub.shape[1]), "insufficient": True}

    w = np.array([weights.get(u, 0.0) for u in uids])
    if w.sum() > 0:
        w = w / w.sum()
    port_ret_series = pd.Series((sub.to_numpy() * w[:, None]).sum(axis=0), index=sub.columns)
    cum = (1 + port_ret_series).cumprod()
    port_ret = float(cum.iloc[-1] - 1)
    port_mdd = float((cum / cum.cummax() - 1).min())

    indiv_mdds = []
    for u in uids:
        c = (1 + sub.loc[u]).cumprod()
        indiv_mdds.append(float((c / c.cummax() - 1).min()))
    avg_indiv_mdd = float(np.mean(indiv_mdds))

    return {"n_months": int(sub.shape[1]), "insufficient": False,
           "portfolio_ret": port_ret, "portfolio_mdd": port_mdd,
           "avg_individual_mdd": avg_indiv_mdd,
           # MDD 是負數（跌幅），portfolio「較不深」代表 portfolio_mdd 較接近0（較大）。
           # 正值＝分散有效：portfolio_mdd > avg_individual_mdd。
           "free_lunch_mdd_gain": port_mdd - avg_indiv_mdd}


# ============================================================================
# 主流程
# ============================================================================

def _wide_returns(market: str) -> pd.DataFrame:
    rl = T._returns_monthly()
    sub = rl[rl.strategy_uid.str.startswith(f"{market}::")]
    return sub.pivot(index="strategy_uid", columns="month", values="ret")


def build_one(market: str, seg, invest_type: str, wide: pd.DataFrame, log=print) -> dict:
    label = seg.label
    row = {"market": market, "seg_start": seg.start, "seg_end": seg.end, "label": label,
          "market_pct_change": float(seg.pct_change), "days": int(seg.days),
          "invest_type": invest_type}

    rec = T.t1_get_recommended_criteria(invest_type, label, market)
    base = T.t2_filter_pool(rec["criteria"], market=market, uid_whitelist=rec["uid_whitelist"],
                            usable_only=True)
    row["n_candidates"] = base["n_matched"]
    if base["n_matched"] == 0:
        row.update({"note": f"該情境×類型無適配策略（矩陣條件下0個候選，本身是有價值的發現）",
                   "n_selected": 0})
        return row

    sm = T._strategy_map()
    pool = sm[sm.strategy_uid.isin(base["strategy_uids"])][
        ["strategy_uid", "credibility_score_pct", "cluster_L1"]]

    if rec["method"] == "cluster_diversify":
        picked, note = _pick_cluster_diversified_crisis(pool, market)
    else:
        picked = _pick_diversified(pool, GROUP_SIZE)
        note = ""
    row["n_selected"] = len(picked)
    row["selected_uids"] = "|".join(picked)
    row["n_clusters_selected"] = int(pool[pool.strategy_uid.isin(picked)]["cluster_L1"].nunique())

    if len(picked) < 2:
        row["note"] = (note + "；候選不足2檔，無法算組合/HRP權重").strip("；")
        return row

    w = T.t9_compute_weights(picked)
    eq_weights = w.get("equal_weight")
    hrp_weights = w.get("hrp_weight")   # None：該組共同月數<6，HRP不可靠（T9自己的門檻）
    row["weight_scheme"] = "equal"      # S-04定案：等權為主，見模組docstring

    months = _month_range(seg.start, seg.end)
    stats = _segment_stats(wide, picked, months, eq_weights)
    row.update(stats)
    if stats.get("insufficient"):
        row["note"] = (note + f"；情境窗內共同月數過少({stats['n_months']})，不評估組合表現").strip("；")
        return row

    row["beats_market"] = bool(stats["portfolio_ret"] > row["market_pct_change"])

    # HRP 對照組（S-04：不影響上面的判定，純粹留做「等權 vs 風險平價」的論文對照）
    if hrp_weights is not None:
        hrp_stats = _segment_stats(wide, picked, months, hrp_weights)
        if not hrp_stats.get("insufficient"):
            row["portfolio_ret_hrp"] = hrp_stats["portfolio_ret"]
            row["portfolio_mdd_hrp"] = hrp_stats["portfolio_mdd"]
            row["free_lunch_mdd_gain_hrp"] = hrp_stats["free_lunch_mdd_gain"]
            row["beats_market_hrp"] = bool(hrp_stats["portfolio_ret"] > row["market_pct_change"])

    row["note"] = note
    return row


def run(log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE4)
    rows = []
    for market in C.MARKETS:
        seg_table = pd.read_parquet(paths.STAGE2 / "regime" / f"regime_table_{market}.parquet")
        wide = _wide_returns(market)
        log(f"[{market}] {len(seg_table)} 段 × 3 類型 = {len(seg_table)*3} 格")
        for seg in seg_table.itertuples():
            for itype in ("保守型", "積極型", "全天候"):
                rows.append(build_one(market, seg, itype, wide, log))
    out = pd.DataFrame(rows)

    n_ok = int(out["beats_market"].notna().sum()) if "beats_market" in out else 0
    n_beats = int(out["beats_market"].sum()) if "beats_market" in out else 0
    log(f"\n可評估的情境×類型格：{n_ok}/{len(out)}｜其中贏大盤：{n_beats}/{n_ok if n_ok else 1}"
       f"（{n_beats/n_ok:.1%})" if n_ok else "")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "scenario_table.parquet"
    out.to_parquet(p, compression="zstd", index=False)
    freeze.write_manifest(
        "output_a", OUT_DIR,
        inputs=[paths.STAGE4 / "strategy_map.parquet",
               paths.STAGE3 / "cluster_assign.parquet", paths.STAGE3 / "co_fail_regimes.parquet"]
              + [paths.STAGE2 / "regime" / f"regime_table_{m}.parquet" for m in C.MARKETS],
        outputs=[p],
        params={"group_size": GROUP_SIZE, "k_crisis_groups": K_CRISIS_GROUPS,
               "min_segment_months": MIN_SEGMENT_MONTHS,
               "selection_method": "純量化規則代替Agent1 Step3，見模組docstring",
               "weight_scheme": "equal（S-04定案，HRP降為*_hrp對照欄位，不影響beats_market）"},
        notes="產出A：20年情境×策略表現對照表，3類投資人×全部regime段。"
              "選兵用純量化規則(baseline①)，非真正agentic LLM(Agent1尚未建置)",
    )
    log(f"→ {p}  {len(out):,} 列")
    return out


def _report(df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 66)
    log("產出A · 驗收摘要")
    log("=" * 66)
    for market in C.MARKETS:
        for itype in ("保守型", "積極型", "全天候"):
            sub = df[(df.market == market) & (df.invest_type == itype)]
            n_no_cand = int((sub.n_candidates == 0).sum())
            n_eval = int(sub["beats_market"].notna().sum()) if "beats_market" in sub else 0
            n_beat = int(sub["beats_market"].sum()) if "beats_market" in sub else 0
            hrp_line = ""
            if "beats_market_hrp" in sub:
                n_eval_hrp = int(sub["beats_market_hrp"].notna().sum())
                n_beat_hrp = int(sub["beats_market_hrp"].sum())
                hrp_line = f"｜HRP對照{n_beat_hrp}/{n_eval_hrp if n_eval_hrp else 1}"
            log(f"[{market}/{itype}] {len(sub)}段｜無候選{n_no_cand}｜可評估{n_eval}｜"
               f"贏大盤(等權){n_beat}/{n_eval if n_eval else 1}{hrp_line}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.output_a")
    ap.parse_args(argv)
    out = run()
    _report(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
