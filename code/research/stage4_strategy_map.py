# -*- coding: utf-8 -*-
"""階段 4 · 彙整成 strategy_map（W-13）

輸入 ← `_frozen/stage0/candidate_index.parquet`（身份/結構/整段績效）
        `_frozen/stage1/strategy_scan.parquet`（C-3~C-5 原始指標）
        `_frozen/stage1/strategy_marks.parquet`（等級/is_usable）
        `_frozen/stage1/returns_monthly.parquet`（regime_fit／macro_fit 的原料）
        `_frozen/stage2/regime/regime_table_{market}.parquet`（2a）
        `_frozen/stage2/macro/macro_history_{market}.parquet`（2b）
        `_frozen/stage3/cluster_assign.parquet` + `co_fail_regimes.parquet`（階段3）
輸出 → `_frozen/stage4/strategy_map.parquet`（① 主表，一列一策略）
        `_frozen/stage4/regime_performance.parquet`（regime_fit 的完整數字）
        `_frozen/stage4/macro_performance.parquet`（macro_fit 的完整數字）

⚠️ **本階段是彙整層，不重算**（研究部 v9：「階段4 本質是彙整，非分類」）：
   絕大多數欄位是把階段0/1/3已經算好的東西 join 起來。真正**新算**的只有三類：
     1. `regime_fit`／`regime_performance`：策略逐月報酬 × 該市場自己的 regime_table。
     2. `macro_best_cell`／`macro_performance`：策略逐月報酬 × 該市場自己的四格表。
     3. `v1_beneficial`：同一 (market, f_combo, C_id) 下 v1 的 CAGR 是否優於 v0。
   ②③④⑤（returns/、clusters/、macro/、regime/）**不複製**，本階段只讀取、
   引用既有的凍結目錄——它們已經是階段1/2/3自己的凍結產物，見 run() 的
   freeze.verify_inputs 呼叫與 manifest 的 inputs 清單。

⚠️ **regime_fit／macro_fit 用策略「全部」報酬歷史，不受 DD-03 共同窗限制**——
   這跟 cluster_L1/L2/L3（DD-03 窗內才有）是不同範疇：regime_fit 是「這個策略
   自己的逐月報酬碰到熊市時表現如何」，跟 HRP 要求「所有策略同一段窗才能算
   相關」無關，用愈長的歷史愈準，沒有理由裁短。

⚠️ **regime_fit 標籤定義是本階段對 GateC「含『熊市抗跌』等」的解讀，非架構
   文件的精確公式**（GateC 原文沒給數字門檻）：
     該策略在某 regime 標籤的月份（跨所有該標籤的歷史段）平均報酬 >= 0，
     且月數 >= `contracts.REGIME_FIT_MIN_MONTHS`（樣本太少不下判斷），才貼
     「熊市抗跌」/「危機抗跌」標籤。只對熊/危機兩個標籤貼標——GateC 原文
     舉的例子都是防禦性標籤，牛市/盤整表現好不算賣點，不需要對稱地貼
     「牛市強勢」；完整的四格數字仍留在 regime_performance 附屬表可查。

⚠️ **macro_fit 信心分級門檻同樣是本階段的解讀**（GateC「信心門檻n：待資料
   看分布定」，`contracts.MACRO_CONFIDENCE_CUTS`）：n>=12（一年）=高、
   n>=6=中、其餘=低。

⚠️ **cluster_L1/L2/L3/co_fail_peers 只投影「該策略自己市場的常態樹」**——
   每個策略同時屬於最多 4 棵樹（自己市場的 normal/crisis + XM 的
   normal/crisis），主表放不下 4 套群 id，這裡選最直接相關的一套當便利欄位，
   完整六棵樹的結果仍以 `cluster_assign.parquet`／`co_fail_regimes.parquet`
   為準（附屬表，見模組開頭）。DD-03 窗外的少量策略（hist_start 晚於共同窗
   起點）沒有 cluster_id，留 NaN。

❌ **本階段明確不做（範疇超出目前資料/未經授權觸發LLM，非遺漏）**：
   - `return_story` 四道安檢判決：其中「靠少數股」「規模效應」可用既有的
     top1_share/effective_n/smallcap_share 表達，但「產業β」需要持股的產業
     分類逐月資料——這條資料在目前的資料庫/管線裡完全不存在（沒有任何
     industry/sector 欄位被收集或計算過），硬做會是編造，不做。
   - `cluster_story`（LLM 點③，Opus 群間互補因果解釋）：這是要花錢呼叫
     LLM 的步驟，且研究部v9定位它是「離線一次性」產物，不該由本階段自動
     觸發——留給使用者決定何時要跑。

用法：
    cd code
    python -m research.stage4_strategy_map
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, paths

CO_FAIL_LEVEL = "L1"   # 跟 stage3_hrp.CO_FAIL_LEVEL 一致


# ============================================================================
# 月份 → 標籤 的查表（regime_fit／macro_fit 共用的展開邏輯）
# ============================================================================

def _regime_month_labels(market: str) -> pd.Series:
    """該市場全部月份 → regime 標籤（牛/熊/危機/盤整），index=Period[M]。"""
    seg = pd.read_parquet(paths.STAGE2 / "regime" / f"regime_table_{market}.parquet")
    seg = seg.sort_values("start")
    out: dict[pd.Period, str] = {}
    for r in seg.itertuples():
        months = pd.period_range(r.start.to_period("M"), r.end.to_period("M"), freq="M")
        for m in months:
            out[m] = r.label   # 段界共用月份採「較晚的段」，跟 stage2c 的 month_range 同慣例
    return pd.Series(out)


def _macro_month_cells(market: str) -> pd.Series:
    """該市場全部月份 → 投資時鐘四格，index=Period[M]（缺值月份自然不在裡面）。"""
    mh = pd.read_parquet(paths.STAGE2 / "macro" / f"macro_history_{market}.parquet")
    mh = mh.dropna(subset=["clock_cell"])
    return pd.Series(mh.clock_cell.values, index=mh.month.values)


# ============================================================================
# 新算 1：regime_fit
# ============================================================================

def compute_regime_performance(months_long: pd.DataFrame, idx: pd.DataFrame,
                               log=print) -> pd.DataFrame:
    """每策略 × regime標籤 的表現統計（完整網格：每策略必有4個標籤列）。"""
    market_of = idx.set_index(C.PK)["market"]
    rows = []
    for m in C.MARKETS:
        label_map = _regime_month_labels(m)
        uids = market_of[market_of == m].index
        sub = months_long[months_long.strategy_uid.isin(uids)].copy()
        sub["label"] = sub["month"].map(label_map)
        sub = sub.dropna(subset=["label"])
        g = sub.groupby(["strategy_uid", "label"], observed=True)["ret"]
        stat = g.agg(n_months="count", avg_ret="mean",
                    win_ratio=lambda s: float((s > 0).mean())).reset_index()
        stat.insert(1, "market", m)
        rows.append(stat)
        log(f"  [{m}] regime_performance：{stat.strategy_uid.nunique():,} 策略"
            f"｜{stat.n_months.sum():,} 筆(策略×標籤)月數加總")
    perf = pd.concat(rows, ignore_index=True)

    # 補完整網格：每個策略對每個 regime 標籤都要有一列（沒重疊的填 n_months=0/NaN），
    # 讓下游查詢不用處理「這個組合到底存不存在」的例外。
    full_idx = pd.MultiIndex.from_product(
        [idx[C.PK].unique(), C.REGIME_LABELS], names=["strategy_uid", "label"])
    perf = perf.set_index(["strategy_uid", "label"]).reindex(full_idx).reset_index()
    perf["market"] = perf["strategy_uid"].map(market_of)
    perf["n_months"] = perf["n_months"].fillna(0).astype(int)
    perf["market"] = perf["market"].astype("category")
    perf["label"] = pd.Categorical(perf["label"], categories=C.REGIME_LABELS)
    return perf


def derive_regime_fit(perf: pd.DataFrame) -> pd.Series:
    """`regime_fit` 標籤字串（見模組開頭定義）：pipe-joined，可能為空字串。"""
    ok = perf[(perf.n_months >= C.REGIME_FIT_MIN_MONTHS) & (perf.avg_ret >= 0)]
    tag_of = {"熊": "熊市抗跌", "危機": "危機抗跌"}
    ok = ok[ok.label.isin(tag_of)]
    tags = ok.assign(tag=ok["label"].map(tag_of)).groupby("strategy_uid")["tag"] \
             .apply(lambda s: "|".join(s))
    return tags


# ============================================================================
# 新算 2：macro_fit
# ============================================================================

def _confidence(n: int) -> str:
    if n >= C.MACRO_CONFIDENCE_CUTS["高"]:
        return "高"
    if n >= C.MACRO_CONFIDENCE_CUTS["中"]:
        return "中"
    return "低"


def compute_macro_performance(months_long: pd.DataFrame, idx: pd.DataFrame,
                              log=print) -> pd.DataFrame:
    """每策略 × 投資時鐘四格 的表現統計（完整網格：每策略必有4個格列）。"""
    market_of = idx.set_index(C.PK)["market"]
    rows = []
    for m in C.MARKETS:
        cell_map = _macro_month_cells(m)
        uids = market_of[market_of == m].index
        sub = months_long[months_long.strategy_uid.isin(uids)].copy()
        sub["clock_cell"] = sub["month"].map(cell_map)
        sub = sub.dropna(subset=["clock_cell"])
        g = sub.groupby(["strategy_uid", "clock_cell"], observed=True)["ret"]
        stat = g.agg(n_months="count", avg_ret="mean",
                    win_ratio=lambda s: float((s > 0).mean())).reset_index()
        stat.insert(1, "market", m)
        rows.append(stat)
        log(f"  [{m}] macro_performance：{stat.strategy_uid.nunique():,} 策略"
            f"｜{stat.n_months.sum():,} 筆(策略×格)月數加總")
    perf = pd.concat(rows, ignore_index=True)

    full_idx = pd.MultiIndex.from_product(
        [idx[C.PK].unique(), C.CLOCK_CELLS], names=["strategy_uid", "clock_cell"])
    perf = perf.set_index(["strategy_uid", "clock_cell"]).reindex(full_idx).reset_index()
    perf["market"] = perf["strategy_uid"].map(market_of)
    perf["n_months"] = perf["n_months"].fillna(0).astype(int)
    perf["confidence"] = perf["n_months"].apply(_confidence)
    perf.loc[perf.n_months == 0, "confidence"] = None
    perf["market"] = perf["market"].astype("category")
    perf["clock_cell"] = pd.Categorical(perf["clock_cell"], categories=C.CLOCK_CELLS)
    perf["confidence"] = pd.Categorical(perf["confidence"], categories=C.CONFIDENCE)
    return perf


def derive_macro_summary(perf: pd.DataFrame) -> pd.DataFrame:
    """每策略的「表現最好的格」摘要（主表便利欄位；完整4格數字在附屬表）。"""
    has_data = perf[perf.n_months > 0]
    best = has_data.loc[has_data.groupby("strategy_uid")["avg_ret"].idxmax()]
    return best.set_index("strategy_uid")[["clock_cell", "avg_ret"]].rename(
        columns={"clock_cell": "macro_best_cell", "avg_ret": "macro_best_cell_avg_ret"})


# ============================================================================
# 新算 3：v1_beneficial
# ============================================================================

def compute_v1_beneficial(idx: pd.DataFrame, log=print) -> pd.Series:
    """同一 (market, f_combo, C_id) 下，v1 的 CAGR 是否優於 v0——group級事實，
    廣播給該組的 v0/v1 兩列（兩列拿到相同值，讓查任一個V變體都能看到答案）。
    只有一半（一個V）的組合留 NaN（沒有對照組可比）。
    """
    key = ["market", "f_combo", "C_id"]
    g = idx.groupby(key, dropna=False)
    cagr_v0 = g.apply(lambda s: s.loc[s.V == "v0", "CAGR"].iloc[0] if (s.V == "v0").any() else np.nan)
    cagr_v1 = g.apply(lambda s: s.loc[s.V == "v1", "CAGR"].iloc[0] if (s.V == "v1").any() else np.nan)
    both = cagr_v0.notna() & cagr_v1.notna()
    beneficial = pd.Series(pd.NA, index=cagr_v0.index, dtype="boolean")
    beneficial[both] = (cagr_v1[both] > cagr_v0[both]).astype("boolean")

    tmp = idx[key].drop_duplicates().set_index(key)
    tmp["v1_beneficial"] = beneficial.reindex(tmp.index)
    out = idx.merge(tmp.reset_index(), on=key, how="left")["v1_beneficial"]
    out.index = idx[C.PK]
    n_comparable = int(both.sum()) * 2
    log(f"  v1_beneficial：{n_comparable:,}/{len(idx):,} 列有對照組可比"
        f"（{beneficial[both].mean():.1%} 的子樹 v1 優於 v0）")
    return out


# ============================================================================
# HRP 投影：自己市場的常態樹
# ============================================================================

def project_cluster_info(idx: pd.DataFrame, log=print) -> pd.DataFrame:
    assign = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")
    co_fail = pd.read_parquet(paths.STAGE3 / "co_fail_regimes.parquet")
    co_fail = co_fail[co_fail.level == CO_FAIL_LEVEL]

    rows = []
    for m in C.MARKETS:
        a = assign[assign.tree_id == f"{m}_normal"][
            [C.PK, "cluster_L1", "cluster_L3"]].copy()   # cluster_L2 移除，見 H-04
        cf = co_fail[co_fail.tree_key == m].set_index("cluster_normal")["co_fail_peers"]
        a["co_fail_peers"] = a["cluster_L1"].map(cf)
        rows.append(a)
        log(f"  [{m}] cluster 投影：{len(a):,}/{(idx.market==m).sum():,} 策略有 normal 樹群 id")
    out = pd.concat(rows, ignore_index=True).set_index(C.PK)
    return out.reindex(idx[C.PK])


# ============================================================================
# 主流程
# ============================================================================

def build(log=print) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # ⚠️ 兩個獨立manifest都要驗，理由同 stage3_hrp.py 的同一處註解——
    # STAGE1根目錄=stage1_scan（strategy_scan/returns_monthly），
    # _marks/=stage1_marks（strategy_marks），本函式兩者都讀。
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE1 / "_marks")
    freeze.verify_inputs(paths.STAGE2 / "regime")
    freeze.verify_inputs(paths.STAGE2 / "macro")
    freeze.verify_inputs(paths.STAGE3)

    log("載入 candidate_index / strategy_scan / strategy_marks / returns_monthly …")
    idx = pd.read_parquet(paths.STAGE0 / "candidate_index.parquet")
    scan = pd.read_parquet(paths.STAGE1 / "strategy_scan.parquet")
    marks = pd.read_parquet(paths.STAGE1 / "strategy_marks.parquet")
    months_long = pd.read_parquet(paths.STAGE1 / "returns_monthly.parquet")
    log(f"  candidate_index {len(idx):,}｜strategy_scan {len(scan):,}｜"
        f"strategy_marks {len(marks):,}｜returns_monthly {len(months_long):,}\n")

    df = idx.merge(scan.drop(columns=["market"]), on=C.PK, how="inner", validate="one_to_one")
    df = df.merge(marks.drop(columns=["market"]), on=C.PK, how="inner", validate="one_to_one")
    if len(df) != len(idx):
        raise AssertionError(
            f"三表 join 後列數不符（{len(df)} vs {len(idx)}）——candidate_index/"
            f"strategy_scan/strategy_marks 是否用了不同批次的候選池？")

    log("計算 regime_performance / regime_fit …")
    regime_perf = compute_regime_performance(months_long, idx, log)
    C.validate(regime_perf, C.REGIME_PERFORMANCE)
    df["regime_fit"] = df[C.PK].map(derive_regime_fit(regime_perf)).fillna("")
    df.loc[df["regime_fit"] == "", "regime_fit"] = None
    log(f"  regime_fit 非空比例：{df['regime_fit'].notna().mean():.1%}\n")

    log("計算 macro_performance / macro_fit …")
    macro_perf = compute_macro_performance(months_long, idx, log)
    C.validate(macro_perf, C.MACRO_PERFORMANCE)
    macro_summary = derive_macro_summary(macro_perf)
    df = df.merge(macro_summary, left_on=C.PK, right_index=True, how="left")
    log(f"  macro_best_cell 覆蓋率：{df['macro_best_cell'].notna().mean():.1%}\n")

    log("計算 v1_beneficial …")
    df["v1_beneficial"] = df[C.PK].map(compute_v1_beneficial(idx, log))

    log("投影 HRP cluster 資訊（自己市場的常態樹）…")
    cluster_info = project_cluster_info(idx, log)
    df = df.merge(cluster_info, left_on=C.PK, right_index=True, how="left")

    df["market"] = df["market"].astype("category")
    for cat_col in ("return_shape", "risk_shape", "credibility_grade", "factor_type"):
        df[cat_col] = df[cat_col].astype("category")
    df["stability_grade"] = pd.Categorical(df["stability_grade"], categories=C.STABILITY_GRADES)
    df["macro_best_cell"] = pd.Categorical(df["macro_best_cell"], categories=C.CLOCK_CELLS)
    df["is_usable"] = df["is_usable"].astype(bool)
    df["v1_beneficial"] = df["v1_beneficial"].astype("boolean")

    strategy_map = df[[c.name for c in C.STRATEGY_MAP.columns]].copy()
    C.validate(strategy_map, C.STRATEGY_MAP, strict_columns=True)
    log("✓ strategy_map 契約通過")
    return strategy_map, regime_perf, macro_perf


def run(log=print) -> pd.DataFrame:
    strategy_map, regime_perf, macro_perf = build(log)
    paths.STAGE4.mkdir(parents=True, exist_ok=True)

    outs = []
    p = paths.STAGE4 / "strategy_map.parquet"
    strategy_map.to_parquet(p, compression="zstd", index=False); outs.append(p)
    p = paths.STAGE4 / "regime_performance.parquet"
    regime_perf.to_parquet(p, compression="zstd", index=False); outs.append(p)
    p = paths.STAGE4 / "macro_performance.parquet"
    macro_perf.to_parquet(p, compression="zstd", index=False); outs.append(p)

    freeze.write_manifest(
        "stage4_strategy_map", paths.STAGE4,
        inputs=[paths.STAGE0 / "candidate_index.parquet",
               paths.STAGE1 / "strategy_scan.parquet",
               paths.STAGE1 / "strategy_marks.parquet",
               paths.STAGE1 / "returns_monthly.parquet",
               paths.STAGE3 / "cluster_assign.parquet",
               paths.STAGE3 / "co_fail_regimes.parquet"]
              + [paths.STAGE2 / "regime" / f"regime_table_{m}.parquet" for m in C.MARKETS]
              + [paths.STAGE2 / "macro" / f"macro_history_{m}.parquet" for m in C.MARKETS],
        outputs=outs,
        params={"regime_fit_min_months": C.REGIME_FIT_MIN_MONTHS,
               "macro_confidence_cuts": C.MACRO_CONFIDENCE_CUTS,
               "co_fail_level": CO_FAIL_LEVEL},
        notes="彙整層，不重算既有產物；②③④⑤(returns/clusters/macro/regime)引用"
              "階段1/2/3的既有凍結目錄，不複製。②returns/=stage1/returns_monthly.parquet、"
              "③clusters/=stage3/*、④macro/=stage2/macro/*、⑤regime/=stage2/regime+consistency/*。"
              "return_story四道安檢與cluster_story(LLM點③)未做，理由見模組開頭",
    )
    log(f"\n→ strategy_map.parquet  {len(strategy_map):,} 列, {(paths.STAGE4/'strategy_map.parquet').stat().st_size/1024:.0f} KB")
    return strategy_map


def _report(df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 62)
    log("階段4 strategy_map · 驗收報告")
    log("=" * 62)
    log(f"總列數 {len(df):,}｜is_usable {df.is_usable.sum():,} ({df.is_usable.mean():.1%})")
    log(f"\nregime_fit 分布（非空才算，可多標籤）：")
    tags = df["regime_fit"].dropna().str.split("|").explode()
    log(tags.value_counts().to_string() if len(tags) else "  （無）")
    log(f"\nmacro_best_cell × market：\n{pd.crosstab(df.market, df.macro_best_cell).to_string()}")
    log(f"\nv1_beneficial（僅算有對照組的）：\n"
        f"{df['v1_beneficial'].value_counts(dropna=True).to_string()}")
    log(f"\ncluster_L1 覆蓋率（自己市場常態樹）：{df['cluster_L1'].notna().mean():.1%}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.stage4_strategy_map")
    ap.parse_args(argv)
    out = run()
    _report(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
