# -*- coding: utf-8 -*-
"""S-02 · 決策層資訊源設計（開發待辦追蹤.md 方向C 第二步）

老師走方向C 的第二個接縫：決策層的輸入是「總經狀態 × 群特性」，我們沒有新聞這類
非結構化資料。這裡定案三件事：
  1. 總經狀態怎麼表述給 LLM
  2. 群特性給到多細
  3. 要不要附歷史績效

⚠️ **這一項的設計品質直接決定方向C 站不站得住**（S-02原始記錄的警語）。

---------------------------------------------------------------------------
🔴 核心決策：決策層看得到「條件式」績效，看不到「無條件」績效
---------------------------------------------------------------------------
`S-01`（`cluster_macro_interface.parquet`）已經有 CAGR_median／MDD_median／
annual_ret_mean 這些欄位，但那些是「這個群整體表現好不好」的**無條件**統計量。
若決策層直接看到「群X的CAGR中位數23.5%」，幾乎必然導向「總是選CAGR最高的群」
——這正是 H-10/H-12 已經證實的陷阱在總經決策層的翻版（H-12實測：無多樣性限制
的純品質排序，在US/XM會塌縮成集中在單一群的賭注，OOS表現由一次歷史運氣主導，
不是穩健策略，見H-12結果）。

故本模組新增 `cluster_macro_conditional.parquet`（`contracts.
CLUSTER_MACRO_CONDITIONAL`）——群在投資時鐘四格（復甦/過熱/停滯性通膨/衰退）
各自的**條件式**平均報酬/勝率，把 `macro_performance.parquet`（階段4已算好的
策略層級數字）用跟 H-06/stage3_hrp 同一套「群代表=成員簡單平均」邏輯彙整到
群層級。決策層要做的判斷因此變成「現在總經狀態對應哪個clock_cell、這個clock_cell
下哪些群條件式表現較好」，而不是「無條件挑歷史最強的群」。

**群定義沿用主線六棵樹（`_frozen/stage3/cluster_assign.parquet`，全時間窗）**，
跟 H-06/H-07/H-08/S-01 用同一份，不是 H-11 的 IS-only 群——理由見 H-02 條目的
結案記錄：主線六棵樹是唯一的正式群定義，IS/OOS 是獨立的驗證分支，不取代它。

---------------------------------------------------------------------------
決策層 prompt 組裝的兩塊素材（本模組提供純資料，不含 prompt/LLM 呼叫本身）
---------------------------------------------------------------------------
  `macro_state_snapshot(market, month)` —— 總經狀態怎麼表述：只回傳 z-score
      （growth_z/inflation_z/rate_level_z/rate_direction_z，用平滑版_s3）+
      clock_cell 分類，**不回傳 month 本身**（S-03：決策層不給日期）。`month`
      是輸入參數（供未來 walk-forward 測試逐月呼叫用），但絕不出現在回傳值裡。

  `group_decision_context(tree_id, cluster_id)` —— 群特性給到多細：組合
      identity_label + 純結構欄位（市場組成、因子/估值濾網構成、群內平均相關）
      + 本模組新算的條件式績效（四格 avg_ret/win_ratio）。**明確排除**
      CLUSTER_MACRO_INTERFACE 裡的無條件績效欄位（CAGR_median/MDD_median/
      annual_ret_mean/annual_ret_std/quarterly_ret_std/best_year_ret/
      worst_year_ret/n_years_positive/pct_years_positive）——見上方核心決策。

用法：
    cd code
    python -m research.macro_decision_input
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import contracts as C
from . import freeze, paths

DEFAULT_TREES = ("TW_normal", "US_normal", "XM_normal")
LEVEL = "L1"

#: group_decision_context() 會從 CLUSTER_MACRO_INTERFACE 排除的欄位——
#: 全部是「無條件」績效統計，見模組開頭的核心決策說明。
_EXCLUDED_INTERFACE_FIELDS = (
    "CAGR_median", "MDD_median", "annual_ret_mean", "annual_ret_std",
    "quarterly_ret_std", "best_year_ret", "worst_year_ret",
    "n_years_positive", "pct_years_positive", "n_years",
)


# ============================================================================
# 群的投資時鐘四格條件式績效（純程式，無LLM）
# ============================================================================

def build_conditional_performance(trees=DEFAULT_TREES, log=print) -> pd.DataFrame:
    assign = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")
    mp = pd.read_parquet(paths.STAGE4 / "macro_performance.parquet")

    rows = []
    for tree_id in trees:
        a = assign[assign.tree_id == tree_id][[C.PK, f"cluster_{LEVEL}"]].rename(
            columns={f"cluster_{LEVEL}": "_cl"})
        merged = a.merge(mp, on=C.PK, how="left")
        n_missing = int(merged["clock_cell"].isna().sum())
        if n_missing:
            log(f"  ⚠️ [{tree_id}] {n_missing} 筆策略×clock_cell在macro_performance"
                f"查無資料（該策略可能不在DD-03共同窗內），彙總時自動排除")
        merged = merged.dropna(subset=["clock_cell"])

        for (cid, cell), g in merged.groupby(["_cl", "clock_cell"], observed=True):
            valid = g.dropna(subset=["avg_ret"])
            rows.append({
                "tree_id": tree_id, "level": LEVEL, "cluster_id": int(cid),
                "clock_cell": cell,
                "n_members_with_data": int(len(valid)),
                "avg_ret_mean": float(valid["avg_ret"].mean()) if len(valid) else None,
                "avg_ret_median": float(valid["avg_ret"].median()) if len(valid) else None,
                "win_ratio_mean": float(valid["win_ratio"].mean()) if len(valid) else None,
                "pct_high_confidence": (float((g["confidence"] == "高").mean())
                                        if len(g) else None),
            })
        log(f"[{tree_id}] {a['_cl'].nunique()} 群 × 4 個clock_cell 彙總完成")
    return pd.DataFrame(rows)


def run(trees=DEFAULT_TREES, log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE3)
    freeze.verify_inputs(paths.STAGE4)
    df = build_conditional_performance(trees=trees, log=log)
    df["tree_id"] = df["tree_id"].astype("category")
    df["level"] = df["level"].astype("category")
    df["clock_cell"] = df["clock_cell"].astype("category")
    C.validate(df, C.CLUSTER_MACRO_CONDITIONAL, strict_columns=True)
    log(f"✓ cluster_macro_conditional 契約通過（{len(df)} 列＝群數×4格）")

    p = paths.STAGE3 / "cluster_macro_conditional.parquet"
    df.to_parquet(p, compression="zstd", index=False)
    freeze.write_manifest(
        "cluster_macro_conditional", paths.STAGE3 / "_macro_conditional",
        inputs=[paths.STAGE3 / "cluster_assign.parquet",
               paths.STAGE4 / "macro_performance.parquet"],
        outputs=[p],
        params={"trees": list(trees), "level": LEVEL},
        notes="S-02：群的投資時鐘四格條件式績效，純程式算(H-06同一套群代表=成員"
              "簡單平均邏輯)，無LLM。決策層看條件式績效、不看無條件績效，理由見"
              "模組docstring。",
    )
    log(f"→ {p}")
    return df


# ============================================================================
# 決策層 prompt 素材組裝（純資料函式，不含 LLM 呼叫）
# ============================================================================

def macro_state_snapshot(market: str, month: str) -> dict:
    """給定市場與月份，回傳**不含月份本身**的總經狀態描述（S-03：決策層不給日期）。

    `month` 是查表用的參數（供未來逐月 walk-forward 測試呼叫），刻意不放進回傳值——
    呼叫端若要記錄「這是哪個月的查詢」，須自行在函式呼叫之外另外保存，不能靠這個
    回傳值反查，否則等於繞了一圈還是把日期交給了下游。
    """
    mh = pd.read_parquet(paths.STAGE2 / "macro" / f"macro_history_{market}.parquet")
    row = mh[mh.month == pd.Period(month, "M")]
    if row.empty:
        raise ValueError(f"{market} 的 macro_history 查無 {month} 這個月份")
    r = row.iloc[0]
    return {
        "growth_z": None if pd.isna(r.growth_z_s3) else round(float(r.growth_z_s3), 3),
        "inflation_z": None if pd.isna(r.inflation_z_s3) else round(float(r.inflation_z_s3), 3),
        "rate_level_z": None if pd.isna(r.rate_level_z_s3) else round(float(r.rate_level_z_s3), 3),
        "rate_direction_z": (None if pd.isna(r.rate_direction_z_s3)
                             else round(float(r.rate_direction_z_s3), 3)),
        "clock_cell": None if pd.isna(r.clock_cell) else str(r.clock_cell),
    }


def group_decision_context(tree_id: str, cluster_id: int) -> dict:
    """給定群，回傳決策層看得到的完整素材：結構特徵 + identity_label（皆來自S-01，
    排除無條件績效欄位）+ 四格條件式績效（本模組新算）。
    """
    interface = pd.read_parquet(paths.STAGE3 / "cluster_macro_interface.parquet")
    row = interface[(interface.tree_id == tree_id) & (interface.cluster_id == cluster_id)]
    if row.empty:
        raise ValueError(f"{tree_id} 群{cluster_id} 不存在於 cluster_macro_interface")
    r = row.iloc[0]
    # ⚠️ 逐值轉成原生 Python 型別（int/float/str），不能直接把 numpy 純量塞進payload：
    # numpy.int64/float64 不是原生JSON可序列化型別，`json.dumps()`會直接炸掉
    # `TypeError: Object of type int64 is not JSON serializable`——這是2026-08-31
    # 開發時實測抓到的真實bug，其他LLM prompt組裝函式(cluster_story/cluster_identity
    # 的build_prompt)都用`json.dumps()`且未加`default=str`，本函式的payload最終
    # 也是要被同樣方式序列化進prompt，此處不轉型會在下游（S-05/H-17~H-21真的組
    # prompt時）才炸開，先在源頭修掉。
    payload = {}
    for k, v in r.items():
        if k in _EXCLUDED_INTERFACE_FIELDS or k in ("tree_id", "level", "cluster_id"):
            continue
        if pd.isna(v):
            payload[k] = None
        elif isinstance(v, (bool,)):
            payload[k] = bool(v)
        elif hasattr(v, "item"):   # numpy 純量（int64/float64/bool_...）
            payload[k] = v.item()
        else:
            payload[k] = v

    cond = pd.read_parquet(paths.STAGE3 / "cluster_macro_conditional.parquet")
    cond = cond[(cond.tree_id == tree_id) & (cond.cluster_id == cluster_id)]
    payload["conditional_performance"] = {
        str(cr.clock_cell): {
            "avg_ret_median": (None if pd.isna(cr.avg_ret_median)
                              else round(float(cr.avg_ret_median), 4)),
            "win_ratio_mean": (None if pd.isna(cr.win_ratio_mean)
                              else round(float(cr.win_ratio_mean), 4)),
            "n_members_with_data": int(cr.n_members_with_data),
        }
        for cr in cond.itertuples()
    }
    return payload


def _report(df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 78)
    log("S-02 · 群的投資時鐘四格條件式績效 驗收摘要")
    log("=" * 78)
    piv = df.pivot_table(index=["tree_id", "cluster_id"], columns="clock_cell",
                         values="avg_ret_median")
    log(piv.round(4).to_string())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.macro_decision_input")
    ap.add_argument("--trees", nargs="+", default=list(DEFAULT_TREES))
    a = ap.parse_args(argv)
    df = run(trees=a.trees)
    _report(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
