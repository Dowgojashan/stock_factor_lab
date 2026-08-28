# -*- coding: utf-8 -*-
"""H-06 · 群的定量特徵表（老師 2026-08-26 意見，論文兩大重點之一，見開發待辦追蹤.md）

老師具體要求的角度：
  「某幾年都固定賺錢、某幾年賠錢」
  「在季度或年份，他們的獲利表現是不是不太一樣」

純程式算，**沒有LLM**——這是給論文直接呈現的定量素材，也是之後 H-08（單群LLM解釋）
的輸入。現有 `cluster_story.py` 的 `_cluster_profiles` 是為餵LLM設計的，只有橫斷面
的成分側寫（因子/市場/CAGR/MDD），完全沒有時間維度，故另外做這支。

輸出（三張表，`_frozen/stage3/`）：
  cluster_annual_returns.parquet    群代表序列的逐年複利報酬（回答「哪幾年賺賠」）
  cluster_quarterly_returns.parquet 群代表序列的逐季複利報酬（回答「季度表現是否不同」）
  cluster_profile_quant.parquet     每群一列的彙總表（成分側寫 + 時間型態統計量）

群代表序列＝成員報酬簡單平均（跟 `stage3_hrp._cluster_meta_and_corr` 同一定義，
群間相關矩陣也是拿這條序列算的，口徑一致）。只做 L1（給LLM讀/論文呈現的粗粒度，
L3群數太多不適合逐群描述，跟 cluster_story.py 同樣的理由）。只做 normal 樹——
crisis 樹只有17-26個月、不到完整2年，做不出「哪幾年賺賠」這種分析（且H-14已把
crisis樹的定位限縮在描述性揭露，不需要在這裡重複）。

用法：
    cd code
    python -m research.cluster_temporal_profile
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, paths
from .cluster_story import _cluster_profiles

DEFAULT_TREES = ("TW_normal", "US_normal", "XM_normal")
LEVEL = "L1"


def _tree_key(tree_id: str) -> str:
    return tree_id.rsplit("_", 1)[0]


def _member_wide_returns(tree_id: str, months_long: pd.DataFrame,
                         assign: pd.DataFrame) -> pd.DataFrame:
    """該樹全部成員的月報酬（策略×月），裁到該樹的 DD-03 共同窗。"""
    window_start, window_end = C.HRP_WINDOWS[_tree_key(tree_id)]
    uids = assign.loc[assign.tree_id == tree_id, C.PK]
    w = months_long[months_long.strategy_uid.isin(set(uids))]
    w = w[(w.month >= pd.Period(window_start, "M")) & (w.month <= pd.Period(window_end, "M"))]
    return w.pivot(index="strategy_uid", columns="month", values="ret")


def _annual_quarterly(rep: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """群代表序列（index=月, PeriodIndex[M]）→ 逐年／逐季複利報酬表。"""
    years = rep.index.year
    ann_rows = []
    for y, idx in pd.Series(rep.index, index=rep.index).groupby(years):
        sub = rep.loc[idx]
        ann_rows.append({"year": int(y), "ret": float((1 + sub).prod() - 1),
                         "n_months": int(len(sub))})
    ann = pd.DataFrame(ann_rows).sort_values("year").reset_index(drop=True)

    quarters = rep.index.quarter
    q_rows = []
    for (y, q), idx in pd.Series(rep.index, index=rep.index).groupby([years, quarters]):
        sub = rep.loc[idx]
        q_rows.append({"year": int(y), "quarter": int(q),
                       "ret": float((1 + sub).prod() - 1), "n_months": int(len(sub))})
    qtr = pd.DataFrame(q_rows).sort_values(["year", "quarter"]).reset_index(drop=True)
    return ann, qtr


def _flatten_composition(prof: dict) -> dict:
    """`_cluster_profiles` 給的巢狀 dict → 攤平成表格欄位（H-06只留top1，不像LLM版留top3）。"""
    n = prof["n_members"]
    mkt = prof["market_mix"]

    def top1(d: dict) -> tuple[str | None, float | None]:
        if not d:
            return None, None
        k = max(d, key=d.get)
        return k, round(d[k] / n, 4)

    f_type, f_type_pct = top1(prof["top_factor_types"])
    f1, f1_pct = top1(prof["top_F1"])
    c_src, c_src_pct = top1(prof["top_C_source"])
    v1_count = prof["V_mix"].get("v1", 0)

    return {
        "n_members": n,
        "pct_TW": round(mkt.get("TW", 0) / n, 4),
        "pct_US": round(mkt.get("US", 0) / n, 4),
        "top1_factor_type": f_type, "top1_factor_type_pct": f_type_pct,
        "top1_F1": f1, "top1_F1_pct": f1_pct,
        "top1_C_source": c_src, "top1_C_source_pct": c_src_pct,
        "pct_v1": round(v1_count / n, 4),
        "CAGR_median": prof["CAGR_median"], "MDD_median": prof["MDD_median"],
        "smallcap_share_median": prof["smallcap_share_median"],
        "avg_intra_corr": prof["avg_intra_corr"],
    }


def build(trees=DEFAULT_TREES, log=print) -> dict[str, pd.DataFrame]:
    freeze.verify_inputs(paths.STAGE3)
    freeze.verify_inputs(paths.STAGE4)

    months_long = pd.read_parquet(paths.STAGE1 / "returns_monthly.parquet")
    assign = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")

    annual_rows, quarterly_rows, profile_rows = [], [], []
    for tree_id in trees:
        log(f"[{tree_id}] 計算群代表序列的年/季報酬 …")
        wide = _member_wide_returns(tree_id, months_long, assign)
        a = assign[(assign.tree_id == tree_id)][[C.PK, f"cluster_{LEVEL}"]].rename(
            columns={f"cluster_{LEVEL}": "_cl"})
        comp_profiles = _cluster_profiles(tree_id)

        for cid, g in a.groupby("_cl"):
            members = [u for u in g[C.PK] if u in wide.index]
            if not members:
                continue
            rep = wide.loc[members].mean(axis=0).dropna()
            rep.index = pd.PeriodIndex(rep.index, freq="M")
            ann, qtr = _annual_quarterly(rep)

            for r in ann.itertuples():
                annual_rows.append({"tree_id": tree_id, "level": LEVEL, "cluster_id": int(cid),
                                    "year": r.year, "ret": r.ret, "n_months": r.n_months})
            for r in qtr.itertuples():
                quarterly_rows.append({"tree_id": tree_id, "level": LEVEL, "cluster_id": int(cid),
                                       "year": r.year, "quarter": r.quarter, "ret": r.ret,
                                       "n_months": r.n_months})

            n_pos = int((ann["ret"] > 0).sum())
            best = ann.loc[ann["ret"].idxmax()]
            worst = ann.loc[ann["ret"].idxmin()]
            summary = {
                "tree_id": tree_id, "level": LEVEL, "cluster_id": int(cid),
                **_flatten_composition(comp_profiles[int(cid)]),
                "window_start_year": int(ann["year"].min()),
                "window_end_year": int(ann["year"].max()),
                "n_years": int(len(ann)),
                "n_years_positive": n_pos,
                "pct_years_positive": round(n_pos / len(ann), 4),
                "best_year": int(best["year"]), "best_year_ret": round(float(best["ret"]), 4),
                "worst_year": int(worst["year"]), "worst_year_ret": round(float(worst["ret"]), 4),
                "annual_ret_mean": round(float(ann["ret"].mean()), 4),
                "annual_ret_std": (round(float(ann["ret"].std(ddof=0)), 4)
                                  if len(ann) > 1 else None),
                "quarterly_ret_std": (round(float(qtr["ret"].std(ddof=0)), 4)
                                      if len(qtr) > 1 else None),
            }
            profile_rows.append(summary)
        log(f"  {a['_cl'].nunique()} 群完成")

    annual = pd.DataFrame(annual_rows)
    quarterly = pd.DataFrame(quarterly_rows)
    profile = pd.DataFrame(profile_rows)
    for df in (annual, quarterly, profile):
        df["tree_id"] = df["tree_id"].astype("category")
        df["level"] = df["level"].astype("category")

    C.validate(annual, C.CLUSTER_ANNUAL_RETURNS, strict_columns=True)
    C.validate(quarterly, C.CLUSTER_QUARTERLY_RETURNS, strict_columns=True)
    C.validate(profile, C.CLUSTER_PROFILE_QUANT, strict_columns=True)
    log("✓ 三張表契約皆通過")
    return {"annual": annual, "quarterly": quarterly, "profile": profile}


def run(trees=DEFAULT_TREES, log=print) -> dict[str, pd.DataFrame]:
    tables = build(trees=trees, log=log)
    outs = []
    for key, fname in (("annual", "cluster_annual_returns.parquet"),
                       ("quarterly", "cluster_quarterly_returns.parquet"),
                       ("profile", "cluster_profile_quant.parquet")):
        p = paths.STAGE3 / fname
        tables[key].to_parquet(p, compression="zstd", index=False)
        outs.append(p)
        log(f"→ {fname}  {len(tables[key]):,} 列")

    # ⚠️ 不寫進 stage3 的 MANIFEST：那是 stage3_hrp.py 專屬的凍結鏈（DD-08），
    # 這裡是附加分析、非stage3本體的輸出，跟 cluster_story.py 選獨立側錄同樣理由。
    freeze.write_manifest(
        "cluster_temporal_profile", paths.STAGE3 / "_temporal_profile",
        inputs=[paths.STAGE1 / "returns_monthly.parquet",
               paths.STAGE3 / "cluster_assign.parquet",
               paths.STAGE3 / "cluster_meta.parquet",
               paths.STAGE4 / "strategy_map.parquet"],
        outputs=outs,
        params={"trees": list(trees), "level": LEVEL},
        notes="H-06：群的定量特徵表，純程式算無LLM。群代表序列=成員報酬簡單平均"
              "（同stage3_hrp._cluster_meta_and_corr口徑）。crisis樹不做（樣本不足2年）。",
    )
    return tables


def _report(tables: dict[str, pd.DataFrame], log=print) -> None:
    log("\n" + "=" * 66)
    log("H-06 · 群定量特徵表 驗收摘要")
    log("=" * 66)
    profile = tables["profile"]
    for tid, g in profile.groupby("tree_id", observed=True):
        log(f"\n[{tid}]（{len(g)} 群，{g['window_start_year'].iloc[0]}"
            f"~{g['window_end_year'].iloc[0]}）")
        show = g[["cluster_id", "n_members", "pct_TW", "top1_factor_type",
                  "CAGR_median", "n_years_positive", "n_years", "best_year",
                  "best_year_ret", "worst_year", "worst_year_ret", "annual_ret_std"]]
        log(show.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.cluster_temporal_profile")
    ap.add_argument("--trees", nargs="+", default=list(DEFAULT_TREES))
    a = ap.parse_args(argv)
    tables = run(trees=a.trees)
    _report(tables)
    return 0


if __name__ == "__main__":
    sys.exit(main())
