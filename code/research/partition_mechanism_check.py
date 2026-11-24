# -*- coding: utf-8 -*-
"""隨機分群為何在美股、跨市場勝過 HRP 分群：機制查證（2026-11-22，論文 4-4；使用者授權，經寫作 session 轉達）。

背景：`walkforward_partition_compare.csv` 的 CAGR 勝出比例，台股 71.8%、美股 14.3%、跨市場 15.1%。
論文的推論：隨機分出的群混合各類策略，各群依 Calmar 挑出的代表，結果接近只依 Calmar 排序挑選（E_top_calmar）。

⚠️ 最直接的驗證是比較「隨機分群名單」和 E 名單的重疊，但 `walkforward_partition` 沒有存名單。
   要重現名單，得對每個窗口重建 HRP 樹（取群大小），上次全跑 6.9 小時，超出授權範圍，故不做。
   本檔改用已存的結果做三項間接查證，只讀現有檔案，不建樹、不呼叫 LLM：

① 績效距離（同一格比較）：R＝隨機分群精選（30 次抽樣平均），A＝HRP 精選，E＝E_top_calmar。
   若推論成立，R 應比 A 更接近 E。
② 名單組成：
   - HRP 與 E 名單的重疊（Jaccard、重疊檔數÷A 的檔數）
   - 跨市場：各組名單中台股策略的比例（HRP 的群大致依市場切開，配額會強迫按市場分配）
   - 台股、美股：A、E 名單在 L1 群的分布。⚠️ 用全期凍結樹的 L1 標籤近似（各窗口的樹不同），
     只限 k_mode=fixed
③ 補位比例：A 和 R 的 n_backfilled÷n_members，排除「多樣性門檻造成的補位差異」這個解釋

輸出（`_analysis_outputs_robustness/`）：
  partition_mechanism_distance.csv、partition_mechanism_composition.csv、
  partition_mechanism_backfill.csv、_partition_mechanism_check_manifest/

用法（cwd=code/）：python -m research.partition_mechanism_check
"""
from __future__ import annotations

import sys

import pandas as pd

from . import freeze, paths

OUT = paths.ROOT / "_analysis_outputs_robustness"
DETAIL = OUT / "walkforward_matrix_detail.csv"
PART = OUT / "walkforward_partition.csv"
MEMBERS = OUT / "walkforward_members.parquet"
KEY = ["tree_key", "scheme", "k_mode", "ratio", "allocation", "window_no"]


def _load():
    d = pd.read_csv(DETAIL)
    d["ratio"] = d.ratio.astype(str)
    r = pd.read_csv(PART)
    r["ratio"] = r.ratio.astype(str)
    return d, r


def distance(d: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    a = d[d.group == "A_hrp"].set_index(KEY).oos_cagr.rename("A")
    e = d[d.group == "E_top_calmar"].set_index(KEY).oos_cagr.rename("E")
    rr = r.set_index(KEY).oos_cagr.rename("R")
    x = pd.concat([a, e, rr], axis=1)
    if x.isna().any().any():
        raise AssertionError("A／E／R 有格子對不上")
    x = x.reset_index()
    x["dRE"] = (x.R - x.E).abs()
    x["dAE"] = (x.A - x.E).abs()
    rows = []
    groups = ([((t, "ALL"), g) for t, g in x.groupby("tree_key")]
              + [((t, rt), g) for (t, rt), g in x.groupby(["tree_key", "ratio"])])
    for (t, ratio), g in groups:
        rows.append({"tree_key": t, "ratio": ratio, "n_cells": len(g),
                     "A_oos_cagr_mean": g.A.mean(), "R_oos_cagr_mean": g.R.mean(),
                     "E_oos_cagr_mean": g.E.mean(),
                     "pct_E_beats_A": (g.E > g.A).mean(), "pct_R_beats_A": (g.R > g.A).mean(),
                     "median_abs_R_minus_E": g.dRE.median(), "median_abs_A_minus_E": g.dAE.median(),
                     "pct_R_closer_to_E_than_A": (g.dRE < g.dAE).mean(),
                     "corr_R_E": g.R.corr(g.E), "corr_A_E": g.A.corr(g.E)})
    return pd.DataFrame(rows)


def composition() -> pd.DataFrame:
    m = pd.read_parquet(MEMBERS)
    rows = []
    # HRP 與 E 名單的重疊
    a = m[m.group == "A_hrp"].set_index(KEY).members
    e = m[m.group == "E_top_calmar"].set_index(KEY).members
    j = pd.concat([a.rename("a"), e.rename("e")], axis=1).dropna().reset_index()
    j["jac"] = [len(set(p) & set(q)) / len(set(p) | set(q)) for p, q in zip(j.a, j.e)]
    j["ovl"] = [len(set(p) & set(q)) / len(set(p)) for p, q in zip(j.a, j.e)]
    for (t, rt), g in [((t, "ALL"), g) for t, g in j.groupby("tree_key")] + \
            [((t, rt), g) for (t, rt), g in j.groupby(["tree_key", "ratio"])]:
        rows.append({"item": "A_vs_E_overlap", "tree_key": t, "ratio": rt, "group": "A_hrp∩E",
                     "cluster_id": None, "share": g.ovl.mean(), "jaccard": g.jac.mean(), "n_cells": len(g)})
    # 跨市場：台股策略比例
    xm = m[m.tree_key == "XM"].copy()
    xm["tw"] = xm.members.map(lambda L: sum(u.startswith("TW::") for u in L) / len(L))
    for (grp, rt), g in [((grp, "ALL"), g) for grp, g in xm.groupby("group")] + \
            [((grp, rt), g) for (grp, rt), g in xm.groupby(["group", "ratio"])]:
        rows.append({"item": "XM_tw_share", "tree_key": "XM", "ratio": rt, "group": grp,
                     "cluster_id": None, "share": g.tw.mean(), "jaccard": None, "n_cells": len(g)})
    ci = pd.read_parquet(paths.STAGE0 / "candidate_index.parquet", columns=["market"])
    rows.append({"item": "XM_tw_share", "tree_key": "XM", "ratio": "ALL", "group": "candidate_pool",
                 "cluster_id": None, "share": float((ci.market == "TW").mean()), "jaccard": None,
                 "n_cells": None})
    # 台股、美股：L1 群分布（全期凍結樹近似，k_mode=fixed）
    ca = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")
    for t in ("TW", "US"):
        lab = ca[ca.tree_id == f"{t}_normal"].set_index("strategy_uid").cluster_L1
        sub = m[(m.tree_key == t) & (m.k_mode == "fixed")]
        dists = {"candidate_pool": lab.value_counts(normalize=True)}
        for grp in ("A_hrp", "E_top_calmar"):
            dists[grp] = sub[sub.group == grp].members.explode().map(lab).value_counts(normalize=True)
        for grp, s in dists.items():
            s = s.reindex(sorted(lab.unique())).fillna(0.0)
            for cid, v in s.items():
                rows.append({"item": "L1_cluster_share", "tree_key": t, "ratio": "ALL", "group": grp,
                             "cluster_id": int(cid), "share": float(v), "jaccard": None, "n_cells": None})
            rows.append({"item": "L1_cluster_HHI", "tree_key": t, "ratio": "ALL", "group": grp,
                         "cluster_id": None, "share": float((s ** 2).sum()), "jaccard": None,
                         "n_cells": None})
    return pd.DataFrame(rows)


def backfill(d: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    a = d[d.group == "A_hrp"].assign(bf=lambda x: x.n_backfilled / x.n_members)
    rr = r.assign(bf=lambda x: x.n_backfilled_mean / x.n_members_mean)
    rows = []
    for name, df in (("A_hrp", a), ("R_random_partition", rr)):
        for (t, rt), g in [((t, "ALL"), g) for t, g in df.groupby("tree_key")] + \
                [((t, rt), g) for (t, rt), g in df.groupby(["tree_key", "ratio"])]:
            rows.append({"group": name, "tree_key": t, "ratio": rt, "backfill_share_mean": g.bf.mean(),
                         "n_cells": len(g)})
    return pd.DataFrame(rows)


def run(log=print):
    freeze.verify_inputs(OUT / "_walkforward_matrix_manifest")
    freeze.verify_inputs(OUT / "_walkforward_partition_manifest")
    d, r = _load()
    dist, comp, bf = distance(d, r), composition(), backfill(d, r)
    ps = [OUT / "partition_mechanism_distance.csv", OUT / "partition_mechanism_composition.csv",
          OUT / "partition_mechanism_backfill.csv"]
    for df, p in zip((dist, comp, bf), ps):
        df.to_csv(p, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "partition_mechanism_check", OUT / "_partition_mechanism_check_manifest",
        inputs=[DETAIL, PART, MEMBERS, paths.STAGE0 / "candidate_index.parquet",
                paths.STAGE3 / "cluster_assign.parquet"],
        outputs=ps,
        params={"note": "L1 群分布以全期凍結樹近似，只用 k_mode=fixed；隨機分群名單未存，未直接比對"},
        notes="論文 4-4：隨機分群勝過 HRP（美股、跨市場）的機制查證。")
    log(dist[dist.ratio == "ALL"].round(4).to_string(index=False))
    log(comp[comp.ratio == "ALL"].round(4).to_string(index=False))
    log(bf[bf.ratio == "ALL"].round(4).to_string(index=False))
    for p in ps:
        log(f"→ {p}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run()
