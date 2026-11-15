# -*- coding: utf-8 -*-
"""C1 主因子門檻敏感度，用現行候選池重做（2026-11-15，使用者授權，經寫作 session 轉達）。

舊版（`code/robustness_checks.py` 的 C1，2026-09-13）用的是 openSec 變體的舊池，
本檔改讀現行輸入（`paths.VARIANT`／`paths.PHASE4_DIR_BY_MARKET`）：
  台股 openSec_boost：`_analysis_outputs_phase2_twfix`、`_analysis_outputs_phase4_twfix`（11-09）
  美股 openSec_balanced：`_analysis_outputs_phase2`、`_analysis_outputs_phase4`（10-07～08）

做法（與舊版相同）：只往嚴格方向調門檻（2%／2.5%／3%／4%／5%）。
  提高門檻只會把主因子名單縮成現行名單的子集，所以子集內的策略都已有回測，
  直接從現行最終候選池挑出「主因子桶仍過門檻」的策略即可，不用重跑回測。
  放寬（例如 1%）會產生沒進過第三、四階段的組合，必須補回測，本檔不做。

台股有人工強制納入的主因子桶（`phase2_analyze.FORCE_PRIMARY_OVERRIDE`），分兩種處理：
  強制保留：強制桶不受門檻影響（＝現行規則）
  強制也受門檻：強制桶與其他桶一樣要過門檻

另算（不建樹）：
  - 落在現行平時樹宇宙內的檔數（`stage3_hrp.rebuild_tree_returns`）
  - 子集的策略層 ENB（`enb_null.fast_enb`，與 H-09 同一算法；2% 強制保留必須等於
    `effective_number_of_bets.csv` 的 enb_raw）
  - 子集涵蓋現行 L1 群的數目、最大群佔比（群標籤沿用現行整棵樹，不是重新分群）
不建樹、不跑 walk-forward、不呼叫 LLM；不覆寫舊版 `{TW,US}_primary_margin_sensitivity.csv`。
輸出：`_analysis_outputs_robustness/primary_margin_sensitivity_current.csv`
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import contracts as C
from . import enb_null as EN
from . import freeze, paths
from . import stage3_hrp as S3

sys.path.insert(0, str(paths.CODE_DIR))
from phase2_analyze import FORCE_PRIMARY_OVERRIDE  # noqa: E402

OUT = paths.ROOT / "_analysis_outputs_robustness"
MARGINS = (0.02, 0.025, 0.03, 0.04, 0.05)
P2DIR = {"TW": paths.ROOT / "_analysis_outputs_phase2_twfix", "US": paths.ROOT / "_analysis_outputs_phase2"}
TREATMENTS = ("強制保留", "強制也受門檻")


def _inputs(m: str):
    v = paths.VARIANT[m]
    solo = pd.read_csv(P2DIR[m] / f"{m}_L2_{v}_solo_buckets.csv", encoding="utf-8-sig")
    comb = pd.read_csv(P2DIR[m] / f"{m}_L2_{v}_all_combos.csv", encoding="utf-8-sig")
    fin = pd.read_csv(paths.PHASE4_DIR_BY_MARKET[m] / f"{m}_L4_{v}_final_candidates.csv", encoding="utf-8-sig")
    return v, solo, comb, fin


def _primary_map(comb: pd.DataFrame) -> pd.Series:
    """F組合 → 主因子桶。策略名稱裡兩個因子的順序不是「主因子在前」，
    必須用第二階段晉升表的 primary 欄對應（與舊版 C1 相同）。"""
    p = comb[comb["晉升"]].copy()
    p["fc"] = p.strategy.str.split("__").str[:2].str.join("__")
    assert p.groupby("fc").primary.nunique().max() == 1, "有組合對應到兩個主因子"
    s = p.drop_duplicates("fc").set_index("fc").primary
    return s.map(lambda x: (x.rsplit("_qb", 1)[0], int(x.rsplit("_qb", 1)[1])))


def subsets(m: str) -> tuple[list[dict], dict]:
    v, solo, comb, fin = _inputs(m)
    force = FORCE_PRIMARY_OVERRIDE.get(v, set())
    cur = solo[solo["可當primary"]]
    cur_set = set(zip(cur.F1, cur.k1))
    pm = _primary_map(comb)
    assert fin["F組合"].isin(pm.index).all(), f"{m} 現行池有組合不在晉升表"
    pb = fin["F組合"].map(pm)
    # 對帳：現行池每一檔的主因子桶都必須在現行主因子名單裡
    assert pb.map(lambda t: t in cur_set).all(), f"{m} 現行池有主因子桶不在現行名單"
    forced_hit = {t for t in force if t in cur_set and float(cur.set_index(["F1", "k1"]).loc[t, "贏大盤"]) < 0.02}
    rows, keep = [], {}
    for tr in TREATMENTS if forced_hit else TREATMENTS[:1]:
        for mg in MARGINS:
            ok = {t for t, e in zip(zip(cur.F1, cur.k1), cur["贏大盤"]) if e >= mg}
            if tr == "強制保留":
                ok |= forced_hit
            sel = pb.map(lambda t, _ok=ok: t in _ok)
            sub = fin[sel]
            keep[(tr, mg)] = set(sub["strategy"])
            rows.append({"市場": m, "變體": v, "強制桶處理": tr, "主因子門檻": mg,
                         "主因子桶數": len(ok), "候選池": int(sel.sum()), "佔現行池": float(sel.mean()),
                         "CAGR中位": float(sub.CAGR.median()), "CAGR_p90": float(sub.CAGR.quantile(.9)),
                         "CAGR最高": float(sub.CAGR.max()), "MDD中位": float(sub.max_drawdown.median()),
                         "強制桶": ";".join(f"{f}_qb{k}" for f, k in sorted(forced_hit)) or "無"})
    base = keep[("強制保留", 0.02)]
    assert len(base) == len(fin), f"{m} 2% 強制保留應等於現行池 {len(fin)}，得到 {len(base)}"
    return rows, keep


def main(argv=None) -> int:
    for d in (paths.STAGE0, paths.STAGE1, paths.STAGE1 / "_marks", paths.STAGE3):
        freeze.verify_inputs(d)
    idx = pd.read_parquet(paths.STAGE0 / "candidate_index.parquet")
    assign = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")
    ref = pd.read_csv(OUT / "effective_number_of_bets.csv").set_index("tree_id")

    rows, keep = [], {}
    for m in ("TW", "US"):
        r, k = subsets(m)
        rows += r
        # 候選池的 strategy 名稱 → strategy_uid
        im = idx[idx.market == m]
        name_col = next(c for c in ("strategy", "strategy_name", "name") if c in im.columns)
        n2u = im.set_index(name_col)[C.PK]
        assert len(n2u) == len(im) and n2u.index.is_unique
        keep[m] = {key: set(n2u.reindex(list(s)).dropna()) for key, s in k.items()}
        assert all(len(u) == len(s) for u, s in zip(keep[m].values(), k.values())), f"{m} 名稱對不到 uid"
    df = pd.DataFrame(rows)

    extra = []
    for tree in ("TW_normal", "US_normal", "XM_normal"):
        tk = tree.split("_")[0]
        wide = S3.rebuild_tree_returns(tree, log=lambda *a, **k: None)
        lab = assign[assign.tree_id == tree].set_index(C.PK)["cluster_L1"]
        mk = ("TW", "US") if tk == "XM" else (tk,)
        keys = sorted({key for m in mk for key in keep[m]})
        for tr, mg in keys:
            uids = set().union(*(keep[m].get((tr, mg), keep[m][("強制保留", mg)]) for m in mk))
            w = wide[wide.index.isin(uids)]
            z = EN._standardise(w.to_numpy(dtype=np.float64))
            cl = lab.reindex(w.index)
            share = cl.value_counts(normalize=True)
            extra.append({"樹": tk, "強制桶處理": tr, "主因子門檻": mg, "樹宇宙內檔數": len(w),
                          "樹宇宙佔現行": len(w) / len(wide), "策略層ENB": EN.fast_enb(z),
                          "現行ENB": float(ref.loc[tree, "enb_raw"]),
                          "涵蓋L1群數": int(cl.nunique()), "現行L1群數": int(lab.nunique()),
                          "最大群佔比": float(share.max())})
            print(f"{tk} {tr} {mg:.1%}：{len(w):,} 檔｜ENB {extra[-1]['策略層ENB']:.4f}", flush=True)
    ex = pd.DataFrame(extra)
    b = ex[(ex.強制桶處理 == "強制保留") & (ex.主因子門檻 == 0.02)]
    assert np.allclose(b.策略層ENB, b.現行ENB, atol=1e-3), b[["樹", "策略層ENB", "現行ENB"]]
    assert (b.樹宇宙佔現行 == 1).all()
    print("✓ 2% 強制保留＝現行池、現行 ENB")

    p1 = OUT / "primary_margin_sensitivity_current.csv"
    p2 = OUT / "primary_margin_sensitivity_current_enb.csv"
    df.to_csv(p1, index=False, encoding="utf-8-sig")
    ex.to_csv(p2, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "primary_margin_current", OUT / "_primary_margin_current_manifest",
        inputs=[P2DIR[m] / f"{m}_L2_{paths.VARIANT[m]}_solo_buckets.csv" for m in ("TW", "US")]
        + [paths.PHASE4_DIR_BY_MARKET[m] / f"{m}_L4_{paths.VARIANT[m]}_final_candidates.csv" for m in ("TW", "US")]
        + [paths.STAGE0 / "candidate_index.parquet", paths.STAGE1 / "returns_monthly.parquet",
           paths.STAGE3 / "cluster_assign.parquet", OUT / "effective_number_of_bets.csv"],
        outputs=[p1, p2],
        params={"margins": list(MARGINS), "treatments": list(TREATMENTS)},
        notes="C1 用現行池重做；只往嚴格方向；不建樹，群標籤沿用現行整棵樹。")
    pd.set_option("display.width", 220)
    print(df.drop(columns=["變體"]).to_string(index=False))
    print(ex.to_string(index=False))
    print(p1, p2, sep="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
