# -*- coding: utf-8 -*-
"""群數選擇的退化門檻敏感度（2026-11-15，使用者授權，經寫作 session 轉達；論文附錄 C 參數表的「檢驗」欄）。

現行 `cluster_count_selection.recommend_k(max_share_cap=0.5)`：最大群佔比 > 50% 的 k 視為退化解排除。
(a) 用現有 `cluster_count_silhouette_scan.csv`（k=3..20）改成 0.4／0.6 重選，看三棵平時樹的 k 是否改變；
    0.5 必須選出現行 `stage3_hrp.L1_TARGET`，否則中止。
(b) 把掃描延伸到 k=3..30：沿用凍結 linkage，輪廓係數用分塊計算（距離矩陣不整個放進記憶體，
    XM 整個要約 20GB）。分塊只改變計算順序，k=3..20 必須與現有掃描表一致（容差 1e-8），否則中止。
各 IS 窗：`k_stability.csv` 沒有存各窗的完整掃描（只存選中的 k 與現行 k 的輪廓係數），依指示不做。
不建樹、不呼叫 LLM。
輸出：`_analysis_outputs_robustness/k_cap_sensitivity.csv`（各樹 × 門檻 × 掃描範圍的選擇）、
      `cluster_count_silhouette_scan_k30.csv`（延伸掃描，新檔名，不覆寫原掃描表）
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from . import cluster_count_selection as CCS
from . import enb_null as EN
from . import freeze, hrp, paths
from . import stage3_hrp as S3

OUT = paths.ROOT / "_analysis_outputs_robustness"
CAPS = (0.4, 0.5, 0.6)
K_MAX = 30
TREES = ("TW_normal", "US_normal", "XM_normal")
BLOCK = 1000


def chunked_scan(tree_id: str, ks: range) -> pd.DataFrame:
    wide = S3.rebuild_tree_returns(tree_id, log=lambda *a, **k: None)
    link = np.load(paths.STAGE3 / f"linkage_{tree_id}.npy")
    x = wide.to_numpy(dtype=np.float64)
    z = EN._standardise(x)
    assert z.shape[0] == x.shape[0], "重建的報酬矩陣仍含零變異數策略"
    n, t = z.shape
    labs = {k: hrp.cut_clusters(link, k) for k in ks}
    uniq = {k: np.unique(v) for k, v in labs.items()}
    onehot = np.concatenate([(labs[k][:, None] == uniq[k][None, :]).astype(np.float64) for k in ks], axis=1)
    offs = np.cumsum([0] + [len(uniq[k]) for k in ks])
    sums = np.empty_like(onehot)
    t0 = time.time()
    for s in range(0, n, BLOCK):
        e = min(s + BLOCK, n)
        corr = (z[s:e] @ z.T) / t
        d = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
        d[np.arange(e - s), np.arange(s, e)] = 0.0
        sums[s:e] = d @ onehot
    rows = []
    for j, k in enumerate(ks):
        lab, u = labs[k], uniq[k]
        stc = sums[:, offs[j]:offs[j + 1]]
        sizes = np.bincount(np.searchsorted(u, lab), minlength=len(u)).astype(float)
        own = np.searchsorted(u, lab)
        own_size = sizes[own]
        a = np.where(own_size > 1, stc[np.arange(n), own] / np.maximum(own_size - 1, 1), 0.0)
        m = stc / sizes[None, :]
        m[np.arange(n), own] = np.inf
        b = m.min(axis=1)
        sil = float(np.where(own_size > 1, (b - a) / np.maximum(a, b), 0.0).mean()) if len(u) >= 2 else np.nan
        rows.append({"tree_id": tree_id, "k": k, "silhouette": sil, "n_clusters_actual": len(u),
                     "max_share": float(sizes.max() / n), "n_singleton": int((sizes == 1).sum())})
    print(f"[{tree_id}] 分塊掃描 k={ks.start}..{ks.stop-1}｜N={n:,}｜{time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE1 / "_marks")
    freeze.verify_inputs(paths.STAGE3)
    base = pd.read_csv(OUT / "cluster_count_silhouette_scan.csv")
    ext = pd.concat([chunked_scan(tid, range(3, K_MAX + 1)) for tid in TREES], ignore_index=True)
    chk = ext.merge(base, on=["tree_id", "k"], suffixes=("", "_base"))
    assert len(chk) == len(base)
    gap = (chk.silhouette - chk.silhouette_base).abs().max()
    assert gap < 1e-8 and (chk.max_share - chk.max_share_base).abs().max() < 1e-12 \
        and (chk.n_clusters_actual == chk.n_clusters_actual_base).all(), f"分塊掃描與現有掃描表不一致（最大差 {gap:.2e}）"
    print(f"✓ 分塊掃描 k=3..20 與現有掃描表一致（輪廓係數最大差 {gap:.1e}）")

    rows = []
    for tid in TREES:
        tk = tid.split("_")[0]
        for rng_lab, sc in (("3..20（現有掃描表）", base[base.tree_id == tid]), (f"3..{K_MAX}（延伸）", ext[ext.tree_id == tid])):
            sc = sc.reset_index(drop=True)
            top = sc.loc[sc.silhouette.idxmax()]
            for cap in CAPS:
                r = CCS.recommend_k(sc, cap, log=lambda *a, **k: None)
                sel = sc[sc.k == r["k"]].iloc[0]
                rows.append({"tree_id": tid, "掃描範圍": rng_lab, "max_share_cap": cap, "選出的k": r["k"],
                             "輪廓係數": r["silhouette"], "最大群佔比": float(sel.max_share),
                             "現行L1_TARGET": S3.L1_TARGET[tk], "與現行相同": r["k"] == S3.L1_TARGET[tk],
                             "全域最高k": int(top.k), "全域最高輪廓係數": float(top.silhouette),
                             "全域最高的最大群佔比": float(top.max_share), "退化退讓": r["degenerate_fallback"]})
                if cap == 0.5 and rng_lab.startswith("3..20"):
                    assert r["k"] == S3.L1_TARGET[tk], (tid, r["k"], S3.L1_TARGET[tk])
    print("✓ 0.5 門檻在 k=3..20 選出現行 L1_TARGET")
    res = pd.DataFrame(rows)
    p1, p2 = OUT / "k_cap_sensitivity.csv", OUT / "cluster_count_silhouette_scan_k30.csv"
    res.to_csv(p1, index=False, encoding="utf-8-sig")
    ext.to_csv(p2, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "k_cap_sensitivity", OUT / "_k_cap_sensitivity_manifest",
        inputs=[OUT / "cluster_count_silhouette_scan.csv", paths.STAGE1 / "returns_monthly.parquet",
                paths.STAGE1 / "strategy_marks.parquet"] + [paths.STAGE3 / f"linkage_{t}.npy" for t in TREES],
        outputs=[p1, p2], params={"caps": list(CAPS), "k_max_ext": K_MAX, "block": BLOCK},
        notes="退化門檻敏感度；k=3..20 用現有掃描表，3..30 用凍結 linkage 分塊重算（3..20 已與原表對帳）。"
              "各 IS 窗未做：k_stability.csv 沒有存完整掃描。")
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(index=False))
    print(ext[ext.k > 20].round(4).to_string(index=False))
    print("→", p1, p2, sep="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
