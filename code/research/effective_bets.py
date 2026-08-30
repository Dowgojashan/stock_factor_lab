# -*- coding: utf-8 -*-
"""H-09 · 有效獨立賭注數（Effective Number of Bets，開發待辦追蹤.md 第四階段）

老師原話：「HRP到底幫你少了多少東西」——量化「N個策略實際上等於幾個獨立賭注」，
也是本專案「免費午餐」大小的**理論上限**：不管怎麼挑選子集，能拿到的分散效果
都不可能超過整個宇宙的有效獨立賭注數（ENB）。方法見 `hrp.effective_number_of_bets()`
（Meucci 2009 / López de Prado 2016 的PCA熵版定義）。

三個口徑一起看，才能回答老師的問題：
  enb_raw      = ENB(全部N個策略的完整相關矩陣)   —— 這個宇宙裡「真正」有幾個獨立訊號
  n_clusters   = HRP分出的L1群數（k）             —— 目前實際分出來的群數
  enb_clusters = ENB(k個群代表序列的k×k相關矩陣)  —— 群代表彼此還剩多少獨立性
                 （沿用stage3既有的 cluster_corr_matrix_{tree_id}.parquet，不必重算）

由此導出兩個對照指標：
  redundancy_ratio   = N / enb_raw     —— headline數字：平均每個「真正獨立」的訊號
                        背後，藏了幾個彼此高度重複的策略。這是回答老師問題最直接的數字。
  k_vs_enb_raw        = k / enb_raw     —— HRP現有的群數，佔了理論上限的幾成。
                        遠小於1代表理論上還能再分出更多群；接近1代表k已經逼近上限。
  cluster_independence = enb_clusters / k —— 現有k個群代表彼此還有沒有殘餘冗餘。
                        接近1代表群與群幾乎正交（HRP分群把可分散的結構抓乾淨了）；
                        遠小於1代表雖然分了k群，但群跟群之間還是高度重疊。

只做 normal 樹（TW/US/XM）——crisis 樹樣本量太小、H-14已限定描述性用途，
不需要在這裡重複算 ENB（少量月份下corr的估計本來就不穩，算出來的ENB沒有解讀意義）。

⚠️ enb_raw 需要重建完整N×N相關矩陣的特徵分解——這步 stage3_hrp.py 建樹時本來就
做過一次（PSD檢查用的 `np.linalg.eigvalsh`），這裡是同一計算量級的重算，不是新
量級：XM樹 N≈15,800 在同一套管線下已驗證可行（stage3_hrp.py 實測六棵樹合計548秒，
含這一步）。

用法：
    cd code
    python -m research.effective_bets
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, hrp, paths
from . import stage3_hrp as S3

DEFAULT_TREES = ("TW_normal", "US_normal", "XM_normal")


def _tree_corr(tree_id: str, log=print) -> tuple[np.ndarray, pd.Index]:
    """重建某棵樹的完整N×N相關矩陣。跟 cluster_count_selection._rebuild_dist_matrix
    走同一段資料準備（同一份usable_pool、同一段共同窗），只是這裡要的是corr本身
    而非距離矩陣，corr在建樹當下應與 stage3_hrp.py 逐位元一致。
    """
    tree_key, kind = tree_id.rsplit("_", 1)
    months_long = pd.read_parquet(paths.STAGE1 / "returns_monthly.parquet")
    meta = pd.read_parquet(paths.STAGE1 / "returns_meta.parquet")
    marks = pd.read_parquet(paths.STAGE1 / "strategy_marks.parquet")
    usable = set(marks.loc[marks.is_usable, C.PK])
    meta = meta[meta.strategy_uid.isin(usable)]

    window_start, window_end = C.HRP_WINDOWS[tree_key]
    uids = S3._tree_universe(tree_key, window_start, meta)
    if kind == "normal":
        wide = S3._pivot_window(months_long, uids, window_start, window_end)
    else:
        crisis_months = S3._load_crisis_months(tree_key)
        wide = S3._pivot_months(months_long, uids, crisis_months)

    std0 = wide.std(axis=1) == 0
    if std0.any():
        log(f"  ⚠️ 排除 {int(std0.sum())} 個零變異數策略（同stage3_hrp.py的排除規則）")
        wide = wide.loc[~std0]

    t0 = time.time()
    corr = np.corrcoef(wide.to_numpy(dtype=np.float64))
    log(f"[{tree_id}] 重建相關矩陣 {corr.shape}｜{time.time()-t0:.0f}s")
    return corr, wide.index


def compute(trees=DEFAULT_TREES, log=print) -> pd.DataFrame:
    rows = []
    for tree_id in trees:
        tree_key = tree_id.rsplit("_", 1)[0]
        corr, uids = _tree_corr(tree_id, log)
        n = len(uids)

        t0 = time.time()
        enb_raw = hrp.effective_number_of_bets(corr)
        log(f"[{tree_id}] ENB(全部{n:,}個策略) = {enb_raw:.2f}｜{time.time()-t0:.0f}s")

        gcorr = pd.read_parquet(paths.STAGE3 / f"cluster_corr_matrix_{tree_id}.parquet")
        gcorr.columns = gcorr.columns.astype(int)
        gcorr.index = gcorr.index.astype(int)
        k = len(gcorr)
        enb_clusters = hrp.effective_number_of_bets(gcorr.to_numpy())
        log(f"[{tree_id}] ENB(L1群代表, k={k}) = {enb_clusters:.2f}")

        rows.append({
            "tree_id": tree_id, "tree_key": tree_key,
            "n_strategies": n, "enb_raw": round(enb_raw, 4),
            "n_clusters_l1": k, "enb_clusters": round(enb_clusters, 4),
            "redundancy_ratio": round(n / enb_raw, 2) if enb_raw > 0 else None,
            "k_vs_enb_raw": round(k / enb_raw, 4) if enb_raw > 0 else None,
            "cluster_independence": round(enb_clusters / k, 4) if k > 0 else None,
        })
    return pd.DataFrame(rows)


OUT_DIR = paths.ROOT / "_analysis_outputs_robustness"


def run(trees=DEFAULT_TREES, log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE3)
    out = compute(trees=trees, log=log)
    for col in ("tree_id", "tree_key"):
        out[col] = out[col].astype("category")
    C.validate(out, C.EFFECTIVE_BETS, strict_columns=True)
    log("✓ effective_bets 契約通過")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "effective_number_of_bets.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    # ⚠️ 不寫進 stage3 的 MANIFEST：同 cluster_temporal_profile.py 的理由——
    # 這是附加分析、非stage3本體，不佔用 DD-08 凍結鏈的雜湊驗證範圍。
    freeze.write_manifest(
        "effective_bets", OUT_DIR / "_effective_bets_manifest",
        inputs=[paths.STAGE1 / "returns_monthly.parquet",
               paths.STAGE1 / "returns_meta.parquet",
               paths.STAGE1 / "strategy_marks.parquet"]
              + [paths.STAGE3 / f"cluster_corr_matrix_{t}.parquet" for t in trees],
        outputs=[p],
        params={"trees": list(trees), "method": "PCA熵版ENB（Meucci 2009/López de Prado 2016）"},
        notes="H-09：有效獨立賭注數，只做normal樹（crisis樣本量太小、H-14已限定描述性用途）",
    )
    log(f"→ {p}")
    return out


def _report(df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 78)
    log("H-09 · 有效獨立賭注數 驗收摘要")
    log("=" * 78)
    for r in df.itertuples():
        log(f"[{r.tree_id}] N={r.n_strategies:,} 策略 → ENB={r.enb_raw:.1f}"
            f"（平均每個獨立訊號背後藏了 {r.redundancy_ratio:.1f} 個高度重複的策略）")
        log(f"           HRP分出 k={r.n_clusters_l1} 群，佔理論上限 {r.k_vs_enb_raw:.1%}"
            f"｜k個群代表彼此的獨立性 {r.cluster_independence:.1%}"
            f"（ENB(群代表)={r.enb_clusters:.2f} / k={r.n_clusters_l1}）")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.effective_bets")
    ap.add_argument("--trees", nargs="+", default=list(DEFAULT_TREES))
    a = ap.parse_args(argv)
    df = run(trees=a.trees)
    _report(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
