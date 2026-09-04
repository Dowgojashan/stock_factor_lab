# -*- coding: utf-8 -*-
"""H-26d · 共同窗選擇的穩健性（2026-09-04）

**這是 `落差處理方案_v1.md` 落差3 當初規劃的「方案 E」，程式參數
（`contracts.HRP_ROBUSTNESS_WINDOW_TW`）留了下來但從來沒有執行過**——
2026-09-04 code review 查全域搜尋確認該常數零引用。

原始規劃的原話：
  「主分析用方案 B，另外用 2003-01 窗（台股保留 48.3%）跑一次台股樹，
    比較兩個窗下的**群結構一致性（ARI）**。若一致性高 → 證明群結構不是窗選擇的
    產物；若低 → 這件事本身要寫進 limitations。成本很低，但論文防禦力提升很多。」

---------------------------------------------------------------------------
要回答的問題
---------------------------------------------------------------------------
台股的 HRP 共同窗定在 2007-01，是因為「再往前每多一個月都要犧牲大量策略」
（2006-01 起算會少 522 個、2007-01 只少 1 個）。但選窗畢竟是個判斷，
**若口試被問「換個窗口分出來會不會完全不一樣」，目前答不出來。**

本模組用 2003-01 窗（只剩 3,272 檔，49%）重建一次台股樹，跟主線 2007-01 窗的
官方分群結果比對 ARI（Adjusted Rand Index，1=完全相同、0=隨機）。

⚠️ **只做台股**：美股主線窗已是 2002-01（幾乎最早），再往前的 2000-02 只剩
459 檔（5.5%），樣本太少、比較沒有意義。跨市場窗跟著台股走，同理。

---------------------------------------------------------------------------
比較設計
---------------------------------------------------------------------------
  - 兩棵樹在**共同策略**上比對（2003 窗宇宙是 2007 窗宇宙的子集，共 3,272 檔）
  - **兩邊都切同一個 k**（主線的 L1_TARGET=6）——若各自選 k，ARI 會同時混進
    「窗不同」與「k 不同」兩個因素，無法歸因
  - 另外掃描 k=3..12，確認結論不是只在 k=6 成立
  - **附隨機基準線**：把其中一邊的標籤打散重算 ARI，給讀者一個「0 附近長怎樣」
    的參照，避免只看到一個沒有尺度的數字

用法：
    cd code
    python -m research.window_robustness
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
from .walkforward_matrix import _load_inputs

TREE_KEY = "TW"
MAIN_TREE_ID = "TW_normal"
K_SCAN = range(3, 13)
N_SHUFFLE = 20          # 隨機基準線的重抽次數
SHUFFLE_SEED = 42


def build_alt_tree(log=print) -> dict:
    """用替代窗（2003-01~2025-12）重建台股樹，走跟主線完全相同的程式路徑。"""
    alt_start, alt_end = C.HRP_ROBUSTNESS_WINDOW_TW
    months_long, meta, f_combo_map = _load_inputs()
    uids = S3._tree_universe(TREE_KEY, alt_start, meta)
    wide = S3._pivot_window(months_long, uids, alt_start, alt_end)
    log(f"[替代窗 {alt_start}~{alt_end}] 策略 {len(wide):,} 檔｜{wide.shape[1]} 個月")
    t0 = time.time()
    tree = S3._build_tree(f"{TREE_KEY}_alt_{alt_start}", TREE_KEY, wide, f_combo_map,
                          log=lambda *a, **k: None)
    log(f"  建樹完成 {time.time()-t0:.0f}s｜linkage={tree['method']}")
    return tree


def compare(tree_alt: dict, log=print) -> pd.DataFrame:
    """替代窗 vs 主線窗的群結構一致性（共同策略上的 ARI）。"""
    freeze.verify_inputs(paths.STAGE3)
    main = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")
    main = main[main.tree_id == MAIN_TREE_ID].set_index(C.PK)

    alt_assign = tree_alt["assign"].set_index(C.PK)
    common = main.index.intersection(alt_assign.index)
    log(f"  共同策略 {len(common):,}（主線 {len(main):,}／替代窗 {len(alt_assign):,}）")
    if len(common) < 100:
        raise AssertionError(f"共同策略只有 {len(common)}，無法比較")

    # 主線的 linkage 也要拿來重切，才能在同一個 k 上比
    link_main = np.load(paths.STAGE3 / f"linkage_{MAIN_TREE_ID}.npy")
    wide_main = S3.rebuild_tree_returns(MAIN_TREE_ID, log=lambda *a, **k: None)
    if len(link_main) + 1 != len(wide_main):
        raise AssertionError("主線 linkage 葉節點數與報酬矩陣列數不符")

    rng = np.random.default_rng(SHUFFLE_SEED)
    rows = []
    for k in K_SCAN:
        lab_main = pd.Series(hrp.cut_clusters(link_main, k), index=wide_main.index)
        lab_alt = pd.Series(hrp.cut_clusters(tree_alt["link"], k),
                            index=tree_alt["assign"][C.PK].to_numpy())
        a, b = lab_main.reindex(common), lab_alt.reindex(common)
        ari = hrp.adjusted_rand_index(a, b)
        # 隨機基準線：打散替代窗的標籤，看 ARI 會落在哪
        floor = float(np.mean([
            hrp.adjusted_rand_index(a, pd.Series(rng.permutation(b.to_numpy()),
                                                 index=b.index))
            for _ in range(N_SHUFFLE)]))
        rows.append({"tree_key": TREE_KEY, "k": int(k), "n_common": len(common),
                    "ari": float(ari), "ari_random_floor": floor,
                    "n_main": int(len(main)), "n_alt": int(len(alt_assign)),
                    "is_main_k": bool(k == S3.L1_TARGET[TREE_KEY])})
        log(f"  k={k:>2}  ARI={ari:.4f}  （隨機基準 {floor:+.4f}）"
            f"{'  ← 主線採用的 k' if k == S3.L1_TARGET[TREE_KEY] else ''}")
    return pd.DataFrame(rows)


def run(log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE0)
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE1 / "_marks")

    tree_alt = build_alt_tree(log)
    df = compare(tree_alt, log)
    df["tree_key"] = df["tree_key"].astype("category")
    C.validate(df, C.WINDOW_ROBUSTNESS, strict_columns=True)
    log(f"✓ window_robustness 契約通過（{len(df)} 個 k）")

    out_dir = paths.ROOT / "_analysis_outputs_robustness"
    p = out_dir / "window_robustness.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "window_robustness", out_dir / "_window_robustness_manifest",
        inputs=[paths.STAGE1 / "returns_monthly.parquet",
               paths.STAGE1 / "returns_meta.parquet",
               paths.STAGE1 / "strategy_marks.parquet",
               paths.STAGE3 / "cluster_assign.parquet"],
        outputs=[p],
        params={"main_window": C.HRP_WINDOWS[TREE_KEY],
               "alt_window": C.HRP_ROBUSTNESS_WINDOW_TW,
               "k_scan": [int(k) for k in K_SCAN],
               "n_shuffle": N_SHUFFLE, "shuffle_seed": SHUFFLE_SEED},
        notes="H-26d（＝落差3 的方案E，2026-09-04 首次執行）：台股共同窗 2007-01 "
              "vs 2003-01 的群結構一致性，用來回答「群結構是不是窗選擇的產物」。",
    )
    log(f"→ {p}")
    return df


def _report(df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 74)
    log("H-26d · 共同窗選擇的穩健性（台股 2007-01 vs 2003-01）")
    log("=" * 74)
    m = df[df.is_main_k].iloc[0]
    log(f"共同策略 {m.n_common:,}（主線 {m.n_main:,}／替代窗 {m.n_alt:,}）")
    log(f"主線採用的 k={int(m.k)}：**ARI = {m.ari:.4f}**"
        f"（隨機基準 {m.ari_random_floor:+.4f}）")
    log(f"k=3~12 的 ARI 範圍：{df.ari.min():.4f} ~ {df.ari.max():.4f}"
        f"，平均 {df.ari.mean():.4f}")
    log("")
    if m.ari > 0.5:
        log("→ 一致性高：群結構**不是**窗口選擇的產物，可寫進論文當防禦")
    elif m.ari > 0.2:
        log("→ 一致性中等：結構有共通性但也有實質差異，須在 limitations 誠實說明")
    else:
        log("→ ⚠️ 一致性低：換窗會得到相當不同的分群，**必須**寫進 limitations")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.window_robustness")
    ap.parse_args(argv)
    df = run()
    _report(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
