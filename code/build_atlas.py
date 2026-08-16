# -*- coding: utf-8 -*-
"""
把 SOP 流程（Phase 3 = v0、Phase 4 = v1）的結果餵給 `analyze_batch.py`，
產出學姊論文第四章的 18 張圖圖鑑。

## 為什麼需要這一層

`analyze_batch.py` 是為「舊的批次實驗」寫的，它假設**一個 label 裡同時有 v0 與 v1**：

    v0 = df[df["V"] == "v0"];  v1 = df[df["V"] == "v1"]
    common = v0.index.intersection(v1.index)          # ← 兩邊都要有才配得起來

但 SOP 流程**故意把 v0/v1 拆成兩個 label**（Phase 3 只跑 v0、Phase 4 只補 v1），
為的是省掉一半的重複回測。直接跑會讓 4-20 ~ 4-24（V 相關的 5 張圖）全部是空的。

## 這支做什麼（不修改 analyze_batch.py 一個字）

1. 把 `{市場}_L3{變體}_M` 與 `{市場}_L4{變體}_M` 的 `stats.parquet` 合併，
   寫進一個新的輕量 label 目錄（**只有 stats.parquet，沒有逐策略子目錄**）
2. **攔截 `analyze_batch.load_strategy_artifacts`**，把逐策略的路徑依 `__v0`/`__v1`
   後綴轉回原本的 L3 / L4 目錄——避免建立上萬個 junction 或複製上百 GB
3. 借 `analyze_batch.main()` 跑完整套圖，再把成品搬到 `_analysis_outputs_atlas/`

用法（cwd=code/）：
  python build_atlas.py --market TW --variant openSec
  python build_atlas.py --market US --variant openSec --skip-credibility
"""
import sys
import shutil
import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import analyze_batch as ab            # noqa: E402

ART = HERE / "results_artifacts"
ATLAS_OUT = ROOT / "_analysis_outputs_atlas"


def phase_label(market, variant, phase):
    """Phase 3/4 的 label 命名規則（與 phase3_conditions / phase4_valuation 一致）。"""
    return f"{market}_L{phase}_M" if variant == "strict" else f"{market}_L{phase}_{variant}_M"


def build_merged_stats(market, variant, merged_label):
    """合併 L3(v0) + L4(v1) 的 stats，寫進 merged_label 目錄。"""
    l3, l4 = phase_label(market, variant, 3), phase_label(market, variant, 4)
    for lab in (l3, l4):
        p = ART / lab / "stats.parquet"
        if not p.exists():
            raise FileNotFoundError(f"找不到 {p}——請先跑完 Phase 3/4（{market} / {variant}）")
    a = pd.read_parquet(ART / l3 / "stats.parquet")
    b = pd.read_parquet(ART / l4 / "stats.parquet")

    # 防呆：Phase 3 應該全是 v0、Phase 4 全是 v1，混到就代表 label 給錯了
    bad3 = (~a["strategy"].str.endswith("__v0")).sum()
    bad4 = (~b["strategy"].str.endswith("__v1")).sum()
    if bad3 or bad4:
        raise AssertionError(f"[{market}/{variant}] L3 有 {bad3} 個非 v0、L4 有 {bad4} 個非 v1，"
                             f"label 可能給錯")

    m = pd.concat([a, b], ignore_index=True)
    out = ART / merged_label
    out.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out / "stats.parquet")
    print(f">> 合併 {l3}（{len(a)} 個 v0）+ {l4}（{len(b)} 個 v1）= {len(m)} 個策略")
    return l3, l4, m


def patch_artifact_resolver(l3, l4, merged_label):
    """把逐策略路徑依 v0/v1 導回原目錄。

    analyze_batch 內部一律用 `HERE/results_artifacts/{label}/{strategy}` 組路徑，
    這裡只認 strategy 名字的後綴，其餘完全不動它的邏輯。
    """
    orig = ab.load_strategy_artifacts

    def resolved(strat_dir):
        strat_dir = Path(strat_dir)
        if strat_dir.parent.name == merged_label:          # 只攔我們造出來的假 label
            real = l4 if strat_dir.name.endswith("__v1") else l3
            strat_dir = ART / real / strat_dir.name
        return orig(strat_dir)

    ab.load_strategy_artifacts = resolved


def main():
    ap = argparse.ArgumentParser(description="產出第四章 18 張圖圖鑑（SOP 版）")
    ap.add_argument("--market", default="TW", choices=["TW", "US"])
    ap.add_argument("--variant", default="openSec")
    ap.add_argument("--skip-credibility", action="store_true",
                    help="跳過逐策略可信度掃描（4-5~4-10），可省下約 10-15 分鐘")
    ap.add_argument("--keep-temp", action="store_true", help="保留暫時的合併 label 目錄")
    args = ap.parse_args()

    merged_label = f"{args.market}_ATLAS_{args.variant}_M"
    print(f"=== 第四章圖鑑｜{args.market}／{args.variant} ===")

    l3, l4, _ = build_merged_stats(args.market, args.variant, merged_label)
    patch_artifact_resolver(l3, l4, merged_label)

    # 借 analyze_batch.main()：它從 argv 讀參數，這樣行為與原本完全一致
    argv = ["analyze_batch.py", "--label", merged_label]
    if args.skip_credibility:
        argv.append("--skip-credibility")
    old_argv, sys.argv = sys.argv, argv
    try:
        ab.main()
    finally:
        sys.argv = old_argv

    # 成品搬到專屬目錄（analyze_batch 預設寫到 gitignore 的 _analysis_outputs/）
    src = ROOT / "_analysis_outputs" / merged_label
    dst = ATLAS_OUT / f"{args.market}_{args.variant}"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        n = len(list((dst / "figures").glob("*.png"))) if (dst / "figures").exists() else 0
        print(f"\n>> 圖鑑已存至 {dst}（{n} 張圖）")

    if not args.keep_temp:
        shutil.rmtree(ART / merged_label, ignore_errors=True)
        print(f">> 已清除暫時目錄 {merged_label}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
