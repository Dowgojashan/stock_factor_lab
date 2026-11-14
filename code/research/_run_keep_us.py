# -*- coding: utf-8 -*-
"""只重算台股／跨市場、美股沿用上一輪結果（2026-11-11 使用者同意）。

背景：2026-11-09 台股 Phase 2 白名單更新後要重跑研究部主鏈，但美股的價格、候選池、樹都沒有變，
上一輪（2026-11-01～11-08）已驗證美股結果逐位不變。`k_stability`／`walkforward_matrix`／`walkforward_partition`
預設三棵樹一起跑、輸出整檔覆寫，只跑 TW／XM 會把美股的列清掉；美股部分合計要 20 小時以上。

做法（不改原始模組）：照原本的 `main()` 跑全部三棵樹，但把「逐窗計算」的函式換成——
  台股／跨市場：照常計算；美股：直接回傳上一輪結果檔裡同一窗的列（不建樹、不計算）。
因此彙總、契約檢查、輸出與 manifest 全部由模組自己照原流程產生，不需要事後改檔或重寫 manifest。

美股的來源：`_analysis_outputs_robustness/_舊版備份_20261109_白名單更新前/`（2026-11-04～11-08 的結果）。
驗證：先用 `--trees US` 跑一次，輸出的美股列必須與來源逐列相同（`verify` 子指令）。

用法（cwd＝code/）：
    python -m research._run_keep_us k_stability [--trees US]
    python -m research._run_keep_us walkforward_matrix [--trees US]
    python -m research._run_keep_us walkforward_partition [--trees US]
    python -m research._run_keep_us effective_bets [--trees US_normal]     # 2026-11-11 加
    python -m research._run_keep_us turnover_cost [--trees US_normal]      # 2026-11-11 加（turnover_cost.run 新增 reuse 參數）
    python -m research._run_keep_us verify <模組名>
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import pandas as pd  # noqa: E402

from research import paths  # noqa: E402

OUT = paths.ROOT / "_analysis_outputs_robustness"
SRC = OUT / "_舊版備份_20261109_白名單更新前"
KEEP = "US"
STR_COLS = ["tree_key", "scheme", "mode", "k_mode", "ratio", "allocation", "group",
            "is_start", "is_end", "oos_start", "oos_end", "linkage_method"]


def _csv(name: str) -> pd.DataFrame:
    head = pd.read_csv(SRC / name, nrows=0).columns
    # float_precision="round_trip"：pandas 預設的快速浮點解析會差最後 1 個位元（實測 partition 4 格差 1e-16），
    # 用精確解析才能讓美股列寫回後逐位相同
    return pd.read_csv(SRC / name, dtype={c: str for c in STR_COLS if c in head}, float_precision="round_trip")


def _records(df: pd.DataFrame) -> list[dict]:
    import numpy as np
    return [{k: (list(v) if isinstance(v, np.ndarray)
                 else None if (not isinstance(v, (list, tuple)) and pd.isna(v)) else v) for k, v in r.items()}
            for r in df.to_dict("records")]


def patch_k_stability(mod):
    old = _csv("k_stability.csv")
    old = old[old.tree_key == KEEP]
    orig = mod.scan_window

    def scan_window(tree_key, is_start, is_end, months_long, meta, log=print):
        if tree_key != KEEP:
            return orig(tree_key, is_start, is_end, months_long, meta, log)
        r = old[(old.is_start == is_start) & (old.is_end == is_end)]
        assert len(r) == 1, f"[{KEEP} {is_start}~{is_end}] 上一輪結果有 {len(r)} 列"
        log(f"  [{KEEP}] IS {is_start}~{is_end}｜沿用上一輪結果（美股資料未變）")
        return _records(r)[0]
    mod.scan_window = scan_window


def _window_filter(df: pd.DataFrame, srow) -> pd.DataFrame:
    return df[(df.tree_key == KEEP) & (df.scheme == str(srow.scheme))
              & (df.window_no.astype(int) == int(srow.window_no))]


def patch_walkforward_matrix(mod):
    det = _csv("walkforward_matrix_detail.csv")
    mem = pd.read_parquet(SRC / "walkforward_members.parquet")
    for c in ("tree_key", "scheme", "k_mode", "ratio", "allocation", "group"):
        mem[c] = mem[c].astype(str)
    orig_build, orig_one = mod.build_tree_for_window, mod.run_one_window

    def build_tree_for_window(tree_key, *a, **k):
        return {} if tree_key == KEEP else orig_build(tree_key, *a, **k)

    def run_one_window(tree_key, srow, tree, months_long, k_table, log=print):
        if tree_key != KEEP:
            return orig_one(tree_key, srow, tree, months_long, k_table, log)
        r, p = _window_filter(det, srow), _window_filter(mem, srow)
        assert len(r) and len(p), f"[{KEEP}/{srow.scheme}/w{srow.window_no}] 上一輪結果缺列"
        return _records(r), _records(p)
    mod.build_tree_for_window, mod.run_one_window = build_tree_for_window, run_one_window


def patch_walkforward_partition(mod):
    old = _csv("walkforward_partition.csv")
    orig_build, orig_one = mod.build_tree_for_window, mod.run_one_window

    def build_tree_for_window(tree_key, *a, **k):
        return {} if tree_key == KEEP else orig_build(tree_key, *a, **k)

    def run_one_window(tree_key, srow, tree, months_long, k_table, n_draws, log=print):
        if tree_key != KEEP:
            return orig_one(tree_key, srow, tree, months_long, k_table, n_draws, log)
        r = _window_filter(old, srow)
        assert len(r), f"[{KEEP}/{srow.scheme}/w{srow.window_no}] 上一輪結果缺列"
        return _records(r)
    mod.build_tree_for_window, mod.run_one_window = build_tree_for_window, run_one_window


def patch_effective_bets(mod):
    old = _csv("effective_number_of_bets.csv")
    orig = mod.compute

    def compute(trees=mod.DEFAULT_TREES, log=print):
        rest = [x for x in trees if not x.startswith(KEEP + "_")]
        new = orig(trees=rest, log=log) if rest else pd.DataFrame()
        keep = old[old.tree_id.isin([x for x in trees if x.startswith(KEEP + "_")])]
        log(f"[{KEEP}] 沿用上一輪結果 {len(keep)} 列（美股資料未變）")
        out = pd.concat([new, keep[new.columns]] if len(new) else [keep], ignore_index=True)
        order = {t: i for i, t in enumerate(trees)}
        return out.sort_values("tree_id", key=lambda s: s.map(order)).reset_index(drop=True)
    mod.compute = compute


def patch_turnover_cost(mod):
    import functools
    old = _csv("turnover_cost.csv")
    old = old[old.tree_key == KEEP][["tree_key", "ratio", "allocation", "n_strategies", "n_stocks_avg",
                                      "monthly_turnover", "annual_turnover", "gross_cagr"]]
    mod.run = functools.partial(mod.run, reuse=old)


PATCH = {"k_stability": patch_k_stability, "walkforward_matrix": patch_walkforward_matrix,
         "walkforward_partition": patch_walkforward_partition,
         "effective_bets": patch_effective_bets, "turnover_cost": patch_turnover_cost}
FILES = {"k_stability": ["k_stability.csv"],
         "walkforward_matrix": ["walkforward_matrix_detail.csv", "walkforward_members.parquet"],
         "walkforward_partition": ["walkforward_partition.csv"],
         "effective_bets": ["effective_number_of_bets.csv"], "turnover_cost": ["turnover_cost.csv"]}


def verify(name: str) -> int:
    """現行輸出裡的美股列，必須與上一輪來源逐列相同。"""
    ok = True
    for f in FILES[name]:
        assert (OUT / f).stat().st_mtime > (SRC / f).stat().st_mtime, f"{f} 不是剛產生的（還是舊檔），驗證無效"
        rd = (lambda p: pd.read_parquet(p)) if f.endswith(".parquet") else (lambda p: pd.read_csv(p, float_precision="round_trip"))
        a, b = rd(SRC / f), rd(OUT / f)
        a, b = a[a.tree_key.astype(str) == KEEP], b[b.tree_key.astype(str) == KEEP]
        if f.endswith(".parquet"):
            a = a.assign(members=a.members.map(tuple)); b = b.assign(members=b.members.map(tuple))
        for d in (a, b):
            for c in d.columns:
                d[c] = d[c].astype(str)
        a, b = a.sort_values(list(a.columns)).reset_index(drop=True), b.sort_values(list(b.columns)).reset_index(drop=True)
        same = list(a.columns) == list(b.columns) and a.equals(b)
        ok &= same
        print(f"{'✓' if same else '✗'} {f}：美股 上一輪 {len(a)} 列／現行 {len(b)} 列｜逐列相同={same}")
    return 0 if ok else 1


def main() -> int:
    assert Path.cwd().resolve() == paths.ROOT.joinpath("code").resolve(), "請在 code/ 底下執行"
    if len(sys.argv) >= 3 and sys.argv[1] == "verify":
        return verify(sys.argv[2])
    name = sys.argv[1]
    mod = importlib.import_module(f"research.{name}")
    PATCH[name](mod)
    print(f">> {name}：台股／跨市場照常計算，美股沿用 {SRC.name}", flush=True)
    return mod.main(sys.argv[2:]) or 0


if __name__ == "__main__":
    sys.exit(main())
