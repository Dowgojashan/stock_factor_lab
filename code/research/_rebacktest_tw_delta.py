# -*- coding: utf-8 -*-
"""台股股價修正後 Phase 2 白名單變動的增量評估（2026-11-09 使用者授權「先做增量評估」）。

背景：股價修正後重跑台股 Phase 2（`_analysis_outputs_phase2_twfix/`），F1×F2 白名單從 521 組變成 577 組
（新增 57、移除 1）。但 2026-10-31～11-01 重跑的 Phase 3／4 沿用舊白名單，候選池（29,642）少了新增組合。
本腳本只回測「新增的組合」，算出它們會讓候選池多出多少策略，用來決定要不要整條重跑。

做法（不改原始程式、不動任何既有目錄）：
  1. 新增組合＝新白名單 − 舊白名單；移除組合＝舊 − 新
  2. Phase 3（v0）、Phase 4（v1）回測：呼叫 `phase3_conditions.main()`／`phase4_valuation.main()`，
     白名單換成新增組合，輸出到 `code/results_artifacts_twfix_delta/`
  3. 候選判定沿用 `phase4_analyze` 的最終候選規則：v0、v1 各自 `平均持股數 >= MIN_HOLDINGS` 且 `CAGR > 基準`
     （基準＝股價修正後的自建宇宙基準，`results_artifacts_twfix/TW_UNIVERSE_M`）
  4. 移除組合在現行候選池中的策略數，從 `_analysis_outputs_phase4_twfix/` 的最終候選數出

用法（cwd＝code/）：
    python -m research._rebacktest_tw_delta 3        # Phase 3 回測
    python -m research._rebacktest_tw_delta 4        # Phase 4 回測
    python -m research._rebacktest_tw_delta report   # 候選判定與彙總
"""
from __future__ import annotations

import functools
import importlib
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = CODE.parent
FIX = CODE / "results_artifacts_twfix"
DELTA = CODE / "results_artifacts_twfix_delta"
OUT = ROOT / "_analysis_outputs_robustness" / "_chain_20261101" / "phase2_delta"
VARIANT = "openSec_boost"
OLD_P2 = ROOT / "_analysis_outputs_phase2" / f"TW_L2_{VARIANT}_體質檢查表.csv"
NEW_P2 = ROOT / "_analysis_outputs_phase2_twfix" / f"TW_L2_{VARIANT}_體質檢查表.csv"
MODULES = {"3": "phase3_conditions", "4": "phase4_valuation"}


def _pairs(p: Path) -> set[str]:
    return {"__".join(s.split("__")[:2]) for s in pd.read_csv(p, encoding="utf-8-sig")["strategy"]}


def delta_pairs() -> tuple[set[str], set[str]]:
    old, new = _pairs(OLD_P2), _pairs(NEW_P2)
    return new - old, old - new


def run_phase(phase: str) -> int:
    assert Path.cwd().resolve() == CODE.resolve(), f"請在 code/ 底下執行（現在 {Path.cwd()}）"
    added, removed = delta_pairs()
    print(f">> 新增組合 {len(added)}｜移除組合 {len(removed)}｜只回測新增的部分 → {DELTA}", flush=True)
    DELTA.mkdir(parents=True, exist_ok=True)
    mod = importlib.import_module(MODULES[phase])
    df = pd.read_csv(NEW_P2, encoding="utf-8-sig")
    mod.load_allowed_pairs = lambda market, variant="strict": (set(added), df)
    mod.ART_DIR = DELTA
    mod.run_spec = functools.partial(fcv_core.run_spec, art_dir=DELTA)
    mod.is_done = functools.partial(fcv_core.is_done, art_dir=DELTA)
    mod.CATALOG_DIR = DELTA
    mod.RUN_LOG = DELTA / f"_delta_phase{phase}_log.txt"
    sys.argv = [mod.__file__, "--market", "TW", "--variant", VARIANT]
    return mod.main()


def report() -> int:
    import phase2_analyze as P2A
    import universe_benchmark as UB
    from phase3_analyze import parse
    UB.ART_DIR = FIX
    bench, _ = UB.get_bench("TW", "universe")
    added, removed = delta_pairs()
    P2A.ART = DELTA
    rows = []
    for lab, v in (("L3", "v0"), ("L4", "v1")):
        name = f"TW_{lab}_{VARIANT}_M"
        st = pd.read_parquet(DELTA / name / "stats.parquet")
        st["F組合"] = st["strategy"].map(lambda s: parse(s)[0])
        st = st[st["F組合"].isin(added)].copy()
        st["持股數"] = [P2A.avg_holdings(name, s) for s in st["strategy"]]
        st["V"] = v
        st["候選"] = (st["持股數"] >= P2A.MIN_HOLDINGS) & (st["CAGR"] > bench)
        rows.append(st)
    d = pd.concat(rows, ignore_index=True)
    cur = pd.read_csv(ROOT / "_analysis_outputs_phase4_twfix" / f"TW_L4_{VARIANT}_final_candidates.csv",
                      encoding="utf-8-sig")
    cur_removed = int(cur["F組合"].isin(removed).sum())
    by = (d.groupby("F組合").agg(回測=("strategy", "size"), 候選=("候選", "sum"),
                               CAGR中位=("CAGR", "median")).sort_values("候選", ascending=False))
    OUT.mkdir(parents=True, exist_ok=True)
    by.to_csv(OUT / "delta_by_combo.csv", encoding="utf-8-sig")
    d[d["候選"]].to_csv(OUT / "delta_candidates.csv", index=False, encoding="utf-8-sig")
    s = {"基準CAGR": bench, "新增組合": len(added), "移除組合": len(removed),
         "新增組合回測策略數": len(d), "新增組合候選數": int(d["候選"].sum()),
         "移除組合在現行候選池的策略數": cur_removed, "現行候選池": len(cur),
         "淨增加": int(d["候選"].sum()) - cur_removed}
    s["淨增加佔現行候選池"] = s["淨增加"] / len(cur)
    (OUT / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(s, ensure_ascii=False, indent=2))
    print(by.head(20).to_string())
    return 0


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg in MODULES:
        return run_phase(arg)
    if arg == "report":
        return report()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
