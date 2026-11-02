# -*- coding: utf-8 -*-
"""台股價格修正後重跑 Phase 3／Phase 4 的回測 job（2026-10-31 使用者授權「重跑全部回測」）。

不改原始程式：直接呼叫 `phase3_conditions.main()`／`phase4_valuation.main()`，只把輸出目錄、
完成判斷與執行紀錄改到新目錄 `code/results_artifacts_twfix/`。舊的 `results_artifacts/` 一個檔都不動
（備份原則；新舊並存，之後研究部切換 TW 讀新目錄、US 照舊）。
其餘完全沿用原流程：同一份 Phase 2 白名單、同一個 spec（因子池、分桶、27 個 C）、同一個
MarketData 起訖（MARKET_START～IN_SAMPLE_END）、同樣的 MIN_TRADES 篩選與 batch 落地格式。

用法（cwd 必須是 code/，config.ini 用相對路徑）：
    python -m research._rebacktest_tw_jobs 3 --market TW --variant openSec_boost [--dry-run]
    python -m research._rebacktest_tw_jobs 4 --market TW --variant openSec_boost [--dry-run]
"""
from __future__ import annotations

import functools
import importlib
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
import fcv_core  # noqa: E402

NEW = CODE / "results_artifacts_twfix"
NEW_P4 = CODE.parent / "_analysis_outputs_phase4_twfix"
MODULES = {"3": "phase3_conditions", "4": "phase4_valuation"}


def run_universe(rest) -> int:
    """自建宇宙基準（`universe_benchmark.py`）改寫到新目錄（它的寫入與讀取都用模組層的 ART_DIR）。"""
    import universe_benchmark as UB
    UB.ART_DIR = NEW
    print(f">> 重跑自建宇宙基準｜輸出 → {NEW}", flush=True)
    sys.argv = [UB.__file__] + rest
    return UB.main() or 0


def run_p4analyze(rest) -> int:
    """Phase 4 候選篩選（`phase4_analyze.py`）：讀新目錄的回測與宇宙基準，輸出到 `_analysis_outputs_phase4_twfix/`
    （不覆蓋原本的 `_analysis_outputs_phase4/`）。Phase 2 白名單照舊讀原檔。"""
    import phase2_analyze as P2
    import universe_benchmark as UB
    import phase4_analyze as P4
    P2.ART = NEW; P4.ART = NEW; UB.ART_DIR = NEW
    NEW_P4.mkdir(parents=True, exist_ok=True)
    P4.OUT = NEW_P4
    print(f">> 重跑 Phase 4 候選篩選｜讀 {NEW}｜輸出 → {NEW_P4}", flush=True)
    sys.argv = [P4.__file__] + rest
    return P4.main() or 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] in ("universe", "p4analyze"):
        assert Path.cwd().resolve() == CODE.resolve(), f"請在 code/ 底下執行（現在 {Path.cwd()}）"
        rest = sys.argv[2:]
        if "--market" in rest and rest[rest.index("--market") + 1] != "TW":
            raise SystemExit("這支只用來重跑台股")
        return run_universe(rest) if sys.argv[1] == "universe" else run_p4analyze(rest)
    if len(sys.argv) < 2 or sys.argv[1] not in MODULES:
        print(__doc__); return 2
    phase, rest = sys.argv[1], sys.argv[2:]
    if "--market" in rest and rest[rest.index("--market") + 1] != "TW":
        raise SystemExit("這支只用來重跑台股（美股資料沒有修正，不要重跑）")
    old = (Path.cwd() / fcv_core.ART_DIR).resolve()
    assert old != NEW.resolve(), "新舊輸出目錄不可相同"
    assert Path.cwd().resolve() == CODE.resolve(), f"請在 code/ 底下執行（現在 {Path.cwd()}）"
    NEW.mkdir(parents=True, exist_ok=True)
    mod = importlib.import_module(MODULES[phase])
    # 原腳本以 `from fcv_core import run_spec, is_done, ART_DIR` 取得這三個名字，
    # run_spec／is_done 的 art_dir 預設值在定義時就綁死成舊目錄，所以要換成 partial。
    mod.ART_DIR = NEW
    mod.run_spec = functools.partial(fcv_core.run_spec, art_dir=NEW)
    mod.is_done = functools.partial(fcv_core.is_done, art_dir=NEW)
    mod.CATALOG_DIR = NEW                                   # log() 會 mkdir 這個目錄
    mod.RUN_LOG = NEW / f"_rerun_phase{phase}_log.txt"      # 不寫進既有的 _catalog/phase*_run_log.txt
    print(f">> 重跑 Phase {phase}｜輸出 → {NEW}｜舊目錄 {old} 不動", flush=True)
    sys.argv = [mod.__file__] + rest
    return mod.main()


if __name__ == "__main__":
    sys.exit(main())
