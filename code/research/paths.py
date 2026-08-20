# -*- coding: utf-8 -*-
"""路徑與 sys.path 掛載（本套件所有模組的共同前置）。

為什麼需要這支：`database.py` / `get_data.py` / `combinations.py` 等在**根目錄**，
但工作腳本在 `code/` 底下執行。`fcv_core.py` 開頭會把 ROOT 與 code/ 都加進
sys.path，所以只要 import 過一次 fcv_core，後面 `from database import Database`
才會找得到。本模組把這件事集中處理，避免每支腳本各自重複。
"""
import sys
from pathlib import Path

# code/research/paths.py -> code/research -> code -> repo root
RESEARCH_DIR = Path(__file__).resolve().parent
CODE_DIR = RESEARCH_DIR.parent
ROOT = CODE_DIR.parent

for _p in (str(ROOT), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- 輸入（階段 −1 產物，唯讀） ----
PHASE4_DIR = ROOT / "_analysis_outputs_phase4"
ARTIFACTS_DIR = CODE_DIR / "results_artifacts"

# ---- 輸出（研究部凍結產物） ----
FROZEN = ROOT / "_frozen"
STAGE0 = FROZEN / "stage0"
STAGE1 = FROZEN / "stage1"
STAGE2 = FROZEN / "stage2"
STAGE3 = FROZEN / "stage3"
STAGE4 = FROZEN / "stage4"

MARKETS = ("TW", "US")
VARIANT = "openSec"          # 採用的正式設計（研究部 v9 定案）
IN_SAMPLE_END = "2025-12"

# 落差2（SDD DD-01）：v0 的產物留在 L3，v1 在 L4。
#   L4 是 v0/v1 對照實驗、只跑 v1，故 v0 的回測產物留在 Phase 3 的 job 目錄。
#   實測覆蓋率 100%（14,078/14,078）。
JOB_BY_V = {"v0": "L3", "v1": "L4"}


def job_label(market: str, v: str) -> str:
    """回傳該策略回測產物所在的 job 目錄名，如 `TW_L3_openSec_M`。"""
    if v not in JOB_BY_V:
        raise ValueError(f"未知的 V 值: {v!r}（僅接受 v0/v1）")
    return f"{market}_{JOB_BY_V[v]}_{VARIANT}_M"


def artifacts_path(market: str, strategy: str, v: str) -> Path:
    """單一策略的回測產物目錄（絕對路徑）。"""
    return ARTIFACTS_DIR / job_label(market, v) / strategy


def candidates_csv(market: str) -> Path:
    """階段 −1 的候選池 CSV。注意：UTF-8 with BOM，須 encoding='utf-8-sig'。"""
    return PHASE4_DIR / f"{market}_L4_{VARIANT}_final_candidates.csv"


def stats_parquet(market: str, v: str) -> Path:
    """job 層級的 stats.parquet（sharpe_ann / daily_sharpe / avg_drawdown 的來源）。"""
    return ARTIFACTS_DIR / job_label(market, v) / "stats.parquet"


def ensure_dirs() -> None:
    for d in (FROZEN, STAGE0, STAGE1, STAGE2, STAGE3, STAGE4):
        d.mkdir(parents=True, exist_ok=True)
