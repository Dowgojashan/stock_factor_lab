# -*- coding: utf-8 -*-
"""凍結、版本與指紋機制（SDD DD-08）。

要解決的問題：凍結原則若只靠人工紀律，一旦有人改了上游而下游沒重跑，
會**靜默產出錯誤且無法察覺**——這對「凍結地圖不作弊」的論文主張是致命的。

做法：每階段產出旁附 MANIFEST.json，記錄輸入檔的 sha256、參數、git SHA。
下游執行前呼叫 verify_inputs()，重算上游指紋並比對，不符即拒絕執行。

附帶效益：這份 manifest 就是論文「可複現性」章節的直接證據，
也是老師要親自跑程式時的環境對帳單。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from . import paths

MANIFEST_NAME = "MANIFEST.json"
_CHUNK = 1 << 20


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=paths.ROOT, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _describe(p: Path) -> dict[str, Any]:
    d: dict[str, Any] = {
        "path": str(p.relative_to(paths.ROOT)) if p.is_relative_to(paths.ROOT) else str(p),
        "sha256": sha256_file(p),
        "bytes": p.stat().st_size,
    }
    if p.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            d["rows"] = pq.ParquetFile(p).metadata.num_rows
        except Exception:
            pass
    return d


def write_manifest(stage: str, out_dir: Path, *,
                   inputs: Iterable[Path], outputs: Iterable[Path],
                   params: dict[str, Any] | None = None,
                   notes: str | None = None) -> Path:
    """為一個階段的產出寫 manifest。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": stage,
        "produced_at": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "params": params or {},
        "inputs": [_describe(Path(p)) for p in inputs],
        "outputs": [_describe(Path(p)) for p in outputs],
    }
    if notes:
        manifest["notes"] = notes
    dest = out_dir / MANIFEST_NAME
    dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def read_manifest(stage_dir: Path) -> dict[str, Any]:
    p = Path(stage_dir) / MANIFEST_NAME
    if not p.exists():
        raise FileNotFoundError(
            f"找不到 {p}。上游階段尚未執行，或產物被移動過。")
    return json.loads(p.read_text(encoding="utf-8"))


class FreezeViolation(RuntimeError):
    """上游產物與 manifest 記錄不符——凍結被破壞。"""


def verify_inputs(stage_dir: Path) -> dict[str, Any]:
    """驗證某階段的產出仍與其 manifest 相符（下游執行前呼叫）。

    不符即 raise FreezeViolation——**拒絕執行下游**，而不是印個警告就繼續。
    """
    m = read_manifest(stage_dir)
    bad: list[str] = []
    for rec in m.get("outputs", []):
        p = paths.ROOT / rec["path"]
        if not p.exists():
            bad.append(f"{rec['path']}：檔案不存在")
            continue
        got = sha256_file(p)
        if got != rec["sha256"]:
            bad.append(f"{rec['path']}：sha256 不符"
                       f"（manifest={rec['sha256'][:12]}… 實際={got[:12]}…）")
    if bad:
        raise FreezeViolation(
            f"[{m['stage']}] 凍結產物已被改動，下游拒絕執行：\n  - "
            + "\n  - ".join(bad)
            + "\n請重跑該階段，或確認是否誤動了 _frozen/ 底下的檔案。"
        )
    return m
