# -*- coding: utf-8 -*-
"""研究部管線的統一進入點。

設計意圖（SDD 8.2）：老師要親自跑程式，他拿到的應該是「一行指令跑一個階段」，
而不是一堆散落的腳本。

用法：
    cd code
    python -m research.cli list             # 列出各階段狀態
    python -m research.cli stage0           # 跑階段0
    python -m research.cli verify stage0    # 驗證凍結產物未被改動
"""
from __future__ import annotations

import argparse
import sys

from . import freeze, paths

STAGES = {
    "stage0": ("階段0 · 格式整備 → candidate_index", paths.STAGE0, "research.stage0_index"),
    "stage1": ("階段1 · 標記關卡 A/B/C → strategy_marks", paths.STAGE1, "research.stage1_scan"),
    "stage2": ("階段2 · regime + 總經對應", paths.STAGE2, None),
    "stage3": ("階段3 · HRP 階層聚類", paths.STAGE3, None),
    "stage4": ("階段4 · strategy_map 彙整凍結", paths.STAGE4, None),
}


def cmd_list() -> int:
    print(f"{'階段':<8} {'狀態':<10} 說明")
    print("-" * 64)
    for key, (desc, d, mod) in STAGES.items():
        if mod is None:
            status = "未開發"
        elif (d / freeze.MANIFEST_NAME).exists():
            m = freeze.read_manifest(d)
            status = f"已完成"
            desc = f"{desc}   [{m['produced_at'][:16]} @ {m['git_sha']}]"
        else:
            status = "未執行"
        print(f"{key:<8} {status:<10} {desc}")
    return 0


def cmd_run(stage: str) -> int:
    if stage not in STAGES:
        print(f"未知階段 {stage}；可用：{', '.join(STAGES)}", file=sys.stderr)
        return 2
    desc, _, mod = STAGES[stage]
    if mod is None:
        print(f"{stage} 尚未開發（{desc}）", file=sys.stderr)
        return 2
    import importlib
    return importlib.import_module(mod).main()


def cmd_verify(stage: str) -> int:
    _, d, _ = STAGES[stage]
    try:
        m = freeze.verify_inputs(d)
    except (FileNotFoundError, freeze.FreezeViolation) as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1
    rows = ", ".join(f"{o['path'].split(chr(92))[-1]}={o.get('rows','?')}列"
                     for o in m["outputs"])
    print(f"✓ {stage} 凍結產物完好（{m['produced_at'][:16]} @ {m['git_sha']}）：{rows}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.cli", description="研究部管線")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="列出各階段狀態")
    p_run = sub.add_parser("run", help="執行某階段")
    p_run.add_argument("stage")
    p_ver = sub.add_parser("verify", help="驗證某階段的凍結產物")
    p_ver.add_argument("stage")
    for s in STAGES:                       # 便捷寫法：cli stage0 == cli run stage0
        sub.add_parser(s, help=STAGES[s][0])

    a = ap.parse_args(argv)
    if a.cmd == "list":
        return cmd_list()
    if a.cmd == "run":
        return cmd_run(a.stage)
    if a.cmd == "verify":
        return cmd_verify(a.stage)
    return cmd_run(a.cmd)


if __name__ == "__main__":
    sys.exit(main())
