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
    # ⚠️ 2026-08-25 code review 修正：原本這裡的說明文字寫「→ strategy_marks」，
    # 但實際掛的模組是 stage1_scan（產出 strategy_scan，不是 strategy_marks）——
    # 兩者共用同一份manifest時這個誤導不明顯，拆開manifest後說明必須跟模組對齊。
    "stage1": ("階段1a · 單趟掃描 → strategy_scan", paths.STAGE1, "research.stage1_scan"),
    "stage1_marks": ("階段1b · 標記關卡 A/B/C → strategy_marks",
                     paths.STAGE1 / "_marks", "research.stage1_marks"),
    "stage2a": ("階段2a · Regime Dating（牛熊切割）", paths.STAGE2 / "regime", "research.stage2a_regime"),
    "stage2b": ("階段2b · 總經 → 月頻特徵表", paths.STAGE2 / "macro", "research.stage2b_macro"),
    "stage2c": ("階段2c · 交叉佐證（2a×2b 匯流）", paths.STAGE2 / "consistency", "research.stage2c_consistency"),
    "stage3": ("階段3 · HRP 階層聚類", paths.STAGE3, "research.stage3_hrp"),
    "stage4": ("階段4 · strategy_map 彙整凍結", paths.STAGE4, "research.stage4_strategy_map"),
}

#: W-08：資料品質防線常設化。不是 STAGES（不寫 _frozen manifest——新資料來了本來就
#: 該掃出不同結果，不是「重跑要位元相同」的凍結產物語意），獨立走 dataquality 子指令，
#: 讓它跟其他階段一樣「一行指令跑一次」，而不是只能用 python -m research.diagnose_price_anomalies
#: 這種散落的呼叫方式（見同名模組 docstring 的完整判定方法論）。
DATAQUALITY_DESC = "資料品質防線 · 原始價格異常深度掃描 + 與 stage1 data_glitch 交叉核對"


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

    dq_file = paths.ROOT / "_analysis_outputs_dataquality" / "price_anomaly_events.csv"
    if dq_file.exists():
        import datetime
        ts = datetime.datetime.fromtimestamp(dq_file.stat().st_mtime).strftime("%Y-%m-%dT%H:%M")
        status, desc = "已執行過", f"{DATAQUALITY_DESC}   [{ts}]"
    else:
        status, desc = "未執行", DATAQUALITY_DESC
    print(f"{'dataq':<8} {status:<10} {desc}")
    return 0


def cmd_run(stage: str, extra: list[str] | None = None) -> int:
    if stage not in STAGES:
        print(f"未知階段 {stage}；可用：{', '.join(STAGES)}", file=sys.stderr)
        return 2
    desc, _, mod = STAGES[stage]
    if mod is None:
        print(f"{stage} 尚未開發（{desc}）", file=sys.stderr)
        return 2
    import importlib
    # ⚠️ 子模組的 main(argv=None) 預設會自己讀 sys.argv[1:]——但那份 argv
    # 是「research.cli stage1」這一整串，子模組的 argparse 看到多出來的
    # "stage1" 會直接報 unrecognized arguments 而中止。必須明確傳入剩餘參數
    # （沒有就傳空 list），子模組才不會誤讀 cli.py 自己已經解析掉的部分。
    return importlib.import_module(mod).main(extra or [])


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


def cmd_dataquality() -> int:
    from . import diagnose_price_anomalies
    return diagnose_price_anomalies.main()


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
    sub.add_parser("dataquality", help=DATAQUALITY_DESC)

    a = ap.parse_args(argv)
    if a.cmd == "list":
        return cmd_list()
    if a.cmd == "run":
        return cmd_run(a.stage)
    if a.cmd == "verify":
        return cmd_verify(a.stage)
    if a.cmd == "dataquality":
        return cmd_dataquality()
    return cmd_run(a.cmd)


if __name__ == "__main__":
    sys.exit(main())
