# -*- coding: utf-8 -*-
"""執行陰性對照 A（scheme E window 3，12季）與 B（scheme A window 2，8季）。
用法：python -m app._run_negative_controls [A|B|both] [--real]
預設 dry_run=True；加 --real 才會真的燒 token（需使用者授權）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import _negative_control as nc  # noqa: E402
from app import simulate  # noqa: E402
from utils.config import Config  # noqa: E402


def summarize(run_id: str):
    cps = simulate.load_checkpoints(run_id)
    print(f"\n=== {run_id}：{len(cps)} 筆 checkpoint ===")
    for c in sorted(cps, key=lambda x: (x["arm"], x["quarter_end"])):
        print(f"  {c['arm']:<9s} {c['quarter_end']}  狀態={c['m1d']['state']:<10s}"
             f" 決策={c['decision']['decision']}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    dry_run = "--real" not in sys.argv

    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")
    print(f"模型={model}  dry_run={dry_run}")

    if which in ("A", "both"):
        nc.run_negative_control("neg_control_A_E_w3", scheme="E", window_no=3,
                                model=model, api_key=api_key, dry_run=dry_run)
        summarize("neg_control_A_E_w3")

    if which in ("B", "both"):
        nc.run_negative_control("neg_control_B_A_w2", scheme="A", window_no=2,
                                model=model, api_key=api_key, dry_run=dry_run)
        summarize("neg_control_B_A_w2")


if __name__ == "__main__":
    main()
