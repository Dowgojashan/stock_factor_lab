# -*- coding: utf-8 -*-
"""2026-10-05：通用版——跑指定品質窗口長度的exclude_v1正式8季模擬。

用法：
    PYTHONIOENCODING=utf-8 python -m app._run_recency_quality_variant --months 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import simulate  # noqa: E402
from app._check_rolling_production_simulation import run_rolling_simulation  # noqa: E402
from utils.config import Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, required=True)
    args = ap.parse_args()
    m = args.months

    run_id_prefix = f"anchored_w4_openSec_boost_exclude_v1_recent{m}mo"
    picks_path = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
                 / f"anchored_w4_openSec_boost_exclude_v1_recent{m}mo.parquet")

    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")
    print(f"使用模型：{model}")
    print(f"=== exclude_v1+近{m}個月品質：control0 + L2，8季真實呼叫 ===")

    run_rolling_simulation(run_id_prefix=run_id_prefix, model=model, api_key=api_key,
                           dry_run=False, picks_path=picks_path,
                           tree_key="TW", scheme="E", window_no=4)

    print("\n--- 完成，逐季狀態/決策總覽 ---")
    for arm in simulate.ARMS:
        run_id = f"{run_id_prefix}_{arm}"
        checkpoints = simulate.load_checkpoints(run_id)
        print(f"\n[{arm}] checkpoint 數量：{len(checkpoints)}（預期 8）")
        for c in sorted(checkpoints, key=lambda x: x["quarter_end"]):
            o = c["outcome"]
            pr = o.get("portfolio_realized_return")
            pr_str = f"{pr:+.2%}" if pr is not None else "None"
            print(f"  {c['quarter_end']}  狀態={c['m1d']['state']:<10s}"
                 f" 決策={c['decision']['decision']}｜投組報酬={pr_str}")


if __name__ == "__main__":
    main()
