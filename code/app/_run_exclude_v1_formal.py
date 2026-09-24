# -*- coding: utf-8 -*-
"""§3.14：anchored、rolling window6 exclude_v1 兩組候選池的正式8季實驗
（2026-09-24，使用者授權執行），control0+L2兩臂×8季，真實LLM呼叫。

動機：§3.12的rolling window6正式模擬（+20.19%）使用者發現候選池其實V1比例
高達56%（equal allocation），既然§3.8已經證實排除V1同時改善composition跟
績效，這裡把exclude_v1接進完整實戰管線——anchored、rolling都要跑，才能跟
既有的baseline（anchored+24.70%、rolling+20.19%）公平對照，湊齊4格矩陣
（anchored/rolling × baseline/exclude_v1）在**真實production管線**下的結果
（§3.8/3.9那組2×2矩陣是純離線composition/CAGR測試，這裡是第一次在完整
monitor→triggers→diagnose→facts_lean→agents→simulate管線下驗證）。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8；先跑過
`_build_exclude_v1_silhouette_picks.py`產生兩組候選池才能跑這支）：
    PYTHONIOENCODING=utf-8 python -m app._run_exclude_v1_formal
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import simulate  # noqa: E402
from app._check_rolling_production_simulation import run_rolling_simulation  # noqa: E402
from utils.config import Config  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"

RUNS = [
    dict(name="anchored_ev1", run_id_prefix="anchored_ev1",
        picks_path=OUT_DIR / "anchored_w4_silhouette_exclude_v1.parquet",
        scheme="E", window_no=4),
    dict(name="rolling_w6_ev1", run_id_prefix="rolling_w6_ev1",
        picks_path=OUT_DIR / "rolling_w6_silhouette_exclude_v1.parquet",
        scheme="rolling_6_2", window_no=6),
]


def main():
    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")
    print(f"使用模型：{model}")

    for run in RUNS:
        print(f"\n=== {run['name']} 正式實驗：control0 + L2，8季真實呼叫（使用者已授權）===")
        run_rolling_simulation(run_id_prefix=run["run_id_prefix"], model=model, api_key=api_key,
                               dry_run=False, picks_path=run["picks_path"],
                               scheme=run["scheme"], window_no=run["window_no"])

        print(f"\n--- {run['name']} 完成，逐季狀態/決策總覽 ---")
        for arm in simulate.ARMS:
            run_id = f"{run['run_id_prefix']}_{arm}"
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
