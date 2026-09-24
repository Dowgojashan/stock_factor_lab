# -*- coding: utf-8 -*-
"""§4④b：rolling(window6, silhouette_is)候選池的正式8季實驗（2026-09-23，
使用者授權執行），control0+L2兩臂×8季，真實LLM呼叫。

跟`_run_formal_8q_v3.py`（anchored版，9/22報告那組結果）完全同一套跑法，
只換候選池來源（`_check_rolling_production_simulation.run_rolling_simulation()`，
見該檔docstring的安全設計說明）跟RUN_ID_PREFIX。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._run_rolling_w6_formal
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import simulate  # noqa: E402
from app._check_rolling_production_simulation import run_rolling_simulation  # noqa: E402
from utils.config import Config  # noqa: E402

RUN_ID_PREFIX = "rolling_w6"


def main():
    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")

    print(f"使用模型：{model}")
    print("=== rolling window6 正式實驗：control0 + L2，8季真實呼叫（使用者已授權）===")
    run_rolling_simulation(run_id_prefix=RUN_ID_PREFIX, model=model, api_key=api_key, dry_run=False)

    print("\n=== 完成，逐季狀態/決策總覽 ===")
    for arm in simulate.ARMS:
        run_id = f"{RUN_ID_PREFIX}_{arm}"
        checkpoints = simulate.load_checkpoints(run_id)
        print(f"\n[{arm}] checkpoint 數量：{len(checkpoints)}（預期 8）")
        for c in sorted(checkpoints, key=lambda x: x["quarter_end"]):
            print(f"  {c['quarter_end']}  狀態={c['m1d']['state']:<10s}"
                 f" 決策={c['decision']['decision']}"
                 f"｜OOS超額(vs B_all)={c['outcome']['excess_vs_ball']:+.2%}")


if __name__ == "__main__":
    main()
