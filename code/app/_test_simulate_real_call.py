# -*- coding: utf-8 -*-
"""simulate.py 第一次真實單季測試（會燒真實 token）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import simulate  # noqa: E402
from utils.config import Config  # noqa: E402

RUN_ID = "_test_real_q1"


def main():
    p = simulate.checkpoint_path(RUN_ID)
    if p.exists():
        p.unlink()

    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")

    print(f"使用模型：{model}")
    print("=== 真實單季測試：control0 + L2，只跑第 1 季（會燒真實 token）===")
    simulate.run_simulation(RUN_ID, model=model, api_key=api_key,
                            dry_run=False, max_quarters=1)

    print("\n=== 檢查落盤結果 ===")
    checkpoints = simulate.load_checkpoints(RUN_ID)
    print(f"checkpoint 數量：{len(checkpoints)}（預期 2）")
    for c in checkpoints:
        print(f"\n--- {c['arm']} / {c['quarter_end']} ---")
        print(f"狀態={c['m1d']['state']}  決策={c['decision']['decision']}")
        if c["arm"] == "L2":
            print(f"decision keys: {list(c['decision'].keys())}")
            print(f"summary keys: {list(c['summary'].keys())}")


if __name__ == "__main__":
    main()
