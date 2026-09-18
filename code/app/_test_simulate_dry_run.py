# -*- coding: utf-8 -*-
"""simulate.py 的 dry_run 驗證：不燒 token，確認整條管線串得起來、
checkpoint 落盤／續跑機制正常運作。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import simulate  # noqa: E402

RUN_ID = "_test_dry_run"


def main():
    # 清掉舊的測試 checkpoint，從乾淨狀態開始
    p = simulate.checkpoint_path(RUN_ID)
    if p.exists():
        p.unlink()

    print("=== 第一次執行：跑前 2 季（dry_run，不燒 token）===")
    simulate.run_simulation(RUN_ID, model="gpt-5", api_key="fake-key-dry-run",
                            dry_run=True, max_quarters=2)

    checkpoints = simulate.load_checkpoints(RUN_ID)
    print(f"\n寫入 checkpoint 數量：{len(checkpoints)}（預期 2 臂 × 2 季 = 4）")
    for c in checkpoints:
        print(f"  {c['arm']} / {c['quarter_end']}：狀態={c['m1d']['state']}，"
             f"決策={c['decision']['decision']}")

    print("\n=== 第二次執行：續跑同一個 run_id（應該全部跳過，不重算）===")
    simulate.run_simulation(RUN_ID, model="gpt-5", api_key="fake-key-dry-run",
                            dry_run=True, max_quarters=2)
    checkpoints2 = simulate.load_checkpoints(RUN_ID)
    print(f"\n續跑後 checkpoint 數量：{len(checkpoints2)}（應該還是 4，沒有重複寫入）")
    assert len(checkpoints2) == len(checkpoints), "續跑不應該產生新的 checkpoint"

    print("\n全部測試通過。")


if __name__ == "__main__":
    main()
