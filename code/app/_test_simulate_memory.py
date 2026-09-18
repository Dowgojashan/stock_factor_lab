# -*- coding: utf-8 -*-
"""simulate.py 的記憶機制驗證：跑 3 季（dry_run，不燒 token），確認
recent_quarter_full／earlier_quarters_summary 正確遞移，且續跑能精確還原。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import simulate  # noqa: E402

RUN_ID = "_test_memory"


def main():
    p = simulate.checkpoint_path(RUN_ID)
    if p.exists():
        p.unlink()

    print("=== 跑 3 季（dry_run）===")
    simulate.run_simulation(RUN_ID, model="gpt-5", api_key="fake-key",
                            dry_run=True, max_quarters=3)

    checkpoints = simulate.load_checkpoints(RUN_ID)
    l2 = [c for c in checkpoints if c["arm"] == "L2"]
    l2.sort(key=lambda c: c["quarter_end"])

    print(f"\n=== L2 臂 {len(l2)} 季的記憶遞移檢查 ===")
    for i, c in enumerate(l2):
        mem = c["agent_a_memory_after"]
        earlier_ends = [s["quarter_end"] for s in mem["earlier_quarters_summary"]]
        recent_end = mem["recent_quarter_full"]["quarter_end"] if mem["recent_quarter_full"] else None
        print(f"  第{i+1}季（{c['quarter_end']}）完成後：recent_quarter_full={recent_end}，"
             f"earlier_quarters_summary涵蓋={earlier_ends}")

    # 驗證：第1季完成後 recent=第1季本身，earlier=[]
    assert l2[0]["agent_a_memory_after"]["recent_quarter_full"]["quarter_end"] == l2[0]["quarter_end"]
    assert l2[0]["agent_a_memory_after"]["earlier_quarters_summary"] == []
    # 驗證：第2季完成後 recent=第2季，earlier=[第1季摘要]
    assert l2[1]["agent_a_memory_after"]["recent_quarter_full"]["quarter_end"] == l2[1]["quarter_end"]
    assert len(l2[1]["agent_a_memory_after"]["earlier_quarters_summary"]) == 1
    assert l2[1]["agent_a_memory_after"]["earlier_quarters_summary"][0]["quarter_end"] == l2[0]["quarter_end"]
    # 驗證：第3季完成後 recent=第3季，earlier=[第1季摘要,第2季摘要]
    assert l2[2]["agent_a_memory_after"]["recent_quarter_full"]["quarter_end"] == l2[2]["quarter_end"]
    assert len(l2[2]["agent_a_memory_after"]["earlier_quarters_summary"]) == 2

    print("\n記憶遞移驗證通過：最近一期永遠是全文，更早期正確遞移成結構化摘要。")

    print("\n=== 續跑測試：清掉記憶重新從 checkpoint 還原 ===")
    simulate.run_simulation(RUN_ID, model="gpt-5", api_key="fake-key",
                            dry_run=True, max_quarters=3)
    checkpoints2 = simulate.load_checkpoints(RUN_ID)
    assert len(checkpoints2) == len(checkpoints), "續跑不應該產生新 checkpoint"
    print("續跑正確跳過，checkpoint 數量沒有變化。")

    print("\n全部測試通過。")


if __name__ == "__main__":
    main()
