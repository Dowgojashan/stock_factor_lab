# -*- coding: utf-8 -*-
"""正式 8 季實驗，第四版（2026-10-04，使用者授權重跑）：control0＋L2 兩臂 ×
8 季（2024-Q1~2025-Q4），真實 LLM 呼叫。

🔴 重跑原因：台股候選池這次全面換成 openSec_boost 變體（新增
PROX_52WK_HIGH/MOM_3M 動能因子為F1、Phase2強制納入ROE/EPS/ROIC/REV_G/
MOM_3M當primary、C因子改成六大類各一代表），候選池 15,009→29,255，HRP
L1群數也從k=7改成重新計算出的k=6（見`文件/完整操作流程.md`§2.13、
CLAUDE.md §0b）。`walkforward_members.parquet`（scheme=E/window_no=4/
k_mode=silhouette_is/ratio=legacy/group=A_hrp，IS=2007-01~2023-12、
OOS=2024-01~2025-12）已確認是用這批新資料重算過的。

使用者的目的：看等權基準下 2024-2025 表現依然遠低於TAIEX（+70.56%），
這次要驗證「進入實戰調整權重」（W2c等機制）有沒有更大的改善空間——
不是在測試新因子池本身，是測試新因子池 + 實戰權重調整機制疊加的效果。

跟 `_run_formal_8q_v3.py` 完全同一套跑法，只換 RUN_ID——舊版（v3，舊池
15,009/k=7）保留不覆蓋，當作新舊池子的對照紀錄。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import simulate  # noqa: E402
from utils.config import Config  # noqa: E402

RUN_ID = "formal_8q_control0_L2_openSec_boost_v4"


def main():
    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")

    print(f"使用模型：{model}")
    print("=== 正式實驗第四版：openSec_boost新候選池，control0 + L2，8季真實呼叫 ===")
    simulate.run_simulation(RUN_ID, model=model, api_key=api_key, dry_run=False)

    print("\n=== 完成，逐季狀態/決策總覽 ===")
    checkpoints = simulate.load_checkpoints(RUN_ID)
    print(f"checkpoint 數量：{len(checkpoints)}（預期 16 = 2臂 × 8季）")
    for c in sorted(checkpoints, key=lambda x: (x["arm"], x["quarter_end"])):
        print(f"  {c['arm']:<9s} {c['quarter_end']}  狀態={c['m1d']['state']:<10s}"
             f" 決策={c['decision']['decision']}")


if __name__ == "__main__":
    main()
