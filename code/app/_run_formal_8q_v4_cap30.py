# -*- coding: utf-8 -*-
"""2026-10-13：台股 openSec_boost baseline（含V1），重跑以套用使用者剛拍板
的W2c單股上限新值（7%→30%，見`weights.py`開頭docstring）。

跟`_run_formal_8q_v4.py`（舊版，7%上限，headline數字control0+11.16%/
L2+32.73%）完全同一套跑法，只換RUN_ID，避免撞checkpoint被誤判成「已完成」
整批跳過（`run_rolling_simulation`／`run_simulation`都是用run_id查checkpoint
決定要不要跳過）。舊版checkpoint保留不刪，當作7%/30%上限前後對照用。

🔴🔴 2026-10-19補：原RUN_ID（`formal_8q_control0_L2_openSec_boost_v4_cap30`）
已經在10/13跑完一輪（16筆checkpoint，2臂×8季），但那是M2（2026-10-17才接線
進`simulate.run_simulation()`）接線**之前**跑的——直接查checkpoint內容確認
完全沒有`m2`欄位。若現在沿用同一個RUN_ID重跑，會被resume邏輯判定「16筆都
已存在」整批跳過，M2根本不會被真的補進去（這個專案已經踩過的「checkpoint
撞名造成靜默跳過」陷阱）。RUN_ID加`_m2`後綴，強制這次重新跑一輪真正含M2的
版本。舊的（無M2）checkpoint保留不刪，當作有無M2前後對照用。

🔴🔴 2026-10-19再補：`_m2`版本跑完後發現L2從+38.65%掉到+19.57%，追查發現
主因不是M2，是agent在理由裡直接引用`w2c_reference_json`裡「23.48/21.98pp
數字是7%舊上限算出來的」那條caveat，連續好幾季選保守的A5、放棄W2c。查證
後補了第5條caveat（引用`研究進度報告_2026-10-05.md`§3.3/§3.3d既有的cap
敏感度測試，說明「cap參數從7%放寬到30%」這個調整本身風險溫和、已驗證過，
跟「這組pp數字能不能套用」是兩個不同問題——這是補齊本來就存在但被忽略的
證據，不是刪警告或作弊。RUN_ID再加`_v2`後綴，這次要看補了這條caveat之後
agent的決策會不會不一樣。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import simulate  # noqa: E402
from utils.config import Config  # noqa: E402

RUN_ID = "formal_8q_control0_L2_openSec_boost_v4_cap30_m2_v2"


def main():
    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")

    print(f"使用模型：{model}")
    print("=== 台股openSec_boost baseline（W2c上限30%重跑）：control0 + L2，8季真實呼叫 ===")
    simulate.run_simulation(RUN_ID, model=model, api_key=api_key, dry_run=False)

    print("\n=== 完成，逐季狀態/決策總覽 ===")
    checkpoints = simulate.load_checkpoints(RUN_ID)
    print(f"checkpoint 數量：{len(checkpoints)}（預期 16 = 2臂 × 8季）")
    for c in sorted(checkpoints, key=lambda x: (x["arm"], x["quarter_end"])):
        o = c["outcome"]
        pr = o.get("portfolio_realized_return")
        pr_str = f"{pr:+.2%}" if pr is not None else "None"
        print(f"  {c['arm']:<9s} {c['quarter_end']}  狀態={c['m1d']['state']:<10s}"
             f" 決策={c['decision']['decision']}｜投組報酬={pr_str}")


if __name__ == "__main__":
    main()
