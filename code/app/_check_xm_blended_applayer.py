# -*- coding: utf-8 -*-
"""2026-10-14：跨市場（XM）實戰應用層——不發明新的「跨市場集中度」概念，
改成把XM投組拆成TW子帳本＋US子帳本，各自套用各自市場已校準的M1-D／Hot
Segment／W2c門檻跟變體（TW單檔／US籃子），分開監控、分開決策，最後依資本
佔比混合成XM整體報酬。

設計理由（2026-10-14使用者核准的方向）：
  - XM沒有自己的「市場指數」可以監控集中度——M1-D的概念本來就綁定單一
    市場的指數結構（`monitor.fetch_mcap_wide()`的exchange對照表只有TW/US），
    發明一個「跨市場混合集中度」指標沒有真實對應物，反而是憑空杜撰
  - 查`walkforward_members.parquet`（XM／scheme E／window4／legacy ratio／
    equal分配）：40檔代表策略剛好20檔TW::／20檔US::，1:1分割，跟HRP分群
    本身的性質吻合（M-01已證實：XM的群幾乎都是單一市場群，分散來源是市場
    邊界不是因子邊界）——拆成兩個子帳本不是簡化，是忠實反映HRP樹本身的
    真實結構
  - 兩個子帳本各自套用的監控/動作機制（TW單檔W2c、US籃子W2c）都已經在
    2026-10-14的cap30重跑裡完整驗證過，**不需要重新花LLM額度**——直接
    重用已經跑完的`anchored_w4_openSec_boost_exclude_v1_cap30`（TW）與
    `anchored_w4_openSec_balanced_exclude_v1_cap30`（US）checkpoint，依
    50/50資本佔比混合逐季報酬

用法：
    cd code
    python -m app._check_xm_blended_applayer
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import simulate  # noqa: E402

TW_WEIGHT = 0.5  # 查walkforward_members.parquet(XM/legacy/equal)實測是20:20=1:1
US_WEIGHT = 0.5

# 🔴 2026-10-20更新：改指向含M2+Hot Segment pool校準+完整5條W2c caveat的
# `_hotseg`版本checkpoint（舊版`_cap30`不含M2，已過期，見`文件/實驗收尾
# 追蹤.md`§一）。
# 🔴🔴 2026-10-22更新：US側再改指向`_m1dfix`版本——`_hotseg`checkpoint建立
# 在過期的「M1-D全程NONE」假設上（universe_clause bug＋2009-2012市值資料
# 本身有誤，兩輪修正後美股window4真實在2025Q3進入TRIGGERED）。TW側不受
# 這個bug影響，維持`_hotseg`版本不變。
RUN_IDS = {
    "control0": ("anchored_w4_openSec_boost_exclude_v1_cap30_hotseg_control0",
                "anchored_w4_openSec_balanced_exclude_v1_cap30_hotseg_m1dfix_control0"),
    "L2": ("anchored_w4_openSec_boost_exclude_v1_cap30_hotseg_L2",
          "anchored_w4_openSec_balanced_exclude_v1_cap30_hotseg_m1dfix_L2"),
}


def cum(returns: list[float]) -> float:
    v = 1.0
    for r in returns:
        v *= (1 + r)
    return (v - 1) * 100


def main():
    print(f"{'arm':>10} {'quarter':>12} {'TW報酬':>10} {'US報酬':>10} {'XM混合(50/50)':>14}")
    for arm, (tw_run, us_run) in RUN_IDS.items():
        tw = {c["quarter_end"]: c["outcome"]["portfolio_realized_return"]
             for c in simulate.load_checkpoints(tw_run)}
        us = {c["quarter_end"]: c["outcome"]["portfolio_realized_return"]
             for c in simulate.load_checkpoints(us_run)}
        quarters = sorted(set(tw) & set(us))
        xm_returns = []
        for q in quarters:
            r_tw, r_us = tw[q], us[q]
            r_xm = TW_WEIGHT * r_tw + US_WEIGHT * r_us
            xm_returns.append(r_xm)
            print(f"{arm:>10} {q:>12} {r_tw:>+9.2%} {r_us:>+9.2%} {r_xm:>+13.2%}")
        print(f"  ── {arm} 累積：XM(混合)={cum(xm_returns):+.2f}%／"
             f"純TW={cum(list(tw[q] for q in quarters)):+.2f}%／"
             f"純US={cum(list(us[q] for q in quarters)):+.2f}%\n")


if __name__ == "__main__":
    main()
