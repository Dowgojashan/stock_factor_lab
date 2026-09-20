# -*- coding: utf-8 -*-
"""幫 α=0＋cap-and-redistribute（目前找到最好、且合規的條件式W2版本）建模真實
交易成本，重新驗證改善是否經得起考驗（2026-09-18，使用者要求）。

交易成本沿用本專案已查證過的真實台股成本（見 `_prelim_k_anchor_full45_netcost.py`
／`_prelim_k_anchor_turnover_cost.py`）：
    FEE_RATIO = 1.425/1000（買賣手續費，雙邊都要付）
    TAX_RATIO = 3/1000（證交稅，只有賣出那一邊要付）
    ROUNDTRIP_COST = 2*FEE_RATIO + TAX_RATIO = 0.585%（換手一單位的成本）

🔴 關鍵：基準線（現況、α=1）本身也有真實換手成本——每季「快時鐘重解持股」
本身就會換股，不是條件式W2獨有的成本。之前所有分析（含 §11.5 限制8）都沒有
對任何版本扣過這筆成本。這裡兩條路徑（基準線／條件式W2）都各自扣自己的真實
換手成本，才是公平比較「加這個機制的淨效益」，不是只扣條件式那邊、放過基準線。

換手率定義：跟 `monitor.process_layer()` 的 `continuity_vs_prev` 同一套
（turnover = 1 - sum(min(w_new, w_old))），前後兩期都是同一條路徑自己的權重
（不是每季重新比較到基準線），才是真實部位變動的量。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from database import Database  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402

from app import monitor, triggers  # noqa: E402
from app._conditional_w2 import (ALL_QUARTERS, CALIBRATION_QUARTERS,  # noqa: E402
                                 FORWARD_QUARTERS, REGISTRATION_DATE,
                                 get_window4_members)
from app._conditional_w2_capped import cap_and_redistribute  # noqa: E402
from app._prelim_a5_w2_prevalidation import resolve_w0, tilt_weights  # noqa: E402
from app.performance import measure  # noqa: E402

FEE_RATIO = 1.425 / 1000
TAX_RATIO = 3 / 1000
ROUNDTRIP_COST = 2 * FEE_RATIO + TAX_RATIO


def turnover(w_old: dict[str, float], w_new: dict[str, float]) -> float:
    keys = set(w_old) | set(w_new)
    overlap = sum(min(w_old.get(s, 0.0), w_new.get(s, 0.0)) for s in keys)
    return 1.0 - overlap


def main():
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    print(">> 載入 TW MarketData ...")
    md = MarketData("TW")
    md_map = {"TW": md}
    uids = get_window4_members()

    db = Database("TW")
    conn = db.create_connection()
    mcap_wide = monitor.fetch_mcap_wide(conn, start="2007-01-01")
    cond = triggers.register_m1d(mcap_wide, REGISTRATION_DATE)

    checkpoints = [REGISTRATION_DATE] + ALL_QUARTERS
    state = "NONE"
    quarter_info = []
    for i in range(len(checkpoints) - 1):
        as_of, end = checkpoints[i], checkpoints[i + 1]
        m1d = triggers.evaluate_quarter(cond, mcap_wide, end, state)
        state = m1d["state"]
        w0 = resolve_w0(md, idx, uids, as_of)
        mcap_row = monitor.asof_row(mcap_wide, as_of)
        quarter_info.append({"as_of": as_of, "end": end, "state": state,
                             "w0": w0, "mcap_row": mcap_row})

    def gross_return(weights: dict[str, float], as_of: str, end: str) -> float:
        return measure(md_map, weights, as_of, end)["portfolio_realized_return"]

    # 起點：登記時點的權重（兩條路徑此時完全相同，w0 at registration）
    reg_w0 = resolve_w0(md, idx, uids, REGISTRATION_DATE)
    reg_w1 = tilt_weights(reg_w0, monitor.asof_row(mcap_wide, REGISTRATION_DATE), 1.0)

    prev_base_w = reg_w1
    prev_cond_w = reg_w1

    rows = []
    for q in quarter_info:
        # 基準線：永遠 α=1
        base_target = tilt_weights(q["w0"], q["mcap_row"], 1.0)
        base_gross = gross_return(base_target, q["as_of"], q["end"])
        base_to = turnover(prev_base_w, base_target)
        base_net = base_gross - base_to * ROUNDTRIP_COST

        # 條件式 W2：NONE 時等於基準線；觸發時 α=0 + cap-and-redistribute
        if q["state"] == "NONE":
            cond_target = base_target
        else:
            raw = tilt_weights(q["w0"], q["mcap_row"], 0.0)
            cond_target = cap_and_redistribute(raw)
        cond_gross = gross_return(cond_target, q["as_of"], q["end"])
        cond_to = turnover(prev_cond_w, cond_target)
        cond_net = cond_gross - cond_to * ROUNDTRIP_COST

        rows.append({"end": q["end"], "state": q["state"],
                    "base_gross": base_gross, "base_turnover": base_to, "base_net": base_net,
                    "cond_gross": cond_gross, "cond_turnover": cond_to, "cond_net": cond_net})
        prev_base_w, prev_cond_w = base_target, cond_target

    df = pd.DataFrame(rows)
    print("\n=== 逐季：毛報酬／換手率／淨報酬（基準線 vs 條件式W2）===")
    for _, r in df.iterrows():
        print(f"  {r['end']} [{r['state']:<10s}] 基準:毛={r['base_gross']:+.4%} 換手={r['base_turnover']:.1%} 淨={r['base_net']:+.4%}"
             f"  ｜ 條件式:毛={r['cond_gross']:+.4%} 換手={r['cond_turnover']:.1%} 淨={r['cond_net']:+.4%}")

    def cum(col: str, quarters: set[str]) -> float:
        vals = [df.loc[df.end == q, col].iloc[0] for q in ALL_QUARTERS if q in quarters]
        return float(pd.Series([1 + v for v in vals]).prod() - 1)

    print("\n=== 扣真實交易成本後的累計改善 ===")
    for label, quarters in [("校準期(2024-2025)", CALIBRATION_QUARTERS), ("前瞻驗證期(2026H1)", FORWARD_QUARTERS)]:
        base_gross_cum = cum("base_gross", quarters)
        cond_gross_cum = cum("cond_gross", quarters)
        base_net_cum = cum("base_net", quarters)
        cond_net_cum = cum("cond_net", quarters)
        print(f"\n{label}：")
        print(f"  毛報酬改善（不扣成本，原本的結論）：{cond_gross_cum - base_gross_cum:+.2%}")
        print(f"  淨報酬改善（扣真實換手成本後）：{cond_net_cum - base_net_cum:+.2%}")
        print(f"  （基準線累計：毛={base_gross_cum:+.2%} 淨={base_net_cum:+.2%}｜"
             f"條件式累計：毛={cond_gross_cum:+.2%} 淨={cond_net_cum:+.2%}）")

    out = (Path(__file__).resolve().parent.parent.parent
          / "_analysis_outputs_applayer" / "conditional_w2_netcost.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out}")


if __name__ == "__main__":
    main()
