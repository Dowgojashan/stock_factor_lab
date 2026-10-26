# -*- coding: utf-8 -*-
"""2026-10-27：台美股W2c單股上限敏感度分析（window4, 2024-2025, exclude_v1
候選池），不呼叫LLM、不跑L2——跟`_check_w2c_basket_offline.py`同一套離線
反事實手法：每季**強制**套用W2c市值加權（忽略M1-D真實觸發狀態），只改變
`cap_and_redistribute()`的cap參數，看報酬/MDD隨上限鬆緊怎麼變。

背景：台股這個分析2026-10-05~06做過一次（`multi_window_cap_sweep_forced.csv`，
原始產生腳本已不在repo），當時只測了TW、用的是舊的日頻NAV MDD算法。這支腳本
①補上美股版本（美股現在真的有一次真實觸發，window4本身更有意義）②重新驗證
TW window4這個子集的數字在候選池/M2/Hot Segment都定案之後是否仍然一致
③MDD改用跟本次10/27報告其餘圖表一致的季頻取樣算法（9個季末點，不是日頻
NAV），兩者量級不直接可比，差異已在輸出裡註明。

用法：cd code; PYTHONIOENCODING=utf-8 python -m app._cap_sweep_window4
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from app import monitor, simulate  # noqa: E402
from app import weights as weights_mod  # noqa: E402
from app._check_rolling_production_simulation import build_rolling_inputs  # noqa: E402

APPLAYER = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
CAPS = [0.07, 0.10, 0.15, 0.20, 0.30, 0.40]

MARKETS = {
    "TW": APPLAYER / "anchored_w4_openSec_boost_exclude_v1.parquet",
    "US": APPLAYER / "anchored_w4_openSec_balanced_exclude_v1.parquet",
}


def nav_drawdown(rets: list[float]) -> tuple[float, float]:
    nav = [1.0]
    for r in rets:
        nav.append(nav[-1] * (1 + r))
    peak = nav[0]
    mdd = 0.0
    for v in nav:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    return (nav[-1] - 1) * 100, mdd * 100


def sweep_market(tree_key: str, picks_path: Path) -> pd.DataFrame:
    print(f"\n=== {tree_key}：載入window4 exclude_v1候選池 ===")
    inputs = build_rolling_inputs(model="unused", api_key="unused",
                                  picks_path=picks_path, tree_key=tree_key,
                                  scheme="E", window_no=4)
    uids = inputs.uids_by_allocation["equal"]

    periods = [(simulate.REGISTRATION_DATE, simulate.QUARTER_ENDS[0])]
    periods += list(zip(simulate.QUARTER_ENDS[:-1], simulate.QUARTER_ENDS[1:]))

    rows = []
    for cap in CAPS:
        rets = []
        for as_of, end in periods:
            w0 = simulate.resolve_weights(inputs.md, inputs.idx, uids, as_of)
            mcap_row = monitor.asof_row(inputs.mcap_wide, as_of)
            raw = weights_mod.tilt_weights(w0, mcap_row, 0.0)
            w_capped = weights_mod.cap_and_redistribute(raw, cap=cap)
            r = monitor.outcome_layer(inputs.md_map, w_capped, as_of, end)["portfolio_realized_return"]
            rets.append(r)
        cum, mdd = nav_drawdown(rets)
        rows.append(dict(market=tree_key, cap=f"{cap:.0%}", cum_pct=cum, mdd_pct=mdd))
        print(f"  cap={cap:.0%}  累積={cum:+.2f}%  MDD(季頻取樣)={mdd:+.2f}%")

    return pd.DataFrame(rows)


def main():
    all_rows = []
    for tree_key, picks_path in MARKETS.items():
        df = sweep_market(tree_key, picks_path)
        all_rows.append(df)

    out = pd.concat(all_rows, ignore_index=True)
    out_path = APPLAYER / "cap_sweep_window4_2026-10-27.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}")


if __name__ == "__main__":
    main()
