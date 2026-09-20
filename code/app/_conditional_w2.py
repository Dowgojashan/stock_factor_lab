# -*- coding: utf-8 -*-
"""條件式 W2（2026-09-18，使用者要求）：「M1-D 觸發時才啟動」的市值傾斜機制。

背景：原始 W2（永遠開啟的市值傾斜）已於 window 1-3 前置驗證失敗棄用（D29）——
但那是測試「永遠開著」的版本，從沒測過「只在 M1-D 觸發時才啟動」的版本。
window 1-3 裡 M1-D 幾乎從未觸發，所以條件式版本理論上不會拖累正常年份。

路徑 A（查 2000-2026 台股歷史有沒有獨立先例）已查證失敗（09-18）：唯一超過
2024-2025 幅度的兩段都是同一事件本身，沒有獨立樣本。使用者決定：改用
2024-2025 校準傾斜強度 α，用 2026（資料到 07-24，可以拿到完整 Q1/Q2 兩個
乾淨季度）做真正的前瞻驗證——這是誠實的「用測試期校準、用真前瞻期驗證」，
不是重複踩 D29 的「拿答案卷對答案」問題，因為要校準的只有「觸發後傾斜多少」
這一個參數，「什麼時候該觸發」（X=1.86pp 門檻）本來就是用 2007-2023 資料
獨立訂的，沒有碰過 2024-2025。

機制定義：沿用 `_prelim_a5_w2_prevalidation.py` 的 tilt_weights()
（w(s) ∝ w0(s)^α × mktcap(s)^(1-α)），差別是每一季用哪個 α 由當季的
M1-D state 決定：
    state == "NONE"                  -> α = 1.0（不傾斜，維持現況）
    state in ("OBSERVING","TRIGGERED") -> α = 候選值（本腳本逐一測試）
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
from app._prelim_a5_w2_prevalidation import resolve_w0, tilt_weights  # noqa: E402
from app.performance import measure  # noqa: E402

OUT_PATH = (Path(__file__).resolve().parent.parent.parent
           / "_analysis_outputs_applayer" / "conditional_w2.csv")

CANDIDATE_ALPHAS = [0.75, 0.5, 0.25, 0.0]   # 觸發時要測的候選傾斜強度
REGISTRATION_DATE = "2023-12-31"

# 2024Q1 ~ 2026Q2：前 8 季是「校準期」（2024-2025，已知會觸發），
# 後 2 季是「前瞻驗證期」（2026 H1，資料到 2026-07-24，Q1/Q2 皆完整）
ALL_QUARTERS = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
                "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31",
                "2026-03-31", "2026-06-30"]
CALIBRATION_QUARTERS = set(ALL_QUARTERS[:8])
FORWARD_QUARTERS = set(ALL_QUARTERS[8:])


def get_window4_members() -> list[str]:
    m = pd.read_parquet(Path(__file__).resolve().parent.parent.parent
                        / "_analysis_outputs_robustness" / "walkforward_members.parquet")
    key = dict(tree_key="TW", scheme="E", window_no=4, k_mode="silhouette_is",
              ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    assert len(sub) == 1
    return list(sub.iloc[0]["members"])


def main():
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    print(">> 載入 TW MarketData ...")
    md = MarketData("TW")
    md_map = {"TW": md}
    uids = get_window4_members()
    print(f"   window 4 成員數：{len(uids)}")

    db = Database("TW")
    conn = db.create_connection()
    mcap_wide = monitor.fetch_mcap_wide(conn, start="2007-01-01")

    tx = pd.read_sql("SELECT date, close FROM taiex_tr ORDER BY date", conn)
    tx["date"] = pd.to_datetime(tx["date"])
    cap_weight_series = tx.set_index("date")["close"].astype(float).sort_index()

    cond = triggers.register_m1d(mcap_wide, REGISTRATION_DATE)

    # 逐季算 M1-D 狀態（延續 triggers.py 的遲滯規則），並記錄每季的
    # w0（各策略等權彙總的基準持股）與市值列，供後面各 α 重複使用
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
        print(f"   {as_of} -> {end}  M1-D={state}")

    def realized_return(weights: dict[str, float], as_of: str, end: str) -> float:
        res = measure(md_map, weights, as_of, end)
        return res["portfolio_realized_return"]

    def cap_weight_bench(as_of: str, end: str) -> float:
        ts = cap_weight_series.index
        d0 = ts[ts <= pd.Timestamp(as_of)].max()
        d1 = ts[ts <= pd.Timestamp(end)].max()
        return float(cap_weight_series.loc[d1] / cap_weight_series.loc[d0] - 1.0)

    # 基準線（永遠 α=1，等於現況實際發生的事）——每季都算一次，供比較
    baseline_rets = {}
    cap_rets = {}
    for q in quarter_info:
        w1 = tilt_weights(q["w0"], q["mcap_row"], 1.0)
        baseline_rets[q["end"]] = realized_return(w1, q["as_of"], q["end"])
        cap_rets[q["end"]] = cap_weight_bench(q["as_of"], q["end"])

    print("\n=== 基準線（永遠不傾斜，α=1，等於現況）逐季報酬 vs 市值加權大盤 ===")
    for q in quarter_info:
        e = q["end"]
        tag = "校準期" if e in CALIBRATION_QUARTERS else "前瞻驗證期"
        print(f"  {e} [{tag}] state={q['state']:<10s} baseline={baseline_rets[e]:+.4%}"
             f"  cap_weight_bench={cap_rets[e]:+.4%}  gap={baseline_rets[e]-cap_rets[e]:+.4%}")

    # 對每個候選 α：只在 state != NONE 的季度換成該 α，其餘沿用 α=1 的報酬
    rows = []
    for alpha in CANDIDATE_ALPHAS:
        cond_rets = {}
        for q in quarter_info:
            if q["state"] == "NONE":
                cond_rets[q["end"]] = baseline_rets[q["end"]]
            else:
                w = tilt_weights(q["w0"], q["mcap_row"], alpha)
                cond_rets[q["end"]] = realized_return(w, q["as_of"], q["end"])

        def cum(rets: dict[str, float], quarters: set[str]) -> float:
            vals = [rets[q] for q in ALL_QUARTERS if q in quarters]
            return float(pd.Series([1 + r for r in vals]).prod() - 1)

        calib_cond = cum(cond_rets, CALIBRATION_QUARTERS)
        calib_base = cum(baseline_rets, CALIBRATION_QUARTERS)
        fwd_cond = cum(cond_rets, FORWARD_QUARTERS)
        fwd_base = cum(baseline_rets, FORWARD_QUARTERS)

        rows.append({
            "alpha_when_triggered": alpha,
            "calib_conditional_cum": calib_cond, "calib_baseline_cum": calib_base,
            "calib_improvement": calib_cond - calib_base,
            "forward_conditional_cum": fwd_cond, "forward_baseline_cum": fwd_base,
            "forward_improvement": fwd_cond - fwd_base,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("\n=== 校準期（2024-2025，8季累計）：各候選 α 相對現況（α=1）基準線的改善 ===")
    for _, r in df.iterrows():
        print(f"  α={r['alpha_when_triggered']:.2f}（觸發時）：條件式累計={r['calib_conditional_cum']:+.2%}"
             f"  現況累計={r['calib_baseline_cum']:+.2%}  改善={r['calib_improvement']:+.2%}")

    print("\n=== 前瞻驗證期（2026 H1，Q1+Q2，未參與校準）：用校準期表現最好的 α 測 ===")
    best = df.loc[df["calib_improvement"].idxmax()]
    print(f"校準期表現最好的 α = {best['alpha_when_triggered']}"
         f"（校準期改善 {best['calib_improvement']:+.2%}）")
    print(f"用這個 α 在 2026 H1（真前瞻，未參與校準）：條件式累計={best['forward_conditional_cum']:+.2%}"
         f"  現況累計={best['forward_baseline_cum']:+.2%}  改善={best['forward_improvement']:+.2%}")

    print(f"\n寫入 {OUT_PATH}")


if __name__ == "__main__":
    main()
