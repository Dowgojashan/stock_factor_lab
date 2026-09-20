# -*- coding: utf-8 -*-
"""C2 單一股票上限調整的驗證（2026-09-19，使用者要求，跟 W2c 同一等級的
四輪查證：合規性／交易成本／敏感度／季中漂移）。

背景：使用者發現 `weights.C2_CAP = 0.08` 是本專案自訂值（900格歷史反推），
**不是法規要求**——真實法規遠比這寬鬆（台灣舊制10%、新制25%「台積電條款」，
2026-04-24生效）。且 §3.2 已測出等權設計對大型股的低配是12年以上的結構性
常數（2013年至今−65%~−72%，非近期異象），設計文件 §3.3 的結論是「這是一次性
政策決定，不是每季監控決定」——這裡驗證的正是「調整這個一次性政策」的效果。

跟 D50（W2c 驗證）用同一套機制（α=0 完全市值加權 ＋ cap-and-redistribute），
**唯一變數是 cap 值**：
    7%   現行 W2c 目標值（C2=8%自訂上限，留1pp緩衝）
    10%  台灣舊法規
    23%  台灣新法規25%，留2pp緩衝（比照7%相對8%的留buffer邏輯）
    25%  台灣新法規（金管證投字第1150381494號令），無緩衝

四輪查證（每輪都先假設結果是假的，D50 的既有紀律）：
    ①合規性：cap-and-redistribute 後是否確實守住各自的 cap
    ②真實交易成本：兩條路徑（基準線／條件式）都各自扣真實換手成本
    ③季中漂移：季初重平衡後的權重，季底（未再平衡）是否又漂回超過**該 cap
       對應的真實法規硬上限**——7%/10%這兩檔的硬上限仍是本專案自訂的8%或
       舊制10%，有明確緩衝；25%這檔已經是法規硬上限本身，漂移就是真正的
       合規風險，要特別檢查
    ④額外：跟真實市值加權大盤（taiex_tr）比較——這是上一輪對話裡使用者
       關心的「跟大盤走向比」，原始 W2c 驗證沒做這個比較，這裡補上
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
from app._prelim_a5_w2_prevalidation import resolve_w0, tilt_weights  # noqa: E402
from app.performance import measure  # noqa: E402

FEE_RATIO = 1.425 / 1000
TAX_RATIO = 3 / 1000
ROUNDTRIP_COST = 2 * FEE_RATIO + TAX_RATIO

# (標籤, 目標cap, 該檔對應的真實硬上限——用來判斷漂移是否構成真正違規)
CAP_SCENARIOS = [
    ("7%（現行W2c，自訂C2=8%留1pp緩衝）", 0.07, 0.08),
    ("10%（台灣舊法規）", 0.10, 0.10),
    ("23%（台灣新法規25%留2pp緩衝）", 0.23, 0.25),
    ("25%（台灣新法規，無緩衝）", 0.25, 0.25),
]

OUT_PATH = (Path(__file__).resolve().parent.parent.parent
           / "_analysis_outputs_applayer" / "c2_cap_raise_scrutiny.csv")


def cap_and_redistribute(weights: dict[str, float], cap: float, max_iter: int = 50) -> dict[str, float]:
    w = dict(weights)
    for _ in range(max_iter):
        over = {s: v for s, v in w.items() if v > cap}
        if not over:
            break
        excess = sum(v - cap for v in over.values())
        for s in over:
            w[s] = cap
        under = {s: v for s, v in w.items() if v < cap}
        under_total = sum(under.values())
        if under_total <= 0:
            break
        for s in under:
            w[s] += excess * (under[s] / under_total)
    return w


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

    tx = pd.read_sql("SELECT date, close FROM taiex_tr ORDER BY date", conn)
    tx["date"] = pd.to_datetime(tx["date"])
    cap_weight_series = tx.set_index("date")["close"].astype(float).sort_index()

    def cap_weight_bench(as_of: str, end: str) -> float:
        ts = cap_weight_series.index
        d0 = ts[ts <= pd.Timestamp(as_of)].max()
        d1 = ts[ts <= pd.Timestamp(end)].max()
        return float(cap_weight_series.loc[d1] / cap_weight_series.loc[d0] - 1.0)

    checkpoints = [REGISTRATION_DATE] + ALL_QUARTERS
    state = "NONE"
    quarter_info = []
    for i in range(len(checkpoints) - 1):
        as_of, end = checkpoints[i], checkpoints[i + 1]
        m1d = triggers.evaluate_quarter(cond, mcap_wide, end, state)
        state = m1d["state"]
        w0 = resolve_w0(md, idx, uids, as_of)
        mcap_row_asof = monitor.asof_row(mcap_wide, as_of)
        mcap_row_end = monitor.asof_row(mcap_wide, end)
        quarter_info.append({"as_of": as_of, "end": end, "state": state,
                             "w0": w0, "mcap_row_asof": mcap_row_asof,
                             "mcap_row_end": mcap_row_end})
        print(f"   {as_of} -> {end}  M1-D={state}")

    def gross_return(weights: dict[str, float], as_of: str, end: str) -> float:
        return measure(md_map, weights, as_of, end)["portfolio_realized_return"]

    def cum(rets: dict[str, float], quarters: set[str]) -> float:
        vals = [rets[q] for q in ALL_QUARTERS if q in quarters]
        return float(pd.Series([1 + r for r in vals]).prod() - 1)

    # ---- 基準線（α=1，永遠不傾斜＝現況）----
    reg_w0 = resolve_w0(md, idx, uids, REGISTRATION_DATE)
    reg_w1 = tilt_weights(reg_w0, monitor.asof_row(mcap_wide, REGISTRATION_DATE), 1.0)
    base_gross_rets, base_net_rets = {}, {}
    prev_base_w = reg_w1
    for q in quarter_info:
        target = tilt_weights(q["w0"], q["mcap_row_asof"], 1.0)
        g = gross_return(target, q["as_of"], q["end"])
        to = turnover(prev_base_w, target)
        base_gross_rets[q["end"]] = g
        base_net_rets[q["end"]] = g - to * ROUNDTRIP_COST
        prev_base_w = target

    market_rets = {q["end"]: cap_weight_bench(q["as_of"], q["end"]) for q in quarter_info}

    calib_base_gross, fwd_base_gross = cum(base_gross_rets, CALIBRATION_QUARTERS), cum(base_gross_rets, FORWARD_QUARTERS)
    calib_base_net, fwd_base_net = cum(base_net_rets, CALIBRATION_QUARTERS), cum(base_net_rets, FORWARD_QUARTERS)
    calib_mkt, fwd_mkt = cum(market_rets, CALIBRATION_QUARTERS), cum(market_rets, FORWARD_QUARTERS)

    print(f"\n=== 基準線（現況，α=1）===")
    print(f"  校準期：毛={calib_base_gross:+.2%} 淨={calib_base_net:+.2%}  對大盤超額(淨)={calib_base_net-calib_mkt:+.2%}")
    print(f"  前瞻期：毛={fwd_base_gross:+.2%} 淨={fwd_base_net:+.2%}  對大盤超額(淨)={fwd_base_net-fwd_mkt:+.2%}")
    print(f"  （真實大盤市值加權累積：校準期={calib_mkt:+.2%}  前瞻期={fwd_mkt:+.2%}）")

    rows = []
    for label, target_cap, hard_limit in CAP_SCENARIOS:
        print(f"\n=== {label}（目標cap={target_cap:.0%}，對應硬上限={hard_limit:.0%}）===")

        # ①②：合規性 + 交易成本
        cond_gross_rets, cond_net_rets = {}, {}
        prev_cond_w = reg_w1
        breach_after_cap = 0
        for q in quarter_info:
            if q["state"] == "NONE":
                target = tilt_weights(q["w0"], q["mcap_row_asof"], 1.0)
            else:
                raw = tilt_weights(q["w0"], q["mcap_row_asof"], 0.0)
                target = cap_and_redistribute(raw, cap=target_cap)
                if max(target.values()) > target_cap + 1e-6:
                    breach_after_cap += 1
            g = gross_return(target, q["as_of"], q["end"])
            to = turnover(prev_cond_w, target)
            cond_gross_rets[q["end"]] = g
            cond_net_rets[q["end"]] = g - to * ROUNDTRIP_COST
            prev_cond_w = target

        assert breach_after_cap == 0, f"①合規性查證失敗：{label} 重平衡當下仍有 {breach_after_cap} 季超過目標cap"
        print(f"  ①合規性：7個觸發季重平衡當下全部守住 {target_cap:.0%} 目標 ✅")

        calib_cond_gross, fwd_cond_gross = cum(cond_gross_rets, CALIBRATION_QUARTERS), cum(cond_gross_rets, FORWARD_QUARTERS)
        calib_cond_net, fwd_cond_net = cum(cond_net_rets, CALIBRATION_QUARTERS), cum(cond_net_rets, FORWARD_QUARTERS)
        print(f"  ②淨改善（扣真實交易成本）：校準期={calib_cond_net-calib_base_net:+.2%}"
             f"  前瞻期={fwd_cond_net-fwd_base_net:+.2%}")
        print(f"      對大盤超額（淨）：校準期={calib_cond_net-calib_mkt:+.2%}"
             f"  前瞻期={fwd_cond_net-fwd_mkt:+.2%}"
             f"（基準線是 {calib_base_net-calib_mkt:+.2%} / {fwd_base_net-fwd_mkt:+.2%}）")

        # ③季中漂移：用該 cap 對應的真實硬上限判斷是否構成真正違規
        drift_breaches = []
        for q in quarter_info:
            if q["state"] == "NONE":
                continue
            raw = tilt_weights(q["w0"], q["mcap_row_asof"], 0.0)
            capped = cap_and_redistribute(raw, cap=target_cap)
            over_raw = {s: v for s, v in raw.items() if v > target_cap}
            if not over_raw:
                continue
            implied_shares = {s: capped[s] / q["mcap_row_asof"].get(s, float("nan")) for s in capped}
            end_values = {s: implied_shares[s] * q["mcap_row_end"].get(s, float("nan")) for s in capped}
            total_end = sum(v for v in end_values.values() if pd.notna(v))
            end_weights = {s: v / total_end for s, v in end_values.items() if pd.notna(v)}
            for s in over_raw:
                w_end = end_weights.get(s, float("nan"))
                if pd.notna(w_end) and w_end > hard_limit:
                    drift_breaches.append((q["end"], s, w_end))
        if drift_breaches:
            print(f"  ③🔴 季中漂移：{len(drift_breaches)} 個案例季底漂回超過真實硬上限 {hard_limit:.0%}：")
            for e, s, w in drift_breaches:
                print(f"      {e}｜{s}：季底權重 {w:.2%}（超過硬上限 {hard_limit:.0%}）")
        else:
            print(f"  ③季中漂移：全部案例季底仍在真實硬上限 {hard_limit:.0%} 以下 ✅")

        rows.append({
            "label": label, "target_cap": target_cap, "hard_limit": hard_limit,
            "calib_improvement_net": calib_cond_net - calib_base_net,
            "forward_improvement_net": fwd_cond_net - fwd_base_net,
            "calib_excess_vs_market_net": calib_cond_net - calib_mkt,
            "forward_excess_vs_market_net": fwd_cond_net - fwd_mkt,
            "n_drift_breaches": len(drift_breaches),
        })

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {OUT_PATH}")

    print("\n=== 總表：各 cap 情境 vs 基準線 vs 真實大盤 ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
