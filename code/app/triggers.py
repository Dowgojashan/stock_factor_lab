# -*- coding: utf-8 -*-
"""失效條件登記、比對、持續性規則（設計文件 §10 階段1/2、§7.4a）。

範圍（2026-09-17 初版）：**只做 M1-D**。理由：M1-D 是設計裡唯一「預測區可驅動
動作」的機制（§7.4「M1 的三個角色」），也是唯一有完整、已凍結門檻的機制
（X=1.86pp，見 §7.4a）。M2/M3/M4/M5/M7 屬於**診斷**（回答「為什麼」），依
`實戰開發追蹤.md` §3 的檔案分工屬於 `diagnose.py` 的範圍，不在這裡。

🔴 持續性規則（§7.4a 表：「2025Q1 短暫回落（不算誤報，維持觀察而非解除）」）：
採**雙門檻遲滯（hysteresis）**，不是單一門檻的開關：
  - 進場：累計偏離 > p90（X）⇒ TRIGGERED
  - 出場：只有累計偏離**跌破 p75** 才解除回 NONE；介於 p75~p90 之間維持
    現有狀態不降級（TRIGGERED 保持 TRIGGERED，不會因為一次小回落就解除）
  - 這樣「首次觸發後隔季小幅回落」不會被誤判成假警報解除，符合 §7.4a 記錄的
    真實案例（2024Q4 首次觸發 → 2025Q1 回落到 p75~p90 之間仍維持觀察 →
    2025Q2 起再次超過 p90 持續觸發）
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from app import monitor

X_P90 = 0.0186   # 已凍結，見 §7.4a／開發追蹤 D27
X_P75 = 0.012    # ~+1.2pp，設計文件 §7.4a 表列為「依分布內插」的建議 A4 門檻


@dataclasses.dataclass
class M1DCondition:
    """登記時點固定下來的 M1-D 判準——階段 1 產出，之後每季拿它比對。"""
    as_of: str
    registration_q1_weight: float
    p75: float
    p90: float


def register_m1d(mcap_wide: pd.DataFrame, as_of: str,
                 p75: float = X_P75, p90: float = X_P90) -> M1DCondition:
    """階段 1：期初登記。把登記時點的指數 Q1 集中度固定下來，
    之後每季比對「相對這個水準的累計變動」有沒有超過門檻。
    🔴 p75/p90 是外部凍結好的門檻（§11.7 程序），這裡不重新計算，只是登記。
    """
    env = monitor.environment_layer(mcap_wide, as_of)
    return M1DCondition(as_of=as_of, registration_q1_weight=env["q1_weight"],
                        p75=p75, p90=p90)


def evaluate_quarter(cond: M1DCondition, mcap_wide: pd.DataFrame, as_of: str,
                     prev_state: str) -> dict:
    """階段 2：每季比對。`prev_state` 是上一季結束時的狀態
    （"NONE"／"OBSERVING"／"TRIGGERED"），回傳這一季的新狀態與判斷依據。
    """
    env = monitor.environment_layer(mcap_wide, as_of)
    dev = env["q1_weight"] - cond.registration_q1_weight

    if dev > cond.p90:
        new_state = "TRIGGERED"
    elif dev > cond.p75:
        # 介於 p75~p90：如果上一季已經是 TRIGGERED，維持 TRIGGERED（遲滯，
        # 不因一次小回落解除）；否則升級/維持為 OBSERVING
        new_state = "TRIGGERED" if prev_state == "TRIGGERED" else "OBSERVING"
    else:
        new_state = "NONE"

    action = {"NONE": "A0", "OBSERVING": "A4", "TRIGGERED": "A5"}[new_state]
    return {
        "as_of": as_of, "q1_weight": env["q1_weight"],
        "registration_q1_weight": cond.registration_q1_weight,
        "cumulative_deviation": dev, "p75": cond.p75, "p90": cond.p90,
        "prev_state": prev_state, "state": new_state, "action": action,
    }


def run_quarterly_series(mcap_wide: pd.DataFrame, cond: M1DCondition,
                         quarter_ends: list[str]) -> pd.DataFrame:
    """依序跑過一串季度檢查點，維護跨季的持續性狀態（§7.4a 的「觀察期」語意，
    需要記得上一季狀態，不能每季獨立重算）。"""
    rows = []
    state = "NONE"
    for q in quarter_ends:
        r = evaluate_quarter(cond, mcap_wide, q, state)
        rows.append(r)
        state = r["state"]
    return pd.DataFrame(rows)


# ============================================================ Hot Segment（設計文件 §7.4b③，
# 跟 M1-D 平行、獨立觸發，不是誰包含誰——兩者可能同時觸發、只觸發一個、或都不觸發）
#
# 🔴 判斷方向跟 M1-D 相反：M1-D 是「偏離越高越異常」，Hot Segment 是「覆蓋率越低越異常」。
# 門檻用 2015-2023 歷史分布算出（見 `_analysis_outputs_applayer/hot_segment_coverage_test*.csv`，
# `coverage_count`／N=市場已校準值／X=10%／calib_2015_2023 這個切面），XM 尚未校準，
# 沿用時直接報錯，不可假設落在 TW/US 中間（跟 `monitor.hot_segment_layer()` 同一個立場）。

HOT_SEGMENT_P25 = {"TW": 0.3363, "US": 0.3039}  # 觀察門檻：低於此值 → OBSERVING
HOT_SEGMENT_P10 = {"TW": 0.3080, "US": 0.2595}  # 異常門檻：低於此值 → TRIGGERED


@dataclasses.dataclass
class HotSegmentCondition:
    """登記時點固定下來的 Hot Segment 門檻——跟 `M1DCondition` 平行，
    但這裡登記的是門檻本身（市場已校準的歷史分位數），不是登記時的覆蓋率水準
    （M1-D 比的是「相對登記時的累計偏離」，Hot Segment 比的是「相對歷史分布的絕對水位」，
    兩者的判斷邏輯本來就不同，不要誤以為要對齊成同一種比較方式）。"""
    market: str
    p25: float
    p10: float


def register_hot_segment(market: str) -> HotSegmentCondition:
    """階段 1：期初登記。門檻是市場已校準的歷史分位數（見上方常數），不需要
    當下的覆蓋率數字，純粹是選對市場對應的門檻。"""
    if market not in HOT_SEGMENT_P25:
        raise ValueError(f"Hot Segment 尚未替 {market} 校準門檻（目前只有 TW/US）")
    return HotSegmentCondition(market=market, p25=HOT_SEGMENT_P25[market], p10=HOT_SEGMENT_P10[market])


def evaluate_hot_segment_quarter(cond: HotSegmentCondition, coverage_count: float,
                                 prev_state: str) -> dict:
    """階段 2：每季比對。`coverage_count` 是 `monitor.hot_segment_layer()` 算出的
    市場層級覆蓋率（不是策略層級的 coverage_frac）。跟 M1-D 同一套雙門檻遲滯設計，
    但方向相反——低於門檻才算異常。`coverage_count` 若為 NaN（見
    `monitor.hot_segment_layer()` 的誠實回傳）直接視為 NONE 並在 `note` 標記，
    不可讓 NaN 悄悄比較出一個看似正常的結果。
    """
    if coverage_count is None or coverage_count != coverage_count:  # NaN 檢查
        # 🔴 2026-09-30 code review修正：原本這裡強制回傳"NONE"（等同A0，一切
        # 正常）——docstring明明寫「不可讓NaN悄悄比較出一個看似正常的結果」，
        # 但NONE/A0本身就是「看似正常」的結果，兩者矛盾。改成維持prev_state
        # 不變（資料算不出來≠市場恢復正常，尤其若上一季是TRIGGERED、這一季
        # 剛好碰到熱門股全缺市值資料，不該因此悄悄解除傾斜）。
        action = {"NONE": "A0", "OBSERVING": "A4", "TRIGGERED": "A5"}[prev_state]
        return {"coverage_count": coverage_count, "p25": cond.p25, "p10": cond.p10,
               "prev_state": prev_state, "state": prev_state, "action": action,
               "note": "coverage_count 無法計算（熱門股全部缺市值資料），"
                       "本季不判定、維持上一季狀態不變"}

    if coverage_count < cond.p10:
        new_state = "TRIGGERED"
    elif coverage_count < cond.p25:
        new_state = "TRIGGERED" if prev_state == "TRIGGERED" else "OBSERVING"
    else:
        new_state = "NONE"

    action = {"NONE": "A0", "OBSERVING": "A4", "TRIGGERED": "A5"}[new_state]
    return {
        "coverage_count": coverage_count, "p25": cond.p25, "p10": cond.p10,
        "prev_state": prev_state, "state": new_state, "action": action,
    }


# ============================================================ 回撤觸發（Drawdown Escalation，
# 2026-10-08新增，跟 M1-D／Hot Segment 平行、獨立判定——但**用途不同**，見下方
# `evaluate_drawdown_quarter()` docstring）

# 🔴🔴 跟 M1-D(X_P90/X_P75) 或 Hot Segment(HOT_SEGMENT_P25/P10) 不同：這兩組既有
# 門檻都是用真實歷史資料校準出來的凍結值（M1-D 用 2007-2023、Hot Segment 用
# 2015-2023，見各自上方的常數註解）。回撤目前沒有對應的校準資料可用，若直接
# 手動指定一個絕對數字（例如「回撤超過-15%」）等於憑印象發明一個沒查證過的
# 門檻，跟本專案的查證原則衝突。這裡改用**expanding window自我比較**（跟
# `monitor.expanding_percentile()`既有機制同一個精神）——拿「目前回撤相對這次
# 模擬自己過去已實現的回撤分布，落在第幾百分位」當判準，不需要外部基準，門檻
# 本身（0.75/0.90）只是跟M1-D/Hot Segment同一個雙門檻遲滯慣例的百分位切點，
# 不是憑空的新數字。⚠️ 代價：模擬季數少時（本系統正式模擬只有8季）百分位估計
# 統計上很薄——這跟本系統其餘「單一段OOS」的既有限制同一個性質，誠實揭露即可，
# 不是這裡獨有的新問題。
DRAWDOWN_P75 = 0.75
DRAWDOWN_P90 = 0.90


@dataclasses.dataclass
class DrawdownCondition:
    """跟 `M1DCondition`／`HotSegmentCondition` 平行，但這裡不登記任何水準
    （回撤定義上就是『相對歷史峰值的距離』，不需要另外登記基準點）。"""
    p75: float = DRAWDOWN_P75
    p90: float = DRAWDOWN_P90


def register_drawdown() -> DrawdownCondition:
    """階段 1：回撤觸發沒有需要登記的水準，純粹回傳門檻設定——保留跟
    `register_m1d()`/`register_hot_segment()` 一致的呼叫介面，方便 `simulate.py`
    統一處理三組觸發條件的初始化方式。"""
    return DrawdownCondition()


def evaluate_drawdown_quarter(cond: DrawdownCondition, dd_history, as_of: str,
                              current_dd: float, prev_state: str) -> dict:
    """階段 2：每季比對。`dd_history` 是這次模擬到目前為止（不含本季）的回撤序列
    （index=日期，value=當季回撤，<=0），`current_dd` 是這一季的回撤。

    🔴🔴 **這裡的回傳值只能驅動 A0／A4／A5（不改投組），不可驅動 A1／A2／
    W2c／CoverageTilt／RepSwap 這類投組變更動作**——`monitor.drawdown_state()`
    docstring已說明「目前回撤水位」本身視為狀態變數，但它的輸入
    （`outcome["portfolio_realized_return"]`逐季累積）仍是流量變數的產物，為了
    不踩到設計文件§7.0／§9.0「流量變數不可驅動投組變更」這條**實測驗證過**的
    硬規則（體制持續性r=-0.003，用流量變數的落差去推下一季的投組決策等於倒果
    為因），這裡刻意比照 Hot Segment 的 state→action 映射（只給 A0/A4/A5），
    呼叫端（`simulate.py`）不可以、也沒有被設計成可以把這個狀態接到
    `ActiveConfig` 的任何投組變更旗標上——跟 `last_decision_was_w2c`／
    `last_decision_was_coverage_tilt`／`last_decision_was_rep_swap` 不是同一類
    東西，這個狀態只能流進 facts 給決策 agent 當**第三個基準動作參考**
    （`facts_lean.drawdown_baseline_action_csv`），跟 m1d/hot_segment 的基準
    動作並列，不綁定任何特定可選動作。

    資料不足（`dd_history`為空或expanding_percentile回NaN）時維持prev_state不變
    （跟Hot Segment NaN處理同一個誠實立場：資料算不出來≠一切正常）。
    """
    from app import monitor
    if dd_history is None or len(dd_history) == 0:
        pct = float("nan")
    else:
        pct = monitor.expanding_percentile(dd_history.abs(), as_of, abs(current_dd))

    if pct != pct:  # NaN
        action = {"NONE": "A0", "OBSERVING": "A4", "TRIGGERED": "A5"}[prev_state]
        return {"current_dd": current_dd, "percentile": pct, "p75": cond.p75, "p90": cond.p90,
               "prev_state": prev_state, "state": prev_state, "action": action,
               "note": "回撤歷史觀測不足（expanding window仍在累積中），"
                       "本季不判定、維持上一季狀態不變"}

    if pct >= cond.p90:
        new_state = "TRIGGERED"
    elif pct >= cond.p75:
        new_state = "TRIGGERED" if prev_state == "TRIGGERED" else "OBSERVING"
    else:
        new_state = "NONE"

    action = {"NONE": "A0", "OBSERVING": "A4", "TRIGGERED": "A5"}[new_state]
    return {
        "current_dd": current_dd, "percentile": pct, "p75": cond.p75, "p90": cond.p90,
        "prev_state": prev_state, "state": new_state, "action": action,
    }
