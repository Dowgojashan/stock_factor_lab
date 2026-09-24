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
