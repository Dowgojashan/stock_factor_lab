# -*- coding: utf-8 -*-
"""M0~M7 診斷（設計文件 §7.4）。回答「為什麼」——跟 `triggers.py`（回答「要不要
警示」，目前只做 M1-D）分工不同。

🔴 範圍縮減（2026-09-17，實作時發現）：M2／M5／M7 在目前的即時系統量測方法下
**結構性測不到**，不是還沒做完：
  - **M2**（策略群集中度）：需要 HRP 樹的分群指派，跟本系統採用的輕量
    `resolve_strategy_holdings` 持股解析管線不是同一套資料來源，未接
  - **M5**（風格曝險漂移）：`factor_exposure_F1`（見 `ops/tools.py:566`）是算在
    **策略清單**（`strategy_uids`）上的分布，而 scheme E 的 walk-forward 設計裡
    策略清單在整個窗次（8 季）內固定不變，只有窗次邊界才會換。既然這個指標
    算在不會變的東西上，它跟登記時的 L1 距離在窗次內**永遠是 0**——不是資料
    不夠，是這個指標定義方式在本系統裡本來就測不到任何漂移
  - **M7**（候選池整體失效，判準「B_all 相對等權大盤的超額 < 歷史 p10」）：
    A9 已確認 B_all 在即時系統裡**就是**等權大盤本身（`walkforward_matrix.py:62`
    定義 B_all＝全宇宙等權，跟 `performance.measure()` 的等權基準是同一件事）——
    這代表這個判準在數學上**必然恆等於 0**，結構性測不到

本檔案實作 **M3、M4、M6、M0**——這四個在目前系統下是真正可計算、有意義的。
M1-D 在 `triggers.py`（前瞻區、可驅動動作）；M1-R（回顧區、已實現貢獻的基準鏈
拆解）尚未實作，需要另外的逐季基準鏈串接邏輯，留待後續。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# M3 歷史分布：windows 1-3 真實資料（`_prelim_a1_portfolio_noise.py` 的輸出，
# n=36 季度樣本，2015-2023，符合 §11.7 只用訓練期資料的規則）
_M3_HISTORY_PATH = (Path(__file__).resolve().parent.parent.parent
                    / "_analysis_outputs_applayer" / "a1_portfolio_noise_windows123.csv")

# M0／M6 績效層歷史分布：windows 1-3 真實逐季 excess_vs_ball
# （`_prelim_m0_performance_history.py` 的輸出，n=36，2015-2023）
_M0_HISTORY_PATH = (Path(__file__).resolve().parent.parent.parent
                    / "_analysis_outputs_applayer" / "m0_performance_history.csv")


def _m3_history() -> pd.Series:
    df = pd.read_csv(_M3_HISTORY_PATH)
    return df[~df["is_baseline"]]["n_stocks"]


def _m0_history() -> pd.Series:
    df = pd.read_csv(_M0_HISTORY_PATH)
    return df["excess_vs_ball"].dropna()


def is_performance_below_p10(current_excess_vs_ball: float) -> dict:
    """給 `run_diagnosis()` 的 `performance_below_p10` 用：本季 excess_vs_ball
    是否低於 windows 1-3 的歷史 p10（同一個流量變數，跟 M4 角度不同——M4 看
    連續 2 期 <0 的持續性，這裡看單一期本身是不是歷史級的極端值）。"""
    hist = _m0_history()
    p10 = hist.quantile(0.10)
    return {"current": current_excess_vs_ball, "historical_p10": float(p10),
           "below_p10": bool(current_excess_vs_ball < p10)}


def diagnose_m3(n_unique_stocks: int) -> dict:
    """選股廣度異常：`n_unique_stocks` 超出歷史 p10／p90（狀態警示區，無 pp 貢獻）。"""
    hist = _m3_history()
    p10, p90 = hist.quantile(0.10), hist.quantile(0.90)
    triggered = n_unique_stocks < p10 or n_unique_stocks > p90
    return {
        "mechanism": "M3", "name": "選股廣度異常",
        "n_unique_stocks": n_unique_stocks, "historical_p10": float(p10), "historical_p90": float(p90),
        "triggered": bool(triggered), "action": "A4或A5" if triggered else "A0",
        "region": "B",  # 狀態警示區，無 pp 貢獻
    }


def diagnose_m4(excess_vs_ball_history: list[float]) -> dict:
    """選股規則相對候選池失效：相對 B_all（＝等權大盤，見 A9）超額連續 2 期 < 0。
    `excess_vs_ball_history`：依時間順序排列，最新一期在最後。"""
    if len(excess_vs_ball_history) < 2:
        return {"mechanism": "M4", "name": "選股規則相對候選池失效",
               "triggered": False, "reason": "樣本不足（少於 2 期）",
               "action": "A0", "region": "A"}
    last2 = excess_vs_ball_history[-2:]
    triggered = all(e < 0 for e in last2)
    return {
        "mechanism": "M4", "name": "選股規則相對候選池失效",
        "last_2_excess": last2, "triggered": bool(triggered),
        "action": "A5" if triggered else "A0", "region": "A",
    }


def diagnose_m6_m0(performance_below_p10: bool, any_mechanism_triggered: bool) -> dict:
    """M6（僅為隨機波動）／M0（無法歸因）：兩者互斥，且都是「以上皆非」的出口。
    - 績效沒有異常（未低於歷史 p10）且沒有機制觸發 ⇒ M6（正常，什麼都沒發生）
    - 績效確實異常，但沒有任何已知機制的證據判準成立 ⇒ M0（誠實回報無法歸因，
      **不可硬塞一個最像的機制**——這是本設計最重要的一條規則，見 §7.4「M0 是
      本清單最重要的一條」）
    """
    if not performance_below_p10 and not any_mechanism_triggered:
        return {"mechanism": "M6", "name": "僅為隨機波動", "action": "A0"}
    if performance_below_p10 and not any_mechanism_triggered:
        return {"mechanism": "M0", "name": "無法歸因",
               "note": "績效層確實異常，但所有已知機制的證據判準皆不成立——"
                       "誠實回報，不猜一個最像的機制", "action": "A5"}
    return {"mechanism": None, "name": "已有其他機制觸發，M6/M0 不適用"}


def run_diagnosis(*, n_unique_stocks: int, excess_vs_ball_history: list[float]) -> dict:
    """組出完整的 M0~M7 判定（本階段涵蓋 M3/M4/M6/M0；M1-D 另見 triggers.py）。
    回傳格式對齊 §7.4「固定格式」：A 區（可歸因，M1/M4/M7）／B 區（狀態警示，M2/M3/M5）。
    🔴 本階段 A 區只有 M4（M1-R／M7 尚未實作，見檔案開頭說明）；B 區只有 M3
    （M2／M5 結構性測不到，見檔案開頭說明）。

    `performance_below_p10`（M0/M6 用）內部用 `excess_vs_ball_history[-1]`（本季
    的值）自動算，不用呼叫端另外傳——避免 M4 跟 M0/M6 各自被喂到不一致的「本季」
    數字（之前的版本要求呼叫端另外傳一個布林值，容易兜錯）。
    """
    m3 = diagnose_m3(n_unique_stocks)
    m4 = diagnose_m4(excess_vs_ball_history)
    any_triggered = m3["triggered"] or m4["triggered"]

    perf = None
    if excess_vs_ball_history:
        perf = is_performance_below_p10(excess_vs_ball_history[-1])
    m6_m0 = diagnose_m6_m0(bool(perf and perf["below_p10"]), any_triggered)
    if perf is not None:
        m6_m0["performance_check"] = perf

    return {
        "region_a_attributable": {"M4": m4},
        "region_b_state_warning": {"M3": m3},
        "fallback": m6_m0,
        "not_available": {
            "M2": "需要 HRP 樹分群指派，跟輕量持股解析管線不同源，未接",
            "M5": "factor_exposure_F1 算在固定不變的策略清單上，窗次內 L1 距離恆為 0，結構性測不到",
            "M7": "B_all 在即時系統裡即等權大盤本身，「B_all 相對等權大盤超額」數學上恆為 0，結構性測不到",
        },
    }
