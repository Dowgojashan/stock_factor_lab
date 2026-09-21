# -*- coding: utf-8 -*-
"""M0~M7 診斷（設計文件 §7.4）。回答「為什麼」——跟 `triggers.py`（回答「要不要
警示」，目前只做 M1-D）分工不同。

🔴 範圍縮減（2026-09-17，實作時發現）：M2／M5 在目前的即時系統量測方法下
**結構性測不到**，不是還沒做完：
  - **M2**（策略群集中度）：需要 HRP 樹的分群指派，跟本系統採用的輕量
    `resolve_strategy_holdings` 持股解析管線不是同一套資料來源，未接
  - **M5**（風格曝險漂移）：`factor_exposure_F1`（見 `ops/tools.py:566`）是算在
    **策略清單**（`strategy_uids`）上的分布，而 scheme E 的 walk-forward 設計裡
    策略清單在整個窗次（8 季）內固定不變，只有窗次邊界才會換。既然這個指標
    算在不會變的東西上，它跟登記時的 L1 距離在窗次內**永遠是 0**——不是資料
    不夠，是這個指標定義方式在本系統裡本來就測不到任何漂移

🔴🔴 **M7（候選池整體失效）已於 2026-09-22（§8待辦item10）解鎖**：原本判準
「B_all 相對等權大盤的超額 < 歷史 p10」因為 B_all 只是等權大盤的替身（A9），
數學上恆等於 0，結構性測不到。用股票層方法重算真實 B_all
（`monitor.fetch_ball_returns()`，window1~4 全部涵蓋）後，這個判準第一次
真的能算，見 `diagnose_m7()`。

本檔案實作 **M3、M4、M6、M7、M0、M8**——這六個在目前系統下是真正可計算、有意義
的。M1-D 在 `triggers.py`（前瞻區、可驅動動作）；M1-R（回顧區、已實現貢獻的
基準鏈拆解）尚未實作，需要另外的逐季基準鏈串接邏輯，留待後續。

🔴🔴 2026-09-18（開發追蹤 D56）：新增 **M8（產業集中度）**，回應 §7.3 #6
「產業／主題集中」——但範圍上要精確：這裡測的是**投組自身有沒有無意間集中
在單一產業**（狀態警示區，跟 M2/M3/M5 同一類「我自己跑掉了嗎」），不是
§7.3 表格原本想像的「資金集中同一產業、反轉時同時被殺」那種**外部市場現象**
（那種需要資金流資料才測得到，本專案仍然沒有，見 §7.3 #1 因子擁擠同一類
盲區）。用 `code/app/industry.py` 的產業分類（D47/D49 查證過，TW 覆蓋率
100%），算投組裡佔比最大的單一產業，跟 windows 1-3 歷史分布比對
（`_prelim_industry_concentration_history.py`，n=39，p90=16.39%）。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app import industry as industry_mod

# M3 歷史分布：windows 1-3 真實資料（`_prelim_a1_portfolio_noise.py` 的輸出，
# n=36 季度樣本，2015-2023，符合 §11.7 只用訓練期資料的規則）
_M3_HISTORY_PATH = (Path(__file__).resolve().parent.parent.parent
                    / "_analysis_outputs_applayer" / "a1_portfolio_noise_windows123.csv")

# M0／M6 績效層歷史分布：windows 1-3 真實逐季 excess_vs_ball
# 🔴🔴 2026-09-22（§8待辦item10）：改讀「修正版」——`excess_vs_ball` 欄位已用
# 真實股票層 B_all 重算過（原始檔的 excess_vs_ball 是舊 placeholder，等於
# excess_vs_equal_weight，見 `monitor.fetch_ball_returns()` docstring）。
# 修正後 p10 從 -0.16% 變成 -0.35%、中位數從 +2.39% 變成 +0.69%，是實質修正
# 不是誤差範圍內調整。原始檔（`m0_performance_history.csv`）保留不動，
# 供事後稽核比對兩個版本的差異。
_M0_HISTORY_PATH = (Path(__file__).resolve().parent.parent.parent
                    / "_analysis_outputs_applayer" / "m0_performance_history_corrected.csv")

# M7 候選池整體失效歷史分布：跟M0同一份修正檔算出來的
# `ball_excess_vs_equal_weight`（= 真實B_all - 真實等權大盤），n=36，2015-2023。
# 🔴 2026-09-22：這是本欄位第一次真的能算——修正前 B_all 直接借用等權大盤
# 當替身，這個判準數學上恆為0，M7 因此被列為「結構性測不到」（見檔案開頭
# 說明）。B_all 真的重算出來後，M7 現在真的可以判定了。
_M7_HISTORY_PATH = _M0_HISTORY_PATH

# M8 歷史分布：windows 1-3 真實資料（`_prelim_industry_concentration_history.py`
# 的輸出，n=39，2015-2023，符合 §11.7 只用訓練期資料的規則）
_M8_HISTORY_PATH = (Path(__file__).resolve().parent.parent.parent
                    / "_analysis_outputs_applayer" / "industry_concentration_windows123.csv")


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


def _m8_history() -> pd.Series:
    df = pd.read_csv(_M8_HISTORY_PATH)
    return df["max_industry_weight"]


def diagnose_m8(weights: dict[str, float]) -> dict:
    """產業集中度：投組裡佔比最大的單一產業，超出歷史 p90（狀態警示區，
    無 pp 貢獻，跟 M3 同一類）。只設 p90（上界）不設 p10（下界）——產業
    分散不是失效模式，只有「集中」才是要示警的方向，跟 M3 同時檢查
    上下界（太集中或太分散都算異常）不同，這裡刻意不對稱。"""
    hist = _m8_history()
    p90 = hist.quantile(0.90)
    top_industry, top_weight = industry_mod.max_industry_weight(weights)
    triggered = top_weight > p90
    return {
        "mechanism": "M8", "name": "產業集中度",
        "top_industry": top_industry, "max_industry_weight": top_weight,
        "historical_p90": float(p90),
        "triggered": bool(triggered), "action": "A4或A5" if triggered else "A0",
        "region": "B",  # 狀態警示區，無 pp 貢獻
    }


def _m7_history() -> pd.Series:
    df = pd.read_csv(_M7_HISTORY_PATH)
    return (df["ball_benchmark_return"] - df["equal_weight_bench"]).dropna()


def diagnose_m7(ball_benchmark_return: float | None,
                equal_weight_benchmark_return: float | None) -> dict:
    """候選池整體失效：B_all 相對等權大盤的超額 < 歷史 p10（可歸因區，跟 M1-R／M4
    同一條巢狀基準鏈：A_hrp → B_all → 等權大盤 → 市值加權，見設計文件 §7.4）。

    🔴🔴 2026-09-22（§8待辦item10）：這是本判準第一次真的能算——修正前
    `ball_benchmark_return` 直接借用等權大盤當替身，兩者恆等，判準數學上
    恆為0，見 `monitor.fetch_ball_returns()`。若呼叫端沒有提供真實 B_all
    （`ball_benchmark_return is None` 或兩個值剛好相等，例如落在還沒重算
    過真實B_all的窗次），誠實回報「本次無法判定」，不假裝算出一個數字。

    **不會觸發任何配權重動作**——設計文件原話：「此時調任何權重都救不了，
    誠實回報才是正解」，觸發時對應 A0 或 A5，跟 M1 的 W 系列動作不同。
    """
    if ball_benchmark_return is None or equal_weight_benchmark_return is None:
        return {"mechanism": "M7", "name": "候選池整體失效",
               "available": False, "reason": "本次未提供真實B_all，無法判定",
               "region": "A"}
    excess = ball_benchmark_return - equal_weight_benchmark_return
    hist = _m7_history()
    p10 = hist.quantile(0.10)
    triggered = excess < p10
    return {
        "mechanism": "M7", "name": "候選池整體失效", "available": True,
        "ball_excess_vs_equal_weight": excess, "historical_p10": float(p10),
        "triggered": bool(triggered), "action": "A5" if triggered else "A0",
        "region": "A",
        "note": "觸發時不代表任何配權重動作救得了——整個候選池方法論本身"
               "失效，需要人工覆核而非調整投組",
    }


def baseline_chain_decomposition(a_hrp_return: float, ball_return: float | None,
                                 equal_weight_return: float | None,
                                 cap_weight_return: float | None) -> dict | None:
    """§8待辦item9（M1-R回顧區基準鏈拆解正式化，A7一次性分析的正式版）。

    把 A_hrp 對市值加權大盤的總落差，沿著設計文件§7.4的巢狀基準鏈
    （A_hrp → B_all → 等權大盤 → 市值加權大盤）拆成三段可加總的貢獻：

        A_hrp vs B_all        → M4（選股規則相對候選池失效）
        B_all vs 等權大盤      → M7（候選池整體失效）
        等權大盤 vs 市值加權大盤 → M1-R（規模曝險的已實現貢獻）

    🔴🔴 這是**回顧區（3a）專用的解釋工具，不是新的觸發機制**——跟
    `diagnose_m4()`/`diagnose_m7()` 各自的歷史門檻觸發判斷是分開的兩件事，
    這裡純粹是「這一期的總落差，多少可以歸因到哪一段」的算術分解，本身
    不判斷觸發與否、不建議動作（§7.0：回顧區不得驅動動作）。

    任一段缺資料就誠實回傳 None（例如某窗次還沒重算過真實B_all），不硬湊。
    """
    if ball_return is None or equal_weight_return is None or cap_weight_return is None:
        return None
    gap_m4 = a_hrp_return - ball_return
    gap_m7 = ball_return - equal_weight_return
    gap_m1r = equal_weight_return - cap_weight_return
    total_gap = a_hrp_return - cap_weight_return

    def _pct(gap: float) -> float | None:
        return (gap / total_gap * 100.0) if total_gap != 0 else None

    return {
        "a_hrp_return": a_hrp_return, "ball_return": ball_return,
        "equal_weight_return": equal_weight_return, "cap_weight_return": cap_weight_return,
        "total_gap": total_gap,
        "m4_gap": gap_m4, "m4_pct_of_total": _pct(gap_m4),
        "m7_gap": gap_m7, "m7_pct_of_total": _pct(gap_m7),
        "m1r_gap": gap_m1r, "m1r_pct_of_total": _pct(gap_m1r),
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


def run_diagnosis(*, n_unique_stocks: int, excess_vs_ball_history: list[float],
                  weights: dict[str, float] | None = None,
                  ball_benchmark_return: float | None = None,
                  equal_weight_benchmark_return: float | None = None) -> dict:
    """組出完整的 M0~M8 判定（本階段涵蓋 M3/M4/M6/M0/M7/M8；M1-D 另見 triggers.py）。
    回傳格式對齊 §7.4「固定格式」：A 區（可歸因，M1/M4/M7）／B 區（狀態警示，M2/M3/M5/M8）。
    🔴 2026-09-22（§8待辦item10）：**M7 已解鎖**——B_all 真的用股票層方法重算過
    （`monitor.fetch_ball_returns()`），不再是等權大盤的替身，判準不再恆為0。
    B 區仍只有 M3、M8（M2／M5 結構性測不到，見檔案開頭說明）。

    `performance_below_p10`（M0/M6 用）內部用 `excess_vs_ball_history[-1]`（本季
    的值）自動算，不用呼叫端另外傳——避免 M4 跟 M0/M6 各自被喂到不一致的「本季」
    數字（之前的版本要求呼叫端另外傳一個布林值，容易兜錯）。

    `weights`：投組權重（company_symbol -> weight），給 M8 算產業集中度用。
    選填是為了兼容還沒接產業資料的舊呼叫端——不傳的話 M8 直接跳過，不強制
    每個呼叫端都要接，但 `not_available` 會誠實記錄「本次未提供」而不是
    假裝這個機制不存在。`ball_benchmark_return`／`equal_weight_benchmark_return`：
    給 M7 用，同樣選填，沒提供就記在 `not_available`。
    """
    m3 = diagnose_m3(n_unique_stocks)
    m4 = diagnose_m4(excess_vs_ball_history)
    m8 = diagnose_m8(weights) if weights is not None else None
    m7 = diagnose_m7(ball_benchmark_return, equal_weight_benchmark_return)
    any_triggered = (m3["triggered"] or m4["triggered"] or bool(m8 and m8["triggered"])
                     or bool(m7.get("triggered")))

    perf = None
    if excess_vs_ball_history:
        perf = is_performance_below_p10(excess_vs_ball_history[-1])
    m6_m0 = diagnose_m6_m0(bool(perf and perf["below_p10"]), any_triggered)
    if perf is not None:
        m6_m0["performance_check"] = perf

    region_a = {"M4": m4}
    region_b = {"M3": m3}
    not_available = {
        "M2": "需要 HRP 樹分群指派，跟輕量持股解析管線不同源，未接",
        "M5": "factor_exposure_F1 算在固定不變的策略清單上，窗次內 L1 距離恆為 0，結構性測不到",
    }
    if m7.get("available"):
        region_a["M7"] = m7
    else:
        not_available["M7"] = m7.get("reason", "本次未提供真實B_all，無法判定")
    if m8 is not None:
        region_b["M8"] = m8
    else:
        not_available["M8"] = "呼叫端本次未傳入 weights，跳過（非結構性測不到，見 code/app/industry.py）"

    return {
        "region_a_attributable": region_a,
        "region_b_state_warning": region_b,
        "fallback": m6_m0,
        "not_available": not_available,
    }
