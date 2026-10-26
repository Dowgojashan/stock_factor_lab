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

#: 🔴🔴 2026-10-18改版：原本是單一全域常數（兩個市場共用），美股長期借用
#: 台股的校準結果——不是因為美股集中度分布跟台股一樣，是因為美股在
#: 2026-10-17之前完全沒有足夠歷史市值資料做自己的校準（`outstanding_shares`
#: 2022年以前永遠是null，覆蓋率只有200~650家，不夠拿來算15年以上的歷史
#: 分布）。collector完成`USSharesOutstandingBackfiller`回補（2009年起）後，
#: 美股終於有條件做自己的校準（`_prelim_a1_index_concentration_us.py`）。
#: 改成跟`HOT_SEGMENT_P25`/`P10`同一個「按市場分開」的字典慣例。
#:
#: 🔴🔴 2026-10-21第一次重新校準：`monitor.fetch_mcap_wide('US')`原本沒套
#: `database.py`的`_universe_clause()`，量的其實是NASDAQ/NYSE/AMEX三個
#: 交易所全部symbol（11,187檔，母體逐季從547到6,027劇烈變動），不是論文
#: 宇宙定義的Russell 3000（2,994檔）——修好這個bug後用正確母體重新校準，
#: p90從舊值0.0861變成0.1137、p75從0.0142變成0.0629（2009-2023, n=52）。
#: ⚠️ 當時以為這次校準變動不影響「美股M1-D全程NONE」的結論，但後來同儕
#: 複核算術發現：美股2023Q4登記水準約90.2%，最大可能上升空間只有9.78pp，
#: 小於這組p90（11.37pp）——TRIGGERED在數學上不可能發生，NONE是門檻設計
#: 的必然結果，不是實證發現（這個缺陷在修universe_clause之前也一樣存在）。
#:
#: 🔴🔴 2026-10-21第二次重新校準（collector回補美股歷史市值資料後）：
#: 第一次校準的p90/p75異常偏高，根因是2009-2012那幾年市值資料不是覆蓋面
#: 不足（`_universe_clause`修正後`n_universe`其實沒變），而是**個別公司的
#: `market_capital`/`close`/`outstanding_shares`數值本身有誤**，導致前20%
#: 佔比在2009-2012出現多次99%+的假性尖峰，墊高了p90。collector把能修的
#: 資料品質問題修完後，前20%佔比序列變得合理（長期落在67%~92%，不再出現
#: 失真的99%+尖峰），重新校準：p90全樣本0.0835→排除2009-2011噪訊窗後
#: **0.0197**、p75**0.0139**（n=40，`_prelim_a1_index_concentration_us.py`／
#: `_check_us_m1d_narrow_metric.py`重新產出）。這組新門檻下，2023Q4登記
#: 水準89.41%、上升空間10.59pp，遠大於新p90（1.97pp）——TRIGGERED在數學上
#: 重新變得可能，且用真實2024-2025資料模擬`evaluate_quarter`的狀態機，
#: 2025Q1起OBSERVING、2025Q3起TRIGGERED（見`文件/實驗收尾追蹤.md`§七）。
#: **這代表已完成的美股L2正式模擬（`anchored_w4_openSec_balanced_
#: {baseline,exclude_v1}_cap30_hotseg`）建立在已過期的「全程NONE」假設上，
#: 需要使用者裁示是否重跑**——不像第一次校準，這次的改動是真實的、會改變
#: 生產結果，不是「重跑也一樣」的無資訊變動。
X_P90 = {"TW": 0.0186, "US": 0.0197}   # TW已凍結，見§7.4a／開發追蹤D27；
                                        # US見2026-10-21第二次校準（collector
                                        # 回補資料品質後），`_prelim_a1_
                                        # index_concentration_us.py`
X_P75 = {"TW": 0.012, "US": 0.0139}    # 同上


@dataclasses.dataclass
class M1DCondition:
    """登記時點固定下來的 M1-D 判準——階段 1 產出，之後每季拿它比對。"""
    as_of: str
    registration_q1_weight: float
    p75: float
    p90: float
    market: str = "TW"


def register_m1d(mcap_wide: pd.DataFrame, as_of: str, market: str = "TW",
                 p75: float | None = None, p90: float | None = None) -> M1DCondition:
    """階段 1：期初登記。把登記時點的指數 Q1 集中度固定下來，
    之後每季比對「相對這個水準的累計變動」有沒有超過門檻。
    🔴 p75/p90 預設從`X_P75`/`X_P90`依`market`查表（§11.7程序凍結好的門檻，
    這裡不重新計算，只是登記）；明確傳入p75/p90可覆蓋查表結果，供測試用
    合成資料時指定任意門檻，不受限於目前已校準的市場清單。
    """
    if p75 is None or p90 is None:
        if market not in X_P90:
            raise ValueError(f"M1-D尚未替{market}校準門檻（目前只有TW/US），"
                             f"也可以明確傳入p75/p90覆蓋查表")
        p75 = X_P75[market] if p75 is None else p75
        p90 = X_P90[market] if p90 is None else p90
    env = monitor.environment_layer(mcap_wide, as_of)
    return M1DCondition(as_of=as_of, registration_q1_weight=env["q1_weight"],
                        p75=p75, p90=p90, market=market)


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
#
# 🔴🔴 2026-10-14重新校準TW：window2／window4兩次真實LLM模擬都發現現行門檻
# 對我們實際用的exclude_v1名單形同虛設（coverage_count從未跌破），往回查
# 才發現舊門檻是用`walkforward_members.parquet`的**baseline（含V1）**名單
# 校準的，不是我們實際production用的exclude_v1名單——exclude_v1本身就會
# 把覆蓋率推高（見4.2.3：40.1%→53.6%），用baseline校準出來的門檻，對
# coverage天生較高的exclude_v1名單當然形同虛設。改用exclude_v1名單重新
# 校準（`_recalibrate_hot_segment_exclude_v1.py`，2015-2023共36個季底，
# N=3/X=10%不變，只換代表名單）：P25由0.3363→0.5743、P10由0.3080→0.5085。
# 反查2024-2025：新門檻下8季裡有6季低於P25、3季低於P10（舊門檻下0季）——
# 這代表不需要跳到window2，window4本身就能真正測到Hot Segment觸發。
# ⚠️ 2026-10-19當時US尚未做同一輪重新校準（這段文字原寫於此，US曾經
# 維持舊值）——**2026-10-19稍後已補上**，見下方「✅ 2026-10-19：US已補上
# 個別校準」區塊，這裡保留只是交代歷史脈絡，不代表US現在仍是舊值。
#
# 🔴🔴 2026-10-19（論文session核對機制盤點表時連帶抓到的潛在bug，修在這裡）：
# 上面這段只講了「TW換成exclude_v1名單重新校準」，但沒注意到**TW其實有
# 兩個不同的候選池情境**——baseline（含V1）跟exclude_v1——兩者的coverage
# 分布不一樣，需要各自的門檻，而這裡原本只保留了exclude_v1那一組數字
# （0.5743/0.5085），baseline那組（`_recalibrate_hot_segment_baseline.py`
# 算出的0.4269/0.4127）算完之後沒有地方可以存，若未來有人拿baseline名單
# 跑Hot Segment，會悄悄套用exclude_v1的門檻——跟M1-D曾經「美股套用台股
# 門檻」同一個bug類別（市場/情境沒有分流，共用了不該共用的校準值）。
# 改成比照`X_P90`/`X_P75`的市場dict模式，再多一層pool（"baseline"／
# "exclude_v1"）維度，強制呼叫端明確指定，不給預設值去悄悄猜。
# US兩個pool目前共用同一組舊值（尚未個別校準，見上一段），誠實標註在
# dict裡而不是假裝已經分流。
#
# ✅ 2026-10-19：US已補上個別校準（`_recalibrate_hot_segment_baseline_us.py`／
# `_recalibrate_hot_segment_exclude_v1_us.py`，N=6/X=10%、2015-2023共36個
# 季底，代表名單分別來自`anchored_w4_openSec_balanced_baseline.parquet`／
# `_exclude_v1.parquet`）：baseline P25=0.2574/P10=0.2384、exclude_v1
# P25=0.3333/P10=0.3015——方向跟TW一致（exclude_v1 coverage天生比baseline
# 高，門檻也跟著higher），是個合理的交叉驗證，不是巧合湊出來的數字。

HOT_SEGMENT_P25 = {  # 觀察門檻：低於此值 → OBSERVING
    "TW": {"baseline": 0.4269, "exclude_v1": 0.5743},
    "US": {"baseline": 0.2574, "exclude_v1": 0.3333},
}
HOT_SEGMENT_P10 = {  # 異常門檻：低於此值 → TRIGGERED
    "TW": {"baseline": 0.4127, "exclude_v1": 0.5085},
    "US": {"baseline": 0.2384, "exclude_v1": 0.3015},
}


@dataclasses.dataclass
class HotSegmentCondition:
    """登記時點固定下來的 Hot Segment 門檻——跟 `M1DCondition` 平行，
    但這裡登記的是門檻本身（市場已校準的歷史分位數），不是登記時的覆蓋率水準
    （M1-D 比的是「相對登記時的累計偏離」，Hot Segment 比的是「相對歷史分布的絕對水位」，
    兩者的判斷邏輯本來就不同，不要誤以為要對齊成同一種比較方式）。"""
    market: str
    pool: str
    p25: float
    p10: float


def register_hot_segment(market: str, pool: str) -> HotSegmentCondition:
    """階段 1：期初登記。門檻是市場＋候選池情境已校準的歷史分位數（見上方
    常數），不需要當下的覆蓋率數字。`pool` 沒有預設值——2026-10-19起強制
    呼叫端明確指定 "baseline" 或 "exclude_v1"，不接受省略，避免重演「悄悄
    套用錯的候選池門檻」這類靜默錯誤（見上方常數區塊的說明）。"""
    if market not in HOT_SEGMENT_P25:
        raise ValueError(f"Hot Segment 尚未替 {market} 校準門檻（目前只有 TW/US）")
    if pool not in HOT_SEGMENT_P25[market]:
        raise ValueError(f"Hot Segment 的 {market} 門檻尚未替 pool={pool!r} 校準"
                         f"（目前只有 {sorted(HOT_SEGMENT_P25[market])}）")
    return HotSegmentCondition(market=market, pool=pool,
                               p25=HOT_SEGMENT_P25[market][pool], p10=HOT_SEGMENT_P10[market][pool])


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


# ============================================================ M2：策略群集中度
# （2026-10-14新增，跟 M1-D／Hot Segment／回撤觸發平行、獨立判定）
#
# 背景：2026-09-17`開發待辦追蹤.md`曾把這個判準（當時稱M2）判定「結構性
# 測不到」——理由是即時系統用的輕量持股解析管線，當時拿不到HRP分群指派
# 資料。2026-10-08做RepSwap時，已經把`cluster_assign.parquet`的分群指派
# 正式接進即時管線（`weights.load_cluster_membership()`），這個阻礙已經
# 不存在，但沒有人回頭把M2接起來——2026-10-14查證美股exclude_v1的代表
# 策略名單時，意外發現這份名單本身（還沒有任何RepSwap活動介入）就已經
# 集中56%在單一HRP群（4群，理想值25%），直接證實這個監控有實際意義，
# 不是空想。
#
# 🔴🔴 誠實標註：下面的p75/p90門檻**不是從真實歷史分布校準出來的**（跟
# M1-D/Hot Segment不同）——建立這種校準需要在許多歷史窗口上重跑整套
# build_silhouette_picks()流程去累積max_cluster_share的實證分布，是一個
# 獨立的校準專案，這次沒有做。這裡暫時用「相對理論均勻值（1/k）的倍數」
# 當一個**說得出道理但未經驗證**的暫定門檻（1.5倍→觀察，2倍→異常），
# 跟目前已查到的台股1.5倍（不觸發）／美股2.24倍（觸發）剛好一致，但這個
# 一致性是觀察到的巧合，不是校準出來的保證，寫論文時這條限制要誠實揭露，
# 不可宣稱已完成校準。
M2_P75_MULTIPLE = 1.5
M2_P90_MULTIPLE = 2.0


@dataclasses.dataclass
class M2Condition:
    """跟`M1DCondition`平行，但這裡登記的是「理想均勻值」（1/k，k=HRP群數），
    不是某個時間點的水準——群集中度的比較基準本來就是結構性的（跨幾個群
    平均分配），不是時間序列上的某個登記點。"""
    k: int
    ideal_share: float
    p75_multiple: float = M2_P75_MULTIPLE
    p90_multiple: float = M2_P90_MULTIPLE


def register_m2(k: int, p75_multiple: float = M2_P75_MULTIPLE,
               p90_multiple: float = M2_P90_MULTIPLE) -> M2Condition:
    """階段 1：登記這棵樹的群數 k，算出理論均勻值 1/k 當比較基準。"""
    if k <= 0:
        raise ValueError(f"k 必須是正整數，收到 {k}")
    return M2Condition(k=k, ideal_share=1.0 / k, p75_multiple=p75_multiple,
                       p90_multiple=p90_multiple)


def max_cluster_share(cluster_of: dict, member_uids) -> float:
    """目前持股（代表策略清單）裡，跨HRP群分布的最大單群佔比——用策略檔數
    計（適用`allocation="equal"`，每檔策略權重相同時，檔數佔比＝權重佔比；
    `allocation="proportional"`時這個簡化不精確，是已知限制，之後若要支援
    proportional需要改成吃權重字典而非純檔數）。缺分群指派的策略直接排除
    （跟`tilt_weights`等既有函式「量不到就不裝作有」同一個立場）。"""
    from collections import Counter
    labeled = [cluster_of[u] for u in member_uids if u in cluster_of and cluster_of[u] is not None]
    if not labeled:
        return float("nan")
    counts = Counter(labeled)
    return max(counts.values()) / len(labeled)


def evaluate_m2_quarter(cond: M2Condition, cluster_of: dict, member_uids,
                        prev_state: str) -> dict:
    """階段 2：每季比對。跟M1-D/Hot Segment同一套雙門檻遲滯設計：
    超過p90倍數→TRIGGERED；介於p75~p90倍數之間→維持或升級為OBSERVING；
    低於p75倍數→解除回NONE。

    🔴🔴 這裡的回傳值只能驅動A0／A4／A5（不改投組）——群集中度本身是
    「代表策略清單的結構屬性」，不是「市場端的即時訊號」，目前沒有對應的
    投組變更動作設計（跟drawdown同一個定位，見設計文件原表「M2策略群集中度
    過高→A2」雖然列了A2，但A2（切換equal/proportional配置）不解決群集中度
    本身的問題，只是換一種配權方式，兩者關係尚未釐清，這次先只接A0/A4/A5
    這個保守版本，A2要不要真的接上留給之後決定）。
    """
    share = max_cluster_share(cluster_of, member_uids)
    if share != share:  # NaN
        action = {"NONE": "A0", "OBSERVING": "A4", "TRIGGERED": "A5"}[prev_state]
        return {"max_cluster_share": share, "ideal_share": cond.ideal_share,
               "ratio_to_ideal": float("nan"), "k": cond.k,
               "prev_state": prev_state, "state": prev_state, "action": action,
               "note": "代表策略清單缺分群指派資料，本季不判定、維持上一季狀態不變"}

    ratio = share / cond.ideal_share
    if ratio > cond.p90_multiple:
        new_state = "TRIGGERED"
    elif ratio > cond.p75_multiple:
        new_state = "TRIGGERED" if prev_state == "TRIGGERED" else "OBSERVING"
    else:
        new_state = "NONE"

    action = {"NONE": "A0", "OBSERVING": "A4", "TRIGGERED": "A5"}[new_state]
    return {
        "max_cluster_share": round(share, 4), "ideal_share": round(cond.ideal_share, 4),
        "ratio_to_ideal": round(ratio, 2), "k": cond.k,
        "prev_state": prev_state, "state": new_state, "action": action,
    }
