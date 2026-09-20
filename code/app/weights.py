# -*- coding: utf-8 -*-
"""W2c：條件式市值傾斜（設計文件 §7.5，開發追蹤 D50）。

🔴 唯一通過前置驗證、正式加入動作空間的「規模曝險軸」動作——W1／W2（永遠
開啟版）／W3 皆已驗證失敗棄用（D29）。W2c 差別在**只有 M1-D 觸發時才啟動**，
window 1-3（M1-D 幾乎從未觸發）不會被拖累，這是它跟失敗版 W2 的關鍵差異。

驗證歷程（D50，四輪查證都先假設結果是假的）：
  ① 未加上限的 α=0 在多數觸發季度違反 C2=8%（最高 61.36%，單一台積電）——
     加本檔的 `cap_and_redistribute()`（7% 目標上限）後 7 個觸發季全部守住
     （除下面第③點的殘留限制）
  ② 真實交易成本（`FEE_RATIO`／`TAX_RATIO`，見 `_prelim_k_anchor_full45_netcost.py`
     已查證的真實台股成本）幾乎沒有侵蝕改善——條件式傾斜的換手率反而低於
     現況基準線（因為集中在少數大型股比現況每季隨財報重排幾百檔小型股更穩定）
  ③ 🔴 **季度制重新平衡無法 100% 保證任何時刻都不超過硬上限**——重新平衡只在
     季初生效，季中若被砍過的股票大漲，季底可能又漂回超過上限。用 7% 目標
     （留 1pp 緩衝，非直接用 8%）大幅降低但沒有完全消除這個風險，須誠實揭露
  ④ 重分配規則（proportional vs equal-split）敏感度不大，結論不依賴這個細節

**現況（2026-09-18）**：只到「決策記錄」層級——`decision` 選 W2c 之後，
目前**不會**真的改變 `simulate.py` 後續季度的 `resolve_weights()`（跟其他
動作 A0/A2/A4/A5 現況相同，見開發追蹤 D50「新發現的既有缺口」）。執行層
（讓決策真的驅動後續持股計算）列為待辦，尚未實作。
"""
from __future__ import annotations

C2_CAP = 0.08          # 設計文件 §2 C2：單一股票上限（法規/業界對照後的既有政策）
W2C_TARGET_CAP = 0.07  # 實際套用的目標值（留 1pp 緩衝，見上方③）


def tilt_weights(w0: dict[str, float], mcap_row, alpha: float) -> dict[str, float]:
    """`w(s) ∝ w0(s)^alpha × mktcap(s)^(1-alpha)`（設計文件 §7.5 W2c 定義）。
    α=1 還原現有等權基準（不改變現況）；α=0 變成對已選中股票子集的市值加權
    （W2c 觸發時採用的設定）。缺市值的股票在 alpha<1 時無法算比例，直接排除
    （跟現有系統「量不到就不裝作有」的一致立場）。`mcap_row` 為
    `pandas.Series`（index=company_symbol, value=market_capital）。"""
    if alpha >= 0.999:
        total = sum(w0.values())
        return {s: w / total for s, w in w0.items()} if total else {}
    raw = {}
    for s, w in w0.items():
        mc = mcap_row.get(s)
        if mc is None or mc != mc or mc <= 0 or w <= 0:  # mc != mc 偵測 NaN，不額外 import pandas
            continue
        raw[s] = (w ** alpha) * (mc ** (1 - alpha))
    total = sum(raw.values())
    return {s: v / total for s, v in raw.items()} if total else {}


def cap_and_redistribute(weights: dict[str, float], cap: float = W2C_TARGET_CAP,
                         max_iter: int = 50) -> dict[str, float]:
    """反覆把超過 `cap` 的持股砍到 `cap`，多出來的權重按比例分給還沒被砍過的
    持股，直到沒有人超過上限為止（標準的 iterative capping 演算法，MSCI
    25/50 上限型指數同一套邏輯，避免一次性重分配又把別的股票推過上限）。
    D50 驗證：`cap=0.07`（預設，留緩衝）＋ proportional 重分配是最終定案版本；
    敏感度測試過 equal-split 重分配，結論量級相近（見開發追蹤 D50 ④）。"""
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


def apply_w2c(w0: dict[str, float], mcap_row) -> dict[str, float]:
    """W2c 觸發時的完整權重計算：α=0 市值加權 ＋ cap-and-redistribute。
    未觸發時應直接用 `tilt_weights(w0, mcap_row, 1.0)`（不傾斜），不呼叫這支函式。"""
    raw = tilt_weights(w0, mcap_row, 0.0)
    return cap_and_redistribute(raw, cap=W2C_TARGET_CAP)
