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

🔴 **這段曾經過期，2026-09-30訂正**：上面舊版寫「只到決策記錄層級，選W2c
不會真的改變持股」——這句話早在同一天（D51，2026-09-18）就已經修好過期，
但這裡的docstring當時忘記回頭同步，一度造成agent誤讀（連續5季選保守的A5、
一次都沒選W2c，見`actions.w2c_reference()`docstring的完整記錄）。**現況**：
選W2c會真的透過`simulate.ActiveConfig`改變下一季持股（呼叫本檔`apply_w2c()`），
執行層已完成，不是待辦。

下方Coverage Tilt（2026-09-29接線，設計文件§7.4b）是跟W2c平行的第二個傾斜
機制，由Hot Segment觸發（不是M1-D），見`coverage_tilt_weights()`/`apply_
coverage_tilt()`——**跟W2c不是同一個驗證成熟度**，前置驗證效果是null/微負，
不是已證實有效的動作，見`actions.coverage_tilt_reference()`的caveats。
"""
from __future__ import annotations

C2_CAP = 0.08          # 設計文件 §2 C2：單一股票上限（法規/業界對照後的既有政策，
                       # 全投組通用，不是 W2c 專屬——下面 Coverage Tilt 的護欄
                       # 也是套用同一條政策，只是常數名稱歷史上先給了 W2c）
W2C_TARGET_CAP = 0.07  # 實際套用的目標值（留 1pp 緩衝，見上方③）；Coverage Tilt
                       # 的 `apply_coverage_tilt()` 也沿用同一個目標值，見下方


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


# ============================================================ Coverage Tilt（設計文件 §7.4b④，
# 跟 W2c 平行、獨立觸發，由 Hot Segment 而不是 M1-D 觸發——2026-09-29 正式接線）
#
# 🔴 β=5 是已測試網格 {0,1,2,5,10} 的中段值，刻意保守：2026-09-22 的前置驗證
# （`_prelim_coverage_tilt_prevalidation.py`，TW window 1-3）顯示 β 從 0 拉到 10
# 效果幾乎不變、甚至略朝負向（±0.3pp量級），跟 W2c（四輪查證後才拍板）不是同一個
# 成熟度——是否正式納入動作空間原本留給10/7跟老師確認，這次是照使用者的決定接線，
# 這個背景要誠實記在這裡，不要包裝成「已驗證有效」。
COVERAGE_TILT_BETA = 5.0


def coverage_tilt_weights(holdings: dict[str, list[str]], covers: dict[str, float],
                          uids: list[str], beta: float = COVERAGE_TILT_BETA) -> dict[str, float]:
    """`w_r(t) ∝ (1/R) × (1 + β·coverage_frac(r,t))`，β=0 還原純等權，只調策略間
    的相對權重，策略內部持股仍等權（不動選股邏輯本身）。跟上面 `tilt_weights()`
    是**完全不同的公式**（那支是 W2c 的市值傾斜、直接對股票層操作），這裡是策略層
    先算好策略間權重、再攤回股票層，兩者不可混用、不可共用同一個函式名稱
    （沿用 `_prelim_coverage_tilt_prevalidation.py::tilt_weights()` 的邏輯，正式
    接線時改這個名字以免撞名）。

    `holdings`：{uid: [持股代碼,...]}；`covers`：{uid: coverage_frac}，兩者都由
    `resolve_holdings_and_coverage_frac()` 算好傳入，這裡純算術、不重解持股。
    """
    raw_w = {uid: (1.0 + beta * covers[uid]) for uid in uids if holdings.get(uid)}
    total = sum(raw_w.values())
    strategy_w = {uid: v / total for uid, v in raw_w.items()} if total else {}

    out: dict[str, float] = {}
    for uid, sw in strategy_w.items():
        syms = holdings[uid]
        w = sw / len(syms)
        for s in syms:
            out[s] = out.get(s, 0.0) + w
    return out


def resolve_holdings_and_coverage_frac(md, idx, uids: list[str], as_of: str,
                                       hot_segment: list[str]) -> tuple[dict, dict]:
    """解析每個代表策略在 `as_of` 的持股，並算它對 `hot_segment` 的覆蓋比例
    （策略層級，`coverage_frac(r,t) = |holdings ∩ HotSegment| / |HotSegment|`）。
    跟 `monitor.hot_segment_layer()` 的市場層級 `coverage_count` 是不同粒度的計算，
    這裡是 Coverage Tilt 權重公式要用的輸入。"""
    from resolve_strategy_holdings import resolve_holdings
    hot_set = set(hot_segment)
    holdings: dict[str, list[str]] = {}
    covers: dict[str, float] = {}
    for uid in uids:
        row = idx.loc[uid]
        try:
            syms, _ = resolve_holdings(md, row, as_of)
        except (RuntimeError, KeyError, ValueError):
            syms = []
        holdings[uid] = syms
        n_hot_in = len(set(syms) & hot_set) if syms else 0
        covers[uid] = (n_hot_in / len(hot_set)) if hot_set else 0.0
    return holdings, covers


def apply_coverage_tilt(md, idx, uids: list[str], as_of: str, hot_segment: list[str],
                        beta: float = COVERAGE_TILT_BETA) -> dict[str, float]:
    """Coverage Tilt 觸發時的完整權重計算：解析持股＋算覆蓋比例＋套β公式＋
    cap-and-redistribute（2026-09-30補上的護欄，見下方說明）。
    未觸發時應直接用等權（`simulate.resolve_weights()`），不呼叫這支函式。

    🔴 **護欄理由**（設計文件§7.4b④原本列的待辦第3項）：`coverage_tilt_weights()`
    是純比例傾斜，數學上沒有上限——若某季只有極少數代表策略覆蓋 Hot Segment
    （例如30個裡只有2個），這幾個策略的持股會被過度放大，可能讓單一股票權重
    衝到不合理的高點，等同繞過了C2（單一股票上限）這條全投組通用的政策。跟
    `apply_w2c()`共用同一個`cap_and_redistribute()`機制與同一個目標上限
    （`W2C_TARGET_CAP`，7%，留1pp緩衝）——這是既有、已驗證過的護欄機制
    （MSCI 25/50上限型指數同一套邏輯），不是重新發明一個新的。"""
    holdings, covers = resolve_holdings_and_coverage_frac(md, idx, uids, as_of, hot_segment)
    raw = coverage_tilt_weights(holdings, covers, uids, beta)
    return cap_and_redistribute(raw, cap=W2C_TARGET_CAP)
