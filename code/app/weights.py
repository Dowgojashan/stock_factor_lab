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

🔴 **2026-10-13使用者正式拍板：W2c觸發時的單股上限由7%改為30%**（原本7%只是
D50驗證時的保守預設；`文件/研究進度報告_2026-10-05.md`§3.3/§七記錄了「40%
可以打贏大盤」「老師要求補測7~30%中間值」的討論過程，使用者在此基礎上決定
正式採用30%，台股/美股共用同一個數字，不分市場）。既有用7%算出的歷史結果
（含TW headline的control0+11.16%/+14.44%→L2+32.73%/+38.17%、美股的8季
模擬）都要用30%重新驗證才能正式引用，見對應的`_cap30`重跑腳本。
"""
from __future__ import annotations

C2_CAP = 0.08          # 設計文件 §2 C2：單一股票上限（法規/業界對照後的既有政策，
                       # 全投組通用，不是 W2c 專屬——這是C2風控層本身的政策上限，
                       # 跟下面W2c觸發時實際套用的目標上限是兩件事，2026-10-13
                       # 使用者的30%決策只動W2C_TARGET_CAP，不動這個）
W2C_TARGET_CAP = 0.30  # 2026-10-13使用者拍板：由7%改為30%（見上方docstring）。
                       # Coverage Tilt的`apply_coverage_tilt()`也沿用這個常數，
                       # 跟著變成30%——Coverage Tilt本身尚未採用，此處只是共用
                       # 程式碼的連帶影響，不是對Coverage Tilt另外做了新決策


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
    D50 驗證：proportional 重分配是最終定案版本（敏感度測試過 equal-split
    重分配，結論量級相近，見開發追蹤 D50 ④）；`cap` 預設值 2026-10-13起由
    7%改為30%（使用者拍板，見本檔開頭docstring），重分配規則本身的定案
    結論不受這次改動影響。"""
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


# ============================================================ W2c-basket（2026-10-14新增，
# 🔴 2026-10-14使用者正式拍板：台美股改用各自適合的W2c變體，不再共用單一
# 設定（見下方`apply_w2c_for_market()`，是`simulate.py`實際呼叫的介面）。
#
# 背景與離線驗證（`app/_check_w2c_basket_offline.py`／`_check_w2c_basket_tw_
# offline.py`／`_check_w2c_forced_full_period_offline.py`，皆用真實歷史報酬、
# 不呼叫LLM）：
#   - 美股window4真實8季LLM模擬（30%上限）：W2c單檔市值傾斜在真實套用的4季
#     （2025Q1~Q4，注意跟agent「決策」當季有一季延遲，決策季本身還沿用上季
#     設定）拖累表現——control0 +39.50% vs 單檔W2c **-1.45%**；查實際持股發現
#     傾斜對象常是「手上市值最大但不保證是當季領漲」的股票（曾是RPAY這種
#     跟AI行情無關的中小型支付公司，也曾是真正的AAPL/AVGO/TSLA），因為美股
#     AI行情由Nvidia/蘋果/博通/特斯拉等多檔輪動領漲，沒有單一持續主導者。
#   - 台股同一組真實觸發季（2025Q1~Q4）：單檔W2c control0 +14.44% vs
#     **+44.18%**，因為傾斜對象幾乎永遠是台積電，而台積電恰好也是這段期間
#     最強的持續贏家——「押注手上最大市值那一檔」在台股等於「押注真正的
#     贏家」。
#   - 籃子版本（K=3，往市值最大的3檔分散傾斜）：美股修好了（-1.45%→+39.62%，
#     幾乎追平control0，不再拖累），但**台股的獲利也幾乎被稀釋光**
#     （+44.18%→+14.76%，幾乎打回control0原點）——因為台股的獲利來源正是
#     「集中」這件事本身，分散掉集中度＝分散掉獲利來源。
#   ⇒ 籃子版本不是「兩邊都更好」的改良，是用犧牲台股獲利去換美股不再虧，
#     是集中vs分散的取捨，不是免費午餐。兩個市場的最佳設定本來就不同，
#     不應該共用一套參數。
#
# 🔴 這個市場專屬設定**只做過離線（不呼叫LLM）的歷史報酬驗證**，尚未跑過
# 正式8季LLM模擬確認agent在面對美股basket版本時的決策行為是否一致——
# 正式LLM驗證留給使用者決定何時執行（會花額度）。

W2C_BASKET_K = 3  # 籃子預設大小，離線測試顯示K=3跟K=5效果幾乎相同，取較小值

#: 2026-10-14使用者拍板：台美股各用適合自己的W2c變體，不共用單一設定。
#: "single"＝現行`apply_w2c()`（市值最大單檔傾斜）；"basket"＝
#: `apply_w2c_basket()`（市值最大K檔分散傾斜）。XM尚未定義（應用層目前
#: 也不支援XM，見`monitor.fetch_mcap_wide()`的exchange對照表只有TW/US）。
W2C_VARIANT_BY_MARKET = {"TW": "single", "US": "basket"}


def apply_w2c_for_market(w0: dict[str, float], mcap_row, market: str) -> dict[str, float]:
    """W2c觸發時的完整權重計算，依`W2C_VARIANT_BY_MARKET`自動選市場適合的
    變體——`simulate.py`的`resolve_portfolio_weights()`應該呼叫這支，不要
    直接呼叫`apply_w2c()`/`apply_w2c_basket()`，否則會漏掉市場專屬設定。"""
    variant = W2C_VARIANT_BY_MARKET.get(market)
    if variant is None:
        raise ValueError(f"W2c尚未替市場「{market}」定義變體，見W2C_VARIANT_BY_MARKET")
    if variant == "single":
        return apply_w2c(w0, mcap_row)
    if variant == "basket":
        return apply_w2c_basket(w0, mcap_row, basket_k=W2C_BASKET_K)
    raise ValueError(f"未知的W2c變體「{variant}」（市場「{market}」）")


def tilt_weights_basket(w0: dict[str, float], mcap_row, alpha: float,
                        basket_k: int = W2C_BASKET_K) -> dict[str, float]:
    """`tilt_weights()`的多股籃子變體：先依市值排序找出目前持股裡最大的
    `basket_k`檔組成籃子，籃子內部套用跟`tilt_weights()`同樣的
    `w(s) ∝ w0(s)^alpha × mktcap(s)^(1-alpha)`公式重新分配**籃子內部**的
    權重（籃子整體拿到的權重比例＝籃子內所有股票原始w0權重之和，不變），
    籃子外的股票維持原始w0權重不變。alpha=0時，籃子內部變成純市值加權
    （籃子內最大市值的仍拿最多，但不會像`tilt_weights`一樣讓單一一檔
    吃下幾乎全部的傾斜權重）。

    跟`tilt_weights`相同的立場：缺市值/非正權重的股票直接排除。`basket_k`
    大於實際有效候選股數時，等同對全部有效候選股做`tilt_weights`（籃子
    涵蓋全部，退化回原版）。
    """
    valid = {s: w for s, w in w0.items()
            if w > 0 and mcap_row.get(s) is not None
            and mcap_row.get(s) == mcap_row.get(s) and mcap_row.get(s) > 0}
    if not valid:
        return {}
    if alpha >= 0.999:
        total = sum(valid.values())
        return {s: w / total for s, w in valid.items()} if total else {}

    ranked = sorted(valid.items(), key=lambda kv: -mcap_row[kv[0]])
    basket = dict(ranked[:basket_k])
    rest = {s: w for s, w in valid.items() if s not in basket}

    basket_raw = {s: (w ** alpha) * (mcap_row[s] ** (1 - alpha)) for s, w in basket.items()}
    basket_raw_total = sum(basket_raw.values())
    basket_w0_total = sum(basket.values())
    if basket_raw_total > 0:
        basket_scaled = {s: (v / basket_raw_total) * basket_w0_total
                         for s, v in basket_raw.items()}
    else:
        basket_scaled = basket

    merged = {**rest, **basket_scaled}
    total = sum(merged.values())
    return {s: v / total for s, v in merged.items()} if total else {}


def apply_w2c_basket(w0: dict[str, float], mcap_row,
                     basket_k: int = W2C_BASKET_K) -> dict[str, float]:
    """W2c-basket變體的完整權重計算：α=0籃子市值加權＋沿用既有的
    `cap_and_redistribute()`護欄（跟`apply_w2c()`同一個單股上限）。
    🔴 實驗性，見本區塊開頭說明，不可當成已驗證的production機制引用。"""
    raw = tilt_weights_basket(w0, mcap_row, 0.0, basket_k=basket_k)
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
    （`W2C_TARGET_CAP`，2026-10-13起為30%）——這是既有、已驗證過的護欄機制
    （MSCI 25/50上限型指數同一套邏輯），不是重新發明一個新的。"""
    holdings, covers = resolve_holdings_and_coverage_frac(md, idx, uids, as_of, hot_segment)
    raw = coverage_tilt_weights(holdings, covers, uids, beta)
    return cap_and_redistribute(raw, cap=W2C_TARGET_CAP)


# ============================================================ RepSwap（代表策略置換，設計文件
# 外、2026-10-08 新增，跟 Coverage Tilt 平行、同樣由 Hot Segment 觸發，是給決策 agent 的
# 第二個選項——Coverage Tilt 調的是「策略間權重」，RepSwap 調的是「群內誰當代表」，
# 不改配額（m 不變）也不改權重公式（swap 後仍是純等權），力度更小、也更針對 Hot
# Segment 診斷想抓的問題本身（代表策略系統性沒持有熱門股，換成同群裡會持有的）。
#
# 🔴🔴 誠實狀態（2026-10-08，首次實作，尚無任何歷史驗證）：跟 Coverage Tilt 一樣，
# 這是**全新、尚未驗證**的動作，不是像 W2c 那樣走完四輪查證才拍板的。使用者決定
# 「先處理程式，測試等美股重跑完、跟台股一起測」（2026-10-08），所以這裡只做到
# 「程式邏輯正確、可被決策 agent 選用」，沒有任何歷史回測/前瞻驗證支撐效果——
# `actions.rep_swap_reference()` 的 caveats 必須完整反映這一點，不可包裝成已驗證。

REP_SWAP_MAX_SWAPS = 3     # 每季最多置換幾個代表（跨所有群加總），刻意保守、
                           # 限制周轉率——這是跟 Coverage Tilt 的設計差異之一：
                           # Coverage Tilt 調整全部代表的權重比例，RepSwap 只動
                           # 少數「完全沒覆蓋熱門股」的代表，其餘原封不動。
REP_SWAP_SCAN_TOPK = 50    # 候選群可能有幾千個成員（M-01/§3.5②：TW 最大群 8,991），
                           # 不可整群掃描 resolve_holdings（成本隨群大小線性放大）。
                           # 先用 Calmar 品質排序取前 K 名再掃描覆蓋率，跟 H-10
                           # 「品質優先、多樣性為次要篩選條件」同一個精神——這裡
                           # 品質優先、覆蓋率為次要篩選條件。


def _candidate_calmar(idx, uid: str) -> float:
    """品質分數＝Calmar 比率（CAGR/|MDD|），跟 H-10 代表策略挑選規則用同一個定義
    （`research/stage3_hrp.py` 的既有量測），不是重新發明一個新指標。MDD=0 或缺值
    時回傳 -inf，讓它排在品質排序最後面，不會被誤選成「品質無限好」。
    單筆查詢用，供只有幾十筆的場合（例如排 zero_cov 置換順序）；大量候選排序
    請用下面的 `_calmar_series()`（向量化，避免逐列 `.loc` 的效能陷阱）。"""
    row = idx.loc[uid]
    mdd = row.get("max_drawdown")
    cagr = row.get("CAGR")
    if mdd is None or cagr is None or mdd != mdd or cagr != cagr or mdd == 0:
        return float("-inf")
    return float(cagr) / abs(float(mdd))


def _calmar_series(idx, uids: list[str]):
    """`_candidate_calmar()` 的向量化版本——HRP 群可能有上千個成員
    （M-01/§3.5②：TW 最大群 8,991），逐一 `.loc[uid]` 排序會隨群大小線性變慢，
    這裡一次性 `.loc[uids]` 批量取出再做向量運算，回傳 `pandas.Series`
    （index=uid，value=Calmar，缺值/MDD=0 一律填 -inf）。"""
    import numpy as np
    import pandas as pd
    sub = idx.loc[uids, ["CAGR", "max_drawdown"]]
    mdd = sub["max_drawdown"].astype(float)
    cagr = sub["CAGR"].astype(float)
    calmar = cagr / mdd.abs()
    bad = mdd.isna() | cagr.isna() | (mdd == 0)
    calmar = calmar.mask(bad, -np.inf)
    return pd.Series(calmar.values, index=sub.index)


def load_cluster_membership(cluster_assign_path, tree_id: str) -> tuple[dict, dict]:
    """讀 `_frozen/stage3/cluster_assign.parquet`，回傳
    (`cluster_of`: {uid -> cluster_L1}, `members_by_cluster`: {cluster_L1 -> [uid,...]})，
    只取 `tree_id` 這棵樹（例如 "TW_normal"）。呼叫端只需要載入一次（每個市場固定），
    不要每季重讀——跟 `simulate.run_simulation()` 載入 `mcap_wide`／`m`（walkforward
    members）同一個「一次性載入、整個模擬迴圈共用」慣例。"""
    import pandas as pd
    ca = pd.read_parquet(cluster_assign_path)
    sub = ca[ca["tree_id"] == tree_id]
    if len(sub) == 0:
        raise ValueError(f"cluster_assign 裡找不到 tree_id={tree_id!r}")
    cluster_of = dict(zip(sub["strategy_uid"], sub["cluster_L1"]))
    members_by_cluster: dict = {}
    for uid, cid in cluster_of.items():
        members_by_cluster.setdefault(cid, []).append(uid)
    return cluster_of, members_by_cluster


def rep_swap_members(md, idx, uids: list[str], as_of: str, hot_segment: list[str],
                     cluster_of: dict, members_by_cluster: dict,
                     max_swaps: int = REP_SWAP_MAX_SWAPS,
                     scan_topk: int = REP_SWAP_SCAN_TOPK) -> dict:
    """RepSwap 觸發時的完整置換邏輯：找出目前代表裡「完全沒持有任何 Hot Segment
    成分股」的成員（`coverage_frac==0`，重用 Coverage Tilt 既有的
    `resolve_holdings_and_coverage_frac()`，不是另一套計算），逐一嘗試在**同一個
    HRP 群**裡找一個尚未入選、覆蓋率>0 的候選策略換進來（候選依 Calmar 品質由高到低
    掃描，`scan_topk` 檔內找到第一個覆蓋率>0 的就採用，不是找「覆蓋率最高」的——
    品質優先、覆蓋率只是次要篩選門檻，跟 H-10 的精神一致）。

    `cluster_of`／`members_by_cluster`：`load_cluster_membership()` 的輸出，呼叫端
    一次性載入後重複傳入，不在這支函式裡重讀 parquet。

    不在同一個群裡找替代，是因為跨群換會同時改變「群間配額」這個已經驗證過的
    分散機制（M-02/M-03：配額的多樣性限制才是 HRP 分散效果的來源，不是群邊界
    本身）——RepSwap 刻意只動「群內誰當代表」，不動群際配額，才不會意外動到
    已經驗證過的機制。

    回傳 `{"members": 置換後的名單, "swaps": [{"cluster", "out", "in", "out_covers",
    "in_covers"}, ...], "n_candidates_without_cluster": int}`——後者記錄有幾個
    零覆蓋成員因為「在 `cluster_of` 裡查不到自己的群」（候選池版本跟分群凍結時
    不同步時可能發生，見 `cluster_assign` 本身只收 is_usable 子集）而被跳過，
    不是靜默失敗，`swaps` 之外的資訊供稽核用。"""
    holdings, covers = resolve_holdings_and_coverage_frac(md, idx, uids, as_of, hot_segment)
    zero_cov = [uid for uid in uids if covers.get(uid, 0.0) == 0.0]
    # 由品質最差的零覆蓋成員先換（跟原本的代表挑選規則一致：先犧牲品質較弱的）
    zero_cov.sort(key=lambda u: _candidate_calmar(idx, u))

    members = list(uids)
    member_set = set(uids)
    swaps = []
    n_no_cluster = 0

    for uid_out in zero_cov:
        if len(swaps) >= max_swaps:
            break
        cid = cluster_of.get(uid_out)
        if cid is None:
            n_no_cluster += 1
            continue
        pool = [u for u in members_by_cluster.get(cid, []) if u not in member_set]
        if not pool:
            continue
        # 向量化排序（`_calmar_series`），群可能有上千個成員，逐列 `.loc` 排序
        # 會隨群大小線性變慢（已用window4實測抓到：未向量化前單次呼叫~109秒，
        # 大半時間花在這裡跟下面的 resolve_holdings 掃描上）
        pool_calmar = _calmar_series(idx, pool).sort_values(ascending=False)
        pool_sorted = pool_calmar.index.tolist()
        for uid_in in pool_sorted[:scan_topk]:
            row = idx.loc[uid_in]
            try:
                from resolve_strategy_holdings import resolve_holdings
                syms, _ = resolve_holdings(md, row, as_of)
            except (RuntimeError, KeyError, ValueError):
                continue
            if not syms:
                continue
            in_cov = len(set(syms) & set(hot_segment)) / len(hot_segment) if hot_segment else 0.0
            if in_cov > 0.0:
                members[members.index(uid_out)] = uid_in
                member_set.discard(uid_out)
                member_set.add(uid_in)
                swaps.append({"cluster": int(cid), "out": uid_out, "in": uid_in,
                             "out_covers": covers.get(uid_out, 0.0), "in_covers": in_cov})
                break

    return {"members": members, "swaps": swaps, "n_no_cluster": n_no_cluster}
