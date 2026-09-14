# -*- coding: utf-8 -*-
"""L2 · 風控層（應用層開發追蹤.md §1/§3-C，Phase A）

判定門檻（C2/C3，2026-09-09 從 900 格歷史資料反推，見應用層開發追蹤.md）：
    單一股票（本階段＝單一策略，見 engine.py 的範圍限定）> single_stock_cap → 違規
    單一 HRP 群佔比 > cluster_cap（依分配方式自動切換）→ 違規

🔴 **2026-09-09 實跑驗證抓到的教訓，AL-01 的複用建議要修正**：
   群佔比**不能**用 `ops.tools.t8_compute_portfolio_risk` 的 `cluster_coverage`
   算——T8 把 members 對到凍結的全域六棵樹（固定 k=6），但 walk-forward 每一窗
   自己重建的樹（尤其 k_mode="silhouette_is"）群數/分群結構完全不同，兩者對不
   起來，實測算出來的 `max_cluster_share` 差了 2.8~3.6 倍。
   → 群佔比改用 `engine.Holdings.cluster_info`（這一窗自己的樹算出來、
   已經寫進 `walkforward_matrix_detail.csv` 凍結的正確數字）。
   → T8 只用來算**跟分群結構無關**的東西：`portfolio_mdd`／`portfolio_ann_vol`
   是直接從 members 的歷史報酬序列算出來的，不涉及 cluster 對應，這部分可以
   放心複用。

🔴 **2026-09-10（應用層 §6 落差②）**：T8 呼叫時其實同時算出 `market_share`、
   `factor_exposure_F1`、`regime_avg_ret` 三組數字（`ops/tools.py:590-598`），
   但先前只取用了 `portfolio_mdd`／`portfolio_ann_vol`，其餘三組算完就丟——
   等於白算。這三組不涉及 cluster 對應（跟上面 mdd/vol 同一類，可放心複用），
   現在補進 `RiskReport`，讓 memo 能講「這期因子曝險集中在哪個因子」而不是
   只會複誦選了幾檔、CAGR 多少。

⚠️ 這個教訓對 `live` 模式（A2，尚未實作）同樣成立：慢時鐘用 T9 重建的新樹，
   分群結果不會寫回 `cluster_assign.parquet`，屆時一樣不能用 T8 的
   cluster_coverage，要用 T9 重建當下算出來的分群結果。

C4：違規**直接攔下**，不是只標記——`assess()` 只負責算出違規清單，真正「攔下」
的判斷在 `cli.py`（沒有 override_reason 就不准繼續）。

🔴 **2026-09-10（§8-R5，P3）：`assess()` 的 C2 量錯對象了，補上 `assess_stock_level()`**
`assess()` 的 `single_stock` 判定用的是 `1/n_members`＝**單一策略**權重，但
`1/n > 8%` 需要 n < 12.5，而實際組合是 25~668 檔 ⇒ **在任何正常設定下都不可能觸發**。
而且老師 2026-09-08 講的台積電集中度是**股票層級**問題，不是策略層級。
⇒ 新增 `assess_stock_level()`，吃 `resolve_strategy_holdings` 的輸出算真實股票權重。
⚠️ 舊的 `assess()` C2 判定**保留不動**（它仍是「策略數異常少」的安全網），
兩者是不同層級的檢查，不是誰取代誰。

🔴 **2026-09-11（§9.7 I-1，S1）：XM 混合市場不需要改這支檔案**——`assess_stock_level()`
的權重公式 `w(s) = Σ (1/N策略) × (1/n_i)` 純以 `strategy_uid`／`stock_symbol` 計數，
跟股票屬於哪個市場、用哪種幣別完全無關。XM 投組混合台股（TWD）與美股（USD）持股時，
權重本身不受影響；`market`／`stock_market` 只是額外的顯示用欄位，不進權重計算。
"""
from __future__ import annotations

import dataclasses

from ops import tools as T
from .engine import Holdings


@dataclasses.dataclass
class Violation:
    kind: str          # "single_stock" | "cluster_share" | "stock_weight"
    detail: str
    value: float
    limit: float


@dataclasses.dataclass
class StockConcentration:
    """🔴 §8-R5（2026-09-10，P3）：**真正的股票層級**集中度。

    老師 2026-09-08 講的台積電集中度是**股票層級**問題，但 `assess()` 的 C2 量的是
    **單一策略**權重（`1/n_members`）——`1/n > 8%` 需要 n < 12.5，而實際組合是
    25~668 檔，**C2 在任何正常設定下都不可能觸發**，等於形同虛設，而且量錯對象。

    `resolve_strategy_holdings.py` 早就能算出每個策略當下實際持有哪些股票，
    只是 `risk.py` 從來沒有消費它。這個 dataclass 補上這一段。

    權重定義（跟整條研究鏈的等權慣例一致，見 engine.py docstring）：
        w(股票 s) = Σ_{持有 s 的策略 i} (1/N策略) × (1/n_i)
    其中 n_i 是策略 i 當下持有的股票檔數。⇒ Σ_s w(s) = 1。
    ⚠️ 這是「策略等權、且每個策略內部對自己的持股等權」的隱含假設——跟
    `_portfolio_series` 的每月拉回等權是同一套慣例，不是這裡自己發明的。
    """
    as_of: str
    n_unique_stocks: int
    n_strategies: int
    n_empty_strategies: int            # 🔴 2026-09-11：解出 0 檔股票的策略數（見下方 bug 註記）
    weights: dict[str, float]          # stock_symbol -> 權重（auto_trim=True 時是修剪後的）
    names: dict[str, str]              # stock_symbol -> 公司名
    appear_in: dict[str, int]          # stock_symbol -> 被幾個策略選中
    max_stock_weight: float
    max_stock_symbol: str
    violations: list[Violation]
    trimmed: bool = False              # F2：這份 .weights 是不是自動修剪過的
    raw_weights: dict[str, float] | None = None   # F2：修剪前的原始權重（trimmed=True 才有值）

    @property
    def ok(self) -> bool:
        return len(self.violations) == 0

    def top(self, n: int = 15) -> list[dict]:
        rows = sorted(self.weights.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return [{"stock_symbol": s, "company_name": self.names.get(s, ""),
                 "weight": w, "appear_in_n_strategies": self.appear_in.get(s, 0)}
                for s, w in rows]


#: F2：修剪演算法跟違規判定共用同一個浮點容差，避免「trim_to_cap 認為已收斂」
#: 但 `assess_stock_level` 的違規檢查用零容差判定成「還在違規」這種不一致
#: （2026-09-14 code review 抓到的真 bug：兩處容差原本各寫各的，1e-12 vs
#: 零容差，重新分配算出來的浮點數可能落在 (cap, cap+1e-12] 這個縫隙裡，
#: 被 `trim_to_cap` 判定「沒有超標」但被 `assess_stock_level` 判定「超標」）。
_CAP_EPS = 1e-9


def trim_to_cap(weights: dict[str, float], cap: float) -> dict[str, float]:
    """F2（應用層開發追蹤.md §10.2／§10.5-R-A5）：迭代式等比例修剪。

    超過上限的股票砍到上限，砍下來的部分按「原始權重的相對比例」分給還沒
    超過上限的股票；分完可能又有別的股票被推過上限，重複到全部合規為止
    （業界標準的 capped-weight／waterfall 演算法，不是自創）。

    🔴 可行性邊界（R-A5）：把 `sum(weights)` 平分給 `len(weights)` 檔股票，
    每檔至少會拿到 `total/n`——如果 `cap` 比這個數字還低，數學上**不可能**
    修剪到合規（無論怎麼分都會有股票超標）。這裡直接拒絕執行，不讓演算法
    自己去試（試了也不會收斂，只會無限迭代或悄悄回傳一個仍然違規的結果）。

    🔴 2026-09-14 code review 抓到的真 bug：每一輪最多讓「目前還沒被卡住的
    股票裡，權重超標的那些」一次全部被卡住，最壞情況（每輪剛好只多卡一檔）
    需要到 `n` 輪才會收斂——原本寫死 `range(50)`，股票數一多（例如混合多個
    腳位、或 ratio 選比較大的比例，輕易超過 50 檔不重複股票）就會在合法情境
    下誤炸 `AssertionError`。改成 `range(n + 1)`，剛好覆蓋理論最壞情況。
    """
    n = len(weights)
    if n == 0:
        return dict(weights)
    total = sum(weights.values())
    floor = total / n
    if cap < floor - _CAP_EPS:
        raise ValueError(
            f"上限 {cap:.4%} 低於 {n} 檔股票平均分配的最低權重 {floor:.4%}，"
            f"數學上無法修剪到合規，請調高上限或增加候選股票數")

    w = dict(weights)
    capped: set[str] = set()
    for _ in range(n + 1):
        over = {s for s, v in w.items() if s not in capped and v > cap + _CAP_EPS}
        if not over:
            break
        capped |= over
        free_syms = [s for s in w if s not in capped]
        remaining_total = total - cap * len(capped)
        if not free_syms:
            # 全部都被 cap 卡住——只有 total 剛好等於 cap*n 這個邊界情況會走到這裡
            break
        free_raw_total = sum(weights[s] for s in free_syms)
        if free_raw_total <= 0:
            per = max(remaining_total, 0.0) / len(free_syms)
            for s in free_syms:
                w[s] = per
        else:
            for s in free_syms:
                w[s] = remaining_total * (weights[s] / free_raw_total)
        for s in capped:
            w[s] = cap
    else:
        raise AssertionError(f"修剪迭代 {n + 1} 輪仍未收斂，程式邏輯可能有誤")

    assert max(w.values()) <= cap + _CAP_EPS, "修剪後仍有違規，程式邏輯有錯"
    assert abs(sum(w.values()) - total) < 1e-6, "總權重不守恆"
    return w


def blend_stock_weights(items: list[tuple["StockConcentration", float]]) -> dict[str, float]:
    """F1（應用層開發追蹤.md §10.1／§10.5-R-A1）：把各腳位已經算好的股票層級
    權重（`assess_stock_level().weights`）依混合比例線性組合。純算術，不重新
    查資料庫、不重新解析持股——每個腳位的 `StockConcentration` 由呼叫端
    （`ui.py`）先各自解析好再傳進來。
    """
    if abs(sum(w for _, w in items) - 1.0) > 1e-6:
        raise ValueError("混合比例總和必須是 100%")
    blended: dict[str, float] = {}
    for sc, weight in items:
        for sym, w in sc.weights.items():
            blended[sym] = blended.get(sym, 0.0) + weight * w
    return blended


def assess_stock_level(holdings: Holdings, stock_detail, *,
                       as_of: str = "", cap: float | None = None,
                       auto_trim: bool = False) -> StockConcentration:
    """把解析出來的股票明細換算成真實權重，並用 C2 上限判定（§8-R5）。

    `stock_detail`：`resolve_strategy_holdings` / `ui._resolve_stock_holdings` 的輸出，
    需要有 `strategy_uid`／`stock_symbol` 兩欄（`company_name` 選填）。

    `auto_trim`（F2，2026-09-14）：True 時，超過上限的股票會被修剪
    （見 `trim_to_cap`），`.weights`／`.violations`／`.max_stock_weight` 換成
    修剪後的版本（違規理論上會清空），原始未修剪的權重保留在 `.raw_weights`。
    預設 `False`——維持現有「攔下＋人工覆核」路徑不變，不是取代，是並存的選項。
    """
    if stock_detail is None or len(stock_detail) == 0:
        raise ValueError("沒有股票明細可算——請先解析持股")
    cap = holdings.config.single_stock_cap if cap is None else cap

    # 🔴 2026-09-11 code review 抓到的真 bug：分母原本用
    # `stock_detail.groupby("strategy_uid").size()`（即 stock_detail 裡實際
    # 出現的策略數），但如果某個策略在 `as_of` 那天算出來的持股剛好是空的
    # （F1/F2/C/V 條件同時滿足的股票是 0 檔，這是完全合理、真的會發生的
    # 情況——`resolve_holdings()` 對此不做任何非空保證），那個策略根本不會
    # 出現在 stock_detail 裡，分母悄悄縮小，其餘策略的權重被錯誤放大。
    # 這正是本專案最忌諱的一類「靜默算錯」（pitfalls.md §6：不會報錯，只會
    # 給出看起來合理但錯誤的數字）。
    # 正確語意：找不到股票的策略，它的 1/N 份額視同沒有投資到任何股票
    # （類似持有現金），不應該轉嫁給其他策略去放大——分母固定用
    # `holdings.n_members`（投組真正的策略數），不是 stock_detail 裡剛好
    # 出現的策略數。
    per_strategy = stock_detail.groupby("strategy_uid")["stock_symbol"].nunique()
    n_strat = holdings.n_members
    if n_strat == 0:
        raise ValueError("holdings 沒有任何策略成員")
    if int(per_strategy.size) > n_strat:
        raise ValueError(
            f"stock_detail 裡的策略數（{int(per_strategy.size)}）比 "
            f"holdings.n_members（{n_strat}）還多——是不是傳錯 stock_detail？")
    n_empty = n_strat - int(per_strategy.size)   # 沒有解出任何股票的策略數

    weights: dict[str, float] = {}
    for uid, grp in stock_detail.groupby("strategy_uid"):
        syms = grp["stock_symbol"].astype(str).unique()
        if len(syms) == 0:
            continue
        w = (1.0 / n_strat) / len(syms)
        for s in syms:
            weights[s] = weights.get(s, 0.0) + w

    appear = (stock_detail.drop_duplicates(["strategy_uid", "stock_symbol"])
              .groupby("stock_symbol").size().to_dict())
    names = {}
    if "company_name" in stock_detail.columns:
        names = (stock_detail.drop_duplicates("stock_symbol")
                 .set_index("stock_symbol")["company_name"].fillna("").to_dict())

    raw_weights = dict(weights)
    trimmed = False
    if auto_trim and any(w > cap for w in weights.values()):
        weights = trim_to_cap(weights, cap)
        trimmed = True

    top_sym = max(weights, key=weights.get)
    top_w = weights[top_sym]
    violations = []
    for s, w in sorted(weights.items(), key=lambda kv: kv[1], reverse=True):
        if w > cap + _CAP_EPS:   # 跟 trim_to_cap 共用同一個容差，見該函式上方說明
            violations.append(Violation(
                kind="stock_weight",
                detail=(f"單一股票 {s}（{names.get(s, '')}）權重 {w:.2%}，"
                        f"超過上限 {cap:.2%}"),
                value=w, limit=cap))
    return StockConcentration(
        as_of=as_of, n_unique_stocks=len(weights), n_strategies=n_strat,
        n_empty_strategies=n_empty,
        weights=weights, names={k: str(v) for k, v in names.items()},
        appear_in={k: int(v) for k, v in appear.items()},
        max_stock_weight=top_w, max_stock_symbol=top_sym, violations=violations,
        trimmed=trimmed, raw_weights=raw_weights if trimmed else None)


@dataclasses.dataclass
class RiskReport:
    portfolio_mdd: float | None
    portfolio_ann_vol: float | None
    max_single_weight: float
    max_cluster_share: float
    n_clusters_covered: int
    violations: list[Violation]
    # 2026-09-10（應用層 §6 落差②）：T8 本來就算好、先前被丟掉的三組數字。
    market_share: dict[str, float]
    factor_exposure_f1: dict[str, float]
    regime_avg_ret: dict[str, float]

    @property
    def ok(self) -> bool:
        return len(self.violations) == 0


def assess(holdings: Holdings) -> RiskReport:
    cfg = holdings.config
    n = holdings.n_members
    equal_weight = 1.0 / n if n else 0.0

    # portfolio_mdd／portfolio_ann_vol／market_share／factor_exposure_F1／
    # regime_avg_ret 都不涉及分群結構，T8 可以放心複用；
    # 群佔比改讀 holdings.cluster_info（理由見本檔案開頭的 2026-09-09 教訓）。
    raw = T.t8_compute_portfolio_risk(holdings.members)   # 等權，跟 T8 預設一致
    max_share = holdings.cluster_info["max_cluster_share"]

    violations: list[Violation] = []
    if equal_weight > cfg.single_stock_cap:
        violations.append(Violation(
            kind="single_stock",
            detail=f"本次共選 {n} 檔，等權下單檔權重 {equal_weight:.2%}，"
                   f"超過上限 {cfg.single_stock_cap:.2%}",
            value=equal_weight, limit=cfg.single_stock_cap))
    if max_share > cfg.cluster_cap:
        violations.append(Violation(
            kind="cluster_share",
            detail=f"最大單一策略群佔比 {max_share:.2%}，"
                   f"超過上限 {cfg.cluster_cap:.2%}",
            value=max_share, limit=cfg.cluster_cap))

    return RiskReport(
        portfolio_mdd=raw["portfolio_mdd"], portfolio_ann_vol=raw["portfolio_ann_vol"],
        max_single_weight=equal_weight, max_cluster_share=max_share,
        n_clusters_covered=holdings.cluster_info["n_clusters_covered"],
        violations=violations,
        market_share=raw["market_share"],
        factor_exposure_f1=raw["factor_exposure_F1"],
        regime_avg_ret=raw["regime_avg_ret"],
    )
