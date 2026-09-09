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

⚠️ 這個教訓對 `live` 模式（A2，尚未實作）同樣成立：慢時鐘用 T9 重建的新樹，
   分群結果不會寫回 `cluster_assign.parquet`，屆時一樣不能用 T8 的
   cluster_coverage，要用 T9 重建當下算出來的分群結果。

C4：違規**直接攔下**，不是只標記——`assess()` 只負責算出違規清單，真正「攔下」
的判斷在 `cli.py`（沒有 override_reason 就不准繼續）。
"""
from __future__ import annotations

import dataclasses

from ops import tools as T
from .engine import Holdings


@dataclasses.dataclass
class Violation:
    kind: str          # "single_stock" | "cluster_share"
    detail: str
    value: float
    limit: float


@dataclasses.dataclass
class RiskReport:
    portfolio_mdd: float | None
    portfolio_ann_vol: float | None
    max_single_weight: float
    max_cluster_share: float
    n_clusters_covered: int
    violations: list[Violation]

    @property
    def ok(self) -> bool:
        return len(self.violations) == 0


def assess(holdings: Holdings) -> RiskReport:
    cfg = holdings.config
    n = holdings.n_members
    equal_weight = 1.0 / n if n else 0.0

    # portfolio_mdd／portfolio_ann_vol 不涉及分群結構，T8 可以放心複用；
    # 群佔比改讀 holdings.cluster_info（理由見本檔案開頭的 2026-09-09 教訓）。
    raw = T.t8_compute_portfolio_risk(holdings.members)   # 等權，跟 T8 預設一致
    max_share = holdings.cluster_info["max_cluster_share"]

    violations: list[Violation] = []
    if equal_weight > cfg.single_stock_cap:
        violations.append(Violation(
            kind="single_stock",
            detail=f"本次共選 {n} 檔，等權下單檔權重 {equal_weight:.2%}，"
                   f"超過上限 {cfg.single_stock_cap:.2%}（C2）",
            value=equal_weight, limit=cfg.single_stock_cap))
    if max_share > cfg.cluster_cap:
        violations.append(Violation(
            kind="cluster_share",
            detail=f"最大單一 HRP 群佔比 {max_share:.2%}，"
                   f"超過上限 {cfg.cluster_cap:.2%}（C3，{cfg.allocation} 分配）",
            value=max_share, limit=cfg.cluster_cap))

    return RiskReport(
        portfolio_mdd=raw["portfolio_mdd"], portfolio_ann_vol=raw["portfolio_ann_vol"],
        max_single_weight=equal_weight, max_cluster_share=max_share,
        n_clusters_covered=holdings.cluster_info["n_clusters_covered"],
        violations=violations,
    )
