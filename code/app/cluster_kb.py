# -*- coding: utf-8 -*-
"""L1+ · 群知識庫（應用層開發追蹤.md §9.2/§9.4，解釋 agent 的資料來源，2026-09-11）

`_frozen/stage3/` 有 16 個產物，應用層原本只用了 2 個（`cluster_assign`／
`cluster_meta`，且只拿來建樹挑代表）。這支檔案把其餘 8 個接進來，讓解釋 agent
能講「這批策略落在哪些群、這些群什麼性格、彼此像不像、什麼時候賺賠」，
不只是複誦選了幾檔、CAGR 多少（見應用層開發追蹤.md §9.2 資源盤點）。

🔴 **H8（應用層開發追蹤.md §9.7）：一律以主線樹為框架，只支援 `kind="normal"`**
——這些群檔案只對 `_frozen/stage3/` 的六棵主線樹存在，walk-forward 每窗重建的
樹 k 不同（實測 window1=11群、window6=10群），群 id 對不上，沒有對應的群檔案。
正式模式用的就是主線樹，完全吻合；若日後要在驗證模式顯示，必須標注「這是主線樹
的群體知識，不是選股當下那棵樹」——這是本模組刻意不接 walk-forward 樹的原因。

🔴 **R15（§9.7）：群檔案是全樣本算的**——`cluster_profile_quant` 的
`window_end_year`＝2025、`cluster_annual_returns` 的 year 到 2025。正式模式的 IS
本身就是全樣本，沒有前視問題；但這些數字**不可**用來描述任何 IS 結束在 2025 年
之前的驗證模式窗次（那樣會含未來資訊）。這是本模組只給正式模式用的第二個理由。

⚠️ **各檔案的 key 欄位不統一，容易踩雷**：`cluster_identity`／`cluster_profile_quant`／
`cluster_annual_returns`／`cluster_story` 用 `tree_id`（值＝`"TW_normal"` 這種帶
kind 的字串）；但 `co_fail_regimes` 用 `tree_key`（值＝裸市場代號 `"TW"`）。
已用 `.venv` python 實查兩種取值方式，不是憑印象假設。
"""
from __future__ import annotations

import dataclasses

import pandas as pd

from research import freeze, paths

LEVEL = "L1"


def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(paths.STAGE3 / f"{name}.parquet")


@dataclasses.dataclass
class ClusterFootprint:
    """一次執行選出的策略，落在主線樹哪些群、各幾檔，附完整群體知識。"""
    market: str
    tree_id: str                              # 例如 "TW_normal"
    counts: dict[int, int]                    # cluster_id -> 本次選中檔數
    total_in_tree: dict[int, int]             # cluster_id -> 該群在主線樹的總檔數
    identity: dict[int, dict]                 # cluster_id -> cluster_identity 那一列
    profile: dict[int, dict]                  # cluster_id -> cluster_profile_quant 那一列
    corr_normal: pd.DataFrame                 # 群間相關矩陣（平時）index/columns=cluster_id
    corr_crisis: pd.DataFrame | None          # 群間相關矩陣（危機期），可能沒有對應資料
    annual_returns: dict[int, pd.DataFrame]   # cluster_id -> 逐年報酬 [year, ret, n_months]
    co_fail: dict[int, dict]                  # cluster_id -> co_fail_regimes 那一列
    story: pd.DataFrame                       # 本次用到的群之間的 story 配對

    @property
    def clusters_used(self) -> list[int]:
        return sorted(self.counts)

    def share(self, cluster_id: int) -> float:
        """這個群在本次持股裡的佔比（依檔數計算，跟 C3 群佔比同一種算法，
        不是依權重——等權投組下兩者數字相同）。"""
        total = sum(self.counts.values())
        return self.counts[cluster_id] / total if total else float("nan")

    def pair_corr(self, a: int, b: int, *, crisis: bool = False) -> float | None:
        """任兩群的相關係數。`crisis=True` 讀危機期矩陣（可能是 None）。"""
        m = self.corr_crisis if crisis else self.corr_normal
        if m is None or a not in m.index or b not in m.columns:
            return None
        return float(m.loc[a, b])


def build_footprint(market: str, members: list[str]) -> ClusterFootprint:
    """給一批 strategy_uid，組出它們在主線樹（`{market}_normal`）裡的群足跡 +
    對應的全部群體知識。

    ⚠️ 只支援主線樹（H8）——不接受 walk-forward 每窗重建的樹（那些沒有對應的
    群檔案），呼叫端要自己確保這是正式模式，或明確標注「這是主線樹的群體知識，
    不是選股當下那棵樹」。
    """
    tree_id = f"{market}_normal"
    freeze.verify_inputs(paths.STAGE3)   # I-9：app 層先前完全沒做凍結驗證，這裡補上

    assign = _load("cluster_assign")
    assign = assign[assign.tree_id == tree_id]
    if assign.empty:
        raise ValueError(f"cluster_assign.parquet 裡找不到 {tree_id}")
    col = f"cluster_{LEVEL}"
    member_set = set(members)
    sub = assign[assign.strategy_uid.isin(member_set)]
    if len(sub) < len(member_set):
        missing = member_set - set(sub.strategy_uid)
        raise ValueError(f"{len(missing)} 個策略在 {tree_id} 裡找不到分群，"
                         f"例如 {sorted(missing)[:3]}——選股跟建樹用的是不是同一棵樹？")
    counts = {int(k): int(v) for k, v in sub[col].value_counts().items()}
    total_in_tree = {int(k): int(v) for k, v in assign[col].value_counts().items()}

    identity_df = _load("cluster_identity")
    identity_df = identity_df[(identity_df.tree_id == tree_id) & (identity_df.level == LEVEL)]
    identity = {int(r.cluster_id): r._asdict() for r in identity_df.itertuples(index=False)}

    profile_df = _load("cluster_profile_quant")
    profile_df = profile_df[(profile_df.tree_id == tree_id) & (profile_df.level == LEVEL)]
    profile = {int(r.cluster_id): r._asdict() for r in profile_df.itertuples(index=False)}

    corr_normal = _load(f"cluster_corr_matrix_{tree_id}")
    corr_normal.index = corr_normal.index.astype(int)
    corr_normal.columns = corr_normal.columns.astype(int)
    try:
        corr_crisis = _load(f"cluster_corr_matrix_{market}_crisis")
        corr_crisis.index = corr_crisis.index.astype(int)
        corr_crisis.columns = corr_crisis.columns.astype(int)
    except FileNotFoundError:
        corr_crisis = None

    annual_df = _load("cluster_annual_returns")
    annual_df = annual_df[(annual_df.tree_id == tree_id) & (annual_df.level == LEVEL)]
    annual_returns = {int(cid): g[["year", "ret", "n_months"]].reset_index(drop=True)
                      for cid, g in annual_df.groupby("cluster_id")}

    # ⚠️ co_fail_regimes 用 tree_key（裸市場代號），不是 tree_id——跟其他檔案不一樣
    co_fail_df = _load("co_fail_regimes")
    co_fail_df = co_fail_df[(co_fail_df.tree_key == market) & (co_fail_df.level == LEVEL)]
    co_fail = {int(r.cluster_normal): r._asdict() for r in co_fail_df.itertuples(index=False)}

    story_df = _load("cluster_story")
    story_df = story_df[(story_df.tree_id == tree_id) & (story_df.level == LEVEL)]
    used = set(counts)
    story_df = story_df[story_df.cluster_a.isin(used) & story_df.cluster_b.isin(used)]

    return ClusterFootprint(
        market=market, tree_id=tree_id, counts=counts, total_in_tree=total_in_tree,
        identity=identity, profile=profile,
        corr_normal=corr_normal, corr_crisis=corr_crisis,
        annual_returns=annual_returns, co_fail=co_fail,
        story=story_df.reset_index(drop=True),
    )
