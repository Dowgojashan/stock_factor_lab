# -*- coding: utf-8 -*-
"""L0 · 設定層：RunConfig（應用層開發追蹤.md §1 L0，Phase A）

用一個可存可重用的 JSON 檔描述「照哪一組研究結果去運作」，不勉強套進
`research/contracts.py` 的 Schema（那是驗證 DataFrame 用的，這裡只是單一設定物件）。

預設值全部取自研究部已驗證的結論，不是隨便訂的：
    ratio="legacy"、allocation="equal"、k_mode="silhouette_is"
    ← 應用層開發追蹤.md §3 B2／H-26 §1.5／H-27 §2.2
    single_stock_cap／cluster_cap_equal／cluster_cap_proportional
    ← 應用層開發追蹤.md §3 C2/C3（TW）＋ §9.7 I-4（US/XM，2026-09-11 補），
      從 900 格歷史資料反推，**依市場各自不同**（見 `_DEFAULT_*_CAP` 表）

用法：
    cd code
    python -c "from app.config import RunConfig; RunConfig.default_replay('TW').save('my_run.json')"
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

Mode = Literal["live", "replay"]
Market = Literal["TW", "US", "XM"]

#: 🔴 2026-09-11（§9.7 I-4／S0）：C2/C3 門檻依市場各自反推，不可三市場共用同一組
#: 台股數字——實測 US 比例分配的歷史最大 `max_cluster_share`（50.00%）**剛好等於**
#: TW 原上限，沿用會讓門檻對 US 形同虛設。反推方法與 TW 當初相同：C2＝歷史最大值
#: ×2、C3＝歷史最大值＋緩衝取整數（見 `應用層開發追蹤.md` §9.7 I-4 的推導表）。
#:
#: 實測歷史最大值（900 格 legacy~10% 的 `walkforward_matrix_detail.csv`）：
#:   C2（單一策略自然權重 1/n_members）  TW 4.00%／US 5.00%／XM 6.67%（legacy k=3，1/15）
#:   C3 等量分配 max_cluster_share       TW 22.31%／US 25.12%／XM 33.48%
#:   C3 比例分配 max_cluster_share       TW 46.67%／US 50.00%／XM 53.33%
DEFAULT_SINGLE_STOCK_CAP = {"TW": 0.08, "US": 0.10, "XM": 0.14}
DEFAULT_CLUSTER_CAP_EQUAL = {"TW": 0.25, "US": 0.30, "XM": 0.40}
DEFAULT_CLUSTER_CAP_PROPORTIONAL = {"TW": 0.50, "US": 0.55, "XM": 0.60}
Allocation = Literal["equal", "proportional"]
#: `mainline_h03` 是 2026-09-10（§7 P2）為正式模式新增的：k 來自 `_frozen/stage3/`
#: 主線樹，由 H-03 用輪廓係數在**完整共同窗**（TW 2007-01~2025-12，228 個月）上選出
#: （TW=6）。⚠️ 語意上它就是「用該窗 IS 資料選 k」＝`silhouette_is`——因為正式模式的
#: IS 就是完整窗。分開命名是為了**稽核可追溯**（看紀錄就知道 k 是查 `k_stability.csv`
#: 還是讀凍結主線樹），不是兩套不同演算法。見應用層開發追蹤.md §7.4b。
KMode = Literal["fixed", "silhouette_is", "mainline_h03"]
Group = Literal["A_hrp", "D_top_cagr", "E_top_calmar"]


@dataclasses.dataclass
class ReplayAnchor:
    """`mode="replay"` 專用：指到 `walkforward_members.parquet` 的哪一格。

    ⚠️ A4（應用層開發追蹤.md）：選錨點時應避開太接近 LLM 訓練截止日的窗，
    這件事由呼叫端（人）判斷，這裡不做自動檢查。
    """
    scheme: str            # 13 個窗口方案代號之一，見 H-26（A~L、R）
    window_no: int


@dataclasses.dataclass
class RunConfig:
    mode: Mode = "replay"
    market: Market = "TW"
    group: Group = "A_hrp"
    ratio: str = "legacy"
    allocation: Allocation = "equal"
    # B2：慢時鐘重建樹固定用 silhouette_is（每次重選 k），不用 fixed（有前視偏誤）。
    k_mode: KMode = "silhouette_is"

    # C2/C3：風控上限，預設值＝研究結果反推，依市場各自不同（見上方 §9.7 I-4）。
    # None＝依 market 查表帶入預設值；允許使用者覆寫（C1）。
    single_stock_cap: float | None = None
    cluster_cap_equal: float | None = None
    cluster_cap_proportional: float | None = None

    # A3：只有 mode="replay" 需要。
    replay_anchor: ReplayAnchor | None = None

    created_at: str = dataclasses.field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if self.mode == "replay" and self.replay_anchor is None:
            raise ValueError("mode='replay' 必須提供 replay_anchor（scheme + window_no）")
        if self.mode == "live" and self.replay_anchor is not None:
            raise ValueError("mode='live' 不該有 replay_anchor（那是 replay 專用欄位）——"
                             "正式模式的 IS 固定為錨點到最新可用月，不指向任何凍結窗次")
        # k_mode 與 mode 必須配套（2026-09-10 §7 P2）：正式模式讀凍結主線樹的 k，
        # replay 模式查 walkforward 那一格的 k。混用會讓稽核紀錄講不清楚 k 從哪來。
        if self.mode == "live" and self.k_mode != "mainline_h03":
            raise ValueError(
                f"mode='live' 的 k_mode 必須是 'mainline_h03'（目前 {self.k_mode!r}）——"
                "正式模式用 _frozen/stage3 主線樹，k 由 H-03 在完整共同窗上選出")
        if self.mode == "replay" and self.k_mode == "mainline_h03":
            raise ValueError("mode='replay' 不能用 k_mode='mainline_h03'——"
                             "凍結表的每一格各有自己的 k，見 walkforward_matrix_detail.csv")
        # §9.7 I-4：C2/C3 沒有明確指定時，依市場帶入各自反推的預設值——
        # 三市場門檻不同，不可共用 TW 的數字（US 比例分配的歷史最大值剛好等於
        # TW 舊上限，沿用會讓門檻形同虛設）。
        if self.single_stock_cap is None:
            self.single_stock_cap = DEFAULT_SINGLE_STOCK_CAP[self.market]
        if self.cluster_cap_equal is None:
            self.cluster_cap_equal = DEFAULT_CLUSTER_CAP_EQUAL[self.market]
        if self.cluster_cap_proportional is None:
            self.cluster_cap_proportional = DEFAULT_CLUSTER_CAP_PROPORTIONAL[self.market]
        cap = self.cluster_cap_equal if self.allocation == "equal" else self.cluster_cap_proportional
        if not (0 < self.single_stock_cap <= 1) or not (0 < cap <= 1):
            raise ValueError("風控上限必須是 (0, 1] 之間的比例")

    @property
    def cluster_cap(self) -> float:
        """C3：依目前分配方式自動切換的群佔比上限（2026-09-09 review 定案）。"""
        return self.cluster_cap_equal if self.allocation == "equal" else self.cluster_cap_proportional

    @classmethod
    def default_replay(cls, market: Market, scheme: str = "A", window_no: int = 6) -> "RunConfig":
        """給一個馬上能用的 replay 設定：legacy＋等量＋方案 A 的最後一窗。"""
        return cls(mode="replay", market=market,
                   replay_anchor=ReplayAnchor(scheme=scheme, window_no=window_no))

    @classmethod
    def default_live(cls, market: Market = "TW") -> "RunConfig":
        """正式模式（§7 G1/G3）：IS＝錨點到最新可用月，無 OOS，用凍結主線樹。

        ⚠️ 這裡沒有「選窗次」這個概念——正式模式依定義只有一個 IS
        （`contracts.HRP_WINDOWS[market]`），所以也沒有日期參數可傳（§7.5 G5：
        探索模式決定不做）。
        """
        return cls(mode="live", market=market, k_mode="mainline_h03")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(dataclasses.asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        anchor = d.pop("replay_anchor", None)
        return cls(replay_anchor=ReplayAnchor(**anchor) if anchor else None, **d)
