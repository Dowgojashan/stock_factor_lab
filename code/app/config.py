# -*- coding: utf-8 -*-
"""L0 · 設定層：RunConfig（應用層開發追蹤.md §1 L0，Phase A）

用一個可存可重用的 JSON 檔描述「照哪一組研究結果去運作」，不勉強套進
`research/contracts.py` 的 Schema（那是驗證 DataFrame 用的，這裡只是單一設定物件）。

預設值全部取自研究部已驗證的結論，不是隨便訂的：
    ratio="legacy"、allocation="equal"、k_mode="silhouette_is"
    ← 應用層開發追蹤.md §3 B2／H-26 §1.5／H-27 §2.2
    single_stock_cap=0.08、cluster_cap_equal=0.25、cluster_cap_proportional=0.50
    ← 應用層開發追蹤.md §3 C2/C3，從 900 格歷史資料反推

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
Allocation = Literal["equal", "proportional"]
KMode = Literal["fixed", "silhouette_is"]
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

    # C2/C3：風控上限，預設值＝研究結果反推，允許使用者覆寫（C1）。
    single_stock_cap: float = 0.08
    cluster_cap_equal: float = 0.25
    cluster_cap_proportional: float = 0.50

    # A3：只有 mode="replay" 需要。
    replay_anchor: ReplayAnchor | None = None

    created_at: str = dataclasses.field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if self.mode == "replay" and self.replay_anchor is None:
            raise ValueError("mode='replay' 必須提供 replay_anchor（scheme + window_no）")
        if self.mode == "live" and self.replay_anchor is not None:
            raise ValueError("mode='live' 不該有 replay_anchor（那是 replay 專用欄位）")
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

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(dataclasses.asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        anchor = d.pop("replay_anchor", None)
        return cls(replay_anchor=ReplayAnchor(**anchor) if anchor else None, **d)
