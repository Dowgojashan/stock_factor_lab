# -*- coding: utf-8 -*-
"""L1 · 執行引擎（應用層開發追蹤.md §1，Phase A）

讀 RunConfig → 跑對應模式 → 回傳「當期持股 + 績效/風險數字」。

⚠️ **`live` 模式尚未實作**——需要即時查資料庫算最新財報條件（A2），本機資料庫
目前連不上（2026-09-09 查證：MySQL 服務未啟動），先把不依賴資料庫的部分做完
（`replay` 模式 + 風控層），A2 留到資料庫恢復後再接。

⚠️ **本階段的「當期持股」是策略層級**（`strategy_uid` 清單），不是股票層級——
跟 `ops.tools` T8/T9 現有工具的操作對象一致（它們也是拿 strategy_uid 當持倉單位）。
把每個策略解析成「現在實際會買哪些股票」是 A2 live 模式要做的下一步，這裡先
誠實標註這個範圍，不假裝已經做到股票層級。

用法：
    cd code
    python -c "
    from app.config import RunConfig
    from app.engine import run
    cfg = RunConfig.default_replay('TW')
    print(run(cfg))
    "
"""
from __future__ import annotations

import dataclasses

import pandas as pd

from research import paths
from .config import RunConfig

MEMBERS_PATH = paths.ROOT / "_analysis_outputs_robustness" / "walkforward_members.parquet"
DETAIL_PATH = paths.ROOT / "_analysis_outputs_robustness" / "walkforward_matrix_detail.csv"


@dataclasses.dataclass
class Holdings:
    """一次執行的「當期持股」結果（策略層級，見本檔案 docstring 的範圍限定）。"""
    config: RunConfig
    members: list[str]                # strategy_uid 清單，本階段的「持股」單位
    n_members: int
    window_info: dict                 # is_start/is_end/oos_start/oos_end 等背景資訊
    performance: dict                 # oos_cagr/oos_mdd/oos_sharpe 等（replay 才有）
    cluster_info: dict                # {"max_cluster_share", "n_clusters"}——
                                      # 這一窗**自己重建**的樹算出來的，見下方警告


def _load_replay_row(cfg: RunConfig) -> pd.Series:
    """從凍結的 45 窗資料裡，找出跟這份設定完全對應的那一格。"""
    members_df = pd.read_parquet(MEMBERS_PATH)
    detail_df = pd.read_csv(DETAIL_PATH)
    detail_df["ratio"] = detail_df["ratio"].astype(str)

    key = dict(tree_key=cfg.market, scheme=cfg.replay_anchor.scheme,
              window_no=cfg.replay_anchor.window_no, k_mode=cfg.k_mode,
              ratio=cfg.ratio, allocation=cfg.allocation, group=cfg.group)

    m = members_df
    for col, val in key.items():
        m = m[m[col] == val]
    if len(m) == 0:
        raise ValueError(f"walkforward_members.parquet 裡找不到這格設定：{key}")
    if len(m) > 1:
        raise AssertionError(f"設定不足以鎖定唯一一格，撞到 {len(m)} 列：{key}")
    row = m.iloc[0]

    d = detail_df
    for col, val in key.items():
        d = d[d[col] == val]
    if len(d) != 1:
        raise AssertionError(
            f"walkforward_matrix_detail.csv 對不上同一格（{len(d)} 列）：{key}")
    perf = d.iloc[0]
    return row, perf


def _run_replay(cfg: RunConfig) -> Holdings:
    row, perf = _load_replay_row(cfg)
    members = list(row["members"])
    return Holdings(
        config=cfg, members=members, n_members=len(members),
        window_info={"is_start": row["is_start"], "is_end": row["is_end"],
                    "oos_start": row["oos_start"], "oos_end": row["oos_end"],
                    "scheme": row["scheme"], "window_no": int(row["window_no"])},
        performance={"is_cagr": float(perf["is_cagr"]), "is_mdd": float(perf["is_mdd"]),
                    "oos_cagr": float(perf["oos_cagr"]), "oos_mdd": float(perf["oos_mdd"]),
                    "oos_sharpe": float(perf["oos_sharpe"]),
                    "n_backfilled": int(perf["n_backfilled"])},
        # 🔴 實跑驗證抓到的教訓：這裡**不能**用 `ops.tools.t8_compute_portfolio_risk`
        # 的 cluster_coverage 去算群佔比——T8 把 members 對到凍結的全域六棵樹
        # （固定 k=6），但 k_mode="silhouette_is" 時每一窗自己重建的樹群數完全
        # 不同（實測 window1=11群、window6=10群），兩者對不起來，算出來的
        # max_cluster_share 差了 2.8~3.6 倍。這一窗真正的群結構已經在
        # `walkforward_matrix_detail.csv` 裡算好、凍結，直接讀出來才是對的。
        cluster_info={"max_cluster_share": float(perf["max_cluster_share"]),
                     "n_clusters": int(perf["n_clusters"]),
                     "n_clusters_covered": int(perf["n_clusters_covered"])},
    )


def _run_live(cfg: RunConfig) -> Holdings:
    raise NotImplementedError(
        "live 模式需要 A2（即時查資料庫算最新財報條件），尚未實作——"
        "2026-09-09 查證本機資料庫連不上，這塊留到資料庫恢復後再接。"
        "先用 mode='replay' 開發/測試風控層跟其餘管線。")


def run(cfg: RunConfig) -> Holdings:
    return _run_replay(cfg) if cfg.mode == "replay" else _run_live(cfg)
