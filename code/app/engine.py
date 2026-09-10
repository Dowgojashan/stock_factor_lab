# -*- coding: utf-8 -*-
"""L1 · 執行引擎（應用層開發追蹤.md §1，Phase A）

讀 RunConfig → 跑對應模式 → 回傳「當期持股 + 績效/風險數字」。

⚠️ **`live` 模式（策略選擇本身）尚未實作**，但**暫緩的理由已於 2026-09-10 更正**
（見應用層開發追蹤.md §7）：原本寫「要把報酬序列補到今天」，那其實不必要——
慢時鐘依 B2 是 **2 年**重建一次，用資料到 2025-12（`returns_monthly.parquet` 實測
範圍 2000-02~2025-12）建的樹在 2026 年使用，屬於 24 個月週期內的正常狀態，
不是將就。真正還沒做的是**即時建樹這條程式路徑本身**（§7 的 P1）。

🔴 **本檔案最核心的設計問題（應用層 §7.1）**：`_load_replay_row()` 是純
`.loc[]` 查表——「選了哪些策略」這個核心決策 100% 讀自凍結的
`walkforward_matrix_detail.csv`／`walkforward_members.parquet`，本檔案不計算
任何東西。§7 已定案要改成即時建樹（P1：抽 `app/clustering.py`，新增即時路徑但
**不刪除**現有查表路徑，先跑 5~10 格凍結表設定驗收成員清單 100% 一致）。

🔴 **改的時候必守（§8-R2 查證）**：驗證過的 8,370 格**全部是等權組合**
（`walkforward_matrix.py:406` 的 `_portfolio_series` ＝ `mean(axis=0)`），HRP 只
用來分群與配額，**從來沒有拿來配權重**。即時路徑一律維持等權；`ops.tools.t9`
的 HRP 權重只能當並列對照，不得當實際權重（否則 C2 的 8% 上限也失去依據——
那是從 `1/n_members` 反推的）。同理 `allocation="equal"/"proportional"` 決定的是
**各群分到幾個代表名額**，不是投組權重。

🔴 **2026-09-10（應用層 §6 落差⑤）更正**：上面暫緩的只是「重建樹」這一半。
資料庫本身這次已經修好（換成獨立版 MariaDB，見 CLAUDE.md §5），
`resolve_strategy_holdings.py` 已經證實「已凍結驗證的策略清單，現在實際
持有哪些股票」這件事不需要重建樹、不需要 returns_monthly 補到今天——
`fcv_core.MarketData.get_mask()` 對任意日期都成立，`candidate_index` 的策略
定義是規則性的。這一半的「快時鐘」已經在 `app/ui.py`「股票層級持股明細」
的「今天」選項接上，見該檔案對應段落的說明。

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

# H-12 四組對照裡，跟 RunConfig.group 同義的候選方案代號（B_all/C_random 沒有
# 固定的 members 清單可比，不納入並列比較）。
_ALL_GROUPS = ("A_hrp", "D_top_cagr", "E_top_calmar")


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
    alternative_groups: dict          # 2026-09-10（應用層 §6 落差③）：同一格設定下
                                      # 其他候選方案（見 _load_alternative_groups）


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


def _load_alternative_groups(cfg: RunConfig, current_members: list[str]) -> dict:
    """H-12/H-26 四組對照裡，跟目前選定 group 同一格設定的其他候選方案——
    數字全部是研究部已經跑好、寫進 walkforward_matrix_detail.csv 的凍結結果，
    不是重新計算、也不是重新開一次「哪個方法比較好」的辯論（那件事 H-26/H-27/
    M-03 已經用統計證據回答過：精選贏過狂灑，但 HRP 分群本身不提供報酬優勢）。
    這裡純粹把已經算好的比較結果攤開給決策者看，呼應 Self-Driving Portfolio
    論文（Ang et al. 2026）PC Strategy Review 的精神——候選方案要並列可見，
    不是只看得到已經選定的那一個。

    找不到對應資料的方案（例如某些比例/分配組合沒跑）直接跳過，不硬湊。
    """
    members_df = pd.read_parquet(MEMBERS_PATH)
    detail_df = pd.read_csv(DETAIL_PATH)
    detail_df["ratio"] = detail_df["ratio"].astype(str)
    key_base = dict(tree_key=cfg.market, scheme=cfg.replay_anchor.scheme,
                    window_no=cfg.replay_anchor.window_no, k_mode=cfg.k_mode,
                    ratio=cfg.ratio, allocation=cfg.allocation)
    curr = set(current_members)

    out = {}
    for g in _ALL_GROUPS:
        if g == cfg.group:
            continue
        key = {**key_base, "group": g}
        m, d = members_df, detail_df
        for col, val in key.items():
            m = m[m[col] == val]
            d = d[d[col] == val]
        if len(m) != 1 or len(d) != 1:
            continue
        alt_members = set(m.iloc[0]["members"])
        perf = d.iloc[0]
        out[g] = {
            "n_members": len(alt_members),
            "oos_cagr": float(perf["oos_cagr"]), "oos_mdd": float(perf["oos_mdd"]),
            "oos_sharpe": float(perf["oos_sharpe"]),
            "n_would_add": len(alt_members - curr),
            "n_would_remove": len(curr - alt_members),
        }
    return out


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
        alternative_groups=_load_alternative_groups(cfg, members),
    )


def _run_live(cfg: RunConfig) -> Holdings:
    # 2026-09-10 更正訊息：原文寫「本機資料庫連不上」已過時（已換成獨立版
    # MariaDB 並修好，見 CLAUDE.md §5），且「要把 returns_monthly 補到今天」
    # 也不是真正的阻礙（見本檔案 docstring）。真正缺的是即時建樹路徑本身。
    raise NotImplementedError(
        "live 模式（即時建樹選策略）尚未實作——這是應用層 §7 的 P1 工作："
        "抽 app/clustering.py 承接 research/stage3_hrp.py 的建樹邏輯，"
        "改吃任意 IS 區間，並用凍結表 5~10 格驗收成員清單 100% 一致。"
        "⚠️ 股票層級的『快時鐘』不受此限，已可運作："
        "見 resolve_strategy_holdings.py 與 app/ui.py 的『今天』選項。"
        "現階段請用 mode='replay'。")


def run(cfg: RunConfig) -> Holdings:
    return _run_replay(cfg) if cfg.mode == "replay" else _run_live(cfg)
