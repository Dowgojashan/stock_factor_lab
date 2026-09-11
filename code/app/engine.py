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
from research.contracts import BENCHMARK_CAGR   # R8②：基準對照
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
    performance: dict                 # replay：is_*＋oos_*；live：**只有 is_***（無 OOS）
    cluster_info: dict                # {"max_cluster_share", "n_clusters"}——
                                      # 這一窗**自己重建**的樹算出來的，見下方警告
    alternative_groups: dict          # 2026-09-10（應用層 §6 落差③）：同一格設定下
                                      # 其他候選方案（見 _load_alternative_groups）
    # ↓ 2026-09-10（§7 P2）新增，replay 也會填，讓兩種模式的下游程式碼一致
    tree_info: dict = dataclasses.field(default_factory=dict)
                                      # {tree_id, source, k, n_universe, is_start, is_end}
    validation: dict = dataclasses.field(default_factory=dict)
                                      # G7 三態標籤 {label, reason, structural_caveat}
    reference_oos: dict | None = None
                                      # 🔴 R13：正式模式沒有 OOS，這裡放「凍結表同類
                                      # 設定的 OOS 分布」當歷史參照。**這才是該當主要
                                      # 績效顯示的東西**，不是本次的 is_cagr

    @property
    def has_oos(self) -> bool:
        """這次執行有沒有真正的樣本外結果。正式模式恆為 False（依定義沒有）。"""
        return "oos_cagr" in self.performance


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
        tree_info={"tree_id": f"{cfg.market}_wf_{row['is_start']}_{row['is_end']}",
                  "source": "frozen_walkforward", "k": int(perf["n_clusters"]),
                  "n_universe": int(perf["n_universe"]),
                  "is_start": str(row["is_start"]), "is_end": str(row["is_end"]),
                  "n_is_months": int(perf["n_is_months"])},
        validation={
            "label": "✅ 方法已驗證",          # G7：replay 直接讀凍結窗次，定義上已驗證
            "reason": (f"直接讀 H-26/H-27 凍結矩陣的第 {int(row['window_no'])} 窗"
                      f"（方案 {row['scheme']}），該格本身就是驗證資料的一部分"),
            "structural_caveat": (
                "本窗有真實 OOS 可比對，但 OOS 絕對數字已被候選池全樣本篩選高估"
                "（§8-R1），只可作相對比較與校準基準，不可當預期報酬。"),
        },
        reference_oos=reference_oos_distribution(cfg),
    )


def reference_oos_distribution(cfg: RunConfig) -> dict | None:
    """🔴 R13：正式模式沒有 OOS，這裡給「凍結表同類設定的 OOS 分布」當歷史參照。

    **這才是正式模式該當主要績效顯示的東西**——本次的 `is_cagr` 是樣本內配適值，
    實測 900 格台股 A_hrp 顯示 IS CAGR 中位數 23.60% vs OOS 15.98%（高 7.62pp、
    1.48 倍，74.2% 的格子皆然），拿它當預期報酬會直接誤導。

    ⚠️ 這份分布本身也已被高估（§8-R1：候選池是全期間贏過基準的策略），只能當
    **相對比較與校準參照**，不是預期報酬。呼叫端顯示時必須帶上這句話。
    """
    df = pd.read_csv(DETAIL_PATH)
    df["ratio"] = df["ratio"].astype(str)
    sub = df[(df.tree_key == cfg.market) & (df.group == cfg.group)
             & (df.ratio == cfg.ratio) & (df.allocation == cfg.allocation)]
    if sub.empty:
        return None
    return {
        "n_cells": int(len(sub)),
        "oos_cagr_p10": float(sub.oos_cagr.quantile(0.10)),
        "oos_cagr_median": float(sub.oos_cagr.median()),
        "oos_cagr_p90": float(sub.oos_cagr.quantile(0.90)),
        "oos_mdd_median": float(sub.oos_mdd.median()),
        "oos_sharpe_median": float(sub.oos_sharpe.median()),
        "benchmark_cagr": float(BENCHMARK_CAGR.get(cfg.market, float("nan"))),  # R8②
        "caveat": ("歷史參照，非本期預測；且此分布本身已被候選池全樣本篩選高估"
                   "（應用層 §8-R1），只可作相對比較與校準基準"),
    }


def alt_group_win_rates(market: str) -> dict | None:
    """🔴 2026-09-11（§9.7 S2）：§8-R12 定錨句要用的「A_hrp 對其他候選方案的勝率」，
    改成依市場現算，不可沿用寫死的 TW 數字（16.8%／13.2%）——三市場開放後那組
    數字若原樣顯示在 US／XM 的 memo 或 UI 上，會把 TW 專屬事實講成通用事實，
    是真正的正確性錯誤，不只是措辭問題。

    實測三市場都有效（900 格 legacy~10% 逐格對照，`walkforward_matrix_detail.csv`）：
        TW：A_hrp 贏 E_top_calmar  Calmar 16.8%／CAGR 13.2%；贏 D_top_cagr 27.0%／23.1%
        US：A_hrp 贏 E_top_calmar  Calmar  8.0%／CAGR  4.2%；贏 D_top_cagr 21.9%／ 2.4%
        XM：A_hrp 贏 E_top_calmar  Calmar 34.0%／CAGR 17.1%；贏 D_top_cagr 45.0%／17.7%
    """
    df = pd.read_csv(DETAIL_PATH)
    df["ratio"] = df["ratio"].astype(str)
    df["calmar"] = df.oos_cagr / df.oos_mdd.abs().where(df.oos_mdd.abs() != 0)
    key = ["scheme", "window_no", "k_mode", "ratio", "allocation"]
    sub = df[(df.tree_key == market) & df.group.isin(("A_hrp", "E_top_calmar", "D_top_cagr"))]
    piv = sub.pivot_table(index=key, columns="group", values=["calmar", "oos_cagr"],
                          aggfunc="first").dropna()
    if piv.empty:
        return None
    out = {"n_cells": int(len(piv))}
    for other in ("E_top_calmar", "D_top_cagr"):
        for metric in ("calmar", "oos_cagr"):
            out[f"{metric}_vs_{other}"] = float((piv[(metric, "A_hrp")] > piv[(metric, other)]).mean())
    return out


def historical_same_setting_windows(cfg: RunConfig) -> list[dict]:
    """🆕 2026-09-11（§9.4 情境比對②）：同一組設定（market/group/ratio/allocation，
    跟 `reference_oos_distribution()` 用**完全相同的篩選條件**，故列表長度＝
    該函式的 `n_cells`——這裡給的是逐格明細，不是分位數彙總）在凍結矩陣裡逐窗的
    `is_end → oos` 實際結果，讓解釋 agent 能講「這個做法在歷史上 is_end 落在哪些
    年份、那次 oos 實際賺賠多少」，而不是只有一個聚合分布。

    ⚠️ 跟 `reference_oos_distribution()` 一樣只能當**歷史參照**，不是本次預測；
    且這份分布本身已被候選池全樣本篩選高估（§8-R1）。
    """
    df = pd.read_csv(DETAIL_PATH)
    df["ratio"] = df["ratio"].astype(str)
    sub = df[(df.tree_key == cfg.market) & (df.group == cfg.group)
             & (df.ratio == cfg.ratio) & (df.allocation == cfg.allocation)]
    if sub.empty:
        return []
    sub = sub.sort_values("is_end")
    return [
        {"scheme": r.scheme, "window_no": int(r.window_no), "k_mode": r.k_mode,
         "is_end": str(r.is_end), "oos_start": str(r.oos_start), "oos_end": str(r.oos_end),
         "oos_cagr": float(r.oos_cagr), "oos_mdd": float(r.oos_mdd),
         "oos_sharpe": float(r.oos_sharpe)}
        for r in sub.itertuples(index=False)
    ]


def mechanism_consistency(market: str) -> dict | None:
    """🆕 2026-09-11（§9.4 機制二／三：穩定性＋回撤品質）：A_hrp 對 B_all 的
    OOS CAGR 超額分布，以及三組候選方案的 OOS MDD 排名，依市場現算，不寫死
    投影片上那組 TW 專屬數字（+2.79pp~+3.86pp、勝率 85%~100%）——三市場開放後
    這組數字理應跟 R12/`alt_group_win_rates()` 一樣依市場而異。

    ⚠️ 這是**跨全部設定組合**的聚合統計（不是這次 RunConfig 那一格的窄範圍），
    回答的是「這個方法論本身穩不穩」，跟 `reference_oos_distribution()`
    （同一格設定的歷史分布）是互補but不同的問題。

    ⚠️ **B_all 沒有 ratio/allocation 概念**（凍結表裡固定是 `ratio="all"`／
    `allocation="unallocated"`——它是「全買」基準，不分配代表名額），所以只能用
    `(scheme, window_no, k_mode)` 三欄跟 A_hrp/D_top_cagr 對齊，**不能**用
    `reference_oos_distribution()` 那種含 ratio/allocation 的完整 key，
    不然會撞成 0 列（2026-09-11 實跑抓到過這個問題）。
    """
    df = pd.read_csv(DETAIL_PATH)
    df["ratio"] = df["ratio"].astype(str)
    base_key = ["scheme", "window_no", "k_mode"]
    full_key = base_key + ["ratio", "allocation"]

    sub = df[(df.tree_key == market) & df.group.isin(("A_hrp", "D_top_cagr"))]
    piv_cagr = sub.pivot_table(index=full_key, columns="group", values="oos_cagr", aggfunc="first")
    piv_mdd = sub.pivot_table(index=full_key, columns="group", values="oos_mdd", aggfunc="first")
    ball = (df[(df.tree_key == market) & (df.group == "B_all")]
            .drop_duplicates(base_key).set_index(base_key)[["oos_cagr", "oos_mdd"]])
    if piv_cagr.empty or ball.empty:
        return None

    cagr = piv_cagr.join(ball["oos_cagr"].rename("B_all"), on=base_key, how="inner").dropna(
        subset=["A_hrp", "B_all"])
    if cagr.empty:
        return None
    excess = cagr["A_hrp"] - cagr["B_all"]
    out = {
        "n_cells": int(len(cagr)),
        "cagr_excess_vs_Ball_min": float(excess.min()),
        "cagr_excess_vs_Ball_max": float(excess.max()),
        "cagr_excess_vs_Ball_median": float(excess.median()),
        "win_rate_vs_Ball": float((excess > 0).mean()),
    }
    mdd = piv_mdd.join(ball["oos_mdd"].rename("B_all"), on=base_key, how="inner")
    for g in ("A_hrp", "D_top_cagr", "B_all"):
        if g in mdd:
            out[f"oos_mdd_median_{g}"] = float(mdd[g].dropna().median())
    return out


def _run_live(cfg: RunConfig) -> Holdings:
    """正式模式（§7 G1/G3）：用凍結主線樹即時挑代表，無 OOS。

    跟 `_run_replay` 的根本差別：**這裡真的在算**（挑代表、算 IS 績效、算集中度），
    不是查表。那份成員清單不存在於任何檔案——walkforward 的 45 個窗沒有一個是
    完整 228 個月窗（§7.4b）。

    ⚠️ 樹本身不重建：`contracts.HRP_WINDOWS[market]` 就是 `_frozen/stage3` 主線樹的
    建構窗，k 由 H-03 在同一個窗上用輪廓係數選出，重算只會得到同一棵樹。
    """
    from . import clustering as CL   # 延後 import：這支會載 returns_monthly（~35MB）

    months_long, _, _ = CL.load_inputs(log=lambda *a, **k: None)
    tree = CL.load_mainline_tree(cfg.market, months_long, log=lambda *a, **k: None)
    ratio = cfg.ratio if cfg.ratio == "legacy" else float(cfg.ratio)

    cache: dict = {}
    sel = CL.select(tree, ratio, cfg.allocation, cfg.group,
                    k_mode=cfg.k_mode, _cache=cache)

    # 候選方案對照（§6 落差③）：正式模式下三組都要現場算，不能查表
    alts = {}
    for g in _ALL_GROUPS:
        if g == cfg.group:
            continue
        alt = CL.select(tree, ratio, cfg.allocation, g, k_mode=cfg.k_mode, _cache=cache)
        alt_set, cur_set = set(alt.members), set(sel.members)
        alts[g] = {
            "n_members": alt.n_members,
            # ⚠️ 正式模式沒有 OOS，這裡只能給 IS——欄位名刻意不叫 oos_*，
            # 避免下游誤把它當成跟 replay 同一種東西（§8-R13）
            "is_cagr": alt.performance_is["is_cagr"],
            "is_mdd": alt.performance_is["is_mdd"],
            "is_sharpe": alt.performance_is["is_sharpe"],
            "n_would_add": len(alt_set - cur_set),
            "n_would_remove": len(cur_set - alt_set),
        }

    return Holdings(
        config=cfg, members=sel.members, n_members=sel.n_members,
        window_info={"is_start": tree.is_start, "is_end": tree.is_end,
                     "oos_start": None, "oos_end": None,
                     "scheme": None, "window_no": None,
                     "note": "正式模式：IS＝錨點到最新可用月，依定義無 OOS"},
        # 🔴 只有 is_*，沒有 oos_*——`Holdings.has_oos` 因此為 False
        performance={**sel.performance_is,
                     "n_backfilled": sel.n_backfilled,
                     "target_total": sel.target_total},
        cluster_info=sel.cluster_info,
        alternative_groups=alts,
        tree_info={"tree_id": tree.tree_id, "source": tree.source, "k": tree.k,
                   "n_universe": tree.n_universe,
                   "is_start": tree.is_start, "is_end": tree.is_end,
                   "n_is_months": int(tree.wide_is.shape[1])},
        validation={
            "label": "✅ 方法已驗證",              # G7
            "reason": (f"方法論參數（group={cfg.group}／ratio={cfg.ratio}／"
                       f"allocation={cfg.allocation}）在 H-26/H-27 驗證集合內，"
                       f"且樹為 _frozen/stage3 主線樹（{tree.tree_id}，k={tree.k}），"
                       "是 H-03／ENB／群身份／M 系列共同的基礎"),
            "structural_caveat": (
                "本模式的**組合績效**無 OOS 可驗證——IS 用掉全部資料，依定義沒有"
                "樣本外可留（§7.4b）。已驗證的是方法，不是這一期的績效。"),
        },
        reference_oos=reference_oos_distribution(cfg),
    )


def run(cfg: RunConfig) -> Holdings:
    return _run_replay(cfg) if cfg.mode == "replay" else _run_live(cfg)
