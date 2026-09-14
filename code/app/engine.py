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
from research import stage3_hrp as S3
from research import walkforward_matrix as WF
from research.contracts import BENCHMARK_CAGR   # R8②：基準對照
from . import clustering
from .config import GROUP_LABELS, RunConfig

MEMBERS_PATH = paths.ROOT / "_analysis_outputs_robustness" / "walkforward_members.parquet"
DETAIL_PATH = paths.ROOT / "_analysis_outputs_robustness" / "walkforward_matrix_detail.csv"

#: F1（應用層開發追蹤.md §10.1／§10.5-R-A2）：跨市場混合的貨幣揭露——
#: 台股報酬以 TWD 計、美股以 USD 計，混合時直接加權組合，未做匯率調整
#: （v10 §8 既有限制，M-06）。抽成常數讓 XM 既有處與這裡共用同一份文字，
#: 不要各寫一份、以後改一邊忘了改另一邊。
FX_CAVEAT = ("跨市場混合的報酬序列以各自幣別計算後直接加權組合"
            "（台股 TWD／美股 USD），未做匯率調整——真實投資人配置美股"
            "會承擔匯率波動，這裡的混合績效無法反映這個效果。")

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
            "label": "✅ 方法已驗證",
            "reason": (f"這是歷史回測資料庫中第 {int(row['window_no'])} 個時間窗"
                      f"（{row['scheme']} 方案）的實際結果，屬於已驗證的歷史資料"),
            "structural_caveat": (
                "本期有實際的後續表現可供比對，但候選策略池本身是用長期累積績效"
                "篩選出來的，數字會系統性偏高，僅適合用於相對比較與基準校準，"
                "不代表未來實際可達到的報酬。"),
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
        "benchmark_cagr": float(BENCHMARK_CAGR.get(cfg.market, float("nan"))),
        "caveat": ("此為歷史參照數字，並非本期預測；候選策略池本身以長期績效篩選"
                   "而成，分布數字會系統性偏高，僅適合作相對比較與基準校準之用。"),
    }


def alt_group_win_rates(market: str, baseline: str = "A_hrp") -> dict | None:
    """🔴 2026-09-11（§9.7 S2）：§8-R12 定錨句要用的「目前方法對其他候選方案的
    勝率」，改成依市場現算，不可沿用寫死的 TW 數字（16.8%／13.2%）——三市場
    開放後那組數字若原樣顯示在 US／XM 的 memo 或 UI 上，會把 TW 專屬事實講成
    通用事實，是真正的正確性錯誤，不只是措辭問題。

    🔴 2026-09-14 code review 抓到的真 bug：`baseline` 原本寫死是 `"A_hrp"`，
    呼叫端（`memo._alt_context()`／`ui.py` 候選方案對照）卻把結果講成「目前
    採用的邏輯」——如果使用者選的 `group` 其實是 `D_top_cagr`／`E_top_calmar`
    （側邊欄「選股邏輯」下拉選單本來就三個都能選），memo 會講出「目前採用
    hrp」這種跟實際設定不符、甚至自己比自己的荒謬句子。現在 `baseline` 可以
    是任何一個 `_ALL_GROUPS` 成員，「其他兩個」動態算出，不寫死是哪兩個。

    實測三市場都有效（900 格 legacy~10% 逐格對照，`walkforward_matrix_detail.csv`，
    `baseline="A_hrp"` 時）：
        TW：A_hrp 贏 E_top_calmar  Calmar 16.8%／CAGR 13.2%；贏 D_top_cagr 27.0%／23.1%
        US：A_hrp 贏 E_top_calmar  Calmar  8.0%／CAGR  4.2%；贏 D_top_cagr 21.9%／ 2.4%
        XM：A_hrp 贏 E_top_calmar  Calmar 34.0%／CAGR 17.1%；贏 D_top_cagr 45.0%／17.7%
    """
    if baseline not in _ALL_GROUPS:
        raise ValueError(f"baseline 必須是 {_ALL_GROUPS} 之一，收到 {baseline!r}")
    others = [g for g in _ALL_GROUPS if g != baseline]

    df = pd.read_csv(DETAIL_PATH)
    df["ratio"] = df["ratio"].astype(str)
    df["calmar"] = df.oos_cagr / df.oos_mdd.abs().where(df.oos_mdd.abs() != 0)
    key = ["scheme", "window_no", "k_mode", "ratio", "allocation"]
    sub = df[(df.tree_key == market) & df.group.isin(_ALL_GROUPS)]
    piv = sub.pivot_table(index=key, columns="group", values=["calmar", "oos_cagr"],
                          aggfunc="first").dropna()
    if piv.empty:
        return None
    out = {"n_cells": int(len(piv)), "baseline": baseline, "others": others}
    for other in others:
        for metric in ("calmar", "oos_cagr"):
            out[f"{metric}_vs_{other}"] = float((piv[(metric, baseline)] > piv[(metric, other)]).mean())
    return out


def historical_same_setting_windows(cfg: RunConfig, as_of_is_end: str | None = None) -> list[dict]:
    """🆕 2026-09-11（§9.4 情境比對②）：同一組設定（market/group/ratio/allocation，
    跟 `reference_oos_distribution()` 用**完全相同的篩選條件**，故列表長度＝
    該函式的 `n_cells`——這裡給的是逐格明細，不是分位數彙總）在凍結矩陣裡逐窗的
    `is_end → oos` 實際結果，讓解釋 agent 能講「這個做法在歷史上 is_end 落在哪些
    年份、那次 oos 實際賺賠多少」，而不是只有一個聚合分布。

    ⚠️ 跟 `reference_oos_distribution()` 一樣只能當**歷史參照**，不是本次預測；
    且這份分布本身已被候選池全樣本篩選高估（§8-R1）。

    `as_of_is_end`（2026-09-15，驗證模式解釋 agent 用）：只保留 `is_end` **嚴格
    小於**這個日期的窗次——正式模式不傳（`None`，本來就是全樣本，沒有「未來
    窗次」這回事）；驗證模式必須傳目前這一窗自己的 `is_end`，否則會把還沒發生
    （相對這一窗的決策時點）的其他窗次結果也餵給 LLM，是跟群知識庫全樣本前視
    （R15）同一類問題。

    🔴 2026-09-15 code review 抓到的真 bug：原本用 `<=`（含等號），因為呼叫端
    傳的就是**這一窗自己的** `is_end`，這一窗自己那一列會通過篩選、混進「歷史
    對照窗次」清單——等於這一窗的真實 is_end→oos 結果被算了兩次：一次正確地
    當作「這一窗的績效」（`holdings.performance`），一次又被包裝成「獨立的
    歷史先例」之一餵給 LLM。改成 `<`（嚴格小於）排除自己，跟姊妹函式
    `mechanism_consistency()` 用 `oos_end <= as_of_is_end`（因為比較的是不同
    欄位，天生就不含自己）的安全性質對齊。
    """
    df = pd.read_csv(DETAIL_PATH)
    df["ratio"] = df["ratio"].astype(str)
    sub = df[(df.tree_key == cfg.market) & (df.group == cfg.group)
             & (df.ratio == cfg.ratio) & (df.allocation == cfg.allocation)]
    if as_of_is_end is not None:
        sub = sub[sub.is_end < as_of_is_end]
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


def mechanism_consistency(market: str, as_of_is_end: str | None = None) -> dict | None:
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

    `as_of_is_end`（2026-09-15，驗證模式解釋 agent 用）：只納入 `oos_end` 小於
    等於這個日期的窗次——這個統計本來就是跨全部窗次聚合，若不限制，驗證模式
    引用它等於在講「這個方法論長期穩不穩」時，用到了對這一窗而言還沒發生的
    其他窗次結果，是跟 R15 同一類問題（實作這次補丁時才發現也適用在這裡，
    不是原本 R15/R16/H8 清單裡點名的項目）。早期窗次限制後可能完全沒有
    可比較的「更早」窗次，此時直接回傳 `None`——這是誠實的結果，不是 bug。
    """
    df = pd.read_csv(DETAIL_PATH)
    df["ratio"] = df["ratio"].astype(str)
    if as_of_is_end is not None:
        df = df[df.oos_end <= as_of_is_end]
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
            "label": "✅ 方法已驗證",
            "reason": (f"本次的分群模型（{tree.tree_id}，共 {tree.k} 群）與選股規則"
                       "皆屬於已經過長期歷史回測驗證的方法"),
            "structural_caveat": (
                "本次分群與選股使用了全部可得的歷史資料，沒有保留可驗證的未來"
                "區間，因此**這一期的組合表現**目前無法驗證——已驗證的是方法本身"
                "在過去的有效性，不是這一期實際會賺多少。"),
        },
        reference_oos=reference_oos_distribution(cfg),
    )


def run(cfg: RunConfig) -> Holdings:
    return _run_replay(cfg) if cfg.mode == "replay" else _run_live(cfg)


# ============================================================================
# F1 · 可調式混合（ensemble，應用層開發追蹤.md §10.1／§10.5-R-A1~R-A4）
# ============================================================================
#
# 🔴 設計原則（R-A1）：**不改 `RunConfig`／`Holdings` 本身**。31 處既有程式碼
# （calibration.py／cli.py／engine.py／explain.py／memo.py／ui.py）直接讀
# `.config.market`／`.config.group`，假設剛好一個市場、一個方法——混合物件
# 完全不去碰這些既有路徑，只在新寫的顯示邏輯裡處理，31 處一行都不用改。
#
# 每個 leg 都是一份完整、合法、已經跑過 `run()` 的既有 `Holdings`，混合只發生
# 在算完之後這一層（線性組合已驗證方法的結果，不是重新選股、不是新的統計方法）。

@dataclasses.dataclass
class BlendLeg:
    holdings: Holdings
    weight: float


@dataclasses.dataclass
class BlendHoldings:
    """F1 混合結果。**刻意不繼承 `Holdings`**——下游任何預期單一
    market/group 的程式碼都不會意外吃到這個型別（見上方 R-A1 說明）。
    """
    legs: list[BlendLeg]
    stock_weights: dict[str, float]        # stock_symbol -> 混合後權重
    performance: dict                      # is_*或oos_*（依 has_oos）
    has_oos: bool
    reference_oos: dict | None
    validation: dict
    is_cross_market: bool

    @property
    def markets(self) -> list[str]:
        return sorted({leg.holdings.config.market for leg in self.legs})

    @property
    def n_unique_stocks(self) -> int:
        return len(self.stock_weights)


def _leg_monthly_series(holdings: Holdings, phase: str, months_long) -> pd.Series:
    """單一 leg 在 IS 或 OOS 期間的等權投組逐月報酬序列，供 F1 混合用。

    🔴 重用研究部既有公式（`WF._portfolio_series`），不重刻一份算法——兩份
    算法哪天對不上，會是很難抓到的錯（同一類教訓見 pitfalls.md 六）。
    """
    cfg = holdings.config
    if cfg.mode == "live":
        if phase != "is":
            raise ValueError("正式模式沒有樣本外，只能取 is 序列")
        tree = clustering.load_mainline_tree(cfg.market, months_long, log=lambda *a, **k: None)
        return WF._portfolio_series(tree.wide_is, holdings.members)
    wi = holdings.window_info
    start, end = ((wi["is_start"], wi["is_end"]) if phase == "is"
                 else (wi["oos_start"], wi["oos_end"]))
    wide = S3._pivot_window(months_long, pd.Index(holdings.members), start, end)
    return WF._portfolio_series(wide, holdings.members)


def blend_performance(legs: list[BlendLeg]) -> tuple[dict, bool]:
    """把各 leg 的逐月報酬序列依權重線性組合，回傳 (performance dict, has_oos)。

    🔴 R-A2：CAGR/MDD/Sharpe **不可**直接對點估計做加權平均（v10 §8 已有 XM
    那次的教訓）——必須先混合逐月序列，再用同一套公式重算。跨市場的兩腳位
    建模區間長度不同（TW/XM 228 個月、US 288 個月），混合前先對齊到共同重疊
    月份，不足 12 個月直接拒絕（R-A2 的靜默錯誤防線）。
    """
    if len(legs) < 2:
        raise ValueError("混合至少需要兩個腳位")
    total_w = sum(leg.weight for leg in legs)
    if abs(total_w - 1.0) > 1e-6:
        raise ValueError(f"混合比例總和必須是 100%（目前 {total_w:.1%}）")

    has_oos_flags = {leg.holdings.has_oos for leg in legs}
    if len(has_oos_flags) > 1:
        raise ValueError("要混合的腳位必須同時都有樣本外、或同時都沒有——不能混用"
                         "正式模式（無 OOS）與驗證模式（有 OOS）")
    has_oos = has_oos_flags.pop()
    phase = "oos" if has_oos else "is"

    months_long, _meta, _f_combo_map = clustering.load_inputs(log=lambda *a, **k: None)
    series_list = [(_leg_monthly_series(leg.holdings, phase, months_long), leg.weight)
                   for leg in legs]

    common_idx = series_list[0][0].index
    for s, _w in series_list[1:]:
        common_idx = common_idx.intersection(s.index)
    if len(common_idx) < 12:
        raise ValueError("各腳位的可比較期間不足 12 個月，無法混合"
                         "（可能是跨市場建模區間差異太大）")

    blended = sum(w * s.reindex(common_idx) for s, w in series_list)
    perf = {
        f"{phase}_cagr": WF._cagr(blended),
        f"{phase}_mdd": WF._mdd(blended),
        f"{phase}_sharpe": WF._sharpe(blended),
        "n_months_compared": int(len(common_idx)),
    }
    return perf, has_oos


def blend_reference_oos(legs: list[BlendLeg]) -> dict | None:
    """R-A4：混合後的歷史參照分布——不重新生成聯合分布（那需要真的把每一格
    歷史資料重跑一次），改用各 leg 的既有分布依權重加權平均關鍵分位數，並
    誠實標注這只是加權平均，不是重新算出的聯合分布。

    🔴 2026-09-15 code review 提醒（統計上的正確性，非程式邏輯錯誤）：對
    p10/p90 這種分位數做加權平均，**結果本身不是任何真實或假設聯合分布的
    分位數**（分位數不是線性運算，只有在各腳位完全同分布/完全相關等特殊
    情況下才會剛好等於聯合分布的分位數；一般情況下這只是一個沒有嚴謹統計
    意義的啟發式數字）。目前這組數字**沒有在畫面上直接顯示**（`ui.py` 只
    印出 `caveat` 這句話，不印 `oos_cagr_p10` 等實際數值），所以現況不構成
    誤導使用者的問題；但如果之後有人想把這些數字加進畫面顯示，一定要先
    看這段說明，不能當成「10% 分位數」的正常意義使用。"""
    refs = [(leg.holdings.reference_oos, leg.weight) for leg in legs]
    if any(r is None for r, _ in refs):
        return None
    blended = {
        "oos_cagr_p10": sum(r["oos_cagr_p10"] * w for r, w in refs),
        "oos_cagr_median": sum(r["oos_cagr_median"] * w for r, w in refs),
        "oos_cagr_p90": sum(r["oos_cagr_p90"] * w for r, w in refs),
        "oos_mdd_median": sum(r["oos_mdd_median"] * w for r, w in refs),
        "benchmark_cagr": sum(r["benchmark_cagr"] * w for r, w in refs),
        "n_cells": min(r["n_cells"] for r, _ in refs),
        "caveat": ("此為各腳位歷史分布關鍵分位數的加權平均，不是重新算出的"
                  "聯合分布，加權平均後的 p10／p90 本身也不是任何真實分布的"
                  "分位數，只是一個粗略的啟發式數字；且各腳位本身的分布已被"
                  "候選池全樣本篩選高估，只可作相對比較與基準校準之用，"
                  "不可解讀成「10% 機率會低於這個數字」。"),
    }
    return blended


def blend_holdings(legs: list[BlendLeg]) -> BlendHoldings:
    """組出 F1 的混合結果。呼叫端（`ui.py`）負責先把每個 leg 的 `Holdings`
    跑出來、把股票層級權重（`risk.assess_stock_level().weights`）解析好，
    再包成 `BlendLeg` 傳進來——這裡只做混合這一步的算術。
    """
    perf, has_oos = blend_performance(legs)
    ref = blend_reference_oos(legs) if not has_oos else None
    is_cross_market = len({leg.holdings.config.market for leg in legs}) > 1

    # G7 第四態（R-A4）：混合本身不是被驗證過的獨立方法，各腳位方法／市場
    # 才是已驗證的——措辭不可跟單一方法的「✅ 方法已驗證」混淆。
    leg_desc = "、".join(
        f"{leg.holdings.config.market}/"
        f"{GROUP_LABELS.get(leg.holdings.config.group, leg.holdings.config.group)}"
        f"×{leg.weight:.0%}" for leg in legs)
    validation = {
        "label": "🔷 已驗證方法／市場的加權組合",
        "reason": f"各腳位（{leg_desc}）本身都是已驗證的方法，但這個混合比例"
                 f"是使用者自訂的線性組合",
        "structural_caveat": ("各腳位方法本身都已驗證，但**這個特定的混合比例**"
                             "未被歷史回測驗證過整體表現——這是您自訂的線性組合，"
                             "不是被驗證過的獨立方法。是否採用這個比例屬於"
                             "您自己的判斷，系統不對這個比例的優劣背書。"),
    }
    if is_cross_market:
        validation["structural_caveat"] += "　" + FX_CAVEAT

    return BlendHoldings(
        legs=legs, stock_weights={}, performance=perf, has_oos=has_oos,
        reference_oos=ref, validation=validation, is_cross_market=is_cross_market)
