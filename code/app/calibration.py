# -*- coding: utf-8 -*-
"""L2 · 校準監控（C5，應用層開發追蹤.md §3-C5，Phase B）

老師 9-8 沒直接提到這塊，是我們自己討論 IS/OOS 結構時新增的設計（§0.4）：
「今天」是第 46 個窗，OOS 還沒發生沒辦法驗證方法論——但可以拿它跟**歷史 45 窗
的分布**比，偏離太多就示警，這不是重新做統計檢定，是健康監控。

門檻算法（AL-02 review 定案）：跟 C2/C3 同一套「從 900 格歷史資料反推」的
方法，取這份 RunConfig 對應的 (market, group, ratio, allocation) 組合，
算 OOS CAGR／OOS Calmar 的 **p10 分位數**當警戒線——不是寫死某一組數字
（原本 AL-02 只算了 TW/legacy/equal 這一組），改成動態算，這樣任何 RunConfig
組合都適用。

⚠️ **本版是「一次性」比對，不是真正的「累積」監控**：C5 原始設計是「進行中
窗次的**累積**表現」，但累積需要 live 模式跑過好幾季、從稽核紀錄裡加總才有
意義——`replay` 模式一次就拿到完整 OOS 表現，沒有「進行中」這件事。這裡先用
「這格的完整 OOS 表現 vs 歷史 p10」示範門檻怎麼算，並在 B3 提早觸發時重置起點
（見應用層開發追蹤.md C5 說明）。

🔴 **2026-09-10（§8-R3）更正上一段的後半**：舊版寫「真正的累積邏輯留給 live 模式做
（讀 `audit.py` 的歷史紀錄加總）」——**那件事目前在資料上做不到**。要算實現績效得有
IS 結束之後的月報酬，但 `returns_monthly.parquet` 就到 IS 結束那個月（2025-12）為止。
⇒ 正式模式改成回傳 `status="tracking_started"`：把門檻立起來當日後基準，並誠實說明
為什麼現在不做判定，**不假裝算得出來**。真要做累積監控，前提是先把 `returns_monthly`
延長到 IS 結束之後（collector／研究部的事）。

⚠️ **門檻本身已被高估**（§8-R1）：p10 取自 walk-forward 的 OOS，而候選池是
「全期間 CAGR ≥ 基準」篩出來的（`passes_alpha_gate` 實測 100%），OOS 絕對數字
系統性偏高。方向上這讓門檻偏嚴（較容易示警），是保守的，但**不可宣稱
「只有 10% 的期間會示警」**。
"""
from __future__ import annotations

import dataclasses

import pandas as pd

from research import paths
from .config import RunConfig
from .engine import Holdings

DETAIL_PATH = paths.ROOT / "_analysis_outputs_robustness" / "walkforward_matrix_detail.csv"


@dataclasses.dataclass
class CalibrationThresholds:
    oos_cagr_p10: float
    oos_calmar_p10: float
    n_cells: int          # 這個門檻是從幾格歷史資料算出來的，太少要提醒


@dataclasses.dataclass
class CalibrationResult:
    thresholds: CalibrationThresholds
    oos_cagr: float | None
    oos_calmar: float | None
    below_cagr: bool
    below_calmar: bool
    #: 2026-09-10（§8-R3）：正式模式沒有 OOS，必須能誠實表達「還沒東西可比」
    #:   "evaluated"        replay：有完整 OOS，已實際比對
    #:   "tracking_started" live：門檻已算好當作日後基準，但**還沒有實現績效可比**
    status: str = "evaluated"
    note: str = ""

    @property
    def flagged(self) -> bool:
        return self.below_cagr or self.below_calmar

    @property
    def evaluable(self) -> bool:
        return self.status == "evaluated"


def get_thresholds(cfg: RunConfig) -> CalibrationThresholds:
    df = pd.read_csv(DETAIL_PATH)
    df["ratio"] = df["ratio"].astype(str)
    sub = df[(df.tree_key == cfg.market) & (df.group == cfg.group)
            & (df.ratio == cfg.ratio) & (df.allocation == cfg.allocation)]
    if len(sub) == 0:
        raise ValueError(f"歷史資料裡沒有這組設定可以反推門檻：market={cfg.market}, "
                         f"group={cfg.group}, ratio={cfg.ratio}, allocation={cfg.allocation}")
    # 2026-09-09 code review：MDD=0 除出來是 inf，會悄悄污染 p10 分位數（目前
    # 45 窗歷史資料最小 |oos_mdd|=2.07%，還沒踩到，但沒防呆）——把分母是0的列
    # 排除成 NaN，quantile() 預設會自動跳過 NaN，不是拿 inf 進去算。
    mdd_abs = sub.oos_mdd.abs()
    calmar = sub.oos_cagr / mdd_abs.where(mdd_abs != 0)
    return CalibrationThresholds(
        oos_cagr_p10=float(sub.oos_cagr.quantile(0.10)),
        oos_calmar_p10=float(calmar.quantile(0.10)),
        n_cells=len(sub),
    )


def check(holdings: Holdings) -> CalibrationResult:
    th = get_thresholds(holdings.config)

    # 🔴 2026-09-10（§8-R3）：正式模式依定義沒有 OOS——IS 用掉全部資料。
    # 舊版直接讀 performance["oos_cagr"] 會 KeyError。這裡不假裝算得出來，
    # 改成「把門檻立起來當日後基準」，並誠實說明為什麼現在還不能比。
    #
    # ⚠️ 為什麼不能「從 audit_log 累積實現績效」（本檔案舊 docstring 的規劃）：
    # 要算實現績效得有 IS 結束之後的月報酬，但 returns_monthly.parquet 就到
    # IS 結束那個月（2025-12）為止——資料上根本不存在。要真的做累積監控，
    # 前提是先把 returns_monthly 延長到 IS 結束之後，那是 collector／研究部的事。
    if not holdings.has_oos:
        return CalibrationResult(
            thresholds=th, oos_cagr=None, oos_calmar=None,
            below_cagr=False, below_calmar=False,
            status="tracking_started",
            note=(f"正式模式無 OOS（IS {holdings.window_info['is_start']}~"
                  f"{holdings.window_info['is_end']} 用掉全部資料）。門檻已立起來"
                  f"作為日後基準：OOS CAGR < {th.oos_cagr_p10:.2%} 或 "
                  f"OOS Calmar < {th.oos_calmar_p10:.3f} 時示警（取自 {th.n_cells} "
                  f"格歷史）。⚠️ 實現績效要等 returns_monthly 延長到 IS 結束之後"
                  f"才算得出來，目前資料上不存在，故本次不做判定。"
                  f"⚠️ 另注意此門檻本身已被候選池全樣本篩選高估（§8-R1）。"))

    oos_cagr = holdings.performance["oos_cagr"]
    oos_mdd = holdings.performance["oos_mdd"]
    if oos_mdd == 0:
        # 沒有回撤，Calmar 理論上無限大（正報酬）／無限小（負報酬）——用 inf
        # 表示，不能當成 0 處理（0 會被誤判成「這期表現很差」，語意剛好相反），
        # 也不能直接除以 0（2026-09-09 code review 抓到會 ZeroDivisionError 崩潰）
        oos_calmar = float("inf") if oos_cagr >= 0 else float("-inf")
    else:
        oos_calmar = oos_cagr / abs(oos_mdd)
    return CalibrationResult(
        thresholds=th, oos_cagr=oos_cagr, oos_calmar=oos_calmar,
        below_cagr=oos_cagr < th.oos_cagr_p10,
        below_calmar=oos_calmar < th.oos_calmar_p10,
    )
