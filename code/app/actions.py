# -*- coding: utf-8 -*-
"""§3 動作空間（實戰監控 Agent 系統）。目前只實作 §7.7 要求的 A4 洩題防線：
`get_action_reference()`——動作參照**只能看到 `oos_end < as_of` 的窗次**，
物理上在資料載入器過濾掉，不靠 prompt 約束（設計文件 §7.7）。

🔴 window 4（scheme E，oos 2024-01~2025-12）的 `oos_end`＝實驗最後一天，在
整個實驗期間（2024-Q1~2025-Q4）任何 `as_of` 都不會小於它 ⇒ 用 `oos_end < as_of`
這個一般化的時序過濾規則，window 4 自然、永遠被排除，不需要另外寫「排除
window 4」的特判——這樣同一套邏輯往後也適用於任何新窗次，不必每次改規則。

W1／W3（市值傾斜等其餘維度）驗證失敗未啟用（見 D29）；**W2c（條件式市值
傾斜）已於 2026-09-18 前置驗證通過**（開發追蹤 D50），`w2c_reference()`
提供這個動作**唯一的一次性驗證結果**（不是像 `get_action_reference()` 那樣
隨 `as_of` 變化的即時查詢——W2c 的驗證是固定的歷史校準/前瞻結果，見
D50），供決策 agent 引用時遵守 §7.6「不得自行計算預期效果」的規則。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app import weights as weights_mod

DEFAULT_MATRIX_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "_analysis_outputs_robustness" / "walkforward_matrix_detail.csv"
)


def _oos_end_month_end(oos_end_col: pd.Series) -> pd.Series:
    """矩陣的 `oos_end` 是「YYYY-MM」月份格式，`pd.to_datetime` 會解析成
    **該月第一天**，不是月底。若拿它直接跟day-精度的 `as_of` 比較，同一個月
    裡會出現「窗次的 oos_end（月初）< as_of（月底）」的假象，實際上該窗
    OOS 期間要到月底才真正結束——這裡統一轉成月底日期再比較，
    否則會在月份邊界洩題（已用 as_of=2025-12-31 對 window4 實測抓到過一次）。
    """
    ts = pd.PeriodIndex(oos_end_col, freq="M").to_timestamp(how="end")
    return pd.Series(ts, index=oos_end_col.index)


def get_action_reference(
    ratio: str,
    allocation: str,
    as_of: str,
    *,
    tree_key: str = "TW",
    scheme: str = "E",
    group: str = "A_hrp",
    k_mode: str = "silhouette_is",
    matrix_path: Path = DEFAULT_MATRIX_PATH,
) -> list[dict]:
    """回傳某個動作（`ratio`/`allocation` 組合）在**嚴格早於 `as_of`** 的窗次
    裡的歷史表現，供 agent 評估「這個動作歷史上表現如何」時參照。

    🔴 物理過濾：`oos_end < as_of`（字串可比較的 YYYY-MM 或 YYYY-MM-DD 皆可，
    先轉 `pd.Timestamp`），不是靠 prompt 告訴 agent「不要看某一窗」——
    矩陣裡本來就沒有該窗的列，agent 拿到的 list 物理上不含它。
    """
    df = pd.read_csv(matrix_path)
    df = df[(df.tree_key == tree_key) & (df.scheme == scheme)
            & (df.group == group) & (df.k_mode == k_mode)
            & (df.ratio == ratio) & (df.allocation == allocation)]

    as_of_ts = pd.Timestamp(as_of)
    df = df[_oos_end_month_end(df["oos_end"]).lt(as_of_ts)]

    cols = ["window_no", "is_end", "oos_start", "oos_end",
            "oos_cagr", "oos_sharpe", "oos_mdd"]
    return df[cols].sort_values("window_no").to_dict("records")


def list_available_actions(
    as_of: str,
    *,
    tree_key: str = "TW",
    scheme: str = "E",
    group: str = "A_hrp",
    k_mode: str = "silhouette_is",
    matrix_path: Path = DEFAULT_MATRIX_PATH,
) -> list[dict]:
    """列出 `as_of` 當下每個 (ratio, allocation) 動作**可參照的窗次數**，
    方便 agent／人核對「這次能看到幾窗歷史」，避免誤以為看得到 window 4。
    """
    df = pd.read_csv(matrix_path)
    df = df[(df.tree_key == tree_key) & (df.scheme == scheme)
            & (df.group == group) & (df.k_mode == k_mode)]
    as_of_ts = pd.Timestamp(as_of)
    df = df[_oos_end_month_end(df["oos_end"]).lt(as_of_ts)]

    out = []
    for (r, a), gg in df.groupby(["ratio", "allocation"], observed=True):
        out.append({
            "ratio": r, "allocation": a,
            "n_windows_visible": len(gg),
            "window_nos_visible": sorted(gg["window_no"].tolist()),
            "mean_oos_cagr": float(gg["oos_cagr"].mean()) if len(gg) else None,
        })
    return out


def w2c_reference() -> dict:
    """W2c（條件式市值傾斜）的驗證結果（開發追蹤 D50，2026-09-18 凍結）。
    跟 `get_action_reference()`／`list_available_actions()` 不同：這不是隨
    `as_of` 變化的即時窗次查詢，是**一次性**的校準＋前瞻驗證結果，決策
    agent 引用時只能照抄這個 dict 的數字，不可自行外推到未驗證過的情境
    （§7.6）。`caveats` 欄位須完整帶給 agent，不可只挑正面數字。

    🔴🔴 2026-09-18（D52 抓到）：`caveats` 原本有一條「目前只到決策記錄
    層級，選這個動作不會真的改變後續季度的持股計算」——這是 D50 當時（執行
    層還沒做）的真實狀況，但 D51 做完執行層後這句話已經**過期且錯誤**，
    忘記回頭修正。正式重跑時 agent 在連續 5 個觸發季度全部選了保守的 A5、
    一次都沒選 W2c——很可能就是因為看到這句話，以為選 W2c 等於沒選。已拿掉
    這條過期 caveat，不是為了引導 agent 選 W2c，是修正一個真的過期的事實
    描述（選 W2c 現在真的會透過 `simulate.ActiveConfig` 改變下一季的持股）。

    🔴🔴 2026-10-19（論文session核對機制盤點表時連帶抓到）：下面
    `calibration_net_improvement_pp=23.48`／`forward_net_improvement_pp=21.98`
    兩個數字是 D50（2026-09-18）當時用**7%目標上限**算出來的，`weights.py`
    開頭docstring已經明文警告「2026-10-13使用者正式拍板改成30%後，既有用7%
    算出的歷史結果都要用30%重新驗證才能正式引用」——但這支函式當時忘記
    回頭同步，一直原封不動回傳這兩個舊數字，agent若在30%上限生效後的正式
    重跑裡引用它們，等於拿「舊政策下的結果」當「現行政策的證據」。D50的
    校準期（2024-Q1~2025-Q4）也早於2026-10-05的TW openSec_boost候選池
    重建，候選池版本同樣不保證一致。在30%上限下的新前瞻驗證數字補齊之前，
    這兩個數字**只能當歷史參考，不能當現行系統的證據**，已加進下面
    `caveats` 明確揭露；方法論本身（四輪查證、含真實交易成本）不受影響，
    只是這組具體數字的適用範圍需要重新確認。"""
    return {
        "mechanism": "W2c：M1-D 觸發時啟動市值傾斜（α=0）＋ 目標上限、"
                     "超過部分按比例重分配（cap-and-redistribute）；目標上限"
                     f"現為 {weights_mod.W2C_TARGET_CAP:.0%}（2026-10-13使用者拍板，"
                     "原為7%，見下方caveats）",
        "calibration_period": "2024-Q1~2025-Q4（8季，用於校準傾斜強度與上限；"
                              "🔴這組窗次早於2026-10-05的TW候選池重建）",
        "calibration_net_improvement_pp": 23.48,
        "forward_validation_period": "2026-Q1~2026-Q2（真前瞻，未參與任何校準）",
        "forward_net_improvement_pp": 21.98,
        "net_of_real_transaction_cost": True,
        "numbers_measured_under_cap": 0.07,
        "caveats": [
            "校準期與前瞻期本質上是同一個規模集中事件的不同階段，不是兩個獨立事件樣本",
            "前瞻驗證只有 2 季，統計證據仍薄弱",
            "季度制重新平衡無法 100% 保證任何時刻都不超過 8% 硬上限——"
            "7% 目標上限下仍有 1/6 觸發季度季底微幅超過（8.68%）",
            "🔴🔴 2026-10-19新增：上面calibration_net_improvement_pp/"
            "forward_net_improvement_pp兩個數字是在**7%舊上限**下算出來的，"
            "現行目標上限已改成30%（見mechanism欄位），且校準期可能早於"
            "目前的候選池版本——這兩個數字只能當歷史參考／方法論證據，"
            "不可當成30%上限下的現行預期效果，30%上限下的新前瞻驗證結果"
            "出來之前，選W2c的理由不可宣稱這組pp數字在現行設定下同樣成立",
            "🔴 2026-10-19補充（避免上一條單獨看顯得「上限沒驗證過」而過度"
            "保守）：**上限參數本身**（不是這組pp數字）另外做過獨立的敏感度"
            "掃描（`文件/研究進度報告_2026-10-05.md`§3.3/§3.3d，TW exclude_v1"
            "候選池、window4真實觸發窗口）——結論是①在維持現行gating（只有"
            "真的觸發才啟動）前提下，cap從7%調到30%/40%/不設上限，報酬與MDD"
            "都是平滑單調變化，沒有斷崖式風險跳升；②對從未觸發M1-D的歷史"
            "窗口（window1-3），cap設多寬完全不影響結果（W2c根本沒被啟動）。"
            "這代表「cap上限從7%放寬到30%」這個參數調整本身，在既有證據下"
            "風險是溫和、可預期的，跟「這組+23.48/+21.98pp的改善數字在30%下"
            "是否同樣成立」是兩個不同層次的問題——前者有壓力測試過的證據"
            "支持，後者才是上一條caveat講的、尚待30%上限下重新驗證的部分。"
            "⚠️ 這個敏感度測試用的是TW exclude_v1候選池，跟本次呼叫若為TW"
            "baseline池不保證逐檔持股一致，但cap_and_redistribute()本身的"
            "數學行為（cap越寬、被砍的股票越少）不隨候選池組成改變，定性"
            "結論合理適用。",
        ],
    }


def coverage_tilt_reference() -> dict:
    """Coverage Tilt（覆蓋率傾斜）的驗證結果，跟 `w2c_reference()` 同樣是**一次性**
    校準結果，不是隨 `as_of` 變化的即時查詢，決策 agent 引用時只能照抄這個 dict、
    不可自行外推（§7.6）。

    🔴🔴 跟 W2c 不是同一個成熟度等級，這點必須完整帶給 agent、不可只挑正面數字：
    設計文件§7.4b④記載2026-09-22前置驗證（TW window 1-3，β∈{0,1,2,5,10}）的結果是
    「β從0拉到10，效果幾乎不變、甚至略朝負向」，且「是否正式納入動作空間」原本
    留給10/7跟老師確認的開放問題，還沒有像W2c那樣走完四輪查證才拍板。這裡先接線，
    是遵照使用者2026-09-29的決定，不是因為已經驗證出正面效果。

    🔴 2026-10-19新增美股版（`_prelim_coverage_tilt_prevalidation_us.py`，
    跟TW版同方法論、同window1-3定義，只換market=US、N_MOMENTUM=6）：三窗
    平均年化超額報酬從+7.26%（β=0）**單調升到**+7.59%（β=10），跟TW的
    方向剛好相反（TW是微降、US是微升）。兩邊量級都在個位數bp~數十bp，
    同屬「在雜訊範圍內、不能斷言有效」的等級，但連「方向」都不一致，
    比單純兩邊都是null更值得誠實揭露——不可因為US這次是正向就宣稱
    CoverageTilt對美股「有效」，那會過度解讀一個雜訊量級的方向差異。
    XM仍完全沒有對應驗證。"""
    return {
        "mechanism": "Coverage Tilt：Hot Segment觸發時，依代表策略對Hot Segment的"
                     "覆蓋比例coverage_frac(r,t)加重權重，w_r∝(1/R)×(1+β·coverage_frac)，"
                     "β=5（已測試網格{0,1,2,5,10}的中段值，刻意保守）",
        "prevalidation_period": "TW/US window 1-3（2015-2023），跟W2c前置驗證同一套紀律"
                                "（只能用window 1-3，不可碰window 4）",
        "prevalidation_result": "TW：β從0拉到10，三窗平均年化超額報酬+11.29%→+11.14%"
                                "（降幅15bp/yr）。US：+7.26%→+7.59%（升幅33bp/yr）——"
                                "兩市場方向相反，量級都在雜訊範圍內，不能斷言任一方向有效",
        "root_cause": "window 1-3平均coverage_frac（TW約3.7~4.5%、US約1.9~2.8%）都"
                      "遠低於隨機持股的理論值（約28%），代表這些legacy策略的選股邏輯"
                      "本來就系統性避開動能股（負向選中，不是隨機分布）——這正是Hot"
                      "Segment診斷想抓的現象本身，但也因此window 1-3沒有「覆蓋率從"
                      "正常掉到異常」這種情境可以測，跟已棄用的W2/W3同一類「驗證期"
                      "本身沒有機制發揮作用的場景」問題",
        "caveats": [
            "效果驗證結果是null/微幅（TW微負、US微正，方向不一致）——跟W2c的"
            "+21.98pp前瞻驗證改善不是同一個等級的證據",
            "β=5沒有被驗證過是「最佳」或「有效」的值，只是網格中段的保守選擇",
            "是否正式納入動作空間，設計文件原本記載為「留待10/7跟老師確認」的開放"
            "問題，2026-09-29接線是使用者的決定，不代表已經有方法論上的定論",
            "XM市場完全沒有對應β驗證（TW/US已於2026-10-19都補齊window 1-3版本）",
        ],
    }


def rep_swap_reference() -> dict:
    """RepSwap（代表策略置換）的參照資料，跟 `coverage_tilt_reference()` 同樣是
    **一次性**說明文件，不是隨 `as_of` 變化的即時查詢。

    🔴🔴 2026-10-08 新增，比 Coverage Tilt 更早期的成熟度——**截至目前沒有任何
    歷史回測或前瞻驗證結果**，連 Coverage Tilt 的「β網格掃過、效果null/微負」
    這種程度的證據都還沒有。使用者2026-10-08的決定是「先把程式邏輯寫好、
    可以被選用，測試留到美股候選池重建完成、跟台股一起測」——這裡誠實記錄
    這個狀態，`caveats` 必須完整帶給 agent，不可因為是「新選項」就被選用時
    暗示已經驗證過。"""
    return {
        "mechanism": "RepSwap：Hot Segment觸發時，把目前代表策略裡「完全沒持有"
                     "任何Hot Segment成分股」的成員，換成同一個HRP群內尚未入選、"
                     "且確實持有Hot Segment成分股、品質（Calmar）最高的候選策略。"
                     "只換配額內的『誰』，不改配額本身（群間代表數m不變）、不改"
                     "權重公式（換完仍是純等權）——跟Coverage Tilt（調整策略間"
                     "權重比例）是完全不同的機制，見`weights.rep_swap_members()`。"
                     "每季最多置換3個代表（REP_SWAP_MAX_SWAPS），刻意保守控制周轉率。",
        "validation_status": "🔴 無任何歷史回測/前瞻驗證結果——這是全新動作，"
                             "2026-10-08剛完成程式實作，尚未實際跑過任何一季",
        "caveats": [
            "完全沒有被驗證過有效、無效、或有害——不是像Coverage Tilt那樣「測過"
            "但效果弱」，是「還沒測過」，證據強度比Coverage Tilt更低",
            "選它時理由必須誠實反映「這是未驗證的新動作，選用目的是增加系統"
            "可選動作的多樣性／示範用途，不能宣稱或暗示能改善覆蓋率或報酬」",
            "置換邏輯本身依賴HRP分群凍結結果（`_frozen/stage3/cluster_assign.parquet`），"
            "若候選池與分群建立時的池子不同步，可能有策略查不到自己的群"
            "（`rep_swap_members()`回傳的`n_no_cluster`會誠實記錄這種情況，不是"
            "靜默跳過）",
        ],
    }
