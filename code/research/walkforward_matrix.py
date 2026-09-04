# -*- coding: utf-8 -*-
"""H-26／H-27／H-21 · Anchored walk-forward × 精選比例 × 分配方式 實驗矩陣

**2026-09-02 老師會議定案的畢業必考題主體。** 老師原話：

  「你這**只有移動一次**嘛。我如果能夠移動了兩三次，然後每次的答案都偏向[HRP]
    一次，**那 HRP 就的確比較好**」
  「**我如果 out-sample 多幾次，我的 HRP 精選都會[贏]**⋯那答案就出來了」
  「你這個應該是跑 **Anchored** 就可以了」
  「你的精選是抽 5% 而已？**抽前 5% 還是抽前 3% 還是前 1% 還是前 10%？**
    這個有時候**你越挑精兵反而是越爛**」
  「因為你精選的它是**一個需要研究的對象的參數**」

---------------------------------------------------------------------------
🔴 為什麼要交叉全跑，而不是「先找最優設定再驗證」
---------------------------------------------------------------------------
先跑一輪挑出最好的比例／分配方式，再拿它去跑 walk-forward，等於**用全部資料
挑參數、再用同一批資料驗證**，是上帝視角。本專案已經因為同一個錯誤撤銷過一次
提案（H-11 原提案用「OOS 要涵蓋 COVID 崩盤」選窗，見 contracts.HRP_IS_WINDOWS
註解）。老師也明講精選比例是「需要研究的對象的參數」＝實驗變數，不是待優化值。

故三個維度（窗口方案 × 精選比例 × 分配方式）**全部交叉**，每一格獨立評估，
結論看「A_hrp 在多少比例的格子裡贏」，不看單一最佳格。

實作上交叉也最便宜：**anchored 的樹只由 IS 結束點決定**，跟比例/分配/組別無關，
故 12 個方案共用端點後只需 11 次建樹（見 `unique_is_ends()`）。

---------------------------------------------------------------------------
窗口方案（機械規則，不看發生過什麼事）
---------------------------------------------------------------------------
規則：**IS 至少 min_is 個月 → 之後每 oos_len 個月切一個 OOS 區塊 → 切到資料用完；
剩餘不足 24 個月就併進最後一窗。** 12 個方案＝ min_is∈{72,96,120,144} ×
oos_len∈{24,36,48} 的完全交叉，無人為挑選。

⚠️ **窗口邊界不得用「涵蓋了哪個危機」來合理化**——那是上帝視角。方案剛好涵蓋
2018 貿易戰／2020 疫情／2022 熊市是**機械規則的結果，不是選窗的理由**，論文須
如此表述。

另加 1 組 **rolling** 對照（IS 長度固定、頭尾一起移動）：老師自己說過「台股後來
因為美中關係整個變了」，若結構真的變了，rolling（丟掉舊資料）該表現更好——這一
組直接回答那個問題。

---------------------------------------------------------------------------
精選比例：用「總選取量」當共同座標軸
---------------------------------------------------------------------------
🔴 **兩種分配方式必須釘死同一個總量才可比。** 若等量用「每群 5 支」(總 30)、
比例用「每群 5%」(總 334)，跑出比例較優也無法歸因——不知道是分法好還是多買了
300 支。故一律先定 `總量 = 宇宙 × ratio`，再用兩種方式把這個總量分下去，
兩邊選的檔數完全相同，唯一差別是分佈。

順帶修掉一個既有的不一致：現行 `m=5/群` 在三棵樹上其實是 0.45%(TW)／0.42%(US)／
**0.10%(XM)**——XM 只有 3 群所以只選 15 支，嚴格程度是台美的 4 倍多，三棵樹本來
就不可比。改用比例當座標軸後，群數只影響「怎麼分」，不再影響「分多少」。

`legacy` 是保留的參考點（總量＝現行 m=5 的檔數，TW30/US35/XM15），作用是讓新程式
能跟已報告給老師的 H-12 數字對照驗算——像用已知重量的砝碼校驗新體重計。

---------------------------------------------------------------------------
對照組（4 組，C_random 已移除）
---------------------------------------------------------------------------
  A_hrp        逐群配額 + H-10 貪婪多樣性規則（品質=Calmar，門檻=該群 avg_intra_corr）
  B_all        全宇宙等權（基準線，不受比例/分配影響，每窗只算一次）
  D_top_cagr   純 IS CAGR 前 N 名，無多樣性限制
  E_top_calmar 純 IS Calmar 前 N 名，無多樣性限制

D→E 拆出「品質指標(CAGR vs Calmar)」的貢獻，E→A 拆出「多樣性限制」本身的貢獻。
⚠️ **C_random 依使用者 2026-09-03 決定移除**：H-12 已實測它幾乎等於 B_all
（TW 17.56% vs 17.59%／US 25.33% vs 25.31%／XM 22.36% vs 22.52%），結論可直接
引用；且它每格要抽 200 次，是本矩陣最貴的一組。

---------------------------------------------------------------------------
其他設計決定
---------------------------------------------------------------------------
- **k（群數）是第四個維度**（H-26b，2026-09-04）：`fixed`＝H-03 用完整窗選出的
  6/7/3（形式上有前視偏誤）／`silhouette_is`＝每個 IS 窗只用該窗資料重選。
  🔴 實測**這個前視偏誤是實質的**：台股 15 個 IS 窗只有 4 個選出 k=6，且
  `2010-01~2017-12` 在 k=6 的輪廓係數是**負的**（−0.0046）——6 群在那個訓練窗上
  是錯的分法。故兩個 k_mode 都跑，差距本身即論文內容（同 H-18② 的處理方式）。
  兩者共用同一個 linkage，只是切的位置不同，建樹不需跑兩次。
- **等量分配的天花板**：小群不足配額時給滿，剩餘額度按其餘群大小比例重分配，
  並記錄 `n_capped_clusters` 讓天花板何時作用可查。實測 24 個格子只有
  「TW／10%／等量」一格會觸發（群5 只有 75 檔、配額要 111）。
- 權重一律等權（S-04 定案）。

用法：
    cd code
    python -m research.walkforward_matrix --list-schemes      # 只印窗口方案
    python -m research.walkforward_matrix --dry-run           # 印矩陣規模不建樹
    python -m research.walkforward_matrix --schemes E H       # 只跑指定方案
    python -m research.walkforward_matrix                     # 全跑
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, hrp, paths
from . import stage3_hrp as S3
from .cluster_representatives import select_representatives
from .four_group_control import (_cagr, _cagr_matrix, _mdd, _mdd_matrix, _portfolio_series,
                                 _sharpe, _small_enb)

TREES = ("TW", "US", "XM")
LEVEL = "L1"

#: 各樹的 IS 起點（anchored 的錨）＝ DD-03 共同窗起點，跟主線六棵樹一致。
#: 窗口方案的月數是以**台股的資料長度（228 個月）**定義的；美股 IS 從 2002-01
#: 起算會多吃 5 年，但 **OOS 用完全相同的絕對日曆區間**——沿用 H-11 v2 已定案的
#: 原則（`contracts.HRP_OOS_WINDOW` 註解），避免各市場各自挑窗。
ANCHOR_START = {"TW": "2007-01", "US": "2002-01", "XM": "2007-01"}
#: 方案的月數基準（台股/XM 的共同窗長度）
SCHEME_TOTAL_MONTHS = 228
SCHEME_BASE = pd.Period("2007-01", "M")

MIN_IS_GRID = (72, 96, 120, 144)      # 6/8/10/12 年
OOS_LEN_GRID = (24, 36, 48)           # 2/3/4 年
MIN_TAIL_MONTHS = 24                  # 尾巴不足此長度就併進最後一窗

#: rolling 對照組用的固定 IS 長度（挑 96＝8 年，跟 anchored 方案 D/E/F 的下限一致，
#: 這樣 rolling vs anchored 的第一窗起點相同，差別純粹是「後續窗要不要丟掉舊資料」）
ROLLING_IS_MONTHS = 96
ROLLING_OOS_LEN = 36

#: 精選比例（共同座標軸）。`legacy` 不是比例，是保留現行 m=5/群 的總量當校驗點。
LEGACY_M_PER_CLUSTER = 5
RATIO_GRID = ("legacy", 0.01, 0.03, 0.05, 0.10)
ALLOCATIONS = ("equal", "proportional")
GROUPS = ("A_hrp", "B_all", "D_top_cagr", "E_top_calmar")

#: 🔴 群數 k 的來源（H-26b，2026-09-04 新增的第四個維度）
#:   fixed         `stage3_hrp.L1_TARGET`＝6/7/3，是 H-03 用**完整窗**選出來的
#:                 → 早期 IS 窗等於用了未來資訊，**形式上是前視偏誤**
#:   silhouette_is 每個 IS 窗只用該窗資料重選 k（`k_stability.csv` 已算好並凍結）
#:                 → 乾淨版
#:
#: **兩個都跑，差距本身就是論文內容**——同 H-18② 的處理方式（凍結版 vs 滾動窗
#: 都跑，用差距說明問題嚴重性）。實測台股 15 個 IS 窗只有 4 個選出 k=6，且
#: `2010-01~2017-12` 這個窗在 k=6 的輪廓係數是**負的**（−0.0046），代表 6 群在
#: 那個訓練窗上是錯的分法——這個前視偏誤**是實質的，不是形式的**。
#:
#: ⚠️ 兩種 k_mode **共用同一個 linkage**（同一 IS 窗、同一相關矩陣），只是切的
#: 位置不同，故建樹不需跑兩次，只在 `run_one_window` 重切一次。
K_MODES = ("fixed", "silhouette_is")


# ============================================================================
# 窗口方案
# ============================================================================

def _blocks(total: int, min_is: int, oos_len: int,
            min_tail: int = MIN_TAIL_MONTHS) -> list[tuple[int, int]]:
    """回傳 [(is_end_offset, oos_len), ...]，offset 以月為單位、從 0 起算。

    機械規則，無人為調整：IS 至少 min_is → 每 oos_len 切一塊 → 剩餘不足 min_tail
    就併進本窗。
    """
    out: list[tuple[int, int]] = []
    cur = min_is
    while total - cur > 0:
        rem = total - cur
        L = oos_len if rem >= oos_len else rem
        if 0 < rem - L < min_tail:
            L = rem
        out.append((cur, L))
        cur += L
    return out


def build_schemes() -> pd.DataFrame:
    """12 個 anchored 方案（A~L）+ 1 個 rolling 對照（R）的逐窗明細。"""
    rows = []
    tag = 0
    for min_is in MIN_IS_GRID:
        for oos_len in OOS_LEN_GRID:
            blocks = _blocks(SCHEME_TOTAL_MONTHS, min_is, oos_len)
            if len(blocks) < 2:
                continue          # 只有一窗無法談「多次都贏」，不納入
            name = chr(65 + tag); tag += 1
            for i, (off, L) in enumerate(blocks, 1):
                rows.append({"scheme": name, "mode": "anchored",
                            "min_is_months": min_is, "oos_len_months": oos_len,
                            "window_no": i, "n_windows": len(blocks),
                            "is_end_offset": off, "oos_months": L})

    # rolling 對照：IS 長度固定，頭尾一起移動
    blocks = _blocks(SCHEME_TOTAL_MONTHS, ROLLING_IS_MONTHS, ROLLING_OOS_LEN)
    for i, (off, L) in enumerate(blocks, 1):
        rows.append({"scheme": "R", "mode": "rolling",
                    "min_is_months": ROLLING_IS_MONTHS, "oos_len_months": ROLLING_OOS_LEN,
                    "window_no": i, "n_windows": len(blocks),
                    "is_end_offset": off, "oos_months": L})
    return pd.DataFrame(rows)


def window_dates(row: pd.Series, tree_key: str) -> tuple[str, str, str, str]:
    """一個窗次在某棵樹上的實際 (is_start, is_end, oos_start, oos_end)。

    anchored：is_start 固定在該樹的錨點。
    rolling ：is_start 跟著往後移，維持固定 IS 長度——但**不得早於該樹的錨點**
              （錨點是 DD-03 共同窗起點，早於它就會出現缺值月份，破壞 PSD 保證）。
    """
    off = int(row.is_end_offset)
    is_end = SCHEME_BASE + off - 1
    oos_start = SCHEME_BASE + off
    oos_end = SCHEME_BASE + off + int(row.oos_months) - 1

    anchor = pd.Period(ANCHOR_START[tree_key], "M")
    if row["mode"] == "rolling":
        want = is_end - int(row.min_is_months) + 1
        is_start = max(want, anchor)
    else:
        is_start = anchor
    return str(is_start), str(is_end), str(oos_start), str(oos_end)


def unique_is_ends(schemes: pd.DataFrame) -> pd.DataFrame:
    """不重複的 (mode, is_end_offset, min_is_months) 組合——**枚舉用，不是建樹清單**。

    anchored 的樹只由 is_end 決定（is_start 固定）；rolling 的 is_start 會移動，
    故兩者分開列。

    ⚠️ **這個抽象 key 會高估建樹次數**：rolling 第一窗的 is_start 被夾到錨點後，
    可能跟 anchored 某一窗**是同一個訓練窗**（台股/跨市場都是 2007-01~2014-12），
    但抽象 key 不同。`run()` 已改用實際日期 (is_start, is_end) 當快取 key，
    正確去重（2026-09-04 code review）。本函式目前只供 `k_stability` 枚舉窗口用，
    該模組自己另外用 `seen` 集合依日期去重。
    """
    anc = (schemes[schemes["mode"] == "anchored"][["is_end_offset"]]
           .drop_duplicates().assign(mode="anchored", min_is_months=0))
    rol = (schemes[schemes["mode"] == "rolling"][["is_end_offset", "min_is_months"]]
           .drop_duplicates().assign(mode="rolling"))
    return pd.concat([anc, rol], ignore_index=True)[
        ["mode", "is_end_offset", "min_is_months"]]


# ============================================================================
# 配額分配（共同座標軸的核心）
# ============================================================================

def allocate(sizes: pd.Series, total: int, how: str) -> tuple[pd.Series, int]:
    """把總量 `total` 分配給各群，回傳 (每群配額, 觸發天花板的群數)。

    🔴 **兩種分配方式配出的總量必須完全相等**——這是共同座標軸的全部意義。
    若等量配 30 支、比例配 334 支，跑出比例較優也無法歸因（分不出是分法好還是
    多買了 300 支）。故兩邊都用整數配額 + 餘數分配，確保 `q.sum() == total`
    （唯一例外是 total 超過群成員總數，此時配到滿為止）。

    `equal`        每群 total/k；小群不足配額時給滿，餘額轉給還有空間的群，
                   反覆到沒有群超額為止（`n_capped` 記錄觸頂群數）
    `proportional` 按群大小切，用**最大餘額法**分配整數配額（不是各自四捨五入，
                   那會讓總量偏掉），每群至少 1——否則小群整個消失，
                   多樣性限制就失效了
    """
    sizes = sizes.astype(int)
    cap_total = int(sizes.sum())
    total = int(min(total, cap_total))
    q = pd.Series(0, index=sizes.index, dtype=int)

    if how == "proportional":
        raw = sizes / cap_total * total
        q = np.floor(raw).astype(int).clip(lower=1, upper=sizes)
        # 最大餘額法補足／削減，直到總量剛好等於 total
        while int(q.sum()) < total:
            room = (sizes - q)
            frac = (raw - q).where(room > 0, -np.inf)
            if not np.isfinite(frac).any():
                break
            q[frac.idxmax()] += 1
        while int(q.sum()) > total:
            over = (q - raw).where(q > 1, -np.inf)      # 不可低於下限 1
            if not np.isfinite(over).any():
                break
            q[over.idxmax()] -= 1
        return q, int((q >= sizes).sum())

    # equal：等額分配 + 天花板重分配
    remaining = total
    active = [c for c in sizes.index]
    n_capped = 0
    while remaining > 0 and active:
        share = remaining // len(active)
        if share == 0:
            # 餘數不足每群 1 支：發給「剩餘空間最大」的群，一次 1 支直到發完
            for cid in sorted(active, key=lambda c: sizes[c] - q[c], reverse=True):
                if remaining == 0:
                    break
                if q[cid] < sizes[cid]:
                    q[cid] += 1
                    remaining -= 1
            break
        progressed = False
        for cid in list(active):
            take = min(share, int(sizes[cid]) - int(q[cid]))
            if take > 0:
                q[cid] += take
                remaining -= take
                progressed = True
            if q[cid] >= sizes[cid]:                     # 這群滿了，退出下一輪
                active.remove(cid)
                n_capped += 1
        if not progressed:                               # 全部觸頂，無法再配
            break
    return q, n_capped


def target_total(ratio, n_universe: int, k: int) -> int:
    """該格子的總選取量。`legacy` ＝ 現行 m=5/群 的檔數（保留當校驗點）。"""
    if ratio == "legacy":
        return LEGACY_M_PER_CLUSTER * k
    return max(k, int(round(n_universe * float(ratio))))


# ============================================================================
# 建樹（快取）
# ============================================================================

def _load_inputs():
    months_long = pd.read_parquet(paths.STAGE1 / "returns_monthly.parquet")
    meta = pd.read_parquet(paths.STAGE1 / "returns_meta.parquet")
    idx = pd.read_parquet(paths.STAGE0 / "candidate_index.parquet")
    f_combo_map = (idx.market.astype(str) + "::" + idx.f_combo.astype(str))
    f_combo_map.index = idx.strategy_uid
    marks = pd.read_parquet(paths.STAGE1 / "strategy_marks.parquet")
    usable = set(marks.loc[marks.is_usable, C.PK])
    return months_long, meta[meta.strategy_uid.isin(usable)], f_combo_map


def build_tree_for_window(tree_key: str, is_start: str, is_end: str,
                          months_long, meta, f_combo_map, log=print) -> dict:
    """在指定 IS 窗上建一棵完整的 normal 樹（k 固定為 H-03 的正式群數）。

    策略宇宙的門檻沿用 `S3._tree_universe`（`hist_start <= 窗起點`）——rolling 的
    窗起點晚於錨點時宇宙會**變大**（更多策略來得及有完整歷史），這是真實的、
    不該人為壓回去；`n_strategies` 欄位會記錄實際數量供查。
    """
    uids = S3._tree_universe(tree_key, is_start, meta)
    wide = S3._pivot_window(months_long, uids, is_start, is_end)
    tree_id = f"{tree_key}_wf_{is_start}_{is_end}"
    return S3._build_tree(tree_id, tree_key, wide, f_combo_map, log=lambda *a, **k: None)


# ============================================================================
# 挑選與評估
# ============================================================================

def _pick_a(assign_tree: pd.DataFrame, meta_tree: pd.DataFrame, wide_is: pd.DataFrame,
            quality_is: pd.Series, quota: pd.Series,
            corr_full: np.ndarray, pos: pd.Series) -> tuple[list[str], int]:
    """A_hrp：逐群按配額用 H-10 貪婪多樣性規則挑代表。回傳 (名單, backfill 檔數)。

    ⚠️ `corr_full`／`pos` 由呼叫端傳入，**不在這裡算**——它們只跟 `wide_is` 有關，
    而本函式每窗會被呼叫 10 次（5 比例 × 2 分配）。原本在函式內算，XM 樹等於每窗
    重算 10 次 15,040×15,040 的相關矩陣（每次約 1.8GB），2026-09-04 code review
    抓到並外提。
    """
    picked_all: list[str] = []
    n_backfilled = 0
    for cid, g in assign_tree.groupby(f"cluster_{LEVEL}"):
        m = int(quota.get(int(cid), 0))
        if m <= 0:
            continue
        members = g[C.PK].tolist()
        qsub = quality_is.reindex(members).dropna()
        if qsub.empty:
            continue
        row = meta_tree[meta_tree.cluster_id == int(cid)]
        avg_intra = (float(row["avg_intra_corr"].iloc[0])
                     if len(row) and pd.notna(row["avg_intra_corr"].iloc[0]) else 1.0)
        member_idx = [pos[u] for u in members if u in pos.index]
        sub_corr = corr_full[np.ix_(member_idx, member_idx)]
        sub_index = pd.Index([u for u in members if u in pos.index])
        picked, backfilled = select_representatives(qsub, sub_corr, sub_index, m, avg_intra)
        picked_all += picked
        n_backfilled += len(backfilled)
    return picked_all, n_backfilled


def _evaluate(members: list[str], wide_is: pd.DataFrame, wide_oos: pd.DataFrame,
              cluster_map: pd.Series) -> dict:
    """一組成員在 IS/OOS 的等權組合績效 + 集中度診斷。"""
    if not members:
        # 回傳空 dict 會讓該列缺掉全部指標欄位、在 DataFrame 裡變成 NaN，
        # 而 n_members 是契約的非空欄位——與其讓它在契約驗證時才以難懂的訊息炸開，
        # 不如在源頭明確中止（2026-09-04 code review 補上，目前實測未發生過）。
        raise ValueError("組合成員為空，無法評估——請檢查配額分配或品質分數是否全為 NaN")
    p_is = _portfolio_series(wide_is, members)
    p_oos = _portfolio_series(wide_oos, members)
    cl = cluster_map.reindex(members).dropna()
    vc = cl.value_counts()
    return {
        "n_members": len(members),
        "is_cagr": _cagr(p_is), "is_mdd": _mdd(p_is), "is_sharpe": _sharpe(p_is),
        "oos_cagr": _cagr(p_oos), "oos_mdd": _mdd(p_oos), "oos_sharpe": _sharpe(p_oos),
        "oos_enb": _small_enb(wide_oos, members) if len(members) <= 400 else float("nan"),
        "n_clusters_covered": int(cl.nunique()),
        "max_cluster_share": float(vc.iloc[0] / len(cl)) if len(cl) else float("nan"),
    }


def _load_k_table(log=print) -> dict[tuple[str, str, str], int]:
    """讀 H-26b 診斷產出的「各 IS 窗只用該窗資料選出的 k」。

    刻意讀凍結檔而不是現算——silhouette_scan 在 XM（15,040 檔）上很慢，而
    `k_stability.py` 已經算過並寫了 manifest，直接沿用既省時又讓兩邊必然一致。
    """
    p = paths.ROOT / "_analysis_outputs_robustness" / "k_stability.csv"
    if not p.exists():
        raise FileNotFoundError(
            "找不到 k_stability.csv——請先執行 `python -m research.k_stability`，"
            "它提供 k_mode='silhouette_is' 所需的每窗 k")
    freeze.verify_inputs(paths.ROOT / "_analysis_outputs_robustness" / "_k_stability_manifest")
    df = pd.read_csv(p)
    tbl = {(r.tree_key, r.is_start, r.is_end): int(r.k_is_selected) for r in df.itertuples()}
    log(f"  載入 k_stability：{len(tbl)} 個 IS 窗的 IS-選 k")
    return tbl


def run_one_window(tree_key: str, srow: pd.Series, tree: dict,
                   months_long, k_table: dict, log=print) -> list[dict]:
    """一個 (樹 × 窗次) 下，跑完全部 k_mode × 比例 × 分配 × 組別 的格子。"""
    is_start, is_end, oos_start, oos_end = window_dates(srow, tree_key)
    assign = tree["assign"]
    assign = assign[assign.tree_id == tree["tree_id"]] if "tree_id" in assign.columns else assign
    cmeta = tree["cluster_meta"]
    cmeta = cmeta[cmeta.level == LEVEL] if "level" in cmeta.columns else cmeta

    uids = pd.Index(assign[C.PK])
    wide_is = S3._pivot_window(months_long, uids, is_start, is_end)
    w = months_long[months_long.strategy_uid.isin(set(uids))]
    w = w[(w.month >= pd.Period(oos_start, "M")) & (w.month <= pd.Period(oos_end, "M"))]
    wide_oos = w.pivot(index="strategy_uid", columns="month", values="ret")
    # ⚠️ IS 側由 `_pivot_window` 保證無 NaN，OOS 側原本沒有對應檢查（2026-09-04
    # code review 補上）。若 OOS 有缺值，`_portfolio_series` 的 mean(skipna=True)
    # 會在那些月份**靜默地少平均幾檔策略**，組合報酬悄悄失真而不報錯——正是本專案
    # 最忌諱的一類錯誤。DD-03 共同窗理論上保證涵蓋到 2025-12，這裡是防線不是修錯。
    n_nan = int(wide_oos.isna().sum().sum())
    n_missing = len(set(uids) - set(wide_oos.index))
    if n_nan or n_missing:
        raise ValueError(
            f"[{tree_key}/{srow.scheme}/w{int(srow.window_no)}] OOS 窗 "
            f"{oos_start}~{oos_end} 資料不完整：{n_nan} 個缺值、"
            f"{n_missing} 個策略完全無資料——DD-03 共同窗的保證被打破了")

    cagr_is = _cagr_matrix(wide_is)
    mdd_is = _mdd_matrix(wide_is)
    quality_is = cagr_is / mdd_is.abs().replace(0, np.nan)     # Calmar，同 H-10 口徑
    n_uni = len(uids)

    # 相關矩陣只跟 wide_is 有關，每窗算一次就好——見 `_pick_a` docstring
    corr_full = np.corrcoef(wide_is.to_numpy(dtype=np.float64))
    pos = pd.Series(range(len(wide_is.index)), index=wide_is.index)

    common0 = {"tree_key": tree_key, "scheme": srow.scheme, "mode": srow["mode"],
               "window_no": int(srow.window_no), "n_windows": int(srow.n_windows),
               "min_is_months": int(srow.min_is_months),
               "oos_len_months": int(srow.oos_len_months),
               "is_start": is_start, "is_end": is_end,
               "oos_start": oos_start, "oos_end": oos_end,
               "n_is_months": wide_is.shape[1], "n_oos_months": wide_oos.shape[1],
               "n_universe": n_uni}

    rows = []
    for k_mode in K_MODES:
        if k_mode == "fixed":
            a_tree, m_tree = assign, cmeta
        else:
            # 只用 IS 窗資料選出的 k，重切**同一棵 linkage**（不重建樹）。
            # `cut_clusters` 回傳的標籤順序對應建樹時 wide 的列順序，而 wide_is 是
            # 用 assign[PK] 依序 pivot 出來的（`_pivot_window` 有 .loc[uids] 固定順序），
            # 兩者一致，故可直接對位。
            k_is = k_table.get((tree_key, is_start, is_end))
            if k_is is None:
                raise KeyError(
                    f"k_stability 缺 ({tree_key}, {is_start}, {is_end}) 這個 IS 窗——"
                    f"請重跑 `python -m research.k_stability`")
            # 🔴 對位不變式：linkage 有 N-1 列合併紀錄，N 即建樹時的策略數。
            # 若它跟 wide_is 的列數不符，代表兩者不是同一批策略（例如零變異數
            # 排除規則變了），`labels` 就會**靜默地對錯策略**——分群結果全錯卻不報錯。
            # 這是靠註解維持的脆弱不變式，用執行期斷言鎖住。
            n_leaf = len(tree["link"]) + 1
            if n_leaf != len(wide_is):
                raise AssertionError(
                    f"[{tree_key} {is_start}~{is_end}] linkage 葉節點數 {n_leaf} "
                    f"!= wide_is 列數 {len(wide_is)}——重切的群標籤會對錯策略")
            labels = hrp.cut_clusters(tree["link"], k_is)
            m_tree, _ = S3._cluster_meta_and_corr(wide_is, corr_full, labels,
                                                  LEVEL, tree["tree_id"])
            a_tree = pd.DataFrame({C.PK: wide_is.index.to_numpy(),
                                  f"cluster_{LEVEL}": labels})

        cluster_map = a_tree.set_index(C.PK)[f"cluster_{LEVEL}"]
        sizes = a_tree.groupby(f"cluster_{LEVEL}").size()
        k = len(sizes)
        common = {**common0, "k_mode": k_mode, "n_clusters": k}

        # B_all 不受比例/分配影響，每個 k_mode 只算一次。
        # ⚠️ 兩個 k_mode 的 B_all **報酬類指標相同**（全宇宙不看分群），但
        # `n_clusters_covered`／`max_cluster_share` 會不同——它們是拿不同的群定義
        # 去算集中度。D/E 兩組同理。讀表時別誤以為那兩組完全不受 k_mode 影響。
        # ⚠️ 哨兵值不可用 "n/a"／"NA"／"null" 這類字串——它們在 pandas 的預設 NA 清單裡，
        # 存成 CSV 再 read_csv 回來會變成 NaN，觸發主鍵不得為空的契約違規（2026-09-03
        # 開發時實測踩到，跟 co_fail_peers／stable_core 是同一類往返陷阱）。
        b = _evaluate(list(uids), wide_is, wide_oos, cluster_map)
        rows.append({**common, "ratio": "all", "allocation": "unallocated", "group": "B_all",
                    "target_total": n_uni, "n_capped_clusters": 0, "n_backfilled": 0, **b})

        for ratio in RATIO_GRID:
            tot = target_total(ratio, n_uni, k)
            for how in ALLOCATIONS:
                quota, n_capped = allocate(sizes, tot, how)
                a_members, n_bf = _pick_a(a_tree, m_tree, wide_is, quality_is, quota,
                                          corr_full, pos)
                n_eff = len(a_members)      # D/E 用 A 的實際檔數對齊，比較才公平
                d_members = cagr_is.sort_values(ascending=False).index[:n_eff].tolist()
                e_members = quality_is.dropna().sort_values(ascending=False).index[:n_eff].tolist()
                for gname, mem, bf in (("A_hrp", a_members, n_bf),
                                      ("D_top_cagr", d_members, 0),
                                      ("E_top_calmar", e_members, 0)):
                    rows.append({**common, "ratio": str(ratio), "allocation": how,
                                "group": gname, "target_total": tot,
                                "n_capped_clusters": n_capped, "n_backfilled": bf,
                                **_evaluate(mem, wide_is, wide_oos, cluster_map)})
    return rows


# ============================================================================
# 主流程
# ============================================================================

def run(schemes_filter=None, trees=TREES, log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE0)
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE1 / "_marks")

    k_table = _load_k_table(log)
    schemes = build_schemes()
    if schemes_filter:
        schemes = schemes[schemes.scheme.isin(schemes_filter)]
    months_long, meta, f_combo_map = _load_inputs()

    log(f"窗口方案 {schemes.scheme.nunique()} 個｜總窗次 {len(schemes)}\n")

    # 🔴 先驗 k_table 涵蓋率再開始建樹——否則缺一個窗要等好幾個小時的建樹跑完
    # 才會在 run_one_window 裡拋 KeyError，白等一整輪（2026-09-04 code review 補上）。
    missing = sorted({(t,) + window_dates(r, t)[:2]
                      for t in trees for _, r in schemes.iterrows()} - set(k_table))
    if missing:
        raise KeyError(
            f"k_stability 缺 {len(missing)} 個 IS 窗，例如 {missing[:3]}——"
            f"請重跑 `python -m research.k_stability`（窗口方案改過就要重跑）")

    all_rows = []
    t0 = time.time()
    for tree_key in trees:
        # ⚠️ 快取 key 用**實際日期** (is_start, is_end)，不用抽象的
        # (mode, offset, min_is)——2026-09-04 code review 發現後者會重複建樹：
        # rolling 第一窗的 is_start 被夾到錨點後，跟 anchored 某一窗**是同一個訓練窗**
        # （台股/跨市場都是 2007-01~2014-12），但兩者的抽象 key 不同，各建了一次。
        # 樹只由 (樹, IS起點, IS終點) 決定，用日期當 key 才是它真正的身分。
        # ⚠️ 美股不會發生這種重複——它的錨點是 2002-01，rolling 第一窗是
        # 2007-01~2014-12（96月）跟 anchored 的 2002-01~2014-12（156月）確實不同。
        wanted: dict[tuple[str, str], None] = {}
        for _, srow in schemes.iterrows():
            s, e, _, _ = window_dates(srow, tree_key)
            wanted[(s, e)] = None
        log(f"  [{tree_key}] 需建樹 {len(wanted)} 棵（已依實際日期去重）")

        cache: dict[tuple[str, str], dict] = {}
        for is_start, is_end in wanted:
            tt = time.time()
            cache[(is_start, is_end)] = build_tree_for_window(
                tree_key, is_start, is_end, months_long, meta, f_combo_map, log)
            log(f"  [{tree_key}] 建樹 IS {is_start}~{is_end}  {time.time()-tt:.0f}s")

        for _, srow in schemes.iterrows():
            s, e, _, _ = window_dates(srow, tree_key)
            all_rows += run_one_window(tree_key, srow, cache[(s, e)], months_long,
                                       k_table, log)
        log(f"[{tree_key}] 完成，累計 {time.time()-t0:.0f}s\n")

    df = pd.DataFrame(all_rows)
    for col in ("tree_key", "scheme", "mode", "k_mode", "ratio", "allocation", "group"):
        df[col] = df[col].astype("category")
    C.validate(df, C.WALKFORWARD_MATRIX, strict_columns=True)
    log(f"✓ walkforward_matrix 契約通過（{len(df):,} 格）")

    out_dir = paths.ROOT / "_analysis_outputs_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_detail = out_dir / "walkforward_matrix_detail.csv"
    p_sum = out_dir / "walkforward_matrix_summary.csv"
    df.to_csv(p_detail, index=False, encoding="utf-8-sig")
    summary = summarize(df)
    summary.to_csv(p_sum, index=False, encoding="utf-8-sig")

    freeze.write_manifest(
        "walkforward_matrix", out_dir / "_walkforward_matrix_manifest",
        inputs=[paths.STAGE0 / "candidate_index.parquet",
               paths.STAGE1 / "returns_monthly.parquet",
               paths.STAGE1 / "returns_meta.parquet",
               paths.STAGE1 / "strategy_marks.parquet"],
        outputs=[p_detail, p_sum],
        params={"min_is_grid": list(MIN_IS_GRID), "oos_len_grid": list(OOS_LEN_GRID),
               "ratio_grid": [str(r) for r in RATIO_GRID],
               "allocations": list(ALLOCATIONS), "groups": list(GROUPS),
               "k_modes": list(K_MODES),
               "rolling_is_months": ROLLING_IS_MONTHS,
               "legacy_m_per_cluster": LEGACY_M_PER_CLUSTER,
               "min_tail_months": MIN_TAIL_MONTHS},
        notes="H-26/H-27/H-21：anchored walk-forward × 精選比例 × 分配方式的完整交叉。"
              "三維度不預先挑選（先挑=上帝視角，見模組docstring）。C_random已移除。",
    )
    log(f"→ {p_detail}\n→ {p_sum}")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """A_hrp 在每個維度切面下的 OOS 勝率。

    🔴 **三個指標都要看，不能只看 CAGR。** 老師明講 MDD 是我們唯一的代價
    （「我只有換股的話，我的代價就是我的 MDD 一定會比較慘」），而 H-12 已實測
    E_top_calmar 在 CAGR 上其實贏過 A_hrp——A 的優勢在**風險調整後**，不在
    絕對報酬。只報 CAGR 勝率會得到「HRP 沒用」的錯誤結論。

      win_*_cagr    OOS 年化報酬較高
      win_*_mdd     OOS 最大回撤較淺（MDD 是負數，故比大小）
      win_*_calmar  OOS Calmar（CAGR/|MDD|）較高 ← 老師「績效除以 MDD」的口徑
    """
    # ⚠️ k_mode 必須進 wkeys：加入 k_mode 維度後，B_all 每個窗次有兩列（各 k_mode
    # 一列），不含 k_mode 的索引不再唯一，`index.map()` 會拋 InvalidIndexError
    # （2026-09-04 開發時實測踩到）。
    keys = ["tree_key", "scheme", "k_mode", "ratio", "allocation", "window_no"]
    wkeys = ["tree_key", "scheme", "k_mode", "window_no"]
    piv = {m: df.pivot_table(index=keys, columns="group", values=f"oos_{m}", observed=True)
           for m in ("cagr", "mdd")}
    b = {m: df[df.group == "B_all"].set_index(wkeys)[f"oos_{m}"] for m in ("cagr", "mdd")}

    sub = piv["cagr"].reset_index()
    if "A_hrp" not in sub.columns:
        return pd.DataFrame()
    for m in ("cagr", "mdd"):
        flat = piv[m].reset_index()
        for gname in ("A_hrp", "D_top_cagr", "E_top_calmar"):
            if gname in flat.columns:
                sub[f"{gname}_{m}"] = flat[gname].to_numpy()
        sub[f"B_all_{m}"] = flat.set_index(wkeys).index.map(b[m])
    # ⚠️ 只保留 A 組真的有值的列——B_all 自己那一列的 ratio="all"，A 欄是 NaN，
    # 若不剔除會被 `NaN > x == False` 靜默算成敗場（開發時實測 20/22=90.9% 的來源）。
    sub = sub.dropna(subset=["A_hrp_cagr"])

    def _calmar(c, d):
        return c / d.abs().replace(0, np.nan)

    rows = []
    for dim in ("tree_key", "scheme", "k_mode", "ratio", "allocation", "window_no"):
        for val, g in sub.groupby(dim, observed=True):
            if len(g) == 0:
                continue
            rec = {"dimension": dim, "value": str(val), "n_cells": len(g)}
            for opp in ("B_all", "D_top_cagr", "E_top_calmar"):
                if f"{opp}_cagr" not in g.columns:
                    continue
                short = {"B_all": "B", "D_top_cagr": "D", "E_top_calmar": "E"}[opp]
                rec[f"win_{short}_cagr"] = float((g["A_hrp_cagr"] > g[f"{opp}_cagr"]).mean())
                rec[f"win_{short}_mdd"] = float((g["A_hrp_mdd"] > g[f"{opp}_mdd"]).mean())
                rec[f"win_{short}_calmar"] = float(
                    (_calmar(g["A_hrp_cagr"], g["A_hrp_mdd"])
                     > _calmar(g[f"{opp}_cagr"], g[f"{opp}_mdd"])).mean())
            rec["mean_cagr_A"] = float(g["A_hrp_cagr"].mean())
            rec["mean_mdd_A"] = float(g["A_hrp_mdd"].mean())
            rec["mean_cagr_B"] = float(g["B_all_cagr"].mean())
            rec["mean_mdd_B"] = float(g["B_all_mdd"].mean())
            rows.append(rec)
    out = pd.DataFrame(rows)

    # 🔴 A 組的 backfill 比例——這欄是解讀比例掃描的關鍵，不能只看勝率。
    # backfill = 多樣性門檻擋不住、退回純品質排序的檔數。實測 10% 時高達 56%，
    # 代表 A_hrp 在高比例下已經半退化成 E_top_calmar，「比例愈高表現愈差」有一部分
    # 是**多樣性機制失效**造成的，不只是「買太多稀釋報酬」。
    a = df[df.group == "A_hrp"].copy()
    a["_pct_bf"] = a.n_backfilled / a.n_members
    for dim in ("tree_key", "scheme", "k_mode", "ratio", "allocation", "window_no"):
        m = a.groupby(dim, observed=True)["_pct_bf"].mean()
        sel = out.dimension == dim
        out.loc[sel, "pct_backfilled_A"] = out.loc[sel, "value"].map(
            {str(k_): v for k_, v in m.items()})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.walkforward_matrix")
    ap.add_argument("--list-schemes", action="store_true", help="只印窗口方案明細")
    ap.add_argument("--dry-run", action="store_true", help="印矩陣規模，不建樹")
    ap.add_argument("--schemes", nargs="+", help="只跑指定方案（如 E H R）")
    ap.add_argument("--trees", nargs="+", default=list(TREES))
    a = ap.parse_args(argv)

    schemes = build_schemes()
    if a.schemes:
        schemes = schemes[schemes.scheme.isin(a.schemes)]

    if a.list_schemes or a.dry_run:
        print(f"{'方案':<5}{'模式':<10}{'IS下限':>7}{'OOS':>5}{'窗數':>5}  窗次明細")
        for name, g in schemes.groupby("scheme", observed=True):
            r0 = g.iloc[0]
            det = "  ".join(
                f"IS→{window_dates(r, 'TW')[1]}/OOS {window_dates(r, 'TW')[2]}~"
                f"{window_dates(r, 'TW')[3]}({int(r.oos_months)}m)" for _, r in g.iterrows())
            print(f"{name:<5}{r0['mode']:<10}{int(r0.min_is_months):>7}"
                  f"{int(r0.oos_len_months):>5}{len(g):>5}  {det}")
        # 建樹次數要用**實際日期**去重來數，不能用 unique_is_ends 的抽象 key——
        # 後者會把「rolling 第一窗被夾到錨點後跟 anchored 某窗相同」算成兩棵
        # （2026-09-04 code review 修正；台股/跨市場各少 1 棵）。
        n_build = {t: len({window_dates(r, t)[:2] for _, r in schemes.iterrows()})
                   for t in a.trees}
        n_cells = (len(schemes) * len(a.trees)
                  * len(K_MODES) * (1 + len(RATIO_GRID) * len(ALLOCATIONS) * 3))
        print(f"\n總窗次 {len(schemes)}｜需建樹 {n_build}＝共 {sum(n_build.values())} 次"
              f"｜k_mode {len(K_MODES)} 種｜評估格數 {n_cells:,}")
        return 0

    df = run(schemes_filter=a.schemes, trees=tuple(a.trees))
    s = summarize(df)
    print("\n" + "=" * 100)
    print("A_hrp 的 OOS 勝率（CAGR／MDD／Calmar 三個指標都看，理由見 summarize docstring）")
    print("=" * 100)
    cols = ["dimension", "value", "n_cells",
            "win_B_cagr", "win_B_mdd", "win_B_calmar",
            "win_E_cagr", "win_E_mdd", "win_E_calmar",
            "mean_cagr_A", "mean_mdd_A", "mean_cagr_B", "mean_mdd_B"]
    print(s[[c for c in cols if c in s.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
