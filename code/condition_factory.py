# 建立各種條件的工廠函式
# 注意：本檔案提供兩種數值比較語意——gt(>)/lt(<=) 為寬鬆比較（lt 會與 between 的下界重疊），
#      gte(>=)/lt_strict(<) 為不重疊／不留空的分箱比較；另含「季」語意的條件。

from __future__ import annotations

import weakref

import numpy as np
import pandas as pd


# ---------------------------
# 基礎數值比較（寬鬆比較：gt / lt）
# ---------------------------

def greater_than_factory(threshold):
    """s > threshold"""
    return lambda s: (s > threshold)

def less_than_factory(threshold):
    """s <= threshold（注意：會與 between 的下界重疊）"""
    return lambda s: (s <= threshold)

def between_factory(lower, upper):
    """左閉右開：lower <= s < upper"""
    return lambda s: (s >= lower) & (s < upper)


# ---------------------------
# 基礎數值比較（不重疊分箱：lt_strict / gte，避免重疊與邊界落空）
# ---------------------------

def lt_strict_factory(threshold):
    """s < threshold（用於第一箱，避免與 [threshold, ...) 重疊）"""
    return lambda s: (s < threshold)

def gte_factory(threshold):
    """s >= threshold（用於最後一箱，避免 threshold 本身落空）"""
    return lambda s: (s >= threshold)


# ---------------------------
# FinLabDataFrame 內建條件（保留）
# ---------------------------

def rise_factory(n):
    """FinLabDataFrame: s.rise(n)（注意：若 s 是日頻資料，n=1 代表比前 1 天）"""
    return lambda s: s.rise(n)

def is_largest_factory(n):
    """FinLabDataFrame: s.is_largest(n)（橫斷面：每期取前 n 大股票）"""
    return lambda s: s.is_largest(n)

def diff_abs_lt_factory(threshold):
    """|diff| < threshold（多用於穩定度）"""
    return lambda s: (s.diff().abs() < threshold)


# ---------------------------
# 「季」語意：先壓縮成公告更新點，再做時間序列判斷
#   適用於 report:* 這種財報欄位（通常已被 ffill 成日頻）
# ---------------------------

def _as_df(s) -> pd.DataFrame:
    """將輸入轉成 DataFrame（支援 Series/CustomDataFrame/pd.DataFrame）。保留原本 DataFrame 子類型。"""
    if isinstance(s, pd.Series):
        return s.to_frame()
    if isinstance(s, pd.DataFrame):
        return s  # 保留 CustomDataFrame 等子類型
    return pd.DataFrame(s)

def _compress_report_updates(s) -> pd.DataFrame:
    """
    將日頻（ffill）的財報序列壓縮成「公告更新點」序列：
    - 只保留「值有變動」的日期列（通常是公告日）
    - 再 ffill 一次，避免部分欄位在公告日出現 NaN
    """
    df = _as_df(s)
    changed = df.ne(df.shift(1))
    q = df.where(changed).dropna(how="all")
    return q.ffill()

def rise_q_factory(n: int):
    """較上 n 季（公告更新點往回 n 次）上升：q(t) > q(t-n)"""
    def _cond(s):
        df = _as_df(s)
        q = _compress_report_updates(df)
        cond_q = q > q.shift(n)
        out = cond_q.reindex(df.index, method="ffill").fillna(False)
        return out
    return _cond

def is_highest_q_factory(n: int):
    """近 n 季最大（公告更新點 rolling max，含當期）：q(t) >= max(q(t-n+1..t))"""
    def _cond(s):
        df = _as_df(s)
        q = _compress_report_updates(df)
        roll_max = q.rolling(window=n, min_periods=n).max()
        cond_q = q >= roll_max
        out = cond_q.reindex(df.index, method="ffill").fillna(False)
        return out
    return _cond

def yoy_gt_factory(periods: int = 4):
    """較上年同季上升（公告更新點往回 4 次，近似同季 YoY）：q(t) > q(t-4)"""
    def _cond(s):
        df = _as_df(s)
        q = _compress_report_updates(df)
        base = q.shift(periods)
        cond_q = q > base
        out = cond_q.reindex(df.index, method="ffill").fillna(False)
        return out
    return _cond

def ytd_avg_gt_prev_year_avg_factory():
    """
    今年到目前平均 > 去年『全年』平均
    （注意：這不是「去年同樣時間平均」，若要同期間比較請用 ytd_avg_gt_prev_year_same_period_avg）
    """
    def _cond(s):
        df = _as_df(s)
        q = _compress_report_updates(df)

        # 今年到目前平均（按年 expanding mean）
        ytd_avg = q.groupby(q.index.year).expanding(min_periods=1).mean()
        ytd_avg.index = ytd_avg.index.droplevel(0)  # 去掉 year

        # 去年全年平均（按年 mean）
        year_avg = q.groupby(q.index.year).mean()

        baseline = pd.DataFrame(index=q.index, columns=q.columns, dtype=float)
        years = pd.Index(q.index.year).unique()
        for y in years:
            if (y - 1) in year_avg.index:
                baseline.loc[q.index.year == y] = year_avg.loc[y - 1].values

        cond_q = ytd_avg > baseline
        out = cond_q.reindex(df.index, method="ffill").fillna(False)
        return out
    return _cond

def ytd_avg_gt_prev_year_same_period_avg_factory():
    """
    對齊簡報定義：
    今年到目前平均 > 去年同樣時間平均
    做法：在公告更新點上，按「當年度第 k 次公告」對齊去年的第 k 次公告，
         用 expanding mean（YTD 平均）逐點比較。
    """
    def _cond(s):
        df = _as_df(s)
        q = _compress_report_updates(df)

        cond_parts = []
        years = sorted(pd.Index(q.index.year).unique().tolist())
        for y in years:
            q_y = q[q.index.year == y]
            if q_y.empty:
                continue
            ytd_y = q_y.expanding(min_periods=1).mean()

            prev = q[q.index.year == (y - 1)]
            if prev.empty:
                # 沒有去年資料：全部 False
                cond_y = pd.DataFrame(False, index=q_y.index, columns=q_y.columns)
                cond_parts.append(cond_y)
                continue

            ytd_prev = prev.expanding(min_periods=1).mean()

            baseline_y = pd.DataFrame(np.nan, index=q_y.index, columns=q_y.columns, dtype=float)
            m = min(len(q_y), len(prev))
            # 以「同樣第 k 次公告」對齊
            baseline_y.iloc[:m, :] = ytd_prev.iloc[:m, :].values

            cond_y = (ytd_y > baseline_y).fillna(False)
            cond_parts.append(cond_y)

        if cond_parts:
            cond_q = pd.concat(cond_parts).sort_index()
        else:
            cond_q = pd.DataFrame(False, index=q.index, columns=q.columns)

        out = cond_q.reindex(df.index, method="ffill").fillna(False)
        return out
    return _cond


# ---------------------------
# 橫斷面分位 band（A3：q_band）
#   每期(每列)在同市場內對各公司做「橫斷面」百分位排名，判斷是否落在第 k 個 N 分位 band。
#   語意＝「同期同儕的相對位置」（factor 本義）：無前瞻、桶佔比恆≈1/N、跨市場可比。
#   ⚠️ 這是「橫斷面」運算，與 rise_q / is_highest_q / yoy_gt 等「時序」條件不同——
#      不可壓縮成公告更新點（_compress_report_updates）；那會讓同期各公司錯位。
#      直接對日頻 ffill 後的 frame 排名即可（frame 每格＝該公司該日已公告的最新值，
#      前瞻防護由讀取層 filing_date / adjust_index_of_report 保證）。
# ---------------------------

# frame 身分 → 該 frame 的橫斷面百分位排名。用意：同一因子的 N 個 band 共用同一次 rank
# （每因子只算一次、不重算 N 次）。以 weakref 綁定 frame 生命週期並在回收時自動清除，
# 避免用「殘留的全域 id 快取」造成 id 被回收再用時取到舊 rank。
_QBAND_RANK_CACHE: dict = {}


def _cached_pct_rank(s):
    """回傳 s 的橫斷面百分位排名 s.rank(axis=1, pct=True)；同一個 frame 物件只算一次。"""
    key = id(s)
    entry = _QBAND_RANK_CACHE.get(key)
    if entry is not None and entry[0]() is s:  # 確認是同一個「活著的」物件，非 id 回收再用
        return entry[1]
    pct = s.rank(axis=1, pct=True)
    try:
        _QBAND_RANK_CACHE[key] = (
            weakref.ref(s, lambda _ref: _QBAND_RANK_CACHE.pop(key, None)),
            pct,
        )
    except TypeError:
        pass  # s 不可被 weak-reference：不快取即可，正確性不受影響
    return pct


def q_band_factory(k, n):
    """橫斷面第 k 個 n 分位 band（k=0..n-1）：百分位 pct 落在 (k/n, (k+1)/n]。
    - 不重疊、不漏：N 個 band 剛好把 (0,1] 分成 N 段，每個非 NaN 公司每期恰落一桶。
    - 首桶下界為開區間，但 pct 最小值＝1/有效家數 > 0，故最小者仍入桶 0；末桶上界含 1.0。
    - NaN（無資料 / 已下市）公司 rank 為 NaN → 不入任何桶（天然排除）。
    """
    k = int(k)
    n = int(n)
    lo, hi = k / n, (k + 1) / n

    def _cond(s):
        pct = _cached_pct_rank(s)
        return (pct > lo) & (pct <= hi)

    return _cond


# ---------------------------
# 所有支援的條件工廠函式
# ---------------------------

CONDITION_FACTORY = {
    # 寬鬆比較
    "gt": greater_than_factory,
    "lt": less_than_factory,

    # 不重疊分箱（推薦）
    "gte": gte_factory,
    "lt_strict": lt_strict_factory,

    # 區間
    "between": between_factory,

    # FinLab 內建
    "rise": rise_factory,
    "is_largest": is_largest_factory,
    "diff_abs_lt": diff_abs_lt_factory,

    # 「季」語意
    "rise_q": rise_q_factory,
    "is_highest_q": is_highest_q_factory,
    "yoy_gt": yoy_gt_factory,
    "ytd_avg_gt_prev_year_avg": ytd_avg_gt_prev_year_avg_factory,
    "ytd_avg_gt_prev_year_same_period_avg": ytd_avg_gt_prev_year_same_period_avg_factory,

    # 橫斷面分位 band（A3）
    "q_band": q_band_factory,
}


# ---------------------------
# 自動產生條件名稱（依據因子名與條件類型）
# ---------------------------

def auto_name(factor: str, cond_type: str, args: list) -> str:
    """依因子名與條件類型組出條件名稱，例如 ROE + is_highest_q + [4] 會得到 ROE_qmax4。"""
    factor_upper = factor.upper()
    if cond_type == "between":
        return f"{factor_upper}_{args[0]}_{args[1]}"
    elif cond_type == "gt":
        return f"{factor_upper}_>{args[0]}"
    elif cond_type == "gte":
        return f"{factor_upper}_>={args[0]}"
    elif cond_type == "lt":
        return f"{factor_upper}_<={args[0]}"
    elif cond_type == "lt_strict":
        return f"{factor_upper}_<{args[0]}"
    elif cond_type == "rise":
        return f"{factor_upper}_rise"
    elif cond_type == "rise_q":
        return f"{factor_upper}_riseq{args[0] if args else ''}"
    elif cond_type == "is_largest":
        return f"{factor_upper}_top{args[0]}"
    elif cond_type == "is_highest_q":
        return f"{factor_upper}_qmax{args[0]}"
    elif cond_type == "diff_abs_lt":
        return f"{factor_upper}_stable"
    elif cond_type == "yoy_gt":
        return f"{factor_upper}_yoy"
    elif cond_type == "ytd_avg_gt_prev_year_avg":
        return f"{factor_upper}_ytdavg_gt_lyavg"
    elif cond_type == "ytd_avg_gt_prev_year_same_period_avg":
        return f"{factor_upper}_ytdavg_gt_lyytdavg"
    elif cond_type == "q_band":
        return f"{factor_upper}_qb{args[0]}of{args[1]}"
    else:
        return f"{factor_upper}_{cond_type}_{'_'.join(map(str, args))}"


def build_conditions(grouped_defs: dict) -> list:
    """讀取分組條件定義（factor 對應 field 與 conditions），逐條建立 {name, field, cond} 清單。
    其中 cond 是實際的判斷函式，由 CONDITION_FACTORY 依 type 產生。"""
    conditions = []
    for factor, factor_info in grouped_defs.items():
        field = factor_info["field"]  # 每個因子對應的 dataframe 欄位名稱
        cond_list = factor_info["conditions"]  # 該因子底下的條件清單

        for cond_def in cond_list:
            cond_type = cond_def["type"]  # 條件類型
            args = cond_def.get("args", [])  # 條件所需參數
            prefix = cond_def.get("prefix", "")  # 選用的前綴
            name = auto_name(prefix + factor, cond_type, args)

            factory_func = CONDITION_FACTORY[cond_type]  # 對應的工廠
            cond_func = factory_func(*args)  # 建立條件函式

            conditions.append({
                "name": name,
                "field": field,
                "cond": cond_func
            })
    return conditions
