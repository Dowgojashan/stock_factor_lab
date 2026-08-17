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
# 「季」語意的時序條件
#   吃 **公告點稀疏 frame**（未 ffill，由 MarketData.get_field_ann 提供）：
#   每個非 NaN 就是該股的一次財報公告，NaN＝當天沒有新財報。
#   分派規則見本檔末的 ANNOUNCEMENT_TYPES。
# ---------------------------

def _as_df(s) -> pd.DataFrame:
    """將輸入轉成 DataFrame（支援 Series/CustomDataFrame/pd.DataFrame）。保留原本 DataFrame 子類型。"""
    if isinstance(s, pd.Series):
        return s.to_frame()
    if isinstance(s, pd.DataFrame):
        return s  # 保留 CustomDataFrame 等子類型
    return pd.DataFrame(s)

# 以下「季」語意的條件，一律吃 **公告點稀疏 frame**（未 ffill）：
#   每個非 NaN 就是該股的一次財報公告，NaN 代表當天它沒有新財報。
#   由 `fcv_core.MarketData.get_field_ann()` 提供，`get_mask` 依
#   `condition_factory.ANNOUNCEMENT_TYPES` 分派。理由見該常數的說明。
#
# 因此這裡一律「逐股票沿著自己的公告序列」運算，不再有面板層壓縮、
# 也不再需要用「值有沒有變」去猜公告點——那兩個假設分別在美股與台股會出事。

def _shift_q(sp: pd.DataFrame, n: int) -> pd.DataFrame:
    """每一欄沿著自己的公告序列往回 n 次，再 ffill 成日頻。

    用 numpy 逐欄搬移而非 `sp.apply(lambda c: c.dropna().shift(n))`：
    後者每欄產生的 Series 索引都不同，pandas 要對齊 2,972 個相異索引，
    美股實測一個條件要 1.5 分鐘；numpy 版把時間壓到秒級。
    """
    arr = sp.to_numpy(dtype=float, copy=False)
    out = np.full(arr.shape, np.nan)
    for j in range(arr.shape[1]):
        idx = np.flatnonzero(~np.isnan(arr[:, j]))
        if idx.size > n:
            out[idx[n:], j] = arr[idx[:-n], j]
    return pd.DataFrame(out, index=sp.index, columns=sp.columns).ffill()


def _rollmax_q(sp: pd.DataFrame, n: int) -> pd.DataFrame:
    """每一欄沿著自己的公告序列取近 n 次的最大值（含當期），再 ffill 成日頻。"""
    arr = sp.to_numpy(dtype=float, copy=False)
    out = np.full(arr.shape, np.nan)
    for j in range(arr.shape[1]):
        idx = np.flatnonzero(~np.isnan(arr[:, j]))
        if idx.size >= n:
            v = arr[idx, j]
            # 滑動視窗最大值：用 stride 疊出 (m-n+1, n) 再取列最大，避免 pandas rolling 的開銷
            w = np.lib.stride_tricks.sliding_window_view(v, n).max(axis=1)
            out[idx[n - 1:], j] = w
    return pd.DataFrame(out, index=sp.index, columns=sp.columns).ffill()


def rise_q_factory(n: int):
    """較上 n 季（該股自己的公告更新點往回 n 次）上升：q(t) > q(t-n)"""
    def _cond(s):
        sp = _as_df(s)
        return (sp.ffill() > _shift_q(sp, n)).fillna(False)
    return _cond

def is_highest_q_factory(n: int):
    """近 n 季最大（該股自己的公告更新點 rolling max，含當期）：q(t) >= max(q(t-n+1..t))"""
    def _cond(s):
        sp = _as_df(s)
        return (sp.ffill() >= _rollmax_q(sp, n)).fillna(False)
    return _cond

def yoy_gt_factory(periods: int = 4):
    """較上年同季上升（該股自己的公告更新點往回 4 次，近似同季 YoY）：q(t) > q(t-4)"""
    def _cond(s):
        sp = _as_df(s)
        return (sp.ffill() > _shift_q(sp, periods)).fillna(False)
    return _cond

def _ytd_mean_q(sp: pd.DataFrame) -> pd.DataFrame:
    """每一欄沿著自己的公告序列算「當年至今的平均」，再 ffill 成日頻。"""
    def f(col):
        s = col.dropna()
        if s.empty:
            return s
        return s.groupby(s.index.year).expanding(min_periods=1).mean().droplevel(0)
    return sp.apply(f).ffill()


def _prev_year_mean_q(sp: pd.DataFrame) -> pd.DataFrame:
    """每一欄：去年**全年**的公告平均，對映到今年每個公告點，再 ffill 成日頻。"""
    def per_col(s):
        ym = s.groupby(s.index.year).mean()
        return pd.Series([ym.get(y - 1, np.nan) for y in s.index.year], index=s.index)
    return sp.apply(lambda c: per_col(c.dropna()) if c.notna().any() else c).ffill()


def _prev_year_same_k_q(sp: pd.DataFrame) -> pd.DataFrame:
    """去年「第 k 次公告」當下的 YTD 平均，對映到今年第 k 次公告。

    以「當年度第幾次公告」對齊，而不是日曆日期——各股公告節奏不同時這才對得起來。
    """
    def per_col(s):
        yr, kk = s.index.year, s.to_frame("v").groupby(s.index.year).cumcount()
        ytd = s.groupby(yr).expanding(min_periods=1).mean().droplevel(0)
        src = pd.Series(ytd.values, index=pd.MultiIndex.from_arrays([yr, kk]))
        src = src[~src.index.duplicated()]
        return pd.Series(src.reindex(pd.MultiIndex.from_arrays([yr - 1, kk])).values,
                         index=s.index)

    return sp.apply(lambda c: per_col(c.dropna()) if c.notna().any() else c).ffill()


def ytd_avg_gt_prev_year_avg_factory():
    """
    今年到目前平均 > 去年『全年』平均
    （注意：這不是「去年同樣時間平均」，若要同期間比較請用 ytd_avg_gt_prev_year_same_period_avg）
    """
    def _cond(s):
        sp = _as_df(s)
        return (_ytd_mean_q(sp) > _prev_year_mean_q(sp)).fillna(False)
    return _cond

def ytd_avg_gt_prev_year_same_period_avg_factory():
    """
    對齊簡報定義：
    今年到目前平均 > 去年同樣時間平均
    做法：在公告更新點上，按「當年度第 k 次公告」對齊去年的第 k 次公告，
         用 expanding mean（YTD 平均）逐點比較。
    """
    def _cond(s):
        sp = _as_df(s)
        return (_ytd_mean_q(sp) > _prev_year_same_k_q(sp)).fillna(False)
    return _cond


# ---------------------------
# 橫斷面分位 band（A3：q_band）
#   每期(每列)在同市場內對各公司做「橫斷面」百分位排名，判斷是否落在第 k 個 N 分位 band。
#   語意＝「同期同儕的相對位置」（factor 本義）：無前瞻、桶佔比恆≈1/N、跨市場可比。
#   ⚠️ 這是「橫斷面」運算，與 rise_q / is_highest_q / yoy_gt 等「時序」條件不同——
#      **必須**用密集 frame（get_field），不可用公告點稀疏 frame——
#      稀疏 frame 每列只有當天公告的少數公司，排名母體會爆掉。
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


# 需要「公告點稀疏 frame」（未 ffill）的條件型別。
#
# 為什麼要分兩種 frame（2026-08-17）：
#   `Data.get("report:X")` 在 get_data.py:136 對財報做 ffill，讓每個交易日都有值。
#   這對**橫斷面**條件（q_band）是必要的——每一列都要有完整的同期同儕才能排名。
#   但對**時序**條件（較上季升、近 n 季最高、YoY…）是災難：ffill 之後看不出
#   「這一天有沒有新財報」，只能用「值有沒有變」去猜，而那個猜測在
#   各公司公告日分散的市場（美股用真 filing_date）完全失效：
#       riseq1 通過率 0.8%（應約 50%）、qmax4 通過率 97.5%（應約 25%）
#
#   ffill **之前**的 frame 才有公告結構——每個非 NaN 就是該股的一次財報：
#       台股 106 列 × 1,775 檔（每檔中位 103 筆 ≈ 每季一次）
#       美股 6,944 列 × 2,812 檔（每檔的非 NaN 只落在它自己的 filing_date）
#
# 故 `fcv_core.MarketData.get_mask` 依這個集合分派：
#   在集合內 → `get_field_ann()`（稀疏、未 ffill）
#   不在     → `get_field()`（密集、已對齊交易日）
ANNOUNCEMENT_TYPES = frozenset({
    "rise_q", "is_highest_q", "yoy_gt",
    "ytd_avg_gt_prev_year_avg", "ytd_avg_gt_prev_year_same_period_avg",
})


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
                "cond": cond_func,
                "type": cond_type,
                # 「時序」條件必須拿**未 ffill 的公告點稀疏 frame**，否則在各公司公告日
                # 分散的市場（美股）會靜默失效——見 ANNOUNCEMENT_TYPES 的說明。
                "needs_ann": cond_type in ANNOUNCEMENT_TYPES,
            })
    return conditions
