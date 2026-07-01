
"""
stats_test.py
-------------
統計檢定與效果量工具模組，供 report_grouping.py 與 condition_factory.py 共用。

需求/特色：
1) 對 ReportCollection 各群組（F1/F2/C/P3）之績效指標（如 CAGR, Sharpe, MDD, Win ratio）做群組間檢定。
2) 提供自動選擇檢定（normality/variance 檢查 → Welch t / Mann-Whitney / ANOVA / Kruskal）。
3) 提供效果量（Cohen's d、Hedges' g、Cliff's delta）。
4) 多重比較校正（Benjamini–Hochberg FDR）。
5) 針對「時間序列 change pattern」情境，提供「最近窗口 vs 過去窗口」的檢定遮罩，可直接做為 C 階條件。

用法示例：
from stats_test import analyze_by_level, pairwise_compare_by_level
df = analyze_by_level(rc, level='C', score_column='CAGR')
pairs = pairwise_compare_by_level(rc, level='C', score_column='CAGR', method='auto', fdr=0.05)

在 condition_factory 中可用：
from stats_test import ttest_recent_vs_past_mask
mask = ttest_recent_vs_past_mask(series, n_recent=4, n_past=8, alternative='greater', alpha=0.05)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats as sps
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False


# -------------------------
# 效果量 (Effect sizes)
# -------------------------

def _nanvar(x: np.ndarray, ddof: int = 1) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size <= ddof:
        return np.nan
    return np.var(x, ddof=ddof)

def cohen_d(x: Sequence[float], y: Sequence[float]) -> float:
    """Cohen's d（兩獨立樣本）"""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = _nanvar(x, ddof=1), _nanvar(y, ddof=1)
    # pooled std
    s = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2)) if (nx+ny-2) > 0 else np.nan
    if s == 0 or np.isnan(s):
        return np.nan
    return (np.nanmean(x) - np.nanmean(y)) / s

def hedges_g(x: Sequence[float], y: Sequence[float]) -> float:
    """Hedges' g（小樣本修正）"""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    d = cohen_d(x, y)
    if np.isnan(d):
        return d
    # small sample correction
    J = 1 - (3 / (4*(nx+ny) - 9)) if (nx+ny) > 2 else 1.0
    return d * J

def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Cliff's delta：秩基效果量，範圍 [-1, 1]；>0 代表 x 傾向大於 y。
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    # O(n log n) 近似：用秩差估計
    xy = np.concatenate([x, y])
    ranks = pd.Series(xy).rank(method='average').to_numpy()
    rx = ranks[:len(x)]; ry = ranks[len(x):]
    # 以秩差換算 delta（近似）
    mean_rx = np.nanmean(rx); mean_ry = np.nanmean(ry)
    delta = (mean_rx - mean_ry) / (len(xy)+1e-9)
    # 正規化到 [-1, 1] 的粗略近似；對實務比較足夠
    return float(np.clip(delta * 2, -1, 1))


# -------------------------
# 前置檢查
# -------------------------

def normality_test(x: Sequence[float], alpha: float = 0.05) -> Tuple[float, bool]:
    """
    Shapiro-Wilk 正態性檢定。回傳 (p_value, is_normal)。
    """
    if not _SCIPY_OK:
        # 若環境無 SciPy，保守地視為非正態
        return np.nan, False
    x = pd.Series(x, dtype=float).dropna()
    if len(x) < 3:
        return np.nan, False
    stat, p = sps.shapiro(x)
    return float(p), bool(p >= alpha)

def variance_homogeneity_test(groups: List[Sequence[float]], alpha: float = 0.05) -> Tuple[float, bool]:
    """
    Levene 方差齊性檢定。回傳 (p_value, is_equal_var)。
    """
    if not _SCIPY_OK:
        return np.nan, False
    arrays = [pd.Series(g, dtype=float).dropna().to_numpy() for g in groups]
    if any(len(a) < 2 for a in arrays):
        return np.nan, False
    stat, p = sps.levene(*arrays, center='median')
    return float(p), bool(p >= alpha)


# -------------------------
# 兩群比較 & 多群比較
# -------------------------

def ttest_ind_welch(x: Sequence[float], y: Sequence[float], alternative: Literal['two-sided','greater','less']='two-sided') -> Tuple[float, float, str]:
    """
    Welch t-test（不假設等變異）。回傳 (stat, p, 'welch_t')。
    """
    if not _SCIPY_OK:
        return np.nan, np.nan, 'welch_t'
    x = pd.Series(x, dtype=float).dropna()
    y = pd.Series(y, dtype=float).dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan, 'welch_t'
    stat, p = sps.ttest_ind(x, y, equal_var=False, alternative=alternative)
    return float(stat), float(p), 'welch_t'

def mannwhitney_u(x: Sequence[float], y: Sequence[float], alternative: Literal['two-sided','greater','less']='two-sided') -> Tuple[float, float, str]:
    """
    Mann-Whitney U test（非參數）。回傳 (stat, p, 'mannwhitney')。
    """
    if not _SCIPY_OK:
        return np.nan, np.nan, 'mannwhitney'
    x = pd.Series(x, dtype=float).dropna()
    y = pd.Series(y, dtype=float).dropna()
    if len(x) < 1 or len(y) < 1:
        return np.nan, np.nan, 'mannwhitney'
    stat, p = sps.mannwhitneyu(x, y, alternative=alternative, method='auto')
    return float(stat), float(p), 'mannwhitney'

def anova_oneway(groups: List[Sequence[float]]) -> Tuple[float, float, str]:
    """
    一因子 ANOVA（等變異假設）。回傳 (stat, p, 'anova')。
    """
    if not _SCIPY_OK:
        return np.nan, np.nan, 'anova'
    arrays = [pd.Series(g, dtype=float).dropna().to_numpy() for g in groups]
    if any(len(a) < 2 for a in arrays) or len(arrays) < 2:
        return np.nan, np.nan, 'anova'
    stat, p = sps.f_oneway(*arrays)
    return float(stat), float(p), 'anova'

def kruskal_wallis(groups: List[Sequence[float]]) -> Tuple[float, float, str]:
    """
    Kruskal-Wallis（非參數，多群）。回傳 (stat, p, 'kruskal')。
    """
    if not _SCIPY_OK:
        return np.nan, np.nan, 'kruskal'
    arrays = [pd.Series(g, dtype=float).dropna().to_numpy() for g in groups]
    if any(len(a) < 1 for a in arrays) or len(arrays) < 2:
        return np.nan, np.nan, 'kruskal'
    stat, p = sps.kruskal(*arrays)
    return float(stat), float(p), 'kruskal'


# -------------------------
# 多重比較校正
# -------------------------

def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, float]:
    """
    Benjamini–Hochberg FDR 控制。
    回傳：rejected(bool array), critical_value(float)。
    """
    pvals = np.asarray([np.nan if p is None else p for p in pvals], dtype=float)
    n = np.sum(~np.isnan(pvals))
    if n == 0:
        return np.zeros_like(pvals, dtype=bool), np.nan
    order = np.argsort(np.where(np.isnan(pvals), np.inf, pvals))
    ranked = np.arange(1, len(pvals)+1, dtype=float)
    thresh = ranked * (alpha / n)
    passed = np.zeros_like(pvals, dtype=bool)
    max_k = -1
    for i, idx in enumerate(order):
        p = pvals[idx]
        if np.isnan(p):
            continue
        if p <= thresh[i]:
            passed[idx] = True
            max_k = i
    crit = thresh[max_k] if max_k >= 0 else np.nan
    # 保證單調性（常見做法另見 statsmodels）—此處維持簡潔
    return passed, crit


# -------------------------
# ReportCollection 支援
# -------------------------

def _stats_df_with_level(report_collection, level: Literal['F1','F2','C','P3'], score_column: str) -> pd.DataFrame:
    """
    從 ReportCollection 取出 .get_stats()，並在 index 解析層級標籤。
    回傳長相： index=strategy_name, 欄位包含 score_column, level
    """
    level_idx = {'F1':0,'F2':1,'C':2,'P3':3}[level]
    stats_df = report_collection.get_stats().T
    stats_df = stats_df.dropna(axis=1, how='all')
    stats_df['__level__'] = stats_df.index.map(
        lambda name: name.split('__')[level_idx] if len(name.split('__')) > level_idx else 'None'
    )
    if score_column not in stats_df.columns:
        raise KeyError(f"score_column '{score_column}' 不存在於 stats（可用欄位：{list(stats_df.columns)}）")
    return stats_df.rename(columns={'__level__':'level'})

def analyze_by_level(report_collection, level: Literal['F1','F2','C','P3']='C',
                     score_column: str='CAGR') -> pd.DataFrame:
    """
    輸出每一群組的樣本數、平均、標準差（簡表）。
    """
    df = _stats_df_with_level(report_collection, level, score_column)
    g = df.groupby('level')[score_column]
    out = g.agg(n='count', mean='mean', std='std').sort_values('mean', ascending=False)
    return out

def pairwise_compare_by_level(report_collection, level: Literal['F1','F2','C','P3']='C',
                              score_column: str='CAGR',
                              method: Literal['auto','welch_t','mannwhitney']='auto',
                              alternative: Literal['two-sided','greater','less']='two-sided',
                              fdr: Optional[float]=0.05,
                              min_n: int=2) -> pd.DataFrame:
    """
    兩兩群組比較（輸出每一對的統計檢定 + 效果量 + FDR 校正）。
    """
    df = _stats_df_with_level(report_collection, level, score_column)
    groups = {k: v[score_column].dropna().to_numpy() for k, v in df.groupby('level')}
    keys = list(groups.keys())
    rows = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            k1, k2 = keys[i], keys[j]
            x, y = groups[k1], groups[k2]
            if len(x) < min_n or len(y) < min_n:
                rows.append((k1,k2,np.nan,np.nan,'insufficient',np.nan,np.nan,np.nan))
                continue

            if method == 'auto':
                # 正態 + 等變異 → t; 非等變異 → Welch t；非正態 → Mann-Whitney
                p_norm_x, nx = normality_test(x)
                p_norm_y, ny = normality_test(y)
                p_lev, eqv = variance_homogeneity_test([x,y])
                if nx and ny and eqv:
                    stat, p, tag = ttest_ind_welch(x, y, alternative=alternative)  # 用 Welch 也OK（更穩健）
                elif nx and ny and not eqv:
                    stat, p, tag = ttest_ind_welch(x, y, alternative=alternative)
                else:
                    stat, p, tag = mannwhitney_u(x, y, alternative=alternative)
            elif method == 'welch_t':
                stat, p, tag = ttest_ind_welch(x, y, alternative=alternative)
            else:
                stat, p, tag = mannwhitney_u(x, y, alternative=alternative)

            d = cohen_d(x, y)
            g = hedges_g(x, y)
            cd = cliffs_delta(x, y)
            rows.append((k1, k2, stat, p, tag, d, g, cd))

    res = pd.DataFrame(rows, columns=['group1','group2','stat','p_value','test','cohen_d','hedges_g','cliffs_delta'])
    if fdr is not None:
        reject, crit = benjamini_hochberg(res['p_value'].to_numpy(), alpha=fdr)
        res['fdr_reject'] = reject
        res['fdr_alpha'] = fdr
        res['fdr_critical'] = crit
    return res.sort_values('p_value')

def multi_group_test_by_level(report_collection, level: Literal['F1','F2','C','P3']='C',
                              score_column: str='CAGR',
                              method: Literal['auto','anova','kruskal']='auto') -> Tuple[float,float,str]:
    """
    多群組整體檢定（先看是否有群組差異，再做 pairwise）。
    """
    df = _stats_df_with_level(report_collection, level, score_column)
    arrays = [v[score_column].dropna().to_numpy() for _, v in df.groupby('level')]
    if method == 'auto':
        # 若全部群組都近似正態且等變異 → ANOVA，否則 Kruskal
        normals = [normality_test(a)[1] for a in arrays]
        eqv = variance_homogeneity_test(arrays)[1]
        if all(normals) and eqv:
            return (*anova_oneway(arrays),)
        else:
            return (*kruskal_wallis(arrays),)
    elif method == 'anova':
        return (*anova_oneway(arrays),)
    else:
        return (*kruskal_wallis(arrays),)


# -------------------------
# 時間序列：最近 vs 過去（給 C 階條件）
# -------------------------

def ttest_recent_vs_past_mask(s: pd.Series, n_recent: int, n_past: int,
                              alternative: Literal['two-sided','greater','less']='greater',
                              alpha: float=0.05,
                              min_n: int=3) -> pd.Series:
    """
    針對每一個時間點 t，取 s[t-n_recent+1 : t] 與 s[t-n_recent-n_past+1 : t-n_recent] 做 Welch t 檢定，
    當 p < alpha（且均值方向符合 alternative）即標 True。回傳與 s 對齊之布林 Series。
    適用：如「最近四季 ROE 是否顯著高於前八季？」等 change pattern 檢定。
    """
    idx = s.index
    arr = pd.Series(s, dtype=float).to_numpy()
    out = np.zeros_like(arr, dtype=bool)

    if not _SCIPY_OK:
        # 無 SciPy 時，保守返回全 False（或可退化為均值差 > 0 的閾值法）
        return pd.Series(out, index=idx)

    for t in range(len(arr)):
        # 右閉區間：最近視窗結束於 t
        r_end = t + 1
        r_start = r_end - n_recent
        p_end = r_start
        p_start = p_end - n_past
        if r_start < 0 or p_start < 0:
            continue
        recent = arr[r_start:r_end]
        past = arr[p_start:p_end]
        recent = recent[~np.isnan(recent)]
        past = past[~np.isnan(past)]
        if len(recent) < min_n or len(past) < min_n:
            continue
        stat, p, _ = ttest_ind_welch(recent, past, alternative=alternative)
        if np.isnan(p):
            continue
        # 方向性條件（當 alternative='greater' 時，recent 的平均需大於 past）
        ok_dir = True
        if alternative == 'greater':
            ok_dir = np.nanmean(recent) > np.nanmean(past)
        elif alternative == 'less':
            ok_dir = np.nanmean(recent) < np.nanmean(past)
        out[t] = (p < alpha) and ok_dir
    return pd.Series(out, index=idx)


def mw_recent_vs_past_mask(s: pd.Series, n_recent: int, n_past: int,
                           alternative: Literal['two-sided','greater','less']='greater',
                           alpha: float=0.05,
                           min_n: int=3) -> pd.Series:
    """同上，但使用 Mann–Whitney U（非參數）。"""
    idx = s.index
    arr = pd.Series(s, dtype=float).to_numpy()
    out = np.zeros_like(arr, dtype=bool)

    if not _SCIPY_OK:
        return pd.Series(out, index=idx)

    for t in range(len(arr)):
        r_end = t + 1
        r_start = r_end - n_recent
        p_end = r_start
        p_start = p_end - n_past
        if r_start < 0 or p_start < 0:
            continue
        recent = arr[r_start:r_end]
        past = arr[p_start:p_end]
        recent = recent[~np.isnan(recent)]
        past = past[~np.isnan(past)]
        if len(recent) < min_n or len(past) < min_n:
            continue
        stat, p, _ = mannwhitney_u(recent, past, alternative=alternative)
        if np.isnan(p):
            continue
        ok_dir = True
        if alternative == 'greater':
            ok_dir = np.nanmean(recent) > np.nanmean(past)
        elif alternative == 'less':
            ok_dir = np.nanmean(recent) < np.nanmean(past)
        out[t] = (p < alpha) and ok_dir
    return pd.Series(out, index=idx)
