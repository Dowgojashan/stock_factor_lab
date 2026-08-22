# -*- coding: utf-8 -*-
"""HRP（Hierarchical Risk Parity）演算法本體（W-03）

純演算法層，**與資料內容無關**——輸入是報酬矩陣，不管它來自合成資料還是
真實策略。刻意與 `stage3_hrp.py`（真實資料管線）分開，讓演算法邏輯可以
獨立用合成資料驗證正確性，不必等真實資料到位。

演算法鏈（研究部 v9 階段3 · López de Prado 2016）：
    報酬矩陣 (N策略 × T月)
      → correlation matrix (N×N)
      → distance: d(i,j) = sqrt(0.5*(1-ρ(i,j)))       ← 標準相關距離，滿足距離公理
      → linkage（階層聚類，scipy）
      → quasi-diagonalization（依樹狀圖重排，相似策略相鄰）
      → recursive bisection（由上而下切分、依逆變異數配權）→ HRP 權重

為什麼不能用一般 risk parity：傳統作法需要對相關矩陣求逆，策略高度相關時
數值不穩定（老師：「拿一個錯的就會錯得很離譜」）。HRP 不需要矩陣求逆。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage as scipy_linkage
from scipy.spatial.distance import squareform


# ============================================================================
# 相關 → 距離
# ============================================================================

def corr_to_distance(corr: np.ndarray) -> np.ndarray:
    """d(i,j) = sqrt(0.5*(1-ρ(i,j)))。對角線強制為 0（浮點誤差可能讓它變成極小負數）。"""
    d = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(d, 0.0)
    return d


def check_psd(corr: np.ndarray, tol: float = -1e-8) -> tuple[bool, float]:
    """相關矩陣是否半正定（PSD）。DD-03 的共同窗設計就是為了保證這件事。

    非 PSD 代表輸入用了 pairwise-complete（每對策略各自的重疊窗不同），
    這樣算出的距離可能違反三角不等式，餵給 linkage 只會得到一棵無意義的樹
    ——而且 scipy 不會報錯，錯誤是靜默的。這正是本專案最危險的一類 bug。
    """
    eigvals = np.linalg.eigvalsh(corr)
    min_eig = float(eigvals.min())
    return min_eig >= tol, min_eig


def check_triangle_inequality(dist: np.ndarray, n_samples: int = 2000,
                              seed: int = 0, tol: float = 1e-6) -> tuple[bool, int, float]:
    """對距離矩陣抽樣三元組，檢查 d(i,k) <= d(i,j) + d(j,k)。

    全量檢查是 O(N^3)，對 N~15000 不可行，用隨機抽樣當健康度指標。
    """
    n = dist.shape[0]
    rng = np.random.default_rng(seed)
    violations = 0
    max_excess = 0.0
    for _ in range(n_samples):
        i, j, k = rng.choice(n, size=3, replace=False)
        lhs = dist[i, k]
        rhs = dist[i, j] + dist[j, k]
        if lhs > rhs + tol:
            violations += 1
            max_excess = max(max_excess, lhs - rhs)
    return violations == 0, violations, max_excess


# ============================================================================
# Linkage + 準對角化
# ============================================================================

def build_linkage(dist: np.ndarray, method: str = "ward") -> np.ndarray:
    """對稱距離矩陣 → scipy linkage 矩陣。method: 'single'（HRP 原論文）或 'ward'（較平衡）。"""
    condensed = squareform(dist, checks=False)
    return scipy_linkage(condensed, method=method)


def quasi_diagonal_order(link: np.ndarray) -> list[int]:
    """依樹狀圖重排，相似策略相鄰。

    ⚠️ 不能用 `scipy.cluster.hierarchy.dendrogram()` 取 `leaves`——它內部是
    **遞迴**實作（`_dendrogram_calculate_info`），遞迴深度跟樹的葉節點數成正比。
    N ≈ 7,000 就會撞到 Python 預設遞迴限制（1000）而 RecursionError；本專案
    的樹最大到 N ≈ 15,800（XM），必炸無疑。改用 López de Prado (2016) 原書
    的**迭代版**演算法：從最頂層的合併開始，反覆把「還是群」的節點展開成它的
    兩個子節點，直到全部都是原始葉節點索引（< N）為止，全程用陣列操作、
    不遞迴，複雜度與遞迴版相同但沒有堆疊深度限制。
    """
    link = link.astype(int)
    n = link.shape[0] + 1
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    while sort_ix.max() >= n:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)     # 騰出插入新項目的空位
        expand = sort_ix[sort_ix >= n]                        # 還需要展開的（非葉節點）
        i = expand.index
        j = expand.to_numpy() - n
        sort_ix[i] = link[j, 0]                                # 該群的第一個子節點
        right = pd.Series(link[j, 1], index=i + 1)             # 第二個子節點，插在右邊
        sort_ix = pd.concat([sort_ix, right]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.astype(int).tolist()


def cophenetic_correlation(link: np.ndarray, dist: np.ndarray) -> float:
    """linkage 保真度：樹狀圖隱含的距離 vs 原始距離的相關。用來在 single/ward 間挑一個。

    ⚠️ `cophenet(Z, Y)` 給定 Y 時直接回傳 `(c, d)`——`c` 本身就是相關係數（純量），
    `d` 才是樹狀圖隱含的距離陣列。**不要**再對 `c` 和原始距離跑一次 `np.corrcoef`，
    那是拿純量當陣列用，形狀對不上。
    """
    from scipy.cluster.hierarchy import cophenet
    condensed = squareform(dist, checks=False)
    c, _ = cophenet(link, condensed)
    return float(c)


# ============================================================================
# 遞迴二分配權（HRP 權重核心）
# ============================================================================

def recursive_bisection_weights(cov: np.ndarray, sort_order: list[int]) -> np.ndarray:
    """López de Prado (2016) 遞迴二分：依準對角化順序，由上而下切半，
    每次依兩側「逆變異數」比例分配權重，不需要對協方差矩陣求逆。

    回傳：長度 N 的權重陣列，索引對應**原始**（未重排）順序。
    """
    n = cov.shape[0]
    w = pd.Series(1.0, index=sort_order)
    clusters = [sort_order]

    def cluster_var(items: list[int]) -> float:
        sub = cov[np.ix_(items, items)]
        ivp = 1.0 / np.diag(sub)          # 逆變異數（忽略協方差項，HRP 的簡化假設）
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    while clusters:
        # 每次都把「還大於 1 個成員」的群再切一次
        clusters = [c for c in clusters if len(c) > 1]
        if not clusters:
            break
        new_clusters = []
        for c in clusters:
            mid = len(c) // 2
            left, right = c[:mid], c[mid:]
            v_left, v_right = cluster_var(left), cluster_var(right)
            alpha = 1.0 - v_left / (v_left + v_right)   # 變異數小的一側分到較多權重
            w[left] *= alpha
            w[right] *= (1.0 - alpha)
            new_clusters += [left, right]
        clusters = new_clusters

    out = np.zeros(n)
    out[w.index.to_numpy()] = w.to_numpy()
    return out


# ============================================================================
# 分群層級切割
# ============================================================================

def cut_clusters(link: np.ndarray, n_clusters: int) -> np.ndarray:
    """依目標群數切割 dendrogram（`fcluster` 的 maxclust 準則），回傳 1-indexed 群標籤陣列。"""
    return fcluster(link, t=n_clusters, criterion="maxclust")


def adjusted_rand_index(labels_a: pd.Series, labels_b: pd.Series) -> float:
    """ARI，不依賴 sklearn（環境沒裝）。公式見 Hubert & Arabie (1985)。

    用於 DD-06：驗證 L3 分群是否貼近獨立 F 組合分群（客觀選擇分群層級的依據，
    比單純「看樹好不好看」更能辯護）。
    """
    ct = pd.crosstab(labels_a, labels_b).to_numpy().astype(np.int64)
    n = ct.sum()

    def comb2(x):
        return x * (x - 1) // 2

    sum_ij = comb2(ct).sum()
    sum_a = comb2(ct.sum(axis=1)).sum()
    sum_b = comb2(ct.sum(axis=0)).sum()
    expected = sum_a * sum_b / comb2(n) if n > 1 else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    if denom == 0:
        return 1.0 if sum_ij == expected else 0.0
    return float((sum_ij - expected) / denom)


# ============================================================================
# 一次跑完整套（供合成資料驗證與真實管線共用）
# ============================================================================

@dataclass
class HRPTreeResult:
    corr: np.ndarray
    dist: np.ndarray
    link: np.ndarray
    leaf_order: list[int]
    weights: np.ndarray
    psd_ok: bool
    min_eig: float
    cophenetic: float


def build_tree(returns: np.ndarray, method: str = "ward") -> HRPTreeResult:
    """`returns`：N策略 × T月，**不得含 NaN**（DD-03 的共同窗保證這件事）。"""
    if np.isnan(returns).any():
        raise ValueError("報酬矩陣含 NaN——輸入必須先用共同窗裁到完全對齊（DD-03），"
                         "不可用 pairwise-complete，否則相關矩陣不保證 PSD。")
    corr = np.corrcoef(returns)
    cov = np.cov(returns)
    psd_ok, min_eig = check_psd(corr)
    dist = corr_to_distance(corr)
    link = build_linkage(dist, method=method)
    leaf_order = quasi_diagonal_order(link)
    weights = recursive_bisection_weights(cov, leaf_order)
    coph = cophenetic_correlation(link, dist)
    return HRPTreeResult(corr=corr, dist=dist, link=link, leaf_order=leaf_order,
                         weights=weights, psd_ok=psd_ok, min_eig=min_eig, cophenetic=coph)
