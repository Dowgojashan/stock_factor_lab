# -*- coding: utf-8 -*-
from __future__ import annotations

# 安全載入 ReportCollection：不硬編 sys.path
try:
    from combinations import ReportCollection
except Exception:
    # 容許本檔被放在子資料夾時仍能找到上層模組
    import os, sys
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if base not in sys.path:
        sys.path.append(base)
    from combinations import ReportCollection  # 補上路徑後再載入一次

from typing import Optional, Dict, Callable, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import random

# 分組統計檢定的對外介面，底層運算委派給 stats_test
from stats_test import analyze_by_level as st_analyze_by_level, \
                        pairwise_compare_by_level as st_pairwise_compare_by_level, \
                        multi_group_test_by_level as st_multi_group_test_by_level

def analyze_level_summary(report_collection, level="C", score_column="CAGR"):
    """各群組 n/mean/std 簡表"""
    return st_analyze_by_level(report_collection, level=level, score_column=score_column)

def level_overall_test(report_collection, level="C", score_column="CAGR", method="auto"):
    """多群組整體檢定（ANOVA / Kruskal）"""
    stat, p, tag = st_multi_group_test_by_level(report_collection, level=level, score_column=score_column, method=method)
    print(f"[{tag}] stat={stat:.3f}, p={p:.5f}")
    return stat, p, tag

def level_pairwise_tests(report_collection, level="C", score_column="CAGR",
                         method="auto", alternative="two-sided", fdr=0.05, min_n=2):
    """兩兩群組比較（含效果量與 FDR）"""
    df = st_pairwise_compare_by_level(report_collection, level=level, score_column=score_column,
                                      method=method, alternative=alternative, fdr=fdr, min_n=min_n)
    return df

# ===============================================================
# 基本分組與視覺化
# ===============================================================
def group_report_by_prefix(report_collection: ReportCollection, level: str = "C", display_plot: bool = True):
    """
    依策略名稱中的片段進行分組，例如：F1__F2__C__P3
    level 可為 "F1" / "F2" / "C" / "P3"
    """
    grouped_reports = {}
    level_index_map = {"F1": 0, "F2": 1, "C": 2, "P3": 3}
    assert level in level_index_map, f" 不支援的層級 '{level}'，請使用：{list(level_index_map.keys())}"
    level_index = level_index_map[level]

    for name, report in report_collection.reports.items():
        parts = name.split("__")
        key = parts[level_index] if len(parts) > level_index else "None"
        grouped_reports.setdefault(key, {})[name] = report

    grouped_collections = {}
    for group_key, sub_reports in grouped_reports.items():
        print(f"- 分組：{group_key}")
        sub_collection = ReportCollection(sub_reports)
        grouped_collections[group_key] = sub_collection
        display(sub_collection.get_stats())
        if display_plot:
            sub_collection.plot_stats(mode="bar").show()

    return grouped_collections


def plot_group_stats_heatmap(report_collection: ReportCollection,
                             sort_by: str = 'CAGR',
                             top_n: int = 15,
                             indicators: Optional[List[str]] = None,
                             figsize=(12, 6),
                             normalize: bool = True,
                             annot_raw: bool = True,
                             footnote: bool = True):
    """
    用 heatmap 比較策略績效
    - normalize=True: 顏色用 z-score（跨指標可比）
    - annot_raw=True: 格子內數字用原始值（未標準化）
    """
    stats = report_collection.get_stats().T.dropna(axis=1, how="all")
    stats = stats[indicators] if indicators else stats[['CAGR', 'daily_sharpe', 'max_drawdown', 'win_ratio']]
    stats_sorted = stats.sort_values(by=sort_by, ascending=False).head(top_n)

    if normalize:
        stats_to_plot = (stats_sorted - stats_sorted.mean()) / stats_sorted.std()
        cmap = "coolwarm"
        center = 0
        cbar_kws = {"label": "z-score (colors)"}
        annot_data = stats_sorted if annot_raw else stats_to_plot
    else:
        stats_to_plot = stats_sorted
        cmap = "RdYlGn"
        center = None
        cbar_kws = {"label": "raw value (colors)"}
        annot_data = stats_sorted  # 反正此時就是 raw

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        stats_to_plot,
        annot=annot_data,      # 顏色矩陣與標註數字分開傳入
        fmt=".2f",
        cmap=cmap,
        linewidths=.5,
        cbar=True,
        center=center,
        cbar_kws=cbar_kws
    )

    title = f"Top {top_n} Strategies (sorted by {sort_by})"
    if normalize and annot_raw:
        title += "  |  Colors=z-score, Numbers=raw"
    plt.title(title, fontsize=14)
    plt.xlabel("Indicators")
    plt.ylabel("Strategy Name")
    plt.xticks(rotation=30)
    plt.yticks(rotation=0)

    if footnote and normalize and annot_raw:
        plt.figtext(0.01, -0.02, "Note: cell colors are z-scores within the selected top-N; cell numbers are raw metrics.",
                    ha="left", fontsize=10)

    plt.tight_layout()
    plt.show()


def summarize_subgroup_performance(report_collection: ReportCollection,
                                   level: str = "F1",
                                   score_column: str = "CAGR",
                                   top_n: int = 10,
                                   use_zscore: bool = False):
    """依指定層級分組，計算各組 score_column 的平均、標準差與樣本數，並畫成排名長條圖。"""
    level_index_map = {"F1": 0, "F2": 1, "C": 2, "P3": 3}
    assert level in level_index_map, f"level 必須是 {list(level_index_map.keys())}"
    idx = level_index_map[level]

    stats_df = report_collection.get_stats().T
    stats_df[level] = stats_df.index.map(lambda n: n.split("__")[idx] if len(n.split("__")) > idx else "None")

    group_summary = (stats_df.groupby(level)[score_column]
                            .agg(['mean', 'std', 'count'])
                            .sort_values(by='mean', ascending=False))
    df = group_summary.reset_index()

    if use_zscore:
        df['mean_zscore'] = (df['mean'] - df['mean'].mean()) / df['mean'].std()
        y_col, y_label = 'mean_zscore', f'{score_column} Z-score'
    else:
        y_col, y_label = 'mean', f'平均 {score_column}'

    df["label"] = df[level] + "<br>n=" + df["count"].astype(str)

    fig = px.bar(df.head(top_n), x=level, y=y_col, error_y='std', text=y_col, color=level,
                 title=f"{score_column} 表現排名（依 {level} 分組）",
                 labels={y_col: y_label}, hover_data=["mean", "std", "count"])
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',
                      xaxis_tickangle=-30, yaxis_title=y_label, showlegend=False)
    fig.show()
    return group_summary


def summarize_all_indicators_by_level(report_collection: ReportCollection,
                                      level: str = "F1",
                                      indicators: Optional[List[str]] = None,
                                      top_n: int = 10,
                                      use_zscore: bool = False):
    """對多個指標逐一呼叫 summarize_subgroup_performance，產生各指標的分組排名圖。"""
    indicators = indicators or ["CAGR", "daily_sharpe", "max_drawdown", "win_ratio"]
    for score_column in indicators:
        print(f"\n 指標：{score_column}")
        summarize_subgroup_performance(report_collection, level=level,
                                       score_column=score_column, top_n=top_n,
                                       use_zscore=use_zscore)

# ===============================================================
# Top-K 挑選與跨集合合併
# ===============================================================
@dataclass
class TopPickSpec:
    sort_by: str = "CAGR"   # 例：'CAGR'、'daily_sharpe'、'win_ratio'、'max_drawdown' 等
    top_k: int = 5
    ascending: bool = False # 對某些指標（如 max_drawdown）可設 True

def _normalize_ascending(metric: str, ascending_default) -> bool:
    """
    ascending 可為 bool 或 dict；若是 dict，依 metric 取值。
    預設：max_drawdown / avg_drawdown 視為「越小越好」。
    """
    if isinstance(ascending_default, dict):
        return ascending_default.get(metric, False)
    lower_better = {"max_drawdown", "avg_drawdown"}
    return True if metric in lower_better else bool(ascending_default)

def select_top_strategies(report_collection: ReportCollection,
                          spec: TopPickSpec | None = None,
                          min_trades: int = 0,
                          add_prefix: str | None = None) -> ReportCollection:
    """
    從單一 ReportCollection 取出前 top_k 名策略並回傳子集合。
    - min_trades：過濾交易筆數太少的策略（需要 Report.trades 存在）
    - add_prefix：在策略名前加上標籤，避免跨 JSON 重名
    """
    if spec is None:
        spec = TopPickSpec()

    stats = report_collection.get_stats().T.dropna(how="all").copy()

    # 交易筆數
    trade_counts = {}
    for name, rep in report_collection.reports.items():
        try:
            trade_counts[name] = len(rep.trades) if hasattr(rep, "trades") else 0
        except Exception:
            trade_counts[name] = 0
    stats["trade_count"] = pd.Series(trade_counts)

    if min_trades > 0:
        stats = stats[stats["trade_count"] >= min_trades]

    if stats.empty:
        print("No strategy meets the trade count requirement.")
        return ReportCollection({})

    asc = _normalize_ascending(spec.sort_by, spec.ascending)
    top_names = (stats.sort_values(by=spec.sort_by, ascending=asc)
                      .head(int(spec.top_k)).index)

    sub_reports = {}
    for n in top_names:
        new_name = f"{add_prefix}__{n}" if add_prefix else n
        sub_reports[new_name] = report_collection.reports[n]
    return ReportCollection(sub_reports)

def merge_collections(collections: List[Tuple[str, ReportCollection]]) -> ReportCollection:
    """
    把多個 ReportCollection 合併成一個，並自動在 key 前加上 label 前綴以避免重名。
    collections: List[(label, ReportCollection)]
    """
    merged = {}
    for label, coll in collections:
        for name, rep in coll.reports.items():
            new_name = f"{label}__{name}" if not name.startswith(f"{label}__") else name
            merged[new_name] = rep
    return ReportCollection(merged)

def compare_top_across_collections(labeled_collections: Dict[str, ReportCollection],
                                   pick_spec: TopPickSpec = TopPickSpec(),
                                   min_trades: int = 0) -> ReportCollection:
    """
    對每個標籤（如每個 JSON）各自挑前 top_k，再合併。
    """
    picked = []
    for label, coll in labeled_collections.items():
        top_sub = select_top_strategies(coll, spec=pick_spec, min_trades=min_trades, add_prefix=label)
        picked.append((label, top_sub))
    return merge_collections(picked)

def run_multi_jsons(json_paths: List[str],
                    builder: Callable[[str], ReportCollection],
                    pick_spec: TopPickSpec = TopPickSpec(),
                    min_trades: int = 0,
                    heatmap_top_n: int = 20,
                    heatmap_sort_by: Optional[str] = None,
                    display_plot: bool = True):
    """
    一條龍流程：
      1) 逐檔 JSON 用 builder 生成 ReportCollection
      2) 各檔取前 N（可指定 min_trades）
      3) 合併後畫 heatmap
    """
    labeled = {}
    for p in json_paths:
        label = Path(p).stem
        coll = builder(p)
        labeled[label] = coll
        print(f"[完成] {label} 產生 {len(coll.reports)} 策略")

    combined_top = compare_top_across_collections(labeled, pick_spec, min_trades=min_trades)

    if display_plot:
        print("\n=== 各 JSON 前幾名合併後之 Heatmap 比較 ===")
        plot_group_stats_heatmap(combined_top,
                                 sort_by=heatmap_sort_by or pick_spec.sort_by,
                                 top_n=heatmap_top_n,
                                 indicators=['CAGR', 'daily_sharpe', 'max_drawdown', 'win_ratio'],
                                 normalize=True)
    return labeled, combined_top

# ===============================================================
# 進階小工具：入選股數、排行榜、年度貢獻
# ===============================================================
def compute_top_counts_table(top_rc: ReportCollection) -> pd.DataFrame:
    """
    將 top_rc 中每個策略的每日入選股數(company_count)對齊合併成寬表，並附(合計)欄。
    """
    frames = []
    for strat_name, rep in top_rc.reports.items():
        s = rep.stock_data['company_count'].copy()
        s.name = strat_name
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    counts = pd.concat(frames, axis=1).sort_index()
    counts['(合計)'] = counts.sum(axis=1)
    return counts

def plot_counts_line(counts: pd.DataFrame, title: str = ""):
    """把入選股數寬表的每一欄畫成折線圖。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in counts.columns:
        ax.plot(counts.index, counts[col].values, marker='o', label=col)
    ax.set_xlabel('時間(年)'); ax.set_ylabel('入選股數量')
    ax.set_title(title or '前幾名策略的每日入選股數')
    ax.legend(fontsize=10, ncol=2)
    ymax = float(counts.drop(columns=['(合計)'], errors='ignore').max().max()) if '(合計)' in counts.columns else float(counts.max().max())
    ax.set_ylim(bottom=0, top=ymax + 1)
    ax.grid(True); plt.tight_layout(); plt.show()

def top_stocks_from_report(rep, topN: int = 10) -> pd.DataFrame:
    """從單一策略的 trades 算出各股票的累積報酬，取前 topN 檔。"""
    df = rep.trades.copy()
    if len(df) == 0:
        return df
    df['cum_return'] = (1 + df['return']).groupby(df['stock_id']).cumprod() - 1
    last_by_stock = df.groupby('stock_id').last().reset_index()
    return last_by_stock.sort_values('cum_return', ascending=False).head(topN)[['stock_id', 'cum_return', 'entry_date', 'exit_date']]

def merge_stock_leaderboard(top_rc: ReportCollection, topN_per_strategy: int = 10) -> pd.DataFrame:
    """把每個策略的 top 股票表合併成一張長表，並加上 strategy 欄標示來源策略。"""
    tables: List[pd.DataFrame] = []
    for name, rep in top_rc.reports.items():
        t = top_stocks_from_report(rep, topN=topN_per_strategy).copy()
        if len(t) == 0:
            continue
        t.insert(0, 'strategy', name)
        tables.append(t)
    if not tables:
        return pd.DataFrame(columns=['strategy', 'stock_id', 'cum_return', 'entry_date', 'exit_date'])
    return pd.concat(tables, ignore_index=True)

def resample_counts(counts: pd.DataFrame, mode: str = "mean") -> pd.DataFrame:
    """
    將「每日入選股數」彙整成每月數表。
    mode:
      - 'mean'：每月均值（「月均持股數」）
      - 'eom' ：每月最後一天（「月底持股數」）
      - 'sum' ：每月加總（「每月累計入選數」）
    """
    if counts.empty:
        return counts

    if not isinstance(counts.index, pd.DatetimeIndex):
        counts = counts.copy()
        counts.index = pd.to_datetime(counts.index)

    m = mode.lower()
    if m == 'mean':
        monthly = counts.resample('M').mean()
        monthly.attrs['y_label'] = '月均持股數'
    elif m == 'eom':
        monthly = counts.resample('M').last()
        monthly.attrs['y_label'] = '月底持股數'
    elif m == 'sum':
        monthly = counts.resample('M').sum()
        monthly.attrs['y_label'] = '每月累計入選數'
    else:
        raise ValueError("monthly_mode 只能是 'mean'、'eom' 或 'sum'")
    return monthly

def build_leaderboard_panel(
    report_collection,               # ReportCollection
    sort_by: str = "CAGR",
    top_k: int = 5,
    topN_stocks: int = 10,
    stock_board_merge: bool = False,
    # 每日顯示控制
    show_daily_table: bool = True,
    show_daily_plot: bool = True,
    # 每月顯示控制
    monthly_mode: str = "mean",      # 'mean' | 'eom' | 'sum'
    show_monthly_table: bool = True,
    show_monthly_plot: bool = True,
):
    """
    一鍵輸出排行榜面板，支援『每日/每月』表格與圖表獨立開關。

    產出：
      result = {
        'top_rc': ReportCollection(挑選後),
        'top_counts_daily': DataFrame（每日入選股數；可能為空）,
        'monthly_counts': DataFrame（每月彙整；可能為空）,
        'per_strategy_top_stocks' 或 'merged_top_stocks': DataFrame/Dict,
      }
    """
    # 1) 依排序指標挑 Top K
    spec = TopPickSpec(sort_by=sort_by, top_k=top_k, ascending=False)
    top_rc = select_top_strategies(report_collection, spec)

    # 2) 建每日入選股數表（Top K 策略）
    top_counts_daily = compute_top_counts_table(top_rc)

    # ---- 每日：表格 ----
    if show_daily_table and len(top_counts_daily) > 0:
        try:
            from IPython.display import display
            print("> 每日入選股數（前 K 策略）")
            display(top_counts_daily)
        except Exception:
            pass

    # ---- 每日：折線圖 ----
    if show_daily_plot and len(top_counts_daily) > 0:
        title = f'前 {top_k} 名策略的每日入選股數（排序指標：{sort_by}）'
        plot_counts_line(top_counts_daily, title=title)

    # 3) 每月彙整（只有在需要時才計算）
    monthly_counts = pd.DataFrame()
    need_monthly = (show_monthly_table or show_monthly_plot)
    if need_monthly and len(top_counts_daily) > 0:
        monthly_counts = resample_counts(top_counts_daily, mode=monthly_mode)
        y_label = monthly_counts.attrs.get('y_label', '每月指標')

        # ---- 每月：表格 ----
        if show_monthly_table:
            try:
                from IPython.display import display
                print(f"> 每月彙整（{y_label}）")
                display(monthly_counts)
            except Exception:
                pass

        # ---- 每月：折線圖 ----
        if show_monthly_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            for col in monthly_counts.columns:
                if col == '(合計)':
                    ax.plot(monthly_counts.index, monthly_counts[col].values, linewidth=2, label=col)
                else:
                    ax.plot(monthly_counts.index, monthly_counts[col].values, marker='o', label=col)
            ax.set_xlabel('時間(年月)', fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_title(f'前 {top_k} 名策略的{y_label}', fontsize=14)
            ax.legend(fontsize=10, ncol=2)
            ax.grid(True)
            try:
                ymax = float(monthly_counts.drop(columns=['(合計)'], errors='ignore').max().max())
                ax.set_ylim(bottom=0, top=ymax + 1)
            except Exception:
                pass
            plt.tight_layout()
            plt.show()

    # 4) 股票排行榜
    result = {
        'top_rc': top_rc,
        'top_counts_daily': top_counts_daily,
        'monthly_counts': monthly_counts
    }

    if stock_board_merge:
        merged = merge_stock_leaderboard(top_rc, topN_per_strategy=topN_stocks)
        try:
            from IPython.display import display
            print(f"> 跨策略 Top-{topN_stocks} 股票排行（合併）")
            display(merged)
        except Exception:
            pass
        result['merged_top_stocks'] = merged
    else:
        per_strategy = {}
        for name, rep in top_rc.reports.items():
            per_strategy[name] = top_stocks_from_report(rep, topN=topN_stocks)
        result['per_strategy_top_stocks'] = per_strategy

    return result

# ---- 年度貢獻度（Σ 月度持倉 × 月報酬） ----
def compute_annual_contribution_from_close_and_position(close_df: pd.DataFrame, pos_df: pd.DataFrame, start: Optional[str] = None, end: Optional[str] = None, monthly_shift: int = 1) -> pd.DataFrame:
    """以收盤價與持倉計算每檔股票的年度貢獻度。
    先把日收盤轉成月報酬，月持倉以 monthly_shift 落後一期避免前視，再逐月加權後依年加總。"""
    if start or end:
        close_df = close_df.loc[start:end]

    close_m_first = close_df.resample('M').first()
    close_m_last  = close_df.resample('M').last()
    monthly_returns = (close_m_last / close_m_first) - 1

    if not isinstance(pos_df.index, pd.DatetimeIndex):
        raise ValueError("pos_df.index 必須是 DatetimeIndex")
    pos_m = pos_df.resample('M').last().reindex(monthly_returns.index).fillna(0.0)

    common_cols = sorted(set(monthly_returns.columns) & set(pos_m.columns))
    monthly_returns = monthly_returns[common_cols]
    pos_m = pos_m[common_cols]

    pos_for_return = pos_m.shift(monthly_shift) if monthly_shift and monthly_shift > 0 else pos_m
    weighted_monthly = (monthly_returns * pos_for_return).fillna(0.0)

    return weighted_monthly.resample('Y').sum()

def compute_annual_contribution(change_data: dict,
                                position: pd.DataFrame,
                                start: Optional[str] = None,
                                end: Optional[str] = None,
                                monthly_shift: int = 1,
                                close_key: str = 'price:close') -> pd.DataFrame:
    """從 change_data 取出收盤價後，呼叫 compute_annual_contribution_from_close_and_position 計算年度貢獻度。"""
    close = change_data.get(close_key)
    if close is None:
        raise KeyError(f"找不到 change_data['{close_key}']")
    return compute_annual_contribution_from_close_and_position(close, position, start, end, monthly_shift)

def plot_annual_contribution_stacked(annual_contrib_df: pd.DataFrame,
                                     top_N: Optional[int] = 10,
                                     title: str = '年度股票資金貢獻度分布（加總月度權重報酬）',
                                     width: int = 1500, height: int = 600) -> go.Figure:
    """把年度貢獻度畫成堆疊長條圖，正負值分開堆疊，每年只保留前 top_N 檔，其餘併為「其他」。"""
    years = [int(ts.year) for ts in annual_contrib_df.index]
    stocks = annual_contrib_df.columns.tolist()
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Bold
    while len(colors) < len(stocks):
        colors.append(f'hsla({random.randint(0,360)}, 70%, 50%, 0.85)')
    color_map = dict(zip(stocks, colors[:len(stocks)]))

    pos_traces, neg_traces = [], []
    for stock in stocks:
        s = annual_contrib_df[stock]
        pos = s.clip(lower=0);  neg = s.clip(upper=0)
        for i, v in enumerate(pos):
            if v > 0:
                pos_traces.append((years[i], stock, float(v)))
        for i, v in enumerate(neg):
            if v < 0:
                neg_traces.append((years[i], stock, float(v)))

    fig = go.Figure()

    # 正值
    for year in years:
        ys = [(stk, val) for y, stk, val in pos_traces if y == year]
        ys.sort(key=lambda x: x[1], reverse=True)
        cumulative = 0.0

        if top_N is not None and len(ys) > top_N:
            top_items, other_items = ys[:top_N], ys[top_N:]
            other_sum = sum(v for _, v in other_items)
            if other_sum > 0:
                fig.add_trace(go.Bar(name='其他', x=[year], y=[other_sum], base=[cumulative],
                                     marker=dict(color='white', pattern_shape="/", pattern_size=10, line_color='gray'),
                                     hovertemplate="<b>%{x}</b><br>其他<br><extra></extra>"))
                cumulative += other_sum
            for stock, val in reversed(top_items):
                fig.add_trace(go.Bar(name=stock, x=[year], y=[val], base=[cumulative],
                                     marker_color=color_map[stock],
                                     hovertemplate=f"<b>%{{x}}</b><br>股票: {stock}<br>貢獻: %{{y:.4f}}<extra></extra>"))
                cumulative += val
        else:
            for stock, val in reversed(ys):
                fig.add_trace(go.Bar(name=stock, x=[year], y=[val], base=[cumulative],
                                     marker_color=color_map[stock],
                                     hovertemplate=f"<b>%{{x}}</b><br>股票: {stock}<br>貢獻: %{{y:.4f}}<extra></extra>"))
                cumulative += val

    # 負值
    for year in years:
        ys = [(stk, val) for y, stk, val in neg_traces if y == year]
        ys.sort(key=lambda x: x[1], reverse=True)
        cumulative = 0.0
        for stock, val in ys:
            fig.add_trace(go.Bar(name=stock, x=[year], y=[val], base=[cumulative],
                                 marker_color=color_map[stock],
                                 hovertemplate=f"<b>%{{x}}</b><br>股票: {stock}<br>貢獻: %{{y:.4f}}<extra></extra>"))
            cumulative += val

    fig.update_layout(barmode='relative', title=title, xaxis_title='年份',
                      yaxis_title='年度貢獻度（Σ 月度持倉 × 月報酬）',
                      showlegend=True, legend_title='股票', hovermode='closest',
                      width=width, height=height)
    fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="black")
    return fig

import pandas as pd
import numpy as np
import plotly.express as px

def aggregate_stock_contributions(report_collection,
                                  metric: str = "cum",   # "cum"：以「每檔股在該策略的累積報酬」作為貢獻；"sum"：單筆交易收益相加
                                  agg: str = "mean",     # 合併多策略時的彙整方式："mean" 或 "sum"
                                  by_year: bool = False, # True 會回傳 (stock_id, year) 的分解結果
                                  min_strategies: int = 1 # 至少出現在幾個策略才計入（做濾除用）
                                  ):
    """
    跨所有策略計算「股票貢獻度」並彙整。
    回傳：
      - matrix_df：index=stock_id（或 (stock_id, year)），columns=策略名，值=該策略中此股票的貢獻度（缺值以 0 補）
      - agg_df：在 matrix_df 的基礎上依據 agg（mean/sum）彙整出的單欄 DataFrame，欄名 'aggregate'
    說明：
      - metric='cum'：每檔股在單一策略內做 (1+return) 的群組連乘後 -1，再取最後一筆（代表該策略下的最終累積貢獻）；
      - metric='sum'：單純把該股票在該策略的所有 trade return 加總。
      - by_year=True：會將貢獻度先以 entry_date 的年份切分再彙整，可做「年度 × 股票」雙索引分析。
    """
    contrib_list = []
    idx_name = "stock_id"

    for strat_name, rep in report_collection.reports.items():
        # rep.trades 來源於 backtest，必含欄位：stock_id、entry_date、return 等
        df = getattr(rep, "trades", None)
        if df is None or len(df) == 0:
            continue
        df = df.copy()

        # 取年份（做 by_year 時會用到）
        if by_year:
            df["entry_date"] = pd.to_datetime(df["entry_date"])
            df["year"] = df["entry_date"].dt.year
            grp_keys = ["stock_id", "year"]
            idx_name = ("stock_id", "year")
        else:
            grp_keys = ["stock_id"]

        if metric == "cum":
            # 先依股票(與年份)做「逐筆累乘」，再取最後一筆
            df["_cum"] = (1.0 + df["return"]).groupby(df["stock_id"]).cumprod() - 1.0 if not by_year \
                         else (1.0 + df["return"]).groupby([df["stock_id"], df["year"]]).cumprod() - 1.0
            sub = (df.groupby(grp_keys)["_cum"].last()
                     .to_frame(name=strat_name))
        elif metric == "sum":
            sub = (df.groupby(grp_keys)["return"].sum()
                     .to_frame(name=strat_name))
        else:
            raise ValueError("metric 只支援 'cum' 或 'sum'")

        contrib_list.append(sub)

    if not contrib_list:
        # 空集合時回傳空表
        empty = pd.DataFrame(columns=["aggregate"])
        return empty, empty

    # 以股票(或股票×年度)為 index，策略為 columns
    matrix_df = pd.concat(contrib_list, axis=1).fillna(0.0)

    # 濾除「出現在策略數量 < min_strategies」的標的（以非零視為有出現）
    appear_cnt = (matrix_df != 0).sum(axis=1)
    matrix_df = matrix_df.loc[appear_cnt >= int(min_strategies)]

    # 彙整（整體貢獻度）
    if agg == "sum":
        agg_series = matrix_df.sum(axis=1)
    else:
        agg_series = matrix_df.mean(axis=1)

    agg_df = agg_series.to_frame(name="aggregate").sort_values("aggregate", ascending=False)
    return matrix_df, agg_df


import plotly.express as px
import plotly.graph_objects as go

def plot_aggregate_stock_contributions(agg_df,
                                       top_n: int = 20,
                                       kind: str = "bar",   # "bar" 或 "treemap"
                                       show: bool = True,   # 是否直接 show；False 只回傳 fig
                                       title: str = None):
    """整體股票貢獻度的視覺化，支援長條圖與 treemap 兩種樣式。
    show=False 時只回傳 fig 不直接顯示，可由呼叫端自行決定繪製時機。"""
    df = agg_df.head(top_n).reset_index()

    # 讓欄位名稱一致（index 可能是 stock_id 或 (stock_id, year)）
    if "year" in df.columns and "stock_id" in df.columns:
        path = ["year", "stock_id"]
        x_col = "stock_id"
        subtitle = " (by year)"
    else:
        path = ["stock_id"]
        x_col = "stock_id" if "stock_id" in df.columns else df.columns[0]
        subtitle = ""

    if kind == "treemap":
        fig = px.treemap(
            df, path=path, values="aggregate", color="aggregate",
            color_continuous_scale="Blues",
            title=title or f"Top {top_n} Stock Contributions{subtitle}",
        )
        fig.update_layout(margin=dict(l=30, r=30, t=60, b=20), template="plotly_white")
    else:
        fig = px.bar(
            df, x=x_col, y="aggregate", color="aggregate",
            color_continuous_scale="Blues", text="aggregate",
            title=title or f"Top {top_n} Stock Contributions"
        )
        # 排序、數字格式、外顯標籤
        fig.update_layout(
            template="plotly_white",
            height=520, width=1200,
            margin=dict(l=40, r=30, t=70, b=60),
            xaxis_title="stock_id",
            yaxis_title="Aggregate Contribution",
            xaxis_tickangle=-30,
            xaxis=dict(categoryorder="total descending"),
            uniformtext_minsize=10, uniformtext_mode="hide",
            coloraxis_showscale=False
        )
        fig.update_yaxes(tickformat=".2f", gridcolor="rgba(0,0,0,0.08)")
        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Aggregate: %{y:.4f}<extra></extra>"
        )
        # 平均線 + 註解
        mean_val = df["aggregate"].mean()
        fig.add_hline(y=mean_val, line_dash="dot", line_width=1,
                      annotation_text=f"mean={mean_val:.2f}",
                      annotation_position="top left")

    if show:
        fig.show()
    return fig
