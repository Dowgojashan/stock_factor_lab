# -*- coding: utf-8 -*-
"""H-26／H-27 結果視覺化（2026-09-07）

把 8,370 格的走查矩陣畫成六組圖，每組各出 4 個版本（全部市場 + 三個市場各一張），
共 24 張 PNG，輸出到 `_analysis_outputs_robustness/figures/`。

    圖 A  勝率熱力圖        13 個窗口方案 × 3 指標，A_hrp 對 B_all，一眼看有沒有方案翻盤
    圖 B  超額分布箱型圖    2,700 格的分布本身（均值看不出離散度）
    圖 C  比例掃描雙軸圖    OOS CAGR（左軸）與 backfill 佔比（右軸）同框
    圖 D  IS vs OOS 散佈圖  「IS 亮眼 ≠ OOS 好」與 A/B 的分離
    圖 E  逐窗時間軸        超額隨 OOS 區間推移的變化（2024-2025 尾部）
    圖 F  勝率熱力圖        13 個窗口方案 × 3 指標，**A_hrp 對市值加權報酬指數大盤**
                          （同圖 A 的版面，換一個對照組，見 M-17）
    圖 G  比例×分配表格圖   5 比例 × 2 分配＝10 張，每張列 13 方案的 IS/OOS CAGR/MDD，
                          同表同欄最佳標紅、最差標綠（另計，不算進上面的 4 版 × 6 組）
    圖 H  群數穩健性時間軸  每個 IS 窗動態選出的群數 k 隨時間怎麼變（回應老師 9-8 提問①，
                          回答的是「分群結構穩不穩」，不是市場宇宙大小——那張圖在
                          `research.universe_history`，因為那支要連資料庫查即時資料）

⚠️ 本模組**不產生資料**，只讀既有凍結產物畫圖，故不寫 manifest；
   但仍會 `verify_inputs` 走查矩陣的 manifest，確保畫的是未被改動的資料。

用法：
    cd code
    python -m research.figures_h26_h27
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import freeze, paths

plt.rcParams.update({
    "figure.dpi": 130,
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

OUT_DIR = paths.ROOT / "_analysis_outputs_robustness" / "figures"
SRC_DIR = paths.ROOT / "_analysis_outputs_robustness"

TREE_LABEL = {"TW": "台股", "US": "美股", "XM": "跨市場"}
TREE_COLOR = {"TW": "#c0392b", "US": "#2471a3", "XM": "#1e8449"}
SCOPES = ("ALL", "TW", "US", "XM")

#: 台美的市值加權報酬指數；跨市場基準＝兩者的等權混合（同 `market_benchmark.build`）
TR_TABLES = {"TW": "taiex_tr", "US": "sp500_tr"}

#: 比例的顯示順序與標籤（`legacy` = 每群 5 支，是原設計的校驗點）
RATIO_ORDER = ["legacy", "0.01", "0.03", "0.05", "0.1"]
RATIO_LABEL = {"legacy": "legacy\n(每群5支)", "0.01": "1%", "0.03": "3%",
               "0.05": "5%", "0.1": "10%"}
RATIO_TITLE = {"legacy": "legacy（每群5支）", "0.01": "1%", "0.03": "3%",
              "0.05": "5%", "0.1": "10%"}
RATIO_FILE = {"legacy": "legacy", "0.01": "1pct", "0.03": "3pct",
             "0.05": "5pct", "0.1": "10pct"}
ALLOC_TITLE = {"equal": "等量分配", "proportional": "比例分配"}

METRICS = [("cagr", "CAGR"), ("mdd", "MDD"), ("calmar", "Calmar")]


def scope_title(scope: str) -> str:
    return "全部市場" if scope == "ALL" else TREE_LABEL[scope]


def load_pairs(log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """回傳 (A_hrp 對 B_all 的逐格配對表, 方案定義表)。

    配對鍵＝(市場, 方案, 窗次, 群數來源)——`B_all` 不隨比例／分配變動，
    故同一個 B_all 會對到 20 個 A_hrp 格子，這正是「同窗同基準」的比較。
    """
    freeze.verify_inputs(SRC_DIR / "_walkforward_matrix_manifest")
    det = pd.read_csv(SRC_DIR / "walkforward_matrix_detail.csv")
    det["ratio"] = det["ratio"].astype(str)

    a = det[det.group == "A_hrp"].copy()
    b = (det[det.group == "B_all"]
         [["tree_key", "scheme", "window_no", "k_mode",
           "is_cagr", "is_mdd", "oos_cagr", "oos_mdd"]]
         .rename(columns={"is_cagr": "b_is_cagr", "is_mdd": "b_is_mdd",
                          "oos_cagr": "b_oos_cagr", "oos_mdd": "b_oos_mdd"}))
    m = a.merge(b, on=["tree_key", "scheme", "window_no", "k_mode"], how="left")
    if m.b_oos_cagr.isna().any():
        raise AssertionError("有 A_hrp 格子對不到同窗的 B_all")

    for pre in ("is", "oos"):
        m[f"{pre}_calmar"] = m[f"{pre}_cagr"] / m[f"{pre}_mdd"].abs()
        m[f"b_{pre}_calmar"] = m[f"b_{pre}_cagr"] / m[f"b_{pre}_mdd"].abs()
    m["ex_cagr"] = (m.oos_cagr - m.b_oos_cagr) * 100          # pp
    m["ex_mdd"] = (m.oos_mdd - m.b_oos_mdd) * 100             # pp（正=回撤較淺）
    m["win_cagr"] = m.oos_cagr > m.b_oos_cagr
    m["win_mdd"] = m.oos_mdd > m.b_oos_mdd
    m["win_calmar"] = m.oos_calmar > m.b_oos_calmar
    m["bf_share"] = m.n_backfilled / m.n_members * 100
    m["oos_end_dt"] = pd.to_datetime(m.oos_end, format="%Y-%m")

    schemes = (det[["scheme", "mode", "min_is_months", "oos_len_months"]]
               .drop_duplicates()
               .sort_values(["mode", "min_is_months", "oos_len_months"]))
    log(f"  配對表 {len(m):,} 格｜方案 {len(schemes)} 個")
    return m, schemes


def load_market_pairs(log=print) -> pd.DataFrame:
    """回傳 A_hrp 對「市值加權報酬指數大盤」的逐格配對表（同 M-17 的定義）。

    跟 `load_pairs` 的 B_all 對照不同——這裡的對照組每個 (市場, OOS 區間)
    只有一個值（大盤不分方案、不分比例），故同一個大盤數字會對到很多格 A_hrp。
    """
    freeze.verify_inputs(SRC_DIR / "_walkforward_matrix_manifest")
    freeze.verify_inputs(SRC_DIR / "_market_benchmark_manifest")
    idx = pd.read_parquet(SRC_DIR / "market_index_monthly.parquet")
    wide = idx.pivot(index="month", columns="index_name", values="ret")
    wide.index = pd.PeriodIndex(wide.index, freq="M")
    bench = {"TW": wide[TR_TABLES["TW"]], "US": wide[TR_TABLES["US"]]}
    bench["XM"] = pd.concat([bench["TW"], bench["US"]], axis=1).mean(axis=1)

    det = pd.read_csv(SRC_DIR / "walkforward_matrix_detail.csv")
    det["ratio"] = det["ratio"].astype(str)
    a = det[det.group == "A_hrp"].copy()

    def _cagr(s):
        s = s.dropna()
        return float((1 + s).prod()) ** (12 / len(s)) - 1 if len(s) else float("nan")

    def _mdd(s):
        nav = (1 + s.dropna()).cumprod()
        return float((nav / nav.cummax() - 1).min()) if len(nav) else float("nan")

    wins = a[["tree_key", "oos_start", "oos_end"]].drop_duplicates()
    recs = []
    for r in wins.itertuples():
        s = bench[r.tree_key]
        s = s[(s.index >= pd.Period(r.oos_start, "M")) & (s.index <= pd.Period(r.oos_end, "M"))]
        recs.append({"tree_key": r.tree_key, "oos_start": r.oos_start, "oos_end": r.oos_end,
                     "mkt_cagr": _cagr(s), "mkt_mdd": _mdd(s)})
    mkt = pd.DataFrame(recs)

    m = a.merge(mkt, on=["tree_key", "oos_start", "oos_end"], how="left")
    if m.mkt_cagr.isna().any():
        raise AssertionError("有 A_hrp 格子對不到同窗的大盤——指數可能不涵蓋該 OOS 區間")

    m["oos_calmar"] = m.oos_cagr / m.oos_mdd.abs()
    m["mkt_calmar"] = m.mkt_cagr / m.mkt_mdd.abs()
    m["ex_cagr"] = (m.oos_cagr - m.mkt_cagr) * 100
    m["win_cagr"] = m.oos_cagr > m.mkt_cagr
    m["win_mdd"] = m.oos_mdd > m.mkt_mdd
    m["win_calmar"] = m.oos_calmar > m.mkt_calmar
    log(f"  對大盤配對表 {len(m):,} 格")
    return m


def _sub(m: pd.DataFrame, scope: str) -> pd.DataFrame:
    return m if scope == "ALL" else m[m.tree_key == scope]


def _save(fig, name: str, log=print) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    log(f"    → {p.name}")


# --------------------------------------------------------------- 圖 A
def _scheme_label(r, d: pd.DataFrame, scope: str) -> str:
    """窗口方案的 y 軸標籤：IS 用「起點~終點（共 x 個月）」，OOS 用「每 N 年一段」。

    ⚠️ IS 起點會因市場而異——anchored 窗的 IS 是「從資料起點到 OOS 開始前」，
    美股資料比台股/跨市場早 5 年（2002-01 vs 2007-01），所以美股同一個方案的
    首窗 IS 天生比 min_is_months 長 60 個月。單一市場圖（TW/US/XM）用該市場
    首窗（window_no=1）的真實日期；ALL（三市場合併）圖無法只標一個日期，
    改標方案定義本身的最短月數（`min_is_months`，三市場一致）。
    """
    oos_years = int(r.oos_len_months) // 12
    tag = "（rolling）" if r.mode == "rolling" else ""
    if scope == "ALL":
        is_years = int(r.min_is_months) // 12
        return (f"{r.scheme}｜IS首窗最短{r.min_is_months}個月"
                f"（{is_years}年）{tag}\nOOS 每 {oos_years} 年一段")
    w1 = d[(d.scheme == r.scheme) & (d.window_no == 1)]
    is_start, is_end = w1.is_start.iloc[0], w1.is_end.iloc[0]
    n_is = int(w1.n_is_months.iloc[0])
    return (f"{r.scheme}｜IS {is_start}~{is_end}"
            f"（共{n_is}個月）{tag}\nOOS 每 {oos_years} 年一段")


def _draw_heatmap(m: pd.DataFrame, schemes: pd.DataFrame, scope: str,
                  fig_tag: str, opponent_label: str, out_prefix: str, log=print) -> None:
    """13 方案 × 3 指標的勝率熱力圖（通用版）。色階以 0.5 為中心（藍＝A 贏、紅＝A 輸）。

    `fig_a_heatmap`（對 B_all）與 `fig_f_market_heatmap`（對大盤）共用同一份繪圖邏輯，
    只換對照組名稱與輸出檔名——版面必須完全一致，兩張圖才能直接並排比較。
    """
    d = _sub(m, scope)
    labels, rows = [], []
    for r in schemes.itertuples():
        g = d[d.scheme == r.scheme]
        labels.append(_scheme_label(r, d, scope))
        rows.append([g.win_cagr.mean(), g.win_mdd.mean(), g.win_calmar.mean()])
    arr = np.array(rows)

    fig, ax = plt.subplots(figsize=(7.4, 8.4))
    im = ax.imshow(arr, cmap="RdBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(3), [n for _, n in METRICS])
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v - 0.5) > 0.28 else "black",
                    fontweight="bold" if v >= 0.9 else "normal")
    ax.set_title(f"圖 {fig_tag}｜A_hrp 對{opponent_label}的 OOS 勝率（{scope_title(scope)}）\n"
                 f"13 個窗口方案 × 3 指標；50% 為分界", fontsize=11)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("A_hrp 勝率", fontsize=9)
    cb.ax.axhline(0.5, color="k", lw=1)
    _save(fig, f"{out_prefix}_{scope}.png", log)


def fig_a_heatmap(m: pd.DataFrame, schemes: pd.DataFrame, scope: str, log=print) -> None:
    """對照組＝B_all（候選池全灑）。"""
    _draw_heatmap(m, schemes, scope, "A", " B_all", "A_winrate_heatmap", log)


def fig_f_market_heatmap(m: pd.DataFrame, schemes: pd.DataFrame, scope: str, log=print) -> None:
    """對照組＝市值加權報酬指數大盤（TAIEX-TR／SP500-TR，跨市場＝兩者等權混合，見 M-17）。

    ⚠️ 跟圖 A 的關鍵差異：B_all 是「候選池全灑」，本來就是個弱對照組
    （候選池是全期間贏家）；大盤才是真正難纏的對手。台股尤其會在這張圖上
    比圖 A 明顯轉紅——2021 年之後起點的窗次會輸（台積電 AI 行情），
    這是報告第二部分 §5 要誠實揭露的事，不是圖畫錯了。
    """
    _draw_heatmap(m, schemes, scope, "F", "大盤（市值加權報酬指數）", "F_market_heatmap", log)


# --------------------------------------------------------------- 圖 B
def fig_b_box(m: pd.DataFrame, scope: str, log=print) -> None:
    """超額 CAGR 的分布：均值看不出離散度，這張看得出有多少格子落在 0 以下。"""
    d = _sub(m, scope)
    data = [d[d.ratio == r].ex_cagr.values for r in RATIO_ORDER]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color="black", lw=1.6))
    for patch in bp["boxes"]:
        patch.set_facecolor("#5dade2")
        patch.set_alpha(0.55)
    rng = np.random.default_rng(42)
    for i, v in enumerate(data, start=1):
        x = rng.normal(i, 0.055, size=len(v))
        ax.scatter(x, v, s=3, alpha=0.12, color="#2c3e50", zorder=1)
    ax.axhline(0, color="#c0392b", lw=1.4, ls="--", zorder=3)

    for i, (r, v) in enumerate(zip(RATIO_ORDER, data), start=1):
        below = (v < 0).mean()
        ax.text(i, ax.get_ylim()[1] * 0.97,
                f"低於0：{below:.1%}\n均值 {v.mean():+.2f}pp",
                ha="center", va="top", fontsize=8)
    ax.set_xticks(range(1, 6), [RATIO_LABEL[r] for r in RATIO_ORDER])
    ax.set_xlabel("精選比例")
    ax.set_ylabel("OOS CAGR 超額（A_hrp 減 B_all，百分點）")
    n = len(d)
    ax.set_title(f"圖 B｜{n:,} 個格子的超額分布（{scope_title(scope)}）\n"
                 f"箱＝四分位，點＝個別格子（已抖動）", fontsize=11)
    _save(fig, f"B_excess_box_{scope}.png", log)


# --------------------------------------------------------------- 圖 C
def fig_c_ratio_sweep(m: pd.DataFrame, scope: str, allocation: str | None = None,
                      log=print) -> None:
    """比例掃描：左軸 OOS CAGR（越挑越好），右軸 backfill 佔比（機制失效）。

    `allocation=None`（預設）＝等量與比例分配一起池化，跟原本版本相同；
    傳 `"equal"`／`"proportional"` 則只看那一種分配方式——全部市場版特別容易
    被兩種分配方式的差異蓋掉細節，拆開看才知道 backfill 曲線在哪種分配下
    翹得更早、更陡。
    """
    d = m if allocation is None else m[m.allocation == allocation]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax2 = ax.twinx()
    ax2.grid(False)
    trees = ["TW", "US", "XM"] if scope == "ALL" else [scope]
    x = np.arange(5)

    for t in trees:
        g = d[d.tree_key == t].groupby("ratio")
        cagr = [g.get_group(r).oos_cagr.mean() * 100 for r in RATIO_ORDER]
        bf = [g.get_group(r).bf_share.mean() for r in RATIO_ORDER]
        ax.plot(x, cagr, "-o", color=TREE_COLOR[t], lw=2, ms=6,
                label=f"{TREE_LABEL[t]}｜OOS CAGR")
        ax2.plot(x, bf, "--s", color=TREE_COLOR[t], lw=1.4, ms=4, alpha=0.6,
                 label=f"{TREE_LABEL[t]}｜backfill 佔比")

    ax.set_xticks(x, [RATIO_LABEL[r] for r in RATIO_ORDER])
    ax.set_xlabel("精選比例")
    ax.set_ylabel("OOS CAGR（%，實線）")
    ax2.set_ylabel("backfill 佔比（%，虛線）")
    ax2.set_ylim(0, max(65, ax2.get_ylim()[1]))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best", ncol=2 if scope == "ALL" else 1)
    alloc_tag = f"｜{ALLOC_TITLE[allocation]}" if allocation else ""
    ax.set_title(f"圖 C｜精選比例掃描（{scope_title(scope)}{alloc_tag}）\n"
                 f"實線：挑越少報酬越高　虛線：比例越高、多樣性門檻越擋不住", fontsize=11)
    suffix = f"_{allocation}" if allocation else ""
    _save(fig, f"C_ratio_sweep_{scope}{suffix}.png", log)


# --------------------------------------------------------------- 圖 D
def fig_d_is_oos(m: pd.DataFrame, scope: str, log=print) -> None:
    """IS Calmar vs OOS Calmar：A_hrp 與 B_all 疊在同一張，附 y=x 對角線。

    ⚠️ **對角線位置不是過擬合或表現好壞的證據，只是窗口設計的結構性產物**：
    IS 窗很長且幾乎每格都吃到 2008/2020/2022 系統性崩盤（MDD 深、Calmar 低），
    OOS 窗短很多、常常沒遇到崩盤（MDD 淺、Calmar 高）。連完全沒被訓練過的
    `B_all`（全灑）重心也在對角線以上，證明這是窗口設計造成的，不是選股技術
    的訊號。真正檢驗「IS 選的東西 OOS 有沒有繼續有效」要看 rank IC
    （`research.rank_persistence`，H-28 §2）與 A_hrp 對 B_all／C_random 的
    勝率（圖 A/F、M-02b），不是這張圖的對角線。
    """
    d = _sub(m, scope)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    b = d.drop_duplicates(["tree_key", "scheme", "window_no", "k_mode"])
    if scope == "ALL":
        for t in ["TW", "US", "XM"]:
            g = d[d.tree_key == t]
            ax.scatter(g.is_calmar, g.oos_calmar, s=6, alpha=0.22,
                       color=TREE_COLOR[t], label=f"A_hrp·{TREE_LABEL[t]}", zorder=2)
    else:
        ax.scatter(d.is_calmar, d.oos_calmar, s=6, alpha=0.25, color=TREE_COLOR[scope],
                   label=f"A_hrp（{len(d):,} 點）", zorder=2)
    ax.scatter(b.b_is_calmar, b.b_oos_calmar, s=18, alpha=0.75, facecolor="none",
               edgecolor="#2c3e50", lw=0.8,
               label=f"B_all（全灑，{len(b):,} 點）", zorder=3)

    # 🔴 兩群的重心——散點重疊時，「A 在 B 右上方」只有靠這兩個標記才看得出來
    ax.scatter([d.is_calmar.mean()], [d.oos_calmar.mean()], marker="X", s=190,
               color="#8e44ad", edgecolor="white", lw=1.5, zorder=5,
               label=f"A_hrp 重心（{d.is_calmar.mean():.2f}, {d.oos_calmar.mean():.2f}）")
    ax.scatter([b.b_is_calmar.mean()], [b.b_oos_calmar.mean()], marker="X", s=190,
               color="#2c3e50", edgecolor="white", lw=1.5, zorder=5,
               label=f"B_all 重心（{b.b_is_calmar.mean():.2f}, {b.b_oos_calmar.mean():.2f}）")

    lo = 0.0
    hi = float(np.nanpercentile(pd.concat([d.is_calmar, d.oos_calmar]), 98.5))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y = x（OOS 等於 IS）")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("IS Calmar")
    ax.set_ylabel("OOS Calmar")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.set_title(f"圖 D｜IS 與 OOS 的關係（{scope_title(scope)}）\n"
                 f"對角線位置＝窗口設計的結構性偏差，不是好壞或過擬合的證據"
                 f"（見下方 X 重心對照）", fontsize=11)
    ax.text(0.98, 0.03,
            "⚠️ 連完全沒被訓練過的 B_all 重心也在對角線以上，\n"
            "可見對角線上下不能拿來判斷選股技術或過擬合，\n"
            "真正的檢定看 A_hrp 重心是否在 B_all 重心的右上方。",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color="#555555", bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.85))
    _save(fig, f"D_is_vs_oos_{scope}.png", log)


# --------------------------------------------------------------- 圖 E
def fig_e_timeline(m: pd.DataFrame, scope: str, log=print) -> None:
    """超額 CAGR 隨 OOS 區間結束時點的變化，散點 + 各時點中位線。"""
    d = _sub(m, scope)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    trees = ["TW", "US", "XM"] if scope == "ALL" else [scope]
    for t in trees:
        g = d[d.tree_key == t]
        ax.scatter(g.oos_end_dt, g.ex_cagr, s=6, alpha=0.16, color=TREE_COLOR[t])
        med = g.groupby("oos_end_dt").ex_cagr.median()
        ax.plot(med.index, med.values, "-o", color=TREE_COLOR[t], lw=1.8, ms=5,
                label=f"{TREE_LABEL[t]}（中位）")
    ax.axhline(0, color="#c0392b", lw=1.3, ls="--")
    ax.set_xlabel("OOS 區間結束時點")
    ax.set_ylabel("OOS CAGR 超額（A_hrp 減 B_all，百分點）")
    ax.legend(fontsize=8)
    ax.set_title(f"圖 E｜超額隨時間的變化（{scope_title(scope)}）\n"
                 f"散點＝個別格子，實線＝該時點的中位數", fontsize=11)
    _save(fig, f"E_timeline_{scope}.png", log)


# --------------------------------------------------------------- 圖 G
def load_ratio_alloc_table(log=print) -> pd.DataFrame:
    """A_hrp 的 IS/OOS CAGR/MDD，對「市場 × 群數來源」取平均，保留 (ratio, allocation, scheme)。

    ⚠️ MDD 在資料裡存成負數（例如 -0.15），所以「數值越大＝虧得越少＝越好」
    跟 CAGR 的方向剛好一致——四欄可以用同一條「取最大值＝紅、取最小值＝綠」規則，
    不需要對 MDD 額外反向。
    """
    freeze.verify_inputs(SRC_DIR / "_walkforward_matrix_manifest")
    det = pd.read_csv(SRC_DIR / "walkforward_matrix_detail.csv")
    det["ratio"] = det["ratio"].astype(str)
    a = det[det.group == "A_hrp"]
    g = (a.groupby(["ratio", "allocation", "scheme"])
         [["is_cagr", "is_mdd", "oos_cagr", "oos_mdd"]].mean().reset_index())
    log(f"  比例×分配表：{g.ratio.nunique()} 比例 × {g.allocation.nunique()} 分配 "
        f"× {g.scheme.nunique()} 方案")
    return g


def fig_g_scheme_table(g: pd.DataFrame, schemes: pd.DataFrame, ratio: str, allocation: str,
                       log=print) -> None:
    """單一 (比例, 分配) 組合下，13 個窗口方案的 IS/OOS CAGR/MDD 表格圖。

    🔴 標色標的是**整個窗口方案的綜合表現**，不是逐欄各自的極值——用 **OOS Calmar**
    （OOS CAGR / |OOS MDD|）排名，取表現最好、最差的**整列**分別標紅／綠。
    只看 OOS、不看 IS：IS 的數字是機制性的（A_hrp 本來就是被 IS 排序選出來的，
    IS 好看不是證據），且 Calmar 同時顧到報酬與回撤，是全篇報告一貫使用的
    品質分數，比單獨看 OOS CAGR 更能代表「整體表現」。
    """
    sch_order = schemes.scheme.tolist()
    #: y 軸標籤改用「IS72/OOS24」這種月數表示，比 A/B/C 好對照——
    #: 唯一要注意的是 R（rolling）跟 E 的月數剛好相同（IS96/OOS36），要加註記才不會混淆。
    row_labels = []
    for r in schemes.itertuples():
        tag = "(rolling)" if r.mode == "rolling" else ""
        row_labels.append(f"IS{int(r.min_is_months)}/OOS{int(r.oos_len_months)}{tag}")

    sub = g[(g.ratio == ratio) & (g.allocation == allocation)].set_index("scheme").loc[sch_order]
    sub = sub.assign(oos_calmar=sub.oos_cagr / sub.oos_mdd.abs())
    cols = ["is_cagr", "is_mdd", "oos_cagr", "oos_mdd", "oos_calmar"]
    col_labels = ["IS CAGR", "IS MDD", "OOS CAGR", "OOS MDD", "OOS Calmar"]

    cell_text = [[f"{sub.loc[s, c] * 100:+.2f}%" for c in cols[:-1]]
                + [f"{sub.loc[s, 'oos_calmar']:.3f}"] for s in sch_order]
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(sch_order) + 0.75))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                   loc="upper center", cellLoc="center", bbox=[0.12, 0, 0.88, 0.92])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)

    best_color, worst_color = "#f5b7b1", "#a9dfbf"     # 淡紅／淡綠
    best_s = sub.oos_calmar.idxmax()
    worst_s = sub.oos_calmar.idxmin()
    for j in range(len(cols)):
        tbl[sch_order.index(best_s) + 1, j].set_facecolor(best_color)
        tbl[sch_order.index(worst_s) + 1, j].set_facecolor(worst_color)
        tbl[0, j].set_facecolor("#d6dbdf")

    ax.set_title(f"圖 G｜{RATIO_TITLE[ratio]} × {ALLOC_TITLE[allocation]}\n"
                 f"13 個窗口方案的 IS／OOS CAGR、MDD；紅＝OOS Calmar 最佳的窗口、"
                 f"綠＝最差（整列標色，只依 OOS 排名）", fontsize=10.5, pad=14)
    _save(fig, f"G_table_{RATIO_FILE[ratio]}_{allocation}.png", log)


# --------------------------------------------------------------- 圖 H
def fig_h_k_over_time(log=print) -> None:
    """每個 IS 窗動態選出的群數 k，依 IS 窗結束時點畫成時間軸，對照固定值 k_fixed。

    回應老師 9-8 的提問：「每群後來少多少，你至少要給人家一個[圖]」。
    ⚠️ 這張圖答的是「分群結構隨時間穩不穩定」，不是「市場宇宙大小隨時間怎麼變」
    ——後者是上市公司家數的問題，見 `research.universe_history`。
    """
    freeze.verify_inputs(SRC_DIR / "_k_stability_manifest")
    ks = pd.read_csv(SRC_DIR / "k_stability.csv")
    ks["is_end_dt"] = pd.to_datetime(ks.is_end, format="%Y-%m")
    ks = ks.sort_values(["tree_key", "is_end_dt"])

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for t in ("TW", "US", "XM"):
        g = ks[ks.tree_key == t]
        ax.plot(g.is_end_dt, g.k_is_selected, "-o", ms=5, lw=1.8,
                color=TREE_COLOR[t], label=f"{TREE_LABEL[t]}｜動態選 k")
        ax.axhline(g.k_fixed.iloc[0], color=TREE_COLOR[t], lw=1, ls=":", alpha=0.6)
    ax.set_xlabel("IS 窗結束時點")
    ax.set_ylabel("群數 k")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("圖 H｜群數穩健性時間軸（回應老師 9-8 提問①）\n"
                 "實線＝每個 IS 窗自選的 k；虛線＝固定值（台6／美7／跨3）；"
                 "43 個窗次中多數偏離固定值，但走查矩陣結論不受影響（見 §1.4）",
                 fontsize=10.5)
    _save(fig, "H_k_over_time.png", log)


def run(log=print) -> None:
    m, schemes = load_pairs(log)
    mkt = load_market_pairs(log)
    for scope in SCOPES:
        log(f"  [{scope_title(scope)}]")
        fig_a_heatmap(m, schemes, scope, log)
        fig_b_box(m, scope, log)
        fig_c_ratio_sweep(m, scope, log=log)
        fig_d_is_oos(m, scope, log)
        fig_e_timeline(m, scope, log)
        fig_f_market_heatmap(mkt, schemes, scope, log)
    log(f"\n完成：{len(SCOPES) * 6} 張圖 → {OUT_DIR}")

    log("  [圖 C｜全部市場，拆分配方式]")
    for allocation in ("equal", "proportional"):
        fig_c_ratio_sweep(m, "ALL", allocation=allocation, log=log)

    log("  [圖 G｜10 種比例×分配表格]")
    ratio_alloc = load_ratio_alloc_table(log)
    for ratio in RATIO_ORDER:
        for allocation in ("equal", "proportional"):
            fig_g_scheme_table(ratio_alloc, schemes, ratio, allocation, log)
    log(f"完成：{len(RATIO_ORDER) * 2} 張表格圖 → {OUT_DIR}")

    log("  [圖 H｜群數穩健性時間軸]")
    fig_h_k_over_time(log)


def main(argv: list[str] | None = None) -> int:
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
