# -*- coding: utf-8 -*-
"""
Phase 1 分析：對 9 桶單因子結果做線性/單調性檢定，產出因子去留名單。

對應老師 2026-08-05 的指示：
  「先切割 9 個區間…然後我再去看它**有沒有線性**…**沒有線性表示這個因子不行**」
  「那如果好的話，我再把它拼回（9 拼回 3）」

判準（見 重跑計畫_老師方法論SOP.md §2.4）：
  |ρ| ≥ 0.85 且 p < 0.05          → ✅ 過關
  0.5 ≤ |ρ| < 0.85                 → ⚠️ 邊際（看形狀）
  |ρ| < 0.5                        → ❌ 淘汰（除非極端桶效應顯著＝MOM型）

只讀 Phase 1 既有結果，不重跑回測、不改 daily_sharpe。

用法：python phase1_analyze.py [--market TW]
"""
import re
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase1_linearity import FACTORS, N_BUCKETS, IN_SAMPLE_END, LOOKAHEAD_FLAGGED  # noqa: E402
from sweep_config import MARKET_START, date_range_suffix                          # noqa: E402

ART = HERE / "results_artifacts"
OUT = HERE.parent / "_analysis_outputs_phase1"

OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "axes.grid": True,
    "grid.color": "#DDDDDD", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def log(m):
    print(m, flush=True)


def bench_cagr(market, start=None, end=None):
    """in-sample 期間的大盤年化。start/end 預設沿用 MARKET_START/IN_SAMPLE_END。"""
    start = start or MARKET_START[market]
    end = end or IN_SAMPLE_END
    try:
        import fcv_core  # noqa: F401  bootstrap
        from database import Database
        db = Database(market)
        conn = db.create_connection()
        bm = pd.read_sql("SELECT date, close FROM " + db.BENCHMARK_TABLE[market], conn)
        conn.close()
        bm["date"] = pd.to_datetime(bm["date"])
        bm = bm.set_index("date").sort_index()["close"].astype(float).dropna()
        bm = bm[(bm.index >= start) & (bm.index <= end)]
        yrs = (bm.index[-1] - bm.index[0]).days / 365.25
        return float((bm.iloc[-1] / bm.iloc[0]) ** (1 / yrs) - 1), bm.index[0].date(), bm.index[-1].date()
    except Exception as e:
        # ⚠️ 不可 fallback 回 0：基準是所有判定的門檻，bench=0 會讓每個因子都「贏大盤」，
        #    把資料庫連不上偽裝成「全部過關」。這正是我們要求 collector 不要做的事
        #    （算不出來一律回 None，絕不回 0），自己更不能犯。
        raise RuntimeError(f"大盤基準計算失敗，無法判定：{e}") from e


def load_curve(market, factor, rsfx=""):
    """回傳 v0 的 9 桶 CAGR 曲線（index=桶號）。缺檔或缺桶回傳 None。"""
    p = ART / f"{market}_L1_{factor}_M{rsfx}" / "stats.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df = df[df["strategy"].str.endswith("__v0")].copy()   # Phase 1 只跑 v0，這行是保險
    df["k"] = df["strategy"].map(lambda s: int(re.match(r".+_qb(\d+)of\d+__", s).group(1)))
    df = df.sort_values("k").set_index("k")
    return df


def size_overlap(market, factor, k, ref_pos=None, rsfx=""):
    """該因子第 k 桶的持股，有多少比例同時落在「規模最小的那一桶」(REVENUE_qb0)。

    為什麼要看這個：q_band 是橫斷面分位排名，某因子的極端桶很可能同時就是
    「最小的那些公司」——例如小公司的 ROE 常是負的或極低，故「ROE 最低桶」
    撈到的有一大半其實是小公司。在**等權**回測裡小公司報酬又特別高，
    就會把「小公司溢酬」誤認成「這個因子有效」。

    重疊率高（>40%）＝該桶的表現要打折看；低（<20%）＝是獨立訊號。
    這只是**診斷**（揭露干擾程度），不是控制；真正的控制見 size_control_analysis.py
    （用 F1×F2 把 REVENUE 當其中一腳，等於在固定規模的條件下看該因子）。
    """
    if ref_pos is None:
        return np.nan
    p = (ART / f"{market}_L1_{factor}_M{rsfx}"
         / f"{factor}_qb{k}of{N_BUCKETS}__None__None__v0" / "position.parquet")
    if not p.exists():
        return np.nan
    try:
        h = (pd.read_parquet(p) != 0)
        a = ref_pos.reindex(index=h.index, columns=h.columns, fill_value=False)
        inter = (a & h).sum(axis=1)
        n = h.sum(axis=1)
        return float((inter / n.replace(0, np.nan)).mean())
    except Exception:
        return np.nan


def load_size_ref(market, rsfx=""):
    """規模最小那一桶的持股遮罩（REVENUE_qb0），當作規模效應的參照。"""
    p = (ART / f"{market}_L1_REVENUE_M{rsfx}"
         / f"REVENUE_qb0of{N_BUCKETS}__None__None__v0" / "position.parquet")
    if not p.exists():
        log("⚠️ 找不到 REVENUE_qb0 的持股，略過規模重疊診斷")
        return None
    return pd.read_parquet(p) != 0


def classify(ks, ys, rho):
    """形狀分類：單調↑/單調↓/倒U/U型/無序。用二次項判斷凹凸。"""
    if len(ys) < 5:
        return "樣本不足"
    coef = np.polyfit(ks, ys, 2)
    a = coef[0]
    peak_at = int(np.argmax(ys))
    trough_at = int(np.argmin(ys))
    interior = 1 <= peak_at <= len(ys) - 2
    interior_t = 1 <= trough_at <= len(ys) - 2
    if abs(rho) >= 0.85:
        return "單調↑" if rho > 0 else "單調↓"
    if a < 0 and interior:
        return "倒U（中間最好）"
    if a > 0 and interior_t:
        return "U型（兩端最好）"
    return "無序"


def verdict(rho, p, ys, bench):
    """去留判定。極端桶效應：頭或尾單獨明顯高過其餘（MOM 型）。"""
    if abs(rho) >= 0.85 and p < 0.05:
        return "✅ 過關"
    rest = ys[1:-1]
    hi_edge = max(ys[0], ys[-1])
    edge_gap = hi_edge - (np.mean(rest) if len(rest) else np.nan)
    if abs(rho) < 0.5:
        if edge_gap > 0.03 and hi_edge > bench:
            return "⚠️ 只取極端桶"
        return "❌ 淘汰"
    return "⚠️ 邊際"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="TW", choices=["TW", "US"])
    ap.add_argument("--bench", default="universe", choices=["universe", "index"],
                    help="universe＝基準(含股利，預設)；index＝外部價格指數(不含股利)")
    ap.add_argument("--start", default=None,
                    help="自訂起始日期 YYYY-MM-DD（需與 phase1_linearity.py 執行時相同）")
    ap.add_argument("--end", default=None,
                    help="自訂結束日期 YYYY-MM-DD（需與 phase1_linearity.py 執行時相同）")
    args = ap.parse_args()
    mkt = args.market
    start = args.start or MARKET_START[mkt]
    end = args.end or IN_SAMPLE_END
    rsfx = date_range_suffix(start, end, MARKET_START[mkt], IN_SAMPLE_END)
    sfx = ("_idxbench" if args.bench == "index" else "") + rsfx
    OUT.mkdir(parents=True, exist_ok=True)

    from universe_benchmark import get_bench
    bench, _ = get_bench(mkt, args.bench, start=start, end=end)
    log(f"基準 = {bench:.2%}｜期間 {start}~{end}\n")

    log("載入規模參照（REVENUE_qb0）以計算重疊診斷 …")
    ref = load_size_ref(mkt, rsfx)

    rows, curves, missing = [], {}, []
    for f in FACTORS:
        df = load_curve(mkt, f, rsfx)
        if df is None or len(df) < N_BUCKETS:
            missing.append(f)
            continue
        ks = df.index.values.astype(float)
        ys = df["CAGR"].values.astype(float)
        curves[f] = (ks, ys)
        rho, p = st.spearmanr(ks, ys)
        # 前瞻偏誤的因子照跑照畫圖（供報告說明），但判定一律覆寫、不列入評選
        vd = "🚫 前瞻偏誤" if f in LOOKAHEAD_FLAGGED else verdict(rho, p, ys, bench)
        rows.append({
            "因子": f,
            "Spearman_rho": round(float(rho), 3),
            "p值": round(float(p), 4),
            "形狀": classify(ks, ys, rho),
            "判定": vd,
            "最佳桶": int(np.argmax(ys)),
            "最佳桶CAGR": round(float(ys.max()), 4),
            "最差桶CAGR": round(float(ys.min()), 4),
            "全桶均值": round(float(ys.mean()), 4),
            "贏大盤桶數": int((ys > bench).sum()),
            # 規模干擾診斷：最佳桶的持股有多少比例也是「規模最小的那一桶」
            "最佳桶與最小規模重疊": (np.nan if f == "REVENUE" else
                            round(size_overlap(mkt, f, int(np.argmax(ys)), ref, rsfx), 3)),
            **{f"桶{i}": round(float(v), 4) for i, v in enumerate(ys)},
        })

    if missing:
        log(f"⚠️ 無結果（Phase 1 失敗或資料缺失）：{missing}\n")

    out = pd.DataFrame(rows)
    order = {"✅ 過關": 0, "⚠️ 只取極端桶": 1, "⚠️ 邊際": 2, "❌ 淘汰": 3, "🚫 前瞻偏誤": 4}
    out["_o"] = out["判定"].map(order)
    out = out.sort_values(["_o", "Spearman_rho"], key=lambda s: s.abs() if s.name == "Spearman_rho" else s,
                          ascending=[True, False]).drop(columns="_o")
    out.to_csv(OUT / f"{mkt}_phase1{sfx}_linearity.csv", index=False, encoding="utf-8-sig")

    show = ["因子", "Spearman_rho", "p值", "形狀", "判定", "最佳桶", "最佳桶CAGR",
            "全桶均值", "贏大盤桶數", "最佳桶與最小規模重疊"]
    pd.set_option("display.width", 200)
    log("===== Phase 1 線性檢定結果 =====")
    log(out[show].to_string(index=False))
    log("\n===== 9 桶 CAGR 明細 =====")
    log(out[["因子"] + [f"桶{i}" for i in range(N_BUCKETS)]].to_string(index=False))

    # 圖：每因子一條 9 桶折線（小圖網格）
    n = len(curves)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow), sharex=True)
    vmap = dict(zip(out["因子"], out["判定"]))
    cmap = {"✅ 過關": OI[2], "⚠️ 只取極端桶": OI[4], "⚠️ 邊際": OI[0],
            "❌ 淘汰": OI[1], "🚫 前瞻偏誤": OI[3]}
    # 2026-08-18 第三方審查指出 emoji 在 matplotlib 預設中文字型下會顯示成 □（缺字），
    # 顏色已經在區分判定，標題改用純文字，不依賴字型是否有 emoji 字形。
    vtext = {"✅ 過關": "過關", "⚠️ 只取極端桶": "只取極端桶", "⚠️ 邊際": "邊際",
             "❌ 淘汰": "淘汰", "🚫 前瞻偏誤": "前瞻偏誤"}
    for ax, (f, (ks, ys)) in zip(np.atleast_1d(axes).ravel(), curves.items()):
        v = vmap.get(f, "")
        ax.plot(ks, ys, marker="o", color=cmap.get(v, OI[7]), linewidth=1.6, markersize=4)
        ax.axhline(bench, color="#888888", linestyle="--", linewidth=1.0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        rho = out.loc[out["因子"] == f, "Spearman_rho"].iloc[0]
        rng = (float(np.max(ys)) - float(np.min(ys))) * 100
        ax.set_title(f"{f}  ρ={rho:+.2f}  {vtext.get(v, v)}  全距{rng:.1f}pp", fontsize=9)
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Phase 1  9桶單因子 CAGR 曲線 — {mkt}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / f"{mkt}_phase1{sfx}_curves.png", bbox_inches="tight")
    plt.close(fig)

    log(f"\n輸出：{OUT / f'{mkt}_phase1{sfx}_linearity.csv'}")
    log(f"      {OUT / f'{mkt}_phase1{sfx}_curves.png'}")
    log("\n===== 去留摘要 =====")
    for v in ["✅ 過關", "⚠️ 只取極端桶", "⚠️ 邊際", "❌ 淘汰", "🚫 前瞻偏誤"]:
        fs = out.loc[out["判定"] == v, "因子"].tolist()
        log(f"  {v}（{len(fs)}）：{', '.join(fs) if fs else '—'}")
    if missing:
        log(f"  🔴 無資料（{len(missing)}）：{', '.join(missing)}")


if __name__ == "__main__":
    main()
