# -*- coding: utf-8 -*-
"""
台股三個 CAGR 冠軍策略的集中度可信度檢查：是不是靠單一飆股撐起來的？

只算數字，不重篩、不換冠軍、不改任何原始資料、daily_sharpe 不重算。

指標：
  1. top1/top3/top5_share：前1/3/5大貢獻個股佔總報酬比重
     － 兩種口徑並陳（見下方「口徑說明」），避免只看單一口徑失真
  2. Effective N：貢獻占比的 Herfindahl 倒數（沿用 analyze_batch.py::strategy_credibility 的既有演算法）
  3. 貢獻前10大個股明細（金額式累積報酬 + 佔比）
  4. 拿掉 top1 個股後的 CAGR（反事實模擬）

口徑說明（重要）：
  repo 既有的 strategy_credibility() 只用「正貢獻個股」當分母（對應論文4-9/4-10圖），
  這會系統性高估集中度以外的分散度、也讓分母小於實際總報酬。本腳本兩種都算：
    * share_pos ：正貢獻個股為分母（與既有圖表口徑一致，可跟4-9/4-10對照）
    * share_net ：全部個股淨貢獻加總為分母（更貼近「佔總報酬%」的直覺問法）

反事實 CAGR 的方法與限制（誠實說明）：
  不重跑回測引擎，改用 position.parquet 的月頻權重 × 資料庫 price:close（回測引擎用的同一個
  價格欄位）重建投組月報酬，扣除與引擎相同的手續費/證交稅率後鏈式相乘得 CAGR。
  重建值會與引擎回報值有落差（引擎有 MAE/MFE 窗、停損停利、進出場日對齊等細節），
  故本腳本一律「同方法比較」：反事實 CAGR 與『重建基準 CAGR』相比算跌幅，
  重建誤差在相減時大致抵銷。重建基準 vs 引擎回報值的落差也一併輸出供檢查。

  反事實兩種變體：
    * cash  ：top1 的權重變現金（不再投入），最保守、最直觀＝「當初沒買到這檔」
    * renorm：top1 的權重平均分給同期其他持股（假設換成同期其他選中標的）

用法（cwd 任意）：
  python tw_champion_concentration.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fcv_core  # noqa: F401,E402  bootstrap: chdir 到 code/ 讓 database.py 的 ../config.ini 解析得到
from analyze_batch import load_strategy_artifacts, stock_cum_contrib  # noqa: E402
from database import Database  # noqa: E402

OUT_DIR = HERE.parent / "_analysis_outputs_TW" / "_cross_batch"
ART_DIR = HERE / "results_artifacts"

# 引擎費率（backtest.py::sim 台股預設值），反事實重建時沿用同一組
FEE_RATIO = 1.425 / 1000
TAX_RATIO = 3 / 1000

CHAMPIONS = [
    ("EV_S", "TW_batch_EV_S_M", "FCF_P_qb4of5__EV_S_qb0of5__C10_EPS_DYN_qmax4__v1"),
    ("PB", "TW_batch_PB_M", "FCF_P_qb0of5__PB_qb0of5__C4_ROE_DYN_qmax8__v0"),
    ("P_IC", "TW_batch_P_IC_M", "FCF_P_qb0of5__P_IC_qb0of5__C4_ROE_DYN_qmax8__v0"),
]


def log(msg):
    print(msg, flush=True)


def fetch_close_prices(symbols):
    """只撈需要的個股 close（＝回測引擎 price:close 用的同一欄），避免載入整個 Data('TW')。"""
    db = Database("TW")
    conn = db.create_connection()
    cur = conn.cursor()
    syms = "','".join(sorted(set(str(s) for s in symbols)))
    sql = f"""
        SELECT company_symbol, date, close
        FROM company RIGHT JOIN stock ON company.id = stock.company_id
        WHERE {db._exchange_in_clause()} AND company_symbol IN ('{syms}')
    """
    cur.execute(sql)
    rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["company_symbol", "date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.pivot_table(index="date", columns="company_symbol", values="close").sort_index()


def period_returns(px, rebal_dates):
    """把日收盤壓成『再平衡日→下一個再平衡日』的期間報酬矩陣（rows=期初日, cols=個股）。"""
    aligned = px.reindex(px.index.union(rebal_dates)).ffill().reindex(rebal_dates)
    return aligned.pct_change().shift(-1).iloc[:-1]  # r(t) = 從 t 持有到 t+1 的報酬


def simulate(weights, rets):
    """給定逐期權重與期間報酬，回傳鏈式淨值序列（已扣換手成本）。"""
    nav = [1.0]
    prev_w = pd.Series(0.0, index=weights.columns)
    for t in rets.index:
        w = weights.loc[t].fillna(0.0)
        r = rets.loc[t]
        # 只對有價格的標的計算；缺價格者視為當期未持有（權重歸零，不假造報酬）
        valid = r.notna()
        w_eff = w.where(valid, 0.0)
        gross = float((w_eff * r.fillna(0.0)).sum())
        turnover = float((w_eff - prev_w).abs().sum())
        buy = float((w_eff - prev_w).clip(lower=0).sum())
        sell = float((prev_w - w_eff).clip(lower=0).sum())
        cost = FEE_RATIO * turnover + TAX_RATIO * sell
        nav.append(nav[-1] * (1.0 + gross - cost))
        prev_w = w_eff * (1.0 + r.fillna(0.0))
        s = prev_w.sum()
        prev_w = prev_w / s if s > 0 else prev_w * 0.0  # 期末漂移後的實際權重，供下期算換手
        _ = buy  # 明確保留變數以說明成本組成
    return pd.Series(nav[1:], index=rets.index)


def cagr_from_nav(nav, years):
    if years <= 0 or nav.iloc[-1] <= 0:
        return np.nan
    return float(nav.iloc[-1] ** (1.0 / years) - 1.0)


def analyze_champion(tag, label, strategy):
    log(f"\n{'='*70}\n[{tag}] {strategy}")
    sdir = ART_DIR / label / strategy
    art = load_strategy_artifacts(sdir)
    trades, pos = art["trades"], art["position"]

    # ---------- 1~3. 貢獻集中度（不需價格，純用 trades） ----------
    contrib = stock_cum_contrib(trades)                     # 逐股累積複合報酬
    pos_c = contrib[contrib > 0].sort_values(ascending=False)
    share_pos = pos_c / pos_c.sum()                          # 既有口徑（正貢獻為分母）
    net_total = float(contrib.sum())
    share_net = (contrib.sort_values(ascending=False) / net_total) if net_total > 0 else contrib * np.nan

    hhi = float((share_pos.values ** 2).sum())
    eff_n = 1.0 / hhi if hhi > 0 else np.nan
    top1_stock = str(share_pos.index[0])

    top10 = pd.DataFrame({
        "排名": range(1, min(10, len(share_pos)) + 1),
        "stock_id": share_pos.index[:10],
        "累積報酬(倍)": pos_c.values[:10],
        "佔正貢獻比重": share_pos.values[:10],
    })
    top10["佔淨總報酬比重"] = [float(contrib.get(s, np.nan) / net_total) if net_total > 0 else np.nan
                          for s in top10["stock_id"]]

    log(f"  個股數={contrib.size}｜正貢獻={len(pos_c)}｜top1={top1_stock} "
        f"(累積 {pos_c.iloc[0]:+.1%})｜Effective N={eff_n:.2f}")

    # ---------- 4. 反事實：拿掉 top1 後的 CAGR ----------
    px = fetch_close_prices(pos.columns)
    rebal = pd.DatetimeIndex(pos.index)
    rets = period_returns(px.reindex(columns=pos.columns), rebal)
    W = pos.loc[rets.index].fillna(0.0)

    years = (rebal[-1] - rebal[0]).days / 365.25

    nav_base = simulate(W, rets)
    cagr_recon = cagr_from_nav(nav_base, years)

    # cash 變體：top1 權重歸零、不補回（＝當初沒買這檔，錢閒置）
    W_cash = W.copy()
    if top1_stock in W_cash.columns:
        W_cash[top1_stock] = 0.0
    nav_cash = simulate(W_cash, rets)
    cagr_cash = cagr_from_nav(nav_cash, years)

    # renorm 變體：top1 權重平均分給同期其他持股
    W_ren = W_cash.div(W_cash.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    nav_ren = simulate(W_ren, rets)
    cagr_ren = cagr_from_nav(nav_ren, years)

    drop_cash = (cagr_recon - cagr_cash) / cagr_recon if cagr_recon else np.nan
    drop_ren = (cagr_recon - cagr_ren) / cagr_recon if cagr_recon else np.nan

    log(f"  重建基準CAGR={cagr_recon:.2%}｜去top1(cash)={cagr_cash:.2%} (跌{drop_cash:.1%})"
        f"｜去top1(renorm)={cagr_ren:.2%} (跌{drop_ren:.1%})")

    return {
        "候選因子批": tag,
        "strategy": strategy,
        "回測期(年)": round(years, 2),
        "涉及個股數": int(contrib.size),
        "正貢獻個股數": int(len(pos_c)),
        "平均每月持股數": round(float((pos != 0).sum(axis=1).replace(0, np.nan).mean()), 2),
        "top1個股": top1_stock,
        "top1累積報酬(倍)": round(float(pos_c.iloc[0]), 4),
        "top1_share(正貢獻口徑)": round(float(share_pos.iloc[0]), 4),
        "top3_share(正貢獻口徑)": round(float(share_pos.iloc[:3].sum()), 4),
        "top5_share(正貢獻口徑)": round(float(share_pos.iloc[:5].sum()), 4),
        "top1_share(淨總報酬口徑)": round(float(share_net.iloc[0]), 4) if net_total > 0 else np.nan,
        "top3_share(淨總報酬口徑)": round(float(share_net.iloc[:3].sum()), 4) if net_total > 0 else np.nan,
        "top5_share(淨總報酬口徑)": round(float(share_net.iloc[:5].sum()), 4) if net_total > 0 else np.nan,
        "Effective_N": round(float(eff_n), 3),
        "重建基準CAGR": round(float(cagr_recon), 4),
        "去top1後CAGR(cash)": round(float(cagr_cash), 4),
        "去top1後CAGR(renorm)": round(float(cagr_ren), 4),
        "CAGR跌幅(cash)": round(float(drop_cash), 4),
        "CAGR跌幅(renorm)": round(float(drop_ren), 4),
    }, top10


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = pd.read_csv(OUT_DIR / "D_champion_full_metrics.csv", encoding="utf-8-sig")
    reported = dict(zip(stats["strategy"], stats["CAGR"]))

    rows, top10s = [], []
    for tag, label, strategy in CHAMPIONS:
        r, t10 = analyze_champion(tag, label, strategy)
        r["引擎回報CAGR"] = round(float(reported.get(strategy, np.nan)), 4)
        r["重建 vs 引擎落差"] = round(r["重建基準CAGR"] - r["引擎回報CAGR"], 4)
        rows.append(r)
        t10.insert(0, "候選因子批", tag)
        top10s.append(t10)

    out = pd.DataFrame(rows)
    cols = ["候選因子批", "strategy", "引擎回報CAGR", "重建基準CAGR", "重建 vs 引擎落差",
            "回測期(年)", "平均每月持股數", "涉及個股數", "正貢獻個股數",
            "top1個股", "top1累積報酬(倍)",
            "top1_share(正貢獻口徑)", "top3_share(正貢獻口徑)", "top5_share(正貢獻口徑)",
            "top1_share(淨總報酬口徑)", "top3_share(淨總報酬口徑)", "top5_share(淨總報酬口徑)",
            "Effective_N",
            "去top1後CAGR(cash)", "CAGR跌幅(cash)",
            "去top1後CAGR(renorm)", "CAGR跌幅(renorm)"]
    out = out[cols]
    out.to_csv(OUT_DIR / "champion_concentration_check.csv", index=False, encoding="utf-8-sig")

    top10_all = pd.concat(top10s, ignore_index=True)
    top10_all.to_csv(OUT_DIR / "champion_top10_contributors.csv", index=False, encoding="utf-8-sig")

    log(f"\n{'='*70}\n對照表：")
    log(out.to_string(index=False))
    log(f"\n貢獻前10大個股明細：")
    log(top10_all.to_string(index=False))
    log(f"\n輸出：{OUT_DIR/'champion_concentration_check.csv'}")
    log(f"      {OUT_DIR/'champion_top10_contributors.csv'}")


if __name__ == "__main__":
    main()
