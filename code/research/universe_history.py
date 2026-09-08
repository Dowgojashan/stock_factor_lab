# -*- coding: utf-8 -*-
"""補件 · 老師 9-8 要求的「宇宙隨時間變化」圖（2026-09-08）

老師原話：「臺股在 2008 年的時候好像才 1,200 檔，現在變成 1,700 檔⋯你最後要做一個圖，
就是說你那個每群後來少多少，你至少要給人家一個」。

🔴 這句話講的是**上市公司家數隨時間增加**，不是我們的策略宇宙——策略宇宙
（`n_universe`，台股 6,679／美股 8,360／跨市場 15,040）是凍結的，不隨 IS 窗變動
（見 `walkforward_matrix_detail.csv`，同一市場所有窗的 `n_universe` 只差 ±1）。
`company` 表沒有上市/下市日期欄位，只能用「當年有交易紀錄的公司數」當代理指標
（`stock.date` 落在該年、`company_id` 去重計數）。

⚠️ **這支腳本讀活資料庫，不是凍結產物**——`stock`/`company` 表會隨時間更新，
跟 DD-08 的凍結鏈無關，故不寫 manifest、也不 `verify_inputs`。每次重跑數字可能
因資料庫更新而略有不同，這是預期行為（因為問題本身問的就是「歷史上的動態」）。

用法：
    cd code
    python -m research.universe_history
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from database import Database  # noqa: E402
from . import paths

OUT_DIR = paths.ROOT / "_analysis_outputs_robustness"


def fetch_company_count_by_year(market: str, log=print) -> pd.DataFrame:
    """該市場「當年有交易紀錄」的公司家數，逐年計數（代理『上市家數』的指標）。"""
    db = Database(market)
    conn = db.create_connection()
    cur = conn.cursor()
    extra = db._universe_clause("company.company_symbol") if market == "US" else ""
    sql = f"""
        SELECT YEAR(stock.date) AS yr, COUNT(DISTINCT stock.company_id) AS n_companies
        FROM stock
        JOIN company ON company.id = stock.company_id
        WHERE company.{db._exchange_in_clause()} {extra}
        GROUP BY YEAR(stock.date)
        ORDER BY yr
    """
    cur.execute(sql)
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["year", "n_companies"])
    df["market"] = market
    log(f"  [{market}] {df.year.min()}~{df.year.max()}，{len(df)} 年")
    return df


def build(log=print) -> pd.DataFrame:
    parts = [fetch_company_count_by_year(m, log) for m in ("TW", "US")]
    df = pd.concat(parts, ignore_index=True)
    #: 美股資料庫早年（1980s~1990s）只有個位數公司、是資料稀疏的雜訊，不是真實家數，
    #: 從 2000 年開始才有穩定覆蓋，跟其餘研究線的樣本起點（2000/2002）一致。
    df = df[~((df.market == "US") & (df.year < 2000))].reset_index(drop=True)
    return df


def fig_universe_growth(df: pd.DataFrame, log=print) -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "axes.grid": True, "grid.alpha": 0.25,
    })
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    colors = {"TW": "#c0392b", "US": "#2471a3"}
    for m, g in df.groupby("market"):
        ax.plot(g.year, g.n_companies, "-o", ms=3.5, lw=1.8, color=colors[m], label=m)
    for m, yr, note in [("TW", 2008, "2008：1,133 家"), ("TW", 2022, "2022 峰值：1,775 家")]:
        row = df[(df.market == m) & (df.year == yr)]
        if len(row):
            ax.annotate(note, (yr, row.n_companies.iloc[0]),
                        textcoords="offset points", xytext=(0, 10), fontsize=8, ha="center")
    ax.set_xlabel("年")
    ax.set_ylabel("當年有交易紀錄的公司數")
    ax.legend(fontsize=9)
    ax.set_title("附圖｜市場宇宙隨時間的成長（回應老師 9-8 提問）\n"
                 "【注意】這是「上市公司家數」，不是策略宇宙——策略宇宙（凍結池）不隨時間變動",
                 fontsize=11)
    p = OUT_DIR / "figures" / "UH_universe_growth.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    log(f"    → {p.name}")


def run(log=print) -> None:
    df = build(log)
    p = OUT_DIR / "market_universe_history.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    log(f"  → {p.name}")
    fig_universe_growth(df, log)


def main(argv: list[str] | None = None) -> int:
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
