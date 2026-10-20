# -*- coding: utf-8 -*-
"""2026-10-20：回應同儕審閱（`stock-factor-lab-63`論文寫作session）對美股M1-D
「前20%佔比全程NONE」的質疑——這個指標用前20%（by rank）衡量集中度，範圍
本身可能太粗，測不到「前幾大巨頭」這種更窄的集中度事件（Goldman報告記載的
「Magnificent 7」現象）。

🔴 2026-10-20同儕複核CSV後更正：母體檔數**不是**固定的「約2,800~2,972檔
（Russell 3000）」——`monitor.fetch_mcap_wide('US')`的SQL只篩
`exchange_name IN ('NASDAQ','NYSE','AMEX')`，**沒有套`database.py`的
`_universe_clause()`**（查證：`company`表NASDAQ/NYSE/AMEX共11,187個symbol，
`russell3000`表只有2,994筆；`get_daily_stock()`/`get_finance_report()`/
`resolve_strategy_holdings.py`都有套這個clause，`monitor.py`沒有套，是
CLAUDE.md明文警告過的那個雷，這裡確認monitor.py這條路徑沒防到）。
因此這份診斷（以及沿用同一函式的正式M1-D生產計算）量的其實是「資料庫裡
NASDAQ/NYSE/AMEX三個交易所、有市值資料的全部symbol」，不是Russell 3000——
母體本身的檔數逐季劇烈變動（2009Q4=547、2012Q4=2,175、2015Q4=3,544、
2018Q4=4,567、2021Q4=6,027、2023Q4=4,932、2024Q4=4,666、2025Q4=4,448，
CSV的`n_universe`欄逐季記錄），「前20%」「前1%」的實際檔數因此跟著母體
一起變動（2023-2025年間前20%約890~990檔、前1%約45~50檔，不是本檔案
原先假設的594檔/30檔），不是固定比例對應固定檔數。母體檔數隨時間劇烈
變動（尤其2009-2015那段，很可能是collector歷史資料覆蓋率逐年擴增、
不是真實的美股上市公司數暴增）這件事本身，也是①`_prelim_a1_index_
concentration_us.py`校準出的p90門檻为何在2009-2011段特別不穩定的
一個可能成因，比「早期市值資料覆蓋不全的人為跳動」更具體一層——是否
修正`monitor.fetch_mcap_wide()`補上`_universe_clause()`，是另一個需要
使用者/研究session決定的獨立議題，這支診斷腳本**不代為決定、不改生產
程式**，只誠實記錄現狀。

這支腳本是**唯讀診斷**，不重跑L2、不改`triggers.py`/`monitor.py`現行
生產程式，只回答兩個問題：

①**窄指標的動態範圍，是否明顯大於現用的前20%佔比**？——對照前10檔
  （固定檔數，近似真實「幾大巨頭」的概念）、前1%（by rank，隨母體檔數
  變動，介於前10檔與前20%之間）、前20%（現用`q1_weight`定義，by rank）
  三個指標，在2023Q4~2025Q4這9個window4樣本點上的範圍。

②**p90門檻排除2009-2011噪訊窗後是否仍不觸發**？——`_prelim_a1_index_
  concentration_us.py`算出的p90=+8.61pp，校準窗（2009-2023, n=52）疊了
  2009Q3→2011Q3那幾筆單季±17pp的異常跳動（疑為早期市值資料覆蓋不全／
  母體檔數劇烈變動造成，見上方更正）。這裡重算一次p90，排除任何起訖
  落在2009-2011的8季窗，看門檻校準本身是不是問題所在，還是即使下修
  門檻依然不觸發（若仍不觸發，代表瓶頸是①的指標動態範圍不足，不是
  門檻設太高）。

直接用`monitor.fetch_mcap_wide()`（含`SANITY_MAX_CLOSE`離群值防線），跟
`_prelim_a1_index_concentration_us.py`同一套資料清洗邏輯，不自己重寫SQL
（包括沿用它「沒有套`_universe_clause()`」這個現狀，上方已明確記錄）。

用法：cd code; PYTHONIOENCODING=utf-8 python -m app._check_us_m1d_narrow_metric
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from database import Database  # noqa: E402
from app import monitor  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
TRAIN_START = "2009-01-01"  # 同`_prelim_a1_index_concentration_us.py`：collector回補起點
TRAIN_END = "2023-12-31"    # 只用訓練期資料，不可碰實驗期（2024-2025）
HORIZON = 8                 # 跟window4同長度（8季）


def concentration_at(wide: pd.DataFrame, d: str) -> dict:
    row = monitor.asof_row(wide, d).dropna()
    total = row.sum()
    sorted_desc = row.sort_values(ascending=False)
    ranks = row.rank(pct=True)
    return {
        "n_universe": len(row),
        "top10_weight": float(sorted_desc.iloc[:10].sum() / total),
        "top1pct_weight": float(row[ranks >= 0.99].sum() / total),
        "top20pct_weight": float(row[ranks >= 0.80].sum() / total),  # 現用q1_weight定義
    }


def main():
    db = Database("US")
    conn = db.create_connection()
    print(f"讀取美股市值資料（{TRAIN_START} 起，含SANITY_MAX_CLOSE離群值防線）…")
    wide = monitor.fetch_mcap_wide(conn, market="US", start=TRAIN_START)

    qe = pd.date_range("2009-03-31", "2025-12-31", freq="Q")
    rows = []
    for d in qe:
        c = concentration_at(wide, d.strftime("%Y-%m-%d"))
        rows.append({"q": f"{d.year}Q{d.quarter}", "date": d, **c})
    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "us_m1d_narrow_metric_2026-10-20.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"寫入 {out_path}（{len(df)} 季）\n")

    # --- ①窄指標 vs 現用指標，在window4的9個樣本點（2023Q4起8季末）的動態範圍 ---
    w4 = df[(df.date >= "2023-12-31") & (df.date <= "2025-12-31")]
    print(f"=== ①窄指標 vs 現用前20%佔比，window4樣本點（{len(w4)}季）動態範圍 ===")
    for col, label in [("top10_weight", "前10檔"), ("top1pct_weight", "前1%(~45~50檔)"),
                       ("top20pct_weight", "前20%(~890~990檔,現用)")]:
        lo, hi = w4[col].min(), w4[col].max()
        print(f"  {label:<16} {lo:.2%} ~ {hi:.2%}（範圍 {(hi-lo)*100:.2f}pp）")

    # --- ②p90門檻：全樣本 vs 排除2009-2011噪訊窗 ---
    train = df[df.date <= TRAIN_END].reset_index(drop=True)
    train["chg_8q"] = train["top20pct_weight"].shift(-HORIZON) - train["top20pct_weight"]
    train["window_start"] = train["date"]
    train["window_end"] = train["date"].shift(-HORIZON)
    full = train.dropna(subset=["chg_8q"])

    noisy_mask = (full["window_start"].dt.year.between(2009, 2011)
                 | full["window_end"].dt.year.between(2009, 2011))
    excl = full[~noisy_mask]

    p90_full, p75_full = full["chg_8q"].quantile(.90), full["chg_8q"].quantile(.75)
    p90_excl, p75_excl = excl["chg_8q"].quantile(.90), excl["chg_8q"].quantile(.75)

    v_2023 = df.loc[df.date == "2023-12-31", "top20pct_weight"].iloc[0]
    v_2025 = df.loc[df.date == "2025-12-31", "top20pct_weight"].iloc[0]
    actual_chg = v_2025 - v_2023

    print(f"\n=== ②p90門檻校準：全樣本 vs 排除2009-2011起訖窗 ===")
    print(f"  全樣本(2009-2023) n={len(full)}  p90={p90_full*100:+.2f}pp  p75={p75_full*100:+.2f}pp")
    print(f"  排除2009-2011起訖窗 n={len(excl)}  p90={p90_excl*100:+.2f}pp  p75={p75_excl*100:+.2f}pp")
    print(f"  2023Q4->2025Q4 實際變動={actual_chg*100:+.2f}pp")
    verdict = "觸發" if actual_chg > p90_excl else "不觸發"
    print(f"  用排除後的p90({p90_excl*100:.2f}pp)判斷：{verdict}")


if __name__ == "__main__":
    main()
