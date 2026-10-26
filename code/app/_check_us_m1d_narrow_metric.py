# -*- coding: utf-8 -*-
"""2026-10-20：回應同儕審閱（`stock-factor-lab-63`論文寫作session）對美股M1-D
「前20%佔比全程NONE」的質疑——這個指標用前20%（by rank）衡量集中度，範圍
本身可能太粗，測不到「前幾大巨頭」這種更窄的集中度事件（Goldman報告記載的
「Magnificent 7」現象）。

🔴 本檔案的②已跑過三輪，結論逐輪修正，依時間序：
1. 2026-10-20（原始）：p90=8.61pp，headroom問題尚未被發現，誤以為「全程
   NONE」是乾淨的結論。
2. 2026-10-21第一輪（修`monitor.fetch_mcap_wide('US')`沒套`_universe_
   clause()`的bug後，p90→11.37pp）：同儕複核算術發現美股2023Q4登記水準
   90.2%、上升空間只有9.78pp，小於這組p90——TRIGGERED在數學上不可能發生，
   NONE是門檻設計的必然結果，不是實證發現。
3. 2026-10-21第二輪（collector回補美股歷史市值資料的品質問題後重新校準，
   p90→**1.97pp**，排除2009-2011噪訊窗）：母體檔數本身沒變（universe_
   clause修正後`n_universe`跟第一輪一樣），是2009-2012那幾年**個別公司的
   market_capital/close/outstanding_shares數值本身有誤**，導致前20%佔比
   多次假性衝到99%+，墊高了p90。collector修完資料後，2023Q4登記水準
   89.41%、上升空間10.59pp，遠大於新p90——**TRIGGERED在數學上重新變得
   可能，用真實2024-2025資料跑`triggers.evaluate_quarter()`確認2025Q1起
   OBSERVING、2025Q3起TRIGGERED**（見`文件/實驗收尾追蹤.md`§七/§八）。
   **已完成的美股L2正式模擬建立在過期的「全程NONE」假設上，需要使用者
   裁示是否重跑**——跟第一輪「重跑也一樣」的無資訊結論完全不同。

這支腳本是**唯讀診斷**，不重跑L2、不改`triggers.py`/`monitor.py`現行
生產程式（兩者的修正都是另外獨立執行、經使用者裁示的動作，不是這支診斷
腳本自己做的），只回答兩個問題：

①**窄指標的動態範圍，是否明顯大於現用的前20%佔比**？——對照前10檔
  （固定檔數，近似真實「幾大巨頭」的概念）、前1%（by rank，隨母體檔數
  變動，介於前10檔與前20%之間）、前20%（現用`q1_weight`定義，by rank）
  三個指標，在2023Q4~2025Q4這9個window4樣本點上的範圍。

②**p90門檻排除2009-2011噪訊窗後跟全樣本比較**——`_prelim_a1_index_
  concentration_us.py`是正式的校準腳本（`triggers.X_P90["US"]`/
  `X_P75["US"]`的來源，每次universe_clause或底層資料更新後都要記得
  同步重跑那支，不能只跑這支診斷腳本，見上方②第二輪的教訓）；這裡只是
  用同一套方法在同一支腳本裡重算一次，拿來跟①的動態範圍對照用，不是
  production門檻真正的來源。

直接用`monitor.fetch_mcap_wide()`（含`SANITY_MAX_CLOSE`離群值防線、
`_universe_clause()`Russell 3000篩選），跟`_prelim_a1_index_concentration_
us.py`同一套資料清洗邏輯，不自己重寫SQL。

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
    for col, label in [("top10_weight", "前10檔"), ("top1pct_weight", "前1%(~27~29檔)"),
                       ("top20pct_weight", "前20%(~547~583檔,現用)")]:
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
