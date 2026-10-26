# -*- coding: utf-8 -*-
"""2026-10-18：美股版的M1-D門檻校準，跟台股`_prelim_a1_index_concentration.py`
同一個方法，但這是美股第一次真正做這件事——之前美股M1-D的X=1.86pp是**借用
台股的校準結果**（`triggers.py`的`X_P90`是單一全域常數，沒有分市場），原因
是當時美股`outstanding_shares`在2022年以前完全是空的，市值資料覆蓋率只有
200~650家，不夠拿來做15年以上的歷史分布校準。

2026-10-17 collector完成`USSharesOutstandingBackfiller`回補（`--from-date
2009-01-01`）後，美股市值覆蓋率從2009年開始大幅改善，終於有條件做美股自己的
校準——但只能用**2009-2023（15年）**，不是台股的2007-2023（17年）：
2006-2008這三年collector沒有回補，覆蓋率仍停留在舊的~200家，不足採用。

直接用`monitor.fetch_mcap_wide()`（含`SANITY_MAX_CLOSE`離群值防線、
`database._universe_clause()`Russell 3000篩選），不自己重寫SQL——跟production
用同一套資料清洗邏輯，避免兩邊各算一次又產生新的不一致。

🔴🔴 2026-10-21這支腳本本身的`main()`只印全樣本p90/p75（2009-2023, n=52），
**但`triggers.py`的`X_P90["US"]`/`X_P75["US"]`實際採用的是排除2009-2011
起訖窗後的版本**（n=40）——這是2026-10-21才補上的方法學修正：2009-2011
那幾年即使套了`_universe_clause()`，個別公司的市值數值本身仍有誤（collector
2026-10-21回補前），導致前20%佔比假性衝到99%+，墊高全樣本p90。這裡補上
`exclude_noisy_window()`把這道邏輯併回官方校準腳本，避免「正式門檻的真正
來源其實是另一支side-script」這種不透明的狀態——全樣本版本保留當對照
（print時標註兩個版本），但`main()`最後印的「建議採用值」是排除後的版本，
跟`triggers.py`常數同步。
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
TRAIN_START = "2009-01-01"  # collector回補的實際起點，2006-2008覆蓋率不足不採用
TRAIN_END = "2023-12-31"    # §11.7：只用訓練期資料，不可碰實驗期（2024-2025）


def main():
    db = Database("US")
    conn = db.create_connection()
    print(f"讀取美股市值資料（{TRAIN_START} 起，含SANITY_MAX_CLOSE離群值防線）…")
    wide = monitor.fetch_mcap_wide(conn, market="US", start=TRAIN_START)

    def q1_weight_at(d: str) -> float:
        row = monitor.asof_row(wide, d).dropna()
        ranks = row.rank(pct=True)
        return float(row[ranks >= 0.8].sum() / row.sum())

    qe = pd.date_range("2009-03-31", "2025-12-31", freq="Q")
    rows = []
    for d in qe:
        rows.append({"q": f"{d.year}Q{d.quarter}", "date": d,
                    "idx_q1_weight": q1_weight_at(d.strftime("%Y-%m-%d"))})
    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "a1_index_concentration_series_US.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"寫入 {out_path}（{len(df)} 季）")

    train = df[df.date <= TRAIN_END].reset_index(drop=True)
    HORIZON = 8   # 跟window4同長度（8季）
    train["chg_8q"] = train["idx_q1_weight"].shift(-HORIZON) - train["idx_q1_weight"]
    train["window_start"] = train["date"]
    train["window_end"] = train["date"].shift(-HORIZON)
    dist_full = train.dropna(subset=["chg_8q"])

    noisy_mask = (dist_full["window_start"].dt.year.between(2009, 2011)
                 | dist_full["window_end"].dt.year.between(2009, 2011))
    dist_excl = dist_full[~noisy_mask]

    print(f"\n=== 美股指數端Q1集中度「8季後累計變動」歷史分布（{TRAIN_START[:4]}"
         f"-{TRAIN_END[:4]}）===")
    for label, dist in [("全樣本", dist_full["chg_8q"]), ("排除2009-2011起訖窗", dist_excl["chg_8q"])]:
        print(f"  [{label}] n={len(dist)}  平均:{dist.mean()*100:+.2f}pp  "
             f"標準差:{dist.std()*100:.2f}pp  p75={dist.quantile(.75)*100:+.2f}pp  "
             f"p90={dist.quantile(.90)*100:+.2f}pp  最大:{dist.max()*100:+.2f}pp  "
             f"最小:{dist.min()*100:+.2f}pp")

    print(f"\n對照：台股既有門檻（2007-2023校準）p75=1.20pp／p90=1.86pp（D27）")

    v_2023 = df.loc[df.date == "2023-12-31", "idx_q1_weight"].iloc[0]
    v_2025 = df.loc[df.date == "2025-12-31", "idx_q1_weight"].iloc[0]
    actual_chg = v_2025 - v_2023

    # 🔴 2026-10-21：採用排除2009-2011後的版本當production門檻來源
    # （須跟triggers.py的X_P90["US"]/X_P75["US"]同步，見本檔docstring）。
    p90_new, p75_new = dist_excl["chg_8q"].quantile(.90), dist_excl["chg_8q"].quantile(.75)
    pctile = (dist_excl["chg_8q"] < actual_chg).mean()
    print(f"\n=== 反查：2023-12-31({v_2023:.2%}) -> 2025-12-31({v_2025:.2%})，"
         f"實際變動 {actual_chg*100:+.2f}pp ===")
    print(f"  落在美股自己（排除2009-2011版）的歷史分布第 {pctile*100:.1f} 百分位")
    print(f"  **建議採用值**（排除2009-2011版）：p90={p90_new*100:.4f}pp／"
         f"p75={p75_new*100:.4f}pp——應同步寫入triggers.py的X_P90/X_P75[\"US\"]")
    print(f"  若用此p90（{p90_new*100:.2f}pp）："
         f"{'✅ 會觸發' if actual_chg > p90_new else '❌ 不會觸發'}")

    print("\n=== 逐季相對2023-12-31的累計變動（用排除2009-2011版新門檻判斷）===")
    for _, r in df[(df.date >= "2024-01-01") & (df.date <= "2025-12-31")].iterrows():
        dev = r["idx_q1_weight"] - v_2023
        flag = "🔴 超過新p90" if dev > p90_new else ("🔶 超過新p75" if dev > p75_new else "")
        print(f"  {r['q']}: {r['idx_q1_weight']:.2%}  累計變動 {dev*100:+.2f}pp  {flag}")


if __name__ == "__main__":
    main()
