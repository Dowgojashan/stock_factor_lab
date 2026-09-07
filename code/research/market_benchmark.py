# -*- coding: utf-8 -*-
"""M-17 · 真正的市場基準與「等權 vs 市值加權」分解（2026-09-07）

🔴 **這支腳本要修正一個把污染當優點的論證**。`開發待辦追蹤.md` 12.2 延後 M-07/M-17
的理由是：

> B_all 已是比大盤更難贏的標竿（候選池全是全期間贏家，OOS 19.06% vs 大盤
> 8.43%/11.06%），「贏 B_all」多數情況蘊含「贏大盤」。

**這句話有三個問題**：
① **比錯窗**：19.06% 是 **OOS 窗（2019-2025）**，8.43%/11.06% 是**全期**自建基準
   ——製造出假的 10pp 差距。
② **同窗對比後，台股的 B_all 是輸的**（17.59% vs 台股報酬指數 20.91%）。
③ ⇒ **「贏 B_all 蘊含贏大盤」在台股完全不成立。**

---------------------------------------------------------------------------
🔴 兩個推翻稽核清單前提的查證
---------------------------------------------------------------------------
**① 不需要重建等權全市場序列——真正的指數已經在資料庫裡**
   `taiex`／`sp500`（價格指數）與 **`taiex_tr`／`sp500_tr`（報酬指數）**。

**② 該用報酬指數，不是價格指數**（2026-09-07 三重查證）
   ① `contracts.py:49` 註解：自建宇宙基準「同宇宙、同成本、**含股利**、等權」
   ② `code/universe_benchmark.py` docstring 寫得更直白：
      「我們的策略用 TEJ **還原收盤價**（配息已還原），**含股利**⋯
        一直在拿含息的策略比不含息的大盤，每個『贏大盤』都被**高估約 3~4pp**」
      ——**專案 2026-08-08 就發現這件事**，`taiex_tr`/`sp500_tr` 正是 collector
      於 08-11 為此匯入的。
   ③ 資料佐證：台積電 2018-12-28 在 `stock.close` 是 **203.38**，
      實際未還原收盤約 225.5 ⇒ **表中就是還原價**。
   ⇒ 對照必須用報酬指數（TR）。隱含股利率交叉檢查：
     台股 3.87~4.18pp、美股 1.84~2.33pp（三個不同期間），合理。

---------------------------------------------------------------------------
🔴 核心發現：落差幾乎完全來自「等權 vs 市值加權」
---------------------------------------------------------------------------
台股 2025-12：**台積電一檔佔總市值 40.23%**、前 5 大合計 49.75%。
而候選策略平均持股 **44.2 檔、等權**，單檔權重只有 2.26%——**差 18 倍**。
台股報酬指數本質上是「40% 台積電 + 60% 其他」。

🔴 **`stock.close` 是還原價（見上），所以下面兩個數字本來就含息**——
   不需要再估計股利，可以直接跟報酬指數並排：

    台股全市場 2019-01~2025-12（皆含息）
      TAIEX 報酬指數（市值加權）   20.91%
      本模組算的全市場市值加權      19.90%   ← 差 1pp：TAIEX 是自由流通量加權且只含
                                              上市，本模組含上櫃且 9.2% mcap 缺值
      A_hrp                        19.55%
      B_all（候選池等權）           17.59%
      本模組算的全市場等權          15.02%

⇒ **A_hrp 相對「等權市場」是 +4.53pp**（實測，非估計），
  只是輸給市值加權指數 1.36pp。

**一句話的成因**：市值加權比等權多賺 **4.88pp**，而 A_hrp 只比等權市場多賺
**4.53pp**——**差 0.35pp，剛好不夠**。那 4.88pp 來自台積電佔總市值 40.23%。

⚠️ **這是特定時期的現象，不是永久性質**：全部 45 個窗平均，台股 A_hrp 的超額是
   **+2.60pp、勝率 71.1%**。**所有落敗都集中在 2021 年之後起點、且延伸到 2025 的
   四個窗**（市場 CAGR 16.21~30.89%，台積電 AI 行情）；2013~2020 起點的 17 個窗
   四組全贏。H-12 的單一窗（2019-2025）剛好完整涵蓋那段行情。

⚠️ **美股的市值加權算不出來**：查證 `stock.market_capital` 在美股有 **98.8% 是 NULL**
   （台股只有 9.2%）。任何用它算的美股市值加權都是在 1.2% 的偏誤子集上算的，
   **本模組拒絕輸出該數字**。

用法：
    cd code                      # ⚠️ 需要 config.ini，必須在 code/ 下執行
    python -m research.market_benchmark
    python -m research.market_benchmark --skip-db   # 用已快取的指數序列，不連 DB
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, paths

#: 報酬指數（TR）——策略報酬含股利，故對照必須用 TR 而非價格指數。
TR_TABLES = {"TW": "taiex_tr", "US": "sp500_tr"}
PX_TABLES = {"TW": "taiex", "US": "sp500"}
#: 跨市場樹的基準：兩個報酬指數的**等權混合**，對應組合本身就是跨市場等權。
#: ⚠️ 未做匯率調整（見 M-06）——兩個指數各以自己的幣別計價。
GROUPS = ("A_hrp", "B_all", "D_top_cagr", "E_top_calmar")
METRICS = ("cagr", "mdd")


def _out_dir():
    return paths.ROOT / "_analysis_outputs_robustness"


def fetch_index_monthly(log=print) -> pd.DataFrame:
    """從資料庫拉四個指數的月報酬（TR 與價格各兩個）。長表：index_name × month × ret。"""
    import fcv_core  # noqa: F401  讓根目錄進 sys.path
    from database import Database

    db = Database("TW")
    cur = db.create_connection().cursor()
    rows = []
    for name in list(TR_TABLES.values()) + list(PX_TABLES.values()):
        cur.execute(f"SELECT date, close FROM {name} ORDER BY date")
        d = pd.DataFrame(cur.fetchall(), columns=["date", "close"])
        d["date"] = pd.to_datetime(d["date"])
        m = d.set_index("date")["close"].astype(float).resample("M").last().dropna()
        m.index = m.index.to_period("M")
        r = m.pct_change().dropna()
        rows.append(pd.DataFrame({"index_name": name, "month": r.index.astype(str),
                                  "ret": r.to_numpy()}))
        log(f"  [{name}] {len(r)} 個月報酬（{r.index.min()} ~ {r.index.max()}）")
    return pd.concat(rows, ignore_index=True)


def weighting_decomposition(log=print) -> pd.DataFrame:
    """等權 vs 市值加權的分解（只有台股可算，美股 market_capital 98.8% NULL）。"""
    import fcv_core  # noqa: F401
    from database import Database

    rows = []
    for mk, ex, uni in (("TW", "('TWSE')", ""),
                        ("US", "('NASDAQ','NYSE','AMEX')",
                         " AND c.company_symbol IN (SELECT symbol FROM russell3000)")):
        db = Database(mk)
        cur = db.create_connection().cursor()
        cur.execute(f"""SELECT COUNT(*), SUM(s.market_capital IS NULL)
                        FROM stock s JOIN company c ON s.company_id=c.id
                        WHERE c.exchange_name IN {ex}{uni} AND s.date>='2019-01-01'""")
        tot, nul = cur.fetchone()
        null_share = float(nul) / float(tot)
        log(f"  [{mk}] market_capital NULL 比例 {null_share:.1%}")

        cur.execute(f"""SELECT c.company_symbol, s.date, s.close, s.market_capital
                        FROM stock s JOIN company c ON s.company_id=c.id
                        WHERE c.exchange_name IN {ex}{uni}
                          AND s.date >= '2018-11-01' AND s.close > 0""")
        d = pd.DataFrame(cur.fetchall(), columns=["sym", "date", "close", "mcap"])
        d["date"] = pd.to_datetime(d["date"])
        d["m"] = d["date"].dt.to_period("M")
        d["close"] = d["close"].astype(float)
        d["mcap"] = pd.to_numeric(d["mcap"], errors="coerce")
        last = d.sort_values("date").groupby(["sym", "m"]).last().reset_index()
        px = last.pivot(index="m", columns="sym", values="close").sort_index()
        mc = last.pivot(index="m", columns="sym", values="mcap").sort_index()
        ret = px.pct_change()
        ret = ret[(ret.index >= pd.Period("2019-01", "M"))
                  & (ret.index <= pd.Period("2025-12", "M"))]
        w = mc.shift(1).reindex(ret.index)

        def _cagr(s):
            s = s.dropna()
            return float((1 + s).prod()) ** (12 / len(s)) - 1

        ew = _cagr(ret.mean(axis=1))
        # 🔴 美股的市值加權會落在 1.2% 的偏誤子集上，**拒絕輸出**
        cw = (float("nan") if null_share > 0.5
              else _cagr((ret * w).sum(axis=1) / w.where(ret.notna()).sum(axis=1)))
        # 最大一檔的市值佔比（台股的台積電）
        top = float("nan")
        if null_share <= 0.5:
            lastmc = mc.dropna(how="all").iloc[-1].dropna()
            top = float(lastmc.max() / lastmc.sum())
        rows.append({"market": mk, "window": "2019-01~2025-12",
                     "n_stocks_median": int(ret.notna().sum(axis=1).median()),
                     "mcap_null_share": null_share,
                     "equal_weight_cagr": ew, "cap_weight_cagr": cw,
                     "cap_minus_equal_pp": (cw - ew) * 100 if np.isfinite(cw) else float("nan"),
                     "top1_mcap_share": top})
    return pd.DataFrame(rows)


def _index_monthly(skip_db: bool, log=print) -> pd.DataFrame:
    p = _out_dir() / "market_index_monthly.parquet"
    if skip_db:
        if not p.exists():
            raise FileNotFoundError("沒有快取的指數序列，請先不加 --skip-db 跑一次")
        return pd.read_parquet(p)
    return fetch_index_monthly(log)


def build(skip_db: bool = False, log=print) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = _out_dir()
    freeze.verify_inputs(d / "_walkforward_matrix_manifest")
    idx = _index_monthly(skip_db, log)
    wide = idx.pivot(index="month", columns="index_name", values="ret")
    wide.index = pd.PeriodIndex(wide.index, freq="M")
    #: 跨市場基準 = 兩個 TR 指數的等權混合（對應組合本身就是跨市場等權）
    bench = {"TW": wide[TR_TABLES["TW"]], "US": wide[TR_TABLES["US"]]}
    bench["XM"] = pd.concat([bench["TW"], bench["US"]], axis=1).mean(axis=1)

    det = pd.read_csv(d / "walkforward_matrix_detail.csv")

    def _cagr(s):
        s = s.dropna()
        return float((1 + s).prod()) ** (12 / len(s)) - 1 if len(s) else float("nan")

    def _mdd(s):
        nav = (1 + s.dropna()).cumprod()
        return float((nav / nav.cummax() - 1).min()) if len(nav) else float("nan")

    # 每個 (樹 × OOS 區間) 的市場基準只算一次
    wins = det[["tree_key", "oos_start", "oos_end"]].drop_duplicates()
    recs = []
    for r in wins.itertuples():
        b = bench[r.tree_key]
        s = b[(b.index >= pd.Period(r.oos_start, "M"))
              & (b.index <= pd.Period(r.oos_end, "M"))]
        recs.append({"tree_key": r.tree_key, "oos_start": r.oos_start,
                     "oos_end": r.oos_end, "n_months": int(len(s)),
                     "mkt_cagr": _cagr(s), "mkt_mdd": _mdd(s)})
    mkt = pd.DataFrame(recs)
    log(f"  逐窗市場基準 {len(mkt)} 組 (樹 × OOS 區間)")

    m = det.merge(mkt, on=["tree_key", "oos_start", "oos_end"], how="left")
    if m.mkt_cagr.isna().any():
        raise AssertionError("有格子接不到市場基準——指數序列可能不涵蓋該 OOS 區間")

    rows = []
    for (t, g, ratio), gg in m[m.group.isin(GROUPS)].groupby(
            ["tree_key", "group", "ratio"], observed=True):
        for ratio_tag in (str(ratio),):
            rows.append({
                "tree_key": t, "group": g, "ratio": ratio_tag,
                "n_cells": int(len(gg)),
                "group_cagr_mean": float(gg.oos_cagr.mean()),
                "mkt_cagr_mean": float(gg.mkt_cagr.mean()),
                "excess_cagr_pp": float((gg.oos_cagr - gg.mkt_cagr).mean() * 100),
                "win_mkt_cagr": float((gg.oos_cagr > gg.mkt_cagr).mean()),
                "group_mdd_mean": float(gg.oos_mdd.mean()),
                "mkt_mdd_mean": float(gg.mkt_mdd.mean()),
                "win_mkt_mdd": float((gg.oos_mdd > gg.mkt_mdd).mean()),
            })
    cmp_df = pd.DataFrame(rows)
    for c in ("tree_key", "group", "ratio"):
        cmp_df[c] = cmp_df[c].astype("category")

    dec = weighting_decomposition(log) if not skip_db else pd.read_csv(
        d / "weighting_decomposition.csv")
    dec["market"] = dec["market"].astype("category")
    return idx, cmp_df[C.MARKET_BENCHMARK_COMPARE.names], dec[C.WEIGHTING_DECOMPOSITION.names]


def run(skip_db: bool = False, log=print):
    idx, cmp_df, dec = build(skip_db=skip_db, log=log)
    C.validate(cmp_df, C.MARKET_BENCHMARK_COMPARE, strict_columns=True)
    C.validate(dec, C.WEIGHTING_DECOMPOSITION, strict_columns=True)
    d = _out_dir()
    p0 = d / "market_index_monthly.parquet"
    p1 = d / "market_benchmark_compare.csv"
    p2 = d / "weighting_decomposition.csv"
    idx.to_parquet(p0, index=False)
    cmp_df.to_csv(p1, index=False, encoding="utf-8-sig")
    dec.to_csv(p2, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "market_benchmark", d / "_market_benchmark_manifest",
        inputs=[d / "walkforward_matrix_detail.csv"],
        outputs=[p0, p1, p2],
        params={"tr_tables": TR_TABLES, "px_tables": PX_TABLES,
                "xm_benchmark": "兩個 TR 指數的等權混合（未做匯率調整，見 M-06）",
                "why_tr": "contracts.BENCHMARK_CAGR 註解寫明自建基準含股利"},
        notes="M-17：真正的市場基準（報酬指數）+ 等權 vs 市值加權分解。"
              "🔴 美股的市值加權拒絕輸出——market_capital 98.8% NULL。"
              "⚠️ 資料庫來源，非 _frozen 凍結鏈；重跑需要 DB 連線。",
    )
    log(f"→ {p0.name} / {p1.name} / {p2.name}")
    return cmp_df, dec


def _report(cmp_df: pd.DataFrame, dec: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 96)
    log("M-17 · 真正的市場基準（報酬指數）與「等權 vs 市值加權」分解")
    log("=" * 96)

    log("\n【一】等權 vs 市值加權（同一批股票、同期間、僅價格）")
    log(f"  {'市場':<5}{'股票數':>8}{'mcap NULL':>11}{'等權':>10}{'市值加權':>10}"
        f"{'差(pp)':>9}{'最大一檔佔比':>13}")
    for r in dec.itertuples():
        cw = f"{r.cap_weight_cagr:>10.2%}" if np.isfinite(r.cap_weight_cagr) else f"{'不可算':>10}"
        dd = f"{r.cap_minus_equal_pp:>+9.2f}" if np.isfinite(r.cap_minus_equal_pp) else f"{'—':>9}"
        tp = f"{r.top1_mcap_share:>13.2%}" if np.isfinite(r.top1_mcap_share) else f"{'—':>13}"
        log(f"  {r.market:<5}{r.n_stocks_median:>8,}{r.mcap_null_share:>11.1%}"
            f"{r.equal_weight_cagr:>10.2%}{cw}{dd}{tp}")
    log("  🔴 美股的市值加權**拒絕輸出**——market_capital 98.8% NULL，")
    log("     任何用它算的數字都在 1.2% 的偏誤子集上。")

    log("\n【二】各組 vs 真正的市場報酬指數（逐窗，全 45 窗）")
    log(f"  {'樹':<5}{'組別':<14}{'比例':<9}{'組合 CAGR':>11}{'市場 CAGR':>11}"
        f"{'超額(pp)':>10}{'勝率':>8}{'MDD 勝率':>10}")
    order = {"all": 0, "legacy": 1, "0.01": 2, "0.03": 3, "0.05": 4, "0.1": 5}
    c = cmp_df.copy()
    c["_o"] = c.ratio.astype(str).map(order).fillna(9)
    for (t, g), gg in c.groupby(["tree_key", "group"], observed=True):
        for r in gg.sort_values("_o").itertuples():
            log(f"  {r.tree_key:<5}{r.group:<14}{str(r.ratio):<9}"
                f"{r.group_cagr_mean:>11.2%}{r.mkt_cagr_mean:>11.2%}"
                f"{r.excess_cagr_pp:>+10.2f}{r.win_mkt_cagr:>8.1%}{r.win_mkt_mdd:>10.1%}")
        log("")
    log("⚠️ 市場基準用**報酬指數**（策略報酬含股利，見模組 docstring）。")
    log("⚠️ 跨市場基準 = 兩個 TR 指數等權混合，**未做匯率調整**（見 M-06）。")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.market_benchmark")
    ap.add_argument("--skip-db", action="store_true",
                    help="用已快取的指數序列與分解表，不連資料庫")
    a = ap.parse_args(argv)
    _report(*run(skip_db=a.skip_db))
    return 0


if __name__ == "__main__":
    sys.exit(main())
