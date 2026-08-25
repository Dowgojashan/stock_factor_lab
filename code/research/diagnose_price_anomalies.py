# -*- coding: utf-8 -*-
"""診斷 · 價格資料異常清單（回答「是資料異常還是策略異常」）

起因：階段1 掃描偵測到某美股策略單日淨值 +3908%（NAV 22.58 → 905.0）。
追查 NUTX 的**原始 price:close**：2013-08-06 = 75.0 → 08-07 = 38,250.0，
之後長期停在 38,250（全期最高 56,250）。美股不存在這種價位。
→ **異常在來源價格資料；回測是忠實照著壞價格算的，邏輯無誤。**

方法論（v2 修正）：
  v1 從「交易報酬 > 200%」反推元凶，結果 3,143 檔中 2,885 檔價格其實正常——
  **持有期大漲 200% 對小型股是合法的**，這個門檻分不出「資料壞」與「真的漲」。
  v2 改從**價格端**判定（權威證據），再映射到策略：

  判定條件（任一成立即為資料異常）：
    A. 單日跳動 ≥ +300%  —— 合法個股極少單日 4 倍，且需**持續不回落**才算資料錯
    B. 股價絕對值不合理  —— 美股 > $1,000、台股 > 10,000 元
    C. 單日跳動 ≥ +100% 且 5 日內未回落 —— 未調整的反分割特徵

  「單日 +100% 但隨即回落」歸類為**真實暴漲**（生技解盲等），不列為資料異常。

輸出 → `_analysis_outputs_dataquality/`

用法：
    cd code
    python -m research.diagnose_price_anomalies
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import contracts as C
from . import paths

OUT = paths.ROOT / "_analysis_outputs_dataquality"

#: W-08：與 stage1_marks.py 的 data_glitch 共用同一個數字（contracts.PRICE_JUMP_EXTREME），
#: 避免這支「貴、按需跑」的深度診斷與 stage1 那支「便宜、每次都跑」的標記各自維護一份會漂移。
JUMP_EXTREME = C.PRICE_JUMP_EXTREME   # A：單日 +300%
JUMP_SUSPECT = 1.0        # C：單日 +100%
PERSIST_DAYS = 5          # C：幾日內未回落即視為持續性（＝資料錯，非真實暴漲）
PERSIST_KEEP = 0.5        # 回落判準：5 日後仍保有跳動幅度的 50% 以上
ABSURD_PRICE = {"US": 1_000.0, "TW": 10_000.0}


# ------------------------------------------------------------ 價格端判定

def find_price_anomalies(market: str, log=print) -> tuple[pd.DataFrame, dict]:
    """從原始 price:close 找出資料異常的 (股票, 日期)。"""
    from get_data import Data
    t0 = time.time()
    log(f"   [{market}] 載入原始 price:close …")
    px = Data(market=market).get("price:close")
    px.index = pd.to_datetime(px.index)
    log(f"   [{market}] {px.shape}，{time.time()-t0:.0f}s")

    ret = px.pct_change()
    stats = {"n_stocks": px.shape[1], "n_days": px.shape[0],
             "n_jump_100": int((ret >= JUMP_SUSPECT).sum().sum()),
             "n_jump_300": int((ret >= JUMP_EXTREME).sum().sum())}

    rows = []
    stack = ret[ret >= JUMP_SUSPECT].stack()
    log(f"   [{market}] 單日 ≥+100% 的 (股,日) 共 {len(stack)} 個，逐一取證 …")
    for (d, t), r in stack.items():
        s = px[t]
        pos = s.index.get_loc(d)
        before, after = s.iloc[pos - 1], s.iloc[pos]
        # 之後 PERSIST_DAYS 日的價格是否維持在跳動後的水準
        tail = s.iloc[pos + 1: pos + 1 + PERSIST_DAYS].dropna()
        persist = bool(len(tail) and
                       (tail.min() - before) / (after - before) >= PERSIST_KEEP)
        flags = []
        if r >= JUMP_EXTREME:
            flags.append("單日≥300%")
        if persist:
            flags.append("跳動後未回落")
        absurd = float(s.max()) > ABSURD_PRICE[market]
        if absurd:
            flags.append("股價絕對值不合理")

        # 判定：A 或 B 或 C
        is_anomaly = (r >= JUMP_EXTREME and persist) or absurd or \
                     (r >= JUMP_SUSPECT and persist)
        rows.append({
            "market": market, "stock_id": t, "jump_date": d.date(),
            "price_before": float(before), "price_after": float(after),
            "jump_x": float(after / before) if before else np.nan,
            "day_return": float(r), "persisted": persist,
            "price_max_alltime": float(s.max()), "price_min_alltime": float(s.min()),
            "verdict": "資料異常" if is_anomaly else "真實暴漲（保留）",
            "reason": "｜".join(flags) if flags else "單日≥100% 但隨即回落",
        })
    df = pd.DataFrame(rows)
    stats["n_anomaly_events"] = int((df.verdict == "資料異常").sum()) if len(df) else 0
    stats["n_anomaly_stocks"] = df[df.verdict == "資料異常"]["stock_id"].nunique() if len(df) else 0
    return df, stats


# ------------------------------------------------------- 映射到受影響策略

def map_to_strategies(anom: pd.DataFrame, log=print) -> pd.DataFrame:
    """哪些策略在異常發生當下持有該股票。"""
    bad = anom[anom.verdict == "資料異常"]
    key = {(r.market, r.stock_id): pd.Timestamp(r.jump_date) for r in bad.itertuples()}
    if not key:
        return pd.DataFrame()

    idx = pd.read_parquet(paths.STAGE0 / "candidate_index.parquet")
    rows, t0 = [], time.time()
    for i, r in enumerate(idx.itertuples(), 1):
        try:
            tr = pd.read_parquet(Path(r.artifacts_dir) / "trades.parquet",
                                 columns=["stock_id", "entry_date", "exit_date", "return"])
        except Exception:
            continue
        tr["stock_id"] = tr["stock_id"].astype(str)
        cand = tr[tr["stock_id"].isin(bad[bad.market == r.market]["stock_id"].values)]
        for b in cand.itertuples():
            jd = key.get((r.market, b.stock_id))
            if jd is None:
                continue
            ent, ext = pd.Timestamp(b.entry_date), pd.Timestamp(b.exit_date)
            if pd.notna(ent) and ent <= jd and (pd.isna(ext) or ext >= jd):
                rows.append({"strategy_uid": r.strategy_uid, "market": r.market,
                             "stock_id": b.stock_id, "jump_date": jd.date(),
                             "entry_date": ent.date(),
                             "exit_date": None if pd.isna(ext) else ext.date(),
                             "trade_return": float(b.return_ if hasattr(b, "return_") else b[4])})
        if i % 4000 == 0:
            log(f"      映射 {i}/{len(idx)} ({time.time()-t0:.0f}s)")
    log(f"   映射完成，{time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


def cagr_impact(uids: list[str]) -> pd.DataFrame:
    """把異常月的報酬換成該策略月報酬中位數，重算 CAGR，量化灌水。"""
    mo = pd.read_parquet(paths.STAGE1 / "returns_monthly.parquet")
    mo = mo[mo["strategy_uid"].isin(uids)]
    rep = pd.read_parquet(paths.STAGE0 / "candidate_index.parquet") \
            .set_index("strategy_uid")["CAGR"]
    rows = []
    for uid, g in mo.groupby("strategy_uid"):
        r = g["ret"].to_numpy(dtype=float)
        n_bad = int((r > 1.0).sum())
        med = float(np.median(r))
        fixed = np.where(r > 1.0, med, r)
        yrs = len(r) / 12.0
        raw = float(np.prod(1 + r) ** (1 / yrs) - 1)
        fix = float(np.prod(1 + fixed) ** (1 / yrs) - 1)
        rows.append({"strategy_uid": uid, "CAGR_reported": float(rep.get(uid, np.nan)),
                     "CAGR_without_anomaly": fix, "CAGR_inflation_pp": (raw - fix) * 100,
                     "n_anomaly_months": n_bad, "n_months": len(r)})
    return pd.DataFrame(rows).sort_values("CAGR_inflation_pp", ascending=False)


#: recall 檢查只看「實質造成CAGR灌水」的案例，不是「有沒有碰過異常股票」——
#: 後者範圍太廣（曾實測 49.7% 策略碰過某檔異常股，但中位數CAGR灌水=0，多是
#: 分散組合裡權重小到可忽略），拿它當 data_glitch 的驗收基準會逼出一支
#: 高誤報率的刀。真正該接住的是「灌水到有感」的案例，門檻見 contracts.py
#: MONTHLY_JUMP_EXTREME 的校準說明。
MATERIAL_CAGR_INFLATION_PP = 1.0


def cross_check_data_glitch(hits: pd.DataFrame, impact: pd.DataFrame, log=print) -> None:
    """W-08 · 把這次深度掃描的結果，拿去對 stage1_marks.py 的便宜信號 `data_glitch`。

    兩個方法抓的是同一件事（策略持有過資料異常的股票），但用不同資料、不同目的：
    這支從**原始股價**逐檔判定再映射回策略（貴、權威，含CAGR灌水量化）；
    `data_glitch` 從**策略自己的NAV/月報酬**直接判斷（便宜、stage1_marks每次都算，
    canary用途）。核對重點不是「兩者範圍完全一致」（`data_glitch` 本來就不該對
    每個碰過異常股但沒受影響的策略都標記，否則誤報率會高到沒有canary的意義），
    而是**recall on材質案例**：CAGR灌水>1個百分點的策略，data_glitch 不能漏標。
    """
    p = paths.STAGE1 / "strategy_marks.parquet"
    if not p.exists():
        log("\n⚠️ 找不到 strategy_marks.parquet，跳過 data_glitch 交叉核對"
            "（需先跑 python -m research.stage1_marks）")
        return
    marks = pd.read_parquet(p, columns=["strategy_uid", "data_glitch"])
    deep_hit = set(hits["strategy_uid"].unique()) if len(hits) else set()
    cheap_hit = set(marks.loc[marks.data_glitch, "strategy_uid"])

    log("\n" + "=" * 70)
    log("W-08 · data_glitch 交叉核對（stage1_marks 便宜信號 vs 本次深度掃描）")
    log("=" * 70)
    log(f"深掃命中（曾持有異常股票）{len(deep_hit)} 策略｜data_glitch 標記 {len(cheap_hit)} 策略｜"
        f"兩者重疊 {len(deep_hit & cheap_hit)}")

    if len(impact):
        material = impact[impact["CAGR_inflation_pp"] > MATERIAL_CAGR_INFLATION_PP]
        missed = material[~material["strategy_uid"].isin(cheap_hit)]
        log(f"\n【recall驗收】CAGR灌水 > {MATERIAL_CAGR_INFLATION_PP}pp 的材質案例共 "
            f"{len(material)} 個，data_glitch 命中 {len(material) - len(missed)} 個")
        if len(missed):
            log(f"⚠️ 漏標 {len(missed)} 個（門檻可能需要重新校準）：")
            log(missed[["strategy_uid", "CAGR_inflation_pp"]].head(10).to_string(index=False))
        else:
            log("✓ 材質案例（CAGR灌水>1pp）data_glitch 全數涵蓋（0漏標）")
    log(f"\n（附註：深掃命中的 {len(deep_hit)} 策略中，多數是碰過異常股但被分散組合"
        f"稀釋到CAGR幾無影響——data_glitch 刻意不標這些，避免canary誤報率過高）")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log = print

    log("步驟1 · 從原始價格判定資料異常")
    frames, stats = [], {}
    for m in C.MARKETS:
        df, st = find_price_anomalies(m, log)
        frames.append(df); stats[m] = st
    anom = pd.concat(frames, ignore_index=True)
    anom.sort_values(["verdict", "day_return"], ascending=[True, False]) \
        .to_csv(OUT / "price_anomaly_events.csv", index=False, encoding="utf-8-sig")

    stocks = (anom[anom.verdict == "資料異常"]
              .groupby(["market", "stock_id"])
              .agg(n_events=("jump_date", "size"), first_jump=("jump_date", "min"),
                   max_jump_x=("jump_x", "max"),
                   price_max=("price_max_alltime", "max"),
                   reason=("reason", "first")).reset_index()
              .sort_values("max_jump_x", ascending=False))
    stocks.to_csv(OUT / "price_anomaly_stocks.csv", index=False, encoding="utf-8-sig")

    log("\n步驟2 · 映射到受影響策略")
    hits = map_to_strategies(anom, log)
    if len(hits):
        hits.to_csv(OUT / "price_anomaly_hits.csv", index=False, encoding="utf-8-sig")
        impact = cagr_impact(hits["strategy_uid"].unique().tolist())
        impact = impact.merge(
            hits.groupby("strategy_uid")["stock_id"]
                .agg(lambda s: "｜".join(sorted(set(s)))).rename("anomaly_stocks"),
            on="strategy_uid", how="left")
        impact.to_csv(OUT / "price_anomaly_strategies.csv", index=False, encoding="utf-8-sig")
    else:
        impact = pd.DataFrame()

    _summary(anom, stocks, hits, impact, stats, log)
    cross_check_data_glitch(hits, impact, log)
    log(f"\n產出於 {OUT}")
    return 0


def _summary(anom, stocks, hits, impact, stats, log):
    log("\n" + "=" * 70)
    log("價格異常診斷 · 摘要")
    log("=" * 70)
    log("\n【全市場價格健康度】")
    for m, st in stats.items():
        log(f"  [{m}] {st['n_stocks']} 檔 × {st['n_days']} 交易日 | "
            f"單日≥100%: {st['n_jump_100']} 次 | 單日≥300%: {st['n_jump_300']} 次 | "
            f"判定資料異常: {st['n_anomaly_events']} 次 / {st['n_anomaly_stocks']} 檔")

    log(f"\n【判定分布】\n{anom.verdict.value_counts().to_string()}")
    log(f"\n【資料異常股票 {len(stocks)} 檔 · 跳動最大的 15 檔】")
    log(stocks.head(15).to_string(index=False))

    if len(hits):
        log(f"\n【受影響策略】{hits.strategy_uid.nunique():,} / {C.EXPECTED_ROWS_TOTAL} "
            f"({hits.strategy_uid.nunique()/C.EXPECTED_ROWS_TOTAL:.1%})")
        log(hits.groupby("market")["strategy_uid"].nunique().to_string())
        top = (hits.groupby(["market", "stock_id"])["strategy_uid"].nunique()
               .sort_values(ascending=False).head(10))
        log(f"\n影響最多策略的元凶股票:\n{top.to_string()}")
    if len(impact):
        log(f"\n【CAGR 灌水】中位 {impact.CAGR_inflation_pp.median():.2f}pp / "
            f"P90 {impact.CAGR_inflation_pp.quantile(.9):.2f}pp / "
            f"max {impact.CAGR_inflation_pp.max():.2f}pp")
        log(f"\n灌水最嚴重的 10 個策略:")
        log(impact.head(10)[["strategy_uid", "CAGR_reported", "CAGR_without_anomaly",
                             "CAGR_inflation_pp", "anomaly_stocks"]].to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
