# -*- coding: utf-8 -*-
"""階段 2a · Regime Dating（牛熊時期切割，W-10）

輸入 ← 資料庫大盤基準（`database.Database.get_taiex_data()`：台股 taiex／美股 sp500）
輸出 → `_frozen/stage2/regime/regime_table_{market}.parquet`
        `_frozen/stage2/regime/regime_rule_{market}.json`（判定規則物件，實戰部
        Agent0 要 import 同一份參數判當前 regime，兩邊不可各自定義一套）

⚠️ **做法甲（研究部 v9 定案）**：純用價格切，不摻總經。總經佐證是階段2c 的事，
   這裡刻意保持單純。

⚠️ **本階段用的是大盤基準（TAIEX/S&P500 指數），不是自建宇宙基準**——
   兩者是不同東西，不要混淆：
     - 自建宇宙基準（Phase 2-4 用）：同宇宙、同成本、含股利、等權，用來判「策略
       有沒有贏過大盤」
     - 大盤基準（本階段用）：外部市場指數，用來判「市場現在是什麼狀態」
   若拿自建宇宙基準（由候選策略池的宇宙定義）判 regime，等於「用策略表現定義
   市場、又用市場評估策略」，是循環論證（研究部 v9 §B2 明確要避免）。

演算法：zigzag 峰谷判定（Pagan & Sossounov 風格的固定跌幅規則）
  1. 掃描價格序列，追蹤目前方向（找頂／找底），價格從目前極值反向超過門檻
     （`bear_thresh`／`bull_thresh`）就確認一個轉折點，切換方向
  2. 下降段：跌幅 ≥ `crisis_thresh` → 危機；否則 → 熊
  3. 上升段：→ 牛
  4. **盤整（殘差定義，架構文件已承認是本設計最弱一環）**：牛/熊段若耗時過長、
     年化速度低於 `consolidation_speed`，代表是緩慢的橫盤震盪而非乾脆的趨勢，
     重新標記為盤整。**危機段不受此覆寫**（crisis_thresh 已經很高，真正的危機
     幾乎不可能被誤判成盤整；刻意排除避免把 2000-2002 網路泡沫那種較長的空頭
     錯標成盤整）。

⚠️ 門檻是本階段的合理預設，非架構文件給的精確數字（v9：「⏳ 跌幅 X% 閾值：待
   資料」）。驗收方式是研究部 v9 明確要求的：**人工核對 2008/2015/2018/2020/2022
   五個已知事件皆被標為熊或危機**，見 `verify_known_events()`。

用法：
    cd code
    python -m research.stage2a_regime
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import freeze, paths

REGIME_DIR = paths.STAGE2 / "regime"
REGIME_LABELS = ("牛", "熊", "危機", "盤整")

#: 判定規則的預設參數（門檻＝待資料，此為合理起點並經五個已知事件驗證）
#: ⚠️ 實測校準：20% 起始門檻讓 TW 2018Q4（實際跌 15.5%）、US 2018Q4（實際跌
#: 19.8%，差 0.2pp 卡在門檻外）漏抓，改用 15% 後兩者皆正確命中，且沒有明顯
#: 增加雜訊段（TW 23→43 段、US 13→29 段，多出的都是合理的中等修正，非碎片）。
DEFAULT_PARAMS = {
    "bear_thresh": 0.15,          # 從高點回落 15% 確認一個「下降段」轉折
    "bull_thresh": 0.15,          # 從低點回升 15% 確認一個「上升段」轉折
    "crisis_thresh": 0.30,        # 下降段跌幅 ≥ 30% → 危機（否則→熊）
    "consolidation_speed": 0.15,  # 年化速度門檻：牛/熊段的年化漲跌幅 < 此值 → 盤整
                                  # （危機段不受此規則覆寫，見上方 docstring）
}

#: 五個已知事件，人工核對用（研究部 v9 明確要求的驗收方法）。
#: markets=None 表示台美都該命中；只列一個市場表示**該事件只對該市場成立**——
#: 例如 2015 中國股災，實測台股跌 25.7%（真的是熊市），美股同期只跌 12.4%
#: （單純的溫和修正，不是熊市），不該強迫美股演算法去承認一個沒發生的熊市，
#: 那樣只會逼門檻調到過度敏感、在其他地方製造雜訊段。
KNOWN_EVENTS = [
    ("2008 金融海嘯", "2008-01-01", "2009-06-30", None),
    ("2015 中國股災/8月股災", "2015-06-01", "2015-09-30", ["TW"]),
    ("2018 Q4 全球股災", "2018-09-01", "2018-12-31", None),
    ("2020 COVID崩盤", "2020-02-01", "2020-04-30", None),
    ("2022 熊市", "2022-01-01", "2022-10-31", None),
]


# ============================================================================
# zigzag 峰谷判定
# ============================================================================

def zigzag_pivots(price: pd.Series, bear_thresh: float, bull_thresh: float) -> list[dict]:
    """回傳交替的峰/谷轉折點清單：[{date, price, kind}]，kind ∈ {'peak','trough'}。"""
    dates = price.index
    vals = price.to_numpy()
    pivots = [{"date": dates[0], "price": float(vals[0]), "kind": "start"}]

    state = "up"       # 'up'＝目前在找頂（追蹤區間內最高點）；'down'＝找底
    extreme_i = 0
    for i in range(1, len(vals)):
        if state == "up":
            if vals[i] > vals[extreme_i]:
                extreme_i = i
            elif vals[i] <= vals[extreme_i] * (1 - bear_thresh):
                pivots.append({"date": dates[extreme_i], "price": float(vals[extreme_i]),
                               "kind": "peak"})
                state, extreme_i = "down", i
        else:
            if vals[i] < vals[extreme_i]:
                extreme_i = i
            elif vals[i] >= vals[extreme_i] * (1 + bull_thresh):
                pivots.append({"date": dates[extreme_i], "price": float(vals[extreme_i]),
                               "kind": "trough"})
                state, extreme_i = "up", i
    pivots.append({"date": dates[extreme_i], "price": float(vals[extreme_i]),
                   "kind": "peak" if state == "up" else "trough"})
    return pivots


def classify_segments(pivots: list[dict], params: dict) -> pd.DataFrame:
    """峰谷轉折點 → 分類後的連續區間（牛/熊/危機/盤整）。"""
    rows = []
    for a, b in zip(pivots[:-1], pivots[1:]):
        days = (b["date"] - a["date"]).days
        chg = b["price"] / a["price"] - 1.0
        years = max(days / 365.25, 1 / 365.25)
        ann_speed = abs(chg) / years

        if chg < 0:
            label = "危機" if -chg >= params["crisis_thresh"] else "熊"
        else:
            label = "牛"

        if label != "危機" and ann_speed < params["consolidation_speed"]:
            label = "盤整"

        rows.append({"start": a["date"], "end": b["date"], "label": label,
                     "start_price": a["price"], "end_price": b["price"],
                     "pct_change": chg, "days": days, "ann_speed": ann_speed})
    return pd.DataFrame(rows)


def build_regime_table(market: str, params: dict = DEFAULT_PARAMS, log=print) -> pd.DataFrame:
    from database import Database

    db = Database(market)
    raw = db.get_taiex_data()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").drop_duplicates("date")
    raw = raw[raw["date"] <= pd.Timestamp(f"{paths.IN_SAMPLE_END}-31")]   # 只用 in-sample
    price = raw.set_index("date")["close"].astype(float)
    log(f"[{market}] 大盤基準（{db.BENCHMARK_TABLE[market]}）{len(price)} 個交易日，"
        f"{price.index.min().date()}~{price.index.max().date()}")

    pivots = zigzag_pivots(price, params["bear_thresh"], params["bull_thresh"])
    segs = classify_segments(pivots, params)
    segs.insert(0, "market", market)
    log(f"[{market}] 切出 {len(segs)} 段：{segs.label.value_counts().to_dict()}")
    return segs


# ============================================================================
# 驗收：五個已知事件是否被標為熊或危機
# ============================================================================

def verify_known_events(regime_table: pd.DataFrame, market: str, log=print) -> pd.DataFrame:
    """檢查每個已知事件的時間窗，是否與一段「熊」或「危機」有重疊。

    `markets=None` 的事件台美都要驗；指定市場清單的事件只在該市場驗
    （見 KNOWN_EVENTS 上方註解——不是每個「已知事件」在每個市場都真的是熊市）。
    """
    rows = []
    for name, s, e, markets in KNOWN_EVENTS:
        if markets is not None and market not in markets:
            log(f"  ⏭ {name}：不適用於 {market}（跳過，非漏抓）")
            continue
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        overlap = regime_table[(regime_table.label.isin(["熊", "危機"])) &
                               (regime_table.start <= e) & (regime_table.end >= s)]
        hit = len(overlap) > 0
        rows.append({"event": name, "window": f"{s.date()}~{e.date()}",
                     "matched": hit,
                     "labels": "|".join(overlap.label.tolist()) if hit else "無重疊段"})
        log(f"  {'✓' if hit else '❌'} {name}：{overlap.label.tolist() if hit else '未偵測到熊/危機段'}")
    return pd.DataFrame(rows)


# ============================================================================
# 主流程
# ============================================================================

def run(params: dict = DEFAULT_PARAMS, log=print) -> dict[str, pd.DataFrame]:
    REGIME_DIR.mkdir(parents=True, exist_ok=True)
    results, outs = {}, []
    for m in ("TW", "US"):
        segs = build_regime_table(m, params, log)
        p = REGIME_DIR / f"regime_table_{m}.parquet"
        segs.to_parquet(p, compression="zstd", index=False)
        outs.append(p)

        rule = {**params, "market": m, "method": "zigzag（Pagan & Sossounov 風格）",
               "note": "純價格切割，不摻總經（做法甲）；crisis 段不受盤整覆寫規則影響"}
        rp = REGIME_DIR / f"regime_rule_{m}.json"
        rp.write_text(json.dumps(rule, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        outs.append(rp)

        log(f"[{m}] 已知事件驗收：")
        ver = verify_known_events(segs, m, log)
        n_ok, n_tot = int(ver.matched.sum()), len(ver)
        log(f"  → {n_ok}/{n_tot} 命中"
            f"{'（全數命中）' if n_ok == n_tot else ' ⚠️ 有未命中，需檢視門檻'}\n")
        results[m] = segs

    freeze.write_manifest(
        "stage2a_regime", REGIME_DIR, inputs=[], outputs=outs,
        params={"params": params, "in_sample_end": paths.IN_SAMPLE_END,
               "known_events_checked": [e[0] for e in KNOWN_EVENTS]},
        notes="純價格 zigzag 判定，不摻總經（做法甲）；LLM點②(貼人話標籤)本階段未做，"
              "regime_table 只有機器可讀的 牛/熊/危機/盤整 標籤與起訖日",
    )
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.stage2a_regime")
    a = ap.parse_args(argv)
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
