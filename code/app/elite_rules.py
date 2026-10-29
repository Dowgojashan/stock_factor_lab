# -*- coding: utf-8 -*-
"""優等生權重規則（實戰收尾追蹤.md §四 步驟1，2026-10-27 老師會議後定案）。

老師 10/27 指示：單股上限 30% 是看過 window4 才挑的，論文不能只有這一格；要把
「誰是優等生（排名依據）× 取幾個 × 怎麼分 × 分多少錢」整張表實驗出來。本模組是
那張表的**正式規則實作**——離線網格腳本（`_elite_grid_offline.py`）與之後步驟2
的 `simulate.py` 共用同一份程式碼（避免重演離線腳本自寫權重邏輯、漏了市場分流
的坑，見 交接_2026-10-27.md §4-1）。

三段式（全部是純函式，不碰 DB／LLM）：
  1. `score_*()`：替「目前持股」打分數，分數越高越優先（S1~S6）
  2. `elite_weights()`：取前 N 名、照分法分配，回傳優等生權重（總和 1）
  3. `overlay()`：方案 b 疊加式——
     `最終權重 = 資金比例 × 優等生權重 + (1 − 資金比例) × 現行權重(control0)`
     資金比例=0 時逐位等於 control0；=1 時就是方案 a（只買優等生）

定案細節（出處：實戰收尾追蹤.md §四「步驟1 設計定案」）：
  - 只在目前持股（w0>0）裡挑優等生——老師：「我已經選出好學生來了，要怎麼利用優等生」
  - S2 因子方向＝策略自己選的桶（最低桶越低越好、最高桶越高越好、中間桶越靠近
    桶中心越好）；分數用全市場百分位（跟 q_band 同一個母體）；跨策略**簡單平均**
  - 同分（S1 票數最常見）＝**邊界平分**：同分群佔到的名額權重由群內平分
  - 照分數比例：用原始分數（皆為正值），追強用 1+r、押回彈用 1/(1+r)——
    若先轉成持股內百分位，前 N 名的百分位都約 0.99，比例分配會退化成平均分
  - S5 混合＝0.5×追強規則 + 0.5×押回彈規則（各自取 N 名），N=1 時實際持有 2 檔
  - 不設單股上限（資金比例本身控制集中度），改由呼叫端回報最大單股權重
"""
from __future__ import annotations

import math

import pandas as pd

#: 分法代號（N=1 時所有分法相同，只跑 "equal"）
SCHEMES = ("equal", "top1x2", "top2x2", "proportional", "tier")
SCHEME_LABELS = {"equal": "平均分", "top1x2": "第一名加碼(2倍)", "top2x2": "前兩名加碼(2倍)",
                 "proportional": "照分數比例", "tier": "階梯(N:…:1)"}


# ───────────────────────── 1. 分數 ─────────────────────────

def score_votes(holdings: dict[str, list[str]]) -> pd.Series:
    """S1 票數：每檔股票被幾個代表策略同時選中。"""
    cnt: dict[str, int] = {}
    for syms in holdings.values():
        for s in syms:
            cnt[s] = cnt.get(s, 0) + 1
    return pd.Series(cnt, dtype=float)


def band_direction_score(pct: pd.Series, band: int, nbands: int) -> pd.Series:
    """把全市場百分位（0~1，越大＝因子值越高）依策略自己選的桶轉成「越大越好」的分數。
    最低桶→1−pct；最高桶→pct；中間桶→1−|pct−桶中心|（越靠近桶中心越高）。"""
    band, nbands = int(band), int(nbands)
    if band == 0:
        return 1.0 - pct
    if band == nbands - 1:
        return pct
    center = (band + 0.5) / nbands
    return 1.0 - (pct - center).abs()


def score_factor(holdings: dict[str, list[str]], strat_rows: dict[str, dict],
                 pct_at: "callable", f1_weight: float) -> pd.Series:
    """S2 因子綜合分數。每個策略對它持有的每檔股票打
    `f1_weight×F1方向分數 + (1−f1_weight)×F2方向分數`（F2 缺值或策略無F2時只用F1），
    再對持有該股的所有策略**簡單平均**。

    `strat_rows[uid]`：含 F1_factor/F1_band/F1_nbands/F2_empty/F2_factor/F2_band/F2_nbands。
    `pct_at(factor) -> pd.Series`：該因子在 as_of 當天的全市場百分位（index=股票代號），
    取不到回 None。"""
    per_stock: dict[str, list[float]] = {}
    for uid, syms in holdings.items():
        if not syms:
            continue
        r = strat_rows[uid]
        p1 = pct_at(r["F1_factor"])
        if p1 is None:
            continue
        s1 = band_direction_score(p1.reindex(syms), r["F1_band"], r["F1_nbands"])
        if f1_weight < 1.0 and not r["F2_empty"]:
            p2 = pct_at(r["F2_factor"])
            s2 = (band_direction_score(p2.reindex(syms), r["F2_band"], r["F2_nbands"])
                  if p2 is not None else pd.Series(float("nan"), index=syms))
            comb = f1_weight * s1 + (1.0 - f1_weight) * s2
            sc = comb.where(s2.notna(), s1)          # F2 缺值→只用 F1
        else:
            sc = s1
        for s, v in sc.items():
            if pd.notna(v):
                per_stock.setdefault(s, []).append(float(v))
    return pd.Series({s: sum(v) / len(v) for s, v in per_stock.items()}, dtype=float)


def trailing_return(close: pd.DataFrame, syms: list[str], as_of: pd.Timestamp,
                    months: int) -> pd.Series:
    """as_of（取 <= as_of 最後交易日）往回 `months` 個月的簡單報酬，只用 as_of 以前的價格。"""
    idx = close.index
    d1 = idx[idx <= as_of].max()
    d0_cands = idx[idx <= as_of - pd.DateOffset(months=months)]
    if len(d0_cands) == 0:
        return pd.Series(dtype=float)
    d0 = d0_cands.max()
    cols = [s for s in syms if s in close.columns]
    p0, p1 = close.loc[d0, cols], close.loc[d1, cols]
    r = p1.div(p0).sub(1.0)
    return r[(p0 > 0) & (p1 > 0) & p0.notna() & p1.notna()]   # p1=0 會讓押回彈分數 1/(1+r) 變無限大


def score_greedy(ret: pd.Series) -> pd.Series:
    """S3 追強：近期報酬越高越優先；比例分配用的原始分數＝1+r（恆正）。"""
    return 1.0 + ret


def score_reversal(ret: pd.Series) -> pd.Series:
    """S4 押回彈：近期報酬越低越優先；比例分配用的原始分數＝1/(1+r)（恆正）。"""
    return 1.0 / (1.0 + ret)


def score_mcap(mcap_row: pd.Series, syms: list[str]) -> pd.Series:
    """S6 市值（＝現行市值傾斜的排名依據）。"""
    s = mcap_row.reindex(syms)
    return s[s.notna() & (s > 0)].astype(float)


# ───────────────────────── 2. 取前 N 名＋分配 ─────────────────────────

def slot_weights(n: int, scheme: str) -> list[float]:
    """名次 1..N 各自的權重（總和 1）。proportional 不走這裡。"""
    if n == 1:
        return [1.0]
    if scheme == "equal":
        raw = [1.0] * n
    elif scheme == "top1x2":
        raw = [2.0] + [1.0] * (n - 1)
    elif scheme == "top2x2":
        raw = [2.0, 2.0] + [1.0] * (n - 2)
    elif scheme == "tier":
        raw = [float(n - i) for i in range(n)]
    else:
        raise ValueError(f"未知分法：{scheme}")
    tot = sum(raw)
    return [x / tot for x in raw]


def _tie_groups(scores: pd.Series) -> list[list[str]]:
    """依分數由高到低分成同分群（群內依代號排序，只為了輸出穩定，不影響權重）。"""
    s = scores.dropna()
    groups = []
    for _, grp in s.groupby(s.values, sort=False):
        groups.append((grp.iloc[0], sorted(grp.index)))
    groups.sort(key=lambda t: -t[0])
    return [g for _, g in groups]


def elite_weights(scores: pd.Series, n: int, scheme: str) -> dict[str, float]:
    """取前 n 名並分配，回傳優等生權重（總和 1）。同分＝邊界平分：
    一個同分群佔到名次 [p, p+g)，它能分到的是這些名次中落在前 n 名的權重總和，
    由群內 g 檔平分（proportional 則是群內每檔拿 (落在前n名的名額數/g) × 自己的分數）。"""
    if n < 1:
        raise ValueError("n 至少為 1")
    groups = _tie_groups(scores)
    if not groups:
        return {}
    out: dict[str, float] = {}
    pos = 0
    if scheme == "proportional" and n > 1:
        frac: dict[str, float] = {}
        for g in groups:
            if pos >= n:
                break
            take = min(len(g), n - pos)
            for s in g:
                frac[s] = take / len(g)
            pos += len(g)
        raw = {s: f * float(scores[s]) for s, f in frac.items()}
        tot = sum(raw.values())
        return {s: v / tot for s, v in raw.items()} if tot > 0 else {}
    sw = slot_weights(n, scheme if n > 1 else "equal")
    for g in groups:
        if pos >= n:
            break
        covered = sw[pos:min(pos + len(g), n)]
        share = sum(covered) / len(g)
        for s in g:
            out[s] = share
        pos += len(g)
    tot = sum(out.values())
    return {s: v / tot for s, v in out.items()}


def mixed_elite_weights(score_hi: pd.Series, score_lo: pd.Series, n: int,
                        scheme: str) -> dict[str, float]:
    """S5 混合＝0.5×追強(score_hi 取前 n) + 0.5×押回彈(score_lo 取前 n)。"""
    a = elite_weights(score_hi, n, scheme)
    b = elite_weights(score_lo, n, scheme)
    out: dict[str, float] = {}
    for d in (a, b):
        for s, v in d.items():
            out[s] = out.get(s, 0.0) + 0.5 * v
    tot = sum(out.values())
    return {s: v / tot for s, v in out.items()} if tot > 0 else {}


# ───────────────────────── 3. 疊加 ─────────────────────────

def overlay(w0: dict[str, float], elite: dict[str, float], sleeve: float) -> dict[str, float]:
    """方案 b：`sleeve×elite + (1−sleeve)×w0`。w0 總和可能 <1（某策略當季解不出持股
    ＝視同現金，§8-R5），這裡保留同樣的總和，不偷偷把現金補成股票：優等生那一份
    按 w0 的總和縮放。elite 為空（例如沒有任何持股有分數）時退回 w0。"""
    if not elite or sleeve <= 0:
        return dict(w0)
    total = sum(w0.values())
    out = {s: (1.0 - sleeve) * v for s, v in w0.items()}
    for s, v in elite.items():
        out[s] = out.get(s, 0.0) + sleeve * total * v
    return out


def is_degenerate(n: int, scheme: str) -> bool:
    """N=1 時所有分法結果相同，網格只保留 equal 一格。"""
    return n == 1 and scheme != "equal"


def max_weight(w: dict[str, float]) -> float:
    tot = sum(w.values())
    return max(w.values()) / tot if w and tot > 0 else math.nan


# ───────────────────────── 2026-10-28 新增：三種訊號配對的兩種排名依據 ─────────────────────────
# 使用者裁示：步驟2 的 AI 要有三種「訊號→股票層級調整」可選，不只集中度→市值：
#   集中度警示→市值（S6，既有）／熱門股覆蓋不足→熱門股優等生（S7）／突破廣度→突破股優等生（S8）

def score_hot(mom_row: pd.Series, hot: list[str], syms: list[str]) -> pd.Series:
    """S7 熱門股：持股中屬於「全市場動能前 10%」（monitor.hot_segment）的股票，依動能排序；
    比例分配用的原始分數＝1+動能（恆正）。只用 as_of 以前的月底收盤價。"""
    hs = set(hot)
    keep = [s for s in syms if s in hs]
    m = mom_row.reindex(keep)
    return (1.0 + m[m.notna()]).astype(float)


def breakout_hits(close: pd.DataFrame, syms: list[str], as_of, window: int = 20) -> tuple[list[str], int]:
    """回傳 (創近 `window` 個交易日新高的股票, 有效檢查檔數)。S8 與突破廣度訊號共用這一支，避免定義分歧。
    新高＝as_of 收盤價 ≥ 期間最高價的 99.9%，且期間內價格有變動（最高 > 最低）。
    🔴 2026-10-28 code review：價格資料已 ffill，已下市／停牌股的價格整段不動、恆等於期間最高價，
    原本會被算成「創新高」（台股 2024-06-30 的 108 檔中有 13 檔是這種），因此排除價格完全不動者。"""
    idx = close.index
    i1 = idx.get_loc(idx[idx <= pd.Timestamp(as_of)].max())
    cols = [s for s in syms if s in close.columns]
    win = close.iloc[max(0, i1 - window + 1): i1 + 1][cols]
    last, hi, lo = win.iloc[-1], win.max(), win.min()
    ok = last.notna() & (hi > lo)
    hit = last[ok & (last >= 0.999 * hi)].index.tolist()
    return hit, int(ok.sum())


def score_breakout(close: pd.DataFrame, syms: list[str], as_of: pd.Timestamp, window: int = 20) -> pd.Series:
    """S8 突破股：持股中創近 `window` 個交易日新高者（`breakout_hits`），分數一律 1。
    全部同分 ⇒ 依 elite_weights 的邊界平分規則，等於「突破股之間平均分」（使用者 2026-10-28 確認）。"""
    hit, _ = breakout_hits(close, syms, as_of, window)
    return pd.Series(1.0, index=hit)
