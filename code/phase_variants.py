# -*- coding: utf-8 -*-
"""
實驗變體設定：同一套 SOP，用不同寬鬆度的因子池與門檻各跑一遍。

目的（2026-08-08 與使用者確認）：
  先把「嚴格版」這條線走完（Phase 1~4），再跑放寬版做對照。
  最有價值的問題是：**Phase 1 的線性檢定篩選，到底有沒有讓結果更好？**
  這是對老師方法論的直接檢驗——
    若最強的策略仍來自「過關」的因子 → 篩選有效，方法論站得住
    若最強的策略來自被淘汰的因子     → 篩選反而砍掉了好東西，要重新檢討

因子分組來源：Phase 1 台股結果（_analysis_outputs_phase1/Phase1_結果分析.md）
"""

# ==================== Phase 1 判定結果 ====================
# ⚠️ **必須依市場讀取**：台股與美股的 Phase 1 判定完全不同
#    （台股淘汰 7 個、只取極端桶 2 個；美股淘汰 1 個、只取極端桶 9 個）。
#    早期版本把台股結果寫死，直接拿去跑美股會用錯因子分類，故改為讀取實際輸出。
from pathlib import Path as _Path
import pandas as _pd

_P1DIR = _Path(__file__).resolve().parent.parent / "_analysis_outputs_phase1"
LOOKAHEAD = ["MOM1"]        # 🚫 前瞻偏誤，任何市場、任何變體都不用

_VERDICT_KEY = {"✅ 過關": "PASSED", "⚠️ 只取極端桶": "EDGE_ONLY",
                "⚠️ 邊際": "MARGINAL", "❌ 淘汰": "ELIMINATED"}
_cache = {}


def groups(market="TW", rsfx=""):
    """讀該市場 Phase 1 的判定，回傳 {PASSED/EDGE_ONLY/MARGINAL/ELIMINATED/ALL_USABLE}。
    順序沿用 CSV（判定優先、同判定內依 |ρ| 由大到小）——**順序會影響引擎產生的配對名稱**，
    故不可任意重排，否則既有白名單會對不上。

    rsfx：自訂日期範圍的檔名後綴（見 sweep_config.date_range_suffix）。空字串＝預設範圍。
    """
    key = (market, rsfx)
    if key in _cache:
        return _cache[key]
    p = _P1DIR / f"{market}_phase1{rsfx}_linearity.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"找不到 {p}；請先跑 phase1_linearity.py --market {market} 與 phase1_analyze.py"
            + (f"（自訂日期範圍需帶相同的 --start/--end）" if rsfx else ""))
    df = _pd.read_csv(p, encoding="utf-8-sig")
    g = {v: [] for v in _VERDICT_KEY.values()}
    for _, r in df.iterrows():
        k = _VERDICT_KEY.get(r["判定"])
        if k and r["因子"] not in LOOKAHEAD:
            g[k].append(r["因子"])
    g["ALL_USABLE"] = g["PASSED"] + g["EDGE_ONLY"] + g["MARGINAL"] + g["ELIMINATED"]
    _cache[key] = g
    return g

# ==================== 變體 ====================
def _spec(market, rsfx=""):
    g = groups(market, rsfx)
    P, E, M, A = g["PASSED"], g["EDGE_ONLY"], g["MARGINAL"], g["ALL_USABLE"]
    return {
        "strict": {
            "desc": "嚴格版：只有 Phase 1 過關的因子可當 primary",
            "primary": P, "secondary": M + E,
            "primary_margin": 0.02, "secondary_tol": 0.01,
        },
        "relaxed": {
            "desc": "放寬版：邊際與只取極端桶也可當 primary（淘汰的仍排除）",
            "primary": P + M + E, "secondary": P + M + E,
            "primary_margin": 0.02, "secondary_tol": 0.01,
        },
        "all": {
            "desc": "對照組：全部因子下去，等於不做 Phase 1 篩選（檢驗篩選有沒有用）",
            "primary": A, "secondary": A,
            "primary_margin": 0.02, "secondary_tol": 0.01,
        },
        "openSec": {
            "desc": "修正版：primary 維持 Phase 1 過關者，secondary 開放全部",
            "primary": P, "secondary": A,
            "primary_margin": 0.02, "secondary_tol": 0.01,
        },
    }


VARIANTS = {k: None for k in ("strict", "relaxed", "all", "openSec")}   # 供 argparse choices 用

# ==================== 為什麼有 openSec 這個變體 ====================
# 三方對照（見 _analysis_outputs_variants/）發現：
#   當 primary 用：Phase 1 篩選有效——7 個被淘汰的因子只有 REV_G 勉強擠進 primary 資格，
#                  且放寬後中位數/p90 都下降（16.16%→14.84%），證明拿不合格的當主訊號會稀釋品質
#   當 secondary 用：Phase 1 篩選會**誤殺**——全場最佳策略
#                  `MOM_qb2 × EV_EBITDA_qb0 + C4`（CAGR 38.05%、MDD 僅 −38.4%）
#                  用的正是被淘汰的 EV_EBITDA；7 個被淘汰的因子全都進了最終候選池
#
# 原因：Phase 1 檢定的是「這個因子**自己**能不能單調預測報酬」，
#       但 secondary 的功能不是自己預測，而是**提供不同構面的資訊**。
#       這與 Phase 2 的核心發現一致——單因子強度與配對價值是兩件事。
#
# 故 openSec = 用 Phase 1 篩 primary（有效），但不拿它篩 secondary（會誤殺）。

# 各變體的因子池若相同（含順序），Phase 2 的回測可共用同一份，不需重跑。
_POOL_SHARE = {
    "strict": "strict", "relaxed": "strict",   # 都是同樣 12 個因子
    "all": "all", "openSec": "all",            # 都是同樣 19 個因子
}


def l2_label(market, variant, rsfx=""):
    """該變體的 Phase 2 回測標籤（因子池相同者共用）。

    ⚠️ 共用前先斷言兩者的因子池**完全相同（含順序）**——順序會影響引擎產生的
    配對名稱，若不同卻共用，白名單會對不上而靜默少跑一堆組合。

    rsfx：自訂日期範圍的檔名後綴（見 sweep_config.date_range_suffix）。
    """
    owner = _POOL_SHARE[variant]
    if owner != variant:
        a, b = get(variant, market, rsfx)["factors"], get(owner, market, rsfx)["factors"]
        if a != b:
            raise AssertionError(
                f"[{market}] {variant} 與 {owner} 的因子池不同，不可共用 Phase 2 回測："
                f"\n  {variant}: {a}\n  {owner}: {b}")
    base = f"{market}_L2_M" if owner == "strict" else f"{market}_L2_{owner}_M"
    return base + rsfx

def assert_pool_unchanged(label, factors, art_dir=None):
    """比對既有 `_DONE` 裡記錄的因子池與現在要跑的是否相同，不同就拒絕沿用。

    為什麼需要（2026-08-12）：l2_label() 的標籤**不含因子數**，所以因子池從
    19 個變成 23 個時標籤不變，`is_done()` 會直接判定「已完成」而跳過，
    **靜默拿舊的 19 因子回測當新結果用**。這與本專案已發生過兩次的
    「靜默重用／讀錯來源」是同一類錯誤，故在此主動擋下。

    回傳 True＝可安全沿用；False＝目錄不存在或沒有可比的紀錄（照跑即可）。
    因子池不同則 raise。
    """
    import json
    from pathlib import Path as _P
    if art_dir is None:
        from fcv_core import ART_DIR as art_dir
    p = _P(art_dir) / label / "_DONE"
    if not p.exists():
        return False
    try:
        meta = json.loads(p.read_text(encoding="utf-8")).get("meta", {})
    except Exception:
        return False
    old = list(meta.get("primary", [])) + list(meta.get("secondary", []))
    seen, old_pool = set(), []
    for f in old:
        if f not in seen:
            old_pool.append(f)
            seen.add(f)
    if not old_pool:
        return False
    if old_pool != list(factors):
        raise AssertionError(
            f"[{label}] 既有回測的因子池與現在要跑的不同，**不可沿用**：\n"
            f"  既有（{len(old_pool)}）：{old_pool}\n"
            f"  現在（{len(factors)}）：{list(factors)}\n"
            f"→ 請先把 {art_dir}/{label} 封存或改名，再重跑。")
    return True


N_BUCKETS = 3


def get(name="strict", market="TW", rsfx=""):
    spec = _spec(market, rsfx)
    if name not in spec:
        raise ValueError(f"未知變體 {name}，可用：{list(spec)}")
    v = dict(spec[name])
    v["market"] = market
    # 因子池＝primary ∪ secondary，去重且保持順序（順序影響引擎產生的配對名稱）
    pool, seen = [], set()
    for f in list(v["primary"]) + list(v["secondary"]):
        if f not in seen:
            pool.append(f)
            seen.add(f)
    v["factors"] = pool
    v["name"] = name
    return v


def suffix(name):
    """label 後綴：strict 不加（維持既有 TW_L2_M 等名稱不變），其餘加 _{name}。"""
    return "" if name == "strict" else f"_{name}"


def scale(name, market="TW"):
    """估算該變體的規模。"""
    v = get(name, market)
    K, N = len(v["factors"]), N_BUCKETS
    cond = K * N
    singles = cond
    pairs = cond * (cond - N) // 2
    fcombos = singles + pairs
    return {"因子數": K, "條件數": cond, "單因子": singles, "配對": pairs,
            "F組合": fcombos, "Phase2策略數": fcombos,
            "若全過關的Phase3": fcombos * 21, "再加V": fcombos * 21 * 2}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    mkt = sys.argv[1] if len(sys.argv) > 1 else "TW"
    g = groups(mkt)
    print(f"[{mkt}] Phase 1 分組：過關{len(g['PASSED'])}／極端桶{len(g['EDGE_ONLY'])}"
          f"／邊際{len(g['MARGINAL'])}／淘汰{len(g['ELIMINATED'])}\n")
    print(f"{'變體':10s} {'因子':>4s} {'F組合':>6s} {'Phase2':>7s} "
          f"{'Phase3(全過)':>12s} {'+V':>9s}   說明")
    for n in VARIANTS:
        s = scale(n, mkt)
        print(f"{n:10s} {s['因子數']:4d} {s['F組合']:6d} {s['Phase2策略數']:7,d} "
              f"{s['若全過關的Phase3']:12,d} {s['再加V']:9,d}   {get(n, mkt)['desc']}")
    print("\n註：Phase3/+V 是「假如全部 F 組合都通過 Phase 2」的上限；")
    print("    實際會少很多（strict 版 630 個只有 203 個過關，32%）。")
