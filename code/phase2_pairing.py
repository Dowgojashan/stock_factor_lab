# -*- coding: utf-8 -*-
"""
Phase 2：3 桶 F1×F2 配對（依 2026-08-05 指導教授 meeting 的方法論）。

老師的要求：
  「你一個把它定成 primary，一個是 secondary。primary 一定要是…有意義的，
   secondary 就是要符合某些[寬鬆條件]…至少跟大盤差不多，或者說是輸大盤不多」
  「primary 本身就已經很強了，或 primary 加上某一點點比較寬鬆的 secondary 夠強了，
   這時候我就可以把它晉升」
  「F2 可以是空集合」
  「9 拼回 3…因為這樣子樣本才夠大」

分階段原則（見 重跑計畫_老師方法論SOP.md §0.5）：本階段只多開 F2 一個構面，
**C 與 V 仍然關閉**。

因子池只收 Phase 1 沒被淘汰的（見 phase1_analyze 產出的判定）：
  primary   = Phase 1「✅ 過關」的 5 個
  secondary = Phase 1「⚠️ 邊際」5 個 +「⚠️ 只取極端桶」2 個
  淘汰的 7 個與前瞻偏誤的 MOM1 完全不進來。

⚠️ 角色不對稱的實作方式：
  F1∩F2 與 F2∩F1 的遮罩完全相同、回測結果也相同——**不對稱的是「評選標準」，
  不是「策略本身」**。所以這裡把 12 個因子全放進同一個 P1 讓引擎配對
  （不需改引擎、也不重複回測），角色與門檻留到 phase2_analyze.py 才套用。

⚠️ 與老師的一個技術分歧：老師認為 9 桶湊回 3 桶「不用重算，算數湊回來而已」。
  這對統計量成立、對回測不成立——合併後選股清單變了（三個各 11% 的獨立投組
  變成一個 33% 的投組），持股數/權重/成本全都不同，**必須實跑**。此處照實跑。

用法（cwd 任意）：
  python phase2_pairing.py                 # 台股
  python phase2_pairing.py --market US
  python phase2_pairing.py --dry-run
"""
import sys
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fcv_core import MarketData, run_spec, is_done, ART_DIR   # noqa: E402
from sweep_config import MARKET_START                          # noqa: E402
from condition_factory import build_conditions                 # noqa: E402
from phase1_linearity import IN_SAMPLE_END                      # noqa: E402

# ==================== 設定 ====================
N_BUCKETS = 3            # 老師指定：9 桶檢定完，湊回 3 桶跑策略（樣本才夠大）
V_MODES = ("v0",)        # Phase 2 仍關閉 V

# 因子池由 phase_variants 提供（strict / relaxed / all）。
# ⚠️ **因子順序會影響引擎產生的配對名稱**（dedup 只取無序對 j>i），
#    故 phase_variants.get("strict")["factors"] 的順序必須與本檔原本的
#    PRIMARY+SECONDARY 完全一致，否則既有白名單會對不上。已驗證相同。
import phase_variants  # noqa: E402

_V = phase_variants.get("strict", "TW")
PRIMARY, SECONDARY, FACTORS = _V["primary"], _V["secondary"], _V["factors"]

CATALOG_DIR = Path("_catalog")
RUN_LOG = CATALOG_DIR / "phase2_run_log.txt"


def log(msg):
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def make_spec(market: str, variant: str = "strict") -> dict:
    """該變體的因子各切 3 桶放進同一個 P1；引擎自動產生單因子 + 跨因子無序對。

    variant 預設 "strict" ＝加入本參數前的行為完全相同（12 個因子、同順序）。
    """
    V = phase_variants.get(variant, market)
    return {
        "_meta": {
            "phase": "P2_pairing",
            "market": market,
            "variant": variant,
            "n_buckets": N_BUCKETS,
            "primary": V["primary"],
            "secondary": V["secondary"],
            "in_sample_end": IN_SAMPLE_END,
            "note": "C/V 皆關閉；角色不對稱在分析層套用",
        },
        "P1": {
            f: {
                "field": f"report:{f.lower()}",
                "conditions": [{"type": "q_band", "args": [k, N_BUCKETS]}
                               for k in range(N_BUCKETS)],
            }
            for f in V["factors"]
        },
        "P3": {},          # 空 → 只有 C=None
    }


def main():
    ap = argparse.ArgumentParser(description="Phase 2：3桶 F1×F2 配對")
    ap.add_argument("--market", default="TW", choices=["TW", "US"])
    ap.add_argument("--variant", default="strict", choices=list(phase_variants.VARIANTS))
    ap.add_argument("--batch-size", type=int, default=150)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    V = phase_variants.get(args.variant, args.market)
    PRIMARY, SECONDARY, FACTORS = V["primary"], V["secondary"], V["factors"]
    # 因子池相同的變體共用同一份回測（openSec 與 all 同池 → 不重複跑）
    label = phase_variants.l2_label(args.market, args.variant)
    spec = make_spec(args.market, args.variant)
    P1 = build_conditions(spec["P1"])

    n_cond = len(P1)
    n_single = n_cond
    n_pair = (n_cond * (n_cond - N_BUCKETS)) // 2      # 跨因子無序對（同因子的桶不配對）
    n_expect = (n_single + n_pair) * len(V_MODES)

    log(f"=== Phase 2 配對｜市場={args.market}｜因子={len(FACTORS)}"
        f"（primary {len(PRIMARY)} / secondary {len(SECONDARY)}）｜"
        f"分桶 N={N_BUCKETS}｜V={V_MODES}（C 關閉）｜in-sample 至 {IN_SAMPLE_END} ===")
    log(f"    primary  ：{PRIMARY}")
    log(f"    secondary：{SECONDARY}")
    log(f"    P1 條件數={n_cond}｜單因子={n_single}｜跨因子配對={n_pair}｜預期共 {n_expect} 策略")
    log(f"    {'done' if is_done(label) else 'pending'}  {label}")

    if args.dry_run:
        log("dry-run：不執行。")
        return 0
    if is_done(label):
        log("已完成，無待跑。")
        return 0

    log(f"[{args.market}] 載入資料…")
    md = MarketData(args.market, start=MARKET_START[args.market], end=IN_SAMPLE_END)

    t0 = time.time()
    try:
        meta = run_spec(md, spec, label, rebalance="M", batch_size=args.batch_size,
                        dedup=True, v_modes=V_MODES)
        log(f"✅ {label}｜展開 {meta['n_expanded']}｜回測 {meta['n_backtested']}｜{meta['seconds']}s")
        rc = 0
    except Exception as e:
        d = Path(ART_DIR) / label
        d.mkdir(parents=True, exist_ok=True)
        (d / "_FAILED").write_text(
            f"{datetime.now().isoformat()}\n{e}\n\n{traceback.format_exc()}", encoding="utf-8")
        log(f"❌ {label} 失敗：{e}")
        rc = 1

    md.release()
    log(f"=== Phase 2 結束｜耗時 {(time.time()-t0)/60:.1f} 分鐘 ===")
    return rc


if __name__ == "__main__":
    sys.exit(main())
