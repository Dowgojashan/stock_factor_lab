# -*- coding: utf-8 -*-
"""
Phase 3：對 Phase 2 晉升的 F 組合加上 C（動態條件）。

老師的要求（2026-08-05 meeting）：
  「[F1×F2] OK 了以後，我就直接去檢查 C」——**F 定案後才碰 C**，
  避免同時挑 F 和 C 造成雙重選擇偏誤（老師說的「如果三個因子都作弊，應該會有不少快 40」）。

  兩個**互相獨立**的約束（老師特別強調「他的限制是不一樣的」）：
    「這個剩下的不可以太少…不然後面就沒有 LLM 的必要」  → 策略數
    「但是裡面被選出來的[股票]不能太少，那就是策略不[穩定]」→ 每策略持股數

分階段原則（見 重跑計畫_老師方法論SOP.md §0.5）：本階段只多開 C 一個構面，
**V 仍然關閉**（留給 Phase 4）。

⚠️ C 構面沿用原本的 20 個（`sweep_config.build_p3`，衍生自 ROE/EPS/FCF_P），
   **不預先刪減**。雖然 Phase 1 顯示 FCF_P 當「橫斷面分位因子」沒有訊號（ρ=+0.13）、
   且先前 10 批實驗中 C15~C20（FCF_P 衍生）一致墊底，但：
   (a) Phase 1 檢定的是「分位水準」，不等於「動態條件」也沒用
   (b) 使用者指示「沒有明顯拖累過頭的就全部先跑一跑」，不預先窄化
   實際表現由本階段的資料回答。

F 構面白名單：只跑 Phase 2 晉升的 203 個組合（12 單因子 + 191 配對），
用 fcv_core 新增的 allowed_f_pairs 參數精準展開，
**不是把 630 個全跑完再事後丟棄**（那會多跑 3 倍）。

用法（cwd 任意）：
  python phase3_conditions.py                 # 台股
  python phase3_conditions.py --dry-run
"""
import sys
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fcv_core import MarketData, run_spec, is_done, ART_DIR      # noqa: E402
from sweep_config import COMMON_FACTORS, build_p3, MARKET_START, date_range_suffix  # noqa: E402
from condition_factory import build_conditions                    # noqa: E402
from phase1_linearity import IN_SAMPLE_END                        # noqa: E402
from phase2_pairing import make_spec as make_p2_spec              # noqa: E402

V_MODES = ("v0",)        # Phase 3 仍關閉 V（留給 Phase 4）

CATALOG_DIR = Path("_catalog")
RUN_LOG = CATALOG_DIR / "phase3_run_log.txt"


def log(msg):
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_allowed_pairs(market: str, variant: str = "strict", rsfx: str = ""):
    """從 Phase 2 的體質檢查表取出 F 構面白名單（策略名的前兩段）。

    rsfx：自訂日期範圍的檔名後綴，須與該範圍的 phase2_analyze.py 輸出一致。
    """
    sfx = ("" if variant == "strict" else f"_{variant}") + rsfx
    p = HERE.parent / "_analysis_outputs_phase2" / f"{market}_L2{sfx}_體質檢查表.csv"
    if not p.exists():
        raise FileNotFoundError(f"找不到 Phase 2 體質檢查表：{p}\n請先執行 phase2_analyze.py")
    df = pd.read_csv(p, encoding="utf-8-sig")
    pairs = set()
    for s in df["strategy"]:
        parts = s.split("__")
        pairs.add(f"{parts[0]}__{parts[1]}")     # F1__F2（F2 為 None 時就是字串 "None"）
    return pairs, df


def main():
    ap = argparse.ArgumentParser(description="Phase 3：加入 C 動態條件")
    ap.add_argument("--market", default="TW", choices=["TW", "US"])
    ap.add_argument("--variant", default="strict")
    ap.add_argument("--batch-size", type=int, default=150)
    ap.add_argument("--start", default=None,
                    help="自訂起始日期 YYYY-MM-DD（需與同範圍的 phase2_pairing.py/analyze 結果搭配）")
    ap.add_argument("--end", default=None,
                    help="自訂結束日期 YYYY-MM-DD（需與同範圍的 phase2_pairing.py/analyze 結果搭配）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    market = args.market
    start = args.start or MARKET_START[market]
    end = args.end or IN_SAMPLE_END
    rsfx = date_range_suffix(start, end, MARKET_START[market], IN_SAMPLE_END)
    if rsfx:
        log(f"⚠️ 自訂日期範圍 {start}~{end}，輸出改用獨立標籤（後綴 {rsfx}），不會覆蓋既有正式結果")
    label = (f"{market}_L3_M{rsfx}" if args.variant == "strict"
             else f"{market}_L3_{args.variant}_M{rsfx}")

    allowed, l2 = load_allowed_pairs(market, args.variant, rsfx)
    # P1 沿用 Phase 2 完全相同的定義（同因子池、同 3 桶、同順序）
    # → 引擎產生的 F 組合名稱才會與白名單完全對得起來
    # ⚠️ 必須傳 variant：all 的因子池是 19 個，若沿用 strict 的 12 個，
    #    白名單裡含 REV_G 等因子的組合會在展開時**靜默跳過**（不報錯、策略數變少）。
    spec = make_p2_spec(market, args.variant, rsfx, start, end)
    spec["_meta"] = {
        "phase": "P3_conditions",
        "market": market,
        "start": start,
        "in_sample_end": end,
        "n_allowed_f_pairs": len(allowed),
        "note": "F 白名單來自 Phase 2 體質檢查表；V 仍關閉",
    }
    spec["P3"] = build_p3(COMMON_FACTORS)      # 20 個 C（ROE/EPS/FCF_P 衍生）

    P1 = build_conditions(spec["P1"])
    P3 = build_conditions(spec["P3"])
    n_expect = len(allowed) * (len(P3) + 1) * len(V_MODES)

    n_single = sum(1 for a in allowed if a.endswith("__None"))
    log(f"=== Phase 3 加入 C｜市場={market}｜期間 {start}~{end} ===")
    log(f"    F 白名單（Phase 2 晉升）={len(allowed)}"
        f"（單因子 {n_single} / 配對 {len(allowed)-n_single}）")
    log(f"    P1 條件={len(P1)}｜C={len(P3)}（+None={len(P3)+1} 種 C 狀態）｜V={V_MODES}")
    log(f"    預期共 {n_expect} 策略")
    log(f"    {'done' if is_done(label) else 'pending'}  {label}")

    if args.dry_run:
        log("dry-run：不執行。")
        return 0
    if is_done(label):
        log("已完成，無待跑。")
        return 0

    log(f"[{market}] 載入資料…")
    md = MarketData(market, start=start, end=end)

    t0 = time.time()
    try:
        meta = run_spec(md, spec, label, rebalance="M", batch_size=args.batch_size,
                        dedup=True, v_modes=V_MODES, allowed_f_pairs=allowed)
        log(f"✅ {label}｜展開 {meta['n_expanded']}｜回測 {meta['n_backtested']}｜{meta['seconds']}s")
        # 寫 manifest：記錄這批資料是用「哪個變體 + 哪份白名單」跑的。
        # （2026-08-09 補：先前 strict 線是用舊基準的 203 白名單跑、relaxed 線是新基準的 187，
        #   兩個目錄意義不對稱且光看名稱看不出來，故往後一律留下可追溯的紀錄。）
        import json as _json
        (Path(ART_DIR) / label / "_MANIFEST.json").write_text(_json.dumps({
            "label": label, "market": market, "variant": args.variant,
            "n_allowed_f_pairs": len(allowed), "start": start, "in_sample_end": end,
            "v_modes": list(V_MODES), "n_backtested": meta["n_backtested"],
            "whitelist_from": f"_analysis_outputs_phase2/{market}_L2"
                              f"{'' if args.variant == 'strict' else '_' + args.variant}{rsfx}_體質檢查表.csv",
            "run_at": datetime.now().isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        rc = 0
    except Exception as e:
        d = Path(ART_DIR) / label
        d.mkdir(parents=True, exist_ok=True)
        (d / "_FAILED").write_text(
            f"{datetime.now().isoformat()}\n{e}\n\n{traceback.format_exc()}", encoding="utf-8")
        log(f"❌ {label} 失敗：{e}")
        rc = 1

    md.release()
    log(f"=== Phase 3 結束｜耗時 {(time.time()-t0)/3600:.2f} 小時 ===")
    return rc


if __name__ == "__main__":
    sys.exit(main())
