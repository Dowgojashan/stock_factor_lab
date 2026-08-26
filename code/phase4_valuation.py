# -*- coding: utf-8 -*-
"""
Phase 4：對 Phase 3 的策略加上 V（估值濾網），完成 F/C/V 四構面。

分階段原則（見 重跑計畫_老師方法論SOP.md §0.5）：這是最後一個構面。

⚡ 效率設計：**只跑 v1**。
  Phase 3 已經把同樣的 F×C 組合用 v0 跑完並落地了（TW_L3_M，4,263 個），
  v0 的結果直接沿用即可，不需重跑。本階段只補 v1 那一半（另外 4,263 個），
  分析時再把 L3(v0) 與 L4(v1) 依「F__C」配對比較——這正是論文 4-20~4-24
  在做的 V0/V1 對照。
  → 省下約 1.6 小時的重複回測。

V 的定義（fcv_core.MarketData.get_v_mask，沿用學姊論文）：
  v1 = PE 低於近 4 季均值、但高於近 4 季最低點
     ＝「相對自己歷史便宜、但不是最谷底」——買回檔，不接墜落的刀

用法：python phase4_valuation.py [--market TW] [--dry-run]
"""
import sys
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fcv_core import MarketData, run_spec, is_done, ART_DIR      # noqa: E402
from sweep_config import COMMON_FACTORS, build_p3, MARKET_START, date_range_suffix  # noqa: E402
from condition_factory import build_conditions                    # noqa: E402
from phase1_linearity import IN_SAMPLE_END                        # noqa: E402
from phase2_pairing import make_spec as make_p2_spec              # noqa: E402
from phase3_conditions import load_allowed_pairs                  # noqa: E402

V_MODES = ("v1",)        # 只補 v1；v0 沿用 Phase 3 的結果

CATALOG_DIR = Path("_catalog")
RUN_LOG = CATALOG_DIR / "phase4_run_log.txt"


def log(msg):
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser(description="Phase 4：加入 V 估值濾網")
    ap.add_argument("--market", default="TW", choices=["TW", "US"])
    ap.add_argument("--variant", default="strict")
    ap.add_argument("--batch-size", type=int, default=150)
    ap.add_argument("--start", default=None,
                    help="自訂起始日期 YYYY-MM-DD（需與同範圍的 phase3_conditions.py 結果搭配）")
    ap.add_argument("--end", default=None,
                    help="自訂結束日期 YYYY-MM-DD（需與同範圍的 phase3_conditions.py 結果搭配）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    market = args.market
    start = args.start or MARKET_START[market]
    end = args.end or IN_SAMPLE_END
    rsfx = date_range_suffix(start, end, MARKET_START[market], IN_SAMPLE_END)
    if rsfx:
        log(f"⚠️ 自訂日期範圍 {start}~{end}，輸出改用獨立標籤（後綴 {rsfx}），不會覆蓋既有正式結果")
    label = (f"{market}_L4_M{rsfx}" if args.variant == "strict"
             else f"{market}_L4_{args.variant}_M{rsfx}")

    allowed, _ = load_allowed_pairs(market, args.variant, rsfx)
    # ⚠️ 必須傳 variant：all 的因子池是 19 個，若沿用 strict 的 12 個，
    #    白名單裡含 REV_G 等因子的組合會在展開時**靜默跳過**（不報錯、策略數變少）。
    spec = make_p2_spec(market, args.variant, rsfx, start, end)
    spec["_meta"] = {
        "phase": "P4_valuation",
        "market": market,
        "start": start,
        "in_sample_end": end,
        "n_allowed_f_pairs": len(allowed),
        "note": "只跑 v1；v0 沿用 Phase 3 的 {market}_L3_M",
    }
    spec["P3"] = build_p3(COMMON_FACTORS)

    P3 = build_conditions(spec["P3"])
    n_expect = len(allowed) * (len(P3) + 1) * len(V_MODES)

    log(f"=== Phase 4 加入 V｜市場={market}｜期間 {start}~{end} ===")
    log(f"    F 白名單={len(allowed)}｜C 狀態={len(P3)+1}｜V={V_MODES}（v0 沿用 Phase 3）")
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
    log(f"=== Phase 4 結束｜耗時 {(time.time()-t0)/3600:.2f} 小時 ===")
    return rc


if __name__ == "__main__":
    sys.exit(main())
