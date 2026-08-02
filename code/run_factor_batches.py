# -*- coding: utf-8 -*-
"""
因子候選批次 driver：對照學姊論文 3.6.1 節「設計階段測試 30 餘項候選因子」的方法論，
把這個探索步驟自動化、正式產出——每個候選因子各跑一批（F 池 = 固定 ROE/EPS/FCF_P +
1 個候選），規模比照論文（~2,000~7,000 策略/批，見計畫文件規模驗證），
每批各自存一份 stats/明細供 analyze_batch.py 出圖。

執行引擎沿用 fcv_core.py 的 MarketData/run_spec（= 論文 batch_backtest.py 的既有實作），
不逐一重跑 fcv_backtest.ipynb。11 批共用一次 MarketData(market) 載入。market 可為 TW 或
US（US 的 russell3000 圈定已在 database.py 完成，這裡不需另外處理宇宙）。

C 構面沿用 sweep_config.py 的 20-C 定義（C_TYPES/C_SKIP/build_p3），只綁定在固定的
ROE/EPS/FCF_P 三因子上，候選因子只出現在 F 構面。

用法（cwd 任意，模組會自行 chdir 到 code/）：
  python run_factor_batches.py                          # TW 全部 11 批
  python run_factor_batches.py --market US               # US 全部 11 批
  python run_factor_batches.py --market US --candidates PE   # 只跑一批（測試用）
  python run_factor_batches.py --dry-run                 # 只列清單與策略數估算，不跑
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
from sweep_config import COMMON_FACTORS, build_p1, build_p3, MARKET_START  # noqa: E402
from condition_factory import build_conditions                    # noqa: E402

# PE 已排除（2026-07-29 與老師確認）：PE 同時是候選因子(F)又是 V 構面估值濾網
# （fcv_core.py::get_v_mask 預設抓 report:pe）的依據，兩邊都用PE重複探索同一訊號，故拔除。
CANDIDATES = ["EV_EBITDA", "EV_S", "CROIC", "FCF_OI", "ROIC",
              "PB", "PS", "P_IC", "OCF_E", "MOM"]

CATALOG_DIR = Path("_catalog")
RUN_LOG = CATALOG_DIR / "batch_run_log.txt"


def log(msg):
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def make_batch_spec(candidate: str, market: str) -> dict:
    """F 池 = ROE/EPS/FCF_P + candidate（N=5 q_band）；C = sweep_config 的 20-C，只綁 ROE/EPS/FCF_P。"""
    factors_p1 = dict(COMMON_FACTORS)
    factors_p1[candidate] = f"report:{candidate.lower()}"
    P1 = build_p1(factors_p1, n_list=[5])
    P3 = build_p3(COMMON_FACTORS)  # 不含candidate，比照論文C只衍生自體質三因子
    return {
        "_meta": {"market": market, "candidate": candidate, "factors": list(factors_p1)},
        "P1": P1,
        "P3": P3,
    }


def batch_list(candidates, market):
    jobs = []
    for c in candidates:
        spec = make_batch_spec(c, market)
        jobs.append({"label": f"{market}_batch_{c}_M", "candidate": c, "rebalance": "M", "spec": spec})
    return jobs


def main():
    ap = argparse.ArgumentParser(description="因子候選批次 driver")
    ap.add_argument("--market", default="TW", choices=["TW", "US"], help="市場，預設TW")
    ap.add_argument("--candidates", default=",".join(CANDIDATES),
                    help="逗號分隔候選因子清單，預設全部11個")
    ap.add_argument("--batch-size", type=int, default=150)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="冒煙模式（小宇宙、少量策略，label加_SMOKE）")
    ap.add_argument("--smoke-univ", type=int, default=60)
    ap.add_argument("--smoke-limit", type=int, default=30)
    args = ap.parse_args()

    candidates = [c.strip().upper() for c in args.candidates.split(",") if c.strip()]
    jobs = batch_list(candidates, args.market)
    if args.smoke:
        for j in jobs:
            j["label"] += "_SMOKE"

    log(f"=== 因子候選批次啟動｜市場={args.market}｜批次數={len(jobs)}｜候選={candidates} ===")
    for j in jobs:
        P1 = build_conditions(j["spec"]["P1"])
        P3 = build_conditions(j["spec"]["P3"])
        status = "done" if is_done(j["label"]) else "pending"
        log(f"    {status:8s} {j['label']:24s} P1={len(P1):3d} P3={len(P3):3d}")

    if args.dry_run:
        log("dry-run：不執行。")
        return 0

    pending = [j for j in jobs if not is_done(j["label"])]
    if not pending:
        log("全部已完成，無待跑批次。")
        return 0

    log(f"[{args.market}] 載入資料（{len(pending)}/{len(jobs)} 批待跑）…")
    md = MarketData(args.market, start=MARKET_START[args.market],
                    univ_limit=args.smoke_univ if args.smoke else 0)

    t0 = time.time()
    n_ok = n_fail = 0
    for j in pending:
        lb = j["label"]
        try:
            meta = run_spec(md, j["spec"], lb, rebalance=j["rebalance"],
                            batch_size=args.batch_size, dedup=True,
                            smoke_limit=args.smoke_limit if args.smoke else None)
            n_ok += 1
            log(f"✅ {lb}｜回測 {meta['n_backtested']}｜{meta['seconds']}s")
        except Exception as e:
            n_fail += 1
            d = Path(ART_DIR) / lb
            d.mkdir(parents=True, exist_ok=True)
            (d / "_FAILED").write_text(
                f"{datetime.now().isoformat()}\n{e}\n\n{traceback.format_exc()}", encoding="utf-8")
            log(f"❌ {lb} 失敗（已隔離，繼續下一批）：{e}")

    md.release()
    log(f"=== 批次結束｜成功 {n_ok}｜失敗 {n_fail}｜總耗時 {(time.time()-t0)/3600:.2f}h ===")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
