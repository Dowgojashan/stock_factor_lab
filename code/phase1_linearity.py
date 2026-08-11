# -*- coding: utf-8 -*-
"""
Phase 1：9桶單因子線性檢定（依 2026-08-05 指導教授 meeting 的方法論）。

老師的要求：
  「第一個通常會先判斷**有沒有線性**…有線性的才拿出來，**沒有線性表示這個因子不行**」
  「我大概會先把它切割成 8、9 個區間吧，然後再湊回來。因為 3 的倍數嘛」

作法（不改回測引擎）：
  利用 fcv_core.stream_strategy_masks 既有邏輯「同因子的桶不互相配對」
  （`if p2["field"] == p1["field"]: continue`）——每個因子跑一份只含自己 9 個 q_band
  的 spec，且 P3 留空 → 自動只產生「純單因子、無 C」的策略。

  label : {market}_L1_{因子}_M
  P1    : {因子: [q_band(k,9) for k in 0..8]}
  P3    : {}                      → 只有 C=None
  產出  : 9桶 × 1(None) × 2(v0/v1) = 18 策略/因子

⚠️ in-sample = 2000-01-01 ~ 2025-12-31（2026 保留為樣本外），
   透過 MarketData(end=IN_SAMPLE_END) 強制切斷。

因子池 16 個（DB 實有 17 個，扣掉老師已排除的 PE）。
PE 不進 F 構面，僅保留作為 V 估值濾網的依據（fcv_core.get_v_mask）。

用法（cwd 任意）：
  python phase1_linearity.py                    # 台股 16 因子
  python phase1_linearity.py --market US
  python phase1_linearity.py --dry-run
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

# ==================== 設定 ====================
IN_SAMPLE_END = "2025-12-31"     # ⚠️ 2026 保留為樣本外
N_BUCKETS = 9                    # 老師指定：先切 9 桶（3 的倍數，之後可湊回 3）
V_MODES = ("v0",)                # Phase 1 連 V 也關掉：純單因子，一次只開一個構面

# 🚫 已知有前瞻偏誤、不列入評選的因子（仍會照跑、照畫圖，供報告說明用）
#   MOM1 = (公告日+45天收盤 − 期末日收盤)/期末日收盤，但 lab 的
#   format_data.py::adjust_index_of_report 讓財報在「公告日當天」就可用
#   → 選股時已含未來 45 天的股價，是偷看答案，不是預測能力。
#   要真正救回需回 collector 改定義（改成往「前」推的動能）。
LOOKAHEAD_FLAGGED = {"MOM1"}

# DB 實有 21 個因子（TW/US 相同），扣掉 PE（老師 2026-08-05 排除，只留給 V 用）→ 20 個。
# 2026-08-06 collector 補齊：台股 MOM/MOM1 修復（原 99.93% NULL），
# 並新增 ACCRUAL/REV_G/VOL/NETDEBT_EBITDA 補齊「應計品質/成長/波動」三個空白類別。
FACTORS = [
    "ROE", "EPS", "FCF_P", "DEBTRATIO", "REVENUE",      # 體質/規模
    "EV_EBITDA", "EV_S", "PB", "PS", "P_IC",            # 估值倍數
    "ROIC", "CROIC", "FCF_OI", "OCF_E",                 # 資本報酬/現金流品質
    "MOM", "MOM1",                                       # 動量
    "ACCRUAL",                                           # 應計品質
    "REV_G",                                             # 成長
    "VOL",                                               # 波動
    "NETDEBT_EBITDA",                                    # 財務結構（償債能力）
]

CATALOG_DIR = Path("_catalog")
RUN_LOG = CATALOG_DIR / "phase1_run_log.txt"


def log(msg):
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def make_spec(factor: str, market: str) -> dict:
    """單因子 9 桶、無 C。P1 只放這一個因子 → 引擎的同因子跳過邏輯保證不會配出 F2。"""
    return {
        "_meta": {
            "phase": "P1_linearity",
            "market": market,
            "factor": factor,
            "n_buckets": N_BUCKETS,
            "in_sample_end": IN_SAMPLE_END,
        },
        "P1": {
            factor: {
                "field": f"report:{factor.lower()}",
                "conditions": [{"type": "q_band", "args": [k, N_BUCKETS]}
                               for k in range(N_BUCKETS)],
            }
        },
        "P3": {},          # 空 → 只有 C=None
    }


def job_list(factors, market):
    return [{"label": f"{market}_L1_{f}_M", "factor": f, "rebalance": "M",
             "spec": make_spec(f, market)} for f in factors]


def main():
    ap = argparse.ArgumentParser(description="Phase 1：9桶單因子線性檢定")
    ap.add_argument("--market", default="TW", choices=["TW", "US"])
    ap.add_argument("--factors", default=",".join(FACTORS))
    ap.add_argument("--batch-size", type=int, default=150)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    factors = [f.strip().upper() for f in args.factors.split(",") if f.strip()]
    if "PE" in factors:
        log("⚠️ PE 已由老師排除，自動移除（PE 僅保留為 V 構面估值濾網依據）")
        factors = [f for f in factors if f != "PE"]

    jobs = job_list(factors, args.market)
    log(f"=== Phase 1 線性檢定｜市場={args.market}｜因子數={len(jobs)}｜"
        f"分桶 N={N_BUCKETS}｜V={V_MODES}（F2/C 皆關閉）｜in-sample 至 {IN_SAMPLE_END} ===")
    total = 0
    for j in jobs:
        P1 = build_conditions(j["spec"]["P1"])
        P3 = build_conditions(j["spec"]["P3"])
        n_expect = len(P1) * (len(P3) + 1) * len(V_MODES)   # ×(C含None) ×V
        total += n_expect
        status = "done" if is_done(j["label"]) else "pending"
        log(f"    {status:8s} {j['label']:22s} P1={len(P1):2d} P3={len(P3):2d} → 預期 {n_expect} 策略")
    log(f"    合計預期 {total} 個策略")

    if args.dry_run:
        log("dry-run：不執行。")
        return 0

    pending = [j for j in jobs if not is_done(j["label"])]
    if not pending:
        log("全部已完成。")
        return 0

    log(f"[{args.market}] 載入資料（{len(pending)}/{len(jobs)} 個因子待跑）…")
    md = MarketData(args.market, start=MARKET_START[args.market], end=IN_SAMPLE_END)

    t0 = time.time()
    n_ok = n_fail = 0
    for j in pending:
        lb = j["label"]
        try:
            meta = run_spec(md, j["spec"], lb, rebalance=j["rebalance"],
                            batch_size=args.batch_size, dedup=True, v_modes=V_MODES)
            n_ok += 1
            log(f"✅ {lb}｜回測 {meta['n_backtested']}｜{meta['seconds']}s")
        except Exception as e:
            n_fail += 1
            d = Path(ART_DIR) / lb
            d.mkdir(parents=True, exist_ok=True)
            (d / "_FAILED").write_text(
                f"{datetime.now().isoformat()}\n{e}\n\n{traceback.format_exc()}", encoding="utf-8")
            log(f"❌ {lb} 失敗（已隔離，繼續下一個）：{e}")

    md.release()
    log(f"=== Phase 1 結束｜成功 {n_ok}｜失敗 {n_fail}｜耗時 {(time.time()-t0)/60:.1f} 分鐘 ===")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
