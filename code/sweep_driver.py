# -*- coding: utf-8 -*-
"""
Sweep Driver：掃描實驗矩陣、把結果累積成策略庫（模式 A：全算完）。

設計（見 A4 §4）：
- **載入共享**：同市場的所有 job 共用一次 `MarketData`（載入 ~5 分鐘是最貴的一步）。
- **`_DONE` 續傳**：每個 job 跑完寫 `_DONE`；重跑自動跳過 → 隨時可停可續。
- **崩潰隔離**：單一 job 出錯只寫 `_FAILED` + 記 log，不影響其他 job。
- **可觀測**：run_log.txt（時間戳/耗時/策略數）+ jobs_manifest.json（狀態總表）。

搭配 `sweep_supervisor.py` 可在 worker 整個掛掉（OOM 等）時自動重啟、靠 `_DONE` 接續。

用法（cwd 任意，模組會自行 chdir 到 code/）：
  python sweep_driver.py                          # 全量（US+TW × M,Q）
  python sweep_driver.py --dry-run                # 只列工作清單與狀態，不跑
  python sweep_driver.py --markets US             # 只跑美股
  python sweep_driver.py --smoke                  # 冒煙：小宇宙+少量策略，label 加 _SMOKE
  python sweep_driver.py --batch-size 100
"""
import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fcv_core import MarketData, run_spec, is_done, ART_DIR      # noqa: E402
from sweep_config import job_list, MARKET_START                   # noqa: E402

CATALOG_DIR = Path("_catalog")
RUN_LOG = CATALOG_DIR / "run_log.txt"
MANIFEST = CATALOG_DIR / "jobs_manifest.json"


def log(msg):
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def failed_marker(label, art_dir=ART_DIR):
    return Path(art_dir) / label / "_FAILED"


def write_manifest(jobs, art_dir=ART_DIR):
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for j in jobs:
        lb = j["label"]
        if is_done(lb, art_dir):
            status = "done"
        elif failed_marker(lb, art_dir).exists():
            status = "failed"
        else:
            status = "pending"
        rows.append({"label": lb, "market": j["market"],
                     "rebalance": j["rebalance"], "status": status})
    MANIFEST.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def main():
    ap = argparse.ArgumentParser(description="策略庫 Sweep Driver（模式 A）")
    ap.add_argument("--markets", default="US,TW", help="逗號分隔，如 US 或 US,TW")
    ap.add_argument("--rebalances", default="M,Q", help="逗號分隔，如 M 或 M,Q")
    ap.add_argument("--batch-size", type=int, default=150)
    ap.add_argument("--no-dedup", action="store_true", help="關閉 F1×F2 對稱去重（不建議）")
    ap.add_argument("--dry-run", action="store_true", help="只列清單與狀態，不跑")
    ap.add_argument("--smoke", action="store_true", help="冒煙模式（小宇宙、少量策略、label 加 _SMOKE）")
    ap.add_argument("--smoke-univ", type=int, default=60)
    ap.add_argument("--smoke-limit", type=int, default=30)
    args = ap.parse_args()

    markets = [m.strip().upper() for m in args.markets.split(",") if m.strip()]
    rebs = [r.strip().upper() for r in args.rebalances.split(",") if r.strip()]
    jobs = job_list(markets=markets, rebalances=rebs)
    if args.smoke:
        for j in jobs:
            j["label"] += "_SMOKE"

    rows = write_manifest(jobs)
    log(f"=== Sweep 啟動｜jobs={len(jobs)}｜markets={markets}｜rebalances={rebs}"
        f"｜smoke={args.smoke} ===")
    for r in rows:
        log(f"    {r['status']:8s} {r['label']}")
    if args.dry_run:
        log("dry-run：不執行。")
        return 0

    t0 = time.time()
    n_ok = n_skip = n_fail = 0

    for market in markets:
        group = [j for j in jobs if j["market"] == market]
        pending = [j for j in group if not is_done(j["label"])]
        n_skip += len(group) - len(pending)
        if not pending:
            log(f"[{market}] 全部已完成，略過載入。")
            continue

        # ★ 載入共享：一個市場只載一次，跑完它所有 job
        log(f"[{market}] 載入資料（{len(pending)}/{len(group)} 個 job 待跑）…")
        try:
            md = MarketData(market,
                            start=MARKET_START[market],
                            univ_limit=args.smoke_univ if args.smoke else 0)
        except Exception as e:
            log(f"[{market}] ❌ 載入失敗，跳過此市場：{e}")
            traceback.print_exc()
            n_fail += len(pending)
            continue

        for j in pending:
            lb = j["label"]
            try:
                meta = run_spec(md, j["spec"], lb,
                                rebalance=j["rebalance"],
                                batch_size=args.batch_size,
                                smoke_limit=args.smoke_limit if args.smoke else None,
                                dedup=not args.no_dedup)
                n_ok += 1
                log(f"✅ {lb}｜回測 {meta['n_backtested']}｜{meta['seconds']}s")
                failed_marker(lb).unlink(missing_ok=True)
            except Exception as e:
                n_fail += 1
                d = Path(ART_DIR) / lb
                d.mkdir(parents=True, exist_ok=True)
                failed_marker(lb).write_text(
                    f"{datetime.now().isoformat()}\n{e}\n\n{traceback.format_exc()}",
                    encoding="utf-8")
                log(f"❌ {lb} 失敗（已隔離，繼續下一個）：{e}")
            write_manifest(jobs)

        md.release()
        del md
        log(f"[{market}] 完成，資料已釋放。")

    write_manifest(jobs)
    log(f"=== Sweep 結束｜成功 {n_ok}｜略過(已完成) {n_skip}｜失敗 {n_fail}"
        f"｜總耗時 {(time.time()-t0)/3600:.2f}h ===")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
