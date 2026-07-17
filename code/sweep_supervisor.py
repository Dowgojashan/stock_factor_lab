# -*- coding: utf-8 -*-
"""
Sweep Supervisor：讓策略庫可以「連續跑好幾天、無人值守」。

為什麼需要：worker 可能因 OOM／記憶體碎片／偶發例外**整個掛掉**。
supervisor 偵測到非正常結束就重啟 → 重載一次資料 → 靠 `_DONE` 標記跳過已完成的 job 接著跑。
**連記憶體洩漏都擋不住它跑完。**

防無限迴圈：以「本輪有沒有新增 `_DONE`」判斷是否有進展；連續數輪無進展就停止並報告，
不會在一個壞掉的 job 上永遠重試。

用法：
  python sweep_supervisor.py                       # 全量，掛了自動重啟
  python sweep_supervisor.py -- --markets US       # `--` 之後的參數原樣轉給 driver
  python sweep_supervisor.py --max-restarts 10
"""
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fcv_core import ART_DIR          # noqa: E402

LOG = Path("_catalog") / "supervisor_log.txt"


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [supervisor] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def count_done():
    p = Path(ART_DIR)
    return len(list(p.glob("*/_DONE"))) if p.exists() else 0


def main():
    ap = argparse.ArgumentParser(description="Sweep Supervisor（自動重啟）")
    ap.add_argument("--max-restarts", type=int, default=20)
    ap.add_argument("--max-no-progress", type=int, default=3,
                    help="連續幾輪沒有新增 _DONE 就放棄（防無限重試）")
    ap.add_argument("--backoff", type=int, default=20, help="重啟前等待秒數")
    ap.add_argument("driver_args", nargs=argparse.REMAINDER,
                    help="`--` 之後的參數原樣轉給 sweep_driver.py")
    args = ap.parse_args()

    extra = [a for a in args.driver_args if a != "--"]
    cmd = [sys.executable, "-u", str(HERE / "sweep_driver.py")] + extra
    log(f"啟動：{' '.join(cmd)}")

    restarts = no_progress = 0
    while True:
        before = count_done()
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(HERE))
        after = count_done()
        dt = time.time() - t0
        log(f"driver 結束 rc={rc}｜本輪耗時 {dt/60:.1f} 分｜_DONE {before} → {after}")

        if rc == 0:
            log("✅ driver 正常結束（全部完成、無失敗）。supervisor 收工。")
            return 0

        # 有進展就把 no_progress 歸零；沒進展則累計
        if after > before:
            no_progress = 0
        else:
            no_progress += 1
            log(f"⚠️ 本輪無進展（{no_progress}/{args.max_no_progress}）")
            if no_progress >= args.max_no_progress:
                log("❌ 連續無進展，停止重試。請看 _catalog/run_log.txt 與各 job 的 _FAILED。")
                return 1

        restarts += 1
        if restarts > args.max_restarts:
            log(f"❌ 已達重啟上限 {args.max_restarts}，停止。")
            return 1
        log(f"🔁 {args.backoff}s 後重啟（第 {restarts} 次）…")
        time.sleep(args.backoff)


if __name__ == "__main__":
    sys.exit(main())
