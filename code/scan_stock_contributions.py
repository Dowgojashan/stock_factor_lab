# -*- coding: utf-8 -*-
"""
全域股票貢獻掃描：對11批候選因子的全部策略（~74,518個）逐一讀 trades.parquet，
統計每檔股票「在幾成策略裡是Top1貢獻股」（頻率型指標，較不受策略互相重疊影響），
同時保留原始累加值供對照（會受策略重疊/重複計入影響，見對話中的說明）。

不改動任何既有程式碼；只讀 results_artifacts/ 下已完成批次的 trades.parquet。
可續跑：每批完成後落地一次中間結果，中斷後重跑會跳過已完成的批次。
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_batch import stock_cum_contrib  # noqa: E402

CANDIDATES = ["PE", "EV_EBITDA", "EV_S", "CROIC", "FCF_OI", "ROIC",
              "PB", "PS", "P_IC", "OCF_E", "MOM"]
ART_DIR = HERE / "results_artifacts"
OUT_DIR = HERE.parent / "_analysis_outputs"
CKPT_DIR = OUT_DIR / "_stock_scan_checkpoints"


def scan_batch(label: str, log) -> dict:
    """回傳 {stock_id: {"top1_count", "appearances", "total_cum_contrib"}}，以及本批策略數。"""
    batch_dir = ART_DIR / label
    strat_dirs = [d for d in batch_dir.iterdir() if d.is_dir()]
    agg = defaultdict(lambda: {"top1_count": 0, "appearances": 0, "total_cum_contrib": 0.0})
    t0 = time.time()
    n = 0
    for d in strat_dirs:
        p = d / "trades.parquet"
        if not p.exists():
            continue
        t = pd.read_parquet(p)
        if len(t) == 0 or "stock_id" not in t.columns:
            n += 1
            continue
        contrib = stock_cum_contrib(t)
        pos = contrib[contrib > 0]
        if len(pos) > 0:
            top1 = pos.idxmax()
            agg[str(top1)]["top1_count"] += 1
            for sid, v in pos.items():
                sid = str(sid)
                agg[sid]["appearances"] += 1
                agg[sid]["total_cum_contrib"] += float(v)
        n += 1
        if n % 1000 == 0:
            log(f"   [{label}] {n}/{len(strat_dirs)}（{time.time()-t0:.0f}s）")
    log(f"   [{label}] 完成 {n} 個策略，{time.time()-t0:.0f}s")
    return dict(agg), n


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    overall = defaultdict(lambda: {"top1_count": 0, "appearances": 0, "total_cum_contrib": 0.0})
    total_n = 0
    t0 = time.time()

    for candidate in CANDIDATES:
        label = f"TW_batch_{candidate}_M"
        ckpt = CKPT_DIR / f"{label}.json"
        if ckpt.exists():
            log(f"[{label}] 已有checkpoint，跳過重掃")
            saved = json.loads(ckpt.read_text(encoding="utf-8"))
            agg, n = saved["agg"], saved["n"]
        else:
            log(f"=== 掃描 {label} ===")
            agg, n = scan_batch(label, log)
            ckpt.write_text(json.dumps({"agg": agg, "n": n}, ensure_ascii=False), encoding="utf-8")
            log(f"   [{label}] checkpoint 已存")
        total_n += n
        for sid, v in agg.items():
            overall[sid]["top1_count"] += v["top1_count"]
            overall[sid]["appearances"] += v["appearances"]
            overall[sid]["total_cum_contrib"] += v["total_cum_contrib"]

    rows = []
    for sid, v in overall.items():
        rows.append({
            "stock_id": sid,
            "top1_count": v["top1_count"],
            "top1_rate": v["top1_count"] / total_n if total_n else 0.0,
            "appearances": v["appearances"],
            "total_cum_contrib": v["total_cum_contrib"],
        })
    df = pd.DataFrame(rows).sort_values("top1_count", ascending=False).reset_index(drop=True)
    out_path = OUT_DIR / "stock_contribution_scan.parquet"
    df.to_parquet(out_path)
    log(f"=== 全部完成｜總策略數 {total_n}｜總耗時 {(time.time()-t0)/3600:.2f}h ===")
    log(f"寫入 {out_path}")
    log("\nTop1貢獻股 前10名（依 top1_count）：")
    for _, r in df.head(10).iterrows():
        log(f"   {r['stock_id']:8s} top1_count={r['top1_count']:5d}（{r['top1_rate']:.2%}）"
            f" appearances={r['appearances']:6d} total_cum_contrib={r['total_cum_contrib']:.1f}")
    log("\nTotal累加值 前10名（依 total_cum_contrib，會受策略重疊影響，僅供對照）：")
    for _, r in df.sort_values("total_cum_contrib", ascending=False).head(10).iterrows():
        log(f"   {r['stock_id']:8s} total_cum_contrib={r['total_cum_contrib']:8.1f}"
            f" top1_count={r['top1_count']:5d} appearances={r['appearances']:6d}")


if __name__ == "__main__":
    main()
