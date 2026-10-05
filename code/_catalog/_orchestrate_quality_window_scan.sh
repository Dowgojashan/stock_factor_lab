#!/bin/bash
set -uo pipefail
cd "D:/git/stock_factor_lab/code" || exit 1

MASTER_LOG="_catalog/_quality_window_scan_master.log"
PY="../.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

for MONTHS in 60 24; do
    log "=== 建 ${MONTHS}個月 品質窗口的picks ==="
    "$PY" -m app._build_recency_quality_variant --months "$MONTHS" > "_catalog/_build_recency_${MONTHS}mo.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        log "🔴 build(${MONTHS}mo) 失敗（exit=$rc），中止排程"
        exit 1
    fi
    log "✅ build(${MONTHS}mo) 完成"

    log "=== 跑 ${MONTHS}個月 品質窗口的正式8季模擬 ==="
    "$PY" -m app._run_recency_quality_variant --months "$MONTHS" > "_catalog/_run_recency_${MONTHS}mo.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        log "🔴 run(${MONTHS}mo) 失敗（exit=$rc），中止排程"
        exit 1
    fi
    log "✅ run(${MONTHS}mo) 完成"
done

log "=== 🎉 全部品質窗口掃描完成（60個月、24個月都跑完）==="
