#!/bin/bash
# 排程：等 Phase3(openSec_boost) 回測跑完 → Phase3分析 → 確認Phase1/2分析存在
#       → Phase4(openSec_boost) dry-run → 正式回測 → Phase4分析
# 使用者 2026-10-03 00:xx 交代睡前排程，全程背景執行、不需互動。
set -uo pipefail
cd "D:/git/stock_factor_lab/code" || exit 1

MASTER_LOG="_catalog/_orchestrate_master.log"
PY="../.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

log "=== 排程啟動 ==="

# ---------- 1. 等 Phase3(openSec_boost) 回測結束 ----------
log "等待 Phase3(openSec_boost) 回測完成（PID 8171）…"
while ps -p 8171 > /dev/null 2>&1 && ! grep -q "=== Phase 3 結束" _catalog/_phase3_run_openSec_boost.log 2>/dev/null; do
    sleep 30
done
sleep 5   # 讓log檔寫完再檢查
if grep -q "Traceback" _catalog/_phase3_run_openSec_boost.log 2>/dev/null; then
    log "🔴 Phase3 回測失敗，中止排程，見 _phase3_run_openSec_boost.log"
    exit 1
fi
if ! grep -q "=== Phase 3 結束" _catalog/_phase3_run_openSec_boost.log 2>/dev/null; then
    log "🔴 Phase3 process 已結束但沒看到完成標記，中止排程確認狀況"
    exit 1
fi
log "✅ Phase3(openSec_boost) 回測完成"

# ---------- 2. Phase3 分析 ----------
log "開始 Phase3 分析…"
"$PY" phase3_analyze.py --market TW --variant openSec_boost > _catalog/_phase3_analyze_openSec_boost.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    log "🔴 phase3_analyze.py 失敗（exit=$rc），中止排程"
    exit 1
fi
log "✅ Phase3 分析完成，輸出於 _analysis_outputs_phase3/"

# ---------- 3. 確認 Phase1/Phase2 分析都存在 ----------
log "檢查 Phase1/Phase2 分析產出…"
if [ -f "../_analysis_outputs_phase1/TW_phase1_linearity.csv" ]; then
    log "  ✅ Phase1 分析存在：TW_phase1_linearity.csv"
else
    log "  🔴 Phase1 分析缺失：TW_phase1_linearity.csv，之後的 variant 分組會讀不到資料"
fi
if [ -f "../_analysis_outputs_phase2/TW_L2_openSec_boost_體質檢查表.csv" ]; then
    n=$(( $(wc -l < "../_analysis_outputs_phase2/TW_L2_openSec_boost_體質檢查表.csv") - 1 ))
    log "  ✅ Phase2(openSec_boost) 分析存在：體質檢查表 $n 列"
else
    log "  🔴 Phase2(openSec_boost) 分析缺失，Phase3/4 的白名單會讀不到資料"
fi

# ---------- 4. Phase4 dry-run ----------
log "=== Phase4(openSec_boost) dry-run 確認規模 ==="
"$PY" phase4_valuation.py --market TW --variant openSec_boost --dry-run > _catalog/_phase4_dryrun_openSec_boost.log 2>&1
cat _catalog/_phase4_dryrun_openSec_boost.log | tee -a "$MASTER_LOG"

# ---------- 5. Phase4 正式回測 ----------
log "=== 啟動 Phase4(openSec_boost) 正式回測（預估規模與Phase3相近，可能要數小時）==="
"$PY" phase4_valuation.py --market TW --variant openSec_boost > _catalog/_phase4_run_openSec_boost.log 2>&1
rc=$?
if [ $rc -ne 0 ] || grep -q "Traceback" _catalog/_phase4_run_openSec_boost.log 2>/dev/null; then
    log "🔴 Phase4 回測失敗（exit=$rc），中止排程，見 _phase4_run_openSec_boost.log"
    exit 1
fi
log "✅ Phase4(openSec_boost) 回測完成"

# ---------- 6. Phase4 分析 ----------
log "開始 Phase4 分析…"
"$PY" phase4_analyze.py --market TW --variant openSec_boost > _catalog/_phase4_analyze_openSec_boost.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    log "🔴 phase4_analyze.py 失敗（exit=$rc），中止排程"
    exit 1
fi
log "✅ Phase4 分析完成，輸出於 _analysis_outputs_phase4/"

log "=== 🎉 全部排程完成：Phase3+Phase4(openSec_boost) 回測與分析都跑完了 ==="
