#!/bin/bash
# A1 徹底修正後的重跑 —— **只跑 openSec（正式版）**，其餘三個變體等使用者決定
#
# 為什麼只重跑 Phase 3/4：
#   修正只影響「季」語意的時序條件（riseq/qmax/yoy/ytdavg），它們改吃公告點稀疏 frame。
#   q_band（橫斷面分位）仍用密集 frame，已用逐格比對驗證台美各 12 組遮罩**完全相同**
#   → Phase 1（純 q_band）與 Phase 2（純 q_band）的回測產物可完整保留。
#
# 修正後的 C 通過率（實測，兩個市場都落在合理區間）：
#            台股              美股           預期
#   riseq1   46.7~48.5%       48.9~51.5%     ~50%
#   riseq2   45.1~47.6%       47.9~51.8%     ~50%
#   yoy      39.1~43.0%       45.9~53.8%     ~50%
#   qmax4    21.7~24.2%       23.7~31.0%     25~35%
#   qmax8     9.4~11.9%       11.8~19.6%     12~20%
#   ytdavg   41.8~47.9%       45.4~55.1%     ~50%

cd /d/git/stock_factor_lab/code || exit 1
LOG=_catalog/rerun_a1_status.txt
mkdir -p _catalog
: > "$LOG"

say() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOG"; }
step() {
  local desc="$1"; shift
  say ">>> $desc"
  if "$@" >>"$LOG.detail" 2>&1; then say "    ✅ $desc"; return 0
  else say "    ❌ $desc（詳見 $LOG.detail）"; return 1; fi
}

say "===== A1 修正後重跑（openSec only）====="

# ---------- 0. 封存舊的 openSec Phase 3/4（C 條件已改，數字作廢） ----------
ARC=_ARCHIVE_pre_A1_fix
mkdir -p "$ARC"
for d in results_artifacts/TW_L3_openSec_M results_artifacts/TW_L4_openSec_M \
         results_artifacts/US_L3_openSec_M results_artifacts/US_L4_openSec_M; do
  [ -d "$d" ] && mv "$d" "$ARC"/ 2>/dev/null
done
say "已封存 $(ls "$ARC" 2>/dev/null | wc -l) 個舊產物至 $ARC"
say "⚠️ 其餘變體（strict/relaxed/all）的 Phase 3/4 仍是修正前的，**尚未重跑、數字已作廢**"

# ---------- 1. Phase 2 分析（補 B1 的兩個持股數欄位；回測不動、白名單不變） ----------
step "TW/openSec Phase 2 分析" python -W ignore phase2_analyze.py --market TW --variant openSec
step "US/openSec Phase 2 分析" python -W ignore phase2_analyze.py --market US --variant openSec

# ---------- 2. openSec 全線 ----------
for mkt in TW US; do
  step "$mkt/openSec Phase 3" python phase3_conditions.py --market "$mkt" --variant openSec \
    && step "$mkt/openSec Phase 3 分析" python -W ignore phase3_analyze.py --market "$mkt" --variant openSec \
    && step "$mkt/openSec Phase 4" python phase4_valuation.py --market "$mkt" --variant openSec \
    && step "$mkt/openSec Phase 4 分析" python -W ignore phase4_analyze.py --market "$mkt" --variant openSec
done

# ---------- 3. openSec 的後續分析 ----------
for mkt in TW US; do
  step "$mkt C 相關矩陣" python -W ignore c_correlation.py --market "$mkt" --variant openSec
  step "$mkt 穩健性檢定" python -W ignore robustness_checks.py --market "$mkt" --variant openSec
  step "$mkt/openSec 第四章圖鑑" python -W ignore build_atlas.py --market "$mkt" --variant openSec
done

# ⚠️ variant_compare 需要四個變體的 Phase 4，openSec 以外都還是舊的 → 這輪不跑，
#    避免產生「新舊混用」的對照表（本專案已因混用吃過兩次虧）。
say "⏭ 跳過 variant_compare：其餘變體尚未重跑，跑了會是新舊混用"

say "===== 結束 ====="
say "成功 $(grep -c '✅' "$LOG") 步｜失敗 $(grep -c '❌' "$LOG") 步"
