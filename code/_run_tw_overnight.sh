#!/bin/bash
# 等美股跑完 → 自動跑台股四變體全線（2026-08-14 夜間無人值守）
#
# 設計原則：
#   1. 先跑 openSec（正式版設計），萬一半夜出事至少主線是完整的
#   2. 變體之間用 `;` 不用 `&&`——某個變體失敗不會拖垮其他變體
#   3. 每個步驟的成敗都寫進 _catalog/tw_overnight_status.txt，早上一看就知道
#   4. 等待美股有 4 小時上限，逾時就直接開跑（台股不依賴美股，只是避免搶資源）

cd /d/git/stock_factor_lab/code || exit 1
LOG=_catalog/tw_overnight_status.txt
mkdir -p _catalog
: > "$LOG"

say() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOG"; }

step() {                       # step <說明> <指令...>
  local desc="$1"; shift
  say ">>> 開始：$desc"
  if "$@" >>"$LOG.detail" 2>&1; then
    say "    ✅ 完成：$desc"
    return 0
  else
    say "    ❌ 失敗：$desc（詳見 $LOG.detail）"
    return 1
  fi
}

# ---------- 0. 等美股 ----------
say "等待美股 variant_compare 產出（最多 4 小時）…"
for i in $(seq 1 480); do
  [ -f ../_analysis_outputs_variants/US_variant_compare.csv ] && { say "美股已完成 ✅"; break; }
  sleep 30
done
[ -f ../_analysis_outputs_variants/US_variant_compare.csv ] || say "⚠️ 等待逾時，仍直接開跑台股"

say "===== 台股四變體開始 ====="

# ---------- 1. openSec / all 共用的 19 因子 Phase 2 ----------
step "Phase 2 回測（19 因子池，openSec/all 共用）" \
     python phase2_pairing.py --market TW --variant openSec
step "Phase 2 分析 openSec" python -W ignore phase2_analyze.py --market TW --variant openSec
step "Phase 2 分析 all"     python -W ignore phase2_analyze.py --market TW --variant all

# ---------- 2. openSec 全線（正式版，優先） ----------
step "openSec Phase 3" python phase3_conditions.py --market TW --variant openSec \
  && step "openSec Phase 3 分析" python -W ignore phase3_analyze.py --market TW --variant openSec \
  && step "openSec Phase 4" python phase4_valuation.py --market TW --variant openSec \
  && step "openSec Phase 4 分析" python -W ignore phase4_analyze.py --market TW --variant openSec

# ---------- 3. all ----------
step "all Phase 3" python phase3_conditions.py --market TW --variant all \
  && step "all Phase 3 分析" python -W ignore phase3_analyze.py --market TW --variant all \
  && step "all Phase 4" python phase4_valuation.py --market TW --variant all \
  && step "all Phase 4 分析" python -W ignore phase4_analyze.py --market TW --variant all

# ---------- 4. strict / relaxed 共用的 14 因子 Phase 2 ----------
step "Phase 2 回測（14 因子池，strict/relaxed 共用）" \
     python phase2_pairing.py --market TW --variant strict
step "Phase 2 分析 strict"  python -W ignore phase2_analyze.py --market TW --variant strict
step "Phase 2 分析 relaxed" python -W ignore phase2_analyze.py --market TW --variant relaxed

step "strict Phase 3" python phase3_conditions.py --market TW --variant strict \
  && step "strict Phase 3 分析" python -W ignore phase3_analyze.py --market TW --variant strict \
  && step "strict Phase 4" python phase4_valuation.py --market TW --variant strict \
  && step "strict Phase 4 分析" python -W ignore phase4_analyze.py --market TW --variant strict

step "relaxed Phase 3" python phase3_conditions.py --market TW --variant relaxed \
  && step "relaxed Phase 3 分析" python -W ignore phase3_analyze.py --market TW --variant relaxed \
  && step "relaxed Phase 4" python phase4_valuation.py --market TW --variant relaxed \
  && step "relaxed Phase 4 分析" python -W ignore phase4_analyze.py --market TW --variant relaxed

# ---------- 5. 彙總分析 ----------
step "四變體對照" python -W ignore variant_compare.py --market TW
step "規模控制"   python -W ignore size_control_analysis.py --market TW --variant all
step "C 相關矩陣" python -W ignore c_correlation.py --market TW --variant openSec

say "===== 台股全部結束 ====="
say "成功 $(grep -c '✅ 完成' "$LOG") 步｜失敗 $(grep -c '❌ 失敗' "$LOG") 步"
