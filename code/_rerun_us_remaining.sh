#!/bin/bash
# 美股剩下三個變體（strict / relaxed / all）—— A1 修正後重跑
#
# 使用者指定順序（2026-08-17）：
#   1. 台美股 openSec + 分析      ← _rerun_after_a1.sh（已完成）
#   2. 台股剩下三變體             ← _rerun_tw_remaining.sh（已完成）
#   3. 美股剩下三變體             ← 本檔
#
# 只重跑 Phase 3/4：
#   - Phase 1：不受 A1 影響（無 C 條件）
#   - Phase 2：不受 A1 影響（只有 F1×F2，無 C）；q_band 遮罩已逐格驗證 12/12 相同
#     美股的 L2 回測產物 US_L2_M（strict/relaxed 共用）與 US_L2_all_M（all/openSec 共用）都在
#   - phase2_analyze 仍要重跑：它會套用 phase_variants 判定，而先前的 market bug
#     （phase_variants.get(variant) 預設 market="TW"）讓美股用到台股的因子判定

cd /d/git/stock_factor_lab/code || exit 1
LOG=_catalog/rerun_us_remaining_status.txt
mkdir -p _catalog

say() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOG"; }
step() {
  local desc="$1"; shift
  say ">>> $desc"
  if "$@" >>"$LOG.detail" 2>&1; then say "    ✅ $desc"; return 0
  else say "    ❌ $desc（詳見 $LOG.detail）"; return 1; fi
}

: > "$LOG"
say "===== 美股剩下三變體（all / strict / relaxed）====="

# 先確認 Phase 2 回測產物都在，缺了就不該往下跑
for d in results_artifacts/US_L2_M results_artifacts/US_L2_all_M; do
  [ -d "$d" ] || { say "❌ 缺少 $d，中止"; exit 1; }
done
say "Phase 2 回測產物確認存在 ✅"

# 封存這三個變體修正前的 Phase 3/4（openSec 的已在 _rerun_after_a1.sh 處理過）
ARC=_ARCHIVE_pre_A1_fix
mkdir -p "$ARC"
for d in results_artifacts/US_L3_M results_artifacts/US_L4_M \
         results_artifacts/US_L3_relaxed_M results_artifacts/US_L4_relaxed_M \
         results_artifacts/US_L3_all_M results_artifacts/US_L4_all_M; do
  [ -d "$d" ] && mv "$d" "$ARC"/ 2>/dev/null
done
say "已封存美股三變體的舊 Phase 3/4 → $ARC"

# all 先跑（策略數最多，早點知道會不會出事），再 strict、relaxed
for v in all strict relaxed; do
  step "US/$v Phase 2 分析" python -W ignore phase2_analyze.py --market US --variant "$v"
  step "US/$v Phase 3" python phase3_conditions.py --market US --variant "$v" \
    && step "US/$v Phase 3 分析" python -W ignore phase3_analyze.py --market US --variant "$v" \
    && step "US/$v Phase 4" python phase4_valuation.py --market US --variant "$v" \
    && step "US/$v Phase 4 分析" python -W ignore phase4_analyze.py --market US --variant "$v"
done

# 四個變體都是修正後的了 → 美股的四變體對照才有效
step "US 四變體對照" python -W ignore variant_compare.py --market US

# 三個變體的第四章圖鑑
for v in all strict relaxed; do
  step "US/$v 第四章圖鑑" python -W ignore build_atlas.py --market US --variant "$v"
done

say "===== 結束 ====="
say "成功 $(grep -c '✅' "$LOG") 步｜失敗 $(grep -c '❌' "$LOG") 步"
