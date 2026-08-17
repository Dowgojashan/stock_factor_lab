#!/bin/bash
# 接續：openSec（台美）跑完後 → 台股剩下三個變體（strict / relaxed / all）
#
# 順序由使用者指定（2026-08-17）：
#   1. 台美股 openSec + 分析      ← 由 _rerun_after_a1.sh 執行中
#   2. 台股剩下三變體             ← 本檔
#   3. 其餘（美股三變體、文件改寫…）等使用者決定
#
# 同樣只重跑 Phase 3/4：q_band 遮罩已逐格驗證不變，Phase 1/2 產物保留。

cd /d/git/stock_factor_lab/code || exit 1
PREV=_catalog/rerun_a1_status.txt
LOG=_catalog/rerun_tw_remaining_status.txt
mkdir -p _catalog

say() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOG"; }
step() {
  local desc="$1"; shift
  say ">>> $desc"
  if "$@" >>"$LOG.detail" 2>&1; then say "    ✅ $desc"; return 0
  else say "    ❌ $desc（詳見 $LOG.detail）"; return 1; fi
}

: > "$LOG"
say "等待 openSec 重跑結束（最多 12 小時）…"
for i in $(seq 1 1440); do
  grep -q "===== 結束 =====" "$PREV" 2>/dev/null && { say "openSec 已結束 ✅"; break; }
  sleep 30
done
grep -q "===== 結束 =====" "$PREV" 2>/dev/null || say "⚠️ 等待逾時，仍繼續執行台股三變體"

say "===== 台股剩下三變體（strict / relaxed / all）====="

# 封存這三個變體修正前的 Phase 3/4
ARC=_ARCHIVE_pre_A1_fix
mkdir -p "$ARC"
for d in results_artifacts/TW_L3_M results_artifacts/TW_L4_M \
         results_artifacts/TW_L3_relaxed_M results_artifacts/TW_L4_relaxed_M \
         results_artifacts/TW_L3_all_M results_artifacts/TW_L4_all_M; do
  [ -d "$d" ] && mv "$d" "$ARC"/ 2>/dev/null
done
say "已封存台股三變體的舊 Phase 3/4"

# all 先跑（策略數最多，早點知道會不會出事），再 strict、relaxed
for v in all strict relaxed; do
  step "TW/$v Phase 2 分析" python -W ignore phase2_analyze.py --market TW --variant "$v"
  step "TW/$v Phase 3" python phase3_conditions.py --market TW --variant "$v" \
    && step "TW/$v Phase 3 分析" python -W ignore phase3_analyze.py --market TW --variant "$v" \
    && step "TW/$v Phase 4" python phase4_valuation.py --market TW --variant "$v" \
    && step "TW/$v Phase 4 分析" python -W ignore phase4_analyze.py --market TW --variant "$v"
done

# 台股四個變體都是新的了 → 台股的四變體對照有效（只讀台股檔案，不受美股影響）
step "TW 四變體對照" python -W ignore variant_compare.py --market TW

# 三個變體的圖鑑
for v in all strict relaxed; do
  step "TW/$v 第四章圖鑑" python -W ignore build_atlas.py --market TW --variant "$v"
done

say "⏭ 未跑：美股 strict/relaxed/all（等使用者決定）→ 故**不跑** US 的 variant_compare"
say "===== 結束 ====="
say "成功 $(grep -c '✅' "$LOG") 步｜失敗 $(grep -c '❌' "$LOG") 步"
