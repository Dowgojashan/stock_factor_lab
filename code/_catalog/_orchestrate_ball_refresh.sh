#!/bin/bash
# 排程：等 walkforward_partition（M-03b, PID 9900）跑完（純CPU資源排隊考量，
# 見下方2026-10-18修正說明）→ 重建TW window4候選池樹（取得新
# tsmc_cluster_concentration.csv）→ B_all股票層重算（window4 + windows1-3）
# 使用者 2026-10-17 交代排程，全程背景執行、不需互動。
# 🔴 刻意不包含台/美股L2真實重跑——那會燒真實LLM API成本，需要使用者
# 明確下指令才能啟動，不納入自動排程鏈。
#
# 🔴🔴 2026-10-18修正：原版還包含walkforward_evidence/walkforward_significance/
# complementarity_granularity三步，當初排進「等M-03b跑完」是因為看到這三個的
# log檔跟M-03b同一個時間戳記（2026-10-14 16:32），**憑時間巧合推測它們依賴
# M-03b的輸出**，沒有真的去查原始碼驗證——查證後發現這個假設是錯的：
#   - walkforward_evidence.py / walkforward_significance.py 讀的是
#     `walkforward_matrix_detail.csv`（2026-10-11就已經是新版，跟M-03b
#     完全無關的另一條鏈）
#   - complementarity_granularity.py 讀的是`free_lunch_shortlist.csv`
#     （自己內部重算，一樣不碰`walkforward_partition.csv`）
# 這三支腳本**從頭到尾都不讀`walkforward_partition.csv`**，不需要等M-03b，
# 之前（2026-10-17 23:53）已經用正確的最新輸入跑過一次、結果有效，**不需要
# 重跑**，已經從這支排程移除。真正需要等M-03b的只有下面樹重建＋B_all重算
# 這兩步，而且理由是**CPU資源排隊**（跟M-03b一樣在用`build_tree_for_window()`
# 建樹），不是資料相依。
set -uo pipefail
cd "D:/git/stock_factor_lab/code" || exit 1

MASTER_LOG="_catalog/_orchestrate_ball_refresh.log"
PY="../.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

log "=== 排程啟動（2026-10-18修正版：只剩樹重建＋B_all重算，前三步已確認不需要等M-03b、已於10/17跑完）==="

# ---------- 1. 等 walkforward_partition（M-03b）結束（CPU資源排隊，非資料相依）----------
# 🔴 2026-10-17修正：原本用`ps -p 9900`判斷，但git-bash的`ps`只看得到從
# 這個bash lineage衍生出來的process，看不到PID 9900這種從外部直接起的native
# python.exe——`ps -p 9900`會立刻回傳失敗（找不到），導致while迴圈誤判成
# 「已經跑完」。改用windows原生`tasklist`判斷，這個在git-bash裡能正確看到
# 任意native PID（已測試確認）。
SCRIPT_START_EPOCH=$(date +%s)
log "等待 walkforward_partition 完成（PID 9900，純CPU資源排隊）…"
while tasklist //FI "PID eq 9900" 2>/dev/null | grep -q "python.exe"; do
    sleep 60
done
sleep 5
log "✅ walkforward_partition 進程已結束（這一步只是資源排隊，下面兩步本來就不吃它的輸出檔）"

# ---------- 2. 重建TW window4候選池樹（新tsmc_cluster_concentration.csv）----------
log "=== 開始重建TW window4候選池樹（openSec_boost pool，預估約20分鐘）==="
"$PY" -m app._check_tsmc_cluster_concentration > _catalog/_rebuild_tsmc_cluster_concentration.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    log "🔴 _check_tsmc_cluster_concentration 失敗（exit=$rc），中止排程"
    exit 1
fi
log "✅ tsmc_cluster_concentration.csv 重建完成（新pool）"

# ---------- 3. B_all股票層重算：window4 ----------
log "=== 開始B_all股票層重算（window4，8季，預估數十分鐘）==="
"$PY" -m app._recompute_ball_stock_level > _catalog/_rebuild_ball_stock_level.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    log "🔴 _recompute_ball_stock_level 失敗（exit=$rc），中止排程"
    exit 1
fi
log "✅ B_all（window4）重算完成"

# ---------- 4. B_all股票層重算：windows1-3 ----------
log "=== 開始B_all股票層重算（windows1-3，共36季，預估約2小時）==="
"$PY" -m app._recompute_ball_windows123 > _catalog/_rebuild_ball_windows123.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    log "🔴 _recompute_ball_windows123 失敗（exit=$rc），中止排程"
    exit 1
fi
log "✅ B_all（windows1-3）重算完成"

log "✅ 台股B_all（window4+windows1-3）全部重算完畢"

# ---------- 5. 美股M3（選股廣度）歷史校準 ----------
# 🔴 2026-10-18新增：diagnose.py查證發現M0/M3/M6/M7/M8全部沒分市場，美股
# production run一直在用台股的歷史門檻（見diagnose.py開頭說明）。這一步
# 很輕（只讀walkforward_members.parquet既有的美股代表名單，不用建樹）。
log "=== 開始美股M3（選股廣度）歷史校準 ==="
"$PY" -m app._prelim_a1_portfolio_noise_us > _catalog/_rebuild_a1_portfolio_noise_us.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    log "🔴 _prelim_a1_portfolio_noise_us 失敗（exit=$rc），中止排程"
    exit 1
fi
log "✅ 美股M3校準完成"

# ---------- 6. 美股M8（產業集中度）歷史校準 ----------
log "=== 開始美股M8（產業集中度）歷史校準 ==="
"$PY" -m app._prelim_industry_concentration_history_us > _catalog/_rebuild_industry_concentration_us.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    log "🔴 _prelim_industry_concentration_history_us 失敗（exit=$rc），中止排程"
    exit 1
fi
log "✅ 美股M8校準完成"

# ---------- 7. 美股候選池universe建樹（給M0/M6/M7用）----------
log "=== 開始重建美股window4候選池樹（預估約20分鐘）==="
"$PY" -m app._check_us_cluster_universe > _catalog/_rebuild_us_cluster_universe.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    log "🔴 _check_us_cluster_universe 失敗（exit=$rc），中止排程"
    exit 1
fi
log "✅ 美股候選池universe建樹完成"

# ---------- 8. 美股M0/M6/M7（B_all）歷史校準 ----------
log "=== 開始美股M0/M6/M7（B_all）歷史校準（windows1-3，共36季)==="
"$PY" -m app._recompute_m0_performance_us > _catalog/_rebuild_m0_performance_us.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    log "🔴 _recompute_m0_performance_us 失敗（exit=$rc），中止排程"
    exit 1
fi
log "✅ 美股M0/M6/M7校準完成"

log "=== 🎉 全部排程完成：台股B_all＋美股M3/M8/M0/M6/M7歷史校準全部補齊 ==="
log "    （walkforward_evidence/significance/complementarity_granularity已於10/17用正確輸入跑完，不在這支排程內）"
log "    下一步：台股/美股4+4組L2真實重跑——需要使用者明確授權才會啟動，不含在這支排程內"
