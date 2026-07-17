# A4｜策略庫 Sweep 產生器 開發計畫（v1）

> 範圍：**把「跑一份 spec」升級成「有系統掃描設計空間、把結果累積成庫」**
> 目標產物：**因子百科全書（策略庫）** → 供後續 LLM Agent 查詢、依投資人風險偏好挑策略
> 前置：A3（q_band 分位數 spec）已完成並驗證；spec_US 首片庫已產出（2310 策略）

---

## 0. 定位：這份計畫在整個研究的哪裡

老師（2026-07 逐字稿）定調的三層分工：

| 層 | 做什麼 | 誰做 | 狀態 |
|---|---|---|---|
| 產生器 | 暴力展開 F/C/V 組合、跑回測、**把結果全存起來** | 電腦一直跑 | **← A4 這份** |
| 百科全書 | 累積成一個「很多策略方向」的庫 | 產生器的產物 | A4 產出 |
| **LLM Agent** | 面對投資人需求（保守/激進），從庫裡**挑/組合**出建議 | LLM | **研究真正的貢獻點**（另立） |

**老師原話重點**：
- 「原則上你應該是讓它電腦一直在跑，把它操作爆迴來」→ 模式 A：全算完
- 「不會太久就是你可能跑不是禮拜，反正電費不用久」→ 一週量級可接受
- 「**那個挑才是那個 AI Agent 在幫你的**」「**你的重點在你的 LLM，不在於你跑到最漂亮**」
- 「F 應該要有一百多的組合，C 大概二十幾個，V 你基本上可以抓 1~2 個」

> ⚠️ **不是「產更多圖」**。百科全書的本體是**資料**（每個策略的 stats + 明細表），圖表只是給人看的輔助。LLM 讀結構化資料遠比讀 PNG 準。

---

## 1. 現況盤點

### 已完成（A3 + 首片庫）
| 項目 | 狀態 |
|---|---|
| `condition_factory.py` | q_band（橫斷面分位）已加、驗證分桶各 20% 正確 |
| `spec_generator.py` | 產 spec_TW/spec_US（設定寫死在檔頭） |
| `fcv_us.py` | US 專用 runner，**串流分批回測**（記憶體安全，峰值 ~13GB） |
| `database.py` | US 讀取已 JOIN `russell3000` 圈定宇宙（**market-gated，台股 SQL 逐字元不變**） |
| **spec_US 首片庫** | Russell 3000（2809 檔）× 2000–2026 → **2310 策略**，93 分鐘跑完 |
| `analyze_spec_us.py` | 走 parquet 的分析（8 張圖 + 排行 CSV）→ `_analysis_outputs/spec_US/` |

### 缺口（A4 要補）
1. **runner 寫死 US** — `fcv_us.py` 硬編 russell/US_START，TW 無法共用
2. **spec_generator 設定寫死** — 無法吃 config 產生多變體
3. **只有 6 個 C** — 老師要 ~20
4. **沒有 sweep driver** — 無法自動跑多 config、無續傳、無崩潰韌性
5. **沒有 catalog** — 各 config 的 stats 沒併成單一「百科全書索引」
6. **`report_analysis.ipynb` 指向舊 label** — 未讀 master_index

---

## 2. 定案設定（2026-07 與老師/使用者確認）

| 項目 | 定案 | 備註 |
|---|---|---|
| 跑法 | **模式 A：全算完再找** | 老師：不會太久就全算 |
| 市場 | **US + TW 都跑** | 各自成庫，LLM 各自查 |
| 因子 | **台美都用共同 3：ROE / EPS / FCF_P** | 台股只有這 3（**待 DB 確認**） |
| N 分位 | **{5, 10}** | 粗細兩種granularity |
| C 動態條件 | **20 個**（見 §5） | 不需改 condition_factory |
| V 估值濾網 | **2（v0/v1）** | 維持現有 |
| 換股頻率 | **{M, Q}** | 回測參數，不需重載 |
| 時間窗 | **2000–2026（台美同範圍）** | TW 資料待補（目前僅到 2023） |
| 宇宙 | US=Russell 3000(2809)；TW=全 TWSE | |
| F1×F2 對稱去重 | **要**（展開時就砍） | 省 45%，見 §7 |
| 執行載具 | **腳本 + supervisor（非 notebook）** | notebook 不適合連續跑數天 |

> **台美因子不需一致**（使用者定案）；但因兩邊都用共同 3 因子，實際上是**同一份 recipe 各跑一次**（老師方向1），順便保住跨市場可比。

---

## 3. 關鍵洞察：什麼會逼你重載資料

`Data(market)` 載入 ~5 分鐘、吃 ~7GB，是最貴的一步。

- **F / N / C / V / 換股頻率** → 全部可在**同一份載入的資料**上算，**不需重載**
- **只有 market / 宇宙 / 時間窗改變** → 才需重載

**→ 設計原則：一個市場載入一次，跑完它所有 spec 變體。**
「一週」不是來自重載，而是來自**組合本身的量**。

---

## 4. 架構

```
設定矩陣(experiment matrix)        ← 要掃哪些軸、每軸取哪些值
        │ 展開成一堆 config
        ▼
spec 產生器(參數化)                 ← config → spec（大量 F/C/V 條件）
        │
        ▼
回測引擎(串流分批，已有)            ← 每份 spec → results_artifacts/<label>/
        │
        ▼
主目錄 + 去重登記(catalog)          ← 併成「百科全書索引」master_index
```

### 執行流程（連續、可續傳、無人值守）

```
supervisor（監督器）── 崩了自動重啟
   └─ for 資料群組 in [US, TW]:              # 只有 2 個，各載一次
         worker(群組)
            data = load Data(market)          # 5 分鐘，只做一次
            for spec_variant in 該市場工作清單:
                label = 決定式命名(設定)       # US_f3_N5_c20_v2_M
                if 有 _DONE 標記: continue     # ★ 續傳：跳過已完成
                try:
                    串流分批回測(spec, data, label)
                    寫 _DONE
                except: 寫 _FAILED + log，繼續下一個   # ★ 崩潰隔離
            釋放 data
   彙整所有 label → master_index + 全域去重
```

### 四個「一直順跑」的機制
1. **工作切片**：不做「一份跑一週的巨型 spec」；切成**幾小時級**小 job（按 N / 頻率切）→ 崩潰只損失幾小時
2. **`_DONE` 標記 + 續傳**：隨時可停可續（老師：「暑假跑一次、考前再跑一次」）
3. **supervisor 自動重啟**：worker 因 OOM/記憶體碎片掛掉 → 自動重啟 → 重載一次 → 靠 `_DONE` 接著跑（**連記憶體洩漏都擋不住它跑完**）
4. **崩潰隔離 + 日誌**：單 job 出錯只記 `_FAILED`，不影響其他；log 有時間戳/耗時/策略數，隨時可看進度 ETA

> **效率 vs 穩健的平衡**：同市場多 spec 共用一次載入（省 5 分鐘×N），但用 `_DONE` + 自動重啟拿到韌性。**只有崩潰重啟才重載。**

---

## 5. C 條件盤點與擴充（6 → 20）

### 現有可用的 C「條件型別」（`condition_factory.py` 的時序語意工廠）
| 型別 | 語意 | 參數 |
|---|---|---|
| `rise_q(n)` | 較 n 季前上升 q(t)>q(t-n) | n |
| `is_highest_q(n)` | 近 n 季最高（含當季） | n |
| `yoy_gt(4)` | 較去年同季上升（年增 YoY） | periods=4 |
| `ytd_avg_gt_prev_year_same_period_avg()` | 今年至今均 > 去年同期均 | — |
| `ytd_avg_gt_prev_year_avg()` | 今年至今均 > 去年全年均 | — |

> q_band / gt / lt / between 是 **F（水準）** 用的；`is_largest` 是橫斷面選取 —— 都不放進 C。

### 目前 spec 只用了 6 個
| 代號 | 因子 | 條件 |
|---|---|---|
| C1 | ROE | `is_highest_q(4)` |
| C2 | ROE | `rise_q(1)` |
| C3 | ROE | `yoy_gt(4)` |
| C6 | EPS | `yoy_gt(4)` |
| C7 | EPS | `ytd_avg_gt_prev_year_same_period_avg` |
| C8 | FCF_P | `rise_q(2)` |

（C4/C5 原為 REVENUE，因台股沒有被濾掉。）

### 擴充到 20（**不用改 condition_factory 程式碼**，只是 spec 多列幾條）
| 因子 | riseq1 | riseq2 | qmax4 | qmax8 | yoy | ytd同期 | ytd全年 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ROE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| EPS | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| FCF_P | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |

**= 20 個 C**（3×7 − 1）。台美共用同一套。

---

## 6. 產出結構（庫長什麼樣）

```
results_artifacts/
├─ US_f3_N5_c20_v2_M/           每個 job 一包
│   ├─ stats.parquet
│   └─ <每策略>/{position,trades,stock_data,return_table}.parquet
├─ US_f3_N10_c20_v2_M/
├─ US_f3_N5_c20_v2_Q/
├─ TW_f3_N5_c20_v2_M/
└─ ...
_catalog/
├─ master_index.parquet    ★ 百科全書索引（LLM 查這個）
│     欄位：strategy + F1/F2/C/V + market/N/因子/頻率/窗 + CAGR/sharpe_ann/回撤/勝率 + 指向哪包
└─ dedup_registry.parquet  全域數值指紋去重，一列一個「獨立策略」
run_log.txt / jobs_manifest.parquet   進度、狀態、耗時
```

**`master_index.parquet` 就是給 LLM 的百科全書索引**：一列一個獨立策略，帶完整風險報酬 + 設定標籤。
LLM 服務保守型 → `filter(回撤 < X, sharpe_ann > Y)`；激進型 → 換條件。**這就是老師說的「從庫裡挑菜」。**

---

## 7. 規模與時間試算

以定案設定（3 因子 × N{5,10} × C20 × V2 × 頻率{M,Q}）計算：

| 階段 | 計算 | 數量 |
|---|---|---|
| P1 條件 | 3 因子 × (5+10) 桶 | **45** |
| F（有序） | 45 × (1 + 30異因子) | 1,395 |
| **F（對稱去重後）** | 45 單 + 675 無序對 | **720** |
| × C | × (1 + 20) | 15,120 |
| × V | × 2 | 30,240 |
| × 頻率 | × 2 | **60,480 / 市場** |
| **台美合計** | × 2 市場 | **≈ 120,960 次回測** |

**時間**：實測 ~2.3 秒/策略（US 2809 宇宙、月頻）
→ US ≈ 39 小時、TW ≈ 較快（宇宙較小）
→ **合計 ≈ 2.5–3 天**（符合老師「一週」量級，且模式 A 可全算完）

> **不去重的話 ≈ ×2（~5–6 天）且庫裡一半是重複** → 對稱去重是必要的。

### ⚠️ 磁碟空間（**待實測**）
121k 策略 × 每策略 4 張 parquet（position 最大，寬度=宇宙）
→ 粗估 **60–120 GB**。**開發前務必先實測 spec_US（2310 策略）的實際佔用再外推。**
若太大，選項：sweep 只存 **stats + return_table/stock_data**（小），
`position/trades` 只對 LLM 短名單的策略**按需重生**。

---

## 8. 開發項目

| # | 項目 | 說明 |
|---|---|---|
| 1 | **runner 市場泛化** | `fcv_us.py` → 抽出 `run_spec(market, spec, data, label)`，US/TW 共用；載入與跑 spec 拆開，讓 driver 能載一次餵多個 spec |
| 2 | **spec_generator 參數化** | 吃 config dict（多因子/多 N/多 C）產 spec；決定式 label 命名 |
| 3 | **C 擴充到 20** | 依 §5 表在 spec 鋪滿（**不需改 condition_factory**） |
| 4 | **F1×F2 對稱去重** | 在串流展開時就只產無序對（省 45% 時間與空間） |
| 5 | **sweep driver + supervisor** | 工作清單、載入共享、`_DONE` 續傳、崩潰隔離、自動重啟、log/ETA |
| 6 | **catalog builder** | 併 master_index + 全域去重 + 設定標籤 |
| 7 | **sharpe_ann 內建** | 落地時就算正確年化 Sharpe（見 §10），不要事後補 |
| 8 | **analysis 動態化** | `report_analysis.ipynb` 改讀 `master_index`（分析層動態，非執行層） |

**開發順序建議**：1 → 2 → 3/4 → 5 →（小規模冒煙）→ 全量 → 6 → 8

---

## 9. 待確認 / 阻塞項

- [ ] **台股因子數確認**：假設只有共同 3 個（ROE/EPS/FCF_P），**需 DB 輕量查詢確認**（不可用重查詢，見 §10）
- [ ] **台股資料補到 2026**：目前 DB 台股僅到 2023，使用者待補
- [ ] **磁碟空間實測**：先量 spec_US(2310) 實際佔用再外推 121k（§7）
- [ ] IS/OOS 切分：另立計畫，非本次範圍
- [ ] 宇宙分層（流動性/市值）：之後再加軸

---

## 10. 已知問題與踩過的坑（**開發時必讀**）

| # | 問題 | 處置 |
|---|---|---|
| 🔴 | **`daily_sharpe` 是壞值（7~460）** | finlab `get_stats()` 把 `mean/std×√252` 套在未正規化的 `portfolio_returns`（量級 10⁵）上 → 無意義。**要從 `return_table` 月報酬 `mean/std×√12` 重算**（spec_US 中位 0.44）。已在 `analyze_spec_us.py` 的 `add_sharpe_ann()` 實作，**應內建進落地流程**。 |
| 🔴 | **45% 對稱重複** | `f_factor` 用有序配對 → `EPS__ROE` 與 `ROE__EPS` 交集相同、完全重複。2310 個裡數值唯一僅 1260。**展開時就去重**（開發項目 #4）。 |
| 🔴 | **XAMPP MariaDB 極脆弱** | `innodb_buffer_pool_size=16M`。2026-07-14 對 `factorvalue`(837萬列) 跑全表 `GROUP BY` → **MariaDB 崩潰、tablespace 損毀、整個 DB 起不來**（已修復）。**嚴禁對 factorvalue/stock 跑全表掃描**；查詢一律加 `WHERE` 限縮 + `LIMIT`。 |
| 🟠 | **記憶體：遮罩展開會爆** | 全宇宙下 `final_masks` 同時持有全部策略 ≈ 47GB，且展開/預篩階段有多份副本 → 峰值 73–91GB，**64GB 也會 OOM**。**必須用串流分批**（已實作，峰值 ~13GB）。 |
| 🟠 | **`Data(market)` 不管宇宙限縮都全載** | 已用 `database.py` 的 russell JOIN 把 US 載入從 9200 檔降到 2990（記憶體 17.5GB → 7.2GB）。 |
| 🟠 | **倖存者偏誤（美股）** | 宇宙固定為「今日 Russell 3000」回推 2000 → 只留存活者 + 贏家 backfill。**A3 doc Q6 的『無偏誤』結論在此路徑不成立**，論文 limitations 必須揭露。 |
| 🟡 | **filter-before-rank** | 加宇宙篩選後，q_band **必須在限縮後的 frame 上排名**（否則桶不是「宇宙內前 20%」）。已在 `fcv_us.py` 於 `build_masks` 前 reindex 財報 frame 實作。 |
| 🟡 | **parquet 欄名混型別警告** | 存檔時 pandas 警告 mixed type column names，資料有存下但 roundtrip 可能不完全。LLM 要讀這批 result，**格式可靠性待驗**。 |
| 🟡 | **不要用 notebook 跑庫** | kernel 狀態累積、記憶體不還、一崩全死、難續傳。**跑庫用腳本；notebook 只做分析。** |

---

## 11. 附錄：現有指令

```bash
# 產 spec（目前設定寫死在檔頭）
cd code && ../.venv/Scripts/python.exe spec_generator.py

# 跑一份 spec（US，串流分批）
cd code && ../.venv/Scripts/python.exe -u fcv_us.py            # 全量
BATCH_SIZE=150 US_UNIV_LIMIT=50 SMOKE_LIMIT=20 ... fcv_us.py   # 冒煙

# 分析（走 parquet，不需 pickle）
cd code && ../.venv/Scripts/python.exe analyze_spec_us.py
```

**環境**：`.venv`（Python 3.10）；DB = XAMPP MariaDB 10.4（`C:\xampp\mysql`）；
`config.ini` 為相對路徑 `../config.ini` → **腳本 cwd 必須是 `code/`**。
