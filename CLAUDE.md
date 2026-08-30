# 交接文件（給下一個 Claude Code 對話框）

> 這份文件會在每次對話開始時自動載入。寫於 2026-08-20，因為前一個對話框用太久要換新的。
> 使用者是這個研究計畫的主持人（碩士論文相關研究），中文溝通。

---

## 0. 讀這份文件前，先看兩件事

1. **`C:\Users\iplab\.claude\projects\d--git-stock-factor-lab\memory\MEMORY.md`** —— 使用者的長期 auto memory 索引，應該會自動載入，裡面有更細的專案脈絡跟使用者的溝通偏好，這份文件不重複那邊已經有的內容。
2. **根目錄有三份使用者剛給的新架構文件，使用者明確說「先不要動」，我(上一個對話框)還沒讀過**：
   ```
   GateC_標記清單_v1.md
   實戰部架構_v8.md
   研究部完整流程_v9.md
   ```
   **這極可能就是下一階段（HRP）要做的架構規劃，是你接手後第一件事應該去讀的東西。** 使用者說完成的東西會交給你處理。

---

## 1. 這個專案在做什麼

`stock_factor_lab`：量化因子選股回測系統，碩士論文相關研究。參考學姊余姵穎的論文《宣告式多因子量化回測系統之設計與實作》（PDF 在 `文件/`，但已依使用者要求移除公開版，若要看要另外問使用者要參考資料位置）。指導教授會定期開會給方向，逐字稿在 `文件/8-5咪挺.pdf`、`文件/8-19咪挺.pdf`（老師談話內容，不對外公開，`.gitignore` 已排除 `*咪挺*`）。

兩個市場：**台股**（TWSE 全部上市公司，1,775 家）、**美股**（Russell 3000，2026-07-08 快照，2,972 檔，⚠️ 有已知倖存者偏誤，已量化並寫進報告）。

### FCV 框架
```
F（體質因子，選什麼樣的公司）× C（動態條件，時間點對不對）× V（估值濾網，貴不貴）
```
指導教授的 SOP：分四階段，一次只開一個維度——
```
Phase 1  單因子健檢（9桶線性/單調性檢定，Spearman ρ）
Phase 2  F1×F2 不對稱配對（primary嚴格、secondary寬鬆）
Phase 3  加動態條件 C（20種，衍生自 ROE/EPS/FCF_P）
Phase 4  加估值濾網 V（PE 相對估值）
```

---

## 2. 目前進度：Phase 1~4 已全部完成，老師已核可，等下一步指示

**2026-08-19 已經跟老師報告過（逐字稿 `文件/8-19咪挺.pdf`），老師確認台美股的策略結果都 OK，同意進入下一階段。**

### 已完成的內容
- 台美股 Phase 1~4 全部跑完，四個變體（strict / **openSec**（採用）/ relaxed / all）都跑完並互相對照過
- **openSec** 是採用的正式設計：primary 用 Phase 1 過關的因子嚴格篩，secondary 全部開放（因為 Phase 1 檢定的是「因子自己能不能單調預測」，但 secondary 的功能是「提供不同構面資訊」，兩件事不一樣；台美股都獨立驗證過這個設計是對的）
- 修過一個關鍵 bug（A1 defect）：美股的動態條件 C 原本全部失效（用日頻 ffill 密集 frame 而非公告點稀疏 frame，導致「較上季升」恆假、「近N季最高」恆真），修好後美股 C 的排名完全改變
- 台股 2000-2004 財報資料回補完成（原本是空的，讓早期分位桶假訊號嚴重）
- 做過多項穩健性檢定：D1（9桶算術合併 vs 3桶實跑）、C1（primary門檻敏感度）、B3（規模集中度）、D3（子期間穩健性）——全部台美股都做過
- 最終候選策略池：台股 7,162 個、美股 6,916 個，都贏過各自的自建宇宙基準（台股 8.43%、美股 11.06%，2026-08-22 用修復後價格重算過，見 `code/research/contracts.py` 的 `BENCHMARK_CAGR`；舊數字是 8.67%/12.35%，重算前的）

### 給老師看的核心產出文件（都在 `文件/`）
| 檔案 | 用途 |
|---|---|
| `文件/策略成果報告_2026-08-17.md` | **最重要**，給老師看的正式成果報告：Phase1~4怎麼跑、每階段發現、最終候選策略，結果導向敘事（**不含bug修正過程**） |
| `文件/週進度報告_2026-08-17.md` | 進度追蹤報告，含 bug 修正史 |
| `文件/重跑計畫_老師方法論SOP.md` | 老師方法論的完整逐字稿摘錄 + 對應實作規劃 |

以及對應的 HTML 版本（已發布為 Artifact，若使用者要更新記得用 `url` 參數更新原連結不要開新的）：
- 台股：`https://claude.ai/code/artifact/8ab7bd29-ec6e-44e5-b4bc-e79f601424b0`
- 美股：`https://claude.ai/code/artifact/5b6bd2ec-0be4-4e6d-9531-4d6f5200be2b`

### 各階段細節分析文件（都還在使用中，不要移動/刪除）
```
_analysis_outputs_phase1/  Phase1_結果分析.md + phase1_curves.png（9桶單因子曲線）
_analysis_outputs_phase2/  Phase2_結果分析.md + pairing_gain.png
_analysis_outputs_phase3/  Phase3_結果分析.md + C_gain.png
_analysis_outputs_phase4/  Phase4_結果分析與變體對照.md + V_effect.png + candidate_pool_distribution.png
_analysis_outputs_atlas/   第四章18張圖鑑（TW/US × 4變體 × openSec 都有，每組18張）
_analysis_outputs_ccorr/         C 相關性分析
_analysis_outputs_robustness/    D1/C1/B3/D3 穩健性檢定 + 說明文件
_analysis_outputs_sizecontrol/   規模控制分析（用真市值 MKTCAP 三分位）
_analysis_outputs_variants/      四變體對照（strict/openSec/relaxed/all）
```

---

## 3. 下一步：老師指定做 HRP（階層式風險平價）

2026-08-19 會議裡老師給的明確方向：**Hierarchical Risk Parity（HRP）**。

老師的解釋（口語，詳見 `文件/8-19咪挺.pdf` 第3頁）：直接對一堆策略算兩兩相關性、拿去做資產配置，只要一個相關係數估錯就會錯得很離譜。HRP 的做法是**先把策略做階層式分群**（依相似度分成幾群），再在群與群之間做風險分散配置——這樣抓到的分散效果（老師說的「吃到免費午餐」）比直接硬算相關性矩陣更穩健。老師認為這對我們現在手上幾千個策略（很多獲利模式很像）特別有用：先分群、再從每群挑代表，組合出來的結果比較容易穩健。

老師後續會提供評估標準，之後可能要跟美國公債等資產再做比較分析。

**⚠️ 使用者說接下來的架構會整理好交給你** —— 就是本文件開頭提到的那三份新文件（`GateC_標記清單_v1.md`、`實戰部架構_v8.md`、`研究部完整流程_v9.md`），先讀那三份再開始規劃 HRP 怎麼實作。

**技術上的提示（給你參考，不是指示）**：HRP 是在「策略層級」做分群配置，操作對象是已經跑完的策略報酬序列（`results_artifacts/*/return_table.parquet` 或 `stats.parquet`），**不需要重新回測、也不需要即時連 SQL**——這也是為什麼 MySQL 打不開那陣子完全不影響進度（見第5節）。

---

## 4. 專案結構與執行注意事項

### 目錄結構（2026-08-20 剛整理過）
```
/                          根目錄：核心 .py 檔（database.py/get_data.py/combinations.py等，主動在用，不要動）
├── 文件/                  所有 .md/.pdf/.docx 文件 + 兩個舊範例資料夾（因子結合策略/Quantile_AA）
├── _archive/               封存的舊資料（已 gitignore，含舊版 _analysis_outputs）
├── _analysis_outputs_*/    現役分析結果（見上表，不要動）
├── code/                   主要程式碼（Phase1~4 pipeline、分析腳本）
│   ├── _archive/            封存的舊回測（原本 code/_ARCHIVE_* 已合併改名進來，已 gitignore）
│   │   └── ⚠️ 有一個 L3L4_strict_oldbench203 因權限問題卡在外面沒搬進來，
│   │      跟裡面的副本內容看起來一致但沒完全驗證過，使用者知情、要自己處理，不用主動管
│   ├── _catalog/             執行紀錄（master_index.parquet / dedup_registry.parquet 是功能性索引，不要刪）
│   ├── results_artifacts/    回測原始產物（巨大，已 gitignore，58.5GB+）
│   └── phase{1,2,3,4}_*.py   四階段的執行與分析腳本
├── core/, utils/, TA-LIB/  底層依賴，不要動
└── .venv/                  虛擬環境，不要動
```

### ⚠️ 執行 Python 腳本的路徑陷阱
- `database.py`、`get_data.py`、`combinations.py`、`format_data.py` 等在**根目錄**，但大部分工作腳本在 `code/` 底下執行
- `code/fcv_core.py` 開頭會自動把 ROOT 跟 `code/` 都加進 `sys.path`（用 `_ROOT = Path(__file__).resolve().parent.parent`），所以只要 `import fcv_core` 過一次，後面 `from database import Database` 才會找得到模組
- **如果你自己寫 ad-hoc 查詢腳本**，切記先 `import fcv_core`（哪怕用不到它），或自己手動把根目錄加進 `sys.path`，不然會 `ModuleNotFoundError: No module named 'database'`
- **`config.ini` 的讀取用相對路徑**，指令碼一定要在 `code/` 目錄下執行（`cd code` 再跑），不要在根目錄跑，不然 `KeyError: 'database'`（`utils/config.py` 讀不到 `[database]` 區段）

### 資料庫連線注意事項
- `Database(market)` 只是開連線，**不會自動依市場過濾**——`stock` 表沒有市場欄位，要 JOIN `company` 表用 `db._exchange_in_clause()` 才會篩對市場，這裡踩過雷（兩個市場撈出同一份資料）
- 美股還要另外加 `db._universe_clause('c.company_symbol')` 才會篩進 Russell 3000 名單（台股這個 clause 回傳空字串，不影響）

---

## 5. 2026-08-20 發生過的意外：MySQL 壞過、已修好

電腦在整理檔案時當機過一次，使用者強制重開機後 XAMPP 的 MySQL（實際是 MariaDB 10.4.32）打不開。

**根因**：`mysql` 系統資料庫（存權限設定，跟研究用的 TEJ 資料是分開的表）裡的 `db` 這張表因硬關機震壞了。**這張表是 Aria 格式**（`.MAD`/`.MAI`），MariaDB 10.4 的系統表預設用 Aria 不是 MyISAM，修復要用 `aria_chk`，不是 `myisamchk`：
```
cd C:\xampp\mysql\bin
aria_chk -r "C:\xampp\mysql\data\mysql\db"
```
修好後驗證過台美股的 `company`/`stock` 資料筆數都正常，沒有資料損失。**如果之後又發生類似狀況，先查 log（`C:\xampp\mysql\data\mysql_error.log`），或直接用 `mysqld.exe --console` 在前景跑一次看完整錯誤訊息**（XAMPP 控制台本身的錯誤視窗訊息不完整，之前就是這樣才卡關卡很久）。

這台機器是遠端桌面連線到學校實驗室電腦，Windows 安全性中心的 GUI 部分設定會被 RDP 擋掉，但用系統管理員 PowerShell 下 `Add-MpPreference` 指令通常還是能用。

---

## 6. 使用者的工作風格與偏好（摘要，完整版在 auto memory）

- **不要碰資深研究員/學長姊的舊 code**，先確認過再動（`因子結合策略/`、`Quantile_AA/` 就是這種，已經歸檔到 `文件/` 但沒有動內容）
- **已核准的長工作可以自主推進**，不用每一步都問，但過程要主動回報進度
- **極度重視查證，不接受憑印象回答**——這個對話框好幾次先查了 code / 直接重算才回答，中間也有幾次自己講錯被抓到、誠實承認並更正。**回答任何「為什麼」「數字從哪來」的問題之前，先去查程式碼或重新計算，不要用記憶回答**
- 給老師的報告要「結果導向」，**不要把除錯過程、bug 修正史寫進去**（除非該文件本來就是進度/bug報告，兩種文件要分開）
- 偏好簡潔直接、有數字佐證的回答，不要贅字
- 涉及刪除/移動大量檔案這類操作前，會想先看清單、自己確認過才放心讓你做

---

## 7. 論文文獻位置

學姊論文 PDF：`C:\Users\iplab\Documents\_backup_from_repo\余姵穎_宣告式多因子量化回測系統之設計與實作.pdf`（不在 repo 內，`.gitignore` 也明確排除 `/參考資料/`，不可公開）。用 `fitz`（PyMuPDF，已裝在環境裡）读取，`pdftoppm` 沒裝所以 `Read` 工具的 PDF 頁面渲染會失敗，要用 `python -c "import fitz; ..."` 直接抓文字。
