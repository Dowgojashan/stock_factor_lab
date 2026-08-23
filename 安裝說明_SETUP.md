# stock_factor_lab 安裝與執行說明（Python 3.10 / venv）

## 這個資料夾是什麼

量化因子選股策略回測系統，核心是 Phase 1-4 SOP（因子健檢→F1×F2配對→加動態條件C→加估值濾網V）。
隨附的 `lab_full_dump.sql.gz` 是完整資料庫的壓縮匯出檔，裝好環境、匯入這個檔案後，
`code/` 底下的程式就能直接對真實資料跑。

本說明搭配自動安裝腳本 `setup.bat` 使用。腳本能自動完成大部分環境安裝，但**資料庫**
與**兩項前置需求**需要你先手動處理。

---

## A. 前置需求（執行 setup.bat 前先裝好）

1. **Python 3.10**
   - 安裝時務必勾選 **「Add python.exe to PATH」** 與 **「py launcher」**。
   - 驗證：開 cmd 輸入 `py -3.10 --version` 應顯示 3.10.x。

2. **Microsoft C++ Build Tools**（編譯 `core` 回測核心用）
   - 下載：https://visualstudio.microsoft.com/zh-hant/visual-cpp-build-tools/
   - 安裝時勾選：
     1. MSVC v143 - VS 2022 C++ x64/x86 建置工具
     2. Windows 10/11 SDK（視系統版本）
     3. 適用於 Windows 的 C++ CMake 工具
   - 裝完**重開機**。沒有這個，setup.bat 最後一步（編譯Cython）會失敗（其餘套件仍會裝好，只是回測核心跑不動）。

3. **MySQL**（XAMPP 最簡單）
   - 下載安裝 XAMPP：https://www.apachefriends.org/
   - 啟動 XAMPP 控制台，把 **MySQL** 服務啟動（Apache不需要）。
   - 詳細匯入步驟見下方「B. 建立資料庫」。

---

## B. 建立資料庫

1. 啟動 MySQL（XAMPP控制台點 MySQL 的 Start）。
2. 用命令列建立一個空的 `lab` 資料庫（帳密依你的MySQL設定，XAMPP預設帳號`root`、密碼空白）：
   ```
   "C:\xampp\mysql\bin\mysql.exe" -u root -e "CREATE DATABASE lab CHARACTER SET utf8mb4"
   ```
3. 匯入隨附的 `lab_full_dump.sql.gz`（壓縮檔約953MB，解壓後約7.9GB，匯入需要幾分鐘到十幾分鐘，視電腦效能）：
   - 如果有裝 Git Bash 或 WSL（有 gzip 指令）：
     ```
     gzip -dc lab_full_dump.sql.gz | "C:\xampp\mysql\bin\mysql.exe" -u root lab
     ```
   - 如果沒有，先用 7-Zip 之類的工具把 `lab_full_dump.sql.gz` 解壓成 `lab_full_dump.sql`，再匯入：
     ```
     "C:\xampp\mysql\bin\mysql.exe" -u root lab < lab_full_dump.sql
     ```
4. 驗證匯入成功（應該看到12張表，`stock`表約3,900萬列）：
   ```
   "C:\xampp\mysql\bin\mysql.exe" -u root lab -e "SHOW TABLES; SELECT COUNT(*) FROM stock;"
   ```

---

## C. setup.bat 安裝Python環境

### C.1 setup.bat 做什麼

腳本依序執行：
0. 檢查 `py -3.10` 是否存在，沒有就中止並提示先裝 Python 3.10。
1. 用 Python 3.10 建立專案根目錄下的 `.venv`（已存在則略過）。
2. 啟用該環境並升級 pip。
3. 安裝 `requirements_clean.txt`（已鎖定 numpy 1.24.4）。
4. 安裝 `TA-LIB\ta_lib-0.6.3-cp310-cp310-win_amd64.whl`（以 `--no-deps` 安裝）。
5. 進入 `core` 編譯 `backtest_core`，產生 `.pyd`。缺 C++ Build Tools 時只有這步會失敗，其餘套件仍裝好。

### C.2 怎麼執行

**在專案根目錄**（含 setup.bat 的資料夾）對 `setup.bat` 按右鍵→「以系統管理員身分執行」，或在 cmd 執行：
```
cd /d C:\你放這個資料夾的路徑\stock_factor_lab
setup.bat
```
腳本結尾會 `pause`，看完訊息按任意鍵關閉。

### C.3 設定資料庫連線

```
copy config.ini.example config.ini
```
用文字編輯器打開 `config.ini`，依你的 MySQL 設定填入（XAMPP預設：host=127.0.0.1、port=3306、db=lab、user=root、password留空）。

---

## D. 驗證安裝：跑 example.ipynb

⚠️ **這是舊版兩因子交叉分析的展示（`factor_analysis_two_factor_AA/AND`），用來確認環境跟資料庫連線正常，不是論文正式使用的Phase 1-4 SOP方法論**——正式方法論的跑法見下方「E」。

1. 啟動環境：`.venv\Scripts\activate`
2. 用 VS Code 開 `code\example.ipynb`，右上角 **Kernel** 選 `.venv` 的 Python。
3. 從第一個 cell 開始依序執行，能正常跑出策略回測曲線就代表環境+資料庫都通了。

---

## E. 跑正式的 Phase 1-4 SOP

四個階段的腳本都在 `code/`，都有 `--help` 可查參數，也都支援 `--dry-run`（先看預期跑幾個策略，不真的執行）：

```
cd code
.venv\Scripts\activate     （若尚未啟用環境）

REM Phase 1：9桶單因子線性檢定（先dry-run看看）
python phase1_linearity.py --market TW --dry-run
python phase1_linearity.py --market TW          （拿掉--dry-run才會真的跑）
python phase1_analyze.py --market TW             （分析Phase1結果）

REM Phase 2：F1×F2 配對
python phase2_pairing.py --market TW
python phase2_analyze.py --market TW

REM Phase 3：加動態條件C
python phase3_conditions.py --market TW
python phase3_analyze.py --market TW

REM Phase 4：加估值濾網V
python phase4_valuation.py --market TW
python phase4_analyze.py --market TW
```
美股把 `--market TW` 換成 `--market US`。

以上所有指令都用 `--variant openSec` 這個變體（primary因子嚴格篩、secondary全部開放），是正式採用的設計，其餘變體（strict/relaxed/all）不用理會。

⚠️ **全因子全市場真的跑下去，耗時可能是「一整夜」等級**（台美兩市場、幾千個策略組合）。建議第一次先用 `--dry-run` 確認流程沒問題再往下跑。

### E.2 更簡單的跑法：直接用 notebook

`code/因子選股完整流程.ipynb` 把上面「取得資料 → Phase1 → Phase2 → Phase3 → Phase4」整段流程、
加上每階段的分析結果（表格+圖）跟論文第四章18張圖鑑，全部串成一份notebook，
由上而下依序執行每個cell即可，不用照著上面的指令一條條在命令列打。建議直接用這份notebook，
上面的CLI指令列表當作對照/備查即可。

### E.1 關於「參考結果」資料夾——會不會被覆蓋？

跟這份程式一起交付的還有一個**`參考結果（作者已執行版本）/`資料夾**，是作者已經完整跑過一次、產出的Phase1-4分析結果（圖表+數據），**故意放在跟`stock_factor_lab/`平行的位置，不是放在程式碼資料夾裡面**。

原因：`phase1_analyze.py`等分析腳本執行時，輸出路徑是寫死的固定路徑（例如`_analysis_outputs_phase1/`），**會直接覆蓋掉同路徑下已有的檔案，不會詢問、也不會另存新檔**。如果把參考結果放進`stock_factor_lab/`資料夾裡，你一執行分析腳本，參考結果就會被自己剛跑出來的結果蓋掉，之後想比對兩邊有沒有差異就再也比不了了。

**所以請維持兩個資料夾分開放，不要把「參考結果」複製進`stock_factor_lab/`裡面**：
- 你自己執行後的新結果，會出現在 `stock_factor_lab/_analysis_outputs_phase{1,2,3,4}/` 等路徑（跑完才會生成，一開始沒有）。
- 「參考結果」資料夾是作者的版本，不受你執行的影響，可以直接拿來跟你自己跑出來的對照。
- 理論上兩邊的數字應該完全一致（同一份程式+同一份資料庫，Phase1-4是確定性流程，沒有隨機性），對不起來才需要深入查。

---

## F. 常見錯誤排解

| 症狀 | 原因 / 解法 |
|------|------|
| `error: Microsoft Visual C++ 14.0 or greater is required` | 缺 C++ Build Tools（前置 A-2），裝完重開機再跑 setup.bat |
| `ModuleNotFoundError: get_data` | 執行腳本時的工作目錄不對，`code/`底下的程式需要能找到上層的`get_data`等模組，`fcv_core.py`已自動處理這個路徑，但若自己寫程式import要注意 |
| `ImportError: ... numpy.dtype size changed` | numpy 版本衝突。重裝：`pip install --force-reinstall numpy==1.24.4` |
| 連不上資料庫 | 確認 MySQL 已啟動，且 `config.ini` 的 host/port/db/user/password 正確 |
| `py -3.10` 找不到 | Python 3.10 沒裝或沒勾 py launcher。重裝 Python 3.10 |
| 匯入資料庫很久沒反應 | 正常，7.9GB解壓後的SQL檔匯入本來就要幾分鐘到十幾分鐘，看得到CPU/磁碟在動就是還在跑 |
