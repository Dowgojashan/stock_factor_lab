# stock_factor_lab 環境安裝說明（Python 3.10 / venv）

本說明搭配自動安裝腳本 `setup.bat` 使用。腳本能自動完成大部分工作，但有兩項
需要你**先手動安裝**（無法用腳本代裝），裝好後執行 `setup.bat` 即可。

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
   - 裝完**重開機**。沒有這個，第 5 步 Cython 編譯會失敗（其餘套件仍會裝好）。

3. **MySQL 資料庫**：你已用 xampp 建好並匯入 SQL，這部分免處理。
   - `config.ini` 若你的 xampp 設定不同，請修改 `config.ini`。

---

## B. setup.bat 使用說明

`setup.bat` 是一鍵安裝腳本，把建立虛擬環境到編譯回測核心的流程串起來，省去逐項手動安裝。

### B.1 setup.bat 做什麼

腳本會依序執行下列五步，任何一步失敗就停下來並印出原因：

0. 檢查 `py -3.10` 是否存在，沒有就中止並提示先裝 Python 3.10。
1. 用 Python 3.10 建立專案根目錄下的 `.venv`（已存在則略過）。
2. 啟用該環境並升級 pip。
3. 安裝 `requirements_clean.txt`（已鎖定 numpy 1.24.4）。
4. 安裝 `TA-LIB\ta_lib-0.6.3-cp310-cp310-win_amd64.whl`（以 `--no-deps` 安裝）。
5. 進入 `core` 編譯 `backtest_core`，產生 `.pyd`。缺 C++ Build Tools 時只有這步會失敗，其餘套件仍裝好。

### B.2 執行前需要什麼

先完成 A 段的前置需求，至少要有 **Python 3.10（含 py launcher）**；要編譯回測核心則需 **C++ Build Tools**。MySQL 視需要而定。

### B.3 怎麼執行

**在專案根目錄**（含 setup.bat 的資料夾）對 `setup.bat` 按右鍵→「以系統管理員身分執行」，或在 cmd 執行：

```
cd /d C:\Lab\交接\stock_factor_lab
setup.bat
```

腳本結尾會 `pause`，看完訊息按任意鍵關閉。

### B.4 執行後會產生什麼

- 專案根目錄下的 `.venv\`：虛擬環境，之後用 `.venv\Scripts\activate` 啟用。
- 環境內裝好 `requirements_clean.txt` 列出的套件與 TA-Lib。
- `core\backtest_core` 的編譯產物（`core\backtest_core*.pyd`），第 5 步成功時才有。

---

## C. 驗證安裝是否成功

> ⚠️ 2026-09-09 更新：原本這裡是「跑 `code/example.ipynb`」，但這個範例檔已在
> 整理程式碼時刪除（確認沒有任何地方引用它）。現在最準確的驗證方式是直接跑
> 研究部的自檢測試套件——裝好後應該全數通過。

1. 啟動環境：
   ```
   .venv\Scripts\activate
   ```
2. **切到 `code/` 目錄**（`config.ini` 用相對路徑讀取，一定要在 `code/` 下執行）：
   ```
   cd code
   ```
3. 跑測試套件（Windows 預設 cp950 編碼會讓中文輸出報錯，要加 `PYTHONIOENCODING=utf-8`）：
   ```
   set PYTHONIOENCODING=utf-8
   python -m research.tests
   ```
   結尾應顯示 `131/131 通過`（實際項數可能隨開發增加）。若有測試失敗，通常是
   資料庫連線、`config.ini`、或 `_frozen/` 資料沒帶齊，對照下方 E 段排解。

---

## D. 我已經幫你改好的部分

- `requirements_clean.txt`：原本的 `requirements.txt` 是 UTF-16 編碼，且 TA-Lib 那行
  寫死成別人的路徑 `C:/iplab/...`。已轉成 UTF-8 乾淨版，並移除 TA-Lib 行（改由 whl 單獨裝）。
- `backtest.py` 的 `from core.backtest_core import mae_mfe` 已是正確寫法，無需再改。
- 程式碼中沒有殘留 `iplab` 的 import，無需清理。

---

## E. 常見錯誤排解

| 症狀 | 原因 / 解法 |
|------|------|
| `error: Microsoft Visual C++ 14.0 or greater is required` | 缺 C++ Build Tools（前置 A-2），裝完重開機再跑 setup.bat |
| `ModuleNotFoundError: No module named 'database'`（自己寫的 ad-hoc 腳本）| `database.py`／`get_data.py`／`combinations.py` 等在**根目錄**，自己寫的腳本要先 `import fcv_core`（哪怕用不到它，它會把根目錄加進 `sys.path`），或自己手動加 |
| `ImportError: ... numpy.dtype size changed` | numpy 版本衝突。重裝：`pip install --force-reinstall numpy==1.24.4` |
| 連不上資料庫 | 確認 xampp 的 MySQL 已啟動，且 `config.ini` 的 host/port/db/user/password 正確 |
| `py -3.10` 找不到 | Python 3.10 沒裝或沒勾 py launcher。重裝 Python 3.10 |
