# 交接說明 — stock_factor_lab

## 專案目的

這是一套台股選股策略的批次回測與分析框架。核心想法是把選股條件拆成三個構面：F（體質）看公司的基本財務狀況，例如 ROE、EPS、負債比落在哪個區間；C（動態）看財務數字隨時間的變化型態，例如近四季是否創高、是否較去年同季成長；V（估值）用本益比門檻控制進場時的貴賤。把這三個構面的條件互相組合，會展開出數百到上千組策略，框架對每一組都跑一次回測，再把結果彙整、比較，找出哪些條件組合表現好且穩健。整個流程的設定集中在一個 JSON 檔，跑完的結果存成 pickle 與 parquet 兩種格式供後續分析。

## 快速開始

1. **建環境**：在專案根目錄執行 `setup.bat`。它會用 Python 3.10 建立 `.venv`、裝好 `requirements_clean.txt` 的套件與 TA-Lib，並編譯回測核心 `core/backtest_core`。執行前需先自行安裝 Python 3.10（勾選 py launcher）與 Microsoft C++ Build Tools，細節見 `安裝說明_SETUP.md`。

2. **設定資料源**：本專案的行情與財報資料來自本機 MySQL（以 xampp 架設）。連線資訊寫在根目錄的 `config.ini`（資料庫名、帳號、密碼）。換機器時要改這裡，並確認 MySQL 已啟動、SQL 已匯入。

3. **跑回測**：用 VS Code 開 `code/fcv_backtest.ipynb`，kernel 選 `.venv` 的 Python。要調整因子條件就改 `code/fcv_experiment_spec.json`，然後由上往下逐格執行。跑完會在 `code/results_pickle/` 與 `code/results_artifacts/` 產出結果，並在 `code/timing_logs/` 留下計時記錄。一次完整回測偏久（資料載入加上千組策略回測，動輒數十分鐘以上），記憶體也要夠。

4. **看報告**：開 `code/report_analysis.ipynb`，一樣選 `.venv` kernel，由上往下跑完所有 cell。它會讀回測結果，產出 B（整體概覽）、C（體質）、D（動態）、E（估值）四區的圖表，並把圖檔與摘要 CSV 寫到根目錄的 `_analysis_outputs/`。

> 注意：`setup.bat` 與 `安裝說明_SETUP.md` 結尾示範開的是 `example.ipynb`（原始示範檔），實際主流程請開 `fcv_backtest.ipynb` 與 `report_analysis.ipynb`。

## 資料夾結構

```
stock_factor_lab\
├─ code\                     # 主流程 notebook、本次搬入的自製模組、回測結果
│  ├─ fcv_backtest.ipynb     # 主回測
│  ├─ report_analysis.ipynb  # 分析報告
│  ├─ condition_factory.py   # 條件工廠
│  ├─ io_persistence.py      # 結果存讀
│  ├─ report_grouping.py     # 分組、排行榜、年度貢獻等分析工具
│  ├─ stats_test.py          # 統計檢定工具
│  ├─ fcv_experiment_spec.json  # 回測設定檔
│  ├─ results_pickle\        # 回測結果（pickle，整包 report collection）
│  ├─ results_artifacts\     # 回測結果（parquet/json，每策略輕量素材＋stats）
│  └─ timing_logs\           # 各次回測的計時記錄
├─ core\                     # 編譯式回測核心（backtest_core，Windows .pyd）
├─ get_data.py               # 從 MySQL 取行情/財報
├─ backtest.py               # 回測引擎介面
├─ combinations.py           # 把條件組合成策略並回測（sim_conditions、ReportCollection）
├─ report.py                 # 單一策略績效報表物件 Report
├─ database.py / dataframe.py / format_data.py  # 資料層輔助
├─ _analysis_outputs\        # report_analysis 執行後輸出的圖與 CSV
├─ handover_new              # 交接文件
├─ config.ini                # MySQL 連線設定
├─ setup.bat                 # 一鍵環境安裝腳本
├─ 安裝說明_SETUP.md          # 安裝前置需求與 setup.bat 使用說明
└─ requirements_clean.txt    # 套件清單（setup.bat 使用）
```

## 各檔案用途

### 主流程 notebook

**fcv_backtest.ipynb**　主回測程式。讀 `fcv_experiment_spec.json` 取得各因子的條件定義，依序建出 F 構面遮罩（cell 16 `f_factor`，P1 與 P2 組合）、C 構面遮罩（cell 18 `c_factor`，再疊 P3 動態條件）、V 構面（cell 19，用本益比把每組策略展開成 v0／v1 兩版），合起來就是數百到上千組策略。接著做交易次數預篩（cell 23），對通過的策略呼叫 `combinations.sim_conditions` 實際回測（cell 25），再用 `io_persistence.save_all_for_label` 把結果存成 pickle 與 parquet。cell 1 的 `USE_CACHE` 控制是否啟用快取（會影響計時對照），cell 27/28 輸出各階段耗時與條件重複使用統計到 `timing_logs/`。輸入是 JSON 設定與 MySQL 資料，產出是 `results_pickle/` 與 `results_artifacts/` 下以設定檔名（`fcv_experiment_spec`）為 label 的結果。

**report_analysis.ipynb**　分析報告。輸入是上一支跑出來的回測結果：用 `io_persistence.load_all_report_collections` 從 `results_pickle/` 載入完整 report collections（B 區需要 `Report` 物件的圖會用到），並讀 `results_artifacts/fcv_experiment_spec/stats.parquet` 解析出每組策略的 F1/F2/C/V 標籤、算 z-score 綜合分數，建成主表 `df_scored`。整本依 A（環境與載入）、B（整體策略空間與個股集中度）、C（體質構面）、D（動態構面）、E（估值構面）、F（附錄計算）分區，產出盒鬚圖、熱力圖、散布圖等，圖檔與 CSV 輸出到 `_analysis_outputs/`。圖以區段制編號（B.1、C.1、D.1…），對照表見 `out/cleanup_manifest.md` §5。A 區載入 pickle 吃大量記憶體，需在跑回測的同一台機器執行。

### 自製模組

**condition_factory.py**　把 JSON 裡的因子條件轉成回測能用的判斷函式。每種條件型別（大於、區間、近四季最高、較上年同季成長等）對應一個工廠函式，登記在 `CONDITION_FACTORY`；對外主要用 `build_conditions(設定字典)`，回傳 `{name, field, cond}` 清單，並自動命名（例如 ROE 近四季最高會命名為 `ROE_qmax4`）。只有 `fcv_backtest.ipynb` cell 13 會 import 它。

**io_persistence.py**　回測結果的存與讀。`save_all_for_label` 一次把一組結果存成 pickle（整包 report collection）與 parquet（每策略的 trades/position/stock_data/return_table 加上總表 stats）；讀取端有 `load_all_report_collections`（批次載入 pickle）與 `load_artifacts_stats`、`load_strategy_artifacts`（讀 parquet）。路徑與檔名會自動過濾 Windows 非法字元。`fcv_backtest` 用它存、`report_analysis` 用它讀。

**report_grouping.py**　分析報告用的繪圖與彙整工具，import 時別名為 `rg`。提供 Top-K 策略挑選（`TopPickSpec`、`select_top_strategies`）、策略績效熱力圖（`plot_group_stats_heatmap`）、入選股數與排行榜面板（`build_leaderboard_panel`）、年度個股貢獻度（`compute_annual_contribution`、`plot_annual_contribution_stacked`）等。被 `report_analysis.ipynb` 呼叫；分組統計檢定的部分轉呼叫 `stats_test`。

**stats_test.py**　統計檢定與效果量工具。提供群組間檢定（自動在 Welch t／Mann-Whitney／ANOVA／Kruskal 間選擇）、效果量（Cohen's d、Hedges' g、Cliff's delta）與多重比較校正（BH FDR），另有「最近窗口 vs 過去窗口」的時間序列檢定遮罩。由 `report_grouping.py` import 使用。

### 既有核心模組


**get_data.py**　資料層入口，類別 `Data`。`data.get("price:close")`、`data.get("report:roe")` 這類呼叫從 MySQL 取行情與財報，只支援 `price`／`report`／`taiex` 三種來源。`fcv_backtest` 與 `report_analysis` 都用它取資料。

**backtest.py**　回測引擎介面，內部呼叫編譯式核心 `core.backtest_core`（Windows 編譯的 .pyd）。`combinations.py` 會用到。

**combinations.py**　把條件遮罩組成策略並批次回測的上層，提供 `sim_conditions`（回測一批策略）與 `ReportCollection`（一組策略結果的容器）。`fcv_backtest` 直接呼叫；`report_grouping` 依賴它的 `ReportCollection`。

**report.py**　單一策略的績效報表物件 `Report`，提供 `display()`（淨值曲線、回檔、月報酬等複合圖）與 `get_stats()`。pickle 還原 report collection 時需要它可被 import。

### 設定與環境

**fcv_experiment_spec.json**　回測設定檔，決定要跑哪些因子條件。分兩段：`P1` 是體質因子的單層條件（ROE、EPS、FCF_P、DEBTRATIO 各自的區間切法），`P3` 是動態因子條件（ROE_SEQ、REVENUE_GROWTH 等，帶 `C1_`～`C8_` 前綴）。每個條件有 `type`（對應 condition_factory 的條件型別）與 `args`（參數）。檔案沒有 P2 段與 V 段——P2 由回測程式用 P1 的其他欄位條件動態組，V 由本益比另外建。要改回測範圍就改這裡。

**setup.bat**　一鍵環境安裝腳本。依序檢查 Python 3.10、（可選）git lfs、建 `.venv`、升級 pip、裝 `requirements_clean.txt`、裝 TA-Lib whl、編譯回測核心。任一步失敗會停下並印原因；缺 C++ Build Tools 時只有最後的編譯步驟會失敗，其餘套件仍裝好。

**安裝說明_SETUP.md**　搭配 setup.bat 的說明文件。A 段列出要先手動裝的前置（Python 3.10、Git LFS、C++ Build Tools、MySQL），B 段是 setup.bat 的逐步說明，C 段示範開 notebook，E 段是常見錯誤排解表。

## 模組依賴關係

由上而下，箭頭表示「import / 呼叫」：

```
fcv_backtest.ipynb
  ├─ get_data.Data ──> database / dataframe / format_data
  ├─ backtest ──> get_data, core.backtest_core(.pyd)
  ├─ combinations.sim_conditions ──> backtest, report
  ├─ condition_factory.build_conditions
  └─ io_persistence.save_all_for_label ──> report

report_analysis.ipynb
  ├─ io_persistence.load_all_report_collections / load_artifacts_stats ──> report
  ├─ report_grouping (rg) ──> combinations.ReportCollection, stats_test
  └─ get_data.Data（B 區年度貢獻圖需要 price:close）
```

`condition_factory` 與 `stats_test` 本身只依賴 numpy/pandas（stats_test 另用 scipy）。`core.backtest_core` 是 Windows 編譯模組，換平台要重編。

## 常見操作 FAQ

- **想改回測參數**：因子區間、動態條件改 `fcv_experiment_spec.json`；最低交易筆數、resample 頻率、是否啟用快取改 `fcv_backtest.ipynb`（cell 23 的 `MIN_TRADES`/`TOP_K`、cell 25 的 `resample`、cell 1 的 `USE_CACHE`）。
- **想換不同的因子組合**：改 `fcv_experiment_spec.json` 的 P1／P3 內容；新增條件型別才需要動 `condition_factory.py`。
- **想加新的分析圖**：在 `report_analysis.ipynb` 對應區段加 cell，繪圖工具多半已在 `report_grouping.py` 或 A 區的 `plot_*` 函式裡，能沿用就沿用。
- **想跑單一策略的細部分析**：看 `report_analysis.ipynb` B.2／B.3（cell 24、27），用 `Report.display()` 與年度貢獻分解，把對象換成你要的策略名。
- **想匯出結果給別人看**：圖與摘要 CSV 在 `_analysis_outputs/`；原始結果在 `code/results_artifacts/fcv_experiment_spec/`（parquet，較輕量、可直接讀）與 `code/results_pickle/`（pickle，很大）。
