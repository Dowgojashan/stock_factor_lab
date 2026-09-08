# code/ 程式碼地圖

> 建立於 2026-09-08，純查閱用的分類索引，**不影響任何 import 路徑**（沒有搬動
> 任何檔案）。內容依實際讀過的 docstring 分類，`code/research/` 58 個模組逐一
> 核對過；`code/` 頂層 43 個 Phase 1-4 引擎腳本是照檔名規律 + `CLAUDE.md` 既有
> 記載分類，沒有逐一重讀。
>
> 有新模組加進來記得回來補一行，不然這份文件自己也會變成過時文件。

---

## 一、`research/`——研究部主線（論文主要在用這條）

### 1. 基礎設施
不對應任何研究問題，是所有模組共用的底層機制。

| 檔案 | 功能 |
|---|---|
| `paths.py` | 路徑與 `sys.path` 掛載——所有模組的共同前置 |
| `contracts.py` | 資料契約（schema 定義＋驗證器），單一事實來源 |
| `freeze.py` | 凍結／版本／指紋機制（DD-08），管每個階段的 manifest |
| `cli.py` | 統一指令入口，`python -m research.cli list/stage0/...` |
| `tests.py` | 全套自檢測試（131 項，不依賴 pytest）|
| `__init__.py` | 套件初始化 |

### 2. 主線管線 stage0→4
候選池 → 標記 → 總經 → HRP 分群 → 彙整，這是整條研究的骨幹。

| 檔案 | 對應階段 |
|---|---|
| `stage0_index.py` | 階段0：候選池格式整備 |
| `stage1_scan.py` | 階段1a：單趟掃描 |
| `stage1_marks.py` | 階段1b：標記關卡 A/B/C |
| `stage1_mktcap.py` | 階段1前置：市值分位表 |
| `stage2a_regime.py` | 階段2a：牛熊時期切割 |
| `stage2b_macro.py` | 階段2b：總經月頻特徵表 |
| `stage2c_consistency.py` | 階段2c：2a×2b 交叉佐證 |
| `stage3_hrp.py` | 階段3：HRP 階層聚類（主線六棵樹）|
| `stage3_hrp_isoos.py` | H-11：IS/OOS 分群穩健性驗證（獨立分支，不動主線）|
| `stage4_strategy_map.py` | 階段4：彙整 `strategy_map` |
| `macro_spec.py` | 附屬於階段2b：總經指標規格與發布滯後 |
| `validate_macro_raw.py` | 附屬於階段2b：總經原始資料自我檢查（給 collector 交付前跑）|

### 3. HRP／分群解釋層
分群之後「這群是什麼、怎麼挑代表、怎麼講給人聽」。

| 檔案 | 對應題號 |
|---|---|
| `hrp.py` | HRP 演算法本體（純數學，跟資料無關）|
| `cluster_count_selection.py` | H-03：群數用輪廓係數決定 |
| `cluster_representatives.py` | H-10：群內代表挑選規則 |
| `cluster_temporal_profile.py` | H-06：群的定量特徵表 |
| `cluster_visualizations.py`／`_l3.py` | H-07／H-25c：群的視覺化（L1／L3）|
| `cluster_story.py` | 群間互補的 LLM 解釋 |
| `cluster_identity.py` | H-08：單群身份的 LLM 解釋 |
| `cluster_macro_interface.py` | S-01：群→總經決策層介面（方向C，已終止）|
| `effective_bets.py` | H-09：有效獨立賭注數 ENB |
| `complementarity_granularity.py`／`_sensitivity.py` | H-25：互補性的粒度效應／門檻敏感度 |
| `k_stability.py` | H-26b：群數 k 的前視偏誤診斷 |
| `l3_isoos.py` | H-25d：L3 細粒度互補性的 walk-forward 驗證 |

### 4. 決策層 S 系列 ⚠️ 方向 C 已終止，保留但不是主線
老師 9-2 已裁定「總經→選群」這條路終止，這四個是那個方向的產物。

| 檔案 | 對應題號 |
|---|---|
| `macro_decision_input.py` | S-02：決策層資訊源設計 |
| `decision_layer_arms.py` | S-05：決策層對照組設計 |
| `decision_repeatability.py` | S-07：決策層重複執行穩定度 |
| `macro_rolling_window.py` | H-18②：總經訊號滾動窗版本（唯一被保留當方法論貢獻的）|

### 5. 對照實驗與產出

| 檔案 | 功能 |
|---|---|
| `four_group_control.py` | H-12：四組對照（A_hrp/B_all/C_random/D_top_cagr）|
| `output_a.py` | 產出A：20年情境×策略表現回顧表 |

### 6. H-26／H-27 Walk-forward 主線 —— 這次論文主要在用的部分

| 檔案 | 功能 |
|---|---|
| `walkforward_matrix.py` | 主矩陣：13窗口×5比例×2分配×2群數來源 |
| `walkforward_significance.py` | H-26c：統計檢定 |
| `walkforward_evidence.py` | M-10：證據強度（效果分布/多重比較/窗重疊）|
| `walkforward_random.py` | M-02b：C_random 對照 |
| `walkforward_partition.py` | M-03b：A2_random 對照 |
| `window_robustness.py` | H-26d：共同窗穩健性 |
| `mdd_window_length.py` | M-09：MDD 顯著性按窗長分層 |
| `turnover_cost.py` | H-27b：周轉率與交易成本 |
| `figures_h26_h27.py` | 圖 A~H（勝率熱力圖、超額分布、比例掃描、IS/OOS散佈、時間軸、比例×分配表格、群數時間軸）|

### 7. M 系列方法論稽核 —— 補方法論漏洞用

| 檔案 | 對應 M 編號 |
|---|---|
| `beta_baseline.py` | M-01：市場 beta 基準 |
| `partition_control.py` | M-03：分群依據對照 |
| `enb_null.py` | M-05：ENB 虛無分布 |
| `subject_comparison.py` | M-11：換掉檢定主體 |
| `rank_persistence.py` | M-12：rank IC |
| `threshold_control.py` | M-15：多樣性門檻 |
| `enb_rank_deficiency.py` | M-14：ENB 反轉 vs 秩不足 |
| `market_benchmark.py` | M-17：真正的市場基準 |
| `residual_tree_selection.py` | M-13b：殘差樹選兵 |

### 8. 其他

| 檔案 | 功能 |
|---|---|
| `diagnose_price_anomalies.py` | 資料異常診斷（W-08）|
| `universe_history.py` | 補件：上市公司家數隨時間變化圖（回應老師 9-8 提問）|

---

## 二、`ops/`

| 檔案 | 功能 |
|---|---|
| `tools.py` | T1~T13 工具層——已廢止的實戰部三 Agent 架構留下的，仍在用（`output_a.py` 依賴其中 5 個）|

---

## 三、`code/` 頂層——Phase 1-4 引擎（已凍結完成、不會再跑，但保留供論文方法論交代）

⚠️ 這組是照檔名規律 + `CLAUDE.md` 既有記載分類，沒有逐一重讀 docstring 核對。

| 分類 | 檔案 |
|---|---|
| **核心回測引擎** | `fcv_core.py`／`fcv_us.py`／`condition_factory.py`／`io_persistence.py` |
| **Sweep 驅動** | `sweep_config.py`／`sweep_driver.py`／`sweep_supervisor.py`／`spec_generator.py`／`catalog_builder.py` |
| **Phase1** | `phase1_analyze.py`／`phase1_cleanperiod_check.py`／`phase1_linearity.py`／`phase1_subperiod_check.py` |
| **Phase2** | `phase2_analyze.py`／`phase2_pairing.py` |
| **Phase3** | `phase3_analyze.py`／`phase3_conditions.py`／`phase3_fig416.py` |
| **Phase4** | `phase4_analyze.py`／`phase4_valuation.py` |
| **變體對照** | `phase_variants.py`／`variant_compare.py` |
| **批次分析（11因子候選批次）** | `analyze_batch.py`／`analyze_job.py`／`analyze_spec_us.py`／`run_factor_batches.py` |
| **報告/圖鑑產出** | `build_atlas.py`／`build_comparison_report.py`／`report_grouping.py` |
| **穩健性檢定** | `robustness_checks.py`／`size_control_analysis.py`／`size_control_backtest.py`／`c_correlation.py` |
| **台股批次專用報告** | `tw_batch_sanity_report.py`／`tw_champion_concentration.py`／`tw_cross_batch_report.py`／`tw_report_corrections.py` |
| **基準/其他工具** | `universe_benchmark.py`／`check_surv.py`／`scan_stock_contributions.py`／`stats_test.py` |
| **一次性 json 設定** | `spec_TW.json`／`spec_US.json`／`fcv_experiment_spec.json`（給 sweep 引擎讀的規格檔）|

---

## 四、根目錄（不在 `code/` 裡，但跟 `code/` 互相依賴）

⚠️ 這 7 個確認在用、**不建議搬進 `code/`**——`code/fcv_core.py` 的 `sys.path`
設定寫死假設它們在 ROOT（見該檔第 32 行註解），搬動要牽動整條 import 鏈。

| 檔案 | 角色 |
|---|---|
| `database.py` | 資料庫連線層 |
| `get_data.py` | 資料抓取 |
| `combinations.py` | 因子組合產生 |
| `format_data.py` | 資料格式整理 |
| `dataframe.py` | DataFrame 工具 |
| `backtest.py` | 回測引擎（舊版，`fcv_core.py` 是新版）|
| `report.py` | 報表物件（`Report` class，`io_persistence.py` pickle 還原要用）|

---

## 五、`utils/`

| 檔案 | 功能 |
|---|---|
| `config.py` | 讀 `config.ini`（相對路徑 `../config.ini`，腳本要從 `code/` 執行）|
| `openai_quota.py` | LLM API 額度偵測（供 `cluster_story.py`／`cluster_identity.py` 用）|

---

## 六、`core/`

編譯過的 Cython 回測核心（`backtest_core.cp310-win_amd64.pyd`），底層依賴，不要動。
