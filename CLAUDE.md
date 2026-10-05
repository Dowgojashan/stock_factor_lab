# 交接文件（給下一個 Claude Code 對話框）

> 這份文件會在每次對話開始時自動載入。原寫於 2026-08-20，**2026-09-01 大幅更新**（下方第 2 節）。
> 使用者是這個研究計畫的主持人（碩士論文相關研究），中文溝通。

---

## 0. 讀這份文件前，先看幾件事

0a. **🔴🔴 2026-10-01 踩過的坑，下次重跑研究部主鏈前必看：`k_stability.csv` 不會自動跟著候選池更新。**
   `research/walkforward_matrix.py` 的 `k_mode="silhouette_is"` 不是現場算 k，是讀
   `_analysis_outputs_robustness/k_stability.csv`（由獨立模組 `research.k_stability` 產生）
   這份**凍結檔**。這次台股加成長因子、重建 stage0~3／`stage3_hrp.L1_TARGET` 之後，
   **忘記重跑 `python -m research.k_stability`**，導致已經跑完一整輪（~5.3小時）的
   `walkforward_matrix` 裡，`k_mode="fixed"` 的結果是對的（直接讀當下 `L1_TARGET`，
   不依賴這份凍結檔），但 `k_mode="silhouette_is"` 的結果全部是用舊池（6,679檔）、
   舊 `L1_TARGET=6` 算出來的 k，實質上没跟上新池——這是**靜默的、不會報錯**的資料
   不同步，用 `walkforward_members.parquet` 選策略名單時完全看不出來哪裡不對，
   只有去對比「現場重算」跟「凍結檔」的代表策略數才會發現。
   **教訓／規則（往後每次重跑研究部主鏈都要檢查）**：只要改了
   `stage3_hrp.L1_TARGET` 或重建了候選池（stage0~3 任一環），**`research.k_stability`
   也要跟著重跑**，然後 `research.walkforward_matrix` 才能重跑；順序顛倒或漏掉
   `k_stability` 這一步，`silhouette_is` 的結果會悄悄過期但完全不會報錯、也不會被
   任何契約檢查抓到（`walkforward_matrix.py` 只驗證 `k_stability.csv` 涵蓋了所需的
   IS 窗，不驗證它是不是用同一批候選池算的）。
   **舊檔案備份原則**：發現這類「重跑會覆蓋掉舊版本」的情況，覆蓋前一律先把舊檔案
   複製一份留存（這次的作法：`_analysis_outputs_robustness/_舊版備份_20260930_加成長因子前/`
   存了舊版 `k_stability.csv`／`walkforward_matrix_detail.csv`／`walkforward_members.parquet`
   等 8 個檔案，從 git HEAD 復原），不要只靠 git history——使用者明確要求「舊的檔案
   紀錄也要留著，不要輕易被覆蓋掉」，這是長期原則不是這次特例。
   🔴 **這次修正留下的已知缺口（XM 退步，刻意不修）**：使用者只要求修 TW（「我們主要
   只是要看TW有沒有改善」），US/XM 延用「舊值」——但備份用的是 git HEAD（9/6
   commit，加成長因子**之前**的版本），不是這次 session 稍早 `walkforward_matrix`
   全量重跑（已經套用新 `L1_TARGET={"TW":7,"US":7,"XM":6}`）之後的中繼狀態。結果
   XM 的 `k_mode="fixed"` 從「正確的 L1_TARGET=6」被這次合併**退回成更舊的 3**
   （`t_walkforward_k_mode_uses_is_only_k` 測試已抓到：`n_clusters(3) != L1_TARGET(6)`）。
   US 本身没受影響（候選池從未變動）。下次若要正式處理 US/XM，**不要**沿用
   `_舊版備份_20260930_加成長因子前/` 裡的 XM 資料，要嘛重新對 XM 跑一次完整
   `k_stability`+`walkforward_matrix`（XM 樹最貴，單棵約638s×14窗≈2.5小時起跳），
   要嘛去找有沒有更接近的中繼快照。

0b. **🔴🔴 2026-10-04 使用者明確裁示的長期規則：建 HRP 樹絕對不要用寫死的固定 k，一律先重算再建樹。**
   這次 TW 改用 openSec_boost 變體（候選池 15,009→29,255）後，先用舊的
   `stage3_hrp.L1_TARGET["TW"]=7`（上一輪池子選出來的）把六棵樹全部建完（含 US/XM），
   跑完才發現：拿新建好的 TW linkage 用 `cluster_count_selection.py` 重新掃一次輪廓係數，
   建議值是 **k=6** 不是 7——等於整整一輪 stage3+stage4+k_stability+walkforward_matrix
   的 TW 部分全部要重做（walkforward_matrix 甚至因此直接殺掉重跑，白工約4-5小時）。
   **教訓／規則（往後任何時候候選池或因子池有變動，建樹前一定要做，不要等建完才發現）**：
   1. 先跑 `python -m research.cluster_count_selection --tree {market}_normal`
      （便宜，沿用既有linkage只重算距離矩陣+輪廓係數掃描，數十秒~數分鐘量級，
      不是重新建樹）拿到建議的 k。
   2. 確認跟 `stage3_hrp.L1_TARGET[market]` 現在的值是否相符，不符就先改掉常數。
   3. 改完常數才去跑 `stage3_hrp`／`stage4_strategy_map`／`k_stability`／
      `walkforward_matrix` 這條鏈，不要用舊k建完再補救。
   同一次也學到：改某一個市場的候選池時，`stage3_hrp.run()`／`k_stability.run()`／
   `walkforward_matrix.run()` 都支援 `trees=[...]`／`--trees` 只跑指定市場，但
   `stage3_hrp.run()` 的輸出檔（`cluster_assign.parquet`/`cluster_meta.parquet`/
   `co_fail_regimes.parquet`）是**整批覆寫、不會跟舊資料合併**，只跑單一市場會
   把其他市場的列整個清空——要保留其他市場不動，必須自己寫混合重建腳本
   （讀出舊檔案裡其他市場的列、算出這個市場的新結果、merge、再寫回），
   不能直接呼叫 `run(trees=["TW"])`。

0. 🔴🔴 **2026-10-05 查證：`文件/現況銜接.md` 已不存在**（git history 顯示在某次「123」
   commit裡被刪，不是這次對話刪的，刪除原因未知）。**目前實際扮演「現況快照」角色的是
   `文件/實戰開發追蹤_v2.md` §0**（現況快照＋狀態表＋已決定不做的事），**改讀那份**，
   不要再找 `現況銜接.md`。本 CLAUDE.md 講的是**專案怎麼運作**（長期不太變的事），
   `實戰開發追蹤_v2.md` §0 講的是**現在的狀態**（會一直變）。兩者衝突時以日期新的為準。
1. **🔴 `文件/研究框架總覽_v10.md` —— 現行唯一的完整框架文件，接手後第一件事就讀它。**
   裡面有完整的實驗框架、故事線、五章骨架、每階段的實際結果數字、已知限制、尚未完成的項目。
   本 CLAUDE.md 只講「跟你協作有關的注意事項」，研究內容一律以 v10 為準。
2. **auto memory 索引**（`MEMORY.md`）—— 使用者的長期記憶，應該會自動載入，裡面有更細的專案脈絡跟使用者的溝通偏好，這份文件不重複那邊已經有的內容。
   ⚠️ **2026-09-10 更正**：舊版這裡寫死 `C:\Users\iplab\.claude\projects\d--git-stock-factor-lab\memory\`，那是**舊機器**的路徑，換機器後已失效。新機器的實際路徑見對話開頭自動載入的內容，**不要照抄舊路徑去找**。
3. **`文件/開發待辦追蹤.md`** —— 研究部每一項開發的歷程、bug 修正史、待辦狀態。跟 v10 分工：v10 講「框架與結果」，這份講「怎麼走到這裡、還有什麼沒做」。
4. **`文件/應用層開發追蹤.md`** —— 應用層（`code/app/`）的架構、20 項決定事項、分期狀態。研究部與應用層是兩條線，追蹤文件也分開。

### 文件現況（2026-09-09 全面查證，所有 .md 都實際確認過位置）

**所有專案文件都在 `文件/`**，根目錄只留 `CLAUDE.md`／`README.md`／`安裝說明_SETUP.md`。

| 文件 | 狀態 |
|---|---|
| `文件/現況銜接.md` | 🔴 **新對話框最先讀這份**（現況快照＋下一步＋已決定不做的事＋踩過的坑）|
| `文件/研究框架總覽_v10.md` | ✅ **現行**，研究內容唯一權威 |
| `文件/開發待辦追蹤.md` | ✅ 現行（研究部歷程／待辦；標頭寫「最後更新 2026-09-03」但內容其實含 H-26~H-28、M 系列） |
| `文件/應用層開發追蹤.md` | ✅ 現行（應用層 Phase A~D 與決定事項）|
| `文件/換機器遷移清單_2026-09-09.md` | 🔶 上一趟換機器（重灌）用的，留存備查 |
| `文件/換機器遷移清單_2026-09-13.md` | ✅ **現行**（🔴這趟換機器要看這份——新增 `audit_log.jsonl` 手動備份、6個未commit檔案）|
| `文件/老師9-8意見待報告事項_下次會議9-15.md` | ✅ 現行（2026-09-14 更正：原檔名一度誤植成「10-01」，實際下次會議是 **9/15**——2026-09-09 那次改名判斷錯誤，這次改回正確日期）|
| `文件/研究進度報告_2026-09-01.md`／`_2026-09-07.md` | 🔶 歷史存查（各自對應當時報給老師的內容，不是重複檔）|
| `文件/研究部完整流程_v9.md` | 🔶 已被 v10 取代，降級為歷史紀錄（保留設計理由論證），衝突以 v10 為準 |
| `文件/因子候選批次_F與C因子定義.md` | ✅ 現行（F/C 因子定義規格，引用的三支程式都還在）|
| `文件/系統投資政策聲明_IPS.md` | ✅ 現行（2026-09-10 新增，應用層的投資政策聲明；`RunConfig` 是「政策邊界內的一組執行參數」，不是政策本身）|

⚠️ **2026-09-09 查證：以下檔案已不存在**（舊版 CLAUDE.md 還在指這些檔，已移除那些指路）：
`實戰部架構_v8.md`（架構已廢止，但工具層 `code/ops/tools.py` T1~T13 **仍在使用**，`output_a.py` 依賴其中 5 個，**不可刪除**）、
`GateC_標記清單_v1.md`、`系統設計文件_v1.md`、`落差處理方案_v1.md`、
`策略成果報告_2026-08-17.md`、`週進度報告_2026-08-17.md`、`重跑計畫_老師方法論SOP.md`、
`8-5咪挺.pdf`、`8-19咪挺.pdf`（目前只剩根目錄的 `9-8咪挺.pdf`）。

---

### 可用的 skills（`.claude/skills/`）

`meeting-digest`（開會後整理逐字稿）／`meeting-prep`（開會前準備報告）／
`meeting-followup`（開會前稽核漏辦與文件同步）／`methodology-review`（方法論審視，
附本專案實際踩過的陷阱清單）／`decision-log`（決策留痕）。
用 `/<名稱>` 呼叫，說明見 `文件/現況銜接.md` §7。

---

## 1. 這個專案在做什麼

`stock_factor_lab`：量化因子選股回測系統，碩士論文相關研究。參考學姊余姵穎的論文《宣告式多因子量化回測系統之設計與實作》（不在 repo 內，位置見第 7 節）。指導教授會定期開會給方向，逐字稿目前 repo 內只剩根目錄的 `9-8咪挺.pdf`（老師談話內容，**不對外公開**，`.gitignore` 已排除 `*咪挺*`／`*逐字稿*`，所以 **git 帶不走，換機器要手動複製**）。

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

## 2. 目前進度（2026-09-01）

> 🔴 **完整的框架、故事線與全部結果數字，一律看 `研究框架總覽_v10.md`，這裡只給極簡摘要。**

**進度定位**：Phase 1~4（階段 −1）早已完成並經老師核可；之後的研究部主鏈（階段 0→1→2a/2b/2c→3→4→產出A）
以及 H 系列（H-01~H-28）、S 系列（S-01~S-08，S-06 暫緩）、M 系列**全部完成**。
測試套件在舊機器是 **131/131 通過**（⚠️ 資料庫沒開時會是 130/131，掛掉的 `t_ops_t11_regime_label_valid` 是唯一需要即時 DB 的測試，不是程式壞了）。

🔴 **2026-09-10 新機器實測是 129/131**（資料庫有開）。**不是程式壞了，是換機器沒帶 `results_artifacts/`**：
- `t_stage0_artifacts_exist` — 讀凍結 `candidate_index.parquet` 的 `artifacts_dir` 欄，
  裡面 **100% 是舊機器的絕對路徑**（`D:\git\stock_factor_lab\code\results_artifacts\...`），
  新機器上當然不存在 ⇒ 這是**凍結資料存了絕對路徑**的可攜性缺陷，不是這次改壞的
- `t_stage0_idempotent` — 重跑 `stage0_index.py`，它用 `paths.artifacts_path()` 算出新機器路徑，
  但本機 `code/results_artifacts/` 只有 3 個 spec 目錄（**189GB 的逐策略回測產物沒複製過來**）
- **影響範圍**：僅限「重跑 stage0」。stage0 的產物已凍結，下游全部讀 `_frozen/`，
  **應用層完全不受影響**（實測：`app.cli`／`app.clustering` 全部正常，G4 驗收 14/14 通過）

**應用層（`code/app/`，2026-09-08~10）**：Phase A~D 完成，`replay` 模式可跑（CLI + Streamlit UI）；
2026-09-10 另完成 §6 法人治理五項改動（累積 diff／T8 三組數字／多方法對照／IPS 文件／快慢時鐘拆分）。
🔴 **2026-09-10 兩件事要知道**：①**方法論審視抓到 4 個 🔴 問題**（`應用層開發追蹤.md` §8），
最重要的是**候選池全樣本前視偏誤 ⇒ OOS 絕對數字系統性高估**（已補進 v10 §8，建議 9/15 報告）
②**§7 重新設計討論中**（engine 從查表換成即時建樹，G1~G6 已拍板，**尚未動工**）。
`live` 模式仍未實作，但**暫緩理由已更正**——不是「資料庫連不上」也不是「要補報酬到今天」，
純粹是即時建樹那條程式路徑還沒寫。詳見 `文件/應用層開發追蹤.md` §7/§8。

**⚠️ 論文五章骨架已變動**：舊版寫的「④總經→選群應用」**已終止**——老師 2026-09-02 當場裁示方向 C 不做
（「跟總經沒關係⋯不會有答案」）。現在建議把 H-28（機制解釋，已完成）擴充成第四章，
但**這件事還要 9/15 開會跟老師確認**，不是已定案。詳見 v10 §10。

**還沒完成的**：🔴 2026-09-09 訂正——上一版這裡寫的「H-18③／H-20／H-21 還沒完成」是舊資訊，
v10 §10 已在 2026-09-07 訂正過（那次訂正沒有回頭同步進這份 CLAUDE.md，才會一直傳成過時說法）：
H-18③已被 H-26 取代並完成、H-20 因老師 9/2 裁示方向 C（總經→選群）終止而取消、
H-21 維度已改成「walk-forward窗次×精選比例」並隨 H-26/H-27 完成。
**研究部開發工作（含 M 系列收尾）實際已全部完成，測試 131/131 通過**；
真正還沒收尾的是**要跟老師開會拍板的決策**（不是開發任務）：
①第四章定位（H-28 機制解釋章能否取代原總經應用章）②H-22 名詞確認（HRL是否為HRP口誤）
③M 系列推翻原故事線需要當面報告。細節見 v10 §10。

### ⚠️ 幾個容易記錯的數字（已重新查證，舊版 CLAUDE.md 寫錯過）

🔴🔴 **2026-10-05 更新：TW 候選池已換成 openSec_boost 變體，下面「15,810個」「TW 7,128」
「TW k=7」這幾個數字對 TW 已經過期（US/XM 本輪未動，仍是舊值）**——
Phase2 強制納入 ROE/EPS/ROIC/REV_G/MOM_3M 五個 regime-dependent 因子當 primary
（見 `code/phase2_analyze.py` 的 `FORCE_PRIMARY_OVERRIDE`、`code/phase_variants.py` 的
`openSec_boost` 變體），TW 候選池從 15,009 擴增到 **29,255**，HRP 群數從寫死 k=7
改成用 `cluster_count_selection.py` 重算出的 **k=6**（`stage3_hrp.L1_TARGET["TW"]=6`）。
全部研究部主鏈（stage0~4、`k_stability`、`walkforward_matrix`）已針對 TW 完整重跑過。
這次連帶做的實戰層發現（排除V1、W2c上限放寬、強制提早啟動W2c可讓8季累積報酬贏過TAIEX
+70.56%）記錄在 memory `project_openSec_boost_w2c_early_activation_2026-10.md`，
細節不在這裡重複。US（7,128→不動）/XM 本輪完全沒變，下面數字對 US/XM 仍然有效。

- **候選策略池 15,810 個**（TW~~7,128~~ **→29,255（openSec_boost，僅TW）** + US 8,682），
  原採用 openSec 變體，TW 已升級成 openSec_boost。
  ~~舊寫法「台股 7,162、美股 6,916」是錯的~~，正確數字來自 `_frozen/stage0/candidate_index.parquet`
  與 `_analysis_outputs_phase4/{TW,US}_L4_openSec_final_candidates.csv`
- 自建宇宙基準 CAGR：台股 **8.4256%**、美股 **11.0556%**（`contracts.BENCHMARK_CAGR`；
  舊數字 8.67%/12.35% 是 2026-08-22 價格修復前的）
- HRP L1 群數：TW ~~7~~ **→6**（2026-10-05 換成 openSec_boost 候選池後用
  `cluster_count_selection.py` 重算出來的，不是沿用舊值；上一版 TW=7 是更早一輪
  加成長因子後算出的中繼值，見本節最上方🔴🔴區塊）、US **7**、XM ~~3~~
  **→6**（🔴 2026-10-06訂正：這裡原寫「3」已經過期超過一週——`stage3_hrp.py`
  自己的change log記載，XM早在2026-09-30加成長因子那一輪就已經用
  `cluster_count_selection.py`重算成6了，原因是舊的k=3解在新池組成下退化成
  單一群佔比62.8%，換k=6才是非退化最佳點；這不是這次openSec_boost動的，是
  更早一輪就該同步更新、但沒人回頭改這裡才一直傳成舊數字，見`code/research/
  stage3_hrp.py`的`L1_TARGET`常數註解）
  （H-03 用輪廓係數決定，不是寫死的 8；L2 已移除）
- 總經 clock_cell **已改用 5 年滾動窗**（H-18②），`stage4` 讀的是
  `_frozen/stage2/macro_rolling/`，不是 `_frozen/stage2/macro/`
- 🔴 **HRP 只用來分群，沒有用來配權重**（2026-09-10 查證）——所有組合績效都是
  **等權**（`walkforward_matrix.py:406` `_portfolio_series` ＝ `mean(axis=0)`）。
  `allocation="equal"/"proportional"` 決定的是**各群分到幾個代表名額**，不是投組權重
- 🔴 **候選池 15,810 個策略全部是「全期間贏過基準」篩出來的**（`passes_alpha_gate`
  TW 7,128/7,128、US 8,682/8,682 皆 100%）⇒ **OOS 絕對數字系統性高估，
  不可當預期報酬**（相對比較不受影響）。見 v10 §8

### 階段 −1（Phase 1~4）已完成的內容（保留備查）
- 台美股 Phase 1~4 全部跑完，四個變體（strict / **openSec**（採用）/ relaxed / all）都跑完並互相對照過
- **openSec** 是採用的正式設計：primary 用 Phase 1 過關的因子嚴格篩，secondary 全部開放（因為 Phase 1 檢定的是「因子自己能不能單調預測」，但 secondary 的功能是「提供不同構面資訊」，兩件事不一樣；台美股都獨立驗證過這個設計是對的）
- 修過一個關鍵 bug（A1 defect）：美股的動態條件 C 原本全部失效（用日頻 ffill 密集 frame 而非公告點稀疏 frame，導致「較上季升」恆假、「近N季最高」恆真），修好後美股 C 的排名完全改變
- 台股 2000-2004 財報資料回補完成（原本是空的，讓早期分位桶假訊號嚴重）
- 做過多項穩健性檢定：D1（9桶算術合併 vs 3桶實跑）、C1（primary門檻敏感度）、B3（規模集中度）、D3（子期間穩健性）——全部台美股都做過

### 給老師看的進度報告（都在 `文件/`）
| 檔案 | 用途 |
|---|---|
| `文件/研究進度報告_2026-09-07.md` | **最新一份**：H-26/H-27 兩題結果 + 方法論稽核推翻了哪些說法 |
| `文件/研究進度報告_2026-09-01.md` | 前一份：群數客觀化、免費午餐量化、群身份描述 |
| `文件/老師9-8意見待報告事項_下次會議9-15.md` | 老師 9-8 意見裡「還沒報」的部分 + 9/15 報告建議順序 |

⚠️ 舊版 CLAUDE.md 列的 `策略成果報告_2026-08-17.md`／`週進度報告_2026-08-17.md`／
`重跑計畫_老師方法論SOP.md` **這三份檔案已不存在**（2026-09-09 查證）。

已發布為 Artifact 的 HTML 版策略成果報告（若要更新記得用 `url` 參數更新原連結不要開新的）：
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

## 3. HRP 主線 ✅ 已完成（本節保留老師當初的原始指示，供理解設計動機）

2026-08-19 會議裡老師給的明確方向：**Hierarchical Risk Parity（HRP）**。

老師的解釋（口語，出自 8-19 逐字稿第3頁；⚠️ 該 PDF 已不在 repo 內）：直接對一堆策略算兩兩相關性、拿去做資產配置，只要一個相關係數估錯就會錯得很離譜。HRP 的做法是**先把策略做階層式分群**（依相似度分成幾群），再在群與群之間做風險分散配置——這樣抓到的分散效果（老師說的「吃到免費午餐」）比直接硬算相關性矩陣更穩健。老師認為這對我們現在手上幾千個策略（很多獲利模式很像）特別有用：先分群、再從每群挑代表，組合出來的結果比較容易穩健。

**✅ 已完成並超出原始範圍**：六棵樹已建（`code/research/stage3_hrp.py`），群數用輪廓係數客觀決定，
並延伸出有效獨立賭注數（ENB）、群內代表挑選規則、IS/OOS 驗證分支、四組對照實驗、
群的定量描述與 LLM 解釋、總經決策層。**核心發現與全部數字見 `研究框架總覽_v10.md` §3.5、§4、§6。**

一句話結論：**上萬個策略的有效獨立賭注數只有 3.27（TW）/ 4.36（US）/ 5.77（XM）**，
而且**分散效果的來源是市場邊界不是因子邊界**（同市場群間相關 0.78~0.94，跨市場降到 0.54）。

老師後續會提供評估標準，之後可能要跟美國公債等資產再做比較分析（**尚未進行**）。

**技術上的提示**：HRP 是在「策略層級」做分群配置，操作對象是已經跑完的策略報酬序列，
**不需要重新回測、也不需要即時連 SQL**——這也是為什麼 MySQL 打不開那陣子完全不影響進度（見第5節）。

---

## 4. 專案結構與執行注意事項

### 目錄結構（2026-09-09 全面查證更新）
```
/                          根目錄：核心 .py 檔（database.py/get_data.py/combinations.py等，主動在用，不要動）
│                          + CLAUDE.md / README.md / 安裝說明_SETUP.md（根目錄只有這三份 .md）
├── 文件/                  🔴 所有專案文件都在這裡（v10 框架、兩份追蹤文件、進度報告等，見第0節表格）
├── _frozen/                🔴 研究部主鏈的凍結產物（DD-08 雜湊鏈，stage0~4 + stage3_isoos + output_a）
│                          ✅ 2026-09-09 起已全部進 git（92 檔／97MB），換機器會跟著 clone 回來
├── _archive/               封存的舊資料（已 gitignore，含舊版 _analysis_outputs）
├── _analysis_outputs_*/    現役分析結果（見上表，不要動）
├── code/                   主要程式碼（分類導覽見 `code/README.md`）
│   ├── research/            🔴 研究部主鏈（stage0~4、HRP、群描述、決策層、contracts、tests）
│   ├── app/                 🆕 應用層（RunConfig/engine/risk/calibration/memo/audit/cli/ui）
│   │   └── _runs/            執行時的稽核紀錄 audit_log.jsonl（已 gitignore，新機器從空的開始）
│   ├── ops/tools.py         工具層 T1~T13（實戰部架構已廢止但這層仍在用，output_a 依賴它）
│   ├── utils/openai_quota.py LLM 免費額度煞車與帳本
│   ├── _archive/            封存的舊回測（原本 code/_ARCHIVE_* 已合併改名進來，已 gitignore）
│   │   └── ⚠️ 有一個 L3L4_strict_oldbench203 因權限問題卡在外面沒搬進來，
│   │      跟裡面的副本內容看起來一致但沒完全驗證過，使用者知情、要自己處理，不用主動管
│   ├── _catalog/             執行紀錄（master_index.parquet / dedup_registry.parquet 是功能性索引，不要刪）
│   │                         + llm_usage.jsonl（LLM 用量帳本，論文要交代成本時直接引用）
│   ├── results_artifacts/    回測原始產物（巨大，已 gitignore；⚠️ `.gitignore` 註解寫 58.5GB，
│   │                         但 2026-09-09 實測是 **189GB**，換機器評估容量時以實測為準）
│   └── phase{1,2,3,4}_*.py   四階段的執行與分析腳本
├── core/, utils/, TA-LIB/  底層依賴，不要動
└── .venv/                  虛擬環境，不要動
```

### ⚠️ 執行 Python 腳本的路徑陷阱
- `database.py`、`get_data.py`、`combinations.py`、`format_data.py` 等在**根目錄**，但大部分工作腳本在 `code/` 底下執行
- `code/fcv_core.py` 開頭會自動把 ROOT 跟 `code/` 都加進 `sys.path`（用 `_ROOT = Path(__file__).resolve().parent.parent`），所以只要 `import fcv_core` 過一次，後面 `from database import Database` 才會找得到模組
- **如果你自己寫 ad-hoc 查詢腳本**，切記先 `import fcv_core`（哪怕用不到它），或自己手動把根目錄加進 `sys.path`，不然會 `ModuleNotFoundError: No module named 'database'`
- **`config.ini` 的讀取用相對路徑**，指令碼一定要在 `code/` 目錄下執行（`cd code` 再跑），不要在根目錄跑，不然 `KeyError: 'database'`（`utils/config.py` 讀不到 `[database]` 區段）
- **Windows 終端機預設 cp950**，研究部腳本的中文輸出（含 `✓` 這種符號）會炸 `UnicodeEncodeError`
  ——跑任何 `python -m research.*` 都加 `PYTHONIOENCODING=utf-8`
- `config.ini` 含真實 API key，**絕不可回顯其值**

### 資料庫連線注意事項
- `Database(market)` 只是開連線，**不會自動依市場過濾**——`stock` 表沒有市場欄位，要 JOIN `company` 表用 `db._exchange_in_clause()` 才會篩對市場，這裡踩過雷（兩個市場撈出同一份資料）
- 美股還要另外加 `db._universe_clause('c.company_symbol')` 才會篩進 Russell 3000 名單（台股這個 clause 回傳空字串，不影響）

---

## 5. 資料庫：現況、換機器計畫、以及壞過一次的歷史

### ✅ 2026-09-10 已完成：已脫離 XAMPP，現在跑的是獨立版 MariaDB 11.4.3 LTS

換機器（2026-09-09~10）已執行完畢：資料庫改成**官方獨立版 MariaDB Server 11.4.3 LTS**、
`config.ini` 已還原（含 `[database]` 與 `[openai]`）、`_frozen/` 完整、
`python -m research.tests` 通過、Streamlit 已裝好可跑。
**程式碼完全沒改**——`database.py` 用 `pymysql.connect()` 走標準 MySQL 協定，
新伺服器一樣監聽 `127.0.0.1:3306`，換引擎不影響。

⚠️ **下面這段是當初的決策理由與遷移方式，保留備查**（已執行完，不用再做）：

### （已完成）2026-09-09 決定：換機器時順便脫離 XAMPP，改用獨立安裝的 MariaDB Server

理由：XAMPP 一次裝進 Apache／PHP／phpMyAdmin／FTP／Mercury Mail，但這個專案**只用得到 MySQL**，
其餘都是白白多出來的攻擊面（而且 XAMPP 預設 root 空密碼、常常沒限制只聽本機）。

- 換成**官方獨立版 MariaDB Server**（不是換成 MySQL 品牌）——現在裝的本來就是 MariaDB 10.4.32，
  同一個引擎家族、同一套 SQL 語法，`database.py:73` 用的是 `pymysql.connect()` 走標準 MySQL 協定，
  **只要新伺服器一樣監聽 `127.0.0.1:3306`、帳密一樣，程式碼完全不用改**
- 遷移方式：`mysqldump` 匯出 → 裝獨立版 → 設**真的 root 密碼**（不要沿用空密碼）、只聽 127.0.0.1
  → 匯入 → 更新自己的 `config.ini` → 跑 `python -m research.tests` 確認 131 項都過
- 要圖形介面就裝 HeidiSQL（單一 .exe 原生程式），不要再用 phpMyAdmin 那種要跑在 Apache 上的網頁介面
- Docker 方案討論過但不採用：這台是 RDP 連進來的實驗室電腦，多一層虛擬化容易卡，跟「環境不要差太多」的目標相反

### 2026-08-20 發生過的意外：MySQL 壞過、已修好（原理仍適用，但**路徑已變**）

⚠️ **2026-09-10 注意**：下面所有 `C:\xampp\mysql\...` 路徑都是**舊機器**的，
現在跑的是獨立版 MariaDB 11.4.3，data 目錄與 bin 目錄都不在那裡。
**修復原理（系統表是 Aria 格式、要用 `aria_chk` 不是 `myisamchk`）仍然成立**，
但實際路徑要用新安裝位置，不要照抄。

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

🔴 **2026-09-10：換機器已完成，以下舊路徑確定失效——要用時直接問使用者新位置，不要去試舊路徑**：
學姊論文 PDF 原本在 `C:\Users\iplab\Documents\_backup_from_repo\余姵穎_宣告式多因子量化回測系統之設計與實作.pdf`
（不在 repo 內，`.gitignore` 也明確排除 `/參考資料/`，**不可公開**）。

repo 根目錄另有學長 `郭鎧菘_結合量化交易與多代理人決策之金融交易框架初探.pdf`
（**未公開發表的論文，不可散布/上傳**，已 gitignore，所以 git 帶不走、換機器要手動複製）。

讀 PDF：**有文字層的**用 `fitz`（PyMuPDF，已在 `requirements_clean.txt` 裡）直接抓文字，
`python -c "import fitz; ..."` 最快。
⚠️ **2026-09-10 更正**：舊版寫「`pdftoppm` 沒裝所以 `Read` 工具的 PDF 頁面渲染會失敗」——
**新機器上 `Read` 工具讀 PDF 是可以的**，2026-09-10 就是用它讀完三份掃描版 PDF（`book3-part{1,2,3}.pdf`，
無文字層，`fitz` 抓不到字）。⇒ **有文字層用 `fitz`（快、省 token），掃描版用 `Read` 工具**。

讀 .docx 若環境沒有 `python-docx`：用 `zipfile` 解開後對 `word/document.xml` 下正則抓文字即可
（2026-09-10 讀 `逐字稿 (1).docx` 用的就是這招）。
