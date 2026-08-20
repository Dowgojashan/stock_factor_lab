# 階段 1 · 關卡 C　標記清單設計 v1

> **目的**：反推出 Gate C 必須產出哪些欄位，讓 Agent1 的 Step2 快篩與 Step3 精挑真的跑得動。
> **定位變更**：階段 1 從「篩選關卡」改為「**標記關卡**」——不為了減量而篩，只為了讓後面有東西可查。
> **基準資料**：`_analysis_outputs_phase4/{TW,US}_L4_openSec_final_candidates.csv`（台股 7,162 / 美股 6,916）
>
> ## 🔴 v1.1 修訂（2026-08-20 · 全量實測連動）
> 1. **C-1 的 `strategy_id` 改為 `strategy_uid = market::strategy`**——實測台美策略字串**碰撞 1,381 個**，裸 `strategy` 不能當主鍵。
> 2. **§五-1「`valuation` 定義待釐清」→ ✅ 已定案採方案 (a)**：`valuation` 即 C-1 的 `V` 欄位（v0/v1），階段4 不另產 `valuation_verdict`；「條件反轉」現象改用程式算的布林欄位 `v1_beneficial` 承接。
> 3. **§五-2「是否輪動」→ ✅ 已定案**：`rotation_score` = 年度前三大貢獻股在相鄰年度的 Jaccard 相似度、全期平均（低分=有輪動）。原料 `trades.parquet` 已驗證齊備；有效交易年數 < 3 者記 NaN 且不因此降級。
> 4. **§五-3「分位數市場內 vs 全池」→ ✅ 維持原判**：所有 `*_pct` 一律 `groupby(market)`。
> 5. **C-6 `annual_returns` 的實際可用年份**：實測台股策略起始期橫跨 2000-06~2007-05，**並非所有策略都有 2000–2025 全 26 年**；此欄位須容許前段為 NaN，且**禁止用 0 代表缺值**（台股早期假 0 已有前車之鑑）。
> 詳見 `落差處理方案_v1.md`、`系統設計文件_v1.md`。

---

## 一、反推規則

Agent1 的 **Step2 快篩是純程式**（`T2 filter_pool(條件)`），LLM 尚未登場。

> **所以：12 格矩陣裡出現的每一個名詞，都必須是 strategy_map 的一個可比較欄位。**
> **矩陣提到、但 strategy_map 沒有的 → 快篩那一格就執行不了。**

以下清單就是照這條規則盤出來的。

---

## 二、需求盤點：矩陣提到什麼 → 誰要產

| 矩陣中的名詞 | 需要的欄位 | 產出階段 | 現況 |
|---|---|---|---|
| MDD 標準／更嚴／最嚴 | `max_drawdown` + 市場內分位 | **Gate C** | 有原始值，缺分位 |
| EffN 高／最高／分散 | `effective_n` | **Gate A** | ❌ 未產出 |
| CAGR 前段 | `cagr` + 市場內分位 | **Gate C** | 有原始值，缺分位 |
| top1_share（防假） | `top1_share` | **Gate A** | ❌ 未產出 |
| credibility（防假） | `credibility_grade` | **Gate A** | ❌ 未產出 |
| stability＝高原 | `stability_grade` | Gate B | ❌ 未產出 |
| 報酬形態偏「穩定爬升」 | `return_shape` | **Gate C** | ❌ 未產出 |
| 避開「純趨勢型」 | `factor_type` | **Gate C** | ❌ 未產出（可由 F1 規則導出） |
| 靠選股 alpha 不靠趨勢 | `factor_type` + `return_shape` | **Gate C** | 同上 |
| valuation＝favorable | `valuation` | 階段 4 | ⚠️ 定義待釐清（見 §五-1） |
| regime_fit 含「熊市抗跌」等 | `regime_fit`（台/美各一） | 階段 2a | 不屬 Gate C |
| co_fail_regimes 共跌檢查 | `co_fail_regimes` | 階段 3（HRP） | 不屬 Gate C |
| complement_partners | `complement_partners` | 階段 3（HRP） | 不屬 Gate C |

**結論**：矩陣一半以上的條件卡在 Gate A/C 還沒產出的欄位上。這就是為什麼階段 1 不能跳過。

---

## 三、Gate C 標記清單

**標記欄**：`快篩` ＝ T2 可直接用來過濾（必須是數值或有限類別）；`情報` ＝ 只給 LLM 在 Step3 讀，不參與機械篩選。

### C-1　身份與結構（零成本，從策略字串拆解即可）

| 欄位 | 型別 | 說明 | 用途 |
|---|---|---|---|
| `strategy_id` | str | 主鍵 | — |
| `market` | TW / US | **必要**，因為 regime 台美各一套，快篩不能混market | 快篩 |
| `F1_factor` | str | 例 `P_IC` | 快篩 |
| `F1_band` | 0 / 1 / 2 | 分位桶 | 快篩 |
| `F2_factor` | str / null | | 快篩 |
| `F2_band` | 0/1/2 / null | | 快篩 |
| `F2_empty` | bool | **老師 8/19 指示「F1+C 就好」的分流依據**；台股 384 個、美股 652 個 | 快篩 |
| `C_id` | C1–C20 / null | | 快篩 |
| `C_source` | ROE / EPS / FCF_P | | 快篩 |
| `C_rule` | riseq1 / qmax4 / yoy … | | 快篩 |
| `V` | v0 / v1 | 保留（老師判定無普遍價值，但不移除） | 快篩 |

> 這一組**幾乎零成本**，但價值很高——矩陣的「避開純趨勢型」「靠選股 alpha」直接可以用 `F1_factor in (MOM, VOL)` 表達，不需要等 `factor_type`。

### C-2　因子類型（矩陣直接引用）

| 欄位 | 型別 | 說明 |
|---|---|---|
| `factor_type` | 估值型 / 體質型 / 動能型 / 混合型 | 由 `F1_factor` + `F2_factor` 的因子分類規則導出 |
| `factor_type_basis` | str | 判定依據（哪兩個因子、各屬哪類），供 Agent2 引用 |

**因子分類對照**（沿用簡報 S3 的五分類）：

```
估值型：PB PS P_IC EV_S EV_EBITDA FCF_P FCF_OI
體質型：ROE EPS ROIC CROIC OCF_E ACCRUAL
動能型：MOM VOL
規模型：REVENUE REV_G
結構型：DEBTRATIO NETDEBT_EBITDA
```

**判定規則**：F1 與 F2 同類 → 該類型；不同類 → `混合型`，並在 `factor_type_basis` 記兩者。

> ⚠️ **可簡化**：v8 原設計寫「混合型由 LLM 補判、受約束選擇題」。但實際資料只有 **F1 + F2 兩個因子**，台股 F1 僅 11 個桶、美股 16 個，規則覆蓋率是 100%。**建議 Gate C 這一項純規則，不引入 LLM**，省掉一個幻覺面。

### C-3　報酬與風險形態（整段統計量，⚠️ 不得碰 regime）

> v6 定案的斷循環設計：階段 1 不能依賴階段 2。任何需要分 regime 的判斷一律延到階段 2。

| 欄位 | 型別 | 說明 | 用途 |
|---|---|---|---|
| `cagr` | float | 已有 | 快篩 |
| `cagr_pct` | 0–100 | **市場內分位**，矩陣「CAGR 前段」用這個 | 快篩 |
| `max_drawdown` | float | 已有 | 快篩 |
| `mdd_pct` | 0–100 | 市場內分位 | 快篩 |
| `ann_vol` | float | 整段年化波動 | 快篩 |
| `win_ratio` | float | 已有 | 情報 |
| `sharpe_ann` | float | 已有 | 快篩 |
| `return_shape` | 穩定爬升 / 大起大落 | 建議用**年度報酬標準差的市場內分位**切三段，取兩端 | 快篩 |
| `risk_shape` | 淺回撤 / 中等 / 深回撤 | `mdd_pct` 切三段 | 快篩 |
| `worst_year` | float | 最差年度報酬 | 情報 |
| `neg_year_count` | int | 負報酬年數 | 情報 |
| `max_consec_loss_months` | int | 最長連續虧損月數，保守型很在意 | 情報 |

### C-4　可執行性（🆕 新增，源自武器庫實驗發現）

| 欄位 | 型別 | 說明 | 用途 |
|---|---|---|---|
| `holdings_median` | float | 已有（`持股數`） | 快篩 |
| `holdings_p10` | float | 低分位持股數 | 快篩 |
| `coverage_ratio` | 0–1 | **有持股月份 ÷ 總月份** | 快篩 |
| `turnover_ann` | float | 年換手率 | 情報 |

> **為什麼新增**：`avg_holdings` 目前排除空手月份、且用平均，會把「一半月份 22 檔、一半月份 2 檔」偽裝成正常。台股候選池持股數 p10 是 13 檔。老師的原話是「選出來的股票不能太少，**那就是策略不穩定**」——講的是穩定性，平均恰好把它藏起來。

### C-5　規模依賴（🆕 新增，矩陣沒有但應該有）

| 欄位 | 型別 | 說明 | 用途 |
|---|---|---|---|
| `size_tilt_pct` | 0–100 | 持股的市值分位中位數（用 MKTCAP） | 快篩 |
| `smallcap_share` | 0–1 | 持股落在最小市值三分之一的比例 | 快篩 |

> **為什麼新增**：實驗顯示台美 Top 50 各有 **48% / 58%** 含最小規模桶，而全池佔比只有 3.1% / 10.5%。**保守型不應該拿到重倉微型股的策略**，但目前矩陣沒有任何欄位能表達這個限制。這是實質的流動性與可部署性風險。

### C-6　逐年報酬序列（只存不判，給階段 2 用）

| 欄位 | 型別 | 說明 |
|---|---|---|
| `annual_returns` | dict / 26 欄 | 2000–2025 各年報酬 |

> ⭐ **關鍵約束**：Gate C **只存數字、不貼牛熊標籤**。等階段 2a 的 regime 出來，才由階段 2 把它歸類成 `regime_fit`。這是斷開「階段1 依賴階段2」循環的設計，不能違反。

---

## 四、不屬於 Gate C（劃清界線）

| 欄位 | 歸屬 | 備註 |
|---|---|---|
| `effective_n` / `top1_share` / `credibility_grade` / 是否輪動 | **Gate A** | 矩陣的「防假 alpha」全靠這組，優先度等同 Gate C |
| `stability_grade` / 強因子清單 | Gate B | 建議大幅簡化，相似性交給 HRP |
| `regime_fit`（台/美各一） | 階段 2a | 依賴 regime 切割 |
| `macro_fit` | 階段 2b | 凍結模型 |
| `cluster_id` / `co_fail_regimes` / `complement_partners` | 階段 3（HRP） | 🆕 HRP 的分群結果應成為 strategy_map 欄位 |
| `return_story` 判決 | 階段 4 | 乙案，判決凍結、文字按需生成 |
| `valuation` | 階段 4 | 定義待釐清，見下 |

---

## 五、實作前要先決定的三件事

### 1. `valuation` 的定義要重新想

12 格矩陣的保守型-牛市格寫「`valuation`＝favorable」，但：
- 這個欄位原本掛在階段 4
- 而老師 8/19 已判定 **V 構面「是做好玩的，沒有什麼用」**

**要決定**：`valuation` 是指 (a) V 濾網開關（`V == v1`），還是 (b) 策略當下持股的估值水準（獨立計算）？

如果是 (a)，那它已經在 C-1 的 `V` 欄位裡，矩陣可以直接改寫、階段 4 不用產這個欄位。
**建議選 (a)**，理由是 (b) 需要額外計算、而且與 F 構面的估值因子高度重疊。

### 2. 「是否輪動」的定義沒寫死

Gate A 的「靠單一飆股的假貨（低 EffN + **不輪動**）」，但「輪動」一直沒有操作型定義。

**建議**：定義為「年度前三大貢獻股的重疊程度」——若每年的主要貢獻者都是同一批，就是不輪動。

### 3. 分位數是「市場內」還是「全池」

矩陣的數值門檻「待資料，用分位數」。因為台美的 CAGR 分布差很多（中位 15.53% vs 23.61%），**分位數必須以市場為單位計算**，否則美股策略會整批排在台股前面，快篩會失效。

**所有 `*_pct` 欄位一律 `groupby(market)` 計算。**

---

## 六、驗收檢查

Gate C 做完之後，用這個方式驗收——**把 12 格矩陣逐格拆成 SQL/pandas 條件，看能不能寫得出來**：

```
保守型 × 危機格：
  mdd_pct <= X               ✓ C-3
  effective_n >= Y           ✓ Gate A
  regime_fit 含「危機抗跌」    → 階段 2a
  co_fail_regimes 檢查        → 階段 3
  smallcap_share <= Z        ✓ C-5（新增，原矩陣沒有）

積極型 × 盤整格：
  cagr_pct >= X（盤整期）     ⚠️ 需分 regime → 階段 2a
  stability_grade = 高原      → Gate B
  factor_type != 動能型       ✓ C-2
  return_shape = 穩定爬升      ✓ C-3
```

**若某一格有任何一個條件寫不出來 → 那個欄位就是漏掉的，必須補。**

三類 × 四 regime ＝ 12 格全部走一遍，這是最可靠的完整性檢查。

---

## 七、產出

- `strategy_map_gateC.parquet`：上述所有 C-1 ~ C-6 欄位，逐策略一列
- 台股 7,162 列 + 美股 6,916 列 ＝ 14,078 列
- 凍結，作為階段 2/3/4 的輸入與 `T3 get_strategy_profile` 的後端

**預期淘汰量**：接近 0。階段 1 尾端的寬鬆硬篩，能砍的東西 Phase 1–4 幾乎都已經砍過了：
- 「CAGR < 基準」→ Phase 3 已 gate（候選池 **0 個**低於基準）
- 「持股數 < 10」→ Phase 2 已 gate

**剩下唯一會真正砍到東西的是 Gate A 的「靠單一飆股」與 C-4 的 `coverage_ratio`**，而這兩個是資料品質防線，不是減量手段。
