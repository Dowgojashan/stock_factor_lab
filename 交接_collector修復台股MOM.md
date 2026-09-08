# 交接 Prompt：請 collector 修復台股 MOM/MOM1 + 新增 4 個因子

> 使用方式：在 `D:\git\stock_factor_collector` 開一個 Claude Code 對話，把下面
> `---` 之間的全部內容貼進去。
> 診斷來源：`stock_factor_lab` 於 2026-08-06 執行 Phase 1 九桶線性檢定時發現。
>
> **任務 A**：修復台股 MOM/MOM1（目前被停用、值全 NULL）
> **任務 B**：新增 4 個因子（ACCRUAL / REV_G / VOL / NETDEBT_EBITDA），把因子池補到 20 個

---

# 任務 A：修復台股 MOM / MOM1 動能因子（目前被停用，值全為 NULL）

你在 `stock_factor_collector` repo。下游的 `stock_factor_lab` 做因子實驗時發現
**台股 MOM / MOM1 完全不可用**，需要你協助修復並回補。

### 一、問題現況（已由下游實測確認）

資料庫 `lab.factorvalue` 的 `factor_value` 欄位：

| 市場 | 因子 | 總筆數 | NULL 筆數 | NULL 率 | 有值的公司/日期 |
|---|---|---|---|---|---|
| **TW** | **MOM** | 145,653 | 145,545 | **99.93%** | 只有 27 家公司 × 2023 年的 4 個季末日 |
| **TW** | **MOM1** | 145,545 | 145,545 | **100%** | **完全沒有任何值** |
| TW | ROE（對照組） | 155,834 | 21,327 | 13.69% | 1,775 家，正常 |
| US | MOM（對照組） | 223,790 | 7,533 | 3.37% | 正常 |
| US | MOM1（對照組） | 214,354 | 7,533 | 3.51% | 正常 |

**列是有寫進去的**（日期 2000-03-31 ~ 2026-06-30、1,775 家公司都齊），
**只是 `factor_value` 全部是 NULL**。那 108 筆有值的（27 家 × 4 個 2023 年季末日）
應該是更早期某次執行的殘留，不具代表性。

### 二、根本原因（已定位到程式碼）

`CollectorFactory/TWCollectorFactory/TWFactorCollector.py` 第 165-169 行，
**MOM/MOM1 是被刻意停用、寫死成 `None` 的**：

```python
# MOM / MOM1 動能因子需要精準比對財報發布日附近的收盤價，牽涉的日期比對
# 邏輯還沒有處理好（_calculate_mom 有已知的排序問題），先維持停用（None），
# 不要用之前那個寫死 -1 的假值。
data_to_commit.append(self._to_factorValue_obj(company, None, period_id, factor_obj_dict["MOM"], report_date))
data_to_commit.append(self._to_factorValue_obj(company, None, period_id, factor_obj_dict["MOM1"], report_date))
```

註解提到的「已知的排序問題」在同檔 `_calculate_mom()`（第 198-232 行），
下游檢視後認為至少有這幾個缺陷：

1. **缺 ORDER BY（註解說的排序問題）**：第 213-218 行兩個查詢
   ```python
   select(Stock).join(Company).where(and_(Company.company_symbol == symbol, Stock.date <= report_date))
   ).scalars().first()
   ```
   `.first()` **沒有 `order_by`**，回傳的是任意一列（實務上常是最舊的那筆），
   不是「該日期之前最近的一個交易日」。應為 `.order_by(Stock.date.desc())`。

2. **例外處理會產生假值**：第 211-222 行的 `try/except` 只記 log、沒有 return 或 raise，
   失敗時 `report_day` / `report_announce_day` 未定義 → 落進第 224-230 行的裸 `except`
   → **`mom = 0`**。這會把「算不出來」偽裝成「動能為 0」，比 NULL 更危險。
   算不出來時應該回傳 `None`。

3. **`MOM1` 根本沒有實作**：整個 TW collector 找不到 `_calculate_mom1` 或任何 MOM1 的計算邏輯，
   只有第 169 行那個寫死的 `None`。

### 三、可以參考的正確實作（美股版，已在運作）

`CollectorFactory/USCollectorFactory/USFactorCollectorFMPRaw.py::get_momentum_data()`
（第 242-301 行）是能用的版本，定義為：

```
MOM  = (filing_date 當日或之前最近收盤 − 期末日收盤) / 期末日收盤
MOM1 = (filing_date+45天 當日或之前最近收盤 − 期末日收盤) / 期末日收盤
```

它的關鍵作法值得照抄：
- `price_at_or_before(d)` 用**已排序**的價格序列取 `cands[-1]`，正確拿到「該日或之前最近一筆」
- 取不到價格時給 `None`，**不給 0**
- 寫入前先 `delete` 同 (company, factor, date 區間) 的舊列，達成**冪等**（重跑不會疊加）

### ⚠️ 四、台股與美股的關鍵差異（設計上要先決定）

**台股的 `factorvalue.filing_date` 是 NULL**（美股才有值；下游 `database.py` 有註解說明
「台股為 NULL、美股有值，供前瞻防護分流」）。

所以**台股不能照抄美股「用 filing_date 對價格」的作法**。現有 `_calculate_mom` 是改用
**法定申報期限**做對應（第 206 行）：

```python
report_announce_dates = {12:'03-31', 3:'05-15', 6:'08-14', 9:'11-14'}
```

這是可行的替代方案，但請你確認並決定：

- (a) 沿用法定期限對應（實作簡單，但**所有公司同一天**，與實際發布日有落差）
- (b) 想辦法補上台股真實的 filing_date（較精確，但要額外資料源）
- (c) 其他你認為更合理的定義

**無論選哪個，請把最終定義寫進程式註解與 commit message**，
因為下游要在論文裡說明這個因子怎麼算的。

### 五、需要你做的事

1. 修好 `_calculate_mom()`（至少解掉上面三個缺陷）
2. 補上 MOM1 的實作（台股版定義請比照或說明為何不同）
3. 解除 `TWFactorCollector.py` 第 165-169 行的停用
4. **回補台股全歷史**：2000-03-31 ~ 2026-06-30、1,775 家公司
5. 確保是**冪等寫入**（可重跑、不疊加），參考美股版的 delete-then-insert

### 六、驗收標準（下游會用這個檢查）

```sql
SELECT factor_name, COUNT(*) n,
       SUM(CASE WHEN factor_value IS NULL THEN 1 ELSE 0 END) n_null,
       COUNT(DISTINCT company_id) ncomp, MIN(date), MAX(date)
FROM factor RIGHT JOIN factorvalue ON factor.id = factorvalue.factor_id
LEFT JOIN company ON factorvalue.company_id = company.id
WHERE exchange_name IN ('TWSE') AND factor_name IN ('mom','mom1')
GROUP BY factor_name;
```

- `n_null` 比率應降到與 ROE / EPS 同級（**約 14% 或更低**），不是 99.93%
- `ncomp` 應接近 1,775
- 日期範圍應涵蓋 2000-03-31 ~ 2026-06-30
- 抽查數值應落在合理範圍（季度動能大致在 ±50% 內，不應大量出現剛好 0 或 −1）

### 七、請不要做的事

- ❌ **不要用寫死的假值**（歷史上曾經寫死 −1，現有註解特別警告過）
- ❌ 算不出來時**不要回傳 0**——會被誤認為「動能為零」，比 NULL 更難察覺
- ❌ 不要動美股的 MOM/MOM1（那邊是正常的，NULL 率僅 3.4%）

### 八、這件事的影響（說明急迫性）

下游 `stock_factor_lab` 先前的台股因子實驗中，
「MOM 淨貢獻 −9.54%、10 個候選因子中斷崖式墊底」這個結論，
其實是**用每期平均只有 2.7 家公司的雜訊算出來的**，已確認無效並在報告中更正。

指導教授當時就質疑「MOM 死翹翹這個有點怪」，現在確認他是對的。
在這個因子修好之前，**台股的動量構面完全無法研究**。

---

# 任務 B：新增 4 個因子（因子池 16 → 20）

## 為什麼要加

下游盤點現有 17 個因子（扣掉已停用的 PE 為 16 個可用）的**類別分布**後發現：

| 類別 | 現有因子 | 數量 |
|---|---|---|
| 估值倍數 | EV_EBITDA, EV_S, PB, PS, P_IC | 5（過剩，且實測彼此高度同質） |
| 獲利能力 | ROE, EPS, ROIC, CROIC | 4 |
| 現金流品質 | FCF_P, FCF_OI, OCF_E | 3 |
| 動量 | MOM, MOM1 | 2（任務 A 修復中） |
| 財務結構 | DEBTRATIO | 1（偏薄） |
| 規模 | REVENUE | 1 |
| **成長** | — | **0** 🔴 |
| **波動** | — | **0** 🔴 |
| **應計品質** | — | **0** 🔴 |

指導教授要求因子池要有 20 個左右。**補的重點是填補上面三個空白類別，不是再加估值類。**

## ✅ 好消息：4 個因子都不需要新的資料源

四個因子**全部只用你現在已經抓到的原始欄位**就能算，
**不需要新增 FMP API 呼叫、也不需要重新從 TEJ 匯出**：

- 台股：`singleindicator` 現有的 13 個中文指標 + `stock` 表收盤價
- 美股：`USFactorCollectorFMPRaw` 已經在抓的三大報表欄位 + `_fetch_price_history()`

## 四個新因子的規格

### 1. `ACCRUAL`（應計項目）— 補「應計品質」類別

```
ACCRUAL = (淨利 − 營運現金流) / 總資產
```

| 市場 | 分子 | 分母 |
|---|---|---|
| TW | `常續性稅後淨利` − `來自營運之現金流量` | `負債及股東權益總額` |
| US | `netIncome` − `operatingCashFlow` | `totalAssets` |

- 三個欄位你**現在都已經在用**（TW 算 ROE/OCF_E/ROIC 用的、US 的 FMPRaw 也都有）
- 學術依據：Sloan (1996) 應計異象。**應計項目高 = 盈餘品質差**，預期報酬較低（負向因子）
- 合理範圍：多數落在 −0.3 ~ +0.3；超過 ±1 應視為異常值
- 分母為 0 或 None → 回 `None`

### 2. `REV_G`（營收年成長率）— 補「成長」類別

```
REV_G = (本季營收 − 去年同季營收) / |去年同季營收|
```

| 市場 | 欄位 |
|---|---|
| TW | `營業收入淨額` |
| US | `revenue` |

- ⚠️ **必須是「同季比同季」（往回推 4 個季度）**，不是跟上一季比。
  下游已驗證**台股的營收是單季數字、不是累計**（台積電 2022 各季 491/534/613/625 億，
  非遞增累加），所以同季 YoY 可直接相減，季節性自然被抵銷
- ⚠️ 分母要取**絕對值**，否則去年同季為負時成長率的正負號會反向
- 往回找不到 4 季前的資料（最早那幾季）→ 回 `None`，**不要回 0**
- 合理範圍：−1.0 ~ +3.0；極端值（如 +50）通常是去年基期極小，屬正常但要留意
- 注意：現有的 `REVENUE` 因子是**營收絕對值（＝規模因子）**，與這個成長率意義完全不同，兩個都要保留

### 3. `VOL`（報酬波動度）— 補「波動」類別

```
VOL = 期末日之前 60 個交易日的日報酬標準差 × sqrt(252)
```

- 資料來源：`stock` 表的 `close`（兩個市場都有；美股用 `_fetch_price_history()`）
- **跟任務 A 的 MOM 共用同一套價格對日期邏輯**，建議一起做、共用
  `price_at_or_before()` 這類 helper
- 學術依據：低波動異象（low-volatility anomaly）。**波動高 = 預期報酬低**（負向因子）
- 交易日不足 60 天（新上市公司）→ 回 `None`，不要用不足的天數硬算
- 合理範圍：台股年化波動多在 0.2 ~ 0.8；> 2.0 要檢查是否有價格資料錯誤
- 視窗長度 60 天可以調整，但**請把最終選擇寫進註解**（下游論文要說明）

### 4. `NETDEBT_EBITDA`（淨負債／EBITDA）— 補強「財務結構」

```
NETDEBT_EBITDA = 淨負債 / EBITDA
```

| 市場 | 分子 | 分母 |
|---|---|---|
| TW | `淨負債` | `稅前息前折舊前淨利` |
| US | `netDebt` | `ebitda` |

- 兩個欄位你**現在都已經在用**（就是算 EV_EBITDA 的那兩個）
- 意義：**要幾年的 EBITDA 才還得完淨負債**，衡量償債能力。
  與現有的 DEBTRATIO（負債／資產，只看存量比例）角度不同
- ⚠️ **EBITDA ≤ 0 時這個比值沒有意義**（負債為正、EBITDA 為負會得到負值，
  但那代表「還不起」不是「負債輕」）→ 這種情況請回 `None`，不要回負數
- 合理範圍：−2 ~ +10；> 20 通常是 EBITDA 接近 0 造成的爆炸值

## 任務 B 需要你做的事

1. 在 `factor` 表新增這 4 筆定義（`_get_or_create_factor` 應可處理），
   並填上 `factor_formula` 欄位（下游會拿來寫論文）
2. 台股：在 `TWFactorCollector.get_data()` 加上 4 個計算
   - ACCRUAL / NETDEBT_EBITDA 直接用 `data_dict` 現有欄位
   - REV_G 需要跨季度回看——注意 `by_date` 已經把**該公司所有季度**都載入了，
     可以直接往回找 4 季，不需要額外查詢
   - VOL 需要查 `stock` 表，建議與修好的 MOM 共用價格查詢
3. 美股：在 `USFactorCollectorFMPRaw` 加上對應計算（欄位都已在 `base_rows` / 三大報表裡）
4. 回補兩個市場的全歷史，維持**冪等寫入**（delete-then-insert）

## 任務 B 的驗收標準

```sql
SELECT factor_name, COUNT(*) n,
       SUM(CASE WHEN factor_value IS NULL THEN 1 ELSE 0 END) n_null,
       COUNT(DISTINCT company_id) ncomp, MIN(date), MAX(date),
       MIN(factor_value), MAX(factor_value), AVG(factor_value)
FROM factor RIGHT JOIN factorvalue ON factor.id = factorvalue.factor_id
LEFT JOIN company ON factorvalue.company_id = company.id
WHERE exchange_name IN ('TWSE')   -- 美股改成 ('NASDAQ','NYSE','AMEX')
  AND factor_name IN ('accrual','rev_g','vol','netdebt_ebitda')
GROUP BY factor_name;
```

- `n_null` 率應與 ROE/EPS 同級（**台股約 14%、美股接近 0%**）
- `ncomp` 台股應接近 1,775、美股接近 2,990
- `REV_G` 的 NULL 率會**比其他因子高約 4 季的量**（最早 4 季無法算 YoY），屬正常
- 數值範圍請對照上面每個因子的「合理範圍」抽查，特別注意**沒有大量剛好是 0 的值**
  （那通常代表算不出來被誤填成 0）

## 任務 B 請不要做的事

- ❌ 算不出來時**一律回 `None`，絕不回 0 或其他假值**（同任務 A 的教訓）
- ❌ 不要動現有 16 個因子的公式（下游已有基於它們的實驗結果）
- ❌ 不要為了湊 20 個而加估值類因子——那一類已經 5 個且彼此高度同質

---

## 附註：下游的診斷紀錄（不需貼給 collector）

### 任務 A（MOM）的發現經過
- 發現時機：`stock_factor_lab` 執行 `code/phase1_linearity.py`（9 桶單因子線性檢定），
  16 個因子中 MOM/MOM1 兩個產生 0 個策略、`run_spec` 直接拋錯
- 追查路徑：`report:mom` frame 只有 27 個非空欄位 → 查 DB 發現 `factor_value` 99.93% NULL
  → 查 collector 原始碼發現是刻意停用（`TWFactorCollector.py` L165-169）
- 相關更正：`台股結果解讀_report.md` 頂端已加重大更正區塊
- 美股不受影響：US MOM 在美股實驗中淨貢獻 +0.71%、排名第 1，資料正常

### 任務 B（新增 4 因子）的選擇依據
- 台股 `singleindicator` 只有 **13 個原始指標**，這是能加什麼因子的硬限制。
  **沒有毛利、流動資產／負債、研發費用、存貨、應收帳款**，所以毛利率(Novy-Marx)、
  流動比率、R&D 強度這些經典因子**台股做不到**，除非重新從 TEJ 匯出更多欄位
- 選這 4 個的理由：(a) 各補一個空白類別 (b) 台美**都能算**，跨市場比較不會斷
  (c) 都有事前的預期方向，可用 Phase 1 線性檢定直接驗證 (d) 零額外 API 成本
- 備選（若想替換）：`ASSET_G`（總資產 YoY 成長，資產成長異象 Cooper 2008，預期負向）
  可替換 NETDEBT_EBITDA
- ⚠️ **多重檢定風險**：因子 16→20 等於多開 4 次檢定，偽陽性機會上升。
  跟指導教授報告時的說法：不是為了增加樣本而加，是原本因子池在
  成長／波動／應計三個構面完全空白，補齊後才涵蓋論文 §3.6.1 列出的完整類別；
  且 Phase 1 的線性檢定會當守門員（沒線性就淘汰）

### 完成後下游要做的事
1. 重跑 `python code/phase1_linearity.py --market TW`（因子池 16 → 20）
2. 重跑 `python code/phase1_analyze.py --market TW` 更新去留名單
3. 美股同步跑一次
4. 更新 `重跑計畫_老師方法論SOP.md` 的因子池與規模估算
