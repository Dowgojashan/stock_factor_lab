# 交接 Prompt：TEJ 報酬指數匯入 + （選配）財報欄位擴充

> 使用方式：在 `D:\git\stock_factor_collector` 開一個 Claude Code 對話，
> 把 `---` 之間的內容貼進去。需要中央大學 IP（校外走 SSLVPN: sslvpn.ncu.edu.tw）。
>
> **任務 C（必做）**：匯入台股「加權股價**報酬**指數」，補上不含股利的基準缺口
> **任務 D（選配，先評估再決定）**：從 TEJ 多匯出 5 類財報欄位，解鎖 4 個經典因子

---

# 任務 C：匯入台股加權股價報酬指數（TAIEX Total Return）

## 一、為什麼要這個

下游 `stock_factor_lab` 在 2026-08-08 查證發現**基準與策略不對等**：

| | 含股利？ | 說明 |
|---|---|---|
| 策略績效 | ✅ 含 | 用 TEJ **還原收盤價**（`stock.close`，配息已還原） |
| `taiex` 表 | ❌ 不含 | 年底值 2000→4,739｜2010→8,972｜2024→23,035，是**價格指數** |

→ 一直在拿「含息的策略」比「不含息的大盤」，**每個「贏大盤」都被高估約 3~4pp**。

實測差距：外部價格指數年化 **4.71%**，下游自建的等權宇宙基準 **8.67%**。

下游已用「自建宇宙基準」（把宇宙裡的股票全部等權買下來、同樣月頻換股同樣費率）
繞過這個問題，主線分析已改用 8.67%。**但論文仍需要一個對外的、傳統的基準**
（學姊論文用的就是外部指數），所以還是要把報酬指數補進來。

## 二、要抓什麼

**TEJ Pro → 台灣加權股價報酬指數（發行量加權股價報酬指數，TAIEX Total Return Index）**

- `TaiexImporter.py` 的註解已明載「**加權指數／報酬指數皆同一種版面**」，
  代表現有的匯入器應該不需要改格式，換一個指數代碼／檔案就能吃
- 頻率：**日**（與現有 `taiex` 表一致）
- 期間：**能抓多早抓多早**

## 三、⚠️ 已知限制：2003 年以前沒有

台灣的**報酬指數 2003 年才開始發布**，2000~2002 沒有官方資料。

這是硬限制，不是抓法問題。**請不要用任何方式回補、外推或估算 2000~2002 的值**
（例如拿價格指數加一個假設的股利率），下游要在論文裡誠實說明這段沒有資料。
抓到多早就寫多早，缺的就讓它缺。

## 四、寫進資料庫的方式

現有 `taiex` 表**不要動**（下游還在用它做價格指數對照）。請**新開一張表**：

```sql
CREATE TABLE taiex_tr (
  id    INT AUTO_INCREMENT PRIMARY KEY,
  date  DATE NOT NULL,
  close DOUBLE,
  UNIQUE KEY uk_date (date)
);
```

- **欄位名必須是 `date` / `close`**——下游是用
  `SELECT date, close FROM {BENCHMARK_TABLE[market]}` 讀的（見
  `stock_factor_lab/code/phase1_analyze.py::bench_cagr`），欄位名一致才能直接接上
- 冪等寫入（delete-then-insert 或 upsert），可重跑不疊加

## 五、驗收標準

```sql
SELECT COUNT(*) n, MIN(date), MAX(date), MIN(close), MAX(close) FROM taiex_tr;
```

- 起始日應在 **2003 年附近**（若能更早請說明資料來源）
- **報酬指數的年化必須明顯高於價格指數**——價格指數同期約 4.7%，
  報酬指數合理落在 **7~9%**（差距即股利貢獻，台股股息率約 3~4%）。
  ⚠️ **若兩者年化幾乎相同，代表抓錯成價格指數了**，請重新確認指數代碼
- 交易日數應與同期 `taiex` 表相當（不應差超過幾天）

## 六、順便確認（美股，不是 TEJ）

`sp500` 表同樣是**價格指數**（2000→1,320｜2024→5,882），問題完全相同。
FMP 有 S&P 500 Total Return（`^SP500TR`），若能抓請一併建 `sp500_tr` 表，同樣 schema。
這個優先度低於台股，抓不到就先算了。

---

# 任務 D（選配）：從 TEJ 多匯出財報欄位

> **這一項請先評估可行性與工作量再回報，不要直接動手。**
> 下游會依你的回覆決定要不要做（因為做了要重跑約 5 小時的實驗）。

## 一、目前的限制

台股 `singleindicator` 只有 **13 個原始指標**，這是「台股能做什麼因子」的硬天花板。
**缺的欄位**造成下列經典因子**台股做不到、但美股做得到**（FMP 三大報表都有）：

| 缺的欄位 | 卡住的因子 | 學術依據 |
|---|---|---|
| 營業毛利 | **GP_A**（毛利／總資產） | Novy-Marx (2013)，「最乾淨的獲利能力因子」 |
| 流動資產、流動負債 | **CURRENT_RATIO**（流動比率） | 財務結構／流動性構面 |
| 研發費用 | **RD_S**（研發／營收） | Chan et al. (2001) R&D 強度 |
| 存貨 | **INV_G**（存貨 YoY 成長） | Thomas & Zhang (2002) 存貨異象 |
| 應收帳款 | （可與存貨合併成營運資本應計） | 補強現有 ACCRUAL |

## 二、為什麼值得做

下游的因子池目前 20 個，**台美各自都能算**，跨市場比較才不會斷。
但這 20 個是「**遷就台股能算什麼**」湊出來的——美股其實還能多算 4 個，
只是為了對齊台股而沒開。補上後：

1. **因子池 20 → 24**，且新增的都在台美都能算，跨市場比較維持完整
2. 補上「獲利品質」最主流的 GP_A——目前獲利面只有 ROE/EPS/ROIC/CROIC，全是 ROE 家族
3. 回應指導教授對「因子構面涵蓋度」的要求

## 三、要請你評估的問題

1. TEJ Pro 的財報版面**能不能一次多勾這幾個欄位**？還是要換一個資料集（成本／權限）？
2. 匯出後 `TWFinancialReportCollector` / `singleindicator` 的**對照表要改多少**？
3. **全歷史回補（2000-2026、1,775 家）要多久**？

## 四、若決定要做，欄位規格

TEJ 中文欄位名以實際版面為準，以下是需要的**科目**：

| 需要的科目 | 對應美股欄位（FMP，供比對定義） |
|---|---|
| 營業毛利（或營業收入淨額 − 營業成本） | `grossProfit` |
| 流動資產合計 | `totalCurrentAssets` |
| 流動負債合計 | `totalCurrentLiabilities` |
| 研發費用 | `researchAndDevelopmentExpenses` |
| 存貨 | `inventory` |
| 應收帳款 | `netReceivables` |

因子公式（台美一致）：

```
GP_A          = 營業毛利 / 總資產                    預期正向
CURRENT_RATIO = 流動資產 / 流動負債                  預期正向（偏弱）
RD_S          = 研發費用 / 營業收入淨額              預期正向（產業效應強）
INV_G         = (本季存貨 − 去年同季存貨) / |去年同季存貨|   預期負向
```

- ⚠️ `INV_G` 與 `REV_G` 同樣是**同季比同季（往回 4 季）**，不是比上一季
- ⚠️ 分母為 0 / None / 找不到 4 季前 → 一律回 `None`，**絕不回 0**
  （這是任務 A 的教訓，回 0 會被誤認為「成長為零」，比 NULL 更難察覺）
- ⚠️ 研發費用在台股**很多公司是 0 或未揭露**——請區分「真的是 0」與「沒揭露」，
  沒揭露請回 `None`

## 五、驗收標準

```sql
SELECT factor_name, COUNT(*) n,
       SUM(CASE WHEN factor_value IS NULL THEN 1 ELSE 0 END) n_null,
       COUNT(DISTINCT company_id) ncomp, MIN(date), MAX(date),
       MIN(factor_value), MAX(factor_value), AVG(factor_value)
FROM factor RIGHT JOIN factorvalue ON factor.id = factorvalue.factor_id
LEFT JOIN company ON factorvalue.company_id = company.id
WHERE exchange_name IN ('TWSE')
  AND factor_name IN ('gp_a','current_ratio','rd_s','inv_g')
GROUP BY factor_name;
```

- `n_null` 率應與 ROE/EPS 同級（約 14%），`INV_G` 會再高約 4 季的量（正常）
- `ncomp` 接近 1,775
- 合理範圍：`GP_A` 0~0.6｜`CURRENT_RATIO` 0.5~5｜`RD_S` 0~0.3｜`INV_G` −0.8~+2
- **沒有大量剛好是 0 的值**

---

## 附註：下游收到後要做的事（不需貼給 collector）

### 任務 C 完成後
1. `database.py` 的 `BENCHMARK_TABLE` 加一組 `taiex_tr` 的對應
2. `universe_benchmark.py::get_bench()` 加第三種 `kind="index_tr"`
3. **不需重跑任何回測**——基準只影響分析層的門檻，
   但 Phase 2 的 primary 門檻（單因子贏基準 2pp）會變 → 若改用報酬指數當主基準要重跑分析
4. 論文的基準章節改成三方對照：價格指數 4.71% / 報酬指數 ~8% / 自建宇宙 8.67%
   ——這三個數字擺在一起本身就是個好的說明素材

### 任務 D 完成後（成本較高，先評估）
1. `phase1_linearity.py` 的 `FACTORS` 加 4 個 → 重跑 Phase 1（台美各約 25 分鐘）
2. `phase1_analyze.py` 更新去留名單
3. Phase 2~4 全線重跑（openSec 變體，台股約 4~5 小時）
4. ⚠️ **多重檢定風險**：20→24 又多開 4 次檢定。跟老師的說法同任務 B：
   不是為了增加樣本，是補上台股原本因子池「遷就資料可得性」的缺口，
   且 Phase 1 線性檢定會當守門員
