# 因子候選批次實驗：F、C 因子定義記錄

> 對應程式：`code/run_factor_batches.py`（批次driver）、`code/sweep_config.py`（F1/C 定義來源）、
> `code/analyze_batch.py`（每批圖表分析）。
> 方法論對照學姊論文 3.6.1 節「發表用因子確定前，設計階段測試多項候選因子」。

## 1. 實驗設計概要

每個候選因子各自跑一批獨立的回測 + 20張圖表分析（`TW_batch_{候選}_M` / `US_batch_{候選}_M`），
**10批彼此獨立、不合併成一個大池**——每批都是一次完整的「固定三因子 + 1候選因子」測試。

> 原本規劃11個候選因子（含PE），2026-07-29與老師開會確認後**移除PE**，詳見§2.2下方說明。

單批展開規模：F1×F2組合（~170）× C狀態（20個C + None基準 = 21）× V（v0/v1）≈ **7,140個策略**。

## 2. F（體質）因子

### 2.1 固定三因子（每批都在）

來自 `sweep_config.py::COMMON_FACTORS`：

| 因子 | DB欄位 |
|---|---|
| ROE | report:roe |
| EPS | report:eps |
| FCF_P | report:fcf_p（自由現金流殖利率） |

### 2.2 候選因子（現行10個，每批只加1個）

**選用理由**：不是依財務理論精選，而是資料庫實際可用的因子欄位（`fcv_backtest.ipynb`已驗證過的
`factor_name`清單）扣掉固定三因子後的全部剩餘欄位。對應論文3.6.1節候選因子測試的精神，但範圍
受限於這個資料庫實際收集齊全的欄位，不是論文原始30餘項的完整重現。

| 分類 | 因子 |
|---|---|
| 估值倍數 | EV_EBITDA、EV_S、PB（股價淨值比）、PS（股價營收比）、P_IC（股價/投入資本） |
| 資本報酬/獲利能力 | ROIC（投入資本報酬率）、CROIC（現金投入資本報酬率） |
| 現金流品質 | FCF_OI（自由現金流/營業利益）、OCF_E（營運現金流/盈餘） |
| 動量 | MOM |

**PE 已排除（2026-07-29，與老師開會確認）**：PE同時是候選因子(F)、又是V構面估值濾網
（`fcv_core.py::get_v_mask`預設抓`report:pe`）的依據，兩邊都用PE會重複探索同一個訊號，
故拔除F裡的PE、保留V的PE。原本`TW_batch_PE_M`的回測+圖表資料本身有效，已搬移至
`_archive/TW_PE_excluded_F_V_overlap/`（含說明），其餘10批完全不受影響、不需重跑。

### 2.3 分桶方式

每個F因子用 `q_band`（橫斷面分位，每期依全市場排名）切成 **N=5桶**：

0-20%（最低）、20-40%、40-60%、60-80%、80-100%（最高）。

只用N=5一種粒度（不混N=10），避免圖表軸上N混雜、桶邊界不對齊的問題。

### 2.4 F1×F2 展開

每批4個因子（固定3+候選1）× 5桶 = 20個「F1單層」條件；F1、F2兩兩配對、無序去重
（`ROE×EPS`與`EPS×ROE`只算一次），單層+跨因子對約170組。

## 3. C（動態）因子

固定綁在**三個固定因子**上（候選因子不衍生C），來自 `sweep_config.py::C_TYPES`，
每因子7種型別、共 3×7−1=20 個（FCF_P排除`ytdfull`一種，`C_SKIP`明訂）。

### 3.1 型別定義

| 型別代號 | condition_factory 型別 | 意義 |
|---|---|---|
| riseq1 | rise_q(1) | 較上一季上升 |
| riseq2 | rise_q(2) | 較前兩季上升 |
| qmax4 | is_highest_q(4) | 近4季最高 |
| qmax8 | is_highest_q(8) | 近8季最高 |
| yoy | yoy_gt(4) | 年增（YoY）為正 |
| ytdsame | ytd_avg_gt_prev_year_same_period_avg | 今年至今均 > 去年「同期」均 |
| ytdfull | ytd_avg_gt_prev_year_avg | 今年至今均 > 去年「全年」均 |

概念分組：「短期上升趨勢」（riseq1/riseq2）、「創新高」（qmax4/qmax8）、
「年增/趨勢確認」（yoy/ytdsame/ytdfull）——確認F因子選出的股票體質是否持續改善，而非只看單季靜態值。

### 3.2 完整20個C清單（含編號，對應圖表/策略名稱）

**ROE動態條件（C1-C7）**

| 編號 | 條件 |
|---|---|
| C1_ROE_DYN_riseq1 | ROE較上一季上升 |
| C2_ROE_DYN_riseq2 | ROE較前兩季上升 |
| C3_ROE_DYN_qmax4 | ROE為近4季最高 |
| C4_ROE_DYN_qmax8 | ROE為近8季最高 |
| C5_ROE_DYN_yoy | ROE年增（YoY）為正 |
| C6_ROE_DYN_ytdavg_gt_lyytdavg | ROE今年至今均 > 去年同期均 |
| C7_ROE_DYN_ytdavg_gt_lyavg | ROE今年至今均 > 去年全年均 |

**EPS動態條件（C8-C14）**

| 編號 | 條件 |
|---|---|
| C8_EPS_DYN_riseq1 | EPS較上一季上升 |
| C9_EPS_DYN_riseq2 | EPS較前兩季上升 |
| C10_EPS_DYN_qmax4 | EPS為近4季最高 |
| C11_EPS_DYN_qmax8 | EPS為近8季最高 |
| C12_EPS_DYN_yoy | EPS年增（YoY）為正 |
| C13_EPS_DYN_ytdavg_gt_lyytdavg | EPS今年至今均 > 去年同期均 |
| C14_EPS_DYN_ytdavg_gt_lyavg | EPS今年至今均 > 去年全年均 |

**FCF_P動態條件（C15-C20，少一種）**

| 編號 | 條件 |
|---|---|
| C15_FCF_P_DYN_riseq1 | FCF Yield較上一季上升 |
| C16_FCF_P_DYN_riseq2 | FCF Yield較前兩季上升 |
| C17_FCF_P_DYN_qmax4 | FCF Yield為近4季最高 |
| C18_FCF_P_DYN_qmax8 | FCF Yield為近8季最高 |
| C19_FCF_P_DYN_yoy | FCF Yield年增（YoY）為正 |
| C20_FCF_P_DYN_ytdavg_gt_lyytdavg | FCF Yield今年至今均 > 去年同期均 |

FCF_P沒有「今年至今均 > 去年全年均」版本（`C_SKIP = {("FCF_P", "ytdfull")}`排除）。

另外每批圖表都會加一個 **None基準**（不套用任何C條件，只有F×V），共21種C狀態供對照。

## 4. V（估值）構面

來自 `fcv_core.py::MarketData.get_v_mask`，用PE的近4季滾動均值/最低值當估值濾網：

- **V0**：不套用，原始F×C策略
- **V1**：額外要求「PE低於近4季均值，但高於近4季最低點」——相對自己歷史便宜、但非最谷底的估值濾網

每個F×C組合都會產生v0、v1兩個版本。

## 5. 展開總量計算

```
單批策略數 = F1×F2組合(~170) × C狀態(20個C + None = 21) × V(2) ≈ 7,140
```

實際回測數依 `fcv_core.py::MIN_TRADES` 品質濾網（低於5次交易的策略剔除）略有增減：

- 台股：多數批次6,800~7,100（MOM因子因分桶樣本少，僅4,203）
- 美股：目前10個候選因子皆因`filing_date`資料缺口而無效（原本PE批次唯一有效，
  但PE已排除），詳見 `_archive/US_invalid_filing_date_gap/README.md`；需等美股資料源
  補上這10個因子的`filing_date`後才有真正可用的美股候選因子結果。

## 6. 批次清單

`code/run_factor_batches.py::CANDIDATES`（PE已排除，見§2.2）：

```
EV_EBITDA, EV_S, CROIC, FCF_OI, ROIC, PB, PS, P_IC, OCF_E, MOM
```

每個候選各自一批（`{market}_batch_{候選}_M`），共用同一次`MarketData`資料載入以節省時間，
但backtest與圖表分析都各自獨立進行、獨立產出。
