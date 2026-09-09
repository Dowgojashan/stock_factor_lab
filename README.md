# stock_factor_lab

量化因子選股回測系統（碩士論文相關研究）。

- **環境安裝**：看 [`安裝說明_SETUP.md`](安裝說明_SETUP.md)，或直接跑 `setup.bat`
- **研究內容與結果**：看 [`文件/研究框架總覽_v10.md`](文件/研究框架總覽_v10.md)
- **程式碼分類導覽**：看 [`code/README.md`](code/README.md)
- **協作／環境注意事項**：看 [`CLAUDE.md`](CLAUDE.md)

---

## 環境需求

> ⚠️ 2026-09-09 更正：舊版 README 寫的「用 Anaconda 建 python 3.8 環境、
> `pip install -r requirements.txt`」**已經過時**，照著做會裝錯環境。

- **Python 3.10**（TA-Lib 的 wheel 是 `cp310`，換版本要另外找對應 wheel）
- **虛擬環境用 `.venv`**（不是 conda），`setup.bat` 會自動建立
- **套件清單用 `requirements_clean.txt`**（不是 `requirements.txt`——舊的那份是 UTF-16
  編碼且 TA-Lib 那行有問題，理由見 `安裝說明_SETUP.md`）
- **資料庫**：MySQL 協定（目前是 MariaDB），連線設定放在 `config.ini`
  （gitignored，要自己從 `config.ini.example` 複製一份填）

```bat
setup.bat
```

裝完驗證：

```bash
cd code
python -m research.tests
```

（預期 131 項全過；若只過 130 項且失敗的是 `t_ops_t11_regime_label_valid`，
表示資料庫沒開，不是程式壞了。）

---

## 基礎框架語法（沿用學姊的回測框架）

根目錄的 `database.py`／`get_data.py`／`combinations.py`／`format_data.py` 等是繼承下來的
基礎框架，四大類用法如下：

| | 存取 raw data | dataframe 操作 | 回測 | 顯示 report |
|:---:|---|---|---|---|
| **功能** | 從資料庫取得股價（開高低收）與財務報表，存成 dataframe | 對 dataframe 做處理 | 回測模擬股票部位產生的淨值報酬率 | 策略回測基礎報告 |
| **語法** | 1. 建立 DB 連線<br>`data = Data()`<br>2. 取得資料<br>`data.get("你想取得的資料")` | 1. 移動平均<br>`df.average(n_windows)`<br>2. 買入訊號持續為 True 直到出場<br>`df.hold_until(exits)`<br>3. 依因子切群<br>`df.divide_slice(quantile)` | 1. 回測單一 position，回傳 report<br>`report = backtest.sim(position, resample)`<br>2. 回測多個 position，回傳 report dict<br>`report_conditions = sim_conditions(conditions, resample)`<br>（`conditions` 是選股條件集合，dict 結構）| 1. 單一 position 圖組<br>`report.display()`<br>2. CAGR／MDD／Sharpe<br>`report.get_status()`<br>3. 累計報酬比較<br>`report_collection.plot_creturns().show()`<br>4. 各種數據比較（柱狀圖／熱力圖）<br>`report_collection.plot_stats('bar').show()`<br>`report_collection.plot_stats('heatmap')` |
