# 程式設定
## Step 1 前置作業
* clone `https://github.com/QuenHengLee/stock_factor_lab` 到本地端
* 安裝 Python 3.8.10 (2025/5/2 最終可下載版)
* 安裝 TA-LIB `pip install <你的路徑>/TA-LIB/<選cp38的whl>`
* 下載 [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/zh-hant/visual-cpp-build-tools/)
* 勾選以下套件並重新啟動電腦
    1. MSVC v143 - VS 2022 C++ x64/x86 建置工具
    2. Windows 11 SDK（視系統版本而定）
    3. 適用於Windows C++ 的 CMake 工具 
    4. 測試工具的核心功能 - 建置工具
    5. C++ AddressSanitizer
* 將`Microsoft C++ Build Tools`等相關設定新增到系統環境變數 [參考](https://github.com/bycloudai/InstallVSBuildToolsWindows)
* 安裝 `requirements.txt` 前先確保 Python 版本是3.8
* 先將 `requirements.txt` 中的 `TA-LIB` 那一行拿掉，因為前面已經先裝過了
* 安裝 專案套件 `pip install -r requirements.txt`
* Terminal 執行以下指令
    ``` bash
    cd core
    python setup.py build_ext --inplace
* 執行過後會出現以下檔案
    > ![執行後應該長這樣](/img/after_build_backtest_core.pyx.png)

## Step2 資料庫相關
* 下載`xampp`
* 下載資料庫sql檔並放在同一個資料夾中
* 先建立好空的資料庫，如: `lab`
* 開啟`xampp control pannel` 點選 `Shell`
* 執行下列指令 
    ``` bash
    cd /d <你sql檔的資料夾路徑>
* 執行下列指令，`-p`表示DB password，如果沒有密碼就拿掉；`lab`表示建立好的DB名字
    ``` bash
    for %f in (*.sql) do "C:\xampp\mysql\bin\mysql.exe" -u root -p lab < "%f"
* 等所有sql檔都跑完就匯入成功
* 修改`config.ini`的資料庫連線資訊

## Step3 程式碼示範
* 先把所有程式碼中，所有`import`中有`iplab`都拿掉(因為目前都是本地端執行)
* `backtest.py` 中的 `from core import mae_mfe as maemfe` 改成`from core.backtest_core import mae_mfe as maemfe`
* 執行 `code`資料夾中的程式碼，先執行`example.ipynb`作為示範
* 先在最上面加入以下code，以防抓不到get_data等其他相關模組
    ``` python 
        import sys
        sys.path.insert(0, '../')