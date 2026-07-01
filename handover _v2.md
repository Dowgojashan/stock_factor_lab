# 程式設定
## Step 1 前置作業
* clone `https://github.com/QuenHengLee/stock_factor_lab` 到本地端
* 下載 [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/zh-hant/visual-cpp-build-tools/) 
* 勾選以下套件並重新啟動電腦
    1. MSVC v143 - VS 2022 C++ x64/x86 建置工具
    2. Windows 11 SDK（視系統版本而定）
    3. 適用於Windows C++ 的 CMake 工具 
    4. 測試工具的核心功能 - 建置工具
    5. C++ AddressSanitizer
* 將`Microsoft C++ Build Tools`等相關設定新增到系統環境變數 [參考](https://github.com/bycloudai/InstallVSBuildToolsWindows)
* 安裝 TA-LIB到TA-LIB資料夾中
    1. [前往此網址]https://github.com/cgohlke/talib-build/releases
    2. 安裝對應版本的whl檔到TA-LIB資料夾中 (假設是3.10版本就下載ta_lib-0.6.3-cp310-cp310-win_amd64.whl)
    3. 更改專案資料夾中`requirement.txt`的內容，`TA-Lib @ file:///D:/<路徑>/TA-LIB/<檔案全名.whl>`
* 安裝 Python 3.7 ~ 3.11版本(這是由於Ta-Lib、finlab library只有支援到這些版本，使用python3.12版finlab那邊會有問題)，強烈建議一起安裝python launcher
* (可選，使用anaconda可跳過)使用vs code建立虛擬環境並安裝requirement   Hint:此步驟需要透過python launcher(py)指令完成
    1. 使用vs code開啟專案folder
    2. 在terminal中key入 `py -0p`，看你本機目前有的環境 [參考](https://magicjackting.pixnet.net/blog/post/225113189)
    3. 建立虛擬環境: key入 `py -<版本> -m venv .venv`
    4. 啟動虛擬環境: 使用vs code中的Command Prompt，鍵入`.venv\scripts\activate`，看到(.venv)代表完成此步驟
    6. 安裝 專案套件 `pip install -r requirements.txt`
    7. Command Prompt執行以下指令
        ``` bash
        cd core
        python setup.py build_ext --inplace
    8. 執行過後會出現以下檔案
        > ![執行後應該長這樣](/img/after_build_backtest_core.pyx.png)

## Step2 資料庫相關
* 下載`xampp`
* 下載資料庫sql檔並放在同一個資料夾中 [SQL檔](https://drive.google.com/drive/u/1/folders/1cCkc6JRLiXMEBUER7l8mhC0n7x26L78O)
* 先建立好空的資料庫，如: `lab`
* 更改每個sql檔以下的內容
    1. CREATE DATABASE  IF NOT EXISTS `<資料庫名稱>` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci */ /*!80016 DEFAULT ENCRYPTION='N' */;
    2. USE `<資料庫名稱>`;
* 開啟`xampp control pannel` 點選 `Shell`
* 執行下列指令 
    ``` bash
    cd /d <你sql檔的資料夾路徑>
* 執行下列指令，`-p`表示DB password，如果沒有密碼就拿掉；`lab`表示先前建立好的DB名字
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