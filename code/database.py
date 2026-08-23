import configparser
from utils.config import Config
from dataframe import CustomDataFrame

import pymysql
import pandas as pd


class Database:
    # 市場 → 交易所清單（company.exchange_name）
    MARKET_EXCHANGES = {
        "TW": ["TWSE"],
        "US": ["NASDAQ", "NYSE", "AMEX"],
    }
    # 市場 → 大盤基準所在的資料表
    BENCHMARK_TABLE = {
        "TW": "taiex",
        "US": "sp500",
    }
    # 市場 → 資料起點（回測窗）。None＝不設下界（台股保留 2000+ 全歷史）。
    # 價格用交易日 date 過濾；財報用公告日 filing_date 過濾，並留 2 年緩衝，
    # 以保留跨窗初期 ffill / 季度回看所需的「最近一筆」財報。
    # 美股乾淨基本面(有 filing_date)實測深至 1994/1986（見 collector UPDATE_US.md §0.5），
    # 故回測窗放深至 2000（價格起點 2000、財報起點 1998＝2 年緩衝）。
    MARKET_PRICE_START = {"TW": None, "US": "2000-01-01"}
    MARKET_REPORT_START = {"TW": None, "US": "1998-01-01"}

    def __init__(self, market="TW"):
        self.market = str(market).upper()
        if self.market not in self.MARKET_EXCHANGES:
            raise ValueError(
                f"未知市場 market={market}，目前支援 {list(self.MARKET_EXCHANGES)}"
            )
        self._config = Config()
        self._db_data= self._config.get_database_config()
        # self.connection()

    def _exchange_in_clause(self):
        """依市場產生 exchange_name 的 IN 條件字串（值來自受信任的內建清單）。"""
        exch = "','".join(self.MARKET_EXCHANGES[self.market])
        return f"exchange_name IN ('{exch}')"

    def _start_clause(self, column, start):
        """回傳日期下界 SQL 片段；start 為 None 時不過濾（台股保留全歷史）。
        注意：原本 `AND date > 2018-01-01`（未加引號）是整數算術 2018-1-1=2016，
        與日期(轉 YYYYMMDD 數值)比較後恆真＝過濾失效。此處改為正確的字串比較。"""
        return f" AND {column} >= '{start}'" if start else ""

    def _universe_clause(self, symbol_col="company_symbol"):
        """美股：把宇宙圈定在 Russell 3000 成分表（論文美股宇宙＝Russell 3000）。
        成分表 russell3000 由 collector 端維護（見 stock_factor_collector/UPDATE_US.md §0.6），
        symbol 已正規化為破折號格式，與 company_symbol 對齊。
        台股回傳空字串＝SQL 完全不變（保持全歷史、byte-identical 隔離）。"""
        if self.market == "US":
            return f" AND {symbol_col} IN (SELECT symbol FROM russell3000)"
        return ""

    """
    stock_index = {open, high, low, close, volume, market_capital}
    """

    # 建立與DB的連線
    def create_connection(self):
        # 檢查DB版本&連線成功
        try:
            config_host = self._db_data["host"]
            config_port = int(self._db_data["port"])
            config_user = self._db_data["user"]
            config_password = self._db_data["password"]
            config_db = self._db_data["db"]
            config_charset = self._db_data["charset"]

            db = pymysql.connect(
                host=config_host,
                port=config_port,
                user=config_user,
                passwd=config_password,
                db=config_db,
                charset=config_charset,
            )
            return db
        except Exception as e:
            print(e)
            print("無法連結資料庫")
            return e

    # 取的公司的開高低收(stock)
    def get_daily_stock(self):
        try:
            db = self.create_connection()
            cursor = db.cursor()
            # data = cursor.fetchone()
            # print('連線成功')

            # 依市場選取（台股 TWSE / 美股 NASDAQ,NYSE,AMEX）
            sql = f" SELECT company_symbol,name,date,open,high,low,close,volume,market_capital \
                    FROM company RIGHT JOIN stock ON company.id = stock.company_id \
                    WHERE {self._exchange_in_clause()}\
                    {self._universe_clause('company_symbol')}\
                    {self._start_clause('date', self.MARKET_PRICE_START[self.market])}"
            # AND company_symbol>8700
            # AND company_symbol<9000"

            cursor.execute(sql)
            data = cursor.fetchall()
            columns = [
                "company_symbol",
                "name",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "market_capital",
            ]
            df = CustomDataFrame(data, columns=columns)
            # print(df)
            return df

        except Exception as e:
            print(e)
            print("無法執行SQL語法")
            return e

    # 取得公司的財報(factorvalue)
    def get_finance_report(self):
        try:
            db = self.create_connection()
            cursor = db.cursor()
            # 多帶 filing_date（公告日）：台股為 NULL、美股有值，供前瞻防護分流
            sql = f" SELECT date, company_symbol, factor_name, factor_value, filing_date \
                    FROM factor RIGHT JOIN factorvalue ON factor.id = factorvalue.factor_id  \
                    LEFT JOIN  company ON factorvalue.company_id = company.id \
                    WHERE {self._exchange_in_clause()}\
                    {self._universe_clause('company_symbol')}\
                    {self._start_clause('filing_date', self.MARKET_REPORT_START[self.market])}"
            # AND company_symbol>8700
            # AND company_symbol<9000"

            cursor.execute(sql)
            data = cursor.fetchall()
            columns = ["date", "company_symbol", "factor_name", "factor_value", "filing_date"]
            df = CustomDataFrame(data, columns=columns)
            # print('The raw data get from database:\n')
            # print(df)
            return df

        except Exception as e:
            print(e)
            print("無法執行SQL語法")
            return e

    # 取得大盤基準（台股 taiex / 美股 sp500）
    def get_taiex_data(self):
        try:
            db = self.create_connection()
            cursor = db.cursor()
            table = self.BENCHMARK_TABLE[self.market]
            sql = f" SELECT date, open, high, low, close, volume, market_capital FROM {table}"

            cursor.execute(sql)
            data = cursor.fetchall()    
            columns = ["date", "open", "high", "low", "close", "volume", "market_capital"]
            df = CustomDataFrame(data, columns=columns)
            # print('The raw data get from database:\n')
            # print(df)
            return df

        except Exception as e:
            print(e)
            print("無法執行SQL語法")
            return e
if __name__ == "__main__":
    db = Database()
    db.create_connection()

    daily = db.get_daily_stock()
    # daily.to_csv('./OutputFile/daily.csv')

    # finance = db.get_finance_report()
    # 使用布林索引過濾 DataFrame，擷取 "加工業" 種類的收盤價資料
    filtered_df = daily[daily["company_symbol"] == "8905"]
    filtered_df.set_index("date", inplace=True)
    # 升序排序日期索引
    filtered_df = filtered_df.sort_index(ascending=True)

    filtered_df.to_csv("./OutputFile/filtered_df.csv")

    # db.select_index('open')