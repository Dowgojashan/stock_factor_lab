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
                    AND date > 2018-01-01"
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
                    AND date > 2018-01-01"
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