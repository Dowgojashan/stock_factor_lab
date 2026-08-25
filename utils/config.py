from configparser import ConfigParser

class Config():

    def __init__(self):
        self._parser = ConfigParser()
        self._parser.read("../config.ini", encoding="utf-8")
        
    def get_database_config(self):
        database_config = {}
        database_config["host"]     = str(self._parser["database"]["host"])
        database_config["port"]     = int(self._parser["database"]["port"])
        database_config["user"]     = str(self._parser["database"]["user"])
        database_config["password"] = str(self._parser["database"]["password"])
        database_config["db"]       = str(self._parser["database"]["db"])
        database_config["charset"]  = str(self._parser["database"]["charset"])
        return database_config

    def get_openai_api_key(self):
        key = str(self._parser["openai"]["api_key"]).strip()
        if not key:
            raise RuntimeError(
                "config.ini 的 [openai] api_key 是空的；請去 "
                "https://platform.openai.com/api-keys 申請後填入")
        return key

    def get_openai_admin_key(self):
        """Admin API key（跟 api_key 不同種，需要 api.usage.read scope，
        用於 utils/openai_quota.py 主動查詢當日花費，見該模組 docstring）。
        """
        key = str(self._parser["openai"].get("admin_key", "")).strip()
        if not key:
            raise RuntimeError(
                "config.ini 的 [openai] admin_key 是空的；這把跟 api_key 不同種，"
                "須在 OpenAI org 設定裡另外產生、給 api.usage.read 權限")
        return key

    def get_openai_daily_quota_usd(self):
        """每日免費額度上限（美金）。OpenAI 沒有查詢額度上限的 API，
        這個數字只能由使用者自己告知、填進 config.ini。
        """
        raw = str(self._parser["openai"].get("daily_quota_usd", "")).strip()
        if not raw:
            raise RuntimeError(
                "config.ini 的 [openai] daily_quota_usd 是空的；"
                "OpenAI 沒有API可以查額度上限，須自己填入實際數字")
        return float(raw)

    def get_openai_model(self, purpose):
        """依用途查模型名稱（如 purpose="t10" 讀 [openai] t10_model）。
        不給預設值——模型名稱會過期，寫死猜測比明確要求填入更容易悄悄壞掉。
        """
        raw = str(self._parser["openai"].get(f"{purpose}_model", "")).strip()
        if not raw:
            raise RuntimeError(
                f"config.ini 的 [openai] {purpose}_model 是空的；"
                "模型名稱會過期，不寫死猜測，須自己指定要用哪個模型")
        return raw