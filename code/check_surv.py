# -*- coding: utf-8 -*-
# 診斷：美股財報母體是否含 delisted / 移出成分股（倖存者偏誤檢查）
# 用法：MySQL 開機後，於 code/ 目錄執行  ../.venv/Scripts/python.exe -u check_surv.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from database import Database

db = Database(market="US")
conn = db.create_connection()
if not hasattr(conn, "cursor"):
    print("無法連線 MySQL（請先開機）。"); sys.exit(1)
cur = conn.cursor()

US = "company.exchange_name IN ('NASDAQ','NYSE','AMEX')"

cur.execute(f"""
    SELECT COUNT(DISTINCT company_symbol)
    FROM factorvalue LEFT JOIN company ON factorvalue.company_id = company.id
    WHERE {US}
""")
print("US 財報公司數(distinct):", cur.fetchone()[0])

cur.execute(f"""
    SELECT last_year, COUNT(*) FROM (
        SELECT company_symbol, YEAR(MAX(filing_date)) AS last_year
        FROM factorvalue LEFT JOIN company ON factorvalue.company_id = company.id
        WHERE {US} AND filing_date IS NOT NULL
        GROUP BY company_symbol
    ) t GROUP BY last_year ORDER BY last_year
""")
print("\n各公司『最後 filing_date 年度』分布（有 2019~2023 停更=含移出/下市；幾乎全 2025=現存成分,有倖存者偏誤）:")
for yr, n in cur.fetchall():
    print(f"  最後公告年={yr}: {n} 家")

cur.execute(f"""
    SELECT first_year, COUNT(*) FROM (
        SELECT company_symbol, YEAR(MIN(filing_date)) AS first_year
        FROM factorvalue LEFT JOIN company ON factorvalue.company_id = company.id
        WHERE {US} AND filing_date IS NOT NULL
        GROUP BY company_symbol
    ) t GROUP BY first_year ORDER BY first_year
""")
print("\n各公司『最早 filing_date 年度』分布:")
for yr, n in cur.fetchall():
    print(f"  最早公告年={yr}: {n} 家")

cur.close(); conn.close()
