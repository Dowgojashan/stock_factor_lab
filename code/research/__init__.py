# -*- coding: utf-8 -*-
"""研究部管線（階段 0→1→2→3→4）。

設計文件：`系統設計文件_v1.md`（SDD v1.1）
落差處理：`落差處理方案_v1.md`

⚠️ 路徑陷阱（沿用專案既有慣例，見 CLAUDE.md §4）：
  - 本套件所有模組於 import 時透過 `paths.py` 掛好 sys.path，
    之後 `from database import Database` 才找得到根目錄的模組。
  - 指令碼一律在 `code/` 目錄下執行，否則 config.ini 相對路徑讀不到 [database] 區段。
"""
from . import paths  # noqa: F401  # import 副作用：掛 sys.path，必須最先

__all__ = ["paths", "contracts", "freeze"]
