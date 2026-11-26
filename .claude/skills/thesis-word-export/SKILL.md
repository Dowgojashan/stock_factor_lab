---
name: thesis-word-export
description: 把論文（LaTeX，`論文/`）的指定章節連同參考文獻與附錄匯出成 Word（.docx），並用 Word 轉成 PDF 檢查版面。當使用者說「匯出 Word」「轉成 docx」「把第 X 章變成 Word 檔」「給老師／Kevin 看的 Word 版」時使用。工具已寫好，不要重寫或到處找舊的轉檔程式。
---

# 論文匯出 Word

工具在 `論文/tools/word_export/`（2026-11-24 由暫存區的舊程式整理而成）：

| 檔案 | 作用 |
|---|---|
| `export_docx.py` | 主程式：檢查 PDF 是否最新 → 產生圖片 → 轉檔 → 用 Word 檢查 |
| `docx_core.py` | 轉檔核心：直接寫 OOXML（環境沒有 python-docx、pandoc、node） |
| `render_assets.py` | 把 TikZ 圖、matplotlib 的 PDF 圖與公式轉成 PNG，存在 `_assets/` |

⚠️ `論文/` 整個被 git 忽略，**這些工具只存在本機**，換機器時要連同 `論文/` 一起手動複製。

## 一、步驟

1. **先編譯論文**（`thesis-results-writing` 第 18 條：`xelatex -no-pdf` 兩三次，再 `xdvipdfmx main.xdv`）。轉檔所用的節號、表號、圖號取自 `main.aux`，參考文獻取自 `main.pdf`；兩者比任何章節 .tex、圖的 .tex 或 `references.bib` 舊時，程式會直接停下並列出較新的檔案。
2. **執行**（在 repo 根目錄，用專案的 Python）：
   ```
   PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe 論文/tools/word_export/export_docx.py 4 --refs --app A B C D --out 第四章與附錄.docx --check
   ```
   - 章號 `1`–`5`（或檔名，如 `ch04_validation`），可列多章：`1 2 3`；
   - `--refs`：附參考文獻，只列這幾章與附錄實際引用的文獻，順序與格式照論文 PDF（APA）；
   - `--app A B C D`：附錄；
   - `--out`：檔名，存在 `論文/` 下（既有：`第一至三章與附錄.docx`、`第四章與附錄.docx`）；
   - `--check`：用 Word 轉成 PDF，存在 `_assets/<檔名>_check.pdf`，並回報頁數。
   產生圖片要編譯每張 TikZ 圖與公式，第四章加附錄約需 1 至 2 分鐘。
3. **看輸出訊息**：
   - `leftover latex: []`、`stray $ or braces: 0 0 0`：沒有殘留的 LaTeX 指令；不是空的，就要在 `docx_core.resolve()` 或 `textclean()` 補上該指令的處理；
   - `references: N | missing: []`：`missing` 不是空的，代表該文獻在 PDF 的參考文獻中找不到（通常是論文沒重編譯）；
   - `截掉未撰寫的部分：… 自「…」起`：尚有 `\pending{}` 的節會從該 `\section` 起整段截掉，這是正常的；
   - `table without caption`：附錄 B 的提示詞長表本來就沒有標題，可忽略。
4. **目視檢查**（用 fitz 把 `_check.pdf` 的頁面拼成縮圖，有疑問的頁面放大看）。重點：
   - 章首與含 TikZ 的圖：圖中的節號應為實際編號（程式會先把 `\ref` 換成 `main.aux` 的編號），不可出現「??」；
   - 多層表頭的表（`\multicolumn`、`\cmidrule`）、分組列（`\multicolumn{8}{l}{台股}` 與 `\quad` 縮排）；
   - 參考文獻頁、附錄的公式圖。
5. **回報使用者**：輸出路徑與頁數、包含哪些章節、哪些節因未撰寫而截掉、參考文獻篇數、圖與公式數，以及第二節的已知限制。

## 二、已知限制（回報時照實說明）

- 圖與公式在 Word 中是圖片，不能直接修改；要改就改論文原始檔再匯出。
- 章節標題是粗體字，不是 Word 的「標題 1／2／3」樣式，所以導覽窗格不會列出章節。
- 表格採直接格式（三線表框線），欄寬由程式依內容估計；文字欄在表太寬時會被壓縮而折行（實例：表 20「排名依據」欄）。小表與標題保持同頁，長表可跨頁並重複表頭。
- Word 的分頁與 LaTeX PDF 不同；頁碼置中。
- 版面：A4，左邊界 1701、右 1417 twips；中文標楷體、英數 Times New Roman、12 號字。

## 三、出問題時

- `main.aux/main.pdf 比下列檔案舊`：照第 1 步重新編譯。
- `label not in main.aux`：TikZ 圖引用了不存在或尚未編譯的標籤，重新編譯論文。
- `<圖名> failed`：TikZ 圖單獨編譯失敗，看錯誤訊息末段；`render_assets.PRE` 只載入 tikz 的 `arrows.meta, positioning, calc, fit`，圖用到其他函式庫要加進去。
- `\pending{} outside any \section`：章首（第一個 `\section` 之前）還有待寫註記，先寫完或移除。
- `PermissionError` 寫不進 .docx：該檔正在 Word 中開著，請使用者先關閉。
- Word 檢查卡住：工作管理員結束 `WINWORD.EXE` 後重跑。
- 論文用了新的 LaTeX 寫法（例如新的表格指令），轉檔出現殘留：在 `docx_core.py` 補處理（文字指令在 `resolve()`，表格在 `parse_table()`），改完重跑並重新目視檢查。修改前先複製一份 `docx_core.py` 備份。
