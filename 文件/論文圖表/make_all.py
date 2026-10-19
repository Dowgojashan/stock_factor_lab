# -*- coding: utf-8 -*-
"""一次產出第三章四張圖（向量 PDF＋PNG 預覽）。

用法：
    cd 文件/論文圖表
    ../../.venv/Scripts/python.exe -X utf8 make_all.py

輸出（同目錄）：
    fig_overview.pdf/.png          總流程圖（1.1 與第三章開頭）
    fig_architecture.pdf/.png      圖一 系統架構圖
    fig_research_flow.pdf/.png     圖二-a 研究部建構流程
    fig_quarterly_loop.pdf/.png    圖二-b 每季監控決策迴圈

LaTeX：\\includegraphics[width=\\textwidth]{fig_architecture.pdf}
圖寬 6.3in，約等於 A4 內文寬；字型已嵌入（Type 42）。若內文寬不同，等比縮放即可。
"""
import fig_architecture
import fig_overview
import fig_quarterly_loop
import fig_research_flow

if __name__ == "__main__":
    print(fig_overview.make())
    c, _ = fig_architecture.make()
    print(c.save("fig_architecture"))
    print(fig_research_flow.make().save("fig_research_flow"))
    print(fig_quarterly_loop.make().save("fig_quarterly_loop"))
