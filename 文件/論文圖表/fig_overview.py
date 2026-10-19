# -*- coding: utf-8 -*-
"""總流程圖：FCV 策略引擎 → 研究部 → 實戰應用層（極簡版，用於 1.1 與第三章開頭）。"""
from _figlib import C, Canvas


def make():
    W, H = 6.3, 2.75
    c = Canvas(W, H)

    bw, bh, y0 = 1.55, 1.25, 0.82
    xs = [0.12, 2.375, 4.63]          # 三個大框的左緣
    titles = ["FCV 策略引擎", "研究部", "實戰應用層"]
    fcs = [C["fcv"], C["res"], C["app"]]
    bodies = [
        "單因子健檢\n↓\nF1×F2 配對\n↓\n加動態條件 C\n↓\n加估值濾網 V\n（台股、美股各自獨立）",
        "策略標記\n↓\nHRP 階層分群\n↓\n群定性與驗證\n↓\n代表策略挑選",
        "監控（每季）\n↓\n動作與風控\n↓\n解釋\n↓\n稽核紀錄",
    ]
    for x, t, fc, b in zip(xs, titles, fcs, bodies):
        c.group(x, y0, bw + 0.08, bh + 0.5, t, fc=fc, fs=9.0, title_h=0.26)
        c.box(x + 0.06, y0 + 0.07, bw - 0.04, bh + 0.08, b, fs=6.8, fc="#FFFFFF", minfs=5.2)

    # 箭頭與產物標籤
    ymid = y0 + (bh + 0.5) / 2 - 0.05
    c.arrow(xs[0] + bw + 0.08, ymid, xs[1], ymid, lw=1.4)
    c.arrow(xs[1] + bw + 0.08, ymid, xs[2], ymid, lw=1.4)
    c.label((xs[0] + bw + 0.08 + xs[1]) / 2, ymid + 0.45, "候選\n策略池", fs=6.4, bold=True)
    c.label((xs[1] + bw + 0.08 + xs[2]) / 2, ymid + 0.45, "精選\n持倉", fs=6.4, bold=True)
    c.label((xs[0] + bw + 0.08 + xs[1]) / 2, ymid - 0.42, "TW 29,255\nUS 22,856", fs=5.6,
            color=C["mute"])

    # 底部：樣本外監控與修正橫帶（涵蓋研究部＋實戰層）
    x_band = xs[1]
    w_band = xs[2] + bw + 0.08 - xs[1]
    c.box(x_band, 0.36, w_band, 0.26,
          "樣本外（Out-of-Sample）監控與修正：每個窗次以自身樣本內期間重新選代表策略，逐季滾動更新",
          fs=6.4, fc="#FFFFFF", style="dashed", minfs=5.0)
    # LLM 註記
    c.box(0.12, 0.06, W - 0.24, 0.22,
          "LLM 只負責解釋（群身份描述、決策建議與解釋文字）；持股、權重與報酬數字一律由程式計算，LLM 不參與",
          fs=6.2, fc=C["llm"], minfs=5.0)
    return c.save("fig_overview")


if __name__ == "__main__":
    print(make())
