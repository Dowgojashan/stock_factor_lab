# -*- coding: utf-8 -*-
"""圖二-a：研究部建構流程（一次性，每個窗次重建）。對應第三章 3.2、第四章 4.1／4.2。

每個步驟左上角有「步驟 N」標籤（2026-10-19 依使用者要求加入）。
漏斗數字取自論文大綱 4.1 候選池漏斗（2026-10-13 對照 strategy_marks／cluster_assign 查證）：
  候選池 TW 29,255／US 22,856；可用策略 27,389／21,502；納入分群 26,424／21,379／XM 47,926。
群數：TW 6／US 4／XM 7（`stage3_hrp.L1_TARGET`，輪廓係數決定）。
"""
from _figlib import C, Canvas


def make():
    W, H = 6.3, 7.95
    c = Canvas(W, H)

    cx = 3.0           # 主軸中心 x
    bw = 3.9           # 主軸框寬
    x0 = cx - bw / 2

    def spine(y_top, h, text, step, fc="#FFFFFF", bold=False, fs=6.6):
        c.box(x0, y_top - h, bw, h, text, fs=fs, fc=fc, bold=bold)
        c.badge(x0 + 0.36, y_top, step)
        return y_top - h

    y = H - 0.16
    y1 = spine(y, 0.50, "FCV 候選策略池\n台股 29,255 檔／美股 22,856 檔", "步驟 1", fc=C["fcv"], bold=True)
    c.arrow(cx, y1, cx, y1 - 0.22)
    y2 = spine(y1 - 0.22, 0.36, "格式整備：建立候選策略索引", "步驟 2")
    c.arrow(cx, y2, cx, y2 - 0.22)
    y3 = spine(y2 - 0.22, 0.62,
               "策略標記\n標記 passes_alpha_gate（全期間報酬過基準）與 is_usable（資料品質）\n"
               "可用策略：台股 27,389／美股 21,502", "步驟 3")
    c.arrow(cx, y3, cx, y3 - 0.22)
    y4 = spine(y3 - 0.22, 0.62,
               "納入分群的策略\n扣除共同窗內歷史不足、零變異數的策略\n"
               "台股 26,424／美股 21,379／跨市場 47,926", "步驟 4")
    c.arrow(cx, y4, cx, y4 - 0.22)
    y5 = spine(y4 - 0.22, 0.66,
               "HRP 階層分群（三棵主樹：台股／美股／跨市場）\n群數由輪廓係數客觀決定\n"
               "台股 6 群／美股 4 群／跨市場 7 群", "步驟 5", fc=C["res"])

    # 步驟 6：四條並行分析
    yb_top = y5 - 0.38
    bwid, gap = 1.40, 0.10
    xs = [0.12 + i * (bwid + gap) for i in range(4)]
    heads = [
        ("步驟 6a", "群定性\n定量特徵表 → 視覺化\n→ LLM 身份描述"),
        ("步驟 6b", "逐年賺賠與互補性\n群對相關係數、\n逐年報酬"),
        ("步驟 6c", "有效獨立賭注數\n（ENB）與市場 beta\n解釋比例"),
        ("步驟 6d", "四組對照實驗\nA_hrp／B_all／C_random／\nD／E ＋ 樣本內外與滾動驗證"),
    ]
    bh = 0.84
    yb = yb_top - bh
    for x, (st, t) in zip(xs, heads):
        c.box(x, yb, bwid, bh, t, fs=6.2, fc="#FFFFFF")
        c.badge(x + 0.36, yb_top, st)
    ybar = y5 - 0.19
    c.ax.plot([xs[0] + bwid / 2, xs[3] + bwid / 2], [ybar, ybar], lw=1.0, color=C["arrow"], zorder=4)
    c.ax.plot([cx, cx], [y5, ybar], lw=1.0, color=C["arrow"], zorder=4)
    for x in xs:
        c.arrow(x + bwid / 2, ybar, x + bwid / 2, yb_top, lw=1.0)

    # 步驟 7（四條匯入）
    ybar2 = yb - 0.20
    c.ax.plot([xs[0] + bwid / 2, xs[3] + bwid / 2], [ybar2, ybar2], lw=1.0, color=C["arrow"], zorder=4)
    for x in xs:
        c.ax.plot([x + bwid / 2, x + bwid / 2], [yb, ybar2], lw=1.0, color=C["arrow"], zorder=4)
    y7_top = ybar2 - 0.22
    c.arrow(cx, ybar2, cx, y7_top, lw=1.0)
    y7 = spine(y7_top, 0.50, "群內代表策略挑選\n固定配額（非按群大小比例）", "步驟 7", fc=C["res"])
    c.arrow(cx, y7, cx, y7 - 0.22)
    y8 = spine(y7 - 0.22, 0.50,
               "等權精選持倉\n（HRP 只負責分群、不配權重；各窗次以自身樣本內期間重新挑選）", "步驟 8",
               fc=C["res"])
    c.arrow(cx, y8, cx, y8 - 0.22)
    y9 = spine(y8 - 0.22, 0.46, "輸出：凍結產物＋各窗次代表策略名單\n→ 實戰應用層", "步驟 9",
               fc=C["app"], bold=True)

    # 右側章節對照（灰字）
    rx = x0 + bw + 0.10

    def ref(ytop, h, text):
        c.ax.text(rx, ytop - h / 2, text, fontsize=5.6, color=C["mute"], va="center", ha="left",
                  zorder=6)

    ref(H - 0.16, 0.50, "第四章 4.1")
    ref(y3 + 0.62, 0.62, "第一章 1.1\n（策略標記步驟）")
    ref(y4 + 0.62, 0.62, "第四章 4.1\n候選池漏斗")
    ref(y5 + 0.66, 0.66, "第三章 3.2\n第四章 4.2.1")
    for x, t in zip(xs, ["4.2.1(b)", "4.2.1(c)", "4.2.1(a)", "4.2.2"]):
        c.ax.text(x + 0.02, yb - 0.03, t, fontsize=5.4, color=C["mute"], ha="left", va="top")
    ref(y7 + 0.50, 0.50, "第三章 3.2\n「從群結構到組合決斷」")

    c.box(0.12, 0.10, W - 0.24, 0.22,
          "凍結鏈：每階段產出附雜湊（改上游須連鎖重跑下游）；LLM 僅用於群身份與互補解釋，不參與選股與權重計算",
          fs=5.8, fc="#FFFFFF", style="dashed", minfs=4.8)
    return c


if __name__ == "__main__":
    print(make().save("fig_research_flow"))
