# -*- coding: utf-8 -*-
"""圖二-b：每季監控決策迴圈（實戰應用層）。對應第三章 3.3.1~3.3.5、第四章 4.3~4.6。

每個步驟左上角有「步驟 N」標籤（2026-10-19 依使用者要求加入）。
內容依據（2026-10-19 讀碼核對）：
- 預測區只用狀態變數、回顧區才可用流量變數（simulate.py：prospective_facts／retrospective_facts 分開組裝）
- 決策 enum 共 7 個動作（agents.py）；流量變數只能驅動不改持股的 3 個
- 動作生效有一季延遲（simulate.py：apply_w2c 取自上一季決策且上一季觸發）
- 條件式市值傾斜／覆蓋率傾斜／代表策略置換不持續；權重方案切換持續
- 回顧型診斷含基準鏈歸因分解（diagnose.baseline_chain_decomposition，facts_lean 每季呼叫）
"""
from _figlib import C, Canvas

GRAY_BADGE = "#7A7A7A"


def make():
    W, H = 6.3, 8.5
    c = Canvas(W, H)
    cx = 3.15
    top = H - 0.18

    # --- 輸入 ---
    c.box(0.55, top - 0.42, 2.45, 0.42, "精選持倉\n（來自研究部，等權）", fs=6.4, fc=C["res"], bold=True)
    c.box(3.30, top - 0.42, 2.45, 0.42, "市場資料\n（資料庫：價格、市值）", fs=6.4, fc=C["fcv"], bold=True)
    c.badge(0.55 + 0.28, top, "輸入", w=0.40, fc=GRAY_BADGE)
    c.badge(3.30 + 0.28, top, "輸入", w=0.40, fc=GRAY_BADGE)
    yi = top - 0.42
    c.ax.plot([1.775, 1.775, cx], [yi, yi - 0.12, yi - 0.12], lw=1.0, color=C["arrow"], zorder=4)
    c.ax.plot([4.525, 4.525, cx], [yi, yi - 0.12, yi - 0.12], lw=1.0, color=C["arrow"], zorder=4)

    # 步驟 1 三層指標
    y1_top = yi - 0.28
    c.arrow(cx, yi - 0.12, cx, y1_top)
    c.box(1.05, y1_top - 0.62, 4.2, 0.62,
          "三層指標計算（每季）\n環境層（大盤集中度、市場廣度）／過程層（持股結構）／結果層（報酬、超額）\n"
          "每個指標標明：狀態變數（現在長什麼樣）或流量變數（一段期間實現的結果）",
          fs=6.0, fc="#FFFFFF")
    c.badge(1.05 + 0.36, y1_top, "步驟 1")
    y1 = y1_top - 0.62

    # --- 兩條平行路徑 ---
    ybar = y1 - 0.19
    c.ax.plot([cx, cx], [y1, ybar], lw=1.0, color=C["arrow"], zorder=4)
    lx, rx_, lw_ = 0.34, 3.20, 2.78
    c.ax.plot([lx + lw_ / 2, rx_ + lw_ / 2], [ybar, ybar], lw=1.0, color=C["arrow"], zorder=4)
    lane_top = ybar - 0.19
    lane_h = 1.92
    c.arrow(lx + lw_ / 2, ybar, lx + lw_ / 2, lane_top)
    c.arrow(rx_ + lw_ / 2, ybar, rx_ + lw_ / 2, lane_top)
    c.group(lx, lane_top - lane_h, lw_, lane_h, "預測區（僅狀態變數；確定性門檻、不等 LLM）",
            fc="#F7F7F7", fs=6.2, title_h=0.22)
    c.group(rx_, lane_top - lane_h, lw_, lane_h, "回顧區（可用流量變數；回答「為什麼」）",
            fc="#F7F7F7", fs=6.2, title_h=0.22)
    inner_top = lane_top - 0.36
    # 左：步驟 2a／3a
    c.box(lx + 0.08, inner_top - 0.78, lw_ - 0.16, 0.78,
          "觸發型監控（p75 觀察／p90 觸發，雙門檻遲滯）\n指數集中度、領漲族群覆蓋率、\n回撤觸發、策略群集中度",
          fs=6.0, fc="#FFFFFF")
    c.badge(lx + 0.08 + 0.36, inner_top, "步驟 2a")
    c.arrow(lx + lw_ / 2, inner_top - 0.78, lx + lw_ / 2, inner_top - 0.78 - 0.17)
    ya_top = inner_top - 0.78 - 0.17
    c.box(lx + 0.08, ya_top - 0.46, lw_ - 0.16, 0.46,
          "預測性解釋 Agent（LLM）\n只讀狀態變數事實", fs=6.0, fc=C["llm"])
    c.badge(lx + 0.08 + 0.36, ya_top, "步驟 3a")
    # 右：步驟 2b／3b
    c.box(rx_ + 0.08, inner_top - 0.78, lw_ - 0.16, 0.78,
          "回顧型診斷（含基準鏈歸因分解）\n持股廣度、相對候選池表現、候選池整體衰減、\n產業集中度、績效異常無法歸因／隨機波動",
          fs=6.0, fc="#FFFFFF")
    c.badge(rx_ + 0.08 + 0.36, inner_top, "步驟 2b")
    c.arrow(rx_ + lw_ / 2, inner_top - 0.78, rx_ + lw_ / 2, inner_top - 0.78 - 0.17)
    c.box(rx_ + 0.08, ya_top - 0.46, lw_ - 0.16, 0.46,
          "回顧性解釋 Agent（LLM）\n讀流量變數與診斷結果", fs=6.0, fc=C["llm"])
    c.badge(rx_ + 0.08 + 0.36, ya_top, "步驟 3b")
    y_lane = lane_top - lane_h

    # --- 步驟 4 決策 ---
    yb2 = y_lane - 0.19
    c.ax.plot([lx + lw_ / 2, lx + lw_ / 2], [y_lane, yb2], lw=1.0, color=C["arrow"], zorder=4)
    c.ax.plot([rx_ + lw_ / 2, rx_ + lw_ / 2], [y_lane, yb2], lw=1.0, color=C["arrow"], zorder=4)
    c.ax.plot([lx + lw_ / 2, rx_ + lw_ / 2], [yb2, yb2], lw=1.0, color=C["arrow"], zorder=4)
    yd_top = yb2 - 0.20
    c.arrow(cx, yb2, cx, yd_top)
    c.box(0.55, yd_top - 0.92, 3.55, 0.92,
          "決策 Agent（LLM）：從 7 個動作中選 1 個\n"
          "不改持股（3）：維持現狀、進入觀察名單、升級人工覆核\n"
          "改變持股（4）：權重方案切換、條件式市值傾斜、\n覆蓋率傾斜、代表策略置換",
          fs=6.0, fc=C["llm"])
    c.badge(0.55 + 0.36, yd_top, "步驟 4")
    c.box(4.20, yd_top - 0.92 + 0.38, 1.95, 0.54,
          "規則\n流量變數只能驅動不改持股的動作；\n程式判決不可被 LLM 推翻",
          fs=5.6, fc="#FFFFFF", style="dashed", minfs=4.6)
    y_d = yd_top - 0.92

    # 人工覆核分支
    c.box(4.45, y_d - 1.50, 1.70, 0.62,
          "人工覆核\n（升級人工覆核之後；同時看到\n觸發嚴重度與回顧解釋）", fs=5.6, style="dashed",
          fc="#FFFFFF", minfs=4.6)
    c.badge(4.45 + 0.30, y_d - 0.88, "分支", w=0.40, fc=GRAY_BADGE)
    c.ax.plot([4.10, 5.30], [y_d + 0.20, y_d + 0.20], lw=0.9, color=C["arrow"], ls=(0, (4, 2)),
              zorder=4)
    c.arrow(5.30, y_d + 0.20, 5.30, y_d - 0.88, lw=0.9, ls=(0, (4, 2)))
    c.label(4.70, y_d + 0.20, "選「升級人工覆核」", fs=5.2)

    # --- 步驟 5~8 ---
    c.arrow(cx - 0.55, y_d, cx - 0.55, y_d - 0.22)
    y5_top = y_d - 0.22
    c.box(0.55, y5_top - 0.62, 3.55, 0.62,
          "動作生效（一季決策延遲：本季決策、下一季才生效）\n"
          "權重方案切換具持續性；其餘改持股動作只在觸發的下一季生效",
          fs=5.9, fc="#FFFFFF")
    c.badge(0.55 + 0.36, y5_top, "步驟 5")
    y5 = y5_top - 0.62
    c.arrow(cx - 0.55, y5, cx - 0.55, y5 - 0.22)
    y6_top = y5 - 0.22
    c.box(0.55, y6_top - 0.50, 3.55, 0.50,
          "風控檢查\n持股集中度上限（超限削減並重分配）、回測—實盤偏離校準監控",
          fs=5.9, fc="#FFFFFF")
    c.badge(0.55 + 0.36, y6_top, "步驟 6")
    y6 = y6_top - 0.50
    c.arrow(cx - 0.55, y6, cx - 0.55, y6 - 0.22)
    y7_top = y6 - 0.22
    c.box(0.55, y7_top - 0.52, 3.55, 0.52,
          "七段式解釋與備忘錄（LLM 解釋文字）\n前瞻驗證：訓練截止日之後的真實時點，四層標準",
          fs=5.9, fc=C["llm"])
    c.badge(0.55 + 0.36, y7_top, "步驟 7")
    y7 = y7_top - 0.52
    c.arrow(cx - 0.55, y7, cx - 0.55, y7 - 0.22)
    y8_top = y7 - 0.22
    c.box(0.55, y8_top - 0.40, 3.55, 0.40, "稽核紀錄（每次動作留一行 JSON 軌跡）", fs=6.0,
          fc="#FFFFFF", style="dashed")
    c.badge(0.55 + 0.36, y8_top, "步驟 8")
    y8 = y8_top - 0.40

    # 人工覆核 → 稽核（虛線，折線）
    c.ax.plot([5.30, 5.30], [y_d - 1.50, y8_top - 0.20], lw=0.9, color=C["arrow"],
              ls=(0, (4, 2)), zorder=4)
    c.arrow(5.30, y8_top - 0.20, 4.10, y8_top - 0.20, lw=0.9, ls=(0, (4, 2)))

    # --- 迴圈回到下一季 ---
    xl = 0.14
    c.ax.plot([0.55, xl, xl], [y8_top - 0.20, y8_top - 0.20, top - 0.21], lw=1.1,
              color=C["arrow"], zorder=4)
    c.arrow(xl, top - 0.21, 0.55, top - 0.21, lw=1.1)
    c.label(xl, (y8_top + top) / 2, "進入下一季（每季重複步驟 1～8）", fs=6.0, rot=90, bold=True)

    c.ax.text(W / 2, 0.09,
              "各機制的名稱對照見表 3-1；7 個動作中，只有權重方案切換具持續性。",
              fontsize=5.6, color=C["mute"], ha="center", va="center")
    return c


if __name__ == "__main__":
    print(make().save("fig_quarterly_loop"))
