# -*- coding: utf-8 -*-
"""圖一：系統架構圖（比照郭鎧菘論文圖 2 的元件分層風格）。

三個子系統：① FCV 回測與資料服務（沿用既有系統）、② 研究部分析系統、③ 實戰應用層，
底部是共用的 LLM 模型層。所有模組名稱皆已對照真實檔案（2026-10-19 核對）。
②中帶 stage 編號的檔名刻意不顯示（避免 stage0/1/3/4 斷號），只留中文名稱。
"""
import math

from _figlib import C, Canvas


def layer(c, x, ytop, w, title, items, ncols=2, bh=0.30, fc="#FFFFFF", fs=5.7, tfs=6.6,
          gap=0.045, margin=0.05, th=0.19, caption=None, frame_fc="#FBFBFB"):
    """一個『層』：外框＋標題＋模組格。回傳框底部 y。"""
    rows = math.ceil(len(items) / ncols)
    cap_h = 0.12 if caption else 0.0
    h = th + cap_h + margin + rows * bh + (rows - 1) * gap + margin
    y = ytop - h
    c.group(x, y, w, h, title, fc=frame_fc, fs=tfs, title_h=th, bold=True)
    if caption:
        c.ax.text(x + w / 2, ytop - th - 0.045, caption, fontsize=4.9, ha="center", va="center",
                  color=C["mute"], zorder=6)
    bw = (w - 2 * margin - (ncols - 1) * gap) / ncols
    for i, it in enumerate(items):
        txt, st = (it if isinstance(it, str) else it[0]), "solid"   # 只畫已實作者，不分驗證狀態
        r, col = divmod(i, ncols)
        by = ytop - th - cap_h - margin - (r + 1) * bh - r * gap
        bx = x + margin + col * (bw + gap)
        span = (i == len(items) - 1 and len(items) % ncols == 1 and ncols > 1)
        c.box(bx, by, (w - 2 * margin) if span else bw, bh, txt, fs=fs, style=st, fc=fc, minfs=4.6)
    return y


def make():
    W, H = 6.3, 9.2
    c = Canvas(W, H)

    # ================= ① FCV =================
    x1, w1 = 0.08, 2.92
    top = H - 0.08
    c.group(x1, 4.30, w1, top - 4.30, "① FCV 回測與資料服務系統（沿用既有系統）", fc=C["fcv"],
            fs=7.2, title_h=0.26)
    y = top - 0.30
    y = layer(c, x1 + 0.06, y, w1 - 0.12, "應用層", [
        "市場泛化回測核心\n(fcv_core)", "分位數規格產生器\n(spec_generator)",
        "動態條件工廠\n(condition_factory)", "實驗變體設定\n(phase_variants)",
        "四階段分析\n(phase1~4_analyze)", "實驗掃描與策略索引\n(sweep_driver／catalog_builder)"])
    y = layer(c, x1 + 0.06, y - 0.07, w1 - 0.12, "邏輯運算層", [
        "資料框\n(dataframe)", "因子組合\n(combinations)",
        "回測\n(backtest＋backtest_core.pyx)", "報表\n(report)"])
    y = layer(c, x1 + 0.06, y - 0.07, w1 - 0.12, "資料存取層", [
        "資料取得\n(get_data)", "資料格式化\n(format_data)", "資料庫存取\n(database)"])
    # 資料庫
    db_y = 4.38
    c.cylinder(x1 + 0.55, db_y, 1.85, 0.62, "MariaDB 資料庫\n台股 TWSE 1,775 家\n美股 Russell 3000 2,972 檔",
               fs=5.8)

    # ================= ② 研究部 =================
    x2, w2 = 3.40, 2.82
    c.group(x2, 4.30, w2, top - 4.30, "② 研究部分析系統", fc=C["res"], fs=7.2, title_h=0.26)
    y = top - 0.30
    y = layer(c, x2 + 0.06, y, w2 - 0.12, "策略標記層", [
        "格式整備", "策略標記\n(passes_alpha_gate／is_usable)"])
    y = layer(c, x2 + 0.06, y - 0.06, w2 - 0.12, "分群層", [
        "HRP 階層分群（三棵主樹）\n另有危機期子集樹（輔助）\n(hrp)",
        "群數選擇：輪廓係數\n(cluster_count_selection)"])
    y = layer(c, x2 + 0.06, y - 0.06, w2 - 0.12, "群定性層", [
        "群身份與互補解釋（LLM）\n(cluster_identity／cluster_story)",
        "逐年賺賠與互補性\n(cluster_temporal_profile／\ncomplementarity_granularity)"])
    y = layer(c, x2 + 0.06, y - 0.06, w2 - 0.12, "驗證層", [
        "有效獨立賭注數\n(effective_bets／enb_null)", "四組對照實驗\n(four_group_control)",
        "樣本內外與滾動驗證\n(walkforward_matrix)",
        "代表策略挑選\n(cluster_representatives)"])
    y = layer(c, x2 + 0.06, y - 0.06, w2 - 0.12, "基礎設施", [
        ("凍結與雜湊\n(freeze)", "infra"), ("資料契約\n(contracts)", "infra")])
    c.cylinder(x2 + 0.45, 4.38, 1.95, 0.62, "_frozen/ 凍結產物 (parquet)\n＋滾動驗證輸出\n(walkforward_members)", fs=5.8)

    # 子系統間箭頭（① → ②）
    gx1, gx2 = x1 + w1, x2
    c.arrow(gx1, 7.95, gx2, 7.95, lw=1.2)
    c.label((gx1 + gx2) / 2, 8.18, "候選\n策略池", fs=5.6, bold=True)
    c.arrow(x1 + 0.55 + 1.85, 4.69, gx2, 4.69, lw=1.0, ls=(0, (4, 2)))
    c.label((gx1 + gx2) / 2, 5.02, "價格、財報、\n大盤基準", fs=5.2)

    # ================= ③ 實戰應用層 =================
    x3, w3 = 0.08, 6.14
    y3_top = 3.97
    y3_bot = 0.50
    c.group(x3, y3_bot, w3, y3_top - y3_bot, "③ 實戰應用層（按第三章 3.3 機制分組；圖中只列已實作的部分）", fc=C["app"],
            fs=7.2, title_h=0.26)

    r1_top = y3_top - 0.30
    # 第一列：設定與引擎 | 3.3.1 | 3.3.2
    ya = layer(c, x3 + 0.06, r1_top, 1.42, "設定與引擎", [
        ("設定 (config)", "infra"), ("執行引擎 (engine)\n重放／模擬模式", "infra"),
        ("分群與代表挑選\n(clustering)", "infra"), ("每季模擬主迴圈\n(simulate)", "infra")],
        ncols=1, bh=0.32, caption="")
    yb = layer(c, x3 + 1.54, r1_top, 3.30, "3.3.1 監控與動作", [
        ("三層指標\n環境／過程／結果", "infra"), "指數集中度監控",
        ("領漲族群覆蓋率監控", "dashed"),
        ("回撤觸發監控", "dashed"), ("策略群集中度監控", "dashed"), ("回顧型診斷與\n基準鏈歸因分解", "dashed"),
        "條件式市值傾斜", ("覆蓋率傾斜", "dashed"), ("代表策略置換", "dashed"),
        ("權重方案切換", "dashed"), ("不改持股動作：維持現狀／\n觀察名單／升級人工覆核", "dashed"),
        ("動作空間參照", "infra")], ncols=3, bh=0.29,
        caption="monitor／triggers／diagnose／weights／actions")
    yc = layer(c, x3 + 4.90, r1_top, 1.22, "3.3.2 風控", [
        ("持股集中度上限\n(risk)", "dashed"), ("回測—實盤偏離\n校準監控 (calibration)", "dashed")],
        ncols=1, bh=0.40)
    row1_bottom = min(ya, yb, yc)

    # 第二列：3.3.3 | 3.3.4 | 3.3.5 | 介面
    r2_top = row1_bottom - 0.07
    yd = layer(c, x3 + 0.06, r2_top, 2.62, "3.3.3 解釋", [
        "七段式解釋 Agent\n(explain)", "決策 Agent\n(agents／agent_tools)",
        ("解釋事實組裝\n(facts_lean)", "infra"), ("群知識庫\n(cluster_kb)", "infra"),
        ("情境比對\n(scenario)", "infra"), ("新舊面孔比較\n(new_faces)", "infra")],
        ncols=3, bh=0.30, fs=5.6)
    ye = layer(c, x3 + 2.74, r2_top, 1.0, "3.3.4 前瞻驗證", [
        "重複穩定度\n(stability)", "事後績效衡量\n(performance)"], ncols=1, bh=0.34)
    yf = layer(c, x3 + 3.80, r2_top, 1.30, "3.3.5 治理", [
        ("備忘錄 (memo)", "dashed"), ("稽核紀錄 (audit)", "dashed"),
        ("投資政策聲明 IPS\n（文件）", "dashed")], ncols=1, bh=0.27)
    yg = layer(c, x3 + 5.16, r2_top, 1.0, "介面", [
        ("命令列\n(cli)", "infra"), ("Streamlit\n(ui)", "infra")], ncols=1, bh=0.34)
    row2_bottom = min(yd, ye, yf, yg)

    # ================= LLM 模型層 =================
    c.box(0.08, 0.20, W - 0.16, 0.22,
          "LLM 模型層（OpenAI API；額度帳本 openai_quota）— 供 ② 群身份解釋與 ③ 決策／解釋使用；只做解釋，不參與持股與報酬計算",
          fs=5.8, fc=C["llm"], minfs=4.6)

    # ===== 子系統間縱向箭頭 =====
    # ② → ③
    c.arrow(x2 + w2 / 2 + 0.2, 4.30, x2 + w2 / 2 + 0.2, y3_top, lw=1.2)
    c.label(x2 + w2 / 2 + 0.2, 4.135, "各窗次代表策略名單、群知識庫", fs=5.4, bold=True)
    # ① DB → ③
    c.arrow(x1 + 1.47, 4.30, x1 + 1.47, y3_top, lw=1.0, ls=(0, (4, 2)))
    c.label(x1 + 1.47, 4.135, "價格與市值", fs=5.4)

    # 底部下箭頭：LLM → ③ 解釋（短箭頭）
    return c, row2_bottom


if __name__ == "__main__":
    c, rb = make()
    print("③ 第二列底部 y =", round(rb, 2), "（框底 0.52）")
    print(c.save("fig_architecture"))
