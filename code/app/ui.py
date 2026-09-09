# -*- coding: utf-8 -*-
"""L4 · UI 層（Streamlit，應用層開發追蹤.md §1 L4，Phase C）

設定與執行 / 本期持倉 / 風險儀表板 / AI 解讀 / 歷史版本比較，五個區塊對應
§1 五層架構圖裡 L4 的說明。目前只接 `mode="replay"`（`live` 需要 A2，尚未
實作），對應到底層 `engine.run()` 的既有限制，不在 UI 假裝支援。

用法（config.ini 是相對路徑，一定要在 code/ 目錄下執行）：
    cd code
    streamlit run app/ui.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# 這支檔案是 streamlit 直接執行的進入點，不是用 `python -m` 以套件方式跑，
# 所以不能用相對 import（cli.py 那種 `from .config import ...`）——
# 手動把 code/ 加進 sys.path，改用絕對 import（沿用 fcv_core.py 的慣例）。
_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from app.audit import LOG_PATH, record  # noqa: E402
from app.calibration import check as check_calibration  # noqa: E402
from app.config import ReplayAnchor, RunConfig  # noqa: E402
from app.engine import DETAIL_PATH  # noqa: E402
from app.engine import run as run_engine  # noqa: E402
from app.memo import generate as generate_memo  # noqa: E402
from app.risk import assess  # noqa: E402

st.set_page_config(page_title="因子選股應用層", layout="wide")


@st.cache_data
def _load_detail() -> pd.DataFrame:
    df = pd.read_csv(DETAIL_PATH)
    df["ratio"] = df["ratio"].astype(str)
    return df


detail_df = _load_detail()

st.title("因子選股應用層")
st.caption("mode=replay，讀取凍結的 walk-forward 歷史資料重放；"
          "live 模式（即時查資料庫）需要 A2，本機資料庫恢復前尚未實作")

# ---------- 設定與執行（L0/L1） ----------
st.sidebar.header("設定與執行")

st.sidebar.caption("市場：台股（A1 決定：先只做台股，架構驗證完再擴美股／跨市場）")
market = "TW"

group = st.sidebar.selectbox("group", ["A_hrp", "D_top_cagr", "E_top_calmar"])
allocation = st.sidebar.selectbox("allocation", ["equal", "proportional"])

st.sidebar.caption("k_mode：silhouette_is（B2 決定：慢時鐘固定用這個，"
                   "H-26 驗證勝率 94.3% 高於 fixed 的 92.3%，且無前視偏誤）")
k_mode = "silhouette_is"

def _default_index(options: list, preferred) -> int:
    """優先選研究部驗證過的預設值（RunConfig.default_replay 同一組），沒有才退回第一個。"""
    return options.index(preferred) if preferred in options else 0


_avail = detail_df[(detail_df.tree_key == market) & (detail_df.group == group)
                   & (detail_df.allocation == allocation) & (detail_df.k_mode == k_mode)]
ratio_options = sorted(_avail["ratio"].unique().tolist()) or ["legacy"]
ratio = st.sidebar.selectbox("ratio", ratio_options, index=_default_index(ratio_options, "legacy"))

_avail2 = _avail[_avail.ratio == ratio]
scheme_options = sorted(_avail2["scheme"].unique().tolist()) or ["A"]
scheme = st.sidebar.selectbox("scheme（窗口方案，H-26）", scheme_options,
                              index=_default_index(scheme_options, "A"))

_avail3 = _avail2[_avail2.scheme == scheme]
window_options = sorted(int(w) for w in _avail3["window_no"].unique().tolist()) or [1]
window_no = st.sidebar.selectbox("window_no", window_options,
                                 index=_default_index(window_options, max(window_options)))

st.sidebar.markdown("---")
st.sidebar.caption("風控上限（C1：可自訂，預設值來自研究結果反推，見應用層開發追蹤.md §3 C2/C3）")
single_stock_cap = st.sidebar.number_input(
    "單一股票上限 %", value=8.0, min_value=0.1, max_value=100.0, step=0.5) / 100
cluster_cap_equal = st.sidebar.number_input(
    "單一群佔比上限（等量分配）%", value=25.0, min_value=0.1, max_value=100.0, step=1.0) / 100
cluster_cap_proportional = st.sidebar.number_input(
    "單一群佔比上限（比例分配）%", value=50.0, min_value=0.1, max_value=100.0, step=1.0) / 100

st.sidebar.markdown("---")
st.sidebar.caption("E2：純手動觸發。B2 慢時鐘「距下次重跑還有多久」需要 live 模式"
                   "才有實際時鐘可比對，尚未實作前不顯示假倒數")

run_clicked = st.sidebar.button("執行", type="primary")

if run_clicked:
    try:
        cfg = RunConfig(
            mode="replay", market=market, group=group, ratio=ratio, allocation=allocation,
            k_mode=k_mode, single_stock_cap=single_stock_cap,
            cluster_cap_equal=cluster_cap_equal, cluster_cap_proportional=cluster_cap_proportional,
            replay_anchor=ReplayAnchor(scheme=scheme, window_no=int(window_no)),
        )
        holdings = run_engine(cfg)
        risk = assess(holdings)
        calib = check_calibration(holdings)
    except Exception as e:  # noqa: BLE001 — 顯示給使用者看，不是要吞掉錯誤
        st.sidebar.error(f"執行失敗：{e}")
    else:
        st.session_state["holdings"] = holdings
        st.session_state["risk"] = risk
        st.session_state["calib"] = calib
        st.session_state["recorded"] = False
        st.session_state["recorded_at"] = None
        st.session_state["memo_result"] = None
        # 2026-09-09 code review：新一次執行要清掉上一次違規填的覆核原因，
        # 不然換一組設定觸發不同違規時，輸入框會帶著舊文字，容易誤按提交舊理由
        st.session_state.pop("override_reason", None)

holdings = st.session_state.get("holdings")
risk = st.session_state.get("risk")
calib = st.session_state.get("calib")

if holdings is None:
    st.info("左側設定完成後按「執行」開始。")
    st.stop()

tab_holdings, tab_risk, tab_ai, tab_history = st.tabs(
    ["本期持倉", "風險儀表板", "AI 解讀", "歷史版本比較"])

# ---------- 本期持倉（L1） ----------
with tab_holdings:
    st.subheader(f"{holdings.config.market}｜{holdings.config.group}｜"
                f"{holdings.config.ratio}／{holdings.config.allocation}")
    c1, c2, c3 = st.columns(3)
    c1.metric("持股（策略層級）數", holdings.n_members)
    c2.metric("OOS CAGR", f"{holdings.performance['oos_cagr']:.2%}")
    c3.metric("OOS Sharpe", f"{holdings.performance['oos_sharpe']:.3f}")

    st.markdown("**窗次資訊**")
    st.json(holdings.window_info)
    st.markdown("**績效（IS／OOS）**")
    st.json(holdings.performance)

    st.markdown(f"**持股清單**（{holdings.n_members} 檔 strategy_uid，"
               "⚠️ 策略層級不是股票層級，見 engine.py 的範圍限定）")
    st.dataframe(pd.DataFrame({"strategy_uid": holdings.members}),
                width="stretch", height=320)

# ---------- 風險儀表板（L2） ----------
with tab_risk:
    st.subheader("風控檢查（C2/C3）")
    c1, c2, c3 = st.columns(3)
    c1.metric("單檔權重（等權）", f"{risk.max_single_weight:.2%}",
             f"上限 {holdings.config.single_stock_cap:.2%}")
    c2.metric("最大群佔比", f"{risk.max_cluster_share:.2%}",
             f"上限 {holdings.config.cluster_cap:.2%}（{holdings.config.allocation}）")
    c3.metric("涵蓋群數", risk.n_clusters_covered)
    if risk.portfolio_mdd is not None:
        st.caption(f"組合 MDD：{risk.portfolio_mdd:.2%}　"
                  f"組合年化波動：{risk.portfolio_ann_vol:.2%}")

    st.markdown("---")
    st.subheader("校準監控（C5）")
    st.caption(f"對 {calib.thresholds.n_cells} 個歷史格子的 p10 分位反推的警戒線"
              "（示警用，不是強制關卡）")
    cc1, cc2 = st.columns(2)
    cc1.metric("OOS CAGR", f"{calib.oos_cagr:.2%}",
              f"p10 門檻 {calib.thresholds.oos_cagr_p10:.2%}",
              delta_color="inverse" if calib.below_cagr else "off")
    cc2.metric("OOS Calmar", f"{calib.oos_calmar:.3f}",
              f"p10 門檻 {calib.thresholds.oos_calmar_p10:.3f}",
              delta_color="inverse" if calib.below_calmar else "off")
    if calib.flagged:
        st.warning("⚠️ 這次表現明顯偏離歷史常態分布，只是提醒，不會攔下執行")

    st.markdown("---")
    st.subheader("執行覆核與稽核紀錄（C4/F2）")
    if st.session_state.get("recorded"):
        st.success(f"✅ 已寫入稽核紀錄：{st.session_state['recorded_at']}")
    elif risk.violations:
        st.error("🔴 風控違規，需人工覆核才能繼續：")
        for v in risk.violations:
            st.write(f"- {v.detail}")
        reason = st.text_input("覆核原因（必填，會如實寫進稽核紀錄，不做內容驗證）",
                               key="override_reason")
        if st.button("覆核通過並記錄", disabled=not reason.strip()):
            entry = record(holdings, risk, calibration=calib, override_reason=reason)
            st.session_state["recorded"] = True
            st.session_state["recorded_at"] = entry["recorded_at"]
            st.rerun()
    else:
        st.success("✅ 風控通過，無違規")
        if st.button("寫入稽核紀錄"):
            entry = record(holdings, risk, calibration=calib, override_reason=None)
            st.session_state["recorded"] = True
            st.session_state["recorded_at"] = entry["recorded_at"]
            st.rerun()

# ---------- AI 解讀（L3） ----------
with tab_ai:
    st.subheader("AI 解讀（D1/D2 · IC memo）")
    st.caption("架構比照 cluster_story／cluster_identity：程式先算出判決，"
              "LLM 只負責把「為什麼是這個判決」轉述成人話，不准自己推論、"
              "不准引用沒被餵給它的數字")

    dry_run = st.checkbox("dry-run（不花錢，不真的呼叫 LLM）", value=True)
    purpose = "app_memo"
    if not dry_run:
        st.warning("⚠️ config.ini 還沒設定 `[openai] app_memo_model` 這個 key，"
                  "需要借用既有 purpose 的模型設定跟額度記帳（帳本上會記成那個 "
                  "purpose 的用量）")
        purpose = st.text_input("借用哪個 purpose 的模型設定", value="cluster_story")

    if st.button("產生備忘錄"):
        try:
            result = generate_memo(holdings, risk, calib, dry_run=dry_run, purpose=purpose)
        except RuntimeError as e:
            st.error(f"🔴 {e}")
            st.session_state["memo_result"] = None
        else:
            st.session_state["memo_result"] = result

    memo_result = st.session_state.get("memo_result")
    if memo_result:
        labels = {"summary": "本期選股摘要", "risk_note": "風控說明",
                 "calibration_note": "校準監控說明", "caveat": "限制說明"}
        for k, label in labels.items():
            st.markdown(f"**{label}**")
            st.write(memo_result["memo"][k])
        if memo_result["dry_run"]:
            st.caption("此為 dry-run 內容，未實際呼叫 LLM")
        elif holdings.config.mode == "replay":
            if memo_result["leakage_check"]:
                st.error("D2 洩漏掃描發現問題：\n" + "\n".join(memo_result["leakage_check"]))
            else:
                st.success("✅ D2 洩漏掃描通過（memo 裡的數字都能對回判決資料）")

# ---------- 歷史版本比較（F1） ----------
with tab_history:
    st.subheader("歷史版本比較")
    st.caption("F1：全部保留。讀取 audit_log.jsonl 的每筆執行紀錄")
    rows = []
    if LOG_PATH.exists():
        with open(LOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                cfg = e["config"]
                anchor = cfg.get("replay_anchor") or {}
                rows.append({
                    "執行時間": e["recorded_at"], "市場": cfg["market"], "group": cfg["group"],
                    "ratio": cfg["ratio"], "allocation": cfg["allocation"],
                    "scheme": anchor.get("scheme"), "window_no": anchor.get("window_no"),
                    "持股數": e["n_members"],
                    "OOS CAGR": e["performance"].get("oos_cagr"),
                    "OOS MDD": e["performance"].get("oos_mdd"),
                    "違規數": len(e["risk"]["violations"]),
                    "覆核原因": e.get("override_reason") or "",
                })
    if not rows:
        st.info("目前還沒有任何稽核紀錄")
    else:
        hist_df = pd.DataFrame(rows).sort_values("執行時間", ascending=False)
        st.dataframe(hist_df, width="stretch")
