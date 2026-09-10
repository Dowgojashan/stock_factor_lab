# -*- coding: utf-8 -*-
"""L4 · UI 層（Streamlit，應用層開發追蹤.md §1 L4，Phase C）

設定與執行 / 本期持倉 / 風險儀表板 / AI 解讀 / 歷史版本比較，五個區塊對應
§1 五層架構圖裡 L4 的說明。策略選擇（`engine.run()`）目前只接 `mode="replay"`——
慢時鐘（A2，即時重建 HRP 樹）需要把候選池報酬序列補到今天，仍暫緩，見
應用層開發追蹤.md §3-A2。

⚠️ **2026-09-10（應用層 §6 落差⑤）**：A2 暫緩的只有「重建樹」這一半（慢時鐘）——
「已凍結驗證的策略清單現在實際持有哪些股票」（快時鐘）不需要重建樹，機制
完全成立，見「本期持倉」分頁「股票層級持股明細」的「今天」選項。

用法（config.ini 是相對路徑，一定要在 code/ 目錄下執行）：
    cd code
    streamlit run app/ui.py
"""
from __future__ import annotations

import datetime
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

from app.audit import LOG_PATH, diff_holdings, find_previous, record  # noqa: E402
from app.calibration import check as check_calibration  # noqa: E402
from app.config import ReplayAnchor, RunConfig  # noqa: E402
from app.engine import DETAIL_PATH  # noqa: E402
from app.engine import run as run_engine  # noqa: E402
from app.memo import generate as generate_memo  # noqa: E402
from app.risk import assess  # noqa: E402
# 2026-09-10 使用者回饋：股票層級持股明細（原本是獨立腳本 resolve_strategy_holdings.py，
# 使用者要求接進 UI）。直接重用同一套邏輯，不重寫一份——避免兩邊對 C_rule/F1/F2
# 條件重建規則各自維護一份、日後改一邊忘了改另一邊。
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402
from resolve_strategy_holdings import _load_company_names  # noqa: E402
from resolve_strategy_holdings import resolve_holdings  # noqa: E402

st.set_page_config(page_title="因子選股應用層", layout="wide")


@st.cache_data
def _load_detail() -> pd.DataFrame:
    df = pd.read_csv(DETAIL_PATH)
    df["ratio"] = df["ratio"].astype(str)
    return df


detail_df = _load_detail()


# ---------- 股票層級持股解析（選填，見「股票持股明細」section） ----------
# `MarketData` 讀一次全市場資料要 ~110 秒，用 cache_resource 讓同一個 market
# 在整個 Streamlit session 裡只真的連一次資料庫，不會每次互動都重載。
@st.cache_resource(show_spinner=False)
def _get_market_data(market: str):
    from fcv_core import MarketData
    return MarketData(market)


@st.cache_data(show_spinner=False)
def _get_company_names(market: str) -> pd.Series:
    return _load_company_names(market)


@st.cache_data(show_spinner=False)
def _get_candidate_index() -> pd.DataFrame:
    return pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")


def _resolve_stock_holdings(market: str, members: list[str], as_of: str) -> tuple[pd.DataFrame, str]:
    """把一批 strategy_uid 解析成當天實際持有的股票（跟 resolve_strategy_holdings.py
    同一套機制：從 candidate_index 重建 F1/F2/C/V 條件，丟進 get_mask() 現算）。"""
    md = _get_market_data(market)
    idx = _get_candidate_index()
    names = _get_company_names(market)
    rows = []
    use_date = None
    for uid in members:
        row = idx.loc[uid]
        stocks, use_date = resolve_holdings(md, row, as_of)
        for sym in stocks:
            rows.append({"strategy_uid": uid, "stock_symbol": sym,
                        "company_name": names.get(sym, "")})
    return pd.DataFrame(rows), (use_date.date().isoformat() if use_date is not None else as_of)

st.title("因子選股應用層")
st.caption("mode=replay，讀取凍結的 walk-forward 歷史資料重放；"
          "live 模式（即時查資料庫）需要 A2，本機資料庫恢復前尚未實作")

# ---------- 設定與執行（L0/L1） ----------
st.sidebar.header("設定與執行")

st.sidebar.caption("市場：台股（A1 決定：先只做台股，架構驗證完再擴美股／跨市場）")
market = "TW"

group = st.sidebar.selectbox("group", ["A_hrp", "D_top_cagr", "E_top_calmar"])

# 2026-09-10 使用者回饋：「分配」（等量/比例）改用「採樣」；下拉選單顯示中文說明，
# 底層值（"equal"/"proportional"）不變——RunConfig／CSV 欄位仍用原字串，只改顯示文字。
ALLOCATION_LABELS = {"equal": "等量採樣", "proportional": "比例採樣"}
allocation = st.sidebar.selectbox(
    "採樣方式", ["equal", "proportional"], format_func=lambda a: ALLOCATION_LABELS[a])

st.sidebar.caption("k_mode：silhouette_is（B2 決定：慢時鐘固定用這個，"
                   "H-26 驗證勝率 94.3% 高於 fixed 的 92.3%，且無前視偏誤）")
k_mode = "silhouette_is"

def _default_index(options: list, preferred) -> int:
    """優先選研究部驗證過的預設值（RunConfig.default_replay 同一組），沒有才退回第一個。"""
    return options.index(preferred) if preferred in options else 0


# 2026-09-10 使用者回饋：窗口方案不要只顯示代號（A/B/C...），沒有人會去背每個代號
# 對應的窗長——直接把 IS/OOS 月數跟窗次數顯示出來。底層值仍是代號字串，
# 對應 walkforward_matrix_detail.csv 的 scheme 欄位，不影響資料篩選邏輯。
_scheme_info = detail_df.drop_duplicates("scheme").set_index("scheme")


def _scheme_label(code: str) -> str:
    if code not in _scheme_info.index:
        return code
    row = _scheme_info.loc[code]
    tag = "" if row["mode"] == "anchored" else "・rolling對照組"
    return (f"IS {int(row['min_is_months'])} 個月／OOS {int(row['oos_len_months'])} 個月"
           f"（共 {int(row['n_windows'])} 窗{tag}）")


_avail = detail_df[(detail_df.tree_key == market) & (detail_df.group == group)
                   & (detail_df.allocation == allocation) & (detail_df.k_mode == k_mode)]
ratio_options = sorted(_avail["ratio"].unique().tolist()) or ["legacy"]
ratio = st.sidebar.selectbox("ratio", ratio_options, index=_default_index(ratio_options, "legacy"))

_avail2 = _avail[_avail.ratio == ratio]
scheme_options = sorted(_avail2["scheme"].unique().tolist()) or ["A"]
scheme = st.sidebar.selectbox("窗口方案（H-26）", scheme_options,
                              index=_default_index(scheme_options, "A"),
                              format_func=_scheme_label)

_avail3 = _avail2[_avail2.scheme == scheme]
window_options = sorted(int(w) for w in _avail3["window_no"].unique().tolist()) or [1]
window_no = st.sidebar.selectbox("window_no", window_options,
                                 index=_default_index(window_options, max(window_options)))

st.sidebar.markdown("---")
st.sidebar.caption("風控上限（C1：可自訂，預設值來自研究結果反推，見應用層開發追蹤.md §3 C2/C3）")
single_stock_cap = st.sidebar.number_input(
    "單一股票上限 %", value=8.0, min_value=0.1, max_value=100.0, step=0.5) / 100
cluster_cap_equal = st.sidebar.number_input(
    "單一群佔比上限（等量採樣）%", value=25.0, min_value=0.1, max_value=100.0, step=1.0) / 100
cluster_cap_proportional = st.sidebar.number_input(
    "單一群佔比上限（比例採樣）%", value=50.0, min_value=0.1, max_value=100.0, step=1.0) / 100

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
        # 2026-09-10：換一組策略組合後，上一次解析出來的股票持股明細就不對應了，
        # 清掉逼使用者重新按「解析持股」，不留著舊資料造成誤導。
        st.session_state.pop("stock_detail", None)
        st.session_state.pop("stock_detail_date", None)

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
    c3.metric("OOS Sharpe", f"{holdings.performance['oos_sharpe']:.2f}")

    # 2026-09-10（應用層§6落差①）：跟上一次同一組 RunConfig 身份執行結果的比較。
    _diff = diff_holdings(holdings.members, find_previous(holdings.config))
    if _diff["has_previous"]:
        st.info(f"**跟上次相比**（上一次執行於 {_diff['previous_recorded_at'][:19]}）："
               f"新增 {_diff['n_added']} 檔、剔除 {_diff['n_removed']} 檔、"
               f"不變 {_diff['n_unchanged']} 檔")
    else:
        st.caption("這是這組設定（market/group/ratio/allocation/k_mode）第一次執行，"
                  "沒有前一期可比較。")

    st.markdown("**窗次資訊**")
    st.json(holdings.window_info)
    st.markdown("**績效（IS／OOS）**")
    # 2026-09-10 使用者回饋：數字一律顯示到小數點第二位——原始 dict 是未格式化的
    # 浮點數（例如 0.1424137781102339），直接 st.json 會整串洩出來，這裡先轉成
    # 格式化字串再顯示，底層 holdings.performance 的原始精度不受影響。
    perf = holdings.performance
    st.json({
        "is_cagr": f"{perf['is_cagr']:.2%}",
        "is_mdd": f"{perf['is_mdd']:.2%}",
        "oos_cagr": f"{perf['oos_cagr']:.2%}",
        "oos_mdd": f"{perf['oos_mdd']:.2%}",
        "oos_sharpe": f"{perf['oos_sharpe']:.2f}",
        "n_backfilled": perf["n_backfilled"],
    })

    if holdings.alternative_groups:
        st.markdown("**候選方案對照**（應用層§6落差③：同一格設定下的其他 H-12 對照組，"
                   "純呈現既有比較結果，不代表建議改用哪一個——H-26/H-27/M-03 已用"
                   "統計證據回答過這個問題）")
        alt_rows = [{"group": holdings.config.group, "n_members": holdings.n_members,
                    "oos_cagr": f"{perf['oos_cagr']:.2%}", "oos_mdd": f"{perf['oos_mdd']:.2%}",
                    "oos_sharpe": f"{perf['oos_sharpe']:.2f}",
                    "會增/減幾檔": "（目前選定）"}]
        for g, a in holdings.alternative_groups.items():
            alt_rows.append({
                "group": g, "n_members": a["n_members"],
                "oos_cagr": f"{a['oos_cagr']:.2%}", "oos_mdd": f"{a['oos_mdd']:.2%}",
                "oos_sharpe": f"{a['oos_sharpe']:.2f}",
                "會增/減幾檔": f"+{a['n_would_add']} / -{a['n_would_remove']}",
            })
        st.dataframe(pd.DataFrame(alt_rows), width="stretch", hide_index=True)

    st.markdown(f"**持股清單**（{holdings.n_members} 檔 strategy_uid，"
               "⚠️ 策略層級不是股票層級，見 engine.py 的範圍限定）")
    st.dataframe(pd.DataFrame({"strategy_uid": holdings.members}),
                width="stretch", height=320)

    st.markdown("---")
    st.markdown("**股票層級持股明細**（選填）")
    st.caption("即時查資料庫、用 candidate_index 重建每個策略的選股條件現算——"
              "不是重跑回測。第一次解析要連整個市場的資料，約 1~2 分鐘；"
              "同一個市場之後點擊會用快取，秒開。")

    # 2026-09-10（應用層 §6 落差⑤）：A2「live 模式暫緩」原本把慢時鐘（重建 HRP
    # 樹，卡在 returns_monthly.parquet 只到 2025-12）跟快時鐘（凍結的策略清單
    # 現在實際持有哪些股票）混在一起一起暫緩——但快時鐘完全不需要重建樹，
    # `resolve_holdings()` 對任意日期都成立，「今天」只是眾多可能日期之一。
    # 這裡加這個選項，把已經驗證過的機制接上真正即時的一半。
    _date_options = {
        "IS 結束日（is_end）": holdings.window_info["is_end"],
        "OOS 起始日（oos_start）": holdings.window_info["oos_start"],
        "OOS 結束日（oos_end）": holdings.window_info["oos_end"],
        "今天（即時／快時鐘）": datetime.date.today().isoformat(),
    }
    _date_label = st.selectbox("解析哪個時間點的持股", list(_date_options),
                               index=1)  # 預設 oos_start
    if _date_label == "今天（即時／快時鐘）":
        st.caption("⚠️ 策略選擇仍是這個 replay 窗次凍結驗證過的結果（慢時鐘未重建），"
                  "只有股票持股是即時查資料庫算出來的（快時鐘）——不是對「今天該用"
                  "哪組策略」做即時判斷，那件事仍待 A2 慢時鐘完整實作。")

    if st.button("解析持股"):
        try:
            with st.spinner("查資料庫並重建選股條件中..."):
                stock_detail, use_date = _resolve_stock_holdings(
                    holdings.config.market, holdings.members, _date_options[_date_label])
        except Exception as e:  # noqa: BLE001 — 顯示給使用者看，不是要吞掉錯誤
            st.error(f"解析失敗：{e}")
        else:
            st.session_state["stock_detail"] = stock_detail
            st.session_state["stock_detail_date"] = use_date

    stock_detail = st.session_state.get("stock_detail")
    if stock_detail is not None:
        st.caption(f"實際對到交易日：{st.session_state['stock_detail_date']}"
                  f"（要求日期不是交易日時，取小於等於它的最後一個交易日）")
        n_unique = stock_detail["stock_symbol"].nunique()
        st.write(f"{len(stock_detail)} 筆策略-股票對應，{n_unique} 檔不重複股票")
        st.dataframe(stock_detail, width="stretch", height=280)

        st.markdown("**跨策略重疊度**（同一檔股票被幾個選中策略同時持有——"
                    "呼應老師 9-8 的 fund house 風控意見，見應用層開發追蹤.md §2.4）")
        overlap = (stock_detail.groupby(["stock_symbol", "company_name"])["strategy_uid"]
                  .nunique().reset_index(name="n_strategies")
                  .sort_values("n_strategies", ascending=False))
        st.dataframe(overlap, width="stretch", height=240)

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

    st.markdown("**因子曝險與市場/情境分布**（應用層§6落差②：T8 本來就算好、先前沒接進 UI 的數字）")
    st.json({
        "factor_exposure_F1": {k: f"{v:.2%}" for k, v in risk.factor_exposure_f1.items()},
        "market_share": {k: f"{v:.2%}" for k, v in risk.market_share.items()},
        "regime_avg_ret": {k: f"{v:.2%}" for k, v in risk.regime_avg_ret.items()},
    })

    st.markdown("---")
    st.subheader("校準監控（C5）")
    st.caption(f"對 {calib.thresholds.n_cells} 個歷史格子的 p10 分位反推的警戒線"
              "（示警用，不是強制關卡）")
    cc1, cc2 = st.columns(2)
    cc1.metric("OOS CAGR", f"{calib.oos_cagr:.2%}",
              f"p10 門檻 {calib.thresholds.oos_cagr_p10:.2%}",
              delta_color="inverse" if calib.below_cagr else "off")
    cc2.metric("OOS Calmar", f"{calib.oos_calmar:.2f}",
              f"p10 門檻 {calib.thresholds.oos_calmar_p10:.2f}",
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
                 "calibration_note": "校準監控說明",
                 "change_note": "跟上次相比（應用層§6落差①）",
                 "alternative_note": "候選方案對照（應用層§6落差③）",
                 "caveat": "限制說明"}
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
                    # 2026-09-10 使用者回饋：數字一律顯示到小數點第二位（跟其他分頁一致）。
                    "OOS CAGR": (f"{e['performance']['oos_cagr']:.2%}"
                                if e["performance"].get("oos_cagr") is not None else None),
                    "OOS MDD": (f"{e['performance']['oos_mdd']:.2%}"
                               if e["performance"].get("oos_mdd") is not None else None),
                    "違規數": len(e["risk"]["violations"]),
                    "覆核原因": e.get("override_reason") or "",
                })
    if not rows:
        st.info("目前還沒有任何稽核紀錄")
    else:
        hist_df = pd.DataFrame(rows).sort_values("執行時間", ascending=False)
        st.dataframe(hist_df, width="stretch")
