# -*- coding: utf-8 -*-
"""L4 · UI 層（Streamlit，應用層開發追蹤.md §1 L4，Phase C）

設定與執行 / 本期持倉 / 風險儀表板 / AI 解讀 / 歷史版本比較，五個區塊對應
§1 五層架構圖裡 L4 的說明。

🔴 **2026-09-10（§7 P2）：兩種模式並存**，舊版「只接 mode='replay'」的說法已過時。

    正式模式 (live)   IS＝2007-01~最新可用月（228 個月），用 `_frozen/stage3` 凍結
                      主線樹即時挑代表。**依定義沒有 OOS**——IS 用掉全部資料。
                      這是「現在該持有什麼」。
    驗證模式 (replay) 讀凍結 walk-forward 窗次，有真實 OOS 可比對。這是「方法在
                      歷史上表現如何」，不是現在該買什麼。

⚠️ **探索模式（任意 IS 區間、手動 k）決定不做**，見 §7.5 G5。

🔴 **UI 呈現上必守的三件事**（不要為了畫面好看而拿掉）：
  1. **§8-R13**：`is_*` 是樣本內配適值，**不是預期報酬**（歷史上 IS CAGR 中位數比
     OOS 高 7.62pp）。正式模式的最醒目位置放的是「同類設定的歷史 OOS 分布」，
     不是本次的 `is_cagr`。且 IS 與 OOS **不可並排比較**（方向還不一致：
     IS MDD/Sharpe 反而更差，因為 IS 涵蓋 2008）。
  2. **§8-R12**：候選方案對照下面那句定錨句不能拿掉——A_hrp 在報酬上輸給 D/E 是
     **系統性結果**（900 格只贏 16.8%），不是本期特例，沒這句話讀者會問「為什麼不換」。
  3. **G7**：`✅ 方法已驗證` 旁邊一定要有 `structural_caveat`（組合績效無 OOS 可驗證）。

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
from app.config import DEFAULT_CLUSTER_CAP_EQUAL  # noqa: E402
from app.config import DEFAULT_CLUSTER_CAP_PROPORTIONAL  # noqa: E402
from app.config import DEFAULT_SINGLE_STOCK_CAP  # noqa: E402
from app.config import GROUP_LABELS  # noqa: E402
from app.config import ReplayAnchor, RunConfig  # noqa: E402
from app.engine import DETAIL_PATH  # noqa: E402
from app.engine import BlendLeg, blend_holdings  # noqa: E402
from app.engine import run as run_engine  # noqa: E402
from app.risk import assess, assess_stock_level, blend_stock_weights  # noqa: E402
# 2026-09-10 使用者回饋：股票層級持股明細（原本是獨立腳本 resolve_strategy_holdings.py，
# 使用者要求接進 UI）。直接重用同一套邏輯，不重寫一份——避免兩邊對 C_rule/F1/F2
# 條件重建規則各自維護一份、日後改一邊忘了改另一邊。
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402
from resolve_strategy_holdings import _load_company_names  # noqa: E402
from resolve_strategy_holdings import markets_needed  # noqa: E402
from resolve_strategy_holdings import resolve_holdings_multi  # noqa: E402

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


@st.cache_data(show_spinner=False, ttl=3600)
def _cap_weighted_market_return(market: str, as_of: str, end: str) -> float | None:
    """真實市值加權大盤同期報酬（TR 報酬指數，含股利）——只有 TW／US 有，
    沿用 `research/market_benchmark.py` 的方法論（不可用價格指數，見該檔
    docstring：策略用還原收盤價已含股利，比較基準也必須用報酬指數，否則
    每個「贏大盤」都會被高估 3~4pp）。查詢失敗回傳 None，呼叫端自行處理。
    """
    if market not in ("TW", "US"):
        return None
    try:
        import fcv_core  # noqa: F401  sys.path bootstrap，讓根目錄的 database 找得到
        from database import Database

        table = {"TW": "taiex_tr", "US": "sp500_tr"}[market]
        db = Database("TW")   # 兩個 TR 表都在同一個連線可查，市場只是選表名
        cur = db.create_connection().cursor()
        cur.execute(f"SELECT date, close FROM {table} ORDER BY date")
        rows = cur.fetchall()
        s = pd.DataFrame(rows, columns=["date", "close"])
        s["date"] = pd.to_datetime(s["date"])
        s = (s.sort_values("date").drop_duplicates("date")
             .set_index("date")["close"].astype(float))
        as_of_ts, end_ts = pd.Timestamp(as_of), pd.Timestamp(end)
        s0, s1 = s[s.index <= as_of_ts], s[s.index <= end_ts]
        if s0.empty or s1.empty:
            return None
        return float(s1.iloc[-1] / s0.iloc[-1] - 1.0)
    except Exception:  # noqa: BLE001 — UI 上這是加分資訊，查不到就略過不擋流程
        return None


def _resolve_stock_holdings(members: list[str], as_of: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """把一批 strategy_uid 解析成當天實際持有的股票（跟 resolve_strategy_holdings.py
    同一套機制：從 candidate_index 重建 F1/F2/C/V 條件，丟進 get_mask() 現算）。

    🔴 2026-09-11（§9.7 I-1，S1）：XM 投組混合台美策略，不能只用單一市場的
    `MarketData`——依 `candidate_index` 的 `market` 欄位（不是解析 uid 字串）分派
    到對應的資料庫連線。回傳的第二個值從單一字串改成 `{市場: 實際對到的交易日}`，
    因為台美交易日曆不同，XM 投組可能有兩個不同的「今天」，不可化簡成單一日期。
    """
    idx = _get_candidate_index()
    needed = markets_needed(idx, members)
    md_map = {m: _get_market_data(m) for m in needed}
    names_map = {m: _get_company_names(m) for m in needed}

    rows = []
    market_dates: dict[str, pd.Timestamp] = {}
    for uid in members:
        m = idx.loc[uid, "market"]
        stocks, use_date = resolve_holdings_multi(md_map, idx, uid, as_of)
        market_dates[m] = use_date   # 同市場多次覆寫值相同，不影響
        names = names_map[m]
        for sym in stocks:
            rows.append({"strategy_uid": uid, "stock_symbol": sym,
                        "company_name": names.get(sym, ""), "stock_market": m})
    dates_str = {m: d.date().isoformat() for m, d in market_dates.items()}
    return pd.DataFrame(rows), dates_str


# ---------- 監控 Agent · 人機對話（L3，實戰監控Agent系統設計文件 §10 階段6） ----------
# 🔴 2026-09-19：這是**另一個子系統**——跟上面「本期持倉/風險儀表板/AI解讀/
# 歷史版本比較」四個分頁（應用層開發追蹤.md 的 RunConfig/engine/risk 主線）
# 完全獨立，接的是 `app/simulate.py`／`app/agents.py`／`app/monitor.py` 那條
# 線（實戰監控Agent系統設計文件），使用者選 3 個方案（CLI／擴建 Streamlit／
# 這個對話框本身扮演介面）之後決定扩建這支既有 UI。不依賴左側側邊欄任何
# 設定，也不會被 `holdings is None` 擋住——這裡對話的對象是**已經真實跑完**
# 的 L2 臂決策草案（`_runs/simulate_{run_id}.jsonl`），不是這支 UI 上面選股
# 引擎的結果。
def _render_l3_tab() -> None:
    from app import l3_dialogue

    st.subheader("監控 Agent · 人機對話（L3）")
    st.caption("實戰監控Agent系統設計文件 §10 階段 6：針對某一季**已經真實"
              "跑完**的階段5決策草案，人類覆核者可以跟 Agent-A 進行最多 5 輪"
              "對話（設計文件 §6）。清單外選項可在對話中共同提出，但須明確"
              "聲明尚未經程式回測驗證，不能直接採用。")

    run_ids = l3_dialogue.list_run_ids()
    if not run_ids:
        st.info("`code/app/_runs/` 底下還沒有任何 `simulate_*.jsonl`——"
               "要先跑過 `app/simulate.py` 的 L2 臂（例如 `_run_formal_8q.py`）"
               "才有真實決策草案可以對話。")
        return

    # 有多組真實實驗可選時，預設指向主線 8 季實驗（不是隨機挑到陰性/陽性
    # 對照組那些輔助驗證用的跑法）——跟側邊欄「持股規模」預設選 legacy
    # 是同一個 `_default_index` 慣例。
    run_id = st.selectbox(
        "實驗（run_id）", run_ids,
        index=_default_index(run_ids, "formal_8q_control0_L2_execlayer_v2"),
        key="l3_run_id_select")
    candidates = l3_dialogue.list_dialogue_candidates(run_id)
    if not candidates:
        st.warning(f"`{run_id}` 沒有 L2 臂的 checkpoint"
                  "（control0 臂定義上不經過 agent 決策，沒有草案可對話）。")
        return

    def _q_label(c: dict) -> str:
        return (f"{c['quarter_end']}｜狀態={c['m1d']['state']}｜"
               f"決策={c['decision']['decision']}")

    checkpoint = st.selectbox("季度", candidates, format_func=_q_label,
                              key="l3_quarter_select")
    quarter_end = checkpoint["quarter_end"]

    with st.expander("監控與診斷報告（screen 1，已由程式與 3a/3b 產生，"
                     "對話討論的基礎資料）"):
        st.markdown("**三層指標**")
        st.json({"env": checkpoint["env"], "proc": checkpoint["proc"],
                 "outcome": checkpoint["outcome"], "m1d": checkpoint["m1d"]})
        st.markdown("**回顧診斷**（3a，僅供學習，§7.0 不得驅動動作）")
        st.json(checkpoint["retrospective_output"])
        st.markdown("**前瞻評估**（3b，唯一可驅動動作的區塊）")
        st.json(checkpoint["prospective_output"])

    st.markdown("**階段5 決策草案**（已經真實做出，對話目的是討論它、不是"
               "重新決策——要不要真的改變決策，是人類覆核者核准的事）")
    draft_decision = checkpoint["decision"]
    st.info(f"**{draft_decision['decision']}**　{draft_decision['decision_detail']}")
    st.caption(draft_decision["reasoning"])
    if "summary" in checkpoint:
        with st.expander("季度總結（screen 2，已由階段7產生）"):
            st.json(checkpoint["summary"])

    decision_facts = l3_dialogue.build_decision_facts_for_quarter(checkpoint)

    # 換一組（run_id, 季度）視為開新的一段對話——延續上一組的 transcript
    # 沒有意義（決策草案的客觀資料完全不同）。
    _sel_key = (run_id, quarter_end)
    if st.session_state.get("l3_sel_key") != _sel_key:
        st.session_state["l3_sel_key"] = _sel_key
        st.session_state["l3_transcript"] = []
        st.session_state["l3_session_id"] = l3_dialogue.new_session_id(run_id, quarter_end)

    transcript = st.session_state["l3_transcript"]

    st.markdown("---")
    st.markdown(f"**對話**（第 {len(transcript)}/{l3_dialogue.MAX_ROUNDS} 輪）")

    for turn in transcript:
        with st.chat_message("user"):
            st.write(turn["human_message"])
        with st.chat_message("assistant"):
            resp = turn["agent_response"]
            st.write(resp["response"])
            if not resp["still_recommends_stage5_decision"]:
                st.warning(f"⚠️ 這輪表態不再支持原決策草案，改為建議："
                          f"{resp['revised_recommendation']}"
                          f"（是否真的改變決策，仍須人類核准）")
            if resp["proposes_out_of_list_option"]:
                st.error(f"🔶 提出清單外選項（尚未經程式回測驗證，不能直接"
                        f"採用）：{resp['out_of_list_option_description']}")
            if turn["leakage_check"]:
                st.caption(f"⚠️ D2 一致性檢查：{turn['leakage_check']}")
            if turn.get("dry_run"):
                st.caption("（此輪為 dry-run，未實際呼叫 LLM，未落盤稽核紀錄）")

    if len(transcript) >= l3_dialogue.MAX_ROUNDS:
        st.warning(f"已達到 {l3_dialogue.MAX_ROUNDS} 輪上限（設計文件 §6），"
                  "這段對話結束。可在上面重選季度開新的一段對話。")
    else:
        dry_run = st.checkbox(
            "dry-run（先用假回覆測試介面，不呼叫真實 LLM、不落盤稽核紀錄）",
            value=True, key="l3_dry_run")
        human_message = st.chat_input("針對這季的決策草案提問或提出意見…")
        if human_message:
            try:
                with st.spinner("Agent-A 回覆中…"):
                    result = l3_dialogue.run_dialogue_turn(
                        decision_facts, draft_decision, transcript, human_message,
                        dry_run=dry_run)
            except RuntimeError as e:
                st.error(f"🔴 {e}")
            else:
                round_no = len(transcript) + 1
                if not dry_run:
                    l3_dialogue.append_round(
                        st.session_state["l3_session_id"], run_id=run_id,
                        quarter_end=quarter_end, round_no=round_no,
                        human_message=human_message, agent_result=result)
                transcript.append({
                    "round": round_no, "human_message": human_message,
                    "agent_response": result["explanation"],
                    "leakage_check": result["leakage_check"],
                    "dry_run": result["dry_run"],
                })
                st.rerun()

    if transcript and st.button("開始新的一段對話（清空目前 transcript）"):
        st.session_state["l3_transcript"] = []
        st.session_state["l3_session_id"] = l3_dialogue.new_session_id(run_id, quarter_end)
        st.rerun()


st.title("因子選股應用層")

# ---------- 設定與執行（L0/L1） ----------
st.sidebar.header("設定與執行")

MODE_LABELS = {
    "live": "正式模式（依全部歷史資料，建議目前應持有的組合）",
    "replay": "回測檢視（重現某個歷史時間點，可對照後續實際表現）",
}
mode = st.sidebar.radio("模式", ["live", "replay"],
                        format_func=lambda m: MODE_LABELS[m])

MARKET_LABELS = {"TW": "台股", "US": "美股", "XM": "跨市場（台股＋美股）"}
market = st.sidebar.selectbox("市場", ["TW", "US", "XM"],
                              format_func=lambda m: MARKET_LABELS[m])
if market == "XM":
    st.sidebar.caption("跨市場組合是台股與美股策略的合併（各自依所屬市場分群，"
                       "不混合）。股票明細會分別查詢兩地市場資料，兩地交易日不同，"
                       "對應日期可能不同。")

# 三個市場的建模區間與分群數量不同，各自對應真實的資料涵蓋範圍。
_MARKET_IS_INFO = {
    "TW": {"is_range": "2007-01～最新可用月（228 個月）", "k": 6},
    "US": {"is_range": "2002-01～最新可用月（288 個月）", "k": 7},
    "XM": {"is_range": "2007-01～最新可用月（228 個月）", "k": 3},
}
if mode == "live":
    _info = _MARKET_IS_INFO[market]
    st.caption(f"**正式模式**：以 {_info['is_range']} 的完整歷史資料建立分群模型"
               f"（共 {_info['k']} 群），即時挑選代表策略。因使用全部歷史資料建模，"
               f"本次結果沒有樣本外表現可供驗證。")
else:
    st.caption("**回測檢視**：讀取過去某個時間點的歷史選股結果，並可對照其後續"
               "實際表現。用於檢視方法的歷史績效，不是目前建議持股。")

group = st.sidebar.selectbox("選股邏輯", ["A_hrp", "D_top_cagr", "E_top_calmar"],
                             format_func=lambda g: GROUP_LABELS[g])

ALLOCATION_LABELS = {"equal": "等量採樣", "proportional": "比例採樣"}
allocation = st.sidebar.selectbox(
    "採樣方式", ["equal", "proportional"], format_func=lambda a: ALLOCATION_LABELS[a])

if mode == "live":
    st.sidebar.caption(f"分群數量由統計方法依歷史資料自動決定"
                       f"（{MARKET_LABELS[market]} 目前為 {_MARKET_IS_INFO[market]['k']} 群）")
    k_mode = "mainline_h03"
else:
    st.sidebar.caption("每個歷史時間窗的分群數量，各自依統計方法自動決定")
    k_mode = "silhouette_is"

def _default_index(options: list, preferred) -> int:
    """優先選預設建議值，沒有才退回第一個。"""
    return options.index(preferred) if preferred in options else 0


_scheme_info = detail_df.drop_duplicates("scheme").set_index("scheme")


def _scheme_label(code: str) -> str:
    if code not in _scheme_info.index:
        return code
    row = _scheme_info.loc[code]
    tag = "" if row["mode"] == "anchored" else "・滾動對照組"
    return (f"建模期 {int(row['min_is_months'])} 個月／驗證期 {int(row['oos_len_months'])} 個月"
           f"（共 {int(row['n_windows'])} 期{tag}）")


RATIO_LABELS = {"legacy": "標準配置（每群精選 5 檔代表）", "all": "全部策略（不篩選）"}


def _ratio_label(r: str) -> str:
    if r in RATIO_LABELS:
        return RATIO_LABELS[r]
    try:
        return f"擴大配置（全市場的 {float(r):.0%}）"
    except ValueError:
        return r


# ratio 的可選集合取自歷史回測資料（已驗證過的參數空間），確保「✅ 方法已驗證」
# 這個標籤是有依據的——超出這個範圍的設定不在已驗證的範圍內。
_ratio_src = detail_df[(detail_df.tree_key == market) & (detail_df.group == group)
                       & (detail_df.allocation == allocation)]
ratio_options = sorted(_ratio_src["ratio"].unique().tolist()) or ["legacy"]
ratio = st.sidebar.selectbox("持股規模", ratio_options,
                             index=_default_index(ratio_options, "legacy"),
                             format_func=_ratio_label)

scheme = window_no = None
if mode == "replay":
    _avail = _ratio_src[(_ratio_src.k_mode == k_mode) & (_ratio_src.ratio == ratio)]
    scheme_options = sorted(_avail["scheme"].unique().tolist()) or ["A"]
    scheme = st.sidebar.selectbox("回測方案", scheme_options,
                                  index=_default_index(scheme_options, "A"),
                                  format_func=_scheme_label)
    _avail3 = _avail[_avail.scheme == scheme]
    window_options = sorted(int(w) for w in _avail3["window_no"].unique().tolist()) or [1]
    window_no = st.sidebar.selectbox("回測期次", window_options,
                                     index=_default_index(window_options, max(window_options)))
else:
    st.sidebar.caption(f"正式模式沒有「期次」可選：建模區間固定為"
                       f"{MARKET_LABELS[market]} {_MARKET_IS_INFO[market]['is_range']}")

st.sidebar.markdown("---")
st.sidebar.caption("風控上限（可自行調整，預設值依各市場歷史資料設定）")
single_stock_cap = st.sidebar.number_input(
    "單一股票上限 %", value=DEFAULT_SINGLE_STOCK_CAP[market] * 100,
    min_value=0.1, max_value=100.0, step=0.5, key=f"single_cap_{market}") / 100
cluster_cap_equal = st.sidebar.number_input(
    "單一群佔比上限（等量採樣）%", value=DEFAULT_CLUSTER_CAP_EQUAL[market] * 100,
    min_value=0.1, max_value=100.0, step=1.0, key=f"cluster_equal_{market}") / 100
cluster_cap_proportional = st.sidebar.number_input(
    "單一群佔比上限（比例採樣）%", value=DEFAULT_CLUSTER_CAP_PROPORTIONAL[market] * 100,
    min_value=0.1, max_value=100.0, step=1.0, key=f"cluster_prop_{market}") / 100

st.sidebar.markdown("---")
st.sidebar.caption("系統採手動觸發執行，不會自動排程重跑。")


def _current_config() -> RunConfig:
    """把左側目前的設定組成一份 RunConfig——「執行」跟「加入比較清單」共用，
    避免兩處各寫一份，日後改一邊忘了改另一邊。"""
    return RunConfig(
        mode=mode, market=market, group=group, ratio=ratio, allocation=allocation,
        k_mode=k_mode, single_stock_cap=single_stock_cap,
        cluster_cap_equal=cluster_cap_equal, cluster_cap_proportional=cluster_cap_proportional,
        replay_anchor=(ReplayAnchor(scheme=scheme, window_no=int(window_no))
                      if mode == "replay" else None),
    )


run_clicked = st.sidebar.button("執行", type="primary")

# F3（應用層開發追蹤.md §10.3／§10.5-R-A6，輕量版）：加入比較清單，
# 完全不碰現有的 holdings/risk/calib 等 11 個 session_state 鍵——比較清單
# 跟比較結果各自存在新的 key 裡，不干擾現有單組流程。
_compare_configs: list[RunConfig] = st.session_state.setdefault("compare_configs", [])
add_compare_clicked = st.sidebar.button(
    f"加入比較清單（{len(_compare_configs)}/3）",
    disabled=len(_compare_configs) >= 3)
if add_compare_clicked:
    _compare_configs.append(_current_config())
    st.session_state.pop("compare_results", None)   # 清單變了，舊比較結果過期
    st.rerun()

if run_clicked:
    try:
        cfg = _current_config()
        # 正式模式要現場建相關矩陣＋挑代表（6,679×6,679，約 340MB RAM），
        # 比 replay 的查表慢得多，給個 spinner 免得使用者以為當掉
        with st.spinner("正在載入凍結主線樹並挑選代表策略…"
                        if mode == "live" else "讀取凍結窗次…"):
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
        st.session_state.pop("face_comparison", None)

holdings = st.session_state.get("holdings")
risk = st.session_state.get("risk")
calib = st.session_state.get("calib")

# ---------- F3 並排比較（輕量版，§10.3／§10.5-R-A6） ----------
# 只比較彙總數字＋股票層級持股重疊度，不支援 3 組各自解析持股/AI解釋/寫入
# 稽核（那是「完整版」，這次定案不做）。選定一組後按「採用這組設定」，
# 才切回下面正常的單組流程繼續操作。
if _compare_configs:
    with st.expander(f"並排比較（{len(_compare_configs)}/3 組）", expanded=True):
        for i, c in enumerate(_compare_configs):
            cc1, cc2 = st.columns([5, 1])
            cc1.write(f"{i+1}. {MARKET_LABELS[c.market]}｜"
                     f"{GROUP_LABELS.get(c.group, c.group)}｜{_ratio_label(c.ratio)}／"
                     f"{ALLOCATION_LABELS[c.allocation]}"
                     + (f"｜{c.replay_anchor.scheme}-{c.replay_anchor.window_no}"
                        if c.replay_anchor else ""))
            if cc2.button("移除", key=f"remove_compare_{i}"):
                _compare_configs.pop(i)
                st.session_state.pop("compare_results", None)
                st.rerun()

        if st.button("開始比較（依序執行，正式模式每組約 1-2 分鐘）"):
            _results = []
            for c in _compare_configs:
                try:
                    with st.spinner(f"執行 {MARKET_LABELS[c.market]}／"
                                    f"{GROUP_LABELS.get(c.group, c.group)} 中…"):
                        h = run_engine(c)
                    r = assess(h)
                    cb = check_calibration(h)
                except Exception as e:  # noqa: BLE001 — 顯示給使用者看，不是要吞掉錯誤
                    st.error(f"{MARKET_LABELS[c.market]}／{GROUP_LABELS.get(c.group, c.group)}"
                            f" 執行失敗：{e}")
                    _results.append(None)
                else:
                    _results.append((h, r, cb))
            st.session_state["compare_results"] = _results

        _results = st.session_state.get("compare_results")
        if _results:
            rows = []
            for c, res in zip(_compare_configs, _results):
                if res is None:
                    rows.append({"設定": f"{MARKET_LABELS[c.market]}／{GROUP_LABELS.get(c.group, c.group)}",
                                "狀態": "執行失敗"})
                    continue
                h, r, cb = res
                perf = h.performance
                row = {
                    "設定": f"{MARKET_LABELS[c.market]}／{GROUP_LABELS.get(c.group, c.group)}／"
                            f"{_ratio_label(c.ratio)}",
                    "持股數": h.n_members,
                    "驗證狀態": h.validation.get("label", "") if h.validation else "",
                }
                if h.has_oos:
                    row["樣本外年化報酬"] = f"{perf['oos_cagr']:.2%}"
                    row["樣本外最大回撤"] = f"{perf['oos_mdd']:.2%}"
                    row["樣本外 Sharpe"] = f"{perf['oos_sharpe']:.2f}"
                else:
                    row["建模期年化報酬"] = f"{perf['is_cagr']:.2%}"
                    if h.reference_oos:
                        row["同類設定歷史樣本外中位數"] = f"{h.reference_oos['oos_cagr_median']:.2%}"
                row["最大單檔權重(等權)"] = f"{r.max_single_weight:.2%}"
                row["最大群佔比"] = f"{r.max_cluster_share:.2%}"
                row["風控違規數"] = len(r.violations)
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption("⚠️ 不同市場的建模區間長度、風控上限預設值本來就不同"
                      "（台股／跨市場 228 個月、美股 288 個月），並排呈現不代表"
                      "三者是完全同質的比較基準。")

            _adopt_options = [i for i, res in enumerate(_results) if res is not None]
            if _adopt_options:
                _pick = st.selectbox(
                    "採用哪一組設定，切回單組模式繼續操作（解析持股／AI解讀／寫入稽核）？",
                    _adopt_options,
                    format_func=lambda i: f"{i+1}. {MARKET_LABELS[_compare_configs[i].market]}／"
                                          f"{GROUP_LABELS.get(_compare_configs[i].group, _compare_configs[i].group)}")
                if st.button("採用這組設定"):
                    h, r, cb = _results[_pick]
                    st.session_state["holdings"] = h
                    st.session_state["risk"] = r
                    st.session_state["calib"] = cb
                    st.session_state["recorded"] = False
                    st.session_state["recorded_at"] = None
                    st.session_state["memo_result"] = None
                    st.session_state.pop("override_reason", None)
                    st.session_state.pop("stock_detail", None)
                    st.session_state.pop("stock_detail_date", None)
                    st.session_state.pop("face_comparison", None)
                    # 🔴 2026-09-15 code review 抓到的真 bug：這裡原本沒清掉
                    # 比較清單／比較結果／混合結果——「採用這組設定」的說明文字
                    # 明講是「切回單組模式」，但沒清掉的話，下面「並排比較」
                    # 跟「可調式混合」兩個 expander 還是會留在畫面上，讓人以為
                    # 剛採用的單組結果還跟舊的比較/混合綁在一起。
                    st.session_state["compare_configs"] = []
                    st.session_state.pop("compare_results", None)
                    st.session_state.pop("blend_result", None)
                    st.rerun()

# ---------- F1 可調式混合（ensemble，§10.1／§10.5-R-A1~R-A4） ----------
# 直接沿用上面「加入比較清單」的同一份清單當混合腳位，不用再另外做一套
# 選腳位的介面——使用者本來就要先在側邊欄挑好每一組設定再加進清單。
if len(_compare_configs) >= 2:
    with st.expander(f"可調式混合（從上面 {len(_compare_configs)} 組裡混合）",
                     expanded=False):
        st.caption("拉滑桿決定各腳位的資金比例，總和必須是 100%。混合只是把"
                  "已經驗證過的方法／市場依比例線性組合，不是新的獨立方法，"
                  "混合比例本身也沒有被歷史回測驗證過——結果畫面上會清楚標示。")
        _blend_pcts = []
        # 🔴 2026-09-14 code review 抓到的真 bug：`round(100/n)` 在 n=3 時是
        # 33，三個滑桿預設值加總是 99 不是 100——使用者什麼都沒調就會看到
        # 「必須剛好 100%」的警告、按鈕預設是灰的，違背「預設值就能直接用」
        # 的期待。改成餘數塞給最後一個腳位，保證預設值總和一定是 100。
        _n_legs = len(_compare_configs)
        _base_pct = 100 // _n_legs
        _default_pcts = [_base_pct] * (_n_legs - 1) + [100 - _base_pct * (_n_legs - 1)]
        for i, c in enumerate(_compare_configs):
            p = st.slider(
                f"{i+1}. {MARKET_LABELS[c.market]}／{GROUP_LABELS.get(c.group, c.group)} "
                f"比例 %", 0, 100, _default_pcts[i], key=f"blend_pct_{i}")
            _blend_pcts.append(p)
        _pct_total = sum(_blend_pcts)
        st.caption(f"目前總和：{_pct_total}%" +
                  ("" if _pct_total == 100 else " ⚠️ 必須剛好 100% 才能混合"))

        if st.button("混合並檢視結果", disabled=(_pct_total != 100)):
            try:
                with st.spinner("執行各腳位並混合中（含股票層級持股解析，"
                                "第一次連新市場約 1-2 分鐘）…"):
                    # 🔴 2026-09-15 code review 抓到的真 bug：原本只要清單裡有
                    # 任一個 None（可能是還沒跑過、也可能是曾經執行失敗），
                    # 就把**全部**腳位重跑一次——即使已經成功、花了 1-2 分鐘
                    # 建好樹的正式模式腳位也會被丟掉重建。改成只補跑缺的那
                    # 幾個，已經成功的直接沿用快取結果。
                    _results = st.session_state.get("compare_results")
                    if not _results or len(_results) != len(_compare_configs):
                        _results = [None] * len(_compare_configs)
                    for _i, _c in enumerate(_compare_configs):
                        if _results[_i] is None:
                            _h = run_engine(_c)
                            _results[_i] = (_h, assess(_h), check_calibration(_h))
                    st.session_state["compare_results"] = _results
                    legs = [BlendLeg(holdings=res[0], weight=p / 100)
                           for res, p in zip(_results, _blend_pcts) if res is not None]
                    _blend = blend_holdings(legs)

                    _idx = _get_candidate_index()
                    _scs = []
                    for leg in legs:
                        h = leg.holdings
                        _as_of = (datetime.date.today().isoformat() if h.config.mode == "live"
                                 else h.window_info["is_end"])
                        _sd, _dates = _resolve_stock_holdings(h.members, _as_of)
                        _sc = assess_stock_level(
                            h, _sd, as_of="、".join(f"{m}:{d}" for m, d in _dates.items()))
                        _scs.append((_sc, leg.weight))
                    _blend.stock_weights = blend_stock_weights(_scs)
            except Exception as e:  # noqa: BLE001 — 顯示給使用者看，不是要吞掉錯誤
                st.error(f"混合失敗：{e}")
            else:
                st.session_state["blend_result"] = _blend

        _blend = st.session_state.get("blend_result")
        if _blend is not None:
            st.success(f"**{_blend.validation['label']}**　{_blend.validation['reason']}")
            st.warning(_blend.validation["structural_caveat"])
            perf = _blend.performance
            prefix = "oos" if _blend.has_oos else "is"
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric(f"{'樣本外' if _blend.has_oos else '建模期'}年化報酬",
                      f"{perf[f'{prefix}_cagr']:.2%}")
            bc2.metric(f"{'樣本外' if _blend.has_oos else '建模期'}最大回撤",
                      f"{perf[f'{prefix}_mdd']:.2%}")
            bc3.metric(f"{'樣本外' if _blend.has_oos else '建模期'}Sharpe",
                      f"{perf[f'{prefix}_sharpe']:.2f}")
            st.caption(f"混合序列取各腳位共同重疊的 {perf['n_months_compared']} 個月計算，"
                      f"不是把各腳位的績效點估計直接加權平均。")
            if _blend.reference_oos:
                st.caption(f"⚠️ {_blend.reference_oos['caveat']}")
            st.markdown(f"**混合後前 15 大持股（{_blend.n_unique_stocks} 檔不重複股票）**")
            _blend_top = sorted(_blend.stock_weights.items(), key=lambda kv: kv[1],
                               reverse=True)[:15]
            st.dataframe(pd.DataFrame(
                [{"股票代號": s, "混合後權重": f"{w:.3%}"} for s, w in _blend_top]),
                width="stretch", hide_index=True)

tab_holdings, tab_risk, tab_ai, tab_history, tab_l3 = st.tabs(
    ["本期持倉", "風險儀表板", "AI 解讀", "歷史版本比較", "監控 Agent · 人機對話（L3）"])

# L3 是獨立子系統（見上面 `_render_l3_tab` 註解），不依賴左側設定或
# `holdings`，所以放在 `holdings is None` 的提早返回之前渲染。
with tab_l3:
    _render_l3_tab()

if holdings is None:
    with tab_holdings:
        st.info("左側設定完成後按「執行」開始，或用「加入比較清單」同時比較多組設定。")
    st.stop()

# ---------- 本期持倉（L1） ----------
with tab_holdings:
    st.subheader(f"{MARKET_LABELS[holdings.config.market]}｜"
                f"{GROUP_LABELS.get(holdings.config.group, holdings.config.group)}｜"
                f"{_ratio_label(holdings.config.ratio)}／"
                f"{ALLOCATION_LABELS[holdings.config.allocation]}")
    if holdings.validation:
        st.success(f"**{holdings.validation['label']}**　{holdings.validation['reason']}")
        st.warning(holdings.validation["structural_caveat"])

    _ref = holdings.reference_oos
    if holdings.has_oos:
        c1, c2, c3 = st.columns(3)
        c1.metric("持股（策略層級）數", holdings.n_members)
        c2.metric("樣本外年化報酬", f"{holdings.performance['oos_cagr']:.2%}")
        c3.metric("樣本外 Sharpe", f"{holdings.performance['oos_sharpe']:.2f}")
    else:
        # 正式模式沒有樣本外數字可看，主要顯示改成「同類設定的歷史樣本外表現」，
        # 避免把樣本內配適值誤讀成預期報酬。
        c1, c2, c3 = st.columns(3)
        c1.metric("持股（策略層級）數", holdings.n_members)
        if _ref:
            c2.metric("同類設定歷史樣本外報酬中位數", f"{_ref['oos_cagr_median']:.2%}",
                     help=f"取自 {_ref['n_cells']} 組相同市場與選股設定的歷史資料。"
                          f"這是歷史參照，不是本期預測。")
            c3.metric("歷史樣本外報酬警戒線", f"{_ref['oos_cagr_p10']:.2%}")
        st.info("**本次沒有樣本外數字**——因為建模時已使用全部可得的歷史資料，"
               "沒有保留未來可供驗證。上面兩個數字是「同樣設定在歷史上的實際"
               "表現分布」，**不是這一期的預測**。本期建模期間的數字在下方"
               "「績效」區，並附有為什麼不能當預期報酬的說明。")

    _diff = diff_holdings(holdings.members, find_previous(holdings.config),
                          holdings.window_info)
    if _diff["has_previous"]:
        st.info(f"**跟上次相比**（上一次執行於 {_diff['previous_recorded_at'][:19]}）："
               f"新增 {_diff['n_added']} 檔、剔除 {_diff['n_removed']} 檔、"
               f"不變 {_diff['n_unchanged']} 檔")
        if _diff.get("is_chronological") is False:
            st.warning(_diff["direction_note"])
    else:
        st.caption("這是這組設定第一次執行，沒有前一期可比較。")

    with st.expander("時間區間與分群模型（詳細資訊）"):
        _cw1, _cw2 = st.columns(2)
        with _cw1:
            st.markdown("**時間區間**")
            st.json(holdings.window_info)
        with _cw2:
            st.markdown("**分群模型**")
            st.json(holdings.tree_info)

    st.markdown("**績效**")
    perf = holdings.performance
    _shown = {"建模期年化報酬": f"{perf['is_cagr']:.2%}",
             "建模期最大回撤": f"{perf['is_mdd']:.2%}"}
    if "is_sharpe" in perf:
        _shown["建模期 Sharpe"] = f"{perf['is_sharpe']:.2f}"
    if holdings.has_oos:
        _shown.update({"樣本外年化報酬": f"{perf['oos_cagr']:.2%}",
                      "樣本外最大回撤": f"{perf['oos_mdd']:.2%}",
                      "樣本外 Sharpe": f"{perf['oos_sharpe']:.2f}"})
    _shown["候補遞補檔數"] = perf.get("n_backfilled")
    st.json(_shown)
    st.caption("⚠️ **建模期的數字是樣本內配適值，不是預期報酬**：歷史上建模期年化"
              "報酬中位數比實際樣本外報酬高 7.62 個百分點（1.48 倍，900 組歷史"
              "資料中 74.2% 皆然）。但方向不一致——建模期的最大回撤與 Sharpe 反而"
              "比樣本外**更差**，因為建模期涵蓋了 2008 年金融海嘯而樣本外驗證期"
              "都從 2013 年之後開始。**兩者不可並排比較或直接相減。**")

    if _ref:
        st.markdown("**歷史參照與基準**")
        st.json({
            "同類設定樣本外報酬 p10／中位數／p90":
                f"{_ref['oos_cagr_p10']:.2%} ／ {_ref['oos_cagr_median']:.2%} ／ "
                f"{_ref['oos_cagr_p90']:.2%}",
            "同類設定樣本外最大回撤中位數": f"{_ref['oos_mdd_median']:.2%}",
            "市場長期平均報酬基準": f"{_ref['benchmark_cagr']:.2%}",
            "樣本組數": _ref["n_cells"],
        })
        _cap_note = ("　另注意：台股組合相對**市值加權**大盤是落後的（19.55% vs "
                    "20.91%），贏的是**等權市場**（15.02%）——差異來自加權方式"
                    "（台積電佔指數 40.23%），**不可說成「贏大盤」**。"
                    if market == "TW" else "")
        st.caption(f"⚠️ {_ref['caveat']}{_cap_note}")

    if holdings.alternative_groups:
        st.markdown("**候選方案對照**（同一設定下，其他選股邏輯的表現，"
                   "純呈現既有比較結果，不代表建議改用哪一個——這個問題已用"
                   "長期歷史資料回答過）")
        # 正式模式沒有樣本外數字，替代方案只能給建模期數字；欄位名跟著換，
        # 不硬套同一個名字。
        _p = "oos" if holdings.has_oos else "is"
        def _alt_row(name, n, src, note):
            return {"選股邏輯": GROUP_LABELS.get(name, name), "持股數": n,
                    "年化報酬": f"{src[f'{_p}_cagr']:.2%}",
                    "最大回撤": f"{src[f'{_p}_mdd']:.2%}",
                    "Sharpe": f"{src[f'{_p}_sharpe']:.2f}",
                    "會增/減幾檔": note}
        alt_rows = [_alt_row(holdings.config.group, holdings.n_members, perf, "（目前選定）")]
        for g, a in holdings.alternative_groups.items():
            alt_rows.append(_alt_row(g, a["n_members"], a,
                                     f"+{a['n_would_add']} / -{a['n_would_remove']}"))
        st.dataframe(pd.DataFrame(alt_rows), width="stretch", hide_index=True)

        # 🔴 §8-R12 定錨句：單一窗次「看起來選錯」是常態不是例外。
        # 2026-09-11（§9.7 S2）：改成依市場現算，不可沿用寫死的 TW 數字——三市場
        # 開放後那組數字若原樣顯示在 US/XM 上，會把 TW 專屬事實講成通用事實。
        # 🔴 2026-09-15 code review 抓到的真 bug：這一段原本手動重組跟 memo.py
        # 的 `_alt_context()` 幾乎一樣的句子（含同一個 `group == "A_hrp"` 特例
        # 措辭），兩份各自維護，日後改一邊很容易忘記改另一邊（2026-09-14 的
        # baseline bug 就是這樣同時存在於兩邊）。改成直接呼叫 `_alt_context()`，
        # UI 只負責套上自己的顯示格式，不重寫句子本身。
        from app.engine import alt_group_win_rates
        from app.memo import _alt_context
        if alt_group_win_rates(market, baseline=group) is not None:
            st.caption(f"⚠️ {_alt_context(market, group)}")

    st.markdown(f"**持股清單**（{holdings.n_members} 檔策略，"
               "⚠️ 這是策略層級，不是實際持有的個股）")
    st.dataframe(pd.DataFrame({"策略": holdings.members}),
                width="stretch", height=320)

    st.markdown("---")
    st.markdown("**股票層級持股明細**（選填）")
    st.caption("即時查詢資料庫、依每個策略的選股條件現算實際持股——"
              "不是重跑回測。第一次解析要連整個市場的資料，約 1~2 分鐘；"
              "同一個市場之後點擊會用快取，秒開。")

    # 正式模式沒有樣本外邊界日，那兩個選項要拿掉——留著會出現 None 日期。
    _date_options = {"建模期結束日": holdings.window_info["is_end"]}
    if holdings.has_oos:
        _date_options["驗證期起始日"] = holdings.window_info["oos_start"]
        _date_options["驗證期結束日"] = holdings.window_info["oos_end"]
    _date_options["今天（即時）"] = datetime.date.today().isoformat()

    # ⚠️ 先固定「預設選項」再加自訂日期——不然預設值會被新加的自訂選項擠掉，
    # 改變既有行為（原本正式模式預設選「今天」，不該因為多了一個選項就變成
    # 預設選別的）。
    _default_date_idx = 1 if holdings.has_oos else len(_date_options) - 1

    if not holdings.has_oos:
        _is_end_ym = pd.Period(holdings.window_info["is_end"], "M")
        _min_date = (_is_end_ym + 1).start_time.date()   # 建模期結束後第一天，避免前視
        _custom_date = st.date_input(
            "或自訂日期",
            value=max(_min_date, datetime.date.today()), min_value=_min_date,
            key="custom_resolve_date")
        _date_options[f"自訂：{_custom_date.isoformat()}"] = _custom_date.isoformat()
    _date_label = st.selectbox("解析哪個時間點的持股", list(_date_options),
                               index=_default_date_idx)
    if _date_label == "今天（即時）":
        if holdings.has_oos:
            st.caption("⚠️ 策略選擇是這個**歷史時間點**驗證過的結果，只有股票持股是"
                      "即時算的——這個組合（過去的策略＋今天的股票）**不是**正式模式"
                      "該有的樣子。要看「現在該持有什麼」請切到正式模式。")
        else:
            st.caption("✅ **這就是正式模式的完整輸出**：策略組合用全部歷史資料選出，"
                      "個股持股解析到最新可得的交易日。"
                      "⚠️ 實際落點通常不是日曆上的今天——財報依各市場自己的法定"
                      "公告期限生效，且會被該市場最新股價日截斷。"
                      + ("跨市場組合混合台美策略，兩個市場的截斷點可能不同（見下方"
                         "「實際對到交易日」）。" if market == "XM" else ""))

    if st.button("解析持股"):
        try:
            with st.spinner("查資料庫並重建選股條件中..."):
                stock_detail, market_dates = _resolve_stock_holdings(
                    holdings.members, _date_options[_date_label])
        except Exception as e:  # noqa: BLE001 — 顯示給使用者看，不是要吞掉錯誤
            st.error(f"解析失敗：{e}")
        else:
            st.session_state["stock_detail"] = stock_detail
            st.session_state["stock_detail_date"] = market_dates
            # §9.8 S5：績效量測要用「使用者實際輸入的查詢日」，不是各市場
            # 對齊後的交易日字典（那個是拿來顯示的，XM 兩個市場可能不同天）。
            st.session_state["stock_detail_query_asof"] = _date_options[_date_label]

    stock_detail = st.session_state.get("stock_detail")
    if stock_detail is not None:
        _market_dates = st.session_state["stock_detail_date"]
        _dates_display = "、".join(f"{m}:{d}" for m, d in _market_dates.items())
        st.caption(f"實際對到交易日：{_dates_display}"
                  f"（要求日期不是交易日時，取小於等於它的最後一個交易日；"
                  f"跨市場組合混合台美策略，兩個市場的交易日曆不同，可能對到"
                  f"不同日期）")
        n_unique = stock_detail["stock_symbol"].nunique()
        st.write(f"{len(stock_detail)} 筆策略-股票對應，{n_unique} 檔不重複股票")

        # F2（應用層開發追蹤.md §10.2）：違規時可選擇自動修剪，不是取代
        # 「攔下＋人工覆核」，是並存的選項——這個開關的狀態全程用同一個
        # session_state key，稽核紀錄與 AI 解釋兩處呼叫也讀同一個值，
        # 確保整個畫面對「這次要不要修剪」的認知一致。
        auto_trim = st.checkbox(
            "違規時自動修剪（將超額權重依比例分給其他持股，取代人工覆核）",
            value=False, key="auto_trim_stock_cap")
        try:
            _sc = assess_stock_level(holdings, stock_detail, as_of=_dates_display,
                                     auto_trim=auto_trim)
        except Exception as e:  # noqa: BLE001
            st.warning(f"股票層級集中度算不出來：{e}")
            _sc = None
        if _sc is not None:
            st.markdown("**個股集中度**")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("最大單一股票權重", f"{_sc.max_stock_weight:.2%}",
                      f"上限 {holdings.config.single_stock_cap:.2%}",
                      delta_color="inverse" if _sc.violations else "off")
            sc2.metric("最大持股", f"{_sc.max_stock_symbol} "
                                  f"{_sc.names.get(_sc.max_stock_symbol, '')}")
            sc3.metric("違規檔數", len(_sc.violations))
            st.caption("權重計算方式：每檔策略平分整體資金，再由該策略平分給"
                      "自己選中的股票——即「策略等權、策略內持股也等權」。")
            if _sc.n_empty_strategies:
                st.caption(f"ℹ️ 其中 {_sc.n_empty_strategies}/{_sc.n_strategies} 個策略在"
                          f"{_dates_display} 這天篩選條件湊不出任何股票（視同該份額"
                          f"持有現金，其權重不會轉嫁給其他策略，故總權重可能 < 100%）。")
            if _sc.trimmed:
                _n_trimmed = sum(1 for s, w in (_sc.raw_weights or {}).items()
                                 if w > holdings.config.single_stock_cap)
                st.info(f"✂️ 已自動修剪 {_n_trimmed} 檔超標股票至上限，超額部分已"
                       f"按比例分給其他持股。下面顯示的都是**修剪後**的權重。")
            if _sc.violations:
                st.error("🔴 個股集中度違規：")
                for v in _sc.violations[:10]:
                    st.write(f"- {v.detail}")
            elif _sc.trimmed:
                # 修剪後最大權重會剛好卡在上限（trim_to_cap 的定義就是壓到
                # cap），不能講「遠低於上限」——那句只適用於本來就沒違規的
                # 自然狀態，兩種情況要分開講，不然會誤導成「本來就很分散」。
                st.success(f"✅ 已修剪至合規：{_sc.n_unique_stocks} 檔股票中最大權重"
                          f"為 {_sc.max_stock_weight:.2%}，不超過 "
                          f"{holdings.config.single_stock_cap:.2%} 上限")
            else:
                st.success(f"✅ 通過：{_sc.n_unique_stocks} 檔股票中最大權重僅 "
                          f"{_sc.max_stock_weight:.2%}，遠低於 "
                          f"{holdings.config.single_stock_cap:.2%} 上限")
            st.markdown("**前 15 大持股（真實權重）**")
            _top = pd.DataFrame(_sc.top(15))
            _top["weight"] = _top["weight"].map(lambda x: f"{x:.3%}")
            st.dataframe(_top, width="stretch", hide_index=True)

            st.markdown("**已實現績效**（事後量測，跟上面的解釋／風控判斷無關）")
            st.caption("算的是「這批持股從解析日到某個結束日，實際發生過的股價"
                      "變動下賺了多少」——跟下方 AI 解讀的內容完全獨立，"
                      "只是拿同一批已解析的股票權重去對實際股價。")
            _custom_end = st.checkbox("自訂結束日（預設抓每個市場最新可用價格）",
                                      key="perf_custom_end")
            _perf_end_str = None
            if _custom_end:
                _perf_end_date = st.date_input("結束日", value=datetime.date.today(),
                                               key="perf_end_date")
                _perf_end_str = _perf_end_date.isoformat()
            if st.button("查詢已實現報酬"):
                # 🔴 code review：這個按鈕原本沒有 try/except，是全檔唯一一個沒
                # 照專案既有慣例（「解析持股」「產生解釋」都用 try/except 顯示
                # `st.error`）包起來的動作型按鈕——若 `stock_detail_query_asof`
                # 因故不存在（例如頁面熱重載保留了舊 session_state），
                # `pd.Timestamp(None)` 會直接把整頁弄崩潰而不是顯示錯誤訊息。
                try:
                    from app.performance import measure
                    _query_as_of = st.session_state.get("stock_detail_query_asof")
                    if not _query_as_of:
                        raise ValueError("找不到解析持股當時的起算日，請重新點一次"
                                         "「解析持股」再查詢已實現報酬")
                    _perf_idx = _get_candidate_index()
                    _perf_needed = markets_needed(_perf_idx, holdings.members)
                    _perf_md_map = {m: _get_market_data(m) for m in _perf_needed}

                    # 🔴 `measure()` 回傳的 `by_market_benchmark[m]["as_of"/"end"]`
                    # 只是原樣回顯呼叫參數，不是實際用到的（clip 過的）交易日——
                    # 不能拿它們判斷「起訖日是不是被截斷成同一天」。要判斷，必須
                    # 自己比對每個市場真正的價格資料涵蓋範圍
                    # （`md.price_index.max()`），在呼叫 measure() 之前就先擋掉，
                    # 否則不只顯示會誤導，連下面市值加權大盤那段用同一組（已經
                    # 失真的）as_of/end 字串去查 TR 表，也會算出方向顛倒、看似
                    # 合理但實際上「用晚於資料範圍的日期當起點」的錯誤數字。
                    _as_of_ts = pd.Timestamp(_query_as_of)
                    _end_ts = pd.Timestamp(_perf_end_str) if _perf_end_str else None
                    _out_of_range = []
                    for m, md in _perf_md_map.items():
                        _cutoff = md.price_index.max()
                        _this_end = _end_ts if _end_ts is not None else _cutoff
                        if _as_of_ts >= _cutoff or _as_of_ts >= _this_end:
                            _out_of_range.append((m, _cutoff, _end_ts))

                    if _out_of_range:
                        for m, cutoff, custom_end in _out_of_range:
                            if custom_end is not None and _as_of_ts >= custom_end:
                                st.warning(f"⚠️ {m}：指定的結束日（{custom_end.date()}）"
                                          f"不晚於起算日（{_query_as_of}），沒有可衡量的"
                                          f"區間，請選一個更晚的結束日或更早的起算日。")
                            else:
                                st.warning(f"⚠️ {m} 的起算日（{_query_as_of}）已經等於或"
                                          f"晚於該市場價格資料的最後一天"
                                          f"（{cutoff.date()}），沒有可衡量的區間——"
                                          f"不顯示數字，避免誤把「同一天對自己」的 0% "
                                          f"當成真正的已實現報酬。請選一個更早的起算日。")
                    else:
                        with st.spinner("計算已實現報酬中..."):
                            _perf = measure(_perf_md_map, _sc.weights, as_of=_query_as_of,
                                           end=_perf_end_str)
                        pf1, pf2 = st.columns(2)
                        pf1.metric("投組已實現報酬", f"{_perf['portfolio_realized_return']:.2%}")
                        pf2.metric("權重涵蓋率", f"{_perf['portfolio_weight_measured']:.2%}",
                                  f"/{_perf['portfolio_weight_total']:.2%} 目標")
                        for m, b in _perf["by_market_benchmark"].items():
                            _eq = b["equal_weight_benchmark_return"]
                            _eq_str = f"{_eq:.2%}" if _eq is not None else "無法計算"
                            # 顯示用實際 clip 過的區間，不是原樣回顯的請求參數：
                            # 後者在使用者選「今天」這種常見情境下會跟真正用到的
                            # 交易日不同，顯示出來會讓人誤以為量到的是「到今天」
                            # 而非「到最後可得價」。
                            _md_cutoff = _perf_md_map[m].price_index.max()
                            _actual_end = (_end_ts if _end_ts is not None
                                          and _end_ts < _md_cutoff else _md_cutoff)
                            st.write(f"**{m}**（{_query_as_of} → {_actual_end.date()}）："
                                    f"等權大盤 {_eq_str}")
                            if m in ("TW", "US"):
                                _cap_ret = _cap_weighted_market_return(
                                    m, _query_as_of, _actual_end.date().isoformat())
                                if _cap_ret is not None:
                                    _excess = _perf["portfolio_realized_return"] - _cap_ret
                                    st.write(f"　市值加權大盤（真實指數，含股利）：{_cap_ret:.2%}"
                                            f"　→ 投組對市值加權大盤超額：{_excess:+.2%}")
                                else:
                                    st.caption("　市值加權大盤：資料庫查詢失敗或無資料，略過")
                        if holdings.config.market == "XM":
                            st.caption("⚠️ 跨市場組合混合台美策略，不合成單一「跨市場"
                                      "大盤」數字（報酬無法跨市場線性合成），"
                                      "上面台／美兩個市場基準分開看即可。")
                        st.caption(_perf["caveat"])
                except Exception as e:  # noqa: BLE001 — 顯示給使用者看，不是要吞掉錯誤
                    st.error(f"已實現報酬計算失敗：{e}")

        st.dataframe(stock_detail.rename(columns={
            "strategy_uid": "策略", "stock_symbol": "股票代號",
            "company_name": "公司名稱", "stock_market": "市場"}),
            width="stretch", height=280)

        st.markdown("**跨策略重疊度**（同一檔股票被幾個選中策略同時持有）")
        overlap = (stock_detail.groupby(["stock_symbol", "company_name"])["strategy_uid"]
                  .nunique().reset_index(name="n_strategies")
                  .sort_values("n_strategies", ascending=False)
                  .rename(columns={"stock_symbol": "股票代號", "company_name": "公司名稱",
                                   "n_strategies": "被幾個策略選中"}))
        st.dataframe(overlap, width="stretch", height=240)

        # 拿策略選定當下跟現在的持股做差集，呈現目前持股裡有多少是長期常客、
        # 多少是近期才符合條件的新面孔。只在解析日期不是策略選定當下時顯示。
        st.markdown("---")
        st.markdown("**新面孔 vs 老面孔**（目前持股裡，哪些是長期都在的常客、"
                    "哪些是最近才符合條件的新進成分）")
        _is_end = holdings.window_info.get("is_end")
        if _date_options[_date_label] == _is_end:
            st.caption("目前解析的日期就是策略選定當下，沒有時間差可比較。"
                      "選別的日期（例如「今天」）才能看出新舊面孔。")
        elif _is_end is None:
            st.caption("這組設定沒有可當比較基準的時間點。")
        else:
            if st.button("比較新舊面孔"):
                try:
                    from app.new_faces import compare_faces
                    _idx = _get_candidate_index()
                    _needed = markets_needed(_idx, holdings.members)
                    _md_map = {m: _get_market_data(m) for m in _needed}
                    with st.spinner("比較兩個時間點的持股中…"):
                        _fc = compare_faces(_md_map, _idx, holdings.members,
                                           as_of_current=_date_options[_date_label],
                                           as_of_previous=_is_end)
                except Exception as e:  # noqa: BLE001
                    st.error(f"比較失敗：{e}")
                else:
                    st.session_state["face_comparison"] = _fc

            _fc = st.session_state.get("face_comparison")
            if _fc is not None:
                _prev_disp = "、".join(f"{m}:{d.date()}" for m, d in _fc.previous_dates.items())
                _cur_disp = "、".join(f"{m}:{d.date()}" for m, d in _fc.current_dates.items())
                st.caption(f"比較：{_prev_disp}（策略選定當下）→ "
                          f"{_cur_disp}（{_date_label}）")
                f1, f2, f3 = st.columns(3)
                f1.metric("老面孔", len(_fc.old_faces),
                         f"佔現在持股 {_fc.persistence_rate:.1%}",
                         help="兩個時點都持有——長期都在的常客")
                f2.metric("新面孔", len(_fc.new_faces),
                         f"佔現在持股 {_fc.turnover_rate:.1%}",
                         help="只在現在持有——最近才符合條件的新進成分")
                f3.metric("淡出", len(_fc.exited),
                         help="只在策略選定當下持有，現在已經不符合條件")
                st.caption("⚠️ 這個比較橫跨了財報更新（季度換股），"
                          "高換手率是正常現象，不代表策略異常。")
                # 🔴 2026-09-15 使用者抓到的真 bug：XM 不是資料庫層的真實市場
                # （見 IPS §1），Database("XM") 會直接丟 ValueError。跟上面
                # 解析持股一樣，改用 markets_needed() 依策略實際所屬市場
                # （strategy_uid 前綴）分派，需要幾個市場的公司名稱就查幾個。
                _name_markets = markets_needed(_get_candidate_index(), holdings.members)
                names_map = pd.concat([_get_company_names(m) for m in _name_markets])
                names_map = names_map[~names_map.index.duplicated(keep="first")]
                for label, syms in (("老面孔", _fc.old_faces), ("新面孔", _fc.new_faces),
                                    ("淡出", _fc.exited)):
                    with st.expander(f"{label}（{len(syms)} 檔）"):
                        if syms:
                            st.dataframe(pd.DataFrame(
                                {"股票代號": syms,
                                 "公司名稱": [names_map.get(s, "") for s in syms]}),
                                width="stretch", hide_index=True, height=200)
                        else:
                            st.caption("（無）")

# ---------- 風險儀表板（L2） ----------
with tab_risk:
    st.subheader("風控檢查")
    c1, c2, c3 = st.columns(3)
    c1.metric("單檔權重（等權）", f"{risk.max_single_weight:.2%}",
             f"上限 {holdings.config.single_stock_cap:.2%}")
    c2.metric("最大策略群佔比", f"{risk.max_cluster_share:.2%}",
             f"上限 {holdings.config.cluster_cap:.2%}"
             f"（{ALLOCATION_LABELS[holdings.config.allocation]}）")
    c3.metric("涵蓋群數", risk.n_clusters_covered)
    if risk.portfolio_mdd is not None:
        st.caption(f"組合最大回撤：{risk.portfolio_mdd:.2%}　"
                  f"組合年化波動：{risk.portfolio_ann_vol:.2%}")

    st.markdown("**因子曝險與市場／情境分布**")
    _risk_json = {
        "主力因子曝險": {k: f"{v:.2%}" for k, v in risk.factor_exposure_f1.items()},
        "市場分布": {k: f"{v:.2%}" for k, v in risk.market_share.items()},
    }
    # 🔴 2026-09-15 code review 抓到的真 bug：這個統計是用策略完整歷史算出來的
    # 條件式報酬，驗證模式下對早期窗次而言含這一窗當下還沒發生的未來資料，
    # 跟今天在 explain.py／scenario.py 修過的前視問題是同一類，只是這裡（風控
    # 儀表板的顯示）之前沒同步修。只在正式模式顯示，驗證模式改給說明。
    if holdings.config.mode == "live":
        _risk_json["各景氣情境平均報酬"] = {k: f"{v:.2%}" for k, v in risk.regime_avg_ret.items()}
    st.json(_risk_json)
    if holdings.config.mode == "replay":
        st.caption("ℹ️ 驗證模式不顯示「各景氣情境平均報酬」——該統計用策略完整"
                  "歷史計算，對這一窗而言會含當下還沒發生的未來資料，只在正式"
                  "模式提供。")

    st.markdown("---")
    st.subheader("校準監控")
    st.caption(f"依 {calib.thresholds.n_cells} 組歷史資料設定的警戒線"
              "（示警用，不是強制關卡）")
    if calib.evaluable:
        cc1, cc2 = st.columns(2)
        cc1.metric("樣本外年化報酬", f"{calib.oos_cagr:.2%}",
                  f"警戒線 {calib.thresholds.oos_cagr_p10:.2%}",
                  delta_color="inverse" if calib.below_cagr else "off")
        cc2.metric("樣本外風險調整後報酬", f"{calib.oos_calmar:.2f}",
                  f"警戒線 {calib.thresholds.oos_calmar_p10:.2f}",
                  delta_color="inverse" if calib.below_calmar else "off")
        if calib.flagged:
            st.warning("⚠️ 這次表現明顯偏離歷史常態分布，只是提醒，不會攔下執行")
    else:
        st.info("**狀態：尚無樣本外資料可比對**（警戒線已設定，等待未來實際表現）")
        cc1, cc2 = st.columns(2)
        cc1.metric("年化報酬警戒線（日後基準）", f"{calib.thresholds.oos_cagr_p10:.2%}")
        cc2.metric("風險調整後報酬警戒線（日後基準）", f"{calib.thresholds.oos_calmar_p10:.2f}")
        st.caption(calib.note)

    st.markdown("---")
    st.subheader("執行覆核與稽核紀錄")

    _sd = st.session_state.get("stock_detail")
    _sd_date = st.session_state.get("stock_detail_date")   # {市場: 交易日} dict
    if _sd is not None and len(_sd):
        _sd_date_disp = ("、".join(f"{m}:{d}" for m, d in _sd_date.items())
                         if _sd_date else "")
        st.caption(f"📎 稽核紀錄將一併存入股票層級持股："
                  f"{_sd['stock_symbol'].nunique()} 檔不重複股票（as of {_sd_date_disp}）")
    else:
        st.caption("📎 尚未解析股票層級持股——稽核紀錄會誠實記成「本次未解析」。"
                  "想留完整持股軌跡的話，先到「本期持倉」分頁按「解析持股」。")

    def _do_record(reason):
        _s = st.session_state.get("stock_detail")
        _conc = None
        if _s is not None and len(_s):
            try:      # §8-R5：股票層級集中度判決要一起留痕
                _dates = st.session_state.get("stock_detail_date") or {}
                _conc = assess_stock_level(
                    holdings, _s, as_of="、".join(f"{m}:{d}" for m, d in _dates.items()),
                    auto_trim=st.session_state.get("auto_trim_stock_cap", False))
            except Exception:  # noqa: BLE001 — 算不出來就不記，不能因此擋掉稽核寫入
                _conc = None
        # §9.7 S4：live 模式若已產生過解釋（app/explain.py），一併存進稽核紀錄——
        # §9.8 前瞻驗證要求「解釋先進 audit_log.jsonl 並 commit，才能算績效」。
        _explanation = (st.session_state.get("explain_result")
                       if holdings.config.mode == "live" else None)
        return record(holdings, risk, calibration=calib, override_reason=reason,
                      stock_holdings=(_s.rename(columns={"stock_symbol": "stock_id"})
                                     if _s is not None and len(_s) else None),
                      stock_as_of=st.session_state.get("stock_detail_date"),
                      stock_concentration=_conc, explanation=_explanation)

    if st.session_state.get("recorded"):
        st.success(f"✅ 已寫入稽核紀錄：{st.session_state['recorded_at']}")
    elif risk.violations:
        st.error("🔴 風控違規，需人工覆核才能繼續：")
        for v in risk.violations:
            st.write(f"- {v.detail}")
        reason = st.text_input("覆核原因（必填，會如實寫進稽核紀錄，不做內容驗證）",
                               key="override_reason")
        if st.button("覆核通過並記錄", disabled=not reason.strip()):
            entry = _do_record(reason)
            st.session_state["recorded"] = True
            st.session_state["recorded_at"] = entry["recorded_at"]
            st.rerun()
    else:
        st.success("✅ 風控通過，無違規")
        if st.button("寫入稽核紀錄"):
            entry = _do_record(None)
            st.session_state["recorded"] = True
            st.session_state["recorded_at"] = entry["recorded_at"]
            st.rerun()

# ---------- AI 解讀（L3） ----------
with tab_ai:
    # 🔴 2026-09-15：兩種模式現在共用同一套解釋 agent（原本驗證模式只有六欄位
    # 的轉譯器 memo.py，使用者明確要求「不是我要的」，改成跟正式模式一樣）。
    # 能這樣做是因為 R15（群知識庫全樣本前視）／H8（每窗群 id 對不上）已經在
    # `explain.assemble_facts()` 裡修好：驗證模式改用該窗次自己現場重建的樹
    # 現算群統計，不讀主線樹的全樣本群知識庫。R16（基準涵蓋率）查證後不適用
    # （這裡用的基準跟 R16 講的那組是不同的東西）。細節見
    # 應用層開發追蹤.md §10.7。`memo.py` 六欄位版本仍保留給 `cli.py` 用，
    # 不刪除，只是這個分頁不再呼叫它。
    st.subheader("AI 解讀")
    st.caption("這裡的 LLM **可以做事實推論**（把提供的數字放在一起講出關聯"
              "或型態），但**不能做價值判決**（例如建議換方案）。")
    if holdings.config.mode == "replay":
        st.caption("ℹ️ 驗證模式：群身份與機制敘述改用這一窗自己重建的樹現場"
                  "計算，只涵蓋這一窗的建模期間，不套用正式模式那種全樣本"
                  "群知識庫（避免引用這一窗當下還沒發生的未來資訊）——需要"
                  "現場重建樹，第一次執行約需 1-2 分鐘。")

    stock_detail = st.session_state.get("stock_detail")
    if stock_detail is None:
        st.info("請先到「本期持倉」分頁按「解析持股」——AI 解讀需要股票層級"
                "持股資料才能產生。")
    else:
        # 🔴 2026-09-15 使用者要求拿掉 dry-run 選項——「產生解釋」一律真的
        # 呼叫 LLM。model purpose 固定用 "app_memo"：這是驗證模式／正式模式
        # 大量真呼叫（§10.10／§10.11／§9.8）全程實測過能用的設定，見同日
        # 稍早修過的「尚未設定此功能專用模型」誤導警告 bug。
        dry_run = False
        purpose = "app_memo"

        if st.button("產生解釋"):
            try:
                from app.explain import generate as generate_explain

                _market_dates = st.session_state["stock_detail_date"]
                _sc = assess_stock_level(
                    holdings, stock_detail,
                    as_of="、".join(f"{m}:{d}" for m, d in _market_dates.items()),
                    auto_trim=st.session_state.get("auto_trim_stock_cap", False))
                _idx = _get_candidate_index()
                _needed = markets_needed(_idx, holdings.members)
                _md_map = {m: _get_market_data(m) for m in _needed}
                # ⚠️ 不在這裡先建 footprint——讓 `generate_explain` 內部依
                # `holdings.config.mode` 自己決定要用主線樹（live）還是這一窗
                # 現場重建的樹（replay），呼叫端傳錯樹的風險就不存在。
                with st.spinner("組裝資料並產生解釋中…" if holdings.config.mode == "live"
                                else "重建這一窗的分群樹並組裝資料中（約 1-2 分鐘）…"):
                    result = generate_explain(
                        holdings, risk, calib, stock_detail=stock_detail,
                        stock_concentration=_sc, market_dates=_market_dates,
                        md_map=_md_map,
                        face_comparison=st.session_state.get("face_comparison"),
                        dry_run=dry_run, purpose=purpose)
            except RuntimeError as e:
                st.error(f"🔴 {e}")
                st.session_state["explain_result"] = None
            else:
                st.session_state["explain_result"] = result

        explain_result = st.session_state.get("explain_result")
        if explain_result:
            labels = {
                "strategy_footprint_note": "策略組成與群特性",
                "stock_holdings_note": "持股與重疊度",
                "mechanism_note": "為什麼這樣選股會賺錢",
                "character_note": "追高股 或 長期績優股",
                "concentration_risk_note": "集中度風險",
                "structure_risk_note": "結構風險",
                "scenario_note": "情境比對",
                "change_note": "跟上次相比",
                "alternative_note": "其他選股邏輯比較",
                "caveat": "限制說明",
            }
            for k, label in labels.items():
                st.markdown(f"**{label}**")
                st.write(explain_result["explanation"][k])
            if holdings.config.mode == "replay":
                st.markdown("**事後機制歸因**（已知這一窗實際結果後回頭指出可能"
                           "機制，不是預測；每條皆附可查核的歸因判準）")
                flag_label, crit_label = "觀察", "歸因判準"
            else:
                st.markdown("**風險提示**（每條皆附可事後驗證的檢驗標準）")
                flag_label, crit_label = "風險", "檢驗標準"
            for rf in explain_result["explanation"]["risk_flags"]:
                st.write(f"- **{flag_label}**：{rf['risk']}")
                st.caption(f"　{crit_label}：{rf['verification_criterion']}")
            if explain_result["dry_run"]:
                st.caption("此為 dry-run 內容，未實際呼叫 LLM")
            else:
                # 自動一致性檢查——沒抓到問題不代表這份解釋一定沒錯（只查
                # 數字，不查敘述邏輯），但這是第一道防線，通過是必要條件
                # 不是充分條件。
                if explain_result["leakage_check"]:
                    st.error("自動一致性檢查發現問題：\n" +
                            "\n".join(explain_result["leakage_check"]))
                else:
                    st.success("✅ 自動一致性檢查通過（解釋裡的數字都能對回"
                              "判決資料）")
            with st.expander("查看提供給 AI 的完整客觀資料"):
                # facts 裡有些欄位直接來自 parquet（numpy 純量型別），st.json()
                # 不吃這些型別——跟 build_prompt() 一樣先用 json 往返轉成純
                # Python 型別再顯示，不是顯示邏輯本身能處理，是資料型別問題。
                st.json(json.loads(json.dumps(explain_result["facts"],
                                              ensure_ascii=False, default=str)))

# ---------- 歷史版本比較（F1） ----------
with tab_history:
    st.subheader("歷史版本比較")
    st.caption("每一次執行都會留下紀錄，全部保留、可回頭查詢比對")
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
                    "執行時間": e["recorded_at"],
                    "市場": MARKET_LABELS.get(cfg["market"], cfg["market"]),
                    "選股邏輯": GROUP_LABELS.get(cfg["group"], cfg["group"]),
                    "持股規模": _ratio_label(cfg["ratio"]),
                    "採樣方式": ALLOCATION_LABELS.get(cfg["allocation"], cfg["allocation"]),
                    "回測方案": anchor.get("scheme"), "回測期次": anchor.get("window_no"),
                    "持股數": e["n_members"],
                    "樣本外年化報酬": (f"{e['performance']['oos_cagr']:.2%}"
                                if e["performance"].get("oos_cagr") is not None else None),
                    "樣本外最大回撤": (f"{e['performance']['oos_mdd']:.2%}"
                               if e["performance"].get("oos_mdd") is not None else None),
                    "違規數": len(e["risk"]["violations"]),
                    "覆核原因": e.get("override_reason") or "",
                })
    if not rows:
        st.info("目前還沒有任何稽核紀錄")
    else:
        hist_df = pd.DataFrame(rows).sort_values("執行時間", ascending=False)
        st.dataframe(hist_df, width="stretch")
