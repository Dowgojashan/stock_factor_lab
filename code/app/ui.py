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
from app.config import ReplayAnchor, RunConfig  # noqa: E402
from app.engine import DETAIL_PATH  # noqa: E402
from app.engine import run as run_engine  # noqa: E402
from app.memo import generate as generate_memo  # noqa: E402
from app.risk import assess, assess_stock_level  # noqa: E402
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

st.title("因子選股應用層")

# ---------- 設定與執行（L0/L1） ----------
st.sidebar.header("設定與執行")

# 2026-09-10（§7 P2）：兩種模式並存。正式模式用凍結主線樹即時挑代表（無 OOS），
# 驗證模式讀凍結窗次（有 OOS）。⚠️ 探索模式（任意 IS 區間）決定不做，見 §7.5 G5。
MODE_LABELS = {
    "live": "正式模式（用全歷史建的樹選出現在該持有什麼）",
    "replay": "驗證模式（重放某個歷史窗次，有樣本外可比對）",
}
mode = st.sidebar.radio("模式", ["live", "replay"],
                        format_func=lambda m: MODE_LABELS[m])

# 🔴 2026-09-11（§9.5 H2）：原 A1「先只做台股」已作廢，三市場全開。
# 理由：研究唯一證實的 HRP 貢獻是跨市場回撤控制（M-03），且 XM 樹 3 群完全按
# 市場切開（群1=100%台股／群2/3=100%美股）——只有用 XM 樹才拿得到跨市場分散。
MARKET_LABELS = {"TW": "台股（TW）", "US": "美股（US）", "XM": "跨市場（XM）"}
market = st.sidebar.selectbox("市場", ["TW", "US", "XM"],
                              format_func=lambda m: MARKET_LABELS[m])
if market == "XM":
    st.sidebar.caption("⚠️ XM 不是真實市場，是 TW＋US 策略的聯集（3 群完全按市場"
                       "切開，不混合）。股票層級解析會依策略歸屬分派到兩個市場的"
                       "資料庫，兩者交易日曆不同，可能對到不同日期")

# 三市場的 IS 長度／k 不同（contracts.HRP_WINDOWS／H-03），不可共用同一句文字
_MARKET_IS_INFO = {
    "TW": {"is_range": "2007-01~最新可用月（228 個月）", "k": 6},
    "US": {"is_range": "2002-01~最新可用月（288 個月）", "k": 7},
    "XM": {"is_range": "2007-01~最新可用月（228 個月）", "k": 3},
}
if mode == "live":
    _info = _MARKET_IS_INFO[market]
    st.caption(f"**正式模式**：IS＝{_info['is_range']}，用 `_frozen/stage3` 凍結主線樹"
               f"（k={_info['k']}）即時挑代表。依定義沒有樣本外——IS 用掉全部資料。")
else:
    st.caption("**驗證模式**：讀凍結的 walk-forward 窗次，有真實 OOS 可比對。"
               "用來檢視方法在歷史上的表現，不是「現在該買什麼」。")

group = st.sidebar.selectbox("group", ["A_hrp", "D_top_cagr", "E_top_calmar"])

# 2026-09-10 使用者回饋：「分配」（等量/比例）改用「採樣」；下拉選單顯示中文說明，
# 底層值（"equal"/"proportional"）不變——RunConfig／CSV 欄位仍用原字串，只改顯示文字。
ALLOCATION_LABELS = {"equal": "等量採樣", "proportional": "比例採樣"}
allocation = st.sidebar.selectbox(
    "採樣方式", ["equal", "proportional"], format_func=lambda a: ALLOCATION_LABELS[a])

if mode == "live":
    st.sidebar.caption(f"k_mode：mainline_h03——k 來自 `_frozen/stage3` 主線樹，"
                       f"由 H-03 用輪廓係數在完整共同窗上選出（{MARKET_LABELS[market]} "
                       f"k={_MARKET_IS_INFO[market]['k']}）。"
                       "語意上就是 silhouette_is，因為正式模式的 IS 即完整窗（§7.4b）")
    k_mode = "mainline_h03"
else:
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


# ratio 的可選集合取自凍結表（驗證過的參數空間）。正式模式雖然不查表，但刻意
# 沿用同一組選項——G2/G7 的「✅ 方法已驗證」就是靠「方法論參數落在驗證集合內」
# 成立的，開放表外的比例會讓那個標籤失效。
_ratio_src = detail_df[(detail_df.tree_key == market) & (detail_df.group == group)
                       & (detail_df.allocation == allocation)]
ratio_options = sorted(_ratio_src["ratio"].unique().tolist()) or ["legacy"]
ratio = st.sidebar.selectbox("ratio", ratio_options, index=_default_index(ratio_options, "legacy"))

scheme = window_no = None
if mode == "replay":
    _avail = _ratio_src[(_ratio_src.k_mode == k_mode) & (_ratio_src.ratio == ratio)]
    scheme_options = sorted(_avail["scheme"].unique().tolist()) or ["A"]
    scheme = st.sidebar.selectbox("窗口方案（H-26）", scheme_options,
                                  index=_default_index(scheme_options, "A"),
                                  format_func=_scheme_label)
    _avail3 = _avail[_avail.scheme == scheme]
    window_options = sorted(int(w) for w in _avail3["window_no"].unique().tolist()) or [1]
    window_no = st.sidebar.selectbox("window_no", window_options,
                                     index=_default_index(window_options, max(window_options)))
else:
    # 正式模式依定義只有一個 IS（contracts.HRP_WINDOWS[market]），沒有窗次可選。
    # §7.5 G5：不給日期選擇器——探索模式決定不做。
    st.sidebar.caption(f"正式模式沒有「窗次」可選：IS 依定義就是錨點到最新可用月"
                       f"（{MARKET_LABELS[market]} {_MARKET_IS_INFO[market]['is_range']}）")

st.sidebar.markdown("---")
st.sidebar.caption("風控上限（C1：可自訂，預設值依市場各自反推，見應用層開發追蹤.md §3 C2/C3、"
                   "§9.7 I-4）")
# 🔴 2026-09-11（§9.7 I-4）：三市場門檻不同（US/XM 沿用 TW 數字會讓門檻形同虛設，
# 見 S0 的實測反推），不可寫死同一組預設值。用市場名當 widget key 的一部分，
# 讓 Streamlit 換市場時重新以該市場的預設值渲染（同 key 只在首次渲染吃 value=）。
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
st.sidebar.caption("E2：純手動觸發。B2 慢時鐘「距下次重跑還有多久」需要 live 模式"
                   "才有實際時鐘可比對，尚未實作前不顯示假倒數")

run_clicked = st.sidebar.button("執行", type="primary")

if run_clicked:
    try:
        cfg = RunConfig(
            mode=mode, market=market, group=group, ratio=ratio, allocation=allocation,
            k_mode=k_mode, single_stock_cap=single_stock_cap,
            cluster_cap_equal=cluster_cap_equal, cluster_cap_proportional=cluster_cap_proportional,
            replay_anchor=(ReplayAnchor(scheme=scheme, window_no=int(window_no))
                          if mode == "replay" else None),
        )
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

if holdings is None:
    st.info("左側設定完成後按「執行」開始。")
    st.stop()

tab_holdings, tab_risk, tab_ai, tab_history = st.tabs(
    ["本期持倉", "風險儀表板", "AI 解讀", "歷史版本比較"])

# ---------- 本期持倉（L1） ----------
with tab_holdings:
    st.subheader(f"{holdings.config.market}｜{holdings.config.group}｜"
                f"{holdings.config.ratio}／{holdings.config.allocation}")
    # G7 三態驗證標籤 + 結構性限制（一定要一起顯示，不能只放 ✅）
    if holdings.validation:
        st.success(f"**{holdings.validation['label']}**　{holdings.validation['reason']}")
        st.warning(holdings.validation["structural_caveat"])

    _ref = holdings.reference_oos
    if holdings.has_oos:
        c1, c2, c3 = st.columns(3)
        c1.metric("持股（策略層級）數", holdings.n_members)
        c2.metric("OOS CAGR", f"{holdings.performance['oos_cagr']:.2%}")
        c3.metric("OOS Sharpe", f"{holdings.performance['oos_sharpe']:.2f}")
    else:
        # 🔴 R13：正式模式沒有 OOS。**主要績效顯示改成歷史參照分布**，
        # 不把本次的 is_cagr 放在 metric 位置——那是樣本內配適值，放在最醒目的
        # 位置會被讀成「預期報酬」，而實測 IS CAGR 中位數比 OOS 高 7.62pp。
        c1, c2, c3 = st.columns(3)
        c1.metric("持股（策略層級）數", holdings.n_members)
        if _ref:
            c2.metric("同類設定歷史 OOS 中位數", f"{_ref['oos_cagr_median']:.2%}",
                     help=f"取自凍結表 {_ref['n_cells']} 格同 market/group/ratio/allocation "
                          f"的設定。這是歷史參照，不是本期預測。")
            c3.metric("歷史 OOS p10（警戒線）", f"{_ref['oos_cagr_p10']:.2%}")
        st.info("**本模式沒有樣本外數字**——IS 用掉全部可用資料，依定義沒有 OOS。"
               "上面兩個數字是「同類設定在歷史上的樣本外表現分布」，"
               "**不是這一期的預測**。本期的樣本內數字在下方「績效」區，"
               "並附有為什麼不能當預期報酬的說明。")

    # 2026-09-10（應用層§6落差①）：跟上一次同一組 RunConfig 身份執行結果的比較。
    # §8-R6：傳入 window_info 讓它能判斷是不是逆序比較。
    _diff = diff_holdings(holdings.members, find_previous(holdings.config),
                          holdings.window_info)
    if _diff["has_previous"]:
        st.info(f"**跟上次相比**（上一次執行於 {_diff['previous_recorded_at'][:19]}）："
               f"新增 {_diff['n_added']} 檔、剔除 {_diff['n_removed']} 檔、"
               f"不變 {_diff['n_unchanged']} 檔")
        if _diff.get("is_chronological") is False:
            st.warning(_diff["direction_note"])
    else:
        st.caption("這是這組設定（mode/market/group/ratio/allocation/k_mode）第一次執行，"
                  "沒有前一期可比較。")

    _cw1, _cw2 = st.columns(2)
    with _cw1:
        st.markdown("**窗次資訊**")
        st.json(holdings.window_info)
    with _cw2:
        st.markdown("**樹**")
        st.json(holdings.tree_info)

    st.markdown("**績效**")
    # 2026-09-10 使用者回饋：數字一律顯示到小數點第二位——原始 dict 是未格式化的
    # 浮點數（例如 0.1424137781102339），直接 st.json 會整串洩出來，這裡先轉成
    # 格式化字串再顯示，底層 holdings.performance 的原始精度不受影響。
    perf = holdings.performance
    _shown = {"is_cagr": f"{perf['is_cagr']:.2%}", "is_mdd": f"{perf['is_mdd']:.2%}"}
    if "is_sharpe" in perf:
        _shown["is_sharpe"] = f"{perf['is_sharpe']:.2f}"
    if holdings.has_oos:
        _shown.update({"oos_cagr": f"{perf['oos_cagr']:.2%}",
                      "oos_mdd": f"{perf['oos_mdd']:.2%}",
                      "oos_sharpe": f"{perf['oos_sharpe']:.2f}"})
    _shown["n_backfilled"] = perf.get("n_backfilled")
    st.json(_shown)
    # 🔴 R13：IS 不是預期報酬，而且跟 OOS 不可比——兩種模式都要講，
    # 因為 replay 也會把 is_* 跟 oos_* 並排顯示。
    st.caption("⚠️ **is_\\* 是樣本內配適值，不是預期報酬**：歷史上 IS CAGR 中位數比實際 "
              "OOS 高 7.62pp（1.48 倍，900 格中 74.2% 皆然）。但方向不一致——"
              "IS MDD 與 IS Sharpe 反而比 OOS **更差**，因為 IS 涵蓋 2008 金融海嘯而 "
              "OOS 窗都從 2013 之後開始。**兩者不可並排比較或相減**（應用層 §8-R13）。")

    # R8②：基準對照。之前 UI 完全沒有基準，一份不含基準的投資備忘錄不成立。
    if _ref:
        st.markdown("**歷史參照與基準**（R8②）")
        st.json({
            "同類設定 OOS CAGR p10／中位數／p90":
                f"{_ref['oos_cagr_p10']:.2%} ／ {_ref['oos_cagr_median']:.2%} ／ "
                f"{_ref['oos_cagr_p90']:.2%}",
            "同類設定 OOS MDD 中位數": f"{_ref['oos_mdd_median']:.2%}",
            "自建宇宙基準 CAGR": f"{_ref['benchmark_cagr']:.2%}",
            "樣本格數": _ref["n_cells"],
        })
        _m17_note = ("　另注意：台股 A_hrp 相對**市值加權**大盤是輸的（M-17：19.55% vs "
                    "20.91%），贏的是**等權市場**（15.02%）——差異來自加權方式"
                    "（台積電佔指數 40.23%），**不可說成「贏大盤」**。"
                    if market == "TW" else "")
        st.caption(f"⚠️ {_ref['caveat']}{_m17_note}")

    if holdings.alternative_groups:
        st.markdown("**候選方案對照**（應用層§6落差③：同一格設定下的其他 H-12 對照組，"
                   "純呈現既有比較結果，不代表建議改用哪一個——H-26/H-27/M-03 已用"
                   "統計證據回答過這個問題）")
        # 正式模式沒有 OOS，替代方案只能給 is_*；欄位名跟著換，不硬套同一個名字。
        _p = "oos" if holdings.has_oos else "is"
        def _alt_row(name, n, src, note):
            return {"group": name, "n_members": n,
                    f"{_p}_cagr": f"{src[f'{_p}_cagr']:.2%}",
                    f"{_p}_mdd": f"{src[f'{_p}_mdd']:.2%}",
                    f"{_p}_sharpe": f"{src[f'{_p}_sharpe']:.2f}",
                    "會增/減幾檔": note}
        alt_rows = [_alt_row(holdings.config.group, holdings.n_members, perf, "（目前選定）")]
        for g, a in holdings.alternative_groups.items():
            alt_rows.append(_alt_row(g, a["n_members"], a,
                                     f"+{a['n_would_add']} / -{a['n_would_remove']}"))
        st.dataframe(pd.DataFrame(alt_rows), width="stretch", hide_index=True)

        # 🔴 §8-R12 定錨句：單一窗次「看起來選錯」是常態不是例外。
        # 2026-09-11（§9.7 S2）：改成依市場現算，不可沿用寫死的 TW 數字——三市場
        # 開放後那組數字若原樣顯示在 US/XM 上，會把 TW 專屬事實講成通用事實。
        from app.engine import alt_group_win_rates
        _wr = alt_group_win_rates(market)
        if _wr is not None:
            st.caption(
                f"⚠️ **A_hrp 在報酬類指標上輸給 D／E 是已知的系統性結果，不是本期特例**——"
                f"{_wr['n_cells']} 格{MARKET_LABELS[market]}歷史逐格對照中，A_hrp 僅"
                f"**{_wr['calmar_vs_E_top_calmar']:.1%}**（Calmar）／"
                f"**{_wr['oos_cagr_vs_E_top_calmar']:.1%}**（OOS CAGR）勝過 E_top_calmar。"
                f"選用 A_hrp 的理由不是報酬優勢"
                "（M-03/M-03b 已證實 HRP 分群不提供報酬優勢），而是分散度與跨市場回撤控制。"
                "是否更換方法屬政策層決定，不是單期可以判斷的事。")

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
    # 正式模式沒有 OOS 邊界日，那兩個選項要拿掉——留著會出現 None 日期。
    _date_options = {"IS 結束日（is_end）": holdings.window_info["is_end"]}
    if holdings.has_oos:
        _date_options["OOS 起始日（oos_start）"] = holdings.window_info["oos_start"]
        _date_options["OOS 結束日（oos_end）"] = holdings.window_info["oos_end"]
    _date_options["今天（即時／快時鐘）"] = datetime.date.today().isoformat()

    # ⚠️ 先固定「預設選項」再加自訂日期——不然預設值會被新加的自訂選項擠掉，
    # 改變既有行為（原本正式模式預設選「今天」，不該因為多了一個選項就變成
    # 預設選別的）。
    _default_date_idx = 1 if holdings.has_oos else len(_date_options) - 1

    # 🔴 2026-09-11（§9.8 前瞻驗證）：正式模式需要能指定任意日期（2026-01-01／
    # 03-31／05-15 三次前瞻驗證的執行時點），不是只有「IS結束日」跟「今天」兩個
    # 選項。⚠️ 這不是重開「探索模式」的口子——策略選擇仍固定用主線樹（§7.5 G5
    # 決定不做的是任意 IS 區間／手動 k），這裡動的只是快時鐘的解析日期，
    # `resolve_holdings(md, row, as_of)` 本來就支援任意日期，只是 UI 沒開。
    if not holdings.has_oos:
        _is_end_ym = pd.Period(holdings.window_info["is_end"], "M")
        _min_date = (_is_end_ym + 1).start_time.date()   # IS 結束後第一天，避免前視
        _custom_date = st.date_input(
            "或自訂日期（§9.8 前瞻驗證：2026-01-01／2026-03-31／2026-05-15）",
            value=max(_min_date, datetime.date.today()), min_value=_min_date,
            key="custom_resolve_date")
        _date_options[f"自訂：{_custom_date.isoformat()}"] = _custom_date.isoformat()
    _date_label = st.selectbox("解析哪個時間點的持股", list(_date_options),
                               index=_default_date_idx)
    if _date_label == "今天（即時／快時鐘）":
        if holdings.has_oos:
            st.caption("⚠️ 策略選擇是這個**歷史窗次**凍結驗證過的結果，只有股票持股是"
                      "即時算的——這個組合（舊窗的策略＋今天的股票）**不是**正式模式"
                      "該有的樣子。要看「現在該持有什麼」請切到正式模式。")
        else:
            st.caption("✅ **這就是正式模式的完整輸出**：慢時鐘（策略）用到 2025-12 為止的"
                      "全歷史選出，快時鐘（股票）解析到最新可得的交易日。"
                      "⚠️ 實際落點通常不是日曆上的今天——財報套各市場自己的法定公告"
                      "期限、且會被該市場最新股價日截斷（§7.4）。"
                      + ("XM 混合台美策略，兩個市場的截斷點可能不同（見下方「實際對到"
                         "交易日」）。" if market == "XM" else ""))

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

    stock_detail = st.session_state.get("stock_detail")
    if stock_detail is not None:
        _market_dates = st.session_state["stock_detail_date"]
        _dates_display = "、".join(f"{m}:{d}" for m, d in _market_dates.items())
        st.caption(f"實際對到交易日：{_dates_display}"
                  f"（要求日期不是交易日時，取小於等於它的最後一個交易日；"
                  f"XM 投組混合台美策略，兩個市場的交易日曆不同，可能對到不同日期）")
        n_unique = stock_detail["stock_symbol"].nunique()
        st.write(f"{len(stock_detail)} 筆策略-股票對應，{n_unique} 檔不重複股票")

        # 🔴 §8-R5（P3）：真正的股票層級集中度風控。上面 C2 量的是「單一策略」
        # 權重（1/n_members），在正常組合規模下永遠不會觸發，而且量錯對象——
        # 老師 9-8 講的台積電集中度是股票層級的事。
        # ⚠️ 權重公式（1/N策略 × 1/n_i）純以 uid/股票代號計數，跟市場/幣別無關，
        # XM 混合台美持股不需要另外處理——見 risk.py assess_stock_level docstring。
        try:
            _sc = assess_stock_level(holdings, stock_detail, as_of=_dates_display)
        except Exception as e:  # noqa: BLE001
            st.warning(f"股票層級集中度算不出來：{e}")
            _sc = None
        if _sc is not None:
            st.markdown("**股票層級集中度（C2 真正該量的東西，§8-R5）**")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("最大單一股票權重", f"{_sc.max_stock_weight:.2%}",
                      f"上限 {holdings.config.single_stock_cap:.2%}",
                      delta_color="inverse" if _sc.violations else "off")
            sc2.metric("最大持股", f"{_sc.max_stock_symbol} "
                                  f"{_sc.names.get(_sc.max_stock_symbol, '')}")
            sc3.metric("股票層級違規", len(_sc.violations))
            st.caption("權重定義：w(股票 s) = Σ_{持有 s 的策略 i} (1/N策略) × (1/n_i)，"
                      "即「策略等權、策略內部對自己的持股等權」——跟研究鏈"
                      "`_portfolio_series` 的每月拉回等權是同一套慣例。")
            if _sc.n_empty_strategies:
                st.caption(f"ℹ️ 其中 {_sc.n_empty_strategies}/{_sc.n_strategies} 個策略在"
                          f"{_dates_display} 這天篩選條件湊不出任何股票（視同該份額持有現金，"
                          f"其權重份額不會轉嫁給其他策略，故 Σweight 可能 < 100%）。")
            if _sc.violations:
                st.error("🔴 股票層級集中度違規：")
                for v in _sc.violations[:10]:
                    st.write(f"- {v.detail}")
            else:
                st.success(f"✅ 通過：{_sc.n_unique_stocks} 檔股票中最大權重僅 "
                          f"{_sc.max_stock_weight:.2%}，遠低於 "
                          f"{holdings.config.single_stock_cap:.2%} 上限")
            st.markdown("**前 15 大持股（真實權重）**")
            _top = pd.DataFrame(_sc.top(15))
            _top["weight"] = _top["weight"].map(lambda x: f"{x:.3%}")
            st.dataframe(_top, width="stretch", hide_index=True)
            # 台積電對照——這是老師 9-8 直接問的那件事，也連到 v10 §8 的 M-17
            # ⚠️ XM 投組可能含 TW 策略解出的持股，故 TW／XM 都要檢查，不是只有 TW
            _tsmc = _sc.weights.get("2330")
            if holdings.config.market in ("TW", "XM"):
                if _tsmc:
                    st.info(f"**台積電（2330）在本組合的權重：{_tsmc:.3%}**"
                           f"（被 {_sc.appear_in.get('2330', 0)}/{_sc.n_strategies} 個策略選中）。"
                           f"對照它在台股指數約佔 **40.23%**（v10 §8 M-17）⇒ "
                           f"本組合對台積電是**極度低配**，不是超配。"
                           f"這正是 M-17「輸給市值加權大盤、贏過等權市場」的直接原因。")
                else:
                    st.info("**台積電（2330）不在本組合持股中。** 對照它在台股指數約佔 "
                           "**40.23%**（v10 §8 M-17）⇒ 這是完全的低配，也是 M-17"
                           "「輸給市值加權大盤、贏過等權市場」的直接原因。")

        st.dataframe(stock_detail, width="stretch", height=280)

        st.markdown("**跨策略重疊度**（同一檔股票被幾個選中策略同時持有——"
                    "呼應老師 9-8 的 fund house 風控意見，見應用層開發追蹤.md §2.4）")
        overlap = (stock_detail.groupby(["stock_symbol", "company_name"])["strategy_uid"]
                  .nunique().reset_index(name="n_strategies")
                  .sort_values("n_strategies", ascending=False))
        st.dataframe(overlap, width="stretch", height=240)

        # 🔴 P3：新面孔 vs 老面孔（老師 9-8 逐字稿原話查證，見 app/new_faces.py docstring）
        # 「現在這些股票最火的，還是說以前都還不錯都蠻賺錢的」——拿策略選定當下（is_end）
        # 跟現在的持股做差集回答這個問題。只在有 is_end 可比、且解析日期不是 is_end 本身時顯示。
        st.markdown("---")
        st.markdown("**新面孔 vs 老面孔**（P3，老師 9-8 逐字稿原話：「現在這些股票最火的，"
                    "還是說以前都還不錯都蠻賺錢的」）")
        _is_end = holdings.window_info.get("is_end")
        if _date_options[_date_label] == _is_end:
            st.caption("目前解析的日期就是策略選定當下（is_end），沒有時間差可比較。"
                      "選別的日期（例如「今天」）才能看出新舊面孔。")
        elif _is_end is None:
            st.caption("這組設定沒有 is_end 可當比較基準。")
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
                f1.metric("老面孔（persistent）", len(_fc.old_faces),
                         f"佔現在持股 {_fc.persistence_rate:.1%}",
                         help="兩個時點都持有——對應老師說的「以前都還不錯都蠻賺錢」")
                f2.metric("新面孔（new）", len(_fc.new_faces),
                         f"佔現在持股 {_fc.turnover_rate:.1%}",
                         help="只在現在持有——對應老師說的「現在這些股票最火的」")
                f3.metric("淡出（exited）", len(_fc.exited),
                         help="只在策略選定當下持有，現在已經不符合條件")
                st.caption("⚠️ 這個比較橫跨財報更新（快時鐘 B1，季度），"
                          "高換手率是正常現象，不代表策略異常。")
                names_map = _get_company_names(holdings.config.market)
                for label, syms in (("老面孔", _fc.old_faces), ("新面孔", _fc.new_faces),
                                    ("淡出", _fc.exited)):
                    with st.expander(f"{label}（{len(syms)} 檔）"):
                        if syms:
                            st.dataframe(pd.DataFrame(
                                {"stock_symbol": syms,
                                 "company_name": [names_map.get(s, "") for s in syms]}),
                                width="stretch", hide_index=True, height=200)
                        else:
                            st.caption("（無）")

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
    if calib.evaluable:
        cc1, cc2 = st.columns(2)
        cc1.metric("OOS CAGR", f"{calib.oos_cagr:.2%}",
                  f"p10 門檻 {calib.thresholds.oos_cagr_p10:.2%}",
                  delta_color="inverse" if calib.below_cagr else "off")
        cc2.metric("OOS Calmar", f"{calib.oos_calmar:.2f}",
                  f"p10 門檻 {calib.thresholds.oos_calmar_p10:.2f}",
                  delta_color="inverse" if calib.below_calmar else "off")
        if calib.flagged:
            st.warning("⚠️ 這次表現明顯偏離歷史常態分布，只是提醒，不會攔下執行")
    else:
        # 🔴 §8-R3：正式模式沒有 OOS，不假裝判定得出來。門檻仍顯示，當作日後基準。
        st.info(f"**狀態：{calib.status}**（門檻已立起來，尚未判定）")
        cc1, cc2 = st.columns(2)
        cc1.metric("OOS CAGR 警戒線（日後基準）", f"{calib.thresholds.oos_cagr_p10:.2%}")
        cc2.metric("OOS Calmar 警戒線（日後基準）", f"{calib.thresholds.oos_calmar_p10:.2f}")
        st.caption(calib.note)

    st.markdown("---")
    st.subheader("執行覆核與稽核紀錄（C4/F2）")

    # 🔴 §8-R4：把「這次實際持有哪些股票」一併寫進稽核紀錄。有解析過就帶上，
    # 沒解析就誠實記成「本次未解析」——不假裝有。
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
                    holdings, _s, as_of="、".join(f"{m}:{d}" for m, d in _dates.items()))
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
    # 🔴 2026-09-11（§9）：解釋 agent（10 欄位，能做獨立分析判斷）只在 live 模式
    # 啟用——R15（群檔案全樣本前視）／R16（基準只涵蓋 62.2% 窗次）／H8（每窗 k
    # 不同，群 id 對不上）三個問題全部只存在於 replay 模式。replay 模式繼續用
    # 下面的六欄位 memo（程式判決、LLM 只負責轉述），不套用新邏輯。
    if holdings.config.mode == "live":
        st.subheader("AI 解讀（解釋 Agent · §9，10 欄位獨立分析）")
        st.caption("跟 replay 模式的六欄位備忘錄不同：這裡的 LLM **可以做事實推論**"
                  "（把提供的數字放在一起講出關聯或型態），但**不能做價值判決**"
                  "（例如建議換方案）——界線見應用層開發追蹤.md §9.7 R17。")

        stock_detail = st.session_state.get("stock_detail")
        if stock_detail is None:
            st.info("請先到「本期持倉」分頁按「解析持股」——解釋 agent 需要股票層級"
                    "持股（老師 9-8 原話①②③都要用到）才能產生。")
        else:
            dry_run = st.checkbox("dry-run（不花錢，不真的呼叫 LLM）", value=True,
                                  key="explain_dry_run")
            purpose = "app_memo"
            if not dry_run:
                st.warning("⚠️ config.ini 還沒設定 `[openai] app_memo_model` 這個 key，"
                          "需要借用既有 purpose 的模型設定跟額度記帳")
                purpose = st.text_input("借用哪個 purpose 的模型設定", value="cluster_story",
                                        key="explain_purpose")

            if st.button("產生解釋"):
                try:
                    from app.explain import generate as generate_explain
                    from app.cluster_kb import build_footprint

                    _market_dates = st.session_state["stock_detail_date"]
                    _sc = assess_stock_level(
                        holdings, stock_detail,
                        as_of="、".join(f"{m}:{d}" for m, d in _market_dates.items()))
                    _footprint = build_footprint(holdings.config.market, holdings.members)
                    _idx = _get_candidate_index()
                    _needed = markets_needed(_idx, holdings.members)
                    _md_map = {m: _get_market_data(m) for m in _needed}
                    with st.spinner("組裝資料並產生解釋中…"):
                        result = generate_explain(
                            holdings, risk, calib, stock_detail=stock_detail,
                            stock_concentration=_sc, market_dates=_market_dates,
                            footprint=_footprint, md_map=_md_map,
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
                    "strategy_footprint_note": "一之1｜策略層：群足跡與身份",
                    "stock_holdings_note": "一之2｜股票層：持股與重疊度",
                    "mechanism_note": "二｜為什麼這樣挑會賺錢",
                    "character_note": "三｜這批的性格：追火 vs 一直賺",
                    "concentration_risk_note": "四之1/2｜集中度風險",
                    "structure_risk_note": "四之3/4/5｜結構風險",
                    "scenario_note": "五｜情境比對",
                    "change_note": "六之a｜跟上次相比",
                    "alternative_note": "六之b｜候選方案對照",
                    "caveat": "七｜限制說明",
                }
                for k, label in labels.items():
                    st.markdown(f"**{label}**")
                    st.write(explain_result["explanation"][k])
                if explain_result["dry_run"]:
                    st.caption("此為 dry-run 內容，未實際呼叫 LLM")
                else:
                    # §9.8 層二：D2 數字洩漏掃描——沒抓到問題不代表這份解釋一定
                    # 沒錯（只查數字，見 memo.py docstring 的已知限制），但這是
                    # 第一道防線，通過是必要條件不是充分條件。
                    if explain_result["leakage_check"]:
                        st.error("D2 洩漏掃描發現問題：\n" +
                                "\n".join(explain_result["leakage_check"]))
                    else:
                        st.success("✅ D2 洩漏掃描通過（解釋裡的數字都能對回判決資料）")
                with st.expander("查看餵給 LLM 的完整客觀資料（JSON）"):
                    # facts 裡有些欄位直接來自 parquet（numpy 純量型別），st.json()
                    # 不吃這些型別——跟 build_prompt() 一樣先用 json 往返轉成純
                    # Python 型別再顯示，不是顯示邏輯本身能處理，是資料型別問題。
                    st.json(json.loads(json.dumps(explain_result["facts"],
                                                  ensure_ascii=False, default=str)))
    else:
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
