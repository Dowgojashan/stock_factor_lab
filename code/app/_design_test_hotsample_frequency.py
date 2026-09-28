# -*- coding: utf-8 -*-
"""老師9/22②：「你可以去實驗一下你在做insample、hotsample，你可以抓說前面還有
哪幾次hotsample事實上贏過insample⋯你是到底是，你真的苦等六個月它會真的做掉，
還是它只是還沒發動而已」——這個具體回測目前完全沒做過，跟已經做完的Hot Segment
覆蓋率快照（「當下這一刻選不選得到」）是不同的東西：這裡要算的是**時間序列上，
hot segment的實際報酬贏過我們投組報酬的頻率＋事後追蹤會不會自己趕上**。

範圍（2026-09-27/28使用者拍板）：
  - 完整2013-2025歷史（6個rolling窗次OOS串接起來，共52個季度，不是只測8季）
  - anchored／rolling × 保留V1／排除V1，四個版本都跑，不是只測一個
  - hot segment報酬用等權＋市值加權兩種算法都做（使用者要求：反正實戰後面
    也沒有都用等權，W2c觸發時就是市值加權，所以兩種都要看）

方法（避免前視偏誤）：
  - hot segment在「這一季開始之前」用trailing N=3個月動能定義（TW已校準值，
    見§1.1），不用這一季自己的報酬去定義熱門股
  - hot segment的「贏過我們」是指：用這批已經定義好的熱門股，在**接下來這一季**
    的實際報酬（等權/市值加權皆算），跟我們投組**同一季**的實際報酬比較
  - 自我修正檢查（往後追蹤2季）：兩種操作型定義都算，回答的問題不一樣
    ①報酬層級：相對落後幅度有沒有隨時間縮小（不管有沒有換股）
    ②持股層級：我們的投組後來有沒有真的換成當時那批熱門股（選股邏輯有沒有跟上）

候選池來源（四個版本，anchored為單一名單套用全部52季；rolling依季度所在的
window套用對應名單，6個窗次OOS剛好無重疊串接成完整2013-2025）：
  - anchored+v1：官方凍結`walkforward_members.parquet`（scheme=E/window4/
    silhouette_is/equal/A_hrp）
  - anchored+exclude_v1：`anchored_w4_silhouette_exclude_v1.parquet`
  - rolling+v1／exclude_v1：window6用既有`rolling_window6_silhouette_members.
    parquet`／`rolling_w6_silhouette_exclude_v1.parquet`；windows1-5用這次新建的
    `rolling_windows1to5_silhouette_picks.parquet`（見`_build_rolling_all_
    windows_silhouette_picks.py`）

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._design_test_hotsample_frequency
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH, resolve_holdings  # noqa: E402

from research import walkforward_matrix as WF  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _design_test_hot_segment import hot_segment, monthly_close  # noqa: E402

TREE_KEY = "TW"
N_MONTHS = 3      # TW已校準值（§1.1）
X_PCT = 0.10      # TW已校準值（§1.1）
FORWARD_TRACK_Q = 2   # 「還沒發動 vs 真的選錯」往後追蹤幾季

APPLAYER = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
FROZEN_MEMBERS_PATH = (Path(__file__).resolve().parent.parent.parent
                       / "_analysis_outputs_robustness" / "walkforward_members.parquet")

# 6個rolling窗次OOS邊界（無重疊、剛好串接成2013-01~2025-12），來自
# `_design_test_rolling_is6oos2.window_dates_rolling()`實際算出的真實日期
# （2026-09-28查證，不是憑印象排的）
ROLLING_OOS_BOUNDS = {
    1: ("2013-01", "2014-12"),
    2: ("2015-01", "2016-12"),
    3: ("2017-01", "2018-12"),
    4: ("2019-01", "2020-12"),
    5: ("2021-01", "2022-12"),
    6: ("2023-01", "2025-12"),
}
QUARTERS = pd.period_range("2013Q1", "2025Q4", freq="Q")

# 🔴🔴 anchored候選池是用IS 2007-2023訓練一次選出的單一名單，2013-2023全部落在
# 它自己的IS窗內——把這份名單套到2013-2022算「贏不贏過hot segment」是拿訓練資料
# 驗證訓練資料（挑選時就看過這段報酬），跟rolling每個窗次只套自己的真實OOS完全
# 不是同一件事。anchored只有2024-2025才是真正沒看過的樣本外資料（跟§3.14/§7.5
# 一路驗證用的同一段），這裡anchored兩個變體的頻率統計/自我修正檢查都只能用這
# 8季，不能延伸到2013-2022——樣本數本來就會比rolling的52季小，這是anchored（單一
# 訓練窗）跟rolling（walk-forward逐窗都是OOS）兩種設計本身的結構性差異，不是
# 這次分析故意做小，要老實報告，不能為了湊大樣本就悄悄把anchored套進自己的IS窗。
ANCHORED_OOS_BOUNDS = ("2024-01", "2025-12")


def window_no_for_quarter(q: pd.Period) -> int:
    q_start_month = q.start_time.strftime("%Y-%m")
    for w, (s, e) in ROLLING_OOS_BOUNDS.items():
        if s <= q_start_month <= e:
            return w
    raise ValueError(f"{q}不在任何rolling窗次OOS範圍內")


def load_variant_pools() -> dict:
    """回傳{"anchored_v1": [uid...], "anchored_exclude_v1": [...],
    "rolling_v1": {window_no: [...]}, "rolling_exclude_v1": {window_no: [...]}}"""
    m = pd.read_parquet(FROZEN_MEMBERS_PATH)
    key = dict(tree_key=TREE_KEY, scheme="E", window_no=4, k_mode="silhouette_is",
              ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    assert len(sub) == 1, "官方anchored baseline名單查詢應該剛好一列"
    anchored_v1 = list(sub.iloc[0]["members"])

    ev1 = pd.read_parquet(APPLAYER / "anchored_w4_silhouette_exclude_v1.parquet")
    ev1 = ev1[(ev1.allocation == "equal") & (ev1.variant == "exclude_v1")]
    assert len(ev1) == 1
    anchored_exclude_v1 = list(ev1.iloc[0]["members"])

    w6_base = pd.read_parquet(APPLAYER / "rolling_window6_silhouette_members.parquet")
    w6_base = w6_base[w6_base.allocation == "equal"]
    assert len(w6_base) == 1
    w6_ev1 = pd.read_parquet(APPLAYER / "rolling_w6_silhouette_exclude_v1.parquet")
    w6_ev1 = w6_ev1[(w6_ev1.allocation == "equal") & (w6_ev1.variant == "exclude_v1")]
    assert len(w6_ev1) == 1

    w1to5 = pd.read_parquet(APPLAYER / "rolling_windows1to5_silhouette_picks.parquet")
    w1to5 = w1to5[w1to5.allocation == "equal"]

    rolling_v1: dict[int, list[str]] = {6: list(w6_base.iloc[0]["members"])}
    rolling_exclude_v1: dict[int, list[str]] = {6: list(w6_ev1.iloc[0]["members"])}
    for w in range(1, 6):
        base_row = w1to5[(w1to5.window_no == w) & (w1to5.variant == "baseline")]
        ev1_row = w1to5[(w1to5.window_no == w) & (w1to5.variant == "exclude_v1")]
        assert len(base_row) == 1 and len(ev1_row) == 1, f"window{w}名單缺失"
        rolling_v1[w] = list(base_row.iloc[0]["members"])
        rolling_exclude_v1[w] = list(ev1_row.iloc[0]["members"])

    return {"anchored_v1": anchored_v1, "anchored_exclude_v1": anchored_exclude_v1,
           "rolling_v1": rolling_v1, "rolling_exclude_v1": rolling_exclude_v1}


def uids_for_quarter(pools: dict, variant_key: str, q: pd.Period) -> list[str]:
    if variant_key.startswith("anchored"):
        return pools[variant_key]
    w = window_no_for_quarter(q)
    return pools[variant_key][w]


def portfolio_quarterly_return(uids: list[str], months_long: pd.DataFrame, q: pd.Period) -> float:
    """該季3個月的策略報酬複利後，等權平均——跟`WF._evaluate()`同一套「先複利
    單一策略的季內報酬，再跨策略等權平均」邏輯，不是先平均月報酬再複利（兩者
    對單一策略等價，但跨策略平均的順序不能顛倒，等權平均本身就該用「這一整季
    真正賺了多少」而非月報酬的算術平均）。"""
    months_in_q = [q.start_time.to_period("M") + i for i in range(3)]
    w = months_long[months_long.strategy_uid.isin(set(uids)) & months_long.month.isin(months_in_q)]
    wide = w.pivot(index="strategy_uid", columns="month", values="ret")
    missing = set(uids) - set(wide.index)
    if missing or wide.isna().any().any():
        raise ValueError(f"{q}：{len(missing)}個策略無資料或有缺值，不可靜默跳過")
    compounded = (1.0 + wide).prod(axis=1) - 1.0
    return float(compounded.mean())


def momentum_asof(monthly: pd.DataFrame, asof_month_end: pd.Timestamp, months: int) -> pd.Series:
    idx = monthly.index[monthly.index <= asof_month_end]
    if len(idx) <= months:
        raise ValueError(f"{asof_month_end}之前資料不足{months}個月，無法算動能")
    end_px = monthly.loc[idx[-1]]
    start_px = monthly.loc[idx[-1 - months]]
    return end_px / start_px - 1.0


def hot_segment_forward_return(monthly: pd.DataFrame, mktcap: pd.DataFrame,
                               hot_syms: list[str], asof_month_end: pd.Timestamp,
                               q_end_month_end: pd.Timestamp) -> tuple[float, float]:
    """回傳(等權遠期報酬, 市值加權遠期報酬)——用`asof_month_end`當weight base，
    這是前視偏誤的界線：只能用「熱門股名單定義時點」已知的市值當權重，不能用
    季底（已經包含這一季報酬影響過的市值）當權重，否則報酬跟權重循環相關。"""
    start_px = monthly.loc[asof_month_end, hot_syms]
    end_px = monthly.loc[q_end_month_end, hot_syms]
    ret = (end_px / start_px - 1.0).dropna()
    if ret.empty:
        return float("nan"), float("nan")
    eq_ret = float(ret.mean())

    valid_mk = mktcap.index[mktcap.index <= asof_month_end]
    if len(valid_mk) == 0:
        return eq_ret, float("nan")
    mk = mktcap.loc[valid_mk.max(), ret.index].fillna(0.0)
    if mk.sum() <= 0:
        return eq_ret, float("nan")
    cap_ret = float((ret * mk).sum() / mk.sum())
    return eq_ret, cap_ret


def resolved_holdings_union(md: MarketData, idx: pd.DataFrame, uids: list[str], as_of: str) -> set[str]:
    out: set[str] = set()
    for uid in uids:
        try:
            syms, _ = resolve_holdings(md, idx.loc[uid], as_of)
        except (RuntimeError, KeyError, ValueError):
            continue
        out |= set(syms)
    return out


def main():
    print(">> 載入資料...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    md = MarketData(TREE_KEY, start="2010-01-01")
    mktcap = md.get_field("report:mktcap")
    monthly = monthly_close(md)

    pools = load_variant_pools()
    print(f"anchored+v1：{len(pools['anchored_v1'])}檔｜anchored+exclude_v1："
         f"{len(pools['anchored_exclude_v1'])}檔")
    for w in range(1, 7):
        print(f"rolling window{w}：+v1 {len(pools['rolling_v1'][w])}檔｜"
             f"+exclude_v1 {len(pools['rolling_exclude_v1'][w])}檔")

    variants = ["anchored_v1", "anchored_exclude_v1", "rolling_v1", "rolling_exclude_v1"]
    rows = []
    for q in QUARTERS:
        q_start_month = q.start_time.strftime("%Y-%m")
        anchored_is_oos = ANCHORED_OOS_BOUNDS[0] <= q_start_month <= ANCHORED_OOS_BOUNDS[1]

        asof_month_end = (q.start_time - pd.offsets.MonthEnd(1))
        q_end_month_end = q.end_time.normalize()   # 🔴 修正：原本用to_period("M").to_timestamp("M")
        # 會回傳月初而非月底（`Period.to_timestamp()`預設how="start"），改用
        # `.normalize()`直接把Period.end_time的時間部分砍掉、留日期（本身已經是
        # 該季最後一個日曆日），跟`monthly_close()`用calendar month-end當index
        # label的慣例對齊
        mom = momentum_asof(monthly, asof_month_end, N_MONTHS)
        hot_syms = hot_segment(mom, X_PCT)
        eq_ret, cap_ret = hot_segment_forward_return(monthly, mktcap, hot_syms,
                                                      asof_month_end, q_end_month_end)

        row = {"quarter": str(q), "n_hot": len(hot_syms), "hot_eq_ret": eq_ret, "hot_cap_ret": cap_ret,
              "window_no": window_no_for_quarter(q), "anchored_is_oos": anchored_is_oos}
        for vk in variants:
            if vk.startswith("anchored") and not anchored_is_oos:
                row[f"{vk}_ret"] = float("nan")   # anchored在自己的IS窗內，不可用來算頻率
                continue
            uids = uids_for_quarter(pools, vk, q)
            row[f"{vk}_ret"] = portfolio_quarterly_return(uids, months_long, q)
        rows.append(row)
        a_str = f"{row['anchored_v1_ret']:+.2%}" if anchored_is_oos else "IS窗內，不計"
        print(f"  {q}：hot={len(hot_syms)}檔｜熱門等權={eq_ret:+.2%}｜熱門市值加權={cap_ret:+.2%}｜"
             f"anchored_v1={a_str}｜rolling_v1={row['rolling_v1_ret']:+.2%}")

    out = pd.DataFrame(rows)
    # 🔴 anchored在自己IS窗內的季度`{vk}_ret`是NaN——比較式`hot_ret > NaN`在pandas
    # 裡會算出False，若照原樣存進beat欄位，會被誤讀成「這幾季hot segment沒贏」，
    # 但事實是這幾季根本不該拿來比。用`.where(valid, other=pd.NA)`確保無效季度
    # 的beat欄位也是缺值，不是False，下游一律用`.dropna()`後的子集計算頻率。
    for vk in variants:
        valid = out[f"{vk}_ret"].notna()
        out[f"{vk}_beat_eq"] = (out["hot_eq_ret"] > out[f"{vk}_ret"]).where(valid)
        out[f"{vk}_beat_cap"] = (out["hot_cap_ret"] > out[f"{vk}_ret"]).where(valid)

    out_path = APPLAYER / "hotsample_frequency_2013_2025.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}（{len(out)}季）")

    print("\n=== 頻率統計：hot segment贏過我們投組報酬的季數（等權／市值加權）===")
    print("    🔴 anchored只計自己真正的OOS（2024Q1~2025Q4，8季）——candidate list是用"
         "IS 2007-2023訓練選出的，套進2013-2023算「贏不贏」等於拿訓練資料驗證訓練資料，"
         "跟rolling每個窗次都是真正OOS不是同一件事，樣本數本來就會比rolling小，這是"
         "結構性差異、不是這次分析縮水")
    for vk in variants:
        n_valid = int(out[f"{vk}_ret"].notna().sum())
        n_eq = int(out[f"{vk}_beat_eq"].sum())
        n_cap = int(out[f"{vk}_beat_cap"].sum())
        print(f"  {vk}（n={n_valid}）：等權定義下{n_eq}次（{n_eq/n_valid:.1%}）｜"
             f"市值加權定義下{n_cap}次（{n_cap/n_valid:.1%}）")

    print(f"\n=== 自我修正檢查（用等權定義的hot-beat季，往後追蹤{FORWARD_TRACK_Q}季）===")
    q_list = list(QUARTERS)
    for vk in variants:
        print(f"\n--- {vk} ---")
        beat_idx = out.index[out[f"{vk}_beat_eq"] == True].tolist()  # noqa: E712  # NaN在==True下正確排除
        n_tracked, n_return_narrowed, n_holdings_improved = 0, 0, 0
        detail = []
        for i in beat_idx:
            j = i + FORWARD_TRACK_Q
            if j >= len(out):
                continue
            n_tracked += 1
            q_t, q_t2 = q_list[i], q_list[j]
            excess_t = out.loc[i, f"{vk}_ret"] - out.loc[i, "hot_eq_ret"]
            excess_t2 = out.loc[j, f"{vk}_ret"] - out.loc[j, "hot_eq_ret"]
            return_narrowed = excess_t2 > excess_t
            n_return_narrowed += int(return_narrowed)

            hot_syms_t = hot_segment(momentum_asof(monthly, (q_t.start_time - pd.offsets.MonthEnd(1)),
                                                   N_MONTHS), X_PCT)
            uids_t = uids_for_quarter(pools, vk, q_t)
            uids_t2 = uids_for_quarter(pools, vk, q_t2)
            hold_t = resolved_holdings_union(md, idx, uids_t, q_t.end_time.date().isoformat())
            hold_t2 = resolved_holdings_union(md, idx, uids_t2, q_t2.end_time.date().isoformat())
            cov_t = len(hold_t & set(hot_syms_t)) / len(hot_syms_t) if hot_syms_t else float("nan")
            cov_t2 = len(hold_t2 & set(hot_syms_t)) / len(hot_syms_t) if hot_syms_t else float("nan")
            holdings_improved = cov_t2 > cov_t
            n_holdings_improved += int(holdings_improved)
            detail.append(dict(q_t=str(q_t), q_t2=str(q_t2), excess_t=excess_t, excess_t2=excess_t2,
                               return_narrowed=return_narrowed, cov_t=cov_t, cov_t2=cov_t2,
                               holdings_improved=holdings_improved))
        print(f"  hot-beat季數={len(beat_idx)}｜可追蹤（未超出資料範圍）={n_tracked}")
        if n_tracked:
            print(f"  ①報酬層級：{n_return_narrowed}/{n_tracked}（{n_return_narrowed/n_tracked:.1%}）"
                 f"相對落後幅度在{FORWARD_TRACK_Q}季後縮小（還沒發動的訊號）")
            print(f"  ②持股層級：{n_holdings_improved}/{n_tracked}（{n_holdings_improved/n_tracked:.1%}）"
                 f"投組後來對當時熱門股的覆蓋率提升（選股邏輯真的跟上了）")
        detail_df = pd.DataFrame(detail)
        if not detail_df.empty:
            detail_df.to_csv(APPLAYER / f"hotsample_self_correction_{vk}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
