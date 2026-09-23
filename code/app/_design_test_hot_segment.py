# -*- coding: utf-8 -*-
"""D④第一步：Hot Segment（Momentum）覆蓋率計算——先算真實歷史分布，才能校準
§7.4b 規格裡留白的 N（回顧月數）、X（熱門門檻）、p25/p10 三個數字。

設計依據：`文件/實戰監控Agent系統_設計文件.md` §7.4b（2026-09-22 新增）。
不是憑感覺，是真的照規格公式重算：
  - Hot Segment 定義：近 N 個月累積報酬排名前 X%（N∈{1,3,6}、X∈{10%,20%}都測）
  - Coverage 定義：沿用`_design_test_guaranteed_megacap_slot.py::selects_symbol()`
    同一套「選股條件底下選不選得到」邏輯，算 count 版跟 market-cap 加權版兩個口徑

範圍限定（先做小規模驗證，不是正式產出）：只測 TW 市場、2015-2023（跟`_design_
test_guaranteed_megacap_slot.py`同一批 window4 的 30 個代表策略），共 36 個季底時點
——用意是先看 N/X 的選擇對 coverage 有沒有鑑別力，不是這次就要凍結最終門檻數字
（凍結門檻需要更長歷史，比照 M1-D 用 2007-2023，那是下一步）。

用法（cwd 必須是 code/）：
    python -m app._design_test_hot_segment [--market TW|US] [--full-grid]

    預設只測 TW、N∈{1,3,6}×X∈{10%,20%} 全組合（第一輪已跑過，見開發追蹤§1.1）。
    --market US：US 市場穩健性檢查，預設只測 N∈{3,6}×X∈{10%}（TW 已確認 N=1 是雜訊、
    X=10%略優於20%，US這輪只驗證「同一個選擇在另一個市場站不站得住」，不重測全網格，
    省算力；要重測全網格可加 --full-grid）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _design_test_guaranteed_megacap_slot import selects_symbol  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402

N_CANDIDATES = (1, 3, 6)          # 回顧月數候選
X_CANDIDATES = (0.10, 0.20)       # Hot 門檻候選（報酬排名前 X%）
CALIB_ENDS = pd.date_range("2015-03-31", "2023-12-31", freq="Q")   # 2015-2023：訂歷史分布用
CHECK_ENDS = pd.date_range("2024-03-31", "2025-12-31", freq="Q")    # 2024-2025：反查驗證用（比照M1-D）
QUARTER_ENDS = CALIB_ENDS.union(CHECK_ENDS)


def monthly_close(md: MarketData) -> pd.DataFrame:
    """月底收盤價（price:close 月頻重取樣）——**呼叫端算一次、跨所有 N 值共用**，
    不要在 `_quarterly_returns()` 內部各自重算：這個月頻序列本身跟 N 無關，
    只有下面取「往前第幾個月」的 offset 不同，對全市場多年日頻資料重複 resample
    是白白浪費（2026-09-22 code review 抓到，`--full-grid` 下原本會重算 3 次）。
    """
    close = md.get_field("price:close")
    return close.resample("M").last()


def _quarterly_returns(monthly: pd.DataFrame, months: int) -> pd.DataFrame:
    """算「每個季底 as-of，往前 months 個月的累積報酬」，逐股票逐季底一次算完。

    用 price:close 直接算，不透過 factor pipeline（動能不是候選池因子，母體宇宙
    見 §7.4b①：跟 HRP 樹同一批候選股票，這裡用 md.common 當母體，等同 candidate_index
    出現過的股票宇宙，理論上略寬鬆但不影響「是否有鑑別力」這個第一階段驗證目的）。

    `monthly`：`monthly_close()` 算好的月頻收盤價，呼叫端傳入、跨 N 值共用一份。
    """
    out = {}
    for q in QUARTER_ENDS:
        idx = monthly.index[monthly.index <= q]
        if len(idx) <= months:
            continue
        end_px = monthly.loc[idx[-1]]
        start_px = monthly.loc[idx[-1 - months]]
        ret = (end_px / start_px - 1.0)
        out[q] = ret
    return pd.DataFrame(out).T  # index=季底, columns=股票


def hot_segment(ret_row: pd.Series, x_pct: float) -> list[str]:
    """該季底的動能報酬序列 → 排名前 x_pct 的股票清單（NaN 一律排除）。"""
    valid = ret_row.dropna()
    if valid.empty:
        return []
    cutoff = valid.quantile(1 - x_pct)
    return valid[valid >= cutoff].index.tolist()


def compute_coverage(md: MarketData, idx: pd.DataFrame, rep_uids: list[str],
                     mktcap: pd.DataFrame, hot: list[str], as_of: str) -> tuple[float, float]:
    """給一批 Hot Segment 股票 + 代表策略清單，算 coverage_count／coverage_mktcap。

    抽成共用函式（2026-09-22 code review 抓到 `_design_test_hot_segment_xm.py`
    原本整段複製貼上這段邏輯——之後改覆蓋率算法只會改到一邊、另一邊悄悄變舊，
    見開發追蹤§1.5後的修正紀錄）。`_design_test_hot_segment.py::main()` 跟
    `_design_test_hot_segment_xm.py::_one_side()` 都呼叫這支。
    """
    covered = set()
    for uid in rep_uids:
        row = idx.loc[uid]
        for sym in hot:
            if sym in covered:
                continue
            if selects_symbol(md, row, as_of, sym) is True:
                covered.add(sym)
    cov_count = len(covered) / len(hot)

    q = pd.Timestamp(as_of)
    valid_mk_dates = mktcap.index[mktcap.index <= q]
    if len(valid_mk_dates) == 0:
        return cov_count, float("nan")
    mk = mktcap.loc[valid_mk_dates.max()]
    hot_mk = mk.reindex(hot)
    if hot_mk.isna().all():
        # 🔴 全部熱門股都缺市值資料，不是「覆蓋率0」——是這一項指標本季根本
        # 算不出來，必須回傳 NaN 讓呼叫端誠實標記「無法計算」，不可以讓它被
        # 下游 `v < p10` 這種比較式悄悄吃掉、偽裝成「正常、未觸發」
        # （2026-09-22 code review 抓到的靜默降級問題）。
        return cov_count, float("nan")
    hot_mk = hot_mk.fillna(0.0)
    covered_mk = mk.reindex(list(covered)).fillna(0.0)
    cov_mktcap = (covered_mk.sum() / hot_mk.sum()) if hot_mk.sum() > 0 else float("nan")
    return cov_count, cov_mktcap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["TW", "US"], default="TW")
    ap.add_argument("--full-grid", action="store_true",
                    help="US 預設只測 N∈{3,6}×X∈{10%}（穩健性檢查用）；加這個旗標測完整 N∈{1,3,6}×X∈{10%,20%}")
    args = ap.parse_args()

    n_candidates = N_CANDIDATES if (args.market == "TW" or args.full_grid) else (3, 6)
    x_candidates = X_CANDIDATES if (args.market == "TW" or args.full_grid) else (0.10,)

    print(f">> 載入 {args.market} MarketData（2010起，涵蓋N=6個月的最早回顧期）...")
    md = MarketData(args.market, start="2010-01-01")

    idx_full = pd.read_parquet(CANDIDATE_INDEX_PATH)
    idx = idx_full[idx_full["market"] == args.market].set_index("strategy_uid")

    members_path = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_robustness" / "walkforward_members.parquet"
    m = pd.read_parquet(members_path)
    key = dict(tree_key=args.market, scheme="E", window_no=4, k_mode="silhouette_is",
              ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    rep_uids = list(sub.iloc[0]["members"])
    print(f"代表策略數：{len(rep_uids)}（跟保底名額測試同一批 window4 legacy/equal/A_hrp）")

    mktcap = md.get_field("report:mktcap")

    print(">> 算月頻收盤價（跨所有N值共用一份，不重算）...")
    monthly = monthly_close(md)
    print(">> 逐N值算動能報酬...")
    ret_by_n = {n: _quarterly_returns(monthly, n) for n in n_candidates}

    rows = []
    for n in n_candidates:
        ret_df = ret_by_n[n]
        for x in x_candidates:
            for q in ret_df.index:
                hot = hot_segment(ret_df.loc[q], x)
                if not hot:
                    continue
                as_of = q.date().isoformat()
                cov_count, cov_mktcap = compute_coverage(md, idx, rep_uids, mktcap, hot, as_of)
                period = "calib_2015_2023" if q in CALIB_ENDS else "check_2024_2025"
                rows.append(dict(N=n, X=x, quarter=as_of, period=period, n_hot=len(hot),
                                 coverage_count=cov_count, coverage_mktcap=cov_mktcap))
                mk_str = f"{cov_mktcap:.1%}" if pd.notna(cov_mktcap) else "N/A（缺市值資料）"
                print(f"  N={n} X={x:.0%} {as_of}（{period}）：hot={len(hot)}檔｜"
                     f"coverage_count={cov_count:.1%}｜coverage_mktcap={mk_str}")

    out = pd.DataFrame(rows)
    out_path = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
               / f"hot_segment_coverage_test_{args.market}.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}（{len(out)} 列）")

    print("\n=== N/X 組合的coverage分布摘要（判斷鑑別力，僅用calib期）===")
    calib = out[out["period"] == "calib_2015_2023"]
    summary = calib.groupby(["N", "X"])[["coverage_count", "coverage_mktcap"]].describe()
    print(summary)

    print("\n=== 逐季自相關（lag=1）：驗證N=3/N=6在這個市場是不是也是真正的狀態變數 ===")
    for n in n_candidates:
        for x in x_candidates:
            sub = out[(out["N"] == n) & (out["X"] == x)].sort_values("quarter")
            for col in ["coverage_count", "coverage_mktcap"]:
                ac = sub[col].autocorr(lag=1)
                print(f"  N={n} X={x:.0%} {col}: autocorr(lag1)={ac:.3f}")

    print("\n=== 反查驗證：2024-2025每季coverage，落在2015-2023歷史分布第幾百分位（比照M1-D做法）===")
    for n in n_candidates:
        for x in x_candidates:
            sub_calib = calib[(calib["N"] == n) & (calib["X"] == x)]
            sub_check = out[(out["N"] == n) & (out["X"] == x) & (out["period"] == "check_2024_2025")]
            if sub_calib.empty or sub_check.empty:
                continue
            for col in ["coverage_count", "coverage_mktcap"]:
                p25 = sub_calib[col].quantile(0.25)
                p10 = sub_calib[col].quantile(0.10)
                print(f"  N={n} X={x:.0%} {col}：歷史p25={p25:.1%}／p10={p10:.1%}")
                for _, r in sub_check.iterrows():
                    v = r[col]
                    if pd.isna(v):
                        # 🔴 NaN是「這季算不出來」（例如缺市值資料），不是「正常、未觸發」——
                        # 絕對不可以讓它落進 v<p10/v<p25 比較式（兩者對NaN都會回False，
                        # 會被誤判成「無警示」），2026-09-22 code review 抓到的靜默降級問題
                        print(f"      {r['quarter']}：無法計算（缺資料，非「正常」）")
                        continue
                    pct = (sub_calib[col] < v).mean()
                    flag = "🔴低於p10" if v < p10 else ("🔶低於p25" if v < p25 else "")
                    print(f"      {r['quarter']}：{v:.1%}（歷史分布第{pct:.0%}百分位）{flag}")


if __name__ == "__main__":
    main()
