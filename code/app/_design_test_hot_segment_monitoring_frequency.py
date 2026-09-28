# -*- coding: utf-8 -*-
"""F後續：使用者指出`_design_test_monitoring_frequency.py`只測了M1-D（市場端，
跟候選池版本無關），沒有測Hot Segment覆蓋率的監控頻率——覆蓋率是「代表策略選股
條件底下選不選得到熱門股」，天生依賴候選池版本，這裡補上anchored/rolling×
保留V1/排除V1四個版本都測。

方法（跟M1-D那支腳本同一套精神：沿用既有凍結門檻，不為monthly另外校準）：
1. 每個版本自己的季頻coverage_count歷史（2015Q1~2023Q4，36季）算出這個版本
   自己的p25/p10（不能沿用§1.1只用anchored+v1算出的33.6%/30.8%套到其他版本，
   每個候選池組成不同，基期不一樣，跟k值/X門檻要逐市場分別校準是同一個道理）
2. 用這個版本自己的門檻，分別在季頻(2024Q1~2025Q4)跟月頻(2024-01~2025-12)
   反查驗證，比較「低於p10」第一次出現的時間點
3. 負對照：window3(2021-2023)季頻已知不觸發（§1.1同期間），月頻會不會多抓出
   假警報——跟M1-D那支腳本的負對照邏輯一致

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._design_test_hot_segment_monitoring_frequency
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402

from research import walkforward_matrix as WF  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _design_test_hot_segment import compute_coverage, hot_segment, monthly_close  # noqa: E402
from _design_test_hotsample_frequency import load_variant_pools  # noqa: E402

TREE_KEY = "TW"
N_MONTHS = 3
X_PCT = 0.10
CALIB_Q = pd.date_range("2015-03-31", "2023-12-31", freq="Q")
CHECK_Q = pd.date_range("2024-03-31", "2025-12-31", freq="Q")
CHECK_M = pd.date_range("2024-01-31", "2025-12-31", freq="M")
CHECK_W = pd.date_range("2024-01-05", "2025-12-31", freq="W-FRI")
NEGCTRL_Q = pd.date_range("2021-03-31", "2023-12-31", freq="Q")   # window3同期間，§1.1已知不觸發
NEGCTRL_M = pd.date_range("2021-01-31", "2023-12-31", freq="M")
NEGCTRL_W = pd.date_range("2021-01-01", "2023-12-31", freq="W-FRI")


def momentum_hot_by_date(monthly: pd.DataFrame, dates) -> dict[pd.Timestamp, list[str]]:
    """給一串季底/月底日期，逐一算trailing N個月動能→hot segment名單。"""
    out = {}
    for d in dates:
        idx = monthly.index[monthly.index <= d]
        if len(idx) <= N_MONTHS:
            continue
        end_px = monthly.loc[idx[-1]]
        start_px = monthly.loc[idx[-1 - N_MONTHS]]
        ret = end_px / start_px - 1.0
        out[d] = hot_segment(ret, X_PCT)
    return out


def coverage_series(md: MarketData, idx: pd.DataFrame, rep_uids: list[str],
                    mktcap: pd.DataFrame, hot_by_date: dict) -> pd.DataFrame:
    rows = []
    for d, hot in hot_by_date.items():
        if not hot:
            continue
        as_of = d.date().isoformat()
        cov_count, cov_mktcap = compute_coverage(md, idx, rep_uids, mktcap, hot, as_of)
        rows.append({"date": as_of, "n_hot": len(hot), "coverage_count": cov_count,
                     "coverage_mktcap": cov_mktcap})
    return pd.DataFrame(rows)


def first_below(series_df: pd.DataFrame, p10: float) -> str | None:
    hit = series_df[series_df["coverage_count"] < p10]
    return hit.iloc[0]["date"] if len(hit) else None


def main():
    print(">> 載入資料...")
    md = MarketData(TREE_KEY, start="2010-01-01")
    idx_full = pd.read_parquet(CANDIDATE_INDEX_PATH)
    idx = idx_full[idx_full["market"] == TREE_KEY].set_index("strategy_uid")
    mktcap = md.get_field("report:mktcap")
    monthly = monthly_close(md)

    pools = load_variant_pools()
    variant_uids = {
        "anchored_v1": pools["anchored_v1"],
        "anchored_exclude_v1": pools["anchored_exclude_v1"],
        "rolling_v1": pools["rolling_v1"][6],          # 沿用window6（跟§1.1同一批對照基準）
        "rolling_exclude_v1": pools["rolling_exclude_v1"][6],
    }

    hot_calib = momentum_hot_by_date(monthly, CALIB_Q)
    hot_check_q = momentum_hot_by_date(monthly, CHECK_Q)
    hot_check_m = momentum_hot_by_date(monthly, CHECK_M)
    hot_check_w = momentum_hot_by_date(monthly, CHECK_W)
    hot_negctrl_q = momentum_hot_by_date(monthly, NEGCTRL_Q)
    hot_negctrl_m = momentum_hot_by_date(monthly, NEGCTRL_M)
    hot_negctrl_w = momentum_hot_by_date(monthly, NEGCTRL_W)

    print("\n=== 各版本自己的季頻p25/p10校準（2015Q1~2023Q4）＋反查驗證（正式期＋負對照期）===")
    summary_rows = []
    for vk, rep_uids in variant_uids.items():
        print(f"\n--- {vk}（{len(rep_uids)}檔代表策略）---")
        calib_df = coverage_series(md, idx, rep_uids, mktcap, hot_calib)
        p25 = calib_df["coverage_count"].quantile(0.25)
        p10 = calib_df["coverage_count"].quantile(0.10)
        print(f"  校準：均值={calib_df['coverage_count'].mean():.1%}｜p25={p25:.1%}｜p10={p10:.1%}")

        check_q_df = coverage_series(md, idx, rep_uids, mktcap, hot_check_q)
        check_m_df = coverage_series(md, idx, rep_uids, mktcap, hot_check_m)
        check_w_df = coverage_series(md, idx, rep_uids, mktcap, hot_check_w)
        q_first = first_below(check_q_df, p10)
        m_first = first_below(check_m_df, p10)
        w_first = first_below(check_w_df, p10)
        print(f"  正式期(2024-2025)：季頻首次低於p10={q_first or '（無）'}｜"
             f"月頻首次低於p10={m_first or '（無）'}｜週頻首次低於p10={w_first or '（無）'}")

        negq_df = coverage_series(md, idx, rep_uids, mktcap, hot_negctrl_q)
        negm_df = coverage_series(md, idx, rep_uids, mktcap, hot_negctrl_m)
        negw_df = coverage_series(md, idx, rep_uids, mktcap, hot_negctrl_w)
        n_negq = int((negq_df["coverage_count"] < p10).sum())
        n_negm = int((negm_df["coverage_count"] < p10).sum())
        n_negw = int((negw_df["coverage_count"] < p10).sum())
        print(f"  負對照期(2021-2023)：季頻低於p10次數={n_negq}/{len(negq_df)}｜"
             f"月頻低於p10次數={n_negm}/{len(negm_df)}｜週頻低於p10次數={n_negw}/{len(negw_df)}")

        summary_rows.append(dict(variant=vk, n_rep=len(rep_uids), p25=p25, p10=p10,
                                 q_first_below_p10=q_first, m_first_below_p10=m_first,
                                 w_first_below_p10=w_first,
                                 negctrl_q_hits=n_negq, negctrl_q_total=len(negq_df),
                                 negctrl_m_hits=n_negm, negctrl_m_total=len(negm_df),
                                 negctrl_w_hits=n_negw, negctrl_w_total=len(negw_df)))

    out = pd.DataFrame(summary_rows)
    out_path = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer" / "hot_segment_monitoring_frequency.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}")


if __name__ == "__main__":
    main()
