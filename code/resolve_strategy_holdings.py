# -*- coding: utf-8 -*-
"""獨立分析腳本：把「策略 uid」解析成「某個時間點實際持有哪些股票」。

背景（2026-09-10 使用者需求）：應用層 UI／CLI 目前只做到策略層級（`strategy_uid`
清單），`app/engine.py` docstring 明講「這裡先誠實標註這個範圍，不假裝已經做到股票
層級」——這支腳本就是把那個缺口補上，但刻意做成**獨立腳本**，不接進 app/ 五層架構
（範圍：只給人工分析用，不是正式功能，見與使用者的討論）。

原理：不重新回測、不需要 `results_artifacts/`（openSec 那幾個 job 目錄在這台機器
上本來就不在）。改成直接用 `candidate_index.parquet` 存的策略定義
（F1_factor/F1_band/F1_nbands、F2_*、C_source/C_rule、V）重建 `condition_factory`
條件，丟進 `fcv_core.MarketData.get_mask()`（跟原始 Phase 1-4 SOP 同一套機制）
現算出當時的選股布林遮罩，在指定日期切一刀取出通過的股票代號。

C_rule → condition_factory 型別的對應表是從 `candidate_index.parquet` 實際出現的
7 種值（riseq1/riseq2/qmax4/qmax8/yoy/ytdavg_gt_lyavg/ytdavg_gt_lyytdavg）反推
`condition_factory.auto_name()` 的命名規則得到，跟
`文件/因子候選批次_F與C因子定義.md` §3.1 的 7 型別定義一致（那份文件描述的是另一個
先跑的「候選批次」實驗，不是產生 candidate_index 的正式 SOP，但兩者共用同一套
`condition_factory.py`，型別定義相同）。

用法（cwd 必須是 code/，資料庫要能連得上）：
    # 對到某個 replay 窗次的 IS/OOS 邊界日期（推薦用法，跟 app/cli.py 的設定檔共用）
    python resolve_strategy_holdings.py --config my_run.json --date oos_start

    # 手動指定策略與日期
    python resolve_strategy_holdings.py --market TW --date 2023-01-01 \\
        --uids "TW::MOM_qb2of3__ACCRUAL_qb0of3__C4_ROE_DYN_qmax8__v0"

輸出：
    <out>_holdings.csv    每個策略在該日期的持股清單（一列一檔股票）
    <out>_summary.csv     跨策略彙總：哪些股票被幾個策略同時選中（分析重疊度用）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fcv_core  # noqa: E402  ⚠️ 必須先 import，才會把 ROOT 加進 sys.path（見 CLAUDE.md 路徑陷阱）
from condition_factory import build_conditions  # noqa: E402
from database import Database  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from research import paths as research_paths  # noqa: E402

CANDIDATE_INDEX_PATH = research_paths.STAGE0 / "candidate_index.parquet"

# C_rule（candidate_index 欄位值，= condition_factory.auto_name 的輸出後綴）
# → (condition_factory 型別, args)。
# 反推依據：condition_factory.py:302 auto_name() 的型別-後綴對應規則；
# 7 個值涵蓋 candidate_index.parquet 實際出現的全部 C_rule（見開發時的查證）。
C_RULE_TO_TYPE: dict[str, tuple[str, list]] = {
    "riseq1": ("rise_q", [1]),
    "riseq2": ("rise_q", [2]),
    "qmax4": ("is_highest_q", [4]),
    "qmax8": ("is_highest_q", [8]),
    "yoy": ("yoy_gt", [4]),
    "ytdavg_gt_lyavg": ("ytd_avg_gt_prev_year_avg", []),
    "ytdavg_gt_lyytdavg": ("ytd_avg_gt_prev_year_same_period_avg", []),
}


def _field(factor_name: str) -> str:
    """因子名 → DB 欄位名。通用規則，跟 phase1_linearity.py:108／
    run_factor_batches.py:57 同一條規則，`get_data.Data.get()` 是通用查詢，
    不需要為每個因子另外維護對照表。"""
    return f"report:{factor_name.lower()}"


def _q_band_condition(factor: str, band: int, nbands: int) -> dict:
    """建一個 F1/F2 用的橫斷面分位條件（build_conditions 的 grouped_defs 格式）。"""
    grouped = {factor: {"field": _field(factor),
                        "conditions": [{"type": "q_band", "args": [int(band), int(nbands)]}]}}
    return build_conditions(grouped)[0]


def _c_condition(c_source: str, c_rule: str) -> dict:
    """建 C 動態條件。"""
    if c_rule not in C_RULE_TO_TYPE:
        raise ValueError(f"未知的 C_rule={c_rule!r}，C_RULE_TO_TYPE 需要補這個對應"
                         f"（現有：{sorted(C_RULE_TO_TYPE)}）")
    ctype, args = C_RULE_TO_TYPE[c_rule]
    grouped = {c_source: {"field": _field(c_source),
                          "conditions": [{"type": ctype, "args": args}]}}
    return build_conditions(grouped)[0]


def resolve_holdings(md: MarketData, row: pd.Series, as_of: str) -> list[str]:
    """給一列 candidate_index（單一策略的定義）+ 日期，回傳當天通過全部條件的股票代號。

    `as_of` 不是交易日時，取小於等於它的最後一個交易日（as-of 語意，
    避免剛好挑到假日對不到資料）。
    """
    masks = [md.get_mask(_q_band_condition(row["F1_factor"], row["F1_band"], row["F1_nbands"]))]
    if not row["F2_empty"]:
        masks.append(md.get_mask(_q_band_condition(row["F2_factor"], row["F2_band"], row["F2_nbands"])))
    if pd.notna(row["C_rule"]):
        masks.append(md.get_mask(_c_condition(row["C_source"], row["C_rule"])))

    missing = [i for i, m in enumerate(masks) if m is None]
    if missing:
        raise RuntimeError(f"{row['strategy_uid']}：有條件算不出遮罩（欄位可能在這個市場不存在），"
                           f"不生成部分股票清單以免誤導——第 {missing} 個條件是 None")

    combined = masks[0]
    for m in masks[1:]:
        combined = combined & m
    if row["V"] == "v1":
        combined = combined & md.get_v_mask()

    ts = pd.Timestamp(as_of)
    valid_dates = combined.index[combined.index <= ts]
    if len(valid_dates) == 0:
        raise ValueError(f"{as_of} 之前沒有任何交易日資料，無法解析持股")
    use_date = valid_dates.max()
    day = combined.loc[use_date]
    return sorted(day[day].index.tolist()), use_date


def _load_company_names(market: str) -> pd.Series:
    """公司代號 → 名稱對照（market 篩選跟 database.py 其餘地方同一條 `_exchange_in_clause()`，
    避免重演 CLAUDE.md 記載過的雷：`stock`/`company` 沒篩市場會兩邊資料撈在一起）。"""
    db = Database(market)
    conn = db.create_connection()
    q = f"SELECT company_symbol, name FROM company WHERE {db._exchange_in_clause()}"
    df = pd.read_sql(q, conn).drop_duplicates("company_symbol")
    return df.set_index("company_symbol")["name"]


def _resolve_from_config(config_path: str, date_key: str) -> tuple[str, list[str], str]:
    """從 app/cli.py 產生的 RunConfig JSON 讀出 market／策略清單／指定的邊界日期。"""
    sys.path.insert(0, str(HERE))
    from app.config import RunConfig  # noqa: E402
    from app.engine import _load_replay_row  # noqa: E402

    cfg = RunConfig.load(config_path)
    if cfg.mode != "replay":
        raise ValueError("這支腳本目前只支援 mode='replay' 的設定檔（live 模式尚未實作，見 A2）")
    row, _perf = _load_replay_row(cfg)
    if date_key not in ("is_start", "is_end", "oos_start", "oos_end"):
        raise ValueError("--date 搭配 --config 時，只能用 is_start/is_end/oos_start/oos_end 之一")
    as_of = str(row[date_key])
    return cfg.market, list(row["members"]), as_of


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="app/cli.py 產生的 RunConfig JSON（跟 --date is_start/is_end/... 搭配）")
    ap.add_argument("--market", choices=["TW", "US"], help="手動模式：市場")
    ap.add_argument("--uids", help="手動模式：逗號分隔的 strategy_uid 清單")
    ap.add_argument("--date", required=True,
                   help="有 --config 時填 is_start/is_end/oos_start/oos_end 之一；"
                        "手動模式填實際日期 YYYY-MM-DD")
    ap.add_argument("--out", default="holdings", help="輸出檔名前綴，預設 holdings")
    args = ap.parse_args()

    if args.config:
        market, uids, as_of = _resolve_from_config(args.config, args.date)
    elif args.market and args.uids:
        market, uids, as_of = args.market, [u.strip() for u in args.uids.split(",")], args.date
    else:
        ap.error("要嘛給 --config + --date（窗次邊界），要嘛給 --market + --uids + --date（手動指定日期）")
        return 2

    print(f"市場={market}｜策略數={len(uids)}｜日期={as_of}")

    idx = pd.read_parquet(CANDIDATE_INDEX_PATH)
    idx = idx.set_index("strategy_uid")
    missing_uids = [u for u in uids if u not in idx.index]
    if missing_uids:
        raise ValueError(f"candidate_index 裡找不到 {len(missing_uids)} 個策略，例如 {missing_uids[:3]}")

    print(">> 載入 MarketData（第一次會連資料庫，需要一點時間）...")
    md = MarketData(market)

    rows = []
    for uid in uids:
        row = idx.loc[uid]
        holdings, use_date = resolve_holdings(md, row, as_of)
        print(f"  {uid}：{len(holdings)} 檔（實際對到交易日 {use_date.date()}）")
        for sym in holdings:
            rows.append({"strategy_uid": uid, "as_of_requested": as_of,
                        "as_of_actual": use_date.date().isoformat(), "stock_symbol": sym})

    print(">> 查詢公司名稱對照...")
    names = _load_company_names(market)

    detail = pd.DataFrame(rows)
    detail["company_name"] = detail["stock_symbol"].map(names)
    detail_path = f"{args.out}_holdings.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n已寫入 {detail_path}（{len(detail)} 列）")

    summary = (detail.groupby("stock_symbol")["strategy_uid"]
              .agg(n_strategies="nunique", strategies=lambda s: "; ".join(sorted(set(s))))
              .sort_values("n_strategies", ascending=False))
    summary.insert(0, "company_name", summary.index.map(names))
    summary_path = f"{args.out}_summary.csv"
    summary.to_csv(summary_path, encoding="utf-8-sig")
    print(f"已寫入 {summary_path}（{len(summary)} 檔不重複股票，"
         f"最高重疊 {summary['n_strategies'].max() if len(summary) else 0} 個策略同時選中）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
