# -*- coding: utf-8 -*-
"""D③ 補算anchored側（scheme A，IS=6年/OOS=2年，跟新rolling test window-for-window
完全對齊）的候選池大小型股組成，才能真的回答「rolling有沒有改變組成」——
`_design_test_rolling_is6oos2.py`原本只算了rolling側，沒有anchored對照組，
這支腳本補上，不重建樹（用`walkforward_members.parquet`已經存好的members，
只做股票層解析＋市值百分位查詢，便宜）。

用法（cwd 必須是 code/；⚠️ Windows預設cp950主控台，之後若加中文符號log記得比照
`_design_test_rolling_is6oos2.py`加PYTHONIOENCODING=utf-8，先在這裡順手提醒，
目前本檔log沒有emoji還不會炸，但下次改動時容易忘記補）：
    PYTHONIOENCODING=utf-8 python -m app._compare_rolling_vs_anchored_composition
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH, resolve_holdings  # noqa: E402

from research import paths  # noqa: E402

TREE_KEY = "TW"


def composition_for_members(md: MarketData, idx: pd.DataFrame, members: list[str],
                            as_of: str, mktcap: pd.DataFrame) -> dict:
    all_syms: set[str] = set()
    for uid in members:
        try:
            syms, _ = resolve_holdings(md, idx.loc[uid], as_of)
        except (RuntimeError, KeyError, ValueError):
            continue
        all_syms |= set(syms)

    valid_mk_dates = mktcap.index[mktcap.index <= pd.Timestamp(as_of)]
    if len(valid_mk_dates) == 0:
        return {"n_stocks_held": len(all_syms), "n_stocks_with_mktcap": 0,
               "median_mktcap_pct": float("nan"), "frac_top10pct_mktcap": float("nan")}
    mk_row = mktcap.loc[valid_mk_dates.max()]
    mk_universe = mk_row.reindex(md.common).dropna()
    pct_rank = mk_universe.rank(pct=True)
    held_pct = pct_rank.reindex(list(all_syms)).dropna()
    return {"n_stocks_held": len(all_syms), "n_stocks_with_mktcap": len(held_pct),
           "median_mktcap_pct": float(held_pct.median()) if len(held_pct) else float("nan"),
           "frac_top10pct_mktcap": float((held_pct >= 0.90).mean()) if len(held_pct) else float("nan")}


def main():
    print(">> 載入 members（scheme A，IS=6年/OOS=2年，anchored，跟新rolling test window對齊）...")
    m = pd.read_parquet(paths.ROOT / "_analysis_outputs_robustness" / "walkforward_members.parquet")
    sub = m[(m.tree_key == TREE_KEY) & (m.scheme == "A") & (m.k_mode == "fixed")
           & (m.ratio == "legacy") & (m.allocation == "equal") & (m.group == "A_hrp")]
    sub = sub.sort_values("window_no")
    print(f"scheme A 共 {len(sub)} 窗")

    print(">> 載入 TW MarketData...")
    md = MarketData(TREE_KEY, start="2000-01-01")
    mktcap = md.get_field("report:mktcap")
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    rows = []
    for _, r in sub.iterrows():
        comp = composition_for_members(md, idx, list(r.members), r.is_end, mktcap)
        rows.append({"window_no": int(r.window_no), "is_start": r.is_start, "is_end": r.is_end,
                    "oos_start": r.oos_start, "oos_end": r.oos_end, **comp})
        print(f"  window{int(r.window_no)}（IS {r.is_start}~{r.is_end}）：市值前10%佔比="
             f"{comp['frac_top10pct_mktcap']:.1%}（{comp['n_stocks_with_mktcap']}檔有市值資料"
             f"／{comp['n_stocks_held']}檔持股）")

    out = pd.DataFrame(rows)
    out_path = (Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
               / "anchored_schemeA_composition_TW.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out_path}")

    print("\n=== 直接比較：anchored(scheme A) vs rolling(6:2)，window-for-window ===")
    roll = pd.read_csv(Path(__file__).resolve().parent.parent.parent
                       / "_analysis_outputs_applayer" / "rolling_is72oos24_TW.csv")
    cmp = out.merge(roll[["window_no", "frac_top10pct_mktcap", "a_oos_cagr"]],
                    on="window_no", suffixes=("_anchored", "_rolling"))
    anc_perf = pd.read_csv(paths.ROOT / "_analysis_outputs_robustness" / "walkforward_matrix_detail.csv")
    anc_perf = anc_perf[(anc_perf.tree_key == TREE_KEY) & (anc_perf.scheme == "A")
                        & (anc_perf.k_mode == "fixed") & (anc_perf.ratio == "legacy")
                        & (anc_perf.allocation == "equal") & (anc_perf.group == "A_hrp")]
    cmp = cmp.merge(anc_perf[["window_no", "oos_cagr"]].rename(columns={"oos_cagr": "anchored_oos_cagr"}),
                    on="window_no")
    cmp = cmp.rename(columns={"a_oos_cagr": "rolling_oos_cagr"})
    print(cmp[["window_no", "is_end", "frac_top10pct_mktcap_anchored", "frac_top10pct_mktcap_rolling",
              "anchored_oos_cagr", "rolling_oos_cagr"]].to_string(index=False))


if __name__ == "__main__":
    main()
