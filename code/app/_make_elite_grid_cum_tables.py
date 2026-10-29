# -*- coding: utf-8 -*-
"""優等生權重網格：8季累積報酬對照表（2026-10-28，使用者要「每種規則 8 季累積多少」的直接對照）。

window4（2024Q1~2025Q4，8季）累積報酬＝季報酬複利（與 control0 +14.44% 同一算法）。
未扣成本為主（與正式模擬／10/27報告同口徑），另出扣成本版。

  表A：13種排名依據 × 5種資金比例（固定 前3名・平均分）
  表B：13種排名依據 × 16種「取幾名×分法」（固定 資金30%）
  附錄：1,040組全列（含 2015-23 累積、Sharpe、日頻MDD）
每張表都附 control0／每檔等權／市值傾斜全開／大盤 參考列。
輸出：`_analysis_outputs_applayer/elite_grid_1027/tables_cum/`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from app._elite_grid_offline import OUT_DIR  # noqa: E402
from app._make_elite_grid_figures import (GROUP_LABEL, GROUPS, SCHEME_LABEL, SCORE_LABEL,  # noqa: E402
                                          to_md)

TAB_DIR = OUT_DIR / "tables_cum"
#: --liq：美股讀 `_liq` 結果，輸出到 tables_cum_liq（不覆蓋 tables_cum）
SUFFIX = {"TW": "", "US": ""}
SCORE_ORDER = ["S1_votes", "S2_f1", "S2_f1f2",
               "S3_greedy_L1", "S3_greedy_L3", "S3_greedy_L6",
               "S4_reversal_L1", "S4_reversal_L3", "S4_reversal_L6",
               "S5_mixed_L1", "S5_mixed_L3", "S5_mixed_L6", "S6_mcap"]
SLEEVES = [0.1, 0.2, 0.3, 0.5, 1.0]
NSCH = [(1, "equal")] + [(n, s) for n in (3, 5, 10) for s in ("equal", "top1x2", "top2x2", "proportional", "tier")]


def cum_table(market, pool):
    tag = f"{market}_{pool}{SUFFIX[market]}"
    q = pd.read_parquet(OUT_DIR / f"quarterly_{tag}.parquet")
    q4 = q[q.window == 4].copy()
    q4["ret_net"] = (1 - q4.cost) * (1 + q4.ret_gross) - 1
    cum_g = q4.groupby("combo").ret_gross.apply(lambda r: float(np.prod(1 + r) - 1))
    cum_n = q4.groupby("combo").ret_net.apply(lambda r: float(np.prod(1 + r) - 1))
    combos = pd.read_csv(OUT_DIR / f"combos_{tag}.csv").set_index("combo")
    meta = json.loads((OUT_DIR / f"meta_{tag}.json").read_text(encoding="utf-8"))
    bench = meta["windows"]["4"]["bench_cum"]
    return combos, cum_g, cum_n, bench


def fmt(x):
    return f"{x * 100:+.1f}%"


def key(score, n, sch, sl):
    return f"{score}|N{n}|{sch}|sl{int(round(sl * 100))}"


def ref_rows(cum, bench, market, ncols):
    bl = "大盤（加權指數TAIEX）" if market == "TW" else "大盤（S&P500）"
    rows = [("control0（不調整）", cum["REF_control0"]), ("每檔等權", cum["REF_stock_equal"]),
            ("市值傾斜全開（上限30%）", cum["REF_w2c_full"]), (bl, bench)]
    return [[lbl] + [fmt(v)] * ncols for lbl, v in rows]


def main():
    global TAB_DIR
    if "--liq" in sys.argv:
        SUFFIX["US"] = "_liq"
        TAB_DIR = OUT_DIR / "tables_cum_liq"
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    md_parts = ["# 優等生權重規則：2024Q1~2025Q4 共 8 季累積報酬\n",
                "累積報酬＝8季季報酬複利（與 control0 +14.44% 同一算法）。參考列（control0／每檔等權／"
                "市值傾斜全開／大盤）不隨欄位變動，整列同值。\n"]
    appendix = []
    for g in GROUPS:
        market, pool = g
        combos, cg, cn, bench = cum_table(market, pool)
        for cost_lbl, cum in (("未扣成本", cg), ("扣成本", cn)):
            # 表A
            colsA = [f"資金{int(s*100)}%" for s in SLEEVES]
            rowsA = [[SCORE_LABEL[s]] + [fmt(cum[key(s, 3, "equal", sl)]) for sl in SLEEVES] for s in SCORE_ORDER]
            rowsA += ref_rows(cum, bench, market, len(SLEEVES))
            tA = pd.DataFrame(rowsA, columns=["排名依據（前3名・平均分）"] + colsA)
            # 表B
            colsB = [("前1名" if n == 1 else f"前{n}名・{SCHEME_LABEL[s]}") for n, s in NSCH]
            rowsB = [[SCORE_LABEL[s]] + [fmt(cum[key(s, n, sch, 0.3)]) for n, sch in NSCH] for s in SCORE_ORDER]
            rowsB += ref_rows(cum, bench, market, len(NSCH))
            tB = pd.DataFrame(rowsB, columns=["排名依據（資金30%）"] + colsB)
            stem = f"{market}_{pool}_{'gross' if cost_lbl == '未扣成本' else 'net'}"
            tA.to_csv(TAB_DIR / f"tableA_{stem}.csv", index=False, encoding="utf-8-sig")
            tB.to_csv(TAB_DIR / f"tableB_{stem}.csv", index=False, encoding="utf-8-sig")
            md_parts += [f"\n## {GROUP_LABEL[g]}（{cost_lbl}）\n",
                         "### 表A　排名依據 × 資金比例（固定前3名・平均分）\n", to_md(tA),
                         "\n\n### 表B　排名依據 × 取幾名・分法（固定資金30%）\n", to_md(tB), "\n"]
        # 附錄
        m = pd.read_csv(OUT_DIR / f"metrics_all_{market}_{pool}{SUFFIX[market]}.csv", index_col=0)
        for c, r in combos.iterrows():
            if c.startswith("REF_"):
                continue
            appendix.append({"組別": GROUP_LABEL[g], "排名依據": SCORE_LABEL[r.score], "取幾名": int(r.n),
                             "分法": SCHEME_LABEL[r.scheme] if r.n > 1 else "（只取1名）",
                             "資金比例": f"{int(round(r.sleeve*100))}%",
                             "8季累積（未扣成本）": fmt(cg[c]), "8季累積（扣成本）": fmt(cn[c]),
                             "8季日頻MDD（扣成本）": fmt(m.loc[c, "OOS_net_mdd"]),
                             "2015–23 累積（扣成本）": fmt(m.loc[c, "IS_net_cum"]),
                             "2015–23 Sharpe（扣成本）": f"{m.loc[c, 'IS_net_sharpe']:.2f}",
                             "是否晉級": "是" if m.loc[c, "promoted"] else ""})
    (TAB_DIR / "cum8q_tables.md").write_text("\n".join(md_parts), encoding="utf-8")
    pd.DataFrame(appendix).to_csv(TAB_DIR / "appendix_all_1040x4.csv", index=False, encoding="utf-8-sig")
    print("輸出：", sorted(p.name for p in TAB_DIR.iterdir()))


if __name__ == "__main__":
    main()
