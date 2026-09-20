# -*- coding: utf-8 -*-
"""M8（產業集中度）歷史分布：window 1-3（2015-2023，符合 §11.7 只用訓練期
資料的規則）逐季算最大單一產業佔比，供 `diagnose.py` 訂 p90 門檻用——
跟 M3（`_prelim_a1_portfolio_noise.py`）同一套方法，只是這次算的是產業
而不是股票數。

🔴 已知限制（沿用 `industry.py` 的既有揭露，非本腳本新增）：產業分類是
「現在最新」的靜態快照，用它回溯 2015-2023 的持股，等於用今天的分類套用
到過去——若某公司產業別曾經改變會有標籤不精確的問題。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from database import Database  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402

from app import industry  # noqa: E402
from app._prelim_a5_w2_prevalidation import resolve_w0  # noqa: E402

OUT_PATH = (Path(__file__).resolve().parent.parent.parent
           / "_analysis_outputs_applayer" / "industry_concentration_windows123.csv")

WINDOWS = {
    1: {"is_end": "2014-12-31", "oos_start": "2015-01-01", "oos_end": "2017-12-31"},
    2: {"is_end": "2017-12-31", "oos_start": "2018-01-01", "oos_end": "2020-12-31"},
    3: {"is_end": "2020-12-31", "oos_start": "2021-01-01", "oos_end": "2023-12-31"},
}


def quarter_ends(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="Q")]


def get_members(window_no: int) -> list[str]:
    m = pd.read_parquet(Path(__file__).resolve().parent.parent.parent
                        / "_analysis_outputs_robustness" / "walkforward_members.parquet")
    key = dict(tree_key="TW", scheme="E", window_no=window_no, k_mode="silhouette_is",
              ratio="legacy", allocation="equal", group="A_hrp")
    sub = m.copy()
    for k, v in key.items():
        sub = sub[sub[k] == v]
    assert len(sub) == 1
    return list(sub.iloc[0]["members"])


def main():
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")
    print("載入 TW MarketData…")
    md = MarketData("TW")
    imap = industry.load_industry_map("TW")

    rows = []
    for wno, meta in WINDOWS.items():
        uids = get_members(wno)
        checkpoints = [meta["is_end"]] + quarter_ends(meta["oos_start"], meta["oos_end"])
        for as_of in checkpoints:
            w0 = resolve_w0(md, idx, uids, as_of)
            if not w0:
                continue
            top_industry, top_weight = industry.max_industry_weight(w0, imap)
            exposure = industry.industry_exposure(w0, imap)
            unclassified = exposure.get("未分類", 0.0)
            rows.append({"window_no": wno, "as_of": as_of, "top_industry": top_industry,
                        "max_industry_weight": top_weight, "unclassified_weight": unclassified,
                        "n_industries": len(exposure)})
            print(f"  window {wno} {as_of}: top={top_industry} ({top_weight:.2%})"
                 f" 未分類={unclassified:.2%} 產業數={len(exposure)}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {OUT_PATH}，共 {len(df)} 筆")

    s = df["max_industry_weight"]
    print(f"\n=== 歷史分布（n={len(s)}）===")
    print(f"p10={s.quantile(0.10):.4f}  中位數={s.quantile(0.50):.4f}"
         f"  p75={s.quantile(0.75):.4f}  p90={s.quantile(0.90):.4f}  最大值={s.max():.4f}")


if __name__ == "__main__":
    main()
