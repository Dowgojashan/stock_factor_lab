# -*- coding: utf-8 -*-
"""2026-10-05：代表策略品質指標窗口長度掃描（延續_recency_quality_exclude_v1_picks.py
的方向，36個月版本L2結果+30.77%反而比全樣本品質的+38.17%差，測試是不是3年剛好
選到不好的窗口，還是整個方向都不如預期）。

用法：
    PYTHONIOENCODING=utf-8 python -m app._build_recency_quality_variant --months 60
    PYTHONIOENCODING=utf-8 python -m app._build_recency_quality_variant --months 24
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402

from research import walkforward_matrix as WF  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _silhouette_picks_lib import build_silhouette_picks  # noqa: E402

RATIO = "legacy"
IS_START, IS_END = "2007-01", "2023-12"
OOS_START, OOS_END = "2024-01", "2025-12"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"


def save_picks(out_path: Path, variant: str, result: dict) -> None:
    rows = []
    for allocation in ("equal", "proportional"):
        rows.append({
            "tree_key": "TW", "scheme": "E", "window_no": 4,
            "k_mode": "silhouette_is", "ratio": RATIO, "allocation": allocation,
            "group": "A_hrp", "variant": variant,
            "is_start": IS_START, "is_end": IS_END, "oos_start": OOS_START, "oos_end": OOS_END,
            "members": result[allocation],
        })
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"寫入 {out_path}（{result['_meta']}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, required=True)
    args = ap.parse_args()
    m = args.months

    print(f">> 載入資料(今晚重建的openSec_boost池)，品質窗口={m}個月...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    print(f"\n### anchored window4／exclude_v1＋近{m}個月品質 ###")
    res = build_silhouette_picks("TW", IS_START, IS_END, "exclude_v1",
                                 months_long, meta_pool, f_combo_map, idx,
                                 quality_window_months=m)
    save_picks(OUT_DIR / f"anchored_w4_openSec_boost_exclude_v1_recent{m}mo.parquet",
              f"exclude_v1_recent{m}mo", res)


if __name__ == "__main__":
    main()
