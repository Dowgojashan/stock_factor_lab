# -*- coding: utf-8 -*-
"""2026-10-05：代表策略品質指標改成「近3年(36個月)加權Calmar」，疊加已驗證有效
的exclude_v1，看能不能在openSec_boost新池子下進一步把成長/動能比例拉高、
績效拉高。

動機：使用者指出目前挑代表用的是17年全樣本Calmar，天生偏好長期穩健的估值型
策略（這類策略多年平均表現好，不代表近期好），要求認真設計一個直接針對
「增加成長/動能比例」的新機制，而不是之前測過沒用的放寬數量/類別配額。
這裡改的是`_silhouette_picks_lib.build_silhouette_picks()`新增的
`quality_window_months`參數——樹/分群完全不變，只改「群內挑代表」這一步
排序用的品質分數只看IS窗最後36個月，而非全部17年。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._build_recency_quality_exclude_v1_picks
"""
from __future__ import annotations

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
QUALITY_WINDOW_MONTHS = 36  # 近3年
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
    print(">> 載入資料(今晚重建的openSec_boost池)...")
    months_long, meta_pool, f_combo_map = WF._load_inputs()
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    print(f"\n### anchored window4／exclude_v1＋近{QUALITY_WINDOW_MONTHS}個月品質 ###")
    res = build_silhouette_picks("TW", IS_START, IS_END, "exclude_v1",
                                 months_long, meta_pool, f_combo_map, idx,
                                 quality_window_months=QUALITY_WINDOW_MONTHS)
    save_picks(OUT_DIR / "anchored_w4_openSec_boost_exclude_v1_recent3y.parquet",
              "exclude_v1_recent3y", res)


if __name__ == "__main__":
    main()
