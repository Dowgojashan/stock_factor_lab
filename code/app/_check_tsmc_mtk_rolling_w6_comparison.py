# -*- coding: utf-8 -*-
"""填補§3.10留下的缺口：§3.10查的「台積電/聯發科逐季選中比例」全部只用了
anchored window4候選池（baseline跟exclude_v1兩邊都是），rolling候選池底下
台積電/聯發科具體選中比例還沒有查過。這裡補上——rolling window6（baseline
跟exclude_v1兩個變體都要），跟§3.10同一組8個as_of日期，才能三邊（anchored
baseline／rolling baseline／rolling exclude_v1）直接比較，另外anchored
exclude_v1已經有§3.10的資料可以一起放進來看四邊。

重用`_check_tsmc_mtk_exclude_v1_comparison.py`的`check_members()`／
`asof_bool()`（通用函式，不是anchored專屬），只換members清單來源。

用法（cwd 必須是 code/；PYTHONIOENCODING=utf-8；先跑過
`_check_rolling_window6_silhouette_picks.py`跟`_build_exclude_v1_silhouette_
picks.py`產生兩個候選池parquet才能跑這支）：
    PYTHONIOENCODING=utf-8 python -m app._check_tsmc_mtk_rolling_w6_comparison
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import pandas as pd  # noqa: E402
from fcv_core import MarketData  # noqa: E402
from resolve_strategy_holdings import CANDIDATE_INDEX_PATH  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _check_tsmc_mtk_exclude_v1_comparison import check_members  # noqa: E402

TREE_KEY = "TW"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"


def load_members(path: Path, scheme: str, window_no: int, allocation: str = "equal") -> list[str]:
    m = pd.read_parquet(path)
    sub = m[(m.tree_key == TREE_KEY) & (m.scheme == scheme) & (m.window_no == window_no)
           & (m.k_mode == "silhouette_is") & (m.allocation == allocation) & (m.group == "A_hrp")]
    assert len(sub) == 1, f"{path.name} 缺這一列（scheme={scheme}, window_no={window_no}, allocation={allocation}）"
    return list(sub.iloc[0]["members"])


def main():
    idx = pd.read_parquet(CANDIDATE_INDEX_PATH).set_index("strategy_uid")

    rolling_baseline = load_members(OUT_DIR / "rolling_window6_silhouette_members.parquet", "rolling_6_2", 6)
    rolling_ev1 = load_members(OUT_DIR / "rolling_w6_silhouette_exclude_v1.parquet", "rolling_6_2", 6)
    print(f"rolling window6 baseline: {len(rolling_baseline)}檔｜exclude_v1: {len(rolling_ev1)}檔")

    print(">> 載入 TW MarketData ...")
    md = MarketData(TREE_KEY)

    df_base = check_members(rolling_baseline, idx, md, "rolling_baseline")
    df_ev1 = check_members(rolling_ev1, idx, md, "rolling_exclude_v1")
    df = pd.concat([df_base, df_ev1], ignore_index=True)

    out = OUT_DIR / "tsmc_mtk_rolling_w6_comparison.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n寫入 {out}")

    print("\n=== 逐季選中檔數：rolling baseline vs rolling exclude_v1 ===")
    piv = df.groupby(["as_of", "symbol", "variant"])["selected"].sum().unstack("variant").reindex(
        columns=["rolling_baseline", "rolling_exclude_v1"])
    print(f"（分母：baseline={len(rolling_baseline)}檔，exclude_v1={len(rolling_ev1)}檔）")
    print(piv.to_string())

    print("\n=== 跟§3.10的anchored結果對照（4邊：anchored baseline/exclude_v1 vs rolling baseline/exclude_v1）===")
    anchored_csv = OUT_DIR / "tsmc_mtk_exclude_v1_comparison.csv"
    if anchored_csv.exists():
        adf = pd.read_csv(anchored_csv)
        adf["variant"] = adf["variant"].replace({"baseline": "anchored_baseline",
                                                  "exclude_v1": "anchored_exclude_v1"})
        n_anchored_base = adf[adf.variant == "anchored_baseline"]["uid"].nunique()
        n_anchored_ev1 = adf[adf.variant == "anchored_exclude_v1"]["uid"].nunique()
        combined = pd.concat([adf[["as_of", "symbol", "variant", "selected"]], df[["as_of", "symbol", "variant", "selected"]]],
                             ignore_index=True)
        piv4 = combined.groupby(["as_of", "symbol", "variant"])["selected"].sum().unstack("variant").reindex(
            columns=["anchored_baseline", "anchored_exclude_v1", "rolling_baseline", "rolling_exclude_v1"])
        print(f"（分母：anchored_baseline={n_anchored_base}、anchored_exclude_v1={n_anchored_ev1}、"
             f"rolling_baseline={len(rolling_baseline)}、rolling_exclude_v1={len(rolling_ev1)}）")
        print(piv4.to_string())
    else:
        print(f"找不到 {anchored_csv}，跳過四邊對照")


if __name__ == "__main__":
    main()
