# -*- coding: utf-8 -*-
"""2026-10-18 補充分析：L3層級的逐群逐年報酬（論文session提出，4.2.1(c)需要）。

背景：現有`_frozen/stage3/cluster_annual_returns.parquet`只做L1（TW 6／US 4／
XM 7群），`cluster_temporal_profile.py`開頭docstring寫明理由是「L3群數太多
不適合逐群描述」——但那是對「給LLM讀/論文呈現」的描述性用途而言，不代表L3
的逐年報酬資料「算不出來」。這裡只是要L3（TW 471／US 303／XM 774群，平均每群
數十檔）的逐年報酬本身，拿來檢查「全球下跌年，L3細群之間是否出現比L1更多
『一個賠時另一個賺』」，不需要L1版本那張給LLM用的定性特徵表（_cluster_profiles
那段，跳過不算，省時間也避免無謂依賴）。

完全重用`cluster_temporal_profile._member_wide_returns()`／`_annual_quarterly()`
的邏輯，只把LEVEL從"L1"換成"L3"。不重建任何樹——`cluster_assign.parquet`已有
`cluster_L3`欄位（2026-10-18查證過：TW 471/US 303/XM 774，跟論文大綱現有數字
一致）。

✅ 另存新檔，不覆蓋`_frozen/stage3/cluster_annual_returns.parquet`（那份是L1
專屬的凍結鏈產物），也不走freeze/manifest（跟`cluster_temporal_profile.py`
自己的附加分析一樣的理由：這是補充查核，不是stage3本體）。

用法：
    cd code
    python -m research._check_cluster_annual_returns_l3
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from . import contracts as C
from . import freeze, paths
from .cluster_temporal_profile import DEFAULT_TREES, _annual_quarterly, _member_wide_returns

LEVEL = "L3"


def build(trees=DEFAULT_TREES, log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE3)

    months_long = pd.read_parquet(paths.STAGE1 / "returns_monthly.parquet")
    assign = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")

    annual_rows = []
    for tree_id in trees:
        t0 = time.time()
        wide = _member_wide_returns(tree_id, months_long, assign)
        a = assign[(assign.tree_id == tree_id)][[C.PK, f"cluster_{LEVEL}"]].rename(
            columns={f"cluster_{LEVEL}": "_cl"})

        n_clusters = 0
        for cid, g in a.groupby("_cl"):
            members = [u for u in g[C.PK] if u in wide.index]
            if not members:
                continue
            rep = wide.loc[members].mean(axis=0).dropna()
            rep.index = pd.PeriodIndex(rep.index, freq="M")
            ann, _qtr = _annual_quarterly(rep)
            for r in ann.itertuples():
                annual_rows.append({"tree_id": tree_id, "level": LEVEL, "cluster_id": int(cid),
                                    "year": r.year, "ret": r.ret, "n_months": r.n_months,
                                    "n_members": len(members)})
            n_clusters += 1
        log(f"[{tree_id}] {n_clusters} 群完成｜{time.time()-t0:.1f}s")

    df = pd.DataFrame(annual_rows)
    df["tree_id"] = df["tree_id"].astype("category")
    df["level"] = df["level"].astype("category")
    return df


def main() -> int:
    t0 = time.time()
    df = build()
    out = paths.ROOT / "_analysis_outputs_robustness" / "cluster_annual_returns_L3.parquet"
    df.to_parquet(out, compression="zstd", index=False)
    print(f"\n→ {out}  {len(df):,} 列")
    print(f"總耗時 {time.time()-t0:.1f}s")

    print("\n各樹L3群數：")
    print(df.groupby("tree_id", observed=True)["cluster_id"].nunique().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
