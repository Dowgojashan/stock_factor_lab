# -*- coding: utf-8 -*-
"""規則排名圖的資金比例敏感度（2026-11-15，使用者授權，經寫作 session 轉達；論文附錄 C 參數表的「檢驗」欄）。

`_rank_dynamics` (A)：每季季末，同一資金比例的 208 組依「過去 4 季累積報酬（扣成本）」排名。
論文用資金 30%。本檔對 10／20／30／50／100% 各算：
  - 相鄰兩季排名的 Spearman 相關（208 組，跨 40 對的中位數）。
    ⚠️ 相鄰兩季的回看期重疊 3 季，相關會被機械性墊高；另報相隔 4 季（回看期不重疊）的版本。
  - 13 類「中位數名次」時間序列（41 季）與 30% 版本的相關：每類各算 Pearson，報 13 類的中位數與最小值，
    另報 13×41 格整體的 Spearman。
30% 的中位數名次必須等於 `rank_dynamics/{組}/資金30%/00_總覽熱圖_數值.csv`（11-10），否則中止。
台股步驟1 的季報酬是 2026-11-17 依研究部 11-14 新池重建名單後重跑的版本；美股用 `_liq` 版本。
不呼叫 LLM、不重跑回測。輸出：`_analysis_outputs_applayer/rank_dynamics/sleeve_sensitivity.csv`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app import _rank_dynamics as RD  # noqa: E402

FAMS = [f for f in RD.FAM if f != "REF_control0"]


def fam_median(rank: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({f: rank[[c for c in rank.columns if RD.fam_of(c) == f]].median(axis=1) for f in FAMS}).T


def lag_spearman(rank: pd.DataFrame, lag: int) -> np.ndarray:
    r = rank.drop(columns="REF_control0")
    return np.array([r.iloc[i].rank().corr(r.iloc[i + lag].rank()) for i in range(len(r) - lag)])


def main() -> int:
    rows = []
    for market, pool, suf in RD.GROUPS:
        _, net, meta = RD.load(market, pool, suf)
        ranks = {sl: RD.trailing_rank(net, meta, sleeve=sl) for sl in RD.SLEEVES}
        meds = {sl: fam_median(r) for sl, r in ranks.items()}
        ref = pd.read_csv(RD.OUT / f"{market}_{pool}" / "資金30%" / "00_總覽熱圖_數值.csv", index_col=0)
        chk = meds[0.3].copy()
        chk.index = [RD.FAM[f] for f in chk.index]
        assert np.allclose(chk.to_numpy(), ref.loc[chk.index, list(chk.columns)].to_numpy()), (market, pool)
        base = meds[0.3]
        for sl in RD.SLEEVES:
            r = ranks[sl]
            assert r.shape[1] - 1 == 208, (market, pool, sl, r.shape)
            s1, s4 = lag_spearman(r, 1), lag_spearman(r, 4)
            per_fam = [np.corrcoef(meds[sl].loc[f], base.loc[f])[0, 1] for f in FAMS]
            pooled = pd.Series(meds[sl].to_numpy().ravel()).corr(pd.Series(base.to_numpy().ravel()), method="spearman")
            rows.append({"組別": f"{RD.GLAB[market]}・{RD.GLAB[pool]}", "資金比例": f"{sl:.0%}",
                         "相鄰兩季Spearman_中位": float(np.median(s1)), "相鄰兩季Spearman_P25": float(np.quantile(s1, .25)),
                         "相鄰兩季Spearman_P75": float(np.quantile(s1, .75)),
                         "相隔4季Spearman_中位": float(np.median(s4)), "相隔4季Spearman_為負的比例": float((s4 < 0).mean()),
                         "13類中位名次序列與30%相關_各類中位": float(np.median(per_fam)),
                         "13類中位名次序列與30%相關_最小": float(np.min(per_fam)),
                         "最小的類別": RD.FAM[FAMS[int(np.argmin(per_fam))]],
                         "13類×41季整體Spearman_與30%": float(pooled),
                         "資料": "台股步驟1（11-17 新名單）" if market == "TW" else "美股 _liq 版本"})
        print(f"✓ {market}_{pool}：30% 的中位數名次＝11-10 熱圖數值", flush=True)
    res = pd.DataFrame(rows)
    p = RD.OUT / "sleeve_sensitivity.csv"
    res.to_csv(p, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
    print("→", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
