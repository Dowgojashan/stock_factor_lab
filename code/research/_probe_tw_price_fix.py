# -*- coding: utf-8 -*-
"""台股價格修正的影響評估（2026-10-30，使用者授權「先做評估」；不改任何凍結產物）。

抽 N 個台股候選策略，用修正後的 DB 重新回測（同 Phase SOP：`resolve_strategy_holdings` 的條件重建
＋`combinations.sim_conditions`，調倉頻率照原 job 的 _M／_Q），跟凍結的 `_frozen/stage1/returns_monthly`
比較，回答「研究部要不要重跑、要從哪一關重跑」：

  0. 可重現性：2022-01 以前價格幾乎沒改，新舊月報酬應幾乎相同；不同就代表重建方式有誤、後面都不可信
  1. 月報酬差異：依年份
  2. window4 代表策略的挑選依據（IS 2007-01～2023-12 的 Calmar，同 walkforward_matrix）新舊排名相關、前 10% 重疊
  3. 全期間 CAGR 是否跨過自建基準（候選池門檻；基準本身也會變，這裡先用現行常數，只看跨線的檔數）
  4. 2024–25 報酬的變化

輸出：`_analysis_outputs_robustness/probe_tw_price_fix_2026-10-30/`
用法（cwd＝code/）：python -m research._probe_tw_price_fix --n 300
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from research import contracts as C  # noqa: E402
from research import paths  # noqa: E402
from research.four_group_control import _cagr_matrix, _mdd_matrix  # noqa: E402
from research.stage1_scan import _from_return_table  # noqa: E402

OUT = paths.ROOT / "_analysis_outputs_robustness" / "probe_tw_price_fix_2026-10-30"


def combined_mask(md, row):
    from resolve_strategy_holdings import _c_condition, _q_band_condition
    masks = [md.get_mask(_q_band_condition(row.F1_factor, row.F1_band, row.F1_nbands))]
    if not row.F2_empty:
        masks.append(md.get_mask(_q_band_condition(row.F2_factor, row.F2_band, row.F2_nbands)))
    if pd.notna(row.C_rule):
        masks.append(md.get_mask(_c_condition(row.C_source, row.C_rule)))
    if any(m is None for m in masks):
        return None
    m = masks[0]
    for x in masks[1:]:
        m = m & x
    if row.V == "v1":
        m = m & md.get_v_mask()
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from combinations import sim_conditions
    from fcv_core import MarketData, ensure_dtindex_cols
    OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(paths.STAGE0 / "candidate_index.parquet")
    tw = idx[idx.market == "TW"].sample(a.n, random_state=a.seed).copy()
    tw["reb"] = [Path(p).parent.name.rsplit("_", 1)[-1] for p in tw.artifacts_dir]
    assert set(tw.reb) <= {"M", "Q"}, set(tw.reb)
    t0 = time.time()
    md = MarketData("TW")
    new = {}
    for reb, g in tw.groupby("reb"):
        batch = {}
        for r in g.itertuples():
            m = combined_mask(md, r)
            if m is not None:
                batch[r.strategy_uid] = ensure_dtindex_cols(m, md.common)
        for i in range(0, len(batch), 50):
            keys = list(batch)[i:i + 50]
            rc = sim_conditions(conditions={k: batch[k] for k in keys}, resample=reb, data=md.data)
            for k, rep in rc.reports.items():
                rt = fcv_core.return_table_to_df(getattr(rep, "return_table", None))
                if rt is not None:
                    d = _from_return_table(rt)
                    if not d.get("_empty"):
                        new[k] = d["_monthly"]
            print(f"  [{reb}] {min(i + 50, len(batch))}/{len(batch)}  {time.time() - t0:.0f}s", flush=True)
    new = {k: v for k, v in new.items() if v is not None}
    old_long = pd.read_parquet(paths.STAGE1 / "returns_monthly.parquet", filters=[("strategy_uid", "in", list(new))])
    old = old_long.pivot(index="month", columns="strategy_uid", values="ret")
    old.index = pd.PeriodIndex(old.index, freq="M")
    neww = pd.DataFrame(new)
    common = [c for c in neww.columns if c in old.columns]
    neww, old = neww[common], old[common]
    allm = old.index.union(neww.index)
    neww, old = neww.reindex(allm), old.reindex(allm)
    pd.concat({"new": neww, "old": old}, axis=1).to_parquet(OUT / "monthly_new_old.parquet")

    L = [f"# 台股價格修正影響評估（抽 {len(common)} 個策略，seed={a.seed}）", ""]
    pre = (allm < pd.Period("2022-01", "M"))
    d_pre = (neww[pre] - old[pre]).abs()
    L.append(f"0. 可重現性（2022-01 以前）：月報酬差的最大值 {np.nanmax(d_pre.to_numpy()):.2e}、"
             f"差 >1e-6 的策略 {int((d_pre.max() > 1e-6).sum())}／{len(common)}")
    yr = pd.Series(allm.year, index=allm)
    L.append("1. 各年月報酬差（|新−舊| 的平均，百分點）：" + "、".join(
        f"{y} {np.nanmean((neww[yr == y] - old[yr == y]).abs().to_numpy()) * 100:.2f}" for y in range(2019, 2026)))
    IS = (allm >= pd.Period("2007-01", "M")) & (allm <= pd.Period("2023-12", "M"))
    def calmar(w):
        w = w[IS].T
        return _cagr_matrix(w) / _mdd_matrix(w).abs().replace(0, np.nan)
    qn, qo = calmar(neww), calmar(old)
    ok = qn.notna() & qo.notna()
    rho = spearmanr(qn[ok], qo[ok])[0]
    k = max(1, int(ok.sum() * 0.1))
    topn, topo = set(qn[ok].nlargest(k).index), set(qo[ok].nlargest(k).index)
    L.append(f"2. window4 挑選依據（IS 2007-01～2023-12 Calmar）：新舊排名相關 ρ={rho:.3f}；"
             f"前 10%（{k} 檔）重疊 {len(topn & topo)}／{k}；Calmar 中位數 {qo[ok].median():.3f} → {qn[ok].median():.3f}")
    def cagr_each(w):
        return pd.Series({c: float((1 + w[c].dropna()).prod() ** (12 / max(1, w[c].notna().sum())) - 1) for c in w.columns})
    cn, co = cagr_each(neww), cagr_each(old)
    bench = C.BENCHMARK_CAGR["TW"]
    flip = ((cn >= bench) != (co >= bench))
    L.append(f"3. 全期間 CAGR：中位數 {co.median():.2%} → {cn.median():.2%}；相對現行基準 {bench:.4%} 跨線的策略 "
             f"{int(flip.sum())}／{len(common)}（舊過→新不過 {int(((co >= bench) & (cn < bench)).sum())}）")
    o4 = (allm >= pd.Period("2024-01", "M")) & (allm <= pd.Period("2025-12", "M"))
    c4n = (1 + neww[o4]).prod() - 1; c4o = (1 + old[o4]).prod() - 1
    L.append(f"4. 2024–25 累積：中位數 {c4o.median():+.1%} → {c4n.median():+.1%}；新舊排名相關 {spearmanr(c4n, c4o)[0]:.3f}")
    affected = tw.set_index("strategy_uid").loc[common].apply(
        lambda r: any(str(f) in ("MOM", "MOM1", "MOM_3M", "VOL", "PROX_52WK_HIGH") for f in (r.F1_factor, r.F2_factor)), axis=1)
    dq = (qn - qo).abs()
    L.append(f"5. 用到重算因子的策略 {int(affected.sum())} 檔：IS Calmar |變化| 中位數 {dq[affected].median():.3f}；"
             f"其他 {int((~affected).sum())} 檔：{dq[~affected].median():.3f}")
    L.append(f"\n耗時 {time.time() - t0:.0f}s")
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
