# -*- coding: utf-8 -*-
"""交易成本加倍敏感度・第二步：名單固定，樣本外每月再扣一份成本，重算 RQ2（2026-11-15，使用者授權，經寫作 session 轉達）。

做法：
  - 名單不動：A_hrp／D_top_cagr／E_top_calmar 用 `walkforward_members.parquet`（11-14 這一輪）存的成員；
    B_all 依 `walkforward_matrix` 同一規則重建（可用策略、hist_start ≤ IS 起點、IS 窗零變異數排除），
    並要求檔數＝明細表的 n_universe。
  - 樣本外組合月報酬（等權，同 `_portfolio_series`）先重算一次，必須等於明細表的 oos_cagr／oos_mdd／oos_sharpe（對帳），
    再減去成員當月平均的多扣成本（`cost_double_strategy_cost.parquet`）得到成本加倍版。
  - 顯著性：沿用 `wf_unit_rule_sensitivity` 的三種單位判定（多數決＝現行／平均差／中位數差）、
    二項檢定與族內 BH／Bonferroni；成本加倍前後各算一次。多數決的「加倍前」必須等於現行
    walkforward_significance.csv／wf_multiplicity.csv。
  - C_random 不做（沒有存每次抽取的成員）。
  - 美股另按 A_hrp 成員的小型股持股比例（stage1 `smallcap_share`）分三組，看成本拖累。
只讀現有結果與持股檔推出的成本序列，不建樹、不呼叫 LLM。
輸出（`_analysis_outputs_robustness/`）：
  wf_cost_double_detail.csv（逐格，加倍前後並列）、wf_cost_double_tests.csv、wf_cost_double_summary.csv、
  wf_cost_double_groups.csv（各樹×組別的樣本外指標中位數）、wf_cost_double_us_size.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, paths
from . import stage3_hrp as S3
from . import walkforward_matrix as WM
from . import walkforward_significance as WS
from . import wf_unit_rule_sensitivity as URS
from .four_group_control import _cagr, _mdd, _sharpe

OUT = paths.ROOT / "_analysis_outputs_robustness"
WKEY = ["tree_key", "scheme", "window_no", "k_mode"]
MKEY = ["tree_key", "scheme", "k_mode", "ratio", "allocation", "group", "window_no"]


def _metrics(s: np.ndarray, months) -> tuple[float, float, float]:
    x = pd.Series(s, index=months)
    return _cagr(x), _mdd(x), _sharpe(x)


def recompute(log=print) -> pd.DataFrame:
    det = pd.read_csv(OUT / "walkforward_matrix_detail.csv")
    mem = pd.read_parquet(OUT / "walkforward_members.parquet")
    for c in MKEY[:-1]:
        mem[c] = mem[c].astype(str)
    mem = mem.set_index(MKEY)["members"]
    months_long, meta, _ = WM._load_inputs()
    cost = pd.read_parquet(OUT / "cost_double_strategy_cost.parquet")
    cost.columns = pd.PeriodIndex(cost.columns, freq="M")

    out = []
    for (tk, sch, wn), g in det.groupby(["tree_key", "scheme", "window_no"], sort=False):
        r0 = g.iloc[0]
        uids = S3._tree_universe(tk, r0.is_start, meta)
        wide_is = S3._pivot_window(months_long, uids, r0.is_start, r0.is_end)
        wide_is, _ = S3._drop_zero_variance(wide_is, log=lambda *a, **k: None)
        uni = wide_is.index
        assert (g.n_universe == len(uni)).all(), (tk, sch, wn, len(uni), g.n_universe.unique())
        w = months_long[months_long.strategy_uid.isin(set(uni))]
        w = w[(w.month >= pd.Period(r0.oos_start, "M")) & (w.month <= pd.Period(r0.oos_end, "M"))]
        R = w.pivot(index="strategy_uid", columns="month", values="ret").loc[uni]
        assert not R.isna().any().any()
        Cm = cost.reindex(index=uni, columns=R.columns)
        assert not Cm.isna().any().any(), "成本序列缺策略或月份"
        Rv, Cv = R.to_numpy(np.float64), Cm.to_numpy(np.float64)
        pos = pd.Series(np.arange(len(uni)), index=uni)
        for _, row in g.iterrows():
            if row.group == "B_all":
                ix = np.arange(len(uni))
            else:
                key = (tk, sch, row.k_mode, str(row.ratio), str(row.allocation), row.group, int(wn))
                ix = pos.reindex(list(mem.loc[key])).to_numpy()
                assert not np.isnan(ix.astype(float)).any() and len(ix) == row.n_members, key
                ix = ix.astype(int)
            p = Rv[ix].mean(0)
            c = Cv[ix].mean(0)
            c0, m0, s0 = _metrics(p, R.columns)
            assert np.isclose(c0, row.oos_cagr, atol=1e-10) and np.isclose(m0, row.oos_mdd, atol=1e-10) \
                and np.isclose(s0, row.oos_sharpe, atol=1e-9), ("對帳失敗", tk, sch, wn, row.group)
            c2, m2, s2 = _metrics(p - c, R.columns)
            out.append({**{k: row[k] for k in ("tree_key", "scheme", "window_no", "k_mode", "ratio",
                                                "allocation", "group", "n_members", "oos_start", "oos_end")},
                        "oos_cagr": row.oos_cagr, "oos_mdd": row.oos_mdd, "oos_sharpe": row.oos_sharpe,
                        "oos_cagr_x2": c2, "oos_mdd_x2": m2, "oos_sharpe_x2": s2,
                        "extra_cost_ann": float(c.mean() * 12)})
        log(f"  {tk} {sch} w{wn} 完成（宇宙 {len(uni):,}）", flush=True) if wn == 1 else None
    return pd.DataFrame(out)


def _as_ws(d: pd.DataFrame, x2: bool) -> pd.DataFrame:
    s = "_x2" if x2 else ""
    df = d.rename(columns={f"oos_cagr{s}": "cagr", f"oos_mdd{s}": "mdd"}).copy()
    df["calmar"] = df.cagr / df.mdd.abs()
    for c in ("ratio", "allocation"):
        df[c] = df[c].astype(str)
    return df


def _cell_diffs_noC(df: pd.DataFrame) -> pd.DataFrame:
    a = df[df.group == "A_hrp"].set_index(URS.SETTING)
    rows = []
    for opp in WS.OPPONENTS:
        o = df[df.group == opp]
        for m in URS.METRICS:
            if opp == "B_all":
                ser = o.set_index(URS.WKEY)[m]
                ov = a.reset_index().set_index(URS.WKEY).index.map(ser)
            else:
                ov = o.set_index(URS.SETTING)[m].reindex(a.index).to_numpy()
            rows.append(URS._frame(a, np.asarray(ov, dtype=float), opp, m))
    return pd.concat(rows, ignore_index=True)


def tests(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts, ss = [], []
    for x2, tag in ((False, "原成本"), (True, "成本加倍")):
        t = URS.sign_tests(URS.unit_table(_cell_diffs_noC(_as_ws(d, x2))))
        if not x2:
            URS.reconcile(t)       # 多數決＝現行 walkforward_significance／wf_multiplicity
        s = URS.summary(t)
        t.insert(0, "成本", tag)
        s.insert(0, "成本", tag)
        ts.append(t)
        ss.append(s)
    return pd.concat(ts, ignore_index=True), pd.concat(ss, ignore_index=True)


def main(argv=None) -> int:
    for dd in (paths.STAGE0, paths.STAGE1, paths.STAGE1 / "_marks"):
        freeze.verify_inputs(dd)
    freeze.verify_inputs(OUT / "_walkforward_matrix_manifest")
    freeze.verify_inputs(OUT / "_cost_double_series_manifest")
    d = recompute()
    print("✓ 名單固定重算的樣本外指標＝現行明細表（8,370 格逐格對帳）")
    t, s = tests(d)
    print("✓ 原成本的多數決＝現行 walkforward_significance／wf_multiplicity")

    grp = (d.groupby(["tree_key", "group"])
           [["oos_cagr", "oos_cagr_x2", "oos_sharpe", "oos_sharpe_x2", "oos_mdd", "oos_mdd_x2", "extra_cost_ann"]]
           .median().reset_index())
    grp["cagr_差"] = grp.oos_cagr_x2 - grp.oos_cagr

    scan = pd.read_parquet(paths.STAGE1 / "strategy_scan.parquet").set_index(C.PK)["smallcap_share"]
    mem = pd.read_parquet(OUT / "walkforward_members.parquet")
    us = mem[(mem.tree_key.astype(str) == "US") & (mem.group.astype(str) == "A_hrp")].copy()
    us["小型股持股比例"] = us.members.map(lambda m: float(scan.reindex(list(m)).mean()))
    for c in MKEY[:-1]:
        us[c] = us[c].astype(str)
    dd = d[(d.tree_key == "US") & (d.group == "A_hrp")].copy()
    for c in MKEY[:-1]:
        dd[c] = dd[c].astype(str)
    dd = dd.merge(us[MKEY + ["小型股持股比例"]], on=MKEY, validate="one_to_one")
    dd["小型股組"] = pd.qcut(dd["小型股持股比例"], 3, labels=["低", "中", "高"])
    size = (dd.assign(cagr_差=dd.oos_cagr_x2 - dd.oos_cagr)
            .groupby("小型股組", observed=True)
            .agg(格數=("cagr_差", "size"), 小型股持股比例中位=("小型股持股比例", "median"),
                 年化多扣成本中位=("extra_cost_ann", "median"), CAGR變化中位=("cagr_差", "median"),
                 CAGR原中位=("oos_cagr", "median"), CAGR加倍中位=("oos_cagr_x2", "median"))
            .reset_index())
    bu = d[(d.tree_key == "US") & (d.group == "B_all")]
    size.attrs["B_all"] = float((bu.oos_cagr_x2 - bu.oos_cagr).median())

    ps = {k: OUT / f"wf_cost_double_{k}.csv" for k in ("detail", "tests", "summary", "groups", "us_size")}
    for k, df in (("detail", d), ("tests", t), ("summary", s), ("groups", grp), ("us_size", size)):
        df.to_csv(ps[k], index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "wf_cost_double", OUT / "_wf_cost_double_manifest",
        inputs=[OUT / "walkforward_matrix_detail.csv", OUT / "walkforward_members.parquet",
                OUT / "cost_double_strategy_cost.parquet", OUT / "walkforward_significance.csv",
                OUT / "wf_multiplicity.csv", paths.STAGE1 / "returns_monthly.parquet",
                paths.STAGE1 / "strategy_scan.parquet"],
        outputs=list(ps.values()),
        params={"rules": list(URS.RULES), "alpha": URS.ALPHA, "c_random": "不做（未存抽取成員）"},
        notes="名單固定，樣本外每月再扣一份成本；對帳：原成本重算＝明細表、多數決＝現行檢定。")
    pd.set_option("display.width", 250)
    print(s[["成本", "opponent", "metric", "rule", "n_sig_raw", "n_sig_bh_family", "n_sig_bonferroni_family",
             "units_won_total", "units_total"]].to_string(index=False))
    print(grp.round(4).to_string(index=False))
    print(size.round(4).to_string(index=False), "\nUS B_all CAGR 變化中位", round(size.attrs["B_all"], 4))
    print(*ps.values(), sep="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
