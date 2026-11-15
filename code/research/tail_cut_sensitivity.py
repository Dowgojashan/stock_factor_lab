# -*- coding: utf-8 -*-
"""尾端硬篩門檻敏感度（2026-11-15，使用者授權，經寫作 session 轉達；論文附錄 C 參數表的「檢驗」欄）。

現行（`stage1_marks.py:48-50`）：有效股票數 ≤ 市場內 P10 且 輪動分數 ≥ P95 → 淘汰；空手比例 ≥ P95 → 淘汰。
本檔把有效股票數門檻換成 P5／P10／P15，輪動分數與空手比例門檻一起換成 P90／P95／P99（共 9 組），
**只在記憶體重算 is_usable，不覆寫 `_frozen/stage1`**。每組報：
  - TW／US 的淘汰比例
  - 與現行 is_usable 名單的重疊（Jaccard、現行可用中仍保留的比例、新名單中屬現行可用的比例）
  - 三棵平時樹在全期間共同窗的策略層 ENB（`enb_null.fast_enb`，與 H-09 同一算法；
    現行組合必須等於 `effective_number_of_bets.csv`，否則中止）
不建樹、不呼叫 LLM。
輸出：`_analysis_outputs_robustness/tail_cut_sensitivity.csv`
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import contracts as C
from . import enb_null as EN
from . import freeze, paths
from . import stage1_marks as M
from . import stage3_hrp as S3

OUT = paths.ROOT / "_analysis_outputs_robustness"
EFFN = (0.05, 0.10, 0.15)
HIGH = (0.90, 0.95, 0.99)          # 輪動分數與空手比例共用
TREES = ("TW", "US", "XM")


def usable(df: pd.DataFrame, effn_pct: float, high_pct: float) -> pd.Series:
    """與 stage1_marks.build 的尾端硬篩同一寫法，只換百分位。"""
    g = df.groupby("market")
    effn_cut = g["effective_n"].transform(lambda s: s.quantile(effn_pct))
    rot_cut = g["rotation_score"].transform(lambda s: s.quantile(high_pct))
    empty_cut = g["empty_ratio"].transform(lambda s: s.quantile(high_pct))
    blade1 = (df["effective_n"] <= effn_cut) & (df["rotation_score"] >= rot_cut)
    blade2 = df["empty_ratio"] >= empty_cut.clip(lower=1e-6)
    return pd.Series(~(blade1 | blade2).to_numpy(), index=df[C.PK])


def main(argv=None) -> int:
    for d in (paths.STAGE0, paths.STAGE1, paths.STAGE1 / "_marks"):
        freeze.verify_inputs(d)
    idx = pd.read_parquet(paths.STAGE0 / "candidate_index.parquet")
    scan = pd.read_parquet(paths.STAGE1 / "strategy_scan.parquet")
    df = idx.merge(scan.drop(columns=["market"]), on=C.PK, how="inner", validate="one_to_one")
    assert len(df) == len(idx)
    df["market"] = df["market"].astype(str)
    marks = pd.read_parquet(paths.STAGE1 / "strategy_marks.parquet").set_index(C.PK)["is_usable"]
    cur = usable(df, M.EFFN_LOW_PCT, M.ROTATION_HIGH_PCT)
    assert M.ROTATION_HIGH_PCT == M.EMPTY_HIGH_PCT
    assert cur.equals(marks.reindex(cur.index).astype(bool)), "現行門檻重算的 is_usable ≠ strategy_marks"
    print(f"✓ 現行門檻重算的 is_usable＝strategy_marks（{int(cur.sum()):,}/{len(cur):,}）")

    months_long = pd.read_parquet(paths.STAGE1 / "returns_monthly.parquet")
    meta = pd.read_parquet(paths.STAGE1 / "returns_meta.parquet")
    enb_ref = pd.read_csv(OUT / "effective_number_of_bets.csv").set_index("tree_key")["enb_raw"]
    mk = df.set_index(C.PK)["market"]
    wides = {}
    for t in TREES:
        ws, we = C.HRP_WINDOWS[t]
        uids = S3._tree_universe(t, ws, meta)          # 不先做 usable 過濾，之後按各組取子集
        w = months_long[months_long.strategy_uid.isin(set(uids))]
        w = w[(w.month >= pd.Period(ws, "M")) & (w.month <= pd.Period(we, "M"))]
        wide = w.pivot(index="strategy_uid", columns="month", values="ret")
        bad = wide.isna().any(axis=1)
        print(f"[{t}] 共同窗 {ws}~{we}｜候選 {len(wide):,}｜窗內有缺值 {int(bad.sum())}（排除）")
        wides[t] = wide.loc[~bad]

    rows = []
    for e in EFFN:
        for h in HIGH:
            u = usable(df, e, h)
            row = {"有效股票數門檻": f"P{e*100:.0f}", "輪動與空手門檻": f"P{h*100:.0f}",
                   "現行": (e == M.EFFN_LOW_PCT and h == M.ROTATION_HIGH_PCT)}
            for m in ("TW", "US"):
                s = mk[mk == m].index
                row[f"{m}_淘汰比例"] = float(1 - u.reindex(s).mean())
                a, b = set(s[u.reindex(s).to_numpy()]), set(s[cur.reindex(s).to_numpy()])
                row[f"{m}_Jaccard"] = len(a & b) / len(a | b)
                row[f"{m}_現行可用仍保留"] = len(a & b) / len(b)
                row[f"{m}_新名單屬現行可用"] = len(a & b) / len(a)
            for t in TREES:
                wide = wides[t]
                sub = wide.loc[u.reindex(wide.index).to_numpy()]
                z = EN._standardise(sub.to_numpy(dtype=np.float64))
                row[f"{t}_樹宇宙檔數"] = int(z.shape[0])
                row[f"{t}_ENB"] = EN.fast_enb(z)
                if row["現行"]:
                    assert np.isclose(row[f"{t}_ENB"], enb_ref[t], atol=1e-3), (t, row[f"{t}_ENB"], enb_ref[t])
            rows.append(row)
            print(f"  {row['有效股票數門檻']}×{row['輪動與空手門檻']}：TW 淘汰 {row['TW_淘汰比例']:.2%}｜"
                  f"US 淘汰 {row['US_淘汰比例']:.2%}｜ENB " +
                  "／".join(f"{row[f'{t}_ENB']:.2f}" for t in TREES), flush=True)
    print("✓ 現行組合的 ENB＝effective_number_of_bets.csv")
    res = pd.DataFrame(rows)
    p = OUT / "tail_cut_sensitivity.csv"
    res.to_csv(p, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "tail_cut_sensitivity", OUT / "_tail_cut_sensitivity_manifest",
        inputs=[paths.STAGE0 / "candidate_index.parquet", paths.STAGE1 / "strategy_scan.parquet",
                paths.STAGE1 / "strategy_marks.parquet", paths.STAGE1 / "returns_monthly.parquet",
                paths.STAGE1 / "returns_meta.parquet", OUT / "effective_number_of_bets.csv"],
        outputs=[p], params={"effn_pct": list(EFFN), "rotation_empty_pct": list(HIGH)},
        notes="尾端硬篩百分位敏感度；只在記憶體重算 is_usable，不覆寫 _frozen/stage1；ENB 用 fast_enb。")
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(index=False))
    print("→", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
