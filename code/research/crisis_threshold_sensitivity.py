# -*- coding: utf-8 -*-
"""RQ4 的危機門檻敏感度（2026-11-15，使用者授權，經寫作 session 轉達）：crisis_thresh 30% → 25%／35%。

做法：
  - 不重算大盤峰谷。峰谷只由 bear_thresh／bull_thresh（15%）決定，這兩個不變，
    所以直接拿凍結的 `_frozen/stage2/regime/regime_table_{TW,US}.parquet` 每段的
    pct_change、ann_speed，依 stage2a 的 classify_segments 規則重新貼標：
      下跌且跌幅 ≥ 門檻 → 危機；其他下跌 → 熊；上漲 → 牛；非危機且年化速度 < 15% → 盤整。
    先用 30% 重貼，必須和凍結表逐段相同（對帳）。
  - 不覆寫 `_frozen/stage2/regime/`：把重貼的表在記憶體裡換給 `crisis_diversification._segments`，
    再呼叫同一支 `run_tree`（同 SEED、同 N_NULL／N_BOOT）。
  - D2（主定義）取的是 2020、2022 的「熊或危機」段：門檻只在熊↔危機之間移動，但 35% 時
    原本的危機段若年化速度 < 15% 會被改成盤整而掉出 D2——所以逐門檻檢查 D2 月份是否不變。
    另外「平時」＝扣掉 D1∪D2 的月份，D1 變了平時月份池就會變，D2 的「同長度對照百分位」可能小幅移動。
輸出：`_analysis_outputs_robustness/crisis_threshold_sensitivity.csv`（指標，三個門檻並排）、
      `crisis_threshold_sensitivity_segments.csv`（各門檻下標成危機的段）
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import crisis_diversification as CD
from . import freeze, paths
from . import stage2a_regime as R

OUT = paths.ROOT / "_analysis_outputs_robustness"
THRESHOLDS = (0.25, 0.30, 0.35)
SPEED = R.DEFAULT_PARAMS["consolidation_speed"]
_FROZEN = {m: pd.read_parquet(paths.STAGE2 / "regime" / f"regime_table_{m}.parquet") for m in ("TW", "US")}


def relabel(s: pd.DataFrame, th: float) -> pd.DataFrame:
    s = s.copy()
    chg = s["pct_change"].to_numpy()
    lab = np.where(chg < 0, np.where(-chg >= th, "危機", "熊"), "牛")
    lab = np.where((lab != "危機") & (s.ann_speed.to_numpy() < SPEED), "盤整", lab)
    s["label"] = lab
    return s


def main(argv=None) -> int:
    freeze.verify_inputs(paths.STAGE2 / "regime")
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE1 / "_marks")
    freeze.verify_inputs(paths.STAGE3)
    for m, s in _FROZEN.items():
        r = relabel(s, R.DEFAULT_PARAMS["crisis_thresh"])
        assert (r.label.to_numpy() == s.label.to_numpy()).all(), f"{m} 30% 重貼與凍結表不符"
    print("✓ 30% 重貼與凍結的大盤狀態表逐段相同")

    mem = pd.read_parquet(OUT / "walkforward_members.parquet")
    for c in ("tree_key", "group"):
        mem[c] = mem[c].astype(str)
    rows, segs = [], []
    base_cm = {t: CD.crisis_months(t) for t in CD.TREES}
    for th in THRESHOLDS:
        tabs = {m: relabel(s, th) for m, s in _FROZEN.items()}
        CD._segments = lambda m, _t=tabs: _t[m]
        for m, s in tabs.items():
            for r in s[s.label == "危機"].itertuples():
                segs.append({"門檻": th, "市場": m, "起": r.start.date(), "訖": r.end.date(),
                             "跌幅": r[s.columns.get_loc("pct_change") + 1], "年化速度": r.ann_speed})
        for t in CD.TREES:
            cm = CD.crisis_months(t)
            d2_same = cm["D2"] == base_cm[t]["D2"]
            r_, _ = CD.run_tree(t, mem)
            for row in r_:
                row.update({"危機門檻": th, "D2月份與30%相同": d2_same, "D1月數": len(cm["D1"]),
                            "D1新增月份(相對30%)": len(cm["D1"] - base_cm[t]["D1"]),
                            "D1減少月份(相對30%)": len(base_cm[t]["D1"] - cm["D1"])})
            rows += r_
            print(f"門檻 {th:.0%} {t} 完成｜D1 {len(cm['D1'])} 月｜D2 與 30% 相同：{d2_same}", flush=True)
    df = pd.DataFrame(rows)
    # 30% 這一組要與現行 crisis_diversification.csv 相同（同 SEED）
    cur = pd.read_csv(OUT / "crisis_diversification.csv")
    mine = df[df.危機門檻 == 0.30]
    for col in ("群間平均相關", "策略層ENB", "群間平均相關_落在平時分布百分位", "策略層ENB_落在平時分布百分位"):
        a = cur.set_index(["市場", "期間"])[col]
        b = mine.set_index(["市場", "期間"])[col].reindex(a.index)
        assert np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True, atol=1e-12), col
    print("✓ 30% 結果與現行 crisis_diversification.csv 相同")
    p1, p2 = OUT / "crisis_threshold_sensitivity.csv", OUT / "crisis_threshold_sensitivity_segments.csv"
    df.to_csv(p1, index=False, encoding="utf-8-sig")
    pd.DataFrame(segs).to_csv(p2, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "crisis_threshold_sensitivity", OUT / "_crisis_threshold_sensitivity_manifest",
        inputs=[paths.STAGE2 / "regime" / "regime_table_TW.parquet", paths.STAGE2 / "regime" / "regime_table_US.parquet",
                paths.STAGE1 / "returns_monthly.parquet", paths.STAGE3 / "cluster_assign.parquet",
                OUT / "walkforward_members.parquet", OUT / "crisis_diversification.csv"],
        outputs=[p1, p2],
        params={"thresholds": list(THRESHOLDS), "consolidation_speed": SPEED, "SEED": CD.SEED,
                "N_NULL": CD.N_NULL, "N_BOOT": CD.N_BOOT},
        notes="只換 crisis_thresh；峰谷沿用凍結表（bear/bull 門檻不變），不覆寫 _frozen/stage2/regime/。")
    print(p1, p2, sep="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
