# -*- coding: utf-8 -*-
"""步驟1 延伸：優等生規則「只在警示觸發的季才套用」（2026-10-29 使用者授權）。

🔶 這是看過步驟1、步驟2 結果後才做的延伸，**不是事先寫死的**，報告須註明。
只算 window4（2024–25）：逐季警示狀態只有評估期有（集中度用凍結門檻、三訊號用 `step2_signals`），
window1~3 沒有，無法照步驟1 晉級規則重新挑 ⇒ 只能當 2024–25 的描述，不能當「用歷史挑出來」的證據。

三種套用時機（沒觸發的季＝control0）：
  always：每季都套用（＝步驟1）
  m1d   ：集中度警示「觸發」的季才套用
  trig3 ：集中度觸發、熱門股覆蓋觸發、突破廣度「高」任一成立才套用
結算用 `rule_selector.WeightBook`（實際換手成本；起點＝window3 末 control0 持股漂移到評估期開始）。
驗證：always 的未扣成本累積須與步驟1 `metrics_all_*` 的 OOS_gross_cum 逐組相同。
輸出：`elite_grid_1027/trigger_only_{tag}.csv`、`trigger_only_summary.txt`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app import rule_selector as RS  # noqa: E402
from app._step2_phase2_settle import daily_metrics  # noqa: E402

GROUPS = [("TW", "exclude_v1"), ("TW", "baseline"), ("US", "exclude_v1"), ("US", "baseline")]
MODES = ("always", "m1d", "trig3")
LABEL = {"always": "每季都套用", "m1d": "集中度觸發才套用", "trig3": "三訊號任一觸發才套用"}


def run_group(market: str, pool: str) -> tuple[pd.DataFrame, dict]:
    from fcv_core import MarketData
    from app import step2_signals
    from app._elite_grid_offline import load_benchmark
    tag = RS.TAG[(market, pool)]
    close = MarketData(market).data.get("price:close")
    wb = RS.WeightBook(market, pool, close, with_sig=True)
    states = RS.m1d_states(market, pool)
    sig = step2_signals.compute(market, pool)
    sig.index = sig.index.astype(str)
    trig = {"always": [True] * 8,
            "m1d": [states[a] == "TRIGGERED" for a in wb.as_ofs],
            "trig3": [bool(sig.loc[str(a)].m1d_state == "TRIGGERED" or sig.loc[str(a)].hot_state == "TRIGGERED"
                           or sig.loc[str(a)].breakout_state == "高") for a in wb.as_ofs]}
    ref = pd.concat([pd.read_csv(RS.GRID / f"metrics_all_{tag}.csv"), pd.read_csv(RS.GRID / f"metrics_all_{tag}_sig.csv")])
    ref = ref.drop_duplicates("combo").set_index("combo")
    promoted = set(pd.read_csv(RS.GRID / f"promoted_{tag}.csv").combo) | set(pd.read_csv(RS.GRID / f"promoted_{tag}_sig.csv").combo)
    combos = sorted(c for c in wb.known if not c.startswith("REF_"))
    c0 = wb.settle(["REF_control0"] * 8)
    c0_sh, c0_mdd = daily_metrics(wb.daily(["REF_control0"] * 8).net)
    b = load_benchmark(market)
    d0 = b.index[b.index <= pd.Timestamp(wb.as_ofs[0])].max()
    bb = b.loc[d0:pd.Timestamp(wb.ends[-1])]
    rows, bad = [], []
    for i, c in enumerate(combos):
        for mode in MODES:
            keys = [c if t else "REF_control0" for t in trig[mode]]
            r = wb.settle(keys)
            d = wb.daily(keys)
            assert abs(d.net.iloc[-1] - 1 - r["cum_net"]) < 1e-9
            sh, mdd = daily_metrics(d.net)
            rows.append({"combo": c, "score": c.split("|")[0], "mode": mode, "cum_gross": r["cum_gross"],
                         "cum_net": r["cum_net"], "sharpe_net": sh, "mdd_net": mdd, "promoted": c in promoted})
            if mode == "always" and abs(r["cum_gross"] - ref.loc[c, "OOS_gross_cum"]) > 1e-9:
                bad.append((c, r["cum_gross"], ref.loc[c, "OOS_gross_cum"]))
        for key in [kk for kk in wb._cache if kk[1] == c]:   # 控制記憶體：用完就丟（control0 保留）
            del wb._cache[key]
        if i % 200 == 0:
            print(f"  {market}/{pool} {i}/{len(combos)}", flush=True)
    assert not bad, f"always 的未扣成本與步驟1 不一致：{bad[:5]}（共 {len(bad)} 組）"
    df = pd.DataFrame(rows)
    df.to_csv(RS.GRID / f"trigger_only_{tag}.csv", index=False, encoding="utf-8-sig")
    info = {"control0_net": c0["cum_net"], "control0_sharpe": c0_sh, "control0_mdd": c0_mdd,
            "bench": float(bb.iloc[-1] / bb.iloc[0] - 1), "n_combos": len(combos),
            "n_trig": {m: int(sum(trig[m])) for m in MODES},
            "trig_pattern": {m: "".join("1" if t else "0" for t in trig[m]) for m in MODES}}
    return df, info


def summarize(market, pool, df, info) -> list[str]:
    c0, bench = info["control0_net"], info["bench"]
    L = [f"=== {market}・{pool}（{info['n_combos']} 組規則；control0 扣成本 {c0:+.1%}、Sharpe {info['control0_sharpe']:.2f}、"
         f"MDD {info['control0_mdd']:.1%}；大盤 {bench:+.1%}）",
         "觸發季（第1~8季）：" + "；".join(f"{LABEL[m]} {info['trig_pattern'][m]}（{info['n_trig'][m]} 季）" for m in MODES)]
    w = df.pivot(index="combo", columns="mode", values="cum_net")
    for m in MODES:
        x = df[df["mode"] == m]
        L.append(f"  {LABEL[m]}：扣成本中位數 {x.cum_net.median():+.1%}、贏 control0 {(x.cum_net > c0).mean():.0%}、"
                 f"贏大盤 {(x.cum_net > bench).mean():.0%}、Sharpe 中位數 {x.sharpe_net.median():.2f}、MDD 中位數 {x.mdd_net.median():.1%}")
    for m in ("m1d", "trig3"):
        L.append(f"  {LABEL[m]} 比每季都套用高的組數比例：{(w[m] > w['always']).mean():.0%}（中位數差 {(w[m] - w['always']).median() * 100:+.1f} 個百分點）")
    fam = df.pivot_table(index="score", columns="mode", values="cum_net", aggfunc="median")[list(MODES)]
    L.append("  各排名依據的扣成本中位數：")
    L += ["    " + f"{s:<20}" + "  ".join(f"{LABEL[m]} {fam.loc[s, m]:+.1%}" for m in MODES) for s in fam.index]
    pr = df[df.promoted].pivot(index="combo", columns="mode", values="cum_net")[list(MODES)]
    L.append("  步驟1 晉級組（用 2015–23 挑的，含熱門股／突破股的晉級組）：")
    L += ["    " + f"{c:<36}" + "  ".join(f"{LABEL[m]} {pr.loc[c, m]:+.1%}" for m in MODES) for c in pr.index]
    return L


def main():
    out = ["步驟1 延伸：只在警示觸發的季才套用（2024–25；看過結果後才做的延伸，非事先寫死）", ""]
    for m, p in GROUPS:
        df, info = run_group(m, p)
        out += summarize(m, p, df, info) + [""]
        print("\n".join(out[-30:]), flush=True)
    (RS.GRID / "trigger_only_summary.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
