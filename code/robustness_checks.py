# -*- coding: utf-8 -*-
"""
三項純分析層的穩健性檢定（不需重跑回測）。回應 2026-08-16 第三方審查的 C1 / D1 / B3。

C1  PRIMARY_MARGIN 敏感度
    台股 10 個「可當 primary」的桶裡有 6 個的超額擠在 +2.08 ~ +3.13pp 的窄帶，
    只有 4 個估值 qb0 桶是明顯的（+7.09 ~ +9.51pp）。ROE_qb2 更是靠 +2.29pp
    擦邊過 2pp 門檻。若結論對門檻很敏感就必須在論文揭露；若不敏感則是穩健性證據。
    ⚠️ 只能往**嚴格**方向試（≥ 現行 2%），因為放寬會產生 Phase 3 沒跑過的組合。

D1  9 桶算數合併 vs 3 桶實跑
    指導教授認為「9 拼回 3 不用重算，算數湊回來而已」。SOP §3.2 判斷這對統計量成立、
    對回測不成立（選股清單變了、持股數與交易成本都不同），並承諾做一份對照。
    這裡用既有結果算：把 9 桶的 {0,1,2}/{3,4,5}/{6,7,8} 各取算數平均，
    與 Phase 2 實跑的 3 桶 CAGR 逐因子比較。

B3  最終候選池的規模集中度
    量「Top N 含最小規模桶的比例」相對「全池比例」的倍數，台美並列。

用法：python robustness_checks.py [--market TW]
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

OUT = ROOT / "_analysis_outputs_robustness"
P1D, P2D, P4D = (ROOT / f"_analysis_outputs_phase{i}" for i in (1, 2, 4))


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------- C1
def primary_margin_sensitivity(mkt, bench, variant="openSec"):
    """不同 PRIMARY_MARGIN 下的 primary 桶數／晉升組合數／最終候選池品質。

    做法：用既有的 Phase 2 組合表重新套門檻，再拿新的白名單去篩既有的最終候選池。
    因為提高門檻只會讓白名單變小（是現行 2% 的子集），所以完全用得上既有回測。
    """
    sfx = "" if variant == "strict" else f"_{variant}"
    solo = pd.read_csv(P2D / f"{mkt}_L2{sfx}_solo_buckets.csv", encoding="utf-8-sig")
    comb = pd.read_csv(P2D / f"{mkt}_L2{sfx}_all_combos.csv", encoding="utf-8-sig")
    fin = pd.read_csv(P4D / f"{mkt}_L4{sfx}_final_candidates.csv", encoding="utf-8-sig")
    fin["F組合"] = fin["F組合"].astype(str)

    rows = []
    for margin in [0.02, 0.025, 0.03, 0.04, 0.05]:
        ok = solo[(solo["角色池"] == "primary") & (solo["贏大盤"] >= margin)]
        pri = set(zip(ok["F1"], ok["k1"]))
        # 該門檻下仍具 primary 資格的組合：primary 欄位還在 pri 集合裡
        def keep(p):
            if not isinstance(p, str) or "_qb" not in p:
                return False
            f, k = p.rsplit("_qb", 1)
            return (f, int(k)) in pri
        sub = comb[comb["晉升"] & comb["primary"].map(keep)]
        wl = {"__".join(s.split("__")[:2]) for s in sub["strategy"]}
        cand = fin[fin["F組合"].isin(wl)]
        rows.append({
            "PRIMARY_MARGIN": margin,
            "可當primary的桶": len(ok),
            "晉升組合": len(sub),
            "最終候選": len(cand),
            "CAGR中位": cand["CAGR"].median() if len(cand) else np.nan,
            "CAGR_p90": cand["CAGR"].quantile(.9) if len(cand) else np.nan,
            "CAGR最高": cand["CAGR"].max() if len(cand) else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- D1
def nine_vs_three(mkt, variant="openSec"):
    """9 桶算數平均 vs 3 桶實跑的逐因子對照。"""
    p1 = pd.read_csv(P1D / f"{mkt}_phase1_linearity.csv", encoding="utf-8-sig")
    sfx = "" if variant == "strict" else f"_{variant}"
    solo = pd.read_csv(P2D / f"{mkt}_L2{sfx}_solo_buckets.csv", encoding="utf-8-sig")
    actual = {(r.F1, int(r.k1)): r.CAGR for r in solo.itertuples()}

    rows = []
    for r in p1.itertuples():
        if r.判定 == "🚫 前瞻偏誤":
            continue
        nine = [getattr(r, f"桶{i}") for i in range(9)]
        for k, idx in enumerate([(0, 1, 2), (3, 4, 5), (6, 7, 8)]):
            merged = float(np.mean([nine[i] for i in idx]))
            act = actual.get((r.因子, k))
            if act is None:
                continue
            rows.append({"因子": r.因子, "桶": f"qb{k}of3",
                         "9桶算數合併": merged, "3桶實跑": act,
                         "差異": act - merged})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- B3
def size_concentration(mkt, size_bucket="REVENUE_qb0", variant="openSec"):
    sfx = "" if variant == "strict" else f"_{variant}"
    fin = pd.read_csv(P4D / f"{mkt}_L4{sfx}_final_candidates.csv",
                      encoding="utf-8-sig").sort_values("CAGR", ascending=False)
    has = fin["F組合"].astype(str).str.contains(size_bucket)
    base = float(has.mean())
    rows = []
    for n in [10, 50, 100, 500]:
        top = has.head(n)
        rows.append({"範圍": f"Top {n}", "含最小規模桶": int(top.sum()),
                     "佔比": float(top.mean()),
                     "相對全池倍數": float(top.mean() / base) if base else np.nan})
    rows.append({"範圍": f"全池 {len(fin):,}", "含最小規模桶": int(has.sum()),
                 "佔比": base, "相對全池倍數": 1.0})
    return pd.DataFrame(rows), fin.head(50)["F組合"].value_counts().head(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="TW", choices=["TW", "US"])
    ap.add_argument("--variant", default="openSec")
    args = ap.parse_args()
    mkt = args.market
    OUT.mkdir(parents=True, exist_ok=True)

    from universe_benchmark import get_bench
    bench, bdesc = get_bench(mkt, "universe")
    log(f"=== 穩健性檢定｜{mkt}／{args.variant}｜基準 {bench:.2%} ===\n")

    log("===== C1  PRIMARY_MARGIN 敏感度 =====")
    c1 = primary_margin_sensitivity(mkt, bench, args.variant)
    show = c1.copy()
    show["PRIMARY_MARGIN"] = show["PRIMARY_MARGIN"].map("{:.1%}".format)
    for c in ["CAGR中位", "CAGR_p90", "CAGR最高"]:
        show[c] = show[c].map(lambda v: "—" if pd.isna(v) else f"{v:.2%}")
    log(show.to_string(index=False))
    c1.to_csv(OUT / f"{mkt}_primary_margin_sensitivity.csv", index=False, encoding="utf-8-sig")

    log("\n===== D1  9 桶算數合併 vs 3 桶實跑 =====")
    d1 = nine_vs_three(mkt, args.variant)
    log(f"  可比對 {len(d1)} 組（因子 × 3 桶）")
    log(f"  差異（實跑 − 算數）：中位 {d1['差異'].median():+.2%}｜"
        f"平均 {d1['差異'].mean():+.2%}｜標準差 {d1['差異'].std():.2%}")
    log(f"  絕對差異：中位 {d1['差異'].abs().median():.2%}｜最大 {d1['差異'].abs().max():.2%}")
    log(f"  實跑較高的比例：{float((d1['差異'] > 0).mean()):.1%}")
    log("\n  差異最大的 6 組：")
    t = d1.reindex(d1["差異"].abs().sort_values(ascending=False).index).head(6).copy()
    for c in ["9桶算數合併", "3桶實跑", "差異"]:
        t[c] = t[c].map("{:+.2%}".format)
    log(t.to_string(index=False))
    d1.to_csv(OUT / f"{mkt}_9bucket_vs_3bucket.csv", index=False, encoding="utf-8-sig")

    log("\n===== B3  最終候選池的規模集中度 =====")
    b3, topf = size_concentration(mkt, variant=args.variant)
    s = b3.copy()
    s["佔比"] = s["佔比"].map("{:.1%}".format)
    s["相對全池倍數"] = s["相對全池倍數"].map("{:.1f}x".format)
    log(s.to_string(index=False))
    log("\n  Top 50 最常出現的 F 組合：")
    log(topf.to_string())
    b3.to_csv(OUT / f"{mkt}_size_concentration.csv", index=False, encoding="utf-8-sig")

    log(f"\n輸出於 {OUT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
