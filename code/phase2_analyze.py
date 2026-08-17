# -*- coding: utf-8 -*-
"""
Phase 2 分析：套用 primary/secondary 不對稱門檻，產出「體質檢查表」。

老師的規則（2026-08-05 meeting）：
  F1(primary)   一定要是「有意義的」——自己單獨跑就要夠強
  F2(secondary) 「至少跟大盤差不多，或者說是輸大盤不多」——不拖累即可
  晉升條件：「primary 本身就已經很強了，或 primary 加上某一點點比較寬鬆的
            secondary 夠強了，這時候我就可以把它晉升」
  F2 可以是空集合

因為 F1∩F2 與 F2∩F1 是同一個策略，這裡在**分析層**指派角色：
  一個配對 (A,B)，若 A 在 PRIMARY 池且 A 的單因子夠強 → 視為 (primary=A, secondary=B)
  若兩個都符合，取單因子 CAGR 較高者當 primary（較保守，避免用弱的當主訊號）

另外實作老師另一個獨立約束：
  「裡面被選出來的[股票]不能太少，那就是策略不[穩定]」
  → MIN_HOLDINGS（平均每月持股數）在**分析層**過濾，不寫進回測層
    （回測層過濾的話，之後想調門檻就得重跑）

只讀既有結果，不重跑回測、不改 daily_sharpe。

用法：python phase2_analyze.py [--market TW]
"""
import re
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase2_pairing import N_BUCKETS, IN_SAMPLE_END   # noqa: E402
from universe_benchmark import get_bench               # noqa: E402
import phase_variants                                  # noqa: E402

ART = HERE / "results_artifacts"
OUT = HERE.parent / "_analysis_outputs_phase2"


# ==================== 門檻（可調，改動請同步更新報告） ====================
MIN_HOLDINGS = 10        # 老師：選出的股票不能太少 → 平均每月持股數下限
COMBO_DEGRADE_TOL = 0.005  # 配對後相對 primary 單獨最多退步 0.5 個百分點
# PRIMARY_MARGIN / SECONDARY_TOL 改由 phase_variants 提供（各變體可不同）

OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "axes.grid": True,
    "grid.color": "#DDDDDD", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def log(m):
    print(m, flush=True)


def parse_name(s):
    """'PB_qb0of3__ROE_qb2of3__None__v0' → (PB, 0, ROE, 2)；單因子時 F2 為 None。"""
    parts = s.split("__")
    def one(x):
        m = re.match(r"(.+)_qb(\d+)of(\d+)$", x)
        return (m.group(1), int(m.group(2))) if m else (None, np.nan)
    f1, k1 = one(parts[0])
    f2, k2 = one(parts[1]) if len(parts) > 1 else (None, np.nan)
    return f1, k1, f2, k2


def holdings_stats(label, strategy):
    """持股數的三個指標。回傳 (平均持股數, 持股月份覆蓋率, 持股數p10)。

    ⚠️ 2026-08-16 第三方審查指出既有 `平均持股數` 有兩個缺陷，故補兩個欄位：

    1. **排除空手月份**：`nz = nz[nz > 0]` 讓「前十年完全空手、後十六年正常」的策略
       平均持股看起來完全正常。而「空手期間不虧錢」正是 Phase 1 §7 診斷出的
       「什麼都贏大盤」的成因機制——同一個機制在 MIN_HOLDINGS 這關仍沒被擋。
       → 加 `持股月份覆蓋率`。
    2. **用平均而非低分位**：平均 12 檔可能是「一半月份 22 檔、一半月份 2 檔」。
       老師的原話是「被選出來的股票不能太少，那就是策略**不穩定**」——講的正是穩定性，
       而平均恰好把不穩定藏起來。
       → 加 `持股數p10`（有持股月份的第 10 百分位）。

    **目前只輸出欄位、不改門檻**（門檻仍是 `平均持股數 >= MIN_HOLDINGS`），
    先看分布再決定，避免一次改動兩件事、讓結果無法歸因。
    """
    p = ART / label / strategy / "position.parquet"
    if not p.exists():
        return np.nan, np.nan, np.nan
    try:
        pos = pd.read_parquet(p)
        nz = (pos != 0).sum(axis=1)
        n_total = len(nz)
        held = nz[nz > 0]
        if not len(held):
            return 0.0, 0.0, 0.0
        return (float(held.mean()),
                float(len(held) / n_total) if n_total else np.nan,
                float(held.quantile(0.10)))
    except Exception:
        return np.nan, np.nan, np.nan


def avg_holdings(label, strategy):
    """僅平均持股數（維持既有介面與數值，供 phase3/phase4_analyze 沿用）。"""
    return holdings_stats(label, strategy)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="TW", choices=["TW", "US"])
    ap.add_argument("--variant", default="strict", choices=list(phase_variants.VARIANTS))
    ap.add_argument("--bench", default="universe", choices=["universe", "index"],
                    help="universe＝自建宇宙基準(含股利，預設)；index＝外部價格指數(不含股利)")
    args = ap.parse_args()
    mkt = args.market
    # ⚠️ 必須帶 market：phase_variants.get() 的 market 預設是 "TW"，漏傳會拿台股的
    #    Phase 1 判定去篩美股（2026-08-12 code review 抓到，美股 openSec 曾因此用錯
    #    primary 池——台股過關 5 個 vs 美股過關 8 個，且台股的 MOM 在美股只是「只取極端桶」）。
    V = phase_variants.get(args.variant, mkt)
    PRIMARY, SECONDARY = V["primary"], V["secondary"]
    PRIMARY_MARGIN, SECONDARY_TOL = V["primary_margin"], V["secondary_tol"]
    # 因子池相同的變體共用同一份 Phase 2 回測（差別只在分析層的角色指派）：
    #   strict / relaxed  → 12 因子 → TW_L2_M（630 個）
    #   all    / openSec  → 19 因子 → TW_L2_all_M（1,596 個）
    # 對應關係集中在 phase_variants._POOL_SHARE，避免再出現「讀錯回測」的靜默錯誤。
    label = phase_variants.l2_label(mkt, args.variant)
    sfx = f"_{args.variant}" if args.variant != "strict" else ""
    sfx += "_idxbench" if args.bench == "index" else ""
    OUT.mkdir(parents=True, exist_ok=True)

    bench, bdesc = get_bench(mkt, args.bench)
    log(f"變體＝{args.variant}：{V['desc']}")
    log(f"  primary  ({len(PRIMARY)})：{PRIMARY}")
    log(f"  secondary({len(SECONDARY)})：{SECONDARY}")
    log(f"基準＝{bench:.2%}（{bdesc}）")
    log(f"門檻：primary 需 > 基準+{PRIMARY_MARGIN:.0%}｜secondary 需 > 基準−{SECONDARY_TOL:.0%}｜"
        f"配對退步容忍 {COMBO_DEGRADE_TOL:.1%}｜平均持股數 ≥ {MIN_HOLDINGS}\n")

    df = pd.read_parquet(ART / label / "stats.parquet")
    df[["F1", "k1", "F2", "k2"]] = df["strategy"].apply(lambda s: pd.Series(parse_name(s)))
    df["is_pair"] = df["F2"].notna()
    log(f"讀入 {len(df)} 個策略（單因子 {int((~df['is_pair']).sum())}／配對 {int(df['is_pair'].sum())}）")

    # ---------- 平均持股數（老師的第二個約束） ----------
    log("計算持股數三項指標（平均／覆蓋率／p10）…")
    hs = [holdings_stats(label, s) for s in df["strategy"]]
    df["平均持股數"] = [h[0] for h in hs]
    df["持股月份覆蓋率"] = [h[1] for h in hs]      # 有持股的月份 ÷ 總月份
    df["持股數p10"] = [h[2] for h in hs]           # 有持股月份的第 10 百分位
    # ⚠️ 門檻維持只看平均（見 holdings_stats 說明）：先產出分布再決定要不要收緊
    df["持股數足夠"] = df["平均持股數"] >= MIN_HOLDINGS

    # ---------- 單因子基準表：每個 (因子,桶) 自己有多強 ----------
    solo = df[~df["is_pair"]].set_index(["F1", "k1"])["CAGR"].to_dict()
    solo_tbl = df[~df["is_pair"]][["F1", "k1", "CAGR", "max_drawdown", "win_ratio",
                                    "平均持股數"]].copy()
    solo_tbl["角色池"] = solo_tbl["F1"].map(
        lambda f: "primary" if f in PRIMARY else ("secondary" if f in SECONDARY else "?"))
    solo_tbl["贏大盤"] = solo_tbl["CAGR"] - bench
    solo_tbl["可當primary"] = (solo_tbl["F1"].isin(PRIMARY)) & (solo_tbl["贏大盤"] >= PRIMARY_MARGIN)
    solo_tbl["可當secondary"] = solo_tbl["贏大盤"] >= -SECONDARY_TOL
    solo_tbl = solo_tbl.sort_values("CAGR", ascending=False)
    solo_tbl.to_csv(OUT / f"{mkt}_L2{sfx}_solo_buckets.csv", index=False, encoding="utf-8-sig")

    ok_pri = solo_tbl[solo_tbl["可當primary"]]
    log(f"\n可當 primary 的 (因子,桶)：{len(ok_pri)} 個")
    log(ok_pri[["F1", "k1", "CAGR", "贏大盤", "平均持股數"]].to_string(index=False))

    # ---------- 角色指派 + 晉升判定 ----------
    pri_set = set(zip(ok_pri["F1"], ok_pri["k1"]))
    sec_ok = set(zip(solo_tbl.loc[solo_tbl["可當secondary"], "F1"],
                     solo_tbl.loc[solo_tbl["可當secondary"], "k1"]))

    rows = []
    for _, r in df.iterrows():
        a, ka, b, kb = r["F1"], r["k1"], r["F2"], r["k2"]
        if not r["is_pair"]:
            # F2 = 空集合：primary 自己夠強就晉升
            passed = (a, ka) in pri_set
            rows.append({**r.to_dict(), "primary": f"{a}_qb{int(ka)}", "secondary": "（空集合）",
                         "primary_solo_CAGR": solo.get((a, ka), np.nan),
                         "配對增益": 0.0,
                         "晉升": bool(passed and r["持股數足夠"]),
                         "未過原因": "" if passed else "primary 單獨不夠強"})
            continue
        # 配對：挑符合 primary 資格且單因子較強的那個當 primary
        cands = [(x, kx) for x, kx in ((a, ka), (b, kb)) if (x, kx) in pri_set]
        if not cands:
            rows.append({**r.to_dict(), "primary": "—", "secondary": "—",
                         "primary_solo_CAGR": np.nan, "配對增益": np.nan,
                         "晉升": False, "未過原因": "兩邊都不具 primary 資格"})
            continue
        p = max(cands, key=lambda t: solo.get(t, -9))
        s = (b, kb) if p == (a, ka) else (a, ka)
        p_solo = solo.get(p, np.nan)
        gain = r["CAGR"] - p_solo
        # 退步檢定要跟「兩腳裡最好的單獨表現」比，不是只跟被指派為 primary 的那一腳比。
        # 否則會有漏洞：若某一腳不在 primary 池（故不能當 primary）但單獨表現更強，
        # 配對後就算輸給它，也會因為只跟較弱的 primary 比而過關。
        # （台股實測 0 例，但美股因子強弱分布不同、可能咬到，故一律用嚴格版。）
        best_solo = max(solo.get((a, ka), -9), solo.get((b, kb), -9))
        gain_vs_best = r["CAGR"] - best_solo
        reasons = []
        if s not in sec_ok:
            reasons.append("secondary 輸大盤太多")
        if gain_vs_best < -COMBO_DEGRADE_TOL:
            reasons.append("配對後退步過多")
        if not r["持股數足夠"]:
            reasons.append(f"平均持股數<{MIN_HOLDINGS}")
        rows.append({**r.to_dict(),
                     "primary": f"{p[0]}_qb{int(p[1])}", "secondary": f"{s[0]}_qb{int(s[1])}",
                     "primary_solo_CAGR": p_solo, "配對增益": gain,
                     "最佳單腳CAGR": best_solo, "相對最佳單腳增益": gain_vs_best,
                     "晉升": not reasons, "未過原因": "；".join(reasons)})

    res = pd.DataFrame(rows)
    cols = ["strategy", "primary", "secondary", "CAGR", "primary_solo_CAGR", "配對增益",
            "max_drawdown", "win_ratio", "平均持股數", "持股月份覆蓋率", "持股數p10",
            "晉升", "未過原因"]
    res[cols].sort_values(["晉升", "CAGR"], ascending=[False, False]).to_csv(
        OUT / f"{mkt}_L2{sfx}_all_combos.csv", index=False, encoding="utf-8-sig")

    passed = res[res["晉升"]].sort_values("CAGR", ascending=False)
    passed[cols].to_csv(OUT / f"{mkt}_L2{sfx}_體質檢查表.csv", index=False, encoding="utf-8-sig")

    log(f"\n===== 晉升結果 =====")
    log(f"  總策略 {len(res)}｜晉升 {len(passed)}｜未晉升 {len(res)-len(passed)}")
    log("\n未晉升原因分布：")
    log(res.loc[~res["晉升"], "未過原因"].value_counts().to_string())

    log(f"\n===== 體質檢查表 Top 20（依 CAGR）=====")
    log(passed[["primary", "secondary", "CAGR", "配對增益", "max_drawdown",
                "平均持股數"]].head(20).to_string(index=False))

    # ---------- 圖：配對增益分布（F2 到底有沒有幫助？） ----------
    pair = res[res["is_pair"] & res["配對增益"].notna()]
    if len(pair):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        g = pair["配對增益"].dropna()
        axes[0].hist(g, bins=50, color=OI[0], edgecolor="white", linewidth=0.3)
        axes[0].axvline(0, color=OI[1], linestyle="--", linewidth=1.4,
                        label=f"0（中位數 {g.median():+.2%}）")
        axes[0].legend(frameon=False)
        axes[0].set_title("加上 F2 相對 primary 單獨的 CAGR 增益")
        axes[0].set_xlabel("配對增益")

        # ⚠️ 不能用 split("_")[0]：因子名本身就含底線（NETDEBT_EBITDA / EV_S / P_IC），
        #    會被截成 NETDEBT / EV / P。改成剝掉尾端的 _qbK 才是正確的因子名。
        sec_factor = pair["secondary"].str.replace(r"_qb\d+$", "", regex=True)
        order = pair.groupby(sec_factor)["配對增益"].median().sort_values(ascending=False)
        data = [pair.loc[sec_factor == f, "配對增益"].values for f in order.index]
        bp = axes[1].boxplot(data, labels=list(order.index), showmeans=True, patch_artist=True,
                             medianprops=dict(color="black", linewidth=1.3))
        for box in bp["boxes"]:
            box.set(facecolor=OI[2], alpha=0.35)
        axes[1].axhline(0, color=OI[1], linestyle="--", linewidth=1.2)
        axes[1].tick_params(axis="x", labelrotation=60, labelsize=8)
        axes[1].set_title("各 secondary 因子的配對增益（依中位數排序）")
        fig.suptitle(f"Phase 2  F2 有沒有幫助？— {mkt}／{args.variant}（in-sample 至 {IN_SAMPLE_END}，"
                     f"{N_BUCKETS} 桶、C/V 關閉）", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(OUT / f"{mkt}_L2{sfx}_pairing_gain.png", bbox_inches="tight")
        plt.close(fig)
        log(f"\n配對增益：中位數 {g.median():+.2%}｜為正的比例 {float((g > 0).mean()):.1%}")
        log("\n各 secondary 因子的配對增益中位數：")
        log(order.map(lambda v: f"{v:+.2%}").to_string())

    log(f"\n輸出於 {OUT}")


if __name__ == "__main__":
    main()
