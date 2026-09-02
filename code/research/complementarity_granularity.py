# -*- coding: utf-8 -*-
"""H-25 · 互補性的分群粒度效應（2026-09-01）

**要回答的問題**：H-03 把 L1 群數從寫死的 8 改成輪廓係數決定（TW6/US7/XM3）之後，
39 對 L1 群對裡沒有任何一對達到「高互補」（相關 < `COMPLEMENTARITY_CUTS["高"]` = 0.5），
而舊 k=8 時代 XM_normal 有 5 對高互補。是策略真的沒有互補性、是門檻該改，
還是分群粒度的問題？

**結論：是粒度。** 群代表＝成員報酬的簡單平均，平均的成員愈多，個別策略的特異
成分互相抵消得愈徹底，代表序列就愈趨近該市場的大盤，跨市場相關自然被推高。
舊 k=8 的 5 對高互補全部來自同一個 305 檔的小群（全台股、99.7% 用 v1），
k=3 之後那 305 檔被併進 6,679 檔的全台股群裡，特異成分被稀釋掉了。

**門檻不該改**：0.5 這個門檻在 L3 層級把跨市場／同市場切得非常乾淨（見輸出），
不是門檻校錯。若為了讓 L1 出現高互補而把門檻調到 0.55，同市場的
TW_normal 群2×群5（0.539）也會被判成高互補，反而抹掉「分散來源是市場邊界」
這個核心結論——那是為了結論好看而調參數。

---------------------------------------------------------------------------
方法
---------------------------------------------------------------------------
同一批策略、同一段 DD-03 共同窗、同一套判定門檻，**只改變分群層級**：

  1. `stage3_hrp.rebuild_tree_returns(tree_id)` 取回建樹當初實際用的報酬矩陣
     （策略×月，無 NaN，已排除零變異數列）——單一事實來源，不自己另外 pivot
  2. 對 L1 與 L3 各自算群代表序列（成員列的簡單平均，同 `stage3_hrp.
     _cluster_meta_and_corr` 與 H-06 的口徑）
  3. 群代表兩兩相關，依市場組成標成 same／cross，依 `COMPLEMENTARITY_CUTS` 分級
  4. 用 `min_members` 掃描（1／5／20）做穩健性：小群估計較雜訊，但**同市場配對
     是天然的對照組**（群大小與月份數同量級），若雜訊是主因，同市場也該一起噴出
     大量假高互補——實測沒有，這就排除了雜訊解釋

⚠️ 只跑 normal 樹：crisis 樹的相關行為完全不同（同市場危機時可能大幅解相關、
跨市場危機時反而趨近 1），`COMPLEMENTARITY_CUTS` 明文註記未對 crisis 樹校準過。

用法：
    cd code
    python -m research.complementarity_granularity
"""
from __future__ import annotations

import argparse
import itertools
import sys

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, paths
from . import stage3_hrp as S3

TREES = ("TW_normal", "US_normal", "XM_normal")
LEVELS = ("L1", "L3")
#: 穩健性掃描：只納入成員數 >= 門檻的群。1 = 全納入。
MIN_MEMBERS_GRID = (1, 5, 20)


def _complementarity(corr: float) -> str:
    """跟 cluster_story._complementarity／stage3_hrp_isoos 同一套判定，不另立標準。"""
    if corr < C.COMPLEMENTARITY_CUTS["高"]:
        return "高"
    if corr < C.COMPLEMENTARITY_CUTS["中"]:
        return "中"
    return "低"


def cluster_representatives(wide: pd.DataFrame, assign: pd.DataFrame,
                            level: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """群代表序列（群×月）、每群成員數、每群市場。

    群代表＝成員列的簡單平均，跟 `stage3_hrp._cluster_meta_and_corr` 及 H-06
    的群代表口徑一致——這裡不能自創口徑，否則跟既有的 L1 相關矩陣對不起來。
    """
    col = f"cluster_{level}"
    lab = assign.set_index("strategy_uid")[col].reindex(wide.index)
    valid = lab.notna()
    if not valid.all():
        # 建樹當時排除過零變異數策略，assign 仍可能少數對不上；只取對得上的
        wide, lab = wide[valid.values], lab[valid]

    reps = wide.groupby(lab.astype(int).values).mean()
    sizes = lab.astype(int).value_counts().sort_index()
    mkt = (pd.Series(wide.index, index=wide.index).str.split("::").str[0]
           .groupby(lab.astype(int).values).agg(lambda s: s.mode().iloc[0]))
    return reps, sizes, mkt


def pairs_for_tree(tree_id: str, level: str, log=print) -> pd.DataFrame:
    """一棵樹在某個層級的全部群對 + 相關 + same/cross + 成員數。"""
    wide = S3.rebuild_tree_returns(tree_id, log=lambda *a, **k: None)
    assign = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")
    assign = assign[assign.tree_id == tree_id]

    reps, sizes, mkt = cluster_representatives(wide, assign, level)
    corr = np.corrcoef(reps.values)
    ids = list(reps.index)

    rows = []
    for i, j in itertools.combinations(range(len(ids)), 2):
        a, b = ids[i], ids[j]
        rows.append({
            "tree_id": tree_id, "level": level, "cluster_a": a, "cluster_b": b,
            "corr": float(corr[i, j]),
            "pair_type": "same" if mkt[a] == mkt[b] else "cross",
            "n_members_a": int(sizes[a]), "n_members_b": int(sizes[b]),
            "complementarity": _complementarity(float(corr[i, j])),
        })
    log(f"  [{tree_id}/{level}] {len(ids)} 群（成員數中位 {int(sizes.median())}）"
        f"→ {len(rows)} 對")
    return pd.DataFrame(rows)


def summarize(pairs: pd.DataFrame) -> pd.DataFrame:
    """依 (樹, 層級, min_members, same/cross) 彙總互補分布。"""
    rows = []
    for (tree, level), g0 in pairs.groupby(["tree_id", "level"], observed=True):
        for mm in MIN_MEMBERS_GRID:
            g = g0[(g0.n_members_a >= mm) & (g0.n_members_b >= mm)]
            n_cl = len(set(g.cluster_a) | set(g.cluster_b))
            for pt in ("same", "cross"):
                s = g[g.pair_type == pt]
                n = len(s)
                rows.append({
                    "tree_id": tree, "level": level, "min_members": mm, "pair_type": pt,
                    "n_clusters": n_cl, "n_pairs": n,
                    "corr_min": float(s["corr"].min()) if n else None,
                    "corr_median": float(s["corr"].median()) if n else None,
                    "corr_max": float(s["corr"].max()) if n else None,
                    "n_high": int((s.complementarity == "高").sum()),
                    "n_mid": int((s.complementarity == "中").sum()),
                    "n_low": int((s.complementarity == "低").sum()),
                    "pct_high": float((s.complementarity == "高").mean()) if n else None,
                })
    return pd.DataFrame(rows)


#: 免費午餐清單只在跨市場樹做，且只看夠大的群（小群相關估計雜訊大，見 MIN_MEMBERS_GRID
#: 的穩健性掃描——成員>=20 時跨市場高互補仍有 62.6%，結論不靠小群撐著）。
SHORTLIST_TREE = "XM_normal"
SHORTLIST_LEVEL = "L3"
SHORTLIST_MIN_MEMBERS = 20


def _cluster_profiles(tree_id: str, level: str) -> pd.DataFrame:
    """L3 群的成分側寫（市場／因子／估值濾網／績效），供免費午餐清單指認用。"""
    a = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")
    a = a[a.tree_id == tree_id]
    idx = pd.read_parquet(paths.STAGE0 / "candidate_index.parquet")
    m = a.merge(idx, on=C.PK, how="left")

    def _top2(s: pd.Series) -> str:
        v = s.fillna("無").value_counts(normalize=True)
        return "/".join(f"{i}{p:.0%}" for i, p in v.head(2).items())

    g = m.groupby(f"cluster_{level}")
    return pd.DataFrame({
        "n_members": g.size(),
        "market": g["market"].agg(lambda s: s.mode().iloc[0]),
        "top_F1": g["F1_factor"].agg(_top2),
        "top_C_source": g["C_source"].agg(_top2),
        "pct_v1": g["V"].agg(lambda s: float((s == "v1").mean())),
        "CAGR_median": g["CAGR"].median(),
        "MDD_median": g["max_drawdown"].median(),
    })


def free_lunch_shortlist(pairs: pd.DataFrame, log=print) -> pd.DataFrame:
    """把「免費午餐藏在小而特化的群 × 另一個市場」落實成具體可指認的群清單。

    對 XM_normal 樹每一個夠大的 L3 群，算它跟**對面市場**的候選群裡有多少比例
    達到高互補。`pct_high_complement == 1.0`（欄位 `universal`）代表「跟對面市場
    每一群配都能吃到分散」，是配置時最有價值的群。
    """
    prof = _cluster_profiles(SHORTLIST_TREE, SHORTLIST_LEVEL)
    prof = prof[prof.n_members >= SHORTLIST_MIN_MEMBERS]

    x = pairs[(pairs.tree_id == SHORTLIST_TREE) & (pairs.level == SHORTLIST_LEVEL)
             & (pairs.pair_type == "cross")
             & (pairs.n_members_a >= SHORTLIST_MIN_MEMBERS)
             & (pairs.n_members_b >= SHORTLIST_MIN_MEMBERS)]
    high_cut = C.COMPLEMENTARITY_CUTS["高"]

    n_by_mkt = prof.market.value_counts()
    rows = []
    for cid, r in prof.iterrows():
        own = x[(x.cluster_a == cid) | (x.cluster_b == cid)]
        if own.empty:
            continue
        best = own.loc[own["corr"].idxmin()]
        partner = int(best.cluster_b if best.cluster_a == cid else best.cluster_a)
        # 對面市場的候選群數＝對方市場的群數（本群不在其中，不需扣自己）
        n_partners = int(n_by_mkt[[k for k in n_by_mkt.index if k != r.market][0]])
        n_high = int((own["corr"] < high_cut).sum())
        rows.append({
            "tree_id": SHORTLIST_TREE, "level": SHORTLIST_LEVEL, "cluster_id": int(cid),
            "market": r.market, "n_members": int(r.n_members),
            "top_F1": r.top_F1, "top_C_source": r.top_C_source,
            "pct_v1": float(r.pct_v1),
            "CAGR_median": float(r.CAGR_median), "MDD_median": float(r.MDD_median),
            "n_cross_partners": n_partners, "n_high_complement": n_high,
            "pct_high_complement": n_high / n_partners,
            "min_cross_corr": float(best["corr"]), "best_partner_cluster": partner,
            "universal": bool(n_high == n_partners),
        })
    df = pd.DataFrame(rows)
    df["tree_id"] = df["tree_id"].astype("category")
    df["level"] = df["level"].astype("category")
    df["market"] = df["market"].astype("category")
    log(f"  免費午餐清單：{len(df)} 群（成員>={SHORTLIST_MIN_MEMBERS}），"
        f"其中 universal（跟對面市場每一群都高互補）{int(df.universal.sum())} 群")
    return df.sort_values(["market", "pct_high_complement", "CAGR_median"],
                         ascending=[True, False, False]).reset_index(drop=True)


def run(log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE3)

    all_pairs = []
    for tree in TREES:
        for level in LEVELS:
            all_pairs.append(pairs_for_tree(tree, level, log))
    pairs = pd.concat(all_pairs, ignore_index=True)

    summary = summarize(pairs)
    for col in ("tree_id", "level", "pair_type"):
        summary[col] = summary[col].astype("category")
    C.validate(summary, C.COMPLEMENTARITY_GRANULARITY, strict_columns=True)
    log(f"\n✓ complementarity_granularity 契約通過（{len(summary)} 列）")

    shortlist = free_lunch_shortlist(pairs, log)
    C.validate(shortlist, C.FREE_LUNCH_SHORTLIST, strict_columns=True)
    log(f"✓ free_lunch_shortlist 契約通過（{len(shortlist)} 群）")

    out_dir = paths.ROOT / "_analysis_outputs_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_sum = out_dir / "complementarity_granularity_summary.csv"
    p_pairs = out_dir / "complementarity_granularity_pairs.csv"
    p_short = out_dir / "free_lunch_shortlist.csv"
    summary.to_csv(p_sum, index=False, encoding="utf-8-sig")
    pairs.to_csv(p_pairs, index=False, encoding="utf-8-sig")
    shortlist.to_csv(p_short, index=False, encoding="utf-8-sig")

    freeze.write_manifest(
        "complementarity_granularity", out_dir / "_complementarity_granularity_manifest",
        inputs=[paths.STAGE3 / "cluster_assign.parquet",
               paths.STAGE1 / "returns_monthly.parquet",
               paths.STAGE0 / "candidate_index.parquet"],
        outputs=[p_sum, p_pairs, p_short],
        params={"trees": list(TREES), "levels": list(LEVELS),
               "min_members_grid": list(MIN_MEMBERS_GRID),
               "complementarity_cuts": C.COMPLEMENTARITY_CUTS,
               "shortlist_tree": SHORTLIST_TREE, "shortlist_level": SHORTLIST_LEVEL,
               "shortlist_min_members": SHORTLIST_MIN_MEMBERS},
        notes="H-25：互補性的分群粒度效應。L1沒有高互補配對是聚合效應不是門檻問題，"
              "同市場配對當對照組排除小群雜訊解釋。H-25b：免費午餐清單，把結論落實成"
              "具體可指認的群。理由見模組docstring。",
    )
    log(f"→ {p_sum}\n→ {p_pairs}\n→ {p_short}")
    return pairs, summary


def _report(pairs: pd.DataFrame, summary: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 78)
    log("H-25 · 互補性的分群粒度效應")
    log("=" * 78)
    log(f"判定門檻（沿用 contracts.COMPLEMENTARITY_CUTS，未改）：{C.COMPLEMENTARITY_CUTS}")

    for mm in MIN_MEMBERS_GRID:
        log(f"\n【只納入成員數 >= {mm} 的群】")
        s = summary[summary.min_members == mm]
        log(f"{'樹':<12}{'層級':<6}{'類型':<7}{'群數':>5}{'對數':>8}"
            f"{'相關中位':>10}{'最低':>8}{'高互補':>8}{'佔比':>8}")
        for r in s.sort_values(["tree_id", "level", "pair_type"]).itertuples():
            if r.n_pairs == 0:
                continue
            log(f"{r.tree_id:<12}{r.level:<6}{r.pair_type:<7}{r.n_clusters:>5}"
                f"{r.n_pairs:>8,}{r.corr_median:>10.3f}{r.corr_min:>8.3f}"
                f"{r.n_high:>8,}{r.pct_high:>8.1%}")

    log("\n" + "-" * 78)
    log("關鍵對照：同一批策略，只改分群粒度")
    log("-" * 78)
    for tree in TREES:
        for pt in ("cross", "same"):
            a = summary[(summary.tree_id == tree) & (summary.level == "L1")
                       & (summary.pair_type == pt) & (summary.min_members == 1)]
            b = summary[(summary.tree_id == tree) & (summary.level == "L3")
                       & (summary.pair_type == pt) & (summary.min_members == 1)]
            if a.empty or b.empty or a.iloc[0].n_pairs == 0 or b.iloc[0].n_pairs == 0:
                continue
            a, b = a.iloc[0], b.iloc[0]
            log(f"{tree:<12}{pt:<7}L1 中位{a.corr_median:.3f}/高互補{a.pct_high:.1%}"
                f"  →  L3 中位{b.corr_median:.3f}/高互補{b.pct_high:.1%}")


def _report_shortlist(log=print) -> None:
    p = (paths.ROOT / "_analysis_outputs_robustness" / "free_lunch_shortlist.csv")
    df = pd.read_csv(p)
    uni = df[df.universal]
    log("\n" + "=" * 78)
    log(f"H-25b · 免費午餐清單（{SHORTLIST_TREE}／{SHORTLIST_LEVEL}／成員>={SHORTLIST_MIN_MEMBERS}）")
    log("=" * 78)
    log(f"候選群 {len(df)}（TW {int((df.market=='TW').sum())}／US {int((df.market=='US').sum())}）"
        f"｜跟對面市場**每一群**都高互補的群：{len(uni)}")
    for mk in ("TW", "US"):
        g = uni[uni.market == mk]
        if g.empty:
            continue
        log(f"\n【{mk} 側 {len(g)} 群】")
        log(f"{'群':>5}{'檔數':>5}  {'主力因子':<24}{'條件訊號':<20}"
            f"{'v1':>5}{'CAGR':>7}{'MDD':>7}{'最低跨市場相關':>10}")
        for r in g.sort_values("CAGR_median", ascending=False).itertuples():
            log(f"{r.cluster_id:>5}{r.n_members:>5}  {r.top_F1:<24}{r.top_C_source:<20}"
                f"{r.pct_v1:>5.0%}{r.CAGR_median:>7.1%}{r.MDD_median:>7.1%}"
                f"{r.min_cross_corr:>10.3f}（對群{r.best_partner_cluster}）")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.complementarity_granularity")
    ap.parse_args(argv)
    pairs, summary = run()
    _report(pairs, summary)
    _report_shortlist()
    return 0


if __name__ == "__main__":
    sys.exit(main())
