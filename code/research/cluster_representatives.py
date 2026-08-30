# -*- coding: utf-8 -*-
"""H-10 · 群內代表策略挑選規則（開發待辦追蹤.md 第四階段）

老師的實務問題（資金限制）：「這每一群裡面我到底要用績效最好的咧？還是績效
去除以MDD咧？」——這裡處理的是「選誰、選幾個」。**權重已經由S-04定案為等權**
（依學長論文實證：多代理的貢獻偏向標的篩選而非權重配置），不在這裡重複解決。

陷阱（H-10原始記錄）：純挑CAGR（或純挑Calmar）最高，會挑到一批彼此高度相關
的贏家，把HRP分群好不容易找出來的分散性，在群內部又丟掉一次。這是本模組要
避免的核心問題。

規則（貪婪多樣性選擇）：
  1. 品質分數 = CAGR / |MDD|（Calmar比率，直接對應老師「績效去除以MDD」的原話）
  2. 依品質由高到低排序，貪婪納入代表——但新納入者對「已選入的每一個代表」的
     相關係數都不能超過門檻，**門檻取該群自己的 avg_intra_corr**（群內平均相關，
     stage3_hrp.py 已算好、存在 cluster_meta.parquet）——用群自己的相關水準
     當門檻，而非武斷訂一個全域常數（不同群的內部相關水準差異很大）。
  3. 若多樣性門檻太嚴、候選不足m個，退而求其次用品質單獨排序補上剩下的名額
     （不強行湊數但也不留空，跟 output_a.py `_pick_diversified` 同樣的設計慣例）。

`co_fail_regimes` 當警示用（非硬性排除，H-15已定案）：輸出表附上每個群的
co_fail_peers 供人工複查/論文揭露，**不**用它篩選或排除任何代表。

只做 L1、只做 normal 樹——跟 H-06/H-07/cluster_story 同樣的理由（L1是給人讀
的粗粒度；crisis樹樣本量不足、H-14已限定描述性用途）。

`m`（每群代表數）是本模組的自由參數，這裡先跑 m=3 跟 m=5 兩組供比較，不是
定案的資金配置數字——真正要選多少要等 H-12 的四組對照實驗才有依據。

用法：
    cd code
    python -m research.cluster_representatives                # 預設 m=3
    python -m research.cluster_representatives --m 5
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, paths
from .effective_bets import _tree_corr, DEFAULT_TREES

OUT_DIR = paths.ROOT / "_analysis_outputs_robustness"


def select_representatives(quality: pd.Series, corr: np.ndarray, uid_index: pd.Index,
                           m: int, max_pairwise_corr: float) -> tuple[list[str], list[str]]:
    """在同一個群內，依品質（quality，越高越好）貪婪選出最多m個代表。

    `corr`/`uid_index`：該群成員的方陣相關係數（numpy，位置對應`uid_index`），
    由呼叫端從整棵樹的相關矩陣切出子矩陣傳入——不在這裡重算，避免對每個群
    各自重跑一次昂貴的相關性計算。

    回傳 (picked, backfilled)：backfilled 是因為多樣性門檻太嚴、候選不夠時，
    退而求其次用品質單獨排序補上的名單（是picked的子集，供事後區分「真正
    通過多樣性篩選的」vs「補位的」）。
    """
    pos = {u: i for i, u in enumerate(uid_index)}
    order = [u for u in quality.sort_values(ascending=False).index if u in pos]

    picked: list[str] = []
    for uid in order:
        if len(picked) >= m:
            break
        i = pos[uid]
        if not picked or all(abs(corr[i, pos[p]]) <= max_pairwise_corr for p in picked):
            picked.append(uid)

    backfilled: list[str] = []
    if len(picked) < m:
        for uid in order:
            if len(picked) >= m or uid in picked:
                continue
            picked.append(uid)
            backfilled.append(uid)
    return picked, backfilled


def _avg_pairwise_corr(uids: list[str], corr: np.ndarray, uid_index: pd.Index) -> float | None:
    """picked代表集合彼此的平均相關（不含自己），用來對照「有沒有真的比較分散」。"""
    if len(uids) < 2:
        return None
    pos = {u: i for i, u in enumerate(uid_index)}
    idx = [pos[u] for u in uids]
    sub = corr[np.ix_(idx, idx)]
    n = len(idx)
    off_diag = sub[~np.eye(n, dtype=bool)]
    return float(off_diag.mean())


def build(trees=DEFAULT_TREES, m: int = 3, max_share_of_avg: float = 1.0,
         log=print) -> pd.DataFrame:
    """`max_share_of_avg`：多樣性門檻＝該群 avg_intra_corr 的這個比例
    （預設1.0＝用群自己的平均相關當門檻；<1.0代表要求比群平均更分散才收）。
    """
    assign = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")
    meta = pd.read_parquet(paths.STAGE3 / "cluster_meta.parquet")
    meta = meta[meta.level == "L1"].set_index(["tree_id", "cluster_id"])
    sm = pd.read_parquet(paths.STAGE4 / "strategy_map.parquet")[
        [C.PK, "CAGR", "max_drawdown"]]
    cf = pd.read_parquet(paths.STAGE3 / "co_fail_regimes.parquet")
    cf = cf[cf.level == "L1"]

    rows = []
    for tree_id in trees:
        tree_key = tree_id.rsplit("_", 1)[0]
        corr, uid_index = _tree_corr(tree_id, log)
        pos_lookup = pd.Series(range(len(uid_index)), index=uid_index)

        a = assign[assign.tree_id == tree_id][[C.PK, "cluster_L1"]]
        a = a[a[C.PK].isin(uid_index)]     # 對齊 _tree_corr 排除的零變異數策略
        cf_peers = cf[cf.tree_key == tree_key].set_index("cluster_normal")["co_fail_peers"]

        for cid, g in a.groupby("cluster_L1"):
            members = g[C.PK].tolist()
            qsub = sm[sm[C.PK].isin(members)].set_index(C.PK)
            # Calmar比率：max_drawdown是負數(le=0)，|MDD|=0（理論上不該出現，
            # 防禦性處理避免除以零）視為無法評分，quality設為NaN讓它排到最後
            quality = (qsub["CAGR"] / qsub["max_drawdown"].abs()).replace(
                [np.inf, -np.inf], np.nan)

            avg_intra = (meta.loc[(tree_id, int(cid)), "avg_intra_corr"]
                        if (tree_id, int(cid)) in meta.index else np.nan)
            threshold = (float(avg_intra) * max_share_of_avg
                        if pd.notna(avg_intra) else 1.0)   # 群只有1人時avg_intra=NaN，門檻形同虛設

            member_idx = [pos_lookup[u] for u in members]
            sub_corr = corr[np.ix_(member_idx, member_idx)]
            sub_index = pd.Index(members)

            picked, backfilled = select_representatives(
                quality.dropna(), sub_corr, sub_index, m, threshold)
            # 品質全NaN（極端狀況）：quality.dropna()為空，退回全體依CAGR排序選前m個
            if not picked:
                naive_order = qsub["CAGR"].sort_values(ascending=False).index.tolist()
                picked = naive_order[:m]
                backfilled = list(picked)

            naive_top_m = quality.sort_values(ascending=False).index[:m].tolist()

            rows.append({
                "tree_id": tree_id, "level": "L1", "cluster_id": int(cid),
                "n_members": len(members), "m_target": m,
                "n_picked": len(picked), "n_backfilled": len(backfilled),
                "picked_uids": "|".join(picked),
                "naive_top_m_uids": "|".join(naive_top_m),
                "avg_intra_corr_cluster": (round(float(avg_intra), 4)
                                           if pd.notna(avg_intra) else None),
                "avg_pairwise_corr_picked": _avg_pairwise_corr(picked, corr, uid_index),
                "avg_pairwise_corr_naive": _avg_pairwise_corr(naive_top_m, corr, uid_index),
                "co_fail_peers": cf_peers.get(int(cid), ""),   # 警示用，不做篩選（H-15）
            })
        log(f"[{tree_id}] {a['cluster_L1'].nunique()} 群完成挑選")

    return pd.DataFrame(rows)


def run(trees=DEFAULT_TREES, m: int = 3, max_share_of_avg: float = 1.0,
       log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE3)
    freeze.verify_inputs(paths.STAGE4)
    out = build(trees=trees, m=m, max_share_of_avg=max_share_of_avg, log=log)
    for col in ("avg_pairwise_corr_picked", "avg_pairwise_corr_naive"):
        out[col] = out[col].round(4)
    out["tree_id"] = out["tree_id"].astype("category")
    out["level"] = out["level"].astype("category")
    C.validate(out, C.CLUSTER_REPRESENTATIVES, strict_columns=True)
    log("✓ cluster_representatives 契約通過")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"cluster_representatives_m{m}.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    log(f"→ {p}  {len(out)} 群")
    return out


def _report(df: pd.DataFrame, m: int, log=print) -> None:
    log("\n" + "=" * 78)
    log(f"H-10 · 群內代表挑選（m={m}）驗收摘要")
    log("=" * 78)
    diversified = df["avg_pairwise_corr_picked"].notna() & df["avg_pairwise_corr_naive"].notna()
    both = df[diversified]
    if len(both):
        improved = (both["avg_pairwise_corr_picked"] < both["avg_pairwise_corr_naive"]).sum()
        avg_drop = (both["avg_pairwise_corr_naive"] - both["avg_pairwise_corr_picked"]).mean()
        log(f"跟「純品質前{m}名」相比：{improved}/{len(both)} 群的代表彼此平均相關有下降"
            f"（平均降幅 {avg_drop:+.3f}）")
    log(f"\n有動用backfill（多樣性門檻太嚴、候選不足{m}個）的群數："
        f"{(df['n_backfilled'] > 0).sum()}/{len(df)}")
    for r in df.itertuples():
        diff = (r.picked_uids != r.naive_top_m_uids)
        log(f"[{r.tree_id}] 群{r.cluster_id}（{r.n_members}成員）："
            f"選{r.n_picked}個｜picked彼此平均相關={r.avg_pairwise_corr_picked}"
            f"｜naive前{m}名彼此平均相關={r.avg_pairwise_corr_naive}"
            f"｜與naive選法{'不同' if diff else '相同'}"
            f"{'｜co_fail警示:' + r.co_fail_peers if r.co_fail_peers else ''}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.cluster_representatives")
    ap.add_argument("--trees", nargs="+", default=list(DEFAULT_TREES))
    ap.add_argument("--m", type=int, default=3, help="每群目標代表數（預設3，自由參數）")
    ap.add_argument("--max-share-of-avg", type=float, default=1.0,
                    help="多樣性門檻＝群avg_intra_corr的這個比例，預設1.0")
    a = ap.parse_args(argv)
    df = run(trees=a.trees, m=a.m, max_share_of_avg=a.max_share_of_avg)
    _report(df, a.m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
