# -*- coding: utf-8 -*-
"""S-07 · 決策層重複執行穩定度（開發待辦追蹤.md 方向C/D 交界）

學長論文未來工作第2點：同一輸入重複跑 N 次，看決策一致性。LLM 有隨機性，若
同一個總經狀態跑 5 次選出不同的群，方向C 的可信度就有問題——**這項同時是
方向D（抗幻覺）的實證素材**，不是只服務方向C。

跟 S-06（暫緩，見開發待辦追蹤.md S-06條目）比，這裡測的是「LLM決策層本身
可不可靠」，不需要先解決「哪種prompt風格才對」這個現階段無法評估優劣的問題，
故優先做這項。

做法：同一個 (tree_id, market, month) 快照，重複呼叫 `decision_layer_arms.
llm_decision()` N 次（每次都是獨立的真實API呼叫，不是快取重播），量化：

  exact_match_rate      眾數結果（出現最多次的那個選群組合）佔全部N次的比例
  mean_pairwise_jaccard  N次兩兩比較的平均Jaccard相似度（1=完全一樣，0=完全不重疊）
  stable_core            每一次都被選中的群（交集）
  unstable_fringe        只有部分次數被選中的群（聯集減交集）
  rule_in_llm_rate       A_rule（純規則的最佳群）有多少比例的LLM回合有把它包含進去

用法：
    cd code
    python -m research.decision_repeatability --dry-run
    python -m research.decision_repeatability
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter

import pandas as pd

from . import contracts as C
from . import freeze, paths
from .decision_layer_arms import llm_decision, macro_state_snapshot, rule_based_decision

#: 三個情境，涵蓋三棵樹＋三種不同的clock_cell（缺衰退，樣本已足夠看出穩定度趨勢，
#: 不需要窮舉四格；若之後想補齊，直接加進這個tuple即可，函式本身不需要改）。
DEFAULT_SCENARIOS = (
    ("XM_normal", "TW", "2025-12"),   # 復甦
    ("US_normal", "US", "2025-12"),   # 過熱
    ("TW_normal", "TW", "2025-05"),   # 停滯性通膨
)
DEFAULT_N_REPEATS = 5


def repeatability_check(tree_id: str, market: str, month: str, *, n_repeats: int = DEFAULT_N_REPEATS,
                        top_n_rule: int = 1, model: str | None = None,
                        dry_run: bool = False, log=print) -> dict:
    macro_state = macro_state_snapshot(market, month)
    clock_cell = macro_state["clock_cell"]
    if clock_cell is None:
        raise ValueError(f"{market} {month} 沒有clock_cell分類，無法測穩定度")
    a_rule = set(rule_based_decision(tree_id, clock_cell, top_n=top_n_rule))

    runs: list[set[int]] = []
    actual_model = model
    for i in range(n_repeats):
        d = llm_decision(tree_id, market, month, model=model, dry_run=dry_run,
                         log=lambda *a, **k: None)
        s = set(d["selected_clusters"])
        runs.append(s)
        log(f"  [{tree_id}/{market}/{clock_cell}] 第{i+1}/{n_repeats}次：{sorted(s)}")
        if not dry_run and actual_model is None:
            from utils.config import Config
            actual_model = Config().get_openai_model("decision_layer_arm_b")

    if dry_run:
        return {"tree_id": tree_id, "market": market, "clock_cell": clock_cell,
               "n_repeats": n_repeats, "exact_match_rate": None,
               "mean_pairwise_jaccard": None, "stable_core": "", "unstable_fringe": "",
               "rule_in_llm_rate": None, "all_runs": "(dry-run)", "model": "(dry-run)"}

    counts = Counter(frozenset(s) for s in runs)
    mode_set, mode_count = counts.most_common(1)[0]
    exact_match_rate = mode_count / n_repeats

    core = set.intersection(*runs) if runs else set()
    union = set.union(*runs) if runs else set()
    fringe = union - core

    pairs = list(itertools.combinations(runs, 2))
    if pairs:
        jaccards = [len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 1.0 for s1, s2 in pairs]
        mean_jaccard = sum(jaccards) / len(jaccards)
    else:
        mean_jaccard = None

    rule_in_llm_rate = sum(1 for s in runs if a_rule <= s) / n_repeats

    result = {
        "tree_id": tree_id, "market": market, "clock_cell": clock_cell,
        "n_repeats": n_repeats,
        "exact_match_rate": round(exact_match_rate, 4),
        "mean_pairwise_jaccard": round(mean_jaccard, 4) if mean_jaccard is not None else None,
        "stable_core": "|".join(str(c) for c in sorted(core)),
        "unstable_fringe": "|".join(str(c) for c in sorted(fringe)),
        "rule_in_llm_rate": round(rule_in_llm_rate, 4),
        "all_runs": "|".join(str(sorted(s)) for s in runs),
        "model": str(actual_model),
    }
    log(f"  → 完全一致比例{exact_match_rate:.0%}｜平均Jaccard{mean_jaccard:.3f}"
        f"｜穩定核心{sorted(core)}｜規則基準被包含比例{rule_in_llm_rate:.0%}")
    return result


def run(scenarios=DEFAULT_SCENARIOS, n_repeats: int = DEFAULT_N_REPEATS,
       model: str | None = None, log=print) -> pd.DataFrame:
    # ⚠️ 2026-09-01 code review 修正：原本這裡驗 paths.STAGE3（主manifest），
    # 但本模組透過 repeatability_check → rule_based_decision/llm_decision 實際
    # 讀的是 cluster_macro_conditional.parquet／cluster_macro_interface.parquet
    # （附加產物，manifest在各自子目錄）跟 macro_history（STAGE2/macro），
    # 完全不在STAGE3主manifest涵蓋範圍——驗了不相干的東西。這三處下游函式
    # 2026-09-01已補上各自正確的verify_inputs（見decision_layer_arms.py／
    # macro_decision_input.py），這裡不需要重複驗證，改由呼叫鏈本身保證。
    rows = [repeatability_check(t, m, mo, n_repeats=n_repeats, model=model, log=log)
           for t, m, mo in scenarios]
    df = pd.DataFrame(rows)
    df["tree_id"] = df["tree_id"].astype("category")
    df["market"] = df["market"].astype("category")
    df["clock_cell"] = df["clock_cell"].astype("category")
    C.validate(df, C.DECISION_REPEATABILITY, strict_columns=True)
    log(f"\n✓ decision_repeatability 契約通過（{len(df)} 個情境）")

    out_dir = paths.ROOT / "_analysis_outputs_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "decision_repeatability.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    # ⚠️ 之後若要重讀這份CSV：`stable_core`/`unstable_fringe` 存的是"|"分隔的群id
    # 清單，當清單剛好只有1個元素時字串長得跟純數字一樣——若該欄每一列剛好都是
    # 空值或單一數字，pandas的CSV型別推斷會把整欄判成float64而非字串，得寫
    # `pd.read_csv(p, dtype={"stable_core": str, "unstable_fringe": str})` 才安全
    # （2026-08-31開發時實測踩到，不是理論風險，見 t_decision_repeatability_real_data）。
    freeze.write_manifest(
        "decision_repeatability", out_dir / "_decision_repeatability_manifest",
        inputs=[paths.STAGE3 / "cluster_macro_conditional.parquet",
               paths.STAGE3 / "cluster_macro_interface.parquet"],
        outputs=[p],
        params={"scenarios": [list(s) for s in scenarios], "n_repeats": n_repeats},
        notes="S-07：LLM決策層重複執行穩定度，方向C可信度與方向D抗幻覺的共同實證素材。",
    )
    log(f"→ {p}")
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.decision_repeatability")
    ap.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS)
    ap.add_argument("--model", help="覆寫模型（預設讀 config.ini 的 decision_layer_arm_b_model）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.dry_run:
        for t, m, mo in DEFAULT_SCENARIOS:
            repeatability_check(t, m, mo, n_repeats=2, dry_run=True)
        return 0
    df = run(n_repeats=a.n_repeats, model=a.model)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
