# -*- coding: utf-8 -*-
"""階段3 附加 · cluster_identity（研究部 H-08，單群身份的 LLM 解釋）

老師原話：「才知道什麼時候用哪一個」——`cluster_story.py`（LLM點③）只回答「兩群
為什麼互補」，這裡回答不同的問題：**「這一群本身到底是什麼意思」**，單群身份描述，
是不同的產出物。依賴 H-06（`cluster_profile_quant`，時間型態）+ `cluster_story.
_cluster_profiles`（橫斷面成分側寫）+ `co_fail_regimes`（危機期關聯，僅供揭露）。

**這是銜接第六階段（總經→選群，方向C）的介面**：S-01要定義「群→總經層」該餵什麼
過去，這裡產出的身份描述就是那個介面的原始素材。

---------------------------------------------------------------------------
與 cluster_story 相同的抗幻覺設計，額外多一條邊界
---------------------------------------------------------------------------
沿用同一套鐵則（LLM不下判斷、只為程式算好的客觀事實寫文字；只能引用提供的數字；
機械性差異要老實講）。**額外多一條 H-08 專屬的邊界**：

  🔴 **不推論「這群適合什麼總經環境」**——那是S-01/S-02之後決策層的工作。H-08
  完全沒有給LLM任何總經資訊，若這裡就讓LLM猜「這群在升息環境表現好」之類的話，
  等於在總經知識缺席的情況下編造適配性因果，先污染了後面決策層要用的素材。
  這裡只做**客觀身份描述**：這群由什麼因子/市場/估值濾網構成、績效隨時間的
  型態長怎樣，僅此而已。

只做 L1、只做 normal 樹（crisis 樹樣本量太小、H-14/H-16已定案限定描述性用途，
跟 H-06/H-07/cluster_story 同樣的理由）。TW6+US7+XM3 共 16 群。

用法：
    cd code
    python -m research.cluster_identity --dry-run          # 只印prompt不呼叫API、不花錢
    python -m research.cluster_identity --limit 3          # 先跑3群試水溫
    python -m research.cluster_identity                    # 全跑（三棵normal樹，16群）
    python -m research.cluster_identity --resume           # 額度暫停後接續跑
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

from . import contracts as C
from . import freeze, paths
from .cluster_story import _cluster_profiles, _co_fail_lookup

DEFAULT_TREES = ("TW_normal", "US_normal", "XM_normal")
LEVEL = "L1"

RESERVE_RATIO = 0.05
#: 單次呼叫的預估 token 數。🔄 2026-08-30實測3群（TW/US/XM各1）平均約3,627tok
#: （比原本沿用cluster_story的估計值2,500高——單群描述並不比雙群比較省，
#: gpt-5的回覆本身就偏長），取4,000留餘裕。
EST_TOKENS_PER_CALL = 4_000


_SYSTEM_PROMPT = (
    "你是量化研究系統的分析結果轉譯器。你會拿到一群交易策略的**客觀統計側寫**，"
    "你的工作是把這群策略的『身份』用清楚的中文描述出來，不是自己重新判斷或推論。\n"
    "鐵則：\n"
    "1. 只能引用提供的數字，禁止杜撰個股、產業、總體經濟事件，或提供資訊之外的任何因果解釋。\n"
    "2. 若這群的構成主要是機械性的（例如純粹因為同一因子家族、同一估值濾網、"
    "或同一市場），如實說明，不要包裝成深層的投資哲學或風格故事。\n"
    "3. 🔴 禁止推論「這群策略適合什麼樣的總體經濟環境／市場情境」——你沒有拿到"
    "任何總經資訊，這類推論一定是編造的，那是完全不同階段的工作。\n"
    "4. 若群內平均相關（avg_intra_corr）不算很高，代表群內成員異質性不低，"
    "必須在caveat中明講「此描述是群層級的平均特徵，群內成員可能有相當差異」。"
)

_IDENTITY_SCHEMA = {
    "name": "cluster_identity",
    "schema": {
        "type": "object",
        "properties": {
            "identity_label": {
                "type": "string",
                "description": "一句話身份標籤（不超過30字），例如「以估值因子為主"
                               "的台股中型股組合」。只能根據提供的側寫下標籤。"},
            "mechanism_note": {
                "type": "string",
                "description": "這群為什麼長這樣——引用提供的因子/市場/估值濾網"
                               "構成數字說明。若是機械性差異就直說。"},
            "performance_pattern": {
                "type": "string",
                "description": "績效隨時間的型態——引用提供的年度/季度報酬統計"
                               "（正報酬年數、最佳/最差年、年度波動）。"},
            "caveat": {
                "type": "string",
                "description": "此身份描述的限制（例如群內異質性、樣本代表性、"
                               "觀察期間長短）。只能根據提供的資訊寫。"},
        },
        "required": ["identity_label", "mechanism_note", "performance_pattern", "caveat"],
        "additionalProperties": False,
    },
    "strict": True,
}


def _quant_profile(tree_id: str) -> dict[int, dict]:
    """H-06 的時間型態欄位，補進 cluster_story._cluster_profiles 沒有的維度。"""
    q = pd.read_parquet(paths.STAGE3 / "cluster_profile_quant.parquet")
    q = q[(q.tree_id == tree_id) & (q.level == LEVEL)].set_index("cluster_id")
    return q.to_dict("index")


def build_prompt(tree_id: str, comp: dict, quant: dict, co_fail_peers: set[int]) -> str:
    payload = {
        "cluster_id": comp["cluster_id"],
        "n_members": comp["n_members"],
        "market_mix": comp["market_mix"],
        "top_factor_types": comp["top_factor_types"],
        "top_F1": comp["top_F1"],
        "top_C_source": comp["top_C_source"],
        "V_mix": comp["V_mix"],
        "CAGR_median": comp["CAGR_median"],
        "MDD_median": comp["MDD_median"],
        "smallcap_share_median": comp["smallcap_share_median"],
        "avg_intra_corr": comp["avg_intra_corr"],
        "觀察期間": f"{quant.get('window_start_year')}~{quant.get('window_end_year')}"
                  f"（{quant.get('n_years')}年）" if quant else None,
        "正報酬年數": (f"{quant.get('n_years_positive')}/{quant.get('n_years')}"
                    f"（{quant.get('pct_years_positive'):.0%}）"
                   if quant and quant.get("pct_years_positive") is not None else None),
        "最佳年": (f"{quant.get('best_year')}（{quant.get('best_year_ret'):+.1%}）"
                 if quant and quant.get("best_year_ret") is not None else None),
        "最差年": (f"{quant.get('worst_year')}（{quant.get('worst_year_ret'):+.1%}）"
                 if quant and quant.get("worst_year_ret") is not None else None),
        "年度報酬標準差": quant.get("annual_ret_std") if quant else None,
        "季度報酬標準差": quant.get("quarterly_ret_std") if quant else None,
        "危機期co_fail對象": (sorted(co_fail_peers) if co_fail_peers else "無"),
    }
    return (
        f"樹：{tree_id}（層級 {LEVEL}）\n\n"
        f"【群{comp['cluster_id']} 客觀側寫】\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "請依給定的 JSON schema 輸出，只能使用以上資訊。"
    )


def _call_llm(prompt: str, model: str, api_key: str, *, est_tokens: int = 0) -> tuple[dict, dict]:
    import requests
    from utils import openai_quota as OQ
    OQ.check_free_tier_budget(model, estimated_tokens=est_tokens, reserve_ratio=RESERVE_RATIO)
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model,
             "messages": [{"role": "system", "content": _SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}],
             "response_format": {"type": "json_schema", "json_schema": _IDENTITY_SCHEMA}},
        timeout=90,
    )
    OQ.raise_for_openai_response(resp)
    body = resp.json()
    usage = body.get("usage", {})
    OQ.log_usage(model, "cluster_identity", usage)
    return json.loads(body["choices"][0]["message"]["content"]), usage


def build(trees=DEFAULT_TREES, limit=None, dry_run=False, model=None, resume=False,
         resume_path=None, log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE3)
    freeze.verify_inputs(paths.STAGE4)

    if not dry_run:
        from utils.config import Config
        cfg = Config()
        api_key = cfg.get_openai_api_key()
        model = model or cfg.get_openai_model("cluster_identity")
    else:
        api_key, model = None, (model or "(dry-run)")

    from utils import openai_quota as OQ

    rows: list[dict] = []
    done: set[tuple[str, int]] = set()
    if resume:
        p = resume_path or (paths.STAGE3 / "cluster_identity.parquet")
        if p.exists():
            prev = pd.read_parquet(p)
            rows = prev.to_dict("records")
            done = {(str(r["tree_id"]), int(r["cluster_id"])) for r in rows}
            log(f"↻ --resume：讀到既有 {len(rows)} 群，將跳過已完成的、只跑剩下的")
        else:
            log("⚠️ --resume 但找不到既有 cluster_identity.parquet，視同全新執行")

    n_new, total_usage = 0, {"prompt_tokens": 0, "completion_tokens": 0}
    paused = False
    for tree_id in trees:
        if paused:
            break
        profiles = _cluster_profiles(tree_id)
        quant = _quant_profile(tree_id)
        cofail = _co_fail_lookup(tree_id)

        for cid in sorted(profiles):
            if (tree_id, cid) in done:
                continue
            if limit is not None and n_new >= limit:
                break
            comp = profiles[cid]
            prompt = build_prompt(tree_id, comp, quant.get(cid, {}), cofail.get(cid, set()))

            if dry_run:
                log(f"\n{'='*70}\n{tree_id} 群{cid}\n{'='*70}")
                log(prompt)
                story = {"identity_label": "(dry-run)", "mechanism_note": "(dry-run)",
                        "performance_pattern": "(dry-run)", "caveat": "(dry-run)"}
            else:
                t0 = time.time()
                try:
                    story, usage = _call_llm(prompt, model, api_key, est_tokens=EST_TOKENS_PER_CALL)
                except OQ.FreeTierExhaustedError as e:
                    log(f"\n⛔ 免費額度暫停：{e}")
                    log(f"   目前累計已完成 {len(rows)} 群（本次新跑 {n_new} 群），"
                        f"將只寫入這些；額度恢復後可用 --resume 從中斷處接續")
                    paused = True
                    break
                for k in total_usage:
                    total_usage[k] += usage.get(k, 0)
                log(f"  [{tree_id}] 群{cid}｜{usage.get('total_tokens', 0)}tok {time.time()-t0:.1f}s"
                    f"｜{story['identity_label']}")

            rows.append({"tree_id": tree_id, "level": LEVEL, "cluster_id": cid,
                        "identity_label": story["identity_label"],
                        "mechanism_note": story["mechanism_note"],
                        "performance_pattern": story["performance_pattern"],
                        "caveat": story["caveat"], "model": model})
            n_new += 1
        if limit is not None and n_new >= limit:
            break

    if not rows:
        raise ValueError("沒有產生任何群——請檢查 --trees 是否有效、--limit 是否為0、"
                         "今日免費額度是否已用盡，或（--resume 時）是否已全部跑完")
    df = pd.DataFrame(rows)
    for col in ("tree_id", "level"):
        df[col] = df[col].astype("category")
    if not dry_run:
        C.validate(df, C.CLUSTER_IDENTITY, strict_columns=True)
        log(f"✓ cluster_identity 契約通過（累計 {len(df)} 群，本次新跑 {n_new} 群"
            f"{'，⚠️ 因額度暫停未跑完' if paused else ''}）")
        log(f"  本次token合計：prompt {total_usage['prompt_tokens']:,}／"
            f"completion {total_usage['completion_tokens']:,}")
    df.attrs["paused"] = paused
    df.attrs["n_new"] = n_new
    return df


def run(trees=DEFAULT_TREES, limit=None, model=None, resume=False, log=print) -> pd.DataFrame:
    df = build(trees=trees, limit=limit, dry_run=False, model=model, resume=resume, log=log)
    p = paths.STAGE3 / "cluster_identity.parquet"
    df.to_parquet(p, compression="zstd", index=False)
    models_used = sorted(str(m) for m in df["model"].unique()) if len(df) else [model or "(unknown)"]
    actual_model = models_used[0] if len(models_used) == 1 else "|".join(models_used)
    paused = bool(df.attrs.get("paused", False))
    n_new = int(df.attrs.get("n_new", len(df)))
    side = {"produced_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "model": actual_model, "n_clusters": len(df), "n_new_this_run": n_new,
            "resumed": bool(resume), "trees": list(trees), "level": LEVEL,
            "complete": not paused, "paused_by_free_tier_quota": paused,
            "note": "LLM輸出不保證位元可複現，故不納入stage3 MANIFEST的雜湊驗證"
                    + ("。⚠️ 本次因免費額度不足中途暫停，資料不完整" if paused else "")}
    (paths.STAGE3 / "cluster_identity_meta.json").write_text(
        json.dumps(side, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"→ cluster_identity.parquet  累計{len(df)}群（本次新增{n_new}群）, "
        f"{p.stat().st_size/1024:.0f} KB" + ("  ⚠️ 不完整（額度暫停）" if paused else ""))
    return df


def _report(df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 70)
    log("cluster_identity · 驗收摘要")
    log("=" * 70)
    for t, g in df.groupby("tree_id", observed=True):
        log(f"\n[{t}]（{len(g)} 群）")
        for r in g.sort_values("cluster_id").itertuples():
            log(f"  群{r.cluster_id}：{r.identity_label}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.cluster_identity")
    ap.add_argument("--trees", nargs="+", default=list(DEFAULT_TREES))
    ap.add_argument("--limit", type=int, help="只跑前N群（試水溫用）")
    ap.add_argument("--dry-run", action="store_true", help="只印prompt，不呼叫API、不花錢")
    ap.add_argument("--model", help="覆寫模型（預設讀 config.ini 的 cluster_identity_model）")
    ap.add_argument("--resume", action="store_true",
                    help="接續跑：讀既有cluster_identity.parquet，跳過已完成的群")
    a = ap.parse_args(argv)

    if a.dry_run:
        build(trees=a.trees, limit=a.limit, dry_run=True, model=a.model, resume=a.resume)
        return 0
    df = run(trees=a.trees, limit=a.limit, model=a.model, resume=a.resume)
    _report(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
