# -*- coding: utf-8 -*-
"""階段3 附加 · cluster_story（研究部 v9 的 LLM 點③，群間互補因果解釋）

輸入 ← `_frozen/stage3/`（cluster_assign / cluster_meta / cluster_corr_matrix_* /
        co_fail_regimes）+ `_frozen/stage4/strategy_map.parquet`（群的成分側寫）
輸出 → `_frozen/stage3/cluster_story.parquet`（供 T5/T13 的 explanation_text 引用）

v9 定位：**離線一次性產物**。群數只有數十（每棵樹 L1 共 8 群 → 28 對），
成本遠低於 v6 原本設想的「策略配對」版本。

---------------------------------------------------------------------------
🔴 本模組最重要的設計決定：**互補程度由程式判定，LLM 只為既定判決寫字**
---------------------------------------------------------------------------
開發前先實測了六棵樹的群間相關分布（2026-08-25），結果推翻了原本的天真設計：

  市場內幾乎沒有真正低相關的群對——TW 0.539~0.981、US 0.847~0.979。
  跨市場才有——XM 0.372~0.976，且最低的幾對**全部是台↔美**。
  更關鍵：XM 樹的 L1 **完美依市場分裂**（群1-3 全台股、群4-8 全美股、0 混合）。

若照原本想法直接問 LLM「解釋這兩群為何互補」，對一個相關 0.98 的群對，
LLM 幾乎一定會編出一套聽起來合理的經濟故事——那正是論文最不能出現的東西。
故 `complementarity`（高/中/低）改由 `contracts.COMPLEMENTARITY_CUTS` 用實際
相關值算出來，prompt 把這個**既定判決**連同「低＝不可宣稱互補」的指示一起餵給
LLM，LLM 只負責把「為什麼是這個判決」寫成人話。這與 T10 是同一套原則。

第二個防線：群的側寫（factor_type/F1/C_source/V/市場組成…）全部由程式從
strategy_map 算好餵進去，LLM 不得引用未提供的數字，也不得在差異其實是**機械性**
（例如兩群只是 v0 vs v1、或只是市場不同）時硬掰成經濟因果——這在 prompt 裡
是明文要求，因為實測顯示分群結構確實高度受 V 與 F1 家族支配。

用法：
    cd code
    python -m research.cluster_story --dry-run          # 只印prompt不呼叫API、不花錢
    python -m research.cluster_story --limit 3          # 先跑3對試水溫
    python -m research.cluster_story                    # 全跑（三棵normal樹，84對）
    python -m research.cluster_story --trees XM_normal  # 只跑跨市場那棵
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time

import pandas as pd

from . import contracts as C
from . import freeze, paths

#: 預設只跑三棵 normal 樹。crisis 樹的群間互補不是 v9 要的東西——危機期的資訊
#: 已經由 co_fail_regimes（哪些常態群在危機時塌一起）表達，且那是程式算的事實，
#: 不需要 LLM 解釋。跑 crisis 樹只會多燒錢又產出語意重複的文字。
DEFAULT_TREES = ("TW_normal", "US_normal", "XM_normal")
LEVEL = "L1"          # 與 cluster_corr_matrix / co_fail_regimes 的粒度一致


def _complementarity(corr: float) -> str:
    """程式判定互補程度（不交給LLM，理由見模組docstring）。"""
    if corr < C.COMPLEMENTARITY_CUTS["高"]:
        return "高"
    if corr < C.COMPLEMENTARITY_CUTS["中"]:
        return "中"
    return "低"


def _cluster_profiles(tree_id: str) -> dict[int, dict]:
    """每個群的客觀側寫——全部由程式從 strategy_map 算，供LLM引用。"""
    ca = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")
    sm = pd.read_parquet(paths.STAGE4 / "strategy_map.parquet")
    sub = ca[ca.tree_id == tree_id][[C.PK, f"cluster_{LEVEL}"]].rename(
        columns={f"cluster_{LEVEL}": "_cl"})
    df = sm.merge(sub, on=C.PK)
    meta = pd.read_parquet(paths.STAGE3 / "cluster_meta.parquet")
    meta = meta[(meta.tree_id == tree_id) & (meta.level == LEVEL)].set_index("cluster_id")

    out = {}
    for cid, g in df.groupby("_cl"):
        mkt = g[C.PK].str.split("::").str[0].value_counts()
        out[int(cid)] = {
            "cluster_id": int(cid),
            "n_members": int(len(g)),
            "market_mix": {k: int(v) for k, v in mkt.items()},
            # 一律濾掉 count=0 ——categorical 欄位的 value_counts 會把該群根本沒有的
            # 類別也列出來（如 "動能型": 0），列在「top_」底下會誤導LLM以為有這個成分
            "top_factor_types": {k: int(v) for k, v in
                                g.factor_type.value_counts().head(3).items() if v > 0},
            "top_F1": {k: int(v) for k, v in
                      g.F1_factor.value_counts().head(3).items() if v > 0},
            "top_C_source": {("無C" if pd.isna(k) else k): int(v)
                            for k, v in g.C_source.value_counts(dropna=False).head(3).items()
                            if v > 0},
            "V_mix": {k: int(v) for k, v in g.V.value_counts().items() if v > 0},
            "CAGR_median": round(float(g.CAGR.median()), 4),
            "MDD_median": round(float(g.max_drawdown.median()), 4),
            "smallcap_share_median": round(float(g.smallcap_share.median()), 3),
            "avg_intra_corr": (round(float(meta.loc[cid].avg_intra_corr), 4)
                              if cid in meta.index and pd.notna(meta.loc[cid].avg_intra_corr) else None),
        }
    return out


def _co_fail_lookup(tree_id: str) -> dict[int, set[int]]:
    """常態群 → 危機期與它塌進同一群的其他常態群。"""
    if not tree_id.endswith("_normal"):
        return {}
    cf = pd.read_parquet(paths.STAGE3 / "co_fail_regimes.parquet")
    key = tree_id.rsplit("_", 1)[0]
    cf = cf[(cf.tree_key == key) & (cf.level == LEVEL)]
    return {int(r.cluster_normal): {int(x) for x in r.co_fail_peers.split("|") if x}
            for r in cf.itertuples()}


_SYSTEM_PROMPT = (
    "你是量化研究系統的分析結果轉譯器。你會拿到兩群交易策略的**客觀統計側寫**，"
    "以及一個**由程式算出、不可推翻的互補程度判決**。"
    "你的工作只是把「為什麼是這個判決」寫成清楚的中文，不是自己重新判斷。\n"
    "鐵則：\n"
    "1. 禁止推翻或質疑程式給的互補程度判決。\n"
    "2. 禁止引用未提供給你的數字，禁止杜撰個股、產業、總體經濟事件。\n"
    "3. 若兩群的差異主要是**機械性的**（例如只差在估值濾網 v0/v1、只差在因子家族、"
    "或只差在所屬市場），就如實說是機械性差異，**不要包裝成深層的經濟因果故事**。\n"
    "4. 互補程度為「低」時，代表兩群高度重疊，**不可以宣稱它們能分散風險**，"
    "要明講「放在一起不會帶來分散效果」。"
)

_STORY_SCHEMA = {
    "name": "cluster_story",
    "schema": {
        "type": "object",
        "properties": {
            "mechanism_note": {
                "type": "string",
                "description": "兩群的客觀差異在哪（引用提供的側寫數字）。"
                               "若差異是機械性的就直說，不要編經濟故事。"},
            "complement_note": {
                "type": "string",
                "description": "為程式給定的互補程度判決寫說明。判決為「低」時"
                               "必須明講兩群高度重疊、放在一起不會帶來分散效果。"},
            "caveat": {
                "type": "string",
                "description": "此判讀的限制（例如相關係數只反映歷史共同期間、"
                               "群內成員異質性、樣本數少等）。只能根據提供的資訊寫。"},
        },
        "required": ["mechanism_note", "complement_note", "caveat"],
        "additionalProperties": False,
    },
    "strict": True,
}


def build_prompt(tree_id: str, a: dict, b: dict, corr: float,
                 comp: str, co_fail: bool) -> str:
    return (
        f"樹：{tree_id}（層級 {LEVEL}）\n\n"
        f"【程式判決 · 不可推翻】\n"
        f"- 群{a['cluster_id']} 與 群{b['cluster_id']} 的群間相關係數：{corr:.3f}\n"
        f"- 互補程度判決：{comp}"
        f"（判定規則：相關<{C.COMPLEMENTARITY_CUTS['高']}為高、"
        f"<{C.COMPLEMENTARITY_CUTS['中']}為中、否則為低）\n"
        f"- 危機期是否塌進同一群（co_fail）：{co_fail}\n\n"
        f"【群{a['cluster_id']} 客觀側寫】\n{json.dumps(a, ensure_ascii=False, indent=2)}\n\n"
        f"【群{b['cluster_id']} 客觀側寫】\n{json.dumps(b, ensure_ascii=False, indent=2)}\n\n"
        "請依給定的 JSON schema 輸出，只能使用以上資訊。"
    )


def _call_llm(prompt: str, model: str, api_key: str) -> tuple[dict, dict]:
    """回傳 (story, usage)。"""
    import requests
    from utils import openai_quota as OQ
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model,
             "messages": [{"role": "system", "content": _SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}],
             "response_format": {"type": "json_schema", "json_schema": _STORY_SCHEMA}},
        timeout=90,
    )
    OQ.raise_for_openai_response(resp)
    body = resp.json()
    return json.loads(body["choices"][0]["message"]["content"]), body.get("usage", {})


def build(trees=DEFAULT_TREES, limit=None, dry_run=False, model=None, log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE3)
    freeze.verify_inputs(paths.STAGE4)

    if not dry_run:
        from utils.config import Config
        cfg = Config()
        api_key = cfg.get_openai_api_key()
        model = model or cfg.get_openai_model("cluster_story")
    else:
        api_key, model = None, (model or "(dry-run)")

    rows, n_done, total_usage = [], 0, {"prompt_tokens": 0, "completion_tokens": 0}
    for tree_id in trees:
        profiles = _cluster_profiles(tree_id)
        corr = pd.read_parquet(paths.STAGE3 / f"cluster_corr_matrix_{tree_id}.parquet")
        corr.columns = corr.columns.astype(int)
        corr.index = corr.index.astype(int)
        cofail = _co_fail_lookup(tree_id)

        for a_id, b_id in itertools.combinations(sorted(profiles), 2):
            if limit is not None and n_done >= limit:
                break
            c = float(corr.loc[a_id, b_id])
            comp = _complementarity(c)
            cf = b_id in cofail.get(a_id, set())
            prompt = build_prompt(tree_id, profiles[a_id], profiles[b_id], c, comp, cf)

            if dry_run:
                log(f"\n{'='*70}\n{tree_id} 群{a_id}×群{b_id}  corr={c:.3f} 互補={comp} co_fail={cf}\n{'='*70}")
                log(prompt)
                story = {"mechanism_note": "(dry-run)", "complement_note": "(dry-run)",
                        "caveat": "(dry-run)"}
            else:
                t0 = time.time()
                story, usage = _call_llm(prompt, model, api_key)
                for k in total_usage:
                    total_usage[k] += usage.get(k, 0)
                log(f"  [{tree_id}] 群{a_id}×群{b_id} corr={c:.3f} 互補={comp} "
                    f"｜{usage.get('total_tokens', 0)}tok {time.time()-t0:.1f}s")

            rows.append({"tree_id": tree_id, "level": LEVEL,
                        "cluster_a": a_id, "cluster_b": b_id,
                        "corr": round(c, 6), "complementarity": comp, "co_fail": cf,
                        "mechanism_note": story["mechanism_note"],
                        "complement_note": story["complement_note"],
                        "caveat": story["caveat"], "model": model})
            n_done += 1
        if limit is not None and n_done >= limit:
            break

    if not rows:
        # limit=0 或 trees 給空的時候會走到這；空 DataFrame 沒有欄位，
        # 後面 df["tree_id"] 會 KeyError，明確擋下比讓它炸在型別轉換好懂
        raise ValueError("沒有產生任何群對——請檢查 --trees 是否有效、--limit 是否為0")
    df = pd.DataFrame(rows)
    for col in ("tree_id", "level", "complementarity"):
        df[col] = df[col].astype("category")
    df["co_fail"] = df["co_fail"].astype(bool)
    if not dry_run:
        C.validate(df, C.CLUSTER_STORY, strict_columns=True)
        log(f"✓ cluster_story 契約通過（{len(df)} 對）")
        log(f"  token 合計：prompt {total_usage['prompt_tokens']:,}／"
            f"completion {total_usage['completion_tokens']:,}")
    return df


def run(trees=DEFAULT_TREES, limit=None, model=None, log=print) -> pd.DataFrame:
    df = build(trees=trees, limit=limit, dry_run=False, model=model, log=log)
    p = paths.STAGE3 / "cluster_story.parquet"
    df.to_parquet(p, compression="zstd", index=False)
    # 側錄要記**實際用的**模型：`model` 參數可能是 None（表示由 build 去 config 讀），
    # 直接寫參數會變成 "(from config)" 這種沒有稽核價值的字串。這份側錄的存在意義
    # 就是替一個不可完全複現的LLM產物留下溯源紀錄，記錯模型等於失去意義。
    actual_model = str(df["model"].iloc[0]) if len(df) else (model or "(unknown)")
    # ⚠️ 不呼叫 freeze.write_manifest：那會覆蓋 stage3 自己的 MANIFEST.json，
    # 破壞 DD-08 凍結鏈（與 stage2c 當初選擇獨立目錄是同一個理由）。
    # cluster_story 是 stage3 的**附加**產物、且含不可完全複現的LLM輸出，
    # 不納入 stage3 的雜湊驗證範圍，改自帶一份側錄。
    side = {"produced_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "model": actual_model, "n_pairs": len(df),
            "trees": list(trees), "level": LEVEL,
            "complementarity_cuts": C.COMPLEMENTARITY_CUTS,
            "note": "LLM輸出不保證位元可複現，故不納入stage3 MANIFEST的雜湊驗證"}
    (paths.STAGE3 / "cluster_story_meta.json").write_text(
        json.dumps(side, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"→ cluster_story.parquet  {len(df)} 對, {p.stat().st_size/1024:.0f} KB")
    return df


def _report(df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 62)
    log("cluster_story · 驗收摘要")
    log("=" * 62)
    log(f"\n互補程度分布（程式判定）：\n{df.complementarity.value_counts().to_string()}")
    log(f"\n各樹的相關範圍：")
    for t, g in df.groupby("tree_id", observed=True):
        # ⚠️ 必須用 g["corr"] 不能用 g.corr——`corr` 撞到 DataFrame.corr 這個內建方法名，
        # 屬性寫法會取到方法本身而非欄位（2026-08-25 code review 抓到的真實bug）
        log(f"  {t}: {g['corr'].min():.3f} ~ {g['corr'].max():.3f}"
            f"｜高互補 {(g['complementarity'] == '高').sum()} 對")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.cluster_story")
    ap.add_argument("--trees", nargs="+", default=list(DEFAULT_TREES))
    ap.add_argument("--limit", type=int, help="只跑前N對（試水溫用）")
    ap.add_argument("--dry-run", action="store_true", help="只印prompt，不呼叫API、不花錢")
    ap.add_argument("--model", help="覆寫模型（預設讀 config.ini 的 cluster_story_model）")
    a = ap.parse_args(argv)

    if a.dry_run:
        build(trees=a.trees, limit=a.limit, dry_run=True, model=a.model)
        return 0
    df = run(trees=a.trees, limit=a.limit, model=a.model)
    _report(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
