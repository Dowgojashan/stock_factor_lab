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
    python -m research.cluster_story --resume           # 額度暫停後，接續跑剩下的對
                                                          # （讀既有cluster_story.parquet，
                                                          #   跳過已完成的pair，只打尚未
                                                          #   跑過的；--limit在此時代表
                                                          #   「這次最多再打幾對新的」）
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

#: 免費額度預留比例：用到 95% 就踩煞車，留餘裕給正在進行中的請求，
#: 避免剛好卡在邊界被 OpenAI 擋掉（見 utils.openai_quota.check_free_tier_budget）
RESERVE_RATIO = 0.05

#: 單次呼叫的預估 token 數，供**呼叫前**的額度檢查用。
#: 🔄 2026-08-30（H-24換模型後重測）：gpt-5 實測3對 XM_normal（含1對「低」互補、
#: 2對「中」）平均每對3,567 tok（prompt~1,370／completion~2,197），比舊模型sol
#: 的每對1,856 tok高出約92%——gpt-5的回覆明顯更詳盡。取4,000留餘裕。
#: 這只是煞車用的估計值，實際用量以回應的 usage 為準；84對全量預估約
#: 84×3,567≈30萬tok，仍遠低於gpt-5標準池1M/日的額度。
EST_TOKENS_PER_CALL = 4_000


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


def _call_llm(prompt: str, model: str, api_key: str,
              *, est_tokens: int = 0) -> tuple[dict, dict]:
    """回傳 (story, usage)。

    呼叫**前**先查今日免費額度夠不夠（`check_free_tier_budget`，額度不足或模型不在
    免費名單會 raise `FreeTierExhaustedError` 暫停）；呼叫**後**把實際用量寫進帳本。
    """
    import requests
    from utils import openai_quota as OQ
    # 暫停機制：額度不足時在這裡就擋下，不會真的送出請求
    OQ.check_free_tier_budget(model, estimated_tokens=est_tokens,
                              reserve_ratio=RESERVE_RATIO)
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
    usage = body.get("usage", {})
    OQ.log_usage(model, "cluster_story", usage)
    return json.loads(body["choices"][0]["message"]["content"]), usage


def build(trees=DEFAULT_TREES, limit=None, dry_run=False, model=None, resume=False,
         resume_path=None, log=print) -> pd.DataFrame:
    """`resume_path`：--resume 要讀的既有產物路徑，預設是正式輸出位置
    （`_frozen/stage3/cluster_story.parquet`）。開放這個參數只為了讓測試能指向
    temp目錄、不必真的讀寫production路徑。
    """
    freeze.verify_inputs(paths.STAGE3)
    freeze.verify_inputs(paths.STAGE4)

    if not dry_run:
        from utils.config import Config
        cfg = Config()
        api_key = cfg.get_openai_api_key()
        model = model or cfg.get_openai_model("cluster_story")
    else:
        api_key, model = None, (model or "(dry-run)")

    from utils import openai_quota as OQ

    # --resume：讀既有產物，跳過已經打過的pair，只跑剩下的——不然「暫停」就只是
    # 換句話說的「整批重來」，跟原本要解決的問題（84對跑到第80對爆掉不能整批白跑）
    # 是同一件事的兩半，只做前半（部分保存）不做後半（真的接得上）沒有意義。
    rows: list[dict] = []
    done_pairs: set[tuple[str, int, int]] = set()
    if resume:
        p = resume_path or (paths.STAGE3 / "cluster_story.parquet")
        if p.exists():
            prev = pd.read_parquet(p)
            rows = prev.to_dict("records")
            done_pairs = {(str(r["tree_id"]), int(r["cluster_a"]), int(r["cluster_b"]))
                         for r in rows}
            log(f"↻ --resume：讀到既有 {len(rows)} 對，將跳過已完成的、只跑剩下的")
        else:
            log("⚠️ --resume 但找不到既有 cluster_story.parquet，視同全新執行")

    n_new, total_usage = 0, {"prompt_tokens": 0, "completion_tokens": 0}
    paused = False
    for tree_id in trees:
        if paused:
            break
        profiles = _cluster_profiles(tree_id)
        corr = pd.read_parquet(paths.STAGE3 / f"cluster_corr_matrix_{tree_id}.parquet")
        corr.columns = corr.columns.astype(int)
        corr.index = corr.index.astype(int)
        cofail = _co_fail_lookup(tree_id)

        for a_id, b_id in itertools.combinations(sorted(profiles), 2):
            if (tree_id, a_id, b_id) in done_pairs:
                continue
            if limit is not None and n_new >= limit:
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
                try:
                    story, usage = _call_llm(prompt, model, api_key,
                                             est_tokens=EST_TOKENS_PER_CALL)
                except OQ.FreeTierExhaustedError as e:
                    # 暫停機制：免費額度不足時停在這裡，**已完成的部分照樣回傳**
                    # （原本設計是全部跑完才組 DataFrame，中途中斷會整批白跑；
                    #   84對要跑十幾分鐘，跑到第80對才爆掉卻全丟是不能接受的）
                    log(f"\n⛔ 免費額度暫停：{e}")
                    log(f"   目前累計已完成 {len(rows)} 對（本次新跑 {n_new} 對），"
                        f"將只寫入這些；額度恢復後可用 --resume 從中斷處接續")
                    paused = True
                    break
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
            n_new += 1
        if limit is not None and n_new >= limit:
            break

    if not rows:
        # limit=0 或 trees 給空的時候會走到這；空 DataFrame 沒有欄位，
        # 後面 df["tree_id"] 會 KeyError，明確擋下比讓它炸在型別轉換好懂
        # （額度一開始就不足而一對都沒跑成，也會走到這，訊息同樣說得通；
        #   --resume 且該次trees/limit範圍內的pair全部早就跑完，也會走到這）
        raise ValueError("沒有產生任何群對——請檢查 --trees 是否有效、--limit 是否為0、"
                         "今日免費額度是否已用盡，或（--resume 時）是否已全部跑完")
    df = pd.DataFrame(rows)
    for col in ("tree_id", "level", "complementarity"):
        df[col] = df[col].astype("category")
    df["co_fail"] = df["co_fail"].astype(bool)
    if not dry_run:
        C.validate(df, C.CLUSTER_STORY, strict_columns=True)
        log(f"✓ cluster_story 契約通過（累計 {len(df)} 對，本次新跑 {n_new} 對"
            f"{'，⚠️ 因額度暫停未跑完' if paused else ''}）")
        log(f"  本次token合計：prompt {total_usage['prompt_tokens']:,}／"
            f"completion {total_usage['completion_tokens']:,}")
    df.attrs["paused"] = paused
    df.attrs["n_new"] = n_new
    return df


def run(trees=DEFAULT_TREES, limit=None, model=None, resume=False, log=print) -> pd.DataFrame:
    df = build(trees=trees, limit=limit, dry_run=False, model=model, resume=resume, log=log)
    p = paths.STAGE3 / "cluster_story.parquet"
    df.to_parquet(p, compression="zstd", index=False)
    # 側錄的 model 欄位：resume 可能跨次用不同模型，回報所有出現過的模型而非只取
    # 第一列——原本 `iloc[0]` 在非resume情境下沒問題（單一模型跑到底），但resume
    # 後若中途換過模型，只記第一列會漏掉事實，稽核用途就失去意義。
    models_used = sorted(str(m) for m in df["model"].unique()) if len(df) else [model or "(unknown)"]
    actual_model = models_used[0] if len(models_used) == 1 else "|".join(models_used)
    # ⚠️ 不呼叫 freeze.write_manifest：那會覆蓋 stage3 自己的 MANIFEST.json，
    # 破壞 DD-08 凍結鏈（與 stage2c 當初選擇獨立目錄是同一個理由）。
    # cluster_story 是 stage3 的**附加**產物、且含不可完全複現的LLM輸出，
    # 不納入 stage3 的雜湊驗證範圍，改自帶一份側錄。
    paused = bool(df.attrs.get("paused", False))
    n_new = int(df.attrs.get("n_new", len(df)))
    side = {"produced_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "model": actual_model, "n_pairs": len(df), "n_new_this_run": n_new,
            "resumed": bool(resume),
            "trees": list(trees), "level": LEVEL,
            "complementarity_cuts": C.COMPLEMENTARITY_CUTS,
            # ⚠️ 部分完成必須記在側錄裡：否則下次看到這個檔案會誤以為84對全跑完了，
            # 而它其實可能只有前30對——這種靜默的不完整正是本專案踩過的錯誤類型
            "complete": not paused,
            "paused_by_free_tier_quota": paused,
            "note": "LLM輸出不保證位元可複現，故不納入stage3 MANIFEST的雜湊驗證"
                    + ("。⚠️ 本次因免費額度不足中途暫停，資料不完整" if paused else "")}
    (paths.STAGE3 / "cluster_story_meta.json").write_text(
        json.dumps(side, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"→ cluster_story.parquet  累計{len(df)}對（本次新增{n_new}對）, "
        f"{p.stat().st_size/1024:.0f} KB"
        + ("  ⚠️ 不完整（額度暫停）" if paused else ""))
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
    ap.add_argument("--resume", action="store_true",
                    help="接續跑：讀既有cluster_story.parquet，跳過已完成的pair，"
                         "只跑剩下的（--limit在此時代表本次最多再打幾對新的）")
    a = ap.parse_args(argv)

    if a.dry_run:
        build(trees=a.trees, limit=a.limit, dry_run=True, model=a.model, resume=a.resume)
        return 0
    df = run(trees=a.trees, limit=a.limit, model=a.model, resume=a.resume)
    _report(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
