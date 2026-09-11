# -*- coding: utf-8 -*-
"""L3 · 解釋 Agent（應用層開發追蹤.md §9，Phase D 重新設計，2026-09-11）

跟 `memo.py`（D1 六欄位 IC memo，`replay` 用）的根本差異：使用者明確要求
「不要只會統整，要能像學長那樣獨立分析判斷、解釋為什麼這樣操作」（§9 起因）。
`memo.py` 的鐵則「程式判決、LLM 只負責轉述」對這個目標太死板——但這不代表
放棄抗幻覺鐵則，而是照 §9.7-R17 的界線重新畫：

    ✅ **事實推論**：把兩個都餵給它的數字放在一起講出關聯（例如「策略層相關
       0.78 但股票層最大權重僅 1.76%，代表『像』是因為都暴露同一個市場，不是
       買同一批股票」）——這種推論任何人拿到同樣的數字都能覆核，跟
       `cluster_story.py`／投影片本來就在做的事是同一類。
    ❌ **價值判決**：「所以這個組合不好」「應該改用哪個方案」——沒有程式準則
       可覆核，這條線 `memo.py` 的鐵則 7 已經畫過，這裡延續。

只在 `mode="live"` 啟用（§9.7 尾段）：R15（群檔案全樣本前視）／R16（基準涵蓋
62.2%）／H8（walk-forward 每窗 k 不同，群 id 對不上）三個問題全部只存在於
`replay` 模式，`live` 模式完全不受影響——`replay` 模式繼續用 `memo.py` 的
六欄位版本，不套用這裡的邏輯。

資料組裝需要股票層級持股（`resolve_strategy_holdings` 的輸出）跟群知識庫
（`cluster_kb.build_footprint`），這兩件事目前只有 `ui.py` 走過（需要連
資料庫、載入 `MarketData`），CLI 尚未接——`cli.py` 的 `--memo` 目前對 `live`
模式仍是舊版 `memo.py`（策略層級），這是已知的範圍限制，不是遺漏。
"""
from __future__ import annotations

import json

from . import scenario as SC
from .cluster_kb import ClusterFootprint, build_footprint
from .engine import Holdings, alt_group_win_rates, historical_same_setting_windows, mechanism_consistency
from .memo import _call_llm, _fmt_alt, _fmt_diff, _fmt_perf, _fmt_reference, _fmt_violation, _pct
from .risk import RiskReport, StockConcentration
from .calibration import CalibrationResult
from .audit import diff_holdings, find_previous

RESERVE_RATIO = 0.2

_SYSTEM_PROMPT = (
    "你是量化投資系統的資深分析師（不是轉譯器）。你會拿到這一期選股結果的完整"
    "客觀資料：策略層的群知識（身份、機制、歷史績效型態）、股票層的實際持股與"
    "集中度、風控/校準判決、跟情境比對用的歷史數字。\n"
    "你的角色是**做出獨立的分析與判斷**，不是把資料轉述一遍——要能回答"
    "「這批持股是什麼性格」「為什麼這樣挑會賺錢」「現在的情況跟過去哪個時期"
    "像，那時候後來賺賠如何」這類需要綜合判斷的問題，正面地解釋這套方法為什麼"
    "可行、為什麼能賺錢，不要寫成「分散是假的」這種負面診斷語氣。\n\n"
    "但獨立判斷不等於可以杜撰。抗幻覺鐵則的界線畫在這裡：\n"
    "1. **事實推論可以，價值判決不行**。事實推論＝把提供的數字放在一起講出"
    "關聯或型態（例如「策略層相關高但股票層集中度低，代表『像』來自曝險同一個"
    "市場而非買同一批股票」），這種推論任何人拿同樣的數字都能覆核。價值判決＝"
    "「這個組合不好」「應該改用哪個候選方案」，沒有程式準則可覆核，禁止。"
    "是否更換方法屬政策層決定，不在這份分析的權責內。\n"
    "2. **禁止引用未提供的數字**，禁止杜撰個股、產業、總體經濟事件或市場情緒——"
    "包括你自己訓練資料裡可能知道、但沒有出現在下方 JSON 裡的任何事實。\n"
    "3. **禁止推翻或質疑程式給的風控/校準判決**（違規就是違規、通過就是通過）。\n"
    "4. **這是歷史資料與現況的分析，不是預測**——情境比對（scenario）呈現的是"
    "「過去發生過什麼」，禁止寫成「未來會怎樣」「可望達到」等預測性語言，也"
    "禁止用「後來證明」這類暗示你知道後續發展的說法。\n"
    "5. 若有風控違規且已被人工覆核放行，必須如實寫出違規內容跟覆核原因。\n"
    "6. 數字已事先格式化好（百分比/小數位數），請直接照抄，不要自己重新換算。\n"
    "7. **is_* 是樣本內配適值，不是預期報酬**（IS CAGR 中位數歷史上比 OOS 高"
    "1.48 倍）；`live` 模式依定義沒有 OOS，禁止用 is_* 頂替。\n"
    "8. scenario 區塊裡的情境比對**都只是 n=1 或少量歷史實現**，不是統計推論——"
    "描述型態可以，禁止寫成「所以未來也會這樣」。\n"
    "9. alternative_note 只陳述提供的替代方案數字，若替代方案數字比目前選定的好，"
    "必須把 `alternative_context` 那段定錨事實一併寫進去，禁止建議換方案、"
    "禁止寫「屬於正常波動」（研究已證實是系統性結果）。\n"
    "10. change_note 只描述新增/剔除檔數，禁止推測換股原因。"
)

_EXPLAIN_SCHEMA = {
    "name": "portfolio_explanation",
    "schema": {
        "type": "object",
        "properties": {
            "strategy_footprint_note": {
                "type": "string",
                "description": "【一之1｜策略層】這批策略落在主線樹哪些群、各群身份"
                               "（identity_label）、機制（mechanism_note）與定量特徵"
                               "（CAGR/MDD 中位、主力因子）；只能引用提供的群知識，"
                               "不能自己重新詮釋群的意義。"},
            "stock_holdings_note": {
                "type": "string",
                "description": "【一之2｜股票層，老師 9-8 原話①】幾檔不重複股票、"
                               "最大持股權重、跨策略重疊度最高的前幾檔。"},
            "mechanism_note": {
                "type": "string",
                "description": "【二｜為什麼這樣挑會賺錢】依三機制數字（稀有群超配"
                               "倍數、對 B_all 的 CAGR 超額分布與勝率、三組候選方案"
                               "的 OOS MDD 排名）正面解釋這套規則的價值來源，"
                               "只能引用提供的數字。"},
            "character_note": {
                "type": "string",
                "description": "【三｜這批的性格，老師 9-8 原話④】從主力 F1 因子曝險"
                               "（動能 vs 品質估值）跟新舊面孔比例（若有比較資料）"
                               "判斷這批持股偏向「追火」還是「一直賺」，沒有新舊面孔"
                               "資料時明講「本次未比較，只能從因子面判斷」。"},
            "concentration_risk_note": {
                "type": "string",
                "description": "【四之1/2｜集中度風險】股票層最大權重與是否違規、"
                               "特定權值股對照（若提供）；並點出策略層相關高不等於"
                               "股票層集中（把兩個層級的數字放在一起講出這個關聯，"
                               "屬於允許的事實推論）。"},
            "structure_risk_note": {
                "type": "string",
                "description": "【四之3/4/5｜結構風險】平時 vs 危機期群間相關的變化、"
                               "多樣性機制（backfill）是否實際生效、選用的幾個群彼此"
                               "結構重疊程度（story 裡的 complementarity/co_fail）。"},
            "scenario_note": {
                "type": "string",
                "description": "【五｜情境比對，老師 9-8 原話③】綜合四種比對數字"
                               "（群報酬型態最像哪一年、同設定歷史窗次實際 OOS 結果、"
                               "現在的 regime 標籤與歷史平均表現、現有股票在極端年份"
                               "的實際表現）描述現況跟過去哪些時期相似、那些時期後續"
                               "賺賠如何——只能描述型態，禁止預測。"},
            "change_note": {
                "type": "string",
                "description": "跟上一次同一組設定的執行結果相比，新增/剔除了幾檔"
                               "策略；沒有上一筆可比就明講「這是第一次執行」。"},
            "alternative_note": {
                "type": "string",
                "description": "同一格設定下，其他候選方案的檔數與 IS 績效數字，"
                               "並附上 alternative_context 的定錨事實，不建議換方案。"},
            "caveat": {
                "type": "string",
                "description": "本份分析的限制：情境比對是少量歷史實現不是統計推論、"
                               "is_* 非預期報酬、群知識庫止於 2025 年底、正式模式無 "
                               "OOS 等，只能根據提供的資訊寫。"},
        },
        "required": ["strategy_footprint_note", "stock_holdings_note", "mechanism_note",
                    "character_note", "concentration_risk_note", "structure_risk_note",
                    "scenario_note", "change_note", "alternative_note", "caveat"],
        "additionalProperties": False,
    },
    "strict": True,
}


def _cluster_facts(footprint: ClusterFootprint) -> list[dict]:
    out = []
    for cid in footprint.clusters_used:
        prof = footprint.profile.get(cid, {})
        ident = footprint.identity.get(cid, {})
        out.append({
            "cluster_id": cid,
            "n_selected": footprint.counts[cid],
            "n_in_universe": footprint.total_in_tree.get(cid),
            "share_of_this_portfolio": _pct(footprint.share(cid)),
            "identity_label": ident.get("identity_label"),
            "mechanism_note": ident.get("mechanism_note"),
            "performance_pattern": ident.get("performance_pattern"),
            "CAGR_median": _pct(prof["CAGR_median"]) if prof.get("CAGR_median") is not None else None,
            "MDD_median": _pct(prof["MDD_median"]) if prof.get("MDD_median") is not None else None,
            "top1_F1": prof.get("top1_F1"),
            "top1_F1_pct": _pct(prof["top1_F1_pct"]) if prof.get("top1_F1_pct") is not None else None,
            "top1_C_source": prof.get("top1_C_source"),
            "pct_years_positive": _pct(prof["pct_years_positive"]) if prof.get("pct_years_positive") is not None else None,
            "best_year": prof.get("best_year"), "best_year_ret": _pct(prof["best_year_ret"]) if prof.get("best_year_ret") is not None else None,
            "worst_year": prof.get("worst_year"), "worst_year_ret": _pct(prof["worst_year_ret"]) if prof.get("worst_year_ret") is not None else None,
        })
    return out


def _rarest_cluster_protection(footprint: ClusterFootprint, n_universe: int | None) -> dict | None:
    """機制一：稀有但真正不同的群，實際被保障超配了多少倍（依這次真正選中的
    members 現算，不是套用投影片上寫死的 TW 數字）。"""
    if not footprint.clusters_used or not n_universe:
        return None
    total_selected = sum(footprint.counts.values())
    rows = []
    for cid in footprint.clusters_used:
        n_univ = footprint.total_in_tree.get(cid)
        if not n_univ:
            continue
        share = n_univ / n_universe
        expected = total_selected * share
        actual = footprint.counts[cid]
        rows.append({
            "cluster_id": cid, "share_in_universe": _pct(share),
            "n_in_universe": n_univ, "expected_if_random": round(expected, 2),
            "actual_selected": actual,
            "overweight_multiple": round(actual / expected, 1) if expected > 0 else None,
        })
    if not rows:
        return None
    rarest = min(rows, key=lambda r: r["n_in_universe"])
    return {"per_cluster": rows, "rarest_cluster": rarest}


def _structure_facts(footprint: ClusterFootprint) -> dict:
    used = footprint.clusters_used
    pairs_normal, pairs_crisis = [], []
    for i, a in enumerate(used):
        for b in used[i + 1:]:
            pairs_normal.append(footprint.pair_corr(a, b, crisis=False))
            c = footprint.pair_corr(a, b, crisis=True)
            if c is not None:
                pairs_crisis.append(c)
    story_rows = footprint.story.to_dict("records") if not footprint.story.empty else []
    co_fail_rows = {cid: footprint.co_fail.get(cid) for cid in used if footprint.co_fail.get(cid)}
    return {
        "clusters_used": used,
        "avg_pair_corr_normal": round(sum(pairs_normal) / len(pairs_normal), 4) if pairs_normal else None,
        "avg_pair_corr_crisis": round(sum(pairs_crisis) / len(pairs_crisis), 4) if pairs_crisis else None,
        "pair_story": story_rows,
        "co_fail_regimes": co_fail_rows,
    }


def assemble_facts(holdings: Holdings, risk: RiskReport, calib: CalibrationResult, *,
                   stock_detail, stock_concentration: StockConcentration,
                   market_dates: dict[str, str] | None = None,
                   footprint: ClusterFootprint | None = None,
                   md_map: dict | None = None,
                   face_comparison=None) -> dict:
    """組裝解釋 agent 要用的全部事實。`holdings.config.mode` 必須是 `"live"`
    ——R15/R16/H8 三個問題只在 `replay` 模式存在，這裡不接受 `replay`。
    """
    if holdings.config.mode != "live":
        raise ValueError(
            f"解釋 agent 只支援 mode='live'（見應用層開發追蹤.md §9.7），"
            f"目前是 mode={holdings.config.mode!r}——replay 請用 memo.generate()")

    cfg = holdings.config
    if footprint is None:
        footprint = build_footprint(cfg.market, holdings.members)

    overlap = (stock_detail.drop_duplicates(["strategy_uid", "stock_symbol"])
              .groupby(["stock_symbol", "company_name"])["strategy_uid"]
              .nunique().reset_index(name="n_strategies")
              .sort_values("n_strategies", ascending=False).head(10))

    war = SC.weighted_annual_returns(footprint)
    stock_analog = {}
    if war:
        best_year = max(war, key=war.get)
        worst_year = min(war, key=war.get)
        if md_map:
            top_symbols = [r["stock_symbol"] for r in stock_concentration.top(20)]
            stock_analog = {
                "best_year": SC.stock_level_analog(md_map, top_symbols, best_year),
                "worst_year": SC.stock_level_analog(md_map, top_symbols, worst_year),
            }

    regime = None
    if market_dates:
        # as_of 用快時鐘實際解析到的交易日（每個市場自己的），不是 IS 結束日
        regime = SC.regime_snapshot(cfg.market, market_dates, risk.regime_avg_ret)

    facts = {
        "mode": cfg.mode, "market": cfg.market, "group": cfg.group,
        "ratio": cfg.ratio, "allocation": cfg.allocation,
        "window_info": holdings.window_info, "n_members": holdings.n_members,
        "validation": holdings.validation,
        "performance": _fmt_perf(holdings.performance, holdings.has_oos),
        "reference_oos_distribution": _fmt_reference(holdings.reference_oos),

        "section1_strategy_footprint": {
            "tree_id": footprint.tree_id, "clusters": _cluster_facts(footprint),
        },
        "section1_stock_holdings": {
            "n_unique_stocks": stock_concentration.n_unique_stocks,
            "n_strategies": stock_concentration.n_strategies,
            "n_empty_strategies": stock_concentration.n_empty_strategies,
            "max_stock_weight": _pct(stock_concentration.max_stock_weight),
            "max_stock_symbol": stock_concentration.max_stock_symbol,
            "top15": [{**r, "weight": _pct(r["weight"])} for r in stock_concentration.top(15)],
            "most_overlapped": overlap.to_dict("records"),
        },

        "mechanism": {
            "rarest_cluster_protection": _rarest_cluster_protection(
                footprint, holdings.tree_info.get("n_universe")),
            "consistency_and_drawdown": mechanism_consistency(cfg.market),
        },

        "risk": {
            "max_single_weight": _pct(risk.max_single_weight),
            "single_stock_cap": _pct(cfg.single_stock_cap),
            "max_cluster_share": _pct(risk.max_cluster_share),
            "cluster_cap": _pct(cfg.cluster_cap),
            "n_clusters_covered": risk.n_clusters_covered,
            "violations": [_fmt_violation(v) for v in risk.violations],
            "stock_level_violations": [_fmt_violation(v) for v in stock_concentration.violations],
            "factor_exposure_F1": {k: _pct(v) for k, v in risk.factor_exposure_f1.items()},
            "n_backfilled": holdings.performance.get("n_backfilled"),
            "target_total": holdings.performance.get("target_total"),
        },
        "structure": _structure_facts(footprint),

        "character": {
            "factor_exposure_F1": {k: _pct(v) for k, v in risk.factor_exposure_f1.items()},
            "new_vs_old_faces": (
                {"persistence_rate": _pct(face_comparison.persistence_rate),
                 "turnover_rate": _pct(face_comparison.turnover_rate),
                 "n_old": len(face_comparison.old_faces),
                 "n_new": len(face_comparison.new_faces),
                 "n_exited": len(face_comparison.exited)}
                if face_comparison is not None else None),
        },

        "scenario": {
            "weighted_annual_returns_by_year": {str(y): _pct(v) for y, v in war.items()},
            "historical_same_setting_windows": historical_same_setting_windows(cfg),
            "regime": regime,
            "stock_level_analog": stock_analog,
        },

        "change_vs_previous": _fmt_diff(
            diff_holdings(holdings.members, find_previous(cfg), holdings.window_info)),
        "alternative_groups": {g: _fmt_alt(v) for g, v in holdings.alternative_groups.items()},
        "alternative_context": alt_group_win_rates(cfg.market),

        "calibration": {
            "status": calib.status,
            "oos_cagr": _pct(calib.oos_cagr) if calib.oos_cagr is not None else None,
            "oos_cagr_p10": _pct(calib.thresholds.oos_cagr_p10),
            "n_historical_cells": calib.thresholds.n_cells,
            "flagged": calib.flagged, "note": calib.note,
        },
    }
    return facts


def build_prompt(facts: dict) -> str:
    return (
        "【客觀資料 · 由程式算出，不可推翻，數字已格式化，請直接照抄】\n"
        f"{json.dumps(facts, ensure_ascii=False, indent=2, default=str)}\n\n"
        "請依給定的 JSON schema 輸出十個欄位。每個欄位都要做出實質的分析判斷"
        "（不是把數字複誦一遍），但只能根據以上資料——事實推論可以，價值判決"
        "（例如建議換方案）不行，見系統提示的鐵則 1。"
    )


def generate(holdings: Holdings, risk: RiskReport, calib: CalibrationResult, *,
            stock_detail, stock_concentration: StockConcentration,
            market_dates: dict[str, str] | None = None,
            footprint: ClusterFootprint | None = None, md_map: dict | None = None,
            face_comparison=None, dry_run: bool = True, model: str | None = None,
            purpose: str = "app_memo") -> dict:
    facts = assemble_facts(holdings, risk, calib, stock_detail=stock_detail,
                           stock_concentration=stock_concentration, market_dates=market_dates,
                           footprint=footprint, md_map=md_map, face_comparison=face_comparison)
    prompt = build_prompt(facts)

    if dry_run:
        explanation = {k: "(dry-run，未呼叫 LLM)" for k in _EXPLAIN_SCHEMA["schema"]["required"]}
    else:
        from utils.config import Config
        cfg = Config()
        api_key = cfg.get_openai_api_key()
        model = model or cfg.get_openai_model(purpose)
        explanation, _usage = _call_llm(
            prompt, model, api_key, purpose=purpose, est_tokens=len(prompt) // 3,
            system_prompt=_SYSTEM_PROMPT, schema=_EXPLAIN_SCHEMA)

    return {"prompt": prompt, "facts": facts, "explanation": explanation, "dry_run": dry_run}
