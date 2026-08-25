# -*- coding: utf-8 -*-
"""階段1 · 標記（W-07：分位計算 + 等級判定 + 尾端寬鬆硬篩）

輸入 ← `_frozen/stage0/candidate_index.parquet`
        `_frozen/stage1/strategy_scan.parquet`（stage1_scan.py 的原始指標）
輸出 → `_frozen/stage1/strategy_marks.parquet`

本階段把 `strategy_scan.py` 掃出來的**原始數字**轉成**等級/判定**，是 Agent1
快篩真正會用到的欄位（研究部 v9：「階段1 從『篩選關卡』改為『標記關卡』」）。

⚠️ **斷循環鐵則**：本階段不得產出任何需要 regime 的欄位——全部是整段統計量
或市場內分位數，沒有牛熊標籤。

⚠️ **stability_grade 的公式是本階段的解讀，非架構文件的精確定義**：
   GateC 原文只說「算高原寬度分數（`bucket_F1/F2` 的 std、`deltaC_summary`
   的 iqr，小=穩)」，沒有給精確公式。本實作解讀為兩個敏感度分量：
     - bucket 敏感度：同一個 (market, F1_factor, F2_factor, C_id, V) 換
       F1_band/F2_band 時，CAGR 的標準差（換分位桶結果會不會差很多）
     - deltaC 敏感度：同一個 f_combo（即 F1×F2 固定）換 C_id 時，CAGR 的
       標準差（固定F只換C，結果會不會差很多——這正是 Phase 3 C_ranking
       想回答的問題，這裡直接從候選池自己算，不依賴 phase3 的 CSV 輸出）
   兩者各自轉市場內分位、取平均、切三段。若之後與老師確認出更精確的公式，
   改這裡即可，不影響其他欄位。

尾端寬鬆硬篩（研究部 v9 · 「一刀半」）：
  刀1：低 effective_n ＋ 高 rotation_score → 淘汰（靠單一飆股的假貨；
       ⚠️ 方向見 SDD DD-09——不輪動=rotation_score高=靠運氣，不是低）
  刀2：empty_ratio 過高 → 淘汰（明確空手比例高，非 coverage_ratio，見 DD-12）
  預期淘汰 < 5%（Phase 1-4 已經把該篩的都篩過了，這裡只抓資料品質的漏網之魚）

用法：
    cd code
    python -m research.stage1_marks
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, paths

#: 尾端硬篩的分位門檻（P95+，見 DD-09：rotation_score 62.4% 的值恰為 0，
#: 低於 P62 的切點會完全退化，必須用高分位或絕對值）
ROTATION_HIGH_PCT = 0.95      # rotation_score 高於市場內 P95 視為「不輪動」
EFFN_LOW_PCT = 0.10           # effective_n 低於市場內 P10 視為「集中」
EMPTY_HIGH_PCT = 0.95         # empty_ratio 高於市場內 P95 視為「空手期過長」

#: 三段式切法的分位邊界（市場內），取兩端 + 中間
TERTILE = (1 / 3, 2 / 3)


def _pct_rank(s: pd.Series) -> pd.Series:
    """市場內分位（0-100），與其餘 `*_pct` 欄位一致的定義。"""
    return s.rank(pct=True) * 100.0


def _tertile_grade(pct: pd.Series, labels: tuple[str, str, str]) -> pd.Series:
    """依市場內分位切三段：< 1/3 → labels[0]，1/3~2/3 → labels[1]，> 2/3 → labels[2]。"""
    lo, hi = TERTILE[0] * 100, TERTILE[1] * 100
    return pd.cut(pct, bins=[-0.1, lo, hi, 100.1], labels=list(labels)).astype(str)


def build(log=print) -> pd.DataFrame:
    freeze.verify_inputs(paths.STAGE1)
    idx = pd.read_parquet(paths.STAGE0 / "candidate_index.parquet")
    scan = pd.read_parquet(paths.STAGE1 / "strategy_scan.parquet")
    months = pd.read_parquet(paths.STAGE1 / "returns_monthly.parquet",
                             columns=["strategy_uid", "ret"])
    log(f"讀入 candidate_index {len(idx):,} 列、strategy_scan {len(scan):,} 列")

    df = idx.merge(
        scan.drop(columns=["market"]), on=C.PK, how="inner", validate="one_to_one")
    if len(df) != len(idx):
        raise AssertionError(
            f"candidate_index 與 strategy_scan join 後列數不符（{len(df)} vs {len(idx)}）"
            f"——階段0/1 是否用了不同批次的候選池？")
    df["market"] = df["market"].astype(str)

    # ---------------------------------------------------------- C-3：分位與形態
    g = df.groupby("market")
    df["cagr_pct"] = g["CAGR"].transform(_pct_rank)
    # MDD 是負數，絕對值越小（越淺）分位應該越高，故對 -max_drawdown 取分位
    df["mdd_pct"] = g["max_drawdown"].transform(lambda s: _pct_rank(-s))

    return_shape_pct = g["annual_ret_std"].transform(_pct_rank)   # 波動大=分位高
    df["return_shape"] = _tertile_grade(return_shape_pct, ("穩定爬升", "中等", "大起大落"))
    risk_shape_pct = g["max_drawdown"].transform(lambda s: _pct_rank(-s))  # 越淺分位越高
    df["risk_shape"] = _tertile_grade(risk_shape_pct, ("深回撤", "中等", "淺回撤"))

    # ---------------------------------------------------------- 關卡A：可信度
    effn_pct = g["effective_n"].transform(_pct_rank)
    top1_pct_inv = g["top1_share"].transform(lambda s: _pct_rank(-s))   # top1_share低=好
    cred_score = (effn_pct + top1_pct_inv) / 2.0
    df["credibility_score_pct"] = cred_score
    df["credibility_grade"] = _tertile_grade(cred_score, C.CREDIBILITY_GRADES)
    log(f"  credibility_grade 分布: {df.credibility_grade.value_counts().to_dict()}")

    # ---------------------------------------------------------- 關卡B：穩健度
    # bucket 敏感度：同一 (market, F1, F2, C_id, V)，跨 F1_band/F2_band 的 CAGR std
    bucket_key = ["market", "F1_factor", "F2_factor", "C_id", "V"]
    bucket_std = df.groupby(bucket_key)["CAGR"].transform(
        lambda s: s.std(ddof=0) if len(s) > 1 else np.nan)
    # deltaC 敏感度：同一 f_combo（F固定），跨 C_id 的 CAGR std
    deltac_std = df.groupby(["market", "f_combo", "V"])["CAGR"].transform(
        lambda s: s.std(ddof=0) if len(s) > 1 else np.nan)

    bucket_pct = df.assign(_x=bucket_std).groupby("market")["_x"].transform(
        lambda s: _pct_rank(s) if s.notna().any() else pd.Series(np.nan, index=s.index))
    deltac_pct = df.assign(_x=deltac_std).groupby("market")["_x"].transform(
        lambda s: _pct_rank(s) if s.notna().any() else pd.Series(np.nan, index=s.index))
    stability_score = pd.concat([bucket_pct, deltac_pct], axis=1).mean(axis=1, skipna=True)
    # 兩個分量都拿不到（沒有可比較的兄弟策略）時才留 NaN，不強行分級
    has_any = bucket_std.notna() | deltac_std.notna()
    df["stability_grade"] = np.where(
        has_any, _tertile_grade(stability_score, C.STABILITY_GRADES), None)
    log(f"  stability_grade 分布: {pd.Series(df.stability_grade).value_counts(dropna=False).to_dict()}")

    # ---------------------------------------------------------- 尾端寬鬆硬篩
    effn_low_cut = g["effective_n"].transform(lambda s: s.quantile(EFFN_LOW_PCT))
    rot_high_cut = g["rotation_score"].transform(lambda s: s.quantile(ROTATION_HIGH_PCT))
    empty_high_cut = g["empty_ratio"].transform(lambda s: s.quantile(EMPTY_HIGH_PCT))

    blade1 = (df["effective_n"] <= effn_low_cut) & (df["rotation_score"] >= rot_high_cut)
    blade2 = df["empty_ratio"] >= empty_high_cut.clip(lower=1e-6)   # 全池同值(如全0)時不誤殺

    reason = np.select(
        [blade1 & blade2, blade1, blade2],
        ["低EffN+不輪動+空手過長", "低EffN+不輪動(靠單一飆股)", "空手期過長"],
        default=None)
    df["drop_reason"] = reason
    df["is_usable"] = ~(blade1 | blade2)

    drop_rate = 1 - df["is_usable"].mean()
    log(f"  尾端硬篩：淘汰 {(~df['is_usable']).sum()} / {len(df)} ({drop_rate:.2%})"
        f"　{'✅ 符合預期(<5%)' if drop_rate < 0.05 else '⚠️ 超出預期的<5%，需查原因'}")
    for m, sub in df.groupby("market"):
        d = 1 - sub["is_usable"].mean()
        log(f"    [{m}] 淘汰 {(~sub['is_usable']).sum()}/{len(sub)} ({d:.2%})")

    # ---------------------------------------------------------- W-08：資料品質防線
    # 兩個訊號OR合併（見 contracts.py 常數註解——只用單日門檻時，深掃驗出的283個
    # 實質CAGR灌水策略只抓到6個，因異常股在分散組合裡單日被稀釋，但累積到月
    # 報酬仍會超標，故用月報酬門檻補上，門檻本身是拿深掃的CAGR_inflation_pp
    # 反推校準過的，非拍腦袋）：
    #   A. 單一策略NAV的單日跳動 ≥ PRICE_JUMP_EXTREME（極端單一交易日案例）
    #   B. 單一策略月報酬 ≥ MONTHLY_JUMP_EXTREME（異常股票被稀釋後仍看得到的訊號）
    # 只標記、不淘汰——data_glitch=True 不影響 is_usable，供 Agent1/T2 快篩時
    # 自行決定要不要排除，也供 diagnose_price_anomalies.py 交叉核對（見該模組）。
    max_monthly_ret = months.groupby("strategy_uid")["ret"].max().rename("max_monthly_ret")
    df = df.merge(max_monthly_ret, on="strategy_uid", how="left")
    glitch_daily = df["max_daily_ret"] >= C.PRICE_JUMP_EXTREME
    glitch_monthly = df["max_monthly_ret"] >= C.MONTHLY_JUMP_EXTREME
    df["data_glitch"] = (glitch_daily | glitch_monthly).fillna(False)
    n_glitch = int(df["data_glitch"].sum())
    log(f"  data_glitch（單日≥{C.PRICE_JUMP_EXTREME:.0%} 或 單月≥{C.MONTHLY_JUMP_EXTREME:.0%}）："
        f"{n_glitch} / {len(df)} ({n_glitch/len(df):.2%})"
        f"　[單日刀命中{int(glitch_daily.fillna(False).sum())}／單月刀命中"
        f"{int(glitch_monthly.fillna(False).sum())}]")

    out = df[[C.PK, "market"] + [c.name for c in C.STRATEGY_MARKS.columns
                                 if c.name not in (C.PK, "market")]].copy()
    out["market"] = out["market"].astype("category")
    for cat_col in ("return_shape", "risk_shape", "credibility_grade"):
        out[cat_col] = out[cat_col].astype("category")
    out["stability_grade"] = pd.Categorical(out["stability_grade"], categories=C.STABILITY_GRADES)
    out["is_usable"] = out["is_usable"].astype(bool)
    out["data_glitch"] = out["data_glitch"].astype(bool)

    C.validate(out, C.STRATEGY_MARKS, strict_columns=True)
    log("✓ strategy_marks 契約通過")
    return out


def run(log=print) -> pd.DataFrame:
    out = build(log)
    p = paths.STAGE1 / "strategy_marks.parquet"
    out.to_parquet(p, compression="zstd", index=False)
    # ⚠️ 2026-08-25 code review 修正：manifest 寫進獨立子目錄 `_marks/`，不寫
    # `paths.STAGE1` 根目錄——`stage1_scan.py` 也寫那裡，兩者互寫會覆蓋彼此的
    # manifest（只剩最後跑的那個受雜湊保護），導致另一邊的產物完全脫離凍結驗證。
    # 比照 `stage1_mktcap.py` 已有的 `_mktcap/` 前例：manifest 位置獨立，但實際
    # 產物檔案（strategy_marks.parquet）仍留在 STAGE1 根目錄，不移動、不影響
    # 任何下游的讀取路徑。
    freeze.write_manifest(
        "stage1_marks", paths.STAGE1 / "_marks",
        inputs=[paths.STAGE0 / "candidate_index.parquet",
               paths.STAGE1 / "strategy_scan.parquet",
               paths.STAGE1 / "returns_monthly.parquet"],
        outputs=[p],
        params={"rotation_high_pct": ROTATION_HIGH_PCT, "effn_low_pct": EFFN_LOW_PCT,
               "empty_high_pct": EMPTY_HIGH_PCT, "tertile": list(TERTILE),
               "price_jump_extreme": C.PRICE_JUMP_EXTREME,
               "monthly_jump_extreme": C.MONTHLY_JUMP_EXTREME},
        notes="兩把刀方向：低effective_n+高rotation_score（不輪動）；empty_ratio過高。"
              "stability_grade 公式為本階段解讀，見 module docstring。"
              "W-08：data_glitch=單日跳動≥300%或單月報酬≥100%（OR合併，只標記不淘汰，"
              "門檻用diagnose_price_anomalies的CAGR_inflation_pp反推校準，見contracts.py）",
    )
    log(f"→ strategy_marks.parquet  {len(out):,} 列, {p.stat().st_size/1024:.0f} KB")
    return out


def _report(df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 62)
    log("階段1 標記 · 驗收報告")
    log("=" * 62)
    log(f"is_usable: {df.is_usable.sum():,} / {len(df):,} ({df.is_usable.mean():.1%})")
    log(f"data_glitch: {df.data_glitch.sum():,} / {len(df):,} ({df.data_glitch.mean():.2%})")
    log(f"\ndrop_reason 分布:\n{df.drop_reason.value_counts(dropna=False).to_string()}")
    log(f"\ncredibility_grade × market:\n"
        f"{pd.crosstab(df.market, df.credibility_grade).to_string()}")
    log(f"\nstability_grade × market:\n"
        f"{pd.crosstab(df.market, df.stability_grade, dropna=False).to_string()}")
    log(f"\nreturn_shape × market:\n{pd.crosstab(df.market, df.return_shape).to_string()}")
    log(f"\nrisk_shape × market:\n{pd.crosstab(df.market, df.risk_shape).to_string()}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.stage1_marks")
    ap.parse_args(argv)
    out = run()
    _report(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
