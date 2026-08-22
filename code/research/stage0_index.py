# -*- coding: utf-8 -*-
"""階段 0 · 格式整備 → candidate_index.parquet

輸入 ← `_analysis_outputs_phase4/{TW,US}_L4_openSec_final_candidates.csv`（14,078 列）
        + 各 job 的 stats.parquet
輸出 → `_frozen/stage0/candidate_index.parquet`

本階段同時修掉三項「架構↔實驗」落差（見 `落差處理方案_v1.md`）：
  落差1  sharpe_ann/daily_sharpe/avg_drawdown 不在候選 CSV → join stats.parquet + 對帳斷言
  落差2  artifacts_dir 不存在、產物散在 L3(v0)/L4(v1) → 規則解析 + 存在性斷言
  落差5  台美策略字串碰撞 1,381 個 → 主鍵改為 strategy_uid = market::strategy

用法：
    cd code
    python -m research.stage0_index
"""
from __future__ import annotations

import re
import sys
import time

import pandas as pd

from . import contracts as C
from . import freeze, paths

# `MOM_qb2of3` / `NETDEBT_EBITDA_qb1of3`——因子名本身含底線，故從右側的 _qb 錨定
_RE_BAND = re.compile(r"^(?P<factor>.+)_qb(?P<band>\d+)of(?P<nbands>\d+)$")
# `C4_ROE_DYN_qmax8` / `C18_FCF_P_DYN_qmax8` / `C14_EPS_DYN_ytdavg_gt_lyavg`
_RE_C = re.compile(r"^(?P<cid>C\d+)_(?P<source>.+)_DYN_(?P<rule>.+)$")

_NULL_TOKENS = {"None", "none", "NONE", "", "nan"}


def _parse_band(token: str | None):
    """`EV_S_qb0of3` -> ('EV_S', 0, 3)；空集合 -> (None, None, None)。"""
    if token is None or str(token) in _NULL_TOKENS:
        return None, None, None
    m = _RE_BAND.match(str(token))
    if not m:
        raise ValueError(f"無法解析因子桶 token: {token!r}")
    return m["factor"], int(m["band"]), int(m["nbands"])


def _parse_c(token: str | None):
    """`C4_ROE_DYN_qmax8` -> ('C4', 'ROE', 'qmax8')；無 C -> (None, None, None)。"""
    if token is None or str(token) in _NULL_TOKENS:
        return None, None, None
    m = _RE_C.match(str(token))
    if not m:
        raise ValueError(f"無法解析動態條件 token: {token!r}")
    return m["cid"], m["source"], m["rule"]


def _factor_type(f1: str, f2: str | None) -> tuple[str, str]:
    """F1/F2 同類 → 該類；異類 → 混合型。回傳 (type, basis)。

    GateC v1.1 已定案此處**純規則、不引入 LLM**：候選池只有 F1+F2 兩個因子，
    台股 F1 僅 11 個桶、美股 16 個，規則覆蓋率 100%，沒有需要 LLM 補判的邊緣案例。
    """
    t1 = C.FACTOR_TYPE_MAP.get(f1)
    if t1 is None:
        raise ValueError(f"因子 {f1!r} 不在五分類表中，請更新 contracts.FACTOR_TYPE_MAP")
    if f2 is None:
        return t1, f"F1={f1}({t1})；F2=空集合"
    t2 = C.FACTOR_TYPE_MAP.get(f2)
    if t2 is None:
        raise ValueError(f"因子 {f2!r} 不在五分類表中，請更新 contracts.FACTOR_TYPE_MAP")
    basis = f"F1={f1}({t1})；F2={f2}({t2})"
    return (t1 if t1 == t2 else "混合型"), basis


def _load_market(market: str, log) -> pd.DataFrame:
    csv = paths.candidates_csv(market)
    # 落差附註：檔案是 UTF-8 with BOM，欄名含中文
    df = pd.read_csv(csv, encoding="utf-8-sig")
    log(f"  讀入 {csv.name}: {len(df)} 列, 欄位={list(df.columns)}")

    df = df.rename(columns={"F組合": "f_combo", "持股數": "avg_holdings"})
    df["market"] = market
    df[C.PK] = C.make_uid(df["market"], df["strategy"])

    # ---- 拆策略字串 ----
    parsed = []
    for s in df["strategy"]:
        parts = str(s).split("__")
        if len(parts) != 4:
            raise ValueError(f"策略字串不是 4 段: {s!r}")
        f1_tok, f2_tok, c_tok, v_tok = parts
        f1, b1, n1 = _parse_band(f1_tok)
        f2, b2, n2 = _parse_band(f2_tok)
        cid, csrc, crule = _parse_c(c_tok)
        ftype, fbasis = _factor_type(f1, f2)
        parsed.append((f1, b1, n1, f2, b2, n2, f2 is None,
                       cid, csrc, crule, v_tok, ftype, fbasis))
    cols = ["F1_factor", "F1_band", "F1_nbands", "F2_factor", "F2_band", "F2_nbands",
            "F2_empty", "C_id", "C_source", "C_rule", "V_parsed",
            "factor_type", "factor_type_basis"]
    df = pd.concat([df, pd.DataFrame(parsed, columns=cols, index=df.index)], axis=1)

    # 交叉驗證：從策略字串拆出的 V 必須與 CSV 的 V 欄一致（免費的正確性檢查）
    mismatch = (df["V_parsed"] != df["V"]).sum()
    if mismatch:
        raise ValueError(f"[{market}] 策略字串的 V 與 CSV 的 V 欄不符: {mismatch} 列")
    df = df.drop(columns=["V_parsed"])

    # ---- 落差2：解析 artifacts_dir（v0→L3 / v1→L4）並斷言存在 ----
    dirs, missing = [], []
    for s, v in zip(df["strategy"], df["V"]):
        p = paths.artifacts_path(market, s, v)
        dirs.append(str(p))
        if not p.is_dir():
            missing.append(s)
    df["artifacts_dir"] = dirs
    if missing:
        raise FileNotFoundError(
            f"[{market}] {len(missing)} 個策略找不到回測產物目錄，例如 {missing[:3]}。"
            f" 請確認 results_artifacts/ 未被搬動。")
    log(f"  artifacts_dir 解析完成，存在率 100% ({len(df)}/{len(df)})")

    # ---- 落差1：join stats.parquet 補 sharpe，並對帳 ----
    stats = pd.concat(
        [pd.read_parquet(paths.stats_parquet(market, v)) for v in ("v0", "v1")],
        ignore_index=True,
    ).drop_duplicates("strategy").set_index("strategy")

    take = ["sharpe_ann", "daily_sharpe", "avg_drawdown", "CAGR", "max_drawdown", "win_ratio"]
    joined = df.join(stats[take], on="strategy", rsuffix="_st")
    if joined["sharpe_ann"].isna().any():
        n = int(joined["sharpe_ann"].isna().sum())
        raise ValueError(f"[{market}] stats.parquet join 未命中 {n} 列")

    # 對帳斷言：把「stats.parquet 與 CSV 必須同步」這個隱性相依變成可自動偵測
    for col in ("CAGR", "max_drawdown", "win_ratio"):
        C.assert_reconciles(joined[col], joined[f"{col}_st"], name=f"{market}.{col}")
    log(f"  stats.parquet join 100% 命中，CAGR/MDD/win_ratio 對帳通過")

    return joined.drop(columns=[f"{c}_st" for c in ("CAGR", "max_drawdown", "win_ratio")])


def build(log=print) -> pd.DataFrame:
    t0 = time.time()
    paths.ensure_dirs()

    frames = []
    for m in C.MARKETS:
        log(f"[{m}]")
        frames.append(_load_market(m, log))
    df = pd.concat(frames, ignore_index=True)

    df = df[C.CANDIDATE_INDEX.names]                    # 固定欄序
    df["market"] = df["market"].astype("category")
    df["V"] = df["V"].astype("category")

    log("\n驗證契約 …")
    C.validate(df, C.CANDIDATE_INDEX, strict_columns=True)
    log("  ✓ 列數 / 主鍵唯一 / F2_empty 計數 / 型別值域 全部通過")

    out = paths.STAGE0 / "candidate_index.parquet"
    df.to_parquet(out, compression="zstd", index=False)

    freeze.write_manifest(
        "stage0_index", paths.STAGE0,
        inputs=[paths.candidates_csv(m) for m in C.MARKETS]
               + [paths.stats_parquet(m, v) for m in C.MARKETS for v in ("v0", "v1")],
        outputs=[out],
        params={"variant": paths.VARIANT, "pk_rule": C.PK_RULE,
                "job_by_v": paths.JOB_BY_V, "in_sample_end": paths.IN_SAMPLE_END},
        notes="修掉落差1(join stats)/落差2(artifacts_dir 規則解析)/落差5(複合主鍵)",
    )
    log(f"\n產出 {out}  ({len(df)} 列, {out.stat().st_size/1024:.0f} KB, {time.time()-t0:.1f}s)")
    return df


def _report(df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 62)
    log("階段0 驗收報告")
    log("=" * 62)
    log(f"總列數        : {len(df)}  (預期 {C.EXPECTED_ROWS_TOTAL})")
    for m in C.MARKETS:
        sub = df[df.market == m]
        vs = sub["V"].value_counts().to_dict()
        log(f"[{m}] {len(sub):>5} 列 | v0={vs.get('v0',0)} v1={vs.get('v1',0)} "
            f"| F2_empty={int(sub.F2_empty.sum())} (預期 {C.EXPECTED_F2_EMPTY[m]}) "
            f"| 獨立F組合={sub.f_combo.nunique()} (預期 {C.EXPECTED_F_COMBOS[m]})")
    log(f"\n主鍵唯一      : {df[C.PK].is_unique}")
    log(f"裸 strategy 唯一 : {df['strategy'].is_unique}  "
        f"← False 證明落差5 真實存在，複合主鍵是必要的")
    dup = int(len(df) - df["strategy"].nunique())
    log(f"  台美字串碰撞數 : {dup}")
    log(f"\nfactor_type 分布:\n{df.factor_type.value_counts().to_string()}")
    log(f"\nC 條件數      : {df.C_id.nunique()} 種 (含空 C: {int(df.C_id.isna().sum())} 列)")


def main(argv: list[str] | None = None) -> int:
    # 階段0 目前無任何 CLI 參數；接受 argv 只是為了與其他階段模組介面一致，
    # 讓 research.cli 能統一用 `main(extra or [])` 呼叫而不用特判。
    df = build()
    _report(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
