# -*- coding: utf-8 -*-
"""S-01 · 群→總經決策層的介面（開發待辦追蹤.md 方向C 第一步）

老師走方向C（總經→選群）的第一個接縫：H-08 產出的「群身份描述」必須是總經層
能消費的結構化格式，不能只是一段自由文字。這裡把 H-06（`cluster_profile_quant`，
純程式算）與 H-08（`cluster_identity`，LLM 寫的身份描述）合併成一張決策層可以
直接讀的表——純程式組裝，**沒有 LLM**。

🔴 完整設計理由（含為什麼不能直接把 cluster_identity 四個欄位都收進來）見
`contracts.CLUSTER_MACRO_INTERFACE` 的 schema 註解，這裡只重述結論：

  - 只收 `identity_label`（短句、風險最低）
  - `mechanism_note`／`performance_pattern`／`caveat` **一律不收**——
    `cluster_identity.py` 是一次性把含年份的側寫餵給同一個LLM呼叫產生四個欄位，
    即使實測目前沒外洩年份，也只是這次運氣好、不是架構保證（S-02 若要更豐富的
    LLM文字脈絡，須另外設計「輸入從頭就不含年份」的專屬prompt）
  - 其餘欄位來自 `cluster_profile_quant`，**排除**四個帶日曆年份的欄位
    （window_start_year／window_end_year／best_year／worst_year），對應的報酬
    幅度（best_year_ret／worst_year_ret）保留

用法：
    cd code
    python -m research.cluster_macro_interface
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import contracts as C
from . import freeze, paths

#: cluster_profile_quant 裡明確排除的欄位——帶日曆年份，S-03（決策層不給日期）
#: 明文禁止流入總經層。
_EXCLUDED_QUANT_COLUMNS = ("window_start_year", "window_end_year", "best_year", "worst_year")


def build(log=print) -> pd.DataFrame:
    # ⚠️ 2026-09-01 code review 修正：原本驗的是 paths.STAGE3（主manifest，管
    # cluster_assign/cluster_meta等）跟 paths.STAGE4——但本函式實際讀的是
    # cluster_identity.parquet（無正式manifest，見下方說明）跟
    # cluster_profile_quant.parquet（manifest在STAGE3/_temporal_profile子目錄，
    # H-06另開的），兩個都不在原本驗證的範圍內，等於驗了兩個不相干的東西、
    # 真正該驗的都漏掉了。cluster_identity.parquet是LLM輸出，用側錄
    # meta.json（非freeze manifest）記錄完整性，本函式無法用freeze驗證它。
    freeze.verify_inputs(paths.STAGE3 / "_temporal_profile")

    identity = pd.read_parquet(paths.STAGE3 / "cluster_identity.parquet")[
        ["tree_id", "level", "cluster_id", "identity_label"]]
    quant = pd.read_parquet(paths.STAGE3 / "cluster_profile_quant.parquet")
    quant = quant.drop(columns=list(_EXCLUDED_QUANT_COLUMNS))

    df = identity.merge(quant, on=["tree_id", "level", "cluster_id"], how="inner",
                        validate="one_to_one")
    n_dropped = len(identity) - len(df)
    if n_dropped:
        log(f"  ⚠️ {n_dropped} 群在 identity 有但 quant 沒有（或反之），"
            f"merge時被排除——理論上H-06/H-08涵蓋的群應完全一致，若非0須查原因")

    df["tree_id"] = df["tree_id"].astype("category")
    df["level"] = df["level"].astype("category")
    return df[C.CLUSTER_MACRO_INTERFACE.names]


def run(log=print) -> pd.DataFrame:
    df = build(log=log)
    C.validate(df, C.CLUSTER_MACRO_INTERFACE, strict_columns=True)
    log(f"✓ cluster_macro_interface 契約通過（{len(df)} 群）")

    p = paths.STAGE3 / "cluster_macro_interface.parquet"
    df.to_parquet(p, compression="zstd", index=False)
    # ⚠️ 不寫進 stage3 的 MANIFEST：同 cluster_temporal_profile.py 的理由——
    # 這是附加組裝、非stage3本體，不佔用 DD-08 凍結鏈的雜湊驗證範圍。
    freeze.write_manifest(
        "cluster_macro_interface", paths.STAGE3 / "_macro_interface",
        inputs=[paths.STAGE3 / "cluster_identity.parquet",
               paths.STAGE3 / "cluster_profile_quant.parquet"],
        outputs=[p],
        params={"excluded_quant_columns": list(_EXCLUDED_QUANT_COLUMNS),
               "identity_fields_used": ["identity_label"],
               "identity_fields_excluded": ["mechanism_note", "performance_pattern", "caveat"]},
        notes="S-01：群→總經介面。純程式組裝(H-06+H-08合併)，無LLM。"
              "排除規則見模組docstring與contracts.CLUSTER_MACRO_INTERFACE schema註解。",
    )
    log(f"→ {p}")
    return df


def _report(df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 78)
    log("S-01 · 群→總經介面 驗收摘要")
    log("=" * 78)
    for t, g in df.groupby("tree_id", observed=True):
        log(f"\n[{t}]（{len(g)} 群）")
        for r in g.sort_values("cluster_id").itertuples():
            log(f"  群{r.cluster_id}：{r.identity_label}"
               f"｜CAGR中位數{r.CAGR_median:.1%}｜正報酬年比例{r.pct_years_positive:.0%}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.cluster_macro_interface")
    ap.parse_args(argv)
    df = run()
    _report(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
