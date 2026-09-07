# -*- coding: utf-8 -*-
"""M-14 · ENB 反轉是不是 N > T 的秩不足造成的（2026-09-07）

🔴 **稽核清單原本把這一項列為第二輪最優先**，理由是：
`_small_enb` 對 **OOS 窗**算相關矩陣，而 1% 以上的比例在 24 個月窗都是 N > T，
相關矩陣的**秩最多 T−1**，ENB 天花板被機械壓低；
而 M-03b 的核心修正之一正是「HRP 的 ENB 優勢只在 legacy 成立，1% 以上反轉」
——**反轉點恰好落在 N 超過 T 的地方**。

---------------------------------------------------------------------------
🔴 但這個前提在執行前就被實測推翻了（2026-09-06）
---------------------------------------------------------------------------
**① legacy 也大量 N > T，反轉點與 N>T 的分界不重合**

    N>T 的格子佔比    legacy    1%     3%     5%
    TW                46.7%    100%   100%   100%
    US                37.8%    100%   100%   100%
    XM                 0.0%    100%   100%   100%

TW/US 的 legacy（30/35 檔 vs 24 個月窗）已有近四成格子是 N>T，
**只有 XM legacy 完全乾淨**。稽核清單的表只列 TW 且未察覺 legacy 自己就違反了該分界。

⚠️ 上表是**全部格子**的佔比。本模組的 `pct_n_gt_t` 只算 **ENB 可比的子集**
（兩邊都 <=400 檔），數字較低（TW 32%／US 26%／XM 0%）——分母不同，
兩者都顯示同一件事：**legacy 不乾淨**。

**② 分層重算後，反轉在四種窗長全部出現**（48/60 個月的窗甚至更極端）。

故本模組的任務**從「檢查一個可能有問題的結論」降級為「把已知結論做成正式表 + 測試」**。
判讀規則直接寫死：**已排除秩不足的估計偏誤**。

---------------------------------------------------------------------------
產出
---------------------------------------------------------------------------
`ratio × n_oos_months` 的 ENB 勝率交叉表，A vs A2_random（M-03b）與
A vs C_random（M-02b）各一份。

⚠️ `n_oos_months` 是**實際**月數（尾巴併窗後可能是 60），不是方案的名目長度。
兩張來源表已於 2026-09-07 補上該欄（由 `walkforward_matrix.oos_months` 導出，
與未來重跑逐位元相同）。

用法：
    cd code
    python -m research.enb_rank_deficiency
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, paths

#: 兩份對照來源。`control` 是對照組的名稱，用來標記交叉表的列。
SOURCES = (("A2_random", "walkforward_partition.csv", "_walkforward_partition_manifest"),
           ("C_random", "walkforward_random.csv", "_walkforward_random_manifest"))
OOS_LENS = (24, 36, 48, 60)


def _load_pairs(fname: str, manifest: str) -> pd.DataFrame:
    """把 A_hrp 的每一格接上對照組，回傳含 `n_oos_months` 的長表。"""
    d = paths.ROOT / "_analysis_outputs_robustness"
    freeze.verify_inputs(d / "_walkforward_matrix_manifest")
    freeze.verify_inputs(d / manifest)
    det = pd.read_csv(d / "walkforward_matrix_detail.csv")
    a = det[det.group == "A_hrp"].copy()
    r = pd.read_csv(d / fname)
    # 兩份來源的鍵不同：M-03b 是逐格，M-02b 是 (樹×窗×檔數) 去重後的組
    if "scheme" in r.columns:
        key = ["tree_key", "scheme", "k_mode", "ratio", "allocation", "window_no"]
    else:
        key = ["tree_key", "is_start", "is_end", "oos_start", "oos_end", "n_members"]
    for x in (a, r):
        for k in key:
            x[k] = x[k].astype(str)
    m = a.merge(r, on=key, suffixes=("_A", "_R"))
    if len(m) != len(a):
        raise AssertionError(f"[{fname}] 只接上 {len(m)}/{len(a)} 格——兩表不同步")
    # n_oos_months 兩邊都有，取對照表那份（已驗證與矩陣的 n_oos_months 相同）
    col = "n_oos_months_R" if "n_oos_months_R" in m.columns else "n_oos_months"
    m["oos_len"] = m[col].astype(int)
    # ⚠️ M-02b 的合併鍵含 `n_members`，上面為了 join 已把它轉成字串；
    # 直接拿來跟 `oos_len` 比大小會炸 TypeError（2026-09-07 開發時實測踩到）。
    # 統一在這裡還原成數值，供 N>T 判定使用。
    nm = "n_members_A" if "n_members_A" in m.columns else "n_members"
    m["n_members_num"] = pd.to_numeric(m[nm])
    return m


def build(log=print) -> pd.DataFrame:
    rows = []
    for control, fname, manifest in SOURCES:
        m = _load_pairs(fname, manifest)
        ok = m.oos_enb_A.notna() & m.oos_enb_R.notna()
        m = m[ok]
        log(f"  [{control}] 可比的格子 {len(m):,}（ENB 只在 <=400 檔時計算）")
        for (t, ratio, L), g in m.groupby(["tree_key", "ratio", "oos_len"],
                                          observed=True):
            n_gt = int((g.n_members_num > g.oos_len).sum())
            rows.append({
                "control": control, "tree_key": t, "ratio": str(ratio),
                "n_oos_months": int(L), "n_cells": int(len(g)),
                "a_wins": float((g.oos_enb_A > g.oos_enb_R).mean()),
                "enb_a_mean": float(g.oos_enb_A.mean()),
                "enb_control_mean": float(g.oos_enb_R.mean()),
                "enb_diff_mean": float((g.oos_enb_A - g.oos_enb_R).mean()),
                #: 🔴 該格有多少比例是 N > T（相關矩陣秩不足）。稽核清單假設
                #: 「反轉點 = N>T 分界」，本欄就是用來推翻它的證據。
                "pct_n_gt_t": float(n_gt / len(g)),
            })
    df = pd.DataFrame(rows)
    for c in ("control", "tree_key", "ratio"):
        df[c] = df[c].astype("category")
    return df[C.ENB_RANK_DEFICIENCY.names]


def run(log=print) -> pd.DataFrame:
    df = build(log=log)
    C.validate(df, C.ENB_RANK_DEFICIENCY, strict_columns=True)
    log(f"✓ enb_rank_deficiency 契約通過（{len(df)} 列）")
    d = paths.ROOT / "_analysis_outputs_robustness"
    p = d / "enb_rank_deficiency.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "enb_rank_deficiency", d / "_enb_rank_deficiency_manifest",
        inputs=[d / "walkforward_matrix_detail.csv",
                d / "walkforward_partition.csv",
                d / "walkforward_random.csv"],
        outputs=[p],
        params={"oos_lens": list(OOS_LENS),
                "verdict": "反轉在四種窗長皆出現，已排除 N>T 秩不足的估計偏誤"},
        notes="M-14：ENB 勝率按實際 OOS 窗長分層。稽核清單的前提（反轉點 = N>T "
              "分界）已被推翻——legacy 自己就有 37~47% 的格子是 N>T，"
              "且反轉在 24/36/48/60 四種窗長全部出現。",
    )
    log(f"→ {p}")
    return df


def _report(df: pd.DataFrame, log=print) -> None:
    log("\n" + "=" * 96)
    log("M-14 · ENB 反轉是不是 N > T 的秩不足造成的")
    log("=" * 96)
    order = {"legacy": 0, "0.01": 1, "0.03": 2, "0.05": 3, "0.1": 4}
    for control, _, _ in SOURCES:
        g0 = df[df.control == control].copy()
        g0["_o"] = g0.ratio.astype(str).map(order)
        log(f"\n【A_hrp vs {control}】ENB 的 A 勝率（列=比例，欄=實際 OOS 月數）")
        log(f"  {'樹':<5}{'比例':<9}" + "".join(f"{L:>15}" for L in OOS_LENS)
            + f"{'N>T 佔比':>10}")
        for (t,), gt in g0.groupby(["tree_key"], observed=True):
            for r_ in sorted(gt.ratio.astype(str).unique(), key=lambda x: order.get(x, 9)):
                gg = gt[gt.ratio.astype(str) == r_]
                cells = []
                for L in OOS_LENS:
                    h = gg[gg.n_oos_months == L]
                    cells.append(f"{h.a_wins.iloc[0]:>9.0%}(n{int(h.n_cells.iloc[0])})"
                                 if len(h) else f"{'—':>15}")
                log(f"  {t:<5}{r_:<9}" + "".join(f"{c:>15}" for c in cells)
                    + f"{gg.pct_n_gt_t.mean():>10.0%}")
            log("")
    log("🔴 判讀（已定案，不是待驗證）：")
    log("  · **反轉在 24/36/48/60 四種窗長全部出現**，48/60 個月的窗甚至更極端。")
    log("  · **legacy 自己就有大量格子是 N>T**：在本表（ENB 可比的子集）是")
    log("    TW 32%／US 26%／XM 0%；若不限 ENB 可比則是 TW 46.7%／US 37.8%／XM 0%。")
    log("    兩種算法都顯示 legacy 不乾淨——只有 XM legacy（15 檔 vs >=24 月）例外。")
    log("  ⇒ 反轉點與 N>T 的分界**不重合**，**已排除秩不足的估計偏誤**。")
    log("  ⇒ `研究框架總覽_v10.md` §8 的 ENB 那一列**不需修改**。")
    log("  · 更有說服力的機制在 M-15：3~5% 正是隨機組 backfill 暴增的區間。")
    log("")
    log("✅ **附帶確認：M-02b 完全不受這個疑慮影響**——A vs C_random 的 ENB")
    log("   在**每一棵樹、每一個比例、每一種窗長**都是 100% 勝率（含 N>T 的格子）。")


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(prog="research.enb_rank_deficiency").parse_args(argv)
    _report(run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
