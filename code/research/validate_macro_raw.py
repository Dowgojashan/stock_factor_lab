# -*- coding: utf-8 -*-
"""總經原始資料自我檢查（給 collector 交付前跑）

目的：讓 collector 在交付**之前**就能確認資料合規，不必等下游退件才知道哪裡不對。

用法：
    cd code
    python -m research.validate_macro_raw <你的檔案.parquet|.csv>

檢查項目分兩級：
    ❌ 錯誤 —— 不符契約，下游無法使用，必須修
    ⚠️ 警告 —— 可以用，但建議確認（例如覆蓋率偏低、疑似單位錯誤）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import contracts as C
from .macro_spec import AXES, SPEC

REQUIRED_KEYS = ("growth", "inflation", "rate_level", "rate_direction")
OPTIONAL_KEYS = ("growth_alt",)
MIN_START = "1998-01"
NEED_THROUGH = "2025-12"

#: 各指標的合理值域（抓單位錯誤用，例如把 5.25% 寫成 0.0525）
SANITY = {
    "inflation": (-20.0, 30.0, "CPI 年增率應在 ±20% 內；若看到 0.02 這種值，單位可能寫成小數而非百分比"),
    "rate_level": (-5.0, 25.0, "利率應為百分比數值（如 5.25），不是 0.0525"),
    "rate_direction": (-500.0, 500.0, "美股為利差(百分點)、台股為景氣分數(9~45)"),
    "growth": (-30.0, 30.0, "GDP 年增率應在 ±30% 內"),
    "growth_alt": (-30.0, 40.0, "失業率或工業生產指數年增率"),
}


def _load(p: Path) -> pd.DataFrame:
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, encoding="utf-8-sig")
    return pd.read_parquet(p)


def check(path: str | Path, log=print) -> int:
    p = Path(path)
    if not p.exists():
        log(f"❌ 找不到檔案 {p}")
        return 1
    df = _load(p)
    errs: list[str] = []
    warns: list[str] = []

    log(f"檔案：{p}　({len(df):,} 列)")
    log(f"欄位：{list(df.columns)}\n")

    # ---- 1. 契約 ----
    try:
        if "market" in df.columns:
            df["market"] = df["market"].astype("category")
        if "unit" in df.columns:
            df["unit"] = df["unit"].astype("category")
        C.validate(df, C.MACRO_RAW)
        log("✅ 契約驗證通過（欄位／型別／值域／主鍵唯一）")
    except C.ContractError as e:
        errs.append(str(e))
        log(f"❌ 契約驗證失敗：{e}")
        log("\n（以下檢查在契約通過前僅供參考）\n")

    if not {"market", "indicator_key", "period", "value"}.issubset(df.columns):
        log("\n❌ 缺少必要欄位，無法繼續檢查")
        return 1

    # ---- 2. 指標齊全度 ----
    log("\n【指標齊全度】")
    for mkt in C.MARKETS:
        got = set(df[df.market == mkt]["indicator_key"])
        missing = set(REQUIRED_KEYS) - got
        extra = got - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS)
        if missing:
            errs.append(f"{mkt} 缺少必要指標 {sorted(missing)}")
            log(f"  ❌ [{mkt}] 缺少：{sorted(missing)}")
        else:
            log(f"  ✅ [{mkt}] 四個必要軸齊全"
                + (f"（另有 {sorted(got & set(OPTIONAL_KEYS))}）" if got & set(OPTIONAL_KEYS) else ""))
        if extra:
            warns.append(f"{mkt} 有未定義的 indicator_key：{sorted(extra)}")

    # ---- 3. 期間格式與涵蓋範圍 ----
    log("\n【期間格式與涵蓋範圍】")
    bad_fmt = []
    for v in df["period"].astype(str).unique():
        try:
            pd.Period(v)
        except Exception:
            bad_fmt.append(v)
    if bad_fmt:
        errs.append(f"period 格式無法解析：{bad_fmt[:5]}")
        log(f"  ❌ 無法解析的 period：{bad_fmt[:5]}（應為 YYYY-MM 或 YYYYQn）")
    else:
        log("  ✅ period 格式全部可解析")

    for (mkt, key), g in df.groupby(["market", "indicator_key"], observed=True):
        per = pd.PeriodIndex([pd.Period(x) for x in g["period"].astype(str)])
        lo, hi = per.min(), per.max()
        lo_m = lo.asfreq("M", how="start")
        hi_m = hi.asfreq("M", how="end")
        flag = ""
        if lo_m > pd.Period(MIN_START, "M"):
            flag += f"  ⚠️ 起點晚於 {MIN_START}（YoY 與平滑需要暖機期）"
            warns.append(f"{mkt}.{key} 起點 {lo} 晚於 {MIN_START}")
        if hi_m < pd.Period(NEED_THROUGH, "M"):
            flag += f"  ❌ 未涵蓋到 {NEED_THROUGH}"
            errs.append(f"{mkt}.{key} 只到 {hi}，未涵蓋 {NEED_THROUGH}")
        log(f"  [{mkt}] {key:<15} {lo} ~ {hi}  ({len(g):>4} 筆){flag}")

    # ---- 4. 缺值與假 0 ----
    log("\n【缺值與假 0】")
    for (mkt, key), g in df.groupby(["market", "indicator_key"], observed=True):
        n_nan = int(g["value"].isna().sum())
        n_zero = int((g["value"] == 0).sum())
        msg = []
        if n_nan:
            msg.append(f"NaN {n_nan}")
        if n_zero > len(g) * 0.05:
            msg.append(f"**0 值 {n_zero} ({n_zero/len(g):.0%})**")
            warns.append(f"{mkt}.{key} 有 {n_zero} 個 0 值——請確認是真實 0 還是缺值填 0")
        log(f"  [{mkt}] {key:<15} " + ("｜".join(msg) if msg else "無缺值、無異常 0"))
    log("  ⚠️ 提醒：**沒有資料的期間請直接不要那一列，不要填 0 或 NaN 值列**"
        "（`value` 欄位本身不接受 NaN——下游 align_by_lag 對查不到的 period"
        "本來就會回傳 NaN；台股早期財報用 0 頂替缺值已造成過實質錯誤，勿重蹈覆轍）")

    # ---- 5. 值域合理性（抓單位錯誤）----
    log("\n【值域合理性（抓單位錯誤）】")
    for (mkt, key), g in df.groupby(["market", "indicator_key"], observed=True):
        rng = SANITY.get(key)
        v = g["value"].dropna()
        if rng is None or v.empty:
            continue
        lo, hi, hint = rng
        out = v[(v < lo) | (v > hi)]
        u = str(g["unit"].iloc[0]) if "unit" in g.columns else "?"
        if len(out):
            warns.append(f"{mkt}.{key} 有 {len(out)} 筆超出合理值域 [{lo}, {hi}]")
            log(f"  ⚠️ [{mkt}] {key:<15} {len(out)} 筆超出 [{lo},{hi}]（min={v.min():.3g} max={v.max():.3g}, unit={u}）→ {hint}")
            continue

        # 「小數 vs 百分比」錯誤：值域檢查抓不到（0.021 也在 [-20,30] 內），
        # 但百分比欄位若絕大多數絕對值 < 1，幾乎必然是把 2.1% 寫成 0.021。
        if u.startswith("percent") and len(v) >= 12:
            if float(np.nanpercentile(v.abs(), 95)) < 1.0:
                errs.append(
                    f"{mkt}.{key} 宣告 unit={u} 但 95% 的值絕對值 < 1"
                    f"（max={v.abs().max():.4g}）——疑似把百分比寫成小數")
                log(f"  ❌ [{mkt}] {key:<15} unit={u} 但 |值| 幾乎都 < 1（max={v.abs().max():.4g}）"
                    f" → **疑似 2.1% 被寫成 0.021**，請確認")
                continue
        log(f"  ✅ [{mkt}] {key:<15} min={v.min():>9.3g} max={v.max():>9.3g}  unit={u}")

    # ---- 6. release_date ----
    log("\n【release_date（point-in-time 能力）】")
    if "release_date" not in df.columns or df["release_date"].isna().all():
        log("  ⚠️ 未提供 release_date → 下游改用 macro_spec 的**推估**滯後量")
        log("     若資料源拿得到實際發布日（如 FRED/ALFRED），強烈建議補上——")
        log("     那會讓前視偏誤防護從『推估』升級為『實測』")
        warns.append("未提供 release_date")
    else:
        # 覆蓋率**分市場**報告 —— 全體平均會誤導：一個市場 100%、另一個
        # 設計上就該留空（如無 vintage 服務的官方統計），平均後看起來像「一半缺漏」。
        for mkt in C.MARKETS:
            sub = df[df.market == mkt]["release_date"] if "market" in df.columns else df["release_date"]
            cov = sub.notna().mean() if len(sub) else float("nan")
            log(f"  [{mkt}] 覆蓋率 {cov:.0%}"
                + ("（全缺，若該市場資料源無 vintage 服務屬預期）" if cov == 0 else ""))

        rd = pd.to_datetime(df["release_date"], errors="coerce")
        bad = int(rd.isna().sum() - df["release_date"].isna().sum())
        if bad:
            errs.append(f"release_date 有 {bad} 筆無法解析為日期")
            log(f"  ❌ {bad} 筆無法解析為日期")

        # 發布日必須晚於資料期間結束。period 混合月/季頻，PeriodIndex 整批建構
        # 會因頻率不一致而 raise IncompatibleFrequency，故逐筆轉換。
        per_end_month = pd.Series(
            [pd.Period(p).asfreq("M", how="end") for p in df["period"].astype(str)],
            index=df.index)
        has_rd = rd.notna()
        early_mask = has_rd & (rd.dt.to_period("M").values < per_end_month.values)
        early = int(early_mask.sum())
        if early:
            errs.append(f"release_date 有 {early} 筆早於其資料期間（不可能）")
            log(f"  ❌ {early} 筆的發布日早於資料期間結束——不可能，請檢查")
            log("     例：" + "; ".join(
                f"{r.market}.{r.indicator_key} period={r.period} release={r.release_date}"
                for r in df[early_mask].head(3).itertuples()))

    # ---- 總結 ----
    log("\n" + "=" * 60)
    if errs:
        log(f"❌ 不可交付：{len(errs)} 個錯誤")
        for e in errs:
            log(f"   • {e}")
    else:
        log("✅ 可以交付")
    if warns:
        log(f"\n⚠️ {len(warns)} 個警告（可交付但建議確認）：")
        for w in warns:
            log(f"   • {w}")
    log("=" * 60)
    return 1 if errs else 0


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2
    return check(argv[0])


if __name__ == "__main__":
    sys.exit(main())
