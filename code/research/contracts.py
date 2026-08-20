# -*- coding: utf-8 -*-
"""資料契約：schema 定義 + 驗證器（單一事實來源）。

為什麼要有這支（SDD 第三部分）：
  凍結原則規定「改上游 = 連鎖重跑下游」，因此欄位名稱、型別、值域必須先寫死，
  且**由程式強制檢查**，不靠人工記憶。任何產物落地前一律先過 validate()。

設計立場：schema 違規一律**立即中止**（raise），不做 warning 就放行——
  錯的資料比沒資料危險，尤其是靜默錯 join（見 PK_RULE）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

# ============================================================================
# 常數：實測基準（用來當驗收斷言，數字來自 2026-08-20 全量掃描）
# ============================================================================

MARKETS = ("TW", "US")

#: 候選池列數（階段 −1 產物，已向老師報告過的數字）
EXPECTED_ROWS = {"TW": 7162, "US": 6916}
EXPECTED_ROWS_TOTAL = sum(EXPECTED_ROWS.values())          # 14,078

#: v0/v1 拆分（實測）
EXPECTED_V_SPLIT = {"TW": {"v0": 4495, "v1": 2667},
                    "US": {"v0": 3860, "v1": 3056}}

#: F2 空集合策略數（老師「F1+C 就好」的分流依據）。
#: 這個數字對得上，就證明策略字串拆解是正確的——是階段0 最有力的驗收。
EXPECTED_F2_EMPTY = {"TW": 384, "US": 652}

#: 獨立 F 組合數（快篩「多樣性假象」的根源；也是 HRP L3 群數的錨點）
EXPECTED_F_COMBOS = {"TW": 219, "US": 188}

#: 自建宇宙基準年化報酬（研究部 v9 更正版：同宇宙、同成本、含股利、等權）
BENCHMARK_CAGR = {"TW": 0.0867, "US": 0.1235}

#: HRP 每棵樹的共同窗（SDD DD-03 定案，落差3）
#: 台股窗捨棄 2000–2006 是**排除已知不可信資料**（財報洞、90% 假 0），非損失資料。
HRP_WINDOWS = {
    "TW":  ("2007-01", "2025-12"),   # 228 月，保留 7,160 / 7,162
    "US":  ("2002-01", "2025-12"),   # 288 月，保留 6,915 / 6,916
    "XM":  ("2007-01", "2025-12"),   # 228 月，保留 14,076 / 14,078
}
#: 穩健性對照窗（台股樹另跑一次，比較群結構 ARI）
HRP_ROBUSTNESS_WINDOW_TW = ("2003-01", "2025-12")

#: 五分類因子表（研究部 v9 / GateC C-2）
FACTOR_TYPE_MAP = {
    **{f: "估值型" for f in ("PB", "PS", "P_IC", "EV_S", "EV_EBITDA", "FCF_P", "FCF_OI", "PE")},
    **{f: "體質型" for f in ("ROE", "EPS", "ROIC", "CROIC", "OCF_E", "ACCRUAL")},
    **{f: "動能型" for f in ("MOM", "MOM1", "VOL")},
    **{f: "規模型" for f in ("REVENUE", "REV_G")},
    **{f: "結構型" for f in ("DEBTRATIO", "NETDEBT_EBITDA")},
}

#: 主鍵規則。實測台美策略字串**碰撞 1,381 個**（同一套因子命名規則、字串不編碼
#: 市場），若以裸 `strategy` 為鍵合併，約 9.8% 資料會靜默錯配且無錯誤訊息。
PK_RULE = "strategy_uid = market + '::' + strategy"
PK = "strategy_uid"


def make_uid(market: pd.Series | str, strategy: pd.Series | str):
    """組主鍵。市場在前，排序時台美自然分組、debug 時一眼可讀。"""
    if isinstance(market, str):
        return f"{market}::{strategy}"
    return market.astype(str) + "::" + strategy.astype(str)


# ============================================================================
# Schema 描述與驗證
# ============================================================================

@dataclass(frozen=True)
class Column:
    name: str
    kind: str                      # 'str' | 'int' | 'float' | 'bool' | 'cat' | 'period'
    nullable: bool = False
    allowed: Sequence | None = None        # 類別欄的合法值
    ge: float | None = None                # 數值下界（含）
    le: float | None = None                # 數值上界（含）


@dataclass(frozen=True)
class Schema:
    name: str
    columns: list[Column]
    primary_key: list[str] = field(default_factory=lambda: [PK])
    expected_rows: int | None = None
    #: 額外的跨欄檢查，簽章 fn(df) -> None，違規自行 raise
    checks: tuple[Callable[[pd.DataFrame], None], ...] = ()

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]


class ContractError(AssertionError):
    """契約違規。刻意繼承 AssertionError——它代表「這不該發生」，不是可預期的例外。"""


def _fail(schema: str, msg: str) -> None:
    raise ContractError(f"[契約違規 · {schema}] {msg}")


_KIND_CHECK = {
    "str":    lambda s: pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s),
    "int":    lambda s: pd.api.types.is_integer_dtype(s),
    "float":  lambda s: pd.api.types.is_float_dtype(s),
    "bool":   lambda s: pd.api.types.is_bool_dtype(s),
    "cat":    lambda s: True,      # 允許 object/category，值域另外查 allowed
    "period": lambda s: isinstance(s.dtype, pd.PeriodDtype),
}


def validate(df: pd.DataFrame, schema: Schema, *, strict_columns: bool = False) -> pd.DataFrame:
    """驗證 DataFrame 是否符合 schema；不符即 raise ContractError。

    strict_columns=True 時，多出 schema 未宣告的欄位也算違規（用於凍結產物）。
    回傳原 df（方便鏈式呼叫）。
    """
    sn = schema.name

    missing = [c for c in schema.names if c not in df.columns]
    if missing:
        _fail(sn, f"缺少欄位: {missing}")

    if strict_columns:
        extra = [c for c in df.columns if c not in schema.names]
        if extra:
            _fail(sn, f"出現未宣告的欄位: {extra}")

    if schema.expected_rows is not None and len(df) != schema.expected_rows:
        _fail(sn, f"列數 {len(df)} != 預期 {schema.expected_rows}")

    # 主鍵：存在、非空、唯一。這條是防「台美字串碰撞靜默錯 join」的核心防線。
    for k in schema.primary_key:
        if k not in df.columns:
            _fail(sn, f"缺少主鍵欄 {k}")
        if df[k].isna().any():
            _fail(sn, f"主鍵 {k} 有空值 ({int(df[k].isna().sum())} 筆)")
    if schema.primary_key:
        dup = df.duplicated(subset=schema.primary_key)
        if dup.any():
            sample = df.loc[dup, schema.primary_key].head(5).to_dict("records")
            _fail(sn, f"主鍵不唯一：{int(dup.sum())} 筆重複，例如 {sample}。"
                      f" 提醒：主鍵規則為 {PK_RULE}")

    for col in schema.columns:
        s = df[col.name]
        if not col.nullable and s.isna().any():
            _fail(sn, f"欄位 {col.name} 不可為空，但有 {int(s.isna().sum())} 筆空值")

        nn = s.dropna()
        if len(nn) == 0:
            continue

        checker = _KIND_CHECK.get(col.kind)
        if checker is not None and not checker(s):
            _fail(sn, f"欄位 {col.name} 型別為 {s.dtype}，不符宣告的 {col.kind}")

        if col.allowed is not None:
            bad = set(nn.unique()) - set(col.allowed)
            if bad:
                _fail(sn, f"欄位 {col.name} 出現非法值 {sorted(bad)[:5]}，合法值為 {list(col.allowed)}")

        if col.ge is not None and (nn < col.ge).any():
            _fail(sn, f"欄位 {col.name} 有值 < {col.ge}（min={nn.min()}）")
        if col.le is not None and (nn > col.le).any():
            _fail(sn, f"欄位 {col.name} 有值 > {col.le}（max={nn.max()}）")

    for chk in schema.checks:
        chk(df)

    return df


def assert_reconciles(left: pd.Series, right: pd.Series, *, name: str, tol: float = 1e-9) -> None:
    """對帳斷言：兩個來源的同一欄必須相符。

    用於階段0 join stats.parquet 後核對 CAGR/MDD/win_ratio——實測差異僅 9.71e-17，
    證明候選 CSV 與 stats.parquet 同源。這條斷言把「檔案同步」這個隱性相依
    變成可自動偵測的條件：任一列對不上就中止，不同步不可能靜默發生。
    """
    diff = (left.astype(float) - right.astype(float)).abs()
    bad = int((diff > tol).sum())
    if bad:
        _fail("對帳", f"{name} 有 {bad} 列超出容差 {tol:g}（max diff={diff.max():.3e}）。"
                      f" 可能原因：stats.parquet 與候選 CSV 不同步，需重跑上游。")


# ============================================================================
# 各階段 schema
# ============================================================================

def _check_market_counts(df: pd.DataFrame) -> None:
    got = df["market"].value_counts().to_dict()
    for m, n in EXPECTED_ROWS.items():
        if got.get(m, 0) != n:
            _fail("candidate_index", f"{m} 列數 {got.get(m, 0)} != 預期 {n}")


def _check_f2_empty_counts(df: pd.DataFrame) -> None:
    """F2 空集合計數對得上 => 策略字串拆解正確。這是階段0 最有力的驗收。"""
    got = df[df["F2_empty"]]["market"].value_counts().to_dict()
    for m, n in EXPECTED_F2_EMPTY.items():
        if got.get(m, 0) != n:
            _fail("candidate_index",
                  f"{m} 的 F2_empty 計數 {got.get(m, 0)} != 預期 {n}，"
                  f"表示策略字串拆解有誤")


def _check_uid_format(df: pd.DataFrame) -> None:
    expect = make_uid(df["market"], df["strategy"])
    if not (df[PK] == expect).all():
        _fail("candidate_index", f"{PK} 與 market/strategy 不一致，規則應為 {PK_RULE}")


CANDIDATE_INDEX = Schema(
    name="candidate_index",
    expected_rows=EXPECTED_ROWS_TOTAL,
    checks=(_check_market_counts, _check_f2_empty_counts, _check_uid_format),
    columns=[
        Column(PK, "str"),
        Column("strategy", "str"),
        Column("market", "cat", allowed=MARKETS),
        # --- 結構欄（字串拆解） ---
        Column("f_combo", "str"),
        Column("F1_factor", "str"),
        Column("F1_band", "int"),
        Column("F1_nbands", "int"),
        Column("F2_factor", "str", nullable=True),
        Column("F2_band", "float", nullable=True),      # 有 NaN 故為 float
        Column("F2_nbands", "float", nullable=True),
        Column("F2_empty", "bool"),
        Column("C_id", "str", nullable=True),
        Column("C_source", "str", nullable=True),
        Column("C_rule", "str", nullable=True),
        Column("V", "cat", allowed=("v0", "v1")),
        Column("factor_type", "cat",
               allowed=("估值型", "體質型", "動能型", "規模型", "結構型", "混合型")),
        Column("factor_type_basis", "str"),
        # --- 階段 −1 既有指標（候選 CSV） ---
        Column("CAGR", "float"),
        Column("max_drawdown", "float", le=0.0),
        Column("win_ratio", "float", ge=0.0, le=1.0),
        Column("avg_holdings", "float", ge=0.0),
        # --- 落差1：join stats.parquet 補齊 ---
        Column("sharpe_ann", "float"),
        Column("daily_sharpe", "float"),
        Column("avg_drawdown", "float", le=0.0),
        # --- 落差2：規則解析 ---
        Column("artifacts_dir", "str"),
    ],
)


RETURNS_MONTHLY = Schema(
    name="returns_monthly",
    primary_key=[PK, "month"],
    columns=[
        Column(PK, "str"),
        Column("month", "period"),
        Column("ret", "float"),
    ],
)


RETURNS_META = Schema(
    name="returns_meta",
    expected_rows=EXPECTED_ROWS_TOTAL,
    columns=[
        Column(PK, "str"),
        Column("market", "cat", allowed=MARKETS),
        Column("hist_start", "str"),     # YYYY-MM
        Column("hist_end", "str"),
        Column("n_months", "int", ge=1),
    ],
)


ALL_SCHEMAS = {s.name: s for s in (CANDIDATE_INDEX, RETURNS_MONTHLY, RETURNS_META)}
