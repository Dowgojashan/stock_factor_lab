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

#: 候選池列數
#: 🔄 2026-08-22 更新：價格資料異常修復（台股接縫186檔／美股確認26檔）後，
#: openSec 全鏈在台美兩市場重跑；同時修正了 Phase2 因子順序競態 bug
#: （白名單字串比對因子池順序不同，5個OCF_E配對曾被靜默跳過）。
#: 舊值（污染資料，已作廢）：TW 7,162／US 6,916／合計 14,078。
EXPECTED_ROWS = {"TW": 7128, "US": 8682}
EXPECTED_ROWS_TOTAL = sum(EXPECTED_ROWS.values())          # 15,810

#: v0/v1 拆分（實測，2026-08-22 修復後重跑）
EXPECTED_V_SPLIT = {"TW": {"v0": 4451, "v1": 2677},
                    "US": {"v0": 4838, "v1": 3844}}

#: F2 空集合策略數（老師「F1+C 就好」的分流依據）。
#: 這個數字對得上，就證明策略字串拆解是正確的——是階段0 最有力的驗收。
#: ⚠️ 美股從 652 變成 786——F2_empty 策略不涉及 F1×F2 配對，理論上不受
#: primary清單改變影響；差異來自美股基準被污染灌水 1.29pp 導致原本門檻過嚴，
#: 修復後大量 F1+C 策略重新贏過基準而納入候選池（與整體池暴增25.5%同一成因）。
EXPECTED_F2_EMPTY = {"TW": 384, "US": 786}

#: 獨立 F 組合數（快篩「多樣性假象」的根源；也是 HRP L3 群數的錨點）
#: 🔄 台股 primary 清單改變（ROE 出、OCF_E 進）導致組合數微降；
#: 美股因基準修正、候選門檻放寬而組合數上升。
EXPECTED_F_COMBOS = {"TW": 218, "US": 235}

#: 自建宇宙基準年化報酬（研究部 v9 更正版：同宇宙、同成本、含股利、等權）
#: 🔄 2026-08-22 用修復後價格重算：TW 8.67%→8.43%、US 12.35%→11.06%
#: （美股基準原本被污染股票灌水約 1.29pp，是候選池暴增的根本原因）。
BENCHMARK_CAGR = {"TW": 0.084256, "US": 0.110556}

#: HRP 每棵樹的共同窗（SDD DD-03 定案，落差3）
#: 台股窗捨棄 2000–2006 是**排除已知不可信資料**（財報洞、90% 假 0），非損失資料。
HRP_WINDOWS = {
    "TW":  ("2007-01", "2025-12"),   # 228 月，保留 7,160 / 7,162
    "US":  ("2002-01", "2025-12"),   # 288 月，保留 6,915 / 6,916
    "XM":  ("2007-01", "2025-12"),   # 228 月，保留 14,076 / 14,078
}
#: 穩健性對照窗（台股樹另跑一次，比較群結構 ARI）
HRP_ROBUSTNESS_WINDOW_TW = ("2003-01", "2025-12")

#: H-11（2026-08-29 使用者定案 v2）：IS/OOS 切分點，只用於 stage3_hrp_isoos.py，
#: **不影響** `HRP_WINDOWS`（第9節主線六棵樹的既有全時間窗，繼續原樣保留、不動）。
#: 原提案（依市場各自涵蓋COVID/2022熊市選窗）已撤銷——那是先看歷史再回頭挑窗口，
#: 等於用上帝視角選OOS。v2改用「兩市場OOS用同一段絕對日曆時間」：不管各自窗口
#: 多長，OOS一律是2019-01~2025-12；IS佔比因此隨市場窗口長度自然不同（TW/XM
#: 63.2%、US 70.8%），這是窗口長度天生不同的結果，不是選擇性挑出來的。
#: 只做normal樹（crisis樹樣本太小，H-14/H-16已定案不切IS/OOS）。
HRP_IS_WINDOWS = {
    "TW": ("2007-01", "2018-12"),   # 144月／63.2%，IS起點與HRP_WINDOWS相同（共同窗起點不變）
    "US": ("2002-01", "2018-12"),   # 204月／70.8%
    "XM": ("2007-01", "2018-12"),   # 144月／63.2%
}
#: OOS對三市場都相同（同一段絕對日曆時間）
HRP_OOS_WINDOW = ("2019-01", "2025-12")   # 84月，三市場皆同

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
    "date":   lambda s: pd.api.types.is_datetime64_any_dtype(s),
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


#: 三段式等級（C-3 report_shape/risk_shape 共用：市場內分位切三段，取兩端 + 中間）
SHAPE_GRADES = ("低", "中", "高")
CREDIBILITY_GRADES = ("低", "中", "高")
STABILITY_GRADES = ("陡峰", "中", "高原")

STRATEGY_MARKS = Schema(
    name="strategy_marks",
    columns=[
        Column(PK, "str"),
        Column("market", "cat", allowed=MARKETS),
        # C-3：市場內分位（0-100），CAGR 越高分位越高；MDD 用絕對值，越淺分位越高
        Column("cagr_pct", "float", ge=0, le=100),
        Column("mdd_pct", "float", ge=0, le=100),
        Column("return_shape", "cat", allowed=("大起大落", "中等", "穩定爬升")),
        Column("risk_shape", "cat", allowed=("深回撤", "中等", "淺回撤")),
        # 關卡A：可信度（分位數複合評級，見 stage1_marks.py 的組成說明）
        Column("credibility_grade", "cat", allowed=CREDIBILITY_GRADES),
        Column("credibility_score_pct", "float", ge=0, le=100),
        # 關卡B：穩健度（唯一保留的程式指標）
        Column("stability_grade", "cat", allowed=STABILITY_GRADES, nullable=True),
        # 尾端寬鬆硬篩
        Column("is_usable", "bool"),
        Column("drop_reason", "str", nullable=True),
        # W-08：資料品質防線常設化——不淘汰（True 不影響 is_usable），只標記供下游識別
        Column("data_glitch", "bool"),
        # H-01（2026-08-26老師意見）：alpha出賽關卡，獨立欄位不覆寫is_usable語意。
        # CAGR>=自建宇宙基準(BENCHMARK_CAGR)。目前用全樣本CAGR判定，非IS窗——
        # H-11(IS/OOS切分)拍板前的過渡做法，見開發待辦追蹤.md H-01/H-11。
        Column("passes_alpha_gate", "bool"),
    ],
)

#: W-08：單日跳動門檻，與 diagnose_price_anomalies.py 的判定條件A同一個數字
#: （單一策略NAV由多檔股票加權組成，≥300%單日跳動不可能是真實組合報酬，
#: 只可能是底層價格資料錯誤——見該模組docstring起因段）。集中定義於此，
#: 避免stage1_marks.py（便宜、每次stage1都算）與diagnose_price_anomalies.py
#: （貴、按需跑的深度診斷）各自維護一份數字而悄悄漂移。
PRICE_JUMP_EXTREME = 3.0

#: W-08（2026-08-25 校準）：單月報酬門檻。起因：只用 PRICE_JUMP_EXTREME（單日）
#: 上線後拿深度掃描（diagnose_price_anomalies）交叉核對，發現 7,850 個策略被
#: 深掃判定「持有過資料異常股票」，但單日門檻只抓到 6 個——因為異常股票在多檔
#: 分散組合裡通常只佔小權重，單日跳動被稀釋到組合層級遠低於300%，但**累積到
#: 月報酬**仍可能造成實質CAGR灌水。用深掃算出的 CAGR_inflation_pp 當實測基準
#: 反推門檻：CAGR灌水 >1個百分點的283個策略，其 max_monthly_ret **全部**
#: ≥100%（最小值100.85%，283/283 全數命中）；以 100% 為門檻，套用到整個
#: 可用策略池只會標記 2.42%（364/15,041），canary 該有的低誤報率。
#: 兩個門檻用 OR 合併（見 stage1_marks.py），單日門檻保留給極端的單一交易日案例。
MONTHLY_JUMP_EXTREME = 1.0


RETURNS_META = Schema(
    name="returns_meta",
    # ⚠️ 刻意不設 expected_rows：階段1 的設計是「單策略失敗不中止全局」
    # （見 stage1_scan.py 的 DD-05）。若這裡卡死列數必須剛好等於候選總數，
    # 任何一個策略讀檔失敗就會讓 validate() 在寫出任何產物前直接 raise，
    # 連 scan_errors.parquet 都寫不出來，等於「一個失敗、全部陪葬」，
    # 與設計目標矛盾。完整性改由 stage1_scan._merge() 對 candidate_index
    # 做覆蓋率檢查（記警告，不中止）。
    columns=[
        Column(PK, "str"),
        Column("market", "cat", allowed=MARKETS),
        Column("hist_start", "str"),     # YYYY-MM
        Column("hist_end", "str"),
        Column("n_months", "int", ge=1),
    ],
)


# ============================================================================
# 階段 2b · 總經（W-02）
# ============================================================================

#: 投資時鐘四格（成長 × 通膨，各以歷史中位數切）
CLOCK_CELLS = ("復甦", "過熱", "停滯性通膨", "衰退")
#: 信心等級（樣本數不足時標低信心但仍可查，不留白）
CONFIDENCE = ("高", "中", "低")

#: collector 需交付的原始總經表 —— 這是研究部對資料端的**輸入契約**。
#: 刻意用 long 格式：指標數少、頻率不一（月/季/日），寬表會有大量 NaN 且難擴充。
MACRO_RAW = Schema(
    name="macro_raw",
    primary_key=["market", "indicator_key", "period"],
    columns=[
        Column("market", "cat", allowed=MARKETS),
        # 對應 macro_spec.SPEC 的鍵（growth / growth_alt / inflation / rate_level / rate_direction）
        Column("indicator_key", "str"),
        Column("period", "str"),          # 資料所描述的期間：YYYY-MM 或 YYYYQn
        # ⚠️ 刻意 nullable=False：本表用「該期間不存在這一列」表達缺值，
        # 不用 NaN 填值列表達（下游 align_by_lag 對查不到的 period 本來就會
        # 回傳 NaN，不需要來源檔自己先填一個 NaN 值列）。
        # 交付時「缺值請留 NaN」是指**不要用 0 頂替**，不是要求塞 NaN 值列進來——
        # 兩者容易混淆，validate_macro_raw.py 的提示訊息已對齊此說法。
        Column("value", "float"),
        # 單位必須明寫。z-score 對尺度不敏感，單位搞錯**不會在標準化後顯現**，
        # 是典型的靜默錯誤來源，故列為必填而非選填。
        Column("unit", "cat",
               allowed=("index", "percent", "percent_yoy", "percent_annualized", "score")),
        # ⭐ 若資料源提供（如 FRED/ALFRED），填入該筆的**實際發布日**。
        #    有這欄就能做真正的 point-in-time 對齊，不必倚賴 macro_spec 的推估滯後量。
        Column("release_date", "str", nullable=True),
    ],
)

#: 階段2b 產出的月頻特徵表（已套用發布滯後、已標準化、已分格）
MACRO_HISTORY = Schema(
    name="macro_history",
    primary_key=["market", "month"],
    columns=[
        Column("market", "cat", allowed=MARKETS),
        Column("month", "period"),
        # 原始值（已對齊到「該月底可得」的期間）
        Column("growth", "float", nullable=True),
        Column("inflation", "float", nullable=True),
        Column("rate_level", "float", nullable=True),
        Column("rate_direction", "float", nullable=True),
        # z-score（參數只用 in-sample 算並凍結）
        Column("growth_z", "float", nullable=True),
        Column("inflation_z", "float", nullable=True),
        Column("rate_level_z", "float", nullable=True),
        Column("rate_direction_z", "float", nullable=True),
        # 近 3 月平滑版（P4：歷史同步存平滑版，實戰查詢時平滑對平滑）
        Column("growth_z_s3", "float", nullable=True),
        Column("inflation_z_s3", "float", nullable=True),
        Column("rate_level_z_s3", "float", nullable=True),
        Column("rate_direction_z_s3", "float", nullable=True),
        # 投資時鐘四格（方法二）
        Column("clock_cell", "cat", allowed=CLOCK_CELLS, nullable=True),
        # 該月每個特徵實際引用的資料期間，供稽核前視偏誤
        Column("src_periods", "str", nullable=True),
    ],
)


#: H-18②（2026-09-01，軸線二·總經）：既有 `MACRO_HISTORY` 的 z-score／投資時鐘
#: 邊界是拿2000-2025全樣本算一次凍結的——早期月份的分類「偷看」了後期資料。
#: 這張表比較「全樣本凍結版」vs「5年滾動窗版」（每月只用過去5年資料算），
#: 量化兩者的clock_cell分類差多少，這個差距本身就是H-18要的論文內容。
MACRO_CLOCK_COMPARISON = Schema(
    name="macro_clock_comparison",
    primary_key=["market", "month"],
    columns=[
        Column("market", "cat", allowed=MARKETS),
        Column("month", "period"),
        Column("frozen_clock_cell", "cat", allowed=CLOCK_CELLS, nullable=True),
        Column("rolling_clock_cell", "cat", allowed=CLOCK_CELLS, nullable=True),
        # None：滾動窗尚未累積滿5年資料，兩邊無法比較（不是「不同」，是「還不能算」）
        Column("match", "bool", nullable=True),
    ],
)


# ============================================================================
# 階段 2a · Regime Dating（W-10）
# ============================================================================

#: 牛/熊/危機/盤整——zigzag 判定的連續區間標籤
REGIME_LABELS = ("牛", "熊", "危機", "盤整")

#: 階段2a 產出：純價格 zigzag 切割的連續區間（做法甲，不摻總經）
REGIME_TABLE = Schema(
    name="regime_table",
    primary_key=["market", "start"],
    columns=[
        Column("market", "cat", allowed=MARKETS),
        Column("start", "date"),
        Column("end", "date"),
        Column("label", "cat", allowed=REGIME_LABELS),
        Column("start_price", "float", ge=0),
        Column("end_price", "float", ge=0),
        Column("pct_change", "float"),
        Column("days", "int", ge=0),
        Column("ann_speed", "float", ge=0),
    ],
)


# ============================================================================
# 階段 2c · 交叉佐證（2a × 2b 匯流，W-11）
# ============================================================================

#: 牛段預期落在「高成長」格；熊/危機段預期落在「低成長」格。
#: 盤整無方向性預期，2c 刻意不檢查（做法甲精神：價格為主、總經僅對有方向性
#: 的段佐證，勉強配一個「盤整該落在哪格」的預期只是自欺）。
REGIME_EXPECTED_CELLS = {
    "牛": ("復甦", "過熱"),
    "熊": ("停滯性通膨", "衰退"),
    "危機": ("停滯性通膨", "衰退"),
}


def _check_consistency_nullability(df: pd.DataFrame) -> None:
    """checked=False 不該有 consistent 判定；checked=True 且有資料則必須有判定。"""
    bad_unchecked = df[~df["checked"] & df["consistent"].notna()]
    if len(bad_unchecked):
        _fail("regime_consistency", f"{len(bad_unchecked)} 筆 checked=False 卻仍有 consistent 判定")
    bad_checked = df[df["checked"] & (df["n_months_valid"] > 0) & df["consistent"].isna()]
    if len(bad_checked):
        _fail("regime_consistency", f"{len(bad_checked)} 筆 checked=True 且有資料卻缺 consistent 判定")


#: 階段2c 產出：regime 段 × 總經四格 一致性標記。
#: 「背離時段清單」= consistent==False 的子集，不另立 schema（同一張表過濾即得）。
REGIME_CONSISTENCY = Schema(
    name="regime_consistency",
    primary_key=["market", "seg_start"],
    columns=[
        Column("market", "cat", allowed=MARKETS),
        Column("seg_start", "date"),
        Column("seg_end", "date"),
        Column("label", "cat", allowed=REGIME_LABELS),
        Column("checked", "bool"),                                  # 盤整=False，不檢查
        Column("expected_group", "str", nullable=True),              # 未檢查段為 None
        Column("n_months_valid", "int", ge=0),                       # 段內非空月數（含盤整，供參考）
        Column("n_months_matched", "float", nullable=True, ge=0),    # 落入預期格的月數；未檢查/無資料為 None
        Column("pct_match", "float", nullable=True, ge=0, le=100),
        Column("majority_cell", "cat", allowed=CLOCK_CELLS, nullable=True),
        Column("consistent", "bool", nullable=True),                 # pct_match > 50% ；None=未檢查或無總經資料
    ],
    checks=(_check_consistency_nullability,),
)


# ============================================================================
# 階段 3 · HRP 分群（W-03）
# ============================================================================

#: 6 棵樹：3 組（TW/US/XM）× 2 種（normal/crisis）。crisis 需 regime 窗（階段2a／2c
#: 已完成，crisis 樹隨之補建）。
TREE_KEYS = ("TW", "US", "XM")
TREE_IDS = ("TW_normal", "US_normal", "XM_normal",
           "TW_crisis", "US_crisis", "XM_crisis")

CLUSTER_ASSIGN = Schema(
    name="cluster_assign",
    primary_key=[PK, "tree_id"],
    columns=[
        Column(PK, "str"),
        Column("tree_id", "cat", allowed=TREE_IDS),
        Column("cluster_L1", "int"),
        # cluster_L2 移除（H-04，2026-08-28）：從未被下游決策邏輯讀取，見 stage3_hrp.py 常數區註解
        Column("cluster_L3", "int"),
    ],
)

CLUSTER_META = Schema(
    name="cluster_meta",
    primary_key=["tree_id", "level", "cluster_id"],
    columns=[
        Column("tree_id", "cat", allowed=TREE_IDS),
        Column("level", "cat", allowed=("L1", "L3")),   # L2 移除，見 H-04
        Column("cluster_id", "int"),
        Column("n_members", "int", ge=1),
        Column("avg_intra_corr", "float", nullable=True),   # 單一成員的群無群內相關可算
        Column("representative_uid", "str"),
    ],
)

#: 危機期「共跌」的操作型定義（v9）：常態樹的兩個群，若在危機樹裡的成員多數被分
#: 進同一個危機群，代表它們在危機時塌在一起——即使常態時期看起來是不同的一群。
#: 只在 L1（粗層級，給 LLM 讀）算，跟 cluster_corr_matrix 的粒度一致。
CO_FAIL_REGIMES = Schema(
    name="co_fail_regimes",
    primary_key=["tree_key", "level", "cluster_normal"],
    columns=[
        Column("tree_key", "cat", allowed=TREE_KEYS),
        Column("level", "cat", allowed=("L1", "L3")),   # L2 移除，見 H-04
        Column("cluster_normal", "int"),                     # 常態樹的群 id
        Column("n_members", "int", ge=1),
        Column("crisis_dest_cluster", "int"),                # 危機期多數成員的去向（危機樹群id）
        Column("crisis_dest_share", "float", ge=0, le=1),     # 去向集中度
        Column("co_fail_peers", "str"),                       # 同去向的其他常態群，"|" 分隔（可能為空字串）
        Column("n_co_fail_peers", "int", ge=0),
    ],
)


#: LLM點③ cluster_story 的互補程度分級（**程式判定，不由LLM決定**）。
#: 沿用 T10 同一套抗幻覺原則：判斷歸程式、LLM 只為既定判決寫字。
#:
#: 門檻經敏感度分析驗證（2026-08-26，`research/complementarity_sensitivity.py`，
#: 六棵樹168對，非LLM）：
#:   實際使用範圍（3棵normal樹，cluster_story.py只跑normal樹）：
#:     同市場配對最低相關0.5387、跨市場配對最高相關0.5524，兩者有微幅重疊。
#:     high門檻0.5的安全上限是0.5387（同市場最低值）——網格掃描證實high門檻
#:     可以到0.53都還能保證「同市場配對0個被誤判成高互補」，0.5落在安全區內，
#:     margin=0.0387（不寬裕但安全）。
#:   low門檻0.8：落在一段連續分布裡（0.70~0.90之間29個配對，無自然斷點），
#:     不是資料自己浮現的分界，中/低的區分效力比high門檻弱，只能算合理的人為切點。
#:   ⚠️ **這組門檻只驗證過normal樹，不可套用到crisis樹**——crisis樹的相關係數
#:     行為完全不同（同市場配對危機時可能大幅解相關，實測低至0.2556；跨市場配對
#:     危機時反而可能大幅同步，實測高達0.9710，與「危機時全球市場相關性趨近1」
#:     的現象一致）。cluster_story.py 的 DEFAULT_TREES 本來就只含 normal 樹，
#:     此限制目前不影響現有產物，但未來若要對 crisis 樹做互補判定須重新校準。
COMPLEMENTARITY_CUTS = {"高": 0.5, "中": 0.8}

#: LLM點③ 產出：群對層級的互補性解釋（離線一次性凍結，供 T5/T13 的
#: explanation_text 引用）。文字由 LLM 寫，但 complementarity 欄位是程式算的。
CLUSTER_STORY = Schema(
    name="cluster_story",
    primary_key=["tree_id", "level", "cluster_a", "cluster_b"],
    columns=[
        Column("tree_id", "cat", allowed=TREE_IDS),
        Column("level", "cat", allowed=("L1", "L3")),   # L2 移除，見 H-04
        Column("cluster_a", "int"),
        Column("cluster_b", "int"),
        Column("corr", "float", ge=-1, le=1),
        Column("complementarity", "cat", allowed=("高", "中", "低")),   # 程式判定
        Column("co_fail", "bool"),           # 危機期是否塌進同一群（來自 co_fail_regimes）
        Column("mechanism_note", "str"),      # LLM：兩群客觀差異在哪
        Column("complement_note", "str"),     # LLM：為既定的互補程度判決寫說明
        Column("caveat", "str"),              # LLM：此判讀的限制
        Column("model", "str"),
    ],
)


#: H-08（2026-08-30）：單群身份的 LLM 解釋（新產出，非cluster_story）。cluster_story
#: 只做「兩群為什麼互補」，這裡回答「這一群本身到底是什麼意思」——老師原話「才知道
#: 什麼時候用哪一個」，也是銜接S-01（群→總經介面）的素材來源。
#: 輸入全部來自H-06（cluster_profile_quant，定量時間型態）+ cluster_story的
#: _cluster_profiles（橫斷面成分側寫）+ co_fail_regimes（危機期關聯，供參考不代表
#: 選群決策依據，H-15已定案）。跟cluster_story同一套抗幻覺鐵則，額外多一條：
#: **不推論「適合什麼總經環境」**——那是S-01/S-02之後決策層的工作，這裡只做客觀
#: 身份描述，避免在總經知識缺席的情況下編造適配性因果。
CLUSTER_IDENTITY = Schema(
    name="cluster_identity",
    primary_key=["tree_id", "level", "cluster_id"],
    columns=[
        Column("tree_id", "cat", allowed=TREE_IDS),
        Column("level", "cat", allowed=("L1",)),   # 只做L1，跟H-06/H-07/cluster_story同樣理由
        Column("cluster_id", "int"),
        Column("identity_label", "str"),      # LLM：一句話身份標籤
        Column("mechanism_note", "str"),      # LLM：這群為什麼長這樣（成分驅動）
        Column("performance_pattern", "str"), # LLM：績效隨時間的型態（H-06時間序列驅動）
        Column("caveat", "str"),              # LLM：此描述的限制（群內異質性等）
        Column("model", "str"),
    ],
)


#: S-01（2026-08-31）：群→總經決策層的介面。方向C 的接縫——H-08 產出的「群身份
#: 描述」必須是總經層能消費的結構化格式，不能只是一段自由文字。
#:
#: 🔴 **為什麼不直接把 cluster_identity 的四個欄位全部餵過去（S-03 的具體落實）**：
#: `cluster_identity.py` 的 prompt 是**一次性把全部側寫（含觀察期間/最佳年/最差年
#: 等實際年份）餵給同一個LLM呼叫**，`identity_label`/`mechanism_note`
#: /`performance_pattern`/`caveat` 四個輸出欄位共用同一個上下文——即使實測16群裡
#: `performance_pattern`（16/16）與`caveat`（多數）明確引用了真實年份（如「最佳年
#: 2009」），`identity_label`/`mechanism_note`目前沒有外洩年份，**也只是這次運氣好，
#: 不是架構上保證**：LLM在生成這兩欄時技術上仍拿得到年份資訊，只是schema描述引導它
#: 不要用。跟本專案一路的原則（「不給LLM有機會誤用的資訊，而非事後信任它不會誤用」，
#: 見cluster_story/T10的抗幻覺設計）不一致，故不能直接當作決策層的正式介面。
#:
#: **定案做法**：只收 `cluster_identity.identity_label`（短句、風險最低，仍搭配下方
#: 測試做逐群年份比對當第二道防線）；`mechanism_note`/`performance_pattern`/`caveat`
#: **一律不納入**——S-02 若需要更豐富的LLM文字脈絡，須另外設計一個「輸入內容從頭
#: 就不含任何年份/日期」的專屬prompt，不可重用cluster_identity的既有輸出。
#: 其餘欄位全部來自 `CLUSTER_PROFILE_QUANT`（純程式算，非LLM），**排除掉四個帶
#: 日曆年份的欄位**（window_start_year／window_end_year／best_year／worst_year）——
#: 對應的報酬幅度（best_year_ret／worst_year_ret）保留，因為那是純數值統計、
#: 沒有日曆錨點，不會讓LLM反推出「這是哪一年」。
CLUSTER_MACRO_INTERFACE = Schema(
    name="cluster_macro_interface",
    primary_key=["tree_id", "level", "cluster_id"],
    columns=[
        Column("tree_id", "cat", allowed=TREE_IDS),
        Column("level", "cat", allowed=("L1",)),
        Column("cluster_id", "int"),
        Column("identity_label", "str"),
        Column("n_members", "int", ge=1),
        Column("pct_TW", "float", ge=0, le=1),
        Column("pct_US", "float", ge=0, le=1),
        Column("top1_factor_type", "str", nullable=True),
        Column("top1_factor_type_pct", "float", ge=0, le=1, nullable=True),
        Column("top1_F1", "str", nullable=True),
        Column("top1_F1_pct", "float", ge=0, le=1, nullable=True),
        Column("top1_C_source", "str", nullable=True),
        Column("top1_C_source_pct", "float", ge=0, le=1, nullable=True),
        Column("pct_v1", "float", ge=0, le=1),
        Column("CAGR_median", "float"),
        Column("MDD_median", "float"),
        Column("smallcap_share_median", "float", ge=0, le=1),
        Column("avg_intra_corr", "float", nullable=True),
        Column("n_years", "int", ge=1),
        Column("n_years_positive", "int", ge=0),
        Column("pct_years_positive", "float", ge=0, le=1),
        Column("best_year_ret", "float"),      # 幅度保留，年份本身不給（S-03）
        Column("worst_year_ret", "float"),     # 同上
        Column("annual_ret_mean", "float"),
        Column("annual_ret_std", "float", nullable=True),
        Column("quarterly_ret_std", "float", nullable=True),
    ],
)


#: S-02（2026-08-31）：群的投資時鐘四格條件式績效——決策層資訊源設計的核心素材。
#: 純程式算（把 `macro_performance.parquet` 的策略層級數字，用跟
#: `stage3_hrp._cluster_meta_and_corr`／H-06 同一套「群代表=成員簡單平均」邏輯
#: 彙整到群層級），**沒有LLM**。
#:
#: 🔴 **為什麼需要這張表，而不是直接把 CLUSTER_MACRO_INTERFACE 的 CAGR_median 等
#: 欄位當決策依據**：那些是「這個群整體表現好不好」的**無條件**統計量，若決策層
#: 直接看到「群X的CAGR中位數23.5%」，幾乎必然導向「總是選CAGR最高的群」——這正是
#: H-10/H-12已經證實的陷阱在總經決策層的翻版（H-12實測：無多樣性限制的純品質
#: 排序在US/XM會塌縮成集中在單一群的賭注，見H-12結果）。本表提供的是**條件式**
#: 資訊——「這個群在『過熱』時平均月報酬是多少」，決策層要做的判斷才有意義：
#: 拿當下總經狀態對應的clock_cell去查表，而不是無條件地選歷史最強的群。
#:
#: clock_cell（復甦/過熱/停滯性通膨/衰退）是**投資時鐘的regime分類，不是日曆
#: 年份**，符合S-03「決策層看不到日期」的要求。
CLUSTER_MACRO_CONDITIONAL = Schema(
    name="cluster_macro_conditional",
    primary_key=["tree_id", "level", "cluster_id", "clock_cell"],
    columns=[
        Column("tree_id", "cat", allowed=TREE_IDS),
        Column("level", "cat", allowed=("L1",)),
        Column("cluster_id", "int"),
        Column("clock_cell", "cat", allowed=CLOCK_CELLS),
        Column("n_members_with_data", "int", ge=0),
        Column("avg_ret_mean", "float", nullable=True),    # 群內成員avg_ret的平均
        Column("avg_ret_median", "float", nullable=True),
        Column("win_ratio_mean", "float", nullable=True, ge=0, le=1),
        Column("pct_high_confidence", "float", nullable=True, ge=0, le=1),
    ],
)


#: S-07（2026-08-31）：LLM決策層（`decision_layer_arms.llm_decision`）的重複執行
#: 穩定度。同一輸入（同一組總經狀態+候選群素材）重複呼叫N次，量化決策一致性——
#: 也是方向D（抗幻覺）的實證素材，見開發待辦追蹤.md S-07。一列＝一個(tree_id,
#: market,clock_cell)情境的N次重複結果彙總。
DECISION_REPEATABILITY = Schema(
    name="decision_repeatability",
    primary_key=["tree_id", "market", "clock_cell"],
    columns=[
        Column("tree_id", "cat", allowed=TREE_IDS),
        Column("market", "cat", allowed=MARKETS),
        Column("clock_cell", "cat", allowed=CLOCK_CELLS),
        Column("n_repeats", "int", ge=2),
        Column("exact_match_rate", "float", ge=0, le=1),   # 眾數結果佔全部次數的比例
        Column("mean_pairwise_jaccard", "float", ge=0, le=1),
        # nullable=True：內容可能是空字串（完全不穩定時core為空、完全穩定時fringe為空），
        # 空字串經CSV往返讀寫後會變成NaN（pandas.read_csv預設na_values含空字串），
        # 2026-08-31 code review 抓到的真實bug——同類問題H-10的co_fail_peers已踩過一次。
        Column("stable_core", "str", nullable=True),        # 每次都選中的群，"|"分隔
        Column("unstable_fringe", "str", nullable=True),    # 只有部分次數選中的群，"|"分隔
        Column("rule_in_llm_rate", "float", ge=0, le=1),  # A_rule的選擇是否每次都被LLM包含
        Column("all_runs", "str"),           # 逐次結果，供事後稽核，如"[1,3]|[1,2]|[1,3]..."
        Column("model", "str"),
    ],
)


#: H-06（2026-08-28）：群的定量特徵表，老師具體要求的角度——「某幾年固定賺錢、
#: 某幾年賠錢」「季度或年份的獲利表現是不是不太一樣」。純程式算、無LLM，供論文
#: 直接呈現，也是之後 H-08（單群LLM解釋）的輸入素材。只在 L1 算（給LLM讀/論文
#: 呈現的粗粒度，L3群數太多不適合逐群描述，跟 CLUSTER_STORY 同樣的理由）。
#: 群代表序列＝成員報酬簡單平均（與 stage3_hrp._cluster_meta_and_corr 同一定義），
#: 年/季報酬皆為該期間內逐月複利。
CLUSTER_ANNUAL_RETURNS = Schema(
    name="cluster_annual_returns",
    primary_key=["tree_id", "level", "cluster_id", "year"],
    columns=[
        Column("tree_id", "cat", allowed=TREE_IDS),
        Column("level", "cat", allowed=("L1",)),
        Column("cluster_id", "int"),
        Column("year", "int", ge=1999, le=2026),
        Column("ret", "float"),
        Column("n_months", "int", ge=1, le=12),   # 首尾年可能不足12個月
    ],
)

CLUSTER_QUARTERLY_RETURNS = Schema(
    name="cluster_quarterly_returns",
    primary_key=["tree_id", "level", "cluster_id", "year", "quarter"],
    columns=[
        Column("tree_id", "cat", allowed=TREE_IDS),
        Column("level", "cat", allowed=("L1",)),
        Column("cluster_id", "int"),
        Column("year", "int", ge=1999, le=2026),
        Column("quarter", "int", ge=1, le=4),
        Column("ret", "float"),
        Column("n_months", "int", ge=1, le=3),
    ],
)

CLUSTER_PROFILE_QUANT = Schema(
    name="cluster_profile_quant",
    primary_key=["tree_id", "level", "cluster_id"],
    columns=[
        Column("tree_id", "cat", allowed=TREE_IDS),
        Column("level", "cat", allowed=("L1",)),
        Column("cluster_id", "int"),
        Column("n_members", "int", ge=1),
        Column("pct_TW", "float", ge=0, le=1),
        Column("pct_US", "float", ge=0, le=1),
        Column("top1_factor_type", "str", nullable=True),
        Column("top1_factor_type_pct", "float", ge=0, le=1, nullable=True),
        Column("top1_F1", "str", nullable=True),
        Column("top1_F1_pct", "float", ge=0, le=1, nullable=True),
        Column("top1_C_source", "str", nullable=True),
        Column("top1_C_source_pct", "float", ge=0, le=1, nullable=True),
        Column("pct_v1", "float", ge=0, le=1),
        Column("CAGR_median", "float"),
        Column("MDD_median", "float"),
        Column("smallcap_share_median", "float", ge=0, le=1),
        Column("avg_intra_corr", "float", nullable=True),
        Column("window_start_year", "int"),
        Column("window_end_year", "int"),
        Column("n_years", "int", ge=1),
        Column("n_years_positive", "int", ge=0),
        Column("pct_years_positive", "float", ge=0, le=1),
        Column("best_year", "int"),
        Column("best_year_ret", "float"),
        Column("worst_year", "int"),
        Column("worst_year_ret", "float"),
        Column("annual_ret_mean", "float"),
        Column("annual_ret_std", "float", nullable=True),
        Column("quarterly_ret_std", "float", nullable=True),
    ],
)


#: H-09（2026-08-29）：有效獨立賭注數（Effective Number of Bets），一棵normal樹一列。
#: 回答老師「HRP到底幫你少了多少東西」，也是「免費午餐」大小的理論上限。
#: 方法見 hrp.effective_number_of_bets()（PCA熵版，Meucci 2009/López de Prado 2016）。
#: 只做normal樹（crisis樹樣本量太小，見effective_bets.py docstring）。
EFFECTIVE_BETS = Schema(
    name="effective_bets",
    primary_key=["tree_id"],
    columns=[
        Column("tree_id", "cat", allowed=TREE_IDS),
        Column("tree_key", "cat", allowed=("TW", "US", "XM")),
        Column("n_strategies", "int", ge=1),
        Column("enb_raw", "float", ge=1),          # 理論上ENB下限為1（全部完美相關）
        Column("n_clusters_l1", "int", ge=1),
        Column("enb_clusters", "float", ge=1),
        Column("redundancy_ratio", "float", ge=1, nullable=True),
        Column("k_vs_enb_raw", "float", ge=0, nullable=True),
        Column("cluster_independence", "float", ge=0, nullable=True),
    ],
)


#: H-10（2026-08-29）：群內代表策略挑選規則，一群一列。`m_target`/`max_share_of_avg`
#: 是自由參數（每次跑可能不同），主鍵故意不含它們——**同一次跑**（同一個 `m`）內
#: 每群只會出現一列，跨不同`m`的跑是分開的CSV檔（檔名帶m），不會混進同一張表裡。
CLUSTER_REPRESENTATIVES = Schema(
    name="cluster_representatives",
    primary_key=["tree_id", "level", "cluster_id"],
    columns=[
        Column("tree_id", "cat", allowed=TREE_IDS),
        Column("level", "cat", allowed=("L1",)),
        Column("cluster_id", "int"),
        Column("n_members", "int", ge=1),
        Column("m_target", "int", ge=1),
        Column("n_picked", "int", ge=1),
        Column("n_backfilled", "int", ge=0),
        Column("picked_uids", "str"),
        Column("naive_top_m_uids", "str"),
        Column("avg_intra_corr_cluster", "float", nullable=True),
        Column("avg_pairwise_corr_picked", "float", nullable=True),
        Column("avg_pairwise_corr_naive", "float", nullable=True),
        Column("co_fail_peers", "str", nullable=True),   # 警示欄位，非篩選依據（H-15）
    ],
)


# ============================================================================
# H-11 · IS/OOS（stage3_hrp_isoos.py，2026-08-29）
# ============================================================================

#: IS窗建的樹，tree_id 額外加 `_IS` 後綴跟主線六棵樹（TREE_IDS）區隔——
#: **完全獨立的命名空間、獨立的輸出目錄**，不會跟 `_frozen/stage3/` 的正式產物混淆。
ISOOS_TREE_IDS = ("TW_normal_IS", "US_normal_IS", "XM_normal_IS")

CLUSTER_ASSIGN_ISOOS = Schema(
    name="cluster_assign_isoos",
    primary_key=[PK, "tree_id"],
    columns=[
        Column(PK, "str"),
        Column("tree_id", "cat", allowed=ISOOS_TREE_IDS),
        Column("cluster_L1", "int"),
        Column("cluster_L3", "int"),
    ],
)

CLUSTER_META_ISOOS = Schema(
    name="cluster_meta_isoos",
    primary_key=["tree_id", "level", "cluster_id"],
    columns=[
        Column("tree_id", "cat", allowed=ISOOS_TREE_IDS),
        Column("level", "cat", allowed=("L1", "L3")),
        Column("cluster_id", "int"),
        Column("n_members", "int", ge=1),
        Column("avg_intra_corr", "float", nullable=True),
        Column("representative_uid", "str"),
    ],
)

#: H-13 的核心素材：同一批群（IS窗定義、凍結），群間相關係數在 IS 窗跟 OOS 窗
#: 各算一次，比較是否維持同樣的（低）相關——老師原話：「in sample都很低，可是
#: out sample會不會還是很低」。群定義完全不變，唯一差的是拿哪一段月份的報酬去算相關。
ISOOS_CORR_COMPARISON = Schema(
    name="isoos_corr_comparison",
    primary_key=["tree_id", "level", "cluster_a", "cluster_b"],
    columns=[
        Column("tree_id", "cat", allowed=ISOOS_TREE_IDS),
        Column("level", "cat", allowed=("L1",)),   # 只在L1做，跟cluster_story/H-06/H-07同樣理由
        Column("cluster_a", "int"),
        Column("cluster_b", "int"),
        Column("corr_is", "float", ge=-1, le=1),
        Column("corr_oos", "float", ge=-1, le=1, nullable=True),   # OOS月數不足時可能算不出
        Column("delta", "float", nullable=True),                  # corr_oos - corr_is
        Column("complementarity_is", "cat", allowed=("高", "中", "低")),
        Column("complementarity_oos", "cat", allowed=("高", "中", "低"), nullable=True),
        Column("complementarity_stable", "bool", nullable=True),   # IS/OOS互補程度判定是否一致
    ],
)


#: H-25（2026-09-01）：互補性的**分群粒度效應**。
#:
#: 背景：H-03 把 L1 群數從寫死的 8 改成輪廓係數決定（TW6/US7/XM3）之後，
#: 39 對 L1 群對裡**沒有任何一對達到「高互補」**（相關<0.5），而舊 k=8 時代
#: XM_normal 有 5 對高互補。查證後確認**不是策略真的沒有互補性、也不是門檻要改**，
#: 而是分群粒度變粗造成的**聚合效應**：群代表＝成員報酬的簡單平均，平均的成員愈多，
#: 個別策略的特異成分互相抵消得愈徹底，代表序列就愈趨近該市場的大盤，
#: 跨市場相關自然被推高。舊 k=8 的 5 對高互補全部來自同一個 305 檔的小群，
#: k=3 後那 305 檔被併進 6,679 檔的全台股群裡。
#:
#: 本表就是這個效應的量化：同一批策略、同一段共同窗、同一套判定門檻，
#: 只改變分群層級（L1 粗 vs L3 細），比較跨市場／同市場配對的互補分布。
#: **同市場配對是天然的對照組**——群大小與月份數跟跨市場配對同量級，
#: 可排除「小群估計雜訊」這個替代解釋。
COMPLEMENTARITY_GRANULARITY = Schema(
    name="complementarity_granularity",
    primary_key=["tree_id", "level", "min_members", "pair_type"],
    columns=[
        Column("tree_id", "cat", allowed=TREE_IDS),
        Column("level", "cat", allowed=("L1", "L3")),
        Column("min_members", "int", ge=1),      # 納入統計的群最小成員數（穩健性用）
        Column("pair_type", "cat", allowed=("same", "cross")),
        Column("n_clusters", "int", ge=0),
        Column("n_pairs", "int", ge=0),
        Column("corr_min", "float", ge=-1, le=1, nullable=True),
        Column("corr_median", "float", ge=-1, le=1, nullable=True),
        Column("corr_max", "float", ge=-1, le=1, nullable=True),
        Column("n_high", "int", ge=0),           # 相關 < COMPLEMENTARITY_CUTS["高"]
        Column("n_mid", "int", ge=0),
        Column("n_low", "int", ge=0),
        Column("pct_high", "float", ge=0, le=1, nullable=True),
    ],
)

#: H-26／H-27／H-21（2026-09-03）：anchored walk-forward × 精選比例 × 分配方式的
#: 完整交叉矩陣。**2026-09-02 老師會議定案的畢業必考題主體。**
#:
#: 一列 = 一個 (樹 × 窗口方案 × 窗次 × 精選比例 × 分配方式 × 對照組) 的評估格。
#: `B_all` 不受比例/分配影響，每窗只有一列（`ratio="all"`／`allocation="n/a"`）。
#:
#: 🔴 三個維度**刻意不預先挑選最優值**——先挑再驗證＝用同一批資料挑參數又驗證，
#: 是上帝視角（本專案已因同一錯誤撤銷過 H-11 原提案）。老師明講精選比例是
#: 「一個需要研究的對象的參數」＝實驗變數。結論看「A_hrp 在多少比例的格子裡贏」。
#:
#: `ratio` 的 `legacy` 是保留現行 m=5/群 的總量當校驗點（TW30/US35/XM15），
#: 用來驗算新程式能否重現已報告給老師的 H-12 數字。
WALKFORWARD_MATRIX = Schema(
    name="walkforward_matrix",
    primary_key=["tree_key", "scheme", "window_no", "k_mode", "ratio", "allocation", "group"],
    columns=[
        Column("tree_key", "cat", allowed=("TW", "US", "XM")),
        Column("scheme", "str"),                       # A~L（anchored）／R（rolling）
        Column("mode", "cat", allowed=("anchored", "rolling")),
        #: H-26b：群數 k 的來源。`fixed`＝L1_TARGET（H-03 用完整窗選的 6/7/3，
        #: 形式上有前視偏誤）；`silhouette_is`＝每個 IS 窗只用該窗資料重選（乾淨版）。
        #: 兩者共用同一個 linkage，只是切的位置不同。實測前視偏誤**是實質的**——
        #: 台股 15 個 IS 窗只有 4 個選出 k=6，且有一個窗在 k=6 的輪廓係數是負的。
        Column("k_mode", "cat", allowed=("fixed", "silhouette_is")),
        Column("window_no", "int", ge=1),
        Column("n_windows", "int", ge=1),
        Column("min_is_months", "int", ge=1),
        Column("oos_len_months", "int", ge=1),
        Column("is_start", "str"), Column("is_end", "str"),
        Column("oos_start", "str"), Column("oos_end", "str"),
        Column("n_is_months", "int", ge=1),
        Column("n_oos_months", "int", ge=1),
        Column("n_universe", "int", ge=1),
        Column("n_clusters", "int", ge=1),
        Column("ratio", "str"),                        # legacy／0.01…／all(B組)
        Column("allocation", "str"),                   # equal／proportional／n/a(B組)
        Column("group", "cat", allowed=("A_hrp", "B_all", "D_top_cagr", "E_top_calmar")),
        Column("target_total", "int", ge=1),
        #: 等量分配時有多少群撞到「配額 > 群成員數」的天花板（實測僅 TW/10%/equal 觸發）
        Column("n_capped_clusters", "int", ge=0),
        #: A 組因多樣性門檻太嚴而改用純品質補位的檔數（H-10 的 backfill 機制）
        Column("n_backfilled", "int", ge=0),
        Column("n_members", "int", ge=1),
        Column("is_cagr", "float"), Column("is_mdd", "float", le=0),
        Column("is_sharpe", "float", nullable=True),
        Column("oos_cagr", "float"), Column("oos_mdd", "float", le=0),
        Column("oos_sharpe", "float", nullable=True),
        Column("oos_enb", "float", ge=0, nullable=True),   # 成員>400時不算，留NaN
        Column("n_clusters_covered", "int", ge=0),
        Column("max_cluster_share", "float", ge=0, le=1, nullable=True),
    ],
)

#: H-26d（2026-09-04）：共同窗選擇的穩健性——即 `落差處理方案_v1.md` 落差3 規劃的
#: 「方案 E」。`HRP_ROBUSTNESS_WINDOW_TW` 這個常數留了下來但**從未被執行過**
#: （2026-09-04 code review 查全域搜尋確認零引用），本表是它的首次落實。
#:
#: 台股共同窗定在 2007-01 是個判斷（再往前每多一個月要犧牲大量策略）。本表用
#: 2003-01 窗（只剩 3,272 檔、49%）重建一次，跟主線分群比對 ARI，回答
#: 「群結構是不是窗口選擇的產物」。**兩邊切同一個 k**，否則 ARI 會同時混進
#: 「窗不同」與「k 不同」兩個因素。`ari_random_floor` 是把標籤打散後的 ARI，
#: 給讀者一個「0 附近長怎樣」的尺度參照。
WINDOW_ROBUSTNESS = Schema(
    name="window_robustness",
    primary_key=["tree_key", "k"],
    columns=[
        Column("tree_key", "cat", allowed=("TW", "US", "XM")),
        Column("k", "int", ge=2),
        Column("n_common", "int", ge=1),      # 兩窗共同的策略數
        Column("n_main", "int", ge=1),
        Column("n_alt", "int", ge=1),
        Column("ari", "float", ge=-1, le=1),
        Column("ari_random_floor", "float", ge=-1, le=1),
        Column("is_main_k", "bool"),          # 是否為主線採用的 L1_TARGET
    ],
)

#: H-25d（2026-09-04）：L3 細粒度互補性的 walk-forward 驗證。
#:
#: 補 H-25 的缺口——H-25 查出「L1 沒有高互補是聚合效應，降到 L3 後跨市場有 62.6%
#: 高互補」，但那全部用**完整窗**算，從未經 OOS 驗證。本表把老師對 H-13 的原話
#: 「in sample 都很低，可是 out sample 會不會還是很低」從 L1 搬到 L3。
#:
#: 做法比照 H-11：**群定義只用 IS 窗決定並凍結，OOS 只用同一批群重算相關**。
#: `stability_rate` ＝「IS 判定為高互補」的配對中，OOS 仍為高互補的比例。
#: TW/US 樹只有同市場配對，是天然對照組。
L3_ISOOS = Schema(
    name="l3_isoos",
    primary_key=["tree_key", "scheme", "window_no", "pair_type"],
    columns=[
        Column("tree_key", "cat", allowed=("TW", "US", "XM")),
        Column("scheme", "str"),
        Column("window_no", "int", ge=1),
        Column("is_start", "str"), Column("is_end", "str"),
        Column("oos_start", "str"), Column("oos_end", "str"),
        Column("n_clusters", "int", ge=2),     # 成員>=門檻的 L3 群數
        Column("pair_type", "cat", allowed=("same", "cross")),
        Column("n_pairs", "int", ge=1),
        Column("is_pct_high", "float", ge=0, le=1),
        Column("oos_pct_high", "float", ge=0, le=1),
        Column("corr_is_median", "float", ge=-1, le=1),
        Column("corr_oos_median", "float", ge=-1, le=1),
        Column("n_is_high", "int", ge=0),
        Column("n_is_high_stays_high", "int", ge=0),
        Column("stability_rate", "float", ge=0, le=1, nullable=True),  # n_is_high=0 時無定義
    ],
)

#: H-27b（2026-09-04）：周轉率與交易成本。
#:
#: 補 walk-forward 矩陣「全部為毛報酬、未評估實務可行性」的限制。
#: 🔴 **必須算到股票層**：策略之間持股重疊，A 賣出的股票可能正好是 B 買進的，
#: 合併投組裡兩筆會互相抵銷。策略越多、抵銷越多，**每一元的周轉率反而可能下降**，
#: 所以不能用「策略數」當周轉率的代理。
#:
#: `monthly_turnover` = 0.5×Σ|w_t − w_{t−1}|（單邊），未扣價格漂移 → **高估**，
#: 對「成本會不會翻轉結論」是保守方向。
#: `net_cagr_*bp` = 毛 CAGR − 年周轉率 × 單邊成本率 × 2（買賣各一次）。
TURNOVER_COST = Schema(
    name="turnover_cost",
    primary_key=["tree_key", "ratio", "allocation"],
    columns=[
        Column("tree_key", "cat", allowed=("TW", "US", "XM")),
        Column("ratio", "str"),
        Column("allocation", "str"),
        Column("n_strategies", "int", ge=1),
        Column("n_stocks_avg", "float", ge=1),      # 去重後的平均持股檔數
        Column("monthly_turnover", "float", ge=0),
        Column("annual_turnover", "float", ge=0),
        Column("gross_cagr", "float"),
        Column("net_cagr_5bp", "float"),
        Column("net_cagr_10bp", "float"),
        Column("net_cagr_20bp", "float"),
        Column("net_cagr_30bp", "float"),
    ],
)

#: H-26c（2026-09-04）：walk-forward 結果的統計檢定。
#:
#: 🔴 存在理由：`walkforward_matrix` 的 2,700 個 A_hrp 格子**共用同一段歷史**
#: （45 個窗次只對應 21 種不重複的 OOS 區間，且區間互相包含），**2,700 不是樣本數**。
#: 本表只在「方案內部互不重疊的窗次」上做二項檢定——同一方案的窗次依建構方式
#: 必然連續不重疊（見 `t_walkforward_schemes_are_mechanical`），跨方案才重疊。
#:
#: ⚠️ **刻意不合併各方案的 p 值**（Fisher 等方法要求獨立，但方案間共用歷史）。
#: 逐方案各自報告，「每個方案單獨檢定都顯著」比一個合併的小 p 值誠實。
#: ⚠️ 同期的三棵樹不完全獨立（XM 含台美兩市場的策略），`tree_scope="ALL"` 把它們
#: 當獨立單位處理，論文須註明此假設。
WALKFORWARD_SIGNIFICANCE = Schema(
    name="walkforward_significance",
    primary_key=["opponent", "metric", "scheme", "tree_scope"],
    columns=[
        Column("opponent", "cat", allowed=("B_all", "D_top_cagr", "E_top_calmar")),
        Column("metric", "cat", allowed=("cagr", "mdd", "calmar")),
        Column("scheme", "str"),
        Column("tree_scope", "cat", allowed=("ALL", "TW", "US", "XM")),
        Column("n_units", "int", ge=1),          # 單位＝(樹, 方案, 窗次)
        Column("n_wins", "int", ge=0),
        Column("win_rate", "float", ge=0, le=1),
        Column("p_value", "float", ge=0, le=1),
        Column("n_windows_per_tree", "int", ge=1),
        Column("significant_05", "bool"),
    ],
)

#: H-26b（2026-09-04）：群數 k 的前視偏誤診斷。
#:
#: `L1_TARGET`（6/7/3）是 H-03 用**完整窗**的輪廓係數選出來的，但 walk-forward
#: 的早期窗只該知道 IS 期間的資訊——形式上這是前視偏誤，跟 H-18②（總經 z-score
#: 全樣本凍結）、H-11 原提案（挑涵蓋 COVID 的窗）是同一類錯誤。
#: 本表逐 IS 窗只用該窗資料重選 k，看是否仍得到同一個數字：
#: **全數一致 → 前視偏誤形式存在、實質無影響；有不一致 → 須把 k_mode 納入矩陣重跑。**
K_STABILITY = Schema(
    name="k_stability",
    primary_key=["tree_key", "is_start", "is_end"],
    columns=[
        Column("tree_key", "cat", allowed=("TW", "US", "XM")),
        Column("is_start", "str"), Column("is_end", "str"),
        Column("n_is_months", "int", ge=1),
        Column("n_strategies", "int", ge=1),
        Column("linkage_method", "str"),          # single／ward，須與主線同一條選法
        Column("k_fixed", "int", ge=2),           # stage3_hrp.L1_TARGET 的值
        Column("k_is_selected", "int", ge=2),     # 只用 IS 資料選出的 k
        Column("same_as_fixed", "bool"),
        Column("sil_at_selected", "float"),
        Column("sil_at_fixed", "float", nullable=True),
        Column("sil_gap", "float", nullable=True),        # selected − fixed，愈小影響愈輕
        Column("max_share_at_fixed", "float", ge=0, le=1, nullable=True),
        Column("degenerate_fallback", "bool"),    # 全域最佳解是退化解、已被排除
    ],
)

#: H-25b（2026-09-02）：免費午餐清單——把「免費午餐藏在小而特化的群 × 另一個市場」
#: 這句話落實成**具體可指認的群**。一列＝XM_normal 樹的一個 L3 群。
#:
#: `pct_high_complement` 是關鍵欄位：該群跟**對面市場**的候選群裡，有多少比例達到
#: 高互補（相關 < COMPLEMENTARITY_CUTS["高"]）。=1.0 代表「跟對面市場每一群配都
#: 能吃到分散效果」，是配置時最有價值的群。
#:
#: ⚠️ 只在 XM_normal 做——TW_normal/US_normal 樹內全是同市場配對，沒有跨市場對象，
#: 這張表在那兩棵樹上沒有意義。
FREE_LUNCH_SHORTLIST = Schema(
    name="free_lunch_shortlist",
    primary_key=["tree_id", "level", "cluster_id"],
    columns=[
        Column("tree_id", "cat", allowed=TREE_IDS),
        Column("level", "cat", allowed=("L3",)),
        Column("cluster_id", "int"),
        Column("market", "cat", allowed=MARKETS),
        Column("n_members", "int", ge=1),
        Column("top_F1", "str"),               # 前兩大一階因子及佔比
        Column("top_C_source", "str"),         # 前兩大條件訊號來源及佔比
        Column("pct_v1", "float", ge=0, le=1),
        Column("CAGR_median", "float"),
        Column("MDD_median", "float", le=0),
        Column("n_cross_partners", "int", ge=0),      # 對面市場的候選群數
        Column("n_high_complement", "int", ge=0),
        Column("pct_high_complement", "float", ge=0, le=1),
        Column("min_cross_corr", "float", ge=-1, le=1),
        Column("best_partner_cluster", "int"),
        Column("universal", "bool"),           # pct_high_complement == 1.0
    ],
)

#: H-12（2026-08-30）：四組對照實驗——老師的驗證題「選30多支 vs 狂灑下去會不會一樣」。
#: A=HRP跨群選代表／B=全部灑／C=隨機同樣數量（200次抽樣彙總）／D=純CAGR前N名／
#: E=純Calmar前N名（不設多樣性限制，2026-08-30新增，用來把「品質指標選擇」跟
#: 「多樣性限制」兩件事拆開看——A跟D的差異同時混了兩個因素，E只變品質指標不變
#: 多樣性限制，D跟E只變多樣性限制不變品質指標，兩兩對照才能拆解出各自的貢獻）。
#: 一列＝一棵樹×一組。`_std`欄位只有C組（200次抽樣）有值，其餘組是單一次結果、無變異可談。
FOUR_GROUP_CONTROL = Schema(
    name="four_group_control",
    primary_key=["tree_key", "group"],
    columns=[
        Column("tree_key", "cat", allowed=("TW", "US", "XM")),
        Column("group", "cat", allowed=("A_hrp", "B_all", "C_random", "D_top_cagr", "E_top_calmar")),
        Column("n_members", "int", ge=1),
        Column("n_draws", "int", ge=1),
        Column("is_cagr", "float", nullable=True), Column("is_cagr_std", "float", nullable=True),
        Column("is_mdd", "float", nullable=True), Column("is_mdd_std", "float", nullable=True),
        Column("is_sharpe", "float", nullable=True), Column("is_sharpe_std", "float", nullable=True),
        Column("is_enb", "float", nullable=True), Column("is_enb_std", "float", nullable=True),
        Column("oos_cagr", "float", nullable=True), Column("oos_cagr_std", "float", nullable=True),
        Column("oos_mdd", "float", nullable=True), Column("oos_mdd_std", "float", nullable=True),
        Column("oos_sharpe", "float", nullable=True), Column("oos_sharpe_std", "float", nullable=True),
        Column("oos_enb", "float", nullable=True), Column("oos_enb_std", "float", nullable=True),
        # 集中度診斷（2026-08-30新增）：2026-08-30初版跑完後，D組在美股/跨市場的
        # OOS「勝利」被查出其實是選到單一群的集中賭注，不是真的分散——這兩欄
        # 把當時要手動debug才查得到的事實，變成每次跑都自動看得到的欄位。
        Column("n_clusters_covered", "int", ge=1, nullable=True),   # 這組成員橫跨幾個IS群
        Column("max_cluster_share", "float", ge=0, le=1, nullable=True),  # 最大單一群佔比
        Column("note", "str", nullable=True),
    ],
)


# ============================================================================
# 階段 4 · strategy_map 彙整（W-13）
# ============================================================================

#: regime_fit 標籤門檻（GateC「regime_fit 含『熊市抗跌』等」，未給精確公式，
#: 本階段的解讀，見 stage4_strategy_map.py docstring）：該regime標籤月份的
#: 平均報酬 >= 0 且樣本數達門檻，才貼對應標籤。
REGIME_FIT_MIN_MONTHS = 3

#: 四格信心分級門檻（GateC「信心門檻n：待資料看分布定」，本階段的解讀）
MACRO_CONFIDENCE_CUTS = {"高": 12, "中": 6}   # >=12月=高、>=6月=中、其餘=低

#: 階段4 產出：每策略×regime標籤 的表現統計（regime_fit 標籤的計算原料，
#: 也是給 Agent2 查「這個策略在熊市/危機到底表現如何」的完整數字）
REGIME_PERFORMANCE = Schema(
    name="regime_performance",
    primary_key=[PK, "label"],
    columns=[
        Column(PK, "str"),
        Column("market", "cat", allowed=MARKETS),
        Column("label", "cat", allowed=REGIME_LABELS),
        Column("n_months", "int", ge=0),
        Column("avg_ret", "float", nullable=True),     # n_months=0 時無值可算
        Column("win_ratio", "float", nullable=True, ge=0, le=1),
    ],
)

#: 階段4 產出：每策略×投資時鐘四格 的表現統計（v9「四格×該策略 平均報酬/
#: 勝率/樣本數/信心」，見研究部完整流程_v9.md 第五部分欄位表）
MACRO_PERFORMANCE = Schema(
    name="macro_performance",
    primary_key=[PK, "clock_cell"],
    columns=[
        Column(PK, "str"),
        Column("market", "cat", allowed=MARKETS),
        Column("clock_cell", "cat", allowed=CLOCK_CELLS),
        Column("n_months", "int", ge=0),
        Column("avg_ret", "float", nullable=True),
        Column("win_ratio", "float", nullable=True, ge=0, le=1),
        Column("confidence", "cat", allowed=CONFIDENCE, nullable=True),   # n_months=0 時留白
    ],
)

#: 階段4 主表：candidate_index(階段0) + strategy_scan原始指標(階段1) +
#: strategy_marks等級(階段1) + regime_fit(2a彙整) + macro摘要(2b彙整) +
#: cluster投影(階段3) + v1_beneficial(本階段新算)。
#: ⚠️ cluster_L1/L2/L3/co_fail_peers 是「該策略自己市場的常態樹」投影（單一
#: 便利欄位，供快速查詢），完整的六棵樹（含XM、crisis）仍以 cluster_assign/
#: co_fail_regimes 附屬表為準——不是每個策略都在 DD-03 窗內，故這三欄可能是
#: NaN（用 float 裝，不用 int，見 CANDIDATE_INDEX 的 F2_band 同類前例）。
STRATEGY_MAP = Schema(
    name="strategy_map",
    primary_key=[PK],
    columns=[
        # --- 身份與結構（階段0，= CANDIDATE_INDEX 的結構欄位）---
        Column(PK, "str"),
        Column("strategy", "str"),
        Column("market", "cat", allowed=MARKETS),
        Column("f_combo", "str"),
        Column("F1_factor", "str"),
        Column("F1_band", "int"),
        Column("F1_nbands", "int"),
        Column("F2_factor", "str", nullable=True),
        Column("F2_band", "float", nullable=True),
        Column("F2_nbands", "float", nullable=True),
        Column("F2_empty", "bool"),
        Column("C_id", "str", nullable=True),
        Column("C_source", "str", nullable=True),
        Column("C_rule", "str", nullable=True),
        Column("V", "cat", allowed=("v0", "v1")),
        Column("factor_type", "cat",
               allowed=("估值型", "體質型", "動能型", "規模型", "結構型", "混合型")),
        Column("factor_type_basis", "str"),
        # --- 整段績效（階段−1候選CSV / stats.parquet） ---
        Column("CAGR", "float"),
        Column("max_drawdown", "float", le=0.0),
        Column("win_ratio", "float", ge=0.0, le=1.0),
        Column("avg_holdings", "float", ge=0.0),
        Column("sharpe_ann", "float"),
        Column("daily_sharpe", "float"),
        Column("avg_drawdown", "float", le=0.0),
        Column("artifacts_dir", "str"),
        # --- 覆蓋期間（階段1 scan）---
        Column("hist_start", "str"),
        Column("hist_end", "str"),
        Column("n_months", "int", ge=1),
        Column("n_years", "int", ge=1),
        # --- C-3 報酬/風險形態（階段1）---
        Column("cagr_pct", "float", ge=0, le=100),
        Column("mdd_pct", "float", ge=0, le=100),
        Column("return_shape", "cat", allowed=("大起大落", "中等", "穩定爬升")),
        Column("risk_shape", "cat", allowed=("深回撤", "中等", "淺回撤")),
        Column("ann_vol", "float", ge=0),
        Column("ann_vol_monthly", "float", ge=0),
        Column("worst_year", "float"),
        Column("neg_year_count", "int", ge=0),
        Column("annual_ret_std", "float", ge=0),
        Column("max_consec_loss_months", "int", ge=0),
        Column("max_daily_ret", "float"),
        Column("min_daily_ret", "float"),
        # --- 關卡A：可信度（階段1）---
        Column("credibility_grade", "cat", allowed=CREDIBILITY_GRADES),
        Column("credibility_score_pct", "float", ge=0, le=100),
        Column("effective_n", "float", ge=0),
        Column("top1_share", "float", ge=0, le=1),
        Column("top20_cum_share", "float", ge=0, le=1),
        Column("top1_stock", "str"),
        Column("rotation_score", "float", ge=0),
        Column("rotation_n_years", "int", ge=0),
        # --- 關卡B：穩健度（階段1）---
        Column("stability_grade", "cat", allowed=STABILITY_GRADES, nullable=True),
        # --- C-4 可執行性（階段1）---
        Column("holdings_median", "float", ge=0),
        Column("holdings_p10", "float", ge=0),
        Column("empty_ratio", "float", ge=0, le=1),
        Column("n_eval_months", "int", ge=0),
        Column("eval_month_ratio", "float", ge=0, le=1),
        Column("n_holding_months", "int", ge=0),
        Column("turnover_ann", "float", ge=0),
        Column("entries_per_year", "float", ge=0),
        Column("n_trades", "int", ge=0),
        # --- C-5 規模依賴（階段1）---
        Column("size_tilt_pct", "float", ge=0, le=100),
        Column("smallcap_share", "float", ge=0, le=1),
        Column("size_n_obs", "int", ge=0),
        # --- 尾端寬鬆硬篩（階段1）---
        Column("is_usable", "bool"),
        Column("drop_reason", "str", nullable=True),
        # --- W-08 資料品質防線（階段1；見 STRATEGY_MARKS 同名欄位）---
        Column("data_glitch", "bool"),
        # --- H-01 alpha出賽關卡（階段1；見 STRATEGY_MARKS 同名欄位）---
        Column("passes_alpha_gate", "bool"),
        # --- regime_fit（本階段彙整 2a×階段1報酬；完整數字見 regime_performance 附屬表）---
        Column("regime_fit", "str", nullable=True),          # pipe-joined 標籤，見 stage4 docstring
        # --- macro_fit 摘要（本階段彙整 2b×階段1報酬；完整4格數字見 macro_performance 附屬表）---
        Column("macro_best_cell", "cat", allowed=CLOCK_CELLS, nullable=True),
        Column("macro_best_cell_avg_ret", "float", nullable=True),
        # --- HRP 投影（階段3；完整見 cluster_assign/co_fail_regimes 附屬表）---
        # cluster_L2 移除（H-04，2026-08-28），見 stage3_hrp.py 常數區註解
        Column("cluster_L1", "float", nullable=True),
        Column("cluster_L3", "float", nullable=True),
        Column("co_fail_peers", "str", nullable=True),
        # --- v1_beneficial（本階段新算：同一 market×f_combo×C_id 下 v1 CAGR 是否優於 v0）---
        Column("v1_beneficial", "bool", nullable=True),
    ],
)


ALL_SCHEMAS = {s.name: s for s in (CANDIDATE_INDEX, RETURNS_MONTHLY, RETURNS_META,
                                   MACRO_RAW, MACRO_HISTORY, MACRO_CLOCK_COMPARISON,
                                   REGIME_TABLE, REGIME_CONSISTENCY,
                                   CLUSTER_ASSIGN, CLUSTER_META, CO_FAIL_REGIMES, STRATEGY_MARKS,
                                   REGIME_PERFORMANCE, MACRO_PERFORMANCE, STRATEGY_MAP,
                                   CLUSTER_STORY, CLUSTER_IDENTITY, CLUSTER_MACRO_INTERFACE,
                                   CLUSTER_MACRO_CONDITIONAL, DECISION_REPEATABILITY,
                                   CLUSTER_ANNUAL_RETURNS, CLUSTER_QUARTERLY_RETURNS,
                                   CLUSTER_PROFILE_QUANT, EFFECTIVE_BETS, CLUSTER_REPRESENTATIVES,
                                   CLUSTER_ASSIGN_ISOOS, CLUSTER_META_ISOOS, ISOOS_CORR_COMPARISON,
                                   FOUR_GROUP_CONTROL, COMPLEMENTARITY_GRANULARITY,
                                   FREE_LUNCH_SHORTLIST, WALKFORWARD_MATRIX, K_STABILITY,
                                   WALKFORWARD_SIGNIFICANCE, WINDOW_ROBUSTNESS,
                                   L3_ISOOS, TURNOVER_COST)}
