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
    ],
)


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
        Column("cluster_L2", "int"),
        Column("cluster_L3", "int"),
    ],
)

CLUSTER_META = Schema(
    name="cluster_meta",
    primary_key=["tree_id", "level", "cluster_id"],
    columns=[
        Column("tree_id", "cat", allowed=TREE_IDS),
        Column("level", "cat", allowed=("L1", "L2", "L3")),
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
        Column("level", "cat", allowed=("L1", "L2", "L3")),
        Column("cluster_normal", "int"),                     # 常態樹的群 id
        Column("n_members", "int", ge=1),
        Column("crisis_dest_cluster", "int"),                # 危機期多數成員的去向（危機樹群id）
        Column("crisis_dest_share", "float", ge=0, le=1),     # 去向集中度
        Column("co_fail_peers", "str"),                       # 同去向的其他常態群，"|" 分隔（可能為空字串）
        Column("n_co_fail_peers", "int", ge=0),
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
        # --- regime_fit（本階段彙整 2a×階段1報酬；完整數字見 regime_performance 附屬表）---
        Column("regime_fit", "str", nullable=True),          # pipe-joined 標籤，見 stage4 docstring
        # --- macro_fit 摘要（本階段彙整 2b×階段1報酬；完整4格數字見 macro_performance 附屬表）---
        Column("macro_best_cell", "cat", allowed=CLOCK_CELLS, nullable=True),
        Column("macro_best_cell_avg_ret", "float", nullable=True),
        # --- HRP 投影（階段3；完整見 cluster_assign/co_fail_regimes 附屬表）---
        Column("cluster_L1", "float", nullable=True),
        Column("cluster_L2", "float", nullable=True),
        Column("cluster_L3", "float", nullable=True),
        Column("co_fail_peers", "str", nullable=True),
        # --- v1_beneficial（本階段新算：同一 market×f_combo×C_id 下 v1 CAGR 是否優於 v0）---
        Column("v1_beneficial", "bool", nullable=True),
    ],
)


ALL_SCHEMAS = {s.name: s for s in (CANDIDATE_INDEX, RETURNS_MONTHLY, RETURNS_META,
                                   MACRO_RAW, MACRO_HISTORY, REGIME_TABLE, REGIME_CONSISTENCY,
                                   CLUSTER_ASSIGN, CLUSTER_META, CO_FAIL_REGIMES, STRATEGY_MARKS,
                                   REGIME_PERFORMANCE, MACRO_PERFORMANCE, STRATEGY_MAP)}
