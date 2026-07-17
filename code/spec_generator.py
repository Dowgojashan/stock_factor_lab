# -*- coding: utf-8 -*-
"""
A3｜分位數 spec 產生器（方案 A：橫斷面每期排名 q_band）。

把 P1（体质 F 構面）的手寫固定門檻，換成 N=5 的橫斷面分位 band（q_band），
讓「同一個桶」在台美代表同一相對位置 → 因子跨市場可比、不新增前瞻偏誤。

設計原則「預設共用、結構可分」：
- 一份 config → 產出 per-market 檔 spec_TW.json / spec_US.json（不用單一實體檔）。
- 預設兩市場指向同一 config（老師方向1：同 recipe 各市場跑再比）。
- 要走方向2（各市場設不同條件）時，只需讓 CONFIG_BY_MARKET["US"] 指向不同 dict。

spec 內容為純結構（k, N），不需讀 DB → 產生器秒出；分位數在 runtime 由 q_band 計算。
另附 --describe（需 DB）：印各市場各因子各 band 的「實現絕對門檻」對照表（跨市場可比證據）。

用法（cwd 任意；建議在 code/）：
  python spec_generator.py                 # 產出 spec_TW.json / spec_US.json + round-trip 驗證
  python spec_generator.py --describe       # 另印台美門檻對照表（需 MySQL 開著）
"""

import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))     # .../code
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)                              # get_data / dataframe 在 ROOT
sys.path.insert(0, HERE)                              # condition_factory 在 code/
os.chdir(HERE)                                        # 讓 ../config.ini 與 P3 來源檔可解析
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from condition_factory import build_conditions

# ==================== 設定 ====================
N = 5  # 分桶數（→ 每因子 N 個 q_band 條件）

# P1 体质因子 → 讀取層欄位。
# ⚠️ 只用「台美兩市場都有資料」的共同因子，A3 才真的跨市場可比。
#    DEBTRATIO / REVENUE 是美股 collector 新加的、台股沒有 → 不放進共用 recipe（決策：只用共同因子）。
P1_FACTORS = {
    "ROE":   "report:roe",
    "EPS":   "report:eps",
    "FCF_P": "report:fcf_p",
}

# 兩市場共同的財報欄位（用來過濾 P3：台股缺 REVENUE → 自動去掉 REVENUE_GROWTH 的 C4/C5）。
COMMON_REPORT_FIELDS = {"report:roe", "report:eps", "report:fcf_p"}

# P3（動態 C）沿用既有定義：rise_q / is_highest_q / yoy_gt / ytd_avg 皆為相對/排名語意，
# 天生跨市場可比、不需分位數化。來源＝既有 spec 的 P3，並濾成只留共同因子。
P3_SOURCE = "fcv_experiment_spec.json"

# 一份 config → 兩市場。預設 TW/US 指向同一份（方向1）。
# 方向2：把 "US" 換成另一個 {"factors":..., "N":...} 即可發散（注意會犧牲跨市場可比，見計畫 v6）。
_BASE = {"factors": P1_FACTORS, "N": N}
CONFIG_BY_MARKET = {"TW": _BASE, "US": _BASE}


# ==================== 產生器 ====================
def load_p3_template(path=P3_SOURCE, allowed_fields=COMMON_REPORT_FIELDS):
    """從既有 spec 讀取 P3（動態 C）模板，並只保留共同因子的條件群
    （台股缺 REVENUE → 濾掉 REVENUE_GROWTH 的 C4/C5）。"""
    with open(path, encoding="utf-8") as f:
        p3 = json.load(f).get("P3", {})
    return {k: v for k, v in p3.items() if v.get("field") in allowed_fields}


def build_p1(factors, n):
    """對每個体质因子輸出 N 個 q_band 條件（k=0..N-1）。"""
    return {
        name: {
            "field": field,
            "conditions": [{"type": "q_band", "args": [k, n]} for k in range(n)],
        }
        for name, field in factors.items()
    }


def make_spec(market, cfg, p3):
    return {
        "_meta": {
            "generated_by": "code/spec_generator.py",
            "do_not_edit": True,          # 重跑產生器即可覆蓋；勿手改
            "market": market,
            "N": cfg["N"],
            "P1_kind": "q_band (cross-sectional quantile, per-period rank)",
        },
        "P1": build_p1(cfg["factors"], cfg["N"]),
        "P3": p3,
    }


def roundtrip_validate(spec):
    """回讀驗證：build_conditions 讀回每條都建得出函式；且條件名不重複。"""
    p1 = build_conditions(spec["P1"])
    p3 = build_conditions(spec["P3"])
    names = [c["name"] for c in p1 + p3]
    dup = {x for x in names if names.count(x) > 1}
    assert not dup, f"條件名重複：{dup}"
    return p1, p3


def generate(out_dir="."):
    p3 = load_p3_template()
    written = []
    for market, cfg in CONFIG_BY_MARKET.items():
        spec = make_spec(market, cfg, p3)
        p1, p3c = roundtrip_validate(spec)
        out = os.path.join(out_dir, f"spec_{market}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
        print(f"[OK] {out}｜P1={len(p1)} 條（{len(cfg['factors'])} 因子×{cfg['N']} band）"
              f"、P3={len(p3c)} 條｜round-trip 通過")
        written.append(out)
    return written


# ==================== 選用：跨市場門檻對照表（需 DB） ====================
def describe_thresholds(markets=("TW", "US")):
    """對每市場每因子，算各換股點橫斷面 k/N 分位的『實現絕對門檻』再對時間平均。
    純描述、不進回測；證明『同一個桶在台美對應不同絕對值』＝跨市場可比。"""
    import warnings
    import numpy as np
    from get_data import Data

    qs = [k / N for k in range(1, N)]  # 內部斷點 0.2/0.4/0.6/0.8
    for market in markets:
        cfg = CONFIG_BY_MARKET[market]
        print(f"\n===== [{market}] 各因子 band 實現絕對門檻（跨時間平均）=====")
        d = Data(market=market)
        for name, field in cfg["factors"].items():
            s = d.get(field)
            vals = np.asarray(s.values, dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)  # 全 NaN 列
                thr = {f"p{int(q*100)}": np.nanmean(np.nanpercentile(vals, q * 100, axis=1))
                       for q in qs}
            cells = " | ".join(f"{kk}={vv:.3g}" for kk, vv in thr.items())
            print(f"  {name:10s}: {cells}")


def main():
    ap = argparse.ArgumentParser(description="A3 分位數 spec 產生器")
    ap.add_argument("--describe", action="store_true",
                    help="另印台美門檻對照表（需 MySQL 開著，會載入資料、較慢）")
    args = ap.parse_args()

    generate()
    if args.describe:
        describe_thresholds()


if __name__ == "__main__":
    main()
