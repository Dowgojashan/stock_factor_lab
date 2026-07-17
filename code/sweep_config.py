# -*- coding: utf-8 -*-
"""
Sweep 設定層：實驗矩陣 → config → spec（參數化）。

A4 定案（見 A4_策略庫Sweep產生器_開發計畫.md §2）：
  市場 US+TW｜因子=台美共同 3（ROE/EPS/FCF_P）｜N={5,10}｜C=20｜V=2｜頻率={M,Q}｜窗 2000–2026

註：本檔**不動 A3 的 `spec_generator.py`**（那支產出的 spec_US.json 是已完成那批 2310 策略的記錄，
    保留以供追溯）。Sweep 走這裡的參數化產生器。
"""
from pathlib import Path

# ==================== 因子（台美共同 3 個）====================
# ⚠️ 台股實際因子數待 DB 確認（A4 §9）；目前依定案用共同 3 個。
COMMON_FACTORS = {
    "ROE":   "report:roe",
    "EPS":   "report:eps",
    "FCF_P": "report:fcf_p",
}
FACTORS_BY_MARKET = {"US": COMMON_FACTORS, "TW": COMMON_FACTORS}

# ==================== N 分位（粗細兩種）====================
N_LIST = [5, 10]

# ==================== C 動態條件型別（A4 §5）====================
# (tag, condition_factory type, args)；全部用現成型別，不需改 condition_factory。
C_TYPES = [
    ("riseq1",  "rise_q",                                [1]),   # 較上季升
    ("riseq2",  "rise_q",                                [2]),   # 較前2季升
    ("qmax4",   "is_highest_q",                          [4]),   # 近4季最高
    ("qmax8",   "is_highest_q",                          [8]),   # 近8季最高
    ("yoy",     "yoy_gt",                                [4]),   # 年增 YoY
    ("ytdsame", "ytd_avg_gt_prev_year_same_period_avg",  []),    # YTD均 > 去年同期均
    ("ytdfull", "ytd_avg_gt_prev_year_avg",              []),    # YTD均 > 去年全年均
]
# FCF_P 不出 ytdfull → 3×7 − 1 = 20 個 C
C_SKIP = {("FCF_P", "ytdfull")}

# ==================== 換股頻率 ====================
REBALANCES = ["M", "Q"]

# ==================== 時間窗 ====================
MARKET_START = {"US": "2000-01-01", "TW": "2000-01-01"}


# ==================== 產生器 ====================
def build_p1(factors, n_list):
    """每因子對每個 N 出 N 個 q_band（k=0..N-1）。N={5,10} → 每因子 15 條、共 45 條。"""
    return {
        name: {
            "field": field,
            "conditions": [{"type": "q_band", "args": [k, n]}
                           for n in n_list for k in range(n)],
        }
        for name, field in factors.items()
    }


def build_p3(factors, c_types=C_TYPES, skip=C_SKIP):
    """每因子一個群組，條件用 C{idx}_ 前綴確保名稱全域唯一。預設共 20 個 C。"""
    p3, idx = {}, 0
    for fname, field in factors.items():
        conds = []
        for tag, ctype, args in c_types:
            if (fname, tag) in skip:
                continue
            idx += 1
            conds.append({"prefix": f"C{idx}_", "type": ctype, "args": args})
        if conds:
            p3[f"{fname}_DYN"] = {"field": field, "conditions": conds}
    return p3


def count_c(factors, c_types=C_TYPES, skip=C_SKIP):
    return sum(1 for f in factors for t, _, _ in c_types if (f, t) not in skip)


def spec_id(market, factors, n_list, c_types=C_TYPES, skip=C_SKIP):
    """決定式 spec 命名（不含頻率）：US_f3_N5-10_c20"""
    ns = "-".join(str(n) for n in n_list)
    return f"{market}_f{len(factors)}_N{ns}_c{count_c(factors, c_types, skip)}"


def make_spec(market, factors=None, n_list=None):
    factors = factors or FACTORS_BY_MARKET[market]
    n_list = n_list or N_LIST
    return {
        "_meta": {
            "generated_by": "code/sweep_config.py",
            "do_not_edit": True,
            "market": market,
            "factors": list(factors),
            "N_list": n_list,
            "n_C": count_c(factors),
            "P1_kind": "q_band (cross-sectional quantile, per-period rank)",
            "spec_id": spec_id(market, factors, n_list),
        },
        "P1": build_p1(factors, n_list),
        "P3": build_p3(factors),
    }


def job_list(markets=("US", "TW"), rebalances=REBALANCES):
    """展開實驗矩陣 → 工作清單。每個 job = 一份 spec × 一個換股頻率。

    回傳 [{label, market, rebalance, spec}, ...]；label 是決定式的（供 _DONE 續傳比對）。
    同市場的 job 共用一次 Data 載入（見 sweep_driver）。
    """
    jobs = []
    for market in markets:
        spec = make_spec(market)
        sid = spec["_meta"]["spec_id"]
        for reb in rebalances:
            jobs.append({
                "label": f"{sid}_{reb}",
                "market": market,
                "rebalance": reb,
                "start": MARKET_START[market],
                "spec": spec,
            })
    return jobs


if __name__ == "__main__":
    import sys, json
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from condition_factory import build_conditions

    jobs = job_list()
    print(f"工作清單：{len(jobs)} 個 job\n")
    for j in jobs:
        P1 = build_conditions(j["spec"]["P1"])
        P3 = build_conditions(j["spec"]["P3"])
        names = [c["name"] for c in P1 + P3]
        dup = {x for x in names if names.count(x) > 1}
        assert not dup, f"條件名重複：{dup}"
        # 去重後策略數 = (單因子 + 無序異因子對) × (1+C) × 2
        nf = len(j["spec"]["_meta"]["factors"])
        per = len(P1) // nf
        pairs = (nf * (nf - 1) // 2) * per * per
        n_strat = (len(P1) + pairs) * (1 + len(P3)) * 2
        print(f"  {j['label']:22s} P1={len(P1):3d} P3={len(P3):3d} "
              f"→ 去重後策略 {n_strat:,}")
    print("\n[OK] round-trip 通過（每條件都建得出函式、名稱無重複）")
