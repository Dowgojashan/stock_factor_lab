# -*- coding: utf-8 -*-
"""研究部管線的自檢測試（不依賴 pytest，環境沒裝）。

用法：
    cd code
    python -m research.tests

測試分三類（SDD 第七部分）：
  1. 契約層：schema 驗證器本身要抓得到違規（含最關鍵的主鍵重複）
  2. 階段驗收：每階段的硬驗收條件
  3. 冪等性：同輸入重跑，產出位元相同
"""
from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd

from . import contracts as C
from . import freeze, paths

_RESULTS: list[tuple[str, bool, str]] = []


def test(fn):
    """把函式登記為測試。失敗只記錄不中斷，最後一次report。"""
    def run():
        try:
            fn()
            _RESULTS.append((fn.__name__, True, fn.__doc__ or ""))
        except Exception as e:
            _RESULTS.append((fn.__name__, False,
                             f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"))
    run.__name__ = fn.__name__
    return run


def expect_raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return
    raise AssertionError(f"預期 raise {exc.__name__} 但沒有")


# ---------------------------------------------------------------- 契約層

@test
def t_pk_duplicate_is_caught():
    """主鍵重複必須被攔下（防台美字串碰撞靜默錯 join）"""
    df = pd.DataFrame({C.PK: ["TW::a", "TW::a"], "month": [1, 1], "ret": [0.1, 0.2]})
    df["month"] = pd.PeriodIndex(["2020-01", "2020-01"], freq="M")
    expect_raises(C.ContractError, C.validate, df, C.RETURNS_MONTHLY)


@test
def t_missing_column_is_caught():
    """缺欄位必須被攔下"""
    expect_raises(C.ContractError, C.validate,
                  pd.DataFrame({C.PK: ["TW::a"]}), C.RETURNS_META)


@test
def t_illegal_category_is_caught():
    """類別欄出現非法值必須被攔下"""
    df = pd.DataFrame({C.PK: ["XX::a"], "market": ["XX"],
                       "hist_start": ["2007-01"], "hist_end": ["2025-12"], "n_months": [228]})
    expect_raises(C.ContractError, C.validate, df, C.RETURNS_META)


@test
def t_reconcile_detects_drift():
    """對帳斷言必須抓得到兩來源不同步"""
    a = pd.Series([1.0, 2.0]); b = pd.Series([1.0, 2.001])
    expect_raises(C.ContractError, C.assert_reconciles, a, b, name="test")
    C.assert_reconciles(a, pd.Series([1.0, 2.0 + 1e-12]), name="test")   # 容差內應通過


@test
def t_uid_rule():
    """主鍵組成規則"""
    assert C.make_uid("TW", "abc") == "TW::abc"
    s = C.make_uid(pd.Series(["TW", "US"]), pd.Series(["a", "a"]))
    assert list(s) == ["TW::a", "US::a"], "台美同名策略必須產生不同主鍵"


@test
def t_hrp_windows_declared():
    """HRP 三棵樹的窗必須都已宣告（DD-03 定案）"""
    for tree in ("TW", "US", "XM"):
        start, end = C.HRP_WINDOWS[tree]
        assert start < end and end == "2025-12", f"{tree} 窗不合法"
    assert C.HRP_WINDOWS["US"][0] < C.HRP_WINDOWS["TW"][0], \
        "美股窗應比台股早（per-tree 窗的重點：美股不必陪葬）"


# --------------------------------------------------- 階段2b 總經滯後（W-01）

@test
def t_macro_lag_monthly():
    """月頻指標：月底只能用到上個月的資料"""
    from .macro_spec import SPEC, available_period
    import pandas as pd
    cpi = SPEC["US"]["inflation"]
    assert available_period(cpi, "2026-03") == pd.Period("2026-02", "M"), \
        "3 月底時 3 月的 CPI 還沒發布（要 4 月第二週），只能用 2 月"


@test
def t_macro_lag_quarterly_boundary():
    """季頻指標的邊界：GDP 在季末後約 1 個月發布，保守處理不搶當月"""
    from .macro_spec import SPEC, available_period
    import pandas as pd
    gdp = SPEC["US"]["growth"]
    # Q1(3/31 結束) 約 4/30 發布 → 4 月底判定為「還不可用」
    assert available_period(gdp, "2026-04") == pd.Period("2025Q4", "Q")
    # 5 月底才可用
    assert available_period(gdp, "2026-05") == pd.Period("2026Q1", "Q")


@test
def t_macro_lag_no_lookahead():
    """鐵則：任何指標回傳的資料期間都不得晚於決策月"""
    from .macro_spec import SPEC, available_period
    import pandas as pd
    for mkt, inds in SPEC.items():
        for key, ind in inds.items():
            for m in ("2000-01", "2013-07", "2026-03", "2025-12"):
                got = available_period(ind, m)
                end = got.asfreq("M", how="end") if got.freqstr.startswith("Q") else got
                assert end <= pd.Period(m, "M"), \
                    f"{mkt}.{key} 在 {m} 回傳 {got}，晚於決策月＝前視偏誤"


@test
def t_macro_spec_complete():
    """四個概念軸台美都要有指標，且每個都有官方來源"""
    from .macro_spec import SPEC, AXES
    for mkt, inds in SPEC.items():
        axes = {i.axis for i in inds.values()}
        missing = set(AXES) - axes
        assert not missing, f"{mkt} 缺少概念軸: {missing}"
        for key, i in inds.items():
            assert i.source_url.startswith("http"), f"{mkt}.{key} 缺官方來源連結"


# ---------------------------------------------------------------- HRP（W-03）

@test
def t_hrp_psd_detection():
    """check_psd 要能分辨合法相關矩陣與非法（非PSD）矩陣"""
    from . import hrp
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 30))          # 200月 × 30策略，真實資料生成的相關矩陣必為PSD
    corr = np.corrcoef(X.T)
    ok, min_eig = hrp.check_psd(corr)
    assert ok and min_eig > -1e-6, f"真實資料生成的相關矩陣應為PSD，min_eig={min_eig}"

    bad = corr.copy()
    bad[0, 1] = bad[1, 0] = 0.999           # 手動破壞：塞一個不一致的相關值
    bad[0, 2] = bad[2, 0] = 0.999
    bad[1, 2] = bad[2, 0] = -0.999          # 三顆兩兩幾乎完全正相關卻有一對幾乎完全負相關→矛盾
    ok2, min_eig2 = hrp.check_psd(bad)
    assert not ok2, "手動構造的矛盾相關矩陣應被判定為非PSD"


@test
def t_hrp_distance_is_metric():
    """corr_to_distance 產生的距離矩陣需滿足三角不等式（在合法PSD相關矩陣上）"""
    from . import hrp
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 50))
    corr = np.corrcoef(X.T)
    dist = hrp.corr_to_distance(corr)
    ok, violations, max_excess = hrp.check_triangle_inequality(dist, n_samples=3000, seed=1)
    assert ok, f"合法相關矩陣算出的距離違反三角不等式 {violations} 次，最大超出 {max_excess:.2e}"
    assert np.allclose(np.diag(dist), 0.0), "距離矩陣對角線必須是 0"
    assert np.allclose(dist, dist.T), "距離矩陣必須對稱"


@test
def t_hrp_weights_valid():
    """recursive_bisection_weights 權重需為正、加總為 1"""
    from . import hrp
    rng = np.random.default_rng(2)
    n = 40
    X = rng.normal(size=(150, n))
    cov = np.cov(X.T)
    order = list(range(n))
    rng.shuffle(order)
    w = hrp.recursive_bisection_weights(cov, order)
    assert len(w) == n
    assert (w > 0).all(), "HRP 權重不應出現負值或零"
    assert abs(w.sum() - 1.0) < 1e-9, f"權重加總應為 1，實際 {w.sum()}"


@test
def t_hrp_build_tree_end_to_end():
    """build_tree 端到端：合成資料跑完整鏈，權重與PSD檢查皆正常"""
    from . import hrp
    rng = np.random.default_rng(3)
    n, t = 60, 120
    # 造 3 個「真實群」：群內高相關、群間低相關，驗證分群結果非隨機
    base = rng.normal(size=(3, t))
    returns = np.vstack([base[i % 3] + rng.normal(scale=0.3, size=t) for i in range(n)])
    res = hrp.build_tree(returns, method="ward")
    assert res.psd_ok, f"合成資料的相關矩陣應為 PSD，min_eig={res.min_eig}"
    assert abs(res.weights.sum() - 1.0) < 1e-9
    assert len(res.leaf_order) == n and set(res.leaf_order) == set(range(n))
    assert res.cophenetic > 0.3, f"cophenetic 相關過低（{res.cophenetic:.3f}），linkage 可能沒抓到群結構"

    labels = hrp.cut_clusters(res.link, n_clusters=3)
    true_group = np.array([i % 3 for i in range(n)])
    ari = hrp.adjusted_rand_index(pd.Series(labels), pd.Series(true_group))
    assert ari > 0.5, f"3群合成資料切3群，ARI 應明顯 > 0（實際 {ari:.3f}），linkage 未抓到真實群結構"


@test
def t_hrp_build_tree_rejects_nan():
    """build_tree 對含 NaN 的輸入必須拒絕，不能靜默用 pairwise-complete"""
    from . import hrp
    rng = np.random.default_rng(4)
    returns = rng.normal(size=(20, 50))
    returns[3, 10] = np.nan
    expect_raises(ValueError, hrp.build_tree, returns)


@test
def t_hrp_ari_sanity():
    """ARI：相同分群=1，完全打散的隨機分群應接近 0"""
    from . import hrp
    a = pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert abs(hrp.adjusted_rand_index(a, a) - 1.0) < 1e-9, "自己對自己 ARI 應為 1"

    rng = np.random.default_rng(5)
    n = 3000
    a2 = pd.Series(rng.integers(0, 10, size=n))
    b2 = pd.Series(rng.integers(0, 10, size=n))
    ari = hrp.adjusted_rand_index(a2, b2)
    assert abs(ari) < 0.05, f"兩個獨立隨機分群的 ARI 應接近 0，實際 {ari:.3f}"


# ------------------------------------------------------------ 階段0 驗收

def _load_stage0() -> pd.DataFrame:
    p = paths.STAGE0 / "candidate_index.parquet"
    if not p.exists():
        raise AssertionError("階段0 尚未執行，請先 python -m research.cli stage0")
    return pd.read_parquet(p)


@test
def t_stage0_contract():
    """階段0 產物符合契約（列數/主鍵/F2_empty/型別值域）"""
    C.validate(_load_stage0(), C.CANDIDATE_INDEX, strict_columns=True)


@test
def t_stage0_collision_is_real():
    """落差5 實證：裸 strategy 不唯一、複合主鍵唯一

    ⚠️ 不釘死碰撞的確切數字（曾是 1381，openSec 重跑後變 1585）：候選池每次
    重跑組成都會變，釘一個快照數字本身就是 B-01 教訓的同一類錯誤。這裡驗證
    的是**現象存在**（台美用同一套命名規則，碰撞必然 > 0），不是某次的精確值。
    """
    df = _load_stage0()
    assert df[C.PK].is_unique, "複合主鍵必須唯一"
    assert not df["strategy"].is_unique, "裸 strategy 應該不唯一（台美碰撞）"
    collisions = len(df) - df["strategy"].nunique()
    assert collisions > 0, "應該存在台美策略字串碰撞，實測卻是 0——命名規則是否變了？"


@test
def t_stage0_artifacts_exist():
    """落差2 實證：artifacts_dir 全部存在（抽 300 筆）"""
    from pathlib import Path
    df = _load_stage0().sample(300, random_state=0)
    missing = [d for d in df["artifacts_dir"] if not Path(d).is_dir()]
    assert not missing, f"{len(missing)} 個產物目錄不存在，例如 {missing[:2]}"


@test
def t_stage0_v_routes_to_right_job():
    """落差2 規則正確：v0 的產物在 L3、v1 在 L4"""
    df = _load_stage0()
    for v, tag in (("v0", "_L3_"), ("v1", "_L4_")):
        sub = df[df.V == v]["artifacts_dir"]
        assert sub.str.contains(tag).all(), f"{v} 應全部指向 {tag} job 目錄"


@test
def t_stage0_f_combo_count():
    """獨立 F 組合數 = 407（快篩多樣性假象的根源、HRP L3 群數的錨點）"""
    df = _load_stage0()
    for m, n in C.EXPECTED_F_COMBOS.items():
        got = df[df.market == m]["f_combo"].nunique()
        assert got == n, f"{m} 獨立 F 組合 {got} != 預期 {n}"


@test
def t_stage0_beats_benchmark():
    """階段 −1 已 gate：候選池不應有低於自建宇宙基準者（v9 說 0 個）"""
    df = _load_stage0()
    for m, bm in C.BENCHMARK_CAGR.items():
        below = int((df[df.market == m]["CAGR"] < bm).sum())
        assert below == 0, f"{m} 有 {below} 個策略 CAGR 低於基準 {bm:.2%}"


@test
def t_stage0_freeze_intact():
    """凍結產物未被改動"""
    freeze.verify_inputs(paths.STAGE0)


@test
def t_stage0_idempotent():
    """冪等性：重跑階段0 應產出位元相同的檔案"""
    p = paths.STAGE0 / "candidate_index.parquet"
    before = freeze.sha256_file(p)
    from . import stage0_index
    stage0_index.build(log=lambda *a, **k: None)
    assert freeze.sha256_file(p) == before, "重跑產出不同，管線非確定性"


# ---------------------------------------------------------------- runner

def main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("t_") and callable(v)]
    print(f"執行 {len(tests)} 項測試 …\n")
    for t in tests:
        t()
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    for name, ok, msg in _RESULTS:
        mark = "✓" if ok else "✗"
        head = msg.splitlines()[0] if msg else ""
        print(f"  {mark} {name:<32} {head}")
        if not ok:
            print("      " + "\n      ".join(msg.splitlines()[1:6]))
    print(f"\n{passed}/{len(_RESULTS)} 通過")
    return 0 if passed == len(_RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
