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


# --------------------------------------------------------- 階段2a regime

@test
def t_regime_zigzag_basic_shape():
    """合成一段明確的漲跌走勢，驗證 zigzag 抓到正確數量與方向的轉折"""
    from . import stage2a_regime as r2a
    dates = pd.date_range("2020-01-01", periods=400, freq="D")
    # 100 -> 50（跌50%）-> 100（漲100%）-> 60（跌40%），每段線性、明確超過門檻
    seg = np.concatenate([
        np.linspace(100, 50, 100), np.linspace(50, 100, 100),
        np.linspace(100, 60, 100), np.full(100, 60.0),
    ])
    price = pd.Series(seg, index=dates)
    pivots = r2a.zigzag_pivots(price, bear_thresh=0.15, bull_thresh=0.15)
    kinds = [p["kind"] for p in pivots]
    assert kinds[0] == "start"
    assert "peak" in kinds and "trough" in kinds
    # 應該偵測到至少 3 個轉折（跌段底、漲段頂、再跌段底附近）
    assert len(pivots) >= 4, f"轉折點數量過少：{len(pivots)}"


@test
def t_regime_classify_crisis_vs_bear():
    """跌幅超過 crisis_thresh 才判危機，介於 bear_thresh~crisis_thresh 之間判熊"""
    from . import stage2a_regime as r2a
    pivots = [
        {"date": pd.Timestamp("2020-01-01"), "price": 100.0, "kind": "start"},
        {"date": pd.Timestamp("2020-06-01"), "price": 82.0, "kind": "trough"},   # -18%，快速=熊
        {"date": pd.Timestamp("2020-12-01"), "price": 100.0, "kind": "peak"},    # +22%，牛
        {"date": pd.Timestamp("2021-03-01"), "price": 65.0, "kind": "trough"},   # -35%，快速=危機
    ]
    segs = r2a.classify_segments(pivots, r2a.DEFAULT_PARAMS)
    assert segs.iloc[0]["label"] == "熊", f"-18% 快速下跌應判熊，實際 {segs.iloc[0]['label']}"
    assert segs.iloc[1]["label"] == "牛"
    assert segs.iloc[2]["label"] == "危機", f"-35% 快速下跌應判危機，實際 {segs.iloc[2]['label']}"


@test
def t_regime_slow_grind_becomes_consolidation():
    """同樣的跌幅，若耗時極長（年化速度低於門檻），應被重分類為盤整而非熊"""
    from . import stage2a_regime as r2a
    pivots = [
        {"date": pd.Timestamp("2000-01-01"), "price": 100.0, "kind": "start"},
        # -18%，耗時 10 年 → 年化速度 ≈1.8%，遠低於 consolidation_speed(15%) → 應變盤整
        {"date": pd.Timestamp("2010-01-01"), "price": 82.0, "kind": "trough"},
    ]
    segs = r2a.classify_segments(pivots, r2a.DEFAULT_PARAMS)
    assert segs.iloc[0]["label"] == "盤整", \
        f"耗時10年的緩慢-18%應判盤整（非熊），實際 {segs.iloc[0]['label']}"


@test
def t_regime_crisis_not_overridden_by_slow_grind():
    """危機段不受盤整覆寫規則影響，即使耗時很長也維持危機標籤"""
    from . import stage2a_regime as r2a
    pivots = [
        {"date": pd.Timestamp("2000-01-01"), "price": 100.0, "kind": "start"},
        # -35%，耗時 5 年，速度依然偏低，但危機不該被覆寫成盤整
        {"date": pd.Timestamp("2005-01-01"), "price": 65.0, "kind": "trough"},
    ]
    segs = r2a.classify_segments(pivots, r2a.DEFAULT_PARAMS)
    assert segs.iloc[0]["label"] == "危機", \
        f"危機段不該被盤整規則覆寫，實際 {segs.iloc[0]['label']}"


@test
def t_regime_table_no_gaps_no_overlap():
    """真實資料：regime_table 的區間必須連續、無重疊、無空隙（cut_clusters 前提）"""
    p = paths.STAGE2 / "regime" / "regime_table_TW.parquet"
    if not p.exists():
        raise AssertionError("尚未執行 stage2a_regime，請先 python -m research.stage2a_regime")
    t = pd.read_parquet(p).sort_values("start").reset_index(drop=True)
    gap_or_overlap = (t["start"].iloc[1:].to_numpy() != t["end"].iloc[:-1].to_numpy())
    assert not gap_or_overlap.any(), \
        f"{int(gap_or_overlap.sum())} 處銜接不連續（有空隙或重疊）"


@test
def t_regime_known_events_all_hit():
    """五個已知事件（依市場適用性）在真實資料上必須全數命中"""
    from . import stage2a_regime as r2a
    for m in ("TW", "US"):
        p = paths.STAGE2 / "regime" / f"regime_table_{m}.parquet"
        if not p.exists():
            raise AssertionError(f"尚未執行 stage2a_regime（缺 {p}）")
        t = pd.read_parquet(p)
        ver = r2a.verify_known_events(t, m, log=lambda *a, **k: None)
        assert ver.matched.all(), \
            f"[{m}] 已知事件未全數命中：\n{ver[~ver.matched].to_string(index=False)}"


# ------------------------------------------------------------ 階段3 co_fail_regimes

@test
def t_co_fail_regimes_basic_grouping():
    """常態兩個不同群，若危機期成員全被分進同一個危機群，應互為 co_fail peers"""
    from . import stage3_hrp as s3
    normal = pd.DataFrame({
        C.PK: ["TW::a", "TW::b", "TW::c", "TW::d"],
        "tree_id": "TW_normal", "cluster_L1": [1, 1, 2, 2],
    })
    crisis = pd.DataFrame({
        C.PK: ["TW::a", "TW::b", "TW::c", "TW::d"],
        "tree_id": "TW_crisis", "cluster_L1": [9, 9, 9, 9],   # 危機期全塌進同一群
    })
    combined = pd.concat([normal, crisis], ignore_index=True)
    out = s3.build_co_fail_regimes(combined, combined, "TW", "L1")
    row1 = out[out.cluster_normal == 1].iloc[0]
    row2 = out[out.cluster_normal == 2].iloc[0]
    assert row1["crisis_dest_cluster"] == 9 and row2["crisis_dest_cluster"] == 9
    assert row1["co_fail_peers"] == "2"
    assert row2["co_fail_peers"] == "1"
    assert row1["n_co_fail_peers"] == 1 and row2["n_co_fail_peers"] == 1


@test
def t_co_fail_regimes_diverging_dest_means_no_peers():
    """危機期去向不同的常態群，不該被標成 co_fail（避免濫發假的共跌訊號）"""
    from . import stage3_hrp as s3
    normal = pd.DataFrame({C.PK: ["TW::a", "TW::b"], "tree_id": "TW_normal",
                           "cluster_L1": [1, 2]})
    crisis = pd.DataFrame({C.PK: ["TW::a", "TW::b"], "tree_id": "TW_crisis",
                           "cluster_L1": [9, 8]})
    combined = pd.concat([normal, crisis], ignore_index=True)
    out = s3.build_co_fail_regimes(combined, combined, "TW", "L1")
    assert (out["n_co_fail_peers"] == 0).all()
    assert (out["co_fail_peers"] == "").all()


@test
def t_co_fail_regimes_majority_vote_not_unanimous():
    """危機期歸屬取眾數，不要求群內成員 100% 一致才算數"""
    from . import stage3_hrp as s3
    normal = pd.DataFrame({C.PK: ["TW::a", "TW::b", "TW::c"], "tree_id": "TW_normal",
                           "cluster_L1": [1, 1, 1]})
    crisis = pd.DataFrame({C.PK: ["TW::a", "TW::b", "TW::c"], "tree_id": "TW_crisis",
                           "cluster_L1": [9, 9, 5]})   # 2/3 多數去 9
    combined = pd.concat([normal, crisis], ignore_index=True)
    out = s3.build_co_fail_regimes(combined, combined, "TW", "L1")
    row = out[out.cluster_normal == 1].iloc[0]
    assert row["crisis_dest_cluster"] == 9
    assert abs(row["crisis_dest_share"] - 2 / 3) < 1e-9


# ------------------------------------------------------------ 階段2c 交叉佐證

@test
def t_consistency_majority_rule_basic():
    """熊段月份多數落在預期格（低成長）應判一致，比例與月數皆需正確"""
    from . import stage2c_consistency as r2c
    regime = pd.DataFrame([{"market": "TW", "start": pd.Timestamp("2020-01-01"),
                            "end": pd.Timestamp("2020-04-30"), "label": "熊"}])
    macro = pd.DataFrame([
        {"month": pd.Period("2020-01", "M"), "clock_cell": "衰退"},
        {"month": pd.Period("2020-02", "M"), "clock_cell": "衰退"},
        {"month": pd.Period("2020-03", "M"), "clock_cell": "衰退"},
        {"month": pd.Period("2020-04", "M"), "clock_cell": "復甦"},
    ])
    out = r2c.check_segments(regime, macro)
    row = out.iloc[0]
    assert row["checked"]
    assert row["n_months_valid"] == 4
    assert row["n_months_matched"] == 3
    assert abs(row["pct_match"] - 75.0) < 1e-9
    assert row["consistent"] == True
    assert row["majority_cell"] == "衰退"


@test
def t_consistency_bull_expects_high_growth():
    """牛段的預期格是復甦/過熱，跟熊段那組相反——確認方向沒有寫反"""
    from . import stage2c_consistency as r2c
    regime = pd.DataFrame([{"market": "TW", "start": pd.Timestamp("2021-01-01"),
                            "end": pd.Timestamp("2021-03-31"), "label": "牛"}])
    macro = pd.DataFrame([
        {"month": pd.Period("2021-01", "M"), "clock_cell": "過熱"},
        {"month": pd.Period("2021-02", "M"), "clock_cell": "復甦"},
        {"month": pd.Period("2021-03", "M"), "clock_cell": "衰退"},
    ])
    out = r2c.check_segments(regime, macro)
    row = out.iloc[0]
    assert row["expected_group"] == "復甦|過熱"
    assert row["n_months_matched"] == 2
    assert row["consistent"] == True


@test
def t_consistency_consolidation_not_checked():
    """盤整段無方向性預期，不應被檢查、也不該有 consistent 判定"""
    from . import stage2c_consistency as r2c
    regime = pd.DataFrame([{"market": "TW", "start": pd.Timestamp("2020-01-01"),
                            "end": pd.Timestamp("2020-02-29"), "label": "盤整"}])
    macro = pd.DataFrame([
        {"month": pd.Period("2020-01", "M"), "clock_cell": "衰退"},
        {"month": pd.Period("2020-02", "M"), "clock_cell": "衰退"},
    ])
    out = r2c.check_segments(regime, macro)
    row = out.iloc[0]
    assert not row["checked"]
    assert pd.isna(row["consistent"])
    assert row["expected_group"] is None


@test
def t_consistency_no_macro_data_leaves_unjudged():
    """段內完全沒有可用總經資料時，checked=True 但 consistent 應留白，不能瞎猜"""
    from . import stage2c_consistency as r2c
    regime = pd.DataFrame([{"market": "TW", "start": pd.Timestamp("1990-01-01"),
                            "end": pd.Timestamp("1990-03-31"), "label": "熊"}])
    macro = pd.DataFrame([{"month": pd.Period("2020-01", "M"), "clock_cell": "衰退"}])
    out = r2c.check_segments(regime, macro)
    row = out.iloc[0]
    assert row["checked"]
    assert row["n_months_valid"] == 0
    assert pd.isna(row["consistent"])
    assert pd.isna(row["n_months_matched"])


@test
def t_consistency_tie_break_is_alphabetical():
    """眾數格平手時取字母序最小者，確保結果不因雜湊/疊代順序而變（冪等性前提）"""
    from . import stage2c_consistency as r2c
    cells = pd.Series(["過熱", "衰退"])   # 各1票平手
    assert r2c._majority_cell(cells) == sorted(["過熱", "衰退"])[0]


@test
def t_consistency_real_data_contract():
    """真實資料：2c 產物符合契約，且每個市場至少有一段被實際判定過"""
    for m in ("TW", "US"):
        p = paths.STAGE2 / "consistency" / f"regime_consistency_{m}.parquet"
        if not p.exists():
            raise AssertionError(f"尚未執行 stage2c_consistency（缺 {p}）")
        df = pd.read_parquet(p)
        C.validate(df, C.REGIME_CONSISTENCY)
        judged = df[df.checked & df["consistent"].notna()]
        assert len(judged) > 0, f"[{m}] 沒有任何段被判定，2c 形同沒用"


# ------------------------------------------------------------ 階段4 strategy_map

@test
def t_stage4_regime_fit_tags_direction():
    """regime_fit：熊/危機月份平均報酬>=0才貼標籤，方向不能貼反（牛市好不算防禦標籤）"""
    from . import stage4_strategy_map as s4
    perf = pd.DataFrame([
        {"strategy_uid": "TW::a", "market": "TW", "label": "熊", "n_months": 10, "avg_ret": 0.01, "win_ratio": 0.6},
        {"strategy_uid": "TW::a", "market": "TW", "label": "危機", "n_months": 5, "avg_ret": -0.02, "win_ratio": 0.2},
        {"strategy_uid": "TW::a", "market": "TW", "label": "牛", "n_months": 20, "avg_ret": 0.05, "win_ratio": 0.9},
        {"strategy_uid": "TW::a", "market": "TW", "label": "盤整", "n_months": 5, "avg_ret": 0.0, "win_ratio": 0.5},
        {"strategy_uid": "TW::b", "market": "TW", "label": "熊", "n_months": 10, "avg_ret": -0.01, "win_ratio": 0.3},
        {"strategy_uid": "TW::b", "market": "TW", "label": "危機", "n_months": 5, "avg_ret": 0.03, "win_ratio": 0.8},
        {"strategy_uid": "TW::b", "market": "TW", "label": "牛", "n_months": 20, "avg_ret": 0.02, "win_ratio": 0.7},
        {"strategy_uid": "TW::b", "market": "TW", "label": "盤整", "n_months": 5, "avg_ret": 0.0, "win_ratio": 0.5},
    ])
    tags = s4.derive_regime_fit(perf)
    assert tags["TW::a"] == "熊市抗跌"          # 熊>=0通過，危機<0不通過，牛/盤整不參與貼標
    assert tags["TW::b"] == "危機抗跌"          # 反過來


@test
def t_stage4_regime_fit_respects_min_months():
    """月數不足門檻時，即使平均報酬>=0也不該貼標籤（樣本太少不能下判斷）"""
    from . import stage4_strategy_map as s4
    perf = pd.DataFrame([
        {"strategy_uid": "TW::a", "market": "TW", "label": "熊", "n_months": 1, "avg_ret": 0.5, "win_ratio": 1.0},
    ])
    tags = s4.derive_regime_fit(perf)
    assert "TW::a" not in tags.index or pd.isna(tags.get("TW::a"))


@test
def t_stage4_v1_beneficial_group_broadcast():
    """v1_beneficial 是子樹級事實，v0/v1 兩列應拿到相同值；無對照組留 NaN"""
    from . import stage4_strategy_map as s4
    idx = pd.DataFrame({
        C.PK: ["TW::a_v0", "TW::a_v1", "TW::b_v0"],
        "market": ["TW", "TW", "TW"],
        "f_combo": ["fc1", "fc1", "fc2"],
        "C_id": ["C1", "C1", "C1"],
        "V": ["v0", "v1", "v0"],
        "CAGR": [0.10, 0.15, 0.08],
    })
    out = s4.compute_v1_beneficial(idx, log=lambda *a, **k: None)
    assert out["TW::a_v0"] == True and out["TW::a_v1"] == True   # v1(15%) > v0(10%)，兩列同值
    assert pd.isna(out["TW::b_v0"])                              # 沒有 v1 對照組


@test
def t_stage4_real_data_contract():
    """真實資料：strategy_map / regime_performance / macro_performance 契約通過"""
    for name, schema in (("strategy_map", C.STRATEGY_MAP),
                         ("regime_performance", C.REGIME_PERFORMANCE),
                         ("macro_performance", C.MACRO_PERFORMANCE)):
        p = paths.STAGE4 / f"{name}.parquet"
        if not p.exists():
            raise AssertionError(f"尚未執行 stage4_strategy_map（缺 {p}）")
        df = pd.read_parquet(p)
        C.validate(df, schema)
    sm = pd.read_parquet(paths.STAGE4 / "strategy_map.parquet")
    idx = _load_stage0()
    assert len(sm) == len(idx), f"strategy_map 列數({len(sm)}) != candidate_index({len(idx)})"
    assert sm[C.PK].is_unique


# ------------------------------------------------------------ 實戰部工具層 T1-T13

def _ops_examples():
    from ops import tools as T
    sm = T._strategy_map()
    tw = sm[(sm.market == "TW") & (sm.is_usable)].sort_values("CAGR", ascending=False).iloc[0].strategy_uid
    us = sm[(sm.market == "US") & (sm.is_usable)].sort_values("CAGR", ascending=False).iloc[0].strategy_uid
    return tw, us


@test
def t_ops_t1_covers_all_12_cells():
    """T1：3類型×4regime×2市場全部24組合都能產出非空條件，不拋例外（v8完整性驗收方法）"""
    from ops import tools as T
    for itype in ("保守型", "積極型", "全天候"):
        for regime in ("牛", "熊", "危機", "盤整"):
            for market in ("TW", "US"):
                r = T.t1_get_recommended_criteria(itype, regime, market)
                assert len(r["criteria"]) > 0 or r["uid_whitelist"] is not None, \
                    f"{itype}/{regime}/{market} 沒有任何條件"
                assert r["method"] in ("filter", "cluster_diversify")


@test
def t_ops_t1_conservative_crisis_stricter_than_bull():
    """保守型危機格的mdd_pct門檻必須比牛市格嚴（regime惡化收緊，方向不能顛倒）"""
    from ops import tools as T
    bull = {c[0]: c[2] for c in T.t1_get_recommended_criteria("保守型", "牛", "TW")["criteria"]}
    crisis = {c[0]: c[2] for c in T.t1_get_recommended_criteria("保守型", "危機", "TW")["criteria"]}
    assert crisis["mdd_pct"] > bull["mdd_pct"]


@test
def t_ops_t2_cluster_quota_increases_diversity():
    """T2：加cluster配額後，涵蓋群數不該減少（配額防多樣性假象的核心訴求）"""
    from ops import tools as T
    crit = T.t1_get_recommended_criteria("積極型", "牛", "US")["criteria"]
    no_quota = T.t2_filter_pool(crit, market="US")
    with_quota = T.t2_filter_pool(crit, market="US", cluster_quota=2, cluster_level="L3")
    assert with_quota["n_after_quota"] <= no_quota["n_matched"]
    assert with_quota["n_clusters_covered"] is not None and with_quota["n_clusters_covered"] > 0


@test
def t_ops_t3_profile_real_data():
    """T3：真實資料批次查profile，欄位齊全且缺策略時明確報錯（不能靜默回傳空）"""
    from ops import tools as T
    tw, us = _ops_examples()
    prof = T.t3_get_strategy_profile([tw, us])
    assert len(prof) == 2
    assert {"credibility_grade", "regime_fit", "cluster_L1"} <= set(prof[0].keys())
    expect_raises(KeyError, T.t3_get_strategy_profile, ["TW::not_a_real_strategy"])


@test
def t_ops_t13_cluster_info_real_data():
    """T13：查真實策略的群資訊，成員數與co_fail_regimes結構正確"""
    from ops import tools as T
    tw, _ = _ops_examples()
    r = T.t13_get_cluster_info(strategy_uid=tw)
    assert r["n_members"] > 0
    assert r["co_fail_regimes"] is None or "crisis_dest_cluster" in r["co_fail_regimes"]
    assert len(r["regime_performance"]) == 4   # 牛熊危機盤整四列，缺重疊者n_strategies=0而非漏列


@test
def t_ops_t5_xm_scope_shows_cross_market_segregation():
    """T5(xm範圍)：驗證6.2節發現——TW策略在XM樹裡相關最低的群應幾乎全是US群（市場分裂現象）"""
    from ops import tools as T
    tw, _ = _ops_examples()
    r = T.t5_get_complements(tw, scope="xm", k=3)
    ca = T._cluster_assign()
    xm = ca[ca.tree_id == "XM_normal"]
    us_clusters = set(xm[xm.strategy_uid.str.startswith("US::")]["cluster_L1"].unique())
    hit = sum(1 for c in r["lowest_corr_clusters"] if c["cluster_id"] in us_clusters)
    assert hit >= 2, f"預期相關最低的群多數是美股群，實際只有{hit}/3個"


@test
def t_ops_t6_correlation_matrix_symmetric():
    """T6：即時算的相關矩陣須對稱、對角線=1"""
    from ops import tools as T
    tw, us = _ops_examples()
    r = T.t6_check_correlation([tw, us])
    if r["normal"]["corr"] is not None:
        c = r["normal"]["corr"]
        assert abs(c[tw][tw] - 1.0) < 1e-6
        assert abs(c[tw][us] - c[us][tw]) < 1e-9


@test
def t_ops_t7_sector_beta_always_none():
    """T7：產業β欄位必須固定回傳None（資料不存在，不可編造），真alpha判決不可誤植為四道全過"""
    from ops import tools as T
    tw, _ = _ops_examples()
    r = T.t7_get_return_story_verdict(tw)
    assert r["產業β"] is None
    assert "產業β" in r["note"] or "四道" not in r.get("note", "四道")


@test
def t_ops_t8_portfolio_weights_length_check():
    """T8：weights長度須與策略清單一致，長度不符要報錯不能靜默錯位"""
    from ops import tools as T
    tw, us = _ops_examples()
    expect_raises(ValueError, T.t8_compute_portfolio_risk, [tw, us], [0.5])
    r = T.t8_compute_portfolio_risk([tw, us])
    assert abs(sum(r["weights"]) - 1.0) < 1e-6


@test
def t_ops_t9_hrp_weights_sum_to_one():
    """T9：HRP權重與等權baseline都須加總為1"""
    from ops import tools as T
    tw, us = _ops_examples()
    r = T.t9_compute_weights([tw, us])
    assert abs(sum(r["hrp_weight"].values()) - 1.0) < 1e-4
    assert abs(sum(r["equal_weight"].values()) - 1.0) < 1e-6


@test
def t_ops_t11_regime_label_valid():
    """T11：當前regime標籤必須是四個合法值之一，且用的是2a同一套zigzag函式"""
    from ops import tools as T
    for m in ("TW", "US"):
        r = T.t11_get_current_regime(m)
        assert r["label"] in C.REGIME_LABELS
        assert r["provisional"] is True


@test
def t_ops_t12_knn_distances_sorted():
    """T12：k-NN類比月必須依距離由近到遠排序（否則Agent2引用「最像的那個月」會引用錯）"""
    from ops import tools as T
    r = T.t12_query_macro_model("TW", {"growth": 3.0, "inflation": 1.5,
                                       "rate_level": 2.0, "rate_direction": 20.0}, k=8)
    dists = [d["dist"] for d in r["analog_detail"]]
    assert dists == sorted(dists)


# ------------------------------------------------------------ 產出A 20年情境對照表

@test
def t_output_a_segment_stats_free_lunch_math():
    """_segment_stats：組合MDD與個股平均MDD的算法要對——用一組刻意設計的合成資料，
    兩檔策略從不同時間各自單獨大跌、但同時持有時互補（免費午餐應為正值）"""
    from . import output_a as oa
    months = pd.period_range("2020-01", "2020-04", freq="M")
    wide = pd.DataFrame({
        months[0]: [-0.30, 0.05],
        months[1]: [0.10, -0.30],
        months[2]: [0.05, 0.10],
        months[3]: [0.02, 0.02],
    }, index=["A", "B"])
    stats = oa._segment_stats(wide, ["A", "B"], months, {"A": 0.5, "B": 0.5})
    assert not stats["insufficient"]
    # 個別看：A在month0跌30%、B在month1跌30%，各自MDD都接近-30%
    # 一起等權持有：任一月份最大跌幅只有各自的一半，MDD應遠淺於平均個股MDD
    assert stats["free_lunch_mdd_gain"] > 0.03, \
        f"互補設計下免費午餐應明顯為正，實際 {stats['free_lunch_mdd_gain']}"


@test
def t_output_a_segment_stats_insufficient_months():
    """_segment_stats：共同月數低於門檻時要明確標記insufficient，不能硬算出誤導數字"""
    from . import output_a as oa
    months = pd.period_range("2020-01", "2020-01", freq="M")
    wide = pd.DataFrame({months[0]: [0.01, 0.02]}, index=["A", "B"])
    stats = oa._segment_stats(wide, ["A", "B"], months, {"A": 0.5, "B": 0.5})
    assert stats["insufficient"]


@test
def t_output_a_pick_diversified_respects_size_and_uniqueness():
    """_pick_diversified：不重複選同一策略，且不超過size上限"""
    from . import output_a as oa
    pool = pd.DataFrame({
        "strategy_uid": [f"TW::s{i}" for i in range(10)],
        "credibility_score_pct": list(range(10, 0, -1)),
        "cluster_L1": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    })
    picked = oa._pick_diversified(pool, 4)
    assert len(picked) == 4
    assert len(set(picked)) == 4


@test
def t_output_a_real_data_contract():
    """真實資料：產出A涵蓋全部(段×3類型)組合，beats_market只在有效評估時才有值"""
    p = paths.FROZEN / "output_a" / "scenario_table.parquet"
    if not p.exists():
        raise AssertionError("尚未執行 research.output_a")
    df = pd.read_parquet(p)
    for m in ("TW", "US"):
        n_seg = len(pd.read_parquet(paths.STAGE2 / "regime" / f"regime_table_{m}.parquet"))
        assert (df.market == m).sum() == n_seg * 3, f"{m} 列數應為段數×3類型"
    evaluated = df[df["beats_market"].notna()]
    assert (evaluated["n_selected"] >= 2).all(), "有beats_market判定的列，選兵數不該<2"
    no_cand = df[df.n_candidates == 0]
    assert no_cand["beats_market"].isna().all(), "無候選的列不該有beats_market判定"


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
def t_stage3_crisis_trees_real_data():
    """真實資料：六棵樹（3 normal + 3 crisis）都建成，crisis 樹策略宇宙與 normal 相同"""
    p = paths.STAGE3 / "cluster_assign.parquet"
    if not p.exists():
        raise AssertionError("尚未執行 stage3_hrp，請先 python -m research.cli stage3")
    df = pd.read_parquet(p)
    C.validate(df, C.CLUSTER_ASSIGN)
    got = set(df.tree_id.unique())
    assert got == set(C.TREE_IDS), f"樹不齊全：缺 {set(C.TREE_IDS) - got}"
    for m in ("TW", "US", "XM"):
        n_normal = df[df.tree_id == f"{m}_normal"][C.PK].nunique()
        n_crisis = df[df.tree_id == f"{m}_crisis"][C.PK].nunique()
        # crisis 樹可能因零變異數策略被排除而略少於 normal，但不該差太多（<1%）
        assert n_crisis <= n_normal, f"{m} crisis 策略數({n_crisis}) > normal({n_normal})，不合理"
        dropped_pct = (n_normal - n_crisis) / n_normal
        assert dropped_pct < 0.01, f"{m} crisis 樹排除了 {dropped_pct:.1%} 策略，異常偏高"


@test
def t_stage3_universe_excludes_non_usable():
    """回歸測試：v9 規定階段3只對usable_pool算，HRP樹不得含 is_usable=False 的策略

    曾經是真實 bug：`_tree_universe()` 只用 returns_meta 篩市場+歷史起始日，
    從沒 join strategy_marks.is_usable，導致階段1尾端硬篩掉的策略（769個）
    全部漏回六棵樹（TW污染6.26%／US 3.70%／XM 4.85%，比例與階段1淘汰率吻合）。
    """
    p_marks = paths.STAGE1 / "strategy_marks.parquet"
    p_assign = paths.STAGE3 / "cluster_assign.parquet"
    if not (p_marks.exists() and p_assign.exists()):
        raise AssertionError("尚未執行 stage1_marks / stage3_hrp")
    marks = pd.read_parquet(p_marks)
    assign = pd.read_parquet(p_assign)
    not_usable = set(marks.loc[~marks.is_usable, C.PK])
    contaminated = assign[assign[C.PK].isin(not_usable)]
    assert contaminated.empty, \
        f"{len(contaminated)} 筆 is_usable=False 的策略混進了 HRP 樹（{sorted(contaminated.tree_id.unique())}）"


@test
def t_stage3_co_fail_regimes_real_data():
    """真實資料：co_fail_regimes 契約通過，且每個市場都算出 L1 全部群的危機期歸屬"""
    p = paths.STAGE3 / "co_fail_regimes.parquet"
    if not p.exists():
        raise AssertionError("尚未執行 stage3_hrp（缺 co_fail_regimes.parquet）")
    df = pd.read_parquet(p)
    C.validate(df, C.CO_FAIL_REGIMES)
    for m in ("TW", "US", "XM"):
        sub = df[df.tree_key == m]
        assert len(sub) == 8, f"{m} L1 群數應為 8（L1_TARGET），實際 {len(sub)} 筆"
        assert sub["crisis_dest_share"].between(0, 1).all()


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
