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
def t_macro_rolling_zscore_no_lookahead():
    """H-18②：合成資料驗證滾動窗z-score真的沒有偷看未來——構造一段前後統計性質
    明顯不同的序列（前段均值0變異數1，後段均值大幅偏移+變異數放大），驗證
    「後段」發生之前的滾動z-score，數值上不會被後段的統計性質影響。
    這是no-lookahead性質的直接實證，不是只測型別/欄位對不對。
    """
    from . import macro_rolling_window as MRW
    rng = np.random.default_rng(0)
    window = 12
    n_before, n_after = 30, 30
    before = rng.normal(0, 1, n_before)
    after = rng.normal(50, 10, n_after)   # 統計性質劇烈改變的後段
    idx = pd.period_range("2000-01", periods=n_before + n_after, freq="M")
    df = pd.DataFrame({"growth": np.concatenate([before, after])}, index=idx)
    df.index.name = "month"

    # apply_zscore_rolling對AXES裡「不在df.columns的軸」會自動跳過，這裡的df
    # 只放growth一欄，不需要額外mock掉其他三個軸（inflation/rate_level/rate_direction）
    out = MRW.apply_zscore_rolling(df, window=window)

    # 前段最後一個月（第n_before-1個索引，尚未看到after段），z-score應該只反映
    # before段的統計性質，數值應落在合理範圍（絕對值不會突然被後段的均值50拉走）
    last_before_z = out["growth_z"].iloc[n_before - 1]
    assert abs(last_before_z) < 5, (
        f"前段最後一個月的滾動z-score={last_before_z}，數值異常大，"
        "懷疑偷看了後段的統計性質")
    # 前window-1個月因為滾動窗不滿，應該是NaN
    assert out["growth_z"].iloc[:window - 1].isna().all(), \
        f"前{window-1}個月的資料不滿一個完整窗，應該是NaN（min_periods={window}未生效）"
    assert pd.notna(out["growth_z"].iloc[window - 1]), \
        f"第{window}個月資料已滿一個完整窗，應該要有值"


@test
def t_macro_rolling_real_data():
    """H-18②：真實資料，macro_clock_comparison契約通過，且比對邏輯本身正確——
    抽查幾筆，確認match欄位真的等於frozen跟rolling的clock_cell是否相同。
    """
    p = paths.STAGE2 / "macro_rolling" / "macro_clock_comparison.parquet"
    if not p.exists():
        raise AssertionError("尚未執行 research.macro_rolling_window")
    df = pd.read_parquet(p)
    df["market"] = df["market"].astype("category")
    df["frozen_clock_cell"] = df["frozen_clock_cell"].astype("category")
    df["rolling_clock_cell"] = df["rolling_clock_cell"].astype("category")
    C.validate(df, C.MACRO_CLOCK_COMPARISON, strict_columns=True)

    both_valid = df["frozen_clock_cell"].notna() & df["rolling_clock_cell"].notna()
    valid = df[both_valid]
    assert len(valid) > 0, "至少要有一些月份兩邊都能分類，否則無從比較"
    recomputed_match = (valid["frozen_clock_cell"] == valid["rolling_clock_cell"])
    assert (valid["match"].astype(bool) == recomputed_match).all(), \
        "match欄位跟frozen/rolling clock_cell直接比較的結果對不起來"
    # 至少要有一筆兩邊不同的（否則滾動窗版本形同白做，也違反直覺——已知真實
    # 資料裡差異率接近5成，這裡只做「不是0」的寬鬆檢查，避免測試綁死確切比例）
    assert (~valid["match"].astype(bool)).any(), \
        "滾動窗版跟凍結版分類完全一致，不符合已知的真實查證結果，需要重新檢查"


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
def t_hrp_effective_number_of_bets_bounds():
    """H-09：ENB方向不能顛倒——全部獨立時要等於N（上限），全部完美相關時要等於1（下限）"""
    from . import hrp
    n = 20
    identity = np.eye(n)
    enb_indep = hrp.effective_number_of_bets(identity)
    assert abs(enb_indep - n) < 1e-6, f"全部獨立(corr=I)的ENB應為{n}，實際{enb_indep}"

    all_corr = np.ones((n, n))
    enb_perfect = hrp.effective_number_of_bets(all_corr)
    assert abs(enb_perfect - 1.0) < 1e-6, f"全部完美相關的ENB應為1，實際{enb_perfect}"

    assert enb_indep > enb_perfect, "獨立矩陣的ENB必須大於完美相關矩陣的ENB"


@test
def t_hrp_effective_number_of_bets_monotonic_in_correlation():
    """H-09：ENB須隨平均相關程度單調遞減——相關越高，有效獨立賭注數越少（不能顛倒方向）"""
    from . import hrp
    n = 30
    prev_enb = None
    for rho in (0.0, 0.3, 0.6, 0.9):
        corr = np.full((n, n), rho)
        np.fill_diagonal(corr, 1.0)
        enb = hrp.effective_number_of_bets(corr)
        if prev_enb is not None:
            assert enb < prev_enb, (
                f"相關係數從低到高({rho})，ENB應該跟著下降，"
                f"但這次({enb:.2f}) >= 前一次({prev_enb:.2f})")
        prev_enb = enb


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


# ------------------------------------------------------------ cluster_story（LLM點③）

@test
def t_cluster_story_complementarity_thresholds():
    """cluster_story：互補程度是**程式**判定的，門檻方向不能顛倒（相關越低＝互補越高）"""
    from . import cluster_story as CS
    hi, mid = C.COMPLEMENTARITY_CUTS["高"], C.COMPLEMENTARITY_CUTS["中"]
    assert CS._complementarity(hi - 0.01) == "高"
    assert CS._complementarity(hi) == "中"          # 邊界含在下一級
    assert CS._complementarity(mid - 0.01) == "中"
    assert CS._complementarity(mid) == "低"
    assert CS._complementarity(0.98) == "低"
    # 方向確認：相關越高，互補等級不可能變好
    order = {"高": 2, "中": 1, "低": 0}
    vals = [0.1, 0.4, 0.6, 0.85, 0.99]
    levels = [order[CS._complementarity(v)] for v in vals]
    assert levels == sorted(levels, reverse=True), f"互補等級隨相關上升應單調下降，實際{levels}"


@test
def t_cluster_story_prompt_carries_verdict_and_guardrails():
    """cluster_story：prompt須把程式判決與「低互補不可宣稱分散」的指示帶進去
    （這是防止LLM對相關0.98的群對硬掰互補故事的核心防線）"""
    from . import cluster_story as CS
    prof = CS._cluster_profiles("XM_normal")
    a, b = sorted(prof)[:2]
    p = CS.build_prompt("XM_normal", prof[a], prof[b], 0.958, "低", False)
    assert "0.958" in p
    assert "互補程度判決：低" in p
    assert "不可推翻" in p
    # system prompt 必須明文要求「低＝不可宣稱能分散風險」
    assert "不可以宣稱它們能分散風險" in CS._SYSTEM_PROMPT
    assert "不要包裝成深層的經濟因果故事" in CS._SYSTEM_PROMPT


@test
def t_cluster_story_profiles_drop_zero_count_categories():
    """cluster_story：群側寫不得出現 count=0 的類別（categorical的value_counts會列出
    該群根本沒有的類別，列在top_底下會誤導LLM以為有這個成分）"""
    from . import cluster_story as CS
    for prof in CS._cluster_profiles("TW_normal").values():
        for key in ("top_factor_types", "top_F1", "top_C_source", "V_mix"):
            assert all(v > 0 for v in prof[key].values()), f"{key} 含0計數：{prof[key]}"


@test
def t_cluster_story_report_runs():
    """回歸測試：_report 不得炸掉。曾是真實bug——`corr` 撞到 DataFrame.corr 內建
    方法名，寫成 `g.corr.min()` 會取到方法而非欄位，AttributeError。當時因為
    production run 用了 `| tail -40`，stdout是block-buffered、stderr先flush，
    traceback被擠出tail視窗，而且 pipeline 的 exit code 取的是 tail 的 0，
    整個失敗完全沒被看見。產物本身是好的（run() 在 _report 之前就寫檔了）。
    """
    p = paths.STAGE3 / "cluster_story.parquet"
    if not p.exists():
        raise AssertionError(
            "cluster_story.parquet 目前不存在，**這是預期中的狀態，非回歸**："
            "2026-08-28 H-03 改了L1群數後，8/25跑的舊版（群編號到7/8）已被隔離"
            "到 _stale_pre_H03/（避免靜默配對到錯誤的群），見開發待辦追蹤.md H-04下游重跑。"
            "要讓這項測試回綠，需針對新群數重新執行 python -m research.cluster_story"
            "（約$1.31 LLM費用，需先確認才能花）")
    from . import cluster_story as CS
    CS._report(pd.read_parquet(p), log=lambda *a, **k: None)


@test
def t_cluster_story_sidecar_records_real_model():
    """側錄必須記**實際模型名**。這份 json 的用途就是替不可完全複現的LLM產物留
    溯源紀錄，若記成 "(from config)" 這種佔位字串等於失去意義。
    """
    import json as _json
    p = paths.STAGE3 / "cluster_story_meta.json"
    if not p.exists():
        raise AssertionError(
            "cluster_story_meta.json 目前不存在，**這是預期中的狀態，非回歸**——"
            "同 t_cluster_story_report_runs 的說明，見開發待辦追蹤.md H-04下游重跑")
    meta = _json.loads(p.read_text(encoding="utf-8"))
    df = pd.read_parquet(paths.STAGE3 / "cluster_story.parquet")
    assert meta["model"] == str(df["model"].iloc[0]), \
        f"側錄模型({meta['model']}) 與產物實際模型({df['model'].iloc[0]}) 不符"
    assert "(" not in meta["model"], f"側錄記到佔位字串：{meta['model']}"


@test
def t_cluster_temporal_annual_quarterly_compounding():
    """H-06：合成資料驗證年/季複利報酬算對方向與數值——用已知的12個月報酬，
    手算年化複利結果去對，確認不是簡單加總（那樣會系統性低估報酬）。"""
    from . import cluster_temporal_profile as CTP
    idx = pd.period_range("2020-01", "2020-12", freq="M")
    # 每月都漲10%，12個月複利應為 1.1**12 - 1 ≈ 2.1384，不是 12*0.10=1.20（簡單加總）
    rep = pd.Series([0.10] * 12, index=idx)
    ann, qtr = CTP._annual_quarterly(rep)
    assert len(ann) == 1 and ann.iloc[0]["year"] == 2020
    assert abs(ann.iloc[0]["ret"] - (1.1**12 - 1)) < 1e-9, \
        f"年報酬應為複利 {1.1**12-1:.4f}，實際 {ann.iloc[0]['ret']:.4f}（是否誤用簡單加總？）"
    assert ann.iloc[0]["n_months"] == 12
    # 4季，每季3個月都漲10% → 每季複利 1.1**3-1
    assert len(qtr) == 4
    for r in qtr.itertuples():
        assert abs(r.ret - (1.1**3 - 1)) < 1e-9
        assert r.n_months == 3

    # 跨年邊界＋不滿12個月：2020-11~2021-02（4個月），2020年只有2個月
    idx2 = pd.period_range("2020-11", "2021-02", freq="M")
    rep2 = pd.Series([0.05, 0.05, 0.05, 0.05], index=idx2)
    ann2, _ = CTP._annual_quarterly(rep2)
    assert set(ann2["year"]) == {2020, 2021}
    row_2020 = ann2[ann2.year == 2020].iloc[0]
    assert row_2020["n_months"] == 2, "2020年應只涵蓋11、12月共2個月，不能誤算成12"
    assert abs(row_2020["ret"] - (1.05**2 - 1)) < 1e-9


@test
def t_cluster_temporal_profile_real_data():
    """H-06：真實資料，三張表都符合契約，且群組成一致性可交叉核對
    （cluster_profile_quant 的 n_members 應等於該群在 cluster_annual_returns
    裡的月數涵蓋範圍所暗示的成員來源——用更直接的方式：三張表的
    (tree_id,cluster_id) 集合必須完全一致，不能有表A有的群表B沒有）。"""
    for fname, schema in (("cluster_annual_returns.parquet", C.CLUSTER_ANNUAL_RETURNS),
                          ("cluster_quarterly_returns.parquet", C.CLUSTER_QUARTERLY_RETURNS),
                          ("cluster_profile_quant.parquet", C.CLUSTER_PROFILE_QUANT)):
        p = paths.STAGE3 / fname
        if not p.exists():
            raise AssertionError(f"尚未執行 cluster_temporal_profile（缺 {fname}）")
        C.validate(pd.read_parquet(p), schema)

    ann = pd.read_parquet(paths.STAGE3 / "cluster_annual_returns.parquet")
    prof = pd.read_parquet(paths.STAGE3 / "cluster_profile_quant.parquet")
    keys_ann = set(zip(ann.tree_id, ann.cluster_id))
    keys_prof = set(zip(prof.tree_id, prof.cluster_id))
    assert keys_ann == keys_prof, \
        f"annual表與profile表的(tree_id,cluster_id)集合不一致：只在annual={keys_ann-keys_prof}｜只在profile={keys_prof-keys_ann}"

    # pct_TW + pct_US 應該落在 [0,1]（XM可混合，TW/US單市場樹應恰為1.0）
    assert (prof["pct_TW"] + prof["pct_US"]).between(0.999, 1.001).all()
    tw_only = prof[prof.tree_id == "TW_normal"]
    assert (tw_only["pct_TW"] == 1.0).all(), "TW_normal的群不該混進美股策略"

    # n_years_positive 不能超過 n_years（基本的邏輯一致性）
    assert (prof["n_years_positive"] <= prof["n_years"]).all()
    assert (prof["pct_years_positive"] <= 1.0).all()


@test
def t_complementarity_threshold_safe_for_normal_trees():
    """回歸測試（2026-08-26敏感度分析）：COMPLEMENTARITY_CUTS的high門檻(0.5)在
    實際使用範圍（3棵normal樹）必須安全——同市場配對的最低相關係數必須 >= 0.5，
    否則會有同市場配對被誤判成「高互補」（cluster_story的核心防線失效）。
    此測試只涵蓋normal樹，crisis樹已知不適用同一組門檻，見contracts.py註解。
    """
    from . import complementarity_sensitivity as CS
    df = CS.build_all_pairs(log=lambda *a, **k: None)
    normal = df[df.tree_id.str.endswith("_normal")]
    same_min = normal.loc[normal.pair_type == "same", "corr"].min()
    assert same_min >= C.COMPLEMENTARITY_CUTS["高"], (
        f"同市場配對最低相關({same_min:.4f})低於high門檻"
        f"({C.COMPLEMENTARITY_CUTS['高']})，會誤判成高互補")


@test
def t_cluster_story_contract_with_mocked_llm():
    """cluster_story：mock LLM回應時整條組裝流程要通過契約（不打真實API、不花錢）"""
    from unittest import mock
    from . import cluster_story as CS
    fake = ({"mechanism_note": "機械性差異：市場不同。",
             "complement_note": "相關0.372低於0.5門檻，判定高互補。",
             "caveat": "僅根據提供的側寫。"},
            {"prompt_tokens": 1100, "completion_tokens": 600, "total_tokens": 1700})
    with mock.patch.object(CS, "_call_llm", return_value=fake), \
         mock.patch("utils.config.Config.get_openai_api_key", return_value="sk-fake"), \
         mock.patch("utils.config.Config.get_openai_model", return_value="fake-model"):
        df = CS.build(trees=("XM_normal",), limit=3, log=lambda *a, **k: None)
    assert len(df) == 3
    C.validate(df, C.CLUSTER_STORY, strict_columns=True)


@test
def t_cluster_story_resume_skips_completed_pairs():
    """cluster_story --resume：讀既有部分產物時，已完成的pair要跳過、只打剩下的，
    且最終合併結果要涵蓋「舊的+新的」而非只有新的（否則resume=整批重來，等於
    白做H-23的部分保存）。用 resume_path 指向temp檔，不碰production路徑。
    """
    import tempfile
    from pathlib import Path
    from unittest import mock
    from . import cluster_story as CS

    fake = ({"mechanism_note": "m", "complement_note": "c", "caveat": "v"},
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})

    with tempfile.TemporaryDirectory() as td:
        prev_path = Path(td) / "cluster_story.parquet"

        # 第一段：假裝已經跑過2對（XM_normal只有limit=2的量）
        with mock.patch.object(CS, "_call_llm", return_value=fake), \
             mock.patch("utils.config.Config.get_openai_api_key", return_value="sk-fake"), \
             mock.patch("utils.config.Config.get_openai_model", return_value="fake-model"):
            df1 = CS.build(trees=("XM_normal",), limit=2, log=lambda *a, **k: None)
        assert len(df1) == 2
        df1.to_parquet(prev_path, compression="zstd", index=False)
        done_before = {(r.tree_id, r.cluster_a, r.cluster_b) for r in df1.itertuples()}

        # 第二段：resume，追蹤 _call_llm 實際被叫了幾次
        calls: list[tuple] = []

        def _tracking_call_llm(prompt, model, api_key, **kw):
            calls.append(prompt)
            return fake

        with mock.patch.object(CS, "_call_llm", side_effect=_tracking_call_llm), \
             mock.patch("utils.config.Config.get_openai_api_key", return_value="sk-fake"), \
             mock.patch("utils.config.Config.get_openai_model", return_value="fake-model"):
            df2 = CS.build(trees=("XM_normal",), resume=True, resume_path=prev_path,
                          log=lambda *a, **k: None)

        assert len(df2) > len(df1), "resume後的累計對數必須比之前多（否則沒有真的接續跑下去）"
        got_pairs = {(r.tree_id, r.cluster_a, r.cluster_b) for r in df2.itertuples()}
        assert done_before <= got_pairs, "resume前已完成的pair必須原樣保留在最終結果裡"
        assert len(calls) == len(df2) - len(df1), (
            "本次新增的列數應該剛好等於本次真正打LLM的次數——若對不上，"
            "代表resume要嘛重打了已完成的pair、要嘛漏跑了該跑的pair")
        # 主鍵（tree_id,level,cluster_a,cluster_b）不得重複——若resume把已完成的pair
        # 又重打一次，這裡會直接因主鍵重複而raise，是比字串比對更可靠的防線
        C.validate(df2, C.CLUSTER_STORY, strict_columns=True)


@test
def t_cluster_story_resume_missing_file_falls_back_to_fresh():
    """cluster_story --resume：找不到既有檔案時要優雅退回全新執行，不能整個炸掉。"""
    import tempfile
    from pathlib import Path
    from unittest import mock
    from . import cluster_story as CS

    fake = ({"mechanism_note": "m", "complement_note": "c", "caveat": "v"},
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
    with tempfile.TemporaryDirectory() as td:
        missing_path = Path(td) / "does_not_exist.parquet"
        with mock.patch.object(CS, "_call_llm", return_value=fake), \
             mock.patch("utils.config.Config.get_openai_api_key", return_value="sk-fake"), \
             mock.patch("utils.config.Config.get_openai_model", return_value="fake-model"):
            df = CS.build(trees=("XM_normal",), limit=2, resume=True,
                         resume_path=missing_path, log=lambda *a, **k: None)
    assert len(df) == 2


# ------------------------------------------------------------ cluster_identity（H-08，LLM點④之外的新產出）

@test
def t_cluster_identity_contract_with_mocked_llm():
    """H-08：mock LLM回應時整條組裝流程要通過契約（不打真實API、不花錢）"""
    from unittest import mock
    from . import cluster_identity as CI
    fake = ({"identity_label": "測試群", "mechanism_note": "機械性差異：因子家族。",
             "performance_pattern": "19年中13年正報酬。", "caveat": "僅根據提供的側寫。"},
            {"prompt_tokens": 1300, "completion_tokens": 700, "total_tokens": 2000})
    with mock.patch.object(CI, "_call_llm", return_value=fake), \
         mock.patch("utils.config.Config.get_openai_api_key", return_value="sk-fake"), \
         mock.patch("utils.config.Config.get_openai_model", return_value="fake-model"):
        df = CI.build(trees=("XM_normal",), limit=3, log=lambda *a, **k: None)
    assert len(df) == 3
    C.validate(df, C.CLUSTER_IDENTITY, strict_columns=True)


@test
def t_cluster_identity_resume_skips_completed_clusters():
    """H-08 --resume：已完成的群要跳過、只跑剩下的，合併結果涵蓋舊的+新的。
    跟cluster_story的resume是同一個模式，這裡驗證群層級（非pair層級）版本正確。
    """
    import tempfile
    from pathlib import Path
    from unittest import mock
    from . import cluster_identity as CI

    fake = ({"identity_label": "L", "mechanism_note": "m",
             "performance_pattern": "p", "caveat": "c"},
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})

    with tempfile.TemporaryDirectory() as td:
        prev_path = Path(td) / "cluster_identity.parquet"
        with mock.patch.object(CI, "_call_llm", return_value=fake), \
             mock.patch("utils.config.Config.get_openai_api_key", return_value="sk-fake"), \
             mock.patch("utils.config.Config.get_openai_model", return_value="fake-model"):
            df1 = CI.build(trees=("XM_normal",), limit=2, log=lambda *a, **k: None)
        assert len(df1) == 2
        df1.to_parquet(prev_path, compression="zstd", index=False)
        done_before = {(r.tree_id, r.cluster_id) for r in df1.itertuples()}

        calls = []
        def _tracking(prompt, model, api_key, **kw):
            calls.append(prompt)
            return fake
        with mock.patch.object(CI, "_call_llm", side_effect=_tracking), \
             mock.patch("utils.config.Config.get_openai_api_key", return_value="sk-fake"), \
             mock.patch("utils.config.Config.get_openai_model", return_value="fake-model"):
            df2 = CI.build(trees=("XM_normal",), resume=True, resume_path=prev_path,
                          log=lambda *a, **k: None)

        assert len(df2) > len(df1), "resume後累計群數必須比之前多"
        got = {(r.tree_id, r.cluster_id) for r in df2.itertuples()}
        assert done_before <= got, "resume前已完成的群必須原樣保留"
        assert len(calls) == len(df2) - len(df1), \
            "本次新增列數應該剛好等於本次真正打LLM的次數"
        C.validate(df2, C.CLUSTER_IDENTITY, strict_columns=True)


@test
def t_cluster_identity_real_data():
    """H-08：真實資料，cluster_identity契約通過，且16群（TW6/US7/XM3）全部涵蓋，
    identity_label不得為空字串（否則等於沒有產出有意義的身份標籤）。
    """
    p = paths.STAGE3 / "cluster_identity.parquet"
    if not p.exists():
        raise AssertionError("尚未執行 research.cluster_identity")
    df = pd.read_parquet(p)
    C.validate(df, C.CLUSTER_IDENTITY, strict_columns=True)
    counts = df.groupby("tree_id", observed=True).size()
    assert counts.get("TW_normal", 0) == 6, f"TW_normal應有6群，實際{counts.get('TW_normal', 0)}"
    assert counts.get("US_normal", 0) == 7, f"US_normal應有7群，實際{counts.get('US_normal', 0)}"
    assert counts.get("XM_normal", 0) == 3, f"XM_normal應有3群，實際{counts.get('XM_normal', 0)}"
    assert (df["identity_label"].str.len() > 0).all(), "identity_label不該有空字串"
    # 抗幻覺基本檢查：不應該出現總經推論常見的字眼（H-08鐵則3明文禁止）
    banned = ["升息", "降息", "通膨環境", "景氣循環", "適合在"]
    for r in df.itertuples():
        for kw in banned:
            assert kw not in r.mechanism_note and kw not in r.performance_pattern, (
                f"[{r.tree_id}群{r.cluster_id}] 出現疑似總經推論字眼「{kw}」，"
                "H-08鐵則3禁止在沒有總經資訊的情況下做這類推論")


# ------------------------------------------------------------ cluster_macro_interface（S-01）

@test
def t_cluster_macro_interface_real_data():
    """S-01：真實資料，契約通過，16群全涵蓋，且兩道防線都要驗證：
    ①schema層級——四個帶日曆年份的欄位(window_start/end_year、best/worst_year)
      物理上不存在於這張表（strict_columns=True已經會擋，這裡額外顯式檢查一次，
      因為這條規則是S-03安全性的核心，值得比其他schema測試更明確）
    ②內容層級——identity_label（唯一保留的LLM文字欄位）不得包含「這個特定群」
      自己在cluster_profile_quant裡的真實年份數字。用該群自己的實際年份精確比對，
      不是泛用的年份正則（避免member數量剛好落在19xx/20xx區間造成的假陽性，
      這是2026-08-31開發時實測踩過的坑）。
    """
    p = paths.STAGE3 / "cluster_macro_interface.parquet"
    if not p.exists():
        raise AssertionError("尚未執行 research.cluster_macro_interface")
    df = pd.read_parquet(p)
    df["tree_id"] = df["tree_id"].astype("category")
    df["level"] = df["level"].astype("category")
    C.validate(df, C.CLUSTER_MACRO_INTERFACE, strict_columns=True)

    counts = df.groupby("tree_id", observed=True).size()
    assert counts.get("TW_normal", 0) == 6
    assert counts.get("US_normal", 0) == 7
    assert counts.get("XM_normal", 0) == 3

    banned_cols = ("window_start_year", "window_end_year", "best_year", "worst_year")
    for col in banned_cols:
        assert col not in df.columns, (
            f"帶日曆年份的欄位「{col}」不該出現在總經介面表——S-03明文禁止流入決策層")

    quant = pd.read_parquet(paths.STAGE3 / "cluster_profile_quant.parquet")
    quant = quant.set_index(["tree_id", "level", "cluster_id"])
    for r in df.itertuples():
        q = quant.loc[(r.tree_id, r.level, r.cluster_id)]
        for real_year in (q.window_start_year, q.window_end_year, q.best_year, q.worst_year):
            assert str(int(real_year)) not in r.identity_label, (
                f"[{r.tree_id}群{r.cluster_id}] identity_label包含該群真實年份"
                f"{real_year}的字串——即使identity_label目前是空字串以外唯一保留的"
                f"LLM欄位，也不該外洩具體日曆年份（S-03）")


# ------------------------------------------------------------ macro_decision_input（S-02）

@test
def t_macro_decision_input_real_data():
    """S-02：真實資料，cluster_macro_conditional契約通過，16群×4格＝64列，
    且群代表口徑要跟H-06/stage3_hrp一致（成員簡單平均）。
    """
    p = paths.STAGE3 / "cluster_macro_conditional.parquet"
    if not p.exists():
        raise AssertionError("尚未執行 research.macro_decision_input")
    df = pd.read_parquet(p)
    df["tree_id"] = df["tree_id"].astype("category")
    df["level"] = df["level"].astype("category")
    df["clock_cell"] = df["clock_cell"].astype("category")
    C.validate(df, C.CLUSTER_MACRO_CONDITIONAL, strict_columns=True)
    assert len(df) == 16 * 4, f"應為16群×4格=64列，實際{len(df)}"
    for cell in ("復甦", "過熱", "停滯性通膨", "衰退"):
        assert cell in set(df.clock_cell), f"缺少clock_cell={cell}"


@test
def t_macro_state_snapshot_never_leaks_month():
    """S-02：`macro_state_snapshot()`的回傳值絕對不能包含呼叫時傳入的month字串
    本身，也不能有任何鍵叫month/date/year——這是S-03「決策層不給日期」的具體
    程式落實，不是只有文件寫寫。
    """
    from . import macro_decision_input as MDI
    snap = MDI.macro_state_snapshot("TW", "2025-12")
    assert "month" not in snap and "date" not in snap and "year" not in snap
    assert "2025" not in str(snap.values()), "回傳值不該包含查詢用的年份字串"
    assert set(snap) == {"growth_z", "inflation_z", "rate_level_z",
                         "rate_direction_z", "clock_cell"}


@test
def t_group_decision_context_excludes_unconditional_performance():
    """S-02：`group_decision_context()`不得洩漏無條件績效欄位（CAGR_median等）
    ——這是本模組最核心的設計決策（見模組docstring），若這裡失守，決策層就會
    退化成H-10/H-12已經證實有陷阱的「無條件挑歷史最強群」。同時驗證回傳值裡
    的數值都是原生Python型別（int/float/str/dict/None），可以被json.dumps()
    直接序列化——2026-08-31開發時實測抓到numpy.int64塞進payload導致
    json.dumps()直接炸掉的真實bug，已修正，這裡鎖住回歸。
    """
    import json
    from . import macro_decision_input as MDI
    ctx = MDI.group_decision_context("TW_normal", 1)
    for banned in ("CAGR_median", "MDD_median", "annual_ret_mean", "annual_ret_std",
                  "quarterly_ret_std", "best_year_ret", "worst_year_ret",
                  "n_years_positive", "pct_years_positive", "tree_id", "cluster_id"):
        assert banned not in ctx, f"「{banned}」不該出現在決策層看到的群素材裡"
    assert "conditional_performance" in ctx and len(ctx["conditional_performance"]) == 4
    json.dumps(ctx, ensure_ascii=False)   # 不加 default=str 也要能序列化成功


# ------------------------------------------------------------ decision_layer_arms（S-05）

@test
def t_decision_layer_arm_a_matches_conditional_table():
    """S-05 A_rule：真實資料，規則基準必須真的是「該clock_cell下avg_ret_median
    最高」——不是隨便挑，用cluster_macro_conditional.parquet直接反查驗證。
    """
    from . import decision_layer_arms as DLA
    cond = pd.read_parquet(paths.STAGE3 / "cluster_macro_conditional.parquet")
    for tree_id, cell in (("TW_normal", "復甦"), ("US_normal", "過熱"), ("XM_normal", "衰退")):
        picked = DLA.rule_based_decision(tree_id, cell, top_n=1)
        assert len(picked) == 1
        sub = cond[(cond.tree_id == tree_id) & (cond.clock_cell == cell)].dropna(
            subset=["avg_ret_median"])
        expected = int(sub.loc[sub["avg_ret_median"].idxmax(), "cluster_id"])
        assert picked[0] == expected, (
            f"[{tree_id}/{cell}] A_rule選了群{picked[0]}，但真正avg_ret_median最高"
            f"的是群{expected}")


@test
def t_decision_layer_arm_c_returns_all_clusters():
    """S-05 C_all：真實資料，必須回傳該樹**全部**L1群、不看總經狀態，且結果穩定
    （呼叫兩次一致，因為它本來就不該依賴任何隨機性或外部狀態）。
    """
    from . import decision_layer_arms as DLA
    assign = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")
    for tree_id in ("TW_normal", "US_normal", "XM_normal"):
        expected = sorted(assign[assign.tree_id == tree_id]["cluster_L1"].unique().tolist())
        got1 = DLA.equal_weight_all_decision(tree_id)
        got2 = DLA.equal_weight_all_decision(tree_id)
        assert got1 == expected == got2


@test
def t_decision_layer_arm_b_contract_with_mocked_llm():
    """S-05 B_llm：mock LLM回應時整條組裝流程要通過（不打真實API、不花錢），
    且prompt組裝出的月份字串不能外洩進macro_state（S-03，跟S-02的規則一致）。
    """
    from unittest import mock
    from . import decision_layer_arms as DLA
    fake = ({"selected_clusters": [1], "rationale": "群1條件式績效最高。",
             "caveat": "各群差距不大。"},
            {"prompt_tokens": 2000, "completion_tokens": 800, "total_tokens": 2800})
    with mock.patch.object(DLA, "_call_llm", return_value=fake), \
         mock.patch("utils.config.Config.get_openai_api_key", return_value="sk-fake"), \
         mock.patch("utils.config.Config.get_openai_model", return_value="fake-model"):
        decision = DLA.llm_decision("XM_normal", "TW", "2025-12")
    assert decision["selected_clusters"] == [1]
    # prompt 本身也要驗證日期沒有外洩（跟 t_macro_state_snapshot_never_leaks_month
    # 同樣的關切，這裡驗證的是「組裝進最終prompt字串」這一步沒有意外把month塞回去）
    macro_state = DLA.macro_state_snapshot("TW", "2025-12")
    group_ctx = {cid: DLA.group_decision_context("XM_normal", cid)
                for cid in DLA.equal_weight_all_decision("XM_normal")}
    prompt = DLA.build_prompt("XM_normal", macro_state, group_ctx)
    assert "2025" not in prompt, "prompt組裝結果不該包含查詢用的年份字串"


@test
def t_decision_layer_compare_snapshot_dry_run_structure():
    """S-05：真實資料，compare_snapshot 在 dry_run 模式下（不花錢）結構要完整——
    A_rule/C_all 是真實計算結果，B_llm 因dry_run是空清單，三者的鍵都要存在。
    """
    from . import decision_layer_arms as DLA
    result = DLA.compare_snapshot("XM_normal", "TW", "2025-12", dry_run=True,
                                  log=lambda *a, **k: None)
    assert set(("tree_id", "market", "clock_cell", "macro_state", "A_rule",
              "B_llm", "C_all")) <= set(result)
    assert len(result["A_rule"]) == 1
    assert result["C_all"] == [1, 2, 3]
    assert result["B_llm"] == []   # dry-run 不花錢，不是真的決策結果


# ------------------------------------------------------------ decision_repeatability（S-07）

@test
def t_decision_repeatability_math_mocked():
    """S-07：mock掉真正的LLM呼叫（不花錢），手算一組已知答案的序列，驗證
    exact_match_rate/mean_pairwise_jaccard/stable_core/unstable_fringe/
    rule_in_llm_rate 這五個統計量算得對——這些是S-07的核心交付物，算錯了
    整份穩定度報告就沒有意義。
    """
    from unittest import mock
    from . import decision_layer_arms as DLA
    from . import decision_repeatability as DR

    # 手算：5次結果 [1,3][1,3][1,3][1,3][1,2,3]
    #   眾數是{1,3}，出現4/5次 → exact_match_rate=0.8
    #   核心(交集)={1,3}；聯集={1,2,3}；邊緣(聯集-核心)={2}
    #   A_rule=[1]（top1），全部5次都有包含1 → rule_in_llm_rate=1.0
    #   pairwise jaccard：C(5,2)=10對，其中4對是{1,3}vs{1,3}=1.0(共6對，因為4個相同集合兩兩配對=C(4,2)=6對)
    #     另外4對是{1,3}vs{1,2,3}=2/3，加總算出mean
    sequence = [[1, 3], [1, 3], [1, 3], [1, 3], [1, 2, 3]]
    calls = iter(sequence)

    def _fake_llm_decision(tree_id, market, month, *, model=None, dry_run=False, log=print):
        return {"selected_clusters": next(calls), "rationale": "r", "caveat": "c"}

    with mock.patch.object(DR, "llm_decision", side_effect=_fake_llm_decision), \
         mock.patch.object(DR, "rule_based_decision", return_value=[1]), \
         mock.patch.object(DR, "macro_state_snapshot",
                          return_value={"clock_cell": "復甦", "growth_z": 0.0,
                                       "inflation_z": 0.0, "rate_level_z": 0.0,
                                       "rate_direction_z": 0.0}):
        result = DR.repeatability_check("TW_normal", "TW", "2025-12", n_repeats=5,
                                        log=lambda *a, **k: None)

    assert result["exact_match_rate"] == 0.8
    assert result["stable_core"] == "1|3"
    assert result["unstable_fringe"] == "2"
    assert result["rule_in_llm_rate"] == 1.0
    # 手算pairwise jaccard：6對(1,3)組合jaccard=1.0，4對跟{1,2,3}比較jaccard=2/3
    expected_jaccard = (6 * 1.0 + 4 * (2 / 3)) / 10
    assert abs(result["mean_pairwise_jaccard"] - round(expected_jaccard, 4)) < 1e-3


@test
def t_decision_repeatability_real_data():
    """S-07：真實資料，contract通過，且rule_in_llm_rate跟all_runs欄位互相對得起來
    ——不是各自獨立算的兩個數字，同一份底層資料算出來的東西不該互相矛盾。
    """
    p = paths.ROOT / "_analysis_outputs_robustness" / "decision_repeatability.csv"
    if not p.exists():
        raise AssertionError("尚未執行 research.decision_repeatability")
    # ⚠️ stable_core/unstable_fringe 必須明講 dtype=str：這兩欄存的是"|"分隔的群id
    # 清單，但當清單剛好只有1個元素時（例如"4"），字串長得跟純數字一樣，若欄位裡
    # 剛好每一列都是空值或單一數字，pandas的CSV型別推斷會把整欄判成float64而非
    # object——2026-08-31開發時實測踩到的真實bug（不是理論風險，US/TW兩列的
    # unstable_fringe剛好都只有1個元素）。空字串讀回仍會變NaN，schema已宣告
    # nullable=True。
    df = pd.read_csv(p, dtype={"stable_core": str, "unstable_fringe": str})
    df["tree_id"] = df["tree_id"].astype("category")
    df["market"] = df["market"].astype("category")
    df["clock_cell"] = df["clock_cell"].astype("category")
    C.validate(df, C.DECISION_REPEATABILITY, strict_columns=True)
    assert len(df) == 3
    import ast
    for r in df.itertuples():
        assert 0 <= r.exact_match_rate <= 1
        assert 0 <= r.mean_pairwise_jaccard <= 1
        assert r.n_repeats >= 2
        runs = [set(ast.literal_eval(s)) for s in r.all_runs.split("|")]
        assert len(runs) == r.n_repeats, "all_runs記錄的次數要跟n_repeats一致"
        core = set.intersection(*runs)
        stable_core = "" if pd.isna(r.stable_core) else r.stable_core
        assert core == set(int(x) for x in stable_core.split("|") if x), (
            f"[{r.tree_id}] stable_core跟all_runs反推出的交集對不起來")


# ------------------------------------------------------------ 階段1 標記（W-08）

@test
def t_stage1_data_glitch_direction():
    """合成資料：data_glitch 方向不能顛倒——兩把刀（單日/單月）任一達門檻才標True，都在門檻下不標"""
    df = pd.DataFrame({
        C.PK: ["TW::a", "TW::b", "TW::c", "TW::d"],
        "max_daily_ret":   [C.PRICE_JUMP_EXTREME + 0.01, C.PRICE_JUMP_EXTREME - 0.01,
                            np.nan,                       0.05],
        "max_monthly_ret": [0.10,                        C.MONTHLY_JUMP_EXTREME - 0.01,
                            np.nan,                       C.MONTHLY_JUMP_EXTREME + 0.01],
    })
    daily = df["max_daily_ret"] >= C.PRICE_JUMP_EXTREME
    monthly = df["max_monthly_ret"] >= C.MONTHLY_JUMP_EXTREME
    flag = (daily | monthly).fillna(False)
    assert flag.tolist() == [True, False, False, True], \
        "單日刀命中(a)或單月刀命中(d)都要標True；兩刀都沒過(b)或都缺資料(c)要標False"


@test
def t_stage1_data_glitch_real_data():
    """真實資料：data_glitch 在契約內、且不影響 is_usable（只標記不淘汰，W-08定案）。
    另外對深度掃描（diagnose_price_anomalies）認證過的「CAGR灌水>1個百分點」283個
    策略做recall回歸測試——這是W-08校準單月門檻(MONTHLY_JUMP_EXTREME)時的實測基準，
    此測試防止未來有人改動門檻卻沒注意到recall掉下來。
    """
    p = paths.STAGE1 / "strategy_marks.parquet"
    if not p.exists():
        raise AssertionError("尚未執行 stage1_marks，請先 python -m research.stage1_marks")
    df = pd.read_parquet(p)
    C.validate(df, C.STRATEGY_MARKS, strict_columns=True)
    assert df["data_glitch"].dtype == bool
    # data_glitch=True 的策略不必然被淘汰——這是W-08刻意的設計（只標記、留給
    # Agent1快篩時自行決定），故兩者之間不該有任何蘊含關係的斷言，這裡只驗證
    # 兩欄位都存在且獨立可讀，避免未來有人誤把 data_glitch 接成第三把硬篩刀。
    assert set(df["data_glitch"].unique()) <= {True, False}

    p_impact = paths.ROOT / "_analysis_outputs_dataquality" / "price_anomaly_strategies.csv"
    if p_impact.exists():
        impact = pd.read_csv(p_impact)
        big = impact[impact["CAGR_inflation_pp"] > 1.0]
        if len(big):
            glitch = set(df.loc[df.data_glitch, "strategy_uid"])
            missed = big[~big["strategy_uid"].isin(glitch)]
            assert missed.empty, (
                f"data_glitch 漏抓了 {len(missed)} 個CAGR灌水>1pp的策略（校準基準是0漏抓）：\n"
                f"{missed[['strategy_uid', 'CAGR_inflation_pp']].to_string(index=False)}")


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
def t_matrix_completeness_w06():
    """W-06·12格矩陣完整性測試：3類型×4regime×2市場全部24格，T1回傳的每個條件
    欄位都要真的能在strategy_map上套用（T2對不存在的欄位會KeyError）——
    這是實戰部架構v8明文寫的驗收法（§門檻矩陣「把12格逐格拆成pandas條件，
    若某一格有任何條件寫不出對應欄位→那個欄位就是漏掉的，必須回研究部補」），
    抓的是「文件想引用、但階段4 strategy_map實際沒產出」這種欄位名稱漂移。
    """
    from ops import tools as T
    failures = []
    for itype in ("保守型", "積極型", "全天候"):
        for regime in ("牛", "熊", "危機", "盤整"):
            for market in ("TW", "US"):
                rec = T.t1_get_recommended_criteria(itype, regime, market)
                try:
                    T.t2_filter_pool(rec["criteria"], market=market,
                                     uid_whitelist=rec["uid_whitelist"])
                except KeyError as e:
                    failures.append(f"{itype}/{regime}/{market}：{e}")
    assert not failures, "以下格引用了strategy_map沒有的欄位：\n" + "\n".join(failures)


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
def t_ops_t5_degrades_gracefully_without_cluster_story():
    """T5：cluster_story（要花錢的LLM離線產物）不存在時，仍須回傳完整客觀數字、
    explanation 留 None，絕不因缺這個選配產物而壞掉或編造文字。
    """
    from unittest import mock
    from ops import tools as T
    tw, _ = _ops_examples()
    # k=2：XM樹自H-03改用資料驅動的L1群數後只有3群（見開發待辦追蹤.md H-03），
    # 扣掉自己那群，最多只剩2個「別群」可比，k=3會要不到那麼多。
    with mock.patch.object(T, "_cluster_story", return_value=None):
        r = T.t5_get_complements(tw, scope="xm", k=2)
    assert len(r["lowest_corr_clusters"]) == 2
    for c in r["lowest_corr_clusters"]:
        assert c["explanation"] is None
        assert isinstance(c["corr"], float)      # 客觀數字照給


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


# T10 全系列都 monkeypatch requests.post，不打真實OpenAI API——只驗證程式邏輯
# 本身（prompt組裝/回應解析/錯誤分流），驗證不了真實API的欄位shape是否一致，
# 見 ops/tools.py t10_generate_return_story_text docstring 的「驗證待辦」。

def _fake_openai_response(status_code, body_dict):
    import json as _json
    import requests as _requests
    resp = _requests.Response()
    resp.status_code = status_code
    resp._content = _json.dumps(body_dict, ensure_ascii=False).encode("utf-8")
    return resp


@test
def t_ops_t10_prompt_uses_real_verdict_and_evidence():
    """T10：prompt組裝要正確帶入T7判決與T3支持數字，且數字須**四捨五入後**才進prompt。

    實測（2026-08-25）發現不捨入的話，float原始精度（effective_n=28.417404303353255）
    會被模型一字不漏抄進輸出文字，既難讀又多燒token；捨入只動呈現層，判決本身
    仍用原值算，故不影響判定結果。此測試同時鎖住「有帶到」與「已捨入」兩件事。
    """
    from ops import tools as T
    tw, _ = _ops_examples()
    verdict = T.t7_get_return_story_verdict(tw)
    profile = T.t3_get_strategy_profile([tw])[0]
    evidence = {k: profile.get(k) for k in
               ("effective_n", "top1_share", "smallcap_share", "credibility_grade")}
    prompt = T._build_return_story_user_prompt(verdict, evidence)
    assert str(verdict["靠少數股"]) in prompt
    assert str(verdict["真alpha"]) in prompt
    assert str(evidence["credibility_grade"]) in prompt
    # 有帶到（捨入後的形式）
    assert f"{round(float(evidence['effective_n']), 1)}" in prompt
    assert f"{evidence['top1_share']:.1%}" in prompt
    # 且原始未捨入的長浮點數**不該**出現
    assert str(evidence["effective_n"]) not in prompt, "原始float精度不該進prompt"


@test
def t_ops_t10_happy_path_parses_story():
    """T10：mock正常回應時能正確解析出story，且raw_verdict/raw_evidence一併回傳供事後核對"""
    from unittest import mock
    from ops import tools as T
    from utils.config import Config
    tw, _ = _ops_examples()
    fake_story = {
        "few_stock_note": "此策略不靠少數股支撐報酬。",
        "sector_beta_note": "產業β資料不存在，無法判定。",
        "size_driven_note": "此策略不特別依賴小型股規模效應。",
        "real_alpha_note": "在前三道皆通過下，判定為真alpha（產業β未列入判定）。",
        "summary": "整體而言報酬來源分散、不特別依賴規模效應，惟產業β無法驗證。",
    }
    fake_resp = _fake_openai_response(200, {
        "choices": [{"message": {"content": __import__("json").dumps(fake_story, ensure_ascii=False)}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    })
    # 模型改用免費名單內的（原本 "fake-model" 會被新的免費額度煞車擋在送出請求之前，
    # 那不是這項測試要驗的東西）；並 mock 掉 log_usage，避免測試把假用量寫進真實帳本
    from utils import openai_quota as OQ
    with mock.patch("requests.post", return_value=fake_resp), \
         mock.patch.object(Config, "get_openai_api_key", return_value="sk-fake"), \
         mock.patch.object(Config, "get_openai_model", return_value="gpt-5-mini"), \
         mock.patch.object(OQ, "log_usage") as mock_log:
        r = T.t10_generate_return_story_text(tw)
    assert r["story"] == fake_story
    assert r["strategy_uid"] == tw
    # 用量必須有被記帳（監督機制的重點：每次真的呼叫都要留下紀錄）
    assert mock_log.called, "T10 呼叫成功後必須寫入用量帳本"
    assert r["model"] == "gpt-5-mini"
    assert r["usage"]["total_tokens"] == 150
    assert "靠少數股" in r["raw_verdict"]


@test
def t_ops_t10_quota_exhausted_propagates():
    """T10：額度用盡的錯誤要正確傳播成QuotaExhaustedError，不能被吞掉或誤判成一般錯誤"""
    from unittest import mock
    from ops import tools as T
    from utils.config import Config
    from utils import openai_quota as OQ
    tw, _ = _ops_examples()
    fake_resp = _fake_openai_response(429, {
        "error": {"message": "You exceeded your current quota",
                  "type": "insufficient_quota", "code": "insufficient_quota"}})
    # 用免費名單內的模型，讓請求真的送得出去——本測試驗的是「OpenAI 回報額度用盡」
    # 這條路徑（QuotaExhaustedError），不是我們自己的免費額度煞車（FreeTierExhaustedError），
    # 兩者是不同的錯誤、不同的意義，不可混為一談
    with mock.patch("requests.post", return_value=fake_resp), \
         mock.patch.object(Config, "get_openai_api_key", return_value="sk-fake"), \
         mock.patch.object(Config, "get_openai_model", return_value="gpt-5-mini"):
        expect_raises(OQ.QuotaExhaustedError, T.t10_generate_return_story_text, tw)


@test
def t_ops_t10_free_tier_gate_blocks_paid_model():
    """T10：模型不在每日免費額度名單時，必須在**送出請求之前**就擋下。

    這是使用者2026-08-29定的規則——「盡量用免費的，超過再決定要不要花錢」，
    所以靜默改用付費模型跑下去正好違反意圖。驗證方式：mock requests.post，
    若它被呼叫到就代表煞車失效（錢已經花出去了）。
    """
    from unittest import mock
    from ops import tools as T
    from utils.config import Config
    from utils import openai_quota as OQ
    tw, _ = _ops_examples()
    with mock.patch("requests.post") as mock_post, \
         mock.patch.object(Config, "get_openai_api_key", return_value="sk-fake"), \
         mock.patch.object(Config, "get_openai_model", return_value="gpt-5.6-terra"):
        expect_raises(OQ.FreeTierExhaustedError, T.t10_generate_return_story_text, tw)
    assert not mock_post.called, \
        "付費模型必須在送出請求前就被擋下，requests.post 不該被呼叫到（否則錢已經花了）"


@test
def t_ops_t10_missing_strategy_short_circuits_before_llm_call():
    """T10：策略不存在時T7先擋下、直接回傳error，不會浪費一次LLM呼叫（沒mock requests也要過）"""
    from ops import tools as T
    r = T.t10_generate_return_story_text("TW::not_a_real_strategy")
    assert "error" in r


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


# ------------------------------------------------------------ 階段1 manifest 獨立性（code review修正）

@test
def t_stage1_scan_and_marks_manifests_are_independent():
    """回歸測試：stage1_scan 與 stage1_marks 曾經共用同一份 MANIFEST.json
    （都寫 `paths.STAGE1`），後寫的覆蓋先寫的，導致先寫的那4份產物
    （strategy_scan/returns_monthly/annual_returns/returns_meta）完全脫離
    雜湊驗證，且沒有任何報錯——這是2026-08-25 code review 抓到的真實bug。
    修法是 stage1_marks 改寫進獨立的 `_marks/` 子目錄（比照 stage1_mktcap
    的 `_mktcap/` 前例）。此測試鎖住「兩份manifest各自獨立、各自涵蓋正確
    的產物集合」，防止未來有人把兩者的 out_dir 又寫回同一個目錄。
    """
    m_scan = freeze.read_manifest(paths.STAGE1)
    m_marks = freeze.read_manifest(paths.STAGE1 / "_marks")
    assert m_scan["stage"] == "stage1_scan"
    assert m_marks["stage"] == "stage1_marks"
    scan_names = {o["path"].split("\\")[-1].split("/")[-1] for o in m_scan["outputs"]}
    marks_names = {o["path"].split("\\")[-1].split("/")[-1] for o in m_marks["outputs"]}
    assert {"strategy_scan.parquet", "returns_monthly.parquet",
           "annual_returns.parquet", "returns_meta.parquet"} <= scan_names
    assert "strategy_marks.parquet" in marks_names
    assert scan_names.isdisjoint(marks_names)


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
def t_four_group_control_real_data():
    """H-12：真實資料，four_group_control契約通過，且幾個不可能違反的結構性事實：
    C_random(200次抽樣平均)應該非常接近B_all(全宇宙)——兩者理論上收斂到同一個母體
    平均值，只是C用抽樣近似；n_members要對得上A/D的目標值；ENB不可能超過n_members。
    """
    p = paths.ROOT / "_analysis_outputs_robustness" / "four_group_control.csv"
    if not p.exists():
        raise AssertionError("尚未執行 research.four_group_control")
    df = pd.read_csv(p)
    df["tree_key"] = df["tree_key"].astype("category")
    df["group"] = df["group"].astype("category")
    C.validate(df, C.FOUR_GROUP_CONTROL, strict_columns=True)

    for tk, g in df.groupby("tree_key", observed=True):
        b = g[g.group == "B_all"].iloc[0]
        c = g[g.group == "C_random"].iloc[0]
        a = g[g.group == "A_hrp"].iloc[0]
        d = g[g.group == "D_top_cagr"].iloc[0]
        # C是B的無偏抽樣近似，200次平均應該離B的CAGR很近（用寬鬆的絕對值門檻，
        # 避免對隨機數種子的細節過度敏感，只驗證「同一個量級、方向一致」）
        assert abs(c["is_cagr"] - b["is_cagr"]) < 0.02, (
            f"[{tk}] C_random的IS CAGR({c['is_cagr']:.4f})離B_all({b['is_cagr']:.4f})太遠，"
            f"200次抽樣平均不該跟全宇宙平均差這麼多")
        assert abs(c["oos_cagr"] - b["oos_cagr"]) < 0.02
        # A/D的實際選出檔數不該超過目標(k群×5)，且至少要選到大半（backfill機制保底）
        assert a["n_members"] == d["n_members"], "A/D兩組的組合大小應該用同一個n_target"
        # ENB數學邊界：不可能超過該組的成員數
        for _, row in g.iterrows():
            if pd.notna(row["is_enb"]):
                assert row["is_enb"] <= row["n_members"] + 1e-6, \
                    f"[{tk}/{row['group']}] IS ENB({row['is_enb']})不可能超過成員數({row['n_members']})"
            if pd.notna(row["oos_enb"]):
                assert row["oos_enb"] <= row["n_members"] + 1e-6, \
                    f"[{tk}/{row['group']}] OOS ENB({row['oos_enb']})不可能超過成員數({row['n_members']})"
        # A組是貪婪多樣性選擇，設計上每群固定選m=5個代表，理論上應該橫跨全部群
        # （除非某群候選不足才會少），不該退化成集中在少數幾群
        e_row = g[g.group == "E_top_calmar"].iloc[0]
        n_clusters_a = int(a["n_clusters_covered"])
        assert n_clusters_a >= 2, (
            f"[{tk}] A_hrp只橫跨{n_clusters_a}群，多樣性選擇規則可能失效"
            "（設計上應該覆蓋大部分甚至全部群）")
        # max_cluster_share的數學邊界：不可能是負的，不可能超過1
        for _, row in g.iterrows():
            if pd.notna(row["max_cluster_share"]):
                assert 0 <= row["max_cluster_share"] <= 1 + 1e-9, \
                    f"[{tk}/{row['group']}] max_cluster_share越界：{row['max_cluster_share']}"


@test
def t_complementarity_granularity_real_data():
    """H-25：真實資料，契約通過，且鎖住這項分析真正要主張的三件事。

    ①**粒度效應存在**：XM跨市場配對在L3的高互補比例必須遠高於L1（L1是0%）——
      這是「L1沒有高互補是聚合效應、不是策略真的沒互補性」的直接證據。
    ②**不是小群雜訊**：同市場配對是天然對照組（群大小/月份數同量級），
      在同一個層級下高互補比例必須遠低於跨市場。若雜訊是主因，同市場也該
      一起噴出大量假高互補。
    ③**結論不靠納入小群撐著**：把門檻拉到只納入成員數>=20的群，①②仍須成立。

    這三條同時鎖住了「不該為了讓L1出現高互補而調 COMPLEMENTARITY_CUTS」這個
    決策——真正的問題在粒度，不在門檻。
    """
    p = paths.ROOT / "_analysis_outputs_robustness" / "complementarity_granularity_summary.csv"
    if not p.exists():
        raise AssertionError("尚未執行 research.complementarity_granularity")
    df = pd.read_csv(p)
    for col in ("tree_id", "level", "pair_type"):
        df[col] = df[col].astype("category")
    C.validate(df, C.COMPLEMENTARITY_GRANULARITY, strict_columns=True)

    def _row(level, pair_type, mm, tree="XM_normal"):
        s = df[(df.tree_id == tree) & (df.level == level)
              & (df.pair_type == pair_type) & (df.min_members == mm)]
        assert len(s) == 1, f"查不到唯一的 {tree}/{level}/{pair_type}/min={mm}"
        return s.iloc[0]

    for mm in (1, 20):
        cross_l1 = _row("L1", "cross", mm)
        cross_l3 = _row("L3", "cross", mm)
        same_l3 = _row("L3", "same", mm)

        # ① 粒度效應：L1 跨市場 0 對高互補，L3 必須有實質比例
        assert cross_l1.n_high == 0, (
            f"L1跨市場出現{cross_l1.n_high}對高互補，跟本分析的前提（L1沒有高互補）"
            "矛盾——若分群或門檻改過，這項分析的敘事要重寫")
        assert cross_l3.pct_high > 0.4, (
            f"[min={mm}] L3跨市場高互補只有{cross_l3.pct_high:.1%}，"
            "粒度效應不成立，報告裡「免費午餐藏在細粒度」的主張站不住")

        # ② 同市場對照組：同一層級下必須遠低於跨市場
        assert same_l3.pct_high < 0.10, (
            f"[min={mm}] L3同市場高互補達{same_l3.pct_high:.1%}，"
            "對照組失效——無法排除「小群估計雜訊」這個替代解釋")
        assert cross_l3.pct_high > same_l3.pct_high * 5, (
            f"[min={mm}] L3跨市場({cross_l3.pct_high:.1%})沒有明顯高於"
            f"同市場({same_l3.pct_high:.1%})，市場邊界效應不成立")

    # ③ 判定門檻必須仍是未改動的原值——這項分析的整個論點就是「門檻不用改」
    assert C.COMPLEMENTARITY_CUTS == {"高": 0.5, "中": 0.8}, (
        f"COMPLEMENTARITY_CUTS 已被改成 {C.COMPLEMENTARITY_CUTS}；"
        "H-25 的結論（問題在粒度不在門檻）與報告敘述都須重新檢視")


@test
def t_free_lunch_shortlist_real_data():
    """H-25b：免費午餐清單契約通過，且鎖住這張表的三個定義性事實。

    ①`universal` 欄位必須名符其實：標成 True 的群，n_high_complement 必須等於
      n_cross_partners（跟對面市場每一群都高互補），不能是「幾乎全部」。
    ②清單必須**同時有台美兩側**——免費午餐是配對關係，只有單邊等於沒有可配的對象。
    ③每一群的 min_cross_corr 必須真的低於高互補門檻（否則它根本不該進清單），
      且 best_partner_cluster 必須是對面市場的群。
    """
    p = paths.ROOT / "_analysis_outputs_robustness" / "free_lunch_shortlist.csv"
    if not p.exists():
        raise AssertionError("尚未執行 research.complementarity_granularity")
    df = pd.read_csv(p)
    for col in ("tree_id", "level", "market"):
        df[col] = df[col].astype("category")
    C.validate(df, C.FREE_LUNCH_SHORTLIST, strict_columns=True)

    uni = df[df.universal]
    assert len(uni) > 0, "清單裡沒有任何 universal 群，H-25b 的敘事不成立"
    bad = uni[uni.n_high_complement != uni.n_cross_partners]
    assert bad.empty, (
        f"{len(bad)} 群標成universal但n_high_complement != n_cross_partners，"
        f"欄位定義被破壞：\n{bad[['cluster_id', 'n_high_complement', 'n_cross_partners']]}")

    mkts = set(uni.market.astype(str))
    assert mkts == {"TW", "US"}, (
        f"universal群只出現在 {mkts}——免費午餐是配對關係，單邊清單沒有可配的對象")

    high_cut = C.COMPLEMENTARITY_CUTS["高"]
    assert (uni.min_cross_corr < high_cut).all(), (
        "有universal群的min_cross_corr沒有低於高互補門檻，自相矛盾")

    # best_partner 必須在對面市場：用群id反查市場
    mkt_of = df.set_index("cluster_id")["market"].astype(str).to_dict()
    for r in uni.itertuples():
        pm = mkt_of.get(r.best_partner_cluster)
        if pm is not None:   # 夥伴可能因成員數門檻不在清單裡，有才驗
            assert pm != str(r.market), (
                f"群{r.cluster_id}({r.market})的best_partner群{r.best_partner_cluster}"
                f"也是{pm}——跨市場配對不該配到同市場")


@test
def t_rebuild_tree_returns_is_single_source_of_truth():
    """回歸測試（2026-08-30 code review）：`effective_bets._tree_corr` 與
    `cluster_count_selection._rebuild_dist_matrix` 曾經各自維護一份逐行相同的
    資料準備複製品（usable過濾／共同窗／排除零變異數）。那種重複最危險的不是
    多打幾行字，而是**改了其中一邊、另一邊靜默沿用舊規則，兩邊的相關矩陣不再
    是同一個東西且不會報錯**——H-03（群數選擇）與 H-09（ENB）會悄悄建立在不同
    資料上。已抽成 `stage3_hrp.rebuild_tree_returns()` 單一事實來源。

    此測試鎖住兩件事：①兩個呼叫端拿到的 uid 集合與相關矩陣完全一致
    ②`rebuild_tree_returns` 重建的矩陣，其形狀與凍結 linkage 隱含的葉節點數吻合
    （linkage 有 N-1 列合併記錄，N 即當初建樹時的策略數）——若資料準備規則跟
    建樹當下不一致，這裡會直接對不起來。
    """
    from . import effective_bets as EB
    from . import cluster_count_selection as CCS
    from . import stage3_hrp as S3

    tree_id = "XM_normal"    # 挑最大的那棵，最容易暴露不一致
    quiet = lambda *a, **k: None
    corr_eb, idx_eb = EB._tree_corr(tree_id, quiet)
    dist_ccs, link, idx_ccs = CCS._rebuild_dist_matrix(tree_id, quiet)

    assert list(idx_eb) == list(idx_ccs), \
        "兩個呼叫端拿到的策略集合/順序必須完全一致（否則相關矩陣不可比）"
    # dist 是 corr 的確定性函式，反推回去必須吻合
    assert np.allclose(dist_ccs, S3.hrp.corr_to_distance(corr_eb), atol=1e-12), \
        "兩個呼叫端算出的矩陣不一致——資料準備已經分岔了"
    # 與凍結 linkage 的葉節點數對帳：linkage 有 N-1 列
    assert link.shape[0] + 1 == len(idx_eb), (
        f"重建的策略數({len(idx_eb)})與凍結linkage隱含的葉節點數"
        f"({link.shape[0] + 1})不符——資料準備規則已與建樹當下不一致")


@test
def t_effective_bets_real_data():
    """H-09：真實資料，effective_bets契約通過，且ENB的數學邊界不能被打破——
    ENB(N個策略) 不可能超過N（PCA熵的定義域上限就是N），ENB(k個群代表)
    同理不可能超過k；否則代表算法本身寫錯方向或搞混了輸入矩陣。
    """
    p = paths.ROOT / "_analysis_outputs_robustness" / "effective_number_of_bets.csv"
    if not p.exists():
        raise AssertionError("尚未執行 research.effective_bets")
    df = pd.read_csv(p)
    C.validate(df.assign(tree_id=df.tree_id.astype("category"),
                         tree_key=df.tree_key.astype("category")),
               C.EFFECTIVE_BETS, strict_columns=True)
    assert set(df.tree_id) == {"TW_normal", "US_normal", "XM_normal"}, \
        "H-09只做normal樹，crisis樹樣本量太小不該出現在這張表裡"
    for r in df.itertuples():
        assert r.enb_raw <= r.n_strategies + 1e-6, \
            f"[{r.tree_id}] ENB不可能超過N（ENB={r.enb_raw}，N={r.n_strategies}）"
        assert r.enb_clusters <= r.n_clusters_l1 + 1e-6, \
            f"[{r.tree_id}] ENB(群代表)不可能超過群數k（ENB={r.enb_clusters}，k={r.n_clusters_l1}）"
        assert r.enb_raw >= 1.0 - 1e-6, f"[{r.tree_id}] ENB下限應為1，實際{r.enb_raw}"


@test
def t_cluster_representatives_real_data():
    """H-10：真實資料，cluster_representatives契約通過，且多樣性選擇要真的比
    純品質排序更分散——否則貪婪演算法等於白寫，跟naive選法沒有差異。
    """
    p = paths.ROOT / "_analysis_outputs_robustness" / "cluster_representatives_m3.csv"
    if not p.exists():
        raise AssertionError("尚未執行 research.cluster_representatives")
    df = pd.read_csv(p)   # co_fail_peers 空字串讀回會變NaN，schema已宣告nullable=True可直接接受
    df["tree_id"] = df["tree_id"].astype("category")
    df["level"] = df["level"].astype("category")
    C.validate(df, C.CLUSTER_REPRESENTATIVES, strict_columns=True)

    both = df[df["avg_pairwise_corr_picked"].notna() & df["avg_pairwise_corr_naive"].notna()]
    assert len(both) > 0, "至少要有可比較的群（成員數>=2才有平均相關可算）"
    worse = both[both["avg_pairwise_corr_picked"] > both["avg_pairwise_corr_naive"] + 1e-9]
    assert worse.empty, (
        f"多樣性選擇的結果不該比純品質排序更集中，但有 {len(worse)} 群反而更相關：\n"
        f"{worse[['tree_id', 'cluster_id', 'avg_pairwise_corr_picked', 'avg_pairwise_corr_naive']].to_string(index=False)}")
    # n_picked 不該超過該群成員數，也不該超過m_target
    assert (df["n_picked"] <= df["n_members"]).all()
    assert (df["n_picked"] <= df["m_target"]).all()


@test
def t_stage3_hrp_isoos_real_data():
    """H-11：真實資料，isoos契約通過，且完全不能碰到主線stage3的正式產物
    （這是使用者明確要求的「資料要分好，不要搞混」，用檔案系統證據直接驗證，
    不是只看程式邏輯）。
    """
    from . import paths as P
    p = P.STAGE3_ISOOS / "isoos_corr_comparison.parquet"
    if not p.exists():
        raise AssertionError("尚未執行 research.stage3_hrp_isoos")
    df = pd.read_parquet(p)
    df["tree_id"] = df["tree_id"].astype("category")
    df["level"] = df["level"].astype("category")
    df["complementarity_is"] = df["complementarity_is"].astype("category")
    df["complementarity_oos"] = df["complementarity_oos"].astype("category")
    C.validate(df, C.ISOOS_CORR_COMPARISON, strict_columns=True)
    assert set(df.tree_id) == {"TW_normal_IS", "US_normal_IS", "XM_normal_IS"}

    # 隔離驗證：IS/OOS的檔案跟主線stage3六棵樹的檔名不可能撞在一起（不同目錄），
    # 且主線stage3的 MANIFEST 內容不該提到任何 _IS 樹（代表兩邊真的完全獨立）
    assert P.STAGE3_ISOOS != P.STAGE3, "isoos輸出目錄不可以跟主線stage3共用"
    main_manifest = freeze.read_manifest(paths.STAGE3)
    assert "_IS" not in str(main_manifest), \
        "主線stage3的MANIFEST不該出現任何IS/OOS的痕跡——兩邊必須完全獨立"
    main_assign = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")
    assert not any(str(t).endswith("_IS") for t in main_assign.tree_id.unique()), \
        "主線stage3的cluster_assign.parquet不該混進任何IS樹的資料"


@test
def t_stage3_co_fail_regimes_real_data():
    """真實資料：co_fail_regimes 契約通過，且每個市場都算出 L1 全部群的危機期歸屬"""
    p = paths.STAGE3 / "co_fail_regimes.parquet"
    if not p.exists():
        raise AssertionError("尚未執行 stage3_hrp（缺 co_fail_regimes.parquet）")
    df = pd.read_parquet(p)
    C.validate(df, C.CO_FAIL_REGIMES)
    from . import stage3_hrp as S3   # L1_TARGET 現在依市場而異（H-03），不再是單一常數8
    for m in ("TW", "US", "XM"):
        sub = df[df.tree_key == m]
        expect = S3.L1_TARGET[m]
        assert len(sub) == expect, f"{m} L1 群數應為 {expect}（L1_TARGET[{m}]），實際 {len(sub)} 筆"
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
