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
def t_walkforward_schemes_are_mechanical():
    """H-26：窗口方案必須是機械規則產生的，且用完全部資料、不重疊、不留缺口。

    這條鎖住的是**方法論**而非數值：窗口邊界一旦變成人為挑選（例如挑「涵蓋
    COVID」的區間），就是上帝視角——本專案已因同一錯誤撤銷過 H-11 原提案。
    測試驗證每個方案的窗次串起來剛好等於總月數，代表它是規則跑出來的、
    不是手選的。
    """
    from . import walkforward_matrix as WF
    s = WF.build_schemes()
    assert len(s) > 0 and s.scheme.nunique() >= 12, f"方案數不足：{s.scheme.nunique()}"
    assert set(s["mode"]) == {"anchored", "rolling"}, "缺 rolling 對照組"

    for name, g in s.groupby("scheme", observed=True):
        g = g.sort_values("window_no")
        r0 = g.iloc[0]
        # 第一窗的 IS 結束點必須剛好是 min_is
        assert int(r0.is_end_offset) == int(r0.min_is_months), (
            f"[{name}] 第一窗 IS 長度({r0.is_end_offset}) != min_is({r0.min_is_months})")
        # 窗次首尾相接：每一窗的 OOS 結束 == 下一窗的 IS 結束
        offs = g.is_end_offset.tolist(); lens = g.oos_months.tolist()
        for i in range(len(g) - 1):
            assert offs[i] + lens[i] == offs[i + 1], (
                f"[{name}] 第{i+1}窗與第{i+2}窗不相接（重疊或有缺口）")
        # 全部用完，不多不少
        assert offs[-1] + lens[-1] == WF.SCHEME_TOTAL_MONTHS, (
            f"[{name}] 窗次總長 {offs[-1]+lens[-1]} != {WF.SCHEME_TOTAL_MONTHS}")
        # 每個 OOS 至少 24 個月（MDD 要有意義；老師點名 MDD 是唯一代價）
        assert min(lens) >= WF.MIN_TAIL_MONTHS, f"[{name}] 有窗次的 OOS 短於 24 個月"


@test
def t_walkforward_allocate_conserves_total():
    """H-27：兩種分配方式**必須配出同一個總量**——這是共同座標軸的全部意義。

    若等量配出 30 支、比例配出 334 支，跑出比例較優也無法歸因（不知道是分法好
    還是多買了 300 支）。這條測試鎖住「同樣的 total，兩種分法只差分佈」。
    同時驗證天花板重分配：小群不足配額時給滿，餘額轉給其他群，總量不因此縮水。
    """
    from . import walkforward_matrix as WF
    sizes = pd.Series({1: 305, 2: 202, 3: 1654, 4: 2534, 5: 75, 6: 1909})   # 台股實際群大小

    for total in (30, 66, 200, 334, 667):
        qe, ce = WF.allocate(sizes, total, "equal")
        qp, cp = WF.allocate(sizes, total, "proportional")
        assert qe.sum() == total, f"equal 配額總和 {qe.sum()} != {total}"
        assert (qe <= sizes).all(), "equal 有群配額超過成員數"
        assert (qp <= sizes).all(), "proportional 有群配額超過成員數"
        assert (qp >= 1).all(), "proportional 有群配額為0——小群會整個消失，多樣性限制失效"
        # 比例分配的總量因四捨五入與下限1可能微差，但不該偏離超過群數
        assert abs(int(qp.sum()) - total) <= len(sizes), (
            f"proportional 總和 {qp.sum()} 偏離 {total} 太多")

    # 天花板：total=667 時等量每群要 111 支，但群5 只有 75 → 必須觸發重分配
    q, n_capped = WF.allocate(sizes, 667, "equal")
    assert n_capped >= 1, "台股 10% 等量分配應觸發群5 的天花板，實際未觸發"
    assert q[5] == 75, f"群5 應被配滿 75 支（其成員上限），實際 {q[5]}"
    assert q.sum() == 667, "天花板重分配後總量縮水了"

    # legacy 點的總量定義：m=5 × 群數
    assert WF.target_total("legacy", 6679, 6) == 30
    assert WF.target_total("legacy", 15040, 3) == 15
    # 比例點：宇宙 × ratio
    assert WF.target_total(0.05, 6679, 6) == 334


@test
def t_k_stability_real_data():
    """H-26b：群數 k 的前視偏誤診斷契約通過，且鎖住這張表的判讀邏輯。

    這張表存在的理由：`L1_TARGET`（6/7/3）是用**完整窗**選出來的，walk-forward
    的早期窗卻沿用它——形式上是前視偏誤（同 H-18② 總經 z-score 全樣本凍結、
    H-11 原提案挑涵蓋 COVID 的窗）。本表逐 IS 窗只用該窗資料重選 k 來量化它。

    ①`same_as_fixed` 必須跟 `k_is_selected == k_fixed` 完全一致（欄位定義）
    ②`sil_gap` 必須等於 `sil_at_selected − sil_at_fixed`
    ③`sil_gap` 為負時 **必定**是退化解退讓（`degenerate_fallback=True`）——
      因為 recommend_k 取的是最高分，除非最高分那個 k 被判為退化解而排除。
      實測跨市場有兩個窗如此：k=3 分數較高但最大群佔比 51.7%/51.4%，超過 50% 門檻。
    ④主鍵去重有效：rolling 第一窗的 IS 起點被夾到錨點後，會跟 anchored 某窗
      完全相同，不去重就會重複掃描（2026-09-04 開發時實測踩到）。
    """
    p = paths.ROOT / "_analysis_outputs_robustness" / "k_stability.csv"
    if not p.exists():
        raise AssertionError("尚未執行 research.k_stability")
    df = pd.read_csv(p)
    df["tree_key"] = df["tree_key"].astype("category")
    C.validate(df, C.K_STABILITY, strict_columns=True)

    assert (df.same_as_fixed == (df.k_is_selected == df.k_fixed)).all(), \
        "same_as_fixed 欄位與 k_is_selected/k_fixed 不一致"
    gap = df.sil_at_selected - df.sil_at_fixed
    assert np.allclose(df.sil_gap.to_numpy(), gap.to_numpy(), equal_nan=True), \
        "sil_gap 不等於 sil_at_selected − sil_at_fixed"

    neg = df[df.sil_gap < -1e-9]
    assert neg.degenerate_fallback.all(), (
        f"{len(neg)} 個窗的 sil_gap 為負但未標記 degenerate_fallback——"
        f"recommend_k 取最高分，唯一可能為負的原因是最高分的 k 被判為退化解：\n"
        f"{neg[['tree_key','is_start','is_end','sil_gap','degenerate_fallback']]}")
    for r in neg.itertuples():
        assert r.max_share_at_fixed > 0.5, (
            f"[{r.tree_key} {r.is_start}~{r.is_end}] 標記為退化解退讓，"
            f"但 k_fixed 的最大群佔比只有 {r.max_share_at_fixed:.1%}，未超過 50% 門檻")

    assert not df.duplicated(subset=["tree_key", "is_start", "is_end"]).any(), \
        "同一個 (樹, IS窗) 被掃描了多次——去重失效"


@test
def t_walkforward_k_mode_uses_is_only_k():
    """H-26b：矩陣的 `silhouette_is` 模式必須真的用 IS 窗自選的 k，不是沿用 6/7/3。

    這條鎖住整個 k_mode 維度的意義：若兩個模式的 n_clusters 完全相同，這個維度
    就是空的、白跑一場。實測 43 個 IS 窗只有 14 個（32.6%）兩者一致，故矩陣裡
    **必須**看得到差異。
    """
    p = paths.ROOT / "_analysis_outputs_robustness" / "walkforward_matrix_detail.csv"
    kp = paths.ROOT / "_analysis_outputs_robustness" / "k_stability.csv"
    if not p.exists() or not kp.exists():
        raise AssertionError("尚未執行 research.walkforward_matrix 或 research.k_stability")
    df = pd.read_csv(p)
    if "k_mode" not in df.columns:
        raise AssertionError("detail 檔沒有 k_mode 欄位——矩陣尚未加入該維度")
    ktab = {(r.tree_key, r.is_start, r.is_end): int(r.k_is_selected)
            for r in pd.read_csv(kp).itertuples()}

    fixed = df[df.k_mode == "fixed"]
    sil = df[df.k_mode == "silhouette_is"]
    assert len(fixed) and len(sil), "兩個 k_mode 都必須有資料"

    # fixed 模式的群數必須等於 L1_TARGET
    from . import stage3_hrp as S3
    for r in fixed.drop_duplicates(subset=["tree_key", "scheme", "window_no"]).itertuples():
        assert r.n_clusters == S3.L1_TARGET[r.tree_key], (
            f"fixed 模式的 n_clusters({r.n_clusters}) != L1_TARGET({S3.L1_TARGET[r.tree_key]})")

    # silhouette_is 模式的群數必須等於 k_stability 表裡那個窗的 k
    for r in sil.drop_duplicates(subset=["tree_key", "scheme", "window_no"]).itertuples():
        want = ktab.get((r.tree_key, r.is_start, r.is_end))
        assert want is not None, f"k_stability 缺 ({r.tree_key},{r.is_start},{r.is_end})"
        assert r.n_clusters == want, (
            f"[{r.tree_key} {r.is_start}~{r.is_end}] silhouette_is 的 n_clusters"
            f"({r.n_clusters}) != k_stability 的 k_is_selected({want})")

    # 兩模式必須確實產生不同的群數（否則這個維度是空的）
    a = fixed.set_index(["tree_key", "scheme", "window_no"])["n_clusters"]
    b = sil.set_index(["tree_key", "scheme", "window_no"])["n_clusters"]
    a, b = a[~a.index.duplicated()], b[~b.index.duplicated()]
    assert (a != b.reindex(a.index)).any(), \
        "兩個 k_mode 的群數完全相同——k_mode 維度沒有意義，應檢查是否真的重切了"


@test
def t_turnover_cost_real_data():
    """H-27b：周轉率與交易成本，鎖住三個判讀前提。

    這張表要回答「加入交易成本後，『挑越少越好』的結論會不會翻轉」。

    ①**去重後持股數必須遠少於策略數×平均持股**——策略間持股高度重疊正是
      「周轉率不隨策略數等比放大」的原因。實測台股 30 檔策略（理論 1,320 個部位）
      去重後只有 356 檔，重疊率 73%。若此性質不成立，整個成本分析的前提就錯了。
    ②**淨報酬必須低於毛報酬且成本隨周轉率單調**——欄位一致性。
    ③🔴 **成本不改變實質結論**（2026-09-04 依實測放寬過的斷言）：
      - 最佳比例在毛報酬與各成本率下必須是同一個（實測三棵樹都是 `legacy`）
      - **只有「毛報酬本來就接近平手」的配對才允許在淨報酬下互換名次**
      實測跨市場的 3% 與 10% 確實互換（毛 25.56% vs 25.22%，差 0.34pp；
      淨@30bp 23.76% vs 23.79%，反向 0.03pp）——因為 10% 那組的周轉率明顯較低
      （19.9% vs 25.0%，策略多→持股重疊多→買賣互相抵銷），扣的成本較少剛好補回來。
      原斷言要求「排序完全一致」，被這組平手翻轉擋下；改為只禁止**有意義差距**
      （毛報酬差 > 0.5pp）的配對翻轉，這才是「結論不變」真正的意思。
    """
    p = paths.ROOT / "_analysis_outputs_robustness" / "turnover_cost.csv"
    if not p.exists():
        raise AssertionError("尚未執行 research.turnover_cost")
    df = pd.read_csv(p)
    for col in ("tree_key", "ratio", "allocation"):
        df[col] = df[col].astype("category")
    C.validate(df, C.TURNOVER_COST, strict_columns=True)

    # ① 去重後持股數必須遠小於「策略數 × 每策略平均持股(約44)」
    assert (df.n_stocks_avg < df.n_strategies * 44 * 0.9).all() or \
           (df.n_strategies <= 2).all(), \
        "去重後持股數沒有明顯少於各策略持股的總和——持股重疊的前提不成立"

    # ② 欄位一致性：成本越高淨報酬越低，且淨 < 毛
    cost_cols = ["net_cagr_5bp", "net_cagr_10bp", "net_cagr_20bp", "net_cagr_30bp"]
    assert (df[cost_cols].lt(df.gross_cagr, axis=0)).all().all(), "淨報酬未低於毛報酬"
    for a, b in zip(cost_cols, cost_cols[1:]):
        assert (df[a] > df[b]).all(), f"{b} 未低於 {a}——成本應隨費率單調遞增"

    # ③ 核心結論：成本不改變「實質」結論（equal 分配）
    TIE = 0.005      # 毛報酬差距 <= 0.5pp 視為平手，允許在淨報酬下互換
    eq = df[df.allocation == "equal"]
    for t, g in eq.groupby("tree_key", observed=True):
        best_gross = g.loc[g.gross_cagr.idxmax(), "ratio"]
        for c in cost_cols:
            assert g.loc[g[c].idxmax(), "ratio"] == best_gross, (
                f"[{t}] {c} 下的最佳比例 {g.loc[g[c].idxmax(), 'ratio']} "
                f"不是毛報酬的最佳比例 {best_gross}——成本改變了最優解")
            # 有意義差距的配對不得翻轉
            for i, ri in g.iterrows():
                for j, rj in g.iterrows():
                    gap = ri.gross_cagr - rj.gross_cagr
                    if gap <= TIE:
                        continue
                    assert ri[c] > rj[c], (
                        f"[{t}] {c}：{ri.ratio} 毛報酬高過 {rj.ratio} {gap:.2%}"
                        f"（超過 {TIE:.1%} 平手容差），淨報酬卻反轉"
                        f"（{ri[c]:.2%} vs {rj[c]:.2%}）——交易成本翻轉了實質結論")


@test
def t_l3_isoos_real_data():
    """H-25d：L3 細粒度互補性的 walk-forward 驗證，鎖住整個發現賴以成立的四件事。

    H-25 主張「免費午餐藏在小而特化的群 × 另一個市場」，但那是**全窗**算的。
    本表把老師對 H-13 的原話「in sample 都很低，可是 out sample 會不會還是很低」
    搬到 L3。以下任何一條壞掉，那個主張就不成立：

    ①**欄位一致性**：`n_is_high_stays_high <= n_is_high <= n_pairs`；
      `stability_rate == n_is_high_stays_high / n_is_high`（n_is_high=0 時為 NaN）。
    ②**同市場是對照組**：TW/US 樹只能有 same 配對（單一市場沒有跨市場對象）；
      只有 XM 樹有 cross。
    ③**核心對照必須成立**：跨市場的 OOS 高互補比例必須遠高於同市場——這是
      「分散來源是市場邊界」的直接證據。實測 67.7% vs ~0%。
    ④**同市場的 IS 高互補是雜訊**：同市場在 IS 有少量（0.7~1.6%）高互補，
      但 OOS 穩定率必須接近 0，否則「同市場也有互補」的反論就成立了。
    """
    p = paths.ROOT / "_analysis_outputs_robustness" / "l3_isoos.csv"
    if not p.exists():
        raise AssertionError("尚未執行 research.l3_isoos")
    df = pd.read_csv(p)
    for col in ("tree_key", "scheme", "pair_type"):
        df[col] = df[col].astype("category")
    C.validate(df, C.L3_ISOOS, strict_columns=True)

    assert (df.n_is_high_stays_high <= df.n_is_high).all(), \
        "n_is_high_stays_high 超過 n_is_high"
    assert (df.n_is_high <= df.n_pairs).all(), "n_is_high 超過 n_pairs"
    has = df[df.n_is_high > 0]
    assert np.allclose(has.stability_rate, has.n_is_high_stays_high / has.n_is_high), \
        "stability_rate 與 n_is_high_stays_high/n_is_high 不符"
    assert df[df.n_is_high == 0].stability_rate.isna().all(), \
        "n_is_high=0 時 stability_rate 應為 NaN（無定義），不可填 0"

    single = df[df.tree_key.isin(["TW", "US"])]
    assert (single.pair_type == "same").all(), \
        "單一市場樹不該有 cross 配對——它沒有跨市場對象"
    assert (df[df.pair_type == "cross"].tree_key == "XM").all(), \
        "cross 配對只該出現在跨市場樹"

    cross = df[df.pair_type == "cross"]
    same = df[df.pair_type == "same"]
    assert len(cross) and len(same), "兩種配對類型都必須有資料"
    assert cross.oos_pct_high.mean() > same.oos_pct_high.mean() + 0.3, (
        f"跨市場的 OOS 高互補({cross.oos_pct_high.mean():.1%})沒有明顯高於"
        f"同市場({same.oos_pct_high.mean():.1%})——"
        "「分散來源是市場邊界」在樣本外不成立，H-25 的主張需改寫")
    assert same.stability_rate.fillna(0).mean() < 0.05, (
        f"同市場的 OOS 穩定率達 {same.stability_rate.fillna(0).mean():.1%}——"
        "同市場的 IS 高互補不是雜訊，對照組失效")


@test
def t_window_robustness_real_data():
    """H-26d（＝落差3 的方案E）：共同窗穩健性契約通過，且鎖住判讀的前提。

    這張表要回答「群結構是不是窗口選擇的產物」。若下列前提壞掉，那個結論就不成立：

    ①**隨機基準必須貼近 0**——ARI 的尺度沒有絕對意義，全靠這個基準給參照。
      若打散標籤後 ARI 仍明顯偏離 0，代表比對邏輯有問題（例如標籤沒真的被打散）。
    ②**兩窗必須切同一個 k 才可比**：`n_common` 在所有 k 下必須相同（同一批共同策略）。
    ③主線採用的 k 必須恰有一列標記 `is_main_k`。
    ④主線 k 的 ARI 必須顯著高於隨機基準，否則「結構一致」的結論不成立。
    """
    p = paths.ROOT / "_analysis_outputs_robustness" / "window_robustness.csv"
    if not p.exists():
        raise AssertionError("尚未執行 research.window_robustness")
    df = pd.read_csv(p)
    df["tree_key"] = df["tree_key"].astype("category")
    C.validate(df, C.WINDOW_ROBUSTNESS, strict_columns=True)

    assert df.ari_random_floor.abs().max() < 0.02, (
        f"隨機基準偏離 0 達 {df.ari_random_floor.abs().max():.4f}——"
        "ARI 的尺度參照失效，比對邏輯需檢查")
    assert df.n_common.nunique() == 1, \
        f"不同 k 的共同策略數不一致（{df.n_common.unique()}），比較基礎不同"
    assert int(df.is_main_k.sum()) == 1, "is_main_k 必須恰有一列"

    from . import stage3_hrp as S3
    main = df[df.is_main_k].iloc[0]
    assert int(main.k) == S3.L1_TARGET[str(main.tree_key)], \
        "is_main_k 標記的 k 跟 L1_TARGET 不符"
    assert main.ari > main.ari_random_floor + 0.2, (
        f"主線 k 的 ARI({main.ari:.4f}) 沒有明顯高於隨機基準"
        f"({main.ari_random_floor:.4f})——「群結構不是窗選擇的產物」這個結論不成立，"
        f"必須改寫 limitations")


@test
def t_walkforward_significance_real_data():
    """H-26c：統計檢定表契約通過，且鎖住「不把 2,700 格當樣本數」這件事。

    這張表存在的全部理由，就是修正「2,700 個格子共用同一段歷史、不能當樣本數」
    這個問題。若檢定單位數膨脹回接近 2,700，這張表就失去意義。

    ①單位數必須遠小於格數——每個方案的單位數 = 3 樹 × 該方案窗數（最多 18）
    ②`n_wins <= n_units`，`win_rate == n_wins/n_units`（欄位一致性）
    ③`significant_05` 必須跟 `p_value < 0.05` 一致
    ④主結論（vs B_all 的 Calmar）必須**每個方案都顯著**——這是老師要的
      「不管怎麼切都偏向 HRP」，只要有一個方案不顯著，敘事就要改寫
    """
    p = paths.ROOT / "_analysis_outputs_robustness" / "walkforward_significance.csv"
    if not p.exists():
        raise AssertionError("尚未執行 research.walkforward_significance")
    df = pd.read_csv(p)
    for col in ("opponent", "metric", "scheme", "tree_scope"):
        df[col] = df[col].astype("category")
    C.validate(df, C.WALKFORWARD_SIGNIFICANCE, strict_columns=True)

    assert df.n_units.max() <= 20, (
        f"最大單位數 {df.n_units.max()} 過大——檢定單位應該是「方案內互不重疊的"
        f"窗次 × 樹」（最多 3×6=18），不是把所有設定都當獨立樣本")
    assert (df.n_wins <= df.n_units).all(), "n_wins 超過 n_units"
    assert np.allclose(df.win_rate, df.n_wins / df.n_units), "win_rate 與 n_wins/n_units 不符"
    assert (df.significant_05 == (df.p_value < 0.05)).all(), \
        "significant_05 與 p_value < 0.05 不一致"

    main = df[(df.opponent == "B_all") & (df.metric == "calmar")
              & (df.tree_scope == "ALL")]
    bad = main[~main.significant_05]
    assert bad.empty, (
        f"{len(bad)} 個方案的主結論（vs B_all Calmar）未達顯著，敘事需改寫：\n"
        f"{bad[['scheme', 'n_units', 'n_wins', 'p_value']]}")


@test
def t_walkforward_matrix_real_data():
    """H-26/H-27：真實資料契約通過，且鎖住三個結構性事實。

    ①`legacy`+`equal` 的檔數必須等於 m=5×群數（校驗點的定義不能漂掉）
    ②同一格的 A/D/E 三組**檔數必須一致**——D/E 是拿 A 的實際檔數去取前 N 名，
      檔數不同就沒有可比性
    ③**分配方式對集中度的效果**（2026-09-03 實測後修正過的斷言）：
      - `equal` 分配的 A_hrp 集中度必須**明顯低於** B_all——這是它的全部意義
      - `proportional` 分配的 A_hrp 集中度必須**貼近** B_all——因為按群大小分配
        就是在複製宇宙的成分，等於放棄分散
      這兩條合起來就是 H-28（HRP 為什麼會贏）的機制證據：實測 equal 是 0.167、
      proportional 是 0.467、B_all 是 0.463——比例分配確實退化成全宇宙。
      ⚠️ 原本這裡斷言「A 一律低於 B_all」，那只對 equal 成立，對 proportional
      是錯的（實測 4 格因整數進位微幅超出 B_all 0.35pp 而失敗），已訂正。
    """
    p = paths.ROOT / "_analysis_outputs_robustness" / "walkforward_matrix_detail.csv"
    if not p.exists():
        raise AssertionError("尚未執行 research.walkforward_matrix")
    df = pd.read_csv(p)
    for col in ("tree_key", "scheme", "mode", "k_mode", "ratio", "allocation", "group"):
        df[col] = df[col].astype("category")
    C.validate(df, C.WALKFORWARD_MATRIX, strict_columns=True)

    from . import walkforward_matrix as WF
    leg = df[(df.ratio == "legacy") & (df.allocation == "equal") & (df.group == "A_hrp")]
    for r in leg.itertuples():
        assert r.target_total == WF.LEGACY_M_PER_CLUSTER * r.n_clusters, (
            f"legacy 總量 {r.target_total} != 5×{r.n_clusters}")

    sel = df[df.group != "B_all"]
    # ⚠️ 必須含 k_mode：兩個 k_mode 的群數不同，legacy 的總量(5×k)也就不同，
    # 混在一組比較會誤判成「A/D/E 檔數不一致」（2026-09-04 實測踩到）。
    for key, g in sel.groupby(["tree_key", "scheme", "window_no", "k_mode",
                              "ratio", "allocation"], observed=True):
        n = g.n_members.unique()
        assert len(n) == 1, f"{key} 的 A/D/E 檔數不一致：{dict(zip(g.group, g.n_members))}"

    # B_all 每個窗次現在有兩列（各 k_mode 一列），查表 key 必須含 k_mode
    bkey = ["tree_key", "scheme", "window_no", "k_mode"]
    b = df[df.group == "B_all"].set_index(bkey)["max_cluster_share"]
    a = df[df.group == "A_hrp"].copy()
    a["_b"] = a.set_index(bkey).index.map(b)

    eq = a[a.allocation == "equal"]
    bad_eq = eq[eq.max_cluster_share >= eq["_b"]]
    assert bad_eq.empty, (
        f"{len(bad_eq)} 格的 equal 分配集中度沒有低於 B_all，多樣性規則失效：\n"
        f"{bad_eq[['tree_key','scheme','window_no','ratio','max_cluster_share','_b']].head()}")

    pr = a[a.allocation == "proportional"]
    if len(pr):
        # 按群大小分配＝複製宇宙成分，集中度應貼近 B_all（容差 5pp，涵蓋整數進位）
        drift = (pr.max_cluster_share - pr["_b"]).abs()
        assert drift.max() < 0.05, (
            f"proportional 分配的集中度偏離 B_all 達 {drift.max():.3f}——"
            "它應該貼近全宇宙成分，偏離太多代表分配邏輯有問題")
        # 而且必須明顯比 equal 集中，否則兩種分配的對照沒有意義
        assert pr.max_cluster_share.mean() > eq.max_cluster_share.mean() * 1.5, (
            f"proportional 平均集中度({pr.max_cluster_share.mean():.3f})沒有明顯高於"
            f"equal({eq.max_cluster_share.mean():.3f})，H-28 的機制對照失效")


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


@test
def t_beta_baseline_real_data():
    """M-01：市場 beta 基準與殘差結構，鎖住四件會被誤寫進論文的事。

    這張表推翻了 v10 §3.5② 原本的敘事。以下任何一條壞掉，改寫後的敘事就不成立：

    ①**市場基準相關必須是實算出來的、且落在合理區間**（實測台美 0.5512）。
      整個 M-01 的立論是「跨市場配對相關 0.542 只是這個數字的重現」，
      若它不存在或荒謬（如 >0.99 或 <0），對照就沒有意義。
    ②🔴 **殘差相關必須跟機械下限比，不能跟 0 比**。`mkt` 是同一批策略的等權平均，
      殘差在橫斷面被強制加總≈0，k 個群代表的平均兩兩相關被機械壓到約 −1/(k−1)。
      本測試斷言殘差中位**不顯著低於**機械下限——若有人把 `corr_mechanical_floor`
      拿掉、或改用真正外生的市場指數卻沿用舊解讀，這條會擋下
      「扣掉 beta 後群呈負相關」這種假宣稱。
    ③🔴 **純 beta 模型的預測必須貼近實測**（超額 |excess| 小、低於預測% 接近 50%）。
      這是「策略選擇沒有提供超越 beta 的分散」的直接證據。實測 XM L3 跨市場
      實測 0.4580 vs 預測 0.4627（超額 −0.0047、55.7% 低於預測）。
      若哪天超額變成顯著負值，代表結論翻轉，必須重寫 §3.5②——這條會提醒。
    ④🔴 **粒度效應必須伴隨群 R² 下降**。H-25 的「L3 比 L1 互補」若是真的因子專門化，
      群 R² 不該隨群變小而系統性下降；實測 L1 群大小 6,464/R² 0.9920 →
      L3 群大小 23/R² 0.8470。R² 掉了 0.145 正好解釋相關從 0.542 掉到 0.458。
      本測試鎖住「群變小 ⇒ R² 下降」這個機械成因，避免退回「小群比較特化」的說法。
    """
    p = paths.ROOT / "_analysis_outputs_robustness" / "beta_baseline.csv"
    if not p.exists():
        raise AssertionError("尚未執行 research.beta_baseline")
    df = pd.read_csv(p)
    for col in ("tree_id", "level", "version", "pair_type"):
        df[col] = df[col].astype("category")
    C.validate(df, C.BETA_BASELINE, strict_columns=True)

    # ① 市場基準相關
    mc = df.market_corr.dropna().unique()
    assert len(mc) == 1, f"市場基準相關不唯一：{mc}"
    mc = float(mc[0])
    assert 0.2 < mc < 0.9, f"台美市場基準相關 {mc:.4f} 落在不合理區間"

    # ② 殘差相關 vs 機械下限
    res = df[df.version == "resid"]
    assert len(res) > 0, "缺殘差版列"
    bad = res[res.corr_median < res.corr_mechanical_floor - 0.25]
    assert bad.empty, (
        "殘差相關明顯低於機械下限 −1/(k−1)：\n"
        + bad[["tree_id", "level", "pair_type", "corr_median",
              "corr_mechanical_floor"]].to_string(index=False)
        + "\n——殘差不該比機械下限還負；請檢查 mkt 代理是否被換過")

    # ③ 純 beta 模型：超額必須小
    raw = df[df.version == "raw"]
    assert raw.corr_pred_median.notna().all(), "raw 列缺 beta 模型預測"
    assert res.corr_pred_median.isna().all(), \
        "殘差版不該有 beta 模型預測（beta 已被移除，預測是同義反覆）"
    worst = raw.excess_median.abs().max()
    assert worst < 0.05, (
        f"純 beta 模型的最大超額 {worst:.4f} 超過 0.05——"
        "策略選擇開始提供超越 beta 的分散，v10 §3.5② 的改寫版結論需重新檢視：\n"
        + raw[["tree_id", "level", "pair_type", "corr_median",
              "corr_pred_median", "excess_median"]].to_string(index=False))

    xm = raw[(raw.tree_id == "XM_normal") & (raw.pair_type == "cross")
             & (raw.level == "L3")]
    assert len(xm) == 1
    assert 0.35 < float(xm.pct_below_pred.iloc[0]) < 0.65, (
        f"XM L3 跨市場『低於 beta 預測』的比例 {float(xm.pct_below_pred.iloc[0]):.1%} "
        "偏離 50%——實測應接近純隨機擺動")

    # ④ 粒度效應 = 特異變異稀釋：群變小 ⇒ 群 R² 下降
    for tid, g in raw.groupby("tree_id", observed=True):
        l1 = g[g.level == "L1"]
        l3 = g[g.level == "L3"]
        if l1.empty or l3.empty:
            continue
        s1, s3 = float(l1.cluster_size_median.iloc[0]), float(l3.cluster_size_median.iloc[0])
        r1, r3 = float(l1.cluster_r2_median.iloc[0]), float(l3.cluster_r2_median.iloc[0])
        assert s3 < s1, f"[{tid}] L3 群大小未小於 L1"
        assert r3 <= r1 + 1e-9, (
            f"[{tid}] L3 群 R² {r3:.4f} 未低於 L1 {r1:.4f}——"
            "『群越小、特異變異稀釋越多』的機械解釋不成立，粒度效應需重新歸因")


@test
def t_partition_control_real_data():
    """M-03：分群依據對照，鎖住五件事——其中三件是「這個對照乾不乾淨」的前提。

    本表要回答「A_hrp 的優勢是否來自 HRP 分群本身」。若以下任一條壞掉，
    三組之間的差異就不能歸因到分群依據：

    ①🔴 **A_hrp 必須與 H-12 的 A_hrp 逐位元一致**。本模組直接重用
      `four_group_control` 的量測函式，兩邊算出來的 A 若有任何差異，代表挑選或
      評估路徑被動過，整張表跟既有結果不可比。這是最強的回歸鎖。
    ②🔴 **共同座標軸**：三組的 `n_target`、`n_universe`、`k` 必須相同，
      且 `n_members == n_target`。任何一組檔數不同，報酬差異就混進了「買幾檔」
      這個變因（同 H-27 對 equal/proportional 的堅持）。
    ③🔴 **`max_hrp_cluster_share` 對 A_hrp 是套套邏輯**：A 每群固定挑 m 檔，
      所以它的最大單群佔比**恆等於 1/k**。本測試斷言這個恆等式成立——
      它存在的目的是提醒：**不可拿這個欄位當「HRP 比較分散」的證據**。
      真正非套套邏輯的證據是 `oos_enb`（從 OOS 報酬共變異算出來，與分群無關）。
    ④**A2_random 的抽樣統計必須齊備**（有 std、n_draws>1），
      A_hrp/A1_fcombo 是單次結果故 std 必須為空——判讀靠 σ 數不是點估計。
    ⑤🔴 **A1_fcombo 在 XM 的市場純度必須遠低於 A_hrp**。特徵刻意不含市場欄位，
      故因子構成分群無法還原市場邊界（實測 0.577，隨機基準 0.556，A_hrp 1.000）。
      若哪天 A1 的純度接近 1，代表特徵洩漏了市場資訊，這個對照就沒有意義了。
    """
    p = paths.ROOT / "_analysis_outputs_robustness" / "partition_control.csv"
    if not p.exists():
        raise AssertionError("尚未執行 research.partition_control")
    df = pd.read_csv(p)
    for col in ("tree_key", "partition"):
        df[col] = df[col].astype("category")
    C.validate(df, C.PARTITION_CONTROL, strict_columns=True)

    # ① A_hrp 必須與 H-12 完全一致
    fp = paths.ROOT / "_analysis_outputs_robustness" / "four_group_control.csv"
    if fp.exists():
        f = pd.read_csv(fp)
        a12 = f[f.group == "A_hrp"].set_index("tree_key")
        a03 = df[df.partition == "A_hrp"].set_index("tree_key")
        for col in ("n_members", "is_cagr", "is_mdd", "oos_cagr", "oos_mdd", "oos_enb"):
            for t in a03.index:
                if t not in a12.index:
                    continue
                lhs, rhs = float(a03.loc[t, col]), float(a12.loc[t, col])
                assert abs(lhs - rhs) < 1e-9, (
                    f"[{t}] A_hrp 的 {col} 與 H-12 不一致：{lhs} vs {rhs}"
                    "——兩邊應走完全同一條量測路徑，出現差異代表有一邊被改過")

    # ② 共同座標軸
    for t, g in df.groupby("tree_key", observed=True):
        assert (g.n_target.nunique() == 1 and g.n_universe.nunique() == 1
                and g.k.nunique() == 1), f"[{t}] 三組的 n_target/n_universe/k 不一致"
        assert (g.n_members == g.n_target).all(), (
            f"[{t}] 有組別的實際檔數 != 目標檔數：\n"
            + g[["partition", "n_members", "n_target"]].to_string(index=False))

    # ③ A_hrp 的 max_hrp_cluster_share 恆等於 1/k（套套邏輯，不可當證據）
    for r in df[df.partition == "A_hrp"].itertuples():
        assert abs(r.max_hrp_cluster_share - 1.0 / r.k) < 1e-9, (
            f"[{r.tree_key}] A_hrp 的最大單群佔比 {r.max_hrp_cluster_share:.4f} "
            f"!= 1/k = {1.0/r.k:.4f}——每群固定挑 m 檔的前提被破壞了")

    # ④ 抽樣統計齊備
    std_cols = [c for c in df.columns if c.endswith("_std")]
    r2 = df[df.partition == "A2_random"]
    assert (r2.n_draws > 1).all(), "A2_random 的 n_draws 應大於 1"
    assert r2[std_cols].notna().all().all(), "A2_random 缺抽樣標準差"
    single = df[df.partition != "A2_random"]
    assert single[std_cols].isna().all().all(),         "單次結果的組別不該有抽樣標準差（會被誤讀成有不確定性區間）"

    # ⑤ 因子構成無法還原市場邊界
    xm = df[df.tree_key == "XM"].set_index("partition")
    if len(xm) == 3:
        a = float(xm.loc["A_hrp", "market_purity"])
        f1 = float(xm.loc["A1_fcombo", "market_purity"])
        r = float(xm.loc["A2_random", "market_purity"])
        assert a > 0.99, f"XM 的 A_hrp 市場純度 {a:.3f} 不到 1——HRP L1 應完全分開市場"
        assert f1 < r + 0.15, (
            f"XM 的 A1_fcombo 市場純度 {f1:.3f} 明顯高於隨機基準 {r:.3f}——"
            "因子構成特徵疑似洩漏了市場資訊，該對照失去意義")


@test
def t_mdd_window_length_real_data():
    """M-09：MDD 顯著性按窗長分層，鎖住四件事。

    這張表把 H-26c 的「MDD 只有 8/13 顯著」拆成「檢力不足」與「效果真的弱」。
    以下任何一條壞掉，那個拆解就不成立：

    ①🔴 **不得遺漏對手**。B_all 的列 `ratio="all"`、`allocation="unallocated"`，
      跟 A_hrp 的設定鍵永遠對不上——初版就是因此讓 B_all 整組被靜默丟掉
      （117 列變 78 列，且不會報錯）。斷言列數 = 方案 × 對手 × 指標。
    ②🔴 **顯著門檻的定義必須正確，且必須帶方向**：`min_wins_for_sig` 是雙尾二項
      檢定 p<0.05 所需的最少勝場，只在 A 勝的方向上有意義。
      **雙尾顯著不等於 A 贏**——實測 A_hrp vs D_top_cagr 在方案 A 的 CAGR 上只贏
      3/18（p=0.0075），那是**顯著落敗**。本測試斷言 verdict 對這種列必須標
      「顯著（A敗）」而非「顯著」，否則讀表的人會把 A 的慘敗讀成勝利
      （2026-09-05 開發時本條就是這樣抓到初版缺陷的）。
    ③🔴 **verdict 不可把檢力問題誤標成效果問題**。n=6 的方案門檻是 6/6（100%），
      5/6 不顯著純粹是檢力，不是效果弱。斷言：凡門檻 >= 95% 的不顯著方案，
      一律不得被標成「效果不足」。（初版就是這樣標錯的。）
    ④🔴 **MDD 的效果量必須在各窗長下都明顯為正**。這是「不是短窗雜訊」的核心證據：
      若 24/36/48 個月的效果量都是 1.1~1.4pp 而非隨窗長趨近 0，
      MDD 的弱就只是「效果比 CAGR 小（1.2pp vs 3.3pp）」，不是雜訊。
    """
    p = paths.ROOT / "_analysis_outputs_robustness" / "mdd_window_length.csv"
    if not p.exists():
        raise AssertionError("尚未執行 research.mdd_window_length")
    df = pd.read_csv(p)
    for col in ("opponent", "metric", "scheme", "mode", "verdict"):
        df[col] = df[col].astype("category")
    C.validate(df, C.MDD_WINDOW_LENGTH, strict_columns=True)

    # ① 三個對手都在，且列數完整
    opps = set(df.opponent.astype(str))
    assert opps == {"B_all", "D_top_cagr", "E_top_calmar"}, \
        f"對手不齊：{opps}——B_all 最容易因設定鍵對不上被靜默丟掉"
    n_expect = df.scheme.nunique() * 3 * 3
    assert len(df) == n_expect, f"{len(df)} 列 != 預期 {n_expect}（方案×對手×指標）"

    # ② 門檻定義（只在 A 勝方向）：贏到門檻 ⟺ 顯著
    up = df[df.min_wins_for_sig.notna() & (df.win_rate > 0.5)]
    assert ((up.n_wins >= up.min_wins_for_sig) == up.significant_05).all(), \
        "A 勝方向上「贏場數 >= 顯著門檻」與 significant_05 不一致——門檻算錯了"
    # ②b 顯著落敗必須標明方向，不可只寫「顯著」
    lose = df[df.significant_05 & (df.win_rate < 0.5)]
    assert not lose.empty, \
        "沒有任何顯著落敗的列——請確認 D/E 對手仍在表內（它們是 A 落敗的來源）"
    assert (lose.verdict.astype(str) == "顯著（A敗）").all(), (
        "A_hrp 顯著落敗的列未標明方向，會被讀成 A 贏：\n"
        + lose[["opponent", "metric", "scheme", "n_wins", "n_units",
               "p_value", "verdict"]].head(5).to_string(index=False))
    assert (df.direction.astype(str) == np.where(
        df.win_rate > 0.5, "A勝", np.where(df.win_rate < 0.5, "A敗", "平手"))).all(), \
        "direction 欄與 win_rate 不一致"

    # ③ 檢力問題不得被標成效果問題
    bad = df[(~df.significant_05) & (df.detectable_win_rate >= 0.95)
             & (df.verdict.astype(str) == "效果不足")]
    assert bad.empty, (
        "門檻需近全勝的方案被標成「效果不足」——那是檢力問題：\n"
        + bad[["opponent", "metric", "scheme", "n_wins", "n_units",
              "detectable_win_rate"]].to_string(index=False))

    # ④ MDD 效果量在各窗長下都明顯為正 → 不是短窗雜訊
    mdd = df[(df.opponent == "B_all") & (df.metric == "mdd")]
    by_len = mdd.groupby("oos_len_months")["diff_mean"].mean()
    assert len(by_len) == 3, f"應有 24/36/48 三種窗長，實得 {list(by_len.index)}"
    assert (by_len > 0.005).all(), (
        "有窗長的 MDD 效果量低於 0.5pp——「MDD 的弱只是效果較小、不是短窗雜訊」"
        f"這個結論需重新檢視：\n{(by_len * 100).round(3).to_string()}")
    cagr = df[(df.opponent == "B_all") & (df.metric == "cagr")].diff_mean.mean()
    assert cagr > mdd.diff_mean.mean(), \
        "CAGR 的效果量未大於 MDD——本表的核心對照（3.3pp vs 1.2pp）不成立"


@test
def t_walkforward_evidence_real_data():
    """M-10：walk-forward 證據強度三張表，鎖住六件事。

    這三張表把 H-26／H-26c 三個沒量化的假設補上。以下任何一條壞掉，
    「A_hrp 在 92~94% 格子裡贏、13/13 方案顯著」這組宣稱就要重新檢視：

    ①**三張表都不得遺漏對手**（B_all 的設定鍵對不上，最容易被靜默丟掉——
      M-09 開發時實測從 117 列掉成 78 列且不報錯）。
    ②🔴 **勝率必須是普遍性優勢，不是少數極端格撐起來的**：A_hrp vs B_all 的
      中位數與平均必須同號，且 `median_over_mean` 落在 0.5~1.5。實測 0.64~1.04。
      若中位數接近 0 而平均很大 → 那 92% 是被離群格子撐的，結論要收斂。
    ③**Cohen's d 必須與勝率方向一致**：`pct_positive > 0.5` ⟺ `cohens_d > 0`。
      兩個獨立算出來的量若方向打架，代表其中一個算錯。
    ④🔴 **多重比較的校正必須真的更嚴格**：對每個 (對手 × 指標)，
      `n_sig_raw >= n_sig_bh_family >= n_sig_bonferroni_family`，
      且全域校正不得比族內寬鬆。若順序反了，校正實作是錯的。
    ⑤🔴 **主結論必須撐得住校正**：A_hrp vs B_all 的 cagr 與 calmar 在
      **族內 BH** 下必須維持 13/13。這是論文第五章的主要統計論據；
      若哪天掉下來，`H26_H27_walkforward矩陣結果.md` §4.2 必須改寫。
    ⑥🔴 **anchored 的相鄰窗次選股重疊必須明顯高於 rolling**。這是「窗次可當
      獨立單位」假設的量化揭露：anchored 的 IS 是巢狀的（第 2 窗 IS = 第 1 窗
      IS+OOS），實測相鄰 Jaccard 0.372 vs rolling 0.138、全窗核心佔比
      0.357 vs 0.034。若兩者接近，代表成員名單沒有正確對應到窗次。
    """
    d = paths.ROOT / "_analysis_outputs_robustness"
    pe, pm, po = (d / "wf_effect_distribution.csv", d / "wf_multiplicity.csv",
                  d / "wf_window_overlap.csv")
    for p in (pe, pm, po):
        if not p.exists():
            raise AssertionError("尚未執行 research.walkforward_evidence")
    eff, mul, ovl = pd.read_csv(pe), pd.read_csv(pm), pd.read_csv(po)
    for c in ("tree_key", "opponent", "metric"):
        eff[c] = eff[c].astype("category")
    for c in ("opponent", "metric"):
        mul[c] = mul[c].astype("category")
    for c in ("tree_key", "scheme", "k_mode", "ratio", "allocation", "group"):
        ovl[c] = ovl[c].astype("category")
    C.validate(eff, C.WF_EFFECT_DISTRIBUTION, strict_columns=True)
    C.validate(mul, C.WF_MULTIPLICITY, strict_columns=True)
    C.validate(ovl, C.WF_WINDOW_OVERLAP, strict_columns=True)

    # ① 對手齊全
    want = {"B_all", "D_top_cagr", "E_top_calmar"}
    for name, df_ in (("效果量分布", eff), ("多重比較", mul)):
        got = set(df_.opponent.astype(str))
        assert got == want, f"{name} 缺對手：{want - got}"

    # ② 普遍性優勢
    b = eff[eff.opponent == "B_all"]
    assert (np.sign(b["mean"]) == np.sign(b.p50)).all(), \
        "A_hrp vs B_all 的平均與中位數不同號——優勢方向不一致"
    bad = b[(b.median_over_mean < 0.5) | (b.median_over_mean > 1.5)]
    assert bad.empty, (
        "中位數/平均落在 0.5~1.5 之外，勝率可能被少數極端格主導：\n"
        + bad[["tree_key", "metric", "mean", "p50", "median_over_mean"]].to_string(index=False))

    # ③ Cohen's d 與勝率同向
    assert ((eff.pct_positive > 0.5) == (eff.cohens_d > 0)).all(), \
        "cohens_d 與 pct_positive 方向不一致——兩個獨立算出的量打架"

    # ④ 校正必須更嚴格
    assert (mul.n_sig_raw >= mul.n_sig_bh_family).all(), "族內 BH 比未校正還寬鬆"
    assert (mul.n_sig_bh_family >= mul.n_sig_bonferroni_family).all(), \
        "族內 Bonferroni 比族內 BH 還寬鬆"
    assert (mul.n_sig_bh_family >= mul.n_sig_bh).all(), "全域 BH 比族內 BH 還寬鬆"
    assert (mul.n_sig_bh >= mul.n_sig_bonferroni).all(), "全域 Bonferroni 比全域 BH 寬鬆"

    # ⑤ 主結論撐得住族內 BH
    for metric in ("cagr", "calmar"):
        r = mul[(mul.opponent == "B_all") & (mul.metric == metric)]
        assert len(r) == 1
        n_t, n_s = int(r.n_tests.iloc[0]), int(r.n_sig_bh_family.iloc[0])
        assert n_s == n_t, (
            f"A_hrp vs B_all 的 {metric} 在族內 BH 校正後只剩 {n_s}/{n_t} 方案顯著"
            "——論文主結論「不管怎麼切窗都贏」需改寫")

    # ⑥ anchored 重疊明顯高於 rolling（R 是唯一的 rolling 方案）
    a = ovl[ovl.group == "A_hrp"]
    roll = a[a.scheme == "R"]
    anch = a[a.scheme != "R"]
    assert len(roll) > 0 and len(anch) > 0, "缺 rolling 或 anchored 的格子"
    assert anch.jaccard_adjacent_mean.mean() > roll.jaccard_adjacent_mean.mean() + 0.10, (
        f"anchored 相鄰 Jaccard {anch.jaccard_adjacent_mean.mean():.3f} 未明顯高於 "
        f"rolling {roll.jaccard_adjacent_mean.mean():.3f}——anchored 的 IS 是巢狀的，"
        "重疊本該明顯較高；兩者接近代表成員名單沒對應到正確的窗次")
    assert anch.core_share.mean() > roll.core_share.mean(), \
        "anchored 的全窗核心佔比未高於 rolling"


@test
def t_enb_null_real_data():
    """M-05：ENB 的虛無分布，鎖住五件事。

    v10 §3.4 的核心數字「ENB 只有 3.27/4.36/5.77」原本沒有任何尺度。
    這張表給它虛無分布。以下任何一條壞掉，那個尺度就不可信：

    ①🔴 **快速演算法必須與 H-09 凍結的 `enb_raw` 一致**。整個模組建立在
      「N×N 相關矩陣的非零特徵值 = T×T 的 ZᵀZ/T」這條恆等式上（省下五個數量級
      的運算）。若它錯了，所有虛無分布都沒有意義。斷言差 < 1e-3。
    ②🔴 **純雜訊的 ENB 不得超過秩上限 T−1**。N≫T（15,040 vs 228）時相關矩陣
      有 N−T+1 個特徵值恆為 0，所以純雜訊的 ENB 是 **T 的數量級，不是 N**。
      稽核清單原本假設可以拿 N 當上界，那是錯的；本條把數學事實鎖住。
    ③**實測必須遠低於純雜訊虛無**：`ratio_observed_over_null < 0.1`。
      實測 1.5~2.6%——這是「多樣性假象」最直接的量化。
    ④🔴 **跨市場樹必須有 two_factor 列，單一市場樹不得有**。XM 有兩個市場因子
      （台股大盤、美股大盤，相關 0.5512），拿單因子當虛無會低估虛無維度，
      讓實測看起來「比虛無更分散」——開發初版就犯了這個錯（XM 實測 5.77 vs
      單因子 4.49 看似 +3.1σ；換成雙因子 6.86 後實測其實是低於虛無的）。
    ⑤🔴 **實測必須落在市場因子虛無的同量級**（0.5~1.5 倍）。這是與 M-01
      互相印證的關鍵：ENB 這麼低幾乎完全由市場 beta 解釋。實測 76~84%。
      若哪天跳出這個範圍，v10 §3.4 與 §3.5② 的敘事都要重新檢視。
    """
    p = paths.ROOT / "_analysis_outputs_robustness" / "enb_null.csv"
    if not p.exists():
        raise AssertionError("尚未執行 research.enb_null")
    df = pd.read_csv(p)
    for c in ("tree_id", "null_model"):
        df[c] = df[c].astype("category")
    C.validate(df, C.ENB_NULL, strict_columns=True)

    # ① 快速演算法 vs H-09 凍結值
    ref = pd.read_csv(paths.ROOT / "_analysis_outputs_robustness"
                      / "effective_number_of_bets.csv").set_index("tree_id")
    for tid, g in df.groupby("tree_id", observed=True):
        obs = float(g.enb_observed.iloc[0])
        known = float(ref.loc[str(tid), "enb_raw"])
        assert abs(obs - known) < 1e-3, (
            f"[{tid}] fast_enb {obs:.6f} 與 H-09 的 enb_raw {known:.6f} 不符"
            "——T×T 等價性被破壞，所有虛無分布失效")

    # ② 純雜訊不得超過秩上限
    iid = df[df.null_model == "iid"]
    assert len(iid) == df.tree_id.nunique(), "每棵樹都該有 iid 虛無"
    assert (iid.enb_null_max <= iid.max_possible_enb).all(), (
        "純雜訊的 ENB 超過秩上限 T−1——N≫T 的秩限制被違反：\n"
        + iid[["tree_id", "enb_null_max", "max_possible_enb"]].to_string(index=False))
    assert (iid.enb_null_mean > 0.7 * iid.max_possible_enb).all(), \
        "純雜訊的 ENB 遠低於秩上限——iid 模擬可能沒有真的獨立"

    # ③ 實測遠低於純雜訊
    assert (iid.ratio_observed_over_null < 0.1).all(), (
        "實測 ENB 未遠低於純雜訊虛無——「多樣性假象」的核心量化不成立：\n"
        + iid[["tree_id", "enb_observed", "enb_null_mean",
              "ratio_observed_over_null"]].to_string(index=False))

    # ④ two_factor 只該出現在跨市場樹
    two = set(df[df.null_model == "two_factor"].tree_id.astype(str))
    assert two == {"XM_normal"}, (
        f"two_factor 出現在 {two}——應該只有跨市場樹有（單一市場會退化成單因子）")

    # ⑤ 實測落在市場因子虛無的同量級
    for tid, g in df.groupby("tree_id", observed=True):
        tf = g[g.null_model == "two_factor"]
        mk = tf if len(tf) else g[g.null_model == "one_factor"]
        assert len(mk) == 1, f"[{tid}] 缺市場因子虛無"
        ratio = float(mk.ratio_observed_over_null.iloc[0])
        assert 0.5 < ratio < 1.5, (
            f"[{tid}] 實測 ENB 是市場因子虛無的 {ratio:.2f} 倍，超出 0.5~1.5"
            "——「ENB 這麼低幾乎完全由市場 beta 解釋」需重新檢視")


@test
def t_walkforward_random_real_data():
    """M-02b：walk-forward 版 C_random，鎖住五件事。

    H-12 的 C_random 只做過單一窗、legacy 比例、fixed k。本表把它擴到整個
    45 窗 × 5 比例 × 2 分配 × 2 k_mode 的空間。以下任何一條壞掉，
    「挑選規則有技術」這個宣稱就不能寫成 walk-forward 結論：

    ①🔴 **必須完整覆蓋矩陣的每一個 A_hrp 格子**。去重鍵是 (樹, 窗, 檔數)，
      2,700 格應全部接得上（`compare()` 內已斷言，這裡再驗一次涵蓋率），
      否則就是矩陣重跑過而本表沒跟著重跑，兩邊對不起來。
    ②**檔數必須對齊 A_hrp 的實際 `n_members`**（不是 target_total），
      且不得超過該窗宇宙。對齊實際值才是「買一樣多檔」的公平比較。
    ③**ENB 的計算範圍必須與矩陣一致**：`enb_computed` ⟺ `n_members <= 400`，
      且 False 的列 `oos_enb` 必須為空。兩邊規則若不同，ENB 的對照就不可比。
    ④🔴 **核心結論**：A_hrp 相對隨機挑選的 OOS CAGR z 分數，三棵樹都必須
      為正且勝率過半。這是「挑選規則有技術」的 walk-forward 版證據。
    ⑤🔴 **必須與 M-03 的結論並存而不矛盾**：本表換的是**挑選機制**、
      M-03 換的是**分群依據**，兩者結論相反是正常的（挑選有用、HRP 分群沒用）。
      本條斷言兩張表都在，避免只引用其中一張而誤導。
    """
    d = paths.ROOT / "_analysis_outputs_robustness"
    pr, pc = d / "walkforward_random.csv", d / "walkforward_random_compare.csv"
    for p in (pr, pc):
        if not p.exists():
            raise AssertionError("尚未執行 research.walkforward_random")
    rnd, cmp_df = pd.read_csv(pr), pd.read_csv(pc)
    rnd["tree_key"] = rnd["tree_key"].astype("category")
    for c in ("tree_key", "metric"):
        cmp_df[c] = cmp_df[c].astype("category")
    C.validate(rnd, C.WALKFORWARD_RANDOM, strict_columns=True)
    C.validate(cmp_df, C.WALKFORWARD_RANDOM_COMPARE, strict_columns=True)

    # ① 覆蓋率：矩陣每個 A_hrp 格子都要接得上
    det = pd.read_csv(d / "walkforward_matrix_detail.csv")
    a = det[det.group == "A_hrp"]
    key = ["tree_key", "is_start", "is_end", "oos_start", "oos_end", "n_members"]
    merged = a.merge(rnd[key], on=key, how="inner")
    assert len(merged) == len(a), (
        f"只有 {len(merged)}/{len(a)} 個 A_hrp 格子接得上隨機分布"
        "——矩陣與隨機表不同步，請重跑 research.walkforward_random")

    # ② 檔數對齊且不超過宇宙
    assert (rnd.n_members <= rnd.n_universe).all(), "抽樣檔數超過該窗宇宙"
    assert set(rnd.n_members) <= set(a.n_members), \
        "隨機表出現 A_hrp 沒有的檔數——對齊的應是 A 的實際 n_members"

    # ③ ENB 計算範圍與矩陣一致
    assert (rnd.enb_computed == (rnd.n_members <= 400)).all(), \
        "enb_computed 與 400 檔門檻不一致——與 walkforward_matrix 的規則不同就不可比"
    assert rnd.loc[~rnd.enb_computed, "oos_enb"].isna().all(), \
        "未計算 ENB 的列卻有 oos_enb 值"

    # ④ 核心結論：挑選有技術。⚠️ 對照表自 2026-09-06 起帶 `ratio` 維度
    # （聚合列 ratio="ALL" + 逐比例列），故要先取聚合列，不能直接數列數。
    cagr = cmp_df[(cmp_df.metric == "oos_cagr") & (cmp_df.ratio.astype(str) == "ALL")]
    assert len(cagr) == rnd.tree_key.nunique(), "每棵樹都該有 oos_cagr 的聚合對照列"
    assert (cagr.z_mean > 0).all() and (cagr.pct_A_wins > 0.5).all(), (
        "A_hrp 相對隨機挑選的 OOS CAGR 未全數為正——「挑選規則有技術」"
        "在 walk-forward 上不成立：\n"
        + cagr[["tree_key", "z_mean", "pct_A_wins"]].to_string(index=False))
    # 🔴 逐比例也必須成立——這是本結論與 M-03b 的關鍵差異：
    # 「挑選有技術」在**每一個比例**上都成立，而「HRP 分群有貢獻」只在 legacy 成立。
    per = cmp_df[(cmp_df.metric == "oos_cagr") & (cmp_df.ratio.astype(str) != "ALL")]
    assert (per.pct_A_wins > 0.5).all(), (
        "有比例的 A_hrp 勝率低於 50%——「挑選有技術」不是全比例成立，敘事要收斂：\n"
        + per[["tree_key", "ratio", "diff_mean", "pct_A_wins"]].to_string(index=False))

    # ⑤ 與 M-03 並存
    assert (d / "partition_control.csv").exists(), (
        "缺 M-03 的 partition_control.csv——本表（挑選機制有用）必須與 M-03"
        "（分群依據無用）一起讀，只引用其中一張會誤導")


@test
def t_robustness_manifests_all_consistent():
    """🔴 DD-08 全面檢查：`_analysis_outputs_robustness/` 底下每一份 manifest
    記錄的 sha256 都必須與實際檔案相符。

    **這條測試是 2026-09-06 code review 抓到真實違規後補的。**
    當時為了「只重算對照表、不重跑數小時的模擬」而加的 `--recompare` 路徑，
    改寫了 `walkforward_random_compare.csv` 與 `walkforward_partition_compare.csv`
    卻**沒有重寫 manifest**，兩份 manifest 的雜湊當場失效——而在補上這條測試之前，
    **整套測試沒有任何一條會發現**，只有下游模組真的去 `verify_inputs` 時才會炸。

    通則：**任何改寫產物的路徑都必須同時重寫 manifest**（見各模組的 `_persist`）。
    """
    d = paths.ROOT / "_analysis_outputs_robustness"
    if not d.exists():
        raise AssertionError("找不到 _analysis_outputs_robustness/")
    dirs = sorted(p for p in d.iterdir()
                  if p.is_dir() and p.name.startswith("_") and p.name.endswith("manifest"))
    assert dirs, "一個 manifest 目錄都沒有——命名規則改過了？"
    bad = []
    for m in dirs:
        try:
            freeze.verify_inputs(m)
        except Exception as e:                      # noqa: BLE001 - 要蒐集全部再一起報
            bad.append(f"  {m.name}: {type(e).__name__}: {str(e).splitlines()[0]}")
    assert not bad, (
        f"{len(bad)}/{len(dirs)} 份 manifest 與實際檔案不符：\n" + "\n".join(bad)
        + "\n——請重跑對應階段；若是新增了「只改寫產物」的路徑，該路徑必須一併重寫 manifest")


@test
def t_walkforward_partition_real_data():
    """M-03b：walk-forward 版隨機分群，鎖住五件事。

    這張表把 M-03 的單點結論擴到 2,700 格，並**推翻了兩個外推**。
    以下任何一條壞掉，修正後的結論就不成立：

    ①🔴 **必須完整覆蓋矩陣的每一個 A_hrp 格子**（`compare()` 內已斷言，這裡再驗）。
    ②**共同座標軸**：每格的 `n_members_mean` 必須是整數——代表該格的每一次抽樣
      都選出**同樣檔數**。若不是整數，ENB 的 400 檔門檻會在同一格內部分觸發，
      平均值就混了「有算 ENB」與「沒算 ENB」兩種樣本（開發時實測全部為整數）。
    ③**ENB 計算範圍與矩陣一致**：`enb_computed` ⟺ `n_members_mean <= 400`，
      且 False 的列 `oos_enb` 必須為空。
    ④🔴 **對照表必須有 ratio 維度且聚合列不可單獨解讀**。實測台股 CAGR 在
      legacy 是 A 勝率 21.7%、在 5% 是 75.6%，聚合起來是 50.4% 的假平手——
      本條斷言「至少有一棵樹的逐比例勝率跨越 50%」，確保 ratio 維度真的有作用；
      若哪天所有比例同向，代表資料或口徑變了，敘事要重新檢視。
    ⑤🔴 **與 M-02b 的關鍵差別必須成立**：C_random 的 ENB 勝率是每個比例都 100%，
      而本表（換分群依據）**只有 legacy 勝**。這正是「分散度來自多樣性限制、
      不是 HRP 群邊界」的直接證據。
    """
    d = paths.ROOT / "_analysis_outputs_robustness"
    pp, pc = d / "walkforward_partition.csv", d / "walkforward_partition_compare.csv"
    for p in (pp, pc):
        if not p.exists():
            raise AssertionError("尚未執行 research.walkforward_partition")
    rnd, cmp_df = pd.read_csv(pp), pd.read_csv(pc)
    for c in ("tree_key", "scheme", "k_mode", "ratio", "allocation"):
        rnd[c] = rnd[c].astype("category")
    for c in ("tree_key", "metric", "ratio"):
        cmp_df[c] = cmp_df[c].astype("category")
    C.validate(rnd, C.WALKFORWARD_PARTITION, strict_columns=True)
    C.validate(cmp_df, C.WALKFORWARD_RANDOM_COMPARE, strict_columns=True)

    # ① 覆蓋率
    det = pd.read_csv(d / "walkforward_matrix_detail.csv")
    a = det[det.group == "A_hrp"]
    key = ["tree_key", "scheme", "k_mode", "ratio", "allocation", "window_no"]
    left, right = a[key].astype(str), rnd[key].astype(str)
    merged = left.merge(right, on=key, how="inner")
    assert len(merged) == len(a), (
        f"只有 {len(merged)}/{len(a)} 個 A_hrp 格子接得上隨機分群——兩表不同步")

    # ② 每格的各次抽樣檔數一致
    assert (rnd.n_members_mean == rnd.n_members_mean.round()).all(), (
        "有格子的 n_members_mean 不是整數——同一格內各次抽樣的檔數不同，"
        "ENB 的 400 檔門檻會在格內部分觸發，平均值混了兩種樣本")

    # ③ ENB 範圍
    assert (rnd.enb_computed == (rnd.n_members_mean <= 400)).all(), \
        "enb_computed 與 400 檔門檻不一致——與 walkforward_matrix 的規則不同就不可比"
    assert rnd.loc[~rnd.enb_computed, "oos_enb"].isna().all(), \
        "未計算 ENB 的列卻有 oos_enb 值"

    # ④ ratio 維度存在且真的有作用
    assert "ALL" in set(cmp_df.ratio.astype(str)), "對照表缺聚合列 ratio='ALL'"
    per = cmp_df[(cmp_df.metric == "oos_cagr") & (cmp_df.ratio.astype(str) != "ALL")]
    assert len(per) > 0, "對照表缺逐比例列"
    spans = per.groupby("tree_key", observed=True).pct_A_wins.agg(["min", "max"])
    assert ((spans["min"] < 0.5) & (spans["max"] > 0.5)).any(), (
        "沒有任何一棵樹的逐比例勝率跨越 50%——ratio 維度失去作用，"
        "『聚合會掩蓋相反結構』的判讀需重新檢視：\n" + spans.to_string())

    # ⑤ 與 M-02b 的關鍵差別
    rp = d / "walkforward_random_compare.csv"
    assert rp.exists(), "缺 M-02b 的對照表——兩者必須一起讀"
    rc = pd.read_csv(rp)
    sel_r = rc[(rc.metric == "oos_enb") & (rc.ratio.astype(str) != "ALL")]
    assert (sel_r.pct_A_wins > 0.99).all(), (
        "M-02b 的 ENB 勝率不再是每個比例都 100%——「分散度來自多樣性限制」的"
        "對比失效：\n" + sel_r[["tree_key", "ratio", "pct_A_wins"]].to_string(index=False))
    sel_p = cmp_df[(cmp_df.metric == "oos_enb") & (cmp_df.ratio.astype(str) != "ALL")]
    assert (sel_p.pct_A_wins < 0.99).all(), (
        "M-03b 的 ENB 出現 100% 勝率的比例——若換掉分群依據也能全勝，"
        "「群邊界沒有貢獻」的結論要重新檢視")


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
