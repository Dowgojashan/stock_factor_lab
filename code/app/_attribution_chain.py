# -*- coding: utf-8 -*-
"""事後審核：完整版績效歸因鏈（2026-11-15 依使用者選定的 (c) 方案撰寫；**尚未執行正式結果**）。

論文承諾（第一章 1-4、第二章 2-6-3，Brinson et al. 1986 的思路）：區分績效來自「選了哪些策略」還是「權重的調整」。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
一、鏈的定義（每一關只換掉一件事；時間軸＝實戰層步驟1 的 4 窗、44 季，季末對齊）
  L1 大盤            含息指數（台股 taiex_tr、美股 sp500_tr），季末收盤相除
  L2a 全部買進（含V1） 該市場全部可用候選策略（stage1 is_usable）等權，策略層月報酬複利成季 ← 只在 pool=exclude_v1 時另列
  L2 全部買進         該池全部可用候選策略等權（pool=exclude_v1 只取 `__v0` 策略；baseline 取全部）
  L3 精選             實戰層實際使用的代表名單（每窗一份），策略層等權（同 walkforward `_portfolio_series`：
                      當月有資料的成員取平均，每月再平衡、不計策略之間再平衡的成本）
  L4 control0         同一份名單換成股票層持股（`_elite_grid_offline` 的 REF_control0：季初調整、季內買進持有、
                      扣股票層實際換手成本）
  L5 調權重           規則（主：步驟1 晉級第 1 名 `promoted_*.csv` 首列，事前固定）或 LLM 鏈（只有 window4）
  各關貢獻：
    策略池 = L2a−L1（或 L2−L1）  ⚠️ 全期篩選偏誤：候選池 100% 是「全期間贏過基準」篩出來的（H-01 alpha 關卡，
                                  v10 §8），這一段系統性高估，只能當描述，不能解讀為「建池的價值」
    排除V1 = L2−L2a（只有 exclude_v1）
    精選   = L3−L2
    實作落差 = L4−L3  ← 獨立列出，不歸給任何決策：混合了「月 vs 季」再平衡頻率、策略內部換股 vs 季內買進持有、
                        股票層換手成本、策略解不出持股時的現金部位
    調權重 = L5−L4，再拆成（規則鏈）：
        優等生選股 = 資金比例 s ×（優等生組合毛報酬 r_E − control0 毛報酬）
        成本差     = −（規則成本 − control0 成本）
      推導：overlay＝(1−s)·w0＋s·Σw0·elite（`elite_rules.overlay`），季內買進持有 ⇒ 毛報酬對權重線性，
            r_rule = (1−s)·r_c0 + s·r_E ⇒ r_E = (r_rule − (1−s)·r_c0)/s（不需要重產 w0）。
      ⚠️ 不拆 Brinson 的「配置×選股」交互項：overlay 是線性的，直接寫成 s×(r_E−r_c0)，沒有交互項需要分配。

二、linking（事前選定）
  主：**累積報酬逐關相減**（cum_Lj − cum_Lj−1）。逐關相加恆等於 cum_L5 − cum_L1，不需要選 linking 方法。
  輔（看逐季）：逐季算術差 d_j,t = r_j,t − r_j−1,t，用 **Carino (1999) 對數 linking** 調整成可加：
      k_t = ln(1+r_P,t) − ln(1+r_B,t) 除以 (r_P,t − r_B,t)（相等時 = 1/(1+r_P,t)），k 同理用全期累積；
      linked_j = Σ_t d_j,t · k_t / k，P＝最後一關、B＝大盤；Σ_j linked_j ＝ R_P − R_B（恆等）。
  逐窗也各自做一次（窗內累積），因為 2024–25 台股由單一權值股主導，只報全期會被單一年度帶著走。

三、驗證（全部寫成 assert，執行時自動檢查；任何一項失敗即中止，不輸出）
  V1 恆等：Σ 各關累積差 ＝ cum_L5 − cum_L1（誤差 < 1e-12）
  V2 Carino：Σ linked ＝ R_P − R_B（誤差 < 1e-10），全期與逐窗
  V3 線性（驗證拆解的前提）：同一（排名依據, 取幾名, 分法）不同資金比例反推出的 r_E 必須相同（誤差 < 1e-8）
  V4 名單一致：實戰層每窗名單的策略數，必須等於步驟1 執行紀錄印出的「window{w}：N 個代表策略」；
      名單內策略必須全部在 stage1 月報酬裡（缺一檔就中止——名單版本與研究部版本不一致的訊號）
  V5 LLM：L5 為 LLM 鏈時，window4 的季複利必須等於步驟2 結算 `phase2*_settled` 的 cum_net（誤差 < 1e-8）
  V6 研究部對帳（只有 baseline 池、名單取自 walkforward 那一格時）：L3 在該格樣本外期的月複利 CAGR
      必須等於 `walkforward_matrix_detail.csv` 該格的 oos_cagr（誤差 < 1e-9）

四、名單事前固定與各格分布
  名單：exclude_v1＝`anchored_w{w}_openSec_{boost|balanced}_exclude_v1.parquet` 的 allocation=equal 列；
        baseline＝walkforward 的 scheme E、silhouette_is、legacy、equal、A_hrp（同 `_elite_grid_offline.load_window`）。
        ⚠️ baseline 名單必須用「步驟1 執行當時」的 walkforward_members（`--members-file` 明確指定），
           現行檔已被研究部重跑覆寫時，用錯版本會讓 L3 與 L4 的成員不同，V4 會擋下來。
  分布（穩健性，不進鏈）：
    ① 精選貢獻在研究部所有格的分布：`walkforward_matrix_detail.csv` 中該市場每格 A_hrp 與 B_all 的 oos_cagr 差
       （研究部口徑、各格自己的樣本外期），列分位數；鏈的名單只是其中一種做法
    ② 調權重貢獻在步驟1 全部 1,040 格的分布（同池、同期），列分位數與主規則所在的百分位
  ⚠️ 主規則（晉級第 1 名）是用 2015–23 的扣成本 Sharpe 選出的，window1～3 對它而言是樣本內；只有 window4 是樣本外。
     輸出逐窗表時 window1～3 的 L5 一律標「樣本內」。

五、資料版本（執行時寫進輸出與 manifest）
  每個輸入檔記錄 mtime 與 sha256；台股的實戰層名單／步驟1 若早於研究部最後一次重跑，輸出標「名單為舊版」。
  依使用者 2026-11-15 指示：台股正式版等研究部重跑完、名單重建、步驟1 重跑後再執行；美股何時驗證由使用者決定。

用法（cwd＝code/；**尚未執行過**）：
  python -m app._attribution_chain --market US --pool exclude_v1
  python -m app._attribution_chain --market US --pool baseline --members-file <步驟1 當時的 walkforward_members.parquet>
  加 --llm 另外輸出 window4 的 LLM 鏈（每個方案×風格×重複一條）
輸出：`_analysis_outputs_applayer/attribution_chain/{市場}_{池}/`（chain_cum.csv、chain_quarterly.csv、
      chain_linked.csv、dist_selection.csv、dist_rules.csv）＋ manifest
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import pandas as pd  # noqa: E402

from research import freeze, paths  # noqa: E402

ROOT = paths.ROOT
APP = ROOT / "_analysis_outputs_applayer"
GRID = APP / "elite_grid_1027"
STEP2 = APP / "step2"
WF_DETAIL = ROOT / "_analysis_outputs_robustness" / "walkforward_matrix_detail.csv"
OUT_BASE = APP / "attribution_chain"
TAG = {("TW", "exclude_v1"): "TW_exclude_v1", ("TW", "baseline"): "TW_baseline",
       ("US", "exclude_v1"): "US_exclude_v1_liq", ("US", "baseline"): "US_baseline_liq"}
PICKS = {"TW": "anchored_w{w}_openSec_boost_exclude_v1.parquet", "US": "anchored_w{w}_openSec_balanced_exclude_v1.parquet"}
RUN_LOG = {("TW", "exclude_v1"): "_twfix_TW_exclude_v1_log.txt", ("TW", "baseline"): "_twfix_TW_baseline_log.txt",
           ("US", "exclude_v1"): "_run_US_exclude_v1_liq_log.txt", ("US", "baseline"): "_run_US_baseline_liq_log.txt"}
TOL_ID, TOL_LINK, TOL_LIN, TOL_LLM, TOL_WF = 1e-12, 1e-10, 1e-8, 1e-8, 1e-9


# ───────────────────────── 基本工具 ─────────────────────────

def cum(r) -> float:
    return float(np.prod(1 + np.asarray(r, dtype=float)) - 1)


def carino(layers: np.ndarray) -> np.ndarray:
    """layers：J×T 的逐季報酬（第 0 列＝大盤、最後一列＝P）。回傳 J−1 個關卡的 linked 貢獻，總和＝R_P−R_B。"""
    rB, rP = layers[0], layers[-1]

    def kf(p, b):
        return np.where(np.isclose(p, b, rtol=0, atol=1e-15), 1 / (1 + p),
                        (np.log1p(p) - np.log1p(b)) / np.where(np.isclose(p, b, rtol=0, atol=1e-15), 1, p - b))
    kt = kf(rP, rB)
    RP, RB = cum(rP), cum(rB)
    k = float(kf(np.array(RP), np.array(RB)))
    d = np.diff(layers, axis=0)                       # (J−1)×T
    out = (d * kt).sum(axis=1) / k
    assert abs(out.sum() - (RP - RB)) < TOL_LINK, f"V2 Carino 不可加：{out.sum()} vs {RP - RB}"
    return out, kt / k


def quarter_months(as_of: str, end: str) -> pd.PeriodIndex:
    return pd.period_range(pd.Period(as_of, "M") + 1, pd.Period(end, "M"), freq="M")


# ───────────────────────── 各關報酬 ─────────────────────────

def calendar(tag: str) -> pd.DataFrame:
    q = pd.read_parquet(GRID / f"quarterly_{tag}.parquet")
    cal = q[["window", "q", "as_of", "end"]].drop_duplicates().sort_values(["window", "q"]).reset_index(drop=True)
    assert len(cal) == 44 and cal.as_of.is_unique, "步驟1 季曆不是 44 個不重複的季"
    assert (cal.as_of.iloc[1:].values == cal.end.iloc[:-1].values).all(), "季與季之間不連續"
    return cal


def bench_layer(market: str, cal: pd.DataFrame) -> np.ndarray:
    from app._elite_grid_offline import load_benchmark
    b = load_benchmark(market)
    lv = lambda d: float(b.loc[:pd.Timestamp(d)].iloc[-1])  # noqa: E731
    return np.array([lv(e) / lv(a) - 1 for a, e in zip(cal.as_of, cal.end)])


def members_by_window(market: str, pool: str, members_file: Path | None, run_log: str | None = None) -> dict[int, list[str]]:
    out = {}
    if pool == "exclude_v1":
        for w in (1, 2, 3, 4):
            df = pd.read_parquet(APP / PICKS[market].format(w=w))
            sub = df[df.allocation == "equal"]
            assert len(sub) == 1, f"w{w} 名單列數 {len(sub)}"
            out[w] = list(sub.iloc[0].members)
    else:
        assert members_file is not None, "baseline 池必須用 --members-file 指定步驟1 當時的 walkforward_members"
        wf = pd.read_parquet(members_file)
        for c in ("tree_key", "scheme", "k_mode", "ratio", "group", "allocation"):
            wf[c] = wf[c].astype(str)
        for w in (1, 2, 3, 4):
            sub = wf[(wf.tree_key == market) & (wf.scheme == "E") & (wf.k_mode == "silhouette_is") & (wf.ratio == "legacy")
                     & (wf.group == "A_hrp") & (wf.allocation == "equal") & (wf.window_no == w)]
            assert len(sub) == 1, f"w{w} walkforward 名單列數 {len(sub)}"
            out[w] = list(sub.iloc[0].members)
    # V4：與步驟1 執行紀錄的代表策略數一致
    log = (GRID / (run_log or RUN_LOG[(market, pool)])).read_text(encoding="utf-8")
    logged = {int(w): int(n) for w, n in re.findall(r"== window(\d)：(\d+) 個代表策略", log)}
    for w, m in out.items():
        assert logged.get(w) == len(m), f"V4 名單不一致：w{w} 名單 {len(m)} 檔，步驟1 紀錄 {logged.get(w)} 檔（名單版本可能不同）"
    return out


def strategy_monthly(uids: set[str]) -> pd.DataFrame:
    ml = pd.read_parquet(paths.STAGE1 / "returns_monthly.parquet")
    ml = ml[ml.strategy_uid.isin(uids)]
    return ml.pivot(index="strategy_uid", columns="month", values="ret")


def pool_uids(market: str, pool: str, v1_too: bool) -> set[str]:
    marks = pd.read_parquet(paths.STAGE1 / "strategy_marks.parquet")
    u = marks.loc[marks.is_usable, "strategy_uid"]
    u = u[u.str.startswith(market + "::")]
    if pool == "exclude_v1" and not v1_too:
        u = u[u.str.endswith("__v0")]
    return set(u)


def ew_quarterly(wide: pd.DataFrame, uids, cal: pd.DataFrame, per_window: dict[int, list[str]] | None = None) -> np.ndarray:
    """策略層等權（每月取當月有資料的成員平均）→ 複利成季。per_window 給定時每窗用各自的名單。"""
    out = []
    for r in cal.itertuples():
        mem = per_window[r.window] if per_window else uids
        ms = quarter_months(r.as_of, r.end)
        sub = wide.reindex(index=list(mem), columns=ms)
        m = sub.mean(axis=0, skipna=True).values
        assert np.isfinite(m).all(), f"{r.as_of}～{r.end} 有月份沒有任何成員資料"
        out.append(cum(m))
    return np.array(out)


def grid_layers(tag: str, cal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q = pd.read_parquet(GRID / f"quarterly_{tag}.parquet")
    key = ["window", "q"]
    g = q.pivot_table(index=key, columns="combo", values="ret_gross").loc[list(zip(cal.window, cal.q))]
    c = q.pivot_table(index=key, columns="combo", values="cost").loc[list(zip(cal.window, cal.q))]
    meta = pd.read_csv(GRID / f"combos_{tag}.csv").set_index("combo")
    return g, c, meta


def elite_return(g: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """由 r_rule = (1−s)·r_c0 + s·r_E 反推每格的優等生毛報酬 r_E；V3：同 (score, n, scheme) 各資金比例反推值必須相同。"""
    rc0 = g["REF_control0"]
    rows = {}
    for combo, m in meta[meta.sleeve > 0].iterrows():
        rows[combo] = (g[combo] - (1 - m.sleeve) * rc0) / m.sleeve
    rE = pd.DataFrame(rows)
    key = meta.loc[rE.columns, ["score", "n", "scheme"]].astype(str).agg("|".join, axis=1)
    for k, cols in rE.columns.groupby(key).items():
        spread = (rE[cols].max(axis=1) - rE[cols].min(axis=1)).abs().max()
        assert spread < TOL_LIN, f"V3 線性不成立：{k} 各資金比例反推的優等生報酬差 {spread:.2e}"
    return rE


# ───────────────────────── LLM 鏈（只有 window4） ─────────────────────────

def llm_chains(market: str, pool: str, cal: pd.DataFrame) -> dict[str, np.ndarray]:
    """步驟2 各條鏈（方案×風格×重複）在 window4 的逐季淨報酬（由逐日淨值的季末值相除）。"""
    out = {}
    w4 = cal[cal.window == 4]
    for ver in ("", "v2"):
        f = STEP2 / f"phase2{ver}_daily_{market}_{pool}_gpt-5-mini.parquet"
        st = pd.read_csv(STEP2 / f"phase2{ver}_settled_{market}_{pool}_gpt-5-mini.csv")
        nav = pd.read_parquet(f)
        nav.index = pd.to_datetime(nav.index)
        for ch in nav.columns:
            if not str(ch).startswith("方案|"):
                continue
            s = nav[ch].dropna()
            lv = [float(s.loc[:pd.Timestamp(w4.as_of.iloc[0])].iloc[-1])] + [float(s.loc[:pd.Timestamp(e)].iloc[-1]) for e in w4.end]
            r = np.array(lv[1:]) / np.array(lv[:-1]) - 1
            _, plan, style, rep = str(ch).split("|")
            ref = st[(st.kind == "方案") & (st.plan == plan) & (st["style"] == style) & (st.rep.astype(str) == rep)]
            assert len(ref) == 1 and abs(cum(r) - float(ref.cum_net.iloc[0])) < TOL_LLM, f"V5 LLM 鏈 {ver}{ch} 與步驟2 結算不一致"
            out[f"LLM{ver or 'v1'}|{plan}|{style}|{rep}"] = r
    return out


# ───────────────────────── 主流程 ─────────────────────────

def build(market: str, pool: str, members_file: Path | None, rule: str | None, with_llm: bool, log=print,
          run_log: str | None = None):
    tag = TAG[(market, pool)]
    cal = calendar(tag)
    per_w = members_by_window(market, pool, members_file, run_log)
    all_mem = set().union(*per_w.values())
    pool_all = pool_uids(market, pool, v1_too=True)
    pool_main = pool_uids(market, pool, v1_too=False)
    wide = strategy_monthly(all_mem | pool_all)
    missing = all_mem - set(wide.index)
    assert not missing, f"V4 名單內有 {len(missing)} 檔不在 stage1 月報酬（名單與研究部版本不一致），例：{sorted(missing)[:3]}"

    g, c, meta = grid_layers(tag, cal)
    rule = rule or pd.read_csv(GRID / f"promoted_{tag}.csv").iloc[0]["combo"]
    assert rule in g.columns, f"規則 {rule} 不在步驟1 網格"
    s = float(meta.loc[rule, "sleeve"])
    rE = elite_return(g, meta)

    L = {"L1 大盤": bench_layer(market, cal)}
    if pool == "exclude_v1":
        L["L2a 全部買進（含V1）"] = ew_quarterly(wide, pool_all, cal)
    L["L2 全部買進" + ("（排除V1）" if pool == "exclude_v1" else "")] = ew_quarterly(wide, pool_main, cal)
    L["L3 精選（策略層等權）"] = ew_quarterly(wide, None, cal, per_w)
    rc0_net = (g["REF_control0"] - c["REF_control0"]).values
    L["L4 control0（股票層）"] = rc0_net
    L[f"L5 調權重（規則 {rule}）"] = (g[rule] - c[rule]).values
    names = list(L)
    M = np.vstack([L[k] for k in names])                                  # J×44

    # 調權重拆解（逐季）
    sel = s * (rE[rule].values - g["REF_control0"].values)
    dcost = -(c[rule].values - c["REF_control0"].values)
    assert np.allclose(sel + dcost, M[-1] - M[-2], atol=1e-12), "調權重拆解與 L5−L4 不一致"

    NAME = {"L2a": "策略池（⚠️全期篩選偏誤）", "L2": "排除V1" if pool == "exclude_v1" else "策略池（⚠️全期篩選偏誤）",
            "L3": "精選", "L4": "實作落差（不歸給決策）", "L5": "調權重"}
    contrib_name = {j: NAME[names[j].split()[0]] for j in range(1, len(names))}

    def chain_table(idx: np.ndarray, label: str, in_sample: str) -> list[dict]:
        cums = [cum(M[j, idx]) for j in range(len(names))]
        diffs = np.diff(cums)
        assert abs(diffs.sum() - (cums[-1] - cums[0])) < TOL_ID, "V1 恆等不成立"
        linked, wt = carino(M[:, idx])
        rows = [{"期間": label, "L5樣本內外": in_sample, "關卡": names[0], "累積報酬": cums[0]}]
        for j in range(1, len(names)):
            rows.append({"期間": label, "L5樣本內外": in_sample, "關卡": names[j], "累積報酬": cums[j],
                         "貢獻名稱": contrib_name[j], "貢獻_累積相減": float(diffs[j - 1]), "貢獻_Carino": float(linked[j - 1])})
        # 調權重拆解：逐季 L5−L4 ＝ sel＋dcost，用同一組 Carino 權重 linking ⇒ 兩項相加＝調權重的 Carino 貢獻
        ls, lc = float((sel[idx] * wt).sum()), float((dcost[idx] * wt).sum())
        assert abs(ls + lc - linked[-1]) < TOL_LINK, "調權重拆解的 linking 加總不等於調權重貢獻"
        for nm, v in (("優等生選股 s×(r_E−r_c0)", ls), ("成本差", lc)):
            rows.append({"期間": label, "L5樣本內外": in_sample, "關卡": "　└ 調權重拆解", "貢獻名稱": nm, "貢獻_Carino": v})
        return rows

    rows = chain_table(np.arange(44), "全期（44 季）", "window1～3 樣本內、window4 樣本外")
    for w in (1, 2, 3, 4):
        idx = np.where(cal.window.values == w)[0]
        rows += chain_table(idx, f"window{w}", "樣本外" if w == 4 else "樣本內")
    quarterly = pd.DataFrame(M.T, columns=names).assign(window=cal.window.values, as_of=cal.as_of.values, end=cal.end.values,
                                                        調權重_優等生選股=sel, 調權重_成本差=dcost)

    # 分布②：調權重貢獻在全部格的分布（同池同期，扣成本累積差）
    net = g - c
    dist_rules = []
    for label, idx in [("全期", np.arange(44))] + [(f"window{w}", np.where(cal.window.values == w)[0]) for w in (1, 2, 3, 4)]:
        d = (np.prod(1 + net.values[idx], axis=0) - 1) - cum(rc0_net[idx])
        dd = pd.Series(d, index=net.columns)
        dd = dd[~dd.index.str.startswith("REF_")]
        dist_rules.append({"期間": label, "主規則": rule, "主規則_貢獻": float(dd[rule]),
                           "主規則_百分位": float((dd < dd[rule]).mean()), "格數": len(dd),
                           **{f"p{int(p * 100):02d}": float(dd.quantile(p)) for p in (.05, .25, .5, .75, .95)}})

    # 分布①：精選貢獻在研究部各格的分布（研究部口徑）
    det = pd.read_csv(WF_DETAIL)
    det = det[det.tree_key == market]
    piv = det.pivot_table(index=["scheme", "k_mode", "ratio", "allocation", "window_no"], columns="group", values="oos_cagr")
    b = det[det.group == "B_all"].set_index(["scheme", "k_mode", "window_no"]).oos_cagr
    b = b[~b.index.duplicated()]
    diff = piv["A_hrp"] - [b.get((i[0], i[1], i[4]), np.nan) for i in piv.index]
    dist_sel = pd.DataFrame([{"市場": market, "格數": int(diff.notna().sum()), "口徑": "A_hrp 減 B_all 的樣本外 CAGR（研究部，各格自己的樣本外期）",
                              **{f"p{int(p * 100):02d}": float(diff.quantile(p)) for p in (.05, .25, .5, .75, .95)},
                              "大於0的比例": float((diff > 0).mean())}])

    # V6：baseline 名單取自 walkforward 那一格時，L3 與研究部該格對帳
    v6 = "不適用（exclude_v1 名單不是 walkforward 的格）"
    if pool == "baseline":
        cur = ROOT / "_analysis_outputs_robustness" / "walkforward_members.parquet"
        same = freeze.sha256_file(Path(members_file)) == freeze.sha256_file(cur)
        v6 = "通過" if same else "略過：--members-file 不是研究部現行版本，walkforward_matrix_detail 已是另一版名單的結果"
        log(f"V6：{v6}")
    if pool == "baseline" and same:
        for w in (1, 2, 3, 4):
            r = det[(det.scheme == "E") & (det.k_mode == "silhouette_is") & (det.ratio.astype(str) == "legacy")
                    & (det.allocation == "equal") & (det.group == "A_hrp") & (det.window_no == w)]
            if len(r) != 1:
                continue
            ms = pd.period_range(r.oos_start.iloc[0], r.oos_end.iloc[0], freq="M")
            p = wide.reindex(index=per_w[w], columns=ms).mean(axis=0, skipna=True).values
            cagr = float(np.prod(1 + p) ** (12 / len(p)) - 1)
            assert abs(cagr - float(r.oos_cagr.iloc[0])) < TOL_WF, f"V6 w{w} L3 與研究部 oos_cagr 不一致（{cagr} vs {r.oos_cagr.iloc[0]}）"

    llm = llm_chains(market, pool, cal) if with_llm else {}
    llm_rows = []
    w4 = np.where(cal.window.values == 4)[0]
    for ch, r in llm.items():
        Mw = np.vstack([M[:-1, w4], r])
        cums = [cum(x) for x in Mw]
        linked, _ = carino(Mw)
        llm_rows.append({"LLM鏈": ch, "L4 control0 累積": cums[-2], "L5 LLM 累積": cums[-1],
                         "調權重貢獻_累積相減": cums[-1] - cums[-2], "調權重貢獻_Carino": float(linked[-1])})
    checks = pd.DataFrame([{"V1 恆等": "通過", "V2 Carino": "通過", "V3 線性": "通過", "V4 名單一致": "通過",
                            "V5 LLM 對帳": "通過" if with_llm else "未執行（沒有 --llm）", "V6 研究部對帳": v6, "主規則": rule,
                            "主規則資金比例": s}])
    return {"checks": checks, "chain_cum": pd.DataFrame(rows), "chain_quarterly": quarterly, "dist_rules": pd.DataFrame(dist_rules),
            "dist_selection": dist_sel, "llm_window4": pd.DataFrame(llm_rows)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True, choices=["TW", "US"])
    ap.add_argument("--pool", required=True, choices=["exclude_v1", "baseline"])
    ap.add_argument("--members-file", type=Path, default=None)
    ap.add_argument("--rule", default=None, help="預設＝步驟1 晉級第 1 名（事前固定）")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--run-log", default=None, help="步驟1 執行紀錄檔名（V4 用）；步驟1 重跑後要指定新的紀錄檔")
    a = ap.parse_args(argv)
    out = build(a.market, a.pool, a.members_file, a.rule, a.llm, run_log=a.run_log)
    d = OUT_BASE / f"{a.market}_{a.pool}"
    d.mkdir(parents=True, exist_ok=True)
    files = []
    for k, df in out.items():
        p = d / f"{k}.csv"
        df.to_csv(p, index=False, encoding="utf-8-sig")
        files.append(p)
    tag = TAG[(a.market, a.pool)]
    inputs = [paths.STAGE1 / "returns_monthly.parquet", paths.STAGE1 / "strategy_marks.parquet", WF_DETAIL,
              GRID / f"quarterly_{tag}.parquet", GRID / f"combos_{tag}.csv", GRID / f"promoted_{tag}.csv"]
    inputs += [APP / PICKS[a.market].format(w=w) for w in (1, 2, 3, 4)] if a.pool == "exclude_v1" else [a.members_file]
    freeze.write_manifest("attribution_chain", d, inputs=inputs, outputs=files,
                          params={"market": a.market, "pool": a.pool, "rule": a.rule, "llm": a.llm},
                          notes="績效歸因鏈：大盤→全部買進→精選→control0→調權重；主＝累積相減，輔＝Carino linking")
    print(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
