# -*- coding: utf-8 -*-
"""RQ4：危機期間策略之間的分散與互補是否仍能維持（2026-11-14，設計經使用者確認）。

問題：① 危機期間群間相關是否升高、ENB 是否下降；② 組合在危機期間的實際損失。
論文寫明相關不做波動校正（Forbes & Rigobon 2002），解讀以 ② 實際損失為主。

設計（看結果前寫死）：
  危機定義——月份一律取自 stage2a 凍結的大盤狀態表（門檻 2026-08 已定）：
    D2（主，論文 1-5 的樣本外壓力事件）：2020 新冠、2022 升息，取各市場「包含該事件的下跌段」（熊或危機）
      TW：2020-01-14～03-19（熊）、2022-01-04～10-25（危機）
      US：2020-02-19～03-23（危機）、2022-01-03～06-16 與 2022-08-16～10-12（兩段熊）
      XM：同一事件取兩市場聯集
    D1（描述性）：stage2a 的「危機」標籤（與 crisis 樹相同，含 2008）
    平時：樹的共同窗內，扣掉 D1 ∪ D2 的月份
  群——一律用 normal 樹的 L1 分群（`_frozen/stage3/cluster_assign.parquet`），群序列＝成員月報酬平均（同 stage3 口徑），
    危機和平時比的是同一批群。報酬矩陣用 `stage3_hrp.rebuild_tree_returns`（與 H-09 ENB 同一事實來源）。
  指標——群間平均相關（k×k 非對角平均）、群層 ENB（k 個群序列）、策略層 ENB（N 個策略，`enb_null.fast_enb`）。
  比較——危機月份很少（T=11～26），相關矩陣的秩 ≤ T−1，月份越少 ENB 先天越小，所以主要比較是
    「同長度對照」：從平時月份隨機抽「同樣段數、同樣長度的連續區段」1,000 次，看危機值落在第幾百分位；
    另附危機月份內的區塊自助法（區塊長 2 個月）90% 區間。
  損失——每個 D2 事件：事件期間累積報酬、事件內 MDD（月頻，從事件前一月月底起算）、事件後回到事件前高點的月數。
    精選＝walkforward A_hrp：所有「樣本外期完整涵蓋該事件」的格（方案×k_mode×比例×分配），報分布，不挑單一格；
    全部買進＝normal 樹全部策略等權；大盤＝含息指數（XM 為台美各半，月再平衡）。
    實戰層（日頻）：優等生 control0（排除V1）與大盤，用事件的實際起訖日。
資料版本：研究部＝`_frozen/stage1`、`stage3`（2026-11-10 這一輪：TW 32,726、k=8/4/13）；
  實戰層日頻台股＝`elite_grid_1027` 2026-11-17（研究部 11-14 新池重建名單後重跑步驟1），美股 _liq 2026-10-28。

用法（cwd＝code/）：python -m research.crisis_diversification
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from . import freeze, paths
from . import stage3_hrp as S3
from .enb_null import _standardise, fast_enb

OUT_DIR = paths.ROOT / "_analysis_outputs_robustness"
TREES = ("TW", "US", "XM")
N_NULL = 1000
N_BOOT = 1000
BOOT_BLOCK = 2
MIN_T = 6            # 單一事件月數少於此值不算相關（2020 只有 2～3 個月）
SEED = 20261114
MAIN_DATE_NOTE = "研究部 _frozen 2026-11-10 這一輪（TW 32,726、k=8/4/13）"


# ───────────────────────── 危機月份 ─────────────────────────

def _segments(m: str) -> pd.DataFrame:
    return pd.read_parquet(paths.STAGE2 / "regime" / f"regime_table_{m}.parquet")


def d2_segments(m: str) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """D2：2020、2022 事件 → 該市場包含事件的下跌段（熊或危機）的起訖日。"""
    s = _segments(m)
    down = s[s.label.isin(["熊", "危機"])]
    out = {}
    for ev, (a, b) in {"2020新冠": ("2020-01-01", "2020-12-31"), "2022升息": ("2022-01-01", "2022-12-31")}.items():
        seg = down[(down.start >= a) & (down.start <= b)]
        assert len(seg), f"{m} {ev} 找不到下跌段"
        out[ev] = [(r.start, r.end) for r in seg.itertuples()]
    return out


def _months(segs) -> set[pd.Period]:
    ms = set()
    for a, b in segs:
        ms |= set(pd.period_range(a.to_period("M"), b.to_period("M"), freq="M"))
    return ms


def crisis_months(tree: str) -> dict[str, set[pd.Period]]:
    mk = ["TW", "US"] if tree == "XM" else [tree]
    d2 = {ev: set().union(*[_months(d2_segments(m)[ev]) for m in mk]) for ev in ("2020新冠", "2022升息")}
    d1 = set().union(*[_months([(r.start, r.end) for r in _segments(m).query("label == '危機'").itertuples()]) for m in mk])
    return {"D2": d2["2020新冠"] | d2["2022升息"], "D2_2020新冠": d2["2020新冠"], "D2_2022升息": d2["2022升息"], "D1": d1}


def _blocks(idx: list[int]) -> list[int]:
    """連續索引 → 各連續區段長度。"""
    if not idx:
        return []
    out, run = [], 1
    for a, b in zip(idx, idx[1:]):
        if b == a + 1:
            run += 1
        else:
            out.append(run)
            run = 1
    return out + [run]


# ───────────────────────── 指標 ─────────────────────────

def metrics(X: np.ndarray, G: np.ndarray, cols: np.ndarray) -> dict:
    """X：N×T 策略報酬、G：k×T 群序列，cols：要用的月份索引。"""
    g = G[:, cols]
    c = np.corrcoef(g)
    k = c.shape[0]
    off = c[~np.eye(k, dtype=bool)]
    return {"群間平均相關": float(np.nanmean(off)),
            "群層ENB": fast_enb(_standardise(g)),
            "策略層ENB": fast_enb(_standardise(X[:, cols]))}


def null_draws(normal_idx: np.ndarray, lens: list[int], n: int, rng) -> list[np.ndarray]:
    """從平時月份抽 len(lens) 段、長度分別為 lens 的不重疊連續區段（必須整段落在平時月份內）。"""
    ok = np.zeros(normal_idx.max() + 2, dtype=bool)
    ok[normal_idx] = True
    starts = {L: np.array([s for s in normal_idx if ok[s:s + L].all() and s + L <= len(ok) - 1]) for L in set(lens)}
    out = []
    while len(out) < n:
        used, pick = set(), []
        for L in lens:
            s = int(rng.choice(starts[L]))
            seg = set(range(s, s + L))
            if seg & used:
                break
            used |= seg
            pick += sorted(seg)
        else:
            out.append(np.array(sorted(pick)))
    return out


def boot_draws(cidx: np.ndarray, n: int, rng) -> list[np.ndarray]:
    """危機月份內的移動區塊自助法（區塊長 BOOT_BLOCK，區塊不跨事件以外的月份——直接在危機月份序列上取）。"""
    T = len(cidx)
    nb = int(np.ceil(T / BOOT_BLOCK))
    out = []
    for _ in range(n):
        st = rng.integers(0, T - BOOT_BLOCK + 1, size=nb)
        out.append(cidx[np.concatenate([np.arange(s, s + BOOT_BLOCK) for s in st])[:T]])
    return out


# ───────────────────────── 損失 ─────────────────────────

def event_loss(r: pd.Series, months: set[pd.Period]) -> dict:
    """r：月報酬（PeriodIndex）。事件期間累積報酬、事件內 MDD、事件後回到事件前高點的月數。"""
    ms = sorted(months)
    pre = ms[0] - 1
    w = (1 + r).cumprod()
    w0 = w.loc[:pre].iloc[-1] if (r.index <= pre).any() else 1.0
    peak0 = w.loc[:pre].max() if (r.index <= pre).any() else 1.0
    ev = w.loc[ms[0]:ms[-1]]
    path = np.r_[w0, ev.values]
    dd = path / np.maximum.accumulate(path) - 1
    after = w.loc[ms[-1]:]
    rec = after[after >= peak0]
    return {"事件累積報酬": float(ev.iloc[-1] / w0 - 1), "事件內MDD": float(dd.min()),
            "回到前高月數": int((rec.index[0] - ms[-1]).n) if len(rec) else np.nan}


def daily_loss(r: pd.Series, segs) -> dict:
    a, b = segs[0][0], segs[-1][1]
    x = r.loc[(r.index > a) & (r.index <= b)]
    w = np.r_[1.0, (1 + x).cumprod().values]
    return {"事件累積報酬": float(w[-1] - 1), "事件內MDD": float((w / np.maximum.accumulate(w) - 1).min())}


def bench_monthly(tree: str, months: pd.PeriodIndex) -> pd.Series:
    from app._elite_grid_offline import load_benchmark

    def one(m):
        b = load_benchmark(m)
        return b.resample("M").last().pct_change().to_period("M")
    s = 0.5 * one("TW") + 0.5 * one("US") if tree == "XM" else one(tree)
    return s.reindex(months)


# ───────────────────────── 主流程 ─────────────────────────

def run_tree(tree: str, mem: pd.DataFrame, log=print):
    t0 = time.time()
    wide = S3.rebuild_tree_returns(f"{tree}_normal", log=lambda *a, **k: None)
    months = pd.PeriodIndex(wide.columns, freq="M")
    asg = pd.read_parquet(paths.STAGE3 / "cluster_assign.parquet")
    asg = asg[asg.tree_id.astype(str) == f"{tree}_normal"].set_index("strategy_uid").cluster_L1
    lab = asg.reindex(wide.index)
    assert lab.notna().all(), "有策略不在 normal 樹的分群裡"
    X = wide.values.astype(float)
    G = wide.groupby(lab.values).mean().sort_index().values
    k = G.shape[0]
    cm = crisis_months(tree)
    pos = {m: i for i, m in enumerate(months)}
    all_crisis = {pos[m] for m in (cm["D1"] | cm["D2"]) if m in pos}
    normal_idx = np.array(sorted(set(range(len(months))) - all_crisis))
    rng = np.random.default_rng(SEED)

    rows = []
    full = metrics(X, G, np.arange(len(months)))
    base = {"市場": tree, "資料版本": MAIN_DATE_NOTE, "策略數N": X.shape[0], "群數k": k}
    rows.append({**base, "期間": "全期間", "月數T": len(months), **full})
    rows.append({**base, "期間": "平時（扣除D1∪D2）", "月數T": len(normal_idx), **metrics(X, G, normal_idx)})
    for name in ("D2", "D2_2022升息", "D2_2020新冠", "D1"):
        cidx = np.array(sorted(pos[m] for m in cm[name] if m in pos))
        lens = _blocks(list(cidx))
        if len(cidx) < MIN_T:
            rows.append({**base, "期間": name, "月數T": len(cidx), "區段長度": str(lens), "備註": f"月數 < {MIN_T}，不計算相關"})
            continue
        obs = metrics(X, G, cidx)
        nul = [metrics(X, G, ix) for ix in null_draws(normal_idx, lens, N_NULL, rng)]
        bts = [metrics(X, G, ix) for ix in boot_draws(cidx, N_BOOT, rng)]
        row = {**base, "期間": name, "月數T": len(cidx), "區段長度": str(lens),
               "月份": f"{months[cidx[0]]}～{months[cidx[-1]]}"}
        for key, v in obs.items():
            nv = np.array([d[key] for d in nul])
            bv = np.array([d[key] for d in bts])
            row[key] = v
            row[f"{key}_平時同長度中位數"] = float(np.median(nv))
            row[f"{key}_平時同長度p05"], row[f"{key}_平時同長度p95"] = np.quantile(nv, [.05, .95])
            row[f"{key}_落在平時分布百分位"] = float((nv < v).mean() + 0.5 * (nv == v).mean())
            row[f"{key}_自助法p05"], row[f"{key}_自助法p95"] = np.nanquantile(bv, [.05, .95])
        rows.append(row)
        log(f"  [{tree}] {name}：T={len(cidx)} 群間相關 {obs['群間平均相關']:.3f}（平時同長度中位數 "
            f"{row['群間平均相關_平時同長度中位數']:.3f}，百分位 {row['群間平均相關_落在平時分布百分位']:.0%}）"
            f"｜群層ENB {obs['群層ENB']:.2f}｜策略層ENB {obs['策略層ENB']:.2f}")

    # 損失（D2 每個事件）
    loss = []
    r_all = pd.Series(X.mean(axis=0), index=months)
    bm = bench_monthly(tree, months)
    m = mem[(mem.tree_key == tree) & (mem.group == "A_hrp")]
    # 精選成員一律從 stage1 月報酬原檔取（不少成員資料起始晚於 normal 樹共同窗，不在 wide 裡；
    # 2026-11-14 code review：只用 wide 會漏掉台股最多 45% 的成員）。同一月有資料的成員等權（同 walkforward 的 mean）。
    uids = set().union(*m.members.map(set))
    ml = pd.read_parquet(paths.STAGE1 / "returns_monthly.parquet")
    ml = ml[ml.strategy_uid.isin(uids) & ml.month.isin(months)]
    mw = ml.pivot(index="strategy_uid", columns="month", values="ret").reindex(columns=months)
    for ev in ("2020新冠", "2022升息"):
        ms = sorted(mm for mm in cm[f"D2_{ev}"] if mm in pos)
        tag = {"市場": tree, "事件": ev, "月份": f"{ms[0]}～{ms[-1]}", "頻率": "月", "資料版本": MAIN_DATE_NOTE}
        loss.append({**tag, "組合": "全部買進（normal 樹全部策略等權）", **event_loss(r_all, set(ms))})
        loss.append({**tag, "組合": "大盤（含息）" + ("，台美各半" if tree == "XM" else ""), **event_loss(bm, set(ms))})
        cov = m[(m.oos_start.map(lambda s: pd.Period(s, "M")) <= ms[0]) & (m.oos_end.map(lambda s: pd.Period(s, "M")) >= ms[-1])]
        res = []
        for r in cov.itertuples():
            sub = mw.reindex(list(r.members))
            ev_cols = [mm for mm in months if ms[0] <= mm <= ms[-1]]
            res.append({**event_loss(sub.mean(axis=0, skipna=True), set(ms)),
                        "缺漏成員比例": float(sub[ev_cols].isna().any(axis=1).mean())})
        res = pd.DataFrame(res)
        ref = event_loss(r_all, set(ms))
        loss.append({**tag, "組合": f"精選 A_hrp（樣本外涵蓋事件的 {len(res)} 格，中位數）",
                     **{c: float(res[c].median()) for c in ("事件累積報酬", "事件內MDD", "回到前高月數")},
                     "格數": len(res), "事件累積報酬_p10": float(res.事件累積報酬.quantile(.1)),
                     "事件累積報酬_p90": float(res.事件累積報酬.quantile(.9)),
                     "事件內MDD_p10": float(res.事件內MDD.quantile(.1)), "事件內MDD_p90": float(res.事件內MDD.quantile(.9)),
                     "累積報酬勝過全部買進的格數比例": float((res.事件累積報酬 > ref["事件累積報酬"]).mean()),
                     "MDD淺於全部買進的格數比例": float((res.事件內MDD > ref["事件內MDD"]).mean()),
                     "事件期間缺資料成員比例_最大": float(res.缺漏成員比例.max())})
    log(f"  [{tree}] 完成 {time.time() - t0:.0f}s")
    return rows, loss


def practical_daily(log=print) -> list[dict]:
    from app import _rank_dynamics as RD
    from app._elite_grid_offline import load_benchmark
    out = []
    for mkt, suf, ver in (("TW", "", "實戰層 elite_grid_1027 2026-11-17（研究部 11-14 新池重建名單後重跑步驟1）"),
                          ("US", "_liq", "實戰層 elite_grid_1027 2026-10-28 _liq（美股資料未變，有效）")):
        c0 = RD.load_daily(mkt, "exclude_v1", suf)["REF_control0"]
        # 2026-11-24 補：論文 4-5 主名單是含 v1（pool=baseline），寫作 session 要求補成正式列
        c0_in = RD.load_daily(mkt, "baseline", suf)["REF_control0"]
        b = load_benchmark(mkt).pct_change()
        for ev, segs in d2_segments(mkt).items():
            tag = {"市場": mkt, "事件": ev, "月份": f"{segs[0][0].date()}～{segs[-1][1].date()}", "頻率": "日", "資料版本": ver}
            out.append({**tag, "組合": "優等生 control0（含v1，不調整）", **daily_loss(c0_in, segs)})
            out.append({**tag, "組合": "優等生 control0（排除V1，不調整）", **daily_loss(c0, segs)})
            out.append({**tag, "組合": "大盤（含息）", **daily_loss(b, segs)})
    return out


def main(argv=None) -> int:
    freeze.verify_inputs(paths.STAGE1)
    freeze.verify_inputs(paths.STAGE1 / "_marks")
    freeze.verify_inputs(paths.STAGE3)
    mem = pd.read_parquet(OUT_DIR / "walkforward_members.parquet")
    for c in ("tree_key", "group"):
        mem[c] = mem[c].astype(str)
    rows, loss = [], []
    for t in TREES:
        r, l_ = run_tree(t, mem)
        rows += r
        loss += l_
    loss += practical_daily()
    p1, p2 = OUT_DIR / "crisis_diversification.csv", OUT_DIR / "crisis_diversification_loss.csv"
    pd.DataFrame(rows).to_csv(p1, index=False, encoding="utf-8-sig")
    pd.DataFrame(loss).to_csv(p2, index=False, encoding="utf-8-sig")
    freeze.write_manifest(
        "crisis_diversification", OUT_DIR / "_crisis_diversification_manifest",
        inputs=[paths.STAGE1 / "returns_monthly.parquet", paths.STAGE1 / "returns_meta.parquet",
                paths.STAGE1 / "strategy_marks.parquet", paths.STAGE3 / "cluster_assign.parquet",
                paths.STAGE2 / "regime" / "regime_table_TW.parquet", paths.STAGE2 / "regime" / "regime_table_US.parquet",
                OUT_DIR / "walkforward_members.parquet"],
        outputs=[p1, p2],
        params={"N_NULL": N_NULL, "N_BOOT": N_BOOT, "BOOT_BLOCK": BOOT_BLOCK, "SEED": SEED},
        notes="RQ4 危機期間分散：D2（2020、2022 下跌段）主、D1（危機標籤，含2008）描述性；同長度平時區段對照＋區塊自助法")
    print(p1, p2, sep="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
