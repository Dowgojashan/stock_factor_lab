# -*- coding: utf-8 -*-
"""步驟2：每季「從規則選單挑一個」的共用模組（實戰收尾追蹤.md §四 步驟2，2026-10-28）。

老師 10/27：測 AI 要比較不同挑法（policy）的影響；AI 的產出是「時間 × 權重」矩陣。
這裡把每一種挑法都表示成「每季選單中選哪一個規則」，規則本身的每季報酬直接取
步驟1 的網格輸出（`elite_grid_1027/quarterly_*.parquet`），因此：
  - 階段0（不用 LLM）：固定規則、追強／逆勢換法、警示觸發換法、ε-greedy、Thompson、
    隨機、事後最佳，全部在這裡算
  - 階段1+（LLM）：LLM 每季回傳選單中的一個代號，同樣用這裡的報酬表結算

時間慣例跟步驟1／正式模擬一致：第 k 季季初（as_of）用到 as_of 為止的資訊做選擇，
套用整季。window1~3 的季報酬只當「歷史」（bandit 先驗、上季表現），評估只在 window4。
主指標＝未扣成本季報酬複利；近似扣成本＝沿用該規則在步驟1 自己那條路徑的成本
（真正換規則時的換手不同，屬近似，報告須註明）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
GRID = ROOT / "_analysis_outputs_applayer" / "elite_grid_1027"
RUNS = ROOT / "code" / "app" / "_runs"
FAMILIES = ("S1", "S2", "S3", "S4", "S5", "S6")

#: 各組的步驟1 輸出代號（美股用加流動性門檻版，2026-10-28 使用者裁示）
TAG = {("TW", "exclude_v1"): "TW_exclude_v1", ("TW", "baseline"): "TW_baseline",
       ("US", "exclude_v1"): "US_exclude_v1_liq", ("US", "baseline"): "US_baseline_liq"}
#: 正式 L2 checkpoint（取每季季末的集中度警示狀態；警示只看市場，不受持股影響）
CKPT = {("TW", "exclude_v1"): "simulate_anchored_w4_openSec_boost_exclude_v1_cap30_hotseg_L2.jsonl",
        ("TW", "baseline"): "simulate_formal_8q_control0_L2_openSec_boost_v4_cap30_m2_v2.jsonl",
        ("US", "exclude_v1"): "simulate_anchored_w4_openSec_balanced_exclude_v1_cap30_hotseg_m1dfix_L2.jsonl",
        ("US", "baseline"): "simulate_anchored_w4_openSec_balanced_baseline_cap30_hotseg_m1dfix_L2.jsonl"}


# ───────────────────────── 資料 ─────────────────────────

def load_returns(market: str, pool: str, with_sig: bool = False):
    """回傳 (gross, cost)：index=(window, q, as_of)、columns=combo 的季報酬與季初成本。
    with_sig＝合併「熱門股／突破股」網格（`*_sig`，版本2 用；兩份網格的 control0 等對照組相同）。"""
    q = pd.read_parquet(GRID / f"quarterly_{TAG[(market, pool)]}.parquet")
    if with_sig:
        qs = pd.read_parquet(GRID / f"quarterly_{TAG[(market, pool)]}_sig.parquet")
        q = pd.concat([q, qs[~qs.combo.str.startswith("REF_")]])
    q = q.sort_values(["window", "q"])
    g = q.pivot_table(index=["window", "q", "as_of"], columns="combo", values="ret_gross")
    c = q.pivot_table(index=["window", "q", "as_of"], columns="combo", values="cost")
    return g, c


def family_best(market: str, pool: str, with_sig: bool = False) -> dict[str, str]:
    """步驟1 晉級規則裡的「各家族最佳」（2015–23 扣成本 Sharpe），{family: combo}。
    with_sig＝另加 S7 熱門股、S8 突破股（`*_sig` 網格以同一套晉級規則選出）。"""
    t = pd.read_csv(GRID / f"metrics_all_{TAG[(market, pool)]}.csv", index_col=0)
    fams = list(FAMILIES)
    if with_sig:
        t = pd.concat([t, pd.read_csv(GRID / f"metrics_all_{TAG[(market, pool)]}_sig.csv", index_col=0)])
        t = t[~t.index.duplicated()]
        fams += ["S7", "S8"]
    out = {}
    for f in fams:
        hit = t[t.promote_reason.fillna("").str.contains(f"{f}最佳")]
        if len(hit) != 1:
            raise AssertionError(f"{market}/{pool} 找不到唯一的 {f} 家族最佳（{len(hit)}）")
        out[f] = hit.index[0]
    return out


def menu_A(market, pool) -> list[str]:
    fb = family_best(market, pool)
    return ["REF_control0"] + [fb[f] for f in FAMILIES]


def menu_B(market, pool) -> list[str]:
    fb = family_best(market, pool)
    arms = ["REF_control0"]
    for f in FAMILIES:
        score = fb[f].split("|")[0]
        for n in (1, 3, 5, 10):
            for sl in (10, 20, 30, 50):
                arms.append(f"{score}|N{n}|equal|sl{sl}")
    return arms


def menu_C(market, pool) -> list[str]:
    return ["REF_control0"] + [f"S6_mcap|N3|equal|sl{sl}" for sl in (10, 30, 50)]


MENUS = {"A": menu_A, "B": menu_B, "C": menu_C}


# ── 版本2（2026-10-28）：三種訊號配對 → 選單加入熱門股（S7）、突破股（S8） ──
def menu_A2(market, pool) -> list[str]:
    fb = family_best(market, pool, with_sig=True)
    return ["REF_control0"] + [fb[f] for f in list(FAMILIES) + ["S7", "S8"]]


def menu_B2(market, pool) -> list[str]:
    arms = menu_B(market, pool)
    for score in ("S7_hot", "S8_breakout"):
        for n in (1, 3, 5, 10):
            for sl in (10, 20, 30, 50):
                arms.append(f"{score}|N{n}|equal|sl{sl}")
    return arms


def menu_C2(market, pool) -> list[str]:
    arms = ["REF_control0"]
    for rule in ("S6_mcap|N3|equal", "S7_hot|N3|equal", "S8_breakout|N10|equal"):
        arms += [f"{rule}|sl{sl}" for sl in (10, 30, 50)]
    return arms


MENUS2 = {"A": menu_A2, "B": menu_B2, "C": menu_C2}
#: 不用 LLM 的「訊號規則」（版本2 對照組，事先寫死優先順序）：集中度觸發→市值；否則熱門股覆蓋不足（觸發）→熱門股；
#: 否則突破廣度「高」→突破股；都沒有→control0。用選單 C2 的資金30% 版本
SIGNAL_RULE = {"m1d": "S6_mcap|N3|equal|sl30", "hot": "S7_hot|N3|equal|sl30", "breakout": "S8_breakout|N10|equal|sl30"}


def run_signal_rule(sig: pd.DataFrame, as_ofs: list[str]) -> list[str]:
    picks = []
    for a in as_ofs:
        r = sig.loc[a]
        if r.m1d_state == "TRIGGERED":
            picks.append(SIGNAL_RULE["m1d"])
        elif r.hot_state == "TRIGGERED":
            picks.append(SIGNAL_RULE["hot"])
        elif r.breakout_state == "高":
            picks.append(SIGNAL_RULE["breakout"])
        else:
            picks.append("REF_control0")
    return picks


M1D_CACHE = ROOT / "_analysis_outputs_applayer" / "step2" / "m1d_series_{market}.csv"


def m1d_series(market: str, pool: str) -> pd.DataFrame:
    """window4 每季季末的集中度警示（state、cumulative_deviation，index＝quarter_end）。

    台股：市值資料未修改，直接取正式 checkpoint。
    美股：2026-10-28 collector 修正 close 與 outstanding_shares 後，checkpoint 的偏離值已過期
    （狀態相同、數值不同，例 2024Q2 0.817→1.318pp），改用修正後市值＋**凍結門檻**
    （triggers.X_P90/X_P75，使用者裁示不重新校準）重算；警示只看市場，兩個池子共用。結果快取成 CSV。
    """
    from app import simulate, triggers
    if market == "TW":
        rows = {}
        for line in open(RUNS / CKPT[(market, pool)], encoding="utf-8"):
            cp = json.loads(line)
            if cp.get("arm") not in (None, "L2"):
                continue
            rows[cp["quarter_end"]] = {"state": cp["m1d"]["state"],
                                       "cumulative_deviation": cp["m1d"]["cumulative_deviation"]}
        return pd.DataFrame(rows).T.loc[list(simulate.QUARTER_ENDS)]
    path = Path(str(M1D_CACHE).format(market=market))
    if path.exists():
        return pd.read_csv(path, index_col=0)
    from database import Database
    from app import monitor
    wide = monitor.fetch_mcap_wide(Database(market).create_connection(), market=market, start="2009-01-01")
    cond = triggers.register_m1d(wide, simulate.REGISTRATION_DATE, market=market)
    s = triggers.run_quarterly_series(wide, cond, simulate.QUARTER_ENDS).set_index("as_of")
    out = s[["state", "cumulative_deviation"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, encoding="utf-8-sig")
    return out


def m1d_states(market: str, pool: str) -> dict[str, str]:
    """{as_of: 該季季初可見的集中度警示狀態}。第一季季初（登記日）＝NONE。"""
    from app import simulate
    s = m1d_series(market, pool)
    out = {simulate.REGISTRATION_DATE: "NONE"}
    for qe in simulate.QUARTER_ENDS[:-1]:
        out[qe] = s.loc[qe, "state"]
    return out


# ───────────────────────── 挑法（policy） ─────────────────────────

def _split(g: pd.DataFrame, arms):
    hist = g[g.index.get_level_values("window") < 4][arms]
    ev = g[g.index.get_level_values("window") == 4][arms]
    return hist, ev


def run_deterministic(g, arms, rule: str, states=None) -> list[str]:
    """回傳 window4 每季選的代號。rule：fixed:<combo>／winner／loser／trigger／oracle。"""
    hist, ev = _split(g, arms)
    picks = []
    for k, (idx, row) in enumerate(ev.iterrows()):
        prev = hist.iloc[-1] if k == 0 else ev.iloc[k - 1]
        if rule.startswith("fixed:"):
            picks.append(rule.split(":", 1)[1])
        elif rule == "winner":
            picks.append(prev.idxmax())
        elif rule == "loser":
            picks.append(prev.idxmin())
        elif rule == "trigger":
            # 觸發時用的市值規則：選單有「市值・前3名・平均分・資金30%」就用它（B、C），
            # 否則用選單中唯一的市值規則（A 的市值家族最佳）
            mc = [a for a in arms if a.startswith("S6_mcap")]
            arm = "S6_mcap|N3|equal|sl30" if "S6_mcap|N3|equal|sl30" in arms else (mc[0] if len(mc) == 1 else None)
            if arm is None:
                raise AssertionError(f"選單裡找不到唯一的市值規則：{mc}")
            on = states[idx[2]] == "TRIGGERED"
            picks.append(arm if on else "REF_control0")
        elif rule == "oracle":
            picks.append(row.idxmax())
        else:
            raise ValueError(rule)
    return picks


def run_bandit(g, arms, kind: str, seed: int, eps: float = 0.1) -> list[str]:
    """ε-greedy／Thompson（常態近似）。window1~3 的季報酬當先驗觀測。"""
    rng = np.random.default_rng(seed)
    hist, ev = _split(g, arms)
    obs = {a: list(hist[a].dropna()) for a in arms}
    picks = []
    for _, row in ev.iterrows():
        if kind == "random":
            a = arms[rng.integers(len(arms))]
        elif kind == "egreedy":
            a = arms[rng.integers(len(arms))] if rng.random() < eps else max(arms, key=lambda x: np.mean(obs[x]))
        elif kind == "thompson":
            draws = {x: rng.normal(np.mean(obs[x]), np.std(obs[x], ddof=1) / np.sqrt(len(obs[x]))) for x in arms}
            a = max(draws, key=draws.get)
        else:
            raise ValueError(kind)
        picks.append(a)
        obs[a].append(row[a])
    return picks


def settle(g, c, picks: list[str]) -> dict:
    """依每季選擇結算 window4：未扣成本與近似扣成本的累積報酬、每季報酬。"""
    ev_g = g[g.index.get_level_values("window") == 4]
    ev_c = c[c.index.get_level_values("window") == 4]
    rg = np.array([ev_g.iloc[k][p] for k, p in enumerate(picks)])
    rn = np.array([(1 - ev_c.iloc[k][p]) * (1 + ev_g.iloc[k][p]) - 1 for k, p in enumerate(picks)])
    return {"cum_gross": float(np.prod(1 + rg) - 1), "cum_net_approx": float(np.prod(1 + rn) - 1),
            "q_gross": rg.tolist(), "picks": picks}


# ───────────────────────── 實際換手成本（2026-10-28 code review 待辦②） ─────────────────────────

class WeightBook:
    """還原 window4 每季每個規則的實際權重，用實際價格算季內報酬與季末漂移，
    依「上一季季末實際持股 → 這一季目標持股」計算換規則時的真實換手成本。

    起點＝window3 最後一季的 control0 持股漂移到 2023-12-31（跟步驟1 網格「窗間延續持股」一致：
    評估期開始前，手上是研究部現行做法的持股）。成本參數同步驟1（`_elite_grid_offline.COSTS`）。
    """

    def __init__(self, market: str, pool: str, close: pd.DataFrame, with_sig: bool = False):
        from app import simulate
        from app._elite_grid_offline import COSTS
        tag = TAG[(market, pool)]
        w0 = pd.read_parquet(GRID / f"w0_{tag}.parquet")
        el = pd.read_parquet(GRID / f"elite_weights_{tag}.parquet")
        if with_sig:
            el = pd.concat([el, pd.read_parquet(GRID / f"elite_weights_{tag}_sig.parquet")])
        self.close = close
        self.buy_c, self.sell_c = COSTS[market]
        self.as_ofs = [simulate.REGISTRATION_DATE] + list(simulate.QUARTER_ENDS[:-1])
        self.ends = list(simulate.QUARTER_ENDS)
        self.w0 = {a: dict(zip(d.symbol, d.w0)) for a, d in w0[w0.window == 4].groupby("as_of")}
        self.el = {key: dict(zip(d.symbol, d.elite_weight))
                   for key, d in el[el.window == 4].groupby(["as_of", "score", "n", "scheme"])}
        qf = [GRID / f"quarterly_{tag}.parquet"] + ([GRID / f"quarterly_{tag}_sig.parquet"] if with_sig else [])
        self.known = set().union(*[set(pd.read_parquet(f, columns=["combo"]).combo) for f in qf])
        w3 = w0[w0.window == 3]
        last = w3.as_of.max()
        start = dict(zip(w3[w3.as_of == last].symbol, w3[w3.as_of == last].w0))
        _, self.start = self._drift(start, last, self.as_ofs[0])
        self._cache: dict = {}

    def _drift(self, w: dict, as_of: str, end: str):
        from app._elite_grid_offline import price_ratios
        syms = sorted(w)
        R = price_ratios(self.close, syms, as_of, end).iloc[-1]
        vals = {s: w[s] * float(R[s]) for s in syms}
        nav = (1.0 - sum(w.values())) + sum(vals.values())
        return nav - 1.0, {s: v / nav for s, v in vals.items()}

    def target(self, k: int, combo: str) -> dict:
        from app import elite_rules as ER
        a = self.as_ofs[k]
        w0 = self.w0[a]
        if combo == "REF_control0":
            return dict(w0)
        if combo not in self.known:   # 2026-10-28 code review：名稱錯或網格沒載入就報錯（空的優等生是合法情況，名稱不存在不是）
            raise KeyError(f"規則 {combo} 不在已載入的網格裡（with_sig 是否正確？）")
        score, n, scheme, sl = combo.split("|")
        elite = self.el.get((a, score, int(n[1:]), scheme), {})
        return ER.overlay(w0, elite, int(sl[2:]) / 100)

    def outcome(self, k: int, key, weights: dict | None = None):
        """(季報酬, 季末漂移後權重)。key 為規則代號（自動還原權重）或任意可快取的鍵（需給 weights）。"""
        if (k, key) not in self._cache:
            w = weights if weights is not None else self.target(k, key)
            self._cache[(k, key)] = (w, *self._drift(w, self.as_ofs[k], self.ends[k]))
        return self._cache[(k, key)]

    def _cost(self, w: dict, prev: dict) -> float:
        diff = {s: w.get(s, 0.0) - prev.get(s, 0.0) for s in set(w) | set(prev)}
        return sum(max(d, 0) for d in diff.values()) * self.buy_c + sum(max(-d, 0) for d in diff.values()) * self.sell_c

    def _ratios(self, k: int, syms: list[str]) -> pd.DataFrame:
        """第 k 季的逐日價格比值；以該季 control0 持股為底快取（優等生與方案D 候選都是持股的子集）。"""
        from app._elite_grid_offline import price_ratios
        if not hasattr(self, "_rcache"):
            self._rcache = {}
        if k not in self._rcache:
            self._rcache[k] = price_ratios(self.close, sorted(self.w0[self.as_ofs[k]]), self.as_ofs[k], self.ends[k])
        R = self._rcache[k]
        missing = [s for s in syms if s not in R.columns]
        if missing:
            R = pd.concat([R, price_ratios(self.close, missing, self.as_ofs[k], self.ends[k])], axis=1)
        return R[syms]

    def daily(self, keys: list, weights_list: list | None = None) -> pd.DataFrame:
        """逐日淨值（欄：gross、net；起點 1）。成本在季初一次扣，跟 `settle` 同一套權重、價格與成本，
        最後一天＝`settle` 的累積（2026-10-29 新增，供日頻 MDD／Sharpe 與累積曲線圖）。"""
        prev, g, n = self.start, 1.0, 1.0
        parts = []
        for k, key in enumerate(keys):
            w, _, drifted = self.outcome(k, key, None if weights_list is None else weights_list[k])
            cost = self._cost(w, prev)
            syms = sorted(w)
            R = self._ratios(k, syms)
            v = (1.0 - sum(w.values())) + R.to_numpy() @ np.array([w[s] for s in syms])
            q = pd.DataFrame({"gross": g * v, "net": n * (1 - cost) * v}, index=R.index)
            if k == 0:
                q.iloc[0] = [1.0, 1.0]          # 起點（季初建倉前）
                parts.append(q.iloc[:1])
            parts.append(q.iloc[1:])
            g, n = float(q.gross.iloc[-1]), float(q.net.iloc[-1])
            prev = drifted
        return pd.concat(parts)

    def settle(self, keys: list, weights_list: list | None = None) -> dict:
        prev = self.start
        rg, rn, costs = [], [], []
        for k, key in enumerate(keys):
            w, r, drifted = self.outcome(k, key, None if weights_list is None else weights_list[k])
            cost = self._cost(w, prev)
            rg.append(r); rn.append((1 - cost) * (1 + r) - 1); costs.append(cost)
            prev = drifted
        return {"cum_gross": float(np.prod(np.add(rg, 1)) - 1), "cum_net": float(np.prod(np.add(rn, 1)) - 1),
                "q_gross": rg, "q_net": rn, "costs": costs, "picks": list(keys)}
