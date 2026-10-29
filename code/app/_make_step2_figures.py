# -*- coding: utf-8 -*-
"""步驟2（AI 調股票權重）論文用圖表（2026-10-29，實戰收尾追蹤.md §四步驟2）。

讀 `_step2_phase2_settle.py`（逐日淨值、權重矩陣、每條鏈結果）與 `_step2_evaluate.py`（判定）的輸出。
畫哪些線、哪些組在看結果之前就固定（不挑表現好的）：
  圖1 每季報酬長條（老師 10/27「每期各方法並排」）：大盤／control0／訊號規則（不用 LLM）／方案A~D（各 3 風格×5 次平均）
  圖2 2024–25 逐日累積淨值（同一組線）
  圖3 每個方案×風格 5 次重複的扣成本累積（點），對照 control0、隨機挑 95 百分位、不用 LLM 最佳
  圖4 版本2 減版本1（方案×風格的 5 次平均扣成本累積差，個百分點）
  圖5 權重矩陣示例（時間×股票）：兩版都判有效的唯一一項「台股・排除V1 分工混合・追強型」第1次，對照 control0
  表1 每種挑法的扣成本累積、日頻 Sharpe、日頻 MDD 與判定（CSV＋Markdown）
全部扣成本後；大盤為總報酬指數、不扣成本。輸出 `_analysis_outputs_applayer/step2/figs_v{版本}/`。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402,F401
import pandas as pd  # noqa: E402

from app import rule_selector as RS  # noqa: E402
from app._make_elite_grid_figures import BENCH_LABEL, GROUP_LABEL, GROUPS, INK, MUTED, GRID  # noqa: E402

OUT = RS.ROOT / "_analysis_outputs_applayer" / "step2"
MODEL = "gpt-5-mini"
PLAN_LABEL = {"A": "方案A 規則選擇", "B": "方案B 參數設定", "C": "方案C 資金旋鈕", "D": "方案D 個股評分"}
STYLE_LABEL = {"balanced": "平衡型", "momentum": "追強型", "contrarian": "逆勢型"}
SIGNAL_RULE = {1: "警示觸發換市值", 2: "訊號規則（三訊號）"}
SERIES = [("bench", "大盤", "#55554f"), ("c0", "control0", "#2a78d6"), ("sig", None, "#8a8a85"),
          ("A", PLAN_LABEL["A"], "#eb6834"), ("B", PLAN_LABEL["B"], "#1baf7a"),
          ("C", PLAN_LABEL["C"], "#eda100"), ("D", PLAN_LABEL["D"], "#e87ba4")]


def pre(v):
    return "phase2v2" if v == 2 else "phase2"


def load(v, m, p):
    nav = pd.read_parquet(OUT / f"{pre(v)}_daily_{m}_{p}_{MODEL}.parquet")
    st = pd.read_csv(OUT / f"{pre(v)}_settled_{m}_{p}_{MODEL}.csv")
    return nav, st


def series_navs(v, m, p, nav):
    """固定的那幾條線（逐日淨值）。方案＝該方案 15 條鏈（3 風格×5 次）的淨值平均。"""
    cols = pd.Series(nav.columns)
    out = {"bench": nav["大盤|-|-|0"], "c0": nav["不用LLM|A|固定：REF_control0|0"],
           "sig": nav[f"不用LLM|A|{SIGNAL_RULE[v]}|0"]}
    for plan in "ABCD":
        cs = cols[cols.str.startswith(f"方案|{plan}|")]
        assert len(cs) == 15, (m, p, plan, len(cs))
        out[plan] = nav[cs].mean(axis=1)
    return out


def quarter_returns(s: pd.Series):
    from app import simulate
    ends = [pd.Timestamp(e) for e in simulate.QUARTER_ENDS]
    vals = [float(s.iloc[0])] + [float(s[s.index <= e].iloc[-1]) for e in ends]
    return np.array(vals[1:]) / np.array(vals[:-1]) - 1


def fig1_fig2(v, fd):
    fig1, ax1 = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
    fig2, ax2 = plt.subplots(2, 2, figsize=(13, 7.5))
    for i, (m, p) in enumerate(GROUPS):
        nav, _ = load(v, m, p)
        ss = series_navs(v, m, p, nav)
        a1, a2 = ax1.flat[i], ax2.flat[i]
        x = np.arange(8); wdt = 0.8 / len(SERIES)
        for j, (key, lab, col) in enumerate(SERIES):
            lab = lab or SIGNAL_RULE[v] + "（不用LLM）"
            lab = BENCH_LABEL[m] if key == "bench" else lab
            a1.bar(x - 0.4 + wdt * (j + 0.5), quarter_returns(ss[key]) * 100, wdt * 0.92, color=col, label=lab)
            a2.plot(ss[key].index, (ss[key] - 1) * 100, color=col, lw=2.2 if key in ("bench", "c0") else 1.6,
                    ls="--" if key == "sig" else "-", label=f"{lab} {float(ss[key].iloc[-1] - 1):+.1%}")
        a1.axhline(0, color=MUTED, lw=0.8)
        a1.set_xticks(x, [f"第{k}季" for k in range(1, 9)])
        a1.set_title(GROUP_LABEL[(m, p)], loc="left", color=INK); a1.set_ylabel("季報酬（%）")
        a1.yaxis.grid(True, color=GRID, lw=0.6)
        a2.set_title(GROUP_LABEL[(m, p)], loc="left", color=INK); a2.set_ylabel("累積報酬（%）")
        a2.yaxis.grid(True, color=GRID, lw=0.6); a2.legend(fontsize=7.5, frameon=False, loc="upper left")
    h, l = ax1.flat[0].get_legend_handles_labels()
    l = [x.replace(BENCH_LABEL["TW"], "大盤（TAIEX／S&P500）") for x in l]
    fig1.legend(h, l, ncol=7, loc="upper center", frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig1.suptitle(f"圖1　2024–25 每季報酬（扣成本；方案為 3 風格×5 次平均；版本{v}）", y=1.04, color=INK)
    fig1.tight_layout(); fig1.savefig(fd / "fig1_quarterly_bars.png", bbox_inches="tight"); plt.close(fig1)
    fig2.suptitle(f"圖2　2024–25 逐日累積報酬（扣成本；版本{v}）", color=INK)
    fig2.tight_layout(); fig2.savefig(fd / "fig2_cumulative.png", bbox_inches="tight"); plt.close(fig2)


def fig3(v, fd, verdict):
    fig, axs = plt.subplots(2, 2, figsize=(13, 7.5))
    for i, (m, p) in enumerate(GROUPS):
        _, st = load(v, m, p)
        g = verdict[verdict.組別 == f"{m}・{p}"]
        ax = axs.flat[i]
        cats = [(pl, s) for pl in "ABCD" for s in ("balanced", "momentum", "contrarian")]
        for j, (pl, s) in enumerate(cats):
            y = st[(st.kind == "方案") & (st.plan == pl) & (st["style"] == s)].cum_net * 100
            ax.scatter(np.full(len(y), j), y, s=22, color=SERIES[3 + "ABCD".index(pl)][2], zorder=3)
            ax.plot([j - 0.3, j + 0.3], [y.mean()] * 2, color=INK, lw=1.4, zorder=4)
            row = g[(g.類型 == "方案") & (g.方案 == pl) & (g.風格 == STYLE_LABEL[s])].iloc[0]
            ax.plot([j - 0.4, j + 0.4], [row.隨機挑95百分位 * 100] * 2, color="#c0392b", lw=1, ls=":")
            ax.plot([j - 0.4, j + 0.4], [row.不用LLM最佳 * 100] * 2, color=MUTED, lw=1, ls="--")
            if row.判定 == "有效":
                ax.annotate("有效", (j, y.max()), xytext=(0, 6), textcoords="offset points", ha="center",
                            fontsize=8, color="#008300")
        c0 = float(g.control0.iloc[0]) * 100
        ax.axhline(c0, color="#2a78d6", lw=1.2)
        ax.set_xticks(range(len(cats)), [f"{pl}\n{STYLE_LABEL[s]}" for pl, s in cats], fontsize=8)
        ax.set_title(GROUP_LABEL[(m, p)], loc="left", color=INK); ax.set_ylabel("8 季累積（扣成本，%）")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="o", ls="", color=MUTED, label="每次重複"), Line2D([], [], color=INK, label="5 次平均"),
          Line2D([], [], color="#2a78d6", label="control0"), Line2D([], [], color="#c0392b", ls=":", label="隨機挑 95 百分位"),
          Line2D([], [], color=MUTED, ls="--", label="不用 LLM 的方法中最高")]
    fig.legend(handles=hs, ncol=5, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"圖3　各方案×風格 5 次重複的結果（版本{v}；三條標準都過才標「有效」）", y=1.04, color=INK)
    fig.tight_layout(); fig.savefig(fd / "fig3_reps_vs_bars.png", bbox_inches="tight"); plt.close(fig)


def fig4(fd):
    v1 = pd.read_csv(OUT / f"phase2_verdict_v1_{MODEL}.csv"); v2 = pd.read_csv(OUT / f"phase2_verdict_v2_{MODEL}.csv")
    k = ["組別", "類型", "方案", "風格"]
    d = v1[k + ["扣成本平均"]].merge(v2[k + ["扣成本平均"]], on=k, suffixes=("1", "2"))
    d = d[d.類型 == "方案"]; d["差"] = (d.扣成本平均2 - d.扣成本平均1) * 100
    rows = [f"{m}・{p}" for m, p in GROUPS]
    cols = [(pl, s) for pl in "ABCD" for s in ("平衡型", "追強型", "逆勢型")]
    M = np.array([[float(d[(d.組別 == r) & (d.方案 == pl) & (d.風格 == s)].差.iloc[0]) for pl, s in cols] for r in rows])
    lim = np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=(12, 3.6))
    im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.0f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(M[i, j]) > lim * 0.6 else INK)
    ax.set_xticks(range(len(cols)), [f"{pl}\n{s}" for pl, s in cols], fontsize=8)
    ax.set_yticks(range(len(rows)), [GROUP_LABEL[g] for g in GROUPS])
    fig.colorbar(im, ax=ax, label="版本2 減 版本1（個百分點）")
    ax.set_title("圖4　加入三種訊號後的變化（5 次平均扣成本累積；藍＝變好、紅＝變差）", loc="left", color=INK)
    fig.tight_layout(); fig.savefig(fd / "fig4_v2_minus_v1.png", bbox_inches="tight"); plt.close(fig)


def fig5(v, fd):
    wt = pd.read_parquet(OUT / f"{pre(v)}_weights_TW_exclude_v1_{MODEL}.parquet")
    wt["chain"] = wt.chain.astype(str)
    ai = wt[wt.chain == "分工混合|A×C|momentum|1"].pivot_table(index="symbol", columns="k", values="weight", fill_value=0)
    c0 = wt[wt.chain == "不用LLM|A|固定：REF_control0|0"].pivot_table(index="symbol", columns="k", values="weight", fill_value=0)
    ai, c0 = ai.reindex(columns=range(1, 9), fill_value=0), c0.reindex(columns=range(1, 9), fill_value=0)
    top = ai.max(axis=1).sort_values(ascending=False).index[:15]
    A = ai.loc[top] * 100; C = c0.reindex(top).fillna(0) * 100
    vmax = float(A.to_numpy().max())
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    for ax, M, t in ((axs[0], C, "control0（現行，不調整）"), (axs[1], A, "AI：分工混合・追強型（第1次）")):
        ax.imshow(M.to_numpy(), cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M.iat[i, j]
                ax.text(j, i, f"{val:.1f}" if val >= 0.05 else "", ha="center", va="center", fontsize=7,
                        color="white" if val > vmax * 0.6 else INK)
        ax.set_xticks(range(8), [f"第{k}季" for k in range(1, 9)], fontsize=8); ax.set_title(t, loc="left", color=INK)
    axs[0].set_yticks(range(len(top)), list(top), fontsize=8)
    fig.suptitle("圖5　權重矩陣示例：台股・排除V1，AI 權重最高的 15 檔（數字＝權重 %）", color=INK)
    fig.tight_layout(); fig.savefig(fd / "fig5_weight_matrix_example.png", bbox_inches="tight"); plt.close(fig)


def table1(v, fd, verdict):
    t = verdict.copy()
    cols = ["組別", "類型", "方案", "風格", "扣成本平均", "最低", "最高", "日頻Sharpe平均（參考）", "日頻MDD平均（參考）",
            "control0", "隨機挑95百分位", "不用LLM最佳", "贏control0次數", "判定"]
    t = t[cols]
    t.to_csv(fd / "table1_all_policies.csv", index=False, encoding="utf-8-sig")
    s = t.copy()
    for c in ["扣成本平均", "最低", "最高", "日頻MDD平均（參考）", "control0", "隨機挑95百分位", "不用LLM最佳"]:
        s[c] = s[c].map(lambda x: f"{x:+.1%}")
    s["日頻Sharpe平均（參考）"] = s["日頻Sharpe平均（參考）"].map(lambda x: f"{x:.2f}")
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(x) for x in r) + " |" for r in s.itertuples(index=False)]
    (fd / "table1_all_policies.md").write_text(f"# 表1　步驟2 版本{v} 全部挑法（扣成本 8 季累積）\n\n" + "\n".join(md) + "\n",
                                               encoding="utf-8")


def main():
    for v in (1, 2):
        fd = OUT / f"figs_v{v}"; fd.mkdir(parents=True, exist_ok=True)
        verdict = pd.read_csv(OUT / f"phase2_verdict_v{v}_{MODEL}.csv")
        fig1_fig2(v, fd); fig3(v, fd, verdict); fig5(v, fd); table1(v, fd, verdict)
        print("ok", fd)
    fig4(OUT / "figs_v2")


if __name__ == "__main__":
    main()
