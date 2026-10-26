# -*- coding: utf-8 -*-
"""2026-10-27（第三批）：台股「真實L2 vs W2c全開（強制每季套用，cap=30%）
vs 大盤」逐季報酬率比較圖，用來支持「機制本身有效、但agent目前選擇性觸發
（M1-D閘門）比每季強制套用更保守」這個說法。

資料來源：
①真實L2（exclude_v1）—— 直接讀production checkpoint
  `simulate_anchored_w4_openSec_boost_exclude_v1_cap30_hotseg_L2.jsonl`，
  跟`_make_report_charts_1027.py`同一組資料，不重算。
②W2c全開 —— 複用`_cap_sweep_window4.py`的`sweep_market`邏輯，只取
  cap=30%那組，但改成回傳逐季報酬而非只回傳累積數字（該腳本原本只存
  累積cum_pct，這裡重新跑一次抓逐季序列，計算方式完全相同，無LLM、
  無M2，純機械式tilt_weights+cap_and_redistribute）。
③大盤 —— 跟①②同一組9個日期查`taiex_tr`，與`_make_report_charts_1027.py`
  的`benchmark_quarterly_returns`相同手法。

用法（cwd必須是code/；PYTHONIOENCODING=utf-8）：
    PYTHONIOENCODING=utf-8 python -m app._make_report_charts_1027c
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from app import monitor, simulate  # noqa: E402
from app import weights as weights_mod  # noqa: E402
from app._check_rolling_production_simulation import build_rolling_inputs  # noqa: E402
from app._make_report_charts_1027 import (
    Q_LABELS, benchmark_quarterly_returns, quarterly_returns_from_checkpoint,
)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 11,
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

APPLAYER = Path(__file__).resolve().parent.parent.parent / "_analysis_outputs_applayer"
OUT_DIR = APPLAYER / "report_charts_1027"
CAP = 0.30
TW_PICKS = APPLAYER / "anchored_w4_openSec_boost_exclude_v1.parquet"
TW_L2_CKPT = "app/_runs/simulate_anchored_w4_openSec_boost_exclude_v1_cap30_hotseg_L2.jsonl"


def w2c_full_on_quarterly_returns() -> list[float]:
    inputs = build_rolling_inputs(model="unused", api_key="unused",
                                  picks_path=TW_PICKS, tree_key="TW",
                                  scheme="E", window_no=4)
    uids = inputs.uids_by_allocation["equal"]

    periods = [(simulate.REGISTRATION_DATE, simulate.QUARTER_ENDS[0])]
    periods += list(zip(simulate.QUARTER_ENDS[:-1], simulate.QUARTER_ENDS[1:]))

    rets = []
    for as_of, end in periods:
        w0 = simulate.resolve_weights(inputs.md, inputs.idx, uids, as_of)
        mcap_row = monitor.asof_row(inputs.mcap_wide, as_of)
        raw = weights_mod.tilt_weights(w0, mcap_row, 0.0)
        w_capped = weights_mod.cap_and_redistribute(raw, cap=CAP)
        r = monitor.outcome_layer(inputs.md_map, w_capped, as_of, end)["portfolio_realized_return"]
        rets.append(r)
    return rets


def main():
    l2 = quarterly_returns_from_checkpoint(TW_L2_CKPT)
    w2c_full = w2c_full_on_quarterly_returns()
    taiex = benchmark_quarterly_returns("taiex_tr")

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = range(len(Q_LABELS))
    series = {
        "真實L2（M1-D選擇性觸發）": dict(vals=l2, marker="o", color="#2980b9", dy=14),
        "W2c全開（強制每季套用，cap=30%）": dict(vals=w2c_full, marker="D", color="#c0652b", dy=-18),
        "大盤（TAIEX）": dict(vals=taiex, marker="^", color="#7f8c8d", dy=14, linestyle="--"),
    }
    for label, cfg in series.items():
        pct = [v * 100 for v in cfg["vals"]]
        ax.plot(x, pct, label=label, linewidth=2, markersize=6,
                marker=cfg["marker"], color=cfg["color"],
                linestyle=cfg.get("linestyle", "-"))
        for xi, yi in zip(x, pct):
            ax.annotate(f"{yi:+.1f}", (xi, yi), textcoords="offset points",
                       xytext=(0, cfg["dy"]), ha="center", fontsize=7.5,
                       color=cfg["color"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(Q_LABELS)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("季報酬率（%）")
    ax.set_title("台股：真實L2 vs W2c全開 vs 大盤 逐季報酬率（2024Q1~2025Q4）")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path = OUT_DIR / "08_tw_w2c_full_on_quarterly.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"寫入 {out_path}")


if __name__ == "__main__":
    main()
