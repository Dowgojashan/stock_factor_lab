# -*- coding: utf-8 -*-
"""L4 的最小可行版（CLI，UI 還沒做）——Phase A 的可展示成果。

用法：
    cd code
    python -m app.cli new-config out.json --market TW --scheme A --window-no 6
    python -m app.cli run out.json
    python -m app.cli run out.json --override-reason "已知台積電權值集中，本次接受"
    python -m app.cli run out.json --no-record   # 只看數字，不寫稽核紀錄
"""
from __future__ import annotations

import argparse
import sys

from .audit import record
from .calibration import check as check_calibration
from .config import RunConfig, ReplayAnchor
from .engine import run as run_engine
from .memo import generate as generate_memo
from .risk import assess


def cmd_new_config(args: argparse.Namespace) -> int:
    # 2026-09-10（§7 P2）：加上正式模式。正式模式沒有 scheme/window_no
    # （IS 依定義就是錨點到最新可用月），k_mode 固定 mainline_h03。
    if args.mode == "live":
        cfg = RunConfig(mode="live", market=args.market, group=args.group,
                        ratio=args.ratio, allocation=args.allocation,
                        k_mode="mainline_h03")
    else:
        cfg = RunConfig(
            mode="replay", market=args.market, group=args.group,
            ratio=args.ratio, allocation=args.allocation, k_mode=args.k_mode,
            replay_anchor=ReplayAnchor(scheme=args.scheme, window_no=args.window_no),
        )
    cfg.save(args.out)
    print(f"已建立 {args.out}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = RunConfig.load(args.config)
    holdings = run_engine(cfg)
    risk = assess(holdings)

    print(f"=== 執行結果（{cfg.mode}｜{cfg.market}｜{cfg.group}｜"
         f"{cfg.ratio}／{cfg.allocation}）===")
    print(f"窗次：{holdings.window_info}")
    print(f"持股（策略層級）：{holdings.n_members} 檔")
    print(f"績效：{holdings.performance}")
    print()
    print(f"=== 風控檢查 ===")
    print(f"單檔權重（等權）：{risk.max_single_weight:.2%}（上限 {cfg.single_stock_cap:.2%}）")
    print(f"最大群佔比：{risk.max_cluster_share:.2%}（上限 {cfg.cluster_cap:.2%}，"
         f"{cfg.allocation} 分配）")
    print(f"涵蓋群數：{risk.n_clusters_covered}")

    calib = check_calibration(holdings)
    print()
    print(f"=== 校準監控（C5，對 {calib.thresholds.n_cells} 個歷史格子的 p10 分位）===")
    if calib.evaluable:
        print(f"OOS CAGR：{calib.oos_cagr:.2%}（p10 門檻 {calib.thresholds.oos_cagr_p10:.2%}）"
             f"{' ⚠️ 低於門檻' if calib.below_cagr else ''}")
        print(f"OOS Calmar：{calib.oos_calmar:.3f}（p10 門檻 {calib.thresholds.oos_calmar_p10:.3f}）"
             f"{' ⚠️ 低於門檻' if calib.below_calmar else ''}")
        if calib.flagged:
            print("⚠️ 這次表現明顯偏離歷史常態分布，只是提醒，不會攔下執行"
                 "（校準監控是示警，不是 C4 那種強制關卡）")
    else:
        # §8-R3：正式模式沒有 OOS，不假裝判定得出來
        print(f"狀態：{calib.status}（門檻已立起來當日後基準，本次不判定）")
        print(f"  {calib.note}")

    if args.no_record:
        # 2026-09-09 code review：純預覽/探索設定用，不寫稽核紀錄——沒有要
        # 正式執行就不用被 C4 的 override 流程卡住，也不會弄髒 F2 的歷史紀錄
        # （這支 CLI 原本每次 run 都無條件 record，跟 ui.py「執行」/「寫入
        # 稽核紀錄」分開兩顆按鈕的設計不一致，開發期間就因此髒過一次）。
        if risk.violations:
            print()
            print("🔴 風控違規（僅預覽，--no-record 不寫入稽核紀錄，不需要 override-reason）：")
            for v in risk.violations:
                print(f"  - {v.detail}")
        print("\n（--no-record：本次未寫入稽核紀錄）")
    else:
        if risk.violations:
            print()
            print("🔴 風控違規，本次執行被攔下：")
            for v in risk.violations:
                print(f"  - {v.detail}")
            if not args.override_reason:
                print()
                print("如要放行，加 --override-reason \"原因\" 重跑一次"
                     "（C4：人工覆核，會強制寫進稽核紀錄）；"
                     "如果只是想看數字不做正式執行，改用 --no-record")
                return 1
            print(f"\n人工覆核放行，原因：{args.override_reason}")

        entry = record(holdings, risk, calibration=calib, override_reason=args.override_reason)
        print(f"\n已寫入稽核紀錄：{entry['recorded_at']}")

    if args.memo:
        print()
        print(f"=== AI 備忘錄（D1/D2{'，dry-run，未花錢呼叫 LLM' if not args.live_llm else ''}）===")
        try:
            result = generate_memo(holdings, risk, calib, dry_run=not args.live_llm,
                                   purpose=args.llm_purpose)
        except RuntimeError as e:
            # D2：洩漏掃描是強制關卡，攔下就攔下，不印醜的 traceback
            print(f"🔴 {e}")
            return 1
        for k, v in result["memo"].items():
            print(f"[{k}] {v}")
        if args.live_llm and cfg.mode == "replay":
            print("\n✅ D2 洩漏掃描通過（memo 裡的數字都能對回判決資料）")

    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="app.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_new = sub.add_parser("new-config", help="產生一份起始 RunConfig（replay 模式）")
    p_new.add_argument("out")
    p_new.add_argument("--mode", default="replay", choices=["live", "replay"],
                       help="live＝正式模式（全歷史建的樹、無 OOS）；"
                            "replay＝驗證模式（凍結窗次、有 OOS）")
    p_new.add_argument("--market", default="TW", choices=["TW", "US", "XM"])
    p_new.add_argument("--group", default="A_hrp",
                       choices=["A_hrp", "D_top_cagr", "E_top_calmar"])
    p_new.add_argument("--ratio", default="legacy")
    p_new.add_argument("--allocation", default="equal", choices=["equal", "proportional"])
    p_new.add_argument("--k-mode", dest="k_mode", default="silhouette_is",
                       choices=["fixed", "silhouette_is"])
    p_new.add_argument("--scheme", default="A")
    p_new.add_argument("--window-no", dest="window_no", type=int, default=6)
    p_new.set_defaults(func=cmd_new_config)

    p_run = sub.add_parser("run", help="讀 RunConfig 執行一次")
    p_run.add_argument("config")
    p_run.add_argument("--override-reason", default=None)
    p_run.add_argument("--no-record", action="store_true",
                       help="只預覽這組設定的結果，不寫入稽核紀錄（F2）；"
                            "有違規也不需要 --override-reason")
    p_run.add_argument("--memo", action="store_true",
                       help="順便產生 AI 備忘錄（預設 dry-run，不花錢）")
    p_run.add_argument("--live-llm", action="store_true",
                       help="配合 --memo：真的呼叫 LLM（會花錢，需要 config.ini "
                            "設好 [openai] {purpose}_model）")
    p_run.add_argument("--llm-purpose", dest="llm_purpose", default="app_memo",
                       help="決定去 config.ini 讀哪組模型設定跟額度記帳（預設 app_memo，"
                            "尚未設定時可借用既有 purpose，例如 cluster_story）")
    p_run.set_defaults(func=cmd_run)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
