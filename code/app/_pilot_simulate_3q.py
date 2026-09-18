# -*- coding: utf-8 -*-
"""正式實驗前的小規模試跑（2026-09-18，使用者授權）：control0＋L2 兩臂 × 3 季，
真實 LLM 呼叫。目的：驗證 §12.3 的成本估計（單季樣本推算約 0.47 天額度）在
memory 隨季度增長後有沒有顯著偏離——這是使用者在決定要不要直接跑滿 8 季之前
明確要求先做的檢查點，不是要留存的正式產物腳本。

同時驗證 D46 的兩個修正：
  ① control0 臂應該完全不出現在 llm_usage.jsonl 裡（不再燒 token）
  ② excess_vs_ball 欄位存取不炸（M4 診斷正常運作）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fcv_core  # noqa: E402
from app import simulate  # noqa: E402
from utils.config import Config  # noqa: E402

RUN_ID = "_pilot_3q"
LEDGER_PATH = Path(__file__).resolve().parent.parent / "_catalog" / "llm_usage.jsonl"


def _read_ledger_tail(n_bytes_offset: int) -> list[dict]:
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        f.seek(n_bytes_offset)
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = simulate.checkpoint_path(RUN_ID)
    if p.exists():
        p.unlink()

    cfg = Config()
    api_key = cfg.get_openai_api_key()
    model = cfg.get_openai_model("app_memo")

    offset = LEDGER_PATH.stat().st_size if LEDGER_PATH.exists() else 0

    print(f"使用模型：{model}")
    print("=== 小規模試跑：control0 + L2，3 季真實呼叫（使用者已授權）===")
    simulate.run_simulation(RUN_ID, model=model, api_key=api_key,
                            dry_run=False, max_quarters=3)

    print("\n=== D46 驗證①：control0 這次跑完全不應出現在 usage ledger 裡 ===")
    entries = _read_ledger_tail(offset)
    print(f"本次試跑總共 {len(entries)} 次真實呼叫（3 季 × L2 4 次 = 預期 12 次，"
         "control0 應為 0 次）")
    for e in entries:
        print(f"  {e['ts']}  {e['purpose']:<20s} total={e['total_tokens']}")

    print("\n=== D46 驗證②：checkpoint 裡 M4 診斷正常，excess_vs_ball 有值 ===")
    checkpoints = simulate.load_checkpoints(RUN_ID)
    for c in checkpoints:
        m4 = c["diagnosis"]["region_a_attributable"]["M4"]
        print(f"  {c['arm']} / {c['quarter_end']}：excess_vs_ball="
             f"{c['outcome'].get('excess_vs_ball')}, M4={m4.get('triggered', m4.get('reason'))}")

    print("\n=== 依季彙總 token 用量（看 memory 增長的影響）===")
    # L2 每季固定 4 次呼叫（3b/3a/決策/總結），依序切成 3 組
    per_quarter = [entries[i:i + 4] for i in range(0, len(entries), 4)]
    for i, group in enumerate(per_quarter, start=1):
        total = sum(e["total_tokens"] for e in group)
        by_purpose = {e["purpose"]: e["total_tokens"] for e in group}
        print(f"  第{i}季 L2：合計 {total} tokens  {by_purpose}")

    print("\n全部完成。")


if __name__ == "__main__":
    main()
