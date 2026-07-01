# io_persistence.py
from pathlib import Path
import pickle
import pandas as pd
import re
import warnings
from typing import Dict
from report import Report  # 讓 pickle 能正確還原 Report 物件

# -------------------------------
# 內部：路徑/檔名安全處理
# -------------------------------
_ILLEGAL = r'[\/:*?"<>|]'

def _sanitize(s: str) -> str:
    """將路徑/檔名不合法字元替換為底線；順手 strip 空白."""
    if s is None:
        return "None"
    s = str(s).strip()
    s = re.sub(_ILLEGAL, "_", s)
    # 也避免 Windows 尾端是空白或句點的狀況
    return s.rstrip(" .")

def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def _to_parquet_or_csv(df: pd.DataFrame, out_path: Path, to_parquet: bool):
    """
    嘗試輸出為 parquet；若缺依賴或失敗，降級為 csv。
    to_parquet=True 時才會嘗試 parquet。
    """
    if df is None:
        return
    if not isinstance(df, pd.DataFrame):
        return

    if to_parquet:
        try:
            df.to_parquet(out_path.with_suffix(".parquet"))
            return
        except Exception as e:
            warnings.warn(f"[io_persistence] to_parquet 失敗，改用 CSV。原因：{e}")

    # fallback -> CSV
    # trades 是 row-oriented 的事件表，通常不需要 index；其餘 timeseries 建議保留 index
    if "trades" in str(out_path).lower():
        df.to_csv(out_path.with_suffix(".csv"), index=False)
    else:
        df.to_csv(out_path.with_suffix(".csv"))

# -------------------------------
# 1) 存/讀整包 ReportCollection（pickle）
# -------------------------------
def save_report_collection_pickle(report_collection, out_path: Path):
    """
    直接將 ReportCollection Pickle 化存檔。
    注意：out_path 可含未清理的 label/資料夾名稱，此函式會自動淨化上層路徑名稱。
    """
    out_path = Path(_sanitize(out_path.parent.name)) / _sanitize(out_path.name)
    # 若 out_path 不是單層路徑（例如 results_pickle/<label>.pkl），需重建完整路徑：
    # 重新用原始父層組合，逐層 sanitize。
    parts = [p for p in Path(out_path).parts]
    # 將每一層都消毒
    safe_parts = [_sanitize(p) for p in parts]
    safe_path = Path(*safe_parts)

    _ensure_dir(safe_path.parent)
    with open(safe_path, "wb") as f:
        pickle.dump(report_collection, f)
    print(f"[OK] Pickle saved → {safe_path}")

def load_report_collection_pickle(in_path: Path):
    """
    讀取 ReportCollection 的 Pickle。
    ※ 此處不會重寫路徑，呼叫者需給正確檔案路徑。
    """
    with open(in_path, "rb") as f:
        rc = pickle.load(f)
    print(f"[OK] Pickle loaded ← {in_path}")
    return rc

# -------------------------------
# 2) 匯出輕量分析素材（trades/position/stock_data/return_table + stats）
# -------------------------------
def export_report_collection_artifacts(report_collection, out_dir: Path, to_parquet=True):
    """
    會輸出：
    - {out_dir}/stats.parquet（或 .csv）
    - {out_dir}/{strategy}/trades.parquet|csv
                          /position.parquet|csv
                          /stock_data.parquet|csv
                          /return_table.parquet|csv

    注意：out_dir 與 strategy 目錄名稱都會自動 sanitize，避免 Windows 非法字元。
    """
    # 逐層 sanitize out_dir
    safe_dir = Path(*(_sanitize(p) for p in Path(out_dir).parts))
    _ensure_dir(safe_dir)

    stats_rows = []
    for name, rep in report_collection.reports.items():
        safe_name = _sanitize(name)
        strat_dir = _ensure_dir(safe_dir / safe_name)

        # trades
        if hasattr(rep, "trades"):
            _to_parquet_or_csv(rep.trades, strat_dir / "trades", to_parquet)

        # position
        if hasattr(rep, "position"):
            _to_parquet_or_csv(rep.position, strat_dir / "position", to_parquet)

        # stock_data（含 portfolio_returns / cum_returns / company_count）
        if hasattr(rep, "stock_data"):
            _to_parquet_or_csv(rep.stock_data, strat_dir / "stock_data", to_parquet)

        # return_table（dict → DataFrame 後輸出）
        if hasattr(rep, "return_table") and isinstance(rep.return_table, dict):
            try:
                rt = pd.DataFrame(rep.return_table).T
                _to_parquet_or_csv(rt, strat_dir / "return_table", to_parquet)
            except Exception as e:
                warnings.warn(f"[io_persistence] return_table 匯出失敗：{e}")

        # stats（呼叫內建 get_stats）
        try:
            s = rep.get_stats()
            s["strategy"] = name
            stats_rows.append(s)
        except Exception as e:
            warnings.warn(f"[io_persistence] get_stats() 失敗（{name}）：{e}")

    # 收斂整體 stats
    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        _to_parquet_or_csv(stats_df, safe_dir / "stats", to_parquet)
        print(f"[OK] stats saved → {safe_dir}")

    print(f"[OK] Artifacts exported → {safe_dir}")

# -------------------------------
# 3) 讀回輕量素材
# -------------------------------
def load_artifacts_stats(stats_path: Path) -> pd.DataFrame:
    """
    自動判斷副檔名 .parquet / .csv 讀入。
    """
    stats_path = Path(stats_path)
    if stats_path.suffix == ".parquet":
        return pd.read_parquet(stats_path)
    elif stats_path.suffix == ".csv":
        return pd.read_csv(stats_path)
    else:
        # 試圖優先讀 parquet，再讀 csv
        pq = stats_path.with_suffix(".parquet")
        if pq.exists():
            return pd.read_parquet(pq)
        csv = stats_path.with_suffix(".csv")
        if csv.exists():
            return pd.read_csv(csv)
        raise FileNotFoundError(f"找不到 {stats_path}（含 .parquet/.csv）")

def load_strategy_artifacts(strat_dir: Path):
    """
    回傳 dict：{'trades': df, 'position': df, 'stock_data': df, 'return_table': df}，不存在則為 None。
    會自動偵測 parquet/csv。
    """
    strat_dir = Path(*(_sanitize(p) for p in Path(strat_dir).parts))
    out = {}

    def _read_df(base: Path):
        pq = base.with_suffix(".parquet")
        cs = base.with_suffix(".csv")
        if pq.exists():
            return pd.read_parquet(pq)
        if cs.exists():
            # trades 用 csv 可能沒有 index；其他 timeseries 會希望 index 存在
            if base.name == "trades":
                return pd.read_csv(cs)
            else:
                try:
                    return pd.read_csv(cs, index_col=0, parse_dates=True)
                except Exception:
                    return pd.read_csv(cs)
        return None

    out["trades"] = _read_df(strat_dir / "trades")
    out["position"] = _read_df(strat_dir / "position")
    out["stock_data"] = _read_df(strat_dir / "stock_data")
    out["return_table"] = _read_df(strat_dir / "return_table")
    return out

# -------------------------------
# 4) 便利函式：一個呼叫存好兩種格式
# -------------------------------
def save_all_for_label(report_collection, base_pickle_dir: Path, base_artifacts_dir: Path, label: str, to_parquet=True):
    """給定 label，一次完成兩種輸出：
    - 存一份 pickle 到 {base_pickle_dir}/{label}.pkl
    - 輸出 artifacts 到 {base_artifacts_dir}/{label}/...

    兩個路徑與 label 都會自動 sanitize，呼叫端可直接傳原始字串。
    """
    safe_label = _sanitize(label)
    safe_pickle_dir = Path(*(_sanitize(p) for p in Path(base_pickle_dir).parts))
    safe_art_dir = Path(*(_sanitize(p) for p in Path(base_artifacts_dir).parts))

    _ensure_dir(safe_pickle_dir)
    _ensure_dir(safe_art_dir)

    # 1) pickle
    pickle_path = safe_pickle_dir / f"{safe_label}.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(report_collection, f)
    print(f"[OK] Pickle saved → {pickle_path}")

    # 2) artifacts
    export_report_collection_artifacts(report_collection, safe_art_dir / safe_label, to_parquet=to_parquet)


# 讀取單一 pickle
def load_report_collection_pickle(path: Path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError) as e:
        print(f"Warning: Failed to load pickle file {path}: {e}")
        return None

# 批次讀取資料夾中的所有回測結果
def load_all_report_collections(pickle_dir: Path) -> Dict[str, dict]:
    """
    從 pickle_dir 載入所有已存的回測結果
    回傳 dict: {label: report_collection}
    """
    results = {}
    for pkl_file in Path(pickle_dir).glob("*.pkl"):
        label = pkl_file.stem
        collection = load_report_collection_pickle(pkl_file)
        if collection is not None:
            results[label] = collection
    return results

# 載入單一策略的 artifacts
def load_report_artifacts(artifact_dir: Path, label: str):
    """
    讀取某一個策略的 parquet/csv 結果
    """
    strat_dir = artifact_dir / label
    artifacts = {}
    if strat_dir.exists():
        for f in strat_dir.glob("**/*.parquet"):
            artifacts[f.stem] = pd.read_parquet(f)
        for f in strat_dir.glob("**/*.csv"):
            artifacts[f.stem] = pd.read_csv(f, index_col=0, parse_dates=True)
    return artifacts
