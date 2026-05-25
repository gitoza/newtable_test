"""
01_stage1_extract.py
GRIB2 → Stage1 Parquet 変換スクリプト

あと何      レベル値と代表値
レベル値    代表値          意味
0                          欠測値
k（kは
1～200）	k-1	        土砂基準まであと(k-1)ミリ
201		                土砂基準まであと200ミリ以上


処理内容:
  - あと何の GRIB2 を月別に読み込む あと何は日ごとに05UTCのデータを収集している
  - Stage1 フィルタ: 解析雨量 > 0 のグリッド点のみ残す 
  - 4ワーカー並列で各月を処理し、Parquet として出力する

出力ファイル名:
  stage1/{data_type}/{YYYY}_{MM}.parquet
  例: stage1/rain/2023_07.parquet

使用方法:
  python src/01_stage1_extract.py
  python src/01_stage1_extract.py --year 2023 --month 7  # 単月テスト
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
from pathlib import Path

# -----------------------------------------------------------------------
# sys.path を最初に解決する（サードパーティ/utils より前に必須）
# このファイルは src/ にあるため、src/ をパスに追加すれば utils/ が見える
# -----------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils.file_finder import load_config, find_grib2_files, _parse_datetime_from_filename
from utils.grib_reader import check_definition_path, read_grib2_to_dataframe
from utils.scheduler import (
    set_process_priority_low,
    sleep_file_interval,
    run_with_pool,
)

# ---------------------------------------------------------------------------
# ロガー設定
# ---------------------------------------------------------------------------

def setup_logging(settings: dict) -> None:
    log_cfg = settings.get("logging", {})
    level_name = log_cfg.get("level", "INFO")
    log_dir = Path(log_cfg.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    handler_file = logging.handlers.RotatingFileHandler(
        log_dir / "stage1.log",
        maxBytes=log_cfg.get("rotate_mb", 50) * 1024 * 1024,
        backupCount=log_cfg.get("backup_count", 5),
        encoding="utf-8",
    )
    handler_stdout = logging.StreamHandler(sys.stdout)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    for h in (handler_file, handler_stdout):
        h.setFormatter(fmt)

    logging.basicConfig(
        level=getattr(logging, level_name),
        handlers=[handler_file, handler_stdout],
    )


# ---------------------------------------------------------------------------
# 1ワーカータスク: 1データ種別 × 1年月 を処理
# ---------------------------------------------------------------------------

# あと何サブタイプ: tank_id → 出力サブディレクトリ名
_ATONAN_SUBTYPE_DIRS: dict[str, str] = {
    "level2": "level2",  # 注意報基準
    "level3": "level3",  # 警報基準
    "level4": "level4",  # 危険警報基準
    "level5": "level5",  # 特別警報基準
}



# Stage1 出力スキーマ（atonan）
# モジュールレベルで定義することで、子プロセス内からも参照可能にする。
_STAGE1_SCHEMA = pa.schema([
    ("mesh3",    pa.uint32()),
    ("lat",      pa.float32()),
    ("lon",      pa.float32()),
    ("value",    pa.float32()),
    ("datetime", pa.timestamp("s")),
])
_STAGE1_SCHEMA_COLS: list[str] = [f.name for f in _STAGE1_SCHEMA]


def _write_parquet(
    df: "pd.DataFrame",
    out_path: "Path",
    compression: str,
    row_group_size: int,
) -> None:
    """DataFrame を Parquet に書き出す（空 DataFrame の場合はスキーマのみ）。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = _STAGE1_SCHEMA
    if df.empty:
        table = pa.table(
            {n: pa.array([], type=t) for n, t in zip(schema.names, schema.types)},
            schema=schema,
        )
    else:
        table = pa.Table.from_pandas(df[list(schema.names)], preserve_index=False)
    pq.write_table(table, out_path, compression=compression, row_group_size=row_group_size)


def _process_one_month(
    data_type: str,
    year: int,
    month: int,
    filepaths: list[str],           # Path は pickle できない場合があるため str で渡す
    src_dir: str,                   # 子プロセスで sys.path に追加するため渡す
    output_dir: str,
    rain_gt: float,
    sleep_between_files: float,
    parquet_compression: str,
    parquet_row_group_size: int,
    day: int | None = None,
) -> dict:
    """
    1年月分の GRIB2 を読み込み、Parquet に書き出す。
    ProcessPoolExecutor のワーカーとして実行されるため独立したプロセスで動く。

    Returns
    -------
    rain の場合:
        {"status": "ok"|"skipped"|"empty", "path": str, "rows": int}
    soil の場合:
        {"status": "ok"|"skipped"|"empty",
         "subtypes": {
             "swi":   {"status": ..., "path": ..., "rows": ...},
             "tank1": {"status": ..., "path": ..., "rows": ...},
             "tank2": {"status": ..., "path": ..., "rows": ...},
         }}
    
    atonanの場合：
        {"status": "ok"|"skipped"|"empty",
         "subtypes": {
             "level2":   {"status": ..., "path": ..., "rows": ...},
             "level3": {"status": ..., "path": ..., "rows": ...},
             "level4": {"status": ..., "path": ..., "rows": ...},
             "level5": {"status": ..., "path": ..., "rows": ...},
         }}
    
    """
    # ---- 子プロセス内での sys.path 設定 ----
    import sys
    from pathlib import Path as _Path
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    import logging
    import time
    import pandas as pd
    from utils.grib_reader import read_grib2_to_dataframe

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] worker: %(message)s",
    )
    log = logging.getLogger(__name__)

#    is_soil = (data_type == "soil")
    is_atonan = (data_type == "atonan")

    # ---- 出力パス決定 ----
    stem = f"{year}_{month:02d}_{day:02d}" if day is not None else f"{year}_{month:02d}"
    if is_atonan:
        out_paths = {
            tid: _Path(output_dir) / dirname / f"{stem}.parquet"
            for tid, dirname in _ATONAN_SUBTYPE_DIRS.items()
        }
        if all(p.exists() for p in out_paths.values()):
            log.info("スキップ（既出力）: atonan %d-%02d", year, month)
            return {
                "status": "skipped",
                "year": year, "month": month, "data_type": data_type,
                "subtypes": {
                    tid: {"status": "skipped", "path": str(p), "rows": -1}
                    for tid, p in out_paths.items()
                },
            }
        for p in out_paths.values():
            p.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = _Path(output_dir) / data_type / f"{stem}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            log.info("スキップ（既出力）: %s", out_path.name)
            return {"status": "skipped", "path": str(out_path), "rows": -1,
                    "year": year, "month": month, "data_type": data_type}

    n_files = len(filepaths)
    log.info("開始: %s %d-%02d (%d files)", data_type, year, month, n_files)

    from utils.file_finder import _parse_datetime_from_filename as _parse_dt
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = _STAGE1_SCHEMA
    schema_cols = _STAGE1_SCHEMA_COLS

    n_ok_files = 0
    n_err_files = 0
    log_interval = max(1, n_files // 10)  # 約10%ごとに進捗ログ

    # ---- atonan: レベル別に4本の ParquetWriter へ逐次書き込み ----
    if is_atonan:
        tmp_paths = {
            tid: _Path(output_dir) / dirname / f"{stem}.tmp.parquet"
            for tid, dirname in _ATONAN_SUBTYPE_DIRS.items()
        }
        row_counts = {tid: 0 for tid in _ATONAN_SUBTYPE_DIRS}
        writers: dict = {}
        try:
            for tid, tmp_p in tmp_paths.items():
                writers[tid] = pq.ParquetWriter(tmp_p, schema, compression=parquet_compression)

            for i, fp_str in enumerate(filepaths, 1):
                fp = _Path(fp_str)
                valid_dt = _parse_dt(fp)
                try:
                    df = read_grib2_to_dataframe(
                        fp, valid_datetime=valid_dt,
                        rain_gt=None, include_tank_id=True,
                    )
                    if not df.empty:
                        df["datetime"] = pd.to_datetime(df["datetime"])
                        for tid in _ATONAN_SUBTYPE_DIRS:
                            sub = df[df["tank_id"] == tid]
                            if not sub.empty:
                                writers[tid].write_table(
                                    pa.Table.from_pandas(
                                        sub[schema_cols], schema=schema, preserve_index=False
                                    )
                                )
                                row_counts[tid] += len(sub)
                        n_ok_files += 1
                    del df
                except Exception as e:
                    log.warning("読み込みエラー (%s): %s", fp.name, e)
                    n_err_files += 1

                if i % log_interval == 0 or i == n_files:
                    log.info(
                        "  [%s %d-%02d] %d/%d ファイル完了  有効=%d  エラー=%d",
                        data_type, year, month, i, n_files, n_ok_files, n_err_files,
                    )

                if sleep_between_files > 0:
                    time.sleep(sleep_between_files)

        finally:
            for w in writers.values():
                w.close()

        subtype_results: dict = {}
        for tid, dirname in _ATONAN_SUBTYPE_DIRS.items():
            out_p = out_paths[tid]
            tmp_p = tmp_paths[tid]
            if row_counts[tid] > 0:
                tmp_p.replace(out_p)
                log.info("完了(%s): %s  rows=%d", tid, out_p.name, row_counts[tid])
                subtype_results[tid] = {"status": "ok", "path": str(out_p), "rows": row_counts[tid]}
            else:
                if tmp_p.exists():
                    tmp_p.unlink()
                _write_parquet(pd.DataFrame(), out_p, parquet_compression, parquet_row_group_size)
                log.info("完了(%s): %s  rows=0 (empty)", tid, out_p.name)
                subtype_results[tid] = {"status": "empty", "path": str(out_p), "rows": 0}

        overall = "ok" if any(r["status"] == "ok" for r in subtype_results.values()) else "empty"
        return {"status": overall, "year": year, "month": month, "data_type": data_type,
                "subtypes": subtype_results}

    # ---- rain: ParquetWriter へ逐次書き込み ----
    tmp_path = out_path.with_suffix(".tmp.parquet")
    n_rows = 0
    writer = None
    try:
        writer = pq.ParquetWriter(tmp_path, schema, compression=parquet_compression)

        for i, fp_str in enumerate(filepaths, 1):
            fp = _Path(fp_str)
            valid_dt = _parse_dt(fp)
            try:
                df = read_grib2_to_dataframe(
                    fp, valid_datetime=valid_dt, rain_gt=rain_gt,
                )
                if not df.empty:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    writer.write_table(
                        pa.Table.from_pandas(
                            df[schema_cols], schema=schema, preserve_index=False
                        )
                    )
                    n_rows += len(df)
                    n_ok_files += 1
                del df
            except Exception as e:
                log.warning("読み込みエラー (%s): %s", fp.name, e)
                n_err_files += 1

            if i % log_interval == 0 or i == n_files:
                log.info(
                    "  [%s %d-%02d] %d/%d ファイル完了  有効=%d  エラー=%d",
                    data_type, year, month, i, n_files, n_ok_files, n_err_files,
                )

            if sleep_between_files > 0:
                time.sleep(sleep_between_files)

    finally:
        if writer is not None:
            writer.close()

    if n_rows == 0:
        if tmp_path.exists():
            tmp_path.unlink()
        log.info("有効データなし: %s %d-%02d", data_type, year, month)
        _write_parquet(pd.DataFrame(), out_path, parquet_compression, parquet_row_group_size)
        return {"status": "empty", "path": str(out_path), "rows": 0,
                "year": year, "month": month, "data_type": data_type}

    tmp_path.replace(out_path)
    log.info("完了: %s  rows=%d  -> %s", out_path.name, n_rows, out_path)
    return {"status": "ok", "path": str(out_path), "rows": n_rows,
            "year": year, "month": month, "data_type": data_type}


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def main(single_year: int | None = None, single_month: int | None = None, single_day: int | None = None) -> None:
    settings, secret = load_config()
    setup_logging(settings)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Stage1 抽出開始")
    logger.info("=" * 60)

    set_process_priority_low()
    check_definition_path()

    nas_output    = secret["nas"]["output_root"]
    stage1_dir    = str(Path(nas_output) / settings["output"]["stage1_dir"])
    rain_gt       = float(settings["stage1"]["rain_gt"])
    sleep_files   = float(settings["parallel"]["sleep_between_files"])
    compression   = settings["output"]["parquet_compression"]
    row_group_size = int(settings["output"]["parquet_row_group_size"])

    years  = [single_year]  if single_year  else settings["period"]["years"]
    months = [single_month] if single_month else settings["period"]["months"]

    logger.info(
        "処理設定: years=%s  months=%s  day=%s  rain_gt=%.1f  workers=%d",
        years, months, single_day, rain_gt, settings["parallel"]["max_workers"],
    )

    tasks: list[tuple] = []
    total_rain_files = 0
  #  total_soil_files = 0
    total_atonan_files = 0

    for year in years:
        for month in months:
            for data_type in ("rain", "atonan"):
                filepaths = find_grib2_files(
                    data_type, year, month, settings, secret,
                    top_of_hour_only=True, day=single_day,
                )
                if not filepaths:
                    logger.warning("ファイルなし: %s %d-%02d", data_type, year, month)
                    continue
                n = len(filepaths)
                logger.info("  発見: %s %d-%02d  %d files", data_type, year, month, n)
                if data_type == "rain":
                    total_rain_files += n
                else:
                    total_atonan_files += n
                tasks.append((
                    data_type, year, month,
                    [str(p) for p in filepaths],
                    str(_SRC_DIR),          # 子プロセスに src/ パスを渡す
                    stage1_dir,
                    rain_gt,
                    sleep_files,
                    compression,
                    row_group_size,
                    single_day,
                ))

    rain_tasks = sum(1 for t in tasks if t[0] == "rain")
#    soil_tasks = sum(1 for t in tasks if t[0] == "soil")
    atonan_tasks = sum(1 for t in tasks if t[0] == "atonan")
    logger.info(
        "タスク数: %d  (rain: %d月分/%dfiles  atonan: %d月分/%dfiles)",
        len(tasks), rain_tasks, total_rain_files, atonan_tasks, total_atonan_files,
    )

    results = run_with_pool(
        _process_one_month,
        tasks,
        settings,
        desc="Stage1抽出",
    )

    # タスクごとの結果を年月順に並べて表示
    logger.info("-" * 60)
    logger.info("タスク別結果:")
    sorted_results = sorted(
        [r for r in results if r is not None],
        key=lambda x: (x.get("year", 0), x.get("month", 0), x.get("data_type", "")),
    )
    for r in sorted_results:
        y, m, dt = r.get("year", "?"), r.get("month", "?"), r.get("data_type", "?")
        if "subtypes" in r:
            for tid, sub in r["subtypes"].items():
                st = sub["status"]
                rows = sub.get("rows", -1)
                if st == "ok":
                    logger.info("  [OK]      atonan/%-6s %s-%02s  rows=%d", tid, y, m, rows)
                elif st == "skipped":
                    logger.info("  [SKIP]    atonan/%-6s %s-%02s", tid, y, m)
                else:
                    logger.info("  [EMPTY]   atonan/%-6s %s-%02s", tid, y, m)
        else:
            st = r.get("status")
            rows = r.get("rows", -1)
            if st == "ok":
                logger.info("  [OK]      %-10s %s-%02s  rows=%d", dt, y, m, rows)
            elif st == "skipped":
                logger.info("  [SKIP]    %-10s %s-%02s", dt, y, m)
            else:
                logger.info("  [EMPTY]   %-10s %s-%02s", dt, y, m)
    if any(r is None for r in results):
        failed_count = sum(1 for r in results if r is None)
        logger.error("  [FAIL]    %d タスクが例外終了しました", failed_count)

    def _flatten_results(results: list) -> list[dict]:
        """atonan の subtypes を展開して集計用フラットリストに変換する。"""
        flat = []
        for r in results:
            if r is None:
                flat.append(None)
            elif "subtypes" in r:
                flat.extend(r["subtypes"].values())
            else:
                flat.append(r)
        return flat

    flat = _flatten_results(results)
    ok         = sum(1 for r in flat if r and r.get("status") == "ok")
    skipped    = sum(1 for r in flat if r and r.get("status") == "skipped")
    empty      = sum(1 for r in flat if r and r.get("status") == "empty")
    failed     = sum(1 for r in results if r is None)
    total_rows = sum(r.get("rows", 0) for r in flat if r and r.get("rows", 0) > 0)

    logger.info("=" * 60)
    logger.info(
        "Stage1 完了: ok=%d  skipped=%d  empty=%d  failed=%d  total_rows=%d",
        ok, skipped, empty, failed, total_rows,
    )


# ---------------------------------------------------------------------------
# CLI エントリーポイント
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage1: GRIB2 → Parquet 変換")
    parser.add_argument("--year",  type=int, default=None, help="単月/単日テスト用: 対象年")
    parser.add_argument("--month", type=int, default=None, help="単月/単日テスト用: 対象月")
    parser.add_argument("--day",   type=int, default=None, help="単日テスト用: 対象日（--year と --month も必須）")
    args = parser.parse_args()

    if bool(args.year) != bool(args.month):
        parser.error("--year と --month は両方指定するか両方省略してください")
    if args.day is not None and not (args.year and args.month):
        parser.error("--day を使う場合は --year と --month も指定してください")

    main(single_year=args.year, single_month=args.month, single_day=args.day)
