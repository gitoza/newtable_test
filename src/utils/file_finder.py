"""
utils/file_finder.py
NAS上のGRIB2ファイルを日付・データ種別で検索するユーティリティ。

ファイル命名規則（気象庁標準）:
  解析雨量:     Z__C_RJTD_YYYYMMDDHHmm00_SRF_GPV_Ggis1km_Prr60lv_ANAL_grib2.bin
  土壌雨量指数: Z__C_RJTD_YYYYMMDDHHmm00_SRF_GPV_Ggis1km_Dssr_ANAL_grib2.bin
  あと何ミリ: Z__C_RJTD_YYYYMMDDHHmm00_MET_GPV_Ggis1km_Psw_JRltg_Aper10min_ANAL_N1_grib2.bin

NAS上のディレクトリ構成（実際）:
  {root}/{YYYY}/{MM}/{DD}/
  例: \\172....\data\Grib2\swi10\2025\07\01\
"""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Generator

import yaml

try:
    import tomllib          # Python 3.11+
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]  # Python < 3.11
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 設定ロード
# ---------------------------------------------------------------------------

# このファイルは files/src/utils/file_finder.py
# プロジェクトルート(files/)は .parent x3
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_secret(secret_path: Path) -> dict:
    """
    secret ファイルを読み込む。
    拡張子が .toml なら TOML、それ以外（.yaml / .yml）なら YAML として解析する。
    secret.toml が優先: secret_path が .yaml でも同名の .toml が存在すれば .toml を使う。
    """
    # secret.yaml 指定でも secret.toml が隣に存在すれば TOML 優先
    toml_candidate = secret_path.with_suffix(".toml")
    if secret_path.suffix in (".yaml", ".yml") and toml_candidate.exists():
        secret_path = toml_candidate

    if secret_path.suffix == ".toml":
        if tomllib is None:
            raise ImportError(
                "TOML ファイルを読み込むには tomllib (Python 3.11+) または "
                "tomli パッケージが必要です。`pip install tomli` を実行してください。"
            )
        with open(secret_path, "rb") as f:
            return tomllib.load(f)

    with open(secret_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(
    settings_path: str | Path | None = None,
    secret_path: str | Path | None = None,
) -> tuple[dict, dict]:
    """
    settings.yaml と secret ファイルを読み込んで返す。

    secret ファイルの優先順位:
      1. secret.toml  （secret_path 未指定時、または .yaml 指定でも .toml が存在する場合）
      2. secret.yaml

    パスを省略した場合は __file__ 基準でプロジェクトルートの config/ を自動解決する。
    """
    if settings_path is None:
        settings_path = _PROJECT_ROOT / "config" / "settings.yaml"
    if secret_path is None:
        secret_path = _PROJECT_ROOT / "config" / "secret.yaml"
    settings_path = Path(settings_path)
    secret_path   = Path(secret_path)

    with open(settings_path, encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    secret = _load_secret(secret_path)
    return settings, secret


# ---------------------------------------------------------------------------
# 対象年月の全日付を列挙
# ---------------------------------------------------------------------------

def _days_in_month(year: int, month: int) -> list[date]:
    """指定年月の全日付リストを返す。"""
    days = []
    d = date(year, month, 1)
    while d.month == month:
        days.append(d)
        d += timedelta(days=1)
    return days


# ---------------------------------------------------------------------------
# ファイルパターン
# ---------------------------------------------------------------------------

PATTERNS = {
    "atonan": "Z__C_RJTD_*_MET_GPV_Ggis1km_Psw_JRltg_Aper10min_ANAL_N1_grib2.bin",
}


def _parse_datetime_from_filename(filepath: Path) -> datetime | None:
    """
    ファイル名から観測日時を解析する。
    Z__C_RJTD_YYYYMMDDHHmm00_... の形式を想定。
    """
    try:
        dt_str = filepath.name.split("_")[4]   # 例: "20230701000000" 14桁 (index4: Z__C_RJTD_{dt}_...)
        return datetime.strptime(dt_str[:12], "%Y%m%d%H%M")
    except (IndexError, ValueError):
        logger.debug("日時パース失敗: %s", filepath.name)
        return None


def _is_top_of_hour(dt: datetime) -> bool:
    """正時（分=0）かどうか判定する。"""
    return dt.minute == 0
def _is_target_hour(dt: datetime) -> bool:
    """09UTC 14時を判定"""
    return dt.hour == 5 and dt.minute == 0

# ---------------------------------------------------------------------------
# メイン: ファイルリスト取得
# ---------------------------------------------------------------------------

def find_grib2_files(
    data_type: str,
    year: int,
    month: int,
    settings: dict,
    secret: dict,
    top_of_hour_only: bool = True,
    day: int | None = None,
) -> list[Path]:
    """
    指定年月・データ種別のGRIB2ファイルリストを返す。

    NASのフォルダ構成: {root}/{YYYY}/{MM}/{DD}/
    月内の全日付ディレクトリを走査してファイルを収集する。

    Parameters
    ----------
    data_type       : "atonan"
    year, month     : 対象年月
    settings, secret: 設定辞書
    top_of_hour_only: True の場合、正時ファイルのみ返す ←   14時のみ返したい
    day             : 指定した場合、その日のみ処理（単日テスト用）

    Returns
    -------
    ファイルパスのリスト（昇順ソート済み）
    """
    nas = secret["nas"]
    root_key = "atonan_root"
    root = Path(nas[root_key])

    pattern = PATTERNS.get(data_type)
    if pattern is None:
        raise ValueError(f"未知のdata_type: {data_type}")

    found: list[Path] = []
    missing_days: list[str] = []


    # ✅ ------------------------------
    # ケース1: atonan（単一ディレクトリ）
    # ✅ ------------------------------
    if data_type == "atonan":

        for filepath in root.glob(pattern):

            dt = _parse_datetime_from_filename(filepath)

            if dt is None:
                continue

            # 年月フィルタ
            if dt.year != year or dt.month != month:
                continue

            # 時刻フィルタ（14時のみ）
            if top_of_hour_only and not _is_target_hour(dt):
                continue
            

            found.append(filepath)

    else:
        logger.info("%s %d-%02d: エラー ファイル未発見", data_type, year, month)
        

    if missing_days:
        logger.debug("%s %d-%02d: %d日分のディレクトリなし", data_type, year, month, len(missing_days))

    found.sort()
    logger.info("%s %d-%02d: %d ファイル発見", data_type, year, month, len(found))
    return found


# ---------------------------------------------------------------------------
# バッチ用: 全対象年月のファイルを列挙するジェネレータ
# ---------------------------------------------------------------------------

def iter_monthly_file_pairs(
    settings: dict,
    secret: dict,
) -> Generator[tuple[int, int, list[Path], list[Path]], None, None]:
    """
    settings に定義された全年月について (year, month, atonan_files) を yield する。
    """
    years: list[int] = settings["period"]["years"]
    months: list[int] = settings["period"]["months"]

    for year in years:
        for month in months:
            atonan_files = find_grib2_files(
                "atonan", year, month, settings, secret, top_of_hour_only=True
            )
                            
            yield year, month, atonan_files


# ---------------------------------------------------------------------------
# デバッグ用 CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    cfg, sec = load_config()
    for y, m, a in iter_monthly_file_pairs(cfg, sec):
        print(f"{y}-{m:02d}  atonan={len(a)}")
