"""
utils/grib_reader.py
GRIB2ファイルを読み込んで pandas DataFrame に変換する共通処理。

前提:
  - eccodes 2.46.0 (pip) + JMA用定義ファイル配置済み
  - 環境変数 ECCODES_DEFINITION_PATH が設定済み
  - 格子解像度: 1km格子 (Ni × Nj は実データに依存)

出力 DataFrame カラム:
  mesh3, lat, lon, value, datetime
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import eccodes
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ECCODES_DEFINITION_PATH の確認
# ---------------------------------------------------------------------------

def check_definition_path() -> None:
    """定義パスが設定されていない場合に警告を出す。"""
    val = os.environ.get("ECCODES_DEFINITION_PATH", "")
    if not val:
        logger.warning(
            "ECCODES_DEFINITION_PATH が未設定です。"
            "JMA用テンプレートが読み込めない可能性があります。"
        )
    else:
        logger.debug("ECCODES_DEFINITION_PATH: %s", val)


# ---------------------------------------------------------------------------
# テンプレート5.200 tank2 符号絶対値補正用定数
# ---------------------------------------------------------------------------

# 代表値のスケール因子（仕様書より固定値1 → 10^-1 = 0.1）
_TANK2_REPR_SCALE = 0.1

# 符号絶対値表現の符号ビット閾値（0x8000 × 0.1 = 3276.8）
# これを超える代表値はeccodesによる符号ビット誤読と判断する
_TANK2_SIGN_BIT_VALUE = 0x8000 * _TANK2_REPR_SCALE  # = 3276.8


# ---------------------------------------------------------------------------
# GRIB2 Sec4-7繰り返し構造の分割読み込み
# ---------------------------------------------------------------------------

def _parse_section_offsets(raw: bytes) -> list[tuple[int, int, int]]:
    """
    GRIB2バイト列から全セクションの (sec_num, offset, length) を返す。
    Section 0（固定16バイト）はスキップして offset=16 から開始する。
    """
    sections: list[tuple[int, int, int]] = []
    offset = 16  # Section 0 は固定16バイト
    while offset + 4 <= len(raw):
        if raw[offset:offset+4] == b'7777':
            sections.append((8, offset, 4))
            break
        sec_len = int.from_bytes(raw[offset:offset+4], 'big')
        if sec_len < 5:
            break
        sec_num = raw[offset + 4]
        sections.append((sec_num, offset, sec_len))
        offset += sec_len
    return sections


def _split_grib2_messages(raw: bytes) -> list[bytes]:
    """
    1つのGRIBエンベロープ内に Sec4-7 が複数回繰り返されている場合に、
    各 Sec4-7 セットを独立した GRIB2 メッセージバイト列に組み立てて返す。

    繰り返しが1回のみの場合は元のバイト列をそのままリストに包んで返す。

    Returns
    -------
    list[bytes]: 各要素が eccodes.codes_new_from_message に渡せる完結したGRIB2バイト列
    """
    secs = _parse_section_offsets(raw)

    # Sec4 の数を数える
    sec4_entries = [(num, off, ln) for num, off, ln in secs if num == 4]
    if len(sec4_entries) <= 1:
        return [raw]  # 繰り返しなし → そのまま返す

    # 共通ヘッダ: Sec0(16bytes) + Sec1 + Sec3 を抽出
    sec0 = raw[0:16]
    sec1_entry = next((s for s in secs if s[0] == 1), None)
    sec3_entry = next((s for s in secs if s[0] == 3), None)
    if sec1_entry is None or sec3_entry is None:
        return [raw]  # 想定外構造 → フォールバック

    sec1 = raw[sec1_entry[1] : sec1_entry[1] + sec1_entry[2]]
    sec3 = raw[sec3_entry[1] : sec3_entry[1] + sec3_entry[2]]

    # Sec4-7 のグループを抽出
    messages: list[bytes] = []
    i = 0
    while i < len(secs):
        if secs[i][0] == 4:
            # Sec4, Sec5, Sec6, Sec7 を取り出す
            if i + 3 >= len(secs):
                break
            group = []
            for j in range(4):
                num, off, ln = secs[i + j]
                group.append(raw[off : off + ln])
            i += 4

            # 1つのGRIB2メッセージに組み立てる
            body = sec1 + sec3 + group[0] + group[1] + group[2] + group[3] + b'7777'
            total_len = 16 + len(body)
            # Sec0 の全体長フィールド（オクテット9-16）を書き直す
            new_sec0 = sec0[:8] + total_len.to_bytes(8, 'big')
            messages.append(new_sec0 + body)
        else:
            i += 1

    return messages if messages else [raw]


def _get_tank_id_from_sec4(sec4_bytes: bytes) -> str:
    """
    Sec4 のバイト列からタンクIDを直接読む（eccodes キーに依存しない）。

    仕様書(No.10401 別紙１):
      オクテット23 (index 22): typeOfFirstFixedSurface
        200 → "swi"
        201 → タンク番号をオクテット25-28 (index 24:28) から取得
    
    あと何では、
        オクテット11(index 11):パラメータ番号
            219 → level2
            220 → level3
            221 → level4
            238 → level5
                
    """
    """
    if len(sec4_bytes) < 28:
        return "unknown"
    surface_type = sec4_bytes[22]
    if surface_type == 200:
        return "swi"
    if surface_type == 201:
        tank_num = int.from_bytes(sec4_bytes[24:28], 'big')
        if tank_num == 1:
            return "tank1"
        if tank_num == 2:
            return "tank2"
    return "unknown"
    """
    surface_type = sec4_bytes[10]
    if surface_type == 219:
        return "level2"
    elif surface_type == 220:
        return "level3"
    elif surface_type == 221:
        return "level4"
    elif surface_type == 238:
        return "level5"
    return "unknown"
# ---------------------------------------------------------------------------
# 1ファイル読み込み
# ---------------------------------------------------------------------------

def read_grib2_to_dataframe(
    filepath: Path | str,
    valid_datetime: datetime | None = None,
    rain_gt: float | None = None,
    include_tank_id: bool = False,
) -> pd.DataFrame:
    """
    GRIB2ファイルを読み込み、DataFrame を返す。

    Parameters
    ----------
    filepath        : GRIB2ファイルパス
    valid_datetime  : ファイルの観測日時（ファイル名から渡す）
    rain_gt         : None でなければ value > rain_gt のみ残す（Stage1 フィルタ）
    include_tank_id : True の場合、あと何レベル区分を "tank_id" 列として付加する
                      ("level2" / "level3" / "level4" / "level5" / "unknown")

    Returns
    -------
    DataFrame: columns = [mesh3, lat, lon, value, datetime]
               include_tank_id=True の場合は追加で [tank_id]
    """
    filepath = Path(filepath)
    rows: list[pd.DataFrame] = []

    try:
        with open(filepath, "rb") as fh:
            raw = fh.read()
    except OSError as e:
        logger.error("ファイルオープン失敗: %s  %s", filepath, e)
        return pd.DataFrame(columns=["mesh3", "lat", "lon", "value", "datetime"])

    # Sec4-7 繰り返し構造を分割し、各セットを独立したメッセージとして処理する
    sub_messages = _split_grib2_messages(raw)

    for msg_bytes in sub_messages:
        # Sec4 のバイト列を取り出してタンクIDを先に確定する（eccodes キー非依存）
        secs = _parse_section_offsets(msg_bytes)
        sec4_entry = next((s for s in secs if s[0] == 4), None)
        tank_id_raw = (
            _get_tank_id_from_sec4(msg_bytes[sec4_entry[1] : sec4_entry[1] + sec4_entry[2]])
            if sec4_entry else "unknown"
        )

        msg_id = None
        try:
            msg_id = eccodes.codes_new_from_message(msg_bytes)

            values = eccodes.codes_get_values(msg_id)
            lat, lon = _get_latlon_arrays(msg_id)

            missing = eccodes.codes_get(msg_id, "missingValue", ktype=float)

            # Step1: missing判定は補正前に行う（missingValueの符号仕様に依存しない）
            missing_mask = np.isclose(values, missing)
                        
            mask = ~missing_mask

            if rain_gt is not None:
                mask = mask & (values > rain_gt)

            if mask.any():
                lat_masked = lat[mask]
                lon_masked = lon[mask]
                df = pd.DataFrame({
                    "mesh3": latlon_to_mesh3_array(lat_masked, lon_masked),
                    "lat":   lat_masked.astype(np.float32),
                    "lon":   lon_masked.astype(np.float32),
                    "value": values[mask].astype(np.float32),
                })
                if valid_datetime is not None:
                    df["datetime"] = valid_datetime
                if include_tank_id:
                    df["tank_id"] = tank_id_raw
                rows.append(df)

        except eccodes.CodesInternalError as e:
            logger.warning("メッセージ読み込みエラー (%s): %s", filepath.name, e)
        finally:
            if msg_id is not None:
                eccodes.codes_release(msg_id)

    if not rows:
        return pd.DataFrame(columns=["mesh3", "lat", "lon", "value", "datetime"])

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# 格子座標の取得
# ---------------------------------------------------------------------------

def _get_latlon_arrays(msg_id: int) -> tuple[np.ndarray, np.ndarray]:
    """
    GRIB2メッセージから緯度・経度の1D配列を取得する。

    優先順位:
      1. latitudes / longitudes キーを個別取得（最も安全）
      2. Ni/Nj + 格子定義から等間隔メッシュを生成（フォールバック）

    latlonValues は "Unsupported data type" エラーが発生するケースがある
    （eccodes の内部型が float32/float64 以外になる場合）ため使用しない。
    """
    try:
        # eccodes.codes_get_array で個別取得する方法が最も互換性が高い
        lat = eccodes.codes_get_array(msg_id, "latitudes")   # float64[]
        lon = eccodes.codes_get_array(msg_id, "longitudes")  # float64[]
        return lat, lon
    except eccodes.CodesInternalError:
        pass

    # フォールバック: 格子定義から計算
    return _build_latlon_grid(msg_id)


def _build_latlon_grid(msg_id: int) -> tuple[np.ndarray, np.ndarray]:
    """格子定義から緯度・経度メッシュを生成する。"""
    ni        = eccodes.codes_get(msg_id, "Ni", ktype=int)
    nj        = eccodes.codes_get(msg_id, "Nj", ktype=int)
    lat_first = eccodes.codes_get(msg_id, "latitudeOfFirstGridPointInDegrees", ktype=float)
    lon_first = eccodes.codes_get(msg_id, "longitudeOfFirstGridPointInDegrees", ktype=float)
    lat_last  = eccodes.codes_get(msg_id, "latitudeOfLastGridPointInDegrees",  ktype=float)
    lon_last  = eccodes.codes_get(msg_id, "longitudeOfLastGridPointInDegrees", ktype=float)

    lats = np.linspace(lat_first, lat_last, nj)
    lons = np.linspace(lon_first, lon_last, ni)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lat_grid.ravel(), lon_grid.ravel()


# ---------------------------------------------------------------------------
# 複数ファイルの結合読み込み（直列・メモリ節約版）
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3次メッシュコード変換（整数演算）
# ---------------------------------------------------------------------------

def latlon_to_mesh3_array(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    緯度・経度配列（float64）を 3次メッシュコード配列（uint32）に変換する。

    変換は整数演算のみで行い、浮動小数点の丸め誤差を排除する。
    3次メッシュは南北30秒・東西45秒（約1km）の格子で、
    気象庁1kmGPV格子と1対1対応する。

    Parameters
    ----------
    lat : float64 配列（北緯）
    lon : float64 配列（東経）

    Returns
    -------
    uint32 配列: 8桁のメッシュコード
    """
    lat_idx = (lat * 120).astype(np.int32)
    lon_idx = ((lon - 100.0) * 80).astype(np.int32)

    p = lat_idx // 80          # 緯度1次（2桁）
    u = lon_idx // 80          # 経度1次（2桁）
    q = (lat_idx % 80) // 10  # 緯度2次（1桁）
    v = (lon_idx % 80) // 10  # 経度2次（1桁）
    r = lat_idx % 10           # 緯度3次（1桁）
    w = lon_idx % 10           # 経度3次（1桁）

    code = (
        p * 1_000_000
        + u * 10_000
        + q * 1_000
        + v * 100
        + r * 10
        + w
    ).astype(np.uint32)

    return code


def read_multiple_files(
    filepaths: list[Path],
    datetime_map: dict[Path, datetime] | None = None,
    rain_gt: float | None = None,
    sleep_sec: float = 0.0,
) -> pd.DataFrame:
    """複数 GRIB2ファイルを順次読み込み、結合した DataFrame を返す。"""
    import time

    dfs: list[pd.DataFrame] = []
    for fp in filepaths:
        dt = datetime_map.get(fp) if datetime_map else None
        df = read_grib2_to_dataframe(fp, valid_datetime=dt, rain_gt=rain_gt)
        if not df.empty:
            dfs.append(df)
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    if not dfs:
        return pd.DataFrame(columns=["mesh3", "lat", "lon", "value", "datetime"])
    return pd.concat(dfs, ignore_index=True)
