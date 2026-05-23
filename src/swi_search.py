# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 07:54:28 2025

@author: JMASNYOHOG
"""

# -----------------------------------------------------------------------
# sys.path を最初に解決する（サードパーティ/utils より前に必須）
# このファイルは src/ にあるため、src/ をパスに追加すれば utils/ が見える
# -----------------------------------------------------------------------
from pathlib import Path
import sys
from utils.file_finder import load_config

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

secret = load_config()
 
import os
import pandas as pd
import utils.grib_class as gb  # あなたの環境のクラスを使用
import time

# 刻み幅（正規化用）
step_swi = 20
step_r1 = 5
step_total = 20

# SWIファイルのルートディレクトリ
SWI_BASE_DIR = secret["nas"]["soil_root"]
#市町村のメッシュコード格納ディレクトリ
mesh_dir = os.path.join(secret["output_root"],"3dmesh_to_grib\\input")
#土壌雨量指数ファイル名
swi_fname = "Ggis1km_Psw_Aper10min_ANAL"
#新テーブル
table = os.path.join(secret["output_root"],"table_regular_fcst.csv")


#ファイル名はUTC
def build_dt_str(yyyymmdd: str, hour="05", minute="00", second="00"):
    """
    MMDD を受け取り、YYYYMMDDhhmmss を生成
    年は現在年を使用
    """
    yyyy = yyyymmdd[:4] 
    mm = yyyymmdd[4:6]
    dd = yyyymmdd[6:]

    return f"{yyyy}{mm}{dd}{hour}{minute}{second}"

#指定した日時からディレクトリ及びファイルを探す。
def find_corresponding_file(datetime_str, base_dir=SWI_BASE_DIR):
    """
    指定した日時文字列（例: '20240828230000'）を含むSWIファイルを探す。
    """
    y, m, d, h = datetime_str[:4], datetime_str[4:6], datetime_str[6:8], datetime_str[8:10]
    search_dir = os.path.join(base_dir, y, m, d)
    if not os.path.isdir(search_dir):
        print(f"ディレクトリが存在しません: {search_dir}")
        return None
    for fname in os.listdir(search_dir):
        if datetime_str in fname and swi_fname in fname:
            return os.path.join(search_dir, fname)
    print(f"該当ファイルが見つかりませんでした: {datetime_str}")
    return None


"""
指定キーワードを含むCSVファイルをフォルダ内から検索。
"""    
def find_file(keyword: str) -> str:

    for root, _, files in os.walk(mesh_dir):
        for file in files:
            #対象となる県を選出
            if keyword in file and file.lower().endswith(".csv"):
                return os.path.join(root, file)

    raise FileNotFoundError(f"'{keyword}' を含むCSVファイルが見つかりません。")

#対象となる一次細分の3次メッシュ、kp抽出
def extract_mesh_list(filename_keyword: str, city_name: str) -> list:

    # ファイル検索
    filepath = find_file(filename_keyword)
#    print(f"読み込み対象ファイル: {filepath}")

    # CSVファイル読み込み
    # 1行目は説明文なので読み飛ばし、2行目をヘッダー行として扱う
    df = pd.read_csv(filepath, header=1, encoding="sjis")

    # 一次細分が一致する行を抽出
    matched_rows = df[df["一次細分"] == city_name]

    if matched_rows.empty:
        print(f"⚠ 一次細分「{city_name}」に一致するデータが見つかりません。")
        return []

    #####################作業中箇所　格子がずれている可能性
    # 地域メッシュ（3列目）をリストとして抽出
    mesh_series = pd.to_numeric(
        matched_rows["地域メッシュコード（１km格子対応）"],
        errors="coerce"   # 数値でないもの → NaN
    )
    
    mesh_list = mesh_series.tolist()

    #mesh_list = matched_rows["地域メッシュコード（１km格子対応）"].tolist()
    #.dropna()
    # kp抽出（数値以外は NaN にする）
    kp_series = pd.to_numeric(
        matched_rows["土壌雨量指数基準（警報）"],
        errors="coerce"   # 数値でないもの → NaN
    )
    
    kp_list = kp_series.tolist()

#    print(f"抽出された地域メッシュ数: {len(mesh_list)}")
    return mesh_list, kp_list



"""
3次メッシュコード（9桁または10桁）を緯度・経度に変換する
戻り値は (lat, lon)：南西端の座標（左下の点）
"""
def meshcode_to_latlon(meshcode):
    # 入力を文字列にして桁数を確認
    meshcode = str(meshcode)
    if len(meshcode) < 8:
        raise ValueError("メッシュコードが短すぎます（8桁必要）")

    # 第一次メッシュ（最初の4桁）
    lat_deg = int(meshcode[:2]) * 2 / 3
    lon_deg = int(meshcode[2:4]) 

    # 第二次メッシュ（次の2桁）
    lat_min = int(meshcode[4]) * 5.0 / 60.0
    lon_min = int(meshcode[5]) * 7.5 / 60.0

    # 第三次メッシュ（次の2桁）
    lat_sec = int(meshcode[6]) * 30.0 / 3600.0
    lon_sec = int(meshcode[7]) * 45.0 / 3600.0

    lat = lat_deg + lat_min + lat_sec
    lon = lon_deg + lon_min + lon_sec + 100

    return lat, lon



"""
GRIBファイルからSWI配列を抽出。
指定範囲（lat/lon）に対応する部分を抽出して返す。
"""
def extract_swi(file_path, mesh_list: list):
    if file_path is None or not os.path.exists(file_path):
        raise FileNotFoundError(f"指定ファイルが見つかりません: {file_path}")

    # grib_classを使ってデコード
    swi_grib = gb.Grib_Decode(file_path=file_path)
    swi_data = swi_grib.read_grib2()  # 2次元配列を想定
    
        
    # 結果格納用
    swi_sub = []
    
    for lat, lon in mesh_list:        
        # 緯度経度からインデックスを取得
        y_idx, x_idx = swi_grib.get_index(lat, lon)

        # 東北地方など特定範囲のインデックス。土壌雨量指数データは、なぜか地域メッシュコードから西に2格子ずれる。
        value = swi_data[y_idx,x_idx+2]
        
        swi_sub.append(value)
    
#    print(f"抽出完了: min={np.min(swi_sub)}, max={np.max(swi_sub)}")
    return swi_sub


#予想の判定
def forecast_swi(file_path, r1, r24, swi_s_list:list, kp_list:list, mesh_list:list):

    results = []
    printed = False
    jdg_result = 0
    
    for swi_s, kp, mesh in zip(swi_s_list, kp_list, mesh_list):

        # 無効な行はスキップ
        if pd.isna(kp):
            continue   # 空欄は処理しない
            
        # 比較条件設定
        #ここで、指定された行の指数よりもテーブルを省く　table >= index of row
        df_filtered = file_path[
            (file_path['R24'] >= r24) &
            (file_path['R1'] >= r1) &
            (file_path['start_swi'] >= swi_s)
        ]
        
        row = df_filtered.sort_values(
            ["R1", "R24", "start_swi"]
        ).iloc[0]

        
        final_swi = row['fcst_swi']
        meets_condition = final_swi >= kp
        
        if meets_condition:
            jdg_result = 1
                
    return jdg_result



def search(yyyymmdd, region_n, city_n, r1_n, r24_n):
    
    print(yyyymmdd, region_n)
    
    time.sleep(0.5)   # 1秒停止（この間、処理は止まる）
    
    dt_str = build_dt_str(yyyymmdd)

    #指定された日付の土壌雨量指数ファイル
    swi_file = find_corresponding_file(dt_str)
        
    if swi_file:
 #       print(f"見つかったファイル: {swi_file}")
        
        mesh_l,kp_l = extract_mesh_list(region_n, city_n)

        # 緯度・経度リストに変換
        latlon_list = [meshcode_to_latlon(code) for code in mesh_l]
        swi_list = extract_swi(swi_file, latlon_list)
        # fcst テーブル読み込み
        fcst_df = pd.read_csv(table)        
        new_judge = forecast_swi(
            fcst_df,
            float(r1_n),
            float(r24_n),
            swi_list,
                kp_l,
                mesh_l
        )
 
        return new_judge