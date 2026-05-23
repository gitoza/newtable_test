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

import pandas as pd
import os

# --- swi_sarch.py をモジュールとして読み込む ---
import swi_search  # 同じフォルダに置く（またはPYTHONPATHに追加）

# =========================
# 設定
# =========================

INPUT_DIR = Path(os.path.join(secret["output_root"],"test_data"))
INPUT_FILES = [
    "aomori.csv",
    "akita.csv",
    "iwate.csv",
    "yamagata.csv",
    "miyagi.csv",
    "fukushima.csv",
]

OUTPUT_CSV = os.path.join(secret["output_root"],"totalling_result.csv")

OUT_COLUMNS = [
    "県",
    "年月日",
    "一次細分",
    "警報級の可能性(18-06)",
    "警報級の可能性(06-24)",
    "雨予想（R1）",
    "雨予想（R24）",
    "土砂余裕＋20mm",
    "旧R1",
    "旧R24",
    "旧手法",
    "新手法",
    "あと何判定",
]

# 入力から抜き出す列（1始まり: 1,2,3,4,5,6,7,10,11）
# pandasは0始まり -> 0,1,2,3,4,5,6,9,10
USECOLS = [0, 1, 2, 3, 4, 5, 6, 9, 10]

# 抜き出した9列を割り当てる出力列（「年月日」〜「旧R24」）
MAP_TO = OUT_COLUMNS[1:10]


# 数値（整数扱い）列
INT_COLS = ["雨予想（R1）", "雨予想（R24）", "土砂余裕＋20mm", "旧R1", "旧R24"]

# 対象期間（両端含む）
DATE_START = pd.Timestamp("2025-07-01")
DATE_END   = pd.Timestamp("2025-09-30")


def prefecture_from_filename(path: Path) -> str:
    """ aomori.csv -> aomori """
    return path.stem


def calc_new_method_judge(ymd_value, region_n, city_n, r1_n, r24_n):
    """
    新手法：swi_sarch.search() を呼んで judge を返す
    yyyymmdd は "YYYYMMDD"(8桁) で渡す（例：2025/07/01 -> "20250701"）
    """
    dt = pd.to_datetime(ymd_value, errors="coerce")
    if pd.isna(dt):
        return ""

    dt_prev = dt - pd.Timedelta(days=1)
    yyyymmdd = dt_prev.strftime("%Y%m%d")
 
    # r1/r24 は欠損なら空欄
    if pd.isna(r1_n) or pd.isna(r24_n):
        return ""

    # ユーザー指定どおり：search(yyyymmdd, region_n, city_n, r1_n, r24_n) -> judge
    judge = swi_search.search(yyyymmdd, region_n, city_n, int(r1_n), int(r24_n))

   
    return judge


def read_one_file(path: Path) -> pd.DataFrame:
    # 入力：ヘッダーなし、指定列のみ読み込み
    df_in = pd.read_csv(
        path,
        header=None,
        usecols=USECOLS,
        encoding="utf-8",
        engine="python",
    )

    # 9列に命名（年月日〜旧R24）
    df_in.columns = MAP_TO

    # 「一次細分」の表記ゆれ補正
    df_in["一次細分"] = df_in["一次細分"].astype(str).str.strip().replace({"三八": "三八上北"})

    # 期間フィルタ
    df_in["_ymd"] = pd.to_datetime(df_in["年月日"], errors="coerce")
    df_in = df_in[(df_in["_ymd"] >= DATE_START) & (df_in["_ymd"] <= DATE_END)].copy()
    df_in.drop(columns=["_ymd"], inplace=True)

    if df_in.empty:
        return pd.DataFrame(columns=OUT_COLUMNS)

    # 数値列を整数（欠損可）に変換（比較やモジュール入力のため）
    for col in INT_COLS:
        df_in[col] = pd.to_numeric(df_in[col], errors="coerce").astype("Int64")

    # 出力DF（同じ行数で作成）
    df_out = pd.DataFrame(index=df_in.index, columns=OUT_COLUMNS)

    # 県（ファイル名）を全行に付与
    df_out["県"] = prefecture_from_filename(path)

    # 年月日〜旧R24 を転記
    for c in MAP_TO:
        df_out[c] = df_in[c]

    # 旧手法：旧R1<=R1 または 旧R24<=R24 なら 1
    cond_old = (
        df_out["旧R1"].le(df_out["雨予想（R1）"]) |
        df_out["旧R24"].le(df_out["雨予想（R24）"])
    ).fillna(False)
    df_out["旧手法"] = cond_old.astype(int)

    # あと何判定：土砂余裕＋20mm<=R24 なら 1
    cond_ato = (
        df_out["土砂余裕＋20mm"].le(df_out["雨予想（R24）"])
    ).fillna(False)
    df_out["あと何判定"] = cond_ato.astype(int)

    # 新手法：swi_sarch.search() の judge を入れる
    new_vals = []
    for idx in df_out.index:
        try:
            new_vals.append(
                calc_new_method_judge(
                    ymd_value=df_out.at[idx, "年月日"],
                    region_n=df_out.at[idx, "県"],
                    city_n=df_out.at[idx, "一次細分"],
                    r1_n=df_out.at[idx, "雨予想（R1）"],
                    r24_n=df_out.at[idx, "雨予想（R24）"],
                )
            )
        except Exception as e:
            # 失敗時は空欄（止めたいなら raise に変更）
            print(f"[WARN] 新手法計算失敗: file={path.name}, idx={idx}, err={e}")
            new_vals.append("")

    df_out["新手法"] = new_vals

    return df_out.reset_index(drop=True)


def main():
    all_df = []
    for name in INPUT_FILES:
        path = INPUT_DIR / name
        if not path.exists():
            print(f"[WARN] 見つかりません: {path}")
            continue

        try:
            part = read_one_file(path)
            if not part.empty:
                all_df.append(part)
        except Exception as e:
            print(f"[WARN] 読み込み失敗: {path} / {e}")

    if not all_df:
        pd.DataFrame(columns=OUT_COLUMNS).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"有効なデータが無いためヘッダーのみ作成: {OUTPUT_CSV}")
        return

    df = pd.concat(all_df, ignore_index=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"完了: {OUTPUT_CSV}（{len(df)}行）")


if __name__ == "__main__":
    main()