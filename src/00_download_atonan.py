from pathlib import Path
import sys
from utils.file_finder import load_config

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

secret = load_config()


from pathlib import Path
from datetime import date, timedelta
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError
import time

# ==============================
# 設定
# ==============================

# 保存先フォルダ
SAVE_DIR = Path(secret["atonan_root"])

# ベースURL
BASE_URL = secret["sabar"]["atonan_root"]

# 取得する時刻（05:00:00）
TIME_PART = "050000"

# 取得期間
DATE_RANGES = [
    (date(2025, 6, 1), date(2025, 10, 31)),
]


# ==============================
# 関数
# ==============================

def daterange(start_date: date, end_date: date):
    """開始日から終了日まで1日ずつ返す"""
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def build_url_and_filename(target_date: date):
    """対象日からURLとファイル名を生成"""
    yyyy = target_date.strftime("%Y")
    mm = target_date.strftime("%m")
    dd = target_date.strftime("%d")
    yyyymmdd = target_date.strftime("%Y%m%d")

    filename = f"Z__C_RJTD_{yyyymmdd}{TIME_PART}_MET_GPV_Ggis1km_Psw_JRltg_Aper10min_ANAL_N1_grib2.bin"
    url = f"{BASE_URL}/{yyyy}/{mm}/{dd}/{filename}"
    return url, filename

def download_file(url: str, output_path: Path):
    """ファイルをダウンロード"""
    urlretrieve(url, output_path)


# ==============================
# メイン処理
# ==============================

def main():
    # 保存先フォルダを作成
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    downloaded = 0
    skipped = 0
    failed = 0

    for start_date, end_date in DATE_RANGES:
        for target_date in daterange(start_date, end_date):
            total += 1

            url, filename = build_url_and_filename(target_date)
            output_path = SAVE_DIR / filename

            if output_path.exists():
                print(f"[SKIP] 既存ファイル: {filename}")
                skipped += 1
                continue

            try:
                print(f"[DOWNLOADING] {url}")
                download_file(url, output_path)
                print(f"[OK] {filename}")

                downloaded += 1

                # 1件ダウンロードするごとに1秒待機
                time.sleep(1)

            except (HTTPError, URLError, OSError) as e:
                print(f"[FAILED] {url}")
                print(f"         {e}")

                # 失敗時に中途半端なファイルがあれば削除
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except OSError:
                        pass

                failed += 1

    print("\n=== 完了 ===")
    print(f"対象件数   : {total}")
    print(f"成功       : {downloaded}")
    print(f"スキップ   : {skipped}")
    print(f"失敗       : {failed}")


if __name__ == "__main__":
    main()
