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
import numpy as np
import os

METHOD_COLS = ["旧手法", "新手法", "あと何判定"]
TRUTH_COL = "大雨発表"


def _to_binary_series(s: pd.Series, colname: str) -> pd.Series:
    """
    0/1 を想定。文字列 "0"/"1" や True/False にも対応。
    不正値があれば例外を出す。
    """
    # 欠損はそのまま残す（後で除外）
    if s.dtype == "bool":
        out = s.astype("Int64")
    else:
        # 文字列などを数値へ
        out = pd.to_numeric(s, errors="coerce").astype("Int64")

    # 欠損以外で 0/1 以外が混入していないかチェック
    non_na = out.dropna()
    invalid = non_na[~non_na.isin([0, 1])]
    if len(invalid) > 0:
        examples = invalid.unique()[:10]
        raise ValueError(
            f"列 '{colname}' に 0/1 以外の値が含まれています。例: {examples}"
        )
    return out


def safe_div(numer: float, denom: float) -> float:
    """0除算を NaN にする"""
    return float(numer) / float(denom) if denom != 0 else np.nan


def compute_metrics(pred: pd.Series, truth: pd.Series) -> dict:
    """
    pred/truth: 0/1 の Series（欠損あり得る）
    パターン:
      第1: pred=1 & truth=1 (TP)
      第2: pred=1 & truth=0 (FP)
      第3: pred=0 & truth=1 (FN)
      第4: pred=0 & truth=0 (TN)
    """
    # 欠損行を除外（どちらか欠損なら評価不能）
    df = pd.DataFrame({"pred": pred, "truth": truth}).dropna()
    if df.empty:
        # 全部欠損だった場合
        return {
            "第1パターン(TP)": 0,
            "第2パターン(FP)": 0,
            "第3パターン(FN)": 0,
            "第4パターン(TN)": 0,
            "捕捉率": np.nan,
            "見逃し率": np.nan,
            "空振り率": np.nan,
            "スレットスコア": np.nan,
            "適中率": np.nan,
            "予想あり適中率（一致率）": np.nan,
            "評価対象行数": 0,
        }

    pred1 = df["pred"] == 1
    pred0 = df["pred"] == 0
    tru1 = df["truth"] == 1
    tru0 = df["truth"] == 0

    TP = int((pred1 & tru1).sum())
    FP = int((pred1 & tru0).sum())
    FN = int((pred0 & tru1).sum())
    TN = int((pred0 & tru0).sum())

    # 指定の定義で算出
    # 捕捉率: TP/(TP+FN)
    # 見逃し率: FN/(TP+FN)
    # 空振り率: FP/(TP+FP)
    # スレットスコア: TP/(TP+FP+FN)
    # 適中率: (TP+TN)/(TP+FP+FN+TN)
    # 予想あり適中率（一致率）：TP/(TP+FP)
    capture = safe_div(TP, TP + FN)
    miss = safe_div(FN, TP + FN)
    false_alarm = safe_div(FP, TP + FP)
    threat = safe_div(TP, TP + FP + FN)
    accuracy = safe_div(TP + TN, TP + FP + FN + TN)
    match = safe_div(TP, TP + FP)

    return {
        "第1パターン(TP)": TP,
        "第2パターン(FP)": FP,
        "第3パターン(FN)": FN,
        "第4パターン(TN)": TN,
        "捕捉率": capture,
        "見逃し率": miss,
        "空振り率": false_alarm,
        "スレットスコア": threat,
        "適中率": accuracy,
        "予想あり適中率（一致率）":match,
        "評価対象行数": int(len(df)),
    }


def main():
    secret = load_config()

    INPUT_PATH = os.path.join(secret["output_root"],"totalling_result_rjer.csv")
    OUTPUT_PATH = os.path.join(secret["output_root"],"score_rjer.csv")
    # 必要列のみ読み込み（文字化け対策として encoding は環境に応じて調整可）
    df = pd.read_csv(INPUT_PATH)

    required = METHOD_COLS + [TRUTH_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"input.csv に必要な列がありません: {missing}")

    # 0/1 に正規化
    truth = _to_binary_series(df[TRUTH_COL], TRUTH_COL)

    results = []
    for col in METHOD_COLS:
        pred = _to_binary_series(df[col], col)
        metrics = compute_metrics(pred, truth)
        metrics["手法"] = col
        results.append(metrics)

    out_df = pd.DataFrame(results)

    # 列順を見やすく整形
    cols_order = [
        "手法",
        "評価対象行数",
        "第1パターン(TP)", "第2パターン(FP)", "第3パターン(FN)", "第4パターン(TN)",
        "捕捉率", "見逃し率", "空振り率", "スレットスコア", "適中率", "予想あり適中率（一致率）"
    ]
    out_df = out_df[cols_order]

    # 小数表示を整える（CSV自体は数値）
    out_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()