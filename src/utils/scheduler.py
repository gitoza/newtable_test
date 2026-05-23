"""
utils/scheduler.py
サーバー負荷を抑えるためのスリープ制御・優先度設定ユーティリティ。

戦略:
  1. ファイル間スリープ  : 1ファイル処理後に短時間待機（I/O負荷分散）
  2. 月間バッチ間スリープ: 月単位の処理が終わるたびに長めに待機
  3. プロセス優先度の低下: Windows の BELOW_NORMAL / IDLE クラスに設定
  4. ワーカー数制限      : ProcessPoolExecutor の max_workers を設定から取得
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Windows プロセス優先度設定
# ---------------------------------------------------------------------------

# Windows 優先度クラス定数
_PRIORITY_BELOW_NORMAL = 0x00004000
_PRIORITY_IDLE         = 0x00000040
_PRIORITY_NORMAL       = 0x00000020


def set_process_priority_low() -> bool:
    """
    現在のプロセスを Windows BELOW_NORMAL 優先度に設定する。
    Windows 以外では何もしない。

    Returns
    -------
    bool: 設定成功の場合 True
    """
    if platform.system() != "Windows":
        logger.debug("Windows以外のため優先度設定をスキップ")
        return False

    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetCurrentProcess()
        result = kernel32.SetPriorityClass(handle, _PRIORITY_BELOW_NORMAL)
        if result:
            logger.info("プロセス優先度を BELOW_NORMAL に設定しました")
        else:
            logger.warning("SetPriorityClass 失敗 (エラーコード: %d)", kernel32.GetLastError())
        return bool(result)
    except Exception as e:
        logger.warning("優先度設定エラー: %s", e)
        return False


# ---------------------------------------------------------------------------
# スリープヘルパー
# ---------------------------------------------------------------------------

def sleep_file_interval(settings: dict) -> None:
    """ファイル処理間スリープ（設定値が 0 以下の場合はスキップ）。"""
    sec = settings.get("parallel", {}).get("sleep_between_files", 0.1)
    if sec > 0:
        time.sleep(sec)


def sleep_month_interval(settings: dict) -> None:
    """月バッチ間スリープ。"""
    sec = settings.get("parallel", {}).get("sleep_between_months", 2.0)
    if sec > 0:
        logger.debug("月バッチ間スリープ %.1f 秒", sec)
        time.sleep(sec)


# ---------------------------------------------------------------------------
# ProcessPoolExecutor ラッパー
# ---------------------------------------------------------------------------

def get_max_workers(settings: dict) -> int:
    """設定から max_workers を取得する。"""
    return int(settings.get("parallel", {}).get("max_workers", 4))


def run_with_pool(
    func: Callable,
    tasks: Iterable[tuple],
    settings: dict,
    desc: str = "処理",
) -> list[Any]:
    """
    ProcessPoolExecutor でタスクリストを並列実行するラッパー。

    Parameters
    ----------
    func    : 各ワーカーで実行する関数 (*args でアンパック)
    tasks   : 引数タプルのイテラブル
    settings: 設定辞書
    desc    : ログ用の処理名

    Returns
    -------
    list: func の戻り値リスト（完了順、例外発生分は None）
    """
    max_workers = get_max_workers(settings)
    task_list = list(tasks)
    results: list[Any] = []

    logger.info("%s: %d タスクを %d ワーカーで実行", desc, len(task_list), max_workers)

    n_ok = 0
    n_failed = 0

    progress = (
        _tqdm(total=len(task_list), desc=desc, unit="task", dynamic_ncols=True)
        if _HAS_TQDM else None
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(func, *args): i for i, args in enumerate(task_list)}

        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                result = future.result()
                results.append(result)
                n_ok += 1
                if progress is not None:
                    status = result.get("status", "?") if isinstance(result, dict) else "ok"
                    progress.set_postfix(ok=n_ok, fail=n_failed, status=status, refresh=True)
                    progress.update(1)
            except Exception as e:
                logger.error("タスク %d 失敗: %s", idx, e, exc_info=True)
                results.append(None)
                n_failed += 1
                if progress is not None:
                    progress.set_postfix(ok=n_ok, fail=n_failed, status="FAIL", refresh=True)
                    progress.update(1)

            # タスク完了ごとに短いスリープ（ピーク負荷の分散）
            sleep_file_interval(settings)

    if progress is not None:
        progress.close()

    logger.info("%s: 完了 (%d/%d 成功, %d 失敗)", desc, n_ok, len(task_list), n_failed)
    return results
