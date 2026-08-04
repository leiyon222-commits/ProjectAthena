from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

import MetaTrader5 as mt5
import pandas as pd


SYMBOL = "USDJPY"
TIMEFRAME = mt5.TIMEFRAME_M5
TIMEFRAME_NAME = "M5"

DATABASE_PATH = Path("data/athena.db")

# 直近5年分を取得
HISTORY_DAYS = 1825


def initialize_database(connection: sqlite3.Connection) -> None:
    """ローソク足保存用テーブルを作成する。"""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            time INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            tick_volume INTEGER NOT NULL,
            spread INTEGER NOT NULL,
            real_volume INTEGER NOT NULL,
            PRIMARY KEY (symbol, timeframe, time)
        )
        """
    )
    connection.commit()


def fetch_historical_candles() -> pd.DataFrame:
    """MT5から指定期間の過去ローソク足を取得する。"""
    if not mt5.symbol_select(SYMBOL, True):
        raise RuntimeError(
            f"{SYMBOL}を選択できませんでした: {mt5.last_error()}"
        )

    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=HISTORY_DAYS)

    print("取得開始:", date_from)
    print("取得終了:", date_to)

    rates = mt5.copy_rates_range(
        SYMBOL,
        TIMEFRAME,
        date_from,
        date_to,
    )

    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"過去データを取得できませんでした: {mt5.last_error()}"
        )

    return pd.DataFrame(rates)


def save_candles(
    connection: sqlite3.Connection,
    candles: pd.DataFrame,
) -> int:
    """ローソク足をSQLiteへ保存し、新規追加件数を返す。"""
    before_count = connection.total_changes

    rows = [
        (
            SYMBOL,
            TIMEFRAME_NAME,
            int(row.time),
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            int(row.tick_volume),
            int(row.spread),
            int(row.real_volume),
        )
        for row in candles.itertuples(index=False)
    ]

    connection.executemany(
        """
        INSERT OR IGNORE INTO candles (
            symbol,
            timeframe,
            time,
            open,
            high,
            low,
            close,
            tick_volume,
            spread,
            real_volume
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    connection.commit()

    return connection.total_changes - before_count


def main() -> None:
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not mt5.initialize():
        print("MT5への接続に失敗しました")
        print("エラー:", mt5.last_error())
        return

    try:
        candles = fetch_historical_candles()

        with sqlite3.connect(DATABASE_PATH) as connection:
            initialize_database(connection)

            inserted_count = save_candles(
                connection,
                candles,
            )

            total_count = connection.execute(
                """
                SELECT COUNT(*)
                FROM candles
                WHERE symbol = ? AND timeframe = ?
                """,
                (SYMBOL, TIMEFRAME_NAME),
            ).fetchone()[0]

        first_time = pd.to_datetime(
            candles.iloc[0]["time"],
            unit="s",
            utc=True,
        )

        last_time = pd.to_datetime(
            candles.iloc[-1]["time"],
            unit="s",
            utc=True,
        )

        print("\n=== 過去データ取得結果 ===")
        print("銘柄:", SYMBOL)
        print("時間足:", TIMEFRAME_NAME)
        print("取得本数:", len(candles))
        print("今回の新規保存:", inserted_count)
        print("データベース内合計:", total_count)
        print("最古データ:", first_time)
        print("最新データ:", last_time)
        print("保存先:", DATABASE_PATH.resolve())

    except Exception as error:
        print("過去データ取得中にエラーが発生しました")
        print(error)

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()