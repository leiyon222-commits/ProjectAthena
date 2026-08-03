from pathlib import Path
import sqlite3

import MetaTrader5 as mt5
import pandas as pd


SYMBOL = "USDJPY"
TIMEFRAME = mt5.TIMEFRAME_M5
TIMEFRAME_NAME = "M5"
BAR_COUNT = 500

DATABASE_PATH = Path("data/athena.db")


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


def fetch_candles() -> pd.DataFrame:
    """MT5からローソク足を取得する。"""
    if not mt5.symbol_select(SYMBOL, True):
        raise RuntimeError(
            f"{SYMBOL}を選択できませんでした: {mt5.last_error()}"
        )

    rates = mt5.copy_rates_from_pos(
        SYMBOL,
        TIMEFRAME,
        0,
        BAR_COUNT,
    )

    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"ローソク足を取得できませんでした: {mt5.last_error()}"
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


def print_latest_candles(
    connection: sqlite3.Connection,
    limit: int = 5,
) -> None:
    """保存済みの最新ローソク足を表示する。"""
    rows = connection.execute(
        """
        SELECT
            datetime(time, 'unixepoch') AS utc_time,
            open,
            high,
            low,
            close,
            tick_volume,
            spread
        FROM candles
        WHERE symbol = ? AND timeframe = ?
        ORDER BY time DESC
        LIMIT ?
        """,
        (SYMBOL, TIMEFRAME_NAME, limit),
    ).fetchall()

    print("\n=== データベース内の最新ローソク足 ===")

    for row in reversed(rows):
        print(row)


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
        candles = fetch_candles()

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

            print("MT5接続成功")
            print("銘柄:", SYMBOL)
            print("時間足:", TIMEFRAME_NAME)
            print("取得本数:", len(candles))
            print("今回の新規保存:", inserted_count)
            print("データベース内の合計:", total_count)
            print("保存先:", DATABASE_PATH.resolve())

            print_latest_candles(connection)

    except Exception as error:
        print("処理中にエラーが発生しました")
        print(error)

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()