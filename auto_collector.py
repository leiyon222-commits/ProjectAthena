from datetime import datetime
from pathlib import Path
import sqlite3
import time

import MetaTrader5 as mt5
import pandas as pd


SYMBOL = "USDJPY"
TIMEFRAME = mt5.TIMEFRAME_M5
TIMEFRAME_NAME = "M5"

# 1回の取得本数
BAR_COUNT = 10

# 監視間隔。30秒ごとに新しい足があるか確認する
CHECK_INTERVAL_SECONDS = 30

DATABASE_PATH = Path("data/athena.db")


def initialize_database(connection: sqlite3.Connection) -> None:
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

    candles = pd.DataFrame(rates)

    # 0番目には形成途中の足が含まれるため、
    # 最後の1本を除外して確定済みの足だけ保存する
    return candles.iloc[:-1]


def save_candles(
    connection: sqlite3.Connection,
    candles: pd.DataFrame,
) -> int:
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


def run_collection() -> None:
    candles = fetch_candles()

    with sqlite3.connect(DATABASE_PATH) as connection:
        initialize_database(connection)
        inserted_count = save_candles(connection, candles)

        total_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM candles
            WHERE symbol = ? AND timeframe = ?
            """,
            (SYMBOL, TIMEFRAME_NAME),
        ).fetchone()[0]

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if inserted_count > 0:
        print(
            f"[{current_time}] "
            f"新規保存: {inserted_count}本 / 合計: {total_count}本"
        )
    else:
        print(
            f"[{current_time}] "
            f"新しい確定足なし / 合計: {total_count}本"
        )


def main() -> None:
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Athena自動収集を開始します")
    print(f"銘柄: {SYMBOL}")
    print(f"時間足: {TIMEFRAME_NAME}")
    print(f"確認間隔: {CHECK_INTERVAL_SECONDS}秒")
    print("停止するには Ctrl+C を押してください\n")

    if not mt5.initialize():
        print("MT5への接続に失敗しました")
        print("エラー:", mt5.last_error())
        return

    try:
        while True:
            try:
                run_collection()
            except Exception as error:
                print(
                    f"[{datetime.now():%Y-%m-%d %H:%M:%S}] "
                    f"収集エラー: {error}"
                )

            time.sleep(CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nAthena自動収集を停止しました")

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()