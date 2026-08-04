from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import time

import MetaTrader5 as mt5
import numpy as np
import pandas as pd


DATABASE_PATH = Path("data/athena.db")
SUMMARY_PATH = Path(
    "data/related_m5_import_summary.csv"
)

TABLE_NAME = "candles"
TIMEFRAME_NAME = "M5"
MT5_TIMEFRAME = mt5.TIMEFRAME_M5

START_UTC = datetime(
    2021,
    8,
    1,
    tzinfo=timezone.utc,
)

END_UTC = datetime.now(
    timezone.utc
)

# 90日ずつ取得し、巨大な1回取得を避ける
CHUNK_DAYS = 90

# SQLiteへ一度に書き込む件数
INSERT_BATCH_SIZE = 20_000

# 5年分として明らかに短い履歴を誤採用しないための条件
MIN_EXPECTED_BARS = 250_000
MAX_START_DELAY_DAYS = 21
MAX_RECENT_DELAY_DAYS = 10

RELATED_SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "AUDUSD",
    "NZDUSD",
    "USDCAD",
    "USDCHF",
    "EURJPY",
    "GBPJPY",
    "AUDJPY",
    "NZDJPY",
    "CADJPY",
    "CHFJPY",
]

POSSIBLE_INSERT_COLUMNS = [
    "symbol",
    "timeframe",
    "time",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
]

REQUIRED_DATABASE_COLUMNS = {
    "symbol",
    "timeframe",
    "time",
    "open",
    "high",
    "low",
    "close",
    "spread",
}


def initialize_mt5() -> None:
    """起動中のMT5へ接続する。"""
    if not mt5.initialize():
        raise RuntimeError(
            "MT5へ接続できませんでした: "
            f"{mt5.last_error()}"
        )

    account = mt5.account_info()

    if account is None:
        raise RuntimeError(
            "MT5口座情報を取得できませんでした: "
            f"{mt5.last_error()}"
        )

    print(
        f"接続先: {account.server}"
    )


def create_table_if_missing(
    connection: sqlite3.Connection,
) -> None:
    """
    candlesテーブルが存在しない場合だけ作る。
    既存テーブルがある場合は変更しない。
    """
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            time INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            tick_volume INTEGER,
            spread INTEGER NOT NULL,
            real_volume INTEGER,
            UNIQUE(symbol, timeframe, time)
        )
        """
    )

    connection.commit()


def get_database_columns(
    connection: sqlite3.Connection,
) -> list[str]:
    """既存candlesテーブルの列名を取得する。"""
    rows = connection.execute(
        f"PRAGMA table_info({TABLE_NAME})"
    ).fetchall()

    columns = [
        str(row[1])
        for row in rows
    ]

    missing = (
        REQUIRED_DATABASE_COLUMNS
        - set(columns)
    )

    if missing:
        raise RuntimeError(
            "candlesテーブルに必要な列がありません: "
            f"{sorted(missing)}"
        )

    return columns


def fetch_rates_chunk(
    symbol: str,
    date_from: datetime,
    date_to: datetime,
    retry_count: int = 3,
):
    """MT5から1区間を取得し、一時失敗時は再試行する。"""
    last_error = None

    for attempt in range(
        1,
        retry_count + 1,
    ):
        rates = mt5.copy_rates_range(
            symbol,
            MT5_TIMEFRAME,
            date_from,
            date_to,
        )

        if rates is not None:
            return rates

        last_error = mt5.last_error()

        print(
            f"  取得再試行 {attempt}/{retry_count}: "
            f"{last_error}"
        )

        time.sleep(1.0)

    raise RuntimeError(
        f"{symbol}の履歴取得に失敗しました: "
        f"{last_error}"
    )


def fetch_symbol_history(
    symbol: str,
) -> pd.DataFrame:
    """指定銘柄のM5履歴を90日単位で取得する。"""
    if not mt5.symbol_select(
        symbol,
        True,
    ):
        raise RuntimeError(
            f"{symbol}をMarket Watchへ追加できません: "
            f"{mt5.last_error()}"
        )

    frames: list[pd.DataFrame] = []

    chunk_start = START_UTC
    chunk_number = 0

    total_chunks = int(
        np.ceil(
            (
                END_UTC - START_UTC
            ).days
            / CHUNK_DAYS
        )
    )

    while chunk_start < END_UTC:
        chunk_end = min(
            chunk_start
            + timedelta(days=CHUNK_DAYS),
            END_UTC,
        )

        chunk_number += 1

        rates = fetch_rates_chunk(
            symbol=symbol,
            date_from=chunk_start,
            date_to=chunk_end,
        )

        if len(rates) > 0:
            frames.append(
                pd.DataFrame(rates)
            )

        print(
            f"  取得中: "
            f"{chunk_number}/{total_chunks} "
            f"({chunk_start.date()} ～ "
            f"{chunk_end.date()}) / "
            f"{len(rates):,}本"
        )

        # 境界の取りこぼしを避けるため、
        # 次の区間は同じ終了時刻から開始する。
        # 最後にtime重複を削除する。
        chunk_start = chunk_end

    if not frames:
        return pd.DataFrame()

    history = pd.concat(
        frames,
        ignore_index=True,
    )

    required_rate_columns = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "spread",
        "real_volume",
    ]

    missing = [
        column
        for column in required_rate_columns
        if column not in history.columns
    ]

    if missing:
        raise RuntimeError(
            f"{symbol}の取得結果に必要な列がありません: "
            f"{missing}"
        )

    history = (
        history[
            required_rate_columns
        ]
        .drop_duplicates(
            subset=["time"],
            keep="last",
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    numeric_columns = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "spread",
        "real_volume",
    ]

    for column in numeric_columns:
        history[column] = pd.to_numeric(
            history[column],
            errors="coerce",
        )

    history = history.dropna(
        subset=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "spread",
        ]
    ).reset_index(drop=True)

    history["time"] = (
        history["time"].astype(
            "int64"
        )
    )

    history["tick_volume"] = (
        history["tick_volume"]
        .fillna(0)
        .astype("int64")
    )

    history["spread"] = (
        history["spread"]
        .fillna(0)
        .astype("int64")
    )

    history["real_volume"] = (
        history["real_volume"]
        .fillna(0)
        .astype("int64")
    )

    history.insert(
        0,
        "timeframe",
        TIMEFRAME_NAME,
    )

    history.insert(
        0,
        "symbol",
        symbol,
    )

    return history


def validate_history(
    history: pd.DataFrame,
) -> dict:
    """5年学習へ使える長さかを判定する。"""
    if history.empty:
        return {
            "valid": False,
            "bars_ok": False,
            "start_ok": False,
            "recent_ok": False,
            "start_time": None,
            "end_time": None,
        }

    start_time = pd.to_datetime(
        int(history.iloc[0]["time"]),
        unit="s",
        utc=True,
    )

    end_time = pd.to_datetime(
        int(history.iloc[-1]["time"]),
        unit="s",
        utc=True,
    )

    bars_ok = (
        len(history)
        >= MIN_EXPECTED_BARS
    )

    start_ok = (
        start_time
        <= pd.Timestamp(START_UTC)
        + pd.Timedelta(
            days=MAX_START_DELAY_DAYS
        )
    )

    recent_ok = (
        end_time
        >= pd.Timestamp(END_UTC)
        - pd.Timedelta(
            days=MAX_RECENT_DELAY_DAYS
        )
    )

    return {
        "valid": bool(
            bars_ok
            and start_ok
            and recent_ok
        ),
        "bars_ok": bool(bars_ok),
        "start_ok": bool(start_ok),
        "recent_ok": bool(recent_ok),
        "start_time": start_time,
        "end_time": end_time,
    }


def replace_symbol_history(
    connection: sqlite3.Connection,
    history: pd.DataFrame,
    database_columns: list[str],
    symbol: str,
) -> int:
    """
    1銘柄分をトランザクション内で入れ替える。
    途中で失敗した場合は既存データを維持する。
    """
    insert_columns = [
        column
        for column
        in POSSIBLE_INSERT_COLUMNS
        if column in database_columns
    ]

    placeholders = ", ".join(
        ["?"] * len(insert_columns)
    )

    column_sql = ", ".join(
        insert_columns
    )

    insert_sql = (
        f"INSERT INTO {TABLE_NAME} "
        f"({column_sql}) "
        f"VALUES ({placeholders})"
    )

    connection.execute("BEGIN")

    try:
        connection.execute(
            f"""
            DELETE FROM {TABLE_NAME}
            WHERE symbol = ?
              AND timeframe = ?
            """,
            (
                symbol,
                TIMEFRAME_NAME,
            ),
        )

        for start_index in range(
            0,
            len(history),
            INSERT_BATCH_SIZE,
        ):
            batch = history.iloc[
                start_index:
                start_index
                + INSERT_BATCH_SIZE
            ]

            rows = list(
                batch[
                    insert_columns
                ].itertuples(
                    index=False,
                    name=None,
                )
            )

            connection.executemany(
                insert_sql,
                rows,
            )

            print(
                f"  DB保存中: "
                f"{min(start_index + len(batch), len(history)):,}"
                f" / {len(history):,}"
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    row = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TABLE_NAME}
        WHERE symbol = ?
          AND timeframe = ?
        """,
        (
            symbol,
            TIMEFRAME_NAME,
        ),
    ).fetchone()

    return int(row[0])


def format_timestamp(
    value,
) -> str:
    """CSV・画面表示向けに日時を文字列化する。"""
    if value is None:
        return ""

    return str(value)


def main() -> None:
    print(
        "=== Athena 関連12通貨 M5履歴取込 ==="
    )

    print(
        f"対象期間: "
        f"{START_UTC} ～ {END_UTC}"
    )

    print(
        f"対象銘柄: "
        f"{len(RELATED_SYMBOLS)}銘柄"
    )

    print(
        "\n注意:"
    )

    print(
        "5年分として履歴が不足している銘柄は、"
        "SQLiteへ保存せず結果だけ表示します。"
    )

    initialize_mt5()

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows: list[dict] = []

    try:
        with sqlite3.connect(
            DATABASE_PATH
        ) as connection:
            connection.execute(
                "PRAGMA journal_mode=WAL"
            )

            connection.execute(
                "PRAGMA synchronous=NORMAL"
            )

            create_table_if_missing(
                connection
            )

            database_columns = (
                get_database_columns(
                    connection
                )
            )

            print(
                "\nDB列:"
            )

            print(
                ", ".join(
                    database_columns
                )
            )

            for symbol_number, symbol in enumerate(
                RELATED_SYMBOLS,
                start=1,
            ):
                print(
                    "\n"
                    "=============================="
                )

                print(
                    f"{symbol_number}/{len(RELATED_SYMBOLS)} "
                    f"{symbol}"
                )

                try:
                    history = (
                        fetch_symbol_history(
                            symbol
                        )
                    )

                    validation = (
                        validate_history(
                            history
                        )
                    )

                    print(
                        f"  取得合計: "
                        f"{len(history):,}本"
                    )

                    print(
                        f"  取得開始: "
                        f"{format_timestamp(validation['start_time'])}"
                    )

                    print(
                        f"  取得終了: "
                        f"{format_timestamp(validation['end_time'])}"
                    )

                    if not validation["valid"]:
                        print(
                            "  判定: 履歴不足のため保存見送り"
                        )

                        summary_rows.append(
                            {
                                "symbol": symbol,
                                "status": (
                                    "SKIPPED_INCOMPLETE"
                                ),
                                "fetched_bars": (
                                    len(history)
                                ),
                                "database_bars": 0,
                                "start_time": (
                                    format_timestamp(
                                        validation[
                                            "start_time"
                                        ]
                                    )
                                ),
                                "end_time": (
                                    format_timestamp(
                                        validation[
                                            "end_time"
                                        ]
                                    )
                                ),
                                "bars_ok": (
                                    validation[
                                        "bars_ok"
                                    ]
                                ),
                                "start_ok": (
                                    validation[
                                        "start_ok"
                                    ]
                                ),
                                "recent_ok": (
                                    validation[
                                        "recent_ok"
                                    ]
                                ),
                                "error": "",
                            }
                        )

                        continue

                    database_bars = (
                        replace_symbol_history(
                            connection=connection,
                            history=history,
                            database_columns=(
                                database_columns
                            ),
                            symbol=symbol,
                        )
                    )

                    print(
                        f"  判定: 保存完了 "
                        f"({database_bars:,}本)"
                    )

                    summary_rows.append(
                        {
                            "symbol": symbol,
                            "status": "IMPORTED",
                            "fetched_bars": (
                                len(history)
                            ),
                            "database_bars": (
                                database_bars
                            ),
                            "start_time": (
                                format_timestamp(
                                    validation[
                                        "start_time"
                                    ]
                                )
                            ),
                            "end_time": (
                                format_timestamp(
                                    validation[
                                        "end_time"
                                    ]
                                )
                            ),
                            "bars_ok": (
                                validation[
                                    "bars_ok"
                                ]
                            ),
                            "start_ok": (
                                validation[
                                    "start_ok"
                                ]
                            ),
                            "recent_ok": (
                                validation[
                                    "recent_ok"
                                ]
                            ),
                            "error": "",
                        }
                    )

                except Exception as error:
                    print(
                        f"  エラー: {error}"
                    )

                    summary_rows.append(
                        {
                            "symbol": symbol,
                            "status": "ERROR",
                            "fetched_bars": 0,
                            "database_bars": 0,
                            "start_time": "",
                            "end_time": "",
                            "bars_ok": False,
                            "start_ok": False,
                            "recent_ok": False,
                            "error": str(error),
                        }
                    )

    finally:
        mt5.shutdown()

    summary = pd.DataFrame(
        summary_rows
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    imported_count = int(
        (
            summary["status"]
            == "IMPORTED"
        ).sum()
    )

    imported_bars = int(
        summary.loc[
            summary["status"]
            == "IMPORTED",
            "database_bars",
        ].sum()
    )

    print(
        "\n=============================="
    )

    print(
        "=== 取込結果 ==="
    )

    print(
        summary[
            [
                "symbol",
                "status",
                "fetched_bars",
                "database_bars",
                "start_time",
                "end_time",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\n保存成功: "
        f"{imported_count} / "
        f"{len(RELATED_SYMBOLS)}銘柄"
    )

    print(
        f"追加されたM5: "
        f"{imported_bars:,}本"
    )

    print(
        "\n結果CSV:",
        SUMMARY_PATH.resolve(),
    )

    if imported_count != len(
        RELATED_SYMBOLS
    ):
        print(
            "\n一部銘柄は保存できていません。"
        )

        print(
            "その場合は結果を貼ってください。"
        )


if __name__ == "__main__":
    main()
