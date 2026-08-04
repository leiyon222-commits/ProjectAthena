from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


DATABASE_PATH = Path("data/athena.db")
SOURCE_DATASET_PATH = Path(
    "data/market_context_trade_labels.csv"
)
OUTPUT_PATH = Path(
    "data/market_context_expected_r_labels.csv"
)

SYMBOL = "USDJPY"
TIMEFRAME_NAME = "M5"

POINT_SIZE = 0.001
MAX_BAR_GAP_MINUTES = 15

# 採用した出口条件
STOP_LOSS_ATR_MULTIPLIER = 2.0
TAKE_PROFIT_ATR_MULTIPLIER = 3.0
MAX_HOLDING_BARS = 48

OLD_RESULT_COLUMNS = {
    "target",
    "future_change",
    "buy_win",
    "sell_win",
    "buy_exit_reason",
    "sell_exit_reason",
    "buy_holding_bars",
    "sell_holding_bars",
    "buy_trade_r",
    "sell_trade_r",
    "buy_new_exit_reason",
    "sell_new_exit_reason",
    "buy_new_holding_bars",
    "sell_new_holding_bars",
}


def load_candles() -> pd.DataFrame:
    """SQLiteからUSDJPYのM5ローソク足を読み込む。"""
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            "データベースが見つかりません: "
            f"{DATABASE_PATH.resolve()}"
        )

    with sqlite3.connect(DATABASE_PATH) as connection:
        candles = pd.read_sql_query(
            """
            SELECT
                time,
                open,
                high,
                low,
                close,
                spread
            FROM candles
            WHERE symbol = ? AND timeframe = ?
            ORDER BY time ASC
            """,
            connection,
            params=(SYMBOL, TIMEFRAME_NAME),
        )

    if candles.empty:
        raise RuntimeError(
            "ローソク足データがありません"
        )

    candles["time"] = pd.to_datetime(
        candles["time"],
        unit="s",
        utc=True,
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "spread",
    ]

    for column in numeric_columns:
        candles[column] = pd.to_numeric(
            candles[column],
            errors="coerce",
        )

    candles = (
        candles.dropna(
            subset=[
                "time",
                "open",
                "high",
                "low",
                "close",
                "spread",
            ]
        )
        .drop_duplicates(
            subset=["time"],
            keep="last",
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    return candles


def load_source_dataset() -> pd.DataFrame:
    """これまで作成した特徴量データを読み込む。"""
    if not SOURCE_DATASET_PATH.exists():
        raise FileNotFoundError(
            "元データが見つかりません: "
            f"{SOURCE_DATASET_PATH.resolve()}"
        )

    dataset = pd.read_csv(
        SOURCE_DATASET_PATH,
        parse_dates=["time"],
    )

    dataset = (
        dataset.sort_values("time")
        .drop_duplicates(
            subset=["time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if "m5_atr14" not in dataset.columns:
        raise RuntimeError(
            "m5_atr14列がありません"
        )

    return dataset


def simulate_direction(
    open_values: np.ndarray,
    high_values: np.ndarray,
    low_values: np.ndarray,
    close_values: np.ndarray,
    spread_values: np.ndarray,
    time_values: np.ndarray,
    signal_index: int,
    atr: float,
    direction: str,
) -> dict | None:
    """
    次のM5足の始値で仮想エントリーし、
    SL・TP・時間切れのいずれかで決済する。

    BUY:
      Askでエントリーし、Bidで決済する。

    SELL:
      Bidでエントリーし、Askで決済する。

    同じ足でSLとTPの両方へ到達した場合は、
    保守的に損切り扱いとする。
    """
    entry_index = signal_index + 1

    if entry_index >= len(open_values):
        return None

    if not np.isfinite(atr) or atr <= 0:
        return None

    stop_distance = (
        atr * STOP_LOSS_ATR_MULTIPLIER
    )

    target_distance = (
        atr * TAKE_PROFIT_ATR_MULTIPLIER
    )

    final_index = min(
        entry_index + MAX_HOLDING_BARS - 1,
        len(open_values) - 1,
    )

    entry_spread = (
        float(spread_values[entry_index])
        * POINT_SIZE
    )

    if direction == "BUY":
        entry_price = (
            float(open_values[entry_index])
            + entry_spread
        )

        stop_loss = (
            entry_price - stop_distance
        )

        take_profit = (
            entry_price + target_distance
        )

    elif direction == "SELL":
        entry_price = float(
            open_values[entry_index]
        )

        stop_loss = (
            entry_price + stop_distance
        )

        take_profit = (
            entry_price - target_distance
        )

    else:
        raise ValueError(
            f"未対応の方向です: {direction}"
        )

    previous_time = pd.Timestamp(
        time_values[entry_index]
    )

    for check_index in range(
        entry_index,
        final_index + 1,
    ):
        current_time = pd.Timestamp(
            time_values[check_index]
        )

        if check_index > entry_index:
            time_gap = (
                current_time - previous_time
            )

            if time_gap > pd.Timedelta(
                minutes=MAX_BAR_GAP_MINUTES
            ):
                return None

        previous_time = current_time

        bid_high = float(
            high_values[check_index]
        )

        bid_low = float(
            low_values[check_index]
        )

        current_spread = (
            float(spread_values[check_index])
            * POINT_SIZE
        )

        if direction == "BUY":
            stop_hit = (
                bid_low <= stop_loss
            )

            target_hit = (
                bid_high >= take_profit
            )

        else:
            ask_high = (
                bid_high + current_spread
            )

            ask_low = (
                bid_low + current_spread
            )

            stop_hit = (
                ask_high >= stop_loss
            )

            target_hit = (
                ask_low <= take_profit
            )

        holding_bars = (
            check_index - entry_index + 1
        )

        if stop_hit and target_hit:
            return {
                "trade_r": -1.0,
                "exit_reason": "BOTH_HIT",
                "holding_bars": holding_bars,
            }

        if stop_hit:
            return {
                "trade_r": -1.0,
                "exit_reason": "STOP_LOSS",
                "holding_bars": holding_bars,
            }

        if target_hit:
            return {
                "trade_r": (
                    TAKE_PROFIT_ATR_MULTIPLIER
                    / STOP_LOSS_ATR_MULTIPLIER
                ),
                "exit_reason": "TAKE_PROFIT",
                "holding_bars": holding_bars,
            }

    # 4時間以内にSL・TPへ届かなければ、
    # 最終足の終値で実際のRを計算する。
    final_close = float(
        close_values[final_index]
    )

    final_spread = (
        float(spread_values[final_index])
        * POINT_SIZE
    )

    if direction == "BUY":
        exit_price = final_close

        trade_r = (
            exit_price - entry_price
        ) / stop_distance

    else:
        exit_price = (
            final_close + final_spread
        )

        trade_r = (
            entry_price - exit_price
        ) / stop_distance

    return {
        "trade_r": float(trade_r),
        "exit_reason": "TIME_EXIT",
        "holding_bars": (
            final_index - entry_index + 1
        ),
    }


def build_expected_r_labels(
    candles: pd.DataFrame,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """全特徴量行にBUY・SELLそれぞれの実Rを付ける。"""
    candle_index_by_time = {
        timestamp: index
        for index, timestamp
        in enumerate(candles["time"])
    }

    open_values = candles[
        "open"
    ].to_numpy()

    high_values = candles[
        "high"
    ].to_numpy()

    low_values = candles[
        "low"
    ].to_numpy()

    close_values = candles[
        "close"
    ].to_numpy()

    spread_values = candles[
        "spread"
    ].to_numpy()

    time_values = candles[
        "time"
    ].to_numpy()

    label_rows: list[dict] = []

    total_count = len(dataset)

    selected_columns = dataset[
        [
            "time",
            "m5_atr14",
        ]
    ]

    for row_number, row in enumerate(
        selected_columns.itertuples(
            index=False
        ),
        start=1,
    ):
        signal_index = (
            candle_index_by_time.get(
                row.time
            )
        )

        if signal_index is None:
            continue

        atr = float(
            row.m5_atr14
        )

        buy_result = simulate_direction(
            open_values=open_values,
            high_values=high_values,
            low_values=low_values,
            close_values=close_values,
            spread_values=spread_values,
            time_values=time_values,
            signal_index=signal_index,
            atr=atr,
            direction="BUY",
        )

        sell_result = simulate_direction(
            open_values=open_values,
            high_values=high_values,
            low_values=low_values,
            close_values=close_values,
            spread_values=spread_values,
            time_values=time_values,
            signal_index=signal_index,
            atr=atr,
            direction="SELL",
        )

        if (
            buy_result is None
            or sell_result is None
        ):
            continue

        label_rows.append(
            {
                "time": row.time,
                "buy_trade_r": (
                    buy_result["trade_r"]
                ),
                "sell_trade_r": (
                    sell_result["trade_r"]
                ),
                "buy_new_exit_reason": (
                    buy_result[
                        "exit_reason"
                    ]
                ),
                "sell_new_exit_reason": (
                    sell_result[
                        "exit_reason"
                    ]
                ),
                "buy_new_holding_bars": (
                    buy_result[
                        "holding_bars"
                    ]
                ),
                "sell_new_holding_bars": (
                    sell_result[
                        "holding_bars"
                    ]
                ),
            }
        )

        if (
            row_number % 25_000 == 0
            or row_number == total_count
        ):
            print(
                f"処理中: "
                f"{row_number:,} / "
                f"{total_count:,}"
            )

    return pd.DataFrame(
        label_rows
    )


def print_direction_summary(
    dataset: pd.DataFrame,
    direction: str,
) -> None:
    """BUYまたはSELLのR分布を表示する。"""
    r_column = (
        f"{direction.lower()}_trade_r"
    )

    exit_column = (
        f"{direction.lower()}"
        "_new_exit_reason"
    )

    holding_column = (
        f"{direction.lower()}"
        "_new_holding_bars"
    )

    r_values = dataset[
        r_column
    ]

    positive_count = int(
        (r_values > 0).sum()
    )

    negative_count = int(
        (r_values < 0).sum()
    )

    flat_count = int(
        (r_values == 0).sum()
    )

    print(
        f"\n--- {direction}結果 ---"
    )

    print(
        f"平均R: "
        f"{r_values.mean():.4f}R"
    )

    print(
        f"中央値: "
        f"{r_values.median():.4f}R"
    )

    print(
        f"プラス: "
        f"{positive_count:,}件 "
        f"({positive_count / len(dataset):.2%})"
    )

    print(
        f"マイナス: "
        f"{negative_count:,}件 "
        f"({negative_count / len(dataset):.2%})"
    )

    print(
        f"損益なし: "
        f"{flat_count:,}件 "
        f"({flat_count / len(dataset):.2%})"
    )

    print(
        f"平均保有: "
        f"{dataset[holding_column].mean():.1f}本"
    )

    exit_counts = (
        dataset[exit_column]
        .value_counts()
    )

    print(
        "決済理由:"
    )

    for reason, count in (
        exit_counts.items()
    ):
        print(
            f"  {reason}: "
            f"{count:,}件"
        )


def main() -> None:
    try:
        candles = load_candles()
        source_dataset = (
            load_source_dataset()
        )

        print(
            "=== Athena 期待R教師データ作成 ==="
        )

        print(
            f"ローソク足: "
            f"{len(candles):,}本"
        )

        print(
            f"元データ: "
            f"{len(source_dataset):,}件"
        )

        print(
            "\n固定する出口条件:"
        )

        print(
            f"損切り: ATR × "
            f"{STOP_LOSS_ATR_MULTIPLIER}"
        )

        print(
            f"利確: ATR × "
            f"{TAKE_PROFIT_ATR_MULTIPLIER}"
        )

        print(
            f"最大保有: "
            f"{MAX_HOLDING_BARS}本 "
            f"({MAX_HOLDING_BARS * 5 / 60:.1f}時間)"
        )

        print(
            "\nBUY・SELLの実Rを計算中..."
        )

        labels = build_expected_r_labels(
            candles=candles,
            dataset=source_dataset,
        )

        if labels.empty:
            raise RuntimeError(
                "新しい教師ラベルを"
                "作成できませんでした"
            )

        columns_to_drop = [
            column
            for column in source_dataset.columns
            if column in OLD_RESULT_COLUMNS
        ]

        cleaned_features = (
            source_dataset.drop(
                columns=columns_to_drop
            )
        )

        output = cleaned_features.merge(
            labels,
            on="time",
            how="inner",
        )

        output = output.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "\nCSVへ保存中..."
        )

        output.to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "\n=== 期待R教師データ作成結果 ==="
        )

        print(
            f"利用可能データ: "
            f"{len(output):,}件"
        )

        print(
            f"全列数: "
            f"{len(output.columns):,}列"
        )

        print_direction_summary(
            output,
            "BUY",
        )

        print_direction_summary(
            output,
            "SELL",
        )

        both_positive = int(
            (
                (output["buy_trade_r"] > 0)
                & (
                    output[
                        "sell_trade_r"
                    ] > 0
                )
            ).sum()
        )

        neither_positive = int(
            (
                (output["buy_trade_r"] <= 0)
                & (
                    output[
                        "sell_trade_r"
                    ] <= 0
                )
            ).sum()
        )

        print(
            "\n--- 両方向の関係 ---"
        )

        print(
            f"BUY・SELL両方プラス: "
            f"{both_positive:,}件 "
            f"({both_positive / len(output):.2%})"
        )

        print(
            f"どちらもプラスでない: "
            f"{neither_positive:,}件 "
            f"({neither_positive / len(output):.2%})"
        )

        print(
            "期間:",
            output.iloc[0]["time"],
            "～",
            output.iloc[-1]["time"],
        )

        print(
            "\n保存先:",
            OUTPUT_PATH.resolve(),
        )

    except Exception as error:
        print(
            "\n期待R教師データの作成中に"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()
