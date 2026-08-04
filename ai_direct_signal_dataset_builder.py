from pathlib import Path
import sqlite3

import pandas as pd


SYMBOL = "USDJPY"
TIMEFRAME_NAME = "M5"

DATABASE_PATH = Path("data/athena.db")
OUTPUT_PATH = Path(
    "data/ai_direct_signal_dataset.csv"
)

POINT_SIZE = 0.001

EMA_PERIODS = [10, 20, 50, 100, 200]
RSI_PERIOD = 14
ATR_PERIOD = 14

# 今回の仮想売買条件
STOP_LOSS_ATR_MULTIPLIER = 2.0
TAKE_PROFIT_ATR_MULTIPLIER = 3.0
MAX_HOLDING_BARS = 24

# データの重複を減らすため、約1時間ごとに候補作成
SIGNAL_STEP = 12

# 15分を超えるデータ欠損を含む候補は除外
MAX_BAR_GAP_MINUTES = 15

# AIの教師ラベル
TARGET_SELL = 0
TARGET_NO_TRADE = 1
TARGET_BUY = 2


def load_candles() -> pd.DataFrame:
    """SQLiteからUSDJPYのM5データを読み込む。"""
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
                tick_volume,
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

    return candles


def calculate_features(
    candles: pd.DataFrame,
) -> pd.DataFrame:
    """その時点までに分かる特徴量を計算する。"""
    result = candles.copy()

    for period in EMA_PERIODS:
        ema_column = f"ema{period}"

        result[ema_column] = (
            result["close"]
            .ewm(
                span=period,
                adjust=False,
            )
            .mean()
        )

        result[f"{ema_column}_gap_ratio"] = (
            result["close"]
            - result[ema_column]
        ) / result["close"]

        result[f"{ema_column}_slope"] = (
            result[ema_column]
            .pct_change(3)
        )

    result["ema10_20_gap"] = (
        result["ema10"] - result["ema20"]
    ) / result["close"]

    result["ema20_50_gap"] = (
        result["ema20"] - result["ema50"]
    ) / result["close"]

    result["ema50_100_gap"] = (
        result["ema50"] - result["ema100"]
    ) / result["close"]

    result["ema100_200_gap"] = (
        result["ema100"] - result["ema200"]
    ) / result["close"]

    price_change = result["close"].diff()

    gain = price_change.clip(lower=0)
    loss = -price_change.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False,
        min_periods=RSI_PERIOD,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False,
        min_periods=RSI_PERIOD,
    ).mean()

    relative_strength = (
        average_gain / average_loss
    )

    result["rsi14"] = 100 - (
        100 / (1 + relative_strength)
    )

    previous_close = result["close"].shift(1)

    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (
                result["high"]
                - previous_close
            ).abs(),
            (
                result["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    result["atr14"] = true_range.ewm(
        alpha=1 / ATR_PERIOD,
        adjust=False,
        min_periods=ATR_PERIOD,
    ).mean()

    result["atr_ratio"] = (
        result["atr14"] / result["close"]
    )

    result["spread_price"] = (
        result["spread"] * POINT_SIZE
    )

    result["spread_atr_ratio"] = (
        result["spread_price"]
        / result["atr14"]
    )

    result["candle_body"] = (
        result["close"] - result["open"]
    )

    result["candle_range"] = (
        result["high"] - result["low"]
    )

    result["body_ratio"] = (
        result["candle_body"].abs()
        / result["candle_range"].replace(
            0,
            pd.NA,
        )
    )

    result["upper_wick"] = (
        result["high"]
        - result[
            ["open", "close"]
        ].max(axis=1)
    )

    result["lower_wick"] = (
        result[
            ["open", "close"]
        ].min(axis=1)
        - result["low"]
    )

    for period in [1, 3, 6, 12, 24, 48, 96]:
        result[f"return_{period}"] = (
            result["close"]
            .pct_change(period)
        )

    result["volatility_12"] = (
        result["return_1"]
        .rolling(12)
        .std()
    )

    result["volatility_48"] = (
        result["return_1"]
        .rolling(48)
        .std()
    )

    result["volatility_96"] = (
        result["return_1"]
        .rolling(96)
        .std()
    )

    result["high_12"] = (
        result["high"]
        .rolling(12)
        .max()
    )

    result["low_12"] = (
        result["low"]
        .rolling(12)
        .min()
    )

    result["high_48"] = (
        result["high"]
        .rolling(48)
        .max()
    )

    result["low_48"] = (
        result["low"]
        .rolling(48)
        .min()
    )

    result["position_in_range_12"] = (
        result["close"] - result["low_12"]
    ) / (
        result["high_12"]
        - result["low_12"]
    ).replace(0, pd.NA)

    result["position_in_range_48"] = (
        result["close"] - result["low_48"]
    ) / (
        result["high_48"]
        - result["low_48"]
    ).replace(0, pd.NA)

    result["volume_mean_12"] = (
        result["tick_volume"]
        .rolling(12)
        .mean()
    )

    result["volume_mean_48"] = (
        result["tick_volume"]
        .rolling(48)
        .mean()
    )

    result["volume_ratio_12"] = (
        result["tick_volume"]
        / result["volume_mean_12"]
    )

    result["volume_ratio_48"] = (
        result["tick_volume"]
        / result["volume_mean_48"]
    )

    result["hour_utc"] = (
        result["time"].dt.hour
    )

    result["day_of_week"] = (
        result["time"].dt.dayofweek
    )

    return result


def contains_large_time_gap(
    candles: pd.DataFrame,
    start_index: int,
    end_index: int,
) -> bool:
    """取引期間中に大きな時間欠損があるか調べる。"""
    times = candles.iloc[
        start_index:
        end_index + 1
    ]["time"]

    gaps = times.diff()

    return bool(
        (
            gaps
            > pd.Timedelta(
                minutes=MAX_BAR_GAP_MINUTES
            )
        ).any()
    )


def determine_outcome(
    candles: pd.DataFrame,
    signal_index: int,
    direction: str,
) -> dict | None:
    """指定方向の仮想売買結果を判定する。"""
    entry_index = signal_index + 1

    if entry_index >= len(candles):
        return None

    signal_row = candles.iloc[
        signal_index
    ]

    entry_row = candles.iloc[
        entry_index
    ]

    atr = float(
        signal_row["atr14"]
    )

    if pd.isna(atr) or atr <= 0:
        return None

    final_index = min(
        entry_index + MAX_HOLDING_BARS,
        len(candles) - 1,
    )

    if contains_large_time_gap(
        candles,
        entry_index,
        final_index,
    ):
        return None

    spread_price = (
        float(entry_row["spread"])
        * POINT_SIZE
    )

    stop_distance = (
        atr * STOP_LOSS_ATR_MULTIPLIER
    )

    target_distance = (
        atr * TAKE_PROFIT_ATR_MULTIPLIER
    )

    if direction == "BUY":
        entry_price = (
            float(entry_row["open"])
            + spread_price
        )

        stop_loss = (
            entry_price - stop_distance
        )

        take_profit = (
            entry_price + target_distance
        )

    else:
        entry_price = float(
            entry_row["open"]
        )

        stop_loss = (
            entry_price + stop_distance
        )

        take_profit = (
            entry_price - target_distance
        )

    for check_index in range(
        entry_index,
        final_index + 1,
    ):
        current = candles.iloc[
            check_index
        ]

        high = float(current["high"])
        low = float(current["low"])
        close = float(current["close"])

        current_spread = (
            float(current["spread"])
            * POINT_SIZE
        )

        if direction == "BUY":
            stop_hit = (
                low <= stop_loss
            )

            target_hit = (
                high >= take_profit
            )

            if stop_hit:
                return {
                    "win": 0,
                    "r_multiple": -1.0,
                    "exit_reason": "STOP_LOSS",
                }

            if target_hit:
                return {
                    "win": 1,
                    "r_multiple": (
                        TAKE_PROFIT_ATR_MULTIPLIER
                        / STOP_LOSS_ATR_MULTIPLIER
                    ),
                    "exit_reason": "TAKE_PROFIT",
                }

        else:
            stop_hit = (
                high + current_spread
                >= stop_loss
            )

            target_hit = (
                low + current_spread
                <= take_profit
            )

            if stop_hit:
                return {
                    "win": 0,
                    "r_multiple": -1.0,
                    "exit_reason": "STOP_LOSS",
                }

            if target_hit:
                return {
                    "win": 1,
                    "r_multiple": (
                        TAKE_PROFIT_ATR_MULTIPLIER
                        / STOP_LOSS_ATR_MULTIPLIER
                    ),
                    "exit_reason": "TAKE_PROFIT",
                }

        if check_index == final_index:
            if direction == "BUY":
                exit_price = close

                profit_distance = (
                    exit_price - entry_price
                )
            else:
                exit_price = (
                    close + current_spread
                )

                profit_distance = (
                    entry_price - exit_price
                )

            r_multiple = (
                profit_distance
                / stop_distance
            )

            return {
                "win": int(
                    r_multiple > 0
                ),
                "r_multiple": r_multiple,
                "exit_reason": "TIME_EXIT",
            }

    return None


def choose_target(
    buy_result: dict,
    sell_result: dict,
) -> int:
    """BUY・SELLの結果から教師ラベルを決める。"""
    buy_win = (
        buy_result["r_multiple"] > 0
    )

    sell_win = (
        sell_result["r_multiple"] > 0
    )

    if buy_win and not sell_win:
        return TARGET_BUY

    if sell_win and not buy_win:
        return TARGET_SELL

    return TARGET_NO_TRADE


def create_dataset(
    candles: pd.DataFrame,
) -> pd.DataFrame:
    """AIが直接売買判断するデータセットを作る。"""
    rows: list[dict] = []

    feature_columns = [
        "close",
        "ema10_gap_ratio",
        "ema20_gap_ratio",
        "ema50_gap_ratio",
        "ema100_gap_ratio",
        "ema200_gap_ratio",
        "ema10_slope",
        "ema20_slope",
        "ema50_slope",
        "ema100_slope",
        "ema200_slope",
        "ema10_20_gap",
        "ema20_50_gap",
        "ema50_100_gap",
        "ema100_200_gap",
        "rsi14",
        "atr14",
        "atr_ratio",
        "spread",
        "spread_atr_ratio",
        "tick_volume",
        "volume_ratio_12",
        "volume_ratio_48",
        "candle_body",
        "candle_range",
        "body_ratio",
        "upper_wick",
        "lower_wick",
        "return_1",
        "return_3",
        "return_6",
        "return_12",
        "return_24",
        "return_48",
        "return_96",
        "volatility_12",
        "volatility_48",
        "volatility_96",
        "position_in_range_12",
        "position_in_range_48",
        "hour_utc",
        "day_of_week",
    ]

    final_signal_index = (
        len(candles)
        - MAX_HOLDING_BARS
        - 2
    )

    for index in range(
        250,
        final_signal_index,
        SIGNAL_STEP,
    ):
        current = candles.iloc[
            index
        ]

        if any(
            pd.isna(current[column])
            for column in feature_columns
        ):
            continue

        buy_result = determine_outcome(
            candles=candles,
            signal_index=index,
            direction="BUY",
        )

        sell_result = determine_outcome(
            candles=candles,
            signal_index=index,
            direction="SELL",
        )

        if (
            buy_result is None
            or sell_result is None
        ):
            continue

        target = choose_target(
            buy_result,
            sell_result,
        )

        row = {
            "signal_time": current["time"],
        }

        for column in feature_columns:
            row[column] = float(
                current[column]
            )

        row.update(
            {
                "buy_r": float(
                    buy_result["r_multiple"]
                ),
                "sell_r": float(
                    sell_result["r_multiple"]
                ),
                "buy_exit_reason": (
                    buy_result["exit_reason"]
                ),
                "sell_exit_reason": (
                    sell_result["exit_reason"]
                ),
                "target": target,
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    try:
        candles = load_candles()

        print(
            f"読み込んだローソク足: "
            f"{len(candles):,}本"
        )

        candles = calculate_features(
            candles
        )

        dataset = create_dataset(
            candles
        )

        if dataset.empty:
            print(
                "直接判断AI用データを"
                "作成できませんでした"
            )
            return

        dataset = dataset.replace(
            [
                float("inf"),
                float("-inf"),
            ],
            pd.NA,
        ).dropna()

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset.to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        sell_count = int(
            (
                dataset["target"]
                == TARGET_SELL
            ).sum()
        )

        no_trade_count = int(
            (
                dataset["target"]
                == TARGET_NO_TRADE
            ).sum()
        )

        buy_count = int(
            (
                dataset["target"]
                == TARGET_BUY
            ).sum()
        )

        print(
            "\n=== AI直接判断データセット ==="
        )

        print(
            f"全データ: "
            f"{len(dataset):,}件"
        )

        print(
            f"SELL: "
            f"{sell_count:,}件 "
            f"({sell_count / len(dataset):.2%})"
        )

        print(
            f"NO TRADE: "
            f"{no_trade_count:,}件 "
            f"({no_trade_count / len(dataset):.2%})"
        )

        print(
            f"BUY: "
            f"{buy_count:,}件 "
            f"({buy_count / len(dataset):.2%})"
        )

        print(
            "期間:",
            dataset.iloc[0]["signal_time"],
            "～",
            dataset.iloc[-1]["signal_time"],
        )

        print(
            "保存先:",
            OUTPUT_PATH.resolve(),
        )

        print(
            "\nラベル:"
            "\n0 = SELL"
            "\n1 = NO TRADE"
            "\n2 = BUY"
        )

    except Exception as error:
        print(
            "直接判断AIデータセット作成中に"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()