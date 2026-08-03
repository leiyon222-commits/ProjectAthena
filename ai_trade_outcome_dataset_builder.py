from pathlib import Path
import sqlite3

import pandas as pd


SYMBOL = "USDJPY"
TIMEFRAME_NAME = "M5"

DATABASE_PATH = Path("data/athena.db")
OUTPUT_PATH = Path("data/ai_trade_outcome_dataset.csv")

POINT_SIZE = 0.001

EMA_PERIODS = [10, 20, 50, 100]
RSI_PERIOD = 14
ATR_PERIOD = 14

STOP_LOSS_ATR_MULTIPLIER = 1.5
TAKE_PROFIT_ATR_MULTIPLIER = 2.0

# 最大保有時間：M5を12本＝約1時間
MAX_HOLDING_BARS = 12

# 週末などの時間欠損を含む候補は除外
MAX_BAR_GAP_MINUTES = 15

DIRECTION_BUY = 1
DIRECTION_SELL = -1


def load_candles() -> pd.DataFrame:
    """SQLiteからUSDJPYのM5ローソク足を読み込む。"""
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"データベースが見つかりません: "
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
    """売買判断時点で利用できる特徴量を計算する。"""
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
            result["close"] - result[ema_column]
        ) / result["close"]

    result["ema10_20_gap"] = (
        result["ema10"] - result["ema20"]
    ) / result["close"]

    result["ema20_50_gap"] = (
        result["ema20"] - result["ema50"]
    ) / result["close"]

    result["ema50_100_gap"] = (
        result["ema50"] - result["ema100"]
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
                result["high"] - previous_close
            ).abs(),
            (
                result["low"] - previous_close
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

    result["return_1"] = (
        result["close"].pct_change(1)
    )

    result["return_3"] = (
        result["close"].pct_change(3)
    )

    result["return_6"] = (
        result["close"].pct_change(6)
    )

    result["return_12"] = (
        result["close"].pct_change(12)
    )

    result["return_24"] = (
        result["close"].pct_change(24)
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

    result["volume_mean_12"] = (
        result["tick_volume"]
        .rolling(12)
        .mean()
    )

    result["volume_ratio"] = (
        result["tick_volume"]
        / result["volume_mean_12"]
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
    """対象期間内に大きな時間欠損があるか確認する。"""
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


def determine_trade_outcome(
    candles: pd.DataFrame,
    signal_index: int,
    direction: int,
) -> dict | None:
    """
    次の足でエントリーし、
    SL・TPのどちらへ先に到達するか判定する。
    """
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

    entry_spread = (
        float(entry_row["spread"])
        * POINT_SIZE
    )

    stop_distance = (
        atr
        * STOP_LOSS_ATR_MULTIPLIER
    )

    target_distance = (
        atr
        * TAKE_PROFIT_ATR_MULTIPLIER
    )

    if direction == DIRECTION_BUY:
        # 買いはAsk価格でエントリー
        entry_price = (
            float(entry_row["open"])
            + entry_spread
        )

        stop_loss = (
            entry_price - stop_distance
        )

        take_profit = (
            entry_price + target_distance
        )

    else:
        # 売りはBid価格でエントリー
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

        high = float(
            current["high"]
        )

        low = float(
            current["low"]
        )

        close = float(
            current["close"]
        )

        current_spread = (
            float(current["spread"])
            * POINT_SIZE
        )

        if direction == DIRECTION_BUY:
            stop_hit = (
                low <= stop_loss
            )

            target_hit = (
                high >= take_profit
            )

            # 同じ足で両方に到達した場合は
            # 保守的に損切りを先と扱う
            if stop_hit:
                return {
                    "result": 0,
                    "exit_reason": "STOP_LOSS",
                    "holding_bars": (
                        check_index
                        - entry_index
                        + 1
                    ),
                    "entry_price": entry_price,
                    "exit_price": stop_loss,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }

            if target_hit:
                return {
                    "result": 1,
                    "exit_reason": "TAKE_PROFIT",
                    "holding_bars": (
                        check_index
                        - entry_index
                        + 1
                    ),
                    "entry_price": entry_price,
                    "exit_price": take_profit,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }

        else:
            # SELLはAsk価格で決済するため、
            # 高値・安値にスプレッドを加えて判定する
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
                    "result": 0,
                    "exit_reason": "STOP_LOSS",
                    "holding_bars": (
                        check_index
                        - entry_index
                        + 1
                    ),
                    "entry_price": entry_price,
                    "exit_price": stop_loss,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }

            if target_hit:
                return {
                    "result": 1,
                    "exit_reason": "TAKE_PROFIT",
                    "holding_bars": (
                        check_index
                        - entry_index
                        + 1
                    ),
                    "entry_price": entry_price,
                    "exit_price": take_profit,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }

        if check_index == final_index:
            if direction == DIRECTION_BUY:
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

            return {
                "result": int(
                    profit_distance > 0
                ),
                "exit_reason": "TIME_EXIT",
                "holding_bars": (
                    check_index
                    - entry_index
                    + 1
                ),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }

    return None


def create_dataset(
    candles: pd.DataFrame,
) -> pd.DataFrame:
    """
    各ローソク足について、
    BUY候補とSELL候補の2種類を作成する。
    """
    rows: list[dict] = []

    last_signal_index = (
        len(candles)
        - MAX_HOLDING_BARS
        - 2
    )

    for index in range(
        100,
        last_signal_index,
    ):
        current = candles.iloc[
            index
        ]

        feature_values = [
            current["ema10_gap_ratio"],
            current["ema20_gap_ratio"],
            current["ema50_gap_ratio"],
            current["ema100_gap_ratio"],
            current["ema10_20_gap"],
            current["ema20_50_gap"],
            current["ema50_100_gap"],
            current["rsi14"],
            current["atr14"],
            current["atr_ratio"],
            current["spread_atr_ratio"],
            current["volume_ratio"],
            current["body_ratio"],
            current["return_1"],
            current["return_3"],
            current["return_6"],
            current["return_12"],
            current["return_24"],
            current["volatility_12"],
            current["volatility_48"],
        ]

        if any(
            pd.isna(value)
            for value in feature_values
        ):
            continue

        for direction in [
            DIRECTION_BUY,
            DIRECTION_SELL,
        ]:
            outcome = determine_trade_outcome(
                candles,
                index,
                direction,
            )

            if outcome is None:
                continue

            direction_name = (
                "BUY"
                if direction == DIRECTION_BUY
                else "SELL"
            )

            rows.append(
                {
                    "signal_time": current["time"],
                    "direction": direction_name,
                    "direction_value": direction,
                    "close": float(
                        current["close"]
                    ),
                    "ema10_gap_ratio": float(
                        current["ema10_gap_ratio"]
                    ),
                    "ema20_gap_ratio": float(
                        current["ema20_gap_ratio"]
                    ),
                    "ema50_gap_ratio": float(
                        current["ema50_gap_ratio"]
                    ),
                    "ema100_gap_ratio": float(
                        current["ema100_gap_ratio"]
                    ),
                    "ema10_20_gap": float(
                        current["ema10_20_gap"]
                    ),
                    "ema20_50_gap": float(
                        current["ema20_50_gap"]
                    ),
                    "ema50_100_gap": float(
                        current["ema50_100_gap"]
                    ),
                    "rsi14": float(
                        current["rsi14"]
                    ),
                    "atr14": float(
                        current["atr14"]
                    ),
                    "atr_ratio": float(
                        current["atr_ratio"]
                    ),
                    "spread": float(
                        current["spread"]
                    ),
                    "spread_atr_ratio": float(
                        current["spread_atr_ratio"]
                    ),
                    "tick_volume": float(
                        current["tick_volume"]
                    ),
                    "volume_ratio": float(
                        current["volume_ratio"]
                    ),
                    "candle_body": float(
                        current["candle_body"]
                    ),
                    "candle_range": float(
                        current["candle_range"]
                    ),
                    "body_ratio": float(
                        current["body_ratio"]
                    ),
                    "upper_wick": float(
                        current["upper_wick"]
                    ),
                    "lower_wick": float(
                        current["lower_wick"]
                    ),
                    "return_1": float(
                        current["return_1"]
                    ),
                    "return_3": float(
                        current["return_3"]
                    ),
                    "return_6": float(
                        current["return_6"]
                    ),
                    "return_12": float(
                        current["return_12"]
                    ),
                    "return_24": float(
                        current["return_24"]
                    ),
                    "volatility_12": float(
                        current["volatility_12"]
                    ),
                    "volatility_48": float(
                        current["volatility_48"]
                    ),
                    "hour_utc": int(
                        current["hour_utc"]
                    ),
                    "day_of_week": int(
                        current["day_of_week"]
                    ),
                    "entry_price": float(
                        outcome["entry_price"]
                    ),
                    "exit_price": float(
                        outcome["exit_price"]
                    ),
                    "stop_loss": float(
                        outcome["stop_loss"]
                    ),
                    "take_profit": float(
                        outcome["take_profit"]
                    ),
                    "holding_bars": int(
                        outcome["holding_bars"]
                    ),
                    "exit_reason": outcome[
                        "exit_reason"
                    ],
                    "result": int(
                        outcome["result"]
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


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
                "売買結果データセットを"
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

        wins = int(
            dataset["result"].sum()
        )

        losses = (
            len(dataset) - wins
        )

        buy_rows = dataset[
            dataset["direction"] == "BUY"
        ]

        sell_rows = dataset[
            dataset["direction"] == "SELL"
        ]

        print(
            "\n=== 売買結果AIデータセット ==="
        )

        print(
            f"全候補数: "
            f"{len(dataset):,}件"
        )

        print(
            f"勝ち: "
            f"{wins:,}件"
        )

        print(
            f"負け: "
            f"{losses:,}件"
        )

        print(
            f"全体勝率: "
            f"{wins / len(dataset):.2%}"
        )

        print(
            f"BUY候補: "
            f"{len(buy_rows):,}件"
        )

        print(
            f"BUY勝率: "
            f"{buy_rows['result'].mean():.2%}"
        )

        print(
            f"SELL候補: "
            f"{len(sell_rows):,}件"
        )

        print(
            f"SELL勝率: "
            f"{sell_rows['result'].mean():.2%}"
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

    except Exception as error:
        print(
            "売買結果データセット作成中に"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()