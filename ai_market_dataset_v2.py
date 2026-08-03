from pathlib import Path
import sqlite3

import pandas as pd


SYMBOL = "USDJPY"
TIMEFRAME_NAME = "M5"

DATABASE_PATH = Path("data/athena.db")
OUTPUT_PATH = Path("data/ai_market_dataset_v2.csv")

POINT_SIZE = 0.001

EMA_PERIODS = [10, 20, 50, 100]
RSI_PERIOD = 14
ATR_PERIOD = 14

# M5を12本先＝約1時間先
FUTURE_BARS = 12

# 1時間後の変動がATRの0.35倍未満ならNEUTRAL
TARGET_ATR_MULTIPLIER = 0.35

# 教師ラベル
TARGET_DOWN = 0
TARGET_NEUTRAL = 1
TARGET_UP = 2


def load_candles() -> pd.DataFrame:
    """SQLiteからローソク足を読み込む。"""
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
    """各ローソク足時点で分かる特徴量を計算する。"""
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


def create_three_class_target(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    約1時間後の価格から、
    DOWN・NEUTRAL・UPの3分類ラベルを作る。
    """
    result = dataset.copy()

    result["future_close"] = (
        result["close"]
        .shift(-FUTURE_BARS)
    )

    result["future_change"] = (
        result["future_close"]
        - result["close"]
    )

    result["target_threshold"] = (
        result["atr14"]
        * TARGET_ATR_MULTIPLIER
    )

    # 最初はすべてNEUTRAL
    result["target"] = TARGET_NEUTRAL

    result.loc[
        result["future_change"]
        >= result["target_threshold"],
        "target",
    ] = TARGET_UP

    result.loc[
        result["future_change"]
        <= -result["target_threshold"],
        "target",
    ] = TARGET_DOWN

    return result


def main() -> None:
    try:
        candles = load_candles()

        print(
            f"読み込んだローソク足: "
            f"{len(candles):,}本"
        )

        dataset = calculate_features(
            candles
        )

        dataset = create_three_class_target(
            dataset
        )

        columns = [
            "time",
            "close",
            "ema10_gap_ratio",
            "ema20_gap_ratio",
            "ema50_gap_ratio",
            "ema100_gap_ratio",
            "ema10_20_gap",
            "ema20_50_gap",
            "ema50_100_gap",
            "rsi14",
            "atr14",
            "atr_ratio",
            "spread",
            "spread_atr_ratio",
            "tick_volume",
            "volume_ratio",
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
            "volatility_12",
            "volatility_48",
            "hour_utc",
            "day_of_week",
            "future_close",
            "future_change",
            "target_threshold",
            "target",
        ]

        dataset = dataset[
            columns
        ].copy()

        dataset = dataset.replace(
            [
                float("inf"),
                float("-inf"),
            ],
            pd.NA,
        )

        # 未来12本が存在しない末尾行などを削除
        dataset = dataset.dropna()

        dataset["target"] = (
            dataset["target"].astype(int)
        )

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset.to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        down_count = int(
            (dataset["target"] == TARGET_DOWN).sum()
        )

        neutral_count = int(
            (
                dataset["target"]
                == TARGET_NEUTRAL
            ).sum()
        )

        up_count = int(
            (dataset["target"] == TARGET_UP).sum()
        )

        print(
            "\n=== 3分類AIデータセット作成結果 ==="
        )

        print(
            f"利用可能データ: "
            f"{len(dataset):,}件"
        )

        print(
            f"DOWN: "
            f"{down_count:,}件 "
            f"({down_count / len(dataset):.2%})"
        )

        print(
            f"NEUTRAL: "
            f"{neutral_count:,}件 "
            f"({neutral_count / len(dataset):.2%})"
        )

        print(
            f"UP: "
            f"{up_count:,}件 "
            f"({up_count / len(dataset):.2%})"
        )

        print(
            "期間:",
            dataset.iloc[0]["time"],
            "～",
            dataset.iloc[-1]["time"],
        )

        print(
            "保存先:",
            OUTPUT_PATH.resolve(),
        )

        print(
            "\nラベル:"
            "\n0 = DOWN"
            "\n1 = NEUTRAL"
            "\n2 = UP"
        )

    except Exception as error:
        print(
            "3分類AIデータセット作成中に"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()