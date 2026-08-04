from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


SYMBOL = "USDJPY"
TIMEFRAME_NAME = "M5"

DATABASE_PATH = Path("data/athena.db")
OUTPUT_PATH = Path(
    "data/market_context_features.csv"
)

POINT_SIZE = 0.001

RSI_PERIOD = 14
ATR_PERIOD = 14

# 12本先＝約1時間後
FUTURE_BARS = 12

# ATRの0.35倍未満の値動きは中立
TARGET_ATR_MULTIPLIER = 0.35

TARGET_DOWN = 0
TARGET_NEUTRAL = 1
TARGET_UP = 2

# 過去12本分のローソク足形状を個別に保存
CANDLE_LAG_COUNT = 12


def load_m5_candles() -> pd.DataFrame:
    """SQLiteからM5ローソク足を読み込む。"""
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

    candles = candles.drop_duplicates(
        subset=["time"],
        keep="last",
    )

    return candles.reset_index(drop=True)


def calculate_rsi(
    close: pd.Series,
    period: int = RSI_PERIOD,
) -> pd.Series:
    """RSIを計算する。"""
    change = close.diff()

    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = (
        average_gain / average_loss
    )

    return 100 - (
        100 / (1 + relative_strength)
    )


def calculate_atr(
    frame: pd.DataFrame,
    period: int = ATR_PERIOD,
) -> pd.Series:
    """ATRを計算する。"""
    previous_close = frame["close"].shift(1)

    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (
                frame["high"]
                - previous_close
            ).abs(),
            (
                frame["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def add_basic_candle_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """ローソク足の実体・ヒゲなどを数値化する。"""
    result = frame.copy()

    result["body"] = (
        result["close"] - result["open"]
    )

    result["body_abs"] = (
        result["body"].abs()
    )

    result["candle_range"] = (
        result["high"] - result["low"]
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

    safe_range = result[
        "candle_range"
    ].replace(0, np.nan)

    result["body_ratio"] = (
        result["body_abs"] / safe_range
    )

    result["upper_wick_ratio"] = (
        result["upper_wick"] / safe_range
    )

    result["lower_wick_ratio"] = (
        result["lower_wick"] / safe_range
    )

    result["close_position"] = (
        result["close"] - result["low"]
    ) / safe_range

    result["is_bullish"] = (
        result["close"] > result["open"]
    ).astype(int)

    result["is_bearish"] = (
        result["close"] < result["open"]
    ).astype(int)

    result["body_median_20"] = (
        result["body_abs"]
        .rolling(20)
        .median()
    )

    result["range_median_20"] = (
        result["candle_range"]
        .rolling(20)
        .median()
    )

    return result


def add_fourteen_patterns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    記事にある14種類を機械判定できる形にする。

    パターンの名称には複数の定義があるため、
    ここではAthena用の統一基準を使用する。
    """
    result = frame.copy()

    body_reference = (
        result["body_median_20"]
        .replace(0, np.nan)
    )

    # 1～4：大陽線・大陰線・小陽線・小陰線
    result["pattern_large_bullish"] = (
        (result["is_bullish"] == 1)
        & (
            result["body_abs"]
            >= body_reference * 1.5
        )
        & (
            result["body_ratio"] >= 0.60
        )
    ).astype(int)

    result["pattern_large_bearish"] = (
        (result["is_bearish"] == 1)
        & (
            result["body_abs"]
            >= body_reference * 1.5
        )
        & (
            result["body_ratio"] >= 0.60
        )
    ).astype(int)

    result["pattern_small_bullish"] = (
        (result["is_bullish"] == 1)
        & (
            result["body_abs"]
            <= body_reference * 0.50
        )
    ).astype(int)

    result["pattern_small_bearish"] = (
        (result["is_bearish"] == 1)
        & (
            result["body_abs"]
            <= body_reference * 0.50
        )
    ).astype(int)

    # 5～8：上影・下影の陽線と陰線
    long_upper_wick = (
        result["upper_wick"]
        >= result["body_abs"].clip(
            lower=POINT_SIZE
        ) * 2
    )

    long_lower_wick = (
        result["lower_wick"]
        >= result["body_abs"].clip(
            lower=POINT_SIZE
        ) * 2
    )

    result["pattern_upper_shadow_bullish"] = (
        (result["is_bullish"] == 1)
        & long_upper_wick
        & (
            result["upper_wick_ratio"] >= 0.50
        )
    ).astype(int)

    result["pattern_upper_shadow_bearish"] = (
        (result["is_bearish"] == 1)
        & long_upper_wick
        & (
            result["upper_wick_ratio"] >= 0.50
        )
    ).astype(int)

    result["pattern_lower_shadow_bullish"] = (
        (result["is_bullish"] == 1)
        & long_lower_wick
        & (
            result["lower_wick_ratio"] >= 0.50
        )
    ).astype(int)

    result["pattern_lower_shadow_bearish"] = (
        (result["is_bearish"] == 1)
        & long_lower_wick
        & (
            result["lower_wick_ratio"] >= 0.50
        )
    ).astype(int)

    previous_open = result["open"].shift(1)
    previous_close = result["close"].shift(1)
    previous_high = result["high"].shift(1)
    previous_low = result["low"].shift(1)

    previous_bullish = (
        previous_close > previous_open
    )

    previous_bearish = (
        previous_close < previous_open
    )

    # 9・10：スパイクハイ・スパイクロー
    prior_return_12 = (
        result["close"].pct_change(12)
    )

    result["pattern_spike_high"] = (
        long_upper_wick
        & (
            result["upper_wick_ratio"] >= 0.60
        )
        & (
            prior_return_12 > 0
        )
    ).astype(int)

    result["pattern_spike_low"] = (
        long_lower_wick
        & (
            result["lower_wick_ratio"] >= 0.60
        )
        & (
            prior_return_12 < 0
        )
    ).astype(int)

    # 11・12：陽線・陰線の包み足
    current_body_high = result[
        ["open", "close"]
    ].max(axis=1)

    current_body_low = result[
        ["open", "close"]
    ].min(axis=1)

    previous_body_high = pd.concat(
        [
            previous_open,
            previous_close,
        ],
        axis=1,
    ).max(axis=1)

    previous_body_low = pd.concat(
        [
            previous_open,
            previous_close,
        ],
        axis=1,
    ).min(axis=1)

    result["pattern_bullish_engulfing"] = (
        previous_bearish
        & (result["is_bullish"] == 1)
        & (
            current_body_high
            >= previous_body_high
        )
        & (
            current_body_low
            <= previous_body_low
        )
    ).astype(int)

    result["pattern_bearish_engulfing"] = (
        previous_bullish
        & (result["is_bearish"] == 1)
        & (
            current_body_high
            >= previous_body_high
        )
        & (
            current_body_low
            <= previous_body_low
        )
    ).astype(int)

    # 13・14：スラストアップ・スラストダウン
    result["pattern_thrust_up"] = (
        previous_bullish
        & (result["is_bullish"] == 1)
        & (
            result["close"] > previous_high
        )
    ).astype(int)

    result["pattern_thrust_down"] = (
        previous_bearish
        & (result["is_bearish"] == 1)
        & (
            result["close"] < previous_low
        )
    ).astype(int)

    return result


def add_m5_indicators(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """M5のテクニカル・相場環境を追加する。"""
    result = frame.copy()

    for period in [
        10,
        20,
        50,
        100,
        200,
    ]:
        ema = result["close"].ewm(
            span=period,
            adjust=False,
        ).mean()

        result[f"m5_ema{period}_gap"] = (
            result["close"] - ema
        ) / result["close"]

        result[f"m5_ema{period}_slope"] = (
            ema.pct_change(3)
        )

    result["m5_rsi14"] = calculate_rsi(
        result["close"]
    )

    result["m5_atr14"] = calculate_atr(
        result
    )

    result["m5_atr_ratio"] = (
        result["m5_atr14"]
        / result["close"]
    )

    result["spread_price"] = (
        result["spread"] * POINT_SIZE
    )

    result["spread_atr_ratio"] = (
        result["spread_price"]
        / result["m5_atr14"]
    )

    result["return_1"] = (
        result["close"].pct_change(1)
    )

    for period in [
        3,
        6,
        12,
        24,
        48,
        96,
    ]:
        result[f"return_{period}"] = (
            result["close"].pct_change(
                period
            )
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

    result["position_in_range_48"] = (
        result["close"] - result["low_48"]
    ) / (
        result["high_48"]
        - result["low_48"]
    ).replace(0, np.nan)

    result["volume_mean_48"] = (
        result["tick_volume"]
        .rolling(48)
        .mean()
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


def create_higher_timeframe_features(
    m5: pd.DataFrame,
    rule: str,
    prefix: str,
) -> pd.DataFrame:
    """
    M5から上位足を作成する。

    shift(1)により、現在時刻より前に確定した
    上位足だけを使用する。
    """
    indexed = m5.set_index("time")

    higher = indexed.resample(
        rule,
        label="right",
        closed="right",
    ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "tick_volume": "sum",
            "spread": "mean",
        }
    )

    higher = higher.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    higher[f"{prefix}_rsi14"] = (
        calculate_rsi(
            higher["close"]
        )
    )

    higher[f"{prefix}_atr14"] = (
        calculate_atr(higher)
    )

    ema20 = higher["close"].ewm(
        span=20,
        adjust=False,
    ).mean()

    ema50 = higher["close"].ewm(
        span=50,
        adjust=False,
    ).mean()

    ema100 = higher["close"].ewm(
        span=100,
        adjust=False,
    ).mean()

    higher[f"{prefix}_ema20_gap"] = (
        higher["close"] - ema20
    ) / higher["close"]

    higher[f"{prefix}_ema50_gap"] = (
        higher["close"] - ema50
    ) / higher["close"]

    higher[f"{prefix}_ema100_gap"] = (
        higher["close"] - ema100
    ) / higher["close"]

    higher[f"{prefix}_ema20_50_gap"] = (
        ema20 - ema50
    ) / higher["close"]

    higher[f"{prefix}_ema50_100_gap"] = (
        ema50 - ema100
    ) / higher["close"]

    higher[f"{prefix}_return_1"] = (
        higher["close"].pct_change(1)
    )

    higher[f"{prefix}_return_3"] = (
        higher["close"].pct_change(3)
    )

    higher[f"{prefix}_atr_ratio"] = (
        higher[f"{prefix}_atr14"]
        / higher["close"]
    )

    selected_columns = [
        column
        for column in higher.columns
        if column.startswith(
            f"{prefix}_"
        )
    ]

    # 1本前までに確定した上位足のみ利用
    higher = higher[
        selected_columns
    ].shift(1)

    return higher.reset_index()


def merge_higher_timeframe(
    base: pd.DataFrame,
    higher: pd.DataFrame,
) -> pd.DataFrame:
    """時刻以前の最新上位足をM5へ結合する。"""
    return pd.merge_asof(
        base.sort_values("time"),
        higher.sort_values("time"),
        on="time",
        direction="backward",
    )


def add_candle_sequence_lags(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """過去12本のローソク足形状を個別特徴量にする。"""
    result = frame.copy()

    sequence_columns = [
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "close_position",
        "is_bullish",
        "return_1",
        "tick_volume",
    ]

    for lag in range(
        1,
        CANDLE_LAG_COUNT + 1,
    ):
        for column in sequence_columns:
            result[
                f"lag{lag}_{column}"
            ] = result[column].shift(lag)

    return result


def add_target(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """12本後の方向を3分類ラベルにする。"""
    result = frame.copy()

    result["future_close"] = (
        result["close"]
        .shift(-FUTURE_BARS)
    )

    result["future_change"] = (
        result["future_close"]
        - result["close"]
    )

    result["target_threshold"] = (
        result["m5_atr14"]
        * TARGET_ATR_MULTIPLIER
    )

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
        candles = load_m5_candles()

        print(
            f"読み込んだM5ローソク足: "
            f"{len(candles):,}本"
        )

        features = add_basic_candle_features(
            candles
        )

        features = add_fourteen_patterns(
            features
        )

        features = add_m5_indicators(
            features
        )

        print("M15特徴量を作成中...")

        m15 = create_higher_timeframe_features(
            candles,
            rule="15min",
            prefix="m15",
        )

        features = merge_higher_timeframe(
            features,
            m15,
        )

        print("H1特徴量を作成中...")

        h1 = create_higher_timeframe_features(
            candles,
            rule="1h",
            prefix="h1",
        )

        features = merge_higher_timeframe(
            features,
            h1,
        )

        print("H4特徴量を作成中...")

        h4 = create_higher_timeframe_features(
            candles,
            rule="4h",
            prefix="h4",
        )

        features = merge_higher_timeframe(
            features,
            h4,
        )

        print("ローソク足の連続特徴量を作成中...")

        features = add_candle_sequence_lags(
            features
        )

        features = add_target(
            features
        )

        features = features.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        # 補助計算列のうち不要なものを削除
        drop_columns = [
            "body_median_20",
            "range_median_20",
            "high_48",
            "low_48",
            "volume_mean_48",
            "future_close",
            "target_threshold",
        ]

        features = features.drop(
            columns=[
                column
                for column in drop_columns
                if column in features.columns
            ]
        )

        features = features.dropna().copy()

        features["target"] = (
            features["target"].astype(int)
        )

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("CSVへ保存中...")

        features.to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        down_count = int(
            (
                features["target"]
                == TARGET_DOWN
            ).sum()
        )

        neutral_count = int(
            (
                features["target"]
                == TARGET_NEUTRAL
            ).sum()
        )

        up_count = int(
            (
                features["target"]
                == TARGET_UP
            ).sum()
        )

        pattern_columns = [
            column
            for column in features.columns
            if column.startswith(
                "pattern_"
            )
        ]

        pattern_counts = (
            features[pattern_columns]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        print(
            "\n=== 相場環境特徴量の作成結果 ==="
        )

        print(
            f"利用可能データ: "
            f"{len(features):,}件"
        )

        print(
            f"特徴量を含む全列数: "
            f"{len(features.columns):,}列"
        )

        print(
            f"DOWN: "
            f"{down_count:,}件 "
            f"({down_count / len(features):.2%})"
        )

        print(
            f"NEUTRAL: "
            f"{neutral_count:,}件 "
            f"({neutral_count / len(features):.2%})"
        )

        print(
            f"UP: "
            f"{up_count:,}件 "
            f"({up_count / len(features):.2%})"
        )

        print(
            "期間:",
            features.iloc[0]["time"],
            "～",
            features.iloc[-1]["time"],
        )

        print(
            "\n=== ローソク足パターン発生数 ==="
        )

        print(
            pattern_counts.to_string()
        )

        print(
            "\n保存先:",
            OUTPUT_PATH.resolve(),
        )

    except Exception as error:
        print(
            "\n相場環境特徴量の作成中に"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()