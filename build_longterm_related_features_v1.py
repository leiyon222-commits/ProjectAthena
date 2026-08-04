from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


DATABASE_PATH = Path("data/athena.db")

SOURCE_DATASET_PATH = Path(
    "data/market_context_expected_r_labels.csv"
)

OUTPUT_PATH = Path(
    "data/market_context_expected_r_enriched.csv"
)

FEATURE_SUMMARY_PATH = Path(
    "data/market_context_enriched_feature_summary.csv"
)

TIMEFRAME_NAME = "M5"
MAIN_SYMBOL = "USDJPY"

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

# USDが前にある通貨ペアは上昇＝USD強い。
# USDが後ろにある通貨ペアは上昇＝USD弱い。
USD_DIRECTION_SIGNS = {
    "EURUSD": -1.0,
    "GBPUSD": -1.0,
    "AUDUSD": -1.0,
    "NZDUSD": -1.0,
    "USDCAD": 1.0,
    "USDCHF": 1.0,
}

JPY_CROSS_SYMBOLS = [
    "EURJPY",
    "GBPJPY",
    "AUDJPY",
    "NZDJPY",
    "CADJPY",
    "CHFJPY",
]

# M5の本数で表したおおよその取引時間。
RELATED_RETURN_HORIZONS = {
    "1h": 12,
    "4h": 48,
    "1d": 288,
    "1w": 1440,
    "1m": 5760,
}

# 関連銘柄の時刻が少し欠けた場合だけ、
# 直前値を最大15分まで利用する。
RELATED_ASOF_TOLERANCE = pd.Timedelta(
    minutes=15
)


def normalize_time_precision(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    merge_asof用にtime列を
    datetime64[ns, UTC]へ統一する。
    """
    result = frame.copy()

    result["time"] = pd.to_datetime(
        result["time"],
        utc=True,
        errors="coerce",
    ).astype(
        "datetime64[ns, UTC]"
    )

    return result


def load_source_dataset() -> pd.DataFrame:
    """期待Rラベル付きの既存特徴量を読み込む。"""
    if not SOURCE_DATASET_PATH.exists():
        raise FileNotFoundError(
            "元データが見つかりません: "
            f"{SOURCE_DATASET_PATH.resolve()}"
        )

    dataset = pd.read_csv(
        SOURCE_DATASET_PATH,
    )

    if "time" not in dataset.columns:
        raise RuntimeError(
            "元データにtime列がありません"
        )

    dataset["time"] = pd.to_datetime(
        dataset["time"],
        utc=True,
        errors="coerce",
    )

    dataset = normalize_time_precision(
        dataset
    )

    dataset = (
        dataset.dropna(
            subset=["time"]
        )
        .sort_values("time")
        .drop_duplicates(
            subset=["time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return dataset


def load_symbol_candles(
    symbol: str,
    include_ohlc: bool,
) -> pd.DataFrame:
    """SQLiteから指定銘柄のM5を読み込む。"""
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            "データベースが見つかりません: "
            f"{DATABASE_PATH.resolve()}"
        )

    if include_ohlc:
        select_columns = """
            time,
            open,
            high,
            low,
            close
        """
    else:
        select_columns = """
            time,
            close
        """

    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:
        candles = pd.read_sql_query(
            f"""
            SELECT
                {select_columns}
            FROM candles
            WHERE symbol = ?
              AND timeframe = ?
            ORDER BY time ASC
            """,
            connection,
            params=(
                symbol,
                TIMEFRAME_NAME,
            ),
        )

    if candles.empty:
        raise RuntimeError(
            f"{symbol}のM5データがありません"
        )

    candles["time"] = pd.to_datetime(
        candles["time"],
        unit="s",
        utc=True,
        errors="coerce",
    )

    candles = normalize_time_precision(
        candles
    )

    numeric_columns = [
        column
        for column in candles.columns
        if column != "time"
    ]

    for column in numeric_columns:
        candles[column] = pd.to_numeric(
            candles[column],
            errors="coerce",
        )

    candles = (
        candles.dropna()
        .sort_values("time")
        .drop_duplicates(
            subset=["time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return candles


def calculate_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """RSIを計算する。"""
    change = close.diff()

    gain = change.clip(
        lower=0
    )

    loss = (
        -change.clip(
            upper=0
        )
    )

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
        average_gain
        / average_loss.replace(
            0,
            np.nan,
        )
    )

    rsi = (
        100
        - (
            100
            / (
                1
                + relative_strength
            )
        )
    )

    return rsi


def calculate_atr(
    frame: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """ATRを計算する。"""
    previous_close = frame[
        "close"
    ].shift(1)

    true_range = pd.concat(
        [
            frame["high"]
            - frame["low"],
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


def resample_completed_bars(
    candles: pd.DataFrame,
    frequency: str,
) -> pd.DataFrame:
    """
    M5から確定済み上位足を作る。

    例:
    8月1日00:00～8月2日00:00直前の日足は、
    8月2日00:00とラベル付けする。
    そのため、8月2日00:00以降だけが参照できる。
    """
    indexed = candles.set_index(
        "time"
    )

    resampled = indexed.resample(
        frequency,
        closed="left",
        label="right",
    ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
    )

    resampled = (
        resampled.dropna()
        .reset_index()
    )

    resampled = normalize_time_precision(
        resampled
    )

    return resampled


def build_daily_features(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """確定済み日足から長期特徴量を作る。"""
    features = daily[
        ["time"]
    ].copy()

    close = daily["close"]

    for days in [
        1,
        5,
        20,
        60,
        120,
        252,
    ]:
        features[
            f"long_d1_return_{days}"
        ] = close.pct_change(
            periods=days,
            fill_method=None,
        )

    for period in [
        20,
        50,
        200,
    ]:
        ema = close.ewm(
            span=period,
            adjust=False,
            min_periods=period,
        ).mean()

        features[
            f"long_d1_ema{period}_gap"
        ] = (
            close / ema - 1.0
        )

        features[
            f"long_d1_ema{period}_slope5"
        ] = ema.pct_change(
            periods=5,
            fill_method=None,
        )

    ema20 = close.ewm(
        span=20,
        adjust=False,
        min_periods=20,
    ).mean()

    ema50 = close.ewm(
        span=50,
        adjust=False,
        min_periods=50,
    ).mean()

    ema200 = close.ewm(
        span=200,
        adjust=False,
        min_periods=200,
    ).mean()

    features[
        "long_d1_trend_up"
    ] = (
        (close > ema20)
        & (ema20 > ema50)
        & (ema50 > ema200)
    ).astype("int8")

    features[
        "long_d1_trend_down"
    ] = (
        (close < ema20)
        & (ema20 < ema50)
        & (ema50 < ema200)
    ).astype("int8")

    features[
        "long_d1_rsi14"
    ] = calculate_rsi(
        close,
        period=14,
    ) / 100.0

    atr14 = calculate_atr(
        daily,
        period=14,
    )

    features[
        "long_d1_atr14_ratio"
    ] = (
        atr14 / close
    )

    rolling_high = daily[
        "high"
    ].rolling(
        window=20,
        min_periods=20,
    ).max()

    rolling_low = daily[
        "low"
    ].rolling(
        window=20,
        min_periods=20,
    ).min()

    range_width = (
        rolling_high - rolling_low
    ).replace(
        0,
        np.nan,
    )

    features[
        "long_d1_range_position20"
    ] = (
        close - rolling_low
    ) / range_width

    return features


def build_weekly_features(
    weekly: pd.DataFrame,
) -> pd.DataFrame:
    """確定済み週足から特徴量を作る。"""
    features = weekly[
        ["time"]
    ].copy()

    close = weekly["close"]

    for weeks in [
        1,
        4,
        13,
        26,
        52,
    ]:
        features[
            f"long_w1_return_{weeks}"
        ] = close.pct_change(
            periods=weeks,
            fill_method=None,
        )

    for period in [
        10,
        20,
        50,
    ]:
        ema = close.ewm(
            span=period,
            adjust=False,
            min_periods=period,
        ).mean()

        features[
            f"long_w1_ema{period}_gap"
        ] = (
            close / ema - 1.0
        )

        features[
            f"long_w1_ema{period}_slope2"
        ] = ema.pct_change(
            periods=2,
            fill_method=None,
        )

    ema10 = close.ewm(
        span=10,
        adjust=False,
        min_periods=10,
    ).mean()

    ema20 = close.ewm(
        span=20,
        adjust=False,
        min_periods=20,
    ).mean()

    ema50 = close.ewm(
        span=50,
        adjust=False,
        min_periods=50,
    ).mean()

    features[
        "long_w1_trend_up"
    ] = (
        (close > ema10)
        & (ema10 > ema20)
        & (ema20 > ema50)
    ).astype("int8")

    features[
        "long_w1_trend_down"
    ] = (
        (close < ema10)
        & (ema10 < ema20)
        & (ema20 < ema50)
    ).astype("int8")

    return features


def build_monthly_features(
    monthly: pd.DataFrame,
) -> pd.DataFrame:
    """確定済み月足から特徴量を作る。"""
    features = monthly[
        ["time"]
    ].copy()

    close = monthly["close"]

    for months in [
        1,
        3,
        6,
        12,
    ]:
        features[
            f"long_mn1_return_{months}"
        ] = close.pct_change(
            periods=months,
            fill_method=None,
        )

    for period in [
        3,
        6,
        12,
    ]:
        ema = close.ewm(
            span=period,
            adjust=False,
            min_periods=period,
        ).mean()

        features[
            f"long_mn1_ema{period}_gap"
        ] = (
            close / ema - 1.0
        )

        features[
            f"long_mn1_ema{period}_slope1"
        ] = ema.pct_change(
            periods=1,
            fill_method=None,
        )

    ema3 = close.ewm(
        span=3,
        adjust=False,
        min_periods=3,
    ).mean()

    ema6 = close.ewm(
        span=6,
        adjust=False,
        min_periods=6,
    ).mean()

    ema12 = close.ewm(
        span=12,
        adjust=False,
        min_periods=12,
    ).mean()

    features[
        "long_mn1_trend_up"
    ] = (
        (close > ema3)
        & (ema3 > ema6)
        & (ema6 > ema12)
    ).astype("int8")

    features[
        "long_mn1_trend_down"
    ] = (
        (close < ema3)
        & (ema3 < ema6)
        & (ema6 < ema12)
    ).astype("int8")

    return features


def merge_completed_features(
    base_times: pd.DataFrame,
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    """確定済み上位足特徴量をM5時刻へ結合する。"""
    return pd.merge_asof(
        base_times.sort_values(
            "time"
        ),
        feature_frame.sort_values(
            "time"
        ),
        on="time",
        direction="backward",
        allow_exact_matches=True,
    )


def align_related_close(
    base_times: pd.DataFrame,
    symbol_candles: pd.DataFrame,
    symbol: str,
) -> pd.Series:
    """関連銘柄の終値をUSDJPY基準時刻へ合わせる。"""
    renamed = symbol_candles.rename(
        columns={
            "close": (
                f"{symbol}_close"
            )
        }
    )

    aligned = pd.merge_asof(
        base_times.sort_values(
            "time"
        ),
        renamed.sort_values(
            "time"
        ),
        on="time",
        direction="backward",
        tolerance=(
            RELATED_ASOF_TOLERANCE
        ),
        allow_exact_matches=True,
    )

    close = aligned[
        f"{symbol}_close"
    ]

    # 一時的な数本欠損だけ補う。
    close = close.ffill(
        limit=3
    )

    return close


def build_related_features(
    base_times: pd.DataFrame,
) -> pd.DataFrame:
    """12通貨の動きとUSD・JPY強度を作る。"""
    related_features = pd.DataFrame(
        {
            "time": base_times[
                "time"
            ].copy()
        }
    )

    adjusted_usd_returns: dict[
        str,
        list[pd.Series],
    ] = {
        horizon: []
        for horizon
        in RELATED_RETURN_HORIZONS
    }

    jpy_weakness_returns: dict[
        str,
        list[pd.Series],
    ] = {
        horizon: []
        for horizon
        in RELATED_RETURN_HORIZONS
    }

    all_adjusted_returns: dict[
        str,
        list[pd.Series],
    ] = {
        horizon: []
        for horizon
        in RELATED_RETURN_HORIZONS
    }

    for symbol_number, symbol in enumerate(
        RELATED_SYMBOLS,
        start=1,
    ):
        print(
            f"関連銘柄特徴量: "
            f"{symbol_number}/{len(RELATED_SYMBOLS)} "
            f"{symbol}"
        )

        candles = load_symbol_candles(
            symbol=symbol,
            include_ohlc=False,
        )

        close = align_related_close(
            base_times=base_times,
            symbol_candles=candles,
            symbol=symbol,
        )

        for horizon, periods in (
            RELATED_RETURN_HORIZONS.items()
        ):
            log_return = np.log(
                close
                / close.shift(
                    periods
                )
            )

            feature_name = (
                f"rel_{symbol.lower()}"
                f"_logret_{horizon}"
            )

            related_features[
                feature_name
            ] = log_return.astype(
                "float32"
            )

            if symbol in (
                USD_DIRECTION_SIGNS
            ):
                adjusted = (
                    log_return
                    * USD_DIRECTION_SIGNS[
                        symbol
                    ]
                )

                adjusted_usd_returns[
                    horizon
                ].append(
                    adjusted
                )

                all_adjusted_returns[
                    horizon
                ].append(
                    adjusted
                )

            if symbol in JPY_CROSS_SYMBOLS:
                # JPYクロス上昇＝JPYが弱い
                adjusted = log_return

                jpy_weakness_returns[
                    horizon
                ].append(
                    adjusted
                )

                all_adjusted_returns[
                    horizon
                ].append(
                    adjusted
                )

    for horizon in (
        RELATED_RETURN_HORIZONS
    ):
        usd_frame = pd.concat(
            adjusted_usd_returns[
                horizon
            ],
            axis=1,
        )

        jpy_frame = pd.concat(
            jpy_weakness_returns[
                horizon
            ],
            axis=1,
        )

        all_frame = pd.concat(
            all_adjusted_returns[
                horizon
            ],
            axis=1,
        )

        usd_strength = (
            usd_frame.mean(
                axis=1
            )
        )

        jpy_weakness = (
            jpy_frame.mean(
                axis=1
            )
        )

        related_features[
            f"strength_usd_{horizon}"
        ] = usd_strength.astype(
            "float32"
        )

        related_features[
            f"strength_jpy_weakness_{horizon}"
        ] = jpy_weakness.astype(
            "float32"
        )

        related_features[
            f"strength_usdjpy_pressure_{horizon}"
        ] = (
            usd_strength
            + jpy_weakness
        ).astype(
            "float32"
        )

        related_features[
            f"strength_usd_positive_ratio_{horizon}"
        ] = (
            (
                usd_frame > 0
            ).mean(
                axis=1
            )
        ).astype(
            "float32"
        )

        related_features[
            f"strength_jpy_weak_ratio_{horizon}"
        ] = (
            (
                jpy_frame > 0
            ).mean(
                axis=1
            )
        ).astype(
            "float32"
        )

        related_features[
            f"strength_all_agreement_{horizon}"
        ] = (
            (
                all_frame > 0
            ).mean(
                axis=1
            )
        ).astype(
            "float32"
        )

        related_features[
            f"strength_usd_dispersion_{horizon}"
        ] = usd_frame.std(
            axis=1
        ).astype(
            "float32"
        )

        related_features[
            f"strength_jpy_dispersion_{horizon}"
        ] = jpy_frame.std(
            axis=1
        ).astype(
            "float32"
        )

    return related_features


def add_long_alignment_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """日・週・月・年間の方向一致数を作る。"""
    result = features.copy()

    direction_sources = [
        "long_d1_return_1",
        "long_d1_return_5",
        "long_d1_return_20",
        "long_d1_return_60",
        "long_d1_return_252",
        "long_w1_return_4",
        "long_w1_return_13",
        "long_w1_return_52",
        "long_mn1_return_1",
        "long_mn1_return_3",
        "long_mn1_return_12",
    ]

    available_sources = [
        column
        for column in direction_sources
        if column in result.columns
    ]

    up_flags = [
        (
            result[column] > 0
        ).astype("int8")
        for column in available_sources
    ]

    down_flags = [
        (
            result[column] < 0
        ).astype("int8")
        for column in available_sources
    ]

    result[
        "long_up_alignment_count"
    ] = pd.concat(
        up_flags,
        axis=1,
    ).sum(
        axis=1
    ).astype(
        "int8"
    )

    result[
        "long_down_alignment_count"
    ] = pd.concat(
        down_flags,
        axis=1,
    ).sum(
        axis=1
    ).astype(
        "int8"
    )

    result[
        "long_alignment_score"
    ] = (
        result[
            "long_up_alignment_count"
        ]
        - result[
            "long_down_alignment_count"
        ]
    ).astype(
        "int8"
    )

    result[
        "long_all_up"
    ] = (
        result[
            "long_up_alignment_count"
        ]
        == len(
            available_sources
        )
    ).astype(
        "int8"
    )

    result[
        "long_all_down"
    ] = (
        result[
            "long_down_alignment_count"
        ]
        == len(
            available_sources
        )
    ).astype(
        "int8"
    )

    return result


def create_feature_summary(
    enriched: pd.DataFrame,
    original_columns: set[str],
) -> pd.DataFrame:
    """新しく追加した特徴量の欠損率などを保存する。"""
    new_columns = [
        column
        for column in enriched.columns
        if column not in original_columns
        and column != "time"
    ]

    rows = []

    for column in new_columns:
        if column.startswith(
            "long_"
        ):
            group = "LONG_TIMEFRAME"

        elif column.startswith(
            "rel_"
        ):
            group = "RELATED_SYMBOL"

        elif column.startswith(
            "strength_"
        ):
            group = "CURRENCY_STRENGTH"

        else:
            group = "OTHER"

        rows.append(
            {
                "column": column,
                "group": group,
                "dtype": str(
                    enriched[column].dtype
                ),
                "missing_count": int(
                    enriched[column]
                    .isna()
                    .sum()
                ),
                "missing_ratio": float(
                    enriched[column]
                    .isna()
                    .mean()
                ),
                "minimum": float(
                    enriched[column]
                    .min()
                ),
                "maximum": float(
                    enriched[column]
                    .max()
                ),
                "mean": float(
                    enriched[column]
                    .mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    try:
        source = load_source_dataset()

        original_columns = set(
            source.columns
        )

        base_times = source[
            ["time"]
        ].copy()

        base_times = normalize_time_precision(
            base_times
        )

        print(
            "=== Athena 長期足・関連通貨特徴量作成 ==="
        )

        print(
            f"元データ: "
            f"{len(source):,}件"
        )

        print(
            "元期間:",
            source.iloc[0]["time"],
            "～",
            source.iloc[-1]["time"],
        )

        print(
            "\nUSDJPY上位足を作成中..."
        )

        usdjpy = load_symbol_candles(
            symbol=MAIN_SYMBOL,
            include_ohlc=True,
        )

        daily = resample_completed_bars(
            usdjpy,
            "1D",
        )

        weekly = resample_completed_bars(
            usdjpy,
            "W-MON",
        )

        monthly = resample_completed_bars(
            usdjpy,
            "MS",
        )

        daily_features = (
            build_daily_features(
                daily
            )
        )

        weekly_features = (
            build_weekly_features(
                weekly
            )
        )

        monthly_features = (
            build_monthly_features(
                monthly
            )
        )

        long_features = (
            merge_completed_features(
                base_times,
                daily_features,
            )
        )

        long_features = pd.merge_asof(
            long_features.sort_values(
                "time"
            ),
            weekly_features.sort_values(
                "time"
            ),
            on="time",
            direction="backward",
            allow_exact_matches=True,
        )

        long_features = pd.merge_asof(
            long_features.sort_values(
                "time"
            ),
            monthly_features.sort_values(
                "time"
            ),
            on="time",
            direction="backward",
            allow_exact_matches=True,
        )

        long_features = (
            add_long_alignment_features(
                long_features
            )
        )

        print(
            "\n関連12通貨の特徴量を作成中..."
        )

        related_features = (
            build_related_features(
                base_times
            )
        )

        new_features = (
            long_features.merge(
                related_features,
                on="time",
                how="inner",
                validate="one_to_one",
            )
        )

        new_feature_columns = [
            column
            for column in new_features.columns
            if column != "time"
        ]

        print(
            f"\n追加予定特徴量: "
            f"{len(new_feature_columns):,}個"
        )

        enriched = source.merge(
            new_features,
            on="time",
            how="inner",
            validate="one_to_one",
        )

        enriched = enriched.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        before_drop = len(
            enriched
        )

        enriched = enriched.dropna(
            subset=new_feature_columns
        ).reset_index(drop=True)

        removed_count = (
            before_drop - len(enriched)
        )

        print(
            f"年間特徴量などの準備期間で除外: "
            f"{removed_count:,}件"
        )

        float_columns = [
            column
            for column in new_feature_columns
            if pd.api.types.is_float_dtype(
                enriched[column]
            )
        ]

        enriched[
            float_columns
        ] = enriched[
            float_columns
        ].astype(
            "float32"
        )

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "\nCSVへ保存中..."
        )

        enriched.to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        summary = create_feature_summary(
            enriched=enriched,
            original_columns=(
                original_columns
            ),
        )

        summary.to_csv(
            FEATURE_SUMMARY_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        long_count = int(
            sum(
                column.startswith(
                    "long_"
                )
                for column
                in new_feature_columns
            )
        )

        related_count = int(
            sum(
                column.startswith(
                    "rel_"
                )
                for column
                in new_feature_columns
            )
        )

        strength_count = int(
            sum(
                column.startswith(
                    "strength_"
                )
                for column
                in new_feature_columns
            )
        )

        print(
            "\n=== 作成結果 ==="
        )

        print(
            f"利用可能データ: "
            f"{len(enriched):,}件"
        )

        print(
            f"全列数: "
            f"{len(enriched.columns):,}列"
        )

        print(
            f"追加特徴量: "
            f"{len(new_feature_columns):,}個"
        )

        print(
            f"長期足特徴量: "
            f"{long_count:,}個"
        )

        print(
            f"関連銘柄個別特徴量: "
            f"{related_count:,}個"
        )

        print(
            f"通貨強度特徴量: "
            f"{strength_count:,}個"
        )

        print(
            "期間:",
            enriched.iloc[0]["time"],
            "～",
            enriched.iloc[-1]["time"],
        )

        print(
            "\n長期トレンド一致の分布:"
        )

        print(
            enriched[
                "long_alignment_score"
            ].value_counts()
            .sort_index()
            .to_string()
        )

        print(
            "\n保存先:",
            OUTPUT_PATH.resolve(),
        )

        print(
            "特徴量一覧:",
            FEATURE_SUMMARY_PATH.resolve(),
        )

    except Exception as error:
        print(
            "\n長期足・関連通貨特徴量の"
            "作成中にエラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()
