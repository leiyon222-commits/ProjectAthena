from __future__ import annotations

from pathlib import Path
import gc
import math

import lightgbm as lgb
import numpy as np
import pandas as pd


DATASET_PATH = Path(
    "data/market_context_expected_r_enriched.csv"
)

FOLD_RESULTS_PATH = Path(
    "data/athena_feature_ablation_folds.csv"
)

VALIDATION_RESULTS_PATH = Path(
    "data/athena_feature_ablation_validation.csv"
)

GRID_RESULTS_PATH = Path(
    "data/athena_feature_ablation_grid.csv"
)

TEST_TRADES_PATH = Path(
    "data/athena_feature_ablation_trades.csv"
)

MONTHLY_RESULTS_PATH = Path(
    "data/athena_feature_ablation_monthly.csv"
)

INITIAL_TRAIN_RATIO = 0.40
SEGMENT_RATIO = 0.075
NUMBER_OF_FOLDS = 4

# 教師ラベルは最大48本先まで利用する
GAP_ROWS = 48

EARLY_STOP_RATIO = 0.10

ROLLING_SCORE_WINDOW = 5000
MIN_SCORE_HISTORY = 1000
MIN_VALIDATION_TRADES = 30

TOP_PERCENTAGES = [
    0.005,
    0.010,
    0.020,
    0.030,
    0.050,
    0.100,
]

DIRECTION_MARGINS = [
    0.000,
    0.025,
    0.050,
    0.075,
    0.100,
    0.150,
]

RESULT_COLUMNS = {
    "time",
    "buy_trade_r",
    "sell_trade_r",
    "buy_new_exit_reason",
    "sell_new_exit_reason",
    "buy_new_holding_bars",
    "sell_new_holding_bars",
}

RAW_PRICE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
}

FEATURE_SET_ORDER = [
    "BASE",
    "BASE_LONG",
    "BASE_LONG_STRENGTH",
]


def inspect_columns() -> tuple[
    list[str],
    dict[str, list[str]],
]:
    """CSVヘッダーから必要列と特徴量セットを決める。"""
    header = pd.read_csv(
        DATASET_PATH,
        nrows=0,
    ).columns.tolist()

    header_set = set(header)

    missing_results = (
        RESULT_COLUMNS - header_set
    )

    if missing_results:
        raise RuntimeError(
            "必要な結果列がありません: "
            f"{sorted(missing_results)}"
        )

    candidate_features = []

    for column in header:
        if column in RESULT_COLUMNS:
            continue

        if column in RAW_PRICE_COLUMNS:
            continue

        if column.startswith("pattern_"):
            continue

        # 関連銘柄個別60特徴量は今回は保留
        if column.startswith("rel_"):
            continue

        candidate_features.append(column)

    base_features = [
        column
        for column in candidate_features
        if not column.startswith("long_")
        and not column.startswith("strength_")
    ]

    long_features = [
        column
        for column in candidate_features
        if column.startswith("long_")
    ]

    strength_features = [
        column
        for column in candidate_features
        if column.startswith("strength_")
    ]

    feature_sets = {
        "BASE": base_features,
        "BASE_LONG": (
            base_features
            + long_features
        ),
        "BASE_LONG_STRENGTH": (
            base_features
            + long_features
            + strength_features
        ),
    }

    use_columns = list(
        dict.fromkeys(
            list(RESULT_COLUMNS)
            + feature_sets[
                "BASE_LONG_STRENGTH"
            ]
        )
    )

    return use_columns, feature_sets


def load_dataset(
    use_columns: list[str],
    feature_sets: dict[str, list[str]],
) -> pd.DataFrame:
    """必要列だけfloat32中心で読み込む。"""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "データセットが見つかりません: "
            f"{DATASET_PATH.resolve()}"
        )

    all_feature_columns = (
        feature_sets[
            "BASE_LONG_STRENGTH"
        ]
    )

    dtype_map = {
        column: "float32"
        for column in all_feature_columns
    }

    dtype_map.update(
        {
            "buy_trade_r": "float32",
            "sell_trade_r": "float32",
            "buy_new_holding_bars": (
                "float32"
            ),
            "sell_new_holding_bars": (
                "float32"
            ),
        }
    )

    dataset = pd.read_csv(
        DATASET_PATH,
        usecols=use_columns,
        dtype=dtype_map,
        parse_dates=["time"],
        low_memory=False,
    )

    dataset["time"] = pd.to_datetime(
        dataset["time"],
        utc=True,
        errors="coerce",
    )

    dataset = dataset.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    required_numeric = (
        all_feature_columns
        + [
            "buy_trade_r",
            "sell_trade_r",
            "buy_new_holding_bars",
            "sell_new_holding_bars",
        ]
    )

    dataset = (
        dataset.dropna(
            subset=[
                "time",
                *required_numeric,
            ]
        )
        .sort_values("time")
        .drop_duplicates(
            subset=["time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return dataset


def create_regressor(
    random_state: int,
) -> lgb.LGBMRegressor:
    """期待R回帰モデルを作る。"""
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=250,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def split_training_for_early_stop(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """学習期間内で本学習と監視期間を分ける。"""
    early_stop_start = int(
        len(train)
        * (1 - EARLY_STOP_RATIO)
    )

    fit_train = train.iloc[
        :early_stop_start - GAP_ROWS
    ].copy()

    early_stop = train.iloc[
        early_stop_start:
    ].copy()

    if fit_train.empty or early_stop.empty:
        raise RuntimeError(
            "学習期間の内部分割に失敗しました"
        )

    return fit_train, early_stop


def train_regressor(
    fit_train: pd.DataFrame,
    early_stop: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    random_state: int,
) -> lgb.LGBMRegressor:
    """BUYまたはSELL期待Rモデルを学習する。"""
    model = create_regressor(
        random_state=random_state
    )

    model.fit(
        fit_train[feature_columns],
        fit_train[target_column],
        eval_set=[
            (
                early_stop[
                    feature_columns
                ],
                early_stop[
                    target_column
                ],
            )
        ],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                first_metric_only=True,
                verbose=False,
            ),
            lgb.log_evaluation(
                period=0
            ),
        ],
    )

    return model


def calculate_pearson(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """実Rと予測RのPearson相関。"""
    if (
        np.std(actual) == 0
        or np.std(predicted) == 0
    ):
        return 0.0

    return float(
        np.corrcoef(
            actual,
            predicted,
        )[0, 1]
    )


def calculate_spearman(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """順位で見たSpearman相関。"""
    actual_rank = pd.Series(
        actual
    ).rank(
        method="average"
    )

    predicted_rank = pd.Series(
        predicted
    ).rank(
        method="average"
    )

    correlation = actual_rank.corr(
        predicted_rank
    )

    if pd.isna(correlation):
        return 0.0

    return float(correlation)


def create_prediction_arrays(
    rows: pd.DataFrame,
    buy_model: lgb.LGBMRegressor,
    sell_model: lgb.LGBMRegressor,
    feature_columns: list[str],
) -> dict[str, np.ndarray]:
    """BUY・SELL予測と採用方向を作る。"""
    buy_predictions = buy_model.predict(
        rows[feature_columns]
    )

    sell_predictions = sell_model.predict(
        rows[feature_columns]
    )

    selected_scores = np.maximum(
        buy_predictions,
        sell_predictions,
    )

    differences = np.abs(
        buy_predictions
        - sell_predictions
    )

    directions = np.where(
        buy_predictions
        > sell_predictions,
        "BUY",
        "SELL",
    )

    return {
        "buy": buy_predictions,
        "sell": sell_predictions,
        "score": selected_scores,
        "difference": differences,
        "direction": directions,
    }


def create_rolling_thresholds(
    history_scores: np.ndarray,
    current_scores: np.ndarray,
) -> dict[float, np.ndarray]:
    """
    現在より前の予測分布だけで
    上位X％閾値を計算する。
    """
    history_series = pd.Series(
        np.asarray(
            history_scores,
            dtype=float,
        )
    )

    current_series = pd.Series(
        np.asarray(
            current_scores,
            dtype=float,
        )
    )

    combined = pd.concat(
        [
            history_series,
            current_series,
        ],
        ignore_index=True,
    )

    prior_scores = combined.shift(1)

    current_start = len(
        history_series
    )

    thresholds = {}

    for top_percentage in (
        TOP_PERCENTAGES
    ):
        thresholds[
            top_percentage
        ] = (
            prior_scores.rolling(
                window=(
                    ROLLING_SCORE_WINDOW
                ),
                min_periods=(
                    MIN_SCORE_HISTORY
                ),
            )
            .quantile(
                1.0 - top_percentage
            )
            .iloc[
                current_start:
            ]
            .to_numpy()
        )

    return thresholds


def simulate_non_overlapping_trades(
    rows: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    rolling_thresholds: np.ndarray,
    direction_margin: float,
) -> pd.DataFrame:
    """条件を満たす非重複取引を再現する。"""
    trades = []
    next_allowed_time = None

    times = rows["time"].to_numpy()
    buy_r = rows[
        "buy_trade_r"
    ].to_numpy()
    sell_r = rows[
        "sell_trade_r"
    ].to_numpy()
    buy_holding = rows[
        "buy_new_holding_bars"
    ].to_numpy()
    sell_holding = rows[
        "sell_new_holding_bars"
    ].to_numpy()
    buy_reason = rows[
        "buy_new_exit_reason"
    ].to_numpy()
    sell_reason = rows[
        "sell_new_exit_reason"
    ].to_numpy()

    for index in range(len(rows)):
        signal_time = pd.Timestamp(
            times[index]
        )

        if (
            next_allowed_time is not None
            and signal_time
            < next_allowed_time
        ):
            continue

        threshold = float(
            rolling_thresholds[index]
        )

        if not np.isfinite(threshold):
            continue

        selected_score = float(
            predictions["score"][index]
        )

        difference = float(
            predictions[
                "difference"
            ][index]
        )

        if selected_score < threshold:
            continue

        if difference < direction_margin:
            continue

        direction = str(
            predictions[
                "direction"
            ][index]
        )

        if direction == "BUY":
            actual_r = float(
                buy_r[index]
            )
            holding_bars = int(
                buy_holding[index]
            )
            exit_reason = (
                buy_reason[index]
            )
        else:
            actual_r = float(
                sell_r[index]
            )
            holding_bars = int(
                sell_holding[index]
            )
            exit_reason = (
                sell_reason[index]
            )

        trades.append(
            {
                "time": signal_time,
                "direction": direction,
                "predicted_buy_r": float(
                    predictions["buy"][
                        index
                    ]
                ),
                "predicted_sell_r": float(
                    predictions["sell"][
                        index
                    ]
                ),
                "selected_predicted_r": (
                    selected_score
                ),
                "rolling_threshold": (
                    threshold
                ),
                "prediction_difference": (
                    difference
                ),
                "actual_r": actual_r,
                "holding_bars": (
                    holding_bars
                ),
                "exit_reason": (
                    exit_reason
                ),
            }
        )

        # 保守的に、終了足の次から
        # 新しいシグナルを許可する。
        next_allowed_time = (
            signal_time
            + pd.Timedelta(
                minutes=(
                    max(
                        1,
                        holding_bars,
                    )
                    + 1
                )
                * 5
            )
        )

    return pd.DataFrame(
        trades
    )


def calculate_metrics(
    trades: pd.DataFrame,
) -> dict[str, float | int]:
    """Rベースの売買成績を集計する。"""
    if trades.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "flats": 0,
            "win_rate": 0.0,
            "total_r": 0.0,
            "average_r": 0.0,
            "average_r_lcb95": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "buy_count": 0,
            "sell_count": 0,
        }

    positive = trades[
        trades["actual_r"] > 0
    ]

    negative = trades[
        trades["actual_r"] < 0
    ]

    flat = trades[
        trades["actual_r"] == 0
    ]

    gross_profit = float(
        positive["actual_r"].sum()
    )

    gross_loss = abs(
        float(
            negative[
                "actual_r"
            ].sum()
        )
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf")
    )

    average_r = float(
        trades["actual_r"].mean()
    )

    if len(trades) >= 2:
        standard_error = (
            float(
                trades[
                    "actual_r"
                ].std(ddof=1)
            )
            / math.sqrt(
                len(trades)
            )
        )

        lower_bound = (
            average_r
            - 1.96
            * standard_error
        )
    else:
        lower_bound = average_r

    equity = trades[
        "actual_r"
    ].cumsum()

    equity_with_start = pd.concat(
        [
            pd.Series([0.0]),
            equity.reset_index(
                drop=True
            ),
        ],
        ignore_index=True,
    )

    drawdown = (
        equity_with_start
        - equity_with_start.cummax()
    )

    return {
        "trades": int(
            len(trades)
        ),
        "wins": int(
            len(positive)
        ),
        "losses": int(
            len(negative)
        ),
        "flats": int(
            len(flat)
        ),
        "win_rate": float(
            len(positive)
            / len(trades)
            * 100
        ),
        "total_r": float(
            trades["actual_r"].sum()
        ),
        "average_r": average_r,
        "average_r_lcb95": float(
            lower_bound
        ),
        "profit_factor": float(
            profit_factor
        ),
        "max_drawdown_r": abs(
            float(
                drawdown.min()
            )
        ),
        "buy_count": int(
            (
                trades[
                    "direction"
                ]
                == "BUY"
            ).sum()
        ),
        "sell_count": int(
            (
                trades[
                    "direction"
                ]
                == "SELL"
            ).sum()
        ),
    }


def select_condition(
    fold_number: int,
    feature_set_name: str,
    validation: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    threshold_map: dict[
        float,
        np.ndarray,
    ],
) -> tuple[
    pd.Series | None,
    pd.DataFrame,
]:
    """1特徴量セット内で検証条件を決める。"""
    rows = []

    for top_percentage in (
        TOP_PERCENTAGES
    ):
        thresholds = threshold_map[
            top_percentage
        ]

        for margin in (
            DIRECTION_MARGINS
        ):
            trades = (
                simulate_non_overlapping_trades(
                    rows=validation,
                    predictions=predictions,
                    rolling_thresholds=(
                        thresholds
                    ),
                    direction_margin=(
                        margin
                    ),
                )
            )

            metrics = (
                calculate_metrics(
                    trades
                )
            )

            rows.append(
                {
                    "fold": fold_number,
                    "feature_set": (
                        feature_set_name
                    ),
                    "top_percentage": (
                        top_percentage
                    ),
                    "direction_margin": (
                        margin
                    ),
                    **metrics,
                }
            )

    grid = pd.DataFrame(rows)

    eligible = grid[
        (
            grid["trades"]
            >= MIN_VALIDATION_TRADES
        )
        & (
            grid["average_r"] > 0
        )
        & (
            grid["profit_factor"] > 1.0
        )
    ].copy()

    if eligible.empty:
        return None, grid

    eligible = eligible.sort_values(
        by=[
            "average_r_lcb95",
            "average_r",
            "profit_factor",
            "trades",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    )

    return eligible.iloc[0], grid


def choose_feature_set(
    candidates: list[dict],
) -> dict | None:
    """
    検証期間だけで特徴量セットまで選ぶ。
    テスト結果は選択に使わない。
    """
    if not candidates:
        return None

    ordered = sorted(
        candidates,
        key=lambda item: (
            item[
                "validation_average_r_lcb95"
            ],
            item[
                "validation_average_r"
            ],
            item[
                "validation_profit_factor"
            ],
            item[
                "validation_trades"
            ],
        ),
        reverse=True,
    )

    return ordered[0]


def print_metrics(
    metrics: dict[str, float | int],
) -> None:
    """テスト成績を表示する。"""
    print(
        f"取引回数: "
        f"{metrics['trades']:,}回"
    )

    print(
        f"勝率: "
        f"{metrics['win_rate']:.2f}%"
    )

    print(
        f"合計R: "
        f"{metrics['total_r']:.2f}R"
    )

    print(
        f"平均R: "
        f"{metrics['average_r']:.4f}R"
    )

    print(
        "平均Rの95％下限: "
        f"{metrics['average_r_lcb95']:.4f}R"
    )

    print(
        f"PF: "
        f"{metrics['profit_factor']:.2f}"
    )

    print(
        f"最大DD: "
        f"{metrics['max_drawdown_r']:.2f}R"
    )

    print(
        f"BUY: "
        f"{metrics['buy_count']:,}回"
    )

    print(
        f"SELL: "
        f"{metrics['sell_count']:,}回"
    )


def create_monthly_results(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """正式テスト取引を月別集計する。"""
    if trades.empty:
        return pd.DataFrame()

    source = trades.copy()

    source["month"] = (
        source["time"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        source.groupby(
            "month",
            as_index=False,
        )
        .agg(
            trades=("actual_r", "size"),
            total_r=("actual_r", "sum"),
            average_r=(
                "actual_r",
                "mean",
            ),
            wins=(
                "actual_r",
                lambda values: int(
                    (values > 0).sum()
                ),
            ),
            buy_count=(
                "direction",
                lambda values: int(
                    (
                        values == "BUY"
                    ).sum()
                ),
            ),
            sell_count=(
                "direction",
                lambda values: int(
                    (
                        values == "SELL"
                    ).sum()
                ),
            ),
        )
    )

    monthly["win_rate"] = (
        monthly["wins"]
        / monthly["trades"]
        * 100
    )

    return monthly


def main() -> None:
    try:
        (
            use_columns,
            feature_sets,
        ) = inspect_columns()

        dataset = load_dataset(
            use_columns=use_columns,
            feature_sets=feature_sets,
        )

        print(
            "=== Athena 特徴量アブレーション "
            "ウォークフォワード v1 ==="
        )

        print(
            f"全データ: "
            f"{len(dataset):,}件"
        )

        print(
            "期間:",
            dataset.iloc[0]["time"],
            "～",
            dataset.iloc[-1]["time"],
        )

        print(
            "\n比較する特徴量:"
        )

        for feature_set_name in (
            FEATURE_SET_ORDER
        ):
            print(
                f"{feature_set_name}: "
                f"{len(feature_sets[feature_set_name]):,}個"
            )

        total_count = len(dataset)

        initial_train_count = int(
            total_count
            * INITIAL_TRAIN_RATIO
        )

        segment_count = int(
            total_count
            * SEGMENT_RATIO
        )

        fold_rows = []
        validation_rows = []
        all_grid_frames = []
        all_test_trades = []

        for fold_index in range(
            NUMBER_OF_FOLDS
        ):
            fold_number = (
                fold_index + 1
            )

            train_end = (
                initial_train_count
                + fold_index
                * segment_count
                * 2
            )

            validation_end = (
                train_end
                + segment_count
            )

            test_end = min(
                validation_end
                + segment_count,
                total_count,
            )

            train = dataset.iloc[
                :train_end - GAP_ROWS
            ].copy()

            validation = dataset.iloc[
                train_end:
                validation_end - GAP_ROWS
            ].copy()

            test = dataset.iloc[
                validation_end:
                test_end - GAP_ROWS
            ].copy()

            if (
                train.empty
                or validation.empty
                or test.empty
            ):
                continue

            (
                fit_train,
                early_stop,
            ) = split_training_for_early_stop(
                train
            )

            print(
                "\n=============================="
            )

            print(
                f"Fold {fold_number}"
            )

            print(
                "学習期間:",
                train.iloc[0]["time"],
                "～",
                train.iloc[-1]["time"],
            )

            print(
                "検証期間:",
                validation.iloc[0]["time"],
                "～",
                validation.iloc[-1]["time"],
            )

            print(
                "テスト期間:",
                test.iloc[0]["time"],
                "～",
                test.iloc[-1]["time"],
            )

            candidates = []
            trained = {}

            for set_index, (
                feature_set_name
            ) in enumerate(
                FEATURE_SET_ORDER,
                start=1,
            ):
                feature_columns = (
                    feature_sets[
                        feature_set_name
                    ]
                )

                print(
                    "\n------------------------------"
                )

                print(
                    f"{set_index}/"
                    f"{len(FEATURE_SET_ORDER)} "
                    f"{feature_set_name} "
                    f"({len(feature_columns)}特徴量)"
                )

                print(
                    "BUY期待Rモデルを学習中..."
                )

                buy_model = train_regressor(
                    fit_train=fit_train,
                    early_stop=early_stop,
                    feature_columns=(
                        feature_columns
                    ),
                    target_column=(
                        "buy_trade_r"
                    ),
                    random_state=(
                        1000
                        + fold_number
                        * 10
                        + set_index
                    ),
                )

                print(
                    "SELL期待Rモデルを学習中..."
                )

                sell_model = train_regressor(
                    fit_train=fit_train,
                    early_stop=early_stop,
                    feature_columns=(
                        feature_columns
                    ),
                    target_column=(
                        "sell_trade_r"
                    ),
                    random_state=(
                        2000
                        + fold_number
                        * 10
                        + set_index
                    ),
                )

                early_predictions = (
                    create_prediction_arrays(
                        rows=early_stop,
                        buy_model=buy_model,
                        sell_model=(
                            sell_model
                        ),
                        feature_columns=(
                            feature_columns
                        ),
                    )
                )

                validation_predictions = (
                    create_prediction_arrays(
                        rows=validation,
                        buy_model=buy_model,
                        sell_model=(
                            sell_model
                        ),
                        feature_columns=(
                            feature_columns
                        ),
                    )
                )

                buy_pearson = (
                    calculate_pearson(
                        validation[
                            "buy_trade_r"
                        ].to_numpy(),
                        validation_predictions[
                            "buy"
                        ],
                    )
                )

                sell_pearson = (
                    calculate_pearson(
                        validation[
                            "sell_trade_r"
                        ].to_numpy(),
                        validation_predictions[
                            "sell"
                        ],
                    )
                )

                buy_spearman = (
                    calculate_spearman(
                        validation[
                            "buy_trade_r"
                        ].to_numpy(),
                        validation_predictions[
                            "buy"
                        ],
                    )
                )

                sell_spearman = (
                    calculate_spearman(
                        validation[
                            "sell_trade_r"
                        ].to_numpy(),
                        validation_predictions[
                            "sell"
                        ],
                    )
                )

                print(
                    f"BUY相関: "
                    f"Pearson {buy_pearson:.3f} / "
                    f"Spearman {buy_spearman:.3f}"
                )

                print(
                    f"SELL相関: "
                    f"Pearson {sell_pearson:.3f} / "
                    f"Spearman {sell_spearman:.3f}"
                )

                threshold_map = (
                    create_rolling_thresholds(
                        history_scores=(
                            early_predictions[
                                "score"
                            ]
                        ),
                        current_scores=(
                            validation_predictions[
                                "score"
                            ]
                        ),
                    )
                )

                (
                    best_condition,
                    grid,
                ) = select_condition(
                    fold_number=fold_number,
                    feature_set_name=(
                        feature_set_name
                    ),
                    validation=validation,
                    predictions=(
                        validation_predictions
                    ),
                    threshold_map=(
                        threshold_map
                    ),
                )

                all_grid_frames.append(
                    grid
                )

                validation_record = {
                    "fold": fold_number,
                    "feature_set": (
                        feature_set_name
                    ),
                    "feature_count": len(
                        feature_columns
                    ),
                    "buy_pearson": (
                        buy_pearson
                    ),
                    "sell_pearson": (
                        sell_pearson
                    ),
                    "buy_spearman": (
                        buy_spearman
                    ),
                    "sell_spearman": (
                        sell_spearman
                    ),
                }

                if best_condition is None:
                    print(
                        "有効な検証条件なし"
                    )

                    validation_rows.append(
                        {
                            **validation_record,
                            "status": (
                                "NO_CONDITION"
                            ),
                            "top_percentage": (
                                np.nan
                            ),
                            "direction_margin": (
                                np.nan
                            ),
                            "validation_trades": 0,
                            "validation_total_r": 0.0,
                            "validation_average_r": 0.0,
                            "validation_average_r_lcb95": 0.0,
                            "validation_profit_factor": 0.0,
                        }
                    )

                else:
                    print(
                        "最良検証条件: "
                        f"上位"
                        f"{float(best_condition['top_percentage']) * 100:.1f}% / "
                        f"方向差"
                        f"{float(best_condition['direction_margin']):.3f}R"
                    )

                    print(
                        f"検証: "
                        f"{int(best_condition['trades'])}回 / "
                        f"{float(best_condition['total_r']):+.2f}R / "
                        f"平均"
                        f"{float(best_condition['average_r']):+.4f}R / "
                        f"PF "
                        f"{float(best_condition['profit_factor']):.2f}"
                    )

                    candidate = {
                        "feature_set": (
                            feature_set_name
                        ),
                        "feature_count": len(
                            feature_columns
                        ),
                        "top_percentage": float(
                            best_condition[
                                "top_percentage"
                            ]
                        ),
                        "direction_margin": float(
                            best_condition[
                                "direction_margin"
                            ]
                        ),
                        "validation_trades": int(
                            best_condition[
                                "trades"
                            ]
                        ),
                        "validation_total_r": float(
                            best_condition[
                                "total_r"
                            ]
                        ),
                        "validation_average_r": float(
                            best_condition[
                                "average_r"
                            ]
                        ),
                        "validation_average_r_lcb95": float(
                            best_condition[
                                "average_r_lcb95"
                            ]
                        ),
                        "validation_profit_factor": float(
                            best_condition[
                                "profit_factor"
                            ]
                        ),
                    }

                    candidates.append(
                        candidate
                    )

                    validation_rows.append(
                        {
                            **validation_record,
                            "status": (
                                "CANDIDATE"
                            ),
                            **{
                                key: value
                                for key, value
                                in candidate.items()
                                if key not in {
                                    "feature_set",
                                    "feature_count",
                                }
                            },
                        }
                    )

                trained[
                    feature_set_name
                ] = {
                    "buy_model": buy_model,
                    "sell_model": (
                        sell_model
                    ),
                    "validation_predictions": (
                        validation_predictions
                    ),
                    "feature_columns": (
                        feature_columns
                    ),
                }

                del early_predictions
                gc.collect()

            winner = choose_feature_set(
                candidates
            )

            common_fold = {
                "fold": fold_number,
                "train_start": (
                    train.iloc[0]["time"]
                ),
                "train_end": (
                    train.iloc[-1]["time"]
                ),
                "validation_start": (
                    validation.iloc[0]["time"]
                ),
                "validation_end": (
                    validation.iloc[-1]["time"]
                ),
                "test_start": (
                    test.iloc[0]["time"]
                ),
                "test_end": (
                    test.iloc[-1]["time"]
                ),
            }

            if winner is None:
                print(
                    "\n正式採用:"
                )

                print(
                    "A・B・Cすべて有効条件なし"
                )

                fold_rows.append(
                    {
                        **common_fold,
                        "status": (
                            "NO_FEATURE_SET"
                        ),
                        "selected_feature_set": "",
                        "selected_feature_count": 0,
                        "top_percentage": (
                            np.nan
                        ),
                        "direction_margin": (
                            np.nan
                        ),
                        "validation_trades": 0,
                        "validation_total_r": 0.0,
                        "validation_average_r": 0.0,
                        "validation_average_r_lcb95": 0.0,
                        "validation_profit_factor": 0.0,
                        "test_trades": 0,
                        "test_total_r": 0.0,
                        "test_average_r": 0.0,
                        "test_average_r_lcb95": 0.0,
                        "test_profit_factor": 0.0,
                        "test_max_drawdown_r": 0.0,
                        "test_buy_count": 0,
                        "test_sell_count": 0,
                    }
                )

                del trained
                gc.collect()
                continue

            winner_name = winner[
                "feature_set"
            ]

            print(
                "\n正式採用:"
            )

            print(
                f"特徴量セット: "
                f"{winner_name}"
            )

            print(
                f"検証条件: "
                f"上位"
                f"{winner['top_percentage'] * 100:.1f}% / "
                f"方向差"
                f"{winner['direction_margin']:.3f}R"
            )

            selected = trained[
                winner_name
            ]

            test_predictions = (
                create_prediction_arrays(
                    rows=test,
                    buy_model=selected[
                        "buy_model"
                    ],
                    sell_model=selected[
                        "sell_model"
                    ],
                    feature_columns=selected[
                        "feature_columns"
                    ],
                )
            )

            test_threshold_map = (
                create_rolling_thresholds(
                    history_scores=selected[
                        "validation_predictions"
                    ]["score"],
                    current_scores=(
                        test_predictions[
                            "score"
                        ]
                    ),
                )
            )

            test_trades = (
                simulate_non_overlapping_trades(
                    rows=test,
                    predictions=(
                        test_predictions
                    ),
                    rolling_thresholds=(
                        test_threshold_map[
                            winner[
                                "top_percentage"
                            ]
                        ]
                    ),
                    direction_margin=winner[
                        "direction_margin"
                    ],
                )
            )

            test_metrics = (
                calculate_metrics(
                    test_trades
                )
            )

            print(
                "\n未使用テスト結果:"
            )

            print_metrics(
                test_metrics
            )

            if not test_trades.empty:
                test_trades[
                    "fold"
                ] = fold_number

                test_trades[
                    "feature_set"
                ] = winner_name

                test_trades[
                    "top_percentage"
                ] = winner[
                    "top_percentage"
                ]

                test_trades[
                    "direction_margin"
                ] = winner[
                    "direction_margin"
                ]

                all_test_trades.append(
                    test_trades
                )

            fold_rows.append(
                {
                    **common_fold,
                    "status": "EVALUATED",
                    "selected_feature_set": (
                        winner_name
                    ),
                    "selected_feature_count": (
                        winner[
                            "feature_count"
                        ]
                    ),
                    "top_percentage": (
                        winner[
                            "top_percentage"
                        ]
                    ),
                    "direction_margin": (
                        winner[
                            "direction_margin"
                        ]
                    ),
                    "validation_trades": (
                        winner[
                            "validation_trades"
                        ]
                    ),
                    "validation_total_r": (
                        winner[
                            "validation_total_r"
                        ]
                    ),
                    "validation_average_r": (
                        winner[
                            "validation_average_r"
                        ]
                    ),
                    "validation_average_r_lcb95": (
                        winner[
                            "validation_average_r_lcb95"
                        ]
                    ),
                    "validation_profit_factor": (
                        winner[
                            "validation_profit_factor"
                        ]
                    ),
                    "test_trades": (
                        test_metrics[
                            "trades"
                        ]
                    ),
                    "test_total_r": (
                        test_metrics[
                            "total_r"
                        ]
                    ),
                    "test_average_r": (
                        test_metrics[
                            "average_r"
                        ]
                    ),
                    "test_average_r_lcb95": (
                        test_metrics[
                            "average_r_lcb95"
                        ]
                    ),
                    "test_profit_factor": (
                        test_metrics[
                            "profit_factor"
                        ]
                    ),
                    "test_max_drawdown_r": (
                        test_metrics[
                            "max_drawdown_r"
                        ]
                    ),
                    "test_buy_count": (
                        test_metrics[
                            "buy_count"
                        ]
                    ),
                    "test_sell_count": (
                        test_metrics[
                            "sell_count"
                        ]
                    ),
                }
            )

            del trained
            del test_predictions
            gc.collect()

        fold_frame = pd.DataFrame(
            fold_rows
        )

        validation_frame = pd.DataFrame(
            validation_rows
        )

        FOLD_RESULTS_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fold_frame.to_csv(
            FOLD_RESULTS_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        validation_frame.to_csv(
            VALIDATION_RESULTS_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        if all_grid_frames:
            pd.concat(
                all_grid_frames,
                ignore_index=True,
            ).to_csv(
                GRID_RESULTS_PATH,
                index=False,
                encoding="utf-8-sig",
            )

        print(
            "\n=============================="
        )

        print(
            "=== 正式ウォークフォワード集計 ==="
        )

        evaluated = fold_frame[
            fold_frame["status"]
            == "EVALUATED"
        ]

        if evaluated.empty:
            print(
                "評価可能なFoldは"
                "ありませんでした。"
            )

        else:
            print(
                "\n各Foldの採用:"
            )

            print(
                evaluated[
                    [
                        "fold",
                        "selected_feature_set",
                        "validation_trades",
                        "validation_total_r",
                        "test_trades",
                        "test_total_r",
                        "test_profit_factor",
                    ]
                ].to_string(
                    index=False
                )
            )

            print(
                "\n特徴量セット採用回数:"
            )

            print(
                evaluated[
                    "selected_feature_set"
                ].value_counts()
                .to_string()
            )

            positive_folds = int(
                (
                    evaluated[
                        "test_total_r"
                    ] > 0
                ).sum()
            )

            print(
                f"\n評価できたFold: "
                f"{len(evaluated)} / "
                f"{NUMBER_OF_FOLDS}"
            )

            print(
                f"プラスFold: "
                f"{positive_folds} / "
                f"{len(evaluated)}"
            )

        if all_test_trades:
            combined_trades = pd.concat(
                all_test_trades,
                ignore_index=True,
            ).sort_values(
                "time"
            ).reset_index(
                drop=True
            )

            combined_metrics = (
                calculate_metrics(
                    combined_trades
                )
            )

            combined_trades.to_csv(
                TEST_TRADES_PATH,
                index=False,
                encoding="utf-8-sig",
            )

            print(
                "\n全未使用テスト合計:"
            )

            print_metrics(
                combined_metrics
            )

            monthly = (
                create_monthly_results(
                    combined_trades
                )
            )

            if not monthly.empty:
                monthly.to_csv(
                    MONTHLY_RESULTS_PATH,
                    index=False,
                    encoding="utf-8-sig",
                )

                positive_months = int(
                    (
                        monthly[
                            "total_r"
                        ] > 0
                    ).sum()
                )

                print(
                    "\n月別参考値:"
                )

                print(
                    f"取引があった月: "
                    f"{len(monthly)}か月"
                )

                print(
                    f"月平均取引: "
                    f"{monthly['trades'].mean():.2f}回"
                )

                print(
                    f"月平均損益: "
                    f"{monthly['total_r'].mean():+.2f}R"
                )

                print(
                    f"プラス月: "
                    f"{positive_months} / "
                    f"{len(monthly)}"
                )

                print(
                    f"最高月: "
                    f"{monthly['total_r'].max():+.2f}R"
                )

                print(
                    f"最低月: "
                    f"{monthly['total_r'].min():+.2f}R"
                )

        else:
            print(
                "\n未使用テスト取引は"
                "ありませんでした。"
            )

        print(
            "\nFold結果:",
            FOLD_RESULTS_PATH.resolve(),
        )

        print(
            "検証比較:",
            VALIDATION_RESULTS_PATH.resolve(),
        )

        print(
            "条件グリッド:",
            GRID_RESULTS_PATH.resolve(),
        )

        print(
            "テスト取引:",
            TEST_TRADES_PATH.resolve(),
        )

        print(
            "月別集計:",
            MONTHLY_RESULTS_PATH.resolve(),
        )

    except Exception as error:
        print(
            "\n特徴量アブレーション中に"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()
