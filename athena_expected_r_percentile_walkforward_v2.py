from pathlib import Path
import math

import lightgbm as lgb
import numpy as np
import pandas as pd


DATASET_PATH = Path(
    "data/market_context_expected_r_labels.csv"
)

FOLD_RESULTS_PATH = Path(
    "data/athena_expected_r_percentile_walkforward_folds.csv"
)

GRID_RESULTS_PATH = Path(
    "data/athena_expected_r_percentile_walkforward_grid.csv"
)

TEST_TRADES_PATH = Path(
    "data/athena_expected_r_percentile_walkforward_trades.csv"
)

MONTHLY_RESULTS_PATH = Path(
    "data/athena_expected_r_percentile_monthly.csv"
)

INITIAL_TRAIN_RATIO = 0.40
SEGMENT_RATIO = 0.075
NUMBER_OF_FOLDS = 4

# 教師ラベルは最大48本先まで使う
GAP_ROWS = 48

# 各学習期間の最後10％を早期終了監視兼、
# 順位判定の初期履歴として使う
EARLY_STOP_RATIO = 0.10

# 直近何本の予測分布で順位を判断するか
ROLLING_SCORE_WINDOW = 5000
MIN_SCORE_HISTORY = 1000

MIN_VALIDATION_TRADES = 30

# 上位何％の予測だけを取引するか
TOP_PERCENTAGES = [
    0.005,  # 上位0.5％
    0.010,  # 上位1％
    0.020,  # 上位2％
    0.030,  # 上位3％
    0.050,  # 上位5％
    0.100,  # 上位10％
]

# BUY期待RとSELL期待Rの予測差
DIRECTION_MARGINS = [
    0.000,
    0.025,
    0.050,
    0.075,
    0.100,
    0.150,
]

EXCLUDED_COLUMNS = {
    "time",
    "buy_trade_r",
    "sell_trade_r",
    "buy_new_exit_reason",
    "sell_new_exit_reason",
    "buy_new_holding_bars",
    "sell_new_holding_bars",
    "open",
    "high",
    "low",
    "close",
}


def load_dataset() -> pd.DataFrame:
    """期待R教師データを時刻順に読み込む。"""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "データセットが見つかりません: "
            f"{DATASET_PATH.resolve()}"
        )

    dataset = pd.read_csv(
        DATASET_PATH,
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

    dataset = dataset.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    required_columns = {
        "time",
        "buy_trade_r",
        "sell_trade_r",
        "buy_new_exit_reason",
        "sell_new_exit_reason",
        "buy_new_holding_bars",
        "sell_new_holding_bars",
    }

    missing = required_columns - set(
        dataset.columns
    )

    if missing:
        raise RuntimeError(
            "必要な列がありません: "
            f"{sorted(missing)}"
        )

    return dataset


def get_feature_columns(
    dataset: pd.DataFrame,
) -> list[str]:
    """未来の結果や生価格を除外する。"""
    feature_columns: list[str] = []

    for column in dataset.columns:
        if column in EXCLUDED_COLUMNS:
            continue

        if column.startswith("pattern_"):
            continue

        if pd.api.types.is_numeric_dtype(
            dataset[column]
        ):
            feature_columns.append(column)

    if not feature_columns:
        raise RuntimeError(
            "学習に使える特徴量がありません"
        )

    return feature_columns


def create_regressor() -> lgb.LGBMRegressor:
    """BUYまたはSELLの期待R回帰モデルを作る。"""
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=200,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def split_training_for_early_stop(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """学習期間の中だけで本学習と監視期間へ分ける。"""
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
) -> lgb.LGBMRegressor:
    """期待Rモデルを学習する。"""
    model = create_regressor()

    model.fit(
        fit_train[feature_columns],
        fit_train[target_column],
        eval_X=early_stop[feature_columns],
        eval_y=early_stop[target_column],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                first_metric_only=True,
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )

    return model


def calculate_correlation(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """実Rと予測Rの相関を計算する。"""
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


def create_prediction_arrays(
    rows: pd.DataFrame,
    buy_model: lgb.LGBMRegressor,
    sell_model: lgb.LGBMRegressor,
    feature_columns: list[str],
) -> dict[str, np.ndarray]:
    """BUY・SELL予測と方向差をまとめて作る。"""
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


def create_rolling_percentile_thresholds(
    history_scores: np.ndarray,
    current_scores: np.ndarray,
    top_percentages: list[float],
) -> dict[float, np.ndarray]:
    """
    現在時点より前の予測だけを使い、
    上位X％に入るための閾値を作る。

    current_scoresの現在値は、次の時点以降の
    履歴には加わるが、自分自身の閾値計算には使わない。
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

    prior_values = combined.shift(1)

    current_start = len(history_series)

    threshold_map: dict[
        float,
        np.ndarray,
    ] = {}

    for top_percentage in top_percentages:
        quantile = (
            1.0 - top_percentage
        )

        rolling_threshold = (
            prior_values.rolling(
                window=ROLLING_SCORE_WINDOW,
                min_periods=MIN_SCORE_HISTORY,
            )
            .quantile(quantile)
        )

        threshold_map[
            top_percentage
        ] = rolling_threshold.iloc[
            current_start:
        ].to_numpy()

    return threshold_map


def simulate_non_overlapping_trades(
    rows: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    rolling_thresholds: np.ndarray,
    direction_margin: float,
) -> pd.DataFrame:
    """ローリング順位条件を満たす取引だけを再現する。"""
    trades: list[dict] = []

    next_allowed_time = None

    for index in range(len(rows)):
        row = rows.iloc[index]
        signal_time = row["time"]

        if (
            next_allowed_time is not None
            and signal_time < next_allowed_time
        ):
            continue

        rolling_threshold = float(
            rolling_thresholds[index]
        )

        if not np.isfinite(
            rolling_threshold
        ):
            continue

        selected_score = float(
            predictions["score"][index]
        )

        difference = float(
            predictions["difference"][index]
        )

        if selected_score < rolling_threshold:
            continue

        if difference < direction_margin:
            continue

        direction = str(
            predictions["direction"][index]
        )

        if direction == "BUY":
            actual_r = float(
                row["buy_trade_r"]
            )
            holding_bars = int(
                row[
                    "buy_new_holding_bars"
                ]
            )
            exit_reason = row[
                "buy_new_exit_reason"
            ]
        else:
            actual_r = float(
                row["sell_trade_r"]
            )
            holding_bars = int(
                row[
                    "sell_new_holding_bars"
                ]
            )
            exit_reason = row[
                "sell_new_exit_reason"
            ]

        trades.append(
            {
                "time": signal_time,
                "direction": direction,
                "predicted_buy_r": float(
                    predictions["buy"][index]
                ),
                "predicted_sell_r": float(
                    predictions["sell"][index]
                ),
                "selected_predicted_r": (
                    selected_score
                ),
                "rolling_threshold": (
                    rolling_threshold
                ),
                "prediction_difference": (
                    difference
                ),
                "actual_r": actual_r,
                "holding_bars": holding_bars,
                "exit_reason": exit_reason,
            }
        )

        # シグナル足の次の足で入り、
        # holding_bars本目で終了する想定
        next_allowed_time = (
            signal_time
            + pd.Timedelta(
                minutes=max(
                    1,
                    holding_bars,
                ) * 5
            )
        )

    return pd.DataFrame(trades)


def calculate_trade_metrics(
    trades: pd.DataFrame,
) -> dict[str, float | int]:
    """実際のRを使って売買成績を集計する。"""
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
            "average_holding_bars": 0.0,
        }

    wins = trades[
        trades["actual_r"] > 0
    ]

    losses = trades[
        trades["actual_r"] < 0
    ]

    flats = trades[
        trades["actual_r"] == 0
    ]

    gross_profit = float(
        wins["actual_r"].sum()
    )

    gross_loss = abs(
        float(
            losses["actual_r"].sum()
        )
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit / gross_loss
        )
    else:
        profit_factor = float("inf")

    average_r = float(
        trades["actual_r"].mean()
    )

    if len(trades) >= 2:
        standard_deviation = float(
            trades["actual_r"].std(
                ddof=1
            )
        )

        standard_error = (
            standard_deviation
            / math.sqrt(len(trades))
        )

        average_r_lcb95 = (
            average_r
            - 1.96 * standard_error
        )
    else:
        average_r_lcb95 = average_r

    equity = trades[
        "actual_r"
    ].cumsum()

    equity_with_start = pd.concat(
        [
            pd.Series([0.0]),
            equity.reset_index(drop=True),
        ],
        ignore_index=True,
    )

    running_max = (
        equity_with_start.cummax()
    )

    drawdown = (
        equity_with_start
        - running_max
    )

    max_drawdown = abs(
        float(drawdown.min())
    )

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "flats": len(flats),
        "win_rate": (
            len(wins)
            / len(trades)
            * 100
        ),
        "total_r": float(
            trades["actual_r"].sum()
        ),
        "average_r": average_r,
        "average_r_lcb95": (
            average_r_lcb95
        ),
        "profit_factor": (
            profit_factor
        ),
        "max_drawdown_r": (
            max_drawdown
        ),
        "buy_count": int(
            (
                trades["direction"]
                == "BUY"
            ).sum()
        ),
        "sell_count": int(
            (
                trades["direction"]
                == "SELL"
            ).sum()
        ),
        "average_holding_bars": float(
            trades["holding_bars"].mean()
        ),
    }


def select_validation_condition(
    fold_number: int,
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
    """検証期間だけで上位割合と方向差を決める。"""
    results: list[dict] = []

    for top_percentage in TOP_PERCENTAGES:
        rolling_thresholds = (
            threshold_map[top_percentage]
        )

        for margin in DIRECTION_MARGINS:
            trades = (
                simulate_non_overlapping_trades(
                    rows=validation,
                    predictions=predictions,
                    rolling_thresholds=(
                        rolling_thresholds
                    ),
                    direction_margin=margin,
                )
            )

            metrics = (
                calculate_trade_metrics(
                    trades
                )
            )

            results.append(
                {
                    "fold": fold_number,
                    "top_percentage": (
                        top_percentage
                    ),
                    "direction_margin": (
                        margin
                    ),
                    **metrics,
                }
            )

    result_frame = pd.DataFrame(
        results
    )

    eligible = result_frame[
        (
            result_frame["trades"]
            >= MIN_VALIDATION_TRADES
        )
        & (
            result_frame["average_r"] > 0
        )
        & (
            result_frame["profit_factor"] > 1.0
        )
    ].copy()

    if eligible.empty:
        return None, result_frame

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

    return eligible.iloc[0], result_frame


def print_trade_metrics(
    title: str,
    metrics: dict[str, float | int],
) -> None:
    """売買成績を表示する。"""
    print(f"\n{title}")
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
    """未使用テスト取引を月単位へ集計する。"""
    if trades.empty:
        return pd.DataFrame()

    monthly_source = trades.copy()

    monthly_source["month"] = (
        monthly_source["time"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        monthly_source.groupby(
            "month",
            as_index=False,
        )
        .agg(
            trades=("actual_r", "size"),
            total_r=("actual_r", "sum"),
            average_r=("actual_r", "mean"),
            wins=(
                "actual_r",
                lambda values: int(
                    (values > 0).sum()
                ),
            ),
            buy_count=(
                "direction",
                lambda values: int(
                    (values == "BUY").sum()
                ),
            ),
            sell_count=(
                "direction",
                lambda values: int(
                    (values == "SELL").sum()
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
        dataset = load_dataset()

        feature_columns = (
            get_feature_columns(
                dataset
            )
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

        print(
            "=== Athena 期待R順位 "
            "ウォークフォワード v2 ==="
        )

        print(
            f"全データ: "
            f"{total_count:,}件"
        )

        print(
            f"特徴量: "
            f"{len(feature_columns):,}個"
        )

        print(
            f"ローリング順位履歴: "
            f"{ROLLING_SCORE_WINDOW:,}本"
        )

        print(
            f"フォールド数: "
            f"{NUMBER_OF_FOLDS}"
        )

        fold_results: list[dict] = []
        all_grid_results: list[
            pd.DataFrame
        ] = []
        all_test_trades: list[
            pd.DataFrame
        ] = []

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

            test_end = (
                validation_end
                + segment_count
            )

            if test_end > total_count:
                test_end = total_count

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
                print(
                    f"\nFold {fold_number}: "
                    "期間不足のためスキップ"
                )
                continue

            (
                fit_train,
                early_stop,
            ) = split_training_for_early_stop(
                train
            )

            print(
                "\n===================="
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
            )

            early_predictions = (
                create_prediction_arrays(
                    rows=early_stop,
                    buy_model=buy_model,
                    sell_model=sell_model,
                    feature_columns=(
                        feature_columns
                    ),
                )
            )

            validation_predictions = (
                create_prediction_arrays(
                    rows=validation,
                    buy_model=buy_model,
                    sell_model=sell_model,
                    feature_columns=(
                        feature_columns
                    ),
                )
            )

            buy_correlation = (
                calculate_correlation(
                    validation[
                        "buy_trade_r"
                    ].to_numpy(),
                    validation_predictions[
                        "buy"
                    ],
                )
            )

            sell_correlation = (
                calculate_correlation(
                    validation[
                        "sell_trade_r"
                    ].to_numpy(),
                    validation_predictions[
                        "sell"
                    ],
                )
            )

            print(
                f"BUY検証相関: "
                f"{buy_correlation:.3f}"
            )

            print(
                f"SELL検証相関: "
                f"{sell_correlation:.3f}"
            )

            validation_threshold_map = (
                create_rolling_percentile_thresholds(
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
                    top_percentages=(
                        TOP_PERCENTAGES
                    ),
                )
            )

            (
                best_condition,
                grid_results,
            ) = select_validation_condition(
                fold_number=fold_number,
                validation=validation,
                predictions=(
                    validation_predictions
                ),
                threshold_map=(
                    validation_threshold_map
                ),
            )

            all_grid_results.append(
                grid_results
            )

            common_fold_data = {
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
                "buy_validation_correlation": (
                    buy_correlation
                ),
                "sell_validation_correlation": (
                    sell_correlation
                ),
            }

            if best_condition is None:
                print(
                    "検証期間で最低30回、"
                    "平均Rプラス・PF1超の"
                    "順位条件なし。"
                )

                fold_results.append(
                    {
                        **common_fold_data,
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
                        "validation_pf": 0.0,
                        "test_trades": 0,
                        "test_total_r": 0.0,
                        "test_average_r": 0.0,
                        "test_pf": 0.0,
                        "test_max_dd_r": 0.0,
                        "test_buy_count": 0,
                        "test_sell_count": 0,
                    }
                )

                continue

            top_percentage = float(
                best_condition[
                    "top_percentage"
                ]
            )

            direction_margin = float(
                best_condition[
                    "direction_margin"
                ]
            )

            print(
                "\n採用条件:"
            )

            print(
                f"予測順位: "
                f"上位"
                f"{top_percentage * 100:.1f}%"
            )

            print(
                f"方向差: "
                f"{direction_margin:.3f}R以上"
            )

            print(
                f"検証取引数: "
                f"{int(best_condition['trades']):,}回"
            )

            print(
                f"検証合計R: "
                f"{best_condition['total_r']:.2f}R"
            )

            print(
                f"検証平均R: "
                f"{best_condition['average_r']:.4f}R"
            )

            print(
                f"検証PF: "
                f"{best_condition['profit_factor']:.2f}"
            )

            test_predictions = (
                create_prediction_arrays(
                    rows=test,
                    buy_model=buy_model,
                    sell_model=sell_model,
                    feature_columns=(
                        feature_columns
                    ),
                )
            )

            # テスト開始時には検証期間が過去として利用可能。
            # その後はテスト内でも過去の予測だけを履歴へ追加する。
            test_threshold_map = (
                create_rolling_percentile_thresholds(
                    history_scores=(
                        validation_predictions[
                            "score"
                        ]
                    ),
                    current_scores=(
                        test_predictions[
                            "score"
                        ]
                    ),
                    top_percentages=[
                        top_percentage
                    ],
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
                            top_percentage
                        ]
                    ),
                    direction_margin=(
                        direction_margin
                    ),
                )
            )

            test_metrics = (
                calculate_trade_metrics(
                    test_trades
                )
            )

            print_trade_metrics(
                "未使用テスト結果:",
                test_metrics,
            )

            if not test_trades.empty:
                test_trades["fold"] = (
                    fold_number
                )
                test_trades[
                    "top_percentage"
                ] = top_percentage
                test_trades[
                    "direction_margin"
                ] = direction_margin

                all_test_trades.append(
                    test_trades
                )

            fold_results.append(
                {
                    **common_fold_data,
                    "status": "EVALUATED",
                    "top_percentage": (
                        top_percentage
                    ),
                    "direction_margin": (
                        direction_margin
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
                    "validation_pf": float(
                        best_condition[
                            "profit_factor"
                        ]
                    ),
                    "test_trades": (
                        test_metrics["trades"]
                    ),
                    "test_total_r": (
                        test_metrics["total_r"]
                    ),
                    "test_average_r": (
                        test_metrics[
                            "average_r"
                        ]
                    ),
                    "test_pf": (
                        test_metrics[
                            "profit_factor"
                        ]
                    ),
                    "test_max_dd_r": (
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

        fold_frame = pd.DataFrame(
            fold_results
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

        if all_grid_results:
            pd.concat(
                all_grid_results,
                ignore_index=True,
            ).to_csv(
                GRID_RESULTS_PATH,
                index=False,
                encoding="utf-8-sig",
            )

        print(
            "\n===================="
        )
        print(
            "=== ウォークフォワード集計 ==="
        )

        if all_test_trades:
            combined_trades = pd.concat(
                all_test_trades,
                ignore_index=True,
            ).sort_values(
                "time"
            ).reset_index(drop=True)

            combined_metrics = (
                calculate_trade_metrics(
                    combined_trades
                )
            )

            combined_trades.to_csv(
                TEST_TRADES_PATH,
                index=False,
                encoding="utf-8-sig",
            )

            print_trade_metrics(
                "全未使用テスト合計:",
                combined_metrics,
            )

            evaluated = fold_frame[
                fold_frame["status"]
                == "EVALUATED"
            ]

            positive_folds = int(
                (
                    evaluated[
                        "test_total_r"
                    ] > 0
                ).sum()
            )

            print(
                f"評価できたFold: "
                f"{len(evaluated)} / "
                f"{NUMBER_OF_FOLDS}"
            )

            print(
                f"プラスFold: "
                f"{positive_folds} / "
                f"{len(evaluated)}"
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
                        monthly["total_r"] > 0
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
                    f"{monthly['total_r'].mean():.2f}R"
                )

                print(
                    f"プラス月: "
                    f"{positive_months} / "
                    f"{len(monthly)}"
                )

                print(
                    f"最高月: "
                    f"{monthly['total_r'].max():.2f}R"
                )

                print(
                    f"最低月: "
                    f"{monthly['total_r'].min():.2f}R"
                )

        else:
            print(
                "未使用テストで取引は"
                "発生しませんでした。"
            )

        print(
            "\nFold結果:",
            FOLD_RESULTS_PATH.resolve(),
        )

        print(
            "条件比較:",
            GRID_RESULTS_PATH.resolve(),
        )

        print(
            "未使用テスト取引:",
            TEST_TRADES_PATH.resolve(),
        )

        print(
            "月別集計:",
            MONTHLY_RESULTS_PATH.resolve(),
        )

    except Exception as error:
        print(
            "\n期待R順位ウォークフォワード中に"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()
