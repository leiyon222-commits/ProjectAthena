from pathlib import Path
import math

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


DATASET_PATH = Path(
    "data/market_context_expected_r_labels.csv"
)

FOLD_RESULTS_PATH = Path(
    "data/athena_expected_r_walkforward_folds.csv"
)

GRID_RESULTS_PATH = Path(
    "data/athena_expected_r_walkforward_grid.csv"
)

TEST_TRADES_PATH = Path(
    "data/athena_expected_r_walkforward_trades.csv"
)

# 最初の40％を初回学習に使用し、
# その後は検証7.5％・テスト7.5％を4回繰り返す
INITIAL_TRAIN_RATIO = 0.40
SEGMENT_RATIO = 0.075
NUMBER_OF_FOLDS = 4

# 教師ラベルが48本先までを見るため、
# 各期間の境界を48行空ける
GAP_ROWS = 48

# 各学習期間の最後10％を早期終了監視に使う
EARLY_STOP_RATIO = 0.10

# 1つの検証期間で最低限必要な取引数
MIN_VALIDATION_TRADES = 30

# AIが予測した期待Rの最低値
EXPECTED_R_THRESHOLDS = [
    -0.10,
    -0.05,
    0.00,
    0.025,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.30,
]

# BUY期待RとSELL期待Rの最低差
DIRECTION_MARGINS = [
    0.00,
    0.025,
    0.05,
    0.075,
    0.10,
    0.15,
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
    """期待R教師データを読み込む。"""
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

    missing = (
        required_columns
        - set(dataset.columns)
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
    """未来結果と生の価格を除き、数値特徴量だけを使う。"""
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
    """期待Rを予測するLightGBM回帰モデルを作る。"""
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
    """学習期間内だけで本学習と早期終了監視に分ける。"""
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
    """BUYまたはSELLの期待Rモデルを学習する。"""
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


def calculate_prediction_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """回帰モデル自体の予測性能を集計する。"""
    rmse = math.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    if (
        np.std(actual) > 0
        and np.std(predicted) > 0
    ):
        correlation = float(
            np.corrcoef(
                actual,
                predicted,
            )[0, 1]
        )
    else:
        correlation = 0.0

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "correlation": correlation,
    }


def simulate_non_overlapping_trades(
    rows: pd.DataFrame,
    buy_predictions: np.ndarray,
    sell_predictions: np.ndarray,
    expected_r_threshold: float,
    direction_margin: float,
) -> pd.DataFrame:
    """
    全M5足を順番に確認し、
    予測期待Rが高い方向だけ取引する。

    決済までは次のシグナルを無視する。
    """
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

        buy_prediction = float(
            buy_predictions[index]
        )

        sell_prediction = float(
            sell_predictions[index]
        )

        if buy_prediction > sell_prediction:
            direction = "BUY"
            selected_prediction = (
                buy_prediction
            )
            other_prediction = (
                sell_prediction
            )
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

        elif sell_prediction > buy_prediction:
            direction = "SELL"
            selected_prediction = (
                sell_prediction
            )
            other_prediction = (
                buy_prediction
            )
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

        else:
            continue

        prediction_difference = (
            selected_prediction
            - other_prediction
        )

        if (
            selected_prediction
            < expected_r_threshold
        ):
            continue

        if (
            prediction_difference
            < direction_margin
        ):
            continue

        trades.append(
            {
                "time": signal_time,
                "direction": direction,
                "predicted_buy_r": (
                    buy_prediction
                ),
                "predicted_sell_r": (
                    sell_prediction
                ),
                "selected_predicted_r": (
                    selected_prediction
                ),
                "prediction_difference": (
                    prediction_difference
                ),
                "actual_r": actual_r,
                "holding_bars": holding_bars,
                "exit_reason": exit_reason,
            }
        )

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
    buy_model: lgb.LGBMRegressor,
    sell_model: lgb.LGBMRegressor,
    feature_columns: list[str],
) -> tuple[
    pd.Series | None,
    pd.DataFrame,
]:
    """検証期間だけで取引条件を決める。"""
    buy_predictions = buy_model.predict(
        validation[feature_columns]
    )

    sell_predictions = (
        sell_model.predict(
            validation[feature_columns]
        )
    )

    results: list[dict] = []

    for threshold in EXPECTED_R_THRESHOLDS:
        for margin in DIRECTION_MARGINS:
            trades = (
                simulate_non_overlapping_trades(
                    rows=validation,
                    buy_predictions=(
                        buy_predictions
                    ),
                    sell_predictions=(
                        sell_predictions
                    ),
                    expected_r_threshold=(
                        threshold
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
                    "threshold": threshold,
                    "margin": margin,
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
            "=== Athena 期待R AI "
            "ウォークフォワード v1 ==="
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
            f"初回学習: "
            f"{initial_train_count:,}件"
        )

        print(
            f"各検証・テスト: "
            f"約{segment_count:,}件"
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
                f"\n===================="
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

            validation_buy_predictions = (
                buy_model.predict(
                    validation[
                        feature_columns
                    ]
                )
            )

            validation_sell_predictions = (
                sell_model.predict(
                    validation[
                        feature_columns
                    ]
                )
            )

            buy_prediction_metrics = (
                calculate_prediction_metrics(
                    validation[
                        "buy_trade_r"
                    ].to_numpy(),
                    validation_buy_predictions,
                )
            )

            sell_prediction_metrics = (
                calculate_prediction_metrics(
                    validation[
                        "sell_trade_r"
                    ].to_numpy(),
                    validation_sell_predictions,
                )
            )

            print(
                "BUY検証相関: "
                f"{buy_prediction_metrics['correlation']:.3f}"
            )

            print(
                "SELL検証相関: "
                f"{sell_prediction_metrics['correlation']:.3f}"
            )

            (
                best_condition,
                grid_results,
            ) = select_validation_condition(
                fold_number=fold_number,
                validation=validation,
                buy_model=buy_model,
                sell_model=sell_model,
                feature_columns=(
                    feature_columns
                ),
            )

            all_grid_results.append(
                grid_results
            )

            if best_condition is None:
                print(
                    "検証期間で最低30回、"
                    "平均Rプラス・PF1超の"
                    "条件なし。"
                )

                fold_results.append(
                    {
                        "fold": (
                            fold_number
                        ),
                        "status": (
                            "NO_CONDITION"
                        ),
                        "train_start": (
                            train.iloc[0][
                                "time"
                            ]
                        ),
                        "train_end": (
                            train.iloc[-1][
                                "time"
                            ]
                        ),
                        "validation_start": (
                            validation.iloc[0][
                                "time"
                            ]
                        ),
                        "validation_end": (
                            validation.iloc[-1][
                                "time"
                            ]
                        ),
                        "test_start": (
                            test.iloc[0][
                                "time"
                            ]
                        ),
                        "test_end": (
                            test.iloc[-1][
                                "time"
                            ]
                        ),
                        "threshold": np.nan,
                        "margin": np.nan,
                        "validation_trades": 0,
                        "validation_average_r": 0.0,
                        "validation_pf": 0.0,
                        "test_trades": 0,
                        "test_average_r": 0.0,
                        "test_total_r": 0.0,
                        "test_pf": 0.0,
                        "test_max_dd_r": 0.0,
                        "test_buy_count": 0,
                        "test_sell_count": 0,
                        "buy_validation_correlation": (
                            buy_prediction_metrics[
                                "correlation"
                            ]
                        ),
                        "sell_validation_correlation": (
                            sell_prediction_metrics[
                                "correlation"
                            ]
                        ),
                    }
                )

                continue

            threshold = float(
                best_condition[
                    "threshold"
                ]
            )

            margin = float(
                best_condition["margin"]
            )

            print(
                "\n採用条件:"
            )

            print(
                f"予測期待R: "
                f"{threshold:+.3f}R以上"
            )

            print(
                f"方向差: "
                f"{margin:.3f}R以上"
            )

            print(
                f"検証取引数: "
                f"{int(best_condition['trades']):,}回"
            )

            print(
                f"検証平均R: "
                f"{best_condition['average_r']:.4f}R"
            )

            print(
                f"検証PF: "
                f"{best_condition['profit_factor']:.2f}"
            )

            test_buy_predictions = (
                buy_model.predict(
                    test[feature_columns]
                )
            )

            test_sell_predictions = (
                sell_model.predict(
                    test[feature_columns]
                )
            )

            test_trades = (
                simulate_non_overlapping_trades(
                    rows=test,
                    buy_predictions=(
                        test_buy_predictions
                    ),
                    sell_predictions=(
                        test_sell_predictions
                    ),
                    expected_r_threshold=(
                        threshold
                    ),
                    direction_margin=margin,
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
                test_trades[
                    "fold"
                ] = fold_number

                all_test_trades.append(
                    test_trades
                )

            fold_results.append(
                {
                    "fold": fold_number,
                    "status": "EVALUATED",
                    "train_start": (
                        train.iloc[0]["time"]
                    ),
                    "train_end": (
                        train.iloc[-1]["time"]
                    ),
                    "validation_start": (
                        validation.iloc[0][
                            "time"
                        ]
                    ),
                    "validation_end": (
                        validation.iloc[-1][
                            "time"
                        ]
                    ),
                    "test_start": (
                        test.iloc[0]["time"]
                    ),
                    "test_end": (
                        test.iloc[-1]["time"]
                    ),
                    "threshold": threshold,
                    "margin": margin,
                    "validation_trades": int(
                        best_condition[
                            "trades"
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
                    "test_average_r": (
                        test_metrics[
                            "average_r"
                        ]
                    ),
                    "test_total_r": (
                        test_metrics[
                            "total_r"
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
                    "buy_validation_correlation": (
                        buy_prediction_metrics[
                            "correlation"
                        ]
                    ),
                    "sell_validation_correlation": (
                        sell_prediction_metrics[
                            "correlation"
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

    except Exception as error:
        print(
            "\n期待Rウォークフォワード中に"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()
