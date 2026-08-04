from __future__ import annotations

from pathlib import Path
import gc
import math

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DATASET_PATH = Path(
    "data/market_context_expected_r_enriched.csv"
)

FOLD_RESULTS_PATH = Path(
    "data/athena_trend_timing_folds.csv"
)

AI_GRID_PATH = Path(
    "data/athena_trend_timing_grid.csv"
)

BASELINE_TRADES_PATH = Path(
    "data/athena_trend_baseline_trades.csv"
)

AI_TRADES_PATH = Path(
    "data/athena_trend_timing_ai_trades.csv"
)

MONTHLY_RESULTS_PATH = Path(
    "data/athena_trend_timing_monthly.csv"
)

INITIAL_TRAIN_RATIO = 0.40
SEGMENT_RATIO = 0.075
NUMBER_OF_FOLDS = 4

# 最大48本先の結果を教師として使っている
GAP_ROWS = 48

EARLY_STOP_RATIO = 0.10

ROLLING_SCORE_WINDOW = 5000
MIN_SCORE_HISTORY = 500

MIN_VALIDATION_TRADES = 30

# 方向候補の中からAI予測上位何％を使うか
TOP_PERCENTAGES = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
]

# TP +1.5R、SL -1Rを基準にした
# おおよその損益分岐勝率
MIN_WIN_PROBABILITY = 0.40

# 長期方向は固定し、検証で変更しない
MIN_LONG_ALIGNMENT = 5

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

REGIME_COLUMNS = [
    "long_alignment_score",
    "long_d1_return_1",
    "long_d1_return_20",
    "long_d1_return_252",
    "strength_usdjpy_pressure_1d",
    "strength_usdjpy_pressure_1w",
]


def inspect_columns() -> tuple[
    list[str],
    list[str],
]:
    """AI特徴量と読み込み列を決める。"""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "データセットが見つかりません: "
            f"{DATASET_PATH.resolve()}"
        )

    header = pd.read_csv(
        DATASET_PATH,
        nrows=0,
    ).columns.tolist()

    header_set = set(header)

    required = (
        RESULT_COLUMNS
        | set(REGIME_COLUMNS)
    )

    missing = required - header_set

    if missing:
        raise RuntimeError(
            "必要な列がありません: "
            f"{sorted(missing)}"
        )

    # AIはM5・M15・H1・H4などの
    # 従来155特徴量だけを使う。
    feature_columns = []

    for column in header:
        if column in RESULT_COLUMNS:
            continue

        if column in RAW_PRICE_COLUMNS:
            continue

        if column.startswith("pattern_"):
            continue

        if column.startswith("long_"):
            continue

        if column.startswith("strength_"):
            continue

        if column.startswith("rel_"):
            continue

        feature_columns.append(column)

    use_columns = list(
        dict.fromkeys(
            list(RESULT_COLUMNS)
            + REGIME_COLUMNS
            + feature_columns
        )
    )

    return use_columns, feature_columns


def load_dataset(
    use_columns: list[str],
    feature_columns: list[str],
) -> pd.DataFrame:
    """必要列だけ読み込む。"""
    numeric_columns = (
        feature_columns
        + REGIME_COLUMNS
        + [
            "buy_trade_r",
            "sell_trade_r",
            "buy_new_holding_bars",
            "sell_new_holding_bars",
        ]
    )

    dtype_map = {
        column: "float32"
        for column in numeric_columns
        if column != "long_alignment_score"
    }

    dtype_map[
        "long_alignment_score"
    ] = "int8"

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

    dataset = (
        dataset.dropna(
            subset=[
                "time",
                *numeric_columns,
            ]
        )
        .sort_values("time")
        .drop_duplicates(
            subset=["time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    dataset["regime_direction"] = (
        create_regime_direction(
            dataset
        )
    )

    return dataset


def create_regime_direction(
    rows: pd.DataFrame,
) -> pd.Series:
    """
    長期足と通貨圧力だけで
    BUY・SELL・NONEを固定する。
    """
    buy_condition = (
        (
            rows[
                "long_alignment_score"
            ]
            >= MIN_LONG_ALIGNMENT
        )
        & (
            rows[
                "long_d1_return_252"
            ] > 0
        )
        & (
            rows[
                "long_d1_return_20"
            ] > 0
        )
        & (
            rows[
                "long_d1_return_1"
            ] > 0
        )
        & (
            rows[
                "strength_usdjpy_pressure_1d"
            ] > 0
        )
        & (
            rows[
                "strength_usdjpy_pressure_1w"
            ] > 0
        )
    )

    sell_condition = (
        (
            rows[
                "long_alignment_score"
            ]
            <= -MIN_LONG_ALIGNMENT
        )
        & (
            rows[
                "long_d1_return_252"
            ] < 0
        )
        & (
            rows[
                "long_d1_return_20"
            ] < 0
        )
        & (
            rows[
                "long_d1_return_1"
            ] < 0
        )
        & (
            rows[
                "strength_usdjpy_pressure_1d"
            ] < 0
        )
        & (
            rows[
                "strength_usdjpy_pressure_1w"
            ] < 0
        )
    )

    return pd.Series(
        np.select(
            [
                buy_condition,
                sell_condition,
            ],
            [
                "BUY",
                "SELL",
            ],
            default="NONE",
        ),
        index=rows.index,
        dtype="string",
    )


def split_training_for_early_stop(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """学習期間内だけで本学習と監視期間を分ける。"""
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


def create_classifier(
    random_state: int,
) -> lgb.LGBMClassifier:
    """勝ち・負けを判定する分類モデル。"""
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=200,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def prepare_direction_training(
    rows: pd.DataFrame,
    direction: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """長期方向で絞らず、全M5行を指定方向の教師データにする。"""
    selected = rows.copy()

    if direction == "BUY":
        target = (
            selected[
                "buy_trade_r"
            ] > 0
        ).astype("int8")
    else:
        target = (
            selected[
                "sell_trade_r"
            ] > 0
        ).astype("int8")

    return selected, target.to_numpy()


def validate_training_sample(
    rows: pd.DataFrame,
    target: np.ndarray,
    direction: str,
    split_name: str,
) -> None:
    """分類学習に必要な件数と両クラスを確認する。"""
    if len(rows) < 1000:
        raise RuntimeError(
            f"{direction}の{split_name}行が"
            f"少なすぎます: {len(rows):,}件"
        )

    unique_classes = np.unique(
        target
    )

    if len(unique_classes) < 2:
        raise RuntimeError(
            f"{direction}の{split_name}に"
            "勝ち負け両方がありません"
        )


def train_direction_classifier(
    fit_train: pd.DataFrame,
    early_stop: pd.DataFrame,
    feature_columns: list[str],
    direction: str,
    random_state: int,
) -> tuple[
    lgb.LGBMClassifier,
    dict[str, float | int],
]:
    """全M5行で方向別タイミングAIを学習する。"""
    (
        fit_rows,
        fit_target,
    ) = prepare_direction_training(
        fit_train,
        direction,
    )

    (
        stop_rows,
        stop_target,
    ) = prepare_direction_training(
        early_stop,
        direction,
    )

    validate_training_sample(
        fit_rows,
        fit_target,
        direction,
        "本学習",
    )

    validate_training_sample(
        stop_rows,
        stop_target,
        direction,
        "早期終了監視",
    )

    model = create_classifier(
        random_state=random_state
    )

    model.fit(
        fit_rows[feature_columns],
        fit_target,
        eval_X=(
            stop_rows[
                feature_columns
            ]
        ),
        eval_y=stop_target,
        eval_metric="auc",
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

    stop_probability = (
        model.predict_proba(
            stop_rows[
                feature_columns
            ]
        )[:, 1]
    )

    stop_auc = safe_auc(
        stop_target,
        stop_probability,
    )

    stats = {
        "fit_rows": int(
            len(fit_rows)
        ),
        "fit_positive_rate": float(
            fit_target.mean()
        ),
        "stop_rows": int(
            len(stop_rows)
        ),
        "stop_positive_rate": float(
            stop_target.mean()
        ),
        "stop_auc": float(
            stop_auc
        ),
    }

    return model, stats


def safe_auc(
    target: np.ndarray,
    probability: np.ndarray,
) -> float:
    """両クラスがある場合だけAUCを返す。"""
    if len(
        np.unique(target)
    ) < 2:
        return 0.5

    return float(
        roc_auc_score(
            target,
            probability,
        )
    )


def create_candidate_predictions(
    rows: pd.DataFrame,
    buy_model: lgb.LGBMClassifier,
    sell_model: lgb.LGBMClassifier,
    feature_columns: list[str],
) -> pd.DataFrame:
    """方向フィルターを通過した行だけ予測する。"""
    candidates = rows[
        rows["regime_direction"]
        != "NONE"
    ].copy()

    if candidates.empty:
        return candidates

    candidates[
        "win_probability"
    ] = np.nan

    buy_mask = (
        candidates[
            "regime_direction"
        ] == "BUY"
    )

    sell_mask = (
        candidates[
            "regime_direction"
        ] == "SELL"
    )

    if buy_mask.any():
        candidates.loc[
            buy_mask,
            "win_probability",
        ] = buy_model.predict_proba(
            candidates.loc[
                buy_mask,
                feature_columns,
            ]
        )[:, 1]

    if sell_mask.any():
        candidates.loc[
            sell_mask,
            "win_probability",
        ] = sell_model.predict_proba(
            candidates.loc[
                sell_mask,
                feature_columns,
            ]
        )[:, 1]

    candidates = (
        candidates.dropna(
            subset=["win_probability"]
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    return candidates


def candidate_auc(
    candidates: pd.DataFrame,
) -> dict[str, float]:
    """候補期間でBUY・SELL別AUCを計算する。"""
    results = {}

    for direction in [
        "BUY",
        "SELL",
    ]:
        selected = candidates[
            candidates[
                "regime_direction"
            ] == direction
        ]

        if selected.empty:
            results[
                direction
            ] = 0.5
            continue

        if direction == "BUY":
            target = (
                selected[
                    "buy_trade_r"
                ] > 0
            ).astype("int8")
        else:
            target = (
                selected[
                    "sell_trade_r"
                ] > 0
            ).astype("int8")

        results[
            direction
        ] = safe_auc(
            target.to_numpy(),
            selected[
                "win_probability"
            ].to_numpy(),
        )

    return results


def create_rolling_thresholds(
    history_probabilities: np.ndarray,
    current_probabilities: np.ndarray,
) -> dict[float, np.ndarray]:
    """
    現在より前の候補予測だけを使い、
    上位X％閾値を作る。
    """
    history = pd.Series(
        np.asarray(
            history_probabilities,
            dtype=float,
        )
    )

    current = pd.Series(
        np.asarray(
            current_probabilities,
            dtype=float,
        )
    )

    combined = pd.concat(
        [
            history,
            current,
        ],
        ignore_index=True,
    )

    prior = combined.shift(1)

    current_start = len(history)

    thresholds = {}

    for top_percentage in (
        TOP_PERCENTAGES
    ):
        thresholds[
            top_percentage
        ] = (
            prior.rolling(
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


def simulate_candidate_trades(
    candidates: pd.DataFrame,
    rolling_thresholds: np.ndarray | None,
    use_ai_filter: bool,
) -> pd.DataFrame:
    """方向候補を非重複取引として再現する。"""
    trades = []
    next_allowed_time = None

    if candidates.empty:
        return pd.DataFrame()

    for index, row in candidates.iterrows():
        signal_time = pd.Timestamp(
            row["time"]
        )

        if (
            next_allowed_time is not None
            and signal_time
            < next_allowed_time
        ):
            continue

        probability = float(
            row["win_probability"]
        )

        if use_ai_filter:
            threshold = float(
                rolling_thresholds[
                    index
                ]
            )

            if not np.isfinite(
                threshold
            ):
                continue

            if (
                probability
                < MIN_WIN_PROBABILITY
            ):
                continue

            if probability < threshold:
                continue

        else:
            threshold = np.nan

        direction = str(
            row["regime_direction"]
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
                "win_probability": (
                    probability
                ),
                "rolling_threshold": (
                    threshold
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

        # 終了足の次のM5から
        # 新しい取引を許可する。
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
    """Rベースの成績を集計する。"""
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
            negative["actual_r"].sum()
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
    }


def choose_ai_condition(
    fold_number: int,
    validation_candidates: pd.DataFrame,
    threshold_map: dict[
        float,
        np.ndarray,
    ],
    baseline_metrics: dict[
        str,
        float | int,
    ],
) -> tuple[
    pd.Series | None,
    pd.DataFrame,
]:
    """検証期間だけでAIの絞り率を決める。"""
    rows = []

    for top_percentage in (
        TOP_PERCENTAGES
    ):
        trades = simulate_candidate_trades(
            candidates=(
                validation_candidates
            ),
            rolling_thresholds=(
                threshold_map[
                    top_percentage
                ]
            ),
            use_ai_filter=True,
        )

        metrics = calculate_metrics(
            trades
        )

        rows.append(
            {
                "fold": fold_number,
                "top_percentage": (
                    top_percentage
                ),
                "minimum_probability": (
                    MIN_WIN_PROBABILITY
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
        & (
            grid["average_r"]
            > float(
                baseline_metrics[
                    "average_r"
                ]
            )
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


def print_metrics(
    title: str,
    metrics: dict[str, float | int],
) -> None:
    """成績を見やすく表示する。"""
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
        f"{metrics['total_r']:+.2f}R"
    )

    print(
        f"平均R: "
        f"{metrics['average_r']:+.4f}R"
    )

    print(
        "平均Rの95％下限: "
        f"{metrics['average_r_lcb95']:+.4f}R"
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
    strategy_name: str,
) -> pd.DataFrame:
    """月別集計を作る。"""
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

    monthly.insert(
        0,
        "strategy",
        strategy_name,
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
            feature_columns,
        ) = inspect_columns()

        dataset = load_dataset(
            use_columns=use_columns,
            feature_columns=(
                feature_columns
            ),
        )

        print(
            "=== Athena 長期方向＋"
            "M5タイミングAI "
            "ウォークフォワード v1 ==="
        )

        print(
            f"全データ: "
            f"{len(dataset):,}件"
        )

        print(
            f"M5系AI特徴量: "
            f"{len(feature_columns):,}個"
        )

        print(
            "期間:",
            dataset.iloc[0]["time"],
            "～",
            dataset.iloc[-1]["time"],
        )

        print(
            "\n固定方向条件:"
        )

        print(
            "BUY = 長期一致+5以上、"
            "年間・約1か月・日間上昇、"
            "1日・1週間圧力上向き"
        )

        print(
            "SELL = BUY条件の完全な逆"
        )

        print(
            f"AI最低勝率予測: "
            f"{MIN_WIN_PROBABILITY:.0%}"
        )

        total_regime_counts = (
            dataset[
                "regime_direction"
            ].value_counts()
        )

        print(
            "\n全期間の方向候補:"
        )

        print(
            total_regime_counts.to_string()
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
        all_grid_frames = []
        all_baseline_trades = []
        all_ai_trades = []

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

            print(
                "\nBUYタイミングAIを学習中..."
            )

            (
                buy_model,
                buy_stats,
            ) = train_direction_classifier(
                fit_train=fit_train,
                early_stop=early_stop,
                feature_columns=(
                    feature_columns
                ),
                direction="BUY",
                random_state=(
                    1000 + fold_number
                ),
            )

            print(
                f"BUY本学習候補: "
                f"{buy_stats['fit_rows']:,}件 / "
                f"勝率"
                f"{buy_stats['fit_positive_rate']:.2%}"
            )

            print(
                f"BUY監視候補: "
                f"{buy_stats['stop_rows']:,}件 / "
                f"AUC "
                f"{buy_stats['stop_auc']:.3f}"
            )

            print(
                "\nSELLタイミングAIを学習中..."
            )

            (
                sell_model,
                sell_stats,
            ) = train_direction_classifier(
                fit_train=fit_train,
                early_stop=early_stop,
                feature_columns=(
                    feature_columns
                ),
                direction="SELL",
                random_state=(
                    2000 + fold_number
                ),
            )

            print(
                f"SELL本学習候補: "
                f"{sell_stats['fit_rows']:,}件 / "
                f"勝率"
                f"{sell_stats['fit_positive_rate']:.2%}"
            )

            print(
                f"SELL監視候補: "
                f"{sell_stats['stop_rows']:,}件 / "
                f"AUC "
                f"{sell_stats['stop_auc']:.3f}"
            )

            early_candidates = (
                create_candidate_predictions(
                    rows=early_stop,
                    buy_model=buy_model,
                    sell_model=sell_model,
                    feature_columns=(
                        feature_columns
                    ),
                )
            )

            validation_candidates = (
                create_candidate_predictions(
                    rows=validation,
                    buy_model=buy_model,
                    sell_model=sell_model,
                    feature_columns=(
                        feature_columns
                    ),
                )
            )

            validation_auc = (
                candidate_auc(
                    validation_candidates
                )
            )

            print(
                "\n検証候補:"
            )

            print(
                f"BUY "
                f"{int((validation_candidates['regime_direction'] == 'BUY').sum()):,}件 / "
                f"AUC {validation_auc['BUY']:.3f}"
            )

            print(
                f"SELL "
                f"{int((validation_candidates['regime_direction'] == 'SELL').sum()):,}件 / "
                f"AUC {validation_auc['SELL']:.3f}"
            )

            validation_baseline = (
                simulate_candidate_trades(
                    candidates=(
                        validation_candidates
                    ),
                    rolling_thresholds=None,
                    use_ai_filter=False,
                )
            )

            validation_baseline_metrics = (
                calculate_metrics(
                    validation_baseline
                )
            )

            print_metrics(
                "検証・方向フィルターのみ:",
                validation_baseline_metrics,
            )

            validation_threshold_map = (
                create_rolling_thresholds(
                    history_probabilities=(
                        early_candidates[
                            "win_probability"
                        ].to_numpy()
                    ),
                    current_probabilities=(
                        validation_candidates[
                            "win_probability"
                        ].to_numpy()
                    ),
                )
            )

            (
                best_ai_condition,
                ai_grid,
            ) = choose_ai_condition(
                fold_number=fold_number,
                validation_candidates=(
                    validation_candidates
                ),
                threshold_map=(
                    validation_threshold_map
                ),
                baseline_metrics=(
                    validation_baseline_metrics
                ),
            )

            all_grid_frames.append(
                ai_grid
            )

            if best_ai_condition is None:
                print(
                    "\nAIタイミング条件:"
                )

                print(
                    "方向フィルター単体より改善し、"
                    "最低30回・平均Rプラス・"
                    "PF1超を満たす条件なし"
                )

            else:
                print(
                    "\nAIタイミング条件:"
                )

                print(
                    f"予測上位"
                    f"{float(best_ai_condition['top_percentage']) * 100:.0f}%"
                )

                print(
                    f"検証取引: "
                    f"{int(best_ai_condition['trades']):,}回"
                )

                print(
                    f"検証合計: "
                    f"{float(best_ai_condition['total_r']):+.2f}R"
                )

                print(
                    f"検証平均: "
                    f"{float(best_ai_condition['average_r']):+.4f}R"
                )

                print(
                    f"検証PF: "
                    f"{float(best_ai_condition['profit_factor']):.2f}"
                )

            test_candidates = (
                create_candidate_predictions(
                    rows=test,
                    buy_model=buy_model,
                    sell_model=sell_model,
                    feature_columns=(
                        feature_columns
                    ),
                )
            )

            test_auc = candidate_auc(
                test_candidates
            )

            print(
                "\n未使用テスト候補:"
            )

            print(
                f"BUY "
                f"{int((test_candidates['regime_direction'] == 'BUY').sum()):,}件 / "
                f"AUC {test_auc['BUY']:.3f}"
            )

            print(
                f"SELL "
                f"{int((test_candidates['regime_direction'] == 'SELL').sum()):,}件 / "
                f"AUC {test_auc['SELL']:.3f}"
            )

            test_baseline = (
                simulate_candidate_trades(
                    candidates=(
                        test_candidates
                    ),
                    rolling_thresholds=None,
                    use_ai_filter=False,
                )
            )

            test_baseline_metrics = (
                calculate_metrics(
                    test_baseline
                )
            )

            print_metrics(
                "未使用テスト・方向フィルターのみ:",
                test_baseline_metrics,
            )

            if not test_baseline.empty:
                test_baseline[
                    "fold"
                ] = fold_number

                test_baseline[
                    "strategy"
                ] = "DIRECTION_ONLY"

                all_baseline_trades.append(
                    test_baseline
                )

            ai_status = "NO_CONDITION"
            test_ai_metrics = (
                calculate_metrics(
                    pd.DataFrame()
                )
            )

            selected_top_percentage = (
                np.nan
            )

            if best_ai_condition is not None:
                selected_top_percentage = float(
                    best_ai_condition[
                        "top_percentage"
                    ]
                )

                test_threshold_map = (
                    create_rolling_thresholds(
                        history_probabilities=(
                            validation_candidates[
                                "win_probability"
                            ].to_numpy()
                        ),
                        current_probabilities=(
                            test_candidates[
                                "win_probability"
                            ].to_numpy()
                        ),
                    )
                )

                test_ai = (
                    simulate_candidate_trades(
                        candidates=(
                            test_candidates
                        ),
                        rolling_thresholds=(
                            test_threshold_map[
                                selected_top_percentage
                            ]
                        ),
                        use_ai_filter=True,
                    )
                )

                test_ai_metrics = (
                    calculate_metrics(
                        test_ai
                    )
                )

                print_metrics(
                    "未使用テスト・方向＋AI:",
                    test_ai_metrics,
                )

                ai_status = "EVALUATED"

                if not test_ai.empty:
                    test_ai["fold"] = (
                        fold_number
                    )

                    test_ai[
                        "strategy"
                    ] = "DIRECTION_PLUS_AI"

                    test_ai[
                        "top_percentage"
                    ] = (
                        selected_top_percentage
                    )

                    all_ai_trades.append(
                        test_ai
                    )

            fold_rows.append(
                {
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
                    "buy_stop_auc": (
                        buy_stats[
                            "stop_auc"
                        ]
                    ),
                    "sell_stop_auc": (
                        sell_stats[
                            "stop_auc"
                        ]
                    ),
                    "buy_validation_auc": (
                        validation_auc[
                            "BUY"
                        ]
                    ),
                    "sell_validation_auc": (
                        validation_auc[
                            "SELL"
                        ]
                    ),
                    "buy_test_auc": (
                        test_auc[
                            "BUY"
                        ]
                    ),
                    "sell_test_auc": (
                        test_auc[
                            "SELL"
                        ]
                    ),
                    "validation_baseline_trades": (
                        validation_baseline_metrics[
                            "trades"
                        ]
                    ),
                    "validation_baseline_total_r": (
                        validation_baseline_metrics[
                            "total_r"
                        ]
                    ),
                    "validation_baseline_average_r": (
                        validation_baseline_metrics[
                            "average_r"
                        ]
                    ),
                    "validation_baseline_pf": (
                        validation_baseline_metrics[
                            "profit_factor"
                        ]
                    ),
                    "ai_status": ai_status,
                    "selected_top_percentage": (
                        selected_top_percentage
                    ),
                    "validation_ai_trades": (
                        0
                        if best_ai_condition
                        is None
                        else int(
                            best_ai_condition[
                                "trades"
                            ]
                        )
                    ),
                    "validation_ai_total_r": (
                        0.0
                        if best_ai_condition
                        is None
                        else float(
                            best_ai_condition[
                                "total_r"
                            ]
                        )
                    ),
                    "validation_ai_average_r": (
                        0.0
                        if best_ai_condition
                        is None
                        else float(
                            best_ai_condition[
                                "average_r"
                            ]
                        )
                    ),
                    "validation_ai_pf": (
                        0.0
                        if best_ai_condition
                        is None
                        else float(
                            best_ai_condition[
                                "profit_factor"
                            ]
                        )
                    ),
                    "test_baseline_trades": (
                        test_baseline_metrics[
                            "trades"
                        ]
                    ),
                    "test_baseline_total_r": (
                        test_baseline_metrics[
                            "total_r"
                        ]
                    ),
                    "test_baseline_average_r": (
                        test_baseline_metrics[
                            "average_r"
                        ]
                    ),
                    "test_baseline_pf": (
                        test_baseline_metrics[
                            "profit_factor"
                        ]
                    ),
                    "test_baseline_max_dd_r": (
                        test_baseline_metrics[
                            "max_drawdown_r"
                        ]
                    ),
                    "test_baseline_buy_count": (
                        test_baseline_metrics[
                            "buy_count"
                        ]
                    ),
                    "test_baseline_sell_count": (
                        test_baseline_metrics[
                            "sell_count"
                        ]
                    ),
                    "test_ai_trades": (
                        test_ai_metrics[
                            "trades"
                        ]
                    ),
                    "test_ai_total_r": (
                        test_ai_metrics[
                            "total_r"
                        ]
                    ),
                    "test_ai_average_r": (
                        test_ai_metrics[
                            "average_r"
                        ]
                    ),
                    "test_ai_pf": (
                        test_ai_metrics[
                            "profit_factor"
                        ]
                    ),
                    "test_ai_max_dd_r": (
                        test_ai_metrics[
                            "max_drawdown_r"
                        ]
                    ),
                    "test_ai_buy_count": (
                        test_ai_metrics[
                            "buy_count"
                        ]
                    ),
                    "test_ai_sell_count": (
                        test_ai_metrics[
                            "sell_count"
                        ]
                    ),
                }
            )

            del buy_model
            del sell_model
            del early_candidates
            del validation_candidates
            del test_candidates
            gc.collect()

        fold_frame = pd.DataFrame(
            fold_rows
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

        if all_grid_frames:
            pd.concat(
                all_grid_frames,
                ignore_index=True,
            ).to_csv(
                AI_GRID_PATH,
                index=False,
                encoding="utf-8-sig",
            )

        print(
            "\n=============================="
        )

        print(
            "=== 正式ウォークフォワード集計 ==="
        )

        monthly_frames = []

        if all_baseline_trades:
            combined_baseline = (
                pd.concat(
                    all_baseline_trades,
                    ignore_index=True,
                )
                .sort_values("time")
                .reset_index(drop=True)
            )

            combined_baseline.to_csv(
                BASELINE_TRADES_PATH,
                index=False,
                encoding="utf-8-sig",
            )

            baseline_metrics = (
                calculate_metrics(
                    combined_baseline
                )
            )

            print_metrics(
                "方向フィルターのみ・"
                "全未使用テスト:",
                baseline_metrics,
            )

            baseline_positive_folds = int(
                (
                    fold_frame[
                        "test_baseline_total_r"
                    ] > 0
                ).sum()
            )

            print(
                f"プラスFold: "
                f"{baseline_positive_folds} / "
                f"{len(fold_frame)}"
            )

            monthly_frames.append(
                create_monthly_results(
                    combined_baseline,
                    "DIRECTION_ONLY",
                )
            )

        else:
            print(
                "\n方向フィルターのみでは"
                "テスト取引がありませんでした。"
            )

        evaluated_ai_folds = fold_frame[
            fold_frame["ai_status"]
            == "EVALUATED"
        ]

        if all_ai_trades:
            combined_ai = (
                pd.concat(
                    all_ai_trades,
                    ignore_index=True,
                )
                .sort_values("time")
                .reset_index(drop=True)
            )

            combined_ai.to_csv(
                AI_TRADES_PATH,
                index=False,
                encoding="utf-8-sig",
            )

            ai_metrics = (
                calculate_metrics(
                    combined_ai
                )
            )

            print_metrics(
                "方向＋AI・全未使用テスト:",
                ai_metrics,
            )

            ai_positive_folds = int(
                (
                    evaluated_ai_folds[
                        "test_ai_total_r"
                    ] > 0
                ).sum()
            )

            print(
                f"AI評価Fold: "
                f"{len(evaluated_ai_folds)} / "
                f"{NUMBER_OF_FOLDS}"
            )

            print(
                f"AIプラスFold: "
                f"{ai_positive_folds} / "
                f"{len(evaluated_ai_folds)}"
            )

            monthly_frames.append(
                create_monthly_results(
                    combined_ai,
                    "DIRECTION_PLUS_AI",
                )
            )

        else:
            print(
                "\n方向＋AIで評価可能な"
                "未使用テスト取引は"
                "ありませんでした。"
            )

        if monthly_frames:
            monthly = pd.concat(
                monthly_frames,
                ignore_index=True,
            )

            monthly.to_csv(
                MONTHLY_RESULTS_PATH,
                index=False,
                encoding="utf-8-sig",
            )

            print(
                "\n月別集計:"
            )

            for strategy_name in (
                monthly[
                    "strategy"
                ].unique()
            ):
                selected = monthly[
                    monthly["strategy"]
                    == strategy_name
                ]

                positive_months = int(
                    (
                        selected[
                            "total_r"
                        ] > 0
                    ).sum()
                )

                print(
                    f"{strategy_name}: "
                    f"{len(selected)}か月 / "
                    f"月平均"
                    f"{selected['total_r'].mean():+.2f}R / "
                    f"プラス月"
                    f"{positive_months}/"
                    f"{len(selected)}"
                )

        print(
            "\nFold結果:",
            FOLD_RESULTS_PATH.resolve(),
        )

        print(
            "AI条件比較:",
            AI_GRID_PATH.resolve(),
        )

        print(
            "方向のみ取引:",
            BASELINE_TRADES_PATH.resolve(),
        )

        print(
            "方向＋AI取引:",
            AI_TRADES_PATH.resolve(),
        )

        print(
            "月別集計:",
            MONTHLY_RESULTS_PATH.resolve(),
        )

    except Exception as error:
        print(
            "\n長期方向＋タイミングAIの"
            "処理中にエラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()
