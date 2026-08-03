from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight


DATASET_PATH = Path(
    "data/ai_trade_outcome_dataset.csv"
)

MODEL_PATH = Path(
    "data/athena_trade_outcome_ai.joblib"
)

PREDICTIONS_PATH = Path(
    "data/athena_trade_outcome_ai_predictions.csv"
)

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20

# 最大保有12本のため、期間境界を空ける
GAP_ROWS = 24

# AIが勝ちと判断する最低確率候補
PROBABILITY_THRESHOLDS = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]

# 検証期間で最低限必要な選別件数
MIN_VALIDATION_SELECTIONS = 200

FEATURE_COLUMNS = [
    "direction_value",
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
]

TARGET_COLUMN = "result"


def load_dataset() -> pd.DataFrame:
    """売買結果データを時刻順に読み込む。"""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "データセットが見つかりません: "
            f"{DATASET_PATH.resolve()}"
        )

    dataset = pd.read_csv(
        DATASET_PATH,
        parse_dates=["signal_time"],
    )

    dataset = dataset.sort_values(
        [
            "signal_time",
            "direction_value",
        ]
    ).reset_index(drop=True)

    required_columns = (
        ["signal_time"]
        + FEATURE_COLUMNS
        + [
            TARGET_COLUMN,
            "direction",
            "entry_price",
            "exit_price",
            "holding_bars",
            "exit_reason",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"必要な列がありません: {missing_columns}"
        )

    dataset = dataset[
        required_columns
    ].copy()

    dataset = dataset.replace(
        [
            float("inf"),
            float("-inf"),
        ],
        pd.NA,
    ).dropna()

    dataset[TARGET_COLUMN] = (
        dataset[TARGET_COLUMN].astype(int)
    )

    return dataset


def split_dataset(
    dataset: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """時系列順に60％・20％・20％へ分割する。"""
    unique_times = (
        dataset["signal_time"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    train_boundary_index = int(
        len(unique_times) * TRAIN_RATIO
    )

    validation_boundary_index = int(
        len(unique_times)
        * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    train_end_time = unique_times.iloc[
        train_boundary_index - 1
    ]

    validation_start_time = unique_times.iloc[
        min(
            train_boundary_index + GAP_ROWS,
            len(unique_times) - 1,
        )
    ]

    validation_end_time = unique_times.iloc[
        validation_boundary_index - 1
    ]

    test_start_time = unique_times.iloc[
        min(
            validation_boundary_index + GAP_ROWS,
            len(unique_times) - 1,
        )
    ]

    train = dataset[
        dataset["signal_time"]
        <= train_end_time
    ].copy()

    validation = dataset[
        (
            dataset["signal_time"]
            >= validation_start_time
        )
        & (
            dataset["signal_time"]
            <= validation_end_time
        )
    ].copy()

    test = dataset[
        dataset["signal_time"]
        >= test_start_time
    ].copy()

    if train.empty or validation.empty or test.empty:
        raise RuntimeError(
            "学習・検証・テスト分割に失敗しました"
        )

    return train, validation, test


def create_model() -> HistGradientBoostingClassifier:
    """勝敗予測モデルを作成する。"""
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        max_depth=6,
        min_samples_leaf=100,
        l2_regularization=2.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
    )


def get_win_probabilities(
    model: HistGradientBoostingClassifier,
    features: pd.DataFrame,
) -> np.ndarray:
    """勝ちクラスの予測確率を取得する。"""
    probabilities = model.predict_proba(
        features
    )

    win_class_index = list(
        model.classes_
    ).index(1)

    return probabilities[
        :,
        win_class_index,
    ]


def select_one_direction_per_time(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """
    同じ時刻のBUY・SELLから、
    勝率予測が高い方だけを選択する。
    """
    selected = rows[
        [
            "signal_time",
            "direction",
            "direction_value",
            TARGET_COLUMN,
            "entry_price",
            "exit_price",
            "holding_bars",
            "exit_reason",
        ]
    ].copy()

    selected["win_probability"] = (
        probabilities
    )

    selected = selected[
        selected["win_probability"]
        >= threshold
    ].copy()

    if selected.empty:
        return selected

    selected = (
        selected.sort_values(
            [
                "signal_time",
                "win_probability",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset=["signal_time"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return selected


def evaluate_selected(
    selected: pd.DataFrame,
) -> dict[str, float | int]:
    """AIが選択した候補の勝率などを集計する。"""
    if selected.empty:
        return {
            "selected_count": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "buy_count": 0,
            "sell_count": 0,
        }

    wins = int(
        selected[TARGET_COLUMN].sum()
    )

    losses = len(selected) - wins

    return {
        "selected_count": len(selected),
        "wins": wins,
        "losses": losses,
        "win_rate": (
            wins / len(selected) * 100
        ),
        "buy_count": int(
            (
                selected["direction"]
                == "BUY"
            ).sum()
        ),
        "sell_count": int(
            (
                selected["direction"]
                == "SELL"
            ).sum()
        ),
    }


def select_threshold(
    model: HistGradientBoostingClassifier,
    validation: pd.DataFrame,
) -> float:
    """
    検証期間だけを使って、
    勝ち確率の採用基準を決定する。
    """
    probabilities = get_win_probabilities(
        model,
        validation[FEATURE_COLUMNS],
    )

    results: list[dict] = []

    print(
        "\n=== 検証期間で確率基準を選択 ==="
    )

    for threshold in PROBABILITY_THRESHOLDS:
        selected = select_one_direction_per_time(
            rows=validation,
            probabilities=probabilities,
            threshold=threshold,
        )

        evaluation = evaluate_selected(
            selected
        )

        results.append(
            {
                "threshold": threshold,
                **evaluation,
            }
        )

        print(
            f"勝ち確率 {threshold:.0%}以上: "
            f"{evaluation['selected_count']:,}件 / "
            f"勝率 {evaluation['win_rate']:.2f}% / "
            f"BUY {evaluation['buy_count']:,} / "
            f"SELL {evaluation['sell_count']:,}"
        )

    eligible = [
        result
        for result in results
        if result["selected_count"]
        >= MIN_VALIDATION_SELECTIONS
    ]

    if not eligible:
        raise RuntimeError(
            "最低選別件数を満たす確率基準がありません"
        )

    # 勝率優先。同率なら候補数が多いものを採用
    best = max(
        eligible,
        key=lambda result: (
            result["win_rate"],
            result["selected_count"],
        ),
    )

    print(
        "\n採用する勝ち確率:",
        f"{best['threshold']:.0%}",
    )

    print(
        "検証期間の選別後勝率:",
        f"{best['win_rate']:.2f}%",
    )

    print(
        "検証期間の選別件数:",
        f"{best['selected_count']:,}件",
    )

    return float(
        best["threshold"]
    )


def evaluate_final_test(
    model: HistGradientBoostingClassifier,
    test: pd.DataFrame,
    threshold: float,
) -> None:
    """完全未使用期間を評価する。"""
    x_test = test[
        FEATURE_COLUMNS
    ]

    y_test = test[
        TARGET_COLUMN
    ]

    predictions = model.predict(
        x_test
    )

    probabilities = get_win_probabilities(
        model,
        x_test,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    average_precision = (
        average_precision_score(
            y_test,
            probabilities,
        )
    )

    baseline_accuracy = max(
        float((y_test == 0).mean()),
        float((y_test == 1).mean()),
    )

    selected = select_one_direction_per_time(
        rows=test,
        probabilities=probabilities,
        threshold=threshold,
    )

    selected_evaluation = evaluate_selected(
        selected
    )

    selected.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== 完全未使用期間の最終評価 ==="
    )

    print(
        f"候補行数: {len(test):,}件"
    )

    print(
        f"実際の勝率: "
        f"{y_test.mean():.2%}"
    )

    print(
        f"多数派だけを予測: "
        f"{baseline_accuracy:.2%}"
    )

    print(
        f"AI Accuracy: "
        f"{accuracy:.2%}"
    )

    print(
        f"AI Precision: "
        f"{precision:.2%}"
    )

    print(
        f"AI Recall: "
        f"{recall:.2%}"
    )

    print(
        f"AI ROC AUC: "
        f"{roc_auc:.3f}"
    )

    print(
        f"AI Average Precision: "
        f"{average_precision:.3f}"
    )

    print(
        "\n=== 混同行列 ==="
    )

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print(
        "\n=== 詳細評価 ==="
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "LOSS",
                "WIN",
            ],
            zero_division=0,
        )
    )

    print(
        "\n=== AI選別後 ==="
    )

    print(
        f"採用する勝ち確率: "
        f"{threshold:.0%}"
    )

    print(
        f"選別件数: "
        f"{selected_evaluation['selected_count']:,}件"
    )

    print(
        f"勝ち: "
        f"{selected_evaluation['wins']:,}件"
    )

    print(
        f"負け: "
        f"{selected_evaluation['losses']:,}件"
    )

    print(
        f"選別後勝率: "
        f"{selected_evaluation['win_rate']:.2f}%"
    )

    print(
        f"BUY候補: "
        f"{selected_evaluation['buy_count']:,}件"
    )

    print(
        f"SELL候補: "
        f"{selected_evaluation['sell_count']:,}件"
    )

    print(
        "\n最終選別結果保存先:",
        PREDICTIONS_PATH.resolve(),
    )


def main() -> None:
    try:
        dataset = load_dataset()

        (
            train,
            validation,
            test,
        ) = split_dataset(dataset)

        print(
            "=== Athena 売買結果AI ==="
        )

        print(
            f"全データ: "
            f"{len(dataset):,}件"
        )

        print(
            f"学習: "
            f"{len(train):,}件"
        )

        print(
            f"検証: "
            f"{len(validation):,}件"
        )

        print(
            f"最終テスト: "
            f"{len(test):,}件"
        )

        print(
            "学習期間:",
            train.iloc[0]["signal_time"],
            "～",
            train.iloc[-1]["signal_time"],
        )

        print(
            "検証期間:",
            validation.iloc[0]["signal_time"],
            "～",
            validation.iloc[-1]["signal_time"],
        )

        print(
            "最終テスト期間:",
            test.iloc[0]["signal_time"],
            "～",
            test.iloc[-1]["signal_time"],
        )

        model = create_model()

        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=train[TARGET_COLUMN],
        )

        model.fit(
            train[FEATURE_COLUMNS],
            train[TARGET_COLUMN],
            sample_weight=sample_weights,
        )

        threshold = select_threshold(
            model,
            validation,
        )

        evaluate_final_test(
            model=model,
            test=test,
            threshold=threshold,
        )

        MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            {
                "model": model,
                "features": FEATURE_COLUMNS,
                "threshold": threshold,
                "stop_loss_atr": 1.5,
                "take_profit_atr": 2.0,
                "max_holding_bars": 12,
            },
            MODEL_PATH,
        )

        print(
            "モデル保存先:",
            MODEL_PATH.resolve(),
        )

    except Exception as error:
        print(
            "売買結果AIの学習中に"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()