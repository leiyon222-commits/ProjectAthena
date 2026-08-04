from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


DATASET_PATH = Path(
    "data/market_context_features.csv"
)

RESULT_PATH = Path(
    "data/candlestick_pattern_model_comparison.csv"
)

PREDICTION_PATH = Path(
    "data/candlestick_pattern_test_predictions.csv"
)

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20

# 12本後をラベルにしているため、
# 期間の境界で12行空ける
GAP_ROWS = 12

TARGET_DOWN = 0
TARGET_NEUTRAL = 1
TARGET_UP = 2

CLASS_NAMES = [
    "DOWN",
    "NEUTRAL",
    "UP",
]

# 確信度別の方向予測を確認する
CONFIDENCE_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
]

# 時刻や未来情報など、学習に入れてはいけない列
EXCLUDED_COLUMNS = {
    "time",
    "target",
    "future_change",
    "open",
    "high",
    "low",
    "close",
}


def load_dataset() -> pd.DataFrame:
    """特徴量CSVを時刻順に読み込む。"""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "特徴量データが見つかりません: "
            f"{DATASET_PATH.resolve()}"
        )

    dataset = pd.read_csv(
        DATASET_PATH,
        parse_dates=["time"],
    )

    dataset = dataset.sort_values(
        "time"
    ).reset_index(drop=True)

    dataset = dataset.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    dataset["target"] = (
        dataset["target"].astype(int)
    )

    return dataset


def get_feature_columns(
    dataset: pd.DataFrame,
    include_patterns: bool,
) -> list[str]:
    """モデルへ渡す特徴量一覧を作る。"""
    feature_columns = []

    for column in dataset.columns:
        if column in EXCLUDED_COLUMNS:
            continue

        if (
            not include_patterns
            and column.startswith("pattern_")
        ):
            continue

        # 数値列だけを採用する
        if pd.api.types.is_numeric_dtype(
            dataset[column]
        ):
            feature_columns.append(column)

    return feature_columns


def split_dataset(
    dataset: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """古い60％・中間20％・最新20％へ分割する。"""
    total_count = len(dataset)

    train_end = int(
        total_count * TRAIN_RATIO
    )

    validation_end = int(
        total_count
        * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
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
    ].copy()

    if train.empty or validation.empty or test.empty:
        raise RuntimeError(
            "学習・検証・テスト分割に失敗しました"
        )

    return train, validation, test


def create_model() -> lgb.LGBMClassifier:
    """LightGBMの3分類モデルを作る。"""
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def train_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
) -> lgb.LGBMClassifier:
    """検証期間を監視しながら学習する。"""
    model = create_model()

    model.fit(
        train[feature_columns],
        train["target"],
        eval_set=[
            (
                validation[feature_columns],
                validation["target"],
            )
        ],
        eval_metric="multi_logloss",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                verbose=False,
            ),
            lgb.log_evaluation(
                period=0,
            ),
        ],
    )

    return model


def evaluate_confidence(
    probabilities: np.ndarray,
    actual: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """
    DOWNまたはUPを高確信度で予測した場面だけ評価する。
    NEUTRAL予測は取引候補にしない。
    """
    predicted = np.argmax(
        probabilities,
        axis=1,
    )

    confidence = np.max(
        probabilities,
        axis=1,
    )

    directional = (
        (predicted == TARGET_DOWN)
        | (predicted == TARGET_UP)
    )

    selected = (
        directional
        & (confidence >= threshold)
    )

    selected_count = int(
        selected.sum()
    )

    if selected_count == 0:
        return {
            "threshold": threshold,
            "selected_count": 0,
            "direction_accuracy": 0.0,
            "down_count": 0,
            "up_count": 0,
        }

    correct = (
        predicted[selected]
        == actual[selected]
    )

    return {
        "threshold": threshold,
        "selected_count": selected_count,
        "direction_accuracy": (
            float(correct.mean()) * 100
        ),
        "down_count": int(
            (
                predicted[selected]
                == TARGET_DOWN
            ).sum()
        ),
        "up_count": int(
            (
                predicted[selected]
                == TARGET_UP
            ).sum()
        ),
    }


def evaluate_model(
    model_name: str,
    model: lgb.LGBMClassifier,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict, pd.DataFrame]:
    """完全未使用期間でモデルを評価する。"""
    x_test = test[
        feature_columns
    ]

    y_test = test[
        "target"
    ].to_numpy()

    probabilities = model.predict_proba(
        x_test
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions,
        )
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
        multi_class="ovr",
        average="macro",
    )

    majority_accuracy = max(
        float((y_test == TARGET_DOWN).mean()),
        float((y_test == TARGET_NEUTRAL).mean()),
        float((y_test == TARGET_UP).mean()),
    )

    print(
        f"\n=== {model_name} ==="
    )

    print(
        f"特徴量数: "
        f"{len(feature_columns):,}"
    )

    print(
        f"学習反復回数: "
        f"{model.best_iteration_}"
    )

    print(
        f"多数派予測: "
        f"{majority_accuracy:.2%}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.2%}"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy:.2%}"
    )

    print(
        f"Macro ROC AUC: "
        f"{roc_auc:.3f}"
    )

    print("\n混同行列:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[
                TARGET_DOWN,
                TARGET_NEUTRAL,
                TARGET_UP,
            ],
        )
    )

    print("\n詳細評価:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[
                TARGET_DOWN,
                TARGET_NEUTRAL,
                TARGET_UP,
            ],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    confidence_results = []

    print("確信度別の方向予測:")

    for threshold in CONFIDENCE_THRESHOLDS:
        result = evaluate_confidence(
            probabilities=probabilities,
            actual=y_test,
            threshold=threshold,
        )

        confidence_results.append(result)

        print(
            f"{threshold:.0%}以上: "
            f"{result['selected_count']:,}件 / "
            f"的中率 "
            f"{result['direction_accuracy']:.2f}% / "
            f"DOWN "
            f"{result['down_count']:,} / "
            f"UP "
            f"{result['up_count']:,}"
        )

    output = test[
        [
            "time",
            "target",
            "future_change",
        ]
    ].copy()

    output[f"{model_name}_prediction"] = (
        predictions
    )

    output[f"{model_name}_confidence"] = (
        np.max(
            probabilities,
            axis=1,
        )
    )

    output[f"{model_name}_down_probability"] = (
        probabilities[:, TARGET_DOWN]
    )

    output[
        f"{model_name}_neutral_probability"
    ] = probabilities[:, TARGET_NEUTRAL]

    output[f"{model_name}_up_probability"] = (
        probabilities[:, TARGET_UP]
    )

    summary = {
        "model": model_name,
        "feature_count": len(feature_columns),
        "best_iteration": (
            model.best_iteration_
        ),
        "majority_accuracy": (
            majority_accuracy
        ),
        "accuracy": accuracy,
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "macro_roc_auc": roc_auc,
    }

    for result in confidence_results:
        threshold_name = int(
            result["threshold"] * 100
        )

        summary[
            f"selected_{threshold_name}"
        ] = result["selected_count"]

        summary[
            f"direction_accuracy_{threshold_name}"
        ] = result[
            "direction_accuracy"
        ]

    return summary, output


def print_pattern_importance(
    model: lgb.LGBMClassifier,
    feature_columns: list[str],
) -> None:
    """ローソク足パターンの重要度を表示する。"""
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": (
                model.feature_importances_
            ),
        }
    )

    pattern_importance = importance[
        importance["feature"].str.startswith(
            "pattern_"
        )
    ].sort_values(
        "importance",
        ascending=False,
    )

    print(
        "\n=== ローソク足パターン重要度 ==="
    )

    if pattern_importance.empty:
        print(
            "パターン特徴量がありません"
        )
        return

    print(
        pattern_importance.to_string(
            index=False
        )
    )


def main() -> None:
    try:
        dataset = load_dataset()

        train, validation, test = (
            split_dataset(dataset)
        )

        print(
            "=== ローソク足パターン比較 ==="
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
            "最終テスト期間:",
            test.iloc[0]["time"],
            "～",
            test.iloc[-1]["time"],
        )

        without_pattern_features = (
            get_feature_columns(
                dataset,
                include_patterns=False,
            )
        )

        with_pattern_features = (
            get_feature_columns(
                dataset,
                include_patterns=True,
            )
        )

        print(
            "\nパターンなしモデルを学習中..."
        )

        without_pattern_model = train_model(
            train=train,
            validation=validation,
            feature_columns=(
                without_pattern_features
            ),
        )

        print(
            "パターンありモデルを学習中..."
        )

        with_pattern_model = train_model(
            train=train,
            validation=validation,
            feature_columns=(
                with_pattern_features
            ),
        )

        (
            without_pattern_summary,
            without_pattern_predictions,
        ) = evaluate_model(
            model_name="without_patterns",
            model=without_pattern_model,
            test=test,
            feature_columns=(
                without_pattern_features
            ),
        )

        (
            with_pattern_summary,
            with_pattern_predictions,
        ) = evaluate_model(
            model_name="with_patterns",
            model=with_pattern_model,
            test=test,
            feature_columns=(
                with_pattern_features
            ),
        )

        comparison = pd.DataFrame(
            [
                without_pattern_summary,
                with_pattern_summary,
            ]
        )

        RESULT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        comparison.to_csv(
            RESULT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        prediction_output = (
            without_pattern_predictions.merge(
                with_pattern_predictions,
                on=[
                    "time",
                    "target",
                    "future_change",
                ],
                how="inner",
            )
        )

        prediction_output.to_csv(
            PREDICTION_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print_pattern_importance(
            model=with_pattern_model,
            feature_columns=(
                with_pattern_features
            ),
        )

        print(
            "\n=== 比較結果 ==="
        )

        print(
            comparison[
                [
                    "model",
                    "feature_count",
                    "accuracy",
                    "balanced_accuracy",
                    "macro_roc_auc",
                    "selected_60",
                    "direction_accuracy_60",
                    "selected_70",
                    "direction_accuracy_70",
                ]
            ].to_string(
                index=False
            )
        )

        auc_difference = (
            with_pattern_summary[
                "macro_roc_auc"
            ]
            - without_pattern_summary[
                "macro_roc_auc"
            ]
        )

        balanced_difference = (
            with_pattern_summary[
                "balanced_accuracy"
            ]
            - without_pattern_summary[
                "balanced_accuracy"
            ]
        )

        print(
            "\nパターン追加による差:"
        )

        print(
            f"Macro ROC AUC: "
            f"{auc_difference:+.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{balanced_difference:+.2%}"
        )

        if (
            auc_difference > 0
            and balanced_difference > 0
        ):
            print(
                "ローソク足パターンは、"
                "今回の完全未使用期間で"
                "改善に寄与しました。"
            )
        elif (
            auc_difference < 0
            and balanced_difference < 0
        ):
            print(
                "ローソク足パターンは、"
                "今回の完全未使用期間では"
                "成績を悪化させました。"
            )
        else:
            print(
                "評価指標によって結果が異なるため、"
                "ウォークフォワードで追加確認が必要です。"
            )

        print(
            "\n比較結果保存先:",
            RESULT_PATH.resolve(),
        )

        print(
            "予測結果保存先:",
            PREDICTION_PATH.resolve(),
        )

    except Exception as error:
        print(
            "\nパターン比較中に"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()