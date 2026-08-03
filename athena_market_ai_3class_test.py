from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight


DATASET_PATH = Path("data/ai_market_dataset_v2.csv")
MODEL_PATH = Path("data/athena_market_ai_3class.joblib")
RESULT_PATH = Path(
    "data/athena_market_ai_3class_predictions.csv"
)

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20

# ラベル作成で12本先を使っているため、
# 各期間の境界を12行空ける
GAP_ROWS = 12

# 最終評価では予測期間が重ならないよう、
# 12行ごとに評価する
NON_OVERLAP_STEP = 12

TARGET_DOWN = 0
TARGET_NEUTRAL = 1
TARGET_UP = 2

CLASS_NAMES = [
    "DOWN",
    "NEUTRAL",
    "UP",
]

CONFIDENCE_THRESHOLDS = [
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
]

PROBABILITY_MARGINS = [
    0.00,
    0.05,
    0.10,
    0.15,
]

MIN_VALIDATION_SELECTIONS = 200

FEATURE_COLUMNS = [
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

TARGET_COLUMN = "target"


def load_dataset() -> pd.DataFrame:
    """3分類データセットを時刻順に読み込む。"""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "データセットが見つかりません: "
            f"{DATASET_PATH.resolve()}"
        )

    dataset = pd.read_csv(
        DATASET_PATH,
        parse_dates=["time"],
    )

    dataset = dataset.sort_values(
        "time"
    ).reset_index(drop=True)

    required_columns = (
        ["time"]
        + FEATURE_COLUMNS
        + [TARGET_COLUMN]
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
        [float("inf"), float("-inf")],
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
    """60％・20％・20％に時系列分割する。"""
    total_count = len(dataset)

    train_boundary = int(
        total_count * TRAIN_RATIO
    )

    validation_boundary = int(
        total_count
        * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    train = dataset.iloc[
        :train_boundary - GAP_ROWS
    ].copy()

    validation = dataset.iloc[
        train_boundary:
        validation_boundary - GAP_ROWS
    ].copy()

    test = dataset.iloc[
        validation_boundary:
    ].copy()

    if train.empty or validation.empty or test.empty:
        raise RuntimeError(
            "学習・検証・テスト分割に失敗しました"
        )

    return train, validation, test


def create_model() -> HistGradientBoostingClassifier:
    """3分類モデルを作成する。"""
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        max_depth=5,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
    )


def get_class_probability(
    model: HistGradientBoostingClassifier,
    probabilities: np.ndarray,
    target_class: int,
) -> np.ndarray:
    """指定クラスの確率列を取得する。"""
    class_index = list(
        model.classes_
    ).index(target_class)

    return probabilities[:, class_index]


def create_trade_selection(
    model: HistGradientBoostingClassifier,
    probabilities: np.ndarray,
    confidence_threshold: float,
    probability_margin: float,
) -> dict[str, np.ndarray]:
    """
    UPまたはDOWNの確率が高い場面だけを選択する。

    NEUTRAL予測は取引しない。
    1位と2位の確率差が小さい場合も見送る。
    """
    down_probability = get_class_probability(
        model,
        probabilities,
        TARGET_DOWN,
    )

    neutral_probability = get_class_probability(
        model,
        probabilities,
        TARGET_NEUTRAL,
    )

    up_probability = get_class_probability(
        model,
        probabilities,
        TARGET_UP,
    )

    probability_frame = np.column_stack(
        [
            down_probability,
            neutral_probability,
            up_probability,
        ]
    )

    predicted_class = model.classes_[
        np.argmax(probability_frame, axis=1)
    ]

    sorted_probabilities = np.sort(
        probability_frame,
        axis=1,
    )

    top_probability = sorted_probabilities[:, -1]
    second_probability = sorted_probabilities[:, -2]

    margin = (
        top_probability
        - second_probability
    )

    directional_prediction = (
        (predicted_class == TARGET_DOWN)
        | (predicted_class == TARGET_UP)
    )

    selected = (
        directional_prediction
        & (
            top_probability
            >= confidence_threshold
        )
        & (
            margin
            >= probability_margin
        )
    )

    return {
        "predicted_class": predicted_class,
        "selected": selected,
        "top_probability": top_probability,
        "probability_margin": margin,
        "down_probability": down_probability,
        "neutral_probability": neutral_probability,
        "up_probability": up_probability,
    }


def evaluate_selection(
    actual: np.ndarray,
    selection: dict[str, np.ndarray],
) -> dict[str, float | int]:
    """選別した方向予測の成績を集計する。"""
    selected = selection["selected"]
    predicted = selection["predicted_class"]

    selected_count = int(
        selected.sum()
    )

    if selected_count == 0:
        return {
            "selected_count": 0,
            "correct_count": 0,
            "accuracy": 0.0,
            "coverage": 0.0,
            "down_count": 0,
            "up_count": 0,
        }

    correct = (
        predicted[selected]
        == actual[selected]
    )

    correct_count = int(
        correct.sum()
    )

    return {
        "selected_count": selected_count,
        "correct_count": correct_count,
        "accuracy": (
            correct_count
            / selected_count
            * 100
        ),
        "coverage": (
            selected_count
            / len(actual)
            * 100
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


def select_trade_filter(
    model: HistGradientBoostingClassifier,
    validation: pd.DataFrame,
) -> tuple[float, float]:
    """検証期間だけを使い、売買条件を決める。"""
    x_validation = validation[
        FEATURE_COLUMNS
    ]

    y_validation = validation[
        TARGET_COLUMN
    ].to_numpy()

    probabilities = model.predict_proba(
        x_validation
    )

    results: list[dict] = []

    print(
        "\n=== 検証期間で売買条件を選択 ==="
    )

    for confidence_threshold in (
        CONFIDENCE_THRESHOLDS
    ):
        for probability_margin in (
            PROBABILITY_MARGINS
        ):
            selection = create_trade_selection(
                model=model,
                probabilities=probabilities,
                confidence_threshold=(
                    confidence_threshold
                ),
                probability_margin=(
                    probability_margin
                ),
            )

            evaluation = evaluate_selection(
                y_validation,
                selection,
            )

            result = {
                "confidence": confidence_threshold,
                "margin": probability_margin,
                **evaluation,
            }

            results.append(result)

            print(
                f"確信度 "
                f"{confidence_threshold:.0%} / "
                f"差 {probability_margin:.0%}: "
                f"{evaluation['selected_count']:,}件 / "
                f"的中率 "
                f"{evaluation['accuracy']:.2f}% / "
                f"対象率 "
                f"{evaluation['coverage']:.2f}%"
            )

    eligible = [
        result
        for result in results
        if result["selected_count"]
        >= MIN_VALIDATION_SELECTIONS
    ]

    if not eligible:
        raise RuntimeError(
            "最低検証件数を満たす条件がありません"
        )

    # 的中率を優先。
    # 同率なら対象件数が多い条件を採用する。
    best = max(
        eligible,
        key=lambda result: (
            result["accuracy"],
            result["selected_count"],
        ),
    )

    print(
        "\n採用確信度:",
        f"{best['confidence']:.0%}",
    )

    print(
        "採用確率差:",
        f"{best['margin']:.0%}",
    )

    print(
        "検証期間の方向的中率:",
        f"{best['accuracy']:.2f}%",
    )

    print(
        "検証期間の対象件数:",
        f"{best['selected_count']:,}件",
    )

    return (
        float(best["confidence"]),
        float(best["margin"]),
    )


def evaluate_final_test(
    model: HistGradientBoostingClassifier,
    test: pd.DataFrame,
    confidence_threshold: float,
    probability_margin: float,
) -> None:
    """完全未使用期間を一度だけ評価する。"""
    independent_test = test.iloc[
        ::NON_OVERLAP_STEP
    ].copy()

    x_test = independent_test[
        FEATURE_COLUMNS
    ]

    y_test = independent_test[
        TARGET_COLUMN
    ].to_numpy()

    probabilities = model.predict_proba(
        x_test
    )

    predictions = model.predict(
        x_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
        multi_class="ovr",
        average="macro",
    )

    selection = create_trade_selection(
        model=model,
        probabilities=probabilities,
        confidence_threshold=(
            confidence_threshold
        ),
        probability_margin=(
            probability_margin
        ),
    )

    evaluation = evaluate_selection(
        y_test,
        selection,
    )

    output = independent_test[
        [
            "time",
            TARGET_COLUMN,
        ]
    ].copy()

    output["predicted_class"] = (
        selection["predicted_class"]
    )

    output["selected"] = (
        selection["selected"]
    )

    output["confidence"] = (
        selection["top_probability"]
    )

    output["probability_margin"] = (
        selection["probability_margin"]
    )

    output["down_probability"] = (
        selection["down_probability"]
    )

    output["neutral_probability"] = (
        selection["neutral_probability"]
    )

    output["up_probability"] = (
        selection["up_probability"]
    )

    output["actual_name"] = output[
        TARGET_COLUMN
    ].map(
        {
            TARGET_DOWN: "DOWN",
            TARGET_NEUTRAL: "NEUTRAL",
            TARGET_UP: "UP",
        }
    )

    output["predicted_name"] = output[
        "predicted_class"
    ].map(
        {
            TARGET_DOWN: "DOWN",
            TARGET_NEUTRAL: "NEUTRAL",
            TARGET_UP: "UP",
        }
    )

    output["correct"] = (
        output["predicted_class"]
        == output[TARGET_COLUMN]
    )

    output.to_csv(
        RESULT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    majority_accuracy = max(
        float(
            (
                y_test
                == TARGET_DOWN
            ).mean()
        ),
        float(
            (
                y_test
                == TARGET_NEUTRAL
            ).mean()
        ),
        float(
            (
                y_test
                == TARGET_UP
            ).mean()
        ),
    )

    print(
        "\n=== 完全未使用期間の最終結果 ==="
    )

    print(
        f"独立評価件数: "
        f"{len(independent_test):,}件"
    )

    print(
        f"DOWN: "
        f"{int((y_test == TARGET_DOWN).sum()):,}件"
    )

    print(
        f"NEUTRAL: "
        f"{int((y_test == TARGET_NEUTRAL).sum()):,}件"
    )

    print(
        f"UP: "
        f"{int((y_test == TARGET_UP).sum()):,}件"
    )

    print(
        f"\n多数派だけを予測: "
        f"{majority_accuracy:.2%}"
    )

    print(
        f"3分類AI Accuracy: "
        f"{accuracy:.2%}"
    )

    print(
        f"3分類AI ROC AUC: "
        f"{roc_auc:.3f}"
    )

    print(
        "\n=== 混同行列 ==="
    )

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

    print(
        "\n=== 詳細評価 ==="
    )

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

    print(
        "\n=== AI売買候補の評価 ==="
    )

    print(
        f"採用確信度: "
        f"{confidence_threshold:.0%}"
    )

    print(
        f"採用確率差: "
        f"{probability_margin:.0%}"
    )

    print(
        f"AIが選択した件数: "
        f"{evaluation['selected_count']:,}件"
    )

    print(
        f"DOWN予測: "
        f"{evaluation['down_count']:,}件"
    )

    print(
        f"UP予測: "
        f"{evaluation['up_count']:,}件"
    )

    print(
        f"評価期間に対する対象率: "
        f"{evaluation['coverage']:.2f}%"
    )

    print(
        f"選別後の方向的中率: "
        f"{evaluation['accuracy']:.2f}%"
    )

    print(
        "\n最終予測保存先:",
        RESULT_PATH.resolve(),
    )


def main() -> None:
    try:
        dataset = load_dataset()

        train, validation, test = (
            split_dataset(dataset)
        )

        print(
            "=== Athena 3分類市場AI ==="
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

        (
            confidence_threshold,
            probability_margin,
        ) = select_trade_filter(
            model,
            validation,
        )

        evaluate_final_test(
            model=model,
            test=test,
            confidence_threshold=(
                confidence_threshold
            ),
            probability_margin=(
                probability_margin
            ),
        )

        MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            {
                "model": model,
                "features": FEATURE_COLUMNS,
                "confidence_threshold": (
                    confidence_threshold
                ),
                "probability_margin": (
                    probability_margin
                ),
                "future_bars": 12,
                "target_atr_multiplier": 0.35,
                "class_mapping": {
                    0: "DOWN",
                    1: "NEUTRAL",
                    2: "UP",
                },
            },
            MODEL_PATH,
        )

        print(
            "モデル保存先:",
            MODEL_PATH.resolve(),
        )

    except Exception as error:
        print(
            "3分類AI評価中に"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()