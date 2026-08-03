from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


DATASET_PATH = Path("data/ai_market_dataset.csv")
MODEL_PATH = Path("data/athena_market_ai_final.joblib")
RESULT_PATH = Path("data/athena_market_ai_final_predictions.csv")

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20

# ラベル作成に12本先を使っているため、
# 各期間の境界を12行空ける
GAP_ROWS = 12

THRESHOLDS = [
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
]

# 閾値選択時に最低限必要な検証件数
MIN_VALIDATION_SELECTIONS = 200

# 予測対象が12本先なので、最終評価では
# 同じ予測期間が重ならないよう12行ごとに評価する
NON_OVERLAP_STEP = 12

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
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"データセットが見つかりません: "
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
        ["time", "future_change"]
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    if (
        train.empty
        or validation.empty
        or test.empty
    ):
        raise RuntimeError(
            "学習・検証・最終テストの分割に失敗しました"
        )

    return train, validation, test


def create_model() -> HistGradientBoostingClassifier:
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


def evaluate_threshold(
    probabilities,
    actual,
    threshold: float,
) -> dict:
    up_selected = (
        probabilities >= threshold
    )

    down_selected = (
        probabilities <= 1 - threshold
    )

    selected = (
        up_selected | down_selected
    )

    selected_count = int(
        selected.sum()
    )

    if selected_count == 0:
        return {
            "threshold": threshold,
            "selected_count": 0,
            "accuracy": 0.0,
            "up_count": 0,
            "down_count": 0,
        }

    correct = (
        (
            up_selected
            & (actual == 1)
        )
        |
        (
            down_selected
            & (actual == 0)
        )
    )

    accuracy = (
        int(correct.sum())
        / selected_count
        * 100
    )

    return {
        "threshold": threshold,
        "selected_count": selected_count,
        "accuracy": accuracy,
        "up_count": int(up_selected.sum()),
        "down_count": int(down_selected.sum()),
    }


def select_threshold(
    model,
    validation: pd.DataFrame,
) -> float:
    x_validation = validation[
        FEATURE_COLUMNS
    ]

    y_validation = validation[
        TARGET_COLUMN
    ].to_numpy()

    probabilities = model.predict_proba(
        x_validation
    )[:, 1]

    print("\n=== 検証期間で確信度を選択 ===")

    results = []

    for threshold in THRESHOLDS:
        result = evaluate_threshold(
            probabilities,
            y_validation,
            threshold,
        )

        results.append(result)

        print(
            f"確信度 {threshold:.0%}以上: "
            f"{result['selected_count']:,}件 / "
            f"的中率 {result['accuracy']:.2f}%"
        )

    eligible = [
        result
        for result in results
        if result["selected_count"]
        >= MIN_VALIDATION_SELECTIONS
    ]

    if not eligible:
        raise RuntimeError(
            "最低検証件数を満たす確信度がありません"
        )

    # 的中率を優先し、同率なら取引候補数が多い方
    best = max(
        eligible,
        key=lambda result: (
            result["accuracy"],
            result["selected_count"],
        ),
    )

    print(
        "\n採用する確信度:",
        f"{best['threshold']:.0%}",
    )

    print(
        "検証期間の的中率:",
        f"{best['accuracy']:.2f}%",
    )

    print(
        "検証期間の対象件数:",
        f"{best['selected_count']:,}件",
    )

    return float(
        best["threshold"]
    )


def evaluate_final_test(
    model,
    test: pd.DataFrame,
    threshold: float,
) -> None:
    # 予測期間の重複を減らすため12行ごとに抽出
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
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    majority_accuracy = max(
        float((y_test == 0).mean()),
        float((y_test == 1).mean()),
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    result = evaluate_threshold(
        probabilities,
        y_test,
        threshold,
    )

    up_selected = (
        probabilities >= threshold
    )

    down_selected = (
        probabilities <= 1 - threshold
    )

    selected = (
        up_selected | down_selected
    )

    output = independent_test[
        [
            "time",
            "future_change",
            TARGET_COLUMN,
        ]
    ].copy()

    output["up_probability"] = (
        probabilities
    )

    output["selected"] = selected

    output["predicted_direction"] = "HOLD"
    output.loc[
        up_selected,
        "predicted_direction",
    ] = "UP"

    output.loc[
        down_selected,
        "predicted_direction",
    ] = "DOWN"

    output["correct"] = False

    output.loc[
        up_selected,
        "correct",
    ] = (
        output.loc[
            up_selected,
            TARGET_COLUMN,
        ] == 1
    )

    output.loc[
        down_selected,
        "correct",
    ] = (
        output.loc[
            down_selected,
            TARGET_COLUMN,
        ] == 0
    )

    output.to_csv(
        RESULT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== 完全未使用期間の最終結果 ===")
    print(
        f"独立評価件数: "
        f"{len(independent_test):,}件"
    )

    print(
        f"上昇比率: "
        f"{y_test.mean():.2%}"
    )

    print(
        f"多数派だけを予測: "
        f"{majority_accuracy:.2%}"
    )

    print(
        f"AI Accuracy: "
        f"{accuracy:.2%}"
    )

    print(
        f"AI ROC AUC: "
        f"{roc_auc:.3f}"
    )

    print(
        f"\n採用確信度: "
        f"{threshold:.0%}"
    )

    print(
        f"AIが選択した件数: "
        f"{result['selected_count']:,}件"
    )

    print(
        f"UP予測: "
        f"{result['up_count']:,}件"
    )

    print(
        f"DOWN予測: "
        f"{result['down_count']:,}件"
    )

    print(
        f"選別後の方向的中率: "
        f"{result['accuracy']:.2f}%"
    )

    print(
        "\n最終予測保存先:",
        RESULT_PATH.resolve(),
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
            "=== Athena 市場AI "
            "3期間評価 ==="
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

        model.fit(
            train[FEATURE_COLUMNS],
            train[TARGET_COLUMN],
        )

        threshold = select_threshold(
            model,
            validation,
        )

        evaluate_final_test(
            model,
            test,
            threshold,
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
                "future_bars": 12,
                "non_overlap_step":
                    NON_OVERLAP_STEP,
            },
            MODEL_PATH,
        )

        print(
            "モデル保存先:",
            MODEL_PATH.resolve(),
        )

    except Exception as error:
        print(
            "AI最終評価中に"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()