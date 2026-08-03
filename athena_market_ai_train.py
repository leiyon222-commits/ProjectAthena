from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


DATASET_PATH = Path("data/ai_market_dataset.csv")
MODEL_PATH = Path("data/athena_market_ai.joblib")
PREDICTIONS_PATH = Path("data/athena_market_ai_predictions.csv")

TRAIN_RATIO = 0.75

# ラベル作成に未来12本を使っているため、
# 学習期間とテスト期間の境界を12件空ける
GAP_ROWS = 12

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
    """AI市場データを時刻順に読み込む。"""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"データセットが見つかりません: {DATASET_PATH.resolve()}"
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
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    """古い75％を学習、新しい25％をテストに使う。"""
    split_index = int(
        len(dataset) * TRAIN_RATIO
    )

    train_end_index = (
        split_index - GAP_ROWS
    )

    if train_end_index <= 0:
        raise RuntimeError(
            "学習データが不足しています"
        )

    train = dataset.iloc[
        :train_end_index
    ].copy()

    test = dataset.iloc[
        split_index:
    ].copy()

    x_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    x_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    return (
        x_train,
        x_test,
        y_train,
        y_test,
        test,
    )


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> HistGradientBoostingClassifier:
    """上昇・下落予測モデルを学習する。"""
    model = HistGradientBoostingClassifier(
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

    model.fit(
        x_train,
        y_train,
    )

    return model


def evaluate_model(
    model: HistGradientBoostingClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    test_rows: pd.DataFrame,
) -> None:
    """未知期間でAIを評価する。"""
    predictions = model.predict(
        x_test
    )

    probabilities = model.predict_proba(
        x_test
    )[:, 1]

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

    majority_accuracy = max(
        float((y_test == 0).mean()),
        float((y_test == 1).mean()),
    )

    print("\n=== Athena 市場AI 評価結果 ===")
    print(f"テスト件数: {len(y_test):,}件")
    print(
        f"上昇: {int((y_test == 1).sum()):,}件"
    )
    print(
        f"下落: {int((y_test == 0).sum()):,}件"
    )

    print(
        f"\n多数派だけを予測した場合: "
        f"{majority_accuracy:.2%}"
    )

    print(f"AI Accuracy: {accuracy:.2%}")
    print(f"AI Precision: {precision:.2%}")
    print(f"AI Recall: {recall:.2%}")
    print(f"AI ROC AUC: {roc_auc:.3f}")

    print("\n=== 混同行列 ===")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print("\n=== 詳細評価 ===")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "DOWN",
                "UP",
            ],
            zero_division=0,
        )
    )

    result = test_rows[
        ["time", TARGET_COLUMN]
    ].copy()

    result["predicted_target"] = (
        predictions
    )

    result["up_probability"] = (
        probabilities
    )

    result.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== 確信度別の成績 ===")

    for threshold in [
        0.55,
        0.60,
        0.65,
        0.70,
    ]:
        high_confidence_up = (
            probabilities >= threshold
        )

        high_confidence_down = (
            probabilities <= (1 - threshold)
        )

        selected = (
            high_confidence_up
            | high_confidence_down
        )

        selected_count = int(
            selected.sum()
        )

        if selected_count == 0:
            print(
                f"確信度 {threshold:.0%}以上: "
                "該当なし"
            )
            continue

        correct = (
            (
                high_confidence_up
                & (y_test.to_numpy() == 1)
            )
            |
            (
                high_confidence_down
                & (y_test.to_numpy() == 0)
            )
        )

        selected_accuracy = (
            int(correct.sum())
            / selected_count
            * 100
        )

        print(
            f"確信度 {threshold:.0%}以上: "
            f"{selected_count:,}件 / "
            f"的中率 {selected_accuracy:.2f}%"
        )

    print(
        "\n予測結果保存先:",
        PREDICTIONS_PATH.resolve(),
    )


def save_model(
    model: HistGradientBoostingClassifier,
) -> None:
    """学習済みモデルを保存する。"""
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
            "future_bars": 12,
            "target_atr_multiplier": 0.35,
        },
        MODEL_PATH,
    )

    print(
        "モデル保存先:",
        MODEL_PATH.resolve(),
    )


def main() -> None:
    try:
        dataset = load_dataset()

        print("=== Athena 市場AI 学習開始 ===")
        print(
            f"全データ: {len(dataset):,}件"
        )
        print(
            "期間:",
            dataset.iloc[0]["time"],
            "～",
            dataset.iloc[-1]["time"],
        )

        (
            x_train,
            x_test,
            y_train,
            y_test,
            test_rows,
        ) = split_dataset(dataset)

        print(
            f"学習件数: {len(x_train):,}件"
        )
        print(
            f"テスト件数: {len(x_test):,}件"
        )
        print(
            f"境界の除外件数: {GAP_ROWS}件"
        )
        print(
            f"学習側の上昇比率: "
            f"{y_train.mean():.2%}"
        )
        print(
            f"テスト側の上昇比率: "
            f"{y_test.mean():.2%}"
        )

        model = train_model(
            x_train,
            y_train,
        )

        evaluate_model(
            model,
            x_test,
            y_test,
            test_rows,
        )

        save_model(model)

    except Exception as error:
        print(
            "市場AI学習中に"
            "エラーが発生しました"
        )
        print(error)


if __name__ == "__main__":
    main()