from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


DATASET_PATH = Path("data/ai_trade_dataset.csv")
MODEL_PATH = Path("data/athena_random_forest.joblib")
IMPORTANCE_PATH = Path("data/ai_feature_importance.csv")

TRAIN_RATIO = 0.75

FEATURE_COLUMNS = [
    "direction_value",
    "close",
    "ema50",
    "ema100",
    "ema_gap",
    "ema_gap_ratio",
    "rsi14",
    "atr14",
    "atr_ratio",
    "spread",
    "spread_atr_ratio",
    "tick_volume",
    "volume_change",
    "candle_body",
    "candle_range",
    "body_ratio",
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "hour_utc",
    "day_of_week",
]

TARGET_COLUMN = "result"


def load_dataset() -> pd.DataFrame:
    """AI用CSVを時刻順に読み込む。"""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"AIデータセットが見つかりません: "
            f"{DATASET_PATH.resolve()}"
        )

    dataset = pd.read_csv(
        DATASET_PATH,
        parse_dates=["signal_time"],
    )

    dataset = dataset.sort_values(
        "signal_time"
    ).reset_index(drop=True)

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN, "signal_time"]
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

    clean_dataset = dataset[
        required_columns
    ].copy()

    clean_dataset = clean_dataset.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    ).dropna()

    return clean_dataset


def split_dataset(
    dataset: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """古い75％を学習、新しい25％を評価に使う。"""
    split_index = int(
        len(dataset) * TRAIN_RATIO
    )

    if split_index <= 0 or split_index >= len(dataset):
        raise RuntimeError(
            "学習・テスト分割に必要なデータが不足しています"
        )

    train = dataset.iloc[
        :split_index
    ].copy()

    test = dataset.iloc[
        split_index:
    ].copy()

    x_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    x_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    return x_train, x_test, y_train, y_test


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Random Forestを学習する。"""
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        x_train,
        y_train,
    )

    return model


def evaluate_model(
    model: RandomForestClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """未知期間でモデルを評価する。"""
    predictions = model.predict(
        x_test
    )

    win_probabilities = model.predict_proba(
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

    if y_test.nunique() == 2:
        roc_auc = roc_auc_score(
            y_test,
            win_probabilities,
        )
    else:
        roc_auc = float("nan")

    baseline_accuracy = max(
        float((y_test == 0).mean()),
        float((y_test == 1).mean()),
    )

    print("\n=== Athena AI 評価結果 ===")
    print(f"テスト件数: {len(y_test)}件")
    print(
        f"テスト期間の勝ち: "
        f"{int((y_test == 1).sum())}件"
    )
    print(
        f"テスト期間の負け: "
        f"{int((y_test == 0).sum())}件"
    )

    print(
        f"\n多数派だけを予測した場合: "
        f"{baseline_accuracy:.2%}"
    )
    print(f"AI Accuracy: {accuracy:.2%}")
    print(f"AI Precision: {precision:.2%}")
    print(f"AI Recall: {recall:.2%}")
    print(f"AI ROC AUC: {roc_auc:.3f}")

    print("\n=== 混同行列 ===")
    print(confusion_matrix(
        y_test,
        predictions,
    ))

    print("\n=== 詳細評価 ===")
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

    selected_count = int(
        predictions.sum()
    )

    if selected_count > 0:
        selected_actual_wins = int(
            y_test[predictions == 1].sum()
        )

        filtered_win_rate = (
            selected_actual_wins
            / selected_count
            * 100
        )

        print(
            "AIが取引すると判断した件数:",
            selected_count,
        )
        print(
            "その中の実際の勝ち:",
            selected_actual_wins,
        )
        print(
            f"AI選別後の勝率: "
            f"{filtered_win_rate:.2f}%"
        )
    else:
        print(
            "AIはテスト期間の全取引を"
            "見送りと判定しました"
        )


def save_model_and_importance(
    model: RandomForestClassifier,
) -> None:
    """モデルと特徴量重要度を保存する。"""
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        IMPORTANCE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== 特徴量重要度 上位10件 ===")
    print(
        importance.head(10).to_string(
            index=False
        )
    )

    print(
        "\nモデル保存先:",
        MODEL_PATH.resolve(),
    )

    print(
        "特徴量重要度保存先:",
        IMPORTANCE_PATH.resolve(),
    )


def main() -> None:
    try:
        dataset = load_dataset()

        print("=== Athena AI 学習開始 ===")
        print(f"全データ: {len(dataset)}件")
        print(
            "期間:",
            dataset.iloc[0]["signal_time"],
            "～",
            dataset.iloc[-1]["signal_time"],
        )

        (
            x_train,
            x_test,
            y_train,
            y_test,
        ) = split_dataset(dataset)

        print(f"学習件数: {len(x_train)}件")
        print(f"テスト件数: {len(x_test)}件")

        print(
            "学習側の勝率:",
            f"{y_train.mean():.2%}",
        )

        print(
            "テスト側の勝率:",
            f"{y_test.mean():.2%}",
        )

        model = train_model(
            x_train,
            y_train,
        )

        evaluate_model(
            model,
            x_test,
            y_test,
        )

        save_model_and_importance(
            model
        )

    except Exception as error:
        print(
            "AI学習中にエラーが発生しました"
        )
        print(error)


if __name__ == "__main__":
    main()