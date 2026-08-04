from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight


DATASET_PATH = Path(
    "data/ai_direct_signal_dataset.csv"
)

MODEL_PATH = Path(
    "data/athena_direct_signal_ai.joblib"
)

FINAL_SELECTION_PATH = Path(
    "data/athena_direct_signal_ai_final_selections.csv"
)

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20

# 元データは12本ごと、最大保有は24本なので、
# 期間境界で3サンプル分を空ける
GAP_SAMPLES = 3

# 最終評価では予測期間が重ならないよう、
# 2件ごとに評価する
FINAL_TEST_STEP = 2

TARGET_SELL = 0
TARGET_NO_TRADE = 1
TARGET_BUY = 2

CLASS_NAMES = [
    "SELL",
    "NO_TRADE",
    "BUY",
]

CONFIDENCE_THRESHOLDS = [
    0.40,
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
    0.20,
]

MIN_VALIDATION_TRADES = 100

FEATURE_COLUMNS = [
    "close",
    "ema10_gap_ratio",
    "ema20_gap_ratio",
    "ema50_gap_ratio",
    "ema100_gap_ratio",
    "ema200_gap_ratio",
    "ema10_slope",
    "ema20_slope",
    "ema50_slope",
    "ema100_slope",
    "ema200_slope",
    "ema10_20_gap",
    "ema20_50_gap",
    "ema50_100_gap",
    "ema100_200_gap",
    "rsi14",
    "atr14",
    "atr_ratio",
    "spread",
    "spread_atr_ratio",
    "tick_volume",
    "volume_ratio_12",
    "volume_ratio_48",
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
    "return_48",
    "return_96",
    "volatility_12",
    "volatility_48",
    "volatility_96",
    "position_in_range_12",
    "position_in_range_48",
    "hour_utc",
    "day_of_week",
]

TARGET_COLUMN = "target"


def load_dataset() -> pd.DataFrame:
    """AI直接判断用データを時刻順に読み込む。"""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "データセットが見つかりません: "
            f"{DATASET_PATH.resolve()}"
        )

    dataset = pd.read_csv(
        DATASET_PATH,
        parse_dates=["signal_time"],
    )

    required_columns = (
        ["signal_time", "buy_r", "sell_r"]
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

    dataset = dataset.sort_values(
        "signal_time"
    ).reset_index(drop=True)

    return dataset


def split_dataset(
    dataset: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """時系列順に60％・20％・20％へ分割する。"""
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
        :train_boundary - GAP_SAMPLES
    ].copy()

    validation = dataset.iloc[
        train_boundary:
        validation_boundary - GAP_SAMPLES
    ].copy()

    test = dataset.iloc[
        validation_boundary:
    ].copy()

    if train.empty or validation.empty or test.empty:
        raise RuntimeError(
            "学習・検証・最終テストの分割に失敗しました"
        )

    return train, validation, test


def create_model() -> HistGradientBoostingClassifier:
    """SELL・NO TRADE・BUYの3分類モデルを作成する。"""
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        max_depth=6,
        min_samples_leaf=80,
        l2_regularization=2.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
    )


def create_selection(
    model: HistGradientBoostingClassifier,
    rows: pd.DataFrame,
    confidence_threshold: float,
    probability_margin: float,
) -> pd.DataFrame:
    """確信度が高いBUY・SELLだけを選択する。"""
    probabilities = model.predict_proba(
        rows[FEATURE_COLUMNS]
    )

    predicted_indices = np.argmax(
        probabilities,
        axis=1,
    )

    predicted_classes = model.classes_[
        predicted_indices
    ]

    sorted_probabilities = np.sort(
        probabilities,
        axis=1,
    )

    top_probability = sorted_probabilities[:, -1]
    second_probability = sorted_probabilities[:, -2]

    margin = (
        top_probability
        - second_probability
    )

    directional = (
        (predicted_classes == TARGET_BUY)
        | (predicted_classes == TARGET_SELL)
    )

    selected_mask = (
        directional
        & (
            top_probability
            >= confidence_threshold
        )
        & (
            margin
            >= probability_margin
        )
    )

    selected = rows.loc[
        selected_mask
    ].copy()

    if selected.empty:
        return selected

    selected["predicted_class"] = (
        predicted_classes[selected_mask]
    )

    selected["confidence"] = (
        top_probability[selected_mask]
    )

    selected["probability_margin"] = (
        margin[selected_mask]
    )

    selected["predicted_direction"] = (
        selected["predicted_class"].map(
            {
                TARGET_SELL: "SELL",
                TARGET_BUY: "BUY",
            }
        )
    )

    selected["trade_r"] = np.where(
        selected["predicted_class"]
        == TARGET_BUY,
        selected["buy_r"],
        selected["sell_r"],
    )

    return selected.reset_index(
        drop=True
    )


def calculate_metrics(
    selected: pd.DataFrame,
) -> dict[str, float | int]:
    """選択した売買候補のR成績を集計する。"""
    if selected.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_r": 0.0,
            "average_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "buy_count": 0,
            "sell_count": 0,
        }

    wins = selected[
        selected["trade_r"] > 0
    ]

    losses = selected[
        selected["trade_r"] < 0
    ]

    gross_profit = float(
        wins["trade_r"].sum()
    )

    gross_loss = abs(
        float(losses["trade_r"].sum())
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit / gross_loss
        )
    else:
        profit_factor = float("inf")

    equity = selected[
        "trade_r"
    ].cumsum()

    # 最初の残高を0Rとして含める
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

    max_drawdown_r = abs(
        float(drawdown.min())
    )

    return {
        "trades": len(selected),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (
            len(wins)
            / len(selected)
            * 100
        ),
        "total_r": float(
            selected["trade_r"].sum()
        ),
        "average_r": float(
            selected["trade_r"].mean()
        ),
        "profit_factor": profit_factor,
        "max_drawdown_r": max_drawdown_r,
        "buy_count": int(
            (
                selected["predicted_class"]
                == TARGET_BUY
            ).sum()
        ),
        "sell_count": int(
            (
                selected["predicted_class"]
                == TARGET_SELL
            ).sum()
        ),
    }


def select_filter(
    model: HistGradientBoostingClassifier,
    validation: pd.DataFrame,
) -> tuple[float, float]:
    """検証期間だけを使って売買採用条件を決める。"""
    results: list[dict] = []

    print(
        "\n=== 検証期間で売買条件を選択 ==="
    )

    for threshold in CONFIDENCE_THRESHOLDS:
        for margin in PROBABILITY_MARGINS:
            selected = create_selection(
                model=model,
                rows=validation,
                confidence_threshold=threshold,
                probability_margin=margin,
            )

            metrics = calculate_metrics(
                selected
            )

            results.append(
                {
                    "threshold": threshold,
                    "margin": margin,
                    **metrics,
                }
            )

            print(
                f"確信度 {threshold:.0%} / "
                f"差 {margin:.0%}: "
                f"{metrics['trades']:,}件 / "
                f"平均 {metrics['average_r']:.4f}R / "
                f"PF {metrics['profit_factor']:.2f}"
            )

    eligible = [
        result
        for result in results
        if (
            result["trades"]
            >= MIN_VALIDATION_TRADES
        )
        and (
            result["average_r"] > 0
        )
        and (
            result["profit_factor"] > 1.0
        )
    ]

    if not eligible:
        raise RuntimeError(
            "検証期間で平均Rがプラスかつ"
            "PF1.0超になる条件がありませんでした"
        )

    best = max(
        eligible,
        key=lambda result: (
            result["average_r"],
            result["profit_factor"],
            result["trades"],
        ),
    )

    print(
        "\n=== 検証期間で採用する条件 ==="
    )

    print(
        "採用確信度:",
        f"{best['threshold']:.0%}",
    )

    print(
        "採用確率差:",
        f"{best['margin']:.0%}",
    )

    print(
        "検証取引数:",
        f"{best['trades']:,}件",
    )

    print(
        "検証勝率:",
        f"{best['win_rate']:.2f}%",
    )

    print(
        "検証合計R:",
        f"{best['total_r']:.2f}R",
    )

    print(
        "検証平均R:",
        f"{best['average_r']:.4f}R",
    )

    print(
        "検証PF:",
        f"{best['profit_factor']:.2f}",
    )

    return (
        float(best["threshold"]),
        float(best["margin"]),
    )


def print_final_metrics(
    metrics: dict[str, float | int],
) -> None:
    """完全未使用期間の成績を表示する。"""
    print(
        "\n=== 完全未使用期間の売買結果 ==="
    )

    print(
        f"取引回数: "
        f"{metrics['trades']:,}回"
    )

    print(
        f"勝ち: "
        f"{metrics['wins']:,}回"
    )

    print(
        f"負け: "
        f"{metrics['losses']:,}回"
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

        train, validation, test = (
            split_dataset(dataset)
        )

        print(
            "=== Athena 直接判断AI ==="
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

        # クラス件数差を補正する重みを作る
        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=train[TARGET_COLUMN],
        )

        model.fit(
            train[FEATURE_COLUMNS],
            train[TARGET_COLUMN],
            sample_weight=sample_weights,
        )

        validation_predictions = model.predict(
            validation[FEATURE_COLUMNS]
        )

        print(
            "\n=== 検証期間の3分類性能 ==="
        )

        print(
            f"Accuracy: "
            f"{accuracy_score(
                validation[TARGET_COLUMN],
                validation_predictions
            ):.2%}"
        )

        print(
            f"Balanced Accuracy: "
            f"{balanced_accuracy_score(
                validation[TARGET_COLUMN],
                validation_predictions
            ):.2%}"
        )

        threshold, margin = (
            select_filter(
                model,
                validation,
            )
        )

        # 24本保有に対しデータは12本間隔なので、
        # 最終評価では2件ごとに抽出する
        independent_test = test.iloc[
            ::FINAL_TEST_STEP
        ].copy()

        test_predictions = model.predict(
            independent_test[
                FEATURE_COLUMNS
            ]
        )

        print(
            "\n=== 完全未使用期間の3分類性能 ==="
        )

        print(
            f"独立評価件数: "
            f"{len(independent_test):,}件"
        )

        print(
            f"Accuracy: "
            f"{accuracy_score(
                independent_test[TARGET_COLUMN],
                test_predictions
            ):.2%}"
        )

        print(
            f"Balanced Accuracy: "
            f"{balanced_accuracy_score(
                independent_test[TARGET_COLUMN],
                test_predictions
            ):.2%}"
        )

        print(
            "\n=== 混同行列 ==="
        )

        print(
            confusion_matrix(
                independent_test[
                    TARGET_COLUMN
                ],
                test_predictions,
                labels=[
                    TARGET_SELL,
                    TARGET_NO_TRADE,
                    TARGET_BUY,
                ],
            )
        )

        print(
            "\n=== 詳細評価 ==="
        )

        print(
            classification_report(
                independent_test[
                    TARGET_COLUMN
                ],
                test_predictions,
                labels=[
                    TARGET_SELL,
                    TARGET_NO_TRADE,
                    TARGET_BUY,
                ],
                target_names=CLASS_NAMES,
                zero_division=0,
            )
        )

        final_selected = create_selection(
            model=model,
            rows=independent_test,
            confidence_threshold=threshold,
            probability_margin=margin,
        )

        final_metrics = calculate_metrics(
            final_selected
        )

        print_final_metrics(
            final_metrics
        )

        FINAL_SELECTION_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        final_selected.to_csv(
            FINAL_SELECTION_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        joblib.dump(
            {
                "model": model,
                "features": FEATURE_COLUMNS,
                "confidence_threshold": threshold,
                "probability_margin": margin,
                "class_mapping": {
                    TARGET_SELL: "SELL",
                    TARGET_NO_TRADE: "NO_TRADE",
                    TARGET_BUY: "BUY",
                },
                "stop_loss_atr": 2.0,
                "take_profit_atr": 3.0,
                "max_holding_bars": 24,
            },
            MODEL_PATH,
        )

        print(
            "\nモデル保存先:",
            MODEL_PATH.resolve(),
        )

        print(
            "最終選別結果保存先:",
            FINAL_SELECTION_PATH.resolve(),
        )

    except Exception as error:
        print(
            "\n直接判断AIの学習中に"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()