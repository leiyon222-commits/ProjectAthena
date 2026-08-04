from pathlib import Path

import joblib
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
    "data/market_context_trade_labels.csv"
)

MODEL_PATH = Path(
    "data/athena_trade_barrier_lightgbm.joblib"
)

RESULT_PATH = Path(
    "data/athena_trade_barrier_final_selections.csv"
)

THRESHOLD_RESULT_PATH = Path(
    "data/athena_trade_barrier_threshold_results.csv"
)

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20

# 24本先まで確認して教師ラベルを作っているため、
# 期間境界で24行を除外する
GAP_ROWS = 24

# M5 × 24本なので、検証時は24行ごとに評価する
NON_OVERLAP_STEP = 12

TARGET_SELL = 0
TARGET_NO_TRADE = 1
TARGET_BUY = 2

CLASS_NAMES = [
    "SELL",
    "NO_TRADE",
    "BUY",
]

# TP 2.0 / SL 1.5
WIN_R = 2.0 / 1.5
LOSS_R = -1.0

CONFIDENCE_THRESHOLDS = [
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

PROBABILITY_MARGINS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
]

# 閾値を採用するための最低検証取引数
MIN_VALIDATION_TRADES = 30

# 学習させない列
EXCLUDED_COLUMNS = {
    "time",
    "target",
    "buy_win",
    "sell_win",
    "buy_exit_reason",
    "sell_exit_reason",
    "buy_holding_bars",
    "sell_holding_bars",
    "open",
    "high",
    "low",
    "close",
}


def load_dataset() -> pd.DataFrame:
    """売買バリアラベル付きデータを読み込む。"""
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

    dataset["target"] = (
        dataset["target"].astype(int)
    )

    return dataset


def get_feature_columns(
    dataset: pd.DataFrame,
) -> list[str]:
    """未来情報や結果列を除き、数値特徴量だけを選ぶ。"""
    feature_columns = []

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


def split_dataset(
    dataset: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """古い60％・中間20％・最新20％へ分割する。"""
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
            "学習・検証・最終テストの分割に失敗しました"
        )

    return train, validation, test


def create_model() -> lgb.LGBMClassifier:
    """3分類LightGBMモデルを作る。"""
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=150,
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
    """検証損失を監視しながらモデルを学習する。"""
    model = create_model()

    model.fit(
        train[feature_columns],
        train["target"],
        eval_X=validation[feature_columns],
        eval_y=validation["target"],
        eval_metric="multi_logloss",
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


def create_selection(
    model: lgb.LGBMClassifier,
    rows: pd.DataFrame,
    feature_columns: list[str],
    confidence_threshold: float,
    probability_margin: float,
) -> pd.DataFrame:
    """高確信度のBUY・SELL予測だけを取り出す。"""
    probabilities = model.predict_proba(
        rows[feature_columns]
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

    top_probability = (
        sorted_probabilities[:, -1]
    )

    second_probability = (
        sorted_probabilities[:, -2]
    )

    margin = (
        top_probability
        - second_probability
    )

    directional = (
        (predicted_classes == TARGET_SELL)
        | (predicted_classes == TARGET_BUY)
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

    selected["correct"] = (
        selected["predicted_class"]
        == selected["target"]
    )

    # 保守的評価：
    # 正解なら+1.333R、誤りなら-1R
    selected["trade_r"] = np.where(
        selected["correct"],
        WIN_R,
        LOSS_R,
    )

    return selected.reset_index(
        drop=True
    )


def calculate_metrics(
    selected: pd.DataFrame,
) -> dict[str, float | int]:
    """選別後の売買成績を計算する。"""
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
            "sell_count": 0,
            "buy_count": 0,
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
        float(
            losses["trade_r"].sum()
        )
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
        "max_drawdown_r": max_drawdown,
        "sell_count": int(
            (
                selected["predicted_class"]
                == TARGET_SELL
            ).sum()
        ),
        "buy_count": int(
            (
                selected["predicted_class"]
                == TARGET_BUY
            ).sum()
        ),
    }


def select_filter(
    model: lgb.LGBMClassifier,
    validation: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[float, float, pd.DataFrame]:
    """検証期間だけを使って売買条件を決める。"""
    independent_validation = (
        validation.iloc[
            ::NON_OVERLAP_STEP
        ].copy()
    )

    results: list[dict] = []

    print(
        "\n=== 検証期間で売買条件を選択 ==="
    )

    print(
        f"独立評価候補: "
        f"{len(independent_validation):,}件"
    )

    for threshold in CONFIDENCE_THRESHOLDS:
        for margin in PROBABILITY_MARGINS:
            selected = create_selection(
                model=model,
                rows=independent_validation,
                feature_columns=feature_columns,
                confidence_threshold=threshold,
                probability_margin=margin,
            )

            metrics = calculate_metrics(
                selected
            )

            result = {
                "threshold": threshold,
                "margin": margin,
                **metrics,
            }

            results.append(result)

            print(
                f"確信度 {threshold:.0%} / "
                f"差 {margin:.0%}: "
                f"{metrics['trades']:,}回 / "
                f"勝率 {metrics['win_rate']:.2f}% / "
                f"平均 {metrics['average_r']:.4f}R / "
                f"PF {metrics['profit_factor']:.2f}"
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
        raise RuntimeError(
            "検証期間で最低取引数を満たし、"
            "平均Rプラス・PF1超となる条件は"
            "見つかりませんでした"
        )

    eligible = eligible.sort_values(
        by=[
            "average_r",
            "profit_factor",
            "trades",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    best = eligible.iloc[0]

    print(
        "\n=== 採用する売買条件 ==="
    )

    print(
        f"確信度: "
        f"{best['threshold']:.0%}"
    )

    print(
        f"確率差: "
        f"{best['margin']:.0%}"
    )

    print(
        f"検証取引数: "
        f"{int(best['trades']):,}回"
    )

    print(
        f"検証勝率: "
        f"{best['win_rate']:.2f}%"
    )

    print(
        f"検証合計R: "
        f"{best['total_r']:.2f}R"
    )

    print(
        f"検証平均R: "
        f"{best['average_r']:.4f}R"
    )

    print(
        f"検証PF: "
        f"{best['profit_factor']:.2f}"
    )

    return (
        float(best["threshold"]),
        float(best["margin"]),
        result_frame,
    )


def evaluate_classification(
    model: lgb.LGBMClassifier,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """完全未使用期間の3分類性能を表示する。"""
    probabilities = model.predict_proba(
        test[feature_columns]
    )

    predictions = model.classes_[
        np.argmax(
            probabilities,
            axis=1,
        )
    ]

    actual = test[
        "target"
    ].to_numpy()

    accuracy = accuracy_score(
        actual,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            actual,
            predictions,
        )
    )

    roc_auc = roc_auc_score(
        actual,
        probabilities,
        multi_class="ovr",
        average="macro",
    )

    majority_accuracy = max(
        float(
            (
                actual == TARGET_SELL
            ).mean()
        ),
        float(
            (
                actual == TARGET_NO_TRADE
            ).mean()
        ),
        float(
            (
                actual == TARGET_BUY
            ).mean()
        ),
    )

    print(
        "\n=== 完全未使用期間の分類性能 ==="
    )

    print(
        f"多数派だけを予測: "
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
            actual,
            predictions,
            labels=[
                TARGET_SELL,
                TARGET_NO_TRADE,
                TARGET_BUY,
            ],
        )
    )

    print("\n詳細評価:")

    print(
        classification_report(
            actual,
            predictions,
            labels=[
                TARGET_SELL,
                TARGET_NO_TRADE,
                TARGET_BUY,
            ],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )


def print_final_metrics(
    metrics: dict[str, float | int],
) -> None:
    """完全未使用期間の売買成績を表示する。"""
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
        f"SELL: "
        f"{metrics['sell_count']:,}回"
    )

    print(
        f"BUY: "
        f"{metrics['buy_count']:,}回"
    )


def main() -> None:
    try:
        dataset = load_dataset()

        feature_columns = get_feature_columns(
            dataset
        )

        train, validation, test = (
            split_dataset(dataset)
        )

        print(
            "=== Athena 売買バリアAI ==="
        )

        print(
            f"全データ: "
            f"{len(dataset):,}件"
        )

        print(
            f"特徴量: "
            f"{len(feature_columns):,}個"
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

        model = train_model(
            train=train,
            validation=validation,
            feature_columns=feature_columns,
        )

        print(
            f"\n学習反復回数: "
            f"{model.best_iteration_}"
        )

        (
            threshold,
            margin,
            threshold_results,
        ) = select_filter(
            model=model,
            validation=validation,
            feature_columns=feature_columns,
        )

        independent_test = test.iloc[
            ::NON_OVERLAP_STEP
        ].copy()

        evaluate_classification(
            model=model,
            test=independent_test,
            feature_columns=feature_columns,
        )

        final_selected = create_selection(
            model=model,
            rows=independent_test,
            feature_columns=feature_columns,
            confidence_threshold=threshold,
            probability_margin=margin,
        )

        final_metrics = calculate_metrics(
            final_selected
        )

        print_final_metrics(
            final_metrics
        )

        MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            {
                "model": model,
                "features": feature_columns,
                "confidence_threshold": threshold,
                "probability_margin": margin,
                "class_mapping": {
                    TARGET_SELL: "SELL",
                    TARGET_NO_TRADE: "NO_TRADE",
                    TARGET_BUY: "BUY",
                },
                "stop_loss_atr": 1.5,
                "take_profit_atr": 2.0,
                "max_holding_bars": 24,
            },
            MODEL_PATH,
        )

        final_selected.to_csv(
            RESULT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        threshold_results.to_csv(
            THRESHOLD_RESULT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "\nモデル保存先:",
            MODEL_PATH.resolve(),
        )

        print(
            "最終選別結果保存先:",
            RESULT_PATH.resolve(),
        )

        print(
            "閾値比較結果保存先:",
            THRESHOLD_RESULT_PATH.resolve(),
        )

    except Exception as error:
        print(
            "\n売買バリアAIの学習中に"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()