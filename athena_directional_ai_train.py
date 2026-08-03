from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight


DATASET_PATH = Path(
    "data/ai_trade_outcome_dataset.csv"
)

BUY_MODEL_PATH = Path(
    "data/athena_buy_ai.joblib"
)

SELL_MODEL_PATH = Path(
    "data/athena_sell_ai.joblib"
)

FINAL_SELECTION_PATH = Path(
    "data/athena_directional_ai_final_selections.csv"
)

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20

# 12本先までを教師データ作成に使用しているため、
# 学習・検証・テスト期間の境界を空ける
GAP_TIMES = 24

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

MIN_VALIDATION_SELECTIONS = 100

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

TARGET_COLUMN = "result"


def load_dataset() -> pd.DataFrame:
    """売買結果データセットを時刻順に読み込む。"""
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
        ["signal_time", "direction"]
        + FEATURE_COLUMNS
        + [
            TARGET_COLUMN,
            "entry_price",
            "exit_price",
            "stop_loss",
            "take_profit",
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
        [float("inf"), float("-inf")],
        pd.NA,
    ).dropna()

    dataset[TARGET_COLUMN] = (
        dataset[TARGET_COLUMN].astype(int)
    )

    dataset = dataset.sort_values(
        ["signal_time", "direction"]
    ).reset_index(drop=True)

    dataset["r_multiple"] = dataset.apply(
        calculate_r_multiple,
        axis=1,
    )

    return dataset


def calculate_r_multiple(
    row: pd.Series,
) -> float:
    """
    実際の損益を、エントリーから損切りまでの
    距離で割ってRに変換する。
    """
    entry_price = float(
        row["entry_price"]
    )

    exit_price = float(
        row["exit_price"]
    )

    stop_loss = float(
        row["stop_loss"]
    )

    stop_distance = abs(
        entry_price - stop_loss
    )

    if stop_distance <= 0:
        return 0.0

    if row["direction"] == "BUY":
        profit_distance = (
            exit_price - entry_price
        )
    else:
        profit_distance = (
            entry_price - exit_price
        )

    return (
        profit_distance / stop_distance
    )


def split_dataset(
    dataset: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """時刻単位で60％・20％・20％に分割する。"""
    unique_times = (
        dataset["signal_time"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    train_boundary = int(
        len(unique_times) * TRAIN_RATIO
    )

    validation_boundary = int(
        len(unique_times)
        * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    train_end_time = unique_times.iloc[
        train_boundary - 1
    ]

    validation_start_index = min(
        train_boundary + GAP_TIMES,
        len(unique_times) - 1,
    )

    validation_end_time = unique_times.iloc[
        validation_boundary - 1
    ]

    test_start_index = min(
        validation_boundary + GAP_TIMES,
        len(unique_times) - 1,
    )

    validation_start_time = unique_times.iloc[
        validation_start_index
    ]

    test_start_time = unique_times.iloc[
        test_start_index
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
            "学習・検証・最終テストの分割に失敗しました"
        )

    return train, validation, test


def create_model() -> HistGradientBoostingClassifier:
    """BUYまたはSELL専用モデルを作成する。"""
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


def train_model(
    rows: pd.DataFrame,
) -> HistGradientBoostingClassifier:
    """指定方向専用モデルを学習する。"""
    model = create_model()

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=rows[TARGET_COLUMN],
    )

    model.fit(
        rows[FEATURE_COLUMNS],
        rows[TARGET_COLUMN],
        sample_weight=sample_weights,
    )

    return model


def get_win_probabilities(
    model: HistGradientBoostingClassifier,
    rows: pd.DataFrame,
) -> np.ndarray:
    """勝ちクラスの予測確率を取得する。"""
    probabilities = model.predict_proba(
        rows[FEATURE_COLUMNS]
    )

    win_class_index = list(
        model.classes_
    ).index(1)

    return probabilities[
        :,
        win_class_index,
    ]


def evaluate_threshold(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """指定確率以上の候補を集計する。"""
    selected_mask = (
        probabilities >= threshold
    )

    selected = rows.loc[
        selected_mask
    ].copy()

    if selected.empty:
        return {
            "threshold": threshold,
            "count": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_r": 0.0,
            "average_r": 0.0,
            "profit_factor_r": 0.0,
        }

    wins = selected[
        selected["r_multiple"] > 0
    ]

    losses = selected[
        selected["r_multiple"] < 0
    ]

    gross_profit_r = float(
        wins["r_multiple"].sum()
    )

    gross_loss_r = abs(
        float(
            losses["r_multiple"].sum()
        )
    )

    if gross_loss_r > 0:
        profit_factor_r = (
            gross_profit_r
            / gross_loss_r
        )
    else:
        profit_factor_r = float("inf")

    return {
        "threshold": threshold,
        "count": len(selected),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (
            len(wins)
            / len(selected)
            * 100
        ),
        "total_r": float(
            selected["r_multiple"].sum()
        ),
        "average_r": float(
            selected["r_multiple"].mean()
        ),
        "profit_factor_r": profit_factor_r,
    }


def select_threshold(
    direction: str,
    model: HistGradientBoostingClassifier,
    validation_rows: pd.DataFrame,
) -> float:
    """
    検証期間だけを使って、
    BUYまたはSELL専用の採用確率を決める。
    """
    probabilities = get_win_probabilities(
        model,
        validation_rows,
    )

    results: list[dict] = []

    print(
        f"\n=== {direction} 検証期間の確率比較 ==="
    )

    for threshold in PROBABILITY_THRESHOLDS:
        result = evaluate_threshold(
            rows=validation_rows,
            probabilities=probabilities,
            threshold=threshold,
        )

        results.append(result)

        print(
            f"勝ち確率 {threshold:.0%}以上: "
            f"{result['count']:,}件 / "
            f"勝率 {result['win_rate']:.2f}% / "
            f"合計 {result['total_r']:.2f}R / "
            f"平均 {result['average_r']:.4f}R / "
            f"PF {result['profit_factor_r']:.2f}"
        )

    eligible = [
        result
        for result in results
        if (
            result["count"]
            >= MIN_VALIDATION_SELECTIONS
        )
        and (
            result["average_r"] > 0
        )
    ]

    if not eligible:
        print(
            f"\n{direction}は、最低件数を満たし、"
            "平均Rがプラスになる基準がありません。"
        )

        return 1.01

    # 平均Rを優先し、
    # 同率なら件数とPFが高い条件を採用
    best = max(
        eligible,
        key=lambda result: (
            result["average_r"],
            result["profit_factor_r"],
            result["count"],
        ),
    )

    print(
        f"\n{direction}採用確率: "
        f"{best['threshold']:.0%}"
    )

    print(
        f"{direction}検証勝率: "
        f"{best['win_rate']:.2f}%"
    )

    print(
        f"{direction}検証平均R: "
        f"{best['average_r']:.4f}R"
    )

    print(
        f"{direction}検証PF: "
        f"{best['profit_factor_r']:.2f}"
    )

    print(
        f"{direction}検証件数: "
        f"{best['count']:,}件"
    )

    return float(
        best["threshold"]
    )


def evaluate_model_quality(
    direction: str,
    model: HistGradientBoostingClassifier,
    test_rows: pd.DataFrame,
) -> None:
    """方向専用モデル全体の分類性能を表示する。"""
    y_test = test_rows[
        TARGET_COLUMN
    ]

    probabilities = get_win_probabilities(
        model,
        test_rows,
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

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

    print(
        f"\n--- {direction}モデル全体評価 ---"
    )

    print(
        f"候補数: {len(test_rows):,}件"
    )

    print(
        f"元の勝率: "
        f"{y_test.mean():.2%}"
    )

    print(
        f"Precision: "
        f"{precision:.2%}"
    )

    print(
        f"Recall: "
        f"{recall:.2%}"
    )

    print(
        f"ROC AUC: "
        f"{roc_auc:.3f}"
    )

    print(
        f"Average Precision: "
        f"{average_precision:.3f}"
    )


def select_final_rows(
    direction: str,
    model: HistGradientBoostingClassifier,
    test_rows: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """完全未使用期間から採用候補を作る。"""
    if threshold > 1.0:
        return pd.DataFrame()

    probabilities = get_win_probabilities(
        model,
        test_rows,
    )

    selected = test_rows[
        probabilities >= threshold
    ].copy()

    if selected.empty:
        return selected

    selected["win_probability"] = (
        probabilities[
            probabilities >= threshold
        ]
    )

    selected["model_direction"] = direction

    return selected


def combine_directions(
    buy_selected: pd.DataFrame,
    sell_selected: pd.DataFrame,
) -> pd.DataFrame:
    """
    同一時刻でBUYとSELLの両方が採用された場合、
    勝ち確率が高い方だけを残す。
    """
    available = [
        frame
        for frame in [
            buy_selected,
            sell_selected,
        ]
        if not frame.empty
    ]

    if not available:
        return pd.DataFrame()

    combined = pd.concat(
        available,
        ignore_index=True,
    )

    combined = (
        combined.sort_values(
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
        .sort_values(
            "signal_time"
        )
        .reset_index(drop=True)
    )

    return combined


def print_final_results(
    selected: pd.DataFrame,
) -> None:
    """完全未使用期間の選別結果を表示する。"""
    print(
        "\n=== 完全未使用期間の最終選別 ==="
    )

    if selected.empty:
        print(
            "BUY・SELLとも採用候補がありませんでした"
        )
        return

    wins = selected[
        selected["r_multiple"] > 0
    ]

    losses = selected[
        selected["r_multiple"] < 0
    ]

    gross_profit_r = float(
        wins["r_multiple"].sum()
    )

    gross_loss_r = abs(
        float(
            losses["r_multiple"].sum()
        )
    )

    if gross_loss_r > 0:
        profit_factor_r = (
            gross_profit_r
            / gross_loss_r
        )
    else:
        profit_factor_r = float("inf")

    total_r = float(
        selected["r_multiple"].sum()
    )

    average_r = float(
        selected["r_multiple"].mean()
    )

    print(
        f"選別件数: "
        f"{len(selected):,}件"
    )

    print(
        f"勝ち: "
        f"{len(wins):,}件"
    )

    print(
        f"負け: "
        f"{len(losses):,}件"
    )

    print(
        f"選別後勝率: "
        f"{len(wins) / len(selected):.2%}"
    )

    print(
        f"合計R: "
        f"{total_r:.2f}R"
    )

    print(
        f"平均R: "
        f"{average_r:.4f}R"
    )

    print(
        f"R基準PF: "
        f"{profit_factor_r:.2f}"
    )

    print(
        f"BUY採用: "
        f"{int((selected['direction'] == 'BUY').sum()):,}件"
    )

    print(
        f"SELL採用: "
        f"{int((selected['direction'] == 'SELL').sum()):,}件"
    )

    print(
        f"平均予測確率: "
        f"{selected['win_probability'].mean():.2%}"
    )


def save_model(
    model: HistGradientBoostingClassifier,
    threshold: float,
    direction: str,
    path: Path,
) -> None:
    """方向専用モデルを保存する。"""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
            "threshold": threshold,
            "direction": direction,
            "stop_loss_atr": 1.5,
            "take_profit_atr": 2.0,
            "max_holding_bars": 12,
        },
        path,
    )


def main() -> None:
    try:
        dataset = load_dataset()

        train, validation, test = (
            split_dataset(dataset)
        )

        buy_train = train[
            train["direction"] == "BUY"
        ].copy()

        sell_train = train[
            train["direction"] == "SELL"
        ].copy()

        buy_validation = validation[
            validation["direction"] == "BUY"
        ].copy()

        sell_validation = validation[
            validation["direction"] == "SELL"
        ].copy()

        buy_test = test[
            test["direction"] == "BUY"
        ].copy()

        sell_test = test[
            test["direction"] == "SELL"
        ].copy()

        print(
            "=== Athena BUY・SELL専用AI ==="
        )

        print(
            f"全データ: "
            f"{len(dataset):,}件"
        )

        print(
            f"BUY学習: "
            f"{len(buy_train):,}件"
        )

        print(
            f"SELL学習: "
            f"{len(sell_train):,}件"
        )

        print(
            f"BUY検証: "
            f"{len(buy_validation):,}件"
        )

        print(
            f"SELL検証: "
            f"{len(sell_validation):,}件"
        )

        print(
            f"BUY最終テスト: "
            f"{len(buy_test):,}件"
        )

        print(
            f"SELL最終テスト: "
            f"{len(sell_test):,}件"
        )

        buy_model = train_model(
            buy_train
        )

        sell_model = train_model(
            sell_train
        )

        buy_threshold = select_threshold(
            direction="BUY",
            model=buy_model,
            validation_rows=buy_validation,
        )

        sell_threshold = select_threshold(
            direction="SELL",
            model=sell_model,
            validation_rows=sell_validation,
        )

        evaluate_model_quality(
            direction="BUY",
            model=buy_model,
            test_rows=buy_test,
        )

        evaluate_model_quality(
            direction="SELL",
            model=sell_model,
            test_rows=sell_test,
        )

        buy_selected = select_final_rows(
            direction="BUY",
            model=buy_model,
            test_rows=buy_test,
            threshold=buy_threshold,
        )

        sell_selected = select_final_rows(
            direction="SELL",
            model=sell_model,
            test_rows=sell_test,
            threshold=sell_threshold,
        )

        final_selected = combine_directions(
            buy_selected,
            sell_selected,
        )

        print_final_results(
            final_selected
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

        save_model(
            model=buy_model,
            threshold=buy_threshold,
            direction="BUY",
            path=BUY_MODEL_PATH,
        )

        save_model(
            model=sell_model,
            threshold=sell_threshold,
            direction="SELL",
            path=SELL_MODEL_PATH,
        )

        print(
            "\nBUYモデル保存先:",
            BUY_MODEL_PATH.resolve(),
        )

        print(
            "SELLモデル保存先:",
            SELL_MODEL_PATH.resolve(),
        )

        print(
            "最終選別結果保存先:",
            FINAL_SELECTION_PATH.resolve(),
        )

    except Exception as error:
        print(
            "方向別AIの学習中に"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()