from pathlib import Path
import math

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


DATASET_PATH = Path(
    "data/market_context_trade_labels.csv"
)

MODEL_PATH = Path(
    "data/athena_trade_barrier_binary_v2.joblib"
)

FINAL_TRADES_PATH = Path(
    "data/athena_trade_barrier_binary_v2_trades.csv"
)

THRESHOLD_RESULTS_PATH = Path(
    "data/athena_trade_barrier_binary_v2_thresholds.csv"
)

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20

# 最大24本先まで使ってラベルを作成しているため、
# 各期間の境界を空ける
GAP_ROWS = 24

# 学習期間内の最後10％を早期終了の監視に使う
EARLY_STOP_RATIO = 0.10

TARGET_BUY_COLUMN = "buy_win"
TARGET_SELL_COLUMN = "sell_win"

# TP 2.0 / SL 1.5
WIN_R = 2.0 / 1.5
LOSS_R = -1.0

# BUYまたはSELLの予測確率が、
# この値以上の場合だけ候補にする
PROBABILITY_THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]

# BUY確率とSELL確率の差
DIRECTION_MARGINS = [
    0.00,
    0.03,
    0.05,
    0.10,
    0.15,
]

# 検証期間で最低限必要な取引数
MIN_VALIDATION_TRADES = 100

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

    dataset[TARGET_BUY_COLUMN] = (
        dataset[TARGET_BUY_COLUMN].astype(int)
    )

    dataset[TARGET_SELL_COLUMN] = (
        dataset[TARGET_SELL_COLUMN].astype(int)
    )

    return dataset


def get_feature_columns(
    dataset: pd.DataFrame,
) -> list[str]:
    """結果列や未来情報を除外する。"""
    feature_columns: list[str] = []

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

    train_pool = dataset.iloc[
        :train_boundary - GAP_ROWS
    ].copy()

    validation = dataset.iloc[
        train_boundary:
        validation_boundary - GAP_ROWS
    ].copy()

    test = dataset.iloc[
        validation_boundary:
    ].copy()

    if (
        train_pool.empty
        or validation.empty
        or test.empty
    ):
        raise RuntimeError(
            "学習・検証・最終テストの分割に失敗しました"
        )

    return train_pool, validation, test


def split_training_pool(
    train_pool: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    学習用データの中だけで、
    本学習と早期終了監視に分割する。

    売買条件を決めるvalidationは、
    モデルの早期終了には使用しない。
    """
    early_stop_start = int(
        len(train_pool)
        * (1 - EARLY_STOP_RATIO)
    )

    fit_train = train_pool.iloc[
        :early_stop_start - GAP_ROWS
    ].copy()

    early_stop = train_pool.iloc[
        early_stop_start:
    ].copy()

    if fit_train.empty or early_stop.empty:
        raise RuntimeError(
            "学習用データの内部分割に失敗しました"
        )

    return fit_train, early_stop


def create_model() -> lgb.LGBMClassifier:
    """BUYまたはSELL専用の二値分類モデルを作る。"""
    return lgb.LGBMClassifier(
        objective="binary",
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


def train_binary_model(
    fit_train: pd.DataFrame,
    early_stop: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> lgb.LGBMClassifier:
    """BUYまたはSELL専用モデルを学習する。"""
    model = create_model()

    model.fit(
        fit_train[feature_columns],
        fit_train[target_column],
        eval_X=early_stop[feature_columns],
        eval_y=early_stop[target_column],
        eval_metric="binary_logloss",
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


def get_win_probabilities(
    model: lgb.LGBMClassifier,
    rows: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    """勝ちクラスの予測確率を取得する。"""
    probabilities = model.predict_proba(
        rows[feature_columns]
    )

    positive_class_index = list(
        model.classes_
    ).index(1)

    return probabilities[
        :,
        positive_class_index,
    ]


def simulate_non_overlapping_trades(
    rows: pd.DataFrame,
    buy_probabilities: np.ndarray,
    sell_probabilities: np.ndarray,
    probability_threshold: float,
    direction_margin: float,
) -> pd.DataFrame:
    """
    全M5足を時系列順に確認する。

    BUY・SELLのうち勝率予測が高い方を選び、
    ポジション保有中は次のシグナルを無視する。
    """
    trades: list[dict] = []

    next_allowed_index = 0

    for index in range(len(rows)):
        if index < next_allowed_index:
            continue

        buy_probability = float(
            buy_probabilities[index]
        )

        sell_probability = float(
            sell_probabilities[index]
        )

        top_probability = max(
            buy_probability,
            sell_probability,
        )

        probability_difference = abs(
            buy_probability
            - sell_probability
        )

        if (
            top_probability
            < probability_threshold
        ):
            continue

        if (
            probability_difference
            < direction_margin
        ):
            continue

        row = rows.iloc[index]

        if buy_probability > sell_probability:
            direction = "BUY"
            actual_win = int(
                row[TARGET_BUY_COLUMN]
            )

            exit_reason = row[
                "buy_exit_reason"
            ]

            holding_bars = int(
                row["buy_holding_bars"]
            )

        elif sell_probability > buy_probability:
            direction = "SELL"
            actual_win = int(
                row[TARGET_SELL_COLUMN]
            )

            exit_reason = row[
                "sell_exit_reason"
            ]

            holding_bars = int(
                row["sell_holding_bars"]
            )

        else:
            continue

        trade_r = (
            WIN_R
            if actual_win == 1
            else LOSS_R
        )

        trades.append(
            {
                "time": row["time"],
                "direction": direction,
                "buy_probability": (
                    buy_probability
                ),
                "sell_probability": (
                    sell_probability
                ),
                "selected_probability": (
                    top_probability
                ),
                "probability_difference": (
                    probability_difference
                ),
                "actual_win": actual_win,
                "exit_reason": exit_reason,
                "holding_bars": holding_bars,
                "trade_r": trade_r,
            }
        )

        # 実際に決済するまで次のシグナルを無視する
        next_allowed_index = (
            index
            + max(1, holding_bars)
        )

    return pd.DataFrame(trades)


def calculate_metrics(
    trades: pd.DataFrame,
) -> dict[str, float | int]:
    """売買結果を集計する。"""
    if trades.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_r": 0.0,
            "average_r": 0.0,
            "average_r_lcb95": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "buy_count": 0,
            "sell_count": 0,
        }

    wins = trades[
        trades["trade_r"] > 0
    ]

    losses = trades[
        trades["trade_r"] < 0
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

    average_r = float(
        trades["trade_r"].mean()
    )

    if len(trades) >= 2:
        standard_deviation = float(
            trades["trade_r"].std(
                ddof=1
            )
        )

        standard_error = (
            standard_deviation
            / math.sqrt(len(trades))
        )

        average_r_lcb95 = (
            average_r
            - 1.96 * standard_error
        )
    else:
        average_r_lcb95 = average_r

    equity = trades[
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

    max_drawdown_r = abs(
        float(drawdown.min())
    )

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (
            len(wins)
            / len(trades)
            * 100
        ),
        "total_r": float(
            trades["trade_r"].sum()
        ),
        "average_r": average_r,
        "average_r_lcb95": (
            average_r_lcb95
        ),
        "profit_factor": profit_factor,
        "max_drawdown_r": max_drawdown_r,
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


def evaluate_binary_model(
    model_name: str,
    model: lgb.LGBMClassifier,
    rows: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> None:
    """BUY・SELL専用モデルの順位性能を表示する。"""
    actual = rows[
        target_column
    ].to_numpy()

    probabilities = get_win_probabilities(
        model=model,
        rows=rows,
        feature_columns=feature_columns,
    )

    base_win_rate = float(
        actual.mean()
    )

    roc_auc = roc_auc_score(
        actual,
        probabilities,
    )

    average_precision = (
        average_precision_score(
            actual,
            probabilities,
        )
    )

    print(
        f"\n--- {model_name}モデル評価 ---"
    )

    print(
        f"元の勝率: "
        f"{base_win_rate:.2%}"
    )

    print(
        f"ROC AUC: "
        f"{roc_auc:.3f}"
    )

    print(
        f"Average Precision: "
        f"{average_precision:.3f}"
    )

    print(
        f"学習反復回数: "
        f"{model.best_iteration_}"
    )


def select_trade_filter(
    buy_model: lgb.LGBMClassifier,
    sell_model: lgb.LGBMClassifier,
    validation: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[float, float, pd.DataFrame]:
    """検証期間だけを使い、売買条件を決定する。"""
    buy_probabilities = get_win_probabilities(
        model=buy_model,
        rows=validation,
        feature_columns=feature_columns,
    )

    sell_probabilities = get_win_probabilities(
        model=sell_model,
        rows=validation,
        feature_columns=feature_columns,
    )

    results: list[dict] = []

    print(
        "\n=== 検証期間で売買条件を選択 ==="
    )

    for threshold in PROBABILITY_THRESHOLDS:
        for margin in DIRECTION_MARGINS:
            trades = (
                simulate_non_overlapping_trades(
                    rows=validation,
                    buy_probabilities=(
                        buy_probabilities
                    ),
                    sell_probabilities=(
                        sell_probabilities
                    ),
                    probability_threshold=(
                        threshold
                    ),
                    direction_margin=margin,
                )
            )

            metrics = calculate_metrics(
                trades
            )

            results.append(
                {
                    "threshold": threshold,
                    "margin": margin,
                    **metrics,
                }
            )

            print(
                f"確率 {threshold:.0%} / "
                f"方向差 {margin:.0%}: "
                f"{metrics['trades']:,}回 / "
                f"勝率 "
                f"{metrics['win_rate']:.2f}% / "
                f"平均 "
                f"{metrics['average_r']:.4f}R / "
                f"PF "
                f"{metrics['profit_factor']:.2f}"
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
        result_frame.to_csv(
            THRESHOLD_RESULTS_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        raise RuntimeError(
            "検証期間で最低100回を満たし、"
            "平均Rプラス・PF1超となる"
            "売買条件は見つかりませんでした。"
            "最終テストは確認しません。"
        )

    # 条件を多数試したことによる過大評価を抑えるため、
    # 平均Rの95％下限が高い条件を優先する
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

    best = eligible.iloc[0]

    print(
        "\n=== 採用する売買条件 ==="
    )

    print(
        f"最低勝率予測: "
        f"{best['threshold']:.0%}"
    )

    print(
        f"BUY・SELL確率差: "
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
        f"検証平均Rの95％下限: "
        f"{best['average_r_lcb95']:.4f}R"
    )

    print(
        f"検証PF: "
        f"{best['profit_factor']:.2f}"
    )

    print(
        f"検証最大DD: "
        f"{best['max_drawdown_r']:.2f}R"
    )

    print(
        f"検証BUY: "
        f"{int(best['buy_count']):,}回"
    )

    print(
        f"検証SELL: "
        f"{int(best['sell_count']):,}回"
    )

    return (
        float(best["threshold"]),
        float(best["margin"]),
        result_frame,
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
        f"平均Rの95％下限: "
        f"{metrics['average_r_lcb95']:.4f}R"
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

        feature_columns = get_feature_columns(
            dataset
        )

        (
            train_pool,
            validation,
            test,
        ) = split_dataset(dataset)

        (
            fit_train,
            early_stop,
        ) = split_training_pool(
            train_pool
        )

        print(
            "=== Athena BUY・SELL専用AI v2 ==="
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
            f"本学習: "
            f"{len(fit_train):,}件"
        )

        print(
            f"早期終了監視: "
            f"{len(early_stop):,}件"
        )

        print(
            f"売買条件検証: "
            f"{len(validation):,}件"
        )

        print(
            f"完全未使用テスト: "
            f"{len(test):,}件"
        )

        print(
            "本学習期間:",
            fit_train.iloc[0]["time"],
            "～",
            fit_train.iloc[-1]["time"],
        )

        print(
            "売買条件検証期間:",
            validation.iloc[0]["time"],
            "～",
            validation.iloc[-1]["time"],
        )

        print(
            "完全未使用期間:",
            test.iloc[0]["time"],
            "～",
            test.iloc[-1]["time"],
        )

        print(
            "\nBUY専用AIを学習中..."
        )

        buy_model = train_binary_model(
            fit_train=fit_train,
            early_stop=early_stop,
            feature_columns=feature_columns,
            target_column=TARGET_BUY_COLUMN,
        )

        print(
            "SELL専用AIを学習中..."
        )

        sell_model = train_binary_model(
            fit_train=fit_train,
            early_stop=early_stop,
            feature_columns=feature_columns,
            target_column=TARGET_SELL_COLUMN,
        )

        evaluate_binary_model(
            model_name="BUY",
            model=buy_model,
            rows=validation,
            feature_columns=feature_columns,
            target_column=TARGET_BUY_COLUMN,
        )

        evaluate_binary_model(
            model_name="SELL",
            model=sell_model,
            rows=validation,
            feature_columns=feature_columns,
            target_column=TARGET_SELL_COLUMN,
        )

        (
            probability_threshold,
            direction_margin,
            threshold_results,
        ) = select_trade_filter(
            buy_model=buy_model,
            sell_model=sell_model,
            validation=validation,
            feature_columns=feature_columns,
        )

        # 条件決定後に、完全未使用期間を一度だけ評価
        evaluate_binary_model(
            model_name="BUY・最終",
            model=buy_model,
            rows=test,
            feature_columns=feature_columns,
            target_column=TARGET_BUY_COLUMN,
        )

        evaluate_binary_model(
            model_name="SELL・最終",
            model=sell_model,
            rows=test,
            feature_columns=feature_columns,
            target_column=TARGET_SELL_COLUMN,
        )

        test_buy_probabilities = (
            get_win_probabilities(
                model=buy_model,
                rows=test,
                feature_columns=feature_columns,
            )
        )

        test_sell_probabilities = (
            get_win_probabilities(
                model=sell_model,
                rows=test,
                feature_columns=feature_columns,
            )
        )

        final_trades = (
            simulate_non_overlapping_trades(
                rows=test,
                buy_probabilities=(
                    test_buy_probabilities
                ),
                sell_probabilities=(
                    test_sell_probabilities
                ),
                probability_threshold=(
                    probability_threshold
                ),
                direction_margin=(
                    direction_margin
                ),
            )
        )

        final_metrics = calculate_metrics(
            final_trades
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
                "buy_model": buy_model,
                "sell_model": sell_model,
                "features": feature_columns,
                "probability_threshold": (
                    probability_threshold
                ),
                "direction_margin": (
                    direction_margin
                ),
                "stop_loss_atr": 1.5,
                "take_profit_atr": 2.0,
                "max_holding_bars": 24,
            },
            MODEL_PATH,
        )

        final_trades.to_csv(
            FINAL_TRADES_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        threshold_results.to_csv(
            THRESHOLD_RESULTS_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "\nモデル保存先:",
            MODEL_PATH.resolve(),
        )

        print(
            "最終取引履歴保存先:",
            FINAL_TRADES_PATH.resolve(),
        )

        print(
            "条件比較結果保存先:",
            THRESHOLD_RESULTS_PATH.resolve(),
        )

    except Exception as error:
        print(
            "\nBUY・SELL専用AI v2で"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()