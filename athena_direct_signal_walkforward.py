from pathlib import Path

import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

# 既に作成したファイルから共通処理を読み込む
from athena_direct_signal_ai_train import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    calculate_metrics,
    create_model,
    create_selection,
    load_dataset,
)


RESULT_PATH = Path(
    "data/athena_direct_signal_walkforward_results.csv"
)

# 前回の検証で有望だった条件を固定する
CONFIDENCE_THRESHOLD = 0.55
PROBABILITY_MARGIN = 0.20

# 最初の40％を初回学習に使用し、
# その後を12％ずつ5期間で検証する
INITIAL_TRAIN_RATIO = 0.40
TEST_RATIO = 0.12
FOLD_COUNT = 5

# 最大保有24本、データは12本ごとなので、
# 学習と検証の境界を少し空ける
GAP_SAMPLES = 3


def calculate_period_metrics(
    selected: pd.DataFrame,
) -> dict[str, float | int]:
    """1つの検証期間の詳細成績を計算する。"""
    metrics = calculate_metrics(selected)

    if selected.empty:
        return {
            **metrics,
            "positive_months": 0,
            "tested_months": 0,
            "positive_month_ratio": 0.0,
        }

    monthly = selected.copy()

    monthly["month"] = (
        monthly["signal_time"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_result = (
        monthly.groupby("month")["trade_r"]
        .sum()
    )

    tested_months = len(monthly_result)

    positive_months = int(
        (monthly_result > 0).sum()
    )

    if tested_months > 0:
        positive_month_ratio = (
            positive_months
            / tested_months
        )
    else:
        positive_month_ratio = 0.0

    return {
        **metrics,
        "positive_months": positive_months,
        "tested_months": tested_months,
        "positive_month_ratio": (
            positive_month_ratio
        ),
    }


def run_fold(
    dataset: pd.DataFrame,
    fold_number: int,
    train_end: int,
    test_start: int,
    test_end: int,
) -> dict:
    """1つのウォークフォワード期間を実行する。"""
    train = dataset.iloc[
        :train_end
    ].copy()

    test = dataset.iloc[
        test_start:test_end
    ].copy()

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

    selected = create_selection(
        model=model,
        rows=test,
        confidence_threshold=(
            CONFIDENCE_THRESHOLD
        ),
        probability_margin=(
            PROBABILITY_MARGIN
        ),
    )

    metrics = calculate_period_metrics(
        selected
    )

    train_start_time = (
        train.iloc[0]["signal_time"]
    )

    train_end_time = (
        train.iloc[-1]["signal_time"]
    )

    test_start_time = (
        test.iloc[0]["signal_time"]
    )

    test_end_time = (
        test.iloc[-1]["signal_time"]
    )

    print(
        f"\n=== Fold {fold_number} ==="
    )

    print(
        "学習期間:",
        train_start_time,
        "～",
        train_end_time,
    )

    print(
        "検証期間:",
        test_start_time,
        "～",
        test_end_time,
    )

    print(
        f"検証データ: "
        f"{len(test):,}件"
    )

    print(
        f"取引回数: "
        f"{metrics['trades']:,}回"
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
        "プラス月:",
        f"{metrics['positive_months']}"
        f"/{metrics['tested_months']}",
    )

    return {
        "fold": fold_number,
        "train_start": train_start_time,
        "train_end": train_end_time,
        "test_start": test_start_time,
        "test_end": test_end_time,
        "test_rows": len(test),
        **metrics,
    }


def print_summary(
    results: pd.DataFrame,
) -> None:
    """全期間をまとめて評価する。"""
    print(
        "\n=== ウォークフォワード総合結果 ==="
    )

    profitable_folds = int(
        (results["total_r"] > 0).sum()
    )

    pf_over_one_folds = int(
        (
            results["profit_factor"] > 1.0
        ).sum()
    )

    total_trades = int(
        results["trades"].sum()
    )

    total_r = float(
        results["total_r"].sum()
    )

    if total_trades > 0:
        weighted_average_r = (
            total_r / total_trades
        )
    else:
        weighted_average_r = 0.0

    total_positive_months = int(
        results["positive_months"].sum()
    )

    total_tested_months = int(
        results["tested_months"].sum()
    )

    if total_tested_months > 0:
        positive_month_ratio = (
            total_positive_months
            / total_tested_months
        )
    else:
        positive_month_ratio = 0.0

    print(
        f"検証期間数: "
        f"{len(results)}"
    )

    print(
        f"利益プラス期間: "
        f"{profitable_folds}"
        f"/{len(results)}"
    )

    print(
        f"PF1超期間: "
        f"{pf_over_one_folds}"
        f"/{len(results)}"
    )

    print(
        f"合計取引回数: "
        f"{total_trades:,}回"
    )

    print(
        f"合計R: "
        f"{total_r:.2f}R"
    )

    print(
        f"1取引平均R: "
        f"{weighted_average_r:.4f}R"
    )

    print(
        f"PF中央値: "
        f"{results['profit_factor'].median():.2f}"
    )

    print(
        f"平均PF: "
        f"{results['profit_factor'].mean():.2f}"
    )

    print(
        f"最悪期間R: "
        f"{results['total_r'].min():.2f}R"
    )

    print(
        f"最大DDの最悪値: "
        f"{results['max_drawdown_r'].max():.2f}R"
    )

    print(
        f"プラス月率: "
        f"{positive_month_ratio:.2%}"
    )

    print(
        "\n=== 判定 ==="
    )

    passed = (
        profitable_folds >= 4
        and pf_over_one_folds >= 4
        and total_trades >= 100
        and weighted_average_r > 0
        and positive_month_ratio >= 0.60
    )

    if passed:
        print(
            "複数期間で一定の優位性が"
            "確認できる候補です。"
        )
        print(
            "次は1万円・1％リスクで"
            "残高推移を検証します。"
        )
    else:
        print(
            "複数期間で安定した優位性は"
            "確認できませんでした。"
        )
        print(
            "このモデルの条件変更を続けず、"
            "特徴量・教師ラベル・対象時間足を"
            "見直す段階です。"
        )


def main() -> None:
    try:
        dataset = load_dataset()

        total_count = len(dataset)

        initial_train_size = int(
            total_count
            * INITIAL_TRAIN_RATIO
        )

        test_size = int(
            total_count
            * TEST_RATIO
        )

        print(
            "=== Athena 直接判断AI "
            "ウォークフォワード検証 ==="
        )

        print(
            f"全データ: "
            f"{total_count:,}件"
        )

        print(
            f"固定確信度: "
            f"{CONFIDENCE_THRESHOLD:.0%}"
        )

        print(
            f"固定確率差: "
            f"{PROBABILITY_MARGIN:.0%}"
        )

        print(
            f"検証期間数: "
            f"{FOLD_COUNT}"
        )

        results: list[dict] = []

        for fold_number in range(
            1,
            FOLD_COUNT + 1,
        ):
            train_end = (
                initial_train_size
                + test_size
                * (fold_number - 1)
            )

            test_start = (
                train_end
                + GAP_SAMPLES
            )

            test_end = min(
                test_start + test_size,
                total_count,
            )

            if test_start >= total_count:
                break

            if test_end <= test_start:
                break

            result = run_fold(
                dataset=dataset,
                fold_number=fold_number,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )

            results.append(result)

        if not results:
            print(
                "検証期間を作成できませんでした"
            )
            return

        result_frame = pd.DataFrame(
            results
        )

        RESULT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        result_frame.to_csv(
            RESULT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print_summary(
            result_frame
        )

        print(
            "\n結果保存先:",
            RESULT_PATH.resolve(),
        )

    except Exception as error:
        print(
            "\nウォークフォワード検証中に"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()