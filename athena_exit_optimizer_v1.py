from pathlib import Path
import math
import sqlite3

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


DATABASE_PATH = Path("data/athena.db")
DATASET_PATH = Path("data/market_context_trade_labels.csv")
MODEL_PATH = Path("data/athena_trade_barrier_binary_v2.joblib")

VALIDATION_RESULTS_PATH = Path(
    "data/athena_exit_optimizer_validation_results.csv"
)
FINAL_TRADES_PATH = Path(
    "data/athena_exit_optimizer_final_trades.csv"
)
BEST_CONDITION_PATH = Path(
    "data/athena_exit_optimizer_best_condition.csv"
)

SYMBOL = "USDJPY"
TIMEFRAME_NAME = "M5"

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
GAP_ROWS = 24

POINT_SIZE = 0.001
MAX_BAR_GAP_MINUTES = 15
MIN_VALIDATION_TRADES = 50

# 損切り幅：ATRの何倍か
STOP_LOSS_MULTIPLIERS = [
    0.75,
    1.00,
    1.25,
    1.50,
    2.00,
]

# 利確幅：ATRの何倍か
TAKE_PROFIT_MULTIPLIERS = [
    0.75,
    1.00,
    1.50,
    2.00,
    2.50,
    3.00,
    4.00,
]

# M5の本数
# 24本=2時間、48本=4時間、96本=8時間、288本=24時間
MAX_HOLDING_BARS_LIST = [
    24,
    48,
    96,
    288,
]


def load_candles() -> pd.DataFrame:
    """SQLiteからM5ローソク足を読み込む。"""
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"データベースが見つかりません: {DATABASE_PATH.resolve()}"
        )

    with sqlite3.connect(DATABASE_PATH) as connection:
        candles = pd.read_sql_query(
            """
            SELECT
                time,
                open,
                high,
                low,
                close,
                spread
            FROM candles
            WHERE symbol = ? AND timeframe = ?
            ORDER BY time ASC
            """,
            connection,
            params=(SYMBOL, TIMEFRAME_NAME),
        )

    if candles.empty:
        raise RuntimeError("ローソク足データがありません")

    candles["time"] = pd.to_datetime(
        candles["time"],
        unit="s",
        utc=True,
    )

    candles = (
        candles.drop_duplicates(subset=["time"], keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )

    candles["spread"] = (
        pd.to_numeric(candles["spread"], errors="coerce")
        .fillna(0.0)
    )

    return candles


def load_dataset() -> pd.DataFrame:
    """特徴量データを読み込む。"""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"データセットが見つかりません: {DATASET_PATH.resolve()}"
        )

    dataset = pd.read_csv(
        DATASET_PATH,
        parse_dates=["time"],
    )

    dataset = (
        dataset.sort_values("time")
        .drop_duplicates(subset=["time"], keep="last")
        .reset_index(drop=True)
    )

    dataset = dataset.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if "m5_atr14" not in dataset.columns:
        raise RuntimeError("m5_atr14列がありません")

    return dataset


def load_model_bundle() -> dict:
    """保存済みBUY・SELLモデルを読み込む。"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"モデルが見つかりません: {MODEL_PATH.resolve()}"
        )

    bundle = joblib.load(MODEL_PATH)

    required_keys = {
        "buy_model",
        "sell_model",
        "features",
        "probability_threshold",
        "direction_margin",
    }

    missing = required_keys - set(bundle.keys())

    if missing:
        raise RuntimeError(
            f"モデルファイルに必要な情報がありません: {sorted(missing)}"
        )

    return bundle


def split_dataset(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    前回と同じ期間分割を使う。

    中間20％：出口条件の選択
    最新20％：選んだ出口条件の確認
    """
    total_count = len(dataset)

    train_boundary = int(
        total_count * TRAIN_RATIO
    )

    validation_boundary = int(
        total_count
        * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    validation = dataset.iloc[
        train_boundary:
        validation_boundary - GAP_ROWS
    ].copy()

    test = dataset.iloc[
        validation_boundary:
    ].copy()

    if validation.empty or test.empty:
        raise RuntimeError(
            "検証・最終テスト期間の分割に失敗しました"
        )

    return validation, test


def get_positive_probability(
    model: lgb.LGBMClassifier,
    rows: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    """勝ちクラス1の確率を取得する。"""
    probabilities = model.predict_proba(
        rows[feature_columns]
    )

    class_list = list(model.classes_)

    if 1 not in class_list:
        raise RuntimeError(
            "モデルに勝ちクラス1がありません"
        )

    positive_index = class_list.index(1)

    return probabilities[:, positive_index]


def build_signal_candidates(
    rows: pd.DataFrame,
    buy_model: lgb.LGBMClassifier,
    sell_model: lgb.LGBMClassifier,
    feature_columns: list[str],
    probability_threshold: float,
    direction_margin: float,
) -> pd.DataFrame:
    """
    保存済みAIのエントリー条件を固定して、
    BUY・SELL候補を作る。
    """
    buy_probabilities = get_positive_probability(
        buy_model,
        rows,
        feature_columns,
    )

    sell_probabilities = get_positive_probability(
        sell_model,
        rows,
        feature_columns,
    )

    top_probability = np.maximum(
        buy_probabilities,
        sell_probabilities,
    )

    difference = np.abs(
        buy_probabilities - sell_probabilities
    )

    selected = (
        (top_probability >= probability_threshold)
        & (difference >= direction_margin)
        & (buy_probabilities != sell_probabilities)
    )

    candidates = rows.loc[
        selected,
        ["time", "m5_atr14"],
    ].copy()

    candidates["buy_probability"] = (
        buy_probabilities[selected]
    )

    candidates["sell_probability"] = (
        sell_probabilities[selected]
    )

    candidates["selected_probability"] = (
        top_probability[selected]
    )

    candidates["probability_difference"] = (
        difference[selected]
    )

    candidates["direction"] = np.where(
        candidates["buy_probability"]
        > candidates["sell_probability"],
        "BUY",
        "SELL",
    )

    return candidates.reset_index(drop=True)


def simulate_single_trade(
    candles: pd.DataFrame,
    signal_index: int,
    direction: str,
    atr: float,
    stop_loss_multiplier: float,
    take_profit_multiplier: float,
    max_holding_bars: int,
) -> dict | None:
    """
    次のM5足の始値でエントリーする。

    BUY:
      Askでエントリー、Bidで決済

    SELL:
      Bidでエントリー、Askで決済

    同じ足でSLとTPの両方へ到達した場合は、
    保守的にSL扱いとする。
    """
    entry_index = signal_index + 1

    if entry_index >= len(candles):
        return None

    stop_distance = (
        atr * stop_loss_multiplier
    )

    target_distance = (
        atr * take_profit_multiplier
    )

    if (
        not np.isfinite(stop_distance)
        or stop_distance <= 0
        or not np.isfinite(target_distance)
        or target_distance <= 0
    ):
        return None

    final_index = min(
        entry_index + max_holding_bars - 1,
        len(candles) - 1,
    )

    # 週末などの大きな時間欠損をまたぐ取引は除外する
    window_times = candles.loc[
        entry_index:final_index,
        "time",
    ]

    time_gaps = window_times.diff()

    if (
        time_gaps
        > pd.Timedelta(minutes=MAX_BAR_GAP_MINUTES)
    ).any():
        return None

    entry_row = candles.iloc[entry_index]

    entry_spread = (
        float(entry_row["spread"])
        * POINT_SIZE
    )

    if direction == "BUY":
        entry_price = (
            float(entry_row["open"])
            + entry_spread
        )

        stop_loss = (
            entry_price - stop_distance
        )

        take_profit = (
            entry_price + target_distance
        )

    else:
        entry_price = float(
            entry_row["open"]
        )

        stop_loss = (
            entry_price + stop_distance
        )

        take_profit = (
            entry_price - target_distance
        )

    for check_index in range(
        entry_index,
        final_index + 1,
    ):
        row = candles.iloc[check_index]

        bid_high = float(row["high"])
        bid_low = float(row["low"])

        spread = (
            float(row["spread"])
            * POINT_SIZE
        )

        ask_high = bid_high + spread
        ask_low = bid_low + spread

        if direction == "BUY":
            stop_hit = bid_low <= stop_loss
            target_hit = bid_high >= take_profit
        else:
            stop_hit = ask_high >= stop_loss
            target_hit = ask_low <= take_profit

        holding_bars = (
            check_index - entry_index + 1
        )

        if stop_hit and target_hit:
            return {
                "exit_index": check_index,
                "holding_bars": holding_bars,
                "exit_reason": "BOTH_HIT",
                "trade_r": -1.0,
            }

        if stop_hit:
            return {
                "exit_index": check_index,
                "holding_bars": holding_bars,
                "exit_reason": "STOP_LOSS",
                "trade_r": -1.0,
            }

        if target_hit:
            return {
                "exit_index": check_index,
                "holding_bars": holding_bars,
                "exit_reason": "TAKE_PROFIT",
                "trade_r": (
                    take_profit_multiplier
                    / stop_loss_multiplier
                ),
            }

    # 時間切れ時は、最終足の実際の終値で損益を計算する
    final_row = candles.iloc[final_index]

    if direction == "BUY":
        exit_price = float(
            final_row["close"]
        )

        trade_r = (
            exit_price - entry_price
        ) / stop_distance

    else:
        exit_spread = (
            float(final_row["spread"])
            * POINT_SIZE
        )

        exit_price = (
            float(final_row["close"])
            + exit_spread
        )

        trade_r = (
            entry_price - exit_price
        ) / stop_distance

    return {
        "exit_index": final_index,
        "holding_bars": (
            final_index - entry_index + 1
        ),
        "exit_reason": "TIME_EXIT",
        "trade_r": float(trade_r),
    }


def simulate_condition(
    candidates: pd.DataFrame,
    candles: pd.DataFrame,
    candle_index_by_time: dict,
    stop_loss_multiplier: float,
    take_profit_multiplier: float,
    max_holding_bars: int,
) -> pd.DataFrame:
    """1つの出口条件で、重複しない取引を再現する。"""
    trades: list[dict] = []

    next_allowed_signal_index = 0

    for candidate in candidates.itertuples(
        index=False
    ):
        signal_index = candle_index_by_time.get(
            candidate.time
        )

        if signal_index is None:
            continue

        if signal_index < next_allowed_signal_index:
            continue

        result = simulate_single_trade(
            candles=candles,
            signal_index=signal_index,
            direction=candidate.direction,
            atr=float(candidate.m5_atr14),
            stop_loss_multiplier=(
                stop_loss_multiplier
            ),
            take_profit_multiplier=(
                take_profit_multiplier
            ),
            max_holding_bars=(
                max_holding_bars
            ),
        )

        if result is None:
            continue

        exit_index = int(
            result["exit_index"]
        )

        trades.append(
            {
                "signal_time": candidate.time,
                "entry_time": candles.iloc[
                    signal_index + 1
                ]["time"],
                "exit_time": candles.iloc[
                    exit_index
                ]["time"],
                "direction": candidate.direction,
                "buy_probability": (
                    candidate.buy_probability
                ),
                "sell_probability": (
                    candidate.sell_probability
                ),
                "selected_probability": (
                    candidate.selected_probability
                ),
                "probability_difference": (
                    candidate.probability_difference
                ),
                "stop_loss_atr": (
                    stop_loss_multiplier
                ),
                "take_profit_atr": (
                    take_profit_multiplier
                ),
                "max_holding_bars": (
                    max_holding_bars
                ),
                "holding_bars": (
                    result["holding_bars"]
                ),
                "exit_reason": (
                    result["exit_reason"]
                ),
                "trade_r": (
                    result["trade_r"]
                ),
            }
        )

        # 決済足の終値で次の判断が可能
        next_allowed_signal_index = exit_index

    return pd.DataFrame(trades)


def calculate_metrics(
    trades: pd.DataFrame,
) -> dict[str, float | int]:
    """取引成績を集計する。"""
    if trades.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "flats": 0,
            "win_rate": 0.0,
            "total_r": 0.0,
            "average_r": 0.0,
            "average_r_lcb95": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "average_holding_bars": 0.0,
            "take_profit_count": 0,
            "stop_loss_count": 0,
            "time_exit_count": 0,
            "both_hit_count": 0,
            "buy_count": 0,
            "sell_count": 0,
        }

    wins = trades[
        trades["trade_r"] > 0
    ]

    losses = trades[
        trades["trade_r"] < 0
    ]

    flats = trades[
        trades["trade_r"] == 0
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

    average_r = float(
        trades["trade_r"].mean()
    )

    if len(trades) >= 2:
        standard_deviation = float(
            trades["trade_r"].std(ddof=1)
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

    running_max = equity_with_start.cummax()

    drawdown = (
        equity_with_start - running_max
    )

    max_drawdown_r = abs(
        float(drawdown.min())
    )

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "flats": len(flats),
        "win_rate": (
            len(wins) / len(trades) * 100
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
        "average_holding_bars": float(
            trades["holding_bars"].mean()
        ),
        "take_profit_count": int(
            (
                trades["exit_reason"]
                == "TAKE_PROFIT"
            ).sum()
        ),
        "stop_loss_count": int(
            (
                trades["exit_reason"]
                == "STOP_LOSS"
            ).sum()
        ),
        "time_exit_count": int(
            (
                trades["exit_reason"]
                == "TIME_EXIT"
            ).sum()
        ),
        "both_hit_count": int(
            (
                trades["exit_reason"]
                == "BOTH_HIT"
            ).sum()
        ),
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


def optimize_validation(
    candidates: pd.DataFrame,
    candles: pd.DataFrame,
    candle_index_by_time: dict,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """検証期間だけで出口条件を総当たりする。"""
    results: list[dict] = []

    total_combinations = (
        len(STOP_LOSS_MULTIPLIERS)
        * len(TAKE_PROFIT_MULTIPLIERS)
        * len(MAX_HOLDING_BARS_LIST)
    )

    combination_number = 0

    print(
        "\n=== 出口条件の総当たり ==="
    )

    print(
        f"組み合わせ数: "
        f"{total_combinations:,}"
    )

    for stop_loss in STOP_LOSS_MULTIPLIERS:
        for take_profit in TAKE_PROFIT_MULTIPLIERS:
            for max_holding in MAX_HOLDING_BARS_LIST:
                combination_number += 1

                trades = simulate_condition(
                    candidates=candidates,
                    candles=candles,
                    candle_index_by_time=(
                        candle_index_by_time
                    ),
                    stop_loss_multiplier=(
                        stop_loss
                    ),
                    take_profit_multiplier=(
                        take_profit
                    ),
                    max_holding_bars=(
                        max_holding
                    ),
                )

                metrics = calculate_metrics(
                    trades
                )

                results.append(
                    {
                        "stop_loss_atr": (
                            stop_loss
                        ),
                        "take_profit_atr": (
                            take_profit
                        ),
                        "reward_risk_ratio": (
                            take_profit
                            / stop_loss
                        ),
                        "max_holding_bars": (
                            max_holding
                        ),
                        "max_holding_hours": (
                            max_holding * 5 / 60
                        ),
                        **metrics,
                    }
                )

                if (
                    combination_number % 20 == 0
                    or combination_number
                    == total_combinations
                ):
                    print(
                        f"処理中: "
                        f"{combination_number:,} / "
                        f"{total_combinations:,}"
                    )

    result_frame = pd.DataFrame(
        results
    )

    result_frame = result_frame.sort_values(
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
    ).reset_index(drop=True)

    eligible = result_frame[
        result_frame["trades"]
        >= MIN_VALIDATION_TRADES
    ].copy()

    if eligible.empty:
        return result_frame, None

    profitable = eligible[
        (eligible["average_r"] > 0)
        & (eligible["profit_factor"] > 1.0)
    ].copy()

    if profitable.empty:
        return result_frame, None

    best = profitable.iloc[0]

    return result_frame, best


def print_top_conditions(
    results: pd.DataFrame,
) -> None:
    """検証期間の上位10条件を表示する。"""
    display_columns = [
        "stop_loss_atr",
        "take_profit_atr",
        "reward_risk_ratio",
        "max_holding_hours",
        "trades",
        "win_rate",
        "total_r",
        "average_r",
        "average_r_lcb95",
        "profit_factor",
        "max_drawdown_r",
        "time_exit_count",
    ]

    print(
        "\n=== 検証期間 上位10条件 ==="
    )

    print(
        results[
            display_columns
        ].head(10).to_string(
            index=False
        )
    )


def print_metrics(
    title: str,
    metrics: dict[str, float | int],
) -> None:
    """選択条件の成績を表示する。"""
    print(f"\n=== {title} ===")
    print(f"取引回数: {metrics['trades']:,}回")
    print(f"勝率: {metrics['win_rate']:.2f}%")
    print(f"合計R: {metrics['total_r']:.2f}R")
    print(f"平均R: {metrics['average_r']:.4f}R")
    print(
        "平均Rの95％下限: "
        f"{metrics['average_r_lcb95']:.4f}R"
    )
    print(f"PF: {metrics['profit_factor']:.2f}")
    print(
        f"最大DD: "
        f"{metrics['max_drawdown_r']:.2f}R"
    )
    print(
        f"平均保有: "
        f"{metrics['average_holding_bars']:.1f}本"
    )
    print(
        f"利確: "
        f"{metrics['take_profit_count']:,}回"
    )
    print(
        f"損切り: "
        f"{metrics['stop_loss_count']:,}回"
    )
    print(
        f"時間切れ: "
        f"{metrics['time_exit_count']:,}回"
    )
    print(
        f"同一足両方到達: "
        f"{metrics['both_hit_count']:,}回"
    )
    print(f"BUY: {metrics['buy_count']:,}回")
    print(f"SELL: {metrics['sell_count']:,}回")


def main() -> None:
    try:
        candles = load_candles()
        dataset = load_dataset()
        bundle = load_model_bundle()

        feature_columns = bundle["features"]

        missing_features = [
            column
            for column in feature_columns
            if column not in dataset.columns
        ]

        if missing_features:
            raise RuntimeError(
                "データセットにモデル特徴量がありません: "
                f"{missing_features[:10]}"
            )

        validation, test = split_dataset(
            dataset
        )

        probability_threshold = float(
            bundle["probability_threshold"]
        )

        direction_margin = float(
            bundle["direction_margin"]
        )

        print(
            "=== Athena 出口条件オプティマイザー v1 ==="
        )

        print(
            f"ローソク足: "
            f"{len(candles):,}本"
        )

        print(
            f"検証期間: "
            f"{validation.iloc[0]['time']} ～ "
            f"{validation.iloc[-1]['time']}"
        )

        print(
            f"確認期間: "
            f"{test.iloc[0]['time']} ～ "
            f"{test.iloc[-1]['time']}"
        )

        print(
            f"固定エントリー確率: "
            f"{probability_threshold:.0%}"
        )

        print(
            f"固定BUY・SELL確率差: "
            f"{direction_margin:.0%}"
        )

        validation_candidates = (
            build_signal_candidates(
                rows=validation,
                buy_model=bundle["buy_model"],
                sell_model=bundle["sell_model"],
                feature_columns=feature_columns,
                probability_threshold=(
                    probability_threshold
                ),
                direction_margin=(
                    direction_margin
                ),
            )
        )

        test_candidates = (
            build_signal_candidates(
                rows=test,
                buy_model=bundle["buy_model"],
                sell_model=bundle["sell_model"],
                feature_columns=feature_columns,
                probability_threshold=(
                    probability_threshold
                ),
                direction_margin=(
                    direction_margin
                ),
            )
        )

        print(
            f"検証シグナル候補: "
            f"{len(validation_candidates):,}件"
        )

        print(
            f"確認シグナル候補: "
            f"{len(test_candidates):,}件"
        )

        candle_index_by_time = {
            timestamp: index
            for index, timestamp
            in enumerate(candles["time"])
        }

        (
            validation_results,
            best,
        ) = optimize_validation(
            candidates=validation_candidates,
            candles=candles,
            candle_index_by_time=(
                candle_index_by_time
            ),
        )

        VALIDATION_RESULTS_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        validation_results.to_csv(
            VALIDATION_RESULTS_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print_top_conditions(
            validation_results
        )

        if best is None:
            print(
                "\n最低50回を満たし、"
                "平均Rプラス・PF1超となる"
                "出口条件は見つかりませんでした。"
            )

            print(
                "確認期間は評価せず終了します。"
            )

            print(
                "\n検証結果保存先:",
                VALIDATION_RESULTS_PATH.resolve(),
            )

            return

        best_stop_loss = float(
            best["stop_loss_atr"]
        )

        best_take_profit = float(
            best["take_profit_atr"]
        )

        best_holding_bars = int(
            best["max_holding_bars"]
        )

        print(
            "\n=== 採用する出口条件 ==="
        )

        print(
            f"損切り: ATR × "
            f"{best_stop_loss}"
        )

        print(
            f"利確: ATR × "
            f"{best_take_profit}"
        )

        print(
            f"損益比: "
            f"1 : "
            f"{best_take_profit / best_stop_loss:.2f}"
        )

        print(
            f"最大保有: "
            f"{best_holding_bars}本 "
            f"({best_holding_bars * 5 / 60:.1f}時間)"
        )

        validation_best_trades = (
            simulate_condition(
                candidates=(
                    validation_candidates
                ),
                candles=candles,
                candle_index_by_time=(
                    candle_index_by_time
                ),
                stop_loss_multiplier=(
                    best_stop_loss
                ),
                take_profit_multiplier=(
                    best_take_profit
                ),
                max_holding_bars=(
                    best_holding_bars
                ),
            )
        )

        validation_metrics = (
            calculate_metrics(
                validation_best_trades
            )
        )

        print_metrics(
            "検証期間の採用条件",
            validation_metrics,
        )

        final_trades = simulate_condition(
            candidates=test_candidates,
            candles=candles,
            candle_index_by_time=(
                candle_index_by_time
            ),
            stop_loss_multiplier=(
                best_stop_loss
            ),
            take_profit_multiplier=(
                best_take_profit
            ),
            max_holding_bars=(
                best_holding_bars
            ),
        )

        final_metrics = calculate_metrics(
            final_trades
        )

        print_metrics(
            "確認期間の売買結果",
            final_metrics,
        )

        final_trades.to_csv(
            FINAL_TRADES_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        pd.DataFrame(
            [
                {
                    "probability_threshold": (
                        probability_threshold
                    ),
                    "direction_margin": (
                        direction_margin
                    ),
                    "stop_loss_atr": (
                        best_stop_loss
                    ),
                    "take_profit_atr": (
                        best_take_profit
                    ),
                    "reward_risk_ratio": (
                        best_take_profit
                        / best_stop_loss
                    ),
                    "max_holding_bars": (
                        best_holding_bars
                    ),
                    "max_holding_hours": (
                        best_holding_bars
                        * 5 / 60
                    ),
                    **{
                        f"validation_{key}": value
                        for key, value
                        in validation_metrics.items()
                    },
                    **{
                        f"test_{key}": value
                        for key, value
                        in final_metrics.items()
                    },
                }
            ]
        ).to_csv(
            BEST_CONDITION_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "\n検証結果保存先:",
            VALIDATION_RESULTS_PATH.resolve(),
        )

        print(
            "確認期間の取引履歴:",
            FINAL_TRADES_PATH.resolve(),
        )

        print(
            "採用条件の集計:",
            BEST_CONDITION_PATH.resolve(),
        )

        print(
            "\n注意:"
        )

        print(
            "今回は既存AIのエントリー判断を固定し、"
            "出口条件だけを比較した結果です。"
        )

        print(
            "有望な出口条件が見つかった場合でも、"
            "その条件で教師ラベルを作り直して"
            "AIを再学習する必要があります。"
        )

    except Exception as error:
        print(
            "\n出口条件の検証中に"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()
