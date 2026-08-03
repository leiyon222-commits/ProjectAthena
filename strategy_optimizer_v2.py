from itertools import product
from pathlib import Path
import math
import sqlite3

import pandas as pd


SYMBOL = "USDJPY"
TIMEFRAME_NAME = "M5"

DATABASE_PATH = Path("data/athena.db")
RESULTS_CSV_PATH = Path(
    "data/strategy_optimizer_v2_results.csv"
)

INITIAL_CAPITAL = 10_000.0
RISK_PER_TRADE = 0.01

CONTRACT_SIZE = 100_000
MIN_LOT = 0.01
MAX_LOT = 1.00
LOT_STEP = 0.01
POINT_SIZE = 0.001

RSI_PERIOD = 14
ATR_PERIOD = 14
MAX_HOLDING_BARS = 288

# 約5か月で評価し、約1.5か月で検証
# 検証期間を約1.5か月ずつ移動する
TRAIN_BARS = 30_000
TEST_BARS = 10_000
STEP_BARS = 10_000

EMA_COMBINATIONS = [
    (10, 30),
    (20, 50),
    (50, 100),
]

STOP_LOSS_MULTIPLIERS = [
    1.0,
    1.5,
    2.0,
]

TAKE_PROFIT_MULTIPLIERS = [
    1.5,
    2.0,
    3.0,
]

MAX_SPREAD_ATR_RATIOS = [
    0.25,
    0.40,
    0.60,
]

MIN_TEST_TRADES_PER_FOLD = 5


def load_candles() -> pd.DataFrame:
    """SQLiteからローソク足を読み込む。"""
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"データベースが見つかりません: "
            f"{DATABASE_PATH.resolve()}"
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
                tick_volume,
                spread
            FROM candles
            WHERE symbol = ? AND timeframe = ?
            ORDER BY time ASC
            """,
            connection,
            params=(SYMBOL, TIMEFRAME_NAME),
        )

    if candles.empty:
        raise RuntimeError(
            "ローソク足データがありません"
        )

    candles["time"] = pd.to_datetime(
        candles["time"],
        unit="s",
    )

    return candles


def calculate_base_indicators(
    candles: pd.DataFrame,
) -> pd.DataFrame:
    """RSIとATRを計算する。"""
    result = candles.copy()

    price_change = result["close"].diff()

    gain = price_change.clip(lower=0)
    loss = -price_change.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False,
        min_periods=RSI_PERIOD,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False,
        min_periods=RSI_PERIOD,
    ).mean()

    relative_strength = (
        average_gain / average_loss
    )

    result["rsi14"] = 100 - (
        100 / (1 + relative_strength)
    )

    previous_close = result["close"].shift(1)

    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (
                result["high"] - previous_close
            ).abs(),
            (
                result["low"] - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    result["atr14"] = true_range.ewm(
        alpha=1 / ATR_PERIOD,
        adjust=False,
        min_periods=ATR_PERIOD,
    ).mean()

    return result


def add_ema_indicators(
    candles: pd.DataFrame,
    short_period: int,
    long_period: int,
) -> pd.DataFrame:
    """指定期間のEMAを追加する。"""
    result = candles.copy()

    result["ema_short"] = (
        result["close"]
        .ewm(
            span=short_period,
            adjust=False,
        )
        .mean()
    )

    result["ema_long"] = (
        result["close"]
        .ewm(
            span=long_period,
            adjust=False,
        )
        .mean()
    )

    return result


def create_signal(
    previous: pd.Series,
    current: pd.Series,
) -> str:
    """EMAクロスとRSIから判断する。"""
    if (
        pd.isna(current["rsi14"])
        or pd.isna(current["atr14"])
    ):
        return "HOLD"

    bullish_cross = (
        previous["ema_short"]
        <= previous["ema_long"]
        and current["ema_short"]
        > current["ema_long"]
    )

    bearish_cross = (
        previous["ema_short"]
        >= previous["ema_long"]
        and current["ema_short"]
        < current["ema_long"]
    )

    if (
        bullish_cross
        and 50 <= current["rsi14"] < 70
    ):
        return "BUY"

    if (
        bearish_cross
        and 30 < current["rsi14"] <= 50
    ):
        return "SELL"

    return "HOLD"


def floor_lot_to_step(
    lot: float,
) -> float:
    """ロットを0.01単位で切り捨てる。"""
    stepped_lot = (
        math.floor(lot / LOT_STEP)
        * LOT_STEP
    )

    return round(stepped_lot, 2)


def calculate_lot_size(
    balance: float,
    stop_distance: float,
) -> float:
    """残高の1％リスクでロット計算する。"""
    if balance <= 0 or stop_distance <= 0:
        return 0.0

    risk_amount = (
        balance * RISK_PER_TRADE
    )

    raw_lot = (
        risk_amount
        / (
            stop_distance
            * CONTRACT_SIZE
        )
    )

    lot = floor_lot_to_step(raw_lot)

    if lot < MIN_LOT:
        return 0.0

    return min(lot, MAX_LOT)


def calculate_profit(
    direction: str,
    entry_price: float,
    exit_price: float,
    lot_size: float,
) -> float:
    """USDJPYの損益を日本円で計算する。"""
    trade_units = (
        lot_size * CONTRACT_SIZE
    )

    if direction == "BUY":
        difference = (
            exit_price - entry_price
        )
    else:
        difference = (
            entry_price - exit_price
        )

    return difference * trade_units


def calculate_max_drawdown(
    equity_curve: list[float],
) -> float:
    """最大ドローダウン率を計算する。"""
    if not equity_curve:
        return 0.0

    equity = pd.Series(equity_curve)
    running_max = equity.cummax()

    drawdown = (
        equity - running_max
    ) / running_max

    return abs(
        float(drawdown.min())
    ) * 100


def run_backtest(
    candles: pd.DataFrame,
    allowed_entry_start: int,
    stop_loss_multiplier: float,
    take_profit_multiplier: float,
    max_spread_atr_ratio: float,
) -> dict[str, float | int]:
    """
    バックテストを行う。

    allowed_entry_startより前は指標計算用として
    使用するが、取引は開始しない。
    """
    balance = INITIAL_CAPITAL
    equity_curve = [balance]
    profits: list[float] = []

    skipped_spread = 0
    skipped_lot = 0

    index = max(
        allowed_entry_start,
        1,
    )

    while index < len(candles) - 1:
        previous_row = candles.iloc[
            index - 1
        ]
        signal_row = candles.iloc[index]

        signal = create_signal(
            previous_row,
            signal_row,
        )

        if signal == "HOLD":
            index += 1
            continue

        atr = float(
            signal_row["atr14"]
        )

        if pd.isna(atr) or atr <= 0:
            index += 1
            continue

        entry_index = index + 1

        if entry_index >= len(candles):
            break

        entry_row = candles.iloc[
            entry_index
        ]

        spread_price = (
            float(entry_row["spread"])
            * POINT_SIZE
        )

        spread_atr_ratio = (
            spread_price / atr
        )

        if (
            spread_atr_ratio
            > max_spread_atr_ratio
        ):
            skipped_spread += 1
            index += 1
            continue

        stop_distance = (
            atr * stop_loss_multiplier
        )

        lot_size = calculate_lot_size(
            balance=balance,
            stop_distance=stop_distance,
        )

        if lot_size < MIN_LOT:
            skipped_lot += 1
            index += 1
            continue

        if signal == "BUY":
            entry_price = (
                float(entry_row["open"])
                + spread_price
            )

            stop_loss = (
                entry_price
                - stop_distance
            )

            take_profit = (
                entry_price
                + (
                    atr
                    * take_profit_multiplier
                )
            )

        else:
            entry_price = float(
                entry_row["open"]
            )

            stop_loss = (
                entry_price
                + stop_distance
            )

            take_profit = (
                entry_price
                - (
                    atr
                    * take_profit_multiplier
                )
            )

        final_check_index = min(
            entry_index
            + MAX_HOLDING_BARS,
            len(candles) - 1,
        )

        exit_price = None
        exit_index = None

        for check_index in range(
            entry_index,
            final_check_index + 1,
        ):
            current = candles.iloc[
                check_index
            ]

            high = float(
                current["high"]
            )
            low = float(
                current["low"]
            )
            close = float(
                current["close"]
            )

            current_spread = (
                float(current["spread"])
                * POINT_SIZE
            )

            if signal == "BUY":
                stop_hit = (
                    low <= stop_loss
                )
                target_hit = (
                    high >= take_profit
                )

                if stop_hit:
                    exit_price = stop_loss
                    exit_index = check_index
                    break

                if target_hit:
                    exit_price = take_profit
                    exit_index = check_index
                    break

            else:
                stop_hit = (
                    high + current_spread
                    >= stop_loss
                )

                target_hit = (
                    low + current_spread
                    <= take_profit
                )

                if stop_hit:
                    exit_price = stop_loss
                    exit_index = check_index
                    break

                if target_hit:
                    exit_price = take_profit
                    exit_index = check_index
                    break

            if (
                check_index
                == final_check_index
            ):
                if signal == "BUY":
                    exit_price = close
                else:
                    exit_price = (
                        close
                        + current_spread
                    )

                exit_index = check_index

        if (
            exit_price is None
            or exit_index is None
        ):
            break

        profit = calculate_profit(
            direction=signal,
            entry_price=entry_price,
            exit_price=exit_price,
            lot_size=lot_size,
        )

        balance += profit

        if balance < 0:
            balance = 0.0

        profits.append(profit)
        equity_curve.append(balance)

        index = exit_index + 1

    trade_count = len(profits)

    if trade_count == 0:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "return_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "skipped_spread": skipped_spread,
            "skipped_lot": skipped_lot,
        }

    profit_series = pd.Series(
        profits
    )

    wins = profit_series[
        profit_series > 0
    ]

    losses = profit_series[
        profit_series < 0
    ]

    gross_profit = float(
        wins.sum()
    )

    gross_loss = abs(
        float(losses.sum())
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit
            / gross_loss
        )
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    return_rate = (
        balance
        / INITIAL_CAPITAL
        - 1
    ) * 100

    return {
        "trades": trade_count,
        "win_rate": (
            len(wins)
            / trade_count
            * 100
        ),
        "return_rate": return_rate,
        "profit_factor": profit_factor,
        "max_drawdown":
            calculate_max_drawdown(
                equity_curve
            ),
        "skipped_spread":
            skipped_spread,
        "skipped_lot":
            skipped_lot,
    }


def create_folds(
    candle_count: int,
) -> list[dict[str, int]]:
    """ウォークフォワード期間を作成する。"""
    folds: list[dict[str, int]] = []

    train_start = 0
    fold_number = 1

    while True:
        train_end = (
            train_start + TRAIN_BARS
        )

        test_end = (
            train_end + TEST_BARS
        )

        if test_end > candle_count:
            break

        folds.append(
            {
                "fold": fold_number,
                "train_start": train_start,
                "train_end": train_end,
                "test_end": test_end,
            }
        )

        train_start += STEP_BARS
        fold_number += 1

    return folds


def evaluate_combination(
    base_candles: pd.DataFrame,
    folds: list[dict[str, int]],
    ema_short: int,
    ema_long: int,
    sl_multiplier: float,
    tp_multiplier: float,
    spread_limit: float,
) -> list[dict]:
    """1設定を各期間で検証する。"""
    candles = add_ema_indicators(
        base_candles,
        ema_short,
        ema_long,
    )

    results: list[dict] = []

    for fold in folds:
        train_start = fold[
            "train_start"
        ]
        train_end = fold[
            "train_end"
        ]
        test_end = fold[
            "test_end"
        ]

        warmup = max(
            ema_long * 3,
            300,
        )

        train_slice_start = max(
            0,
            train_start - warmup,
        )

        train_candles = candles.iloc[
            train_slice_start:train_end
        ].reset_index(drop=True)

        train_entry_start = (
            train_start
            - train_slice_start
        )

        test_slice_start = max(
            0,
            train_end - warmup,
        )

        test_candles = candles.iloc[
            test_slice_start:test_end
        ].reset_index(drop=True)

        test_entry_start = (
            train_end
            - test_slice_start
        )

        train_result = run_backtest(
            train_candles,
            train_entry_start,
            sl_multiplier,
            tp_multiplier,
            spread_limit,
        )

        test_result = run_backtest(
            test_candles,
            test_entry_start,
            sl_multiplier,
            tp_multiplier,
            spread_limit,
        )

        results.append(
            {
                "fold": fold["fold"],
                "ema_short": ema_short,
                "ema_long": ema_long,
                "sl_atr": sl_multiplier,
                "tp_atr": tp_multiplier,
                "spread_limit": spread_limit,

                "train_trades":
                    train_result["trades"],
                "train_pf":
                    train_result[
                        "profit_factor"
                    ],
                "train_return":
                    train_result[
                        "return_rate"
                    ],
                "train_dd":
                    train_result[
                        "max_drawdown"
                    ],

                "test_trades":
                    test_result["trades"],
                "test_win_rate":
                    test_result[
                        "win_rate"
                    ],
                "test_pf":
                    test_result[
                        "profit_factor"
                    ],
                "test_return":
                    test_result[
                        "return_rate"
                    ],
                "test_dd":
                    test_result[
                        "max_drawdown"
                    ],
            }
        )

    return results


def summarize_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """設定ごとの複数期間成績を集計する。"""
    group_columns = [
        "ema_short",
        "ema_long",
        "sl_atr",
        "tp_atr",
        "spread_limit",
    ]

    summary = (
        results
        .groupby(group_columns)
        .agg(
            folds=("fold", "count"),
            total_test_trades=(
                "test_trades",
                "sum",
            ),
            average_test_trades=(
                "test_trades",
                "mean",
            ),
            median_test_pf=(
                "test_pf",
                "median",
            ),
            average_test_pf=(
                "test_pf",
                "mean",
            ),
            average_test_return=(
                "test_return",
                "mean",
            ),
            median_test_return=(
                "test_return",
                "median",
            ),
            worst_test_return=(
                "test_return",
                "min",
            ),
            average_test_dd=(
                "test_dd",
                "mean",
            ),
            worst_test_dd=(
                "test_dd",
                "max",
            ),
            positive_return_folds=(
                "test_return",
                lambda series: int(
                    (series > 0).sum()
                ),
            ),
            pf_over_one_folds=(
                "test_pf",
                lambda series: int(
                    (series > 1).sum()
                ),
            ),
        )
        .reset_index()
    )

    summary["positive_fold_ratio"] = (
        summary["positive_return_folds"]
        / summary["folds"]
    )

    summary["pf_over_one_ratio"] = (
        summary["pf_over_one_folds"]
        / summary["folds"]
    )

    return summary


def main() -> None:
    try:
        candles = load_candles()
        candles = (
            calculate_base_indicators(
                candles
            )
        )

        folds = create_folds(
            len(candles)
        )

        if not folds:
            print(
                "ウォークフォワード検証に"
                "必要なデータが不足しています。"
            )
            return

        combinations = list(
            product(
                EMA_COMBINATIONS,
                STOP_LOSS_MULTIPLIERS,
                TAKE_PROFIT_MULTIPLIERS,
                MAX_SPREAD_ATR_RATIOS,
            )
        )

        print(
            "=== Project Athena "
            "ウォークフォワード検証 ==="
        )
        print(
            f"ローソク足: "
            f"{len(candles):,}本"
        )
        print(
            f"検証期間数: "
            f"{len(folds)}"
        )
        print(
            f"設定数: "
            f"{len(combinations)}"
        )
        print(
            "処理を開始します。"
            "数分かかる場合があります。\n"
        )

        all_results: list[dict] = []

        for number, combination in enumerate(
            combinations,
            start=1,
        ):
            (
                ema_pair,
                sl_multiplier,
                tp_multiplier,
                spread_limit,
            ) = combination

            ema_short, ema_long = ema_pair

            combination_results = (
                evaluate_combination(
                    base_candles=candles,
                    folds=folds,
                    ema_short=ema_short,
                    ema_long=ema_long,
                    sl_multiplier=
                        sl_multiplier,
                    tp_multiplier=
                        tp_multiplier,
                    spread_limit=
                        spread_limit,
                )
            )

            all_results.extend(
                combination_results
            )

            test_pfs = [
                row["test_pf"]
                for row
                in combination_results
            ]

            median_pf = float(
                pd.Series(test_pfs).median()
            )

            print(
                f"[{number:02d}/"
                f"{len(combinations)}] "
                f"EMA {ema_short}/"
                f"{ema_long} "
                f"SL {sl_multiplier:.1f} "
                f"TP {tp_multiplier:.1f} "
                f"Spread "
                f"{spread_limit:.0%} "
                f"Median Test PF "
                f"{median_pf:.2f}"
            )

        result_frame = pd.DataFrame(
            all_results
        )

        summary = summarize_results(
            result_frame
        )

        valid_summary = summary[
            summary["total_test_trades"]
            >= (
                MIN_TEST_TRADES_PER_FOLD
                * summary["folds"]
            )
        ].copy()

        if valid_summary.empty:
            print(
                "\n必要な取引回数を満たす"
                "設定がありませんでした。"
            )
            return

        valid_summary = (
            valid_summary.sort_values(
                by=[
                    "positive_fold_ratio",
                    "pf_over_one_ratio",
                    "median_test_pf",
                    "average_test_return",
                    "worst_test_dd",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    False,
                    True,
                ],
            )
        )

        RESULTS_CSV_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        valid_summary.to_csv(
            RESULTS_CSV_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "\n=== 上位10設定 ==="
        )

        display_columns = [
            "ema_short",
            "ema_long",
            "sl_atr",
            "tp_atr",
            "spread_limit",
            "folds",
            "total_test_trades",
            "positive_fold_ratio",
            "pf_over_one_ratio",
            "median_test_pf",
            "average_test_return",
            "worst_test_return",
            "worst_test_dd",
        ]

        print(
            valid_summary[
                display_columns
            ]
            .head(10)
            .to_string(index=False)
        )

        best = valid_summary.iloc[0]

        print(
            "\n=== 最も安定した候補 ==="
        )
        print(
            f"EMA: "
            f"{int(best['ema_short'])}"
            f" / "
            f"{int(best['ema_long'])}"
        )
        print(
            f"SL: ATR × "
            f"{best['sl_atr']:.1f}"
        )
        print(
            f"TP: ATR × "
            f"{best['tp_atr']:.1f}"
        )
        print(
            "スプレッド上限: ATRの"
            f"{best['spread_limit']:.0%}"
        )
        print(
            f"検証期間数: "
            f"{int(best['folds'])}"
        )
        print(
            "利益プラス期間率: "
            f"{best['positive_fold_ratio']:.0%}"
        )
        print(
            "PF1超期間率: "
            f"{best['pf_over_one_ratio']:.0%}"
        )
        print(
            f"PF中央値: "
            f"{best['median_test_pf']:.2f}"
        )
        print(
            "平均利益率: "
            f"{best['average_test_return']:.2f}%"
        )
        print(
            "最悪期間利益率: "
            f"{best['worst_test_return']:.2f}%"
        )
        print(
            f"最悪最大DD: "
            f"{best['worst_test_dd']:.2f}%"
        )

        stable = valid_summary[
            (
                valid_summary[
                    "positive_fold_ratio"
                ] >= 0.60
            )
            & (
                valid_summary[
                    "pf_over_one_ratio"
                ] >= 0.60
            )
            & (
                valid_summary[
                    "median_test_pf"
                ] > 1.0
            )
        ]

        if stable.empty:
            print(
                "\n複数期間で安定して"
                "PF1.0を超える設定は"
                "見つかりませんでした。"
            )
        else:
            print(
                "\n安定候補が見つかりました。"
            )

        print(
            "\n集計結果保存先:",
            RESULTS_CSV_PATH.resolve(),
        )

    except Exception as error:
        print(
            "ウォークフォワード検証中に"
            "エラーが発生しました"
        )
        print(error)


if __name__ == "__main__":
    main()