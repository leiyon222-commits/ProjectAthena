from pathlib import Path
import math
import sqlite3

import pandas as pd


DATABASE_PATH = Path("data/athena.db")
PREDICTIONS_PATH = Path(
    "data/athena_market_ai_3class_predictions.csv"
)
TRADES_PATH = Path(
    "data/athena_ai_paper_backtest_v2_trades.csv"
)

SYMBOL = "USDJPY"
TIMEFRAME_NAME = "M5"

INITIAL_CAPITAL = 10_000.0
RISK_PER_TRADE = 0.01

CONTRACT_SIZE = 100_000
MIN_LOT = 0.01
MAX_LOT = 1.00
LOT_STEP = 0.01

POINT_SIZE = 0.001

ATR_PERIOD = 14
STOP_LOSS_ATR_MULTIPLIER = 1.5
TAKE_PROFIT_ATR_MULTIPLIER = 2.0
MAX_SPREAD_ATR_RATIO = 0.40

MAX_HOLDING_BARS = 12
MAX_ELAPSED_MINUTES = 90

TARGET_DOWN = 0
TARGET_UP = 2


def load_candles() -> pd.DataFrame:
    """SQLiteからローソク足を読み込む。"""
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"データベースがありません: "
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
        utc=True,
    )

    return calculate_atr(candles)


def calculate_atr(
    candles: pd.DataFrame,
) -> pd.DataFrame:
    """過去のローソク足だけでATRを計算する。"""
    result = candles.copy()

    previous_close = result["close"].shift(1)

    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (
                result["high"]
                - previous_close
            ).abs(),
            (
                result["low"]
                - previous_close
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


def load_predictions() -> pd.DataFrame:
    """AIが高確信度で選んだ予測だけ読み込む。"""
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"予測結果がありません: "
            f"{PREDICTIONS_PATH.resolve()}"
        )

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["time"],
    )

    selected = predictions[
        predictions["selected"].astype(str).str.lower()
        == "true"
    ].copy()

    selected = selected[
        selected["predicted_class"].isin(
            [TARGET_DOWN, TARGET_UP]
        )
    ].copy()

    return selected.sort_values(
        "time"
    ).reset_index(drop=True)


def floor_lot_to_step(
    lot: float,
) -> float:
    """ロットを0.01単位で切り捨てる。"""
    stepped = (
        math.floor(lot / LOT_STEP)
        * LOT_STEP
    )

    return round(stepped, 2)


def calculate_lot_size(
    balance: float,
    stop_distance: float,
) -> float:
    """残高の1％リスクになるロットを計算する。"""
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

    lot = floor_lot_to_step(
        raw_lot
    )

    if lot < MIN_LOT:
        return 0.0

    return min(
        lot,
        MAX_LOT,
    )


def calculate_profit(
    direction: str,
    entry_price: float,
    exit_price: float,
    lot_size: float,
) -> float:
    """USDJPYの円損益を計算する。"""
    units = (
        lot_size * CONTRACT_SIZE
    )

    if direction == "UP":
        price_difference = (
            exit_price - entry_price
        )
    else:
        price_difference = (
            entry_price - exit_price
        )

    return price_difference * units


def calculate_max_drawdown(
    balances: list[float],
) -> float:
    """最大ドローダウン率を計算する。"""
    if not balances:
        return 0.0

    equity = pd.Series(
        balances
    )

    running_max = equity.cummax()

    drawdown = (
        equity - running_max
    ) / running_max

    return abs(
        float(drawdown.min())
    ) * 100


def contains_large_time_gap(
    candles: pd.DataFrame,
    entry_index: int,
    exit_index: int,
) -> bool:
    """取引期間中に90分超のデータ欠損があるか確認する。"""
    trade_times = candles.iloc[
        entry_index:
        exit_index + 1
    ]["time"]

    gaps = trade_times.diff()

    return bool(
        (
            gaps
            > pd.Timedelta(
                minutes=MAX_ELAPSED_MINUTES
            )
        ).any()
    )


def run_backtest(
    candles: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    float,
    dict[str, int],
]:
    """安全条件を加えたAI仮想売買を実行する。"""
    candle_index = {
        timestamp: index
        for index, timestamp
        in enumerate(candles["time"])
    }

    trades: list[dict] = []

    balance = INITIAL_CAPITAL
    balance_history = [balance]

    last_exit_index = -1

    skipped = {
        "missing_signal_time": 0,
        "invalid_atr": 0,
        "spread_too_wide": 0,
        "lot_too_small": 0,
        "overlapping_trade": 0,
        "time_gap": 0,
        "no_exit": 0,
    }

    for prediction in predictions.itertuples(
        index=False
    ):
        signal_time = prediction.time

        if signal_time not in candle_index:
            skipped["missing_signal_time"] += 1
            continue

        signal_index = candle_index[
            signal_time
        ]

        entry_index = signal_index + 1

        if entry_index >= len(candles):
            continue

        if entry_index <= last_exit_index:
            skipped["overlapping_trade"] += 1
            continue

        signal_row = candles.iloc[
            signal_index
        ]

        entry_row = candles.iloc[
            entry_index
        ]

        atr = float(
            signal_row["atr14"]
        )

        if pd.isna(atr) or atr <= 0:
            skipped["invalid_atr"] += 1
            continue

        entry_spread = (
            float(entry_row["spread"])
            * POINT_SIZE
        )

        spread_atr_ratio = (
            entry_spread / atr
        )

        if (
            spread_atr_ratio
            > MAX_SPREAD_ATR_RATIO
        ):
            skipped["spread_too_wide"] += 1
            continue

        stop_distance = (
            atr
            * STOP_LOSS_ATR_MULTIPLIER
        )

        lot_size = calculate_lot_size(
            balance=balance,
            stop_distance=stop_distance,
        )

        if lot_size < MIN_LOT:
            skipped["lot_too_small"] += 1
            continue

        predicted_class = int(
            prediction.predicted_class
        )

        if predicted_class == TARGET_UP:
            direction = "UP"

            entry_price = (
                float(entry_row["open"])
                + entry_spread
            )

            stop_loss = (
                entry_price
                - stop_distance
            )

            take_profit = (
                entry_price
                + atr
                * TAKE_PROFIT_ATR_MULTIPLIER
            )

        elif predicted_class == TARGET_DOWN:
            direction = "DOWN"

            entry_price = float(
                entry_row["open"]
            )

            stop_loss = (
                entry_price
                + stop_distance
            )

            take_profit = (
                entry_price
                - atr
                * TAKE_PROFIT_ATR_MULTIPLIER
            )

        else:
            continue

        maximum_exit_index = min(
            entry_index
            + MAX_HOLDING_BARS,
            len(candles) - 1,
        )

        if contains_large_time_gap(
            candles,
            entry_index,
            maximum_exit_index,
        ):
            skipped["time_gap"] += 1
            continue

        exit_price = None
        exit_index = None
        exit_reason = None

        for check_index in range(
            entry_index,
            maximum_exit_index + 1,
        ):
            current = candles.iloc[
                check_index
            ]

            elapsed_time = (
                current["time"]
                - entry_row["time"]
            )

            if (
                elapsed_time
                > pd.Timedelta(
                    minutes=MAX_ELAPSED_MINUTES
                )
            ):
                break

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

            if direction == "UP":
                stop_hit = (
                    low <= stop_loss
                )

                target_hit = (
                    high >= take_profit
                )

                if stop_hit:
                    exit_price = stop_loss
                    exit_index = check_index
                    exit_reason = "STOP_LOSS"
                    break

                if target_hit:
                    exit_price = take_profit
                    exit_index = check_index
                    exit_reason = "TAKE_PROFIT"
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
                    exit_reason = "STOP_LOSS"
                    break

                if target_hit:
                    exit_price = take_profit
                    exit_index = check_index
                    exit_reason = "TAKE_PROFIT"
                    break

            if check_index == maximum_exit_index:
                if direction == "UP":
                    exit_price = close
                else:
                    exit_price = (
                        close + current_spread
                    )

                exit_index = check_index
                exit_reason = "TIME_EXIT"

        if (
            exit_price is None
            or exit_index is None
            or exit_reason is None
        ):
            skipped["no_exit"] += 1
            continue

        profit = calculate_profit(
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            lot_size=lot_size,
        )

        balance += profit

        if balance < 0:
            balance = 0.0

        balance_history.append(
            balance
        )

        trades.append(
            {
                "signal_time": signal_time,
                "entry_time": entry_row["time"],
                "exit_time": candles.iloc[
                    exit_index
                ]["time"],
                "direction": direction,
                "confidence": float(
                    prediction.confidence
                ),
                "lot_size": lot_size,
                "atr14": atr,
                "spread_atr_ratio": (
                    spread_atr_ratio
                ),
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "profit_jpy": profit,
                "balance_jpy": balance,
            }
        )

        last_exit_index = exit_index

        if balance <= 0:
            break

    max_drawdown = (
        calculate_max_drawdown(
            balance_history
        )
    )

    return (
        pd.DataFrame(trades),
        max_drawdown,
        skipped,
    )


def calculate_metrics(
    trades: pd.DataFrame,
) -> dict[str, float | int]:
    """取引成績を集計する。"""
    if trades.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_profit": 0.0,
            "final_balance": INITIAL_CAPITAL,
            "return_rate": 0.0,
            "profit_factor": 0.0,
        }

    wins = trades[
        trades["profit_jpy"] > 0
    ]

    losses = trades[
        trades["profit_jpy"] < 0
    ]

    gross_profit = float(
        wins["profit_jpy"].sum()
    )

    gross_loss = abs(
        float(
            losses["profit_jpy"].sum()
        )
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit / gross_loss
        )
    else:
        profit_factor = float("inf")

    total_profit = float(
        trades["profit_jpy"].sum()
    )

    final_balance = (
        INITIAL_CAPITAL
        + total_profit
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
        "total_profit": total_profit,
        "final_balance": final_balance,
        "return_rate": (
            final_balance
            / INITIAL_CAPITAL
            - 1
        ) * 100,
        "profit_factor": profit_factor,
    }


def print_metrics(
    title: str,
    metrics: dict[str, float | int],
    max_drawdown: float | None = None,
) -> None:
    """集計結果を表示する。"""
    print(f"\n=== {title} ===")

    print(
        f"取引回数: "
        f"{metrics['trades']}回"
    )

    print(
        f"勝ち: "
        f"{metrics['wins']}回"
    )

    print(
        f"負け: "
        f"{metrics['losses']}回"
    )

    print(
        f"勝率: "
        f"{metrics['win_rate']:.2f}%"
    )

    print(
        f"合計損益: "
        f"{metrics['total_profit']:,.2f}円"
    )

    print(
        f"最終残高: "
        f"{metrics['final_balance']:,.2f}円"
    )

    print(
        f"利益率: "
        f"{metrics['return_rate']:.2f}%"
    )

    print(
        f"PF: "
        f"{metrics['profit_factor']:.2f}"
    )

    if max_drawdown is not None:
        print(
            f"最大DD: "
            f"{max_drawdown:.2f}%"
        )


def main() -> None:
    try:
        candles = load_candles()
        predictions = load_predictions()

        print(
            f"読み込んだAI売買候補: "
            f"{len(predictions)}件"
        )

        (
            trades,
            max_drawdown,
            skipped,
        ) = run_backtest(
            candles,
            predictions,
        )

        TRADES_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        trades.to_csv(
            TRADES_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        metrics = calculate_metrics(
            trades
        )

        print_metrics(
            "Athena AI 仮想売買v2",
            metrics,
            max_drawdown,
        )

        if not trades.empty:
            largest_profit_index = (
                trades["profit_jpy"].idxmax()
            )

            without_largest = trades.drop(
                index=largest_profit_index
            ).reset_index(drop=True)

            without_largest_metrics = (
                calculate_metrics(
                    without_largest
                )
            )

            print_metrics(
                "最大利益の1取引を除いた結果",
                without_largest_metrics,
            )

            largest_trade = trades.loc[
                largest_profit_index
            ]

            print(
                "\n=== 最大利益の取引 ==="
            )

            print(
                f"日時: "
                f"{largest_trade['entry_time']}"
                f" → "
                f"{largest_trade['exit_time']}"
            )

            print(
                f"方向: "
                f"{largest_trade['direction']}"
            )

            print(
                f"利益: "
                f"{largest_trade['profit_jpy']:,.2f}円"
            )

        print(
            "\n=== 見送り件数 ==="
        )

        for reason, count in skipped.items():
            print(
                f"{reason}: {count}"
            )

        print(
            "\n取引履歴保存先:",
            TRADES_PATH.resolve(),
        )

    except Exception as error:
        print(
            "AI仮想売買v2で"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()