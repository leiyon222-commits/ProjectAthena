from pathlib import Path
import sqlite3

import pandas as pd


DATABASE_PATH = Path("data/athena.db")
PREDICTIONS_PATH = Path(
    "data/athena_market_ai_3class_predictions.csv"
)
TRADES_PATH = Path(
    "data/athena_ai_paper_backtest_trades.csv"
)

SYMBOL = "USDJPY"
TIMEFRAME_NAME = "M5"

INITIAL_CAPITAL = 10_000.0
LOT_SIZE = 0.01
CONTRACT_SIZE = 100_000
POINT_SIZE = 0.001

HOLDING_BARS = 12

TARGET_DOWN = 0
TARGET_NEUTRAL = 1
TARGET_UP = 2


def load_candles() -> pd.DataFrame:
    """SQLiteから過去のローソク足を取得する。"""
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

    return candles


def load_predictions() -> pd.DataFrame:
    """3分類AIの完全未使用期間の予測を読み込む。"""
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"予測結果がありません: "
            f"{PREDICTIONS_PATH.resolve()}"
        )

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["time"],
    )

    predictions = predictions[
        predictions["selected"] == True
    ].copy()

    predictions = predictions[
        predictions["predicted_class"].isin(
            [TARGET_DOWN, TARGET_UP]
        )
    ].copy()

    predictions = predictions.sort_values(
        "time"
    ).reset_index(drop=True)

    return predictions


def calculate_profit(
    direction: str,
    entry_price: float,
    exit_price: float,
) -> float:
    """USDJPY、0.01ロットの円損益を計算する。"""
    units = LOT_SIZE * CONTRACT_SIZE

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

    equity = pd.Series(balances)
    running_max = equity.cummax()

    drawdown = (
        equity - running_max
    ) / running_max

    return abs(float(drawdown.min())) * 100


def run_backtest(
    candles: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """AIが選んだ場面だけ仮想売買する。"""
    candle_index = {
        timestamp: index
        for index, timestamp
        in enumerate(candles["time"])
    }

    trades: list[dict] = []

    balance = INITIAL_CAPITAL
    balance_history = [balance]

    last_exit_index = -1

    for prediction in predictions.itertuples(
        index=False
    ):
        signal_time = prediction.time

        if signal_time not in candle_index:
            continue

        signal_index = candle_index[
            signal_time
        ]

        entry_index = signal_index + 1
        exit_index = (
            entry_index + HOLDING_BARS
        )

        if exit_index >= len(candles):
            continue

        # 同時保有を避ける
        if entry_index <= last_exit_index:
            continue

        entry_row = candles.iloc[
            entry_index
        ]

        exit_row = candles.iloc[
            exit_index
        ]

        entry_spread = (
            float(entry_row["spread"])
            * POINT_SIZE
        )

        exit_spread = (
            float(exit_row["spread"])
            * POINT_SIZE
        )

        if (
            int(prediction.predicted_class)
            == TARGET_UP
        ):
            direction = "UP"

            # 買いはAskで入り、Bidで決済
            entry_price = (
                float(entry_row["open"])
                + entry_spread
            )

            exit_price = float(
                exit_row["close"]
            )

        elif (
            int(prediction.predicted_class)
            == TARGET_DOWN
        ):
            direction = "DOWN"

            # 売りはBidで入り、Askで決済
            entry_price = float(
                entry_row["open"]
            )

            exit_price = (
                float(exit_row["close"])
                + exit_spread
            )

        else:
            continue

        profit = calculate_profit(
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
        )

        balance += profit
        balance_history.append(balance)

        trades.append(
            {
                "signal_time": signal_time,
                "entry_time": entry_row["time"],
                "exit_time": exit_row["time"],
                "direction": direction,
                "confidence": float(
                    prediction.confidence
                ),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "profit_jpy": profit,
                "balance_jpy": balance,
            }
        )

        last_exit_index = exit_index

        if balance <= 0:
            break

    max_drawdown = calculate_max_drawdown(
        balance_history
    )

    return pd.DataFrame(trades), max_drawdown


def print_results(
    trades: pd.DataFrame,
    max_drawdown: float,
) -> None:
    """仮想売買結果を表示する。"""
    print(
        "\n=== Athena AI 仮想売買結果 ==="
    )

    print(
        f"初期資金: "
        f"{INITIAL_CAPITAL:,.0f}円"
    )

    print(
        f"ロット: "
        f"{LOT_SIZE:.2f}"
    )

    print(
        f"保有時間: "
        f"M5 × {HOLDING_BARS}本"
    )

    if trades.empty:
        print(
            "成立した仮想取引はありません"
        )
        return

    trade_count = len(trades)

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

    final_balance = float(
        trades.iloc[-1]["balance_jpy"]
    )

    return_rate = (
        final_balance / INITIAL_CAPITAL
        - 1
    ) * 100

    win_rate = (
        len(wins) / trade_count
        * 100
    )

    print(
        f"取引回数: "
        f"{trade_count}回"
    )

    print(
        f"勝ち: "
        f"{len(wins)}回"
    )

    print(
        f"負け: "
        f"{len(losses)}回"
    )

    print(
        f"勝率: "
        f"{win_rate:.2f}%"
    )

    print(
        f"合計損益: "
        f"{total_profit:,.2f}円"
    )

    print(
        f"最終残高: "
        f"{final_balance:,.2f}円"
    )

    print(
        f"利益率: "
        f"{return_rate:.2f}%"
    )

    print(
        f"PF: "
        f"{profit_factor:.2f}"
    )

    print(
        f"最大DD: "
        f"{max_drawdown:.2f}%"
    )

    print(
        "\n=== 最新5件 ==="
    )

    print(
        trades[
            [
                "entry_time",
                "exit_time",
                "direction",
                "confidence",
                "entry_price",
                "exit_price",
                "profit_jpy",
                "balance_jpy",
            ]
        ]
        .tail()
        .to_string(index=False)
    )


def main() -> None:
    try:
        candles = load_candles()
        predictions = load_predictions()

        print(
            f"読み込んだAI売買候補: "
            f"{len(predictions)}件"
        )

        trades, max_drawdown = (
            run_backtest(
                candles,
                predictions,
            )
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

        print_results(
            trades,
            max_drawdown,
        )

        print(
            "\n取引履歴保存先:",
            TRADES_PATH.resolve(),
        )

    except Exception as error:
        print(
            "AI仮想売買中に"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()