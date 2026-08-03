from pathlib import Path
import sqlite3

import pandas as pd


SYMBOL = "USDJPY"
TIMEFRAME_NAME = "M5"
DATABASE_PATH = Path("data/athena.db")
TRADES_CSV_PATH = Path("data/backtest_trades.csv")

INITIAL_CAPITAL = 10_000.0

# 0.01ロット = 1,000通貨
LOT_SIZE = 0.01
CONTRACT_SIZE = 100_000
TRADE_UNITS = LOT_SIZE * CONTRACT_SIZE

# USDJPYが小数点以下3桁の前提
POINT_SIZE = 0.001

EMA_SHORT_PERIOD = 20
EMA_LONG_PERIOD = 50
RSI_PERIOD = 14
ATR_PERIOD = 14

STOP_LOSS_ATR_MULTIPLIER = 1.0
TAKE_PROFIT_ATR_MULTIPLIER = 1.5


def load_candles() -> pd.DataFrame:
    """SQLiteからローソク足を読み込む。"""
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
        raise RuntimeError("バックテスト対象のローソク足がありません")

    candles["time"] = pd.to_datetime(
        candles["time"],
        unit="s",
    )

    return candles


def calculate_indicators(candles: pd.DataFrame) -> pd.DataFrame:
    """EMA・RSI・ATRを計算する。"""
    result = candles.copy()

    result["ema20"] = result["close"].ewm(
        span=EMA_SHORT_PERIOD,
        adjust=False,
    ).mean()

    result["ema50"] = result["close"].ewm(
        span=EMA_LONG_PERIOD,
        adjust=False,
    ).mean()

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

    relative_strength = average_gain / average_loss

    result["rsi14"] = 100 - (
        100 / (1 + relative_strength)
    )

    previous_close = result["close"].shift(1)

    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    result["atr14"] = true_range.ewm(
        alpha=1 / ATR_PERIOD,
        adjust=False,
        min_periods=ATR_PERIOD,
    ).mean()

    return result


def create_signal(row: pd.Series) -> str:
    """既存ルールからBUY・SELL・HOLDを返す。"""
    if pd.isna(row["rsi14"]) or pd.isna(row["atr14"]):
        return "HOLD"

    if (
        row["ema20"] > row["ema50"]
        and 50 <= row["rsi14"] < 70
    ):
        return "BUY"

    if (
        row["ema20"] < row["ema50"]
        and 30 < row["rsi14"] <= 50
    ):
        return "SELL"

    return "HOLD"


def calculate_profit(
    direction: str,
    entry_price: float,
    exit_price: float,
) -> float:
    """USDJPYの損益を日本円で計算する。"""
    if direction == "BUY":
        price_difference = exit_price - entry_price
    else:
        price_difference = entry_price - exit_price

    return price_difference * TRADE_UNITS


def calculate_max_drawdown(equity_curve: list[float]) -> float:
    """最大ドローダウン率を計算する。"""
    if not equity_curve:
        return 0.0

    equity = pd.Series(equity_curve)
    running_max = equity.cummax()

    drawdown = (
        equity - running_max
    ) / running_max

    return abs(float(drawdown.min())) * 100


def run_backtest(candles: pd.DataFrame) -> pd.DataFrame:
    """バックテストを実行する。"""
    trades: list[dict] = []

    balance = INITIAL_CAPITAL
    equity_curve = [balance]

    index = 0

    while index < len(candles) - 1:
        signal_row = candles.iloc[index]
        signal = create_signal(signal_row)

        if signal == "HOLD":
            index += 1
            continue

        atr = float(signal_row["atr14"])

        if atr <= 0:
            index += 1
            continue

        # シグナル確定後、次の足の始値でエントリーする
        entry_index = index + 1
        entry_row = candles.iloc[entry_index]

        spread_price = (
            float(entry_row["spread"])
            * POINT_SIZE
        )

        if signal == "BUY":
            entry_price = (
                float(entry_row["open"])
                + spread_price
            )

            stop_loss = (
                entry_price
                - atr * STOP_LOSS_ATR_MULTIPLIER
            )

            take_profit = (
                entry_price
                + atr * TAKE_PROFIT_ATR_MULTIPLIER
            )

        else:
            entry_price = float(entry_row["open"])

            stop_loss = (
                entry_price
                + atr * STOP_LOSS_ATR_MULTIPLIER
            )

            take_profit = (
                entry_price
                - atr * TAKE_PROFIT_ATR_MULTIPLIER
            )

        exit_price = None
        exit_reason = None
        exit_index = None

        for check_index in range(
            entry_index,
            len(candles),
        ):
            current = candles.iloc[check_index]

            high = float(current["high"])
            low = float(current["low"])

            current_spread = (
                float(current["spread"])
                * POINT_SIZE
            )

            if signal == "BUY":
                stop_hit = low <= stop_loss
                target_hit = high >= take_profit

                # 同一ローソク足で両方に到達した場合は
                # 保守的に損切りが先だったと扱う
                if stop_hit:
                    exit_price = stop_loss
                    exit_reason = "STOP_LOSS"
                    exit_index = check_index
                    break

                if target_hit:
                    exit_price = take_profit
                    exit_reason = "TAKE_PROFIT"
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
                    exit_reason = "STOP_LOSS"
                    exit_index = check_index
                    break

                if target_hit:
                    exit_price = take_profit
                    exit_reason = "TAKE_PROFIT"
                    exit_index = check_index
                    break

        # 最新足まで決済されなかったポジションは集計しない
        if (
            exit_price is None
            or exit_index is None
            or exit_reason is None
        ):
            break

        profit = calculate_profit(
            direction=signal,
            entry_price=entry_price,
            exit_price=exit_price,
        )

        balance += profit
        equity_curve.append(balance)

        trades.append(
            {
                "direction": signal,
                "signal_time": signal_row["time"],
                "entry_time": entry_row["time"],
                "exit_time": candles.iloc[
                    exit_index
                ]["time"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "exit_reason": exit_reason,
                "profit_jpy": profit,
                "balance_jpy": balance,
                "rsi14": float(signal_row["rsi14"]),
                "ema20": float(signal_row["ema20"]),
                "ema50": float(signal_row["ema50"]),
                "atr14": atr,
            }
        )

        # ポジション保有中の足を飛ばし、
        # 同時に複数ポジションを持たないようにする
        index = exit_index + 1

    result = pd.DataFrame(trades)

    if not result.empty:
        result.attrs["max_drawdown"] = (
            calculate_max_drawdown(equity_curve)
        )

    return result


def print_results(trades: pd.DataFrame) -> None:
    """結果を集計して表示する。"""
    print("\n=== Project Athena バックテスト結果 ===")

    print(f"初期資金: {INITIAL_CAPITAL:,.0f}円")
    print(f"ロット: {LOT_SIZE:.2f}")

    if trades.empty:
        print("成立した取引はありませんでした")
        return

    trade_count = len(trades)
    wins = trades[trades["profit_jpy"] > 0]
    losses = trades[trades["profit_jpy"] < 0]

    win_count = len(wins)
    loss_count = len(losses)

    total_profit = float(
        trades["profit_jpy"].sum()
    )

    final_balance = float(
        trades.iloc[-1]["balance_jpy"]
    )

    return_rate = (
        final_balance / INITIAL_CAPITAL - 1
    ) * 100

    win_rate = (
        win_count / trade_count * 100
    )

    gross_profit = float(
        wins["profit_jpy"].sum()
    )

    gross_loss = abs(
        float(losses["profit_jpy"].sum())
    )

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf")

    max_drawdown = float(
        trades.attrs.get("max_drawdown", 0.0)
    )

    print(f"取引回数: {trade_count}回")
    print(f"勝ち: {win_count}回")
    print(f"負け: {loss_count}回")
    print(f"勝率: {win_rate:.2f}%")
    print(f"合計損益: {total_profit:,.2f}円")
    print(f"最終残高: {final_balance:,.2f}円")
    print(f"利益率: {return_rate:.2f}%")
    print(f"PF: {profit_factor:.2f}")
    print(f"最大DD: {max_drawdown:.2f}%")

    print("\n=== 最新5件の取引 ===")

    columns = [
        "direction",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "exit_reason",
        "profit_jpy",
        "balance_jpy",
    ]

    print(
        trades[columns]
        .tail()
        .to_string(index=False)
    )


def main() -> None:
    try:
        candles = load_candles()
        candles = calculate_indicators(candles)

        print(f"読み込んだローソク足: {len(candles)}本")

        trades = run_backtest(candles)

        TRADES_CSV_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        trades.to_csv(
            TRADES_CSV_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print_results(trades)

        print(
            "\n取引履歴保存先:",
            TRADES_CSV_PATH.resolve(),
        )

    except Exception as error:
        print("バックテスト中にエラーが発生しました")
        print(error)


if __name__ == "__main__":
    main()