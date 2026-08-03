from pathlib import Path
import math
import sqlite3

import pandas as pd


SYMBOL = "USDJPY"
TIMEFRAME_NAME = "M5"

DATABASE_PATH = Path("data/athena.db")
TRADES_CSV_PATH = Path("data/backtest_v3_trades.csv")

INITIAL_CAPITAL = 10_000.0

# 1回の取引で残高の1%まで損失を許容
RISK_PER_TRADE = 0.01

# MT5 Standard口座を想定
CONTRACT_SIZE = 100_000
MIN_LOT = 0.01
MAX_LOT = 1.00
LOT_STEP = 0.01

# USDJPYは通常、小数点以下3桁
POINT_SIZE = 0.001

EMA_SHORT_PERIOD = 20
EMA_LONG_PERIOD = 50
RSI_PERIOD = 14
ATR_PERIOD = 14

STOP_LOSS_ATR_MULTIPLIER = 1.0
TAKE_PROFIT_ATR_MULTIPLIER = 1.5

# スプレッドがATRの25%以上なら見送る
MAX_SPREAD_ATR_RATIO = 0.25

# M5で最大24時間
MAX_HOLDING_BARS = 288


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
        raise RuntimeError("ローソク足データがありません")

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


def create_signal(
    previous: pd.Series,
    current: pd.Series,
) -> str:
    """EMAが交差した瞬間だけシグナルを出す。"""
    if (
        pd.isna(current["rsi14"])
        or pd.isna(current["atr14"])
    ):
        return "HOLD"

    bullish_cross = (
        previous["ema20"] <= previous["ema50"]
        and current["ema20"] > current["ema50"]
    )

    bearish_cross = (
        previous["ema20"] >= previous["ema50"]
        and current["ema20"] < current["ema50"]
    )

    if bullish_cross and 50 <= current["rsi14"] < 70:
        return "BUY"

    if bearish_cross and 30 < current["rsi14"] <= 50:
        return "SELL"

    return "HOLD"


def floor_lot_to_step(lot: float) -> float:
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
    """残高の1%を上限としてロットを計算する。"""
    if balance <= 0 or stop_distance <= 0:
        return 0.0

    risk_amount = balance * RISK_PER_TRADE

    # USDJPYでは、価格差 × 通貨数 = 円損益
    raw_lot = (
        risk_amount
        / (stop_distance * CONTRACT_SIZE)
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
    """USDJPYの損益を円で計算する。"""
    trade_units = lot_size * CONTRACT_SIZE

    if direction == "BUY":
        price_difference = exit_price - entry_price
    else:
        price_difference = entry_price - exit_price

    return price_difference * trade_units


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

    return abs(float(drawdown.min())) * 100


def run_backtest(
    candles: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    float,
    dict[str, int],
]:
    """1%リスク方式でバックテストを実行する。"""
    trades: list[dict] = []

    balance = INITIAL_CAPITAL
    equity_curve = [balance]

    skipped = {
        "spread_too_wide": 0,
        "lot_too_small": 0,
        "invalid_atr": 0,
    }

    index = 1

    while index < len(candles) - 1:
        previous_row = candles.iloc[index - 1]
        signal_row = candles.iloc[index]

        signal = create_signal(
            previous_row,
            signal_row,
        )

        if signal == "HOLD":
            index += 1
            continue

        atr = float(signal_row["atr14"])

        if atr <= 0:
            skipped["invalid_atr"] += 1
            index += 1
            continue

        entry_index = index + 1
        entry_row = candles.iloc[entry_index]

        spread_price = (
            float(entry_row["spread"])
            * POINT_SIZE
        )

        spread_atr_ratio = spread_price / atr

        if spread_atr_ratio > MAX_SPREAD_ATR_RATIO:
            skipped["spread_too_wide"] += 1
            index += 1
            continue

        stop_distance = (
            atr * STOP_LOSS_ATR_MULTIPLIER
        )

        lot_size = calculate_lot_size(
            balance=balance,
            stop_distance=stop_distance,
        )

        if lot_size < MIN_LOT:
            skipped["lot_too_small"] += 1
            index += 1
            continue

        if signal == "BUY":
            entry_price = (
                float(entry_row["open"])
                + spread_price
            )

            stop_loss = (
                entry_price - stop_distance
            )

            take_profit = (
                entry_price
                + atr * TAKE_PROFIT_ATR_MULTIPLIER
            )

        else:
            entry_price = float(entry_row["open"])

            stop_loss = (
                entry_price + stop_distance
            )

            take_profit = (
                entry_price
                - atr * TAKE_PROFIT_ATR_MULTIPLIER
            )

        exit_price = None
        exit_reason = None
        exit_index = None

        final_check_index = min(
            entry_index + MAX_HOLDING_BARS,
            len(candles) - 1,
        )

        for check_index in range(
            entry_index,
            final_check_index + 1,
        ):
            current = candles.iloc[check_index]

            high = float(current["high"])
            low = float(current["low"])
            close = float(current["close"])

            current_spread = (
                float(current["spread"])
                * POINT_SIZE
            )

            if signal == "BUY":
                stop_hit = low <= stop_loss
                target_hit = high >= take_profit

                # 同一足で両方到達した場合は
                # 保守的に損切りを先とする
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

            if check_index == final_check_index:
                if signal == "BUY":
                    exit_price = close
                else:
                    exit_price = close + current_spread

                exit_reason = "TIME_EXIT"
                exit_index = check_index

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
            lot_size=lot_size,
        )

        risk_amount = (
            stop_distance
            * lot_size
            * CONTRACT_SIZE
        )

        balance += profit

        if balance < 0:
            balance = 0.0

        equity_curve.append(balance)

        trades.append(
            {
                "direction": signal,
                "signal_time": signal_row["time"],
                "entry_time": entry_row["time"],
                "exit_time": candles.iloc[
                    exit_index
                ]["time"],
                "lot_size": lot_size,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "exit_reason": exit_reason,
                "risk_amount_jpy": risk_amount,
                "profit_jpy": profit,
                "balance_jpy": balance,
                "spread_atr_ratio": spread_atr_ratio,
                "rsi14": float(signal_row["rsi14"]),
                "ema20": float(signal_row["ema20"]),
                "ema50": float(signal_row["ema50"]),
                "atr14": atr,
            }
        )

        index = exit_index + 1

    result = pd.DataFrame(trades)

    max_drawdown = calculate_max_drawdown(
        equity_curve
    )

    return result, max_drawdown, skipped


def print_results(
    trades: pd.DataFrame,
    max_drawdown: float,
    skipped: dict[str, int],
) -> None:
    """結果を集計して表示する。"""
    print("\n=== Project Athena バックテスト v3 ===")
    print(f"初期資金: {INITIAL_CAPITAL:,.0f}円")
    print(f"リスク上限: 1取引あたり {RISK_PER_TRADE:.0%}")
    print(
        f"ロット範囲: {MIN_LOT:.2f}～{MAX_LOT:.2f}"
    )
    print("条件: EMAクロス発生時のみエントリー")

    if trades.empty:
        print("成立した取引はありませんでした")
        print("\n=== 見送り件数 ===")
        print(skipped)
        return

    trade_count = len(trades)

    wins = trades[
        trades["profit_jpy"] > 0
    ]

    losses = trades[
        trades["profit_jpy"] < 0
    ]

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
        len(wins) / trade_count * 100
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

    average_profit = float(
        trades["profit_jpy"].mean()
    )

    print(f"取引回数: {trade_count}回")
    print(f"勝ち: {len(wins)}回")
    print(f"負け: {len(losses)}回")
    print(f"勝率: {win_rate:.2f}%")
    print(f"合計損益: {total_profit:,.2f}円")
    print(f"平均損益: {average_profit:,.2f}円")
    print(f"最終残高: {final_balance:,.2f}円")
    print(f"利益率: {return_rate:.2f}%")
    print(f"PF: {profit_factor:.2f}")
    print(f"最大DD: {max_drawdown:.2f}%")

    print("\n=== 見送り件数 ===")
    print(
        "スプレッド過大:",
        skipped["spread_too_wide"],
    )
    print(
        "最小ロット未満:",
        skipped["lot_too_small"],
    )
    print(
        "ATR不正:",
        skipped["invalid_atr"],
    )

    print("\n=== 最新5件の取引 ===")

    columns = [
        "direction",
        "entry_time",
        "exit_time",
        "lot_size",
        "entry_price",
        "exit_price",
        "exit_reason",
        "risk_amount_jpy",
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

        print(
            f"読み込んだローソク足: {len(candles)}本"
        )

        (
            trades,
            max_drawdown,
            skipped,
        ) = run_backtest(candles)

        TRADES_CSV_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        trades.to_csv(
            TRADES_CSV_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print_results(
            trades,
            max_drawdown,
            skipped,
        )

        print(
            "\n取引履歴保存先:",
            TRADES_CSV_PATH.resolve(),
        )

    except Exception as error:
        print("バックテスト中にエラーが発生しました")
        print(error)


if __name__ == "__main__":
    main()