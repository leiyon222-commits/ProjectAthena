from pathlib import Path
import sqlite3

import pandas as pd


SYMBOL = "USDJPY"
TIMEFRAME_NAME = "M5"

DATABASE_PATH = Path("data/athena.db")
OUTPUT_PATH = Path("data/ai_trade_dataset.csv")

POINT_SIZE = 0.001

EMA_SHORT_PERIOD = 50
EMA_LONG_PERIOD = 100
RSI_PERIOD = 14
ATR_PERIOD = 14

STOP_LOSS_ATR_MULTIPLIER = 1.5
TAKE_PROFIT_ATR_MULTIPLIER = 2.0
MAX_SPREAD_ATR_RATIO = 0.40
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
        utc=True,
    )

    return candles


def calculate_indicators(candles: pd.DataFrame) -> pd.DataFrame:
    """AI用の特徴量とテクニカル指標を計算する。"""
    result = candles.copy()

    result["ema50"] = result["close"].ewm(
        span=EMA_SHORT_PERIOD,
        adjust=False,
    ).mean()

    result["ema100"] = result["close"].ewm(
        span=EMA_LONG_PERIOD,
        adjust=False,
    ).mean()

    result["ema_gap"] = (
        result["ema50"] - result["ema100"]
    )

    result["ema_gap_ratio"] = (
        result["ema_gap"] / result["close"]
    )

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

    result["atr_ratio"] = (
        result["atr14"] / result["close"]
    )

    result["spread_price"] = (
        result["spread"] * POINT_SIZE
    )

    result["spread_atr_ratio"] = (
        result["spread_price"] / result["atr14"]
    )

    result["candle_body"] = (
        result["close"] - result["open"]
    )

    result["candle_range"] = (
        result["high"] - result["low"]
    )

    result["body_ratio"] = (
        result["candle_body"].abs()
        / result["candle_range"].replace(0, pd.NA)
    )

    result["return_1"] = result["close"].pct_change(1)
    result["return_3"] = result["close"].pct_change(3)
    result["return_6"] = result["close"].pct_change(6)
    result["return_12"] = result["close"].pct_change(12)

    result["volume_change"] = (
        result["tick_volume"].pct_change()
    )

    result["hour_utc"] = result["time"].dt.hour
    result["day_of_week"] = result["time"].dt.dayofweek

    return result


def create_signal(
    previous: pd.Series,
    current: pd.Series,
) -> str:
    """基準戦略のEMAクロス＋RSI条件を判定する。"""
    if (
        pd.isna(current["rsi14"])
        or pd.isna(current["atr14"])
    ):
        return "HOLD"

    bullish_cross = (
        previous["ema50"] <= previous["ema100"]
        and current["ema50"] > current["ema100"]
    )

    bearish_cross = (
        previous["ema50"] >= previous["ema100"]
        and current["ema50"] < current["ema100"]
    )

    if bullish_cross and 50 <= current["rsi14"] < 70:
        return "BUY"

    if bearish_cross and 30 < current["rsi14"] <= 50:
        return "SELL"

    return "HOLD"


def determine_trade_result(
    candles: pd.DataFrame,
    signal_index: int,
    direction: str,
) -> dict | None:
    """候補取引が勝ちか負けかを判定する。"""
    entry_index = signal_index + 1

    if entry_index >= len(candles):
        return None

    signal_row = candles.iloc[signal_index]
    entry_row = candles.iloc[entry_index]

    atr = float(signal_row["atr14"])

    if pd.isna(atr) or atr <= 0:
        return None

    spread_price = (
        float(entry_row["spread"]) * POINT_SIZE
    )

    spread_atr_ratio = spread_price / atr

    if spread_atr_ratio > MAX_SPREAD_ATR_RATIO:
        return None

    if direction == "BUY":
        entry_price = (
            float(entry_row["open"]) + spread_price
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

    final_index = min(
        entry_index + MAX_HOLDING_BARS,
        len(candles) - 1,
    )

    for check_index in range(
        entry_index,
        final_index + 1,
    ):
        current = candles.iloc[check_index]

        high = float(current["high"])
        low = float(current["low"])
        close = float(current["close"])

        current_spread = (
            float(current["spread"]) * POINT_SIZE
        )

        if direction == "BUY":
            if low <= stop_loss:
                return {
                    "result": 0,
                    "exit_reason": "STOP_LOSS",
                    "holding_bars": (
                        check_index - entry_index + 1
                    ),
                }

            if high >= take_profit:
                return {
                    "result": 1,
                    "exit_reason": "TAKE_PROFIT",
                    "holding_bars": (
                        check_index - entry_index + 1
                    ),
                }

        else:
            if high + current_spread >= stop_loss:
                return {
                    "result": 0,
                    "exit_reason": "STOP_LOSS",
                    "holding_bars": (
                        check_index - entry_index + 1
                    ),
                }

            if low + current_spread <= take_profit:
                return {
                    "result": 1,
                    "exit_reason": "TAKE_PROFIT",
                    "holding_bars": (
                        check_index - entry_index + 1
                    ),
                }

        if check_index == final_index:
            if direction == "BUY":
                profit = close - entry_price
            else:
                profit = (
                    entry_price
                    - (close + current_spread)
                )

            return {
                "result": int(profit > 0),
                "exit_reason": "TIME_EXIT",
                "holding_bars": (
                    check_index - entry_index + 1
                ),
            }

    return None


def build_dataset(candles: pd.DataFrame) -> pd.DataFrame:
    """基準戦略の取引候補からAI用データを作る。"""
    rows: list[dict] = []

    for index in range(1, len(candles) - 1):
        previous = candles.iloc[index - 1]
        current = candles.iloc[index]

        direction = create_signal(
            previous,
            current,
        )

        if direction == "HOLD":
            continue

        trade_result = determine_trade_result(
            candles,
            index,
            direction,
        )

        if trade_result is None:
            continue

        direction_value = (
            1 if direction == "BUY" else -1
        )

        rows.append(
            {
                "signal_time": current["time"],
                "direction": direction,
                "direction_value": direction_value,
                "close": float(current["close"]),
                "ema50": float(current["ema50"]),
                "ema100": float(current["ema100"]),
                "ema_gap": float(current["ema_gap"]),
                "ema_gap_ratio": float(
                    current["ema_gap_ratio"]
                ),
                "rsi14": float(current["rsi14"]),
                "atr14": float(current["atr14"]),
                "atr_ratio": float(
                    current["atr_ratio"]
                ),
                "spread": float(current["spread"]),
                "spread_atr_ratio": float(
                    current["spread_atr_ratio"]
                ),
                "tick_volume": float(
                    current["tick_volume"]
                ),
                "volume_change": float(
                    current["volume_change"]
                ),
                "candle_body": float(
                    current["candle_body"]
                ),
                "candle_range": float(
                    current["candle_range"]
                ),
                "body_ratio": float(
                    current["body_ratio"]
                ),
                "return_1": float(current["return_1"]),
                "return_3": float(current["return_3"]),
                "return_6": float(current["return_6"]),
                "return_12": float(
                    current["return_12"]
                ),
                "hour_utc": int(current["hour_utc"]),
                "day_of_week": int(
                    current["day_of_week"]
                ),
                "holding_bars": int(
                    trade_result["holding_bars"]
                ),
                "exit_reason": trade_result[
                    "exit_reason"
                ],
                "result": int(
                    trade_result["result"]
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    try:
        candles = load_candles()
        candles = calculate_indicators(candles)

        print(
            f"読み込んだローソク足: {len(candles):,}本"
        )

        dataset = build_dataset(candles)

        if dataset.empty:
            print(
                "AI用データを作成できませんでした"
            )
            return

        dataset = dataset.replace(
            [float("inf"), float("-inf")],
            pd.NA,
        ).dropna()

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset.to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        wins = int(dataset["result"].sum())
        losses = len(dataset) - wins
        win_rate = wins / len(dataset) * 100

        print("\n=== AIデータセット作成結果 ===")
        print(f"取引候補数: {len(dataset)}件")
        print(f"勝ち: {wins}件")
        print(f"負け: {losses}件")
        print(f"基準勝率: {win_rate:.2f}%")
        print(
            "期間:",
            dataset.iloc[0]["signal_time"],
            "～",
            dataset.iloc[-1]["signal_time"],
        )
        print("保存先:", OUTPUT_PATH.resolve())

    except Exception as error:
        print(
            "AIデータセット作成中に"
            "エラーが発生しました"
        )
        print(error)


if __name__ == "__main__":
    main()