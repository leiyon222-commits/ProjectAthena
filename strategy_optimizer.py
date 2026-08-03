from itertools import product
from pathlib import Path
import math
import sqlite3

import pandas as pd


SYMBOL = "USDJPY"
TIMEFRAME_NAME = "M5"

DATABASE_PATH = Path("data/athena.db")
RESULTS_CSV_PATH = Path("data/strategy_optimizer_results.csv")

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

# 自動比較する設定
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

TRAIN_RATIO = 0.75
MIN_TRAIN_TRADES = 20


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


def add_ema_indicators(
    candles: pd.DataFrame,
    short_period: int,
    long_period: int,
) -> pd.DataFrame:
    """指定された期間のEMAを追加する。"""
    result = candles.copy()

    result["ema_short"] = result["close"].ewm(
        span=short_period,
        adjust=False,
    ).mean()

    result["ema_long"] = result["close"].ewm(
        span=long_period,
        adjust=False,
    ).mean()

    return result


def create_signal(
    previous: pd.Series,
    current: pd.Series,
) -> str:
    """EMAクロスとRSIからシグナルを作る。"""
    if (
        pd.isna(current["rsi14"])
        or pd.isna(current["atr14"])
    ):
        return "HOLD"

    bullish_cross = (
        previous["ema_short"] <= previous["ema_long"]
        and current["ema_short"] > current["ema_long"]
    )

    bearish_cross = (
        previous["ema_short"] >= previous["ema_long"]
        and current["ema_short"] < current["ema_long"]
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
    """残高の1％リスクになるロットを計算する。"""
    if balance <= 0 or stop_distance <= 0:
        return 0.0

    risk_amount = balance * RISK_PER_TRADE

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
    stop_loss_multiplier: float,
    take_profit_multiplier: float,
    max_spread_atr_ratio: float,
) -> dict[str, float | int]:
    """指定された設定でバックテストする。"""
    balance = INITIAL_CAPITAL
    equity_curve = [balance]

    profits: list[float] = []

    skipped_spread = 0
    skipped_lot = 0

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

        if pd.isna(atr) or atr <= 0:
            index += 1
            continue

        entry_index = index + 1
        entry_row = candles.iloc[entry_index]

        spread_price = (
            float(entry_row["spread"])
            * POINT_SIZE
        )

        spread_atr_ratio = spread_price / atr

        if spread_atr_ratio > max_spread_atr_ratio:
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
                entry_price - stop_distance
            )

            take_profit = (
                entry_price
                + atr * take_profit_multiplier
            )

        else:
            entry_price = float(entry_row["open"])

            stop_loss = (
                entry_price + stop_distance
            )

            take_profit = (
                entry_price
                - atr * take_profit_multiplier
            )

        exit_price = None
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

                # 同じ足で両方到達した場合は、
                # 保守的に損切りを先とする
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

            if check_index == final_check_index:
                if signal == "BUY":
                    exit_price = close
                else:
                    exit_price = close + current_spread

                exit_index = check_index

        if exit_price is None or exit_index is None:
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
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_profit": 0.0,
            "final_balance": INITIAL_CAPITAL,
            "return_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "skipped_spread": skipped_spread,
            "skipped_lot": skipped_lot,
        }

    profit_series = pd.Series(profits)

    wins = profit_series[profit_series > 0]
    losses = profit_series[profit_series < 0]

    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    final_balance = balance

    return_rate = (
        final_balance / INITIAL_CAPITAL - 1
    ) * 100

    return {
        "trades": trade_count,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / trade_count * 100,
        "total_profit": float(profit_series.sum()),
        "final_balance": final_balance,
        "return_rate": return_rate,
        "profit_factor": profit_factor,
        "max_drawdown": calculate_max_drawdown(
            equity_curve
        ),
        "skipped_spread": skipped_spread,
        "skipped_lot": skipped_lot,
    }


def evaluate_combination(
    base_candles: pd.DataFrame,
    train_end_index: int,
    ema_short: int,
    ema_long: int,
    stop_loss_multiplier: float,
    take_profit_multiplier: float,
    max_spread_atr_ratio: float,
) -> dict:
    """1つの設定を学習期間と検証期間で評価する。"""
    candles = add_ema_indicators(
        base_candles,
        short_period=ema_short,
        long_period=ema_long,
    )

    train_candles = candles.iloc[
        :train_end_index
    ].reset_index(drop=True)

    # 検証期間の先頭でもEMA等を安定させるため、
    # 少し前のデータを含める
    warmup_bars = max(ema_long * 3, 300)

    test_start_index = max(
        0,
        train_end_index - warmup_bars,
    )

    test_candles = candles.iloc[
        test_start_index:
    ].reset_index(drop=True)

    train_result = run_backtest(
        train_candles,
        stop_loss_multiplier,
        take_profit_multiplier,
        max_spread_atr_ratio,
    )

    test_result = run_backtest(
        test_candles,
        stop_loss_multiplier,
        take_profit_multiplier,
        max_spread_atr_ratio,
    )

    return {
        "ema_short": ema_short,
        "ema_long": ema_long,
        "sl_atr": stop_loss_multiplier,
        "tp_atr": take_profit_multiplier,
        "spread_atr_limit": max_spread_atr_ratio,

        "train_trades": train_result["trades"],
        "train_win_rate": train_result["win_rate"],
        "train_return": train_result["return_rate"],
        "train_pf": train_result["profit_factor"],
        "train_max_dd": train_result["max_drawdown"],

        "test_trades": test_result["trades"],
        "test_win_rate": test_result["win_rate"],
        "test_return": test_result["return_rate"],
        "test_pf": test_result["profit_factor"],
        "test_max_dd": test_result["max_drawdown"],
    }


def print_result(
    title: str,
    row: pd.Series,
) -> None:
    """設定と成績を表示する。"""
    print(f"\n=== {title} ===")
    print(
        f"EMA: {int(row['ema_short'])}"
        f" / {int(row['ema_long'])}"
    )
    print(f"SL: ATR × {row['sl_atr']:.1f}")
    print(f"TP: ATR × {row['tp_atr']:.1f}")
    print(
        "スプレッド上限: ATRの"
        f"{row['spread_atr_limit']:.0%}"
    )

    print("\n--- 前半75％ ---")
    print(f"取引回数: {int(row['train_trades'])}")
    print(f"勝率: {row['train_win_rate']:.2f}%")
    print(f"利益率: {row['train_return']:.2f}%")
    print(f"PF: {row['train_pf']:.2f}")
    print(f"最大DD: {row['train_max_dd']:.2f}%")

    print("\n--- 後半25％ ---")
    print(f"取引回数: {int(row['test_trades'])}")
    print(f"勝率: {row['test_win_rate']:.2f}%")
    print(f"利益率: {row['test_return']:.2f}%")
    print(f"PF: {row['test_pf']:.2f}")
    print(f"最大DD: {row['test_max_dd']:.2f}%")


def main() -> None:
    try:
        candles = load_candles()
        candles = calculate_base_indicators(candles)

        train_end_index = int(
            len(candles) * TRAIN_RATIO
        )

        train_end_time = candles.iloc[
            train_end_index - 1
        ]["time"]

        test_start_time = candles.iloc[
            train_end_index
        ]["time"]

        combinations = list(
            product(
                EMA_COMBINATIONS,
                STOP_LOSS_MULTIPLIERS,
                TAKE_PROFIT_MULTIPLIERS,
                MAX_SPREAD_ATR_RATIOS,
            )
        )

        print("=== Project Athena 戦略探索 ===")
        print(f"全ローソク足: {len(candles):,}本")
        print(f"学習期間終了: {train_end_time}")
        print(f"検証期間開始: {test_start_time}")
        print(f"比較する組み合わせ: {len(combinations)}通り")
        print("\n処理を開始します。しばらくお待ちください。")

        results: list[dict] = []

        for number, combination in enumerate(
            combinations,
            start=1,
        ):
            (
                ema_combination,
                stop_loss_multiplier,
                take_profit_multiplier,
                max_spread_atr_ratio,
            ) = combination

            ema_short, ema_long = ema_combination

            result = evaluate_combination(
                base_candles=candles,
                train_end_index=train_end_index,
                ema_short=ema_short,
                ema_long=ema_long,
                stop_loss_multiplier=stop_loss_multiplier,
                take_profit_multiplier=take_profit_multiplier,
                max_spread_atr_ratio=max_spread_atr_ratio,
            )

            results.append(result)

            print(
                f"[{number:02d}/{len(combinations)}] "
                f"EMA {ema_short}/{ema_long} "
                f"SL {stop_loss_multiplier:.1f} "
                f"TP {take_profit_multiplier:.1f} "
                f"Spread {max_spread_atr_ratio:.0%} "
                f"Train PF {result['train_pf']:.2f} "
                f"Test PF {result['test_pf']:.2f}"
            )

        result_frame = pd.DataFrame(results)

        RESULTS_CSV_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        result_frame.to_csv(
            RESULTS_CSV_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        valid_results = result_frame[
            result_frame["train_trades"]
            >= MIN_TRAIN_TRADES
        ].copy()

        if valid_results.empty:
            print(
                "\n最低取引回数を満たす設定がありませんでした。"
            )
            return

        # 前半データのPFを最優先し、
        # 次に利益率、最大DDの順で並べる
        valid_results = valid_results.sort_values(
            by=[
                "train_pf",
                "train_return",
                "train_max_dd",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )

        best_train_result = valid_results.iloc[0]

        print_result(
            "前半75％で最も良かった設定",
            best_train_result,
        )

        stable_results = valid_results[
            (valid_results["train_pf"] > 1.0)
            & (valid_results["test_pf"] > 1.0)
            & (valid_results["train_return"] > 0)
            & (valid_results["test_return"] > 0)
        ].copy()

        if stable_results.empty:
            print(
                "\n前半・後半の両方で"
                "PF1.0超かつ利益がプラスの設定は"
                "見つかりませんでした。"
            )
        else:
            stable_results["combined_score"] = (
                stable_results["train_pf"]
                + stable_results["test_pf"]
            )

            stable_results = stable_results.sort_values(
                by=[
                    "combined_score",
                    "test_return",
                    "test_max_dd",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
            )

            best_stable_result = stable_results.iloc[0]

            print_result(
                "前半・後半の両方で安定した設定",
                best_stable_result,
            )

        print(
            "\n全結果保存先:",
            RESULTS_CSV_PATH.resolve(),
        )

    except Exception as error:
        print("戦略探索中にエラーが発生しました")
        print(error)


if __name__ == "__main__":
    main()