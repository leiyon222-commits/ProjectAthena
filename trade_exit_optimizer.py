from itertools import product
from pathlib import Path
import sqlite3

import pandas as pd


SYMBOL = "USDJPY"
TIMEFRAME_NAME = "M5"

DATABASE_PATH = Path("data/athena.db")
RESULT_PATH = Path("data/trade_exit_optimizer_results.csv")

POINT_SIZE = 0.001
ATR_PERIOD = 14

# 予測期間が重なりすぎないよう12本ごとに候補を作る
SIGNAL_STEP = 12

# 週末など、大きな時間欠損を含む候補は除外
MAX_BAR_GAP_MINUTES = 15

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

MAX_HOLDING_BARS_LIST = [
    12,
    24,
    48,
]

DIRECTIONS = [
    "BUY",
    "SELL",
]


def load_candles() -> pd.DataFrame:
    """SQLiteからM5ローソク足を読み込む。"""
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
    """ATR14を計算する。"""
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


def contains_large_time_gap(
    candles: pd.DataFrame,
    start_index: int,
    end_index: int,
) -> bool:
    """対象期間内に15分を超える時間欠損があるか確認する。"""
    times = candles.iloc[
        start_index:
        end_index + 1
    ]["time"]

    gaps = times.diff()

    return bool(
        (
            gaps
            > pd.Timedelta(
                minutes=MAX_BAR_GAP_MINUTES
            )
        ).any()
    )


def evaluate_trade(
    candles: pd.DataFrame,
    signal_index: int,
    direction: str,
    stop_loss_multiplier: float,
    take_profit_multiplier: float,
    max_holding_bars: int,
) -> dict | None:
    """1つの仮想取引のR損益を計算する。"""
    entry_index = signal_index + 1

    if entry_index >= len(candles):
        return None

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
        return None

    final_index = min(
        entry_index + max_holding_bars,
        len(candles) - 1,
    )

    if contains_large_time_gap(
        candles,
        entry_index,
        final_index,
    ):
        return None

    entry_spread = (
        float(entry_row["spread"])
        * POINT_SIZE
    )

    stop_distance = (
        atr * stop_loss_multiplier
    )

    target_distance = (
        atr * take_profit_multiplier
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

    exit_price = None
    exit_reason = None

    for check_index in range(
        entry_index,
        final_index + 1,
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

        if direction == "BUY":
            stop_hit = (
                low <= stop_loss
            )

            target_hit = (
                high >= take_profit
            )

            # 同じ足でSL・TPの両方へ到達した場合は
            # 保守的に損切りを先とする
            if stop_hit:
                exit_price = stop_loss
                exit_reason = "STOP_LOSS"
                break

            if target_hit:
                exit_price = take_profit
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
                exit_reason = "STOP_LOSS"
                break

            if target_hit:
                exit_price = take_profit
                exit_reason = "TAKE_PROFIT"
                break

        if check_index == final_index:
            if direction == "BUY":
                exit_price = close
            else:
                exit_price = (
                    close + current_spread
                )

            exit_reason = "TIME_EXIT"

    if (
        exit_price is None
        or exit_reason is None
    ):
        return None

    if direction == "BUY":
        profit_distance = (
            exit_price - entry_price
        )
    else:
        profit_distance = (
            entry_price - exit_price
        )

    r_multiple = (
        profit_distance / stop_distance
    )

    return {
        "r_multiple": r_multiple,
        "exit_reason": exit_reason,
    }


def evaluate_combination(
    candles: pd.DataFrame,
    direction: str,
    stop_loss_multiplier: float,
    take_profit_multiplier: float,
    max_holding_bars: int,
) -> dict:
    """1つの出口条件を全期間で評価する。"""
    r_values: list[float] = []

    stop_loss_count = 0
    take_profit_count = 0
    time_exit_count = 0
    skipped_count = 0

    last_signal_index = (
        len(candles)
        - max_holding_bars
        - 2
    )

    for signal_index in range(
        100,
        last_signal_index,
        SIGNAL_STEP,
    ):
        result = evaluate_trade(
            candles=candles,
            signal_index=signal_index,
            direction=direction,
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
            skipped_count += 1
            continue

        r_values.append(
            float(result["r_multiple"])
        )

        if (
            result["exit_reason"]
            == "STOP_LOSS"
        ):
            stop_loss_count += 1

        elif (
            result["exit_reason"]
            == "TAKE_PROFIT"
        ):
            take_profit_count += 1

        else:
            time_exit_count += 1

    if not r_values:
        return {
            "direction": direction,
            "sl_atr": stop_loss_multiplier,
            "tp_atr": take_profit_multiplier,
            "holding_bars": max_holding_bars,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_r": 0.0,
            "average_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "stop_loss_count": 0,
            "take_profit_count": 0,
            "time_exit_count": 0,
            "skipped_count": skipped_count,
        }

    r_series = pd.Series(
        r_values
    )

    wins = r_series[
        r_series > 0
    ]

    losses = r_series[
        r_series < 0
    ]

    gross_profit = float(
        wins.sum()
    )

    gross_loss = abs(
        float(losses.sum())
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit / gross_loss
        )
    else:
        profit_factor = float("inf")

    equity = r_series.cumsum()
    running_max = equity.cummax()

    drawdown = (
        equity - running_max
    )

    max_drawdown_r = abs(
        float(drawdown.min())
    )

    return {
        "direction": direction,
        "sl_atr": stop_loss_multiplier,
        "tp_atr": take_profit_multiplier,
        "holding_bars": max_holding_bars,
        "trades": len(r_series),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (
            len(wins)
            / len(r_series)
            * 100
        ),
        "total_r": float(
            r_series.sum()
        ),
        "average_r": float(
            r_series.mean()
        ),
        "profit_factor": profit_factor,
        "max_drawdown_r": max_drawdown_r,
        "stop_loss_count": stop_loss_count,
        "take_profit_count": (
            take_profit_count
        ),
        "time_exit_count": (
            time_exit_count
        ),
        "skipped_count": skipped_count,
    }


def print_top_results(
    results: pd.DataFrame,
    direction: str,
) -> None:
    """方向別の上位設定を表示する。"""
    direction_results = results[
        results["direction"] == direction
    ].copy()

    direction_results = (
        direction_results.sort_values(
            by=[
                "average_r",
                "profit_factor",
                "total_r",
                "max_drawdown_r",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
    )

    print(
        f"\n=== {direction} 上位10条件 ==="
    )

    display_columns = [
        "sl_atr",
        "tp_atr",
        "holding_bars",
        "trades",
        "win_rate",
        "total_r",
        "average_r",
        "profit_factor",
        "max_drawdown_r",
        "stop_loss_count",
        "take_profit_count",
        "time_exit_count",
    ]

    print(
        direction_results[
            display_columns
        ]
        .head(10)
        .to_string(index=False)
    )

    best = direction_results.iloc[0]

    print(
        f"\n=== {direction} 最良候補 ==="
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
        f"最大保有: "
        f"{int(best['holding_bars'])}本"
    )

    print(
        f"取引回数: "
        f"{int(best['trades']):,}回"
    )

    print(
        f"勝率: "
        f"{best['win_rate']:.2f}%"
    )

    print(
        f"合計R: "
        f"{best['total_r']:.2f}R"
    )

    print(
        f"平均R: "
        f"{best['average_r']:.4f}R"
    )

    print(
        f"PF: "
        f"{best['profit_factor']:.2f}"
    )

    print(
        f"最大DD: "
        f"{best['max_drawdown_r']:.2f}R"
    )


def main() -> None:
    try:
        candles = load_candles()

        combinations = list(
            product(
                DIRECTIONS,
                STOP_LOSS_MULTIPLIERS,
                TAKE_PROFIT_MULTIPLIERS,
                MAX_HOLDING_BARS_LIST,
            )
        )

        print(
            "=== Athena 出口条件探索 ==="
        )

        print(
            f"ローソク足: "
            f"{len(candles):,}本"
        )

        print(
            f"比較条件: "
            f"{len(combinations)}通り"
        )

        print(
            f"候補間隔: "
            f"{SIGNAL_STEP}本"
        )

        print(
            "\n処理を開始します。"
        )

        results: list[dict] = []

        for number, combination in enumerate(
            combinations,
            start=1,
        ):
            (
                direction,
                stop_loss_multiplier,
                take_profit_multiplier,
                max_holding_bars,
            ) = combination

            result = evaluate_combination(
                candles=candles,
                direction=direction,
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

            results.append(result)

            print(
                f"[{number:02d}/"
                f"{len(combinations)}] "
                f"{direction} "
                f"SL {stop_loss_multiplier:.1f} "
                f"TP {take_profit_multiplier:.1f} "
                f"保有 {max_holding_bars} "
                f"平均R "
                f"{result['average_r']:.4f} "
                f"PF "
                f"{result['profit_factor']:.2f}"
            )

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

        print_top_results(
            result_frame,
            "BUY",
        )

        print_top_results(
            result_frame,
            "SELL",
        )

        positive_results = result_frame[
            result_frame["average_r"] > 0
        ]

        if positive_results.empty:
            print(
                "\n平均Rがプラスの出口条件は"
                "見つかりませんでした。"
            )
        else:
            print(
                "\n平均Rがプラスの出口条件数:",
                len(positive_results),
            )

        print(
            "\n全結果保存先:",
            RESULT_PATH.resolve(),
        )

    except Exception as error:
        print(
            "出口条件探索中に"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()