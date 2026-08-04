from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


SYMBOL = "USDJPY"
TIMEFRAME_NAME = "M5"

DATABASE_PATH = Path("data/athena.db")
FEATURE_PATH = Path("data/market_context_features.csv")
OUTPUT_PATH = Path("data/market_context_trade_labels.csv")

POINT_SIZE = 0.001

# 仮想売買条件
STOP_LOSS_ATR_MULTIPLIER = 1.5
TAKE_PROFIT_ATR_MULTIPLIER = 2.0

# M5 × 24本 = 最大約2時間
MAX_HOLDING_BARS = 24

# 週末などの大きなデータ欠損を含む候補を除外
MAX_BAR_GAP_MINUTES = 15

TARGET_SELL = 0
TARGET_NO_TRADE = 1
TARGET_BUY = 2


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

    candles = (
        candles.drop_duplicates(
            subset=["time"],
            keep="last",
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    return candles


def load_features() -> pd.DataFrame:
    """作成済みの相場環境特徴量を読み込む。"""
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"特徴量CSVが見つかりません: "
            f"{FEATURE_PATH.resolve()}"
        )

    features = pd.read_csv(
        FEATURE_PATH,
        parse_dates=["time"],
    )

    features = (
        features.sort_values("time")
        .drop_duplicates(
            subset=["time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    required_columns = [
        "time",
        "m5_atr14",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in features.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"必要な列がありません: {missing_columns}"
        )

    return features


def determine_direction_outcome(
    open_values: np.ndarray,
    high_values: np.ndarray,
    low_values: np.ndarray,
    close_values: np.ndarray,
    spread_values: np.ndarray,
    time_values: np.ndarray,
    signal_index: int,
    atr: float,
    direction: str,
) -> dict | None:
    """
    指定方向で、SLまたはTPのどちらへ先に到達したか判定する。

    同一足でSL・TPの両方へ到達した場合は、
    保守的に負けとして扱う。
    """
    entry_index = signal_index + 1

    if entry_index >= len(open_values):
        return None

    final_index = min(
        entry_index + MAX_HOLDING_BARS,
        len(open_values) - 1,
    )

    entry_spread = (
        float(spread_values[entry_index])
        * POINT_SIZE
    )

    stop_distance = (
        atr * STOP_LOSS_ATR_MULTIPLIER
    )

    target_distance = (
        atr * TAKE_PROFIT_ATR_MULTIPLIER
    )

    if direction == "BUY":
        entry_price = (
            float(open_values[entry_index])
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
            open_values[entry_index]
        )

        stop_loss = (
            entry_price + stop_distance
        )

        take_profit = (
            entry_price - target_distance
        )

    previous_time = pd.Timestamp(
        time_values[entry_index]
    )

    for check_index in range(
        entry_index,
        final_index + 1,
    ):
        current_time = pd.Timestamp(
            time_values[check_index]
        )

        if check_index > entry_index:
            time_gap = (
                current_time - previous_time
            )

            if time_gap > pd.Timedelta(
                minutes=MAX_BAR_GAP_MINUTES
            ):
                return None

        previous_time = current_time

        high = float(
            high_values[check_index]
        )

        low = float(
            low_values[check_index]
        )

        current_spread = (
            float(spread_values[check_index])
            * POINT_SIZE
        )

        if direction == "BUY":
            stop_hit = low <= stop_loss
            target_hit = high >= take_profit

        else:
            # SELLはAsk価格で決済されるため、
            # 高値と安値へスプレッドを加える
            ask_high = high + current_spread
            ask_low = low + current_spread

            stop_hit = ask_high >= stop_loss
            target_hit = ask_low <= take_profit

        if stop_hit and target_hit:
            return {
                "win": False,
                "ambiguous": True,
                "exit_reason": "BOTH_HIT",
                "holding_bars": (
                    check_index - entry_index + 1
                ),
            }

        if stop_hit:
            return {
                "win": False,
                "ambiguous": False,
                "exit_reason": "STOP_LOSS",
                "holding_bars": (
                    check_index - entry_index + 1
                ),
            }

        if target_hit:
            return {
                "win": True,
                "ambiguous": False,
                "exit_reason": "TAKE_PROFIT",
                "holding_bars": (
                    check_index - entry_index + 1
                ),
            }

    # 保有期限内にTP・SLへ到達しなければNO TRADE用
    return {
        "win": False,
        "ambiguous": False,
        "exit_reason": "TIME_EXIT",
        "holding_bars": (
            final_index - entry_index + 1
        ),
    }


def choose_target(
    buy_result: dict,
    sell_result: dict,
) -> int:
    """BUY・SELLの結果から3分類ラベルを決める。"""
    buy_win = (
        buy_result["win"]
        and not buy_result["ambiguous"]
    )

    sell_win = (
        sell_result["win"]
        and not sell_result["ambiguous"]
    )

    if buy_win and not sell_win:
        return TARGET_BUY

    if sell_win and not buy_win:
        return TARGET_SELL

    return TARGET_NO_TRADE


def build_labels(
    candles: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """全特徴量行に売買バリア方式のラベルを付ける。"""
    candle_time_to_index = {
        timestamp: index
        for index, timestamp
        in enumerate(candles["time"])
    }

    open_values = candles["open"].to_numpy()
    high_values = candles["high"].to_numpy()
    low_values = candles["low"].to_numpy()
    close_values = candles["close"].to_numpy()
    spread_values = candles["spread"].to_numpy()
    time_values = candles["time"].to_numpy()

    rows: list[dict] = []

    total_count = len(features)

    for row_number, feature_row in enumerate(
        features.itertuples(index=False),
        start=1,
    ):
        signal_time = feature_row.time

        signal_index = candle_time_to_index.get(
            signal_time
        )

        if signal_index is None:
            continue

        atr = float(
            feature_row.m5_atr14
        )

        if (
            not np.isfinite(atr)
            or atr <= 0
        ):
            continue

        buy_result = determine_direction_outcome(
            open_values=open_values,
            high_values=high_values,
            low_values=low_values,
            close_values=close_values,
            spread_values=spread_values,
            time_values=time_values,
            signal_index=signal_index,
            atr=atr,
            direction="BUY",
        )

        sell_result = determine_direction_outcome(
            open_values=open_values,
            high_values=high_values,
            low_values=low_values,
            close_values=close_values,
            spread_values=spread_values,
            time_values=time_values,
            signal_index=signal_index,
            atr=atr,
            direction="SELL",
        )

        if (
            buy_result is None
            or sell_result is None
        ):
            continue

        target = choose_target(
            buy_result,
            sell_result,
        )

        rows.append(
            {
                "time": signal_time,
                "target": target,
                "buy_win": int(
                    buy_result["win"]
                ),
                "sell_win": int(
                    sell_result["win"]
                ),
                "buy_exit_reason": (
                    buy_result["exit_reason"]
                ),
                "sell_exit_reason": (
                    sell_result["exit_reason"]
                ),
                "buy_holding_bars": (
                    buy_result["holding_bars"]
                ),
                "sell_holding_bars": (
                    sell_result["holding_bars"]
                ),
            }
        )

        if (
            row_number % 25_000 == 0
            or row_number == total_count
        ):
            print(
                f"処理中: "
                f"{row_number:,} / "
                f"{total_count:,}"
            )

    return pd.DataFrame(rows)


def main() -> None:
    try:
        candles = load_candles()
        features = load_features()

        print(
            f"ローソク足: "
            f"{len(candles):,}本"
        )

        print(
            f"特徴量データ: "
            f"{len(features):,}件"
        )

        print(
            "\n売買バリアラベルを作成中..."
        )

        labels = build_labels(
            candles,
            features,
        )

        if labels.empty:
            print(
                "ラベルを作成できませんでした"
            )
            return

        # 以前の終値方向ラベルを削除し、
        # 今回の売買ラベルへ置き換える
        columns_to_remove = [
            "target",
            "future_change",
        ]

        cleaned_features = features.drop(
            columns=[
                column
                for column in columns_to_remove
                if column in features.columns
            ]
        )

        dataset = cleaned_features.merge(
            labels,
            on="time",
            how="inner",
        )

        # 前回効果がなかった名前付きパターンは除外
        pattern_columns = [
            column
            for column in dataset.columns
            if column.startswith(
                "pattern_"
            )
        ]

        dataset = dataset.drop(
            columns=pattern_columns
        )

        dataset = dataset.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        dataset["target"] = (
            dataset["target"].astype(int)
        )

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("\nCSVへ保存中...")

        dataset.to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        sell_count = int(
            (
                dataset["target"]
                == TARGET_SELL
            ).sum()
        )

        no_trade_count = int(
            (
                dataset["target"]
                == TARGET_NO_TRADE
            ).sum()
        )

        buy_count = int(
            (
                dataset["target"]
                == TARGET_BUY
            ).sum()
        )

        buy_tp = int(
            (
                dataset["buy_exit_reason"]
                == "TAKE_PROFIT"
            ).sum()
        )

        sell_tp = int(
            (
                dataset["sell_exit_reason"]
                == "TAKE_PROFIT"
            ).sum()
        )

        buy_both_hit = int(
            (
                dataset["buy_exit_reason"]
                == "BOTH_HIT"
            ).sum()
        )

        sell_both_hit = int(
            (
                dataset["sell_exit_reason"]
                == "BOTH_HIT"
            ).sum()
        )

        print(
            "\n=== 売買バリアラベル作成結果 ==="
        )

        print(
            f"利用可能データ: "
            f"{len(dataset):,}件"
        )

        print(
            f"特徴量を含む全列数: "
            f"{len(dataset.columns):,}列"
        )

        print(
            f"SELL: "
            f"{sell_count:,}件 "
            f"({sell_count / len(dataset):.2%})"
        )

        print(
            f"NO TRADE: "
            f"{no_trade_count:,}件 "
            f"({no_trade_count / len(dataset):.2%})"
        )

        print(
            f"BUY: "
            f"{buy_count:,}件 "
            f"({buy_count / len(dataset):.2%})"
        )

        print(
            f"\nBUYのTP到達: "
            f"{buy_tp:,}件"
        )

        print(
            f"SELLのTP到達: "
            f"{sell_tp:,}件"
        )

        print(
            f"BUY同一足SL・TP: "
            f"{buy_both_hit:,}件"
        )

        print(
            f"SELL同一足SL・TP: "
            f"{sell_both_hit:,}件"
        )

        print(
            "期間:",
            dataset.iloc[0]["time"],
            "～",
            dataset.iloc[-1]["time"],
        )

        print(
            "\n設定:"
        )

        print(
            f"SL: ATR × "
            f"{STOP_LOSS_ATR_MULTIPLIER}"
        )

        print(
            f"TP: ATR × "
            f"{TAKE_PROFIT_ATR_MULTIPLIER}"
        )

        print(
            f"最大保有: "
            f"{MAX_HOLDING_BARS}本"
        )

        print(
            "\n保存先:",
            OUTPUT_PATH.resolve(),
        )

    except Exception as error:
        print(
            "\n売買バリアラベルの作成中に"
            "エラーが発生しました"
        )

        print(error)


if __name__ == "__main__":
    main()