from pathlib import Path
import math

import numpy as np
import pandas as pd


DATASET_PATH = Path("data/market_context_expected_r_enriched.csv")

INITIAL_TRAIN_RATIO = 0.40
SEGMENT_RATIO = 0.075
NUMBER_OF_FOLDS = 4
GAP_ROWS = 48
MIN_LONG_ALIGNMENT = 5

REQUIRED_COLUMNS = [
    "time",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "m5_ema20_gap",
    "m5_ema50_gap",
    "m5_rsi14",
    "buy_trade_r",
    "sell_trade_r",
    "buy_new_exit_reason",
    "sell_new_exit_reason",
    "buy_new_holding_bars",
    "sell_new_holding_bars",
    "long_alignment_score",
    "long_d1_return_1",
    "long_d1_return_20",
    "long_d1_return_252",
    "strength_usdjpy_pressure_1d",
    "strength_usdjpy_pressure_1w",
]


def load_dataset() -> pd.DataFrame:
    """必要列を時系列順に読み込む。"""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"データセットがありません: {DATASET_PATH.resolve()}"
        )

    header = pd.read_csv(DATASET_PATH, nrows=0).columns
    missing = set(REQUIRED_COLUMNS) - set(header)
    if missing:
        raise RuntimeError(f"必要な列がありません: {sorted(missing)}")

    rows = pd.read_csv(
        DATASET_PATH,
        usecols=REQUIRED_COLUMNS,
        parse_dates=["time"],
        low_memory=False,
    )
    rows["time"] = pd.to_datetime(rows["time"], utc=True, errors="coerce")
    rows = (
        rows.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=REQUIRED_COLUMNS)
        .sort_values("time")
        .drop_duplicates(subset=["time"], keep="last")
        .reset_index(drop=True)
    )
    return rows


def create_regime_direction(rows: pd.DataFrame) -> pd.Series:
    """既存実験と同一の固定長期方向判定を作る。"""
    buy = (
        (rows["long_alignment_score"] >= MIN_LONG_ALIGNMENT)
        & (rows["long_d1_return_252"] > 0)
        & (rows["long_d1_return_20"] > 0)
        & (rows["long_d1_return_1"] > 0)
        & (rows["strength_usdjpy_pressure_1d"] > 0)
        & (rows["strength_usdjpy_pressure_1w"] > 0)
    )
    sell = (
        (rows["long_alignment_score"] <= -MIN_LONG_ALIGNMENT)
        & (rows["long_d1_return_252"] < 0)
        & (rows["long_d1_return_20"] < 0)
        & (rows["long_d1_return_1"] < 0)
        & (rows["strength_usdjpy_pressure_1d"] < 0)
        & (rows["strength_usdjpy_pressure_1w"] < 0)
    )
    return pd.Series(
        np.select([buy, sell], ["BUY", "SELL"], default="NONE"),
        index=rows.index,
        dtype="string",
    )


def create_confirmed_higher_timeframe(
    rows: pd.DataFrame,
    rule: str,
    prefix: str,
) -> pd.DataFrame:
    """M5から上位足を作り、1本前に確定した値だけを返す。"""
    higher = (
        rows.set_index("time")
        .resample(rule, label="right", closed="right")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        )
        .dropna(subset=["open", "high", "low", "close"])
    )

    higher["ema20"] = higher["close"].ewm(span=20, adjust=False).mean()
    higher["ema50"] = higher["close"].ewm(span=50, adjust=False).mean()
    higher["ema20_slope"] = higher["ema20"].diff()

    # その時刻に形成された上位足を使わず、直前の確定足へずらす。
    confirmed = higher[["close", "ema20", "ema50", "ema20_slope"]].shift(1)
    confirmed = confirmed.rename(
        columns={column: f"{prefix}_{column}" for column in confirmed.columns}
    )
    return confirmed.reset_index()


def add_causal_event_conditions(rows: pd.DataFrame) -> pd.DataFrame:
    """確定上位足とM5クロスイベントを追加する。"""
    result = rows.copy()
    h1 = create_confirmed_higher_timeframe(result, "1h", "confirmed_h1")
    h4 = create_confirmed_higher_timeframe(result, "4h", "confirmed_h4")

    result = pd.merge_asof(
        result.sort_values("time"),
        h1.sort_values("time"),
        on="time",
        direction="backward",
    )
    result = pd.merge_asof(
        result.sort_values("time"),
        h4.sort_values("time"),
        on="time",
        direction="backward",
    )

    result["regime_direction"] = create_regime_direction(result)
    result["buy_higher_allowed"] = (
        (result["confirmed_h4_close"] > result["confirmed_h4_ema50"])
        & (result["confirmed_h1_ema20_slope"] > 0)
    )
    result["sell_higher_allowed"] = (
        (result["confirmed_h4_close"] < result["confirmed_h4_ema50"])
        & (result["confirmed_h1_ema20_slope"] < 0)
    )

    # gap=(close-EMA)/closeなので、符号が終値とEMAの上下関係を表す。
    previous_gap20 = result["m5_ema20_gap"].shift(1)
    ema20 = result["close"] * (1.0 - result["m5_ema20_gap"])
    ema50 = result["close"] * (1.0 - result["m5_ema50_gap"])

    result["buy_cross_event"] = (
        (previous_gap20 <= 0)
        & (result["m5_ema20_gap"] > 0)
        & (ema20 > ema50)
        & result["m5_rsi14"].between(45, 65, inclusive="both")
    )
    result["sell_cross_event"] = (
        (previous_gap20 >= 0)
        & (result["m5_ema20_gap"] < 0)
        & (ema20 < ema50)
        & result["m5_rsi14"].between(35, 55, inclusive="both")
    )

    result["buy_signal"] = (
        (result["regime_direction"] == "BUY")
        & result["buy_higher_allowed"]
        & result["buy_cross_event"]
    )
    result["sell_signal"] = (
        (result["regime_direction"] == "SELL")
        & result["sell_higher_allowed"]
        & result["sell_cross_event"]
    )
    return result


def count_stages(rows: pd.DataFrame) -> dict[str, int]:
    """方向、上位足、イベントの段階別件数を数える。"""
    buy_long = rows["regime_direction"] == "BUY"
    sell_long = rows["regime_direction"] == "SELL"
    buy_higher = buy_long & rows["buy_higher_allowed"]
    sell_higher = sell_long & rows["sell_higher_allowed"]
    buy_event = buy_higher & rows["buy_cross_event"]
    sell_event = sell_higher & rows["sell_cross_event"]
    return {
        "long_buy": int(buy_long.sum()),
        "long_sell": int(sell_long.sum()),
        "higher_buy": int(buy_higher.sum()),
        "higher_sell": int(sell_higher.sum()),
        "event_buy": int(buy_event.sum()),
        "event_sell": int(sell_event.sum()),
    }


def simulate_trades(rows: pd.DataFrame, fold: int) -> pd.DataFrame:
    """イベントを時系列処理し、取引終了足の次から再許可する。"""
    trades: list[dict[str, object]] = []
    next_allowed_time: pd.Timestamp | None = None
    candidates = rows[rows["buy_signal"] | rows["sell_signal"]]

    for row in candidates.itertuples(index=False):
        signal_time = pd.Timestamp(row.time)
        if next_allowed_time is not None and signal_time < next_allowed_time:
            continue

        if bool(row.buy_signal):
            direction = "BUY"
            actual_r = float(row.buy_trade_r)
            holding_bars = int(row.buy_new_holding_bars)
            exit_reason = row.buy_new_exit_reason
        else:
            direction = "SELL"
            actual_r = float(row.sell_trade_r)
            holding_bars = int(row.sell_new_holding_bars)
            exit_reason = row.sell_new_exit_reason

        trades.append(
            {
                "fold": fold,
                "time": signal_time,
                "direction": direction,
                "actual_r": actual_r,
                "holding_bars": holding_bars,
                "exit_reason": exit_reason,
            }
        )
        next_allowed_time = signal_time + pd.Timedelta(
            minutes=(max(1, holding_bars) + 1) * 5
        )

    return pd.DataFrame(trades)


def calculate_metrics(trades: pd.DataFrame) -> dict[str, float | int]:
    """Rベースの成績を計算する。"""
    if trades.empty:
        return {
            "trades": 0,
            "buy_count": 0,
            "sell_count": 0,
            "win_rate": 0.0,
            "total_r": 0.0,
            "average_r": 0.0,
            "average_r_lcb95": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
        }

    positive = trades.loc[trades["actual_r"] > 0, "actual_r"]
    negative = trades.loc[trades["actual_r"] < 0, "actual_r"]
    gross_profit = float(positive.sum())
    gross_loss = abs(float(negative.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss else float("inf")
    average_r = float(trades["actual_r"].mean())
    if len(trades) >= 2:
        standard_error = float(trades["actual_r"].std(ddof=1)) / math.sqrt(len(trades))
        lower_bound = average_r - 1.96 * standard_error
    else:
        lower_bound = average_r
    equity = pd.concat(
        [pd.Series([0.0]), trades["actual_r"].cumsum().reset_index(drop=True)],
        ignore_index=True,
    )
    drawdown = equity - equity.cummax()
    return {
        "trades": int(len(trades)),
        "buy_count": int((trades["direction"] == "BUY").sum()),
        "sell_count": int((trades["direction"] == "SELL").sum()),
        "win_rate": float((trades["actual_r"] > 0).mean() * 100),
        "total_r": float(trades["actual_r"].sum()),
        "average_r": average_r,
        "average_r_lcb95": float(lower_bound),
        "profit_factor": float(profit_factor),
        "max_drawdown_r": abs(float(drawdown.min())),
    }


def print_fold_result(
    fold: int,
    rows: pd.DataFrame,
    stages: dict[str, int],
    metrics: dict[str, float | int],
) -> None:
    print("\n==============================")
    print(f"Fold {fold}")
    print(f"テスト期間: {rows.iloc[0]['time']} ～ {rows.iloc[-1]['time']}")
    print(
        f"長期方向条件: {stages['long_buy'] + stages['long_sell']:,}件 "
        f"(BUY {stages['long_buy']:,} / SELL {stages['long_sell']:,})"
    )
    print(
        f"H1・H4条件通過: {stages['higher_buy'] + stages['higher_sell']:,}件 "
        f"(BUY {stages['higher_buy']:,} / SELL {stages['higher_sell']:,})"
    )
    print(
        f"M5クロスイベント: {stages['event_buy'] + stages['event_sell']:,}件 "
        f"(BUY {stages['event_buy']:,} / SELL {stages['event_sell']:,})"
    )
    print(f"実際の取引: {metrics['trades']:,}回")
    print(f"BUY / SELL: {metrics['buy_count']:,} / {metrics['sell_count']:,}")
    print(f"勝率: {metrics['win_rate']:.2f}%")
    print(f"合計R: {metrics['total_r']:+.2f}R")
    print(f"平均R: {metrics['average_r']:+.4f}R")
    print(f"平均Rの95％下限: {metrics['average_r_lcb95']:+.4f}R")
    print(f"PF: {metrics['profit_factor']:.2f}")
    print(f"最大DD: {metrics['max_drawdown_r']:.2f}R")


def main() -> None:
    print("=== Athena 押し目・戻り復帰イベント Walk-Forward v1 ===")
    rows = add_causal_event_conditions(load_dataset())
    total_count = len(rows)
    initial_train_count = int(total_count * INITIAL_TRAIN_RATIO)
    segment_count = int(total_count * SEGMENT_RATIO)

    all_trades: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, float | int]] = []
    total_stages = {key: 0 for key in count_stages(rows)}

    print(f"全データ: {total_count:,}件")
    print(f"期間: {rows.iloc[0]['time']} ～ {rows.iloc[-1]['time']}")
    print("固定ルールのみ。学習・検証選択・パラメーター調整なし。")

    for fold_index in range(NUMBER_OF_FOLDS):
        fold = fold_index + 1
        train_end = initial_train_count + fold_index * segment_count * 2
        validation_end = train_end + segment_count
        test_end = min(validation_end + segment_count, total_count)
        test = rows.iloc[validation_end : test_end - GAP_ROWS].copy()
        if test.empty:
            raise RuntimeError(f"Fold {fold}のテスト期間が空です")

        stages = count_stages(test)
        trades = simulate_trades(test, fold)
        metrics = calculate_metrics(trades)
        metrics["fold"] = fold
        fold_metrics.append(metrics)
        if not trades.empty:
            all_trades.append(trades)
        for key, value in stages.items():
            total_stages[key] += value
        print_fold_result(fold, test, stages, metrics)

    combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    overall = calculate_metrics(combined)
    positive_folds = sum(float(item["total_r"]) > 0 for item in fold_metrics)
    negative_folds = sum(float(item["total_r"]) < 0 for item in fold_metrics)

    print("\n==============================")
    print("=== 全未使用テスト合計 ===")
    print(
        f"長期方向条件: {total_stages['long_buy'] + total_stages['long_sell']:,}件 "
        f"(BUY {total_stages['long_buy']:,} / SELL {total_stages['long_sell']:,})"
    )
    print(
        f"H1・H4条件通過: {total_stages['higher_buy'] + total_stages['higher_sell']:,}件 "
        f"(BUY {total_stages['higher_buy']:,} / SELL {total_stages['higher_sell']:,})"
    )
    print(
        f"M5クロスイベント: {total_stages['event_buy'] + total_stages['event_sell']:,}件 "
        f"(BUY {total_stages['event_buy']:,} / SELL {total_stages['event_sell']:,})"
    )
    print(f"実際の取引: {overall['trades']:,}回")
    print(f"BUY / SELL: {overall['buy_count']:,} / {overall['sell_count']:,}")
    print(f"勝率: {overall['win_rate']:.2f}%")
    print(f"合計R: {overall['total_r']:+.2f}R")
    print(f"平均R: {overall['average_r']:+.4f}R")
    print(f"平均Rの95％下限: {overall['average_r_lcb95']:+.4f}R")
    print(f"PF: {overall['profit_factor']:.2f}")
    print(f"最大DD: {overall['max_drawdown_r']:.2f}R")
    print(f"プラスFold: {positive_folds} / {NUMBER_OF_FOLDS}")

    print("\n月別成績:")
    if combined.empty:
        print("取引なし")
    else:
        monthly = (
            combined.assign(month=combined["time"].dt.strftime("%Y-%m"))
            .groupby("month", sort=True)
            .agg(trades=("actual_r", "size"), total_r=("actual_r", "sum"))
            .reset_index()
        )
        for row in monthly.itertuples(index=False):
            print(f"{row.month}: {row.trades:,}回 / {row.total_r:+.2f}R")

    rejected = overall["profit_factor"] <= 1.0 or negative_folds >= 3
    print("\n判定:", "不採用" if rejected else "採用候補")
    if rejected:
        reasons = []
        if overall["profit_factor"] <= 1.0:
            reasons.append(f"全体PF {overall['profit_factor']:.2f} が1以下")
        if negative_folds >= 3:
            reasons.append(f"マイナスFoldが{negative_folds}/4")
        print("理由:", "、".join(reasons))


if __name__ == "__main__":
    main()
