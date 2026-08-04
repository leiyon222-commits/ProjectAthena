from pathlib import Path
import math

import numpy as np
import pandas as pd


DATASET_PATH = Path("data/market_context_expected_r_enriched.csv")
INITIAL_TRAIN_RATIO = 0.40
SEGMENT_RATIO = 0.075
NUMBER_OF_FOLDS = 4
GAP_ROWS = 48
PRIMARY_LOOKBACK = 20
ROBUSTNESS_LOOKBACKS = (15, 30)
ADDITIONAL_COST_R = 0.05
MINIMUM_TRADES = 200

REQUIRED_COLUMNS = [
    "time", "open", "high", "low", "close",
    "buy_trade_r", "sell_trade_r",
    "buy_new_exit_reason", "sell_new_exit_reason",
    "buy_new_holding_bars", "sell_new_holding_bars",
]


def load_rows() -> pd.DataFrame:
    header = pd.read_csv(DATASET_PATH, nrows=0).columns
    missing = set(REQUIRED_COLUMNS) - set(header)
    if missing:
        raise RuntimeError(f"必要列不足: {sorted(missing)}")
    rows = pd.read_csv(
        DATASET_PATH,
        usecols=REQUIRED_COLUMNS,
        parse_dates=["time"],
        low_memory=False,
    )
    rows["time"] = pd.to_datetime(rows["time"], utc=True, errors="coerce")
    numeric = [
        "open", "high", "low", "close", "buy_trade_r", "sell_trade_r",
        "buy_new_holding_bars", "sell_new_holding_bars",
    ]
    rows = (
        rows.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["time", *numeric])
        .sort_values("time")
        .drop_duplicates("time", keep="last")
        .reset_index(drop=True)
    )
    return rows


def resample_ohlc(rows: pd.DataFrame, rule: str) -> pd.DataFrame:
    """M5終値確定時刻基準で上位足を構築する。"""
    return (
        rows.set_index("time")
        .resample(rule, label="right", closed="right")
        .agg(open=("open", "first"), high=("high", "max"),
             low=("low", "min"), close=("close", "last"))
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )


def build_signals(rows: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """完了H1ブレイクと、その時点より前に確定したH4トレンドを作る。"""
    h1 = resample_ohlc(rows, "1h")
    h4 = resample_ohlc(rows, "4h")

    h4["ema50"] = h4["close"].ewm(span=50, adjust=False).mean()
    h4["ema200"] = h4["close"].ewm(span=200, adjust=False).mean()
    h4["ema50_slope2"] = h4["ema50"].diff(2)
    # H1シグナル時点で形成中かもしれないH4を絶対に使わない。
    confirmed_h4 = h4[["time", "ema50", "ema200", "ema50_slope2"]].copy()
    confirmed_h4[["ema50", "ema200", "ema50_slope2"]] = confirmed_h4[
        ["ema50", "ema200", "ema50_slope2"]
    ].shift(1)

    h1["prior_high"] = h1["high"].shift(1).rolling(lookback).max()
    h1["prior_low"] = h1["low"].shift(1).rolling(lookback).min()
    h1 = pd.merge_asof(
        h1.sort_values("time"),
        confirmed_h4.sort_values("time"),
        on="time",
        direction="backward",
    )
    h1["buy_signal"] = (
        (h1["close"] > h1["prior_high"])
        & (h1["ema50"] > h1["ema200"])
        & (h1["ema50_slope2"] > 0)
    )
    h1["sell_signal"] = (
        (h1["close"] < h1["prior_low"])
        & (h1["ema50"] < h1["ema200"])
        & (h1["ema50_slope2"] < 0)
    )
    signals = h1.loc[
        h1["buy_signal"] | h1["sell_signal"],
        ["time", "buy_signal", "sell_signal"],
    ]
    return rows.merge(signals, on="time", how="left").assign(
        buy_signal=lambda x: x["buy_signal"].fillna(False).astype(bool),
        sell_signal=lambda x: x["sell_signal"].fillna(False).astype(bool),
    )


def simulate(rows: pd.DataFrame, fold: int) -> pd.DataFrame:
    trades = []
    next_allowed_time = None
    for row in rows.loc[rows["buy_signal"] | rows["sell_signal"]].itertuples(index=False):
        time = pd.Timestamp(row.time)
        if next_allowed_time is not None and time < next_allowed_time:
            continue
        if row.buy_signal:
            direction = "BUY"
            raw_r = float(row.buy_trade_r)
            holding = int(row.buy_new_holding_bars)
            reason = row.buy_new_exit_reason
        else:
            direction = "SELL"
            raw_r = float(row.sell_trade_r)
            holding = int(row.sell_new_holding_bars)
            reason = row.sell_new_exit_reason
        trades.append({
            "fold": fold, "time": time, "direction": direction,
            "raw_r": raw_r, "actual_r": raw_r - ADDITIONAL_COST_R,
            "holding_bars": holding, "exit_reason": reason,
        })
        next_allowed_time = time + pd.Timedelta(minutes=(max(1, holding) + 1) * 5)
    return pd.DataFrame(trades)


def maximum_losing_streak(values: pd.Series) -> int:
    longest = current = 0
    for value in values:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def metrics(trades: pd.DataFrame) -> dict[str, float | int]:
    if trades.empty:
        return {key: 0 for key in (
            "trades", "buy_count", "sell_count", "win_rate", "total_r",
            "average_r", "lcb95", "pf", "max_dd", "max_losing_streak",
        )}
    wins = trades.loc[trades["actual_r"] > 0, "actual_r"]
    losses = trades.loc[trades["actual_r"] < 0, "actual_r"]
    avg = float(trades["actual_r"].mean())
    se = float(trades["actual_r"].std(ddof=1)) / math.sqrt(len(trades)) if len(trades) > 1 else 0.0
    equity = pd.concat([pd.Series([0.0]), trades["actual_r"].cumsum().reset_index(drop=True)], ignore_index=True)
    dd = equity - equity.cummax()
    return {
        "trades": len(trades),
        "buy_count": int((trades["direction"] == "BUY").sum()),
        "sell_count": int((trades["direction"] == "SELL").sum()),
        "win_rate": float((trades["actual_r"] > 0).mean() * 100),
        "total_r": float(trades["actual_r"].sum()),
        "average_r": avg,
        "lcb95": avg - 1.96 * se,
        "pf": float(wins.sum() / abs(losses.sum())) if float(losses.sum()) < 0 else float("inf"),
        "max_dd": abs(float(dd.min())),
        "max_losing_streak": maximum_losing_streak(trades["actual_r"]),
    }


def fold_tests(rows: pd.DataFrame) -> list[pd.DataFrame]:
    total = len(rows)
    initial = int(total * INITIAL_TRAIN_RATIO)
    segment = int(total * SEGMENT_RATIO)
    tests = []
    for index in range(NUMBER_OF_FOLDS):
        train_end = initial + index * segment * 2
        validation_end = train_end + segment
        test_end = min(validation_end + segment, total)
        tests.append(rows.iloc[validation_end:test_end - GAP_ROWS].copy())
    return tests


def evaluate(rows: pd.DataFrame, lookback: int, verbose: bool) -> tuple[dict, list[dict], pd.DataFrame]:
    signaled = build_signals(rows, lookback)
    fold_rows = []
    trade_frames = []
    for fold, test in enumerate(fold_tests(signaled), start=1):
        trades = simulate(test, fold)
        result = metrics(trades)
        result["fold"] = fold
        result["signals"] = int((test["buy_signal"] | test["sell_signal"]).sum())
        fold_rows.append(result)
        if not trades.empty:
            trade_frames.append(trades)
        if verbose:
            print(
                f"Fold {fold}: signals {result['signals']:,}, trades {result['trades']:,}, "
                f"BUY/SELL {result['buy_count']}/{result['sell_count']}, "
                f"total {result['total_r']:+.2f}R, PF {result['pf']:.2f}, "
                f"DD {result['max_dd']:.2f}R, losing streak {result['max_losing_streak']}"
            )
    combined = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    return metrics(combined), fold_rows, combined


def print_period_breakdown(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("期間別: 取引なし")
        return
    for label, fmt in (("年別", "%Y"), ("月別", "%Y-%m")):
        print(f"\n{label}:")
        grouped = (
            trades.assign(period=trades["time"].dt.strftime(fmt))
            .groupby("period").agg(trades=("actual_r", "size"), total_r=("actual_r", "sum"))
        )
        for period, row in grouped.iterrows():
            print(f"{period}: {int(row['trades']):,}回 / {row['total_r']:+.2f}R")


def main() -> None:
    print("=== ATH-H1-BREAKOUT-001 ===")
    print("H1 confirmed Donchian breakout / confirmed H4 trend")
    print(f"追加費用: {ADDITIONAL_COST_R:.2f}R/取引（既存ラベル内スプレッドとは別）")
    rows = load_rows()
    print(f"データ: {len(rows):,} M5行 / {rows.iloc[0]['time']} ～ {rows.iloc[-1]['time']}")
    print(f"欠損時刻: {rows['time'].isna().sum()}, 重複時刻: {rows['time'].duplicated().sum()}")

    print("\n主条件（20 H1）:")
    overall, folds, trades = evaluate(rows, PRIMARY_LOOKBACK, verbose=True)
    positive_folds = sum(item["total_r"] > 0 for item in folds)
    print(
        f"全体: {overall['trades']:,}回, BUY/SELL {overall['buy_count']}/{overall['sell_count']}, "
        f"勝率 {overall['win_rate']:.2f}%, 合計 {overall['total_r']:+.2f}R, "
        f"平均 {overall['average_r']:+.4f}R, 95%下限 {overall['lcb95']:+.4f}R, "
        f"PF {overall['pf']:.2f}, 最大DD {overall['max_dd']:.2f}R, "
        f"最大連敗 {overall['max_losing_streak']}, プラスFold {positive_folds}/4"
    )
    print_period_breakdown(trades)

    print("\n事前登録済み近傍安定性（選択には不使用）:")
    neighbor_results = []
    for lookback in ROBUSTNESS_LOOKBACKS:
        result, neighbor_folds, _ = evaluate(rows, lookback, verbose=False)
        neighbor_positive = sum(item["total_r"] > 0 for item in neighbor_folds)
        neighbor_results.append((lookback, result, neighbor_positive))
        print(
            f"{lookback} H1: {result['trades']:,}回, {result['total_r']:+.2f}R, "
            f"PF {result['pf']:.2f}, DD {result['max_dd']:.2f}R, プラスFold {neighbor_positive}/4"
        )

    total_profit = overall["total_r"]
    profitable_fold_values = [max(0.0, float(item["total_r"])) for item in folds]
    fold_concentration = (
        max(profitable_fold_values) / total_profit if total_profit > 0 else float("inf")
    )
    if not trades.empty and total_profit > 0:
        monthly_profit = trades.assign(month=trades["time"].dt.to_period("M")).groupby("month")["actual_r"].sum().clip(lower=0)
        month_concentration = float(monthly_profit.max() / total_profit)
    else:
        month_concentration = float("inf")
    direction_ratio = (
        max(overall["buy_count"], overall["sell_count"]) / overall["trades"]
        if overall["trades"] else 1.0
    )
    neighbors_stable = all(
        result["pf"] >= 1.20 and positives >= 3
        for _, result, positives in neighbor_results
    )
    accepted = all([
        overall["trades"] >= MINIMUM_TRADES,
        positive_folds >= 3,
        overall["pf"] >= 1.20,
        overall["lcb95"] > 0,
        overall["max_dd"] <= 15,
        fold_concentration <= 0.40,
        month_concentration < 1.0,
        direction_ratio <= 0.80,
        neighbors_stable,
    ])
    print("\n判定:", "CANDIDATE_FOUND" if accepted else "REJECTED")
    print(
        f"監査値: fold利益集中 {fold_concentration:.3f}, 月利益集中 {month_concentration:.3f}, "
        f"最大方向比率 {direction_ratio:.3f}, 近傍安定 {neighbors_stable}"
    )


if __name__ == "__main__":
    main()
