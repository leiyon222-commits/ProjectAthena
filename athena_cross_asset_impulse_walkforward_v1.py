import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DATASET = Path("data/market_context_expected_r_enriched.csv")
INITIAL_TRAIN_RATIO = 0.40
SEGMENT_RATIO = 0.075
FOLDS = 4
GAP_ROWS = 48
COST_R = 0.05
PRIMARY_AGREEMENT = 0.75
NEIGHBOR_AGREEMENTS = (2 / 3, 5 / 6)
MINIMUM_TRADES = 200

COLUMNS = [
    "time", "buy_trade_r", "sell_trade_r",
    "buy_new_holding_bars", "sell_new_holding_bars",
    "buy_new_exit_reason", "sell_new_exit_reason",
    "strength_usdjpy_pressure_1h", "strength_all_agreement_1h",
]


def load_rows():
    rows = pd.read_csv(DATASET, usecols=COLUMNS, parse_dates=["time"], low_memory=False)
    rows["time"] = pd.to_datetime(rows["time"], utc=True, errors="coerce")
    rows = rows.replace([np.inf, -np.inf], np.nan).dropna(subset=COLUMNS)
    return rows.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)


def add_signals(rows, agreement):
    result = rows.copy()
    hourly = result[(result["time"].dt.minute == 0)].copy()
    pressure = hourly["strength_usdjpy_pressure_1h"]
    prior_pressure = pressure.shift(1)
    positive_agreement = hourly["strength_all_agreement_1h"] >= agreement
    negative_agreement = hourly["strength_all_agreement_1h"] <= 1.0 - agreement
    hourly["buy_signal"] = (prior_pressure <= 0) & (pressure > 0) & positive_agreement
    hourly["sell_signal"] = (prior_pressure >= 0) & (pressure < 0) & negative_agreement
    signals = hourly.loc[hourly["buy_signal"] | hourly["sell_signal"], ["time", "buy_signal", "sell_signal"]]
    return result.merge(signals, on="time", how="left").assign(
        buy_signal=lambda x: x["buy_signal"].fillna(False).astype(bool),
        sell_signal=lambda x: x["sell_signal"].fillna(False).astype(bool),
    )


def tests(rows):
    total = len(rows)
    initial = int(total * INITIAL_TRAIN_RATIO)
    segment = int(total * SEGMENT_RATIO)
    for index in range(FOLDS):
        train_end = initial + index * segment * 2
        validation_end = train_end + segment
        test_end = min(validation_end + segment, total)
        yield rows.iloc[validation_end:test_end - GAP_ROWS]


def trade_frame(rows, fold):
    trades, next_allowed = [], None
    for row in rows.loc[rows["buy_signal"] | rows["sell_signal"]].itertuples(index=False):
        time = pd.Timestamp(row.time)
        if next_allowed is not None and time < next_allowed:
            continue
        if row.buy_signal:
            direction, raw_r = "BUY", float(row.buy_trade_r)
            holding, reason = int(row.buy_new_holding_bars), row.buy_new_exit_reason
        else:
            direction, raw_r = "SELL", float(row.sell_trade_r)
            holding, reason = int(row.sell_new_holding_bars), row.sell_new_exit_reason
        trades.append({"fold": fold, "time": time, "direction": direction,
                       "actual_r": raw_r - COST_R, "holding_bars": holding,
                       "exit_reason": reason})
        next_allowed = time + pd.Timedelta(minutes=(max(1, holding) + 1) * 5)
    return pd.DataFrame(trades)


def statistics(trades):
    if trades.empty:
        return {"trade_count": 0, "buy_count": 0, "sell_count": 0,
                "total_r": 0.0, "average_r": 0.0, "average_r_95pct_lower": 0.0,
                "profit_factor": 0.0, "max_drawdown_r": 0.0, "max_losing_streak": 0}
    values = trades["actual_r"]
    average = float(values.mean())
    se = float(values.std(ddof=1)) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    wins, losses = values[values > 0], values[values < 0]
    equity = pd.concat([pd.Series([0.0]), values.cumsum().reset_index(drop=True)], ignore_index=True)
    drawdown = equity - equity.cummax()
    longest = current = 0
    for value in values:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return {
        "trade_count": int(len(trades)),
        "buy_count": int((trades["direction"] == "BUY").sum()),
        "sell_count": int((trades["direction"] == "SELL").sum()),
        "total_r": float(values.sum()), "average_r": average,
        "average_r_95pct_lower": average - 1.96 * se,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if float(losses.sum()) < 0 else float("inf"),
        "max_drawdown_r": abs(float(drawdown.min())), "max_losing_streak": longest,
    }


def evaluate(rows, agreement):
    signaled = add_signals(rows, agreement)
    frames, fold_stats = [], []
    for fold, test in enumerate(tests(signaled), 1):
        trades = trade_frame(test, fold)
        value = statistics(trades)
        value["fold"] = fold
        fold_stats.append(value)
        if not trades.empty:
            frames.append(trades)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    overall = statistics(combined)
    overall["positive_folds"] = sum(x["total_r"] > 0 for x in fold_stats)
    overall["folds"] = fold_stats
    return overall, combined


def main():
    rows = load_rows()
    primary, trades = evaluate(rows, PRIMARY_AGREEMENT)
    neighbors = {}
    for value in NEIGHBOR_AGREEMENTS:
        neighbors[f"{value:.6f}"] = evaluate(rows, value)[0]
    primary["neighbors"] = neighbors
    primary["data_rows"] = len(rows)
    primary["missing_times"] = int(rows["time"].isna().sum())
    primary["duplicate_times"] = int(rows["time"].duplicated().sum())
    primary["minimum_trades"] = MINIMUM_TRADES
    print(json.dumps(primary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
