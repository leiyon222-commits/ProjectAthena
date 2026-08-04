from pathlib import Path

import numpy as np
import pandas as pd

from athena_h1_confirmed_breakout_walkforward_v1 import (
    ADDITIONAL_COST_R,
    fold_tests,
    load_rows,
    metrics,
    resample_ohlc,
    simulate,
)


RESULT_PATH = Path("research/final_batch_results.csv")
MONTHLY_PATH = Path("research/final_batch_monthly.csv")


def first_per_day(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[frame["buy_signal"] | frame["sell_signal"]].copy()
    selected["date"] = selected["time"].dt.floor("D")
    return selected.sort_values("time").groupby("date", as_index=False).first()


def to_m5(rows: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    signals = events[["time", "buy_signal", "sell_signal"]]
    return rows.merge(signals, on="time", how="left").assign(
        buy_signal=lambda x: x["buy_signal"].fillna(False).astype(bool),
        sell_signal=lambda x: x["sell_signal"].fillna(False).astype(bool),
    )


def build_all_signals(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    h1 = resample_ohlc(rows, "1h")
    h4 = resample_ohlc(rows, "4h")
    strategies = {}

    prior_h4_high = h4["high"].shift(1).rolling(10).max()
    prior_h4_low = h4["low"].shift(1).rolling(10).min()
    e = h4.copy()
    e["buy_signal"] = e["close"] > prior_h4_high
    e["sell_signal"] = e["close"] < prior_h4_low
    strategies["ATH-H4-BREAKOUT-001"] = to_m5(rows, e)

    mean = h4["close"].shift(1).rolling(12).mean()
    std = h4["close"].shift(1).rolling(12).std()
    z = (h4["close"] - mean) / std
    e = h4.copy()
    e["buy_signal"] = (z < -1.5) & (z.shift(1) >= -1.5)
    e["sell_signal"] = (z > 1.5) & (z.shift(1) <= 1.5)
    strategies["ATH-H4-MEANREV-001"] = to_m5(rows, e)

    pc = h4["close"].shift(1)
    tr = pd.concat([h4["high"]-h4["low"], (h4["high"]-pc).abs(), (h4["low"]-pc).abs()], axis=1).max(axis=1)
    expanded = tr > 1.4 * tr.shift(1).rolling(10).median()
    location = (h4["close"]-h4["low"]) / (h4["high"]-h4["low"]).replace(0, np.nan)
    e = h4.copy()
    first = expanded & ~expanded.shift(1, fill_value=False)
    e["buy_signal"] = first & (e["close"] > e["open"]) & (location >= .75)
    e["sell_signal"] = first & (e["close"] < e["open"]) & (location <= .25)
    strategies["ATH-H4-EXPANSION-001"] = to_m5(rows, e)

    ema20 = h4["close"].ewm(span=20, adjust=False).mean()
    ema50 = h4["close"].ewm(span=50, adjust=False).mean()
    e = h4.copy()
    e["buy_signal"] = (h4["close"].shift(1) <= ema20.shift(1)) & (h4["close"] > ema20) & (ema20 > ema50)
    e["sell_signal"] = (h4["close"].shift(1) >= ema20.shift(1)) & (h4["close"] < ema20) & (ema20 < ema50)
    strategies["ATH-H4-PULLBACK-001"] = to_m5(rows, e)

    ret = h1["close"].diff()
    up3 = (ret > 0) & (ret.shift(1) > 0) & (ret.shift(2) > 0) & ~(ret.shift(3) > 0)
    down3 = (ret < 0) & (ret.shift(1) < 0) & (ret.shift(2) < 0) & ~(ret.shift(3) < 0)
    for name, reverse in (("ATH-H1-3BAR-CONT-001", False), ("ATH-H1-3BAR-REV-001", True)):
        e = h1.copy()
        e["buy_signal"] = down3 if reverse else up3
        e["sell_signal"] = up3 if reverse else down3
        strategies[name] = to_m5(rows, e)

    inside = (h1["high"].shift(1) < h1["high"].shift(2)) & (h1["low"].shift(1) > h1["low"].shift(2))
    e = h1.copy()
    e["buy_signal"] = inside & (h1["close"] > h1["high"].shift(2))
    e["sell_signal"] = inside & (h1["close"] < h1["low"].shift(2))
    strategies["ATH-H1-INSIDE-001"] = to_m5(rows, e)

    outside = (h1["high"] > h1["high"].shift(1)) & (h1["low"] < h1["low"].shift(1))
    location = (h1["close"]-h1["low"]) / (h1["high"]-h1["low"]).replace(0, np.nan)
    e = h1.copy()
    e["buy_signal"] = outside & (location <= .20)
    e["sell_signal"] = outside & (location >= .80)
    strategies["ATH-H1-OUTSIDE-REV-001"] = to_m5(rows, e)

    ranges = h1["high"] - h1["low"]
    med = ranges.shift(1).rolling(20).median()
    compressed = (ranges.shift(1) < med.shift(1)) & (ranges.shift(2) < med.shift(2)) & (ranges.shift(3) < med.shift(3))
    e = h1.copy()
    e["buy_signal"] = compressed & (h1["close"] > h1["high"].shift(1).rolling(3).max())
    e["sell_signal"] = compressed & (h1["close"] < h1["low"].shift(1).rolling(3).min())
    strategies["ATH-H1-COMPRESS-001"] = to_m5(rows, e)

    daily = resample_ohlc(rows, "1D")
    daily["prior_high"] = daily["high"].shift(1)
    daily["prior_low"] = daily["low"].shift(1)
    levels = daily[["time", "prior_high", "prior_low"]]
    hd = pd.merge_asof(h1.sort_values("time"), levels.sort_values("time"), on="time", direction="backward")
    hours = hd["time"].dt.hour.between(6, 16)
    e = hd.copy()
    e["buy_signal"] = hours & (e["close"] > e["prior_high"])
    e["sell_signal"] = hours & (e["close"] < e["prior_low"])
    strategies["ATH-D1-BREAKOUT-001"] = to_m5(rows, first_per_day(e))

    e = hd.copy()
    e["buy_signal"] = hours & (e["low"] < e["prior_low"]) & (e["close"] >= e["prior_low"])
    e["sell_signal"] = hours & (e["high"] > e["prior_high"]) & (e["close"] <= e["prior_high"])
    strategies["ATH-D1-FADE-001"] = to_m5(rows, first_per_day(e))

    session = h1.copy()
    session["date"] = session["time"].dt.floor("D")
    close6 = session[session["time"].dt.hour == 6][["date", "close"]].rename(columns={"close":"close6"})
    e = session.merge(close6, on="date", how="left")
    at8 = e["time"].dt.hour == 8
    for name, reverse in (("ATH-LONDON-MOM-001", False), ("ATH-LONDON-REV-001", True)):
        s = e.copy()
        up = at8 & (s["close"] > s["close6"])
        down = at8 & (s["close"] < s["close6"])
        s["buy_signal"] = down if reverse else up
        s["sell_signal"] = up if reverse else down
        strategies[name] = to_m5(rows, s)
    return strategies


def main():
    print("=== Final preregistered historical batch 012-024 ===")
    rows = load_rows()
    strategies = build_all_signals(rows)
    results, months = [], []
    for experiment_id, signaled in strategies.items():
        frames, fold_metrics = [], []
        for fold, test in enumerate(fold_tests(signaled), 1):
            trades = simulate(test, fold)
            value = metrics(trades)
            fold_metrics.append(value)
            if not trades.empty:
                frames.append(trades)
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        overall = metrics(combined)
        positive_folds = sum(x["total_r"] > 0 for x in fold_metrics)
        minimum = 100 if experiment_id.startswith("ATH-H4") else 200
        passed = (
            overall["trades"] >= minimum and positive_folds >= 3
            and overall["pf"] >= 1.20 and overall["lcb95"] > 0
            and overall["max_dd"] <= 15
        )
        result = {"experiment_id": experiment_id, **overall,
                  "positive_folds": positive_folds, "minimum_trades": minimum,
                  "primary_gates_passed": passed}
        for index, fold_value in enumerate(fold_metrics, 1):
            result[f"fold_{index}_r"] = fold_value["total_r"]
            result[f"fold_{index}_pf"] = fold_value["pf"]
        results.append(result)
        if not combined.empty:
            monthly = combined.assign(month=combined["time"].dt.strftime("%Y-%m")).groupby("month").agg(trades=("actual_r","size"), total_r=("actual_r","sum")).reset_index()
            monthly.insert(0, "experiment_id", experiment_id)
            months.append(monthly)
        print(
            f"{experiment_id}: {overall['trades']} trades, BUY/SELL "
            f"{overall['buy_count']}/{overall['sell_count']}, {overall['total_r']:+.2f}R, "
            f"PF {overall['pf']:.2f}, LCB {overall['lcb95']:+.3f}, "
            f"DD {overall['max_dd']:.2f}, folds {positive_folds}/4, PASS {passed}"
        )
    pd.DataFrame(results).to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    if months:
        pd.concat(months, ignore_index=True).to_csv(MONTHLY_PATH, index=False, encoding="utf-8-sig")
    print(f"Results: {RESULT_PATH.resolve()}")
    print(f"Monthly: {MONTHLY_PATH.resolve()}")


if __name__ == "__main__":
    main()
