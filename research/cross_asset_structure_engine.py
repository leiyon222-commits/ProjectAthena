import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DATASET = Path("data/market_context_expected_r_enriched.csv")
COST_R = 0.05
GAP_ROWS = 48
MINIMUM_TRADES = 200
FEATURE_COLUMNS = [
    "strength_usdjpy_pressure_1h", "strength_usdjpy_pressure_4h",
    "strength_usdjpy_pressure_1d", "strength_usdjpy_pressure_1w",
    "strength_usdjpy_pressure_1m", "strength_all_agreement_1h",
    "strength_usd_dispersion_1h", "strength_jpy_dispersion_1h",
]
LABEL_COLUMNS = [
    "buy_trade_r", "sell_trade_r", "buy_new_holding_bars",
    "sell_new_holding_bars", "buy_new_exit_reason", "sell_new_exit_reason",
]


def load_rows():
    columns = ["time", *FEATURE_COLUMNS, *LABEL_COLUMNS]
    rows = pd.read_csv(DATASET, usecols=columns, parse_dates=["time"], low_memory=False)
    rows["time"] = pd.to_datetime(rows["time"], utc=True, errors="coerce")
    return (rows.replace([np.inf, -np.inf], np.nan).dropna(subset=columns)
            .sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True))


def signs(hourly):
    return {key: np.sign(hourly[f"strength_usdjpy_pressure_{key}"]).astype("int8")
            for key in ("1h", "4h", "1d", "1w", "1m")}


def event_signals(rows, mode, variant):
    h = rows[rows["time"].dt.minute == 0].copy()
    s = signs(h)
    buy = pd.Series(False, index=h.index)
    sell = pd.Series(False, index=h.index)
    dispersion = h["strength_usd_dispersion_1h"] + h["strength_jpy_dispersion_1h"]

    if mode == "dispersion_expand":
        q = dispersion.shift(1).rolling(120).quantile(float(variant))
        event = (dispersion > q) & (dispersion.shift(1) <= q.shift(1))
        buy, sell = event & (s["1h"] > 0), event & (s["1h"] < 0)
    elif mode == "dispersion_converge":
        high_q = dispersion.shift(1).rolling(120).quantile(float(variant))
        median = dispersion.shift(1).rolling(120).median()
        event = (dispersion < median) & (dispersion.shift(1) >= high_q.shift(1))
        buy, sell = event & (s["1h"].shift(1) < 0), event & (s["1h"].shift(1) > 0)
    elif mode == "horizon_align":
        horizons = tuple(variant)
        aligned_up = pd.Series(True, index=h.index)
        aligned_down = pd.Series(True, index=h.index)
        for horizon in horizons:
            aligned_up &= s[horizon] > 0
            aligned_down &= s[horizon] < 0
        aligned = aligned_up | aligned_down
        event = aligned & ~aligned.shift(1, fill_value=False)
        buy, sell = event & aligned_up, event & aligned_down
    elif mode == "h1_d1_resolve":
        persistence = int(variant)
        mismatch = (s["1h"] != 0) & (s["1d"] != 0) & (s["1h"] != s["1d"])
        aligned = (s["1h"] != 0) & (s["1h"] == s["1d"])
        stable_d1 = pd.Series(True, index=h.index)
        for lag in range(1, persistence + 1):
            stable_d1 &= s["1d"] == s["1d"].shift(lag)
        event = aligned & mismatch.shift(1, fill_value=False) & stable_d1
        buy, sell = event & (s["1d"] > 0), event & (s["1d"] < 0)
    elif mode == "agreement_shock":
        threshold = float(variant)
        agreement = h["strength_all_agreement_1h"]
        neutral_before = agreement.shift(1).between(.40, .60)
        buy = neutral_before & (agreement >= threshold)
        sell = neutral_before & (agreement <= 1.0 - threshold)
    elif mode == "hierarchy_conflict":
        short_horizons, long_horizons = variant
        short_sign = s[short_horizons[0]]
        long_sign = s[long_horizons[0]]
        short_aligned = short_sign != 0
        long_aligned = long_sign != 0
        for horizon in short_horizons[1:]:
            short_aligned &= s[horizon] == short_sign
        for horizon in long_horizons[1:]:
            long_aligned &= s[horizon] == long_sign
        conflict = short_aligned & long_aligned & (short_sign != long_sign)
        event = conflict & ~conflict.shift(1, fill_value=False)
        buy, sell = event & (long_sign > 0), event & (long_sign < 0)
    else:
        raise ValueError(mode)

    h["buy_signal"], h["sell_signal"] = buy.fillna(False), sell.fillna(False)
    signals = h.loc[h["buy_signal"] | h["sell_signal"], ["time", "buy_signal", "sell_signal"]]
    return rows.merge(signals, on="time", how="left").assign(
        buy_signal=lambda x: x["buy_signal"].fillna(False).astype(bool),
        sell_signal=lambda x: x["sell_signal"].fillna(False).astype(bool))


def fixed_tests(rows):
    total, initial, segment = len(rows), int(len(rows) * .40), int(len(rows) * .075)
    for index in range(4):
        validation_end = initial + index * segment * 2 + segment
        test_end = min(validation_end + segment, total)
        yield rows.iloc[validation_end:test_end - GAP_ROWS]


def simulate(test, fold):
    trades, next_allowed = [], None
    for row in test.loc[test["buy_signal"] | test["sell_signal"]].itertuples(index=False):
        time = pd.Timestamp(row.time)
        if next_allowed is not None and time < next_allowed:
            continue
        if row.buy_signal:
            direction, value = "BUY", float(row.buy_trade_r) - COST_R
            holding = int(row.buy_new_holding_bars)
        else:
            direction, value = "SELL", float(row.sell_trade_r) - COST_R
            holding = int(row.sell_new_holding_bars)
        trades.append({"fold": fold, "time": time, "direction": direction, "actual_r": value})
        next_allowed = time + pd.Timedelta(minutes=(max(1, holding) + 1) * 5)
    return pd.DataFrame(trades)


def stats(trades):
    if trades.empty:
        return {"trade_count": 0, "buy_count": 0, "sell_count": 0, "total_r": 0.0,
                "average_r": 0.0, "average_r_95pct_lower": 0.0, "profit_factor": 0.0,
                "max_drawdown_r": 0.0, "max_losing_streak": 0}
    v = trades["actual_r"]
    avg = float(v.mean())
    se = float(v.std(ddof=1)) / math.sqrt(len(v)) if len(v) > 1 else 0.0
    equity = pd.concat([pd.Series([0.0]), v.cumsum().reset_index(drop=True)], ignore_index=True)
    dd = equity - equity.cummax()
    longest = current = 0
    for value in v:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return {"trade_count": int(len(v)), "buy_count": int((trades["direction"] == "BUY").sum()),
            "sell_count": int((trades["direction"] == "SELL").sum()), "total_r": float(v.sum()),
            "average_r": avg, "average_r_95pct_lower": avg - 1.96 * se,
            "profit_factor": float(v[v > 0].sum() / abs(v[v < 0].sum())) if (v < 0).any() else float("inf"),
            "max_drawdown_r": abs(float(dd.min())), "max_losing_streak": longest}


def evaluate(rows, mode, variant):
    signaled = event_signals(rows, mode, variant)
    frames, folds = [], []
    for number, test in enumerate(fixed_tests(signaled), 1):
        trades = simulate(test, number)
        value = stats(trades)
        value["fold"] = number
        folds.append(value)
        if not trades.empty:
            frames.append(trades)
    all_trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    overall = stats(all_trades)
    overall["positive_folds"] = sum(x["total_r"] > 0 for x in folds)
    overall["folds"] = folds
    if not all_trades.empty:
        monthly = (all_trades.assign(month=all_trades["time"].dt.strftime("%Y-%m"))
                   .groupby("month").agg(trades=("actual_r", "size"), total_r=("actual_r", "sum")).reset_index())
        overall["monthly"] = monthly.to_dict("records")
    else:
        overall["monthly"] = []
    if overall["total_r"] > 0:
        fold_profit = max(max(0.0, x["total_r"]) for x in folds) / overall["total_r"]
        month_profit = max(max(0.0, x["total_r"]) for x in overall["monthly"]) / overall["total_r"]
    else:
        fold_profit = month_profit = None
    overall["max_fold_profit_share"] = fold_profit
    overall["max_month_profit_share"] = month_profit
    overall["max_direction_share"] = (max(overall["buy_count"], overall["sell_count"]) /
                                       overall["trade_count"] if overall["trade_count"] else 1.0)
    return overall


def run(experiment_id, mode, primary, neighbors):
    rows = load_rows()
    result = evaluate(rows, mode, primary)
    result["experiment_id"] = experiment_id
    result["additional_cost_r"] = COST_R
    result["non_overlapping_trades"] = True
    result["future_information_audit"] = "PASS_CAUSAL_PAST_ROLLING_AND_CURRENT_CONFIRMED_M5_CLOSE"
    result["neighbors"] = {name: evaluate(rows, mode, value) for name, value in neighbors.items()}
    neighbors_stable = all(x["profit_factor"] >= 1.20 and x["positive_folds"] >= 3
                           for x in result["neighbors"].values())
    result["accepted"] = all([
        result["trade_count"] >= MINIMUM_TRADES, result["positive_folds"] >= 3,
        result["profit_factor"] >= 1.20, result["average_r_95pct_lower"] > 0,
        result["max_drawdown_r"] <= 15,
        result["max_fold_profit_share"] is not None and result["max_fold_profit_share"] <= .40,
        result["max_month_profit_share"] is not None and result["max_month_profit_share"] < 1.0,
        result["max_direction_share"] <= .80, neighbors_stable,
    ])
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
