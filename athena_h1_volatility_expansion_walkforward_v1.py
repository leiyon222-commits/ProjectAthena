import pandas as pd

from athena_h1_confirmed_breakout_walkforward_v1 import (
    ADDITIONAL_COST_R,
    MINIMUM_TRADES,
    fold_tests,
    load_rows,
    metrics,
    print_period_breakdown,
    resample_ohlc,
    simulate,
)


LOOKBACK = 20
PRIMARY_MULTIPLE = 1.5
ROBUSTNESS_MULTIPLES = (1.4, 1.6)
CLOSE_EDGE_FRACTION = 0.20


def build_signals(rows: pd.DataFrame, multiple: float) -> pd.DataFrame:
    """過去中央値からの初回H1レンジ拡大を終値確定後に判定する。"""
    h1 = resample_ohlc(rows, "1h")
    previous_close = h1["close"].shift(1)
    true_range = pd.concat(
        [
            h1["high"] - h1["low"],
            (h1["high"] - previous_close).abs(),
            (h1["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    prior_median = true_range.shift(1).rolling(LOOKBACK).median()
    expanded = true_range > multiple * prior_median
    first_expansion = expanded & ~expanded.shift(1, fill_value=False)
    candle_range = (h1["high"] - h1["low"]).replace(0, pd.NA)
    close_location = (h1["close"] - h1["low"]) / candle_range

    h1["buy_signal"] = (
        first_expansion
        & (h1["close"] > h1["open"])
        & (close_location >= 1.0 - CLOSE_EDGE_FRACTION)
    )
    h1["sell_signal"] = (
        first_expansion
        & (h1["close"] < h1["open"])
        & (close_location <= CLOSE_EDGE_FRACTION)
    )
    signals = h1.loc[
        h1["buy_signal"] | h1["sell_signal"],
        ["time", "buy_signal", "sell_signal"],
    ]
    return rows.merge(signals, on="time", how="left").assign(
        buy_signal=lambda x: x["buy_signal"].fillna(False).astype(bool),
        sell_signal=lambda x: x["sell_signal"].fillna(False).astype(bool),
    )


def evaluate(rows, multiple, verbose):
    signaled = build_signals(rows, multiple)
    fold_results, trade_frames = [], []
    for fold, test in enumerate(fold_tests(signaled), start=1):
        trades = simulate(test, fold)
        result = metrics(trades)
        result["fold"] = fold
        result["signals"] = int((test["buy_signal"] | test["sell_signal"]).sum())
        fold_results.append(result)
        if not trades.empty:
            trade_frames.append(trades)
        if verbose:
            print(
                f"Fold {fold}: signals {result['signals']}, trades {result['trades']}, "
                f"BUY/SELL {result['buy_count']}/{result['sell_count']}, "
                f"total {result['total_r']:+.2f}R, PF {result['pf']:.2f}, "
                f"DD {result['max_dd']:.2f}R, losing streak {result['max_losing_streak']}"
            )
    combined = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    return metrics(combined), fold_results, combined


def concentration(overall, folds, trades):
    if overall["total_r"] <= 0:
        return float("inf"), float("inf")
    fold_value = max(max(0.0, x["total_r"]) for x in folds) / overall["total_r"]
    monthly = trades.assign(month=trades["time"].dt.strftime("%Y-%m")).groupby("month")["actual_r"].sum().clip(lower=0)
    return fold_value, float(monthly.max() / overall["total_r"])


def main():
    print("=== ATH-H1-EXPANSION-001 ===")
    print(f"H1 first volatility expansion / cost {ADDITIONAL_COST_R:.2f}R")
    rows = load_rows()
    print(f"データ: {len(rows):,} M5行 / 欠損時刻 {rows['time'].isna().sum()} / 重複 {rows['time'].duplicated().sum()}")

    print("\n主条件（1.5x prior median TR）:")
    overall, folds, trades = evaluate(rows, PRIMARY_MULTIPLE, True)
    positives = sum(x["total_r"] > 0 for x in folds)
    print(
        f"全体: {overall['trades']}回, BUY/SELL {overall['buy_count']}/{overall['sell_count']}, "
        f"勝率 {overall['win_rate']:.2f}%, 合計 {overall['total_r']:+.2f}R, "
        f"平均 {overall['average_r']:+.4f}R, 95%下限 {overall['lcb95']:+.4f}R, "
        f"PF {overall['pf']:.2f}, DD {overall['max_dd']:.2f}R, "
        f"最大連敗 {overall['max_losing_streak']}, プラスFold {positives}/4"
    )
    print_period_breakdown(trades)

    print("\n事前登録済み近傍安定性（選択には不使用）:")
    stable = []
    for value in ROBUSTNESS_MULTIPLES:
        result, neighbor_folds, _ = evaluate(rows, value, False)
        neighbor_positives = sum(x["total_r"] > 0 for x in neighbor_folds)
        stable.append(result["pf"] >= 1.20 and neighbor_positives >= 3)
        print(
            f"{value:.1f}x: {result['trades']}回, {result['total_r']:+.2f}R, "
            f"PF {result['pf']:.2f}, DD {result['max_dd']:.2f}R, プラスFold {neighbor_positives}/4"
        )

    fold_concentration, month_concentration = concentration(overall, folds, trades)
    direction_ratio = max(overall["buy_count"], overall["sell_count"]) / overall["trades"] if overall["trades"] else 1.0
    accepted = all([
        overall["trades"] >= MINIMUM_TRADES, positives >= 3,
        overall["pf"] >= 1.20, overall["lcb95"] > 0,
        overall["max_dd"] <= 15, fold_concentration <= 0.40,
        month_concentration < 1.0, direction_ratio <= 0.80, all(stable),
    ])
    print("\n判定:", "CANDIDATE_FOUND" if accepted else "REJECTED")
    print(
        f"監査値: fold利益集中 {fold_concentration:.3f}, 月利益集中 {month_concentration:.3f}, "
        f"最大方向比率 {direction_ratio:.3f}, 近傍安定 {all(stable)}"
    )


if __name__ == "__main__":
    main()
