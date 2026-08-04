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
PRIMARY_Z = 2.0
ROBUSTNESS_Z = (1.8, 2.2)


def build_signals(rows: pd.DataFrame, z_threshold: float) -> pd.DataFrame:
    """過去20本だけからH1終値の初回分布逸脱を判定する。"""
    h1 = resample_ohlc(rows, "1h")
    prior_mean = h1["close"].shift(1).rolling(LOOKBACK).mean()
    prior_std = h1["close"].shift(1).rolling(LOOKBACK).std(ddof=1)
    h1["z_score"] = (h1["close"] - prior_mean) / prior_std
    previous_z = h1["z_score"].shift(1)
    h1["buy_signal"] = (previous_z >= -z_threshold) & (h1["z_score"] < -z_threshold)
    h1["sell_signal"] = (previous_z <= z_threshold) & (h1["z_score"] > z_threshold)
    signals = h1.loc[
        h1["buy_signal"] | h1["sell_signal"],
        ["time", "buy_signal", "sell_signal"],
    ]
    return rows.merge(signals, on="time", how="left").assign(
        buy_signal=lambda x: x["buy_signal"].fillna(False).astype(bool),
        sell_signal=lambda x: x["sell_signal"].fillna(False).astype(bool),
    )


def evaluate(rows, z_threshold, verbose):
    signaled = build_signals(rows, z_threshold)
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


def main():
    print("=== ATH-H1-MEANREV-001 ===")
    print(f"H1 20-bar distributional first excursion / cost {ADDITIONAL_COST_R:.2f}R")
    rows = load_rows()
    print(f"データ: {len(rows):,} M5行 / 欠損時刻 {rows['time'].isna().sum()} / 重複 {rows['time'].duplicated().sum()}")

    print("\n主条件（2.0 sigma）:")
    overall, folds, trades = evaluate(rows, PRIMARY_Z, True)
    positives = sum(item["total_r"] > 0 for item in folds)
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
    for z_value in ROBUSTNESS_Z:
        result, neighbor_folds, _ = evaluate(rows, z_value, False)
        neighbor_positives = sum(item["total_r"] > 0 for item in neighbor_folds)
        stable.append(result["pf"] >= 1.20 and neighbor_positives >= 3)
        print(
            f"{z_value:.1f} sigma: {result['trades']}回, {result['total_r']:+.2f}R, "
            f"PF {result['pf']:.2f}, DD {result['max_dd']:.2f}R, プラスFold {neighbor_positives}/4"
        )

    if overall["total_r"] > 0:
        fold_concentration = max(max(0.0, x["total_r"]) for x in folds) / overall["total_r"]
        monthly = trades.assign(month=trades["time"].dt.strftime("%Y-%m")).groupby("month")["actual_r"].sum().clip(lower=0)
        month_concentration = float(monthly.max() / overall["total_r"])
    else:
        fold_concentration = month_concentration = float("inf")
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
