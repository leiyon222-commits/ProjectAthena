import numpy as np
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


PRIMARY_BOUNDS = (0.5, 2.0)
ROBUSTNESS_BOUNDS = ((0.4, 2.2), (0.6, 1.8))
ASIA_HOURS = tuple(range(0, 7))
BREAKOUT_HOURS = tuple(range(7, 11))


def calculate_atr(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()


def build_session_signals(
    rows: pd.DataFrame,
    minimum_range_atr: float,
    maximum_range_atr: float,
) -> pd.DataFrame:
    """完全に観測済みの同日アジア時間レンジから最初の突破だけを作る。"""
    h1 = resample_ohlc(rows, "1h")
    h1["date"] = h1["time"].dt.floor("D")
    h1["hour"] = h1["time"].dt.hour
    # レンジ判定時点で現在H1のATRを使わない。
    h1["confirmed_atr14"] = calculate_atr(h1).shift(1)

    asia = h1[h1["hour"].isin(ASIA_HOURS)].groupby("date").agg(
        asia_high=("high", "max"),
        asia_low=("low", "min"),
        asia_bars=("time", "size"),
    )
    h1 = h1.merge(asia, on="date", how="left")
    h1["range_atr"] = (
        (h1["asia_high"] - h1["asia_low"]) / h1["confirmed_atr14"]
    )

    window = h1["hour"].isin(BREAKOUT_HOURS) & (h1["asia_bars"] == len(ASIA_HOURS))
    range_ok = h1["range_atr"].between(
        minimum_range_atr, maximum_range_atr, inclusive="both"
    )
    h1["buy_signal"] = window & range_ok & (h1["close"] > h1["asia_high"])
    h1["sell_signal"] = window & range_ok & (h1["close"] < h1["asia_low"])

    events = h1[h1["buy_signal"] | h1["sell_signal"]].copy()
    events = events.sort_values("time").groupby("date", as_index=False).first()
    signals = events[["time", "buy_signal", "sell_signal"]]
    return rows.merge(signals, on="time", how="left").assign(
        buy_signal=lambda x: x["buy_signal"].fillna(False).astype(bool),
        sell_signal=lambda x: x["sell_signal"].fillna(False).astype(bool),
    )


def evaluate(rows, bounds, verbose):
    signaled = build_session_signals(rows, bounds[0], bounds[1])
    fold_results = []
    trade_frames = []
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
    print("=== ATH-H1-SESSION-001 ===")
    print("Confirmed Asian range / first London-window H1 close breakout")
    print(f"追加費用: {ADDITIONAL_COST_R:.2f}R/取引")
    rows = load_rows()
    print(f"データ: {len(rows):,} M5行 / 欠損時刻 {rows['time'].isna().sum()} / 重複 {rows['time'].duplicated().sum()}")

    print("\n主条件（range/ATR 0.5～2.0）:")
    overall, folds, trades = evaluate(rows, PRIMARY_BOUNDS, True)
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
    neighbor_passes = []
    for bounds in ROBUSTNESS_BOUNDS:
        result, neighbor_folds, _ = evaluate(rows, bounds, False)
        neighbor_positives = sum(item["total_r"] > 0 for item in neighbor_folds)
        neighbor_passes.append(result["pf"] >= 1.20 and neighbor_positives >= 3)
        print(
            f"{bounds[0]:.1f}～{bounds[1]:.1f}: {result['trades']}回, "
            f"{result['total_r']:+.2f}R, PF {result['pf']:.2f}, "
            f"DD {result['max_dd']:.2f}R, プラスFold {neighbor_positives}/4"
        )

    if overall["total_r"] > 0:
        fold_profit = [max(0.0, item["total_r"]) for item in folds]
        fold_concentration = max(fold_profit) / overall["total_r"]
        month_profit = trades.assign(month=trades["time"].dt.strftime("%Y-%m")).groupby("month")["actual_r"].sum().clip(lower=0)
        month_concentration = float(month_profit.max() / overall["total_r"])
    else:
        fold_concentration = month_concentration = float("inf")
    direction_ratio = max(overall["buy_count"], overall["sell_count"]) / overall["trades"] if overall["trades"] else 1.0
    accepted = all([
        overall["trades"] >= MINIMUM_TRADES,
        positives >= 3,
        overall["pf"] >= 1.20,
        overall["lcb95"] > 0,
        overall["max_dd"] <= 15,
        fold_concentration <= 0.40,
        month_concentration < 1.0,
        direction_ratio <= 0.80,
        all(neighbor_passes),
    ])
    print("\n判定:", "CANDIDATE_FOUND" if accepted else "REJECTED")
    print(
        f"監査値: fold利益集中 {fold_concentration:.3f}, 月利益集中 {month_concentration:.3f}, "
        f"最大方向比率 {direction_ratio:.3f}, 近傍安定 {all(neighbor_passes)}"
    )


if __name__ == "__main__":
    main()
