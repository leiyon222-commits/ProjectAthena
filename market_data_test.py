from datetime import datetime
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "USDJPY"
TIMEFRAME = mt5.TIMEFRAME_M5
BAR_COUNT = 100


def main() -> None:
    if not mt5.initialize():
        print("MT5への接続に失敗しました")
        print("エラー:", mt5.last_error())
        return

    try:
        # 気配値表示で使用できる状態にする
        if not mt5.symbol_select(SYMBOL, True):
            print(f"{SYMBOL}を選択できませんでした")
            print("エラー:", mt5.last_error())
            return

        # 現在価格を取得
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            print("現在価格を取得できませんでした")
            print("エラー:", mt5.last_error())
            return

        print("=== 現在価格 ===")
        print("銘柄:", SYMBOL)
        print("時刻:", datetime.fromtimestamp(tick.time))
        print("Bid:", tick.bid)
        print("Ask:", tick.ask)
        print("Spread:", round(tick.ask - tick.bid, 5))

        # M5ローソク足を取得
        rates = mt5.copy_rates_from_pos(
            SYMBOL,
            TIMEFRAME,
            0,
            BAR_COUNT,
        )

        if rates is None or len(rates) == 0:
            print("ローソク足を取得できませんでした")
            print("エラー:", mt5.last_error())
            return

        candles = pd.DataFrame(rates)
        candles["time"] = pd.to_datetime(candles["time"], unit="s")

        print("\n=== 最新5本のM5ローソク足 ===")
        print(
            candles[
                [
                    "time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "tick_volume",
                    "spread",
                ]
            ].tail()
        )

        # dataフォルダを自動作成
        output_path = Path("data/usdjpy_m5_latest.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # CSV保存
        candles.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"\nCSV保存完了: {output_path}")
        print(f"取得本数: {len(candles)}本")

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()