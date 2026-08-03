from datetime import datetime
from pathlib import Path
import sqlite3

import MetaTrader5 as mt5
import pandas as pd
import streamlit as st


SYMBOL = "USDJPY"
TIMEFRAME_NAME = "M5"
DATABASE_PATH = Path("data/athena.db")
SIMULATED_CAPITAL = 10_000

EMA_SHORT_PERIOD = 20
EMA_LONG_PERIOD = 50
RSI_PERIOD = 14
ATR_PERIOD = 14


def get_mt5_status() -> tuple[bool, str]:
    """MT5への接続状態を確認する。"""
    if not mt5.initialize():
        return False, f"接続失敗: {mt5.last_error()}"

    try:
        account = mt5.account_info()

        if account is None:
            return False, f"口座情報取得失敗: {mt5.last_error()}"

        return True, account.server

    finally:
        mt5.shutdown()


def get_current_price() -> dict[str, float | datetime] | None:
    """USDJPYの現在価格を取得する。"""
    if not mt5.initialize():
        return None

    try:
        if not mt5.symbol_select(SYMBOL, True):
            return None

        tick = mt5.symbol_info_tick(SYMBOL)

        if tick is None:
            return None

        return {
            "time": datetime.fromtimestamp(tick.time),
            "bid": float(tick.bid),
            "ask": float(tick.ask),
            "spread": float(tick.ask - tick.bid),
        }

    finally:
        mt5.shutdown()


def load_candles() -> tuple[int, pd.DataFrame]:
    """SQLiteから保存件数とローソク足を取得する。"""
    if not DATABASE_PATH.exists():
        return 0, pd.DataFrame()

    with sqlite3.connect(DATABASE_PATH) as connection:
        total_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM candles
            WHERE symbol = ? AND timeframe = ?
            """,
            (SYMBOL, TIMEFRAME_NAME),
        ).fetchone()[0]

        candles = pd.read_sql_query(
            """
            SELECT
                time,
                open,
                high,
                low,
                close,
                tick_volume,
                spread
            FROM candles
            WHERE symbol = ? AND timeframe = ?
            ORDER BY time ASC
            """,
            connection,
            params=(SYMBOL, TIMEFRAME_NAME),
        )

    if not candles.empty:
        candles["time"] = pd.to_datetime(
            candles["time"],
            unit="s",
        )

    return int(total_count), candles


def calculate_indicators(candles: pd.DataFrame) -> pd.DataFrame:
    """EMA・RSI・ATRを計算する。"""
    result = candles.copy()

    result["ema20"] = result["close"].ewm(
        span=EMA_SHORT_PERIOD,
        adjust=False,
    ).mean()

    result["ema50"] = result["close"].ewm(
        span=EMA_LONG_PERIOD,
        adjust=False,
    ).mean()

    price_change = result["close"].diff()

    gain = price_change.clip(lower=0)
    loss = -price_change.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False,
        min_periods=RSI_PERIOD,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False,
        min_periods=RSI_PERIOD,
    ).mean()

    relative_strength = average_gain / average_loss

    result["rsi14"] = 100 - (
        100 / (1 + relative_strength)
    )

    previous_close = result["close"].shift(1)

    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    result["atr14"] = true_range.ewm(
        alpha=1 / ATR_PERIOD,
        adjust=False,
        min_periods=ATR_PERIOD,
    ).mean()

    return result


def create_signal(
    latest: pd.Series,
) -> tuple[str, list[str]]:
    """テクニカル指標から仮の売買判断を作成する。"""
    reasons: list[str] = []

    ema20 = float(latest["ema20"])
    ema50 = float(latest["ema50"])
    rsi14 = float(latest["rsi14"])
    atr14 = float(latest["atr14"])

    if pd.isna(rsi14) or pd.isna(atr14):
        return "HOLD", ["指標の計算に必要なデータが不足しています"]

    if ema20 > ema50 and 50 <= rsi14 < 70:
        reasons.append("EMA20がEMA50を上回っています")
        reasons.append(f"RSI14が買い優勢の範囲です：{rsi14:.1f}")
        signal = "BUY"

    elif ema20 < ema50 and 30 < rsi14 <= 50:
        reasons.append("EMA20がEMA50を下回っています")
        reasons.append(f"RSI14が売り優勢の範囲です：{rsi14:.1f}")
        signal = "SELL"

    else:
        reasons.append("EMAとRSIの条件が揃っていません")
        signal = "HOLD"

    reasons.append(f"ATR14：{atr14:.3f}")

    return signal, reasons


@st.fragment(run_every="30s")
def live_dashboard() -> None:
    """30秒ごとに自動更新される表示部分。"""
    connected, connection_message = get_mt5_status()
    price = get_current_price()
    candle_count, candles = load_candles()

    status_column, capital_column, count_column = st.columns(3)

    with status_column:
        if connected:
            st.success("MT5：接続中")
            st.caption(connection_message)
        else:
            st.error("MT5：未接続")
            st.caption(connection_message)

    with capital_column:
        st.metric(
            label="Athena運用資金",
            value=f"¥{SIMULATED_CAPITAL:,}",
        )

    with count_column:
        st.metric(
            label="保存済みローソク足",
            value=f"{candle_count:,}本",
        )

    st.divider()

    st.subheader(f"{SYMBOL} 現在価格")

    if price is None:
        st.warning(
            "現在価格を取得できませんでした。"
            "MT5を確認してください。"
        )
    else:
        bid_column, ask_column, spread_column = st.columns(3)

        bid_column.metric(
            "Bid",
            f"{price['bid']:.3f}",
        )
        ask_column.metric(
            "Ask",
            f"{price['ask']:.3f}",
        )
        spread_column.metric(
            "Spread",
            f"{price['spread']:.3f}",
        )

        st.caption(
            "価格更新時刻："
            f"{price['time'].strftime('%Y-%m-%d %H:%M:%S')}"
        )

    st.divider()

    if candles.empty:
        st.warning(
            "ローソク足データがありません。"
            "auto_collector.pyを起動してください。"
        )

        return

    indicators = calculate_indicators(candles)
    latest = indicators.iloc[-1]

    signal, reasons = create_signal(latest)

    st.subheader("Athena暫定判断")

    if signal == "BUY":
        st.success("BUY")
    elif signal == "SELL":
        st.error("SELL")
    else:
        st.info("HOLD")

    for reason in reasons:
        st.write(f"・{reason}")

    st.divider()

    ema20_column, ema50_column, rsi_column, atr_column = st.columns(4)

    ema20_column.metric(
        "EMA20",
        f"{latest['ema20']:.3f}",
    )

    ema50_column.metric(
        "EMA50",
        f"{latest['ema50']:.3f}",
    )

    rsi_column.metric(
        "RSI14",
        f"{latest['rsi14']:.1f}",
    )

    atr_column.metric(
        "ATR14",
        f"{latest['atr14']:.3f}",
    )

    st.divider()

    latest_candles = indicators.tail(50).copy()

    st.subheader(f"最新の{TIMEFRAME_NAME}ローソク足")

    display_columns = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "spread",
        "ema20",
        "ema50",
        "rsi14",
        "atr14",
    ]

    st.dataframe(
        latest_candles[display_columns].tail(10),
        use_container_width=True,
        hide_index=True,
    )

    chart_data = latest_candles.set_index("time")

    st.subheader("終値・EMA推移")

    st.line_chart(
        chart_data[
            [
                "close",
                "ema20",
                "ema50",
            ]
        ]
    )

    st.divider()

    st.caption(
        "Dashboard表示更新："
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


def main() -> None:
    st.set_page_config(
        page_title="Project Athena",
        page_icon="📈",
        layout="wide",
    )

    st.title("Project Athena")
    st.caption("MT5 AI Trading Research Dashboard")

    st.warning(
        "現在のBUY・SELL・HOLDは検証用のルールベース判断です。"
        "注文は実行しません。"
    )

    live_dashboard()


if __name__ == "__main__":
    main()