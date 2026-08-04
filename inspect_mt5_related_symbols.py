from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

import MetaTrader5 as mt5
import pandas as pd


OUTPUT_PATH = Path(
    "data/mt5_related_symbol_inventory.csv"
)

# Athenaで追加候補にしたい関連市場。
# ブローカーごとの別名も同じtargetにまとめる。
TARGET_GROUPS = [
    {
        "category": "USD_PAIRS",
        "target": "EURUSD",
        "aliases": ["EURUSD"],
    },
    {
        "category": "USD_PAIRS",
        "target": "GBPUSD",
        "aliases": ["GBPUSD"],
    },
    {
        "category": "USD_PAIRS",
        "target": "AUDUSD",
        "aliases": ["AUDUSD"],
    },
    {
        "category": "USD_PAIRS",
        "target": "NZDUSD",
        "aliases": ["NZDUSD"],
    },
    {
        "category": "USD_PAIRS",
        "target": "USDCAD",
        "aliases": ["USDCAD"],
    },
    {
        "category": "USD_PAIRS",
        "target": "USDCHF",
        "aliases": ["USDCHF"],
    },
    {
        "category": "JPY_CROSSES",
        "target": "EURJPY",
        "aliases": ["EURJPY"],
    },
    {
        "category": "JPY_CROSSES",
        "target": "GBPJPY",
        "aliases": ["GBPJPY"],
    },
    {
        "category": "JPY_CROSSES",
        "target": "AUDJPY",
        "aliases": ["AUDJPY"],
    },
    {
        "category": "JPY_CROSSES",
        "target": "NZDJPY",
        "aliases": ["NZDJPY"],
    },
    {
        "category": "JPY_CROSSES",
        "target": "CADJPY",
        "aliases": ["CADJPY"],
    },
    {
        "category": "JPY_CROSSES",
        "target": "CHFJPY",
        "aliases": ["CHFJPY"],
    },
    {
        "category": "COMMODITY",
        "target": "XAUUSD",
        "aliases": [
            "XAUUSD",
            "GOLD",
        ],
    },
    {
        "category": "DOLLAR_INDEX",
        "target": "DXY",
        "aliases": [
            "DXY",
            "USDX",
            "USDINDEX",
            "DOLLARINDEX",
        ],
    },
    {
        "category": "US_STOCK_INDEX",
        "target": "US500",
        "aliases": [
            "US500",
            "SPX500",
            "SP500",
            "USA500",
        ],
    },
    {
        "category": "US_STOCK_INDEX",
        "target": "NAS100",
        "aliases": [
            "NAS100",
            "US100",
            "USTEC",
            "NASDAQ100",
        ],
    },
    {
        "category": "JAPAN_STOCK_INDEX",
        "target": "JP225",
        "aliases": [
            "JP225",
            "JPN225",
            "NIKKEI225",
            "NI225",
        ],
    },
]

# 既存USDJPYデータ開始付近に履歴があるかを見る。
OLD_WINDOW_START = datetime(
    2021,
    8,
    9,
    tzinfo=timezone.utc,
)

OLD_WINDOW_END = OLD_WINDOW_START + timedelta(
    days=7
)

NOW_UTC = datetime.now(
    timezone.utc
)

RECENT_WINDOW_START = NOW_UTC - timedelta(
    days=7
)

D1_HISTORY_START = datetime(
    2021,
    8,
    1,
    tzinfo=timezone.utc,
)


def normalize_symbol_name(
    value: str,
) -> str:
    """記号を除き、大文字英数字だけにする。"""
    return re.sub(
        r"[^A-Z0-9]",
        "",
        value.upper(),
    )


def find_candidates(
    all_symbols: tuple,
    aliases: list[str],
) -> list:
    """別名を含むMT5銘柄を検索する。"""
    normalized_aliases = [
        normalize_symbol_name(alias)
        for alias in aliases
    ]

    candidates = []

    for symbol in all_symbols:
        normalized_name = (
            normalize_symbol_name(
                symbol.name
            )
        )

        matched_alias = None

        for alias in normalized_aliases:
            if alias in normalized_name:
                matched_alias = alias
                break

        if matched_alias is None:
            continue

        # 完全一致を最優先し、その後は
        # Market Watch表示中・名前の短さで選ぶ。
        exact_match = int(
            normalized_name
            == matched_alias
        )

        candidates.append(
            {
                "symbol": symbol,
                "matched_alias": (
                    matched_alias
                ),
                "exact_match": exact_match,
                "visible": int(
                    bool(symbol.visible)
                ),
                "selected": int(
                    bool(symbol.select)
                ),
                "name_length": len(
                    symbol.name
                ),
            }
        )

    candidates.sort(
        key=lambda item: (
            -item["exact_match"],
            -item["visible"],
            -item["selected"],
            item["name_length"],
            item["symbol"].name,
        )
    )

    return candidates


def rates_summary(
    symbol_name: str,
    timeframe: int,
    date_from: datetime,
    date_to: datetime,
) -> dict:
    """指定期間のローソク足本数と範囲を返す。"""
    rates = mt5.copy_rates_range(
        symbol_name,
        timeframe,
        date_from,
        date_to,
    )

    if rates is None:
        return {
            "bars": 0,
            "start": "",
            "end": "",
            "error": str(
                mt5.last_error()
            ),
        }

    if len(rates) == 0:
        return {
            "bars": 0,
            "start": "",
            "end": "",
            "error": "",
        }

    times = pd.to_datetime(
        rates["time"],
        unit="s",
        utc=True,
    )

    return {
        "bars": int(
            len(rates)
        ),
        "start": str(
            times.min()
        ),
        "end": str(
            times.max()
        ),
        "error": "",
    }


def inspect_target(
    category: str,
    target: str,
    aliases: list[str],
    all_symbols: tuple,
) -> dict:
    """対象市場の利用可能銘柄と履歴を確認する。"""
    candidates = find_candidates(
        all_symbols,
        aliases,
    )

    if not candidates:
        return {
            "category": category,
            "target": target,
            "selected_symbol": "",
            "all_candidates": "",
            "description": "",
            "path": "",
            "visible_before": False,
            "symbol_select_success": False,
            "m5_old_bars": 0,
            "m5_old_start": "",
            "m5_old_end": "",
            "m5_recent_bars": 0,
            "m5_recent_start": "",
            "m5_recent_end": "",
            "d1_bars_since_2021": 0,
            "d1_start": "",
            "d1_end": "",
            "usable_for_athena": False,
            "error": "候補銘柄なし",
        }

    best = candidates[0]
    symbol = best["symbol"]
    symbol_name = symbol.name

    select_success = mt5.symbol_select(
        symbol_name,
        True,
    )

    if not select_success:
        return {
            "category": category,
            "target": target,
            "selected_symbol": (
                symbol_name
            ),
            "all_candidates": ", ".join(
                item["symbol"].name
                for item in candidates
            ),
            "description": (
                symbol.description or ""
            ),
            "path": (
                symbol.path or ""
            ),
            "visible_before": bool(
                symbol.visible
            ),
            "symbol_select_success": False,
            "m5_old_bars": 0,
            "m5_old_start": "",
            "m5_old_end": "",
            "m5_recent_bars": 0,
            "m5_recent_start": "",
            "m5_recent_end": "",
            "d1_bars_since_2021": 0,
            "d1_start": "",
            "d1_end": "",
            "usable_for_athena": False,
            "error": (
                "symbol_select失敗: "
                f"{mt5.last_error()}"
            ),
        }

    old_m5 = rates_summary(
        symbol_name=symbol_name,
        timeframe=mt5.TIMEFRAME_M5,
        date_from=OLD_WINDOW_START,
        date_to=OLD_WINDOW_END,
    )

    recent_m5 = rates_summary(
        symbol_name=symbol_name,
        timeframe=mt5.TIMEFRAME_M5,
        date_from=RECENT_WINDOW_START,
        date_to=NOW_UTC,
    )

    daily = rates_summary(
        symbol_name=symbol_name,
        timeframe=mt5.TIMEFRAME_D1,
        date_from=D1_HISTORY_START,
        date_to=NOW_UTC,
    )

    error_parts = [
        value
        for value in [
            old_m5["error"],
            recent_m5["error"],
            daily["error"],
        ]
        if value
    ]

    usable = bool(
        old_m5["bars"] > 0
        and recent_m5["bars"] > 0
        and daily["bars"] > 0
    )

    return {
        "category": category,
        "target": target,
        "selected_symbol": (
            symbol_name
        ),
        "all_candidates": ", ".join(
            item["symbol"].name
            for item in candidates
        ),
        "description": (
            symbol.description or ""
        ),
        "path": (
            symbol.path or ""
        ),
        "visible_before": bool(
            symbol.visible
        ),
        "symbol_select_success": True,
        "m5_old_bars": (
            old_m5["bars"]
        ),
        "m5_old_start": (
            old_m5["start"]
        ),
        "m5_old_end": (
            old_m5["end"]
        ),
        "m5_recent_bars": (
            recent_m5["bars"]
        ),
        "m5_recent_start": (
            recent_m5["start"]
        ),
        "m5_recent_end": (
            recent_m5["end"]
        ),
        "d1_bars_since_2021": (
            daily["bars"]
        ),
        "d1_start": daily["start"],
        "d1_end": daily["end"],
        "usable_for_athena": usable,
        "error": " | ".join(
            error_parts
        ),
    }


def main() -> None:
    print(
        "=== MT5 関連銘柄・履歴確認 ==="
    )

    if not mt5.initialize():
        print(
            "MT5へ接続できませんでした"
        )
        print(
            "エラー:",
            mt5.last_error(),
        )
        return

    try:
        terminal = mt5.terminal_info()
        account = mt5.account_info()
        all_symbols = mt5.symbols_get()

        if all_symbols is None:
            print(
                "銘柄一覧を取得できませんでした"
            )
            print(
                "エラー:",
                mt5.last_error(),
            )
            return

        print(
            f"MT5パッケージ: "
            f"{mt5.__version__}"
        )

        if terminal is not None:
            print(
                f"会社: "
                f"{terminal.company}"
            )

        if account is not None:
            print(
                f"サーバー: "
                f"{account.server}"
            )

        print(
            f"MT5内の全銘柄: "
            f"{len(all_symbols):,}件"
        )

        print(
            "\n関連銘柄を確認中..."
        )

        rows = []

        for index, group in enumerate(
            TARGET_GROUPS,
            start=1,
        ):
            result = inspect_target(
                category=group["category"],
                target=group["target"],
                aliases=group["aliases"],
                all_symbols=all_symbols,
            )

            rows.append(result)

            selected_name = (
                result["selected_symbol"]
                or "見つからない"
            )

            status = (
                "利用可能"
                if result[
                    "usable_for_athena"
                ]
                else "要確認"
            )

            print(
                f"{index:02d}. "
                f"{group['target']:<8} "
                f"→ {selected_name:<20} "
                f"[{status}]"
            )

        result_frame = pd.DataFrame(
            rows
        )

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        result_frame.to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "\n=== Athenaで利用可能 ==="
        )

        usable = result_frame[
            result_frame[
                "usable_for_athena"
            ]
        ]

        if usable.empty:
            print(
                "5年前・直近M5・D1の"
                "すべてを確認できた銘柄は"
                "ありませんでした。"
            )
        else:
            print(
                usable[
                    [
                        "category",
                        "target",
                        "selected_symbol",
                        "m5_old_bars",
                        "m5_recent_bars",
                        "d1_bars_since_2021",
                    ]
                ].to_string(
                    index=False
                )
            )

        print(
            "\n=== 見つからない／履歴不足 ==="
        )

        unavailable = result_frame[
            ~result_frame[
                "usable_for_athena"
            ]
        ]

        if unavailable.empty:
            print(
                "なし"
            )
        else:
            print(
                unavailable[
                    [
                        "target",
                        "selected_symbol",
                        "all_candidates",
                        "m5_old_bars",
                        "m5_recent_bars",
                        "d1_bars_since_2021",
                        "error",
                    ]
                ].to_string(
                    index=False
                )
            )

        print(
            "\n保存先:",
            OUTPUT_PATH.resolve(),
        )

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
