import MetaTrader5 as mt5


def main() -> None:
    if not mt5.initialize():
        print("MT5への接続に失敗しました")
        print("エラー:", mt5.last_error())
        return

    try:
        print("MT5接続成功")
        print("MT5バージョン:", mt5.version())

        account = mt5.account_info()
        if account is None:
            print("口座情報を取得できませんでした")
            print("エラー:", mt5.last_error())
            return

        print("口座番号:", account.login)
        print("サーバー:", account.server)
        print("残高:", account.balance)
        print("証拠金通貨:", account.currency)

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()