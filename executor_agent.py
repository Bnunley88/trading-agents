import os
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
from datetime import date

load_dotenv()

api = tradeapi.REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    os.getenv("ALPACA_BASE_URL")
)

def already_traded_today(symbol):
    today = str(date.today())
    try:
        orders = api.list_orders(status='all', limit=10)
        for order in orders:
            order_date = order.created_at.strftime("%Y-%m-%d")
            if order.symbol == symbol and order_date == today:
                print(f"EXECUTOR: Already traded {symbol} today — skipping.")
                return True
    except Exception as e:
        print(f"EXECUTOR CHECK ERROR: {e}")
    return False

def execute_trade(symbol, shares):
    if already_traded_today(symbol):
        return None
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=shares,
            side='buy',
            type='market',
            time_in_force='day'
        )
        print(f"EXECUTOR AGENT: Order placed!")
        print(f"Symbol: {symbol}")
        print(f"Shares: {shares}")
        print(f"Order ID: {order.id}")
        return order
    except Exception as e:
        print(f"EXECUTOR AGENT ERROR: {e}")
        return None

if __name__ == "__main__":
    execute_trade("MSFT", 1)