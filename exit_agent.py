import os
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

load_dotenv()

api = tradeapi.REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    os.getenv("ALPACA_BASE_URL")
)

def exit_trade(symbol):
    try:
        api.close_position(symbol)
        print(f"EXIT AGENT: Position closed!")
        print(f"Symbol: {symbol} sold successfully")
    except Exception as e:
        print(f"EXIT AGENT ERROR: {e}")

if __name__ == "__main__":
    exit_trade("MSFT")