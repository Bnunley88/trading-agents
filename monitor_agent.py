import os
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

load_dotenv()

api = tradeapi.REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    os.getenv("ALPACA_BASE_URL")
)

def monitor_position(symbol):
    try:
        position = api.get_position(symbol)
        entry_price = float(position.avg_entry_price)
        current_price = float(position.current_price)
        profit_loss = float(position.unrealized_pl)
        profit_loss_pct = float(position.unrealized_plpc) * 100

        print(f"MONITOR AGENT REPORT:")
        print(f"Symbol: {symbol}")
        print(f"Entry Price: ${entry_price}")
        print(f"Current Price: ${current_price}")
        print(f"Profit/Loss: ${profit_loss:.2f}")
        print(f"Profit/Loss %: {profit_loss_pct:.2f}%")

        return {
            "symbol": symbol,
            "entry_price": entry_price,
            "current_price": current_price,
            "profit_loss": profit_loss,
            "profit_loss_pct": profit_loss_pct
        }
    except Exception as e:
        print(f"MONITOR AGENT ERROR: {e}")
        return None

if __name__ == "__main__":
    monitor_position("MSFT")