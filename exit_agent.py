import os
import time
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

load_dotenv()

api = tradeapi.REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    os.getenv("ALPACA_BASE_URL")
)


def cancel_open_orders(symbol):
    """
    Cancels any open orders for this symbol before we try to close the position —
    specifically the broker-level stop-loss order executor_agent.py now attaches at
    entry (OTO order). Alpaca won't cleanly close a position while a protective order
    is still open on it, so this has to happen first or exit_trade() will fail.
    """
    try:
        open_orders = api.list_orders(status="open", symbols=[symbol])
        if not open_orders:
            return
        for order in open_orders:
            try:
                api.cancel_order(order.id)
                print(f"EXIT AGENT: Cancelled open order {order.id} "
                      f"({order.side} {order.type}) for {symbol}")
            except Exception as e:
                print(f"EXIT AGENT: Could not cancel order {order.id} for {symbol}: {e}")
        # Brief pause so the cancellation registers at Alpaca before we try to close —
        # cancels aren't guaranteed instant, and closing right away can race it.
        time.sleep(1)
    except Exception as e:
        print(f"EXIT AGENT: Could not list open orders for {symbol} ({e}) — "
              f"proceeding to close anyway, but this may fail if a stop order is still open.")


def exit_trade(symbol):
    cancel_open_orders(symbol)
    try:
        api.close_position(symbol)
        print(f"EXIT AGENT: Position closed!")
        print(f"Symbol: {symbol} sold successfully")
    except Exception as e:
        print(f"EXIT AGENT ERROR: {e}")


if __name__ == "__main__":
    exit_trade("MSFT")
