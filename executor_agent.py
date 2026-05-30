import os
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

load_dotenv()

api = tradeapi.REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    os.getenv("ALPACA_BASE_URL"),
    api_version="v2"
)

def get_position_size(symbol, risk_pct=0.02):
    """Calculate shares to buy based on 2% portfolio risk"""
    try:
        account = api.get_account()
        portfolio_value = float(account.portfolio_value)
        risk_amount = portfolio_value * risk_pct

        quote = api.get_latest_trade(symbol)
        price = float(quote.price)

        shares = int(risk_amount / price)
        shares = max(1, shares)  # Always buy at least 1 share

        print(f"💼 Portfolio Value: ${portfolio_value:,.2f}")
        print(f"⚠️ Risk Amount (2%): ${risk_amount:,.2f}")
        print(f"💲 {symbol} Price: ${price:,.2f}")
        print(f"📦 Shares to Buy: {shares}")

        return shares
    except Exception as e:
        print(f"Position sizing error: {e}, defaulting to 1 share")
        return 1

def execute_trade(symbol, shares=None):
    try:
        if shares is None:
            shares = get_position_size(symbol)

        order = api.submit_order(
            symbol=symbol,
            qty=shares,
            side="buy",
            type="market",
            time_in_force="day"
        )

        print(f"EXECUTOR AGENT: Order placed!")
        print(f"Symbol: {symbol}")