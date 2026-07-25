"""
options_agent.py — Options trading layer for the trading bot.

Currently DORMANT. Controlled by ENABLE_OPTIONS = False in scheduler.py.
When flipped on, this buys ATM call options on the top pick instead of
(or in addition to) shares.

Prerequisites before enabling:
  - Live Alpaca account with options trading approved (Level 2 minimum)
  - Proven consistent paper trading results (at least 4 weeks)
  - DO NOT enable on paper account — Alpaca paper options API is limited

Strategy when enabled:
  - Buy 1 ATM (at-the-money) call contract on the #1 GPT-4o pick
  - Requires CONSENSUS: both technical signals (VWAP/MACD/momentum) AND backtest
    calibration must agree strongly before buying — options are higher risk/reward
    than equity, so the bar here is intentionally stricter than the equity filter
  - Expiry: next weekly expiration (7-14 days out) — enough time for the
    thesis to play out without paying too much theta decay
  - Position size: 1% of portfolio value (half the normal 2% equity size
    because options already provide leverage), with a $2-4/contract slippage
    buffer built in so sizing accounts for real execution cost, not the stale quote
  - Exit: same hard ceiling and stop loss logic as equity, but applied to the
    OPTION's P&L, not the underlying stock price
  - Never hold options overnight into earnings — theta + IV crush kills you

Why calls on the #1 pick only (not all 3):
  - Options are higher risk/reward — concentrate on the highest conviction name
  - Spreads and liquidity are better on liquid large-caps
  - Keeps total risk exposure manageable alongside the equity positions
"""

import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Headers for Alpaca REST calls
HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json"
}

# ---- CONFIG ----
OPTIONS_RISK_PCT = 0.01        # 1% of portfolio per options trade
MIN_DAYS_TO_EXPIRY = 7         # don't buy weeklies expiring in less than 7 days
MAX_DAYS_TO_EXPIRY = 14        # stay in the next 1-2 weekly cycles
MAX_SPREAD_PCT = 0.05          # skip if bid/ask spread > 5% of mid price
MIN_OPEN_INTEREST = 100        # skip illiquid contracts
MIN_VOLUME = 10                # skip contracts with no volume today
DEFAULT_MAX_CONTRACTS = 1      # fallback cap if scheduler.py doesn't pass one

# ---- SLIPPAGE BUFFER (new) ----
# Reddit copy-trading research: options spreads eat $2-4/contract on average.
# Building this into both the position-sizing math and the limit price so we're
# not sizing/pricing off the stale quoted ask.
OPTIONS_SLIPPAGE_BUFFER_PER_CONTRACT = 3.0   # dollars, midpoint of the $2-4 range

# ---- CONSENSUS SIGNAL REQUIREMENT (new) ----
# Only buy calls when BOTH technical signals AND backtest calibration agree strongly.
# Matches research_agent.py's MIN_CONVICTION_SCORE — keep these in sync if you change either.
CONSENSUS_MIN_CONVICTION = 0.5


def get_next_friday(days_min=MIN_DAYS_TO_EXPIRY, days_max=MAX_DAYS_TO_EXPIRY):
    """
    Returns the next weekly expiration date (Friday) that falls within
    the [days_min, days_max] window from today.
    """
    today = datetime.now().date()
    for offset in range(days_min, days_max + 1):
        candidate = today + timedelta(days=offset)
        if candidate.weekday() == 4:  # Friday
            return candidate.strftime("%Y-%m-%d")
    # Fallback — just go 14 days out and take whatever expiry is closest
    return (today + timedelta(days=days_max)).strftime("%Y-%m-%d")


def check_consensus(symbol, conviction_score):
    """
    Requires BOTH technical signals AND backtest calibration to agree strongly
    before buying an option. Technical side reuses research_agent's
    check_signal_confirmation (VWAP/MACD/momentum majority vote, same function
    scheduler.py uses for the conviction trailing stop). Backtest side checks the
    conviction_score (avg backtest P&L) passed in from scheduler.py against the
    same bar research_agent.py uses to admit a stock in the first place.
    """
    technical_confirmed = None
    try:
        from research_agent import check_signal_confirmation
        technical_confirmed = check_signal_confirmation(symbol)
    except Exception as e:
        print(f"⚠️ OPTIONS: Could not check technical consensus for {symbol}: {e}")

    backtest_confirmed = conviction_score is not None and conviction_score >= CONSENSUS_MIN_CONVICTION

    if technical_confirmed is True and backtest_confirmed:
        print(f"✅ OPTIONS: Consensus met for {symbol} — technical signals confirm, "
              f"backtest conviction {conviction_score:+.2f}% >= {CONSENSUS_MIN_CONVICTION}%")
        return True

    print(f"⚠️ OPTIONS: Consensus NOT met for {symbol} "
          f"(technical confirmed={technical_confirmed}, "
          f"backtest conviction={conviction_score}, required >= {CONSENSUS_MIN_CONVICTION}%) "
          f"— skipping options trade")
    return False


def get_atm_call(symbol, portfolio_value, max_contracts=DEFAULT_MAX_CONTRACTS):
    """
    Finds the best ATM call contract for the given symbol and sizes the trade.

    Returns a dict with contract details and share count, or None if no
    suitable contract is found.
    """
    try:
        # Step 1: get current stock price
        quote_url = f"{ALPACA_BASE_URL}/v2/stocks/{symbol}/quotes/latest"
        quote_resp = requests.get(quote_url, headers=HEADERS, timeout=10)
        quote_data = quote_resp.json()
        current_price = float(quote_data["quote"]["ap"])  # ask price

        # Step 2: find the target expiry
        target_expiry = get_next_friday()

        # Step 3: fetch the options chain
        chain_url = f"{ALPACA_BASE_URL}/v2/options/contracts"
        params = {
            "underlying_symbol": symbol,
            "expiration_date": target_expiry,
            "type": "call",
            "status": "active",
            "limit": 100
        }
        chain_resp = requests.get(chain_url, headers=HEADERS, params=params, timeout=15)
        contracts = chain_resp.json().get("option_contracts", [])

        if not contracts:
            print(f"⚠️ OPTIONS: No call contracts found for {symbol} expiring {target_expiry}")
            return None

        # Step 4: find the ATM contract (closest strike to current price)
        best_contract = None
        smallest_diff = float("inf")
        for c in contracts:
            strike = float(c["strike_price"])
            diff = abs(strike - current_price)
            if diff < smallest_diff:
                smallest_diff = diff
                best_contract = c

        if not best_contract:
            print(f"⚠️ OPTIONS: Could not find ATM contract for {symbol}")
            return None

        symbol_id = best_contract["symbol"]
        strike = float(best_contract["strike_price"])

        # Step 5: check liquidity on the contract
        snapshot_url = f"{ALPACA_BASE_URL}/v2/options/snapshots/{symbol_id}"
        snap_resp = requests.get(snapshot_url, headers=HEADERS, timeout=10)
        snap = snap_resp.json().get("snapshot", {})

        greeks = snap.get("greeks", {})
        quote = snap.get("latestQuote", {})
        bid = float(quote.get("bp", 0))
        ask = float(quote.get("ap", 0))
        mid = (bid + ask) / 2 if (bid + ask) > 0 else 0

        open_interest = int(snap.get("openInterest", 0))
        volume = int(snap.get("dailyBar", {}).get("v", 0))

        if open_interest < MIN_OPEN_INTEREST:
            print(f"⚠️ OPTIONS: {symbol_id} open interest too low ({open_interest}), skipping")
            return None

        if volume < MIN_VOLUME:
            print(f"⚠️ OPTIONS: {symbol_id} volume too low ({volume}), skipping")
            return None

        if mid > 0:
            spread_pct = (ask - bid) / mid
            if spread_pct > MAX_SPREAD_PCT:
                print(f"⚠️ OPTIONS: {symbol_id} spread too wide ({spread_pct:.1%}), skipping")
                return None

        # Step 6: slippage buffer — pad both the sizing math and the limit price so
        # we're planning around real execution cost, not the stale quoted ask.
        slippage_per_share = OPTIONS_SLIPPAGE_BUFFER_PER_CONTRACT / 100
        effective_ask       = ask + slippage_per_share
        limit_price          = round(effective_ask, 2)

        # Step 7: size the trade — 1% of portfolio / cost per contract (100 shares),
        # capped at max_contracts (fixes a prior mismatch — scheduler.py has always
        # called this with a max_contracts cap; this file wasn't previously honoring it)
        risk_amount        = portfolio_value * OPTIONS_RISK_PCT
        cost_per_contract  = effective_ask * 100  # includes slippage buffer
        num_contracts       = max(1, int(risk_amount / cost_per_contract))
        num_contracts       = min(num_contracts, max_contracts)

        delta = greeks.get("delta", "N/A")
        print(f"📊 OPTIONS: {symbol_id} | Strike ${strike} | Expiry {target_expiry}")
        print(f"   Bid/Ask: ${bid:.2f}/${ask:.2f} | Mid: ${mid:.2f} | Delta: {delta}")
        print(f"   Slippage-adjusted ask: ${effective_ask:.2f} "
              f"(+${OPTIONS_SLIPPAGE_BUFFER_PER_CONTRACT:.2f}/contract buffer)")
        print(f"   OI: {open_interest} | Vol: {volume} | Contracts: {num_contracts} "
              f"(capped at {max_contracts})")

        return {
            "contract_symbol": symbol_id,
            "underlying": symbol,
            "strike": strike,
            "expiry": target_expiry,
            "ask": ask,
            "bid": bid,
            "limit_price": limit_price,
            "num_contracts": num_contracts,
            "cost_per_contract": cost_per_contract,
            "total_cost": num_contracts * cost_per_contract,
            "delta": delta
        }

    except Exception as e:
        print(f"⚠️ OPTIONS: Error finding contract for {symbol}: {e}")
        return None


def buy_call_option(symbol, portfolio_value, max_contracts=DEFAULT_MAX_CONTRACTS, conviction_score=None):
    """
    Main entry point called by scheduler.py when ENABLE_OPTIONS is True.
    Requires consensus (technical signals + backtest calibration) before doing
    anything, then finds the best ATM call and submits a limit order at the
    slippage-adjusted price.

    Returns the order object or None.
    """
    if not check_consensus(symbol, conviction_score):
        return None

    print(f"\n📈 OPTIONS AGENT: Finding call for {symbol}...")
    contract = get_atm_call(symbol, portfolio_value, max_contracts=max_contracts)
    if not contract:
        print(f"⚠️ OPTIONS: No suitable contract found for {symbol}, skipping options trade.")
        return None

    try:
        order_url = f"{ALPACA_BASE_URL}/v2/orders"
        order_body = {
            "symbol": contract["contract_symbol"],
            "qty": str(contract["num_contracts"]),
            "side": "buy",
            "type": "limit",
            "limit_price": str(contract["limit_price"]),
            "time_in_force": "day"
        }

        resp = requests.post(order_url, headers=HEADERS, json=order_body, timeout=10)
        order = resp.json()

        if "id" in order:
            print(f"✅ OPTIONS: Order placed!")
            print(f"   Contract: {contract['contract_symbol']}")
            print(f"   Contracts: {contract['num_contracts']} x ${contract['limit_price']:.2f} "
                  f"(slippage-adjusted)")
            print(f"   Total outlay: ${contract['total_cost']:,.2f}")
            print(f"   Order ID: {order['id']}")
            return order
        else:
            print(f"⚠️ OPTIONS: Order rejected: {order}")
            return None

    except Exception as e:
        print(f"⚠️ OPTIONS: Order submission failed: {e}")
        return None


def exit_option(contract_symbol):
    """
    Closes an open options position (market sell).
    Called by scheduler.py when exit conditions fire on an options position.
    """
    try:
        url = f"{ALPACA_BASE_URL}/v2/positions/{contract_symbol}"
        resp = requests.delete(url, headers=HEADERS, timeout=10)
        if resp.status_code in (200, 204):
            print(f"✅ OPTIONS: Position closed for {contract_symbol}")
        else:
            print(f"⚠️ OPTIONS: Could not close {contract_symbol}: {resp.text}")
    except Exception as e:
        print(f"⚠️ OPTIONS: Exit failed for {contract_symbol}: {e}")


if __name__ == "__main__":
    # Quick sanity check — prints what contract it would buy for AAPL
    # without actually placing an order. Safe to run anytime.
    # NOTE: consensus check is bypassed here (conviction_score=999) since this is
    # just a contract-selection/sizing dry run, not a trade-decision test.
    print("OPTIONS AGENT — dry run for AAPL (no order placed, consensus check bypassed)")
    contract = get_atm_call("AAPL", portfolio_value=99000)
    if contract:
        print(f"\nWould buy: {contract['num_contracts']} contract(s) of {contract['contract_symbol']}")
        print(f"Strike: ${contract['strike']} | Expiry: {contract['expiry']}")
        print(f"Limit price: ${contract['limit_price']:.2f} (slippage-adjusted)")
        print(f"Cost: ${contract['total_cost']:,.2f}")
    else:
        print("No suitable contract found.")
