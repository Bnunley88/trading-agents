"""
short_agent.py — Short-selling layer for the trading bot. NEW FILE, longer-term item.

NOT YET WIRED INTO scheduler.py. This is a standalone module, structurally ready to
plug in, but scheduler.py doesn't call it and there's no ENABLE_SHORTS flag checked
anywhere yet. Wiring it in requires two things this session didn't build:
  1. A way for research_agent.py to surface "weakest stocks" candidates (today it
     only surfaces buy candidates) — needs its own filter pass, probably inverted
     versions of the existing quality filters (uptrend-blocked instead of
     downtrend-blocked, negative momentum required instead of rejected, etc.)
  2. A decision in scheduler.py's morning session about how shorts interact with
     the existing TRENDING DOWN equity logic (reduce to 1-2 picks) — do shorts
     replace those picks, run alongside them, or only fire when equity sits out
     entirely? That's a real risk-budgeting decision, not a default I should pick
     for you.

HARD SAFETY GATE: every entry point in this file calls verify_margin_account()
first and refuses to trade if the account isn't confirmed to support shorting.
Per your own pre-live checklist, the live account needs to be opened as a margin
account (not cash) before shorting can ever work for real — this gate makes that
failure loud and safe instead of a confusing order rejection from Alpaca.

Mirrors executor_agent.py's conviction-weighted position sizing, inverted: the
WEAKEST stocks (most negative score) get the largest allocation, not the strongest.
"""

import os
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
from executor_agent import compute_conviction_weights

load_dotenv()

api = tradeapi.REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    os.getenv("ALPACA_BASE_URL"),
    api_version="v2"
)

# ---- POSITION SIZING CONSTANTS ----
# Deliberately separate from executor_agent.py's TOTAL_RISK_PCT rather than reusing
# it — if shorts ever run in the same session as the existing 3 long positions,
# reusing the same budget would silently double your total risk exposure. Size
# this independently and consciously once you decide how shorts and longs interact.
TOTAL_SHORT_RISK_PCT = 0.06   # total capital at risk across all short positions
MIN_SHORT_RISK_PCT   = 0.005  # floor: never allocate less than 0.5% to any single short
MAX_SHORT_RISK_PCT   = 0.035  # ceiling: never allocate more than 3.5% to any single short


def verify_margin_account():
    """
    Hard safety gate — every function below calls this first and refuses to trade
    if it doesn't come back True. Alpaca cash accounts can't open short positions;
    your pre-live checklist item ('open live account as margin, not cash') has to
    be done before this will ever pass on a real account.
    """
    try:
        account = api.get_account()
        shorting_enabled = getattr(account, "shorting_enabled", None)
        multiplier        = getattr(account, "multiplier", None)

        if shorting_enabled is False:
            print("🚫 SHORT AGENT: Account does not have shorting enabled — refusing "
                  "to short. Open a margin account with shorting approval first.")
            return False

        if multiplier is not None and float(multiplier) <= 1.0:
            print(f"🚫 SHORT AGENT: Account multiplier is {multiplier} (cash account) — "
                  f"shorting requires margin. Refusing to short.")
            return False

        return True

    except Exception as e:
        print(f"🚫 SHORT AGENT: Could not verify account type ({e}) — refusing to "
              f"short as a safety default.")
        return False


def compute_short_weights(symbols, weakness_scores, risk_multiplier=1.0):
    """
    Reuses executor_agent's conviction-weighting math, inverted: that function gives
    more allocation to higher scores, so we flip the sign here — the most negative
    weakness_scores (weakest stocks) end up with the highest allocation.
    """
    inverted_scores = {s: -weakness_scores.get(s, 0.0) for s in symbols}

    # Temporarily borrow the shared allocation curve but with our own risk budget —
    # compute_conviction_weights reads module-level constants from executor_agent,
    # so we scale its output into our own TOTAL/MIN/MAX_SHORT_RISK_PCT range instead
    # of importing and overriding its constants directly.
    raw_weights = compute_conviction_weights(symbols, inverted_scores, risk_multiplier=1.0)
    raw_total   = sum(raw_weights.values()) or 1.0

    effective_total = TOTAL_SHORT_RISK_PCT * risk_multiplier
    effective_min   = MIN_SHORT_RISK_PCT   * risk_multiplier
    effective_max   = MAX_SHORT_RISK_PCT   * risk_multiplier

    scaled = {s: (w / raw_total) * effective_total for s, w in raw_weights.items()}

    clamped   = {}
    remainder = 0.0
    free      = []
    for s, alloc in scaled.items():
        if alloc < effective_min:
            clamped[s] = effective_min
            remainder -= effective_min - alloc
        elif alloc > effective_max:
            clamped[s] = effective_max
            remainder += alloc - effective_max
        else:
            clamped[s] = alloc
            free.append(s)

    if remainder != 0 and free:
        free_total = sum(clamped[s] for s in free)
        for s in free:
            if free_total > 0:
                clamped[s] += remainder * (clamped[s] / free_total)
            else:
                clamped[s] += remainder / len(free)

    return clamped


def get_short_position_size(symbol, risk_pct):
    try:
        account         = api.get_account()
        portfolio_value = float(account.portfolio_value)
        risk_amount     = portfolio_value * risk_pct
        quote           = api.get_latest_trade(symbol)
        price           = float(quote.price)
        shares          = int(risk_amount / price)
        shares          = max(1, shares)

        print(f"💼 Portfolio Value : ${portfolio_value:,.2f}")
        print(f"⚠️  Risk Allocation : {risk_pct*100:.2f}% = ${risk_amount:,.2f}")
        print(f"💲 {symbol} Price  : ${price:,.2f}")
        print(f"📦 Shares to Short : {shares}")
        return shares

    except Exception as e:
        print(f"Short position sizing error for {symbol}: {e}, defaulting to 1 share")
        return 1


def execute_short(symbol, shares=None, risk_pct=None):
    """Opens a short position (sell order on a symbol you don't hold — Alpaca
    handles this as a short open on marginable/shortable symbols)."""
    if not verify_margin_account():
        return None

    try:
        if shares is None:
            rp     = risk_pct if risk_pct is not None else 0.02
            shares = get_short_position_size(symbol, rp)

        order = api.submit_order(
            symbol=symbol,
            qty=shares,
            side="sell",
            type="market",
            time_in_force="day"
        )
        print(f"✅ SHORT AGENT: Order placed!")
        print(f"   Symbol   : {symbol}")
        print(f"   Shares   : {shares} (short)")
        print(f"   Order ID : {order.id}")
        return order

    except Exception as e:
        print(f"SHORT AGENT ERROR for {symbol}: {e}")
        return None


def cover_short(symbol, shares=None):
    """Closes a short position (buy to cover). Mirrors exit_agent.py's role for
    the equity side — kept self-contained here since exit_agent.py wasn't part of
    this session's file set."""
    if not verify_margin_account():
        print(f"🚫 SHORT AGENT: Skipping cover for {symbol} — could not verify margin account.")
        return None

    try:
        if shares is None:
            position = api.get_position(symbol)
            shares   = abs(int(float(position.qty)))

        order = api.submit_order(
            symbol=symbol,
            qty=shares,
            side="buy",
            type="market",
            time_in_force="day"
        )
        print(f"✅ SHORT AGENT: Covered {symbol}")
        print(f"   Shares   : {shares}")
        print(f"   Order ID : {order.id}")
        return order

    except Exception as e:
        print(f"SHORT AGENT ERROR covering {symbol}: {e}")
        return None


def execute_shorts(symbols, weakness_scores=None, risk_multiplier=1.0):
    """
    Mirrors executor_agent.execute_trades(). symbols should be the WEAKEST stocks
    (research_agent doesn't produce this list yet — see module docstring), and
    weakness_scores should be oriented so more negative = weaker.
    """
    if not verify_margin_account():
        print("🚫 SHORT AGENT: Margin account not confirmed — skipping all shorts this session.")
        return

    if not symbols:
        print("SHORT AGENT: No symbols to short.")
        return

    weights = compute_short_weights(symbols, weakness_scores or {}, risk_multiplier=risk_multiplier)

    cooloff_label = f" [COOL-OFF: {risk_multiplier*100:.0f}% sizing]" if risk_multiplier < 1.0 else ""
    print(f"\n📊 SHORT CONVICTION-WEIGHTED ALLOCATION{cooloff_label}:")
    print(f"   Total short risk budget: {TOTAL_SHORT_RISK_PCT * risk_multiplier * 100:.1f}% of portfolio")
    for s in symbols:
        score = weakness_scores.get(s, 0.0) if weakness_scores else 0.0
        print(f"   {s}: {weights[s]*100:.2f}% allocation  (weakness score: {score:+.2f}%)")

    for symbol in symbols:
        print(f"\n⚡ Shorting: {symbol}")
        execute_short(symbol, risk_pct=weights[symbol])


if __name__ == "__main__":
    # Dry run — tests the weighting math only, does not place orders or call Alpaca
    # (except for the account check inside execute_shorts, which will refuse safely
    # if margin isn't confirmed).
    test_symbols = ["XYZ", "ABC", "DEF"]
    test_scores  = {"XYZ": -0.12, "ABC": -1.56, "DEF": -1.12}   # more negative = weaker

    print("Short weighting (risk_multiplier=1.0):")
    weights = compute_short_weights(test_symbols, test_scores, risk_multiplier=1.0)
    for s, w in weights.items():
        print(f"  {s}: {w*100:.2f}%  (weakness score: {test_scores[s]:+.2f}%)")
    print(f"  Total: {sum(weights.values())*100:.2f}%")
