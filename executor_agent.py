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

# ---- POSITION SIZING CONSTANTS ----
TOTAL_RISK_PCT  = 0.06   # total capital at risk across all 3 positions (6%)
MIN_RISK_PCT    = 0.005  # floor: never allocate less than 0.5% to any single pick
MAX_RISK_PCT    = 0.035  # ceiling: never allocate more than 3.5% to any single pick


def compute_conviction_weights(symbols, conviction_scores):
    """
    Given a list of symbols and their backtest avg P&L scores, return a
    dict of { symbol: risk_pct } that sums to TOTAL_RISK_PCT.

    Weighting logic:
      - Scores are shifted so the minimum is 0 (no negatives in the weight calc)
      - Each symbol's weight = its shifted score / sum of all shifted scores
      - Weight is then scaled to TOTAL_RISK_PCT and clamped to [MIN, MAX]
      - Any remainder after clamping is distributed evenly across unclamped symbols

    Falls back to equal weighting if scores are missing or all identical.
    """
    if not conviction_scores or len(symbols) == 0:
        equal = TOTAL_RISK_PCT / len(symbols)
        return {s: equal for s in symbols}

    scores = [conviction_scores.get(s, 0.0) for s in symbols]

    # If all scores are identical or zero, equal weight
    if len(set(scores)) == 1 or sum(s for s in scores if s > 0) == 0:
        equal = TOTAL_RISK_PCT / len(symbols)
        return {s: equal for s in symbols}

    # Shift so minimum score becomes 0, preserving relative differences
    min_score = min(scores)
    shifted   = [s - min_score for s in scores]
    total     = sum(shifted)

    if total == 0:
        equal = TOTAL_RISK_PCT / len(symbols)
        return {s: equal for s in symbols}

    # Raw weighted allocations
    raw = {s: (shifted[i] / total) * TOTAL_RISK_PCT
           for i, s in enumerate(symbols)}

    # Clamp to [MIN, MAX]
    clamped   = {}
    remainder = 0.0
    free      = []

    for s, alloc in raw.items():
        if alloc < MIN_RISK_PCT:
            clamped[s] = MIN_RISK_PCT
            remainder += MIN_RISK_PCT - alloc  # deficit pulled from remainder pool
        elif alloc > MAX_RISK_PCT:
            clamped[s] = MAX_RISK_PCT
            remainder += alloc - MAX_RISK_PCT  # surplus goes back to remainder pool
        else:
            clamped[s] = alloc
            free.append(s)

    # Distribute remainder across unclamped symbols proportionally
    if remainder != 0 and free:
        free_total = sum(clamped[s] for s in free)
        for s in free:
            if free_total > 0:
                clamped[s] += remainder * (clamped[s] / free_total)
            else:
                clamped[s] += remainder / len(free)

    return clamped


def get_position_size(symbol, risk_pct):
    """
    Convert a risk % allocation into a share count for the given symbol.
    Always uses portfolio_value (not buying_power) for correct sizing.
    """
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
        print(f"📦 Shares to Buy  : {shares}")
        return shares

    except Exception as e:
        print(f"Position sizing error for {symbol}: {e}, defaulting to 1 share")
        return 1


def execute_trade(symbol, shares=None, risk_pct=None):
    """
    Place a market buy order for symbol.
    If shares is provided, use it directly.
    If risk_pct is provided, calculate shares from portfolio value.
    Falls back to flat 2% if neither is provided.
    """
    try:
        if shares is None:
            rp     = risk_pct if risk_pct is not None else 0.02
            shares = get_position_size(symbol, rp)

        order = api.submit_order(
            symbol=symbol,
            qty=shares,
            side="buy",
            type="market",
            time_in_force="day"
        )
        print(f"✅ EXECUTOR AGENT: Order placed!")
        print(f"   Symbol   : {symbol}")
        print(f"   Shares   : {shares}")
        print(f"   Order ID : {order.id}")
        return order

    except Exception as e:
        print(f"EXECUTOR AGENT ERROR for {symbol}: {e}")
        return None


def execute_trades(symbols, conviction_scores=None):
    """
    Main entry point called by scheduler.py.

    symbols           — list of tickers to buy (already filtered, max 3)
    conviction_scores — dict of { symbol: best_avg_pnl } from backtest calibration
                        If None or missing, falls back to equal 2% per position.

    Prints the full allocation table before placing any orders so you can
    see exactly how capital is being split in the Railway logs.
    """
    if not symbols:
        print("EXECUTOR AGENT: No symbols to trade.")
        return

    weights = compute_conviction_weights(symbols, conviction_scores or {})

    print("\n📊 CONVICTION-WEIGHTED ALLOCATION:")
    print(f"   Total risk budget: {TOTAL_RISK_PCT*100:.1f}% of portfolio")
    for s in symbols:
        score = conviction_scores.get(s, 0.0) if conviction_scores else 0.0
        print(f"   {s}: {weights[s]*100:.2f}% allocation  "
              f"(backtest avg P&L: {score:+.2f}%)")

    for symbol in symbols:
        print(f"\n⚡ Executing: {symbol}")
        execute_trade(symbol, risk_pct=weights[symbol])


if __name__ == "__main__":
    # Test with the picks from tonight — shows allocation without placing orders
    # (swap in real conviction scores from your backtest output)
    test_symbols = ["CRM", "SNOW", "SHOP"]
    test_scores  = {"CRM": 0.12, "SNOW": 1.56, "SHOP": 1.12}
    weights      = compute_conviction_weights(test_symbols, test_scores)
    print("Conviction-weighted allocation test:")
    for s, w in weights.items():
        print(f"  {s}: {w*100:.2f}%  (score: {test_scores[s]:+.2f}%)")
