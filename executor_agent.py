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


def compute_conviction_weights(symbols, conviction_scores, risk_multiplier=1.0):
    effective_total = TOTAL_RISK_PCT * risk_multiplier
    effective_min   = MIN_RISK_PCT   * risk_multiplier
    effective_max   = MAX_RISK_PCT   * risk_multiplier

    if not conviction_scores or len(symbols) == 0:
        equal = effective_total / len(symbols)
        return {s: equal for s in symbols}

    scores = [conviction_scores.get(s, 0.0) for s in symbols]

    if len(set(scores)) == 1 or sum(s for s in scores if s > 0) == 0:
        equal = effective_total / len(symbols)
        return {s: equal for s in symbols}

    min_score = min(scores)
    shifted   = [s - min_score for s in scores]
    total     = sum(shifted)

    if total == 0:
        equal = effective_total / len(symbols)
        return {s: equal for s in symbols}

    raw = {s: (shifted[i] / total) * effective_total
           for i, s in enumerate(symbols)}

    clamped   = {}
    remainder = 0.0
    free      = []

    for s, alloc in raw.items():
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


def get_position_size(symbol, risk_pct):
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


def execute_trades(symbols, conviction_scores=None, risk_multiplier=1.0):
    if not symbols:
        print("EXECUTOR AGENT: No symbols to trade.")
        return

    weights = compute_conviction_weights(symbols, conviction_scores or {},
                                         risk_multiplier=risk_multiplier)

    cooloff_label = f" [COOL-OFF: {risk_multiplier*100:.0f}% sizing]" if risk_multiplier < 1.0 else ""
    print(f"\n📊 CONVICTION-WEIGHTED ALLOCATION{cooloff_label}:")
    print(f"   Total risk budget: {TOTAL_RISK_PCT * risk_multiplier * 100:.1f}% of portfolio")
    for s in symbols:
        score = conviction_scores.get(s, 0.0) if conviction_scores else 0.0
        print(f"   {s}: {weights[s]*100:.2f}% allocation  "
              f"(backtest avg P&L: {score:+.2f}%)")

    for symbol in symbols:
        print(f"\n⚡ Executing: {symbol}")
        execute_trade(symbol, risk_pct=weights[symbol])


if __name__ == "__main__":
    test_symbols = ["CRM", "SNOW", "SHOP"]
    test_scores  = {"CRM": 0.12, "SNOW": 1.56, "SHOP": 1.12}

    print("Normal sizing (risk_multiplier=1.0):")
    weights = compute_conviction_weights(test_symbols, test_scores, risk_multiplier=1.0)
    for s, w in weights.items():
        print(f"  {s}: {w*100:.2f}%  (score: {test_scores[s]:+.2f}%)")
    print(f"  Total: {sum(weights.values())*100:.2f}%")

    print("\nCool-off sizing (risk_multiplier=0.5):")
    weights = compute_conviction_weights(test_symbols, test_scores, risk_multiplier=0.5)
    for s, w in weights.items():
        print(f"  {s}: {w*100:.2f}%  (score: {test_scores[s]:+.2f}%)")
    print(f"  Total: {sum(weights.values())*100:.2f}%")
