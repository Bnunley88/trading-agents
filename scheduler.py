import os
import json
import schedule
import time
import threading
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from research_agent import research_stocks
from analyst_agent import analyze_recommendation
from executor_agent import execute_trades
from monitor_agent import monitor_position
from exit_agent import exit_trade

load_dotenv()

api = tradeapi.REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    os.getenv("ALPACA_BASE_URL")
)

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")
    def log_message(self, format, *args):
        pass

def run_health_server():
    server = HTTPServer(("0.0.0.0", 8080), HealthHandler)
    server.serve_forever()

threading.Thread(target=run_health_server, daemon=True).start()
print("✅ Health server running on port 8080")

# ---- OPTIONS FLAG ----
# Set to False until live account with options Level 2 approval
# Import is conditional so a missing/broken options_agent won't crash the bot
ENABLE_OPTIONS = False

if ENABLE_OPTIONS:
    from options_agent import buy_call_option

# ---- STATE ----
todays_symbols        = []
high_water_marks      = {}
entry_times           = {}
trailing_stops        = {}
consecutive_loss_days = 0
daily_start_value     = 0.0
yesterdays_exits      = []
options_daily_cost    = 0.0

# ---- CONSTANTS ----
HARD_PROFIT_CEILING       = 3.0
STOP_LOSS_PCT             = 2.0
TRAILING_STOP_PCT_DEFAULT = 2.0
GRACE_PERIOD_MINUTES      = 90
MAX_POSITIONS             = 3
MIN_BUYING_POWER          = 1000.0

# ---- DAILY LOSS LIMIT ----
DAILY_LOSS_LIMIT_PCT    = 2.0
COOLOFF_LOSS_DAYS       = 3
COOLOFF_RISK_MULTIPLIER = 0.5

# ---- OPTIONS GUARDRAILS ----
MAX_OPTIONS_BUDGET_PCT = 0.02
MAX_OPTIONS_CONTRACTS  = 1
OPTIONS_DAILY_LOSS_PCT = 1.0

# ---- MARKET HOURS (Central Time) ----
MARKET_OPEN_HOUR   = 9
MARKET_OPEN_MIN    = 35
MARKET_CLOSE_HOUR  = 15
MARKET_CLOSE_MIN   = 45

# ---- CALIBRATION CACHE ----
CACHE_FILE    = "calibration_cache.json"
CACHE_MAX_AGE = timedelta(hours=23)


def load_calibration_cache():
    try:
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
        saved_at = datetime.fromisoformat(data["saved_at"])
        if datetime.now() - saved_at > CACHE_MAX_AGE:
            print("📐 Calibration cache is stale — will recompute.")
            return {}
        stops = data["trailing_stops"]
        print(f"📐 Loaded calibration cache from {saved_at.strftime('%Y-%m-%d %H:%M')} "
              f"({len(stops)} tickers)")
        return stops
    except Exception:
        return {}


def save_calibration_cache(stops_dict):
    try:
        data = {
            "saved_at": datetime.now().isoformat(),
            "trailing_stops": stops_dict
        }
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"💾 Calibration cache saved ({len(stops_dict)} tickers).")
    except Exception as e:
        print(f"⚠️ Could not save calibration cache: {e}")


def get_trailing_stop(symbol):
    return trailing_stops.get(symbol, TRAILING_STOP_PCT_DEFAULT)


def is_in_grace_period(symbol):
    if symbol not in entry_times:
        return False
    elapsed = (datetime.now() - entry_times[symbol]).total_seconds() / 60
    return elapsed < GRACE_PERIOD_MINUTES


def is_market_hours():
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=MARKET_OPEN_HOUR,  minute=MARKET_OPEN_MIN,  second=0)
    market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MIN, second=0)
    return market_open <= now <= market_close


def is_daily_loss_limit_hit():
    if daily_start_value <= 0:
        return False
    try:
        account       = api.get_account()
        current_value = float(account.portfolio_value)
        daily_pct     = ((current_value - daily_start_value) / daily_start_value) * 100
        if daily_pct <= -DAILY_LOSS_LIMIT_PCT:
            print(f"🛑 DAILY LOSS LIMIT HIT: portfolio down {daily_pct:.2f}% today "
                  f"(limit: -{DAILY_LOSS_LIMIT_PCT}%) — pausing new trades.")
            return True
        return False
    except Exception as e:
        print(f"⚠️ Could not check daily loss limit: {e}")
        return False


def is_options_budget_ok(portfolio_value):
    max_spend = portfolio_value * MAX_OPTIONS_BUDGET_PCT
    if options_daily_cost >= max_spend:
        print(f"⚠️ OPTIONS: Daily budget cap reached "
              f"(${options_daily_cost:.2f} / ${max_spend:.2f}) — skipping")
        return False
    return True


def get_risk_multiplier():
    if consecutive_loss_days >= COOLOFF_LOSS_DAYS:
        print(f"⚠️ COOL-OFF ACTIVE: {consecutive_loss_days} consecutive losing days — "
              f"position sizes reduced to {COOLOFF_RISK_MULTIPLIER * 100:.0f}%")
        return COOLOFF_RISK_MULTIPLIER
    return 1.0


def position_exists_in_alpaca(symbol):
    """Verify position exists before attempting exit — prevents ghost position errors."""
    try:
        api.get_position(symbol)
        return True
    except Exception:
        return False


def sync_positions_from_alpaca():
    global todays_symbols
    try:
        open_positions = api.list_positions()
        open_symbols   = [p.symbol for p in open_positions]

        for symbol in open_symbols:
            if symbol not in todays_symbols:
                print(f"🔄 Sync: adding {symbol} to monitoring (found in Alpaca positions)")
                todays_symbols.append(symbol)

        for symbol in list(todays_symbols):
            if symbol not in open_symbols:
                print(f"🔄 Sync: removing {symbol} from monitoring (no longer in Alpaca)")
                todays_symbols.remove(symbol)

    except Exception as e:
        print(f"⚠️ Position sync failed ({e}), using cached todays_symbols")


# ---- SESSIONS ----

def run_morning_session():
    global todays_symbols, high_water_marks, trailing_stops, entry_times
    global daily_start_value, consecutive_loss_days, yesterdays_exits, options_daily_cost

    print("\n🌅 MORNING SESSION - 9:35 AM")
    print("=" * 50)

    options_daily_cost = 0.0

    if yesterdays_exits:
        print(f"🔄 Yesterday's exits (re-entry candidates): {yesterdays_exits}")

    try:
        account           = api.get_account()
        daily_start_value = float(account.portfolio_value)
        buying_power      = float(account.buying_power)
        print(f"📊 Day opening portfolio value: ${daily_start_value:,.2f}")
    except Exception as e:
        print(f"⚠️ Could not get account info: {e}")
        daily_start_value = 0.0
        buying_power      = 0.0

    risk_mult = get_risk_multiplier()
    if consecutive_loss_days > 0:
        print(f"📉 Consecutive losing days: {consecutive_loss_days} "
              f"({'COOL-OFF ACTIVE' if consecutive_loss_days >= COOLOFF_LOSS_DAYS else 'watching'})")

    try:
        existing_positions = api.list_positions()
        existing_symbols   = [p.symbol for p in existing_positions]
        print(f"💼 Open positions: {existing_symbols}")
        print(f"💵 Buying power: ${buying_power:,.2f}")

        if buying_power < MIN_BUYING_POWER:
            print(f"⚠️ Buying power below ${MIN_BUYING_POWER:,.2f} — skipping new buys.")
            todays_symbols = existing_symbols
            for symbol in todays_symbols:
                check_position(symbol)
            print("\n✅ MORNING SESSION COMPLETE (no new buys — low buying power)")
            return

        if len(existing_positions) >= MAX_POSITIONS:
            print(f"⚠️ Already at max {MAX_POSITIONS} positions — skipping new buys.")
            todays_symbols = existing_symbols
            for symbol in todays_symbols:
                check_position(symbol)
            print("\n✅ MORNING SESSION COMPLETE (no new buys — max positions reached)")
            return

    except Exception as e:
        print(f"⚠️ Could not check positions ({e}), proceeding normally.")
        existing_symbols = []

    cached = load_calibration_cache()
    if cached:
        trailing_stops.update(cached)
        print(f"📐 Using cached trailing stops: {trailing_stops}")

    print("\n📊 STEP 1: RESEARCHING STOCKS + CALIBRATING TRAILING STOPS...")
    research_result = research_stocks(previously_held=yesterdays_exits)
    research_report = research_result["report"]
    top_symbols     = research_result["symbols"]
    new_stops       = research_result.get("trailing_stops", {})

    trailing_stops.update(new_stops)
    print(f"📐 Active trailing stops: {trailing_stops}")

    if not top_symbols:
        print("🛑 No quality picks today — sitting on cash, monitoring existing positions only.")
        todays_symbols = existing_symbols
        for symbol in todays_symbols:
            check_position(symbol)
        print("\n✅ MORNING SESSION COMPLETE (no new buys — no quality picks today)")
        return

    if is_daily_loss_limit_hit():
        todays_symbols = existing_symbols
        for symbol in todays_symbols:
            check_position(symbol)
        print("\n✅ MORNING SESSION COMPLETE (no new buys — daily loss limit hit)")
        return

    slots_available = MAX_POSITIONS - len(existing_symbols)
    new_symbols     = [s for s in top_symbols if s not in existing_symbols][:slots_available]

    if not new_symbols:
        print("⚠️ All top picks already held — skipping new buys.")
        todays_symbols = existing_symbols
        for symbol in todays_symbols:
            check_position(symbol)
        print("\n✅ MORNING SESSION COMPLETE (no new buys — all picks already held)")
        return

    todays_symbols   = existing_symbols + new_symbols
    high_water_marks = {}

    print("\n🧠 STEP 2: ANALYZING OPPORTUNITIES...")
    analyze_recommendation(research_report)

    print("\n⚡ STEP 3: EXECUTING TRADES...")
    print(f"Already holding: {existing_symbols}")
    print(f"New trades today: {new_symbols}")
    conviction_scores = research_result.get("conviction_scores", {})
    execute_trades(new_symbols, conviction_scores=conviction_scores,
                   risk_multiplier=risk_mult)

    now = datetime.now()
    for symbol in new_symbols:
        entry_times[symbol] = now

    # Options layer — only runs when ENABLE_OPTIONS = True
    if ENABLE_OPTIONS and new_symbols:
        try:
            account         = api.get_account()
            portfolio_value = float(account.portfolio_value)
            top_pick        = new_symbols[0]

            if is_options_budget_ok(portfolio_value):
                print(f"\n📈 OPTIONS: Buying call on top pick {top_pick} "
                      f"(max {MAX_OPTIONS_CONTRACTS} contract, "
                      f"budget cap: {MAX_OPTIONS_BUDGET_PCT*100:.0f}% of portfolio)...")
                result = buy_call_option(top_pick, portfolio_value,
                                         max_contracts=MAX_OPTIONS_CONTRACTS)
                if result:
                    options_daily_cost += portfolio_value * MAX_OPTIONS_BUDGET_PCT
        except Exception as e:
            print(f"⚠️ OPTIONS: Skipping — {e}")

    time.sleep(3)
    sync_positions_from_alpaca()

    print("\n👁️ STEP 4: MONITORING POSITIONS...")
    for symbol in todays_symbols:
        check_position(symbol)

    print("\n✅ MORNING SESSION COMPLETE!")


def run_intraday_check():
    if not is_market_hours():
        return

    if todays_symbols and is_daily_loss_limit_hit():
        print("🛑 Daily loss limit active — monitoring only, no new buys.")

    sync_positions_from_alpaca()

    if not todays_symbols:
        return

    now_str = datetime.now().strftime("%H:%M")
    print(f"\n🔄 INTRADAY CHECK - {now_str}")
    print("=" * 50)
    for symbol in list(todays_symbols):
        check_position(symbol)


def run_closing_check():
    global todays_symbols, high_water_marks, trailing_stops, entry_times
    global consecutive_loss_days, daily_start_value, yesterdays_exits

    sync_positions_from_alpaca()

    print("\n🌆 CLOSING CHECK - 3:45 PM")
    print("=" * 50)

    if todays_symbols:
        for symbol in list(todays_symbols):
            check_position(symbol)
    else:
        print("⚠️ No positions to monitor at close.")

    _update_loss_streak()

    if trailing_stops:
        save_calibration_cache(trailing_stops)

    print(f"🔄 Re-entry candidates saved for tomorrow: {yesterdays_exits}")

    todays_symbols   = []
    high_water_marks = {}
    entry_times      = {}


def _update_loss_streak():
    global consecutive_loss_days, daily_start_value
    if daily_start_value <= 0:
        return
    try:
        account       = api.get_account()
        closing_value = float(account.portfolio_value)
        daily_pnl_pct = ((closing_value - daily_start_value) / daily_start_value) * 100

        if daily_pnl_pct < 0:
            consecutive_loss_days += 1
            print(f"📉 Day closed down {daily_pnl_pct:.2f}% — "
                  f"consecutive losing days: {consecutive_loss_days}")
            if consecutive_loss_days >= COOLOFF_LOSS_DAYS:
                print(f"⚠️ {COOLOFF_LOSS_DAYS}+ consecutive losing days — "
                      f"cool-off active tomorrow (position sizes at 50%)")
        else:
            if consecutive_loss_days > 0:
                print(f"✅ Day closed up {daily_pnl_pct:.2f}% — "
                      f"resetting consecutive loss streak (was {consecutive_loss_days})")
            consecutive_loss_days = 0

    except Exception as e:
        print(f"⚠️ Could not update loss streak: {e}")


def check_position(symbol):
    global high_water_marks, todays_symbols, trailing_stops, entry_times, yesterdays_exits

    position = monitor_position(symbol)
    if not position:
        print(f"ℹ️ No open position found for {symbol}.")
        return

    current_price = position.get("current_price", 0)
    entry_price   = position.get("entry_price", 0)
    pnl           = position.get("profit_loss_pct", 0)

    if symbol not in high_water_marks or current_price > high_water_marks[symbol]:
        high_water_marks[symbol] = current_price

    peak_price         = high_water_marks[symbol]
    drop_from_peak_pct = ((current_price - peak_price) / peak_price) * 100
    trailing_stop_pct  = get_trailing_stop(symbol)
    grace_active       = is_in_grace_period(symbol)

    grace_label = f" [GRACE {GRACE_PERIOD_MINUTES}min active]" if grace_active else ""
    print(f"📈 {symbol} | Current: ${current_price:.2f} | Entry: ${entry_price:.2f} | Peak: ${peak_price:.2f}")
    print(f"   P&L: {pnl:.2f}% | Drop from peak: {drop_from_peak_pct:.2f}% | "
          f"Trailing stop: {trailing_stop_pct}%{grace_label}")

    def _exit(reason_label):
        if not position_exists_in_alpaca(symbol):
            print(f"ℹ️ {symbol} already closed in Alpaca — skipping exit call")
        else:
            exit_trade(symbol)

        if symbol in todays_symbols:
            todays_symbols.remove(symbol)
        trailing_stops.pop(symbol, None)
        entry_times.pop(symbol, None)
        high_water_marks.pop(symbol, None)

        if symbol not in yesterdays_exits:
            yesterdays_exits.append(symbol)

    if pnl >= HARD_PROFIT_CEILING:
        print(f"💰 HARD PROFIT CEILING HIT! {symbol} +{pnl:.2f}%")
        _exit("HARD_CEILING")
        return

    if pnl <= -STOP_LOSS_PCT:
        print(f"🚨 STOP LOSS TRIGGERED! {symbol} {pnl:.2f}%")
        _exit("STOP_LOSS")
        return

    if (not grace_active
            and peak_price > entry_price
            and drop_from_peak_pct <= -trailing_stop_pct):
        print(f"🛡️ TRAILING STOP TRIGGERED! {symbol} locked in {pnl:.2f}% "
              f"(down {abs(drop_from_peak_pct):.2f}% from peak ${peak_price:.2f}, "
              f"stop was {trailing_stop_pct}%)")
        _exit("TRAILING_STOP")
        return

    print(f"⏳ Holding {symbol}. P&L within range.")


# ---- SCHEDULE ----
for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
    getattr(schedule.every(), day).at("09:35").do(run_morning_session)

for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
    getattr(schedule.every(), day).at("15:45").do(run_closing_check)

schedule.every(5).minutes.do(run_intraday_check)

print("⏰ SCHEDULER RUNNING")
print("🌅 Morning session: 9:35 AM (calibration loaded from cache if fresh)")
print("🔄 Intraday checks: every 5 min during market hours (silent when no positions)")
print("🌆 Closing check: 3:45 PM (saves calibration cache for tomorrow)")
print(f"🛡️  Stop Loss: -{STOP_LOSS_PCT}% (always active)")
print(f"⏸️  Grace period: {GRACE_PERIOD_MINUTES} min after entry (trailing stop suppressed)")
print(f"📉 Trailing Stop: per-stock calibrated (default fallback: {TRAILING_STOP_PCT_DEFAULT}%)")
print(f"💰 Hard Profit Ceiling: +{HARD_PROFIT_CEILING}%")
print(f"🛑 Daily loss limit: -{DAILY_LOSS_LIMIT_PCT}% (pauses new trades if hit)")
print(f"❄️  Cool-off: after {COOLOFF_LOSS_DAYS} consecutive losing days (position sizes at 50%)")
print(f"📈 Options trading: {'ENABLED' if ENABLE_OPTIONS else 'DISABLED (flip ENABLE_OPTIONS to True when ready)'}")

while True:
    schedule.run_pending()
    time.sleep(30)
