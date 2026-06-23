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

# ---- STATE ----
todays_symbols  = []
high_water_marks = {}
entry_times      = {}   # { symbol: datetime } — used for grace period tracking
trailing_stops   = {}   # { symbol: float }    — per-stock calibrated trailing stop %

# ---- CONSTANTS ----
HARD_PROFIT_CEILING      = 5.0
STOP_LOSS_PCT            = 2.0
TRAILING_STOP_PCT_DEFAULT = 2.0   # fallback when no calibration available
GRACE_PERIOD_MINUTES     = 90     # trailing stop suppressed for 90 min after entry
MAX_POSITIONS            = 3
MIN_BUYING_POWER         = 1000.0

# ---- OPTIONS FLAG ----
# Set to True only after:
#   - Live Alpaca account with options approval
#   - Proven 4+ weeks of consistent paper results
#   - Confirmed portfolio_value sizing (not buying_power)
ENABLE_OPTIONS = False

if ENABLE_OPTIONS:
    from options_agent import buy_call_option

# ---- CALIBRATION CACHE ----
CACHE_FILE    = "calibration_cache.json"
CACHE_MAX_AGE = timedelta(hours=23)   # refresh daily, stale after 23h

def load_calibration_cache():
    """Load cached trailing stops from disk. Returns {} if missing or stale."""
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
    """Save trailing stops to disk with a timestamp."""
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
    """Per-stock calibrated stop, falling back to global default."""
    return trailing_stops.get(symbol, TRAILING_STOP_PCT_DEFAULT)


def is_in_grace_period(symbol):
    """True if the position was entered less than GRACE_PERIOD_MINUTES ago."""
    if symbol not in entry_times:
        return False
    elapsed = (datetime.now() - entry_times[symbol]).total_seconds() / 60
    return elapsed < GRACE_PERIOD_MINUTES


# ---- SESSIONS ----

def run_morning_session():
    global todays_symbols, high_water_marks, trailing_stops, entry_times
    print("\n🌅 MORNING SESSION - 9:35 AM")
    print("=" * 50)

    try:
        existing_positions = api.list_positions()
        existing_symbols   = [p.symbol for p in existing_positions]
        account            = api.get_account()
        buying_power       = float(account.buying_power)

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
        print(f"⚠️ Could not check positions/buying power ({e}), proceeding normally.")
        existing_symbols = []

    # ---- Load calibration from cache (fast) or recompute (slow) ----
    cached = load_calibration_cache()
    if cached:
        trailing_stops.update(cached)
        print(f"📐 Using cached trailing stops: {trailing_stops}")
    # research_stocks() will run its own calibration and we merge below

    print("\n📊 STEP 1: RESEARCHING STOCKS + CALIBRATING TRAILING STOPS...")
    research_result  = research_stocks()
    research_report  = research_result["report"]
    top_symbols      = research_result["symbols"]
    new_stops        = research_result.get("trailing_stops", {})

    trailing_stops.update(new_stops)
    print(f"📐 Active trailing stops: {trailing_stops}")

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
    execute_trades(new_symbols, conviction_scores=conviction_scores)

    # Record entry times for grace period tracking
    now = datetime.now()
    for symbol in new_symbols:
        entry_times[symbol] = now

    # ---- Options layer (dormant until ENABLE_OPTIONS = True) ----
    if ENABLE_OPTIONS and new_symbols:
        try:
            account        = api.get_account()
            portfolio_value = float(account.portfolio_value)
            top_pick       = new_symbols[0]
            print(f"\n📈 OPTIONS: Buying call on top pick {top_pick}...")
            buy_call_option(top_pick, portfolio_value)
        except Exception as e:
            print(f"⚠️ OPTIONS: Skipping — {e}")

    print("\n👁️ STEP 4: MONITORING POSITIONS...")
    time.sleep(2)
    for symbol in todays_symbols:
        check_position(symbol)

    print("\n✅ MORNING SESSION COMPLETE!")


def run_intraday_check(label):
    if not todays_symbols:
        print(f"⚠️ No positions to monitor at {label}.")
        return
    print(f"\n🔄 INTRADAY CHECK - {label}")
    print("=" * 50)
    for symbol in todays_symbols:
        check_position(symbol)


def run_closing_check():
    global todays_symbols, high_water_marks, trailing_stops, entry_times
    if not todays_symbols:
        print("⚠️ No positions to monitor at close.")
        return
    print("\n🌆 CLOSING CHECK - 3:45 PM")
    print("=" * 50)

    for symbol in list(todays_symbols):
        check_position(symbol)

    # Save calibration cache at end of day so tomorrow's morning session
    # can skip recomputing and go straight to the research loop.
    if trailing_stops:
        save_calibration_cache(trailing_stops)

    # Reset intraday state but keep trailing_stops — overnight positions
    # need their calibrated stop available at tomorrow morning's first check.
    todays_symbols   = []
    high_water_marks = {}
    entry_times      = {}


def check_position(symbol):
    global high_water_marks, todays_symbols, trailing_stops, entry_times

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
        exit_trade(symbol)
        if symbol in todays_symbols:
            todays_symbols.remove(symbol)
        trailing_stops.pop(symbol, None)
        entry_times.pop(symbol, None)
        high_water_marks.pop(symbol, None)

    # Hard profit ceiling — always active
    if pnl >= HARD_PROFIT_CEILING:
        print(f"💰 HARD PROFIT CEILING HIT! {symbol} +{pnl:.2f}%")
        _exit("HARD_CEILING")
        return

    # Hard stop loss — always active (even in grace period)
    if pnl <= -STOP_LOSS_PCT:
        print(f"🚨 STOP LOSS TRIGGERED! {symbol} {pnl:.2f}%")
        _exit("STOP_LOSS")
        return

    # Trailing stop — suppressed during grace period
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

intraday_times = [
    "10:00", "10:30", "11:00", "11:30",
    "12:00", "12:30", "13:00", "13:30",
    "14:00", "14:30", "15:00", "15:30"
]
for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
    for t in intraday_times:
        getattr(schedule.every(), day).at(t).do(run_intraday_check, label=t)

for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
    getattr(schedule.every(), day).at("15:45").do(run_closing_check)

print("⏰ SCHEDULER RUNNING")
print("🌅 Morning session: 9:35 AM (calibration loaded from cache if fresh)")
print("🔄 Intraday checks: every 30 min, 10:00 AM - 3:30 PM")
print("🌆 Closing check: 3:45 PM (saves calibration cache for tomorrow)")
print(f"🛡️ Stop Loss: -{STOP_LOSS_PCT}% (always active)")
print(f"⏸️  Grace period: {GRACE_PERIOD_MINUTES} min after entry (trailing stop suppressed)")
print(f"📉 Trailing Stop: per-stock calibrated (default fallback: {TRAILING_STOP_PCT_DEFAULT}%)")
print(f"💰 Hard Profit Ceiling: +{HARD_PROFIT_CEILING}%")
print(f"📈 Options trading: {'ENABLED' if ENABLE_OPTIONS else 'DISABLED (flip ENABLE_OPTIONS to True when ready)'}")

while True:
    schedule.run_pending()
    time.sleep(60)
