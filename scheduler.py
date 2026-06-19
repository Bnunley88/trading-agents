import schedule
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from research_agent import research_stocks
from analyst_agent import analyze_recommendation
from executor_agent import execute_trades
from monitor_agent import monitor_position
from exit_agent import exit_trade

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

todays_symbols = []
high_water_marks = {}  # tracks the highest price seen per symbol since entry today

TRAILING_STOP_PCT = 2.0   # sell if price drops this % from the peak since entry
HARD_PROFIT_CEILING = 5.0  # backstop: sell immediately if gain ever hits this %, no matter what

def run_morning_session():
    global todays_symbols, high_water_marks
    print("\n🌅 MORNING SESSION - 9:35 AM")
    print("="*50)

    print("\n📊 STEP 1: RESEARCHING STOCKS...")
    research_result = research_stocks()
    research_report = research_result["report"]
    top_symbols = research_result["symbols"]
    todays_symbols = top_symbols
    high_water_marks = {}  # reset tracking for the new day's positions

    print("\n🧠 STEP 2: ANALYZING OPPORTUNITIES...")
    decision = analyze_recommendation(research_report)

    print("\n⚡ STEP 3: EXECUTING TRADES...")
    print(f"Trading today's top 3 picks: {top_symbols}")
    execute_trades(top_symbols)

    print("\n👁️ STEP 4: MONITORING POSITIONS...")
    time.sleep(2)
    for symbol in todays_symbols:
        check_position(symbol)

    print("\n✅ MORNING SESSION COMPLETE!")

def run_intraday_check(label):
    global todays_symbols
    if not todays_symbols:
        print(f"⚠️ No positions to monitor at {label}.")
        return
    print(f"\n🔄 INTRADAY CHECK - {label}")
    print("="*50)
    for symbol in todays_symbols:
        check_position(symbol)

def run_closing_check():
    global todays_symbols, high_water_marks
    if not todays_symbols:
        print("⚠️ No positions to monitor at close.")
        return
    print("\n🌆 CLOSING CHECK - 3:45 PM")
    print("="*50)
    for symbol in todays_symbols:
        check_position(symbol)
    todays_symbols = []
    high_water_marks = {}

def check_position(symbol):
    global high_water_marks
    position = monitor_position(symbol)
    if not position:
        print(f"ℹ️ No open position found for {symbol}.")
        return

    current_price = position.get("current_price", 0)
    entry_price = position.get("entry_price", 0)
    pnl = position.get("profit_loss_pct", 0)

    # Update the high water mark (highest price seen since entry today)
    if symbol not in high_water_marks or current_price > high_water_marks[symbol]:
        high_water_marks[symbol] = current_price

    peak_price = high_water_marks[symbol]
    drop_from_peak_pct = ((current_price - peak_price) / peak_price) * 100

    print(f"📈 {symbol} | Current: ${current_price:.2f} | Entry: ${entry_price:.2f} | Peak: ${peak_price:.2f}")
    print(f"   P&L from entry: {pnl:.2f}% | Drop from peak: {drop_from_peak_pct:.2f}%")

    # Hard backstop: sell immediately if gain ever hits the ceiling, regardless of trailing stop
    if pnl >= HARD_PROFIT_CEILING:
        print(f"💰 HARD PROFIT CEILING HIT! {symbol} Gain: {pnl:.2f}%")
        exit_trade(symbol)
        todays_symbols.remove(symbol)
        return

    # Original fixed stop loss: protects against a position that never gained ground
    if pnl <= -2.0:
        print(f"🚨 STOP LOSS TRIGGERED! {symbol} Loss: {pnl:.2f}%")
        exit_trade(symbol)
        todays_symbols.remove(symbol)
        return

    # Trailing stop: protects gains once the position has moved up from entry
    if peak_price > entry_price and drop_from_peak_pct <= -TRAILING_STOP_PCT:
        print(f"🛡️ TRAILING STOP TRIGGERED! {symbol} Locked in {pnl:.2f}% (down {abs(drop_from_peak_pct):.2f}% from peak ${peak_price:.2f})")
        exit_trade(symbol)
        todays_symbols.remove(symbol)
        return

    print(f"⏳ Holding {symbol}. P&L within range.")

# Morning session: research + buy + first check
for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
    getattr(schedule.every(), day).at("09:35").do(run_morning_session)

# Intraday checks every 30 minutes from 10:00 AM to 3:30 PM Central
intraday_times = [
    "10:00", "10:30", "11:00", "11:30",
    "12:00", "12:30", "13:00", "13:30",
    "14:00", "14:30", "15:00", "15:30"
]
for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
    for t in intraday_times:
        getattr(schedule.every(), day).at(t).do(run_intraday_check, label=t)

# Closing check
for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
    getattr(schedule.every(), day).at("15:45").do(run_closing_check)

print("⏰ SCHEDULER RUNNING")
print("🌅 Morning buy: 9:35 AM")
print("🔄 Intraday checks: every 30 min, 10:00 AM - 3:30 PM")
print("🌆 Closing check: 3:45 PM")
print(f"🛡️ Stop Loss: -2% | 📉 Trailing Stop: {TRAILING_STOP_PCT}% from peak | 💰 Hard Profit Ceiling: {HARD_PROFIT_CEILING}%")

while True:
    schedule.run_pending()
    time.sleep(60)
