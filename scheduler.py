import schedule
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from research_agent import research_stocks
from analyst_agent import analyze_recommendation
from executor_agent import execute_trade
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

todays_symbol = None

def run_morning_session():
    global todays_symbol
    print("\n🌅 MORNING SESSION - 9:35 AM")
    print("="*50)
    print("\n📊 STEP 1: RESEARCHING STOCKS...")
    research_result = research_stocks()
    research_report = research_result["report"]
    best_symbol = research_result["symbol"]
    todays_symbol = best_symbol
    print("\n🧠 STEP 2: ANALYZING OPPORTUNITIES...")
    decision = analyze_recommendation(research_report)
    print("\n⚡ STEP 3: EXECUTING TRADE...")
    print(f"Trading today's best pick: {best_symbol}")
    execute_trade(best_symbol, 1)
    print("\n👁️ STEP 4: MONITORING POSITION...")
    time.sleep(2)
    check_position(best_symbol)
    print("\n✅ MORNING SESSION COMPLETE!")

def run_midday_check():
    global todays_symbol
    if not todays_symbol:
        print("⚠️ No position to monitor at midday.")
        return
    print("\n☀️ MIDDAY CHECK - 1:00 PM")
    print("="*50)
    check_position(todays_symbol)

def run_closing_check():
    global todays_symbol
    if not todays_symbol:
        print("⚠️ No position to monitor at close.")
        return
    print("\n🌆 CLOSING CHECK - 3:45 PM")
    print("="*50)
    check_position(todays_symbol)
    todays_symbol = None

def check_position(symbol):
    position = monitor_position(symbol)
    if position:
        pnl = position.get("profit_loss_pct", 0)
        print(f"📈 Current P&L: {pnl:.2f}%")
        if pnl < -2.0:
            print(f"🚨 STOP LOSS TRIGGERED! Loss: {pnl:.2f}%")
            exit_trade(symbol)
        elif pnl > 5.0:
            print(f"💰 PROFIT TARGET HIT! Gain: {pnl:.2f}%")
            exit_trade(symbol)
        else:
            print(f"⏳ Holding position. P&L within range.")
    else:
        print("ℹ️ No open position found.")

for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
    getattr(schedule.every(), day).at("09:35").do(run_morning_session)
    getattr(schedule.every(), day).at("13:00").do(run_midday_check)
    getattr(schedule.every(), day).at("15:45").do(run_closing_check)

print("⏰ SCHEDULER RUNNING")
print("🌅 Morning buy: 9:35 AM")
print("☀️ Midday check: 1:00 PM")
print("🌆 Closing check: 3:45 PM")
print("🛡️ Stop Loss: 2% | 💰 Profit Target: 5%")

while True:
    schedule.run_pending()
    time.sleep(60)