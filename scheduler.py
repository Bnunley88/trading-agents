import schedule
import time
from research_agent import research_stocks
from analyst_agent import analyze_recommendation
from executor_agent import execute_trade
from monitor_agent import monitor_position
from exit_agent import exit_trade

# Track today's position globally
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