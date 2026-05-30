import schedule
import time
from research_agent import research_stocks
from analyst_agent import analyze_recommendation
from executor_agent import execute_trade
from monitor_agent import monitor_position
from exit_agent import exit_trade

def run_trading_system():
    print("\n🤖 AUTOMATED TRADING SYSTEM RUNNING...")
    print("="*50)
    
    # Step 1 - Research and get best stock
    print("\n📊 STEP 1: RESEARCHING STOCKS...")
    research_result = research_stocks()
    research_report = research_result["report"]
    best_symbol = research_result["symbol"]
    
    # Step 2 - Analyze
    print("\n🧠 STEP 2: ANALYZING OPPORTUNITIES...")
    decision = analyze_recommendation(research_report)
    
    # Step 3 - Execute with dynamic symbol
    print("\n⚡ STEP 3: EXECUTING TRADE...")
    print(f"Trading today's best pick: {best_symbol}")
    execute_trade(best_symbol, 1)
    
    # Step 4 - Monitor and check stop loss
    print("\n👁️ STEP 4: MONITORING POSITION...")
    time.sleep(2)
    position = monitor_position(best_symbol)
    
    # Step 5 - Auto exit if loss exceeds 2%
    if position and position["profit_loss_pct"] < -2.0:
        print(f"\n🚨 STOP LOSS TRIGGERED! Loss exceeded 2%")
        exit_trade(best_symbol)
    elif position and position["profit_loss_pct"] > 5.0:
        print(f"\n💰 PROFIT TARGET HIT! Taking profits!")
        exit_trade(best_symbol)
    
    print("\n✅ TRADING CYCLE COMPLETE!")

schedule.every().monday.at("09:35").do(run_trading_system)
schedule.every().tuesday.at("09:35").do(run_trading_system)
schedule.every().wednesday.at("09:35").do(run_trading_system)
schedule.every().thursday.at("09:35").do(run_trading_system)
schedule.every().friday.at("09:35").do(run_trading_system)

print("⏰ SCHEDULER RUNNING - Trading every weekday at 9:35 AM")
print("🛡️ Stop Loss: 2% | 💰 Profit Target: 5%")
print("Press Ctrl+C to stop")

while True:
    schedule.run_pending()
    time.sleep(60)