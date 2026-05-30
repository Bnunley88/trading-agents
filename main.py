import time
from research_agent import research_stocks
from analyst_agent import analyze_recommendation
from executor_agent import execute_trade
from monitor_agent import monitor_position

print("🤖 AI TRADING SYSTEM STARTING...")
print("="*50)

# Step 1 - Research
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

# Step 4 - Monitor
print("\n👁️ STEP 4: MONITORING POSITION...")
time.sleep(2)
monitor_position(best_symbol)

print("\n✅ SYSTEM COMPLETE!")