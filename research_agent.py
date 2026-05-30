import os
import yfinance as yf
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def research_stocks():
    watchlist = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META"]
    results = []

    for symbol in watchlist:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        name = info.get("longName", symbol)
        price = info.get("currentPrice", "N/A")
        pe_ratio = info.get("trailingPE", "N/A")
        week_high = info.get("fiftyTwoWeekHigh", "N/A")
        week_low = info.get("fiftyTwoWeekLow", "N/A")

        summary = f"{name} ({symbol}): Price=${price}, PE={pe_ratio}, 52W High={week_high}, 52W Low={week_low}"
        results.append({"symbol": symbol, "summary": summary})

    research_data = "\n".join([r["summary"] for r in results])

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a stock research analyst. Always end your response with BEST_PICK: followed by just the ticker symbol of your top recommendation."},
            {"role": "user", "content": f"Analyze these stocks and identify the single best buy opportunity today:\n{research_data}"}
        ]
    )

    recommendation = response.choices[0].message.content
    
    # Extract the best pick symbol
    best_pick = "MSFT"
    if "BEST_PICK:" in recommendation:
        best_pick = recommendation.split("BEST_PICK:")[-1].strip().split()[0]

    print("RESEARCH AGENT REPORT:")
    print(recommendation)
    print(f"\nBEST PICK TODAY: {best_pick}")
    
    return {"report": recommendation, "symbol": best_pick}

if __name__ == "__main__":
    research_stocks()