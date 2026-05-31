import os
import yfinance as yf
import requests
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def get_news_sentiment(symbol, company_name):
    if not NEWS_API_KEY:
        return "No news data available"
    try:
        url = f"https://newsapi.org/v2/everything?q={company_name}&from={datetime.now().strftime('%Y-%m-%d')}&sortBy=publishedAt&apiKey={NEWS_API_KEY}&pageSize=5"
        response = requests.get(url, timeout=5)
        articles = response.json().get("articles", [])
        if not articles:
            return "No recent news"
        headlines = [a["title"] for a in articles[:5]]
        return " | ".join(headlines)
    except:
        return "News fetch failed"

def research_stocks():
    watchlist = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META"]
    results = []

    for symbol in watchlist:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="10d")

            name = info.get("longName", symbol)
            price = info.get("currentPrice", "N/A")
            pe_ratio = info.get("trailingPE", "N/A")
            week_high = info.get("fiftyTwoWeekHigh", "N/A")
            week_low = info.get("fiftyTwoWeekLow", "N/A")
            volume = info.get("volume", "N/A")
            avg_volume = info.get("averageVolume", "N/A")
            market_cap = info.get("marketCap", "N/A")
            rsi = "N/A"
            momentum = "N/A"

            if len(hist) >= 5:
                closes = hist["Close"].tolist()
                momentum = round(((closes[-1] - closes[0]) / closes[0]) * 100, 2)

            if len(hist) >= 10:
                delta = hist["Close"].diff()
                gain = delta.clip(lower=0).rolling(window=7).mean()
                loss = (-delta.clip(upper=0)).rolling(window=7).mean()
                rs = gain / loss
                rsi_series = 100 - (100 / (1 + rs))
                rsi = round(rsi_series.iloc[-1], 1)

            volume_ratio = "N/A"
            if volume != "N/A" and avg_volume and avg_volume > 0:
                volume_ratio = round(volume / avg_volume, 2)

            news = get_news_sentiment(symbol, name)

            summary = (
                f"{name} ({symbol}): Price=${price}, PE={pe_ratio}, "
                f"52W High={week_high}, 52W Low={week_low}, "
                f"RSI={rsi}, 5-Day Momentum={momentum}%, "
                f"Volume Ratio={volume_ratio}x avg, "
                f"Market Cap={market_cap}, "
                f"Recent News: {news}"
            )
            results.append({"symbol": symbol, "summary": summary})

        except Exception as e:
            print(f"Error fetching {symbol}: {e}")

    research_data = "\n".join([r["summary"] for r in results])

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """You are an expert stock research analyst. 
Analyze stocks using price action, RSI, momentum, volume, and news sentiment.
RSI below 30 = oversold (potential buy). RSI above 70 = overbought (avoid).
High volume ratio = strong interest. Positive momentum = uptrend.
Always end your response with BEST_PICK: followed by just the ticker symbol."""},
            {"role": "user", "content": f"Analyze these stocks and pick the single best buy opportunity today:\n{research_data}"}
        ]
    )

    recommendation = response.choices[0].message.content

    best_pick = "MSFT"
    if "BEST_PICK:" in recommendation:
        best_pick = recommendation.split("BEST_PICK:")[-1].strip().split()[0]

    print("RESEARCH AGENT REPORT:")
    print(recommendation)
    print(f"\nBEST PICK TODAY: {best_pick}")

    return {"report": recommendation, "symbol": best_pick}

if __name__ == "__main__":
    research_stocks()