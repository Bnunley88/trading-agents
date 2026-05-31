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

def get_earnings_info(ticker_obj):
    try:
        calendar = ticker_obj.calendar
        if calendar is None or calendar.empty:
            return "No earnings data"
        earnings_date = calendar.iloc[0]["Earnings Date"]
        if isinstance(earnings_date, list):
            earnings_date = earnings_date[0]
        days_until = (earnings_date - datetime.now()).days
        if days_until < 0:
            return "Earnings passed"
        elif days_until <= 3:
            return f"⚠️ EARNINGS IN {days_until} DAYS - HIGH VOLATILITY EXPECTED"
        elif days_until <= 7:
            return f"📅 Earnings in {days_until} days - watch closely"
        else:
            return f"Earnings in {days_until} days"
    except:
        return "Earnings data unavailable"

def get_insider_activity(ticker_obj):
    try:
        insider = ticker_obj.insider_purchases
        if insider is None or insider.empty:
            return "No insider data"
        recent = insider.head(3)
        buys = recent[recent["Transaction"] == "Buy"]
        if not buys.empty:
            total = buys["Value"].sum()
            return f"🟢 Insider buying detected! ${total:,.0f} purchased recently"
        return "No recent insider buying"
    except:
        return "Insider data unavailable"

def get_analyst_rating(ticker_obj):
    try:
        recommendations = ticker_obj.recommendations
        if recommendations is None or recommendations.empty:
            return "No analyst data"
        recent = recommendations.tail(5)
        upgrades = recent[recent["To Grade"].isin(["Buy", "Strong Buy", "Outperform", "Overweight"])]
        if not upgrades.empty:
            latest = upgrades.iloc[-1]
            return f"🟢 {latest['Firm']} rates {latest['To Grade']}"
        return "No recent upgrades"
    except:
        return "Analyst data unavailable"

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
            earnings = get_earnings_info(ticker)
            insider = get_insider_activity(ticker)
            analyst = get_analyst_rating(ticker)

            summary = (
                f"{name} ({symbol}): Price=${price}, PE={pe_ratio}, "
                f"52W High={week_high}, 52W Low={week_low}, "
                f"RSI={rsi}, 5-Day Momentum={momentum}%, "
                f"Volume Ratio={volume_ratio}x avg, "
                f"Market Cap={market_cap}\n"
                f"  Earnings: {earnings}\n"
                f"  Insider Activity: {insider}\n"
                f"  Analyst Rating: {analyst}\n"
                f"  News: {news}"
            )
            results.append({"symbol": symbol, "summary": summary})

        except Exception as e:
            print(f"Error fetching {symbol}: {e}")

    research_data = "\n".join([r["summary"] for r in results])

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """You are an expert stock research analyst.
Analyze stocks using technical indicators, earnings timing, insider activity, and analyst ratings.
RSI below 30 = oversold (potential buy). RSI above 70 = overbought (avoid).
High volume ratio = strong institutional interest.
Insider buying = very bullish signal.
Analyst upgrades = momentum catalyst.
Stocks with earnings in 1-3 days = HIGH RISK, avoid unless strong conviction.
Stocks with earnings in 4-7 days = potential opportunity for pre-earnings run.
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