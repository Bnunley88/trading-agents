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