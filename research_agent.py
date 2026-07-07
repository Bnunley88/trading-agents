import os
import re
import yfinance as yf
import requests
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from backtest import calibrate_trailing_stop, TRAILING_STOP_PCT

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
NEWS_API_KEY      = os.getenv("NEWS_API_KEY")
FINNHUB_API_KEY   = os.getenv("FINNHUB_API_KEY")
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

MIN_MARKET_CAP = 10_000_000_000

# ---- QUALITY FILTERS ----
MIN_CONVICTION_SCORE = 0.25
MIN_TRAILING_STOP    = 2.5
EARNINGS_BLOCK_DAYS  = 5

# ---- DOWNTREND FILTER ----
# Block stocks where price is this % below their 30-day high
# Catches falling knives like IREN that look oversold but are in freefall
DOWNTREND_BLOCK_PCT  = 15.0

# ---- HIGH CONVICTION ANCHOR LIST ----
# Trimmed to top 12 highest conviction names based on backtest results
# Sorted by historical avg P&L
HIGH_CONVICTION_ANCHORS = [
    "ARM",   # +2.43%
    "HOOD",  # +2.43%
    "DASH",  # +2.73%
    "SHOP",  # +2.34%
    "JPM",   # +2.18%
    "UNH",   # +1.95%
    "GS",    # +1.54%
    "SNOW",  # +1.52%
    "CAT",   # +1.44%
    "META",  # +1.73%
    "JNJ",   # +1.45%
    "NOW",   # +1.02%
]

FALLBACK_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "META", "TSLA", "AMZN", "GOOGL",
    "AMD", "AVGO", "MU", "CRM", "ORCL", "NOW", "UBER", "SHOP",
    "PLTR", "COIN", "ARM", "SMCI", "NFLX",
    "JPM", "V", "MA", "GS", "PYPL",
    "UNH", "LLY", "JNJ",
    "XOM", "CVX",
    "BA", "CAT", "HD", "WMT", "NKE", "DIS", "DASH", "SNOW",
    "MARA", "HOOD"
]

def get_dynamic_watchlist(top_n=20):
    """
    Builds a blended watchlist:
    1. Alpaca most-actives screener (live market momentum)
    2. HIGH_CONVICTION_ANCHORS (trimmed to top 12 proven stocks)
    """
    screener_stocks = []
    try:
        url = "https://data.alpaca.markets/v1beta1/screener/stocks/most-actives"
        headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
        }
        params   = {"top": 50, "by": "volume"}
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data     = response.json()

        candidates = data.get("most_actives", [])
        for stock in candidates:
            symbol = stock.get("symbol", "")
            if not symbol or "." in symbol:
                continue
            try:
                info       = yf.Ticker(symbol).info
                market_cap = info.get("marketCap", 0) or 0
                if market_cap >= MIN_MARKET_CAP:
                    screener_stocks.append(symbol)
                if len(screener_stocks) >= top_n:
                    break
            except:
                continue

        if screener_stocks:
            print(f"📋 Alpaca screener: {len(screener_stocks)} stocks — {screener_stocks}")
        else:
            print("⚠️ Alpaca screener returned no results.")

    except Exception as e:
        print(f"⚠️ Alpaca screener failed ({e})")

    combined = list(screener_stocks)
    for symbol in HIGH_CONVICTION_ANCHORS:
        if symbol not in combined:
            combined.append(symbol)

    if not combined:
        print("⚠️ Using full fallback watchlist.")
        return FALLBACK_WATCHLIST

    print(f"📋 Blended watchlist ({len(combined)} stocks): {combined}")
    return combined


def check_earnings_block(symbol):
    try:
        ticker    = yf.Ticker(symbol)
        calendar  = ticker.calendar
        if calendar is None or calendar.empty:
            return False, None
        earnings_date = calendar.iloc[0]["Earnings Date"]
        if isinstance(earnings_date, list):
            earnings_date = earnings_date[0]
        days_until = (earnings_date - datetime.now()).days
        if 0 <= days_until <= EARNINGS_BLOCK_DAYS:
            return True, days_until
        return False, days_until
    except:
        return False, None


def check_downtrend(symbol):
    """
    Returns (blocked: bool, pct_below_high: float)
    Blocks stocks that are more than DOWNTREND_BLOCK_PCT% below their 30-day high.
    Catches falling knives that look oversold on RSI but are in a clear downtrend.
    A stock can be RSI=20 and still be a falling knife — this filter catches that.
    """
    try:
        hist = yf.Ticker(symbol).history(period="30d")
        if hist.empty or len(hist) < 5:
            return False, 0.0
        high_30d     = hist["Close"].max()
        current      = hist["Close"].iloc[-1]
        pct_below    = ((high_30d - current) / high_30d) * 100
        if pct_below >= DOWNTREND_BLOCK_PCT:
            return True, round(pct_below, 1)
        return False, round(pct_below, 1)
    except:
        return False, 0.0


def get_vwap(hist):
    """
    Calculates VWAP (Volume Weighted Average Price) for the last 5 days
    of intraday-equivalent data using daily OHLC.
    Returns (vwap, pct_above_vwap) where positive = bullish, negative = bearish.
    """
    try:
        if hist.empty or len(hist) < 5:
            return "N/A", "N/A"
        recent       = hist.tail(5)
        typical_price = (recent["High"] + recent["Low"] + recent["Close"]) / 3
        vwap         = (typical_price * recent["Volume"]).sum() / recent["Volume"].sum()
        current      = hist["Close"].iloc[-1]
        pct_vs_vwap  = round(((current - vwap) / vwap) * 100, 2)
        return round(vwap, 2), pct_vs_vwap
    except:
        return "N/A", "N/A"


def get_news_sentiment(symbol, company_name):
    if not NEWS_API_KEY:
        return "No news data available"
    try:
        url      = (f"https://newsapi.org/v2/everything?q={company_name}"
                    f"&from={datetime.now().strftime('%Y-%m-%d')}"
                    f"&sortBy=publishedAt&apiKey={NEWS_API_KEY}&pageSize=5")
        response = requests.get(url, timeout=5)
        articles = response.json().get("articles", [])
        if not articles:
            return "No recent news"
        return " | ".join(a["title"] for a in articles[:5])
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
        buys   = recent[recent["Transaction"] == "Buy"]
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
        recent   = recommendations.tail(5)
        upgrades = recent[recent["To Grade"].isin(
            ["Buy", "Strong Buy", "Outperform", "Overweight"])]
        if not upgrades.empty:
            latest = upgrades.iloc[-1]
            return f"🟢 {latest['Firm']} rates {latest['To Grade']}"
        return "No recent upgrades"
    except:
        return "Analyst data unavailable"

def get_finnhub_data(symbol):
    if not FINNHUB_API_KEY:
        return "No Finnhub data available"
    try:
        quote_resp  = requests.get(
            f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}",
            timeout=5)
        quote       = quote_resp.json()
        current     = quote.get("c", "N/A")
        prev_close  = quote.get("pc", "N/A")
        day_high    = quote.get("h", "N/A")
        day_low     = quote.get("l", "N/A")
        change_pct  = "N/A"
        if current not in ("N/A", None) and prev_close not in ("N/A", None, 0):
            change_pct = round(((current - prev_close) / prev_close) * 100, 2)

        metric_resp = requests.get(
            f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={FINNHUB_API_KEY}",
            timeout=5)
        metrics     = metric_resp.json().get("metric", {})
        beta        = metrics.get("beta", "N/A")
        week52_high = metrics.get("52WeekHigh", "N/A")
        week52_low  = metrics.get("52WeekLow",  "N/A")

        return (f"Finnhub Price=${current} (Day Change={change_pct}%), "
                f"Day Range=${day_low}-${day_high}, "
                f"Beta={beta}, 52W High/Low=${week52_high}/${week52_low}")
    except Exception as e:
        return f"Finnhub data unavailable ({e})"


def calibrate_watchlist_parallel(watchlist):
    print(f"\n📐 Calibrating trailing stops for {len(watchlist)} stocks in parallel...")
    calibration = {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_symbol = {
            executor.submit(calibrate_trailing_stop, symbol): symbol
            for symbol in watchlist
        }
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                optimal_pct, report, best_avg_pnl = future.result()
                calibration[symbol] = (optimal_pct, report, best_avg_pnl)
            except Exception as e:
                print(f"⚠️ Calibration error for {symbol}: {e}")
                calibration[symbol] = (TRAILING_STOP_PCT,
                                       f"{symbol}: error, using default {TRAILING_STOP_PCT}%",
                                       None)

    print("✅ Calibration complete.\n")
    return calibration


def research_stocks():
    print("🔍 Building blended watchlist (Alpaca screener + top 12 conviction anchors)...")
    watchlist = get_dynamic_watchlist(top_n=20)

    calibration = calibrate_watchlist_parallel(watchlist)

    # ---- FILTER 1: Hard block earnings ----
    earnings_blocked     = []
    post_earnings_filter = []
    for symbol in watchlist:
        blocked, days = check_earnings_block(symbol)
        if blocked:
            earnings_blocked.append(f"{symbol} (earnings in {days}d)")
        else:
            post_earnings_filter.append(symbol)

    if earnings_blocked:
        print(f"🚫 Earnings blocked: {', '.join(earnings_blocked)}")

    # ---- FILTER 2: Downtrend filter (falling knife protection) ----
    downtrend_blocked      = []
    post_downtrend_filter  = []
    for symbol in post_earnings_filter:
        blocked, pct_below = check_downtrend(symbol)
        if blocked:
            downtrend_blocked.append(f"{symbol} ({pct_below}% below 30d high)")
        else:
            post_downtrend_filter.append(symbol)

    if downtrend_blocked:
        print(f"🚫 Downtrend blocked: {', '.join(downtrend_blocked)}")

    # ---- FILTERS 3-5: Negative P&L, low conviction, choppy ----
    filtered_watchlist = []
    rejected           = []
    for symbol in post_downtrend_filter:
        optimal_stop, _, best_avg_pnl = calibration.get(symbol, (TRAILING_STOP_PCT, "", None))
        pnl = float(best_avg_pnl) if best_avg_pnl is not None else None

        if pnl is None:
            filtered_watchlist.append(symbol)
            continue
        if pnl <= 0:
            rejected.append(f"{symbol} (negative P&L: {pnl:+.2f}%)")
            continue
        if pnl < MIN_CONVICTION_SCORE:
            rejected.append(f"{symbol} (low conviction: {pnl:+.2f}% < {MIN_CONVICTION_SCORE}%)")
            continue
        if optimal_stop < MIN_TRAILING_STOP:
            rejected.append(f"{symbol} (choppy: stop {optimal_stop}% < {MIN_TRAILING_STOP}% min)")
            continue

        filtered_watchlist.append(symbol)

    if rejected:
        print(f"🚫 Filtered out: {', '.join(rejected)}")

    # Relax to positive P&L only if nothing passes
    if not filtered_watchlist:
        print("⚠️ All stocks filtered out — relaxing conviction threshold, keeping positive P&L only")
        for symbol in post_downtrend_filter:
            _, _, best_avg_pnl = calibration.get(symbol, (TRAILING_STOP_PCT, "", None))
            pnl = float(best_avg_pnl) if best_avg_pnl is not None else None
            if pnl is None or pnl > 0:
                filtered_watchlist.append(symbol)

    print(f"✅ Passing {len(filtered_watchlist)} stocks to GPT-4o: {filtered_watchlist}\n")

    if not filtered_watchlist:
        print("🛑 No quality stocks found today — skipping trades.")
        return {
            "report":            "No quality stocks passed filters today.",
            "symbols":           [],
            "trailing_stops":    {},
            "conviction_scores": {}
        }

    results = []
    for symbol in filtered_watchlist:
        try:
            ticker = yf.Ticker(symbol)
            info   = ticker.info
            hist   = ticker.history(period="30d")

            name       = info.get("longName", symbol)
            price      = info.get("currentPrice", "N/A")
            pe_ratio   = info.get("trailingPE", "N/A")
            week_high  = info.get("fiftyTwoWeekHigh", "N/A")
            week_low   = info.get("fiftyTwoWeekLow",  "N/A")
            volume     = info.get("volume", "N/A")
            avg_volume = info.get("averageVolume", "N/A")
            market_cap = info.get("marketCap", "N/A")
            rsi        = "N/A"
            momentum   = "N/A"

            if len(hist) >= 5:
                closes   = hist["Close"].tolist()
                momentum = round(((closes[-1] - closes[0]) / closes[0]) * 100, 2)

            if len(hist) >= 15:
                delta      = hist["Close"].diff()
                gain       = delta.clip(lower=0).rolling(window=14).mean()
                loss       = (-delta.clip(upper=0)).rolling(window=14).mean()
                rs         = gain / loss
                rsi_series = 100 - (100 / (1 + rs))
                rsi        = round(rsi_series.iloc[-1], 1)

            volume_ratio = "N/A"
            if volume != "N/A" and avg_volume and avg_volume > 0:
                volume_ratio = round(volume / avg_volume, 2)

            # VWAP calculation
            vwap, pct_vs_vwap = get_vwap(hist)
            if pct_vs_vwap != "N/A":
                vwap_label = f"VWAP=${vwap} | Price vs VWAP: {pct_vs_vwap:+.2f}% " \
                             f"({'above — bullish' if pct_vs_vwap > 0 else 'below — bearish'})"
            else:
                vwap_label = "VWAP: unavailable"

            news     = get_news_sentiment(symbol, name)
            earnings = get_earnings_info(ticker)
            insider  = get_insider_activity(ticker)
            analyst  = get_analyst_rating(ticker)
            finnhub  = get_finnhub_data(symbol)

            optimal_stop, _, _ = calibration.get(symbol, (TRAILING_STOP_PCT, "", None))
            _, _, best_avg_pnl = calibration.get(symbol, (TRAILING_STOP_PCT, "", None))
            pnl_display        = f"{float(best_avg_pnl):+.2f}%" if best_avg_pnl is not None else "N/A"

            summary = (
                f"{name} ({symbol}): Price=${price}, PE={pe_ratio}, "
                f"52W High={week_high}, 52W Low={week_low}, "
                f"RSI={rsi}, 5-Day Momentum={momentum}%, "
                f"Volume Ratio={volume_ratio}x avg, Market Cap={market_cap}\n"
                f"  Earnings: {earnings}\n"
                f"  Insider Activity: {insider}\n"
                f"  Analyst Rating: {analyst}\n"
                f"  News: {news}\n"
                f"  Cross-check: {finnhub}\n"
                f"  {vwap_label}\n"
                f"  Backtest-calibrated trailing stop: {optimal_stop}% "
                f"(avg P&L: {pnl_display}, 90-min grace period active after entry)"
            )
            results.append({"symbol": symbol, "summary": summary,
                            "optimal_stop": optimal_stop})

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
Every stock shown has already passed multiple quality filters:
- Positive historical avg P&L of at least +0.25%
- Calibrated trailing stop of at least 2.5% (trending cleanly, not choppy)
- No earnings within 5 days
- Not more than 15% below its 30-day high (no falling knives)
These are pre-vetted quality candidates. Pick the best 3 based on today's setup.
VWAP (Volume Weighted Average Price) is a key signal:
- Price above VWAP = bullish, institutional buying pressure
- Price below VWAP = bearish, selling pressure
- Prefer stocks trading above their VWAP with positive momentum
If fewer than 3 look genuinely good today, pick fewer —
it is better to sit on cash than force a bad trade.
The "Backtest-calibrated trailing stop" and avg P&L show historical performance.
Higher avg P&L = stronger historical edge. Higher trailing stop = cleaner trend.
Always end your response with EXACTLY this format, no bold, no company names, no periods:
TOP_PICKS:
1: TICKER
2: TICKER
3: TICKER
If fewer than 3 qualify today, still use the format but only list the ones that do:
TOP_PICKS:
1: TICKER
2: TICKER"""},
            {"role": "user", "content": f"Analyze these stocks and pick the top buy opportunities today:\n{research_data}"}
        ]
    )

    recommendation = response.choices[0].message.content

    top_picks = []
    if "TOP_PICKS:" in recommendation:
        lines = recommendation.split("TOP_PICKS:")[-1].strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            line    = line.replace("**", "").replace("*", "")
            matches = re.findall(r'\b[A-Z]{2,5}\b', line)
            for match in matches:
                if match not in ("TOP", "PICKS", "RSI", "CEO", "ETF", "NYSE", "NA", "VWAP"):
                    if match not in top_picks:
                        top_picks.append(match)
                        break
            if len(top_picks) >= 3:
                break

    if not top_picks:
        print("🛑 GPT-4o found no quality picks today — sitting on cash.")
        return {
            "report":            recommendation,
            "symbols":           [],
            "trailing_stops":    {},
            "conviction_scores": {}
        }

    trailing_stops    = {}
    conviction_scores = {}
    for symbol in top_picks:
        optimal, _, best_avg_pnl = calibration.get(symbol, (TRAILING_STOP_PCT, "", 0.0))
        trailing_stops[symbol]    = optimal
        conviction_scores[symbol] = float(best_avg_pnl) if best_avg_pnl is not None else 0.0

    print("RESEARCH AGENT REPORT:")
    print(recommendation)
    print(f"\nTOP PICKS TODAY: {top_picks}")
    print(f"📐 Calibrated trailing stops : {trailing_stops}")
    print(f"🎯 Conviction scores         : {conviction_scores}")

    return {
        "report":            recommendation,
        "symbols":           top_picks,
        "trailing_stops":    trailing_stops,
        "conviction_scores": conviction_scores
    }

if __name__ == "__main__":
    research_stocks()
