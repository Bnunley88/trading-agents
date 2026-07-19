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
DOWNTREND_BLOCK_PCT  = 15.0

# ---- VOLUME CONFIRMATION ----
MIN_VOLUME_RATIO     = 1.3
VOLUME_FILTER_ENABLED = True

# ---- SECTOR DIVERSIFICATION ----
MAX_SAME_SECTOR = 1

# ---- UNUSUAL OPTIONS THRESHOLDS ----
UNUSUAL_OPTIONS_VOLUME_RATIO = 2.0   # call volume > 2x open interest = unusual
UNUSUAL_OPTIONS_MIN_VOLUME   = 500   # ignore very low volume contracts

# ---- SHORT INTEREST THRESHOLD ----
HIGH_SHORT_INTEREST_PCT = 10.0  # >10% short float = squeeze potential

# ---- SECTOR ETF MAP for relative strength ----
SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Finance":    "XLF",
    "Healthcare": "XLV",
    "Consumer":   "XLY",
    "Industrial": "XLI",
    "Energy":     "XLE",
    "Cloud":      "IGV",
    "Automotive": "XLY",
    "Crypto":     "BITQ",
    "Unknown":    "SPY",
}

# ---- HIGH CONVICTION ANCHOR LIST ----
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

SECTOR_MAP = {
    "ARM": "Technology", "NVDA": "Technology", "AMD": "Technology",
    "MSFT": "Technology", "AAPL": "Technology", "GOOGL": "Technology",
    "META": "Technology", "NOW": "Technology", "CRM": "Technology",
    "SHOP": "Technology", "ORCL": "Technology", "AVGO": "Technology",
    "MU": "Technology", "SMCI": "Technology", "PLTR": "Technology",
    "HOOD": "Finance", "GS": "Finance", "JPM": "Finance",
    "V": "Finance", "MA": "Finance", "PYPL": "Finance", "COIN": "Finance",
    "BAC": "Finance", "T": "Finance",
    "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare",
    "SNOW": "Cloud", "DASH": "Consumer", "CAT": "Industrial",
    "TSLA": "Automotive", "RIVN": "Automotive",
    "XOM": "Energy", "CVX": "Energy",
    "AMZN": "Consumer", "WMT": "Consumer", "HD": "Consumer",
    "MARA": "Crypto",
}

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
    try:
        hist = yf.Ticker(symbol).history(period="30d")
        if hist.empty or len(hist) < 5:
            return False, 0.0
        high_30d  = hist["Close"].max()
        current   = hist["Close"].iloc[-1]
        pct_below = ((high_30d - current) / high_30d) * 100
        if pct_below >= DOWNTREND_BLOCK_PCT:
            return True, round(pct_below, 1)
        return False, round(pct_below, 1)
    except:
        return False, 0.0


def get_vwap(hist):
    try:
        if hist.empty or len(hist) < 5:
            return "N/A", "N/A"
        recent        = hist.tail(5)
        typical_price = (recent["High"] + recent["Low"] + recent["Close"]) / 3
        vwap          = (typical_price * recent["Volume"]).sum() / recent["Volume"].sum()
        current       = hist["Close"].iloc[-1]
        pct_vs_vwap   = round(((current - vwap) / vwap) * 100, 2)
        return round(vwap, 2), pct_vs_vwap
    except:
        return "N/A", "N/A"


def get_macd(hist):
    try:
        if hist.empty or len(hist) < 30:
            return "N/A", "N/A", "N/A"
        close       = hist["Close"]
        ema12       = close.ewm(span=12, adjust=False).mean()
        ema26       = close.ewm(span=26, adjust=False).mean()
        macd_line   = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram   = macd_line - signal_line

        macd_val    = round(macd_line.iloc[-1], 3)
        signal_val  = round(signal_line.iloc[-1], 3)

        prev_hist   = histogram.iloc[-2]
        curr_hist   = histogram.iloc[-1]
        if prev_hist < 0 and curr_hist > 0:
            crossover = "🟢 BULLISH CROSSOVER — MACD crossed above signal"
        elif prev_hist > 0 and curr_hist < 0:
            crossover = "🔴 BEARISH CROSSOVER — MACD crossed below signal"
        elif curr_hist > 0:
            crossover = "Bullish momentum (MACD above signal)"
        else:
            crossover = "Bearish momentum (MACD below signal)"

        return macd_val, signal_val, crossover
    except:
        return "N/A", "N/A", "N/A"


def get_bollinger_bands(hist, window=20, num_std=2):
    try:
        if hist.empty or len(hist) < window:
            return "N/A", "N/A", "N/A", "N/A"
        close      = hist["Close"]
        sma        = close.rolling(window=window).mean()
        std        = close.rolling(window=window).std()
        upper      = sma + (std * num_std)
        lower      = sma - (std * num_std)
        current    = close.iloc[-1]
        upper_val  = round(upper.iloc[-1], 2)
        lower_val  = round(lower.iloc[-1], 2)
        sma_val    = round(sma.iloc[-1], 2)

        band_width = upper_val - lower_val
        pct_b      = round((current - lower_val) / band_width * 100, 1) if band_width > 0 else 50.0

        if pct_b > 80:
            bb_signal = f"Near upper band ({pct_b}%B) — overbought, caution"
        elif pct_b < 20:
            bb_signal = f"Near lower band ({pct_b}%B) — oversold, potential bounce"
        else:
            bb_signal = f"Middle of bands ({pct_b}%B) — neutral"

        return upper_val, lower_val, sma_val, bb_signal
    except:
        return "N/A", "N/A", "N/A", "N/A"


def get_market_regime():
    try:
        spy  = yf.Ticker("SPY")
        hist = spy.history(period="30d")
        if hist.empty or len(hist) < 20:
            return "Unknown"

        close      = hist["Close"]
        sma20      = close.rolling(window=20).mean().iloc[-1]
        current    = close.iloc[-1]
        momentum5d = ((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]) * 100
        pct_vs_sma = ((current - sma20) / sma20) * 100

        if pct_vs_sma > 1.0 and momentum5d > 0.5:
            return f"TRENDING UP (SPY {pct_vs_sma:+.1f}% above 20d SMA, +{momentum5d:.1f}% 5d momentum) — favor momentum picks"
        elif pct_vs_sma < -1.0 and momentum5d < -0.5:
            return f"TRENDING DOWN (SPY {pct_vs_sma:+.1f}% below 20d SMA, {momentum5d:.1f}% 5d momentum) — be selective, reduce exposure"
        else:
            return f"RANGE-BOUND (SPY {pct_vs_sma:+.1f}% vs 20d SMA, {momentum5d:.1f}% 5d momentum) — favor mean reversion picks"
    except:
        return "Unknown"


def get_weekly_trend(symbol):
    try:
        hist_weekly = yf.Ticker(symbol).history(period="90d", interval="1wk")
        if hist_weekly.empty or len(hist_weekly) < 4:
            return "Weekly data unavailable"
        close       = hist_weekly["Close"]
        momentum_4w = round(((close.iloc[-1] - close.iloc[-4]) / close.iloc[-4]) * 100, 1)
        sma4w       = close.tail(4).mean()
        current     = close.iloc[-1]

        if current > sma4w and momentum_4w > 0:
            return f"Weekly uptrend ({momentum_4w:+.1f}% 4wk) — aligned with daily ✅"
        elif current < sma4w and momentum_4w < 0:
            return f"Weekly downtrend ({momentum_4w:+.1f}% 4wk) — caution ⚠️"
        else:
            return f"Weekly mixed ({momentum_4w:+.1f}% 4wk) — no clear direction"
    except:
        return "Weekly data unavailable"


def get_unusual_options(symbol):
    """
    Detects unusual call options activity using yfinance options chain.
    Compares today's call volume to open interest.
    Volume > 2x open interest on near-term expiry = institutional buying signal.
    Uses only yfinance — no new API needed.
    """
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            return "No options data"

        # Check nearest 2 expirations
        unusual_flags = []
        for expiry in expirations[:2]:
            chain = ticker.option_chain(expiry)
            calls = chain.calls

            if calls.empty:
                continue

            # Filter for meaningful volume
            active_calls = calls[calls["volume"] > UNUSUAL_OPTIONS_MIN_VOLUME]
            if active_calls.empty:
                continue

            # Check for unusual volume vs open interest
            for _, row in active_calls.iterrows():
                vol = row.get("volume", 0) or 0
                oi  = row.get("openInterest", 0) or 0
                if oi > 0 and vol / oi >= UNUSUAL_OPTIONS_VOLUME_RATIO:
                    strike = row.get("strike", "?")
                    unusual_flags.append(
                        f"${strike} calls (exp {expiry}): vol={int(vol)}, OI={int(oi)}, "
                        f"ratio={vol/oi:.1f}x"
                    )

        if unusual_flags:
            return f"🔥 UNUSUAL CALL ACTIVITY: {' | '.join(unusual_flags[:2])}"
        return "No unusual options activity"

    except Exception as e:
        return f"Options data unavailable ({e})"


def get_short_interest(ticker_obj):
    """
    Pulls short interest ratio from yfinance.
    High short interest (>10%) + positive momentum = squeeze potential.
    """
    try:
        info         = ticker_obj.info
        short_float  = info.get("shortPercentOfFloat", None)
        short_ratio  = info.get("shortRatio", None)

        if short_float is None:
            return "Short interest data unavailable"

        short_pct = round(short_float * 100, 1)
        ratio_str = f", days-to-cover: {short_ratio:.1f}" if short_ratio else ""

        if short_pct >= HIGH_SHORT_INTEREST_PCT:
            return f"🔥 HIGH SHORT INTEREST: {short_pct}% float shorted{ratio_str} — squeeze potential"
        elif short_pct >= 5.0:
            return f"Moderate short interest: {short_pct}% float shorted{ratio_str}"
        else:
            return f"Low short interest: {short_pct}% float shorted{ratio_str}"
    except:
        return "Short interest data unavailable"


# Cache for sector ETF data — pulled once per session, reused for all stocks
_etf_cache = {}

def get_etf_5d_return(sector_etf):
    """Pull sector ETF 5-day return, cached so we only fetch each ETF once."""
    if sector_etf in _etf_cache:
        return _etf_cache[sector_etf]
    try:
        etf_hist = yf.Ticker(sector_etf).history(period="10d")
        if etf_hist.empty or len(etf_hist) < 6:
            _etf_cache[sector_etf] = None
            return None
        val = ((etf_hist["Close"].iloc[-1] - etf_hist["Close"].iloc[-5]) / etf_hist["Close"].iloc[-5]) * 100
        _etf_cache[sector_etf] = round(val, 2)
        return _etf_cache[sector_etf]
    except:
        _etf_cache[sector_etf] = None
        return None


def get_relative_strength(symbol, hist):
    """
    Compares stock's 5-day return vs its sector ETF.
    ETF data is cached — only fetched once per ETF per session.
    """
    try:
        if hist.empty or len(hist) < 6:
            return "Relative strength data unavailable"

        sector     = SECTOR_MAP.get(symbol, "Unknown")
        sector_etf = SECTOR_ETF_MAP.get(sector, "SPY")
        stock_5d   = ((hist["Close"].iloc[-1] - hist["Close"].iloc[-5]) / hist["Close"].iloc[-5]) * 100
        etf_5d     = get_etf_5d_return(sector_etf)

        if etf_5d is None:
            return f"Sector ETF data unavailable (sector: {sector})"

        rs = round(stock_5d - etf_5d, 2)

        if rs > 1.0:
            return (f"✅ Outperforming {sector_etf} by {rs:+.2f}% over 5d "
                    f"(stock: {stock_5d:+.2f}%, sector: {etf_5d:+.2f}%) — relative strength")
        elif rs < -1.0:
            return (f"⚠️ Underperforming {sector_etf} by {rs:+.2f}% over 5d "
                    f"(stock: {stock_5d:+.2f}%, sector: {etf_5d:+.2f}%) — relative weakness")
        else:
            return f"In line with {sector_etf} ({rs:+.2f}% vs sector over 5d) — neutral"
    except Exception as e:
        return f"Relative strength unavailable ({e})"


def get_price_volume_divergence(hist):
    """
    Detects price/volume divergence — a key institutional signal.
    Price rising + volume falling = weak move, likely to reverse (bearish divergence)
    Price rising + volume rising = institutional accumulation (strong move)
    Price falling + volume rising = institutional distribution (selling pressure)
    """
    try:
        if hist.empty or len(hist) < 6:
            return "Price/volume data unavailable"

        price_5d  = hist["Close"].tail(5)
        volume_5d = hist["Volume"].tail(5)

        price_trend  = price_5d.iloc[-1] - price_5d.iloc[0]
        volume_trend = volume_5d.mean() - volume_5d.iloc[0]

        price_up   = price_trend > 0
        volume_up  = volume_trend > 0

        if price_up and volume_up:
            return "✅ Price up + Volume up — institutional accumulation, strong move"
        elif price_up and not volume_up:
            return "⚠️ Price up + Volume down — weak move, possible reversal ahead"
        elif not price_up and volume_up:
            return "🔴 Price down + Volume up — institutional distribution, selling pressure"
        else:
            return "Price down + Volume down — low conviction move, indecisive"
    except:
        return "Price/volume divergence data unavailable"


def get_fear_greed():
    """
    Fetches CNN Fear & Greed Index.
    Free public endpoint, no API key needed.
    0-25 = Extreme Fear (buy signal), 75-100 = Extreme Greed (caution)
    """
    try:
        url      = "https://fear-and-greed-index.p.rapidapi.com/v1/fgi"
        headers  = {
            "x-rapidapi-host": "fear-and-greed-index.p.rapidapi.com",
            "x-rapidapi-key": "SIGN-UP-FOR-KEY"
        }
        # Try alternative free endpoint first
        alt_url  = "https://api.alternative.me/fng/"
        response = requests.get(alt_url, timeout=5)
        data     = response.json()

        if "data" in data and len(data["data"]) > 0:
            value       = int(data["data"][0]["value"])
            rating      = data["data"][0]["value_classification"]

            if value <= 25:
                signal = f"🟢 EXTREME FEAR ({value}/100 — {rating}) — historically strong buy signal"
            elif value <= 45:
                signal = f"🟡 Fear ({value}/100 — {rating}) — cautious buy opportunity"
            elif value <= 55:
                signal = f"⚪ Neutral ({value}/100 — {rating})"
            elif value <= 75:
                signal = f"🟡 Greed ({value}/100 — {rating}) — be selective"
            else:
                signal = f"🔴 EXTREME GREED ({value}/100 — {rating}) — caution, market may be overextended"

            return signal

        return "Fear & Greed data unavailable"
    except Exception as e:
        return f"Fear & Greed unavailable ({e})"


def get_sector(symbol):
    return SECTOR_MAP.get(symbol, "Unknown")


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


def apply_sector_diversification(picks, max_same_sector=MAX_SAME_SECTOR):
    sector_counts = {}
    diversified   = []
    removed       = []

    for symbol in picks:
        sector = get_sector(symbol)
        count  = sector_counts.get(sector, 0)
        if sector == "Unknown" or count < max_same_sector:
            diversified.append(symbol)
            sector_counts[sector] = count + 1
        else:
            removed.append(f"{symbol} ({sector} already at max {max_same_sector})")

    if removed:
        print(f"🔀 Sector diversification removed: {', '.join(removed)}")

    return diversified


def research_stocks(previously_held=None):
    print("🔍 Building blended watchlist (Alpaca screener + top 12 conviction anchors)...")
    watchlist = get_dynamic_watchlist(top_n=20)

    # Market regime + Fear & Greed (run once, used in prompt)
    market_regime = get_market_regime()
    fear_greed    = get_fear_greed()
    print(f"📊 Market regime: {market_regime}")
    print(f"😨 Fear & Greed: {fear_greed}")

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

    # ---- FILTER 2: Downtrend filter ----
    downtrend_blocked     = []
    post_downtrend_filter = []
    for symbol in post_earnings_filter:
        blocked, pct_below = check_downtrend(symbol)
        if blocked:
            downtrend_blocked.append(f"{symbol} ({pct_below}% below 30d high)")
        else:
            post_downtrend_filter.append(symbol)

    if downtrend_blocked:
        print(f"🚫 Downtrend blocked: {', '.join(downtrend_blocked)}")

    # ---- FILTERS 3-6: P&L, conviction, choppy, zero score ----
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
        if pnl == 0.0:
            rejected.append(f"{symbol} (zero conviction score — inconclusive)")
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

    # ---- Re-entry candidates ----
    reentry_candidates = []
    if previously_held:
        reentry_candidates = [s for s in previously_held if s in filtered_watchlist]
        if reentry_candidates:
            print(f"🔄 Re-entry candidates (passed filters again): {reentry_candidates}")

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

            volume_flag = ""
            if VOLUME_FILTER_ENABLED and volume_ratio != "N/A":
                if volume_ratio < MIN_VOLUME_RATIO:
                    volume_flag = f" ⚠️ LOW VOLUME ({volume_ratio}x avg)"
                else:
                    volume_flag = f" ✅ Volume confirmed ({volume_ratio}x avg)"

            # All signals
            vwap, pct_vs_vwap   = get_vwap(hist)
            vwap_label          = (f"VWAP=${vwap} | Price vs VWAP: {pct_vs_vwap:+.2f}% "
                                   f"({'above — bullish' if pct_vs_vwap != 'N/A' and pct_vs_vwap > 0 else 'below — bearish'})"
                                   if pct_vs_vwap != "N/A" else "VWAP: unavailable")

            macd_val, signal_val, macd_signal = get_macd(hist)
            macd_label = (f"MACD={macd_val} | Signal={signal_val} | {macd_signal}"
                          if macd_val != "N/A" else "MACD: unavailable")

            bb_upper, bb_lower, bb_sma, bb_signal = get_bollinger_bands(hist)
            bb_label = (f"Bollinger Bands: Upper=${bb_upper} | SMA=${bb_sma} | Lower=${bb_lower} | {bb_signal}"
                        if bb_upper != "N/A" else "Bollinger Bands: unavailable")

            weekly_trend    = get_weekly_trend(symbol)
            unusual_options = get_unusual_options(symbol)
            short_interest  = get_short_interest(ticker)
            rel_strength    = get_relative_strength(symbol, hist)
            pv_divergence   = get_price_volume_divergence(hist)

            reentry_flag = " 🔄 RE-ENTRY CANDIDATE" if symbol in reentry_candidates else ""

            news     = get_news_sentiment(symbol, name)
            earnings = get_earnings_info(ticker)
            insider  = get_insider_activity(ticker)
            analyst  = get_analyst_rating(ticker)
            finnhub  = get_finnhub_data(symbol)

            optimal_stop, _, _ = calibration.get(symbol, (TRAILING_STOP_PCT, "", None))
            _, _, best_avg_pnl = calibration.get(symbol, (TRAILING_STOP_PCT, "", None))
            pnl_display        = f"{float(best_avg_pnl):+.2f}%" if best_avg_pnl is not None else "N/A"

            summary = (
                f"{name} ({symbol}){reentry_flag}: Price=${price}, PE={pe_ratio}, "
                f"52W High={week_high}, 52W Low={week_low}, "
                f"RSI={rsi}, 5-Day Momentum={momentum}%, "
                f"Volume Ratio={volume_ratio}x avg{volume_flag}, Market Cap={market_cap}\n"
                f"  Earnings: {earnings}\n"
                f"  Insider Activity: {insider}\n"
                f"  Analyst Rating: {analyst}\n"
                f"  News: {news}\n"
                f"  Cross-check: {finnhub}\n"
                f"  {vwap_label}\n"
                f"  {macd_label}\n"
                f"  {bb_label}\n"
                f"  Weekly trend: {weekly_trend}\n"
                f"  Unusual Options: {unusual_options}\n"
                f"  Short Interest: {short_interest}\n"
                f"  Relative Strength: {rel_strength}\n"
                f"  Price/Volume: {pv_divergence}\n"
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
            {"role": "system", "content": f"""You are an expert stock research analyst with institutional-grade signal access.
Current market regime: {market_regime}
Current Fear & Greed Index: {fear_greed}

Analyze stocks using all available signals. Signal priority order:
1. Unusual Options Activity — institutional money moving before price. 🔥 flag = strong buy signal
2. MACD — 🟢 BULLISH CROSSOVER is the strongest momentum confirmation
3. Relative Strength vs Sector — stock outperforming its sector = real institutional interest
4. VWAP — price above VWAP = institutional buying pressure
5. Short Interest — 🔥 HIGH SHORT INTEREST + positive momentum = squeeze potential
6. Price/Volume divergence — price up + volume up = strong; price up + volume down = weak
7. Bollinger Bands — near lower band = oversold bounce; near upper = overbought
8. Volume confirmation — prefer 1.3x+ average volume
9. RSI — secondary confirmation only (below 30 = oversold, above 70 = overbought, avoid)
10. Weekly trend — aligned daily + weekly = stronger signal
11. Re-entry candidates — passed filters yesterday AND today = extra confirmation

Fear & Greed context:
- Extreme Fear (0-25): buy quality dips aggressively
- Fear (25-45): good buying opportunity
- Neutral (45-55): normal selection
- Greed (55-75): be selective, require stronger signals
- Extreme Greed (75-100): very selective, only highest conviction picks

Market regime — regime changes HOW you trade, not WHETHER you trade:
- TRENDING UP: favor momentum picks, MACD bullish crossover, price above VWAP. Pick all 3 slots.
- TRENDING DOWN: reduce to 1-2 picks max, require RSI below 50 and price above VWAP.
- RANGE-BOUND: favor mean reversion — RSI below 40, near Bollinger lower band, above VWAP.
  Still pick 2-3 stocks. Stocks with strong backtest avg P&L (+1.0%+) and good Sharpe (0.4+)
  have proven edge even in sideways conditions. DO NOT sit on cash just because it is range-bound.

IMPORTANT: You MUST pick at least 1 stock unless EVERY candidate has:
- Negative backtest avg P&L, AND
- RSI above 70, AND
- Price below VWAP, AND
- No unusual options activity
Only then is sitting on cash acceptable.

Always end your response with EXACTLY this format, no bold, no company names, no periods:
TOP_PICKS:
1: TICKER
2: TICKER
3: TICKER
If fewer than 3 qualify:
TOP_PICKS:
1: TICKER"""},
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
                if match not in ("TOP", "PICKS", "RSI", "CEO", "ETF", "NYSE", "NA",
                                 "VWAP", "MACD", "SMA", "EMA", "BB", "OI"):
                    if match not in top_picks:
                        top_picks.append(match)
                        break
            if len(top_picks) >= 3:
                break

    if top_picks:
        top_picks = apply_sector_diversification(top_picks)

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
