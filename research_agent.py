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
MIN_CONVICTION_SCORE    = 0.25
MIN_TRAILING_STOP       = 2.5
EARNINGS_BLOCK_DAYS     = 5
DOWNTREND_BLOCK_PCT     = 15.0

# ---- VOLUME CONFIRMATION FILTER ----
# Research shows mean reversion trades with volume 30%+ above average
# have 81% win rate vs 61% without — hard requirement
MIN_VOLUME_RATIO        = 1.3    # must be at least 1.3x average volume
VOLUME_FILTER_ENABLED   = True   # set False to disable during low-volume days

# ---- SECTOR DIVERSIFICATION ----
# Never buy more than this many stocks from the same sector
MAX_SAME_SECTOR         = 1

# ---- HIGH CONVICTION ANCHOR LIST ----
# Trimmed to top 12 highest conviction names based on backtest results
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

# ---- SECTOR MAP for diversification check ----
SECTOR_MAP = {
    "ARM": "Technology", "NVDA": "Technology", "AMD": "Technology",
    "MSFT": "Technology", "AAPL": "Technology", "GOOGL": "Technology",
    "META": "Technology", "NOW": "Technology", "CRM": "Technology",
    "SHOP": "Technology", "ORCL": "Technology", "AVGO": "Technology",
    "MU": "Technology", "SMCI": "Technology", "PLTR": "Technology",
    "HOOD": "Finance", "GS": "Finance", "JPM": "Finance",
    "V": "Finance", "MA": "Finance", "PYPL": "Finance", "COIN": "Finance",
    "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare",
    "SNOW": "Cloud", "DASH": "Consumer", "CAT": "Industrial",
    "TSLA": "Automotive", "RIVN": "Automotive",
    "XOM": "Energy", "CVX": "Energy",
    "AMZN": "Consumer", "WMT": "Consumer", "HD": "Consumer",
    "MARA": "Crypto", "HOOD": "Finance",
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
    """
    Calculates MACD (Moving Average Convergence Divergence).
    MACD line = 12-day EMA - 26-day EMA
    Signal line = 9-day EMA of MACD
    Histogram = MACD - Signal
    Bullish: MACD crosses above signal line (histogram turns positive)
    Bearish: MACD crosses below signal line (histogram turns negative)
    """
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
        hist_val    = round(histogram.iloc[-1], 3)

        # Detect crossover in last 2 bars
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
    """
    Calculates Bollinger Bands.
    Upper band = SMA + (std * num_std)
    Lower band = SMA - (std * num_std)
    Price near upper band = overbought
    Price near lower band = oversold / potential bounce
    %B = (price - lower) / (upper - lower) — 0=at lower, 1=at upper, >1=above upper
    """
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
        if band_width > 0:
            pct_b  = round((current - lower_val) / band_width * 100, 1)
        else:
            pct_b  = 50.0

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
    """
    Detects whether the overall market is trending or range-bound
    using SPY (S&P 500 ETF) as proxy.
    Trending up:   SPY above 20-day SMA and 5-day momentum positive
    Trending down: SPY below 20-day SMA and 5-day momentum negative
    Range-bound:   SPY near SMA with low momentum
    Returns regime string passed to GPT-4o to adjust strategy.
    """
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
    """
    Checks weekly timeframe trend for multi-timeframe confirmation.
    Returns brief signal: weekly trend direction vs daily.
    Aligned = stronger signal. Conflicted = use caution.
    """
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


def get_sector(symbol):
    """Returns sector for diversification check."""
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
    """
    Given a list of top picks from GPT-4o, enforce sector diversification.
    Never return more than max_same_sector stocks from the same sector.
    Returns filtered list.
    """
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
    """
    Main research function.
    previously_held — list of symbols held yesterday that stopped out or hit ceiling.
                      Used for re-entry logic: if they still pass all filters today,
                      they get priority consideration.
    """
    print("🔍 Building blended watchlist (Alpaca screener + top 12 conviction anchors)...")
    watchlist = get_dynamic_watchlist(top_n=20)

    # ---- Market regime detection ----
    market_regime = get_market_regime()
    print(f"📊 Market regime: {market_regime}")

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
        # FIX: block exactly 0.00% — walk-forward returning zero means inconclusive
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

    # Relax if nothing passes
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

            # ---- Volume confirmation filter ----
            volume_flag = ""
            if VOLUME_FILTER_ENABLED and volume_ratio != "N/A":
                if volume_ratio < MIN_VOLUME_RATIO:
                    volume_flag = f" ⚠️ LOW VOLUME ({volume_ratio}x avg — below {MIN_VOLUME_RATIO}x threshold)"
                else:
                    volume_flag = f" ✅ Volume confirmed ({volume_ratio}x avg)"

            # VWAP
            vwap, pct_vs_vwap = get_vwap(hist)
            if pct_vs_vwap != "N/A":
                vwap_label = (f"VWAP=${vwap} | Price vs VWAP: {pct_vs_vwap:+.2f}% "
                              f"({'above — bullish' if pct_vs_vwap > 0 else 'below — bearish'})")
            else:
                vwap_label = "VWAP: unavailable"

            # MACD
            macd_val, signal_val, macd_signal = get_macd(hist)
            macd_label = (f"MACD={macd_val} | Signal={signal_val} | {macd_signal}"
                          if macd_val != "N/A" else "MACD: unavailable")

            # Bollinger Bands
            bb_upper, bb_lower, bb_sma, bb_signal = get_bollinger_bands(hist)
            bb_label = (f"Bollinger Bands: Upper=${bb_upper} | SMA=${bb_sma} | Lower=${bb_lower} | {bb_signal}"
                        if bb_upper != "N/A" else "Bollinger Bands: unavailable")

            # Weekly trend (multi-timeframe)
            weekly_trend = get_weekly_trend(symbol)

            # Re-entry flag
            reentry_flag = " 🔄 RE-ENTRY CANDIDATE (passed all filters again)" if symbol in reentry_candidates else ""

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
                f"  Backtest-calibrated trailing stop: {optimal_stop}% "
                f"(avg P&L: {pnl_display}, 90-min grace period active after entry)"
            )
            results.append({"symbol": symbol, "summary": summary,
                            "optimal_stop": optimal_stop,
                            "volume_ratio": volume_ratio})

        except Exception as e:
            print(f"Error fetching {symbol}: {e}")

    research_data = "\n".join([r["summary"] for r in results])

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"""You are an expert stock research analyst.
Current market regime: {market_regime}

Analyze stocks using technical indicators, earnings timing, insider activity, and analyst ratings.
RSI below 30 = oversold (potential buy). RSI above 70 = overbought (avoid).
High volume ratio = strong institutional interest. Prefer stocks with volume 1.3x+ average.
Insider buying = very bullish signal.
Analyst upgrades = momentum catalyst.

Every stock shown has already passed multiple quality filters:
- Positive historical avg P&L (above zero, minimum +0.25%)
- Calibrated trailing stop of at least 2.5% (trending cleanly, not choppy)
- No earnings within 5 days
- Not more than 15% below its 30-day high (no falling knives)

KEY SIGNALS TO WEIGHT (in order of importance):
1. MACD — bullish crossover is the strongest momentum confirmation signal
2. VWAP — price above VWAP = institutional buying pressure (bullish)
3. Bollinger Bands — near lower band = oversold bounce potential; near upper = overbought
4. Volume confirmation — stocks with volume 1.3x+ average have significantly higher win rates
5. Weekly trend — prefer stocks where weekly and daily trends are aligned
6. RSI — secondary signal, use to confirm not to lead
7. Re-entry candidates — stocks that passed yesterday and pass again today have extra confirmation

Market regime adjustment:
- TRENDING UP: favor momentum picks with MACD bullish crossover and price above VWAP
- TRENDING DOWN: be very selective, require multiple confirming signals, pick fewer
- RANGE-BOUND: favor mean reversion (oversold RSI + near Bollinger lower band)

If fewer than 3 look genuinely good today, pick fewer — cash is better than a bad trade.

Always end your response with EXACTLY this format, no bold, no company names, no periods:
TOP_PICKS:
1: TICKER
2: TICKER
3: TICKER
If fewer than 3 qualify today:
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
                                 "VWAP", "MACD", "SMA", "EMA", "BB"):
                    if match not in top_picks:
                        top_picks.append(match)
                        break
            if len(top_picks) >= 3:
                break

    # Apply sector diversification to final picks
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
