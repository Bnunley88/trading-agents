import os
import json
import schedule
import time
import threading
import alpaca_trade_api as tradeapi
import yfinance as yf
from dotenv import load_dotenv
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from research_agent import research_stocks
from analyst_agent import analyze_recommendation
from executor_agent import execute_trades
from monitor_agent import monitor_position
from exit_agent import exit_trade

load_dotenv()

api = tradeapi.REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    os.getenv("ALPACA_BASE_URL")
)

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")
    def log_message(self, format, *args):
        pass

def run_health_server():
    server = HTTPServer(("0.0.0.0", 8080), HealthHandler)
    server.serve_forever()

threading.Thread(target=run_health_server, daemon=True).start()
print("✅ Health server running on port 8080")

# ---- OPTIONS FLAG ----
# Set to False until live account with options Level 2 approval
# Import is conditional so a missing/broken options_agent won't crash the bot
ENABLE_OPTIONS = False

if ENABLE_OPTIONS:
    from options_agent import buy_call_option

# ---- SHORTS FLAG ----
# Set to False until you've decided how shorts interact with the existing equity logic
# (replace reduced TRENDING DOWN picks? run alongside them? only fire when equity sits
# out entirely?) and verified the live/paper account actually has shorting enabled —
# short_agent.verify_margin_account() will refuse safely either way, but this flag
# means the research call and Alpaca hit don't even happen until you're ready.
ENABLE_SHORTS = False

if ENABLE_SHORTS:
    from research_agent import research_shorts
    from short_agent import execute_shorts

# ---- STATE ----
todays_symbols          = []
high_water_marks        = {}   # peak price since entry (per symbol) — used for trailing stop
low_water_marks         = {}   # trough price since entry (per symbol) — used for MAE tracking
entry_times              = {}   # datetime of entry — used for grace period (reset daily, intraday-only)
entry_dates              = {}   # date of entry — used for time-limit exit (NOT reset daily, spans the hold)
trailing_stops            = {}   # calibrated base trailing stop % per symbol (from research_agent/backtest)
effective_trailing_stops  = {}   # conviction-adjusted trailing stop % actually in force per symbol
last_signal_confirmation  = {}   # last known True/False/None from get_signal_confirmation per symbol —
                                  # used so the stop only moves when this actually CHANGES, not every poll
profit_ceilings           = {}   # per-stock MFE-based profit ceiling % (from research_agent/backtest)
consecutive_loss_days    = 0
daily_start_value        = 0.0
yesterdays_exits         = []
options_daily_cost       = 0.0

# ---- CONSTANTS ----
HARD_PROFIT_CEILING       = 3.0   # fallback used when a symbol has no calibrated profit_ceiling yet
STOP_LOSS_PCT             = 3.0   # widened from 2.0 — MAE research: 8/9 stopped trades would've recovered at 2%
TRAILING_STOP_PCT_DEFAULT = 2.0
GRACE_PERIOD_MINUTES      = 90
MAX_POSITIONS             = 5   # nudged up from 3 — days with 9+ candidates passing filters were
                                 # only using 2-3 slots; lets more of those get taken without
                                 # changing anything about how each individual trade is sized/exited
MIN_BUYING_POWER          = 1000.0

# ---- DAILY LOSS LIMIT ----
DAILY_LOSS_LIMIT_PCT    = 2.0
COOLOFF_LOSS_DAYS       = 3
COOLOFF_RISK_MULTIPLIER = 0.5

# ---- TIME-BASED EXITS (new) ----
TIME_LIMIT_TRADING_DAYS   = 2     # exit if held this many trading days and still under threshold
TIME_LIMIT_PNL_THRESHOLD  = 0.5   # % P&L below which a stale position gets freed up
FRIDAY_CLOSE_PNL_THRESHOLD = 0.5  # % P&L below which a position gets closed before the weekend

# ---- VIX / VOLATILITY REGIME (new) ----
VIX_ELEVATED_THRESHOLD = 25.0   # above this, cut new position sizes
VIX_EXTREME_THRESHOLD  = 35.0   # above this, sit out new buys entirely
VIX_RISK_MULTIPLIER    = 0.5

# ---- CONVICTION-BASED TRAILING STOP (new) ----
# Standard trailing stops ratchet tighter on every new high, which stops trades out on
# normal noise. Instead: only adjust the stop when intraday signals (VWAP/MACD/momentum,
# exposed by research_agent.check_signal_confirmation) actually re-confirm or break the thesis.
# If that function isn't available yet, this degrades safely to the old fixed calibrated stop.
TRAILING_STOP_LOOSEN_FACTOR  = 1.15   # signals still confirm thesis -> give it a bit more room
TRAILING_STOP_TIGHTEN_FACTOR = 0.6    # signals deteriorating -> protect gains/limit loss faster
TRAILING_STOP_MIN_PCT        = 1.0
TRAILING_STOP_MAX_PCT        = 4.0

# ---- OPTIONS GUARDRAILS ----
MAX_OPTIONS_BUDGET_PCT = 0.02
MAX_OPTIONS_CONTRACTS  = 1
OPTIONS_DAILY_LOSS_PCT = 1.0

# ---- MARKET HOURS (Central Time) ----
MARKET_OPEN_HOUR   = 8    # was 9 — real market opens 8:30 AM Central (9:30 AM ET). The old
MARKET_OPEN_MIN    = 35   # value (9:35) was an Eastern-Time buffer sitting in Central-Time
                           # code — the bot was starting its morning session a full hour
                           # after the real open, missing the most active hour of the day.
MARKET_CLOSE_HOUR  = 14   # was 15 — real market closes 3:00 PM Central (4:00 PM ET). The old
MARKET_CLOSE_MIN   = 45   # value (15:45) meant the bot thought the market was still open for
                           # 45 minutes after it had actually closed — confirmed live via DASH's
                           # price freezing at the same value for 35+ straight minutes on Aug 6.

# ---- CALIBRATION CACHE ----
CACHE_FILE    = "calibration_cache.json"
CACHE_MAX_AGE = timedelta(hours=23)

# ---- POSITION STATE PERSISTENCE (new) ----
# Railway restarts wipe all in-memory Python state. entry_dates (time-limit exit) and
# low_water_marks (MAE tracking) were deliberately NOT reset daily because they need to
# survive across days for a multi-day hold — but they were never surviving a container
# restart either, which the logs show happening multiple times. This file makes them
# (plus trailing_stops/profit_ceilings/conviction-stop state, so a mid-day restart
# doesn't fall back to flat defaults until the next morning session) restart-safe.
POSITION_STATE_FILE = "position_state_cache.json"

# ---- MAE/MFE LOG (new — feeds backtest.py's per-stock ceiling/stop calibration) ----
MAE_MFE_LOG_FILE = "mae_mfe_log.json"


def load_calibration_cache():
    """Returns (trailing_stops_dict, profit_ceilings_dict). Old cache files won't have
    a 'profit_ceilings' key — .get() with a default handles that safely."""
    try:
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
        saved_at = datetime.fromisoformat(data["saved_at"])
        if datetime.now() - saved_at > CACHE_MAX_AGE:
            print("📐 Calibration cache is stale — will recompute.")
            return {}, {}
        stops    = data["trailing_stops"]
        ceilings = data.get("profit_ceilings", {})
        print(f"📐 Loaded calibration cache from {saved_at.strftime('%Y-%m-%d %H:%M')} "
              f"({len(stops)} tickers)")
        return stops, ceilings
    except Exception:
        return {}, {}


def save_calibration_cache(stops_dict, ceilings_dict):
    try:
        data = {
            "saved_at": datetime.now().isoformat(),
            "trailing_stops": stops_dict,
            "profit_ceilings": ceilings_dict
        }
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"💾 Calibration cache saved ({len(stops_dict)} tickers).")
    except Exception as e:
        print(f"⚠️ Could not save calibration cache: {e}")


def save_position_state():
    """Persists the position-lifecycle state that needs to survive a Railway restart,
    not just a daily reset. Called after every position check (both the exit path and
    the holding path), so it's always close to current — cheap, small file."""
    try:
        data = {
            "entry_dates":              {s: d.isoformat() for s, d in entry_dates.items()},
            "high_water_marks":         high_water_marks,
            "low_water_marks":          low_water_marks,
            "trailing_stops":           trailing_stops,
            "profit_ceilings":          profit_ceilings,
            "effective_trailing_stops": effective_trailing_stops,
            "last_signal_confirmation": last_signal_confirmation,
        }
        with open(POSITION_STATE_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save position state cache: {e}")


def load_position_state():
    """Called once at startup. Restores position-lifecycle state from before a restart.
    sync_positions_from_alpaca() (called right after this, in the schedule) is the
    source of truth for WHICH positions are actually open — if Alpaca no longer shows a
    symbol this restores, it just sits unused in these dicts until the next exit/entry
    touches it, no risk from stale entries."""
    try:
        with open(POSITION_STATE_FILE, "r") as f:
            data = json.load(f)

        for s, d in data.get("entry_dates", {}).items():
            try:
                entry_dates[s] = datetime.fromisoformat(d).date()
            except Exception:
                pass

        high_water_marks.update(data.get("high_water_marks", {}))
        low_water_marks.update(data.get("low_water_marks", {}))
        trailing_stops.update(data.get("trailing_stops", {}))
        profit_ceilings.update(data.get("profit_ceilings", {}))
        effective_trailing_stops.update(data.get("effective_trailing_stops", {}))
        last_signal_confirmation.update(data.get("last_signal_confirmation", {}))

        print(f"🔁 Restored position state from cache: {len(entry_dates)} entry date(s), "
              f"{len(trailing_stops)} trailing stop(s), {len(profit_ceilings)} profit ceiling(s)")
    except FileNotFoundError:
        print("🔁 No position state cache found — starting fresh.")
    except Exception as e:
        print(f"⚠️ Could not load position state cache ({e}) — starting fresh.")


def log_mae_mfe(symbol, entry_price, low_price, high_price, exit_price, exit_reason):
    """Append MAE/MFE data for a completed trade. backtest.py reads this to calibrate
    per-stock stop losses (from MAE) and profit ceilings (from MFE) instead of flat %."""
    try:
        try:
            with open(MAE_MFE_LOG_FILE, "r") as f:
                log = json.load(f)
        except Exception:
            log = []

        mae_pct = ((entry_price - low_price) / entry_price) * 100 if entry_price else 0.0
        mfe_pct = ((high_price - entry_price) / entry_price) * 100 if entry_price else 0.0
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price else 0.0

        log.append({
            "symbol": symbol,
            "entry_price": entry_price,
            "low_price": low_price,
            "high_price": high_price,
            "exit_price": exit_price,
            "mae_pct": round(mae_pct, 3),
            "mfe_pct": round(mfe_pct, 3),
            "pnl_pct": round(pnl_pct, 3),
            "exit_reason": exit_reason,
            "exit_time": datetime.now().isoformat(),
        })

        with open(MAE_MFE_LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)

        print(f"📝 MAE/MFE logged for {symbol}: MAE={mae_pct:.2f}% MFE={mfe_pct:.2f}% "
              f"PnL={pnl_pct:.2f}% (reason: {exit_reason})")
    except Exception as e:
        print(f"⚠️ Could not log MAE/MFE for {symbol}: {e}")


def get_vix():
    """Pull current VIX level. Returns None if unavailable (never blocks trading on failure)."""
    try:
        vix_data = yf.Ticker("^VIX").history(period="1d")
        if vix_data.empty:
            return None
        return float(vix_data["Close"].iloc[-1])
    except Exception as e:
        print(f"⚠️ Could not fetch VIX: {e}")
        return None


def get_signal_confirmation(symbol):
    """
    Returns True if intraday signals still confirm the position's thesis (price above VWAP,
    MACD still bullish, momentum positive), False if they've deteriorated, or None if
    research_agent doesn't expose this yet. Safe no-op until research_agent.py adds
    check_signal_confirmation() — trailing stop falls back to the old fixed calibrated % here.
    """
    try:
        from research_agent import check_signal_confirmation
        return check_signal_confirmation(symbol)
    except ImportError:
        return None
    except Exception as e:
        print(f"⚠️ Signal confirmation check failed for {symbol}: {e}")
        return None


def count_trading_days_held(symbol):
    """Weekday count between entry_date and today. Doesn't account for market holidays —
    close enough for a 2-day threshold."""
    if symbol not in entry_dates:
        return 0
    entry_date = entry_dates[symbol]
    today = datetime.now().date()
    if today <= entry_date:
        return 0
    days = 0
    d = entry_date
    while d < today:
        d += timedelta(days=1)
        if d.weekday() < 5:
            days += 1
    return days


def get_trailing_stop(symbol):
    return trailing_stops.get(symbol, TRAILING_STOP_PCT_DEFAULT)


def get_profit_ceiling(symbol):
    return profit_ceilings.get(symbol, HARD_PROFIT_CEILING)


def is_in_grace_period(symbol):
    if symbol not in entry_times:
        return False
    elapsed = (datetime.now() - entry_times[symbol]).total_seconds() / 60
    return elapsed < GRACE_PERIOD_MINUTES


def is_market_hours():
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=MARKET_OPEN_HOUR,  minute=MARKET_OPEN_MIN,  second=0)
    market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MIN, second=0)
    return market_open <= now <= market_close


def is_daily_loss_limit_hit():
    if daily_start_value <= 0:
        return False
    try:
        account       = api.get_account()
        current_value = float(account.portfolio_value)
        daily_pct     = ((current_value - daily_start_value) / daily_start_value) * 100
        if daily_pct <= -DAILY_LOSS_LIMIT_PCT:
            print(f"🛑 DAILY LOSS LIMIT HIT: portfolio down {daily_pct:.2f}% today "
                  f"(limit: -{DAILY_LOSS_LIMIT_PCT}%) — pausing new trades.")
            return True
        return False
    except Exception as e:
        print(f"⚠️ Could not check daily loss limit: {e}")
        return False


def is_options_budget_ok(portfolio_value):
    max_spend = portfolio_value * MAX_OPTIONS_BUDGET_PCT
    if options_daily_cost >= max_spend:
        print(f"⚠️ OPTIONS: Daily budget cap reached "
              f"(${options_daily_cost:.2f} / ${max_spend:.2f}) — skipping")
        return False
    return True


def get_risk_multiplier():
    if consecutive_loss_days >= COOLOFF_LOSS_DAYS:
        print(f"⚠️ COOL-OFF ACTIVE: {consecutive_loss_days} consecutive losing days — "
              f"position sizes reduced to {COOLOFF_RISK_MULTIPLIER * 100:.0f}%")
        return COOLOFF_RISK_MULTIPLIER
    return 1.0


def position_exists_in_alpaca(symbol):
    """Verify position exists before attempting exit — prevents ghost position errors."""
    try:
        api.get_position(symbol)
        return True
    except Exception:
        return False


def sync_positions_from_alpaca():
    global todays_symbols
    try:
        open_positions = api.list_positions()
        open_symbols   = [p.symbol for p in open_positions]

        for symbol in open_symbols:
            if symbol not in todays_symbols:
                print(f"🔄 Sync: adding {symbol} to monitoring (found in Alpaca positions)")
                todays_symbols.append(symbol)

        for symbol in list(todays_symbols):
            if symbol not in open_symbols:
                print(f"🔄 Sync: removing {symbol} from monitoring (no longer in Alpaca)")
                todays_symbols.remove(symbol)

    except Exception as e:
        print(f"⚠️ Position sync failed ({e}), using cached todays_symbols")


# ---- SESSIONS ----

def run_morning_session():
    global todays_symbols, high_water_marks, low_water_marks, trailing_stops
    global entry_times, entry_dates, effective_trailing_stops, profit_ceilings
    global daily_start_value, consecutive_loss_days, yesterdays_exits, options_daily_cost

    print("\n🌅 MORNING SESSION - 8:35 AM")
    print("=" * 50)

    options_daily_cost = 0.0

    if yesterdays_exits:
        print(f"🔄 Yesterday's exits (re-entry candidates): {yesterdays_exits}")

    try:
        account           = api.get_account()
        daily_start_value = float(account.portfolio_value)
        buying_power      = float(account.buying_power)
        print(f"📊 Day opening portfolio value: ${daily_start_value:,.2f}")
    except Exception as e:
        print(f"⚠️ Could not get account info: {e}")
        daily_start_value = 0.0
        buying_power      = 0.0

    try:
        existing_positions = api.list_positions()
        existing_symbols   = [p.symbol for p in existing_positions]
        print(f"💼 Open positions: {existing_symbols}")
        print(f"💵 Buying power: ${buying_power:,.2f}")

        if buying_power < MIN_BUYING_POWER:
            print(f"⚠️ Buying power below ${MIN_BUYING_POWER:,.2f} — skipping new buys.")
            todays_symbols = existing_symbols
            for symbol in todays_symbols:
                check_position(symbol)
            print("\n✅ MORNING SESSION COMPLETE (no new buys — low buying power)")
            return

        if len(existing_positions) >= MAX_POSITIONS:
            print(f"⚠️ Already at max {MAX_POSITIONS} positions — skipping new buys.")
            todays_symbols = existing_symbols
            for symbol in todays_symbols:
                check_position(symbol)
            print("\n✅ MORNING SESSION COMPLETE (no new buys — max positions reached)")
            return

    except Exception as e:
        print(f"⚠️ Could not check positions ({e}), proceeding normally.")
        existing_symbols = []

    # ---- COOL-OFF + VIX RISK SIZING ----
    if consecutive_loss_days > 0:
        print(f"📉 Consecutive losing days: {consecutive_loss_days} "
              f"({'COOL-OFF ACTIVE' if consecutive_loss_days >= COOLOFF_LOSS_DAYS else 'watching'})")

    vix = get_vix()
    vix_risk_mult = 1.0
    sit_out_vix   = False
    if vix is not None:
        if vix > VIX_EXTREME_THRESHOLD:
            sit_out_vix = True
            print(f"🌪️ VIX EXTREME: {vix:.1f} — sitting out new buys entirely today.")
        elif vix > VIX_ELEVATED_THRESHOLD:
            vix_risk_mult = VIX_RISK_MULTIPLIER
            print(f"🌪️ VIX ELEVATED: {vix:.1f} (>{VIX_ELEVATED_THRESHOLD}) — "
                  f"reducing new position sizes to {VIX_RISK_MULTIPLIER * 100:.0f}%")
        else:
            print(f"🌪️ VIX: {vix:.1f} (normal range)")
    else:
        print("🌪️ VIX: unavailable — proceeding without volatility adjustment")

    if sit_out_vix:
        todays_symbols = existing_symbols
        for symbol in todays_symbols:
            check_position(symbol)
        print("\n✅ MORNING SESSION COMPLETE (no new buys — VIX extreme)")
        return

    risk_mult = get_risk_multiplier() * vix_risk_mult

    cached_stops, cached_ceilings = load_calibration_cache()
    if cached_stops:
        trailing_stops.update(cached_stops)
        print(f"📐 Using cached trailing stops: {trailing_stops}")
    if cached_ceilings:
        profit_ceilings.update(cached_ceilings)
        print(f"📐 Using cached profit ceilings: {profit_ceilings}")

    print("\n📊 STEP 1: RESEARCHING STOCKS + CALIBRATING TRAILING STOPS...")
    research_result = research_stocks(previously_held=yesterdays_exits)
    research_report = research_result["report"]
    top_symbols     = research_result["symbols"]
    new_stops       = research_result.get("trailing_stops", {})

    trailing_stops.update(new_stops)
    print(f"📐 Active trailing stops: {trailing_stops}")

    new_ceilings = research_result.get("profit_ceilings", {})
    profit_ceilings.update(new_ceilings)
    print(f"📐 Active profit ceilings: {profit_ceilings}")

    if not top_symbols:
        print("🛑 No quality picks today — sitting on cash, monitoring existing positions only.")
        todays_symbols = existing_symbols
        for symbol in todays_symbols:
            check_position(symbol)
        print("\n✅ MORNING SESSION COMPLETE (no new buys — no quality picks today)")
        return

    if is_daily_loss_limit_hit():
        todays_symbols = existing_symbols
        for symbol in todays_symbols:
            check_position(symbol)
        print("\n✅ MORNING SESSION COMPLETE (no new buys — daily loss limit hit)")
        return

    slots_available = MAX_POSITIONS - len(existing_symbols)
    new_symbols     = [s for s in top_symbols if s not in existing_symbols][:slots_available]

    if not new_symbols:
        print("⚠️ All top picks already held — skipping new buys.")
        todays_symbols = existing_symbols
        for symbol in todays_symbols:
            check_position(symbol)
        print("\n✅ MORNING SESSION COMPLETE (no new buys — all picks already held)")
        return

    todays_symbols   = existing_symbols + new_symbols
    # high_water_marks intentionally NOT reset here anymore — it used to wipe every
    # morning, which meant a position held more than one day lost its true peak and
    # the trailing stop understated real risk (see SOFI Aug 4: true peak was $18.70
    # from the day before, but a nightly/morning reset would've shown $18.20 instead).
    # It's now only cleared per-symbol on actual exit (see _exit() below), same as the
    # other position-lifecycle state.

    print("\n🧠 STEP 2: ANALYZING OPPORTUNITIES...")
    analyze_recommendation(research_report)

    print("\n⚡ STEP 3: EXECUTING TRADES...")
    print(f"Already holding: {existing_symbols}")
    print(f"New trades today: {new_symbols}")
    conviction_scores = research_result.get("conviction_scores", {})
    execute_trades(new_symbols, conviction_scores=conviction_scores,
                   risk_multiplier=risk_mult)

    now = datetime.now()
    for symbol in new_symbols:
        entry_times[symbol] = now
        entry_dates[symbol] = now.date()

    # Options layer — only runs when ENABLE_OPTIONS = True
    if ENABLE_OPTIONS and new_symbols:
        try:
            account         = api.get_account()
            portfolio_value = float(account.portfolio_value)
            top_pick        = new_symbols[0]

            if is_options_budget_ok(portfolio_value):
                print(f"\n📈 OPTIONS: Buying call on top pick {top_pick} "
                      f"(max {MAX_OPTIONS_CONTRACTS} contract, "
                      f"budget cap: {MAX_OPTIONS_BUDGET_PCT*100:.0f}% of portfolio)...")
                result = buy_call_option(top_pick, portfolio_value,
                                         max_contracts=MAX_OPTIONS_CONTRACTS,
                                         conviction_score=conviction_scores.get(top_pick))
                if result:
                    options_daily_cost += portfolio_value * MAX_OPTIONS_BUDGET_PCT
        except Exception as e:
            print(f"⚠️ OPTIONS: Skipping — {e}")

    time.sleep(3)
    sync_positions_from_alpaca()

    # ---- SHORTS (dormant, ENABLE_SHORTS=False) ----
    # Entry side only — see the ENABLE_SHORTS comment above and short_agent.py's module
    # docstring for what's still missing before this can safely go live.
    if ENABLE_SHORTS:
        regime = research_result.get("market_regime", "")
        if "TRENDING DOWN" in regime:
            print("\n📉 STEP 3c: REGIME IS TRENDING DOWN — EVALUATING SHORTS...")
            short_result    = research_shorts()
            short_symbols   = short_result.get("symbols", [])
            weakness_scores = short_result.get("weakness_scores", {})
            if short_symbols:
                print(f"🔻 Short candidates: {short_symbols}")
                execute_shorts(short_symbols, weakness_scores=weakness_scores, risk_multiplier=risk_mult)
                print("⚠️ NOTE: short positions are intentionally NOT added to todays_symbols and are "
                      "NOT monitored by check_position()'s exit logic — that logic assumes long "
                      "positions (P&L sign, trailing stop direction) and hasn't been adapted for "
                      "shorts yet. Until that exists, open shorts rely only on short_agent.py's own "
                      "safeguards (margin gate) — nothing will trail-stop or time-limit-exit them.")
            else:
                print("🔻 No qualifying short candidates today.")

    print("\n🔍 STEP 3b: VERIFYING FILLS...")
    for symbol in new_symbols:
        if position_exists_in_alpaca(symbol):
            print(f"✅ Confirmed: {symbol} position exists in Alpaca")
        else:
            print(f"❌ WARNING: {symbol} order submitted but no position found in Alpaca "
                  f"— possible fill failure, check Alpaca dashboard")

    print("\n👁️ STEP 4: MONITORING POSITIONS...")
    for symbol in todays_symbols:
        check_position(symbol)

    print("\n✅ MORNING SESSION COMPLETE!")


def run_intraday_check():
    if not is_market_hours():
        return

    if todays_symbols and is_daily_loss_limit_hit():
        print("🛑 Daily loss limit active — monitoring only, no new buys.")

    sync_positions_from_alpaca()

    if not todays_symbols:
        return

    now_str = datetime.now().strftime("%H:%M")
    print(f"\n🔄 INTRADAY CHECK - {now_str}")
    print("=" * 50)
    for symbol in list(todays_symbols):
        check_position(symbol)


def run_closing_check():
    global todays_symbols, high_water_marks, entry_times
    global consecutive_loss_days, daily_start_value, yesterdays_exits

    sync_positions_from_alpaca()

    print("\n🌆 CLOSING CHECK - 2:45 PM")
    print("=" * 50)

    if todays_symbols:
        # at_close=True activates the Friday-close rule below
        for symbol in list(todays_symbols):
            check_position(symbol, at_close=True)
    else:
        print("⚠️ No positions to monitor at close.")

    _update_loss_streak()

    if trailing_stops:
        save_calibration_cache(trailing_stops, profit_ceilings)

    print(f"🔄 Re-entry candidates saved for tomorrow: {yesterdays_exits}")

    # NOTE: entry_dates / low_water_marks / high_water_marks / effective_trailing_stops
    # are intentionally NOT reset here — they need to persist across days for the
    # time-limit exit, MAE tracking, and true peak-based trailing stop to work on
    # multi-day holds. Only entry_times (grace period, genuinely intraday-only) resets.
    todays_symbols   = []
    entry_times      = {}


def _update_loss_streak():
    global consecutive_loss_days, daily_start_value
    if daily_start_value <= 0:
        return
    try:
        account       = api.get_account()
        closing_value = float(account.portfolio_value)
        daily_pnl_pct = ((closing_value - daily_start_value) / daily_start_value) * 100

        if daily_pnl_pct < 0:
            consecutive_loss_days += 1
            print(f"📉 Day closed down {daily_pnl_pct:.2f}% — "
                  f"consecutive losing days: {consecutive_loss_days}")
            if consecutive_loss_days >= COOLOFF_LOSS_DAYS:
                print(f"⚠️ {COOLOFF_LOSS_DAYS}+ consecutive losing days — "
                      f"cool-off active tomorrow (position sizes at 50%)")
        else:
            if consecutive_loss_days > 0:
                print(f"✅ Day closed up {daily_pnl_pct:.2f}% — "
                      f"resetting consecutive loss streak (was {consecutive_loss_days})")
            consecutive_loss_days = 0

    except Exception as e:
        print(f"⚠️ Could not update loss streak: {e}")


def check_position(symbol, at_close=False):
    global high_water_marks, low_water_marks, todays_symbols, trailing_stops
    global effective_trailing_stops, entry_times, entry_dates, yesterdays_exits, profit_ceilings
    global last_signal_confirmation

    position = monitor_position(symbol)
    if not position:
        print(f"ℹ️ No open position found for {symbol}.")
        return

    current_price = position.get("current_price", 0)
    entry_price   = position.get("entry_price", 0)
    pnl           = position.get("profit_loss_pct", 0)

    if symbol not in high_water_marks or current_price > high_water_marks[symbol]:
        high_water_marks[symbol] = current_price
    if symbol not in low_water_marks or current_price < low_water_marks[symbol]:
        low_water_marks[symbol] = current_price

    peak_price          = high_water_marks[symbol]
    drop_from_peak_pct  = ((current_price - peak_price) / peak_price) * 100
    grace_active        = is_in_grace_period(symbol)

    # ---- CONVICTION-BASED TRAILING STOP ----
    # Only tighten/loosen when intraday signals actually CHANGE state from the last check —
    # not on every 5-min poll, even if the signal comes back the same. Recomputing fresh
    # from base_stop_pct every poll was causing the stop to whipsaw back and forth on
    # noisy signal reads even when nothing had really changed. Falls back to the fixed
    # calibrated stop if research_agent.check_signal_confirmation isn't available.
    base_stop_pct      = get_trailing_stop(symbol)
    confirmation       = get_signal_confirmation(symbol)
    prior_confirmation = last_signal_confirmation.get(symbol)

    if confirmation != prior_confirmation:
        if confirmation is True:
            trailing_stop_pct = min(base_stop_pct * TRAILING_STOP_LOOSEN_FACTOR, TRAILING_STOP_MAX_PCT)
            conviction_label = " [signals CONFIRM — stop loosened]"
        elif confirmation is False:
            trailing_stop_pct = max(base_stop_pct * TRAILING_STOP_TIGHTEN_FACTOR, TRAILING_STOP_MIN_PCT)
            conviction_label = " [signals DETERIORATING — stop tightened]"
        else:
            trailing_stop_pct = base_stop_pct
            conviction_label = ""
        effective_trailing_stops[symbol] = trailing_stop_pct
        last_signal_confirmation[symbol] = confirmation
    else:
        # No change since last check — hold steady rather than re-deriving from
        # base_stop_pct again (that repeated re-derivation was the whipsaw).
        trailing_stop_pct = effective_trailing_stops.get(symbol, base_stop_pct)
        conviction_label  = " [signals unchanged — stop held]" if symbol in last_signal_confirmation else ""

    profit_ceiling = get_profit_ceiling(symbol)

    grace_label = f" [GRACE {GRACE_PERIOD_MINUTES}min active]" if grace_active else ""
    print(f"📈 {symbol} | Current: ${current_price:.2f} | Entry: ${entry_price:.2f} | Peak: ${peak_price:.2f}")
    print(f"   P&L: {pnl:.2f}% | Drop from peak: {drop_from_peak_pct:.2f}% | "
          f"Trailing stop: {trailing_stop_pct:.2f}% | Profit ceiling: {profit_ceiling:.2f}%"
          f"{conviction_label}{grace_label}")

    def _exit(reason_label):
        if not position_exists_in_alpaca(symbol):
            print(f"ℹ️ {symbol} already closed in Alpaca — skipping exit call")
        else:
            exit_trade(symbol)

        log_mae_mfe(
            symbol=symbol,
            entry_price=entry_price,
            low_price=low_water_marks.get(symbol, entry_price),
            high_price=peak_price,
            exit_price=current_price,
            exit_reason=reason_label,
        )

        if symbol in todays_symbols:
            todays_symbols.remove(symbol)
        trailing_stops.pop(symbol, None)
        effective_trailing_stops.pop(symbol, None)
        profit_ceilings.pop(symbol, None)
        last_signal_confirmation.pop(symbol, None)
        entry_times.pop(symbol, None)
        entry_dates.pop(symbol, None)
        high_water_marks.pop(symbol, None)
        low_water_marks.pop(symbol, None)

        if symbol not in yesterdays_exits:
            yesterdays_exits.append(symbol)

        save_position_state()

    if pnl >= profit_ceiling:
        print(f"💰 PROFIT CEILING HIT! {symbol} +{pnl:.2f}% (ceiling: {profit_ceiling:.2f}%)")
        _exit("HARD_CEILING")
        return

    if pnl <= -STOP_LOSS_PCT:
        print(f"🚨 STOP LOSS TRIGGERED! {symbol} {pnl:.2f}%")
        _exit("STOP_LOSS")
        return

    trading_days_held = count_trading_days_held(symbol)
    if trading_days_held >= TIME_LIMIT_TRADING_DAYS and pnl < TIME_LIMIT_PNL_THRESHOLD:
        print(f"⏱️ TIME LIMIT HIT! {symbol} held {trading_days_held} trading days at "
              f"{pnl:.2f}% P&L (threshold: {TIME_LIMIT_PNL_THRESHOLD}%) — freeing up capital")
        _exit("TIME_LIMIT")
        return

    if at_close and datetime.now().weekday() == 4 and pnl < FRIDAY_CLOSE_PNL_THRESHOLD:
        print(f"📅 FRIDAY CLOSE RULE! {symbol} at {pnl:.2f}% P&L "
              f"(<{FRIDAY_CLOSE_PNL_THRESHOLD}%) — exiting before weekend gap risk")
        _exit("FRIDAY_CLOSE")
        return

    if (not grace_active
            and peak_price > entry_price
            and drop_from_peak_pct <= -trailing_stop_pct):
        print(f"🛡️ TRAILING STOP TRIGGERED! {symbol} locked in {pnl:.2f}% "
              f"(down {abs(drop_from_peak_pct):.2f}% from peak ${peak_price:.2f}, "
              f"stop was {trailing_stop_pct:.2f}%)")
        _exit("TRAILING_STOP")
        return

    print(f"⏳ Holding {symbol}. P&L within range.")
    save_position_state()


# ---- SCHEDULE ----
load_position_state()

for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
    getattr(schedule.every(), day).at("08:35").do(run_morning_session)

for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
    getattr(schedule.every(), day).at("14:45").do(run_closing_check)

schedule.every(5).minutes.do(run_intraday_check)

print("⏰ SCHEDULER RUNNING")
print("🌅 Morning session: 8:35 AM Central (calibration loaded from cache if fresh, VIX-adjusted sizing)")
print("🔄 Intraday checks: every 5 min during market hours (silent when no positions)")
print("🌆 Closing check: 2:45 PM Central (saves calibration cache, applies Friday-close rule)")
print(f"🛡️  Stop Loss: -{STOP_LOSS_PCT}% (widened from 2% per MAE research, always active)")
print(f"⏸️  Grace period: {GRACE_PERIOD_MINUTES} min after entry (trailing stop suppressed)")
print(f"📉 Trailing Stop: per-stock calibrated, conviction-adjusted "
      f"(default fallback: {TRAILING_STOP_PCT_DEFAULT}%)")
print(f"💰 Profit Ceiling: per-stock calibrated from MFE (fallback: +{HARD_PROFIT_CEILING}%)")
print(f"⏱️  Time limit: {TIME_LIMIT_TRADING_DAYS}+ trading days under {TIME_LIMIT_PNL_THRESHOLD}% P&L -> exit")
print(f"📅 Friday close rule: under {FRIDAY_CLOSE_PNL_THRESHOLD}% P&L at Friday close -> exit")
print(f"🌪️  VIX check: >{VIX_ELEVATED_THRESHOLD} reduces size to {VIX_RISK_MULTIPLIER*100:.0f}%, "
      f">{VIX_EXTREME_THRESHOLD} sits out entirely")
print(f"🛑 Daily loss limit: -{DAILY_LOSS_LIMIT_PCT}% (pauses new trades if hit)")
print(f"❄️  Cool-off: after {COOLOFF_LOSS_DAYS} consecutive losing days (position sizes at 50%)")
print(f"📈 Options trading: {'ENABLED' if ENABLE_OPTIONS else 'DISABLED (flip ENABLE_OPTIONS to True when ready)'}")

while True:
    schedule.run_pending()
    time.sleep(30)
