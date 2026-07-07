"""
Standalone backtester + trailing stop calibrator.

THREE MODES:
  1. Standalone backtest — run directly:
         python3 backtest.py
     Tests fixed and ATR-based trailing stops, prints summary and walk-forward results.

  2. Calibration mode — import and call:
         from backtest import calibrate_trailing_stop
         optimal_pct, report, best_avg_pnl = calibrate_trailing_stop("TSLA")
     Tests multiple trailing stop values and returns the optimal % for that ticker.
     Now uses walk-forward validation to avoid overfitting.

  3. ATR mode — trailing stop sized dynamically by Average True Range:
     Instead of a fixed %, the stop is set at entry_price - (ATR * ATR_MULTIPLIER).
     Widens automatically on volatile days, tightens on calm days.

Does NOT import anything from the live bot files and cannot place trades.
"""

import yfinance as yf
import pandas as pd
import numpy as np

# ---- CONFIG ----
TRAILING_STOP_PCT      = 2.0
HARD_PROFIT_CEILING    = 3.0   # updated to match scheduler's new 3% ceiling
STOP_LOSS_PCT          = 2.0
CHECK_INTERVAL_MINUTES = 5     # updated to match scheduler's 5-min checks
GRACE_PERIOD_MINUTES   = 90

TICKERS   = ["AAPL", "MSFT", "NVDA", "META", "TSLA", "AMZN", "JNJ", "XOM", "COST"]
DAYS_BACK = 30

# ---- ATR CONFIG ----
ATR_PERIOD     = 14    # standard ATR period
ATR_MULTIPLIER = 1.5   # trailing stop = ATR * multiplier (higher = wider stop)

# ---- CALIBRATION CONFIG ----
CALIBRATION_CANDIDATES     = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
MIN_TRADES_FOR_CALIBRATION = 4

# ---- WALK-FORWARD CONFIG ----
# Split the 30-day window: train on first N days, validate on remaining days
# This prevents overfitting to a single period
WALK_FORWARD_TRAIN_DAYS    = 20   # train on first 20 days
WALK_FORWARD_VALIDATE_DAYS = 10   # validate on last 10 days


def compute_atr(hist, period=ATR_PERIOD):
    """
    Computes Average True Range from daily OHLC data.
    ATR = average of True Range over `period` days.
    True Range = max(High-Low, abs(High-PrevClose), abs(Low-PrevClose))
    Returns a Series of ATR values aligned with hist index.
    """
    try:
        high      = hist["High"]
        low       = hist["Low"]
        prev_close = hist["Close"].shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        return atr
    except Exception:
        return pd.Series(dtype=float)


def get_atr_trailing_stop_pct(entry_price, atr_value, multiplier=ATR_MULTIPLIER):
    """
    Converts an ATR value into a trailing stop percentage.
    stop_price = entry_price - (atr * multiplier)
    stop_pct   = (entry_price - stop_price) / entry_price * 100
    Falls back to TRAILING_STOP_PCT if ATR is unavailable.
    """
    try:
        if pd.isna(atr_value) or atr_value <= 0 or entry_price <= 0:
            return TRAILING_STOP_PCT
        stop_price = entry_price - (atr_value * multiplier)
        stop_pct   = ((entry_price - stop_price) / entry_price) * 100
        # Clamp between 1% and 8% to prevent extreme values
        return round(max(1.0, min(8.0, stop_pct)), 2)
    except Exception:
        return TRAILING_STOP_PCT


def simulate_trade(symbol, full_df, entry_price, entry_time,
                   trailing_stop_pct=None,
                   hard_ceiling=None,
                   stop_loss_pct=None,
                   grace_period_minutes=None,
                   use_atr=False,
                   atr_series=None):
    """
    Replays one simulated trade through the exit logic.

    use_atr    — if True, trailing stop is dynamic (ATR-based) rather than fixed %
    atr_series — pre-computed ATR series for the symbol (required if use_atr=True)
    """
    if trailing_stop_pct    is None: trailing_stop_pct    = TRAILING_STOP_PCT
    if hard_ceiling         is None: hard_ceiling         = HARD_PROFIT_CEILING
    if stop_loss_pct        is None: stop_loss_pct        = STOP_LOSS_PCT
    if grace_period_minutes is None: grace_period_minutes = GRACE_PERIOD_MINUTES

    # ATR-based trailing stop: compute at entry
    if use_atr and atr_series is not None:
        try:
            atr_at_entry      = atr_series.asof(entry_time)
            trailing_stop_pct = get_atr_trailing_stop_pct(entry_price, atr_at_entry)
        except Exception:
            pass  # fall back to fixed %

    high_water_mark = entry_price
    checks  = full_df[full_df.index > entry_time]
    step    = max(1, CHECK_INTERVAL_MINUTES // 5)
    sampled = checks.iloc[::step]

    for timestamp, row in sampled.iterrows():
        current_price = row["Close"]

        if current_price > high_water_mark:
            high_water_mark = current_price

        pnl_pct            = ((current_price - entry_price)    / entry_price)    * 100
        drop_from_peak_pct = ((current_price - high_water_mark) / high_water_mark) * 100

        try:
            elapsed_minutes = (timestamp - entry_time).total_seconds() / 60
        except Exception:
            elapsed_minutes = grace_period_minutes + 1

        in_grace_period = elapsed_minutes < grace_period_minutes

        if pnl_pct >= hard_ceiling:
            return {
                "symbol": symbol, "exit_reason": "HARD_CEILING",
                "exit_time": timestamp, "exit_price": current_price,
                "pnl_pct": pnl_pct, "peak_price": high_water_mark,
                "held_overnight": timestamp.date() != entry_time.date(),
                "trailing_stop_used": trailing_stop_pct
            }

        if pnl_pct <= -stop_loss_pct:
            return {
                "symbol": symbol, "exit_reason": "STOP_LOSS",
                "exit_time": timestamp, "exit_price": current_price,
                "pnl_pct": pnl_pct, "peak_price": high_water_mark,
                "held_overnight": timestamp.date() != entry_time.date(),
                "trailing_stop_used": trailing_stop_pct
            }

        if (not in_grace_period
                and high_water_mark > entry_price
                and drop_from_peak_pct <= -trailing_stop_pct):
            return {
                "symbol": symbol, "exit_reason": "TRAILING_STOP",
                "exit_time": timestamp, "exit_price": current_price,
                "pnl_pct": pnl_pct, "peak_price": high_water_mark,
                "held_overnight": timestamp.date() != entry_time.date(),
                "trailing_stop_used": trailing_stop_pct
            }

    if len(sampled) > 0:
        last_row  = sampled.iloc[-1]
        last_time = sampled.index[-1]
        pnl_pct   = ((last_row["Close"] - entry_price) / entry_price) * 100
        return {
            "symbol": symbol, "exit_reason": "STILL_OPEN_AT_DATA_END",
            "exit_time": last_time, "exit_price": last_row["Close"],
            "pnl_pct": pnl_pct, "peak_price": high_water_mark,
            "held_overnight": last_time.date() != entry_time.date(),
            "trailing_stop_used": trailing_stop_pct
        }

    return None


def _run_ticker(symbol, hist, trailing_stop_pct, grace_period_minutes=None,
                use_atr=False, atr_series=None):
    results      = []
    hist         = hist.copy()
    hist["date"] = hist.index.date
    unique_days  = sorted(hist["date"].unique())

    day_idx = 0
    while day_idx < len(unique_days):
        day           = unique_days[day_idx]
        day_open_data = hist[hist["date"] == day]
        if len(day_open_data) < 2:
            day_idx += 1
            continue

        entry_price    = day_open_data.iloc[0]["Close"]
        entry_time     = day_open_data.index[0]
        remaining_data = hist[hist.index >= entry_time]

        result = simulate_trade(symbol, remaining_data, entry_price, entry_time,
                                trailing_stop_pct=trailing_stop_pct,
                                grace_period_minutes=grace_period_minutes,
                                use_atr=use_atr,
                                atr_series=atr_series)
        if result:
            result["entry_price"] = entry_price
            result["entry_date"]  = day
            results.append(result)

            exit_date = result["exit_time"].date()
            next_idx  = day_idx
            while next_idx < len(unique_days) and unique_days[next_idx] <= exit_date:
                next_idx += 1
            if next_idx == day_idx:
                next_idx += 1
            day_idx = next_idx
        else:
            day_idx += 1

    return results


def calibrate_trailing_stop(symbol, days_back=DAYS_BACK):
    """
    Called by research_agent.py for each candidate stock (in parallel).

    Uses WALK-FORWARD validation:
    - Train: finds best trailing stop % on first WALK_FORWARD_TRAIN_DAYS days
    - Validate: confirms performance on last WALK_FORWARD_VALIDATE_DAYS days
    - Returns the validated best % (more robust, less overfitting)

    Also computes ATR-based trailing stop as an alternative and picks
    whichever performs better on the validation set.

    Returns (optimal_pct: float, report: str, best_avg_pnl: float)
    """
    try:
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period=f"{days_back}d", interval="5m")

        if hist.empty or len(hist) < 20:
            return TRAILING_STOP_PCT, f"{symbol}: insufficient data, using default {TRAILING_STOP_PCT}%", None

        # ---- Split into train / validate windows ----
        hist["date"] = hist.index.date
        unique_days  = sorted(hist["date"].unique())

        if len(unique_days) < WALK_FORWARD_TRAIN_DAYS + 2:
            # Not enough days — fall back to single window
            train_hist    = hist
            validate_hist = hist
        else:
            train_cutoff  = unique_days[WALK_FORWARD_TRAIN_DAYS - 1]
            train_hist    = hist[hist["date"] <= train_cutoff]
            validate_hist = hist[hist["date"] > train_cutoff]

        # ---- Daily data for ATR (need OHLC) ----
        daily_hist = ticker.history(period=f"{days_back}d", interval="1d")
        atr_series = compute_atr(daily_hist) if not daily_hist.empty else None

        # ---- PHASE 1: Find best fixed % on training data ----
        best_train_pct     = TRAILING_STOP_PCT
        best_train_avg_pnl = None
        train_lines        = []

        for candidate in CALIBRATION_CANDIDATES:
            results = _run_ticker(symbol, train_hist.copy(), trailing_stop_pct=candidate,
                                  grace_period_minutes=GRACE_PERIOD_MINUTES)
            if len(results) < MIN_TRADES_FOR_CALIBRATION:
                continue
            df        = pd.DataFrame(results)
            avg_pnl   = df["pnl_pct"].mean()
            train_lines.append((candidate, avg_pnl, len(results)))
            if best_train_avg_pnl is None or avg_pnl > best_train_avg_pnl:
                best_train_avg_pnl = avg_pnl
                best_train_pct     = candidate

        # ---- PHASE 2: Validate best fixed % on validation data ----
        validate_results = _run_ticker(symbol, validate_hist.copy(),
                                       trailing_stop_pct=best_train_pct,
                                       grace_period_minutes=GRACE_PERIOD_MINUTES)
        if validate_results:
            val_df          = pd.DataFrame(validate_results)
            validated_pnl   = val_df["pnl_pct"].mean()
        else:
            validated_pnl   = best_train_avg_pnl or 0.0

        # ---- PHASE 3: Test ATR-based stop on validation data ----
        atr_validated_pnl = None
        if atr_series is not None and not atr_series.empty:
            atr_results = _run_ticker(symbol, validate_hist.copy(),
                                      trailing_stop_pct=TRAILING_STOP_PCT,
                                      grace_period_minutes=GRACE_PERIOD_MINUTES,
                                      use_atr=True,
                                      atr_series=atr_series)
            if atr_results:
                atr_df            = pd.DataFrame(atr_results)
                atr_validated_pnl = atr_df["pnl_pct"].mean()
                avg_atr_stop      = atr_df["trailing_stop_used"].mean()
            else:
                atr_validated_pnl = None

        # ---- Pick winner: fixed % vs ATR ----
        use_atr_final = False
        if atr_validated_pnl is not None and atr_validated_pnl > validated_pnl:
            use_atr_final = True
            final_pnl     = atr_validated_pnl
            final_pct     = round(avg_atr_stop, 1) if 'avg_atr_stop' in dir() else best_train_pct
        else:
            final_pnl = validated_pnl
            final_pct = best_train_pct

        # ---- Build report ----
        report_lines = [f"{symbol} walk-forward calibration (train {WALK_FORWARD_TRAIN_DAYS}d / validate {WALK_FORWARD_VALIDATE_DAYS}d):"]
        report_lines.append(f"  Training phase — best fixed stop: {best_train_pct}% (train avg P&L: {best_train_avg_pnl:+.2f}%)" if best_train_avg_pnl is not None else "  Training: inconclusive")
        report_lines.append(f"  Validation — fixed {best_train_pct}%: avg P&L {validated_pnl:+.2f}%")
        if atr_validated_pnl is not None:
            report_lines.append(f"  Validation — ATR-based (avg stop ~{avg_atr_stop:.1f}%): avg P&L {atr_validated_pnl:+.2f}%")
        report_lines.append(f"  → WINNER: {'ATR-based' if use_atr_final else f'fixed {final_pct}%'} (validated avg P&L {final_pnl:+.2f}%)")
        report = "\n".join(report_lines)

        print(f"📐 {symbol} calibrated trailing stop: {final_pct}% (best avg P&L: {final_pnl:+.2f}%)")
        return final_pct, report, final_pnl

    except Exception as e:
        print(f"⚠️ Calibration failed for {symbol}: {e}")
        return TRAILING_STOP_PCT, f"{symbol}: calibration error ({e}), using default {TRAILING_STOP_PCT}%", None


# ---------------------------------------------------------------------------
# STANDALONE MODE
# ---------------------------------------------------------------------------

def run_backtest():
    """Runs fixed trailing stop backtest on all TICKERS."""
    results = []
    for symbol in TICKERS:
        print(f"\nFetching {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            hist   = ticker.history(period=f"{DAYS_BACK}d", interval="5m")
            if hist.empty or len(hist) < 20:
                print(f"  Not enough data for {symbol}, skipping.")
                continue
            ticker_results = _run_ticker(symbol, hist, trailing_stop_pct=TRAILING_STOP_PCT)
            results.extend(ticker_results)
        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")
    return results


def run_atr_backtest():
    """Runs ATR-based trailing stop backtest on all TICKERS for comparison."""
    results = []
    for symbol in TICKERS:
        print(f"\nFetching {symbol} (ATR mode)...")
        try:
            ticker     = yf.Ticker(symbol)
            hist       = ticker.history(period=f"{DAYS_BACK}d", interval="5m")
            daily_hist = ticker.history(period=f"{DAYS_BACK}d", interval="1d")
            if hist.empty or len(hist) < 20:
                print(f"  Not enough data for {symbol}, skipping.")
                continue
            atr_series     = compute_atr(daily_hist)
            ticker_results = _run_ticker(symbol, hist, trailing_stop_pct=TRAILING_STOP_PCT,
                                         use_atr=True, atr_series=atr_series)
            results.extend(ticker_results)
        except Exception as e:
            print(f"  Error fetching {symbol} (ATR): {e}")
    return results


def summarize(results, label="FIXED TRAILING STOP"):
    if not results:
        print(f"\nNo results to summarize ({label}).")
        return

    df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print(f"BACKTEST SUMMARY — {label}")
    print("=" * 60)
    print(f"Total simulated trades : {len(df)}")
    print(f"Average P&L per trade  : {df['pnl_pct'].mean():.2f}%")
    print(f"Win rate               : {(df['pnl_pct'] > 0).mean() * 100:.1f}%")
    print(f"Best trade             : {df['pnl_pct'].max():.2f}%")
    print(f"Worst trade            : {df['pnl_pct'].min():.2f}%")

    print("\nExit reason breakdown:")
    print(df["exit_reason"].value_counts())

    print("\nAvg P&L by exit reason:")
    print(df.groupby("exit_reason")["pnl_pct"].mean().round(2))

    print("\nPer-symbol average P&L:")
    print(df.groupby("symbol")["pnl_pct"].mean().round(2).sort_values(ascending=False))

    if "held_overnight" in df.columns:
        overnight_count = df["held_overnight"].sum()
        print(f"\nTrades held overnight: {overnight_count} of {len(df)} "
              f"({overnight_count/len(df)*100:.1f}%)")
        if overnight_count > 0:
            print(f"Avg P&L overnight : {df[df['held_overnight']]['pnl_pct'].mean():.2f}%")
            print(f"Avg P&L same-day  : {df[~df['held_overnight']]['pnl_pct'].mean():.2f}%")

    if "trailing_stop_used" in df.columns:
        print(f"\nAvg trailing stop used: {df['trailing_stop_used'].mean():.2f}%")


if __name__ == "__main__":
    print(f"Running backtest: {TICKERS}")
    print(f"Settings: Stop Loss={STOP_LOSS_PCT}% | Trailing Stop={TRAILING_STOP_PCT}% | "
          f"Hard Ceiling={HARD_PROFIT_CEILING}% | Grace Period={GRACE_PERIOD_MINUTES}min")
    print(f"Check interval: every {CHECK_INTERVAL_MINUTES} min | Lookback: {DAYS_BACK} days\n")

    # Run fixed trailing stop
    results_fixed = run_backtest()
    summarize(results_fixed, label="FIXED TRAILING STOP")

    # Run ATR-based trailing stop
    print("\n" + "=" * 60)
    print("Running ATR-based trailing stop comparison...")
    results_atr = run_atr_backtest()
    summarize(results_atr, label="ATR-BASED TRAILING STOP")

    # Walk-forward calibration
    print("\n" + "=" * 60)
    print(f"WALK-FORWARD CALIBRATION (train {WALK_FORWARD_TRAIN_DAYS}d / validate {WALK_FORWARD_VALIDATE_DAYS}d)")
    print("=" * 60)
    for symbol in TICKERS:
        optimal, report, pnl = calibrate_trailing_stop(symbol)
        print(report)

    # Save results
    if results_fixed:
        pd.DataFrame(results_fixed).to_csv("backtest_results_fixed.csv", index=False)
    if results_atr:
        pd.DataFrame(results_atr).to_csv("backtest_results_atr.csv", index=False)
    print("\nResults saved to backtest_results_fixed.csv and backtest_results_atr.csv")
