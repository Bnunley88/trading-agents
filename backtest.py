"""
Standalone backtester + trailing stop calibrator.

TWO MODES:
  1. Standalone backtest — run directly:
         python3 backtest.py
     Tests the configured TRAILING_STOP_PCT against all TICKERS and prints a summary.
     Also runs calibration for each ticker and prints a comparison table.

  2. Calibration mode — import and call:
         from backtest import calibrate_trailing_stop
         optimal_pct, report = calibrate_trailing_stop("TSLA")
     Tests multiple trailing stop values and returns the optimal % for that ticker.

Does NOT import anything from the live bot files and cannot place trades.
"""

import yfinance as yf
import pandas as pd

# ---- CONFIG ----
TRAILING_STOP_PCT   = 2.0
HARD_PROFIT_CEILING = 5.0
STOP_LOSS_PCT       = 2.0
CHECK_INTERVAL_MINUTES = 30

# Grace period: trailing stop is suppressed for this many minutes after entry.
# The hard stop loss still fires immediately — this only delays the trailing stop.
# Kills intraday shake-outs on volatile names.
GRACE_PERIOD_MINUTES = 90

TICKERS   = ["AAPL", "MSFT", "NVDA", "META", "TSLA", "AMZN", "JNJ", "XOM", "COST"]
DAYS_BACK = 30

CALIBRATION_CANDIDATES     = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
MIN_TRADES_FOR_CALIBRATION = 4


def simulate_trade(symbol, full_df, entry_price, entry_time,
                   trailing_stop_pct=None,
                   hard_ceiling=None,
                   stop_loss_pct=None,
                   grace_period_minutes=None):
    """
    Replays one simulated trade through the exit logic checked every
    CHECK_INTERVAL_MINUTES.

    Grace period: the trailing stop will NOT fire within grace_period_minutes
    of entry_time. The hard stop loss still fires immediately throughout.
    """
    if trailing_stop_pct    is None: trailing_stop_pct    = TRAILING_STOP_PCT
    if hard_ceiling         is None: hard_ceiling         = HARD_PROFIT_CEILING
    if stop_loss_pct        is None: stop_loss_pct        = STOP_LOSS_PCT
    if grace_period_minutes is None: grace_period_minutes = GRACE_PERIOD_MINUTES

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

        # Hard ceiling — always active
        if pnl_pct >= hard_ceiling:
            return {
                "symbol": symbol, "exit_reason": "HARD_CEILING",
                "exit_time": timestamp, "exit_price": current_price,
                "pnl_pct": pnl_pct, "peak_price": high_water_mark,
                "held_overnight": timestamp.date() != entry_time.date()
            }

        # Hard stop loss — always active
        if pnl_pct <= -stop_loss_pct:
            return {
                "symbol": symbol, "exit_reason": "STOP_LOSS",
                "exit_time": timestamp, "exit_price": current_price,
                "pnl_pct": pnl_pct, "peak_price": high_water_mark,
                "held_overnight": timestamp.date() != entry_time.date()
            }

        # Trailing stop — suppressed during grace period
        if (not in_grace_period
                and high_water_mark > entry_price
                and drop_from_peak_pct <= -trailing_stop_pct):
            return {
                "symbol": symbol, "exit_reason": "TRAILING_STOP",
                "exit_time": timestamp, "exit_price": current_price,
                "pnl_pct": pnl_pct, "peak_price": high_water_mark,
                "held_overnight": timestamp.date() != entry_time.date()
            }

    if len(sampled) > 0:
        last_row  = sampled.iloc[-1]
        last_time = sampled.index[-1]
        pnl_pct   = ((last_row["Close"] - entry_price) / entry_price) * 100
        return {
            "symbol": symbol, "exit_reason": "STILL_OPEN_AT_DATA_END",
            "exit_time": last_time, "exit_price": last_row["Close"],
            "pnl_pct": pnl_pct, "peak_price": high_water_mark,
            "held_overnight": last_time.date() != entry_time.date()
        }

    return None


def _run_ticker(symbol, hist, trailing_stop_pct, grace_period_minutes=None):
    results    = []
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
                                grace_period_minutes=grace_period_minutes)
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
    Returns (optimal_pct: float, report: str).
    """
    try:
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period=f"{days_back}d", interval="5m")

        if hist.empty or len(hist) < 20:
            return TRAILING_STOP_PCT, f"{symbol}: insufficient data, using default {TRAILING_STOP_PCT}%"

        best_pct      = TRAILING_STOP_PCT
        best_avg_pnl  = None
        summary_lines = []

        for candidate in CALIBRATION_CANDIDATES:
            results = _run_ticker(symbol, hist.copy(), trailing_stop_pct=candidate,
                                  grace_period_minutes=GRACE_PERIOD_MINUTES)
            if len(results) < MIN_TRADES_FOR_CALIBRATION:
                summary_lines.append(f"  {candidate}%: too few trades ({len(results)})")
                continue

            df       = pd.DataFrame(results)
            avg_pnl  = df["pnl_pct"].mean()
            win_rate = (df["pnl_pct"] > 0).mean() * 100
            ts_hits  = (df["exit_reason"] == "TRAILING_STOP").sum()
            summary_lines.append(
                f"  {candidate}%: avg P&L={avg_pnl:+.2f}%  win={win_rate:.0f}%  "
                f"trailing_stop_hits={ts_hits}/{len(results)}"
            )

            if best_avg_pnl is None or avg_pnl > best_avg_pnl:
                best_avg_pnl = avg_pnl
                best_pct     = candidate

        if best_avg_pnl is not None:
            report = (
                f"{symbol} trailing stop calibration (last {days_back}d, "
                f"{GRACE_PERIOD_MINUTES}min grace):\n"
                + "\n".join(summary_lines)
                + f"\n  → OPTIMAL: {best_pct}% (avg P&L {best_avg_pnl:+.2f}%)"
            )
        else:
            report = f"{symbol}: calibration inconclusive, using default {TRAILING_STOP_PCT}%"

        print(f"📐 {symbol} calibrated trailing stop: {best_pct}% (best avg P&L: {best_avg_pnl:+.2f}%)" if best_avg_pnl is not None else f"📐 {symbol}: calibration inconclusive")
        return best_pct, report, best_avg_pnl

    except Exception as e:
        print(f"⚠️ Calibration failed for {symbol}: {e}")
        return TRAILING_STOP_PCT, f"{symbol}: calibration error ({e}), using default {TRAILING_STOP_PCT}%", None


# ---------------------------------------------------------------------------
# STANDALONE MODE
# ---------------------------------------------------------------------------

def run_backtest():
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


def summarize(results):
    if not results:
        print("\nNo results to summarize.")
        return

    df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
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

    print("\n" + "=" * 60)
    print(f"TRAILING STOP CALIBRATION (grace period: {GRACE_PERIOD_MINUTES} min)")
    print("=" * 60)
    for symbol in TICKERS:
        optimal, report, _ = calibrate_trailing_stop(symbol)
        print(report)

    df.to_csv("backtest_results.csv", index=False)
    print("\nFull results saved to backtest_results.csv")


if __name__ == "__main__":
    print(f"Running backtest: {TICKERS}")
    print(f"Settings: Stop Loss={STOP_LOSS_PCT}% | Trailing Stop={TRAILING_STOP_PCT}% | "
          f"Hard Ceiling={HARD_PROFIT_CEILING}% | Grace Period={GRACE_PERIOD_MINUTES}min")
    print(f"Check interval: every {CHECK_INTERVAL_MINUTES} min | Lookback: {DAYS_BACK} days\n")
    results = run_backtest()
    summarize(results)
