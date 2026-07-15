"""
Standalone backtester + trailing stop calibrator.

MODES:
  1. Standalone backtest — run directly:
         python3 backtest.py
     Tests fixed and ATR-based trailing stops, prints summary, walk-forward
     calibration, Sharpe ratios, and Monte Carlo simulation results.

  2. Calibration mode — import and call:
         from backtest import calibrate_trailing_stop
         optimal_pct, report, best_avg_pnl = calibrate_trailing_stop("TSLA")
     Uses walk-forward validation + Sharpe ratio to pick the optimal stop.

  3. ATR mode — trailing stop sized dynamically by Average True Range.

Does NOT import anything from the live bot files and cannot place trades.
"""

import yfinance as yf
import pandas as pd
import numpy as np

# ---- CONFIG ----
TRAILING_STOP_PCT      = 2.0
HARD_PROFIT_CEILING    = 3.0
STOP_LOSS_PCT          = 2.0
CHECK_INTERVAL_MINUTES = 5
GRACE_PERIOD_MINUTES   = 90

TICKERS   = ["AAPL", "MSFT", "NVDA", "META", "TSLA", "AMZN", "JNJ", "XOM", "COST"]
DAYS_BACK = 30

# ---- ATR CONFIG ----
ATR_PERIOD     = 14
ATR_MULTIPLIER = 1.5

# ---- CALIBRATION CONFIG ----
CALIBRATION_CANDIDATES     = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
MIN_TRADES_FOR_CALIBRATION = 4

# ---- WALK-FORWARD CONFIG ----
WALK_FORWARD_TRAIN_DAYS    = 20
WALK_FORWARD_VALIDATE_DAYS = 10

# ---- SHARPE RATIO CONFIG ----
RISK_FREE_RATE_ANNUAL    = 0.05
RISK_FREE_RATE_PER_TRADE = RISK_FREE_RATE_ANNUAL / (252 * 2)
MIN_SHARPE_FOR_PREFERENCE = 0.3

# ---- MONTE CARLO CONFIG ----
MONTE_CARLO_SIMULATIONS  = 500
MONTE_CARLO_SEQUENCE_LEN = 20


def fmt_sharpe(val):
    """Safe Sharpe formatter — avoids conditional f-string issues in Python 3.13."""
    if val is None:
        return "N/A"
    return f"{val:.3f}"


def compute_sharpe(pnl_series, risk_free_per_trade=RISK_FREE_RATE_PER_TRADE):
    try:
        if len(pnl_series) < 4:
            return None
        returns = pd.Series(pnl_series)
        avg     = returns.mean()
        std     = returns.std()
        if std == 0:
            return None
        sharpe = (avg - risk_free_per_trade) / std
        return round(float(sharpe), 3)
    except Exception:
        return None


def compute_atr(hist, period=ATR_PERIOD):
    try:
        high       = hist["High"]
        low        = hist["Low"]
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
    try:
        if pd.isna(atr_value) or atr_value <= 0 or entry_price <= 0:
            return TRAILING_STOP_PCT
        stop_price = entry_price - (atr_value * multiplier)
        stop_pct   = ((entry_price - stop_price) / entry_price) * 100
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
    if trailing_stop_pct    is None: trailing_stop_pct    = TRAILING_STOP_PCT
    if hard_ceiling         is None: hard_ceiling         = HARD_PROFIT_CEILING
    if stop_loss_pct        is None: stop_loss_pct        = STOP_LOSS_PCT
    if grace_period_minutes is None: grace_period_minutes = GRACE_PERIOD_MINUTES

    if use_atr and atr_series is not None:
        try:
            atr_at_entry      = atr_series.asof(entry_time)
            trailing_stop_pct = get_atr_trailing_stop_pct(entry_price, atr_at_entry)
        except Exception:
            pass

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


def run_monte_carlo(pnl_list, n_simulations=MONTE_CARLO_SIMULATIONS,
                    sequence_len=MONTE_CARLO_SEQUENCE_LEN):
    if len(pnl_list) < sequence_len:
        return None

    pnl_array     = np.array(pnl_list)
    final_returns = []
    max_drawdowns = []
    sharpe_scores = []

    for _ in range(n_simulations):
        sample     = np.random.choice(pnl_array, size=sequence_len, replace=True)
        cum_return = sample.sum()
        final_returns.append(cum_return)

        equity = np.cumsum(sample)
        peak   = np.maximum.accumulate(equity)
        dd     = equity - peak
        max_drawdowns.append(dd.min())

        if sample.std() > 0:
            sharpe_scores.append(
                (sample.mean() - RISK_FREE_RATE_PER_TRADE) / sample.std()
            )

    final_returns = np.array(final_returns)
    max_drawdowns = np.array(max_drawdowns)

    return {
        "n_simulations":    n_simulations,
        "sequence_len":     sequence_len,
        "median_return":    round(float(np.median(final_returns)), 2),
        "mean_return":      round(float(np.mean(final_returns)), 2),
        "best_case":        round(float(np.percentile(final_returns, 95)), 2),
        "worst_case":       round(float(np.percentile(final_returns, 5)), 2),
        "pct_profitable":   round(float((final_returns > 0).mean() * 100), 1),
        "avg_max_drawdown": round(float(np.mean(max_drawdowns)), 2),
        "worst_drawdown":   round(float(np.min(max_drawdowns)), 2),
        "avg_sharpe":       round(float(np.mean(sharpe_scores)), 3) if sharpe_scores else None,
    }


def calibrate_trailing_stop(symbol, days_back=DAYS_BACK):
    """
    Called by research_agent.py for each candidate stock (in parallel).
    Uses walk-forward validation + Sharpe ratio to pick optimal stop.
    Returns (optimal_pct: float, report: str, best_avg_pnl: float)
    """
    try:
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period=f"{days_back}d", interval="5m")

        if hist.empty or len(hist) < 20:
            return TRAILING_STOP_PCT, f"{symbol}: insufficient data, using default {TRAILING_STOP_PCT}%", None

        hist["date"] = hist.index.date
        unique_days  = sorted(hist["date"].unique())

        if len(unique_days) < WALK_FORWARD_TRAIN_DAYS + 2:
            train_hist    = hist
            validate_hist = hist
        else:
            train_cutoff  = unique_days[WALK_FORWARD_TRAIN_DAYS - 1]
            train_hist    = hist[hist["date"] <= train_cutoff]
            validate_hist = hist[hist["date"] > train_cutoff]

        daily_hist = ticker.history(period=f"{days_back}d", interval="1d")
        atr_series = compute_atr(daily_hist) if not daily_hist.empty else None

        # ---- PHASE 1: Train ----
        train_candidates = []
        for candidate in CALIBRATION_CANDIDATES:
            results = _run_ticker(symbol, train_hist.copy(),
                                  trailing_stop_pct=candidate,
                                  grace_period_minutes=GRACE_PERIOD_MINUTES)
            if len(results) < MIN_TRADES_FOR_CALIBRATION:
                continue
            pnl_list = [r["pnl_pct"] for r in results]
            avg_pnl  = float(np.mean(pnl_list))
            sharpe   = compute_sharpe(pnl_list)
            train_candidates.append((candidate, avg_pnl, sharpe))

        if not train_candidates:
            return TRAILING_STOP_PCT, f"{symbol}: insufficient trades in training, using default", None

        # ---- PHASE 2: Validate fixed stops by Sharpe ----
        best_fixed_pct  = TRAILING_STOP_PCT
        best_val_sharpe = None
        best_val_pnl    = None
        validate_lines  = []

        for candidate, train_pnl, train_sharpe in train_candidates:
            val_results = _run_ticker(symbol, validate_hist.copy(),
                                      trailing_stop_pct=candidate,
                                      grace_period_minutes=GRACE_PERIOD_MINUTES)
            if not val_results:
                validate_lines.append(f"  {candidate}%: no validation trades")
                continue

            val_pnl_list = [r["pnl_pct"] for r in val_results]
            val_avg_pnl  = float(np.mean(val_pnl_list))
            val_sharpe   = compute_sharpe(val_pnl_list)

            validate_lines.append(
                f"  {candidate}%: val avg P&L={val_avg_pnl:+.2f}%  "
                f"val Sharpe={fmt_sharpe(val_sharpe)}  "
                f"({len(val_results)} trades)"
            )

            if val_sharpe is not None:
                if best_val_sharpe is None or val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    best_fixed_pct  = candidate
                    best_val_pnl    = val_avg_pnl
            elif best_val_pnl is None or val_avg_pnl > best_val_pnl:
                best_val_pnl   = val_avg_pnl
                best_fixed_pct = candidate

        # ---- PHASE 3: Validate ATR stop ----
        atr_val_sharpe = None
        atr_val_pnl    = None
        avg_atr_stop   = TRAILING_STOP_PCT

        if atr_series is not None and not atr_series.empty:
            atr_results = _run_ticker(symbol, validate_hist.copy(),
                                      trailing_stop_pct=TRAILING_STOP_PCT,
                                      grace_period_minutes=GRACE_PERIOD_MINUTES,
                                      use_atr=True, atr_series=atr_series)
            if atr_results:
                atr_pnl_list   = [r["pnl_pct"] for r in atr_results]
                atr_val_pnl    = float(np.mean(atr_pnl_list))
                atr_val_sharpe = compute_sharpe(atr_pnl_list)
                avg_atr_stop   = float(np.mean([r["trailing_stop_used"] for r in atr_results]))

        # ---- Pick winner by Sharpe ----
        use_atr_final = False
        if (atr_val_sharpe is not None and best_val_sharpe is not None
                and atr_val_sharpe > best_val_sharpe):
            use_atr_final = True
            final_pnl     = atr_val_pnl
            final_pct     = round(avg_atr_stop, 1)
            final_sharpe  = atr_val_sharpe
        elif (atr_val_pnl is not None and best_val_pnl is None and atr_val_pnl > 0):
            use_atr_final = True
            final_pnl     = atr_val_pnl
            final_pct     = round(avg_atr_stop, 1)
            final_sharpe  = atr_val_sharpe
        else:
            final_pnl    = best_val_pnl if best_val_pnl is not None else 0.0
            final_pct    = best_fixed_pct
            final_sharpe = best_val_sharpe

        # ---- Build report ----
        report_lines = [
            f"{symbol} walk-forward calibration "
            f"(train {WALK_FORWARD_TRAIN_DAYS}d / validate {WALK_FORWARD_VALIDATE_DAYS}d):"
        ]
        report_lines.extend(validate_lines)
        if atr_val_pnl is not None:
            report_lines.append(
                f"  ATR-based (avg stop ~{avg_atr_stop:.1f}%): "
                f"val avg P&L={atr_val_pnl:+.2f}%  "
                f"val Sharpe={fmt_sharpe(atr_val_sharpe)}"
            )
        winner_type = f"ATR-based (~{final_pct}%)" if use_atr_final else f"fixed {final_pct}%"
        report_lines.append(
            f"  → WINNER: {winner_type} | "
            f"validated avg P&L {final_pnl:+.2f}% | "
            f"Sharpe {fmt_sharpe(final_sharpe)}"
        )
        report = "\n".join(report_lines)

        print(f"📐 {symbol} calibrated trailing stop: {final_pct}% "
              f"(val avg P&L: {final_pnl:+.2f}%, Sharpe: {fmt_sharpe(final_sharpe)})")
        return final_pct, report, final_pnl

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


def run_atr_backtest():
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

    overall_sharpe = compute_sharpe(df["pnl_pct"].tolist())
    print(f"Overall Sharpe ratio   : {fmt_sharpe(overall_sharpe)}")

    print("\nExit reason breakdown:")
    print(df["exit_reason"].value_counts())

    print("\nAvg P&L by exit reason:")
    print(df.groupby("exit_reason")["pnl_pct"].mean().round(2))

    print("\nPer-symbol average P&L and Sharpe:")
    for symbol in df["symbol"].unique():
        sym_df  = df[df["symbol"] == symbol]
        avg_pnl = sym_df["pnl_pct"].mean()
        sharpe  = compute_sharpe(sym_df["pnl_pct"].tolist())
        print(f"  {symbol}: avg P&L={avg_pnl:+.2f}%  Sharpe={fmt_sharpe(sharpe)}  "
              f"({len(sym_df)} trades)")

    if "held_overnight" in df.columns:
        overnight_count = df["held_overnight"].sum()
        print(f"\nTrades held overnight: {overnight_count} of {len(df)} "
              f"({overnight_count/len(df)*100:.1f}%)")
        if overnight_count > 0:
            print(f"Avg P&L overnight : {df[df['held_overnight']]['pnl_pct'].mean():.2f}%")
            print(f"Avg P&L same-day  : {df[~df['held_overnight']]['pnl_pct'].mean():.2f}%")

    if "trailing_stop_used" in df.columns:
        print(f"\nAvg trailing stop used: {df['trailing_stop_used'].mean():.2f}%")

    print("\n" + "=" * 60)
    print(f"MONTE CARLO SIMULATION ({MONTE_CARLO_SIMULATIONS} runs, "
          f"{MONTE_CARLO_SEQUENCE_LEN} trades each)")
    print("=" * 60)
    mc = run_monte_carlo(df["pnl_pct"].tolist())
    if mc:
        print(f"Median total return    : {mc['median_return']:+.2f}%")
        print(f"Mean total return      : {mc['mean_return']:+.2f}%")
        print(f"Best case (95th pct)   : {mc['best_case']:+.2f}%")
        print(f"Worst case (5th pct)   : {mc['worst_case']:+.2f}%")
        print(f"% of runs profitable   : {mc['pct_profitable']:.1f}%")
        print(f"Avg max drawdown       : {mc['avg_max_drawdown']:+.2f}%")
        print(f"Worst drawdown seen    : {mc['worst_drawdown']:+.2f}%")
        print(f"Avg Sharpe across runs : {fmt_sharpe(mc['avg_sharpe'])}")
    else:
        print("Insufficient data for Monte Carlo simulation.")


if __name__ == "__main__":
    print(f"Running backtest: {TICKERS}")
    print(f"Settings: Stop Loss={STOP_LOSS_PCT}% | Trailing Stop={TRAILING_STOP_PCT}% | "
          f"Hard Ceiling={HARD_PROFIT_CEILING}% | Grace Period={GRACE_PERIOD_MINUTES}min")
    print(f"Check interval: every {CHECK_INTERVAL_MINUTES} min | Lookback: {DAYS_BACK} days\n")

    results_fixed = run_backtest()
    summarize(results_fixed, label="FIXED TRAILING STOP")

    print("\n" + "=" * 60)
    print("Running ATR-based trailing stop comparison...")
    results_atr = run_atr_backtest()
    summarize(results_atr, label="ATR-BASED TRAILING STOP")

    print("\n" + "=" * 60)
    print(f"WALK-FORWARD CALIBRATION WITH SHARPE "
          f"(train {WALK_FORWARD_TRAIN_DAYS}d / validate {WALK_FORWARD_VALIDATE_DAYS}d)")
    print("=" * 60)
    for symbol in TICKERS:
        optimal, report, pnl = calibrate_trailing_stop(symbol)
        print(report)

    if results_fixed:
        pd.DataFrame(results_fixed).to_csv("backtest_results_fixed.csv", index=False)
    if results_atr:
        pd.DataFrame(results_atr).to_csv("backtest_results_atr.csv", index=False)
    print("\nResults saved to backtest_results_fixed.csv and backtest_results_atr.csv")
