import math
import re
import random as rand
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplfinance as mpf
import os
from pathlib import Path
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from market_analysis import analyze_looped_results
from backtest_compare import run_backtest_comparison
import price_open
import argparse
import json
import secrets
import shutil
import zipfile
import numpy as np
import backtest_compare as bc
from backtest_compare import load_config
from market_analysis import realised_vol_feedback

# ─── Command line, config.toml and the volatility-search block ────────────────
_parser = argparse.ArgumentParser(add_help=True, description="Random OHLC price simulator with self-correcting volatility.")
_parser.add_argument("--config", default=None, help="Path to config.toml (default: next to this script).")
_parser.add_argument("--seed", type=int, default=None, help="Re-use a previous run's random seed (-10,000,000 .. 10,000,000).")
_parser.add_argument("--auto-recal", action="store_true", help="Forward runs: re-calibrate the stored volatility when it is stale.")
_parser.add_argument("--auto-correct-vol", action="store_true",
                     help="Re-generate runs whose realised volatility differs >20%% from what vol_index should give.")
_parser.add_argument("--archive", action="store_true", help="Zip the per-run CSV/PNG files after the analysis.")
ARGS, _UNKNOWN_ARGS = _parser.parse_known_args()
CFG = load_config(ARGS.config)

# Volatility self-correction settings (edit config.toml [volatility_search], or override here).
VOLATILITY_SEARCH = CFG["volatility_search"]
VOL_TARGET = VOLATILITY_SEARCH["target"]          # accuracy_index the search tries to reach
VOL_STEP_ABS = VOLATILITY_SEARCH["step_abs"]      # absolute step (step_mode = "abs")
VOL_STEP_PCT = VOLATILITY_SEARCH["step_pct"]      # relative step in % (step_mode = "pct")
MAX_VOL_ITERS = VOLATILITY_SEARCH["max_iters"]    # iteration cap per search
MC_PATHS = VOLATILITY_SEARCH["mc_paths"]          # paths per cycle per evaluation (median used)
ACCURACY_DECAY_THRESHOLD = VOLATILITY_SEARCH["accuracy_decay_threshold"]

# ─── Tutorial modes (beginner / advanced): explain each step before it is asked ─
TUTORIAL_MODE = str(CFG["tutorial"]["mode"]).lower()
TUTORIAL = {
    "welcome": ("This program invents random price charts, then (optionally) checks them against real prices and "
                "learns how volatile the invented prices should be. You answer a few questions; Enter accepts the default.",
                "Pipeline: price_modeling.py generates data_N.csv/image_N.png -> market_analysis.py computes indicators -> "
                "backtest_compare.py scores against Yahoo prices and searches generator parameters -> ml_model.py trains on the result."),
    "asset_type": ("The type of market. It only changes how the Yahoo Finance symbol is written (e.g. GBP/USD becomes GBPUSD=X).",
                   "asset_type feeds _yfinance_symbol(): fx -> '=X', futures -> '=F', crypto '/'->'-'. It is also the transfer-learning "
                   "class: a new instrument seeds its volatility search from the nearest instrument of the same asset_type."),
    "instrument": ("The name of the thing to simulate, e.g. GBP/USD or AAPL.",
                   "ticker_name labels outputs; chart_symbol (Yahoo symbol) drives real-price downloads and vol_equilibrium.csv keys."),
    "start_price": ("Where the invented price starts. Option 2 downloads today's real price.",
                    "starting_price seeds every cycle; real data is rebased to the same start so price LEVEL never affects scores."),
    "timeframe": ("How much time one chart covers (1d = one day ... 5y = five years).",
                  "TIMEFRAME_CONFIG[tf].span_days sets the window; the vol_index sigmoid is calibrated on ONE-DAY % change, so multi-day "
                  "changes are divided by sqrt(trading days) when SCALE_PERIOD_CHANGE_TO_DAILY is on."),
    "interval": ("How long each candle is. More candles = better indicators (aim for 30-100).",
                 "SMA 12, Bollinger 20 and MACD (12-26-9, 35 candles to complete) need history; bars_for() shows the exact count."),
    "price_mode": ("Whether the invented price may go below zero (some futures can) or is stopped at zero.",
                   "allow_negative disables the zero floor on close and low; ML drawdown features become NaN for non-positive paths."),
    "backtest": ("Backtesting replays past periods and compares the invented price with what really happened.",
                 "Cycle 1 = most recent window, cycle N = oldest. Each cycle is scored by score_cycle(): path, direction, net move, range, "
                 "plus [PRICE] net_price_match and [BARS] direction_match_pct."),
    "cycles": ("How many past periods to replay. More cycles = more reliable learning.",
               "The newest holdout_frac of cycles is kept out of calibration and used as the out-of-sample check (oos_F)."),
    "ga_timeframe": ("A genetic algorithm tries timeframe/candle combinations and keeps the one the simulation tracks best.",
                     "Fitness = composite F of the last 3 cycles at a range-matched vol (sigma_real / 0.2309). Population mutates "
                     "interval within a timeframe or hops timeframe; results are cached per (tf, interval)."),
    "self_correct": ("The program learns: it changes the volatility, re-scores against real prices and repeats until it stops improving.",
                     "run_self_correction(): range discovery [0.1, 5] x sigma_real -> seed (history / transfer / formula) -> null baseline -> "
                     "global search (consensus) -> decay lambda -> GA over shape genes -> per-cycle learning -> fresh-seed validation."),
    "method": ("How the search moves: 'auto' combines several methods and is a good default.",
               "step = LR-scheduled +/-10%; bisect = root of log(range ratio); grid/coarse_fine; cobyla; bayes (GP + EI); ga; mc = fresh-seed "
               "Monte-Carlo with eps-convergence. All use common random numbers except mc."),
    "loops": ("How many times the per-cycle learning is repeated. Each loop refines what was learned.",
              "Per-cycle vols are a running log-mean across loops AND across program runs (vol_equilibrium.csv), so repeated runs keep learning."),
    "plot_mode": ("One chart, or many runs to build a bigger data set for the ML step.", 
                  "run_count >= 100 offers six output modes; mode 6 (CSV + metrics only) is the fastest and smallest."),
    "output_mode": ("Choose how much to save. Charts are slow and large; CSVs are small.",
                    "Modes toggle (source PNG, analysis PNG, ML). Archive/zip options live in config.toml [storage]."),
    "learned_vol": ("A previous self-correction learned a better volatility for this market. Using it makes new charts more realistic.",
                    "forward vol = the learned global vol_index (vol_equilibrium.csv) instead of the one-period market-change formula."),
    "ml": ("A machine-learning model tries to predict whether each chart ends up or down. On random data it should score about 50%.",
           "Purged k-fold CV, accuracy-weighted samples, optional RF tuning (rf_tuning.toml), model comparison, conformal bounds, "
           "adversarial validation, SHAP-style contributions and an HTML report are produced by run_ml_pipeline()."),
}


def tut(key: str) -> None:
    """Print the beginner / advanced explanation for a step (nothing when tutorial mode is off)."""
    if TUTORIAL_MODE in ("beginner", "advanced") and key in TUTORIAL:
        text = TUTORIAL[key][0 if TUTORIAL_MODE == "beginner" else 1]
        print(f"\n  [{TUTORIAL_MODE} tip] {text}")


def ask(prompt: str, key: str | None = None) -> str:
    """input() preceded by the tutorial text for this step."""
    if key:
        tut(key)
    return input(prompt)

# ─── Timeframe configuration ──────────────────────────────────────────────────
#
# Each timeframe has a default candle interval (what the program always used) plus
# a menu of alternatives, so the SMA 12 / Bollinger 20 / MACD (needs 35 candles)
# have enough data. Aim for roughly 30-100 candles; the menu shows the exact
# count for every option.
#
#   span_days         trading days covered (None = Jan 1 -> today for ytd)
#   default_interval  used when you just press Enter
#   intervals         every interval offered for that timeframe
#   description       one-line summary of the DEFAULT, shown in the timeframe menu
#
# Interval keys:  "15m" / "1h" = intraday candles inside the trading session
#                 "3d"         = candles made of N business days (closing at 16:00)
#                 "2w"         = candles made of N weeks (Monday 09:00 open)
#                 "auto"       = ytd only: ~50 evenly spaced days (the original behaviour)
TIMEFRAME_CONFIG: dict = { 
    "1d":  {"span_days": 1,    "span_text": "one trading day",
            "default_interval": "10m", "intervals": ["5m", "10m", "15m", "20m", "30m", "1h"],
            "description": "54 bars × 10 min  (09:00–17:50, one trading day)"},
    "5d":  {"span_days": 5,    "span_text": "5 trading days (Mon–Fri)",
            "default_interval": "1h",  "intervals": ["30m", "45m", "1h", "90m", "2h", "3h"],
            "description": "45 bars × 1 hour  (09:00–17:00, Mon–Fri)"},
    "1m":  {"span_days": 22,   "span_text": "22 trading days",
            "default_interval": "4h",  "intervals": ["2h", "3h", "4h", "6h"],
            "description": "44 bars × 4 hours (09:00 + 13:00, ~22 trading days)"},
    "3m":  {"span_days": 65,   "span_text": "65 trading days (~1 quarter)",
            "default_interval": "1d",  "intervals": ["1d", "2d", "3d"],
            "description": "65 bars × 1 day   (~1 quarter of trading days)"},
    "1y":  {"span_days": 260,  "span_text": "52 weeks",
            "default_interval": "1w",  "intervals": ["3d", "1w", "2w"],
            "description": "52 bars × 1 week  (Monday open, calendar year)"},
    "5y":  {"span_days": 1300, "span_text": "5 years (260 weeks)",
            "default_interval": "1w",  "intervals": ["1w", "2w", "3w", "4w", "6w", "8w"],
            "description": "260 bars × 1 week  (Monday open, 5 calendar years)"},
    "ytd": {"span_days": None, "span_text": "January 1 to today",
            "default_interval": "auto", "intervals": ["auto", "1d", "2d", "3d", "5d"],
            "description": "~50 bars, auto-scaled (January 1 to today)"},
}

# Simulated trading session used by the intraday timeframes (1d, 5d, 1m).
# Default 09:00-18:00. For a 24-hour market (FX, crypto) set
#   SESSION_START_HOUR = 0   and   SESSION_MINUTES = 1440
# so, for example, a 1d chart gives 96 candles at 15m, 48 at 30m, 24 at 1h.
# Keep the default for stocks: Yahoo only has stock prices during market hours, so a
# 24-hour simulated day would leave most backtest candles without real data.
SESSION_START_HOUR = 9
SESSION_MINUTES    = 540

# Candle-count guide-rails shown in the interval menu.
MIN_CANDLES_ANALYSIS = 12    # analysis needs SMA 12 -> fewer candles cannot be analysed
MIN_CANDLES_MACD     = 35    # MACD 12-26 + 9 signal needs 35 candles to complete
MAX_CANDLES_COMFY    = 100   # beyond this the chart gets crowded

VALID_TIMEFRAMES = set(TIMEFRAME_CONFIG)

# ─── Market-change input to the volatility formula ────────────────────────────
# The % change fed to the vol_index formula now spans the chosen timeframe
# (1d = last day, 5d, 1m, 3m, 1y, 5y, ytd = Jan 1 -> now) instead of always
# being the last day's move.
#
# SCALE_PERIOD_CHANGE_TO_DAILY
#   The vol_index sigmoid is calibrated on a ONE-DAY % change: +1% already gives
#   roughly +/-25% per simulated bar and +2% roughly +/-200%. A raw multi-month
#   change (e.g. +8% YTD) saturates it and the simulated price hits zero or goes
#   flat. True  -> use change / sqrt(trading days in the span), which is the
#   formula's native scale and is identical to the raw value for 1d (default).
#   False -> feed the raw period % change straight into the formula.
SCALE_PERIOD_CHANGE_TO_DAILY = True
# BACKTEST_POINT_IN_TIME_CHANGE
#   True  -> each backtest cycle uses the same-length period that ENDED just
#            before the cycle began, so the simulation never sees the outcome it
#            is later scored against (default).
#   False -> every cycle uses the change measured up to today, like forward runs.
BACKTEST_POINT_IN_TIME_CHANGE = True


def normalise_timeframe(raw: str) -> str | None:
    """Return the canonical timeframe key, or None if the input is invalid."""
    t = raw.strip().lower().replace(" ", "").replace("-", "")
    mapping = {
        "1d": "1d",  "1day": "1d",
        "5d": "5d",  "5day": "5d",  "5days": "5d",
        "1m": "1m",  "1month": "1m", "1mo": "1m",
        "3m": "3m",  "3month": "3m", "3mo": "3m",
        "1q": "3m",  "1quarter": "3m",
        "1y": "1y",  "1year": "1y",  "1yr": "1y",
        "5y": "5y",  "5year": "5y", "5years": "5y", "5yr": "5y", "5yrs": "5y",
        "ytd": "ytd","yeartodate": "ytd", "year": "ytd",
    }
    return mapping.get(t)


_INTERVAL_UNITS = {
    "m": "m", "min": "m", "mins": "m", "minute": "m", "minutes": "m",
    "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
    "d": "d", "day": "d", "days": "d",
    "w": "w", "wk": "w", "wks": "w", "week": "w", "weeks": "w",
}


def _canonical_interval(value: int, unit: str) -> str:
    """Canonical key: 60m -> 1h, 120m -> 2h, 90m stays 90m."""
    if unit == "m" and value % 60 == 0:
        return f"{value // 60}h"
    return f"{value}{unit}"


def normalise_interval(raw: str, timeframe: str) -> str | None:
    """Return the canonical interval key for `timeframe`, or None if invalid.

    Accepts '15m', '15 min', '1h', '1 hour', '3d', '2 weeks', 'auto' (ytd), or a bare
    number when it matches exactly one option (e.g. '30' on a 1d chart -> '30m').
    """
    cfg = TIMEFRAME_CONFIG[timeframe]
    options = cfg["intervals"]
    t = raw.strip().lower().replace(" ", "")
    if t in ("", "default"):
        return cfg["default_interval"]
    if t == "auto":
        return "auto" if "auto" in options else None
    match = re.fullmatch(r"(\d+)([a-z]*)", t)
    if not match:
        return None
    value, unit = int(match.group(1)), match.group(2)
    if unit:
        canonical_unit = _INTERVAL_UNITS.get(unit)
        if canonical_unit is None:
            return None
        if canonical_unit == "h":
            key = _canonical_interval(value * 60, "m")   # 2 hours -> 120m -> "2h"
        else:
            key = _canonical_interval(value, canonical_unit)
        return key if key in options else None
    # Bare number: only interpreted when every option uses the same kind of unit
    # (all intraday, all days or all weeks). A mixed menu like 1y (3d / 1w / 2w) is
    # ambiguous, so the user must type the unit.
    numeric = [k for k in options if k != "auto"]
    families = {"d" if k.endswith("d") else "w" if k.endswith("w") else "intraday" for k in numeric}
    if len(families) != 1:
        return None
    if families == {"intraday"}:
        as_minutes = _canonical_interval(value, "m")          # "60" -> "1h", "90" -> "90m"
        if as_minutes in options:
            return as_minutes
    hits = [k for k in numeric if int(re.match(r"\d+", k).group()) == value]   # "3" -> "3h"
    return hits[0] if len(hits) == 1 else None


def interval_spec(timeframe: str, interval: str) -> dict:
    """Describe one candle interval in a form the generator and backtester share.

    kind "intraday": candles of `minutes` inside the trading session
    kind "bday":     candles of `days` business days, stamped at the 16:00 close
                     (days=None, auto=True is the ytd ~50-point sampling)
    kind "week":     candles of `weeks` weeks, stamped at the Monday 09:00 open
    """
    if interval == "auto":
        return {"kind": "bday", "days": None, "auto": True}
    match = re.fullmatch(r"(\d+)([mhdw])", interval)
    if not match:
        raise ValueError(f"Bad interval key: {interval!r}")
    value, unit = int(match.group(1)), match.group(2)
    if unit == "m":
        return {"kind": "intraday", "minutes": value}
    if unit == "h":
        return {"kind": "intraday", "minutes": value * 60}
    if unit == "d":
        return {"kind": "bday", "days": value}
    return {"kind": "week", "weeks": value}


def interval_label(interval: str) -> str:
    """Human-readable label: 15m -> '15 min', 2h -> '2 hours', 3d -> '3 days'."""
    if interval == "auto":
        return "auto (~50 candles)"
    spec = interval_spec("1d", interval)  # timeframe is irrelevant for labelling
    if spec["kind"] == "intraday":
        minutes = spec["minutes"]
        if minutes % 60 == 0:
            hours = minutes // 60
            return f"{hours} hour{'s' if hours != 1 else ''}"
        return f"{minutes} min"
    unit, count = ("day", spec["days"]) if spec["kind"] == "bday" else ("week", spec["weeks"])
    return f"{count} {unit}{'s' if count != 1 else ''}"


def ytd_business_days(ref: date | None = None) -> int:
    """Business days from the first business day of the year up to `ref` (today)."""
    ref = ref or date.today()
    return len(pd.bdate_range(start=_next_business_day(date(ref.year, 1, 1)), end=ref))


def bars_for(timeframe: str, interval: str, ref_today: date | None = None) -> int:
    """Number of candles a simulation produces for this timeframe + interval."""
    cfg = TIMEFRAME_CONFIG[timeframe]
    spec = interval_spec(timeframe, interval)
    if timeframe == "ytd":
        n = ytd_business_days(ref_today)
        return min(n, 50) if spec.get("auto") else math.ceil(n / spec["days"])
    span = cfg["span_days"]
    if spec["kind"] == "intraday":
        return span * max(1, SESSION_MINUTES // spec["minutes"])
    if spec["kind"] == "bday":
        return math.ceil(span / spec["days"])
    return math.ceil(span / 5 / spec["weeks"])


def candle_notes(count: int) -> str:
    """Short guidance for a candle count (empty when it is comfortably in range)."""
    if count < MIN_CANDLES_ANALYSIS:
        return f"too few: SMA 12 needs {MIN_CANDLES_ANALYSIS} (not selectable)"
    if count < MIN_CANDLES_MACD:
        return f"under {MIN_CANDLES_MACD}: MACD signal line will not complete"
    if count > MAX_CANDLES_COMFY:
        return f"over {MAX_CANDLES_COMFY}: crowded chart"
    return ""


def choose_interval(timeframe: str) -> str:
    """Show the interval menu for a timeframe and return the chosen interval key."""
    cfg = TIMEFRAME_CONFIG[timeframe]
    default = cfg["default_interval"]
    print(f"\nCandle interval for {timeframe} ({cfg['span_text']}).")
    print(f"About 30-100 candles gives SMA 12, Bollinger 20 and the MACD signal (35) enough history:")
    for key in cfg["intervals"]:
        count = bars_for(timeframe, key)
        note = candle_notes(count)
        tags = ("  <- default" if key == default else "") + (f"   [{note}]" if note else "")
        print(f"  {key:<5} {interval_label(key):<20} {count:>4} candles{tags}")
    while True:
        tut("interval")
        raw = input(f"Candle interval [{default}]: ")
        key = normalise_interval(raw, timeframe)
        if key is None:
            print(f"  '{raw.strip()}' is not an option. Choose from: {', '.join(cfg['intervals'])}")
            continue
        count = bars_for(timeframe, key)
        if count < MIN_CANDLES_ANALYSIS:
            print(f"  {key} gives only {count} candles; at least {MIN_CANDLES_ANALYSIS} are needed. Pick another.")
            continue
        note = candle_notes(count)
        print(f"  Selected: {key} ({interval_label(key)}) -> {count} candles" + (f"  [{note}]" if note else ""))
        return key


def _next_business_day(d: date) -> date:
    """Advance d forward until it falls on a weekday."""
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _prev_business_day(d: date) -> date:
    """Step d backward until it falls on a weekday."""
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def period_start_for_cycle(
    timeframe: str, cycle_x: int, ref_today: date
) -> tuple[date, date | None]:
    """
    Return (sim_start, sim_end) for backtest cycle x.

    Ordering convention
    -------------------
    Cycle 1  → most recent period, ending at ref_today.
    Cycle x  → period that ends at (ref_today − (x−1) × period_length).

    sim_end is only set for 'ytd' (where the period length varies by year).
    For every other timeframe sim_end is None; build_timestamps() derives the
    end from sim_start + a fixed number of bars.
    """
    if timeframe == "1d":
        end_ref = _prev_business_day(ref_today - timedelta(days=cycle_x - 1))
        return end_ref, None

    elif timeframe == "5d":
        end_day = _prev_business_day(ref_today - timedelta(weeks=cycle_x - 1))
        bdays   = pd.bdate_range(end=end_day, periods=5)
        return bdays[0].date(), None

    elif timeframe == "1m":
        return _next_business_day(ref_today - relativedelta(months=cycle_x)), None

    elif timeframe == "3m":
        return _next_business_day(ref_today - relativedelta(months=3 * cycle_x)), None

    elif timeframe == "1y":
        return _next_business_day(ref_today - relativedelta(years=cycle_x)), None

    elif timeframe == "5y":
        return _next_business_day(ref_today - relativedelta(years=5 * cycle_x)), None

    elif timeframe == "ytd":
        # Cycle x  = Jan 1 through same MM-DD in year (today.year − x + 1).
        # This lets cycles=5 compare this year's YTD against the previous 4.
        year      = ref_today.year - cycle_x + 1
        sim_start = date(year, 1, 1)
        try:
            sim_end = date(year, ref_today.month, ref_today.day)
        except ValueError:
            # Feb 29 target in a non-leap year → fall back to Feb 28
            sim_end = date(year, ref_today.month, ref_today.day - 1)
        sim_end = _prev_business_day(sim_end)
        return sim_start, sim_end

    raise ValueError(f"Unknown timeframe: {timeframe!r}")


def _group_close_stamps(bdays: pd.DatetimeIndex, days_per_candle: int) -> list[pd.Timestamp]:
    """One stamp per group of `days_per_candle` business days, at the group's last day 16:00.

    days_per_candle=1 stamps every business day (the original 3m behaviour); a final
    partial group is stamped at the last available day.
    """
    n = len(bdays)
    picks = [min(days_per_candle * i + days_per_candle - 1, n - 1)
             for i in range(math.ceil(n / days_per_candle))]
    return [bdays[i].replace(hour=16, minute=0, second=0) for i in picks]


def build_timestamps(
    timeframe: str,
    sim_start: date,
    sim_end: date | None = None,
    interval: str | None = None,
) -> list[pd.Timestamp]:
    """
    Generate a list of bar timestamps for one simulation run.

    `interval` picks the candle size from TIMEFRAME_CONFIG[timeframe]["intervals"]
    (None = the timeframe's default interval, i.e. the original behaviour).

    Historical vs. future simulation horizon are kept separate:
    - Non-backtest runs use today as sim_start (forward simulation).
    - Backtest runs pass a historical sim_start computed by period_start_for_cycle().

    The timestamp list is a label for each bar's position in calendar time;
    the actual price path is always random regardless of the dates chosen.
    """
    cfg = TIMEFRAME_CONFIG[timeframe]
    spec = interval_spec(timeframe, interval or cfg["default_interval"])
    sim_start = _next_business_day(sim_start)

    if timeframe == "ytd":
        # Jan 1 (sim_start) through sim_end (or today). YTD is the only timeframe
        # whose length depends on the calendar date, so the year's first business
        # day is used as the start (forward runs pass today as sim_start).
        end = pd.Timestamp(sim_end) if sim_end else pd.Timestamp(date.today())
        ytd_start = _next_business_day(date(sim_start.year, 1, 1))
        bdays = pd.bdate_range(start=ytd_start, end=end)
        n = len(bdays)
        if n == 0:
            return [pd.Timestamp(ytd_start).replace(hour=16, minute=0, second=0)]
        if not spec.get("auto"):
            return _group_close_stamps(bdays, spec["days"])
        target = 50
        # Fewer business days than the target: return them all.
        if n <= target:
            return [d.replace(hour=16, minute=0, second=0) for d in bdays]
        # Otherwise pick `target` evenly-spaced business days across the period.
        indices = [int(round(i * (n - 1) / (target - 1))) for i in range(target)]
        uniq = []
        last = -1
        for idx in indices:
            if idx <= last:
                idx = last + 1
            uniq.append(idx)
            last = idx
        uniq = uniq[:target]
        return [bdays[i].replace(hour=16, minute=0, second=0) for i in uniq]

    if spec["kind"] == "intraday":
        # Candles of `minutes` from the session start; a candle must fit inside the
        # session (e.g. 4h in a 9h session -> 09:00 + 13:00).
        per_day = max(1, SESSION_MINUTES // spec["minutes"])
        days = pd.bdate_range(start=sim_start, periods=cfg["span_days"])
        return [
            d + pd.Timedelta(hours=SESSION_START_HOUR, minutes=j * spec["minutes"])
            for d in days
            for j in range(per_day)
        ]

    if spec["kind"] == "bday":
        bdays = pd.bdate_range(start=sim_start, periods=cfg["span_days"])
        return _group_close_stamps(bdays, spec["days"])

    # kind == "week": Monday 09:00 open, `weeks` apart
    day = pd.Timestamp(sim_start)
    while day.dayofweek != 0:
        day += pd.Timedelta(days=1)
    count = math.ceil(cfg["span_days"] / 5 / spec["weeks"])
    return [
        day.replace(hour=9, minute=0, second=0) + pd.Timedelta(weeks=spec["weeks"] * i)
        for i in range(count)
    ]


# ─── Instrument setup ─────────────────────────────────────────────────────────
tut("welcome")
asset_type = ask(
    "Asset type [forex/futures/stock/crypto/other] (default forex): ", "asset_type"
).strip().lower() or "forex"
ticker_name = ask(
    "Instrument or pair to simulate: ", "instrument"
).strip() or "GBP/USD"

price_option = ask(
    f"Starting price: [1] enter manually or [2] use current {ticker_name}? ", "start_price"
).strip()
if price_option == "2":
    if asset_type == "forex":
        default_price_symbol = ticker_name.upper().replace("/", "") + "=X"
    else:
        default_price_symbol = ticker_name
    price_symbol = input(
        f"Yahoo Finance symbol for {ticker_name} [{default_price_symbol}]: "
    ).strip() or default_price_symbol
    price_data = price_open.get_starting_price(price_symbol)
    if isinstance(price_data, dict):
        starting_price = float(price_data["current"])
    else:
        print(f"Could not get {ticker_name} price: {price_data}")
        exit()
else:
    starting_price = float(input("Starting price: "))

# Symbol shown on the analysis / backtest plots and used to pull real prices for
# backtests: the Yahoo Finance symbol typed at the prompt above (or its default).
# If the prompt was skipped (manual starting price), derive it from the
# instrument name using the same mapping price_change.py uses.
try:
    from .price_change import _yfinance_symbol
except ImportError:
    from price_change import _yfinance_symbol

if price_option == "2":
    chart_symbol = _yfinance_symbol(price_symbol, asset_type)
else:
    chart_symbol = _yfinance_symbol(ticker_name, asset_type)

# ─── Directory setup (idea 83: folders are created when missing) ──────────────
master_dir   = os.path.dirname(os.path.abspath(__file__))
results_dir  = os.path.join(master_dir, "looped_results")
analysis_dir = os.path.join(master_dir, "looped_analysis")

for folder, label in [(results_dir, "Output"), (analysis_dir, "Analysis output")]:
    if not os.path.isdir(folder):
        if CFG["storage"]["auto_create_folders"]:
            os.makedirs(folder, exist_ok=True)
            print(f"{label} folder did not exist - created: {folder}")
        else:
            print(f"{label} folder does not exist: {folder}")
            print("Create the folder manually (or set [storage] auto_create_folders = true in config.toml).")
            exit()


def clear_generated_files(directory, patterns, keep=bc.PERSISTENT_FILES):
    """Delete generated files; the learned-volatility memory files are never touched."""
    for pattern in patterns:
        for path in Path(directory).glob(pattern):
            if path.is_file() and path.name not in keep:
                path.unlink()


# ─── Timeframe selection ──────────────────────────────────────────────────────
print("\nAvailable timeframes:")
for tf, cfg in TIMEFRAME_CONFIG.items():
    print(f"  {tf:<4} — {cfg['description']}")
tut("timeframe")
tf_raw    = input("Timeframe [5d]: ").strip() or "5d"
timeframe = normalise_timeframe(tf_raw)
while timeframe is None:
    print(f"  '{tf_raw}' is not a valid timeframe. Choose from: {', '.join(VALID_TIMEFRAMES)}")
    tf_raw    = input("Timeframe [5d]: ").strip() or "5d"
    timeframe = normalise_timeframe(tf_raw)
print(f"  Selected: {timeframe} — {TIMEFRAME_CONFIG[timeframe]['description']}")

# Use real-time date whenever backtesting is not active.
today = date.today()

# ─── Simulation parameters ────────────────────────────────────────────────────
# A random seed between -10,000,000 and 10,000,000 is drawn for every run and shown; pass it back with
# --seed N to reproduce the run exactly (the engine's numpy seed is derived from it).
RUN_SEED = ARGS.seed if ARGS.seed is not None else secrets.SystemRandom().randint(-10_000_000, 10_000_000)
rand.seed(RUN_SEED)
ENGINE_SEED = (RUN_SEED + 10_000_000) % (2 ** 32)
print(f"\nRandom seed for this run: {RUN_SEED}   (re-run with --seed {RUN_SEED} to reproduce it)")

# Retired manual volatility prompt:
# question = float(input("\nVolatility of market: "))
try:
    from .price_change import get_timeframe_change_details, get_regime
except ImportError:
    from price_change import get_timeframe_change_details, get_regime

import math

_change_cache: dict = {}


def market_change_input(tf: str, as_of: date | None = None, span_days: int | None = None):
    """Return (value fed to the vol_index formula, details) for a timeframe.

    as_of=None  -> measured up to today ("Jan 1 -> now" for ytd).
    as_of=date  -> point-in-time: the same-length period ending before `as_of`.
    """
    key = (tf, as_of, span_days)
    if key not in _change_cache:
        _change_cache[key] = get_timeframe_change_details(
            chart_symbol, asset_type, tf, as_of=as_of, span_days=span_days, today=today
        )
    details = _change_cache[key]
    value = details["daily_equivalent_pct"] if SCALE_PERIOD_CHANGE_TO_DAILY else details["pct"]
    return value, details


def describe_change(tf: str, value: float, d: dict) -> str:
    text = (
        f"{tf.upper()} change {d['pct']:+.2f}%  "
        f"({d['base_date']} close {d['base_price']:.6g} -> {d['end_date']} close {d['end_price']:.6g}, "
        f"{d['trading_days']} trading day{'s' if d['trading_days'] != 1 else ''})"
    )
    if SCALE_PERIOD_CHANGE_TO_DAILY and d["trading_days"] > 1:
        text += f"  ->  daily-equivalent {value:+.4f}% used in the volatility formula"
    else:
        text += f"  ->  {value:+.4f}% used in the volatility formula"
    return text


def run_market_change(tf: str, sim_start: date, sim_end: date | None):
    """Market change for one run (point-in-time for backtests, else up to today)."""
    if backtest and BACKTEST_POINT_IN_TIME_CHANGE:
        span = (sim_end - sim_start).days if (tf == "ytd" and sim_end) else None
        try:
            return market_change_input(tf, as_of=sim_start, span_days=span)
        except ValueError as error:
            print(f"  Warning: no point-in-time {tf.upper()} change before {sim_start} ({error}) "
                  f"- using the change measured up to today instead.")
    try:
        return market_change_input(tf)
    except Exception as error:
        raise SystemExit(
            f"Could not get Yahoo Finance {tf.upper()} change for {ticker_name} ({chart_symbol}): {error}"
        )


def vol_from_market_change(change_pct: float) -> float:
    """The original sigmoid: daily-equivalent % change -> vol_index (x8 compatibility scaling)."""
    formula_exponent = -(2.98228 * change_pct - 5.4098)
    if formula_exponent > 709:
        vol = 0.0
    elif formula_exponent < -709:
        vol = 99.81711
    else:
        vol = 99.81711 / (1 + math.exp(formula_exponent))
    return vol * 8


try:
    market_change_pct, change_info = market_change_input(timeframe)
    print(f"Yahoo {describe_change(timeframe, market_change_pct, change_info)}")
except ValueError as error:
    # Not enough history for this span. A backtest may use a different timeframe
    # or earlier windows, so only stop later if the value is actually needed.
    print(f"Warning: could not measure the {timeframe.upper()} change for {ticker_name} up to today: {error}")
except Exception as error:  # network / Yahoo failure: nothing can run
    raise SystemExit(
        f"Could not get Yahoo Finance {timeframe.upper()} change for {ticker_name} ({chart_symbol}): {error}"
    )
# Note: vol_index is computed per run (below) from the timeframe's % change.
# The Yahoo history is downloaded once and cached, so there are no repeated
# network calls.

tut("price_mode")
price_mode = input(
    "Price behaviour: [1] allow negative futures prices or [2] floor prices at zero? "
).strip().lower()
if price_mode in ("1", "futures"):
    allow_negative = True
elif price_mode in ("2", "floor"):
    allow_negative = False
else:
    print("Choose 1 (allow negative) or 2 (floor at zero).")
    exit()


# ─── Genetic algorithm over timeframe + candle interval ───────────────────────
def ga_timeframe_interval(n_cycles: int = 3, population: int = 8, generations: int = 3, seed: int = 0):
    """Search (timeframe, interval) pairs for the best-tracking simulation.

    Fitness = composite F over the `n_cycles` most recent windows, simulated at the range-matched vol
    (sigma_real / 0.2309) so the comparison is not dominated by a badly chosen volatility.
    Genes are discrete: crossover swaps timeframe/interval between parents, mutation changes the
    interval inside a timeframe or jumps to another timeframe's default.
    """
    rng_ga = np.random.default_rng(seed + ENGINE_SEED)
    pairs = [(tf_, iv_) for tf_, conf in TIMEFRAME_CONFIG.items() for iv_ in conf["intervals"]
             if bars_for(tf_, iv_, today) >= MIN_CANDLES_ANALYSIS]
    history_cache: dict = {}

    def cached_fetcher(symbol, interval_, start_, end_):
        key = (symbol, interval_, str(start_.date()), str(end_.date()))
        if key not in history_cache:
            history_cache[key] = bc._fetch_history(symbol, interval_, start_, end_)
        return history_cache[key]

    fitness_cache: dict = {}
    params_ga = bc.params_from_config(CFG, 3.6, SESSION_START_HOUR, SESSION_MINUTES)

    def fitness(pair):
        if pair in fitness_cache:
            return fitness_cache[pair]
        tf_, iv_ = pair
        spec_ = interval_spec(tf_, iv_)
        ctxs = []
        for x in range(1, n_cycles + 1):
            try:
                s_, e_ = period_start_for_cycle(tf_, x, today)
                ts_ = pd.DatetimeIndex(build_timestamps(tf_, s_, e_, iv_))
                real_, _, _ = bc.fetch_real_for_cycle(chart_symbol, spec_, ts_, today, cached_fetcher)
            except Exception:
                continue
            if real_ is not None:
                ctxs.append(bc.CycleContext(cycle=x, start=s_, timestamps=ts_, real=real_,
                                            intraday=(spec_["kind"] == "intraday"), start_price=starting_price))
        ctxs = [c for c in ctxs if c.scoreable]
        score = -9.0
        if len(ctxs) >= 2:
            sigma_ = bc.real_sigma_pct(ctxs)
            if np.isfinite(sigma_) and sigma_ > 0:
                stats_ = [bc.evaluate_cycle(c, params_ga.with_(vol_index=sigma_ / 0.2309), CFG, 6,
                                            bc.cycle_seed(ENGINE_SEED, c.cycle, 77), False, None, allow_negative)
                          for c in ctxs]
                stats_ = [s for s in stats_ if s]
                if stats_:
                    score = float(np.mean([s["F"] for s in stats_]))
        fitness_cache[pair] = score
        print(f"    {tf_:<4} {iv_:<4} {bars_for(tf_, iv_, today):>4} candles   F = {score:+.3f}"
              + ("" if score > -9 else "   (no real data at this candle size)"))
        return score

    def mutate(pair):
        tf_, iv_ = pair
        if rng_ga.random() < 0.6:
            options = [p for p in pairs if p[0] == tf_ and p != pair]
            if options:
                return options[int(rng_ga.integers(len(options)))]
        options = [p for p in pairs if p[0] != tf_ and p[1] == TIMEFRAME_CONFIG[p[0]]["default_interval"]]
        return options[int(rng_ga.integers(len(options)))] if options else pair

    pop = [pairs[i] for i in rng_ga.choice(len(pairs), size=min(population, len(pairs)), replace=False)]
    print(f"\nGenetic search over {len(pairs)} timeframe/interval pairs (population {len(pop)}, {generations} generations):")
    for gen in range(1, generations + 1):
        print(f"  generation {gen}:")
        ranked = sorted(pop, key=fitness, reverse=True)
        elite = ranked[: max(2, len(ranked) // 2)]
        children = []
        while len(children) < len(pop) - len(elite):
            a, b = (elite[int(rng_ga.integers(len(elite)))] for _ in range(2))
            child = (a[0], b[1]) if (a[0], b[1]) in pairs else a          # crossover
            if rng_ga.random() < 0.5:
                child = mutate(child)
            children.append(child)
        pop = elite + children
    best = max(fitness_cache, key=fitness_cache.get)
    print(f"  Best: {best[0]} / {best[1]} (F = {fitness_cache[best]:+.3f})")
    if fitness_cache[best] <= -9:
        return None
    return best


# ─── Backtest OR forward simulation ──────────────────────────────────────────
tut("backtest")
backtest = ask("\nEnable backtesting? (y/n) [n]: ").strip().lower() in ("y", "yes")
self_correct = False
sc_method = VOLATILITY_SEARCH["method"]
sc_loops = int(VOLATILITY_SEARCH["loops"])
SEARCH_METHODS = ["auto", "step", "bisect", "grid", "coarse_fine", "cobyla", "bayes", "ga", "mc"]

if backtest:
    print(f"\nBacktest timeframes: {', '.join(VALID_TIMEFRAMES)}")
    ga_choice = ask("Let a genetic algorithm search the best timeframe + candle interval? (y/n) [n]: ",
                    "ga_timeframe").strip().lower() in ("y", "yes")
    picked = ga_timeframe_interval() if ga_choice else None
    if picked:
        tf_bt, backtest_interval = picked
        print(f"  Using the GA's choice: {tf_bt} / {backtest_interval}")
    else:
        if ga_choice:
            print("  The search found no usable pair (no real data); choose manually.")
        tf_bt_raw = input(f"Backtest timeframe [{timeframe}]: ").strip() or timeframe
        tf_bt     = normalise_timeframe(tf_bt_raw)
        while tf_bt is None:
            print(f"  Invalid. Choose from: {', '.join(VALID_TIMEFRAMES)}")
            tf_bt_raw = input(f"Backtest timeframe [{timeframe}]: ").strip() or timeframe
            tf_bt     = normalise_timeframe(tf_bt_raw)

        backtest_interval = choose_interval(tf_bt)
    print(
        f"\n  Cycle 1 = most recent {tf_bt} period ending {today}.\n"
        f"  Cycle N = oldest.  All cycles use {bars_for(tf_bt, backtest_interval, today)} candles "
        f"× {interval_label(backtest_interval)}."
    )
    try:
        cycles = int(ask("Number of backtest cycles: ", "cycles").strip())
    except ValueError:
        print("Enter a whole number ≥ 1.")
        exit()
    if cycles < 1:
        print("Cycles must be ≥ 1.")
        exit()

    run_count   = cycles
    sim_windows: list[tuple[date, date | None]] = []
    # Build oldest→newest so run 1 = oldest cycle
    for x in range(cycles, 0, -1):
        s, e = period_start_for_cycle(tf_bt, x, today)
        sim_windows.append((s, e))

    print(f"\n  Backtest schedule ({cycles} cycles, oldest first):")
    for i, (s, e) in enumerate(sim_windows, start=1):
        end_str = str(e) if e else f"+ {bars_for(tf_bt, backtest_interval, today)} bars"
        print(f"    Cycle {i}: {s}  →  {end_str}")

    backtest_timeframe = tf_bt

    if VOLATILITY_SEARCH["enabled"]:
        raw = ask("\nSelf-correct the volatility against the real prices? (y/n) [y]: ", "self_correct").strip().lower()
        self_correct = raw in ("", "y", "yes")
        if self_correct:
            print(f"  Search methods: {', '.join(SEARCH_METHODS)}")
            raw_method = ask(f"  Search method [{sc_method}]: ", "method").strip().lower() or sc_method
            sc_method = raw_method if raw_method in SEARCH_METHODS else sc_method
            try:
                sc_loops = max(1, int(ask(f"  Learning loops [{sc_loops}]: ", "loops").strip() or sc_loops))
            except ValueError:
                print(f"  Not a number; keeping {sc_loops}.")

else:
    backtest_timeframe = timeframe
    backtest_interval  = choose_interval(backtest_timeframe)
    sim_windows = []
    plot_mode   = ask("\nCreate [1] one plot or [M] many plots? ", "plot_mode").strip().lower()
    if plot_mode in ("1", "one", "single"):
        run_count = 1
    elif plot_mode in ("m", "many"):
        try:
            run_count = int(input("How many plots? "))
        except ValueError:
            print("Enter a whole number greater than zero.")
            exit()
        if run_count < 1:
            print("Enter a whole number greater than zero.")
            exit()
    else:
        print("Choose 1 for one plot or M for many plots.")
        exit()

# ─── Output mode (applies to both paths when run_count ≥ 100) ─────────────────
render_analysis_charts = True
render_source_charts   = True
run_ml = False

if backtest:
    # Backtests produce ONE combined plot (backtest_plot.png) instead of
    # per-cycle source/analysis charts and contact sheets.
    render_analysis_charts = False
    render_source_charts   = False

if run_count >= 100 and not backtest:
    label = f"{'backtest cycle' if backtest else 'run'}"
    print(f"\nChoose outputs for {run_count:,} {label}s:")
    print("  Source PNGs: the original OHLC chart for each run.")
    print("  Analysis PNGs: the indicator and trade chart for each run.")
    print("  Per-run CSVs and combined analysis metrics are always saved.")
    print("  1) Source + analysis charts")
    print("  2) Source + analysis charts + ML (longest processing time)")
    print("  3) Analysis charts + ML (no source charts)")
    print("  4) Analysis charts only (no source charts or ML)")
    print("  5) ML only (no charts; required analysis metrics are still calculated)")
    print("  6) CSVs + metrics only (no charts or ML; fastest) [default]")
    output_choice = ask("Choose 1-6 [6]: ", "output_mode").strip() or "6"
    output_modes = {
        "1": (True,  True,  False),
        "2": (True,  True,  True),
        "3": (False, True,  True),
        "4": (False, True,  False),
        "5": (False, False, True),
        "6": (False, False, False),
    }
    if output_choice not in output_modes:
        print("Choose one of the output modes 1-6.")
        exit()
    render_source_charts, render_analysis_charts, run_ml = output_modes[output_choice]

clear_generated_files(results_dir,  ("data_*.csv", "image_*.png"))
clear_generated_files(analysis_dir, ("*.csv", "*.png"))      # the learned-vol memory files are kept (PERSISTENT_FILES)

# ─── Generator, learned-volatility memory, empirical shocks ───────────────────
memory = bc.VolMemory(analysis_dir)
base_params = bc.params_from_config(CFG, vol_index=3.6, session_start=SESSION_START_HOUR,
                                    session_minutes=SESSION_MINUTES)
_spec = interval_spec(backtest_timeframe, backtest_interval)
IS_INTRADAY = _spec["kind"] == "intraday"
SEARCH_META = {"instrument": ticker_name, "asset_type": asset_type, "timeframe": backtest_timeframe,
               "interval": backtest_interval}
if base_params.streak_on:
    print(f"Streak model: P(another bar the same way) = {base_params.streak_c1:.3f} x {base_params.streak_decay:.4f}"
          f"^(run//{base_params.streak_every}); P(10 in a row) = "
          f"{bc.streak_run_probability(base_params.streak_c1, base_params.streak_decay, base_params.streak_every):.4f}")

EMPIRICAL = None
if base_params.dist in ("empirical", "gbm_quantile"):
    try:
        from price_change import _daily_closes
        daily_returns = 100.0 * _daily_closes(chart_symbol, asset_type).pct_change().dropna().tail(1500)
        EMPIRICAL = bc.empirical_from_returns(daily_returns.to_numpy())
        print(f"Empirical shock pool: {0 if EMPIRICAL is None else len(EMPIRICAL['z'])} real daily returns.")
    except Exception as error:
        print(f"Could not build the empirical pool ({error}); falling back to uniform shocks.")
        base_params = base_params.with_(dist="uniform")

# ─── Periodic re-calibration (forward runs; --auto-recal or [volatility_search] auto_recalibrate) ──
RECAL_STATE_PATH = Path(analysis_dir) / "recal_state.json"
RECAL_KEY = f"{ticker_name}|{backtest_timeframe}|{backtest_interval}"


def _load_recal_state() -> dict:
    try:
        return json.loads(RECAL_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def bump_recal_counter(new_runs: int) -> None:
    """Count forward runs since the last calibration (persisted in recal_state.json)."""
    state = _load_recal_state()
    entry = state.setdefault(RECAL_KEY, {"runs_since": 0})
    entry["runs_since"] = int(entry.get("runs_since", 0)) + int(new_runs)
    RECAL_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def build_recal_contexts(n_cycles: int) -> list:
    """The n most recent windows with real data, as search contexts (oldest first)."""
    contexts = []
    for x in range(n_cycles, 0, -1):
        try:
            s_, e_ = period_start_for_cycle(backtest_timeframe, x, today)
            ts_ = pd.DatetimeIndex(build_timestamps(backtest_timeframe, s_, e_, backtest_interval))
            real_, _, _ = bc.fetch_real_for_cycle(chart_symbol, _spec, ts_, today)
            span_ = (e_ - s_).days if (backtest_timeframe == "ytd" and e_) else None
            change_, _ = market_change_input(backtest_timeframe, as_of=s_, span_days=span_)
        except Exception as error:
            print(f"  recalibration: window {x} skipped ({type(error).__name__}: {error})")
            continue
        try:
            regime_ = get_regime(chart_symbol, asset_type, as_of=s_)["regime"]
        except Exception:
            regime_ = "unknown"
        contexts.append(bc.CycleContext(cycle=n_cycles - x + 1, start=s_, timestamps=ts_, real=real_,
                                        intraday=IS_INTRADAY, start_price=starting_price, regime=regime_,
                                        base_vol=vol_from_market_change(change_)))
    return contexts


def maybe_recalibrate() -> None:
    """Re-run a light volatility search when the stored optimum is stale or N new runs have accumulated."""
    if backtest or not (ARGS.auto_recal or VOLATILITY_SEARCH["auto_recalibrate"]):
        return
    since = int(_load_recal_state().get(RECAL_KEY, {}).get("runs_since", 0))
    stale = bc.recalibration_due(memory, SEARCH_META, VOLATILITY_SEARCH["recal_days"])
    if not (stale or since >= VOLATILITY_SEARCH["recal_every_runs"]):
        print(f"Recalibration not due ({since} runs since the last one, optimum younger than "
              f"{VOLATILITY_SEARCH['recal_days']} days).")
        return
    print(f"\nRe-calibrating the volatility ({'no / stale optimum' if stale else f'{since} runs since the last calibration'})...")
    contexts = build_recal_contexts(8)
    result = bc.run_self_correction(contexts, base_params, CFG, SEARCH_META, analysis_dir, ENGINE_SEED,
                                    method="step", loops=1, empirical=EMPIRICAL, allow_negative=allow_negative,
                                    lightweight=True)
    if result.get("status") == "ok":
        state = _load_recal_state()
        state[RECAL_KEY] = {"runs_since": 0, "last_recalibration": str(pd.Timestamp.now())}
        RECAL_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(result["line"])
    else:
        print(f"Recalibration skipped: {result.get('status')}")


maybe_recalibrate()

# Forward runs can start from the volatility a previous self-correction learned (absolute vol_index: the
# market-change formula swings from ~0.1 to ~60 on small changes, so a ratio learned elsewhere would not transfer).
learned_vol_fixed = None
if not backtest:
    _row = memory.global_row(ticker_name, asset_type, backtest_timeframe, backtest_interval)
    if _row is not None and np.isfinite(pd.to_numeric(pd.Series([_row.get("learned_vol")]), errors="coerce").iloc[0]):
        _learned = float(_row["learned_vol"])
        tut("learned_vol")
        _use = input(f"A learned volatility exists for this market (vol_index {_learned:.3f}, "
                     f"saved {str(_row.get('updated'))[:10]}). Use it instead of the market-change formula? (y/n) [y]: ").strip().lower()
        if _use in ("", "y", "yes"):
            learned_vol_fixed = _learned

# ─── Bar generation ───────────────────────────────────────────────────────────
show_run_details   = run_count < 100
progress_interval  = max(1, run_count // 100)


def generate_daily(
    timestamps: list[pd.Timestamp],
    vol_index: float,
    verbose: bool = False,
    params: "bc.GenParams | None" = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate one OHLC(V) run over the provided timestamp sequence (engine: backtest_compare.generate_ohlc).

    vol_index is computed once by the caller (market_change_pct -> formula, or the learned value) and
    passed in. `seed` makes the run reproducible; params carry the tunable shape (wicks, jumps, AR(1),
    streak decay, volume ...).
    """
    p = (params or base_params).with_(vol_index=float(vol_index))
    rng = np.random.default_rng(ENGINE_SEED if seed is None else seed)
    daily = bc.generate_ohlc(timestamps, starting_price, p, rng, allow_negative, IS_INTRADAY, EMPIRICAL)
    if verbose:
        print("\nCandlestick table:")
        print(daily.head(12).to_string())
    return daily


black_background_style = mpf.make_mpf_style(
    base_mpf_style="charles",
    figcolor="black",
    facecolor="black",
    y_on_right=True,
    rc={
        "axes.labelcolor":  "white",
        "axes.titlecolor":  "white",
        "text.color":       "white",
        "xtick.color":      "white",
        "ytick.color":      "white",
    },
)

# Per-run bookkeeping shared by the generation, analysis, self-correction and ML steps.
run_vols: dict[int, float] = {}
run_params: dict[int, "bc.GenParams"] = {}
run_timestamps: dict[int, pd.DatetimeIndex] = {}
run_windows: dict[int, tuple] = {}
run_augmented: dict[int, bool] = {}
run_ensemble_label: dict[int, float] = {}
run_daily: dict[int, pd.DataFrame] = {}


def write_run_outputs(run_number: int, daily: pd.DataFrame, verbose: bool) -> None:
    """data_N.csv (+ image_N.png when source charts are on) and the row-count validation."""
    csv_file   = os.path.join(results_dir, f"data_{run_number}.csv")
    image_file = os.path.join(results_dir, f"image_{run_number}.png")

    lowest_low    = daily["Low"].min()
    open_01       = daily["Open"].iloc[0]
    close_01      = daily["Close"].iloc[-1]
    highest_high  = daily["High"].max()

    if lowest_low <= 0:
        low_to_high_label = "Low to high: N/A"
    else:
        percent_change    = abs((highest_high - lowest_low) / lowest_low * 100)
        low_to_high_label = f"Low to high: {percent_change:.2f}%"
    market_change = (close_01 - open_01) / open_01 * 100

    if render_source_charts:
        figure, axes = mpf.plot(
            daily,
            type="candle",
            style=black_background_style,
            volume="Volume" in daily.columns,
            figsize=(12, 6),
            ylabel="Price",
            title=f"{ticker_name} — {backtest_timeframe.upper()} simulation ({interval_label(backtest_interval)} candles)",
            returnfig=True,
        )
        axes[0].legend(
            handles=[
                Line2D([], [], linestyle="None", color="none"),
                Line2D([], [], linestyle="None", color="none"),
            ],
            labels=[
                low_to_high_label,
                f"Period change: {market_change:.2f}%",
            ],
            loc="upper left",
            handlelength=0,
            handletextpad=0,
        )
        figure.savefig(image_file, facecolor=figure.get_facecolor())
        plt.close(figure)

    daily.to_csv(csv_file)

    # Validate generated CSV row count matches timeframe expectation. ytd is date-dependent, so its
    # expected row count is the number of timestamps generated.
    expected = len(run_timestamps[run_number]) if backtest_timeframe == "ytd" else bars_for(backtest_timeframe, backtest_interval, today)
    df_check = pd.read_csv(csv_file, parse_dates=[0])
    if len(df_check) != expected:
        raise RuntimeError(
            f"Generated CSV rows ({len(df_check)}) != expected for timeframe '{backtest_timeframe}' at {backtest_interval} candles ({expected}) - file: {csv_file}"
        )

    if verbose:
        if render_source_charts:
            print(f"\nSaved chart to : {image_file}")
        print(f"Saved data to  : {csv_file}")


def make_run(run_number: int, vol: float, params: "bc.GenParams", seed: int, verbose: bool = False,
             augmented: bool = False) -> pd.DataFrame:
    daily = generate_daily(run_timestamps[run_number], vol, verbose, params, seed)
    run_vols[run_number], run_params[run_number], run_augmented[run_number] = float(vol), params, augmented
    run_daily[run_number] = daily
    write_run_outputs(run_number, daily, verbose)
    return daily

# ─── Run loop ─────────────────────────────────────────────────────────────────
daily = pd.DataFrame()
backtest_cycles: list[dict] = []
aug_base: dict[int, int] = {}
for run_number in range(1, run_count + 1):

    # Determine simulation window for this run
    if backtest and sim_windows:
        sim_start, sim_end = sim_windows[run_number - 1]
        cycle_label = f" [cycle {run_number}/{run_count}: {sim_start}]"
    else:
        sim_start   = today      # real-time date when not backtesting
        sim_end     = None
        cycle_label = ""

    if show_run_details:
        verb = "backtest cycle" if backtest else "run"
        print(f"\nGenerating {verb} {run_number} of {run_count}{cycle_label}...")
    elif (
        run_number == 1
        or run_number % progress_interval == 0
        or run_number == run_count
    ):
        print(f"Generating run {run_number:,} of {run_count:,}{cycle_label}...")

    timestamps = build_timestamps(backtest_timeframe, sim_start, sim_end, backtest_interval)
    run_timestamps[run_number], run_windows[run_number] = pd.DatetimeIndex(timestamps), (sim_start, sim_end)
    # Compute vol_index once per run and pass it to the generator to avoid
    # repeated calls and to make behavior deterministic for the whole run.
    market_change_pct, change_info = run_market_change(backtest_timeframe, sim_start, sim_end)
    if show_run_details and backtest:
        print(f"  Market change: {describe_change(backtest_timeframe, market_change_pct, change_info)}")
    run_vol_index = vol_from_market_change(market_change_pct)
    if learned_vol_fixed is not None:
        run_vol_index = learned_vol_fixed
    daily = make_run(run_number, run_vol_index, base_params, bc.cycle_seed(ENGINE_SEED, run_number, 0),
                     verbose=show_run_details)
    if backtest:
        backtest_cycles.append(
            {"cycle": run_number, "sim": daily.copy(), "start": sim_start, "end": sim_end, "vol_index": run_vol_index}
        )

# ─── Analysis ─────────────────────────────────────────────────────────────────
OHLC = ["Open", "High", "Low", "Close"]


def run_analysis() -> dict:
    """market_analysis over every data_N.csv, then the metadata columns every downstream step expects."""
    result = analyze_looped_results(
        results_dir,
        analysis_dir,
        render_charts=render_analysis_charts,
        render_source_charts=render_source_charts,
        timeframe=backtest_timeframe,
        chart_label=f"{chart_symbol} ({interval_label(backtest_interval)})",
        vol_by_run=dict(run_vols),
    )
    summary_path = Path(analysis_dir) / "analysis_summary.csv"
    table = pd.read_csv(summary_path)
    table.insert(1, "instrument",  ticker_name)
    table.insert(2, "asset_type",  asset_type)
    table.insert(4, "sim_date",    today.isoformat())
    table.insert(5, "is_backtest", int(backtest))
    table["is_augmented"] = table["run"].map(lambda r: int(run_augmented.get(int(r), False)))
    table["sample_weight"] = np.where(table["is_augmented"] == 1, VOLATILITY_SEARCH["aug_weight"], 1.0)
    if run_ensemble_label:
        table["ensemble_label"] = table["run"].map(lambda r: run_ensemble_label.get(int(r), np.nan))
    table.to_csv(summary_path, index=False)
    return result


_expected_cache: dict = {}


def expected_sigma_for_run(run: int, vol: float) -> float:
    key = (round(vol, 6), run_params[run].streak_c1, run_params[run].jump_prob, run_params[run].ar1, run_params[run].dist)
    if key not in _expected_cache:
        _expected_cache[key] = bc.expected_bar_sigma_pct(run_params[run].with_(vol_index=vol), IS_INTRADAY, empirical=EMPIRICAL)
    return _expected_cache[key]


def realised_vol_step(allow_fix: bool) -> None:
    """Compare realised vol with what each run's vol_index should give; optionally re-generate the outliers."""
    summary_path = Path(analysis_dir) / "analysis_summary.csv"
    feedback = realised_vol_feedback(pd.read_csv(summary_path), dict(run_vols), expected_sigma_for_run,
                                     VOLATILITY_SEARCH["realised_vol_tolerance"])
    feedback.to_csv(Path(analysis_dir) / "realised_vol_feedback.csv", index=False)
    flagged = feedback[feedback["flagged"]] if len(feedback) else feedback
    print(f"Realised-vol feedback: {len(flagged)} of {len(feedback)} runs differ by more than "
          f"{VOLATILITY_SEARCH['realised_vol_tolerance']:.0%} from the volatility their vol_index implies "
          f"(realised_vol_feedback.csv).")
    if len(flagged) and allow_fix:
        for _, row in flagged.iterrows():
            run = int(row["run"])
            new_vol = float(np.clip(row["suggested_vol_index"], run_vols[run] * 0.25, run_vols[run] * 4.0))
            daily_fixed = make_run(run, new_vol, run_params[run], bc.cycle_seed(ENGINE_SEED, run, 2),
                                   augmented=run_augmented.get(run, False))
            if backtest and run <= run_count:
                backtest_cycles[run - 1].update({"sim": daily_fixed.copy(), "vol_index": new_vol})
        print(f"  Re-generated {len(flagged)} flagged run(s) once with the suggested vol_index; re-analysing...")
        run_analysis()


analysis_result = run_analysis()
print(
    f"\nAnalysis complete: {analysis_result['runs']} pairs, "
    f"{analysis_result['bars']} bars, {analysis_result['trades']} crossover trades."
)
realised_vol_step(ARGS.auto_correct_vol)


def merge_scores(items: list[dict]) -> None:
    """Add accuracy_index / bar_direction_match_pct / net_price_error_pct / exact_score to analysis_summary.csv."""
    summary_path = Path(analysis_dir) / "analysis_summary.csv"
    table = pd.read_csv(summary_path)
    rows: dict[int, dict] = {}
    real_by_cycle = {it["cycle"]: it.get("real") for it in items}
    for it in items:
        sc = it.get("score")
        if sc and sc.get("status") == "scored":
            rows[int(it["cycle"])] = {"accuracy_index": sc["index"], "bar_direction_match_pct": 100.0 * sc["direction_match_pct"],
                                      "net_price_error_pct": sc["net_price_error_pct"], "exact_score": sc["exact_score"]}
    for run, is_aug in run_augmented.items():           # augmented runs are scored against their base cycle's real bars
        real = real_by_cycle.get(aug_base.get(run))
        if is_aug and real is not None:
            sc = bc.score_cycle(run_daily[run][OHLC], real)
            if sc.get("status") == "scored":
                rows[int(run)] = {"accuracy_index": sc["index"], "bar_direction_match_pct": 100.0 * sc["direction_match_pct"],
                                  "net_price_error_pct": sc["net_price_error_pct"], "exact_score": sc["exact_score"]}
    for column in ("accuracy_index", "bar_direction_match_pct", "net_price_error_pct", "exact_score"):
        table[column] = table["run"].map(lambda r, c=column: rows.get(int(r), {}).get(c, np.nan))
    table.to_csv(summary_path, index=False)


def build_contexts(items: list[dict], base_vols: dict[int, float]) -> list:
    contexts = []
    for it in items:
        regime = "unknown"
        if VOLATILITY_SEARCH["use_regimes"]:
            try:
                regime = get_regime(chart_symbol, asset_type, as_of=it["start"])["regime"]
            except Exception:
                pass
        contexts.append(bc.CycleContext(cycle=it["cycle"], start=it["start"], timestamps=it["sim"].index,
                                        real=it.get("real"), intraday=IS_INTRADAY, start_price=starting_price,
                                        regime=regime, base_vol=base_vols[it["cycle"]]))
    return contexts


def build_validation(items: list[dict]) -> dict:
    """Rebased real vs simulated frames (adversarial validation) and the search's (vol, F) points (conformal)."""
    sims, reals = [], []
    for it in items:
        if it.get("real") is None:
            continue
        paired = bc._paired_rebased(it["sim"], it["real"])
        if paired is not None:
            _, s_frame, r_frame = paired
            sims.append(s_frame.reset_index(drop=True))
            reals.append(r_frame.reset_index(drop=True))
    audit = memory._read(memory.AUDIT)
    points = pd.DataFrame()
    if len(audit) and {"master_seed", "scope", "vol_index", "composite_F"}.issubset(audit.columns):
        mine = audit[(audit["master_seed"] == ENGINE_SEED) & (audit["scope"] == "global")]
        points = mine[["vol_index", "composite_F"]].dropna()
    return {"real_frames": reals, "sim_frames": sims, "vol_points": points}


# ─── Backtest: real price ghost halo + accuracy index (one plot) ──────────────
final_items = None
if backtest:
    bar_spec = interval_spec(backtest_timeframe, backtest_interval)
    bt = run_backtest_comparison(
        backtest_cycles,
        symbol     = chart_symbol,
        timeframe  = backtest_timeframe,
        output_dir = analysis_dir,
        today      = today,
        bar        = bar_spec,
        interval   = backtest_interval,
    )
    merge_scores(bt["items"])
    final_items = bt["items"]

    # Automatic re-run trigger: has the accuracy decayed versus the stored optimum?
    if VOLATILITY_SEARCH["enabled"] and not self_correct and bc.accuracy_decayed(
            memory, SEARCH_META, bt["overall_index"], ACCURACY_DECAY_THRESHOLD):
        print(f"\nAccuracy index {bt['overall_index']:+.3f} is more than {ACCURACY_DECAY_THRESHOLD} below the stored "
              f"optimum for this market: starting the volatility search automatically.")
        self_correct = True

    if self_correct:
        analysis_path = Path(analysis_dir)
        for name in ("backtest_plot.png", "backtest_accuracy.csv", "rewind_accuracy_summary.csv", "backtest_real_prices.csv"):
            source = analysis_path / name
            if source.exists():
                os.replace(source, analysis_path / name.replace(".", "_null.", 1))      # keep the baseline pass
        null_vols = dict(run_vols)
        contexts = build_contexts(bt["items"], null_vols)
        print(f"\nSelf-correcting volatility ({sc_method}, {sc_loops} loop(s)) against "
              f"{sum(c.scoreable for c in contexts)} of {len(contexts)} cycles with real data...")
        sc = bc.run_self_correction(contexts, base_params, CFG, SEARCH_META, analysis_dir, ENGINE_SEED,
                                    method=sc_method, loops=sc_loops, empirical=EMPIRICAL,
                                    allow_negative=allow_negative)
        if sc.get("status") == "ok":
            # Make the certificate self-sufficient for reproduction: the user-facing seed to pass back.
            cert_file = Path(sc["certificate"])
            certificate = json.loads(cert_file.read_text(encoding="utf-8"))
            certificate["run_seed"] = RUN_SEED
            certificate["reproduce_with"] = f"python price_modeling.py --seed {RUN_SEED}"
            cert_file.write_text(json.dumps(certificate, indent=2), encoding="utf-8")
        if sc.get("status") != "ok":
            print(f"Self-correction skipped: {sc.get('status')}. Keeping the baseline pass.")
            for name in ("backtest_plot", "backtest_accuracy", "rewind_accuracy_summary", "backtest_real_prices"):
                ext = ".png" if name == "backtest_plot" else ".csv"
                if (analysis_path / f"{name}_null{ext}").exists():
                    os.replace(analysis_path / f"{name}_null{ext}", analysis_path / f"{name}{ext}")
        else:
            by_cycle = {c.cycle: c for c in contexts}
            clear_generated_files(results_dir, ("data_*.csv", "image_*.png"))
            for stale in [r for r in run_vols if r > run_count]:
                for registry in (run_vols, run_params, run_timestamps, run_windows, run_augmented, run_daily):
                    registry.pop(stale, None)
            run_ensemble_label.clear()
            new_cycles = []
            for run_number in range(1, run_count + 1):
                ctx = by_cycle.get(run_number)
                vol = sc["per_cycle_vol"].get(run_number) or sc["vol"] * (ctx.scale if ctx else 1.0)
                daily = make_run(run_number, vol, sc["params"], bc.cycle_seed(ENGINE_SEED, run_number, 1))
                new_cycles.append({"cycle": run_number, "sim": daily.copy(), "start": sim_windows[run_number - 1][0],
                                   "end": sim_windows[run_number - 1][1], "vol_index": vol})
            k_ens = int(VOLATILITY_SEARCH["ensemble_k"])
            if k_ens > 1:
                agree = []
                for ctx in contexts:
                    vote = bc.ensemble_vote(ctx, sc["params"].with_(vol_index=run_vols[ctx.cycle]), k_ens,
                                            bc.cycle_seed(ENGINE_SEED, ctx.cycle, 5), EMPIRICAL, allow_negative)
                    if vote:
                        run_ensemble_label[ctx.cycle] = vote["ensemble_net_up"]
                        agree.append((vote["ensemble_direction_match"], vote["mean_path_direction_match"]))
                if agree:
                    print(f"Ensemble vote (K={k_ens}): majority-direction match {100 * np.mean([a[0] for a in agree]):.1f}% "
                          f"vs single path {100 * np.mean([a[1] for a in agree]):.1f}%  (ensemble_label feeds the ML target)")
            n_aug = int(VOLATILITY_SEARCH["n_aug"])
            if n_aug > 0:
                for i, item in enumerate(bc.plan_augmentation(sc, contexts, n_aug, ENGINE_SEED)):
                    aug_run = run_count + 1 + i
                    run_timestamps[aug_run], run_windows[aug_run] = run_timestamps[item["cycle"]], run_windows[item["cycle"]]
                    aug_base[aug_run] = item["cycle"]
                    make_run(aug_run, item["vol_index"], sc["params"], item["seed"], augmented=True)
                print(f"Added {n_aug} augmentation run(s) at the learned vol (sample weight x{VOLATILITY_SEARCH['aug_weight']}).")
            analysis_result = run_analysis()
            realised_vol_step(ARGS.auto_correct_vol)
            prefetched = {it["cycle"]: (it.get("real"), it.get("interval", ""), it.get("note", "")) for it in bt["items"]}
            bt2 = run_backtest_comparison(new_cycles, symbol=chart_symbol, timeframe=backtest_timeframe,
                                          output_dir=analysis_dir, today=today, bar=bar_spec,
                                          interval=backtest_interval, prefetched=prefetched)
            merge_scores(bt2["items"])
            final_items = bt2["items"]

            print("\n══ Self-correction summary ══════════════════════════════════════")
            print(sc["line"])
            print(f"  accuracy index    baseline {bt['overall_index']:+.3f}  ->  self-corrected {bt2['overall_index']:+.3f}")
            print(f"  direction match   baseline {100 * bt['overall_direction_match_pct']:.1f}%  ->  "
                  f"self-corrected {100 * bt2['overall_direction_match_pct']:.1f}%   (50% = coin flip)")
            print(f"  composite F       null {sc['null']['F']:+.3f}  ->  {sc['final']['F']:+.3f} (global vol, fresh seeds)   "
                  f"improvement {sc['improvement_over_null']:+.0%}")
            if sc["test_cycles"]:
                print(f"  out-of-sample     newest {len(sc['test_cycles'])} cycle(s) never used to calibrate: "
                      f"F {sc['null_oos'].get('F', float('nan')):+.3f} -> {sc['oos'].get('F', float('nan')):+.3f} "
                      f"(improvement {sc['oos_improvement']:+.0%})")
            print(f"  vol {100 * VOLATILITY_SEARCH['ci_level']:.0f}% CI      [{sc['ci'][0]:.3f}, {sc['ci'][1]:.3f}]   "
                  f"consensus vols {', '.join(f'{v:.3f}' for v in sc['consensus_vols'])}")
            crit_text = ", ".join(k + "=" + ("met" if v else "not met") for k, v in sc["criteria"].items())
            print(f"  criteria          {crit_text}")
            print(f"  certificate       {sc['certificate']}  (seed {RUN_SEED})")
            if 100 * bt2["overall_direction_match_pct"] < 55:
                print("  note: a symmetric random generator cannot beat 50% direction match by tuning volatility;\n"
                      "        the per-cycle numbers above are in-sample for the learned vols - judge generalisation by the\n"
                      "        out-of-sample line.")
            print("═════════════════════════════════════════════════════════════════")

    # ML on the backtest cycles (uses vol_index, accuracy_index, direction match from this run)
    if run_count >= 10:
        ml_choice = ask("\nRun ML model on these backtest cycles? (y/n): ", "ml").strip().lower()
        if ml_choice in ("y", "yes"):
            from ml_model import run_ml_pipeline
            try:
                run_ml_pipeline(analysis_dir, instrument=ticker_name, asset_type=asset_type, timeframe=backtest_timeframe,
                                validation=build_validation(final_items or bt["items"]), config=CFG)
            except (ValueError, FileNotFoundError) as ml_error:
                print(f"[ML] Skipped: {ml_error}")

if not backtest:
# ─── ML option ────────────────────────────────────────────────────────────────
    if run_count < 100:
        ml_choice = ask("\nRun ML model on these results? (y/n): ", "ml").strip().lower()
        run_ml    = ml_choice in ("y", "yes")

    if run_ml:
        from ml_model import run_ml_pipeline
        try:
            run_ml_pipeline(
                analysis_dir,
                instrument  = ticker_name,
                asset_type  = asset_type,
                timeframe   = backtest_timeframe,
                config      = CFG,
            )
        except (ValueError, FileNotFoundError) as ml_error:
            print(f"[ML] Skipped: {ml_error}")

    bump_recal_counter(run_count)
    df = daily

# ─── Report, archive ──────────────────────────────────────────────────────────
try:
    from ml_model import write_html_report
    print(f"\nHTML report: {write_html_report(analysis_dir)}")
except Exception as error:
    print(f"HTML report skipped ({type(error).__name__}: {error})")


def archive_results() -> None:
    """Zip the per-run CSV/PNG files (about 10x smaller than loose files, far faster to sync or commit)."""
    storage = CFG["storage"]
    target = Path(storage["archive_dir"]) if storage["archive_dir"] else Path(master_dir)
    target.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in ticker_name)
    archive_path = target / f"looped_results_{safe_name}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.zip"
    members = [p for p in Path(results_dir).iterdir()
               if p.is_file() and (storage["keep_png"] or p.suffix.lower() != ".png")]
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in members:
            bundle.write(path, path.name)
    loose = sum(p.stat().st_size for p in members)
    print(f"Archived {len(members):,} files ({loose / 1e6:.1f} MB loose) -> {archive_path} "
          f"({archive_path.stat().st_size / 1e6:.1f} MB)")
    if storage["delete_loose_after_archive"]:
        for path in members:
            path.unlink()
        print("  loose files removed (set delete_loose_after_archive = false to keep them).")


if ARGS.archive or CFG["storage"]["archive_after_run"]:
    archive_results()