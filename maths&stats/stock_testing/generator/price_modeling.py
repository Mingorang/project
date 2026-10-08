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
asset_type = input(
    "Asset type [forex/futures/stock/crypto/other] (default forex): "
).strip().lower() or "forex"
ticker_name = input(
    "Instrument or pair to simulate (default GBP/USD): "
).strip() or "GBP/USD"

price_option = input(
    f"Starting price: [1] enter manually or [2] use current {ticker_name}? "
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

# ─── Directory setup ──────────────────────────────────────────────────────────
master_dir   = os.path.dirname(os.path.abspath(__file__))
results_dir  = os.path.join(master_dir, "looped_results")
analysis_dir = os.path.join(master_dir, "looped_analysis")

for folder, label in [(results_dir, "Output"), (analysis_dir, "Analysis output")]:
    if not os.path.isdir(folder):
        print(f"{label} folder does not exist: {folder}")
        print("Create the folder manually before generating data.")
        exit()


def clear_generated_files(directory, patterns):
    for pattern in patterns:
        for path in Path(directory).glob(pattern):
            if path.is_file():
                path.unlink()


# ─── Timeframe selection ──────────────────────────────────────────────────────
print("\nAvailable timeframes:")
for tf, cfg in TIMEFRAME_CONFIG.items():
    print(f"  {tf:<4} — {cfg['description']}")
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
rand.seed(rand.randint(-1_000_000, 1_000_000))

# Retired manual volatility prompt:
# question = float(input("\nVolatility of market: "))
try:
    from .price_change import get_timeframe_change_details
except ImportError:
    from price_change import get_timeframe_change_details

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

# ─── Backtest OR forward simulation ──────────────────────────────────────────
backtest = input("\nEnable backtesting? (y/n) [n]: ").strip().lower() in ("y", "yes")

if backtest:
    print(f"\nBacktest timeframes: {', '.join(VALID_TIMEFRAMES)}")
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
        cycles = int(input("Number of backtest cycles: ").strip())
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

else:
    backtest_timeframe = timeframe
    backtest_interval  = choose_interval(backtest_timeframe)
    sim_windows = []
    plot_mode   = input("\nCreate [1] one plot or [M] many plots? ").strip().lower()
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
    output_choice = input("Choose 1-6 [6]: ").strip() or "6"
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
clear_generated_files(analysis_dir, ("*.csv", "*.png"))

# ─── Bar generation ───────────────────────────────────────────────────────────
show_run_details   = run_count < 100
progress_interval  = max(1, run_count // 100)


def generate_daily(
    timestamps: list[pd.Timestamp],
    vol_index: float,
    verbose: bool = False,
) -> pd.DataFrame:
    """Simulate one OHLC run over the provided timestamp sequence.

    vol_index is computed once by the caller (based on market_change_pct and
    any global scaling) and passed in to avoid repeated external calls inside
    the inner loop.
    """
    rows = []
    prev_close = starting_price
    for ts in timestamps:
        open_price = prev_close
        # Simulate close based on the vol_index provided by the caller.
        simulated_change_pct = rand.uniform(-vol_index, vol_index) * (6 / 15)
        close_price = open_price * (1 + simulated_change_pct / 100)

        # Enforce non-negative floor only if configured.
        if not allow_negative:
            close_price = max(0.0, close_price)

        wick_range = (3.5 / 15) * (prev_close / 100) * vol_index
        high = max(open_price, close_price) + abs(rand.normalvariate(0, wick_range / 2))
        low = min(open_price, close_price) - abs(rand.normalvariate(0, wick_range / 2))
        if not allow_negative:
            low = max(0.0, low)

        row = {
            "Open": round(open_price, 9),
            "High": round(high, 9),
            "Low": round(low, 9),
            "Close": round(close_price, 9),
        }
        rows.append((ts, row))
        prev_close = close_price

        if verbose:
            print(f"{ts}: O={row['Open']} H={row['High']} L={row['Low']} C={row['Close']}")

    daily = pd.DataFrame(
        [r for _, r in rows],
        index=pd.DatetimeIndex([ts for ts, _ in rows], name="Date"),
    )
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

# ─── Run loop ─────────────────────────────────────────────────────────────────
daily = pd.DataFrame()
backtest_cycles: list[dict] = []
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
    # Compute vol_index once per run and pass it to the generator to avoid
    # repeated calls and to make behavior deterministic for the whole run.
    market_change_pct, change_info = run_market_change(backtest_timeframe, sim_start, sim_end)
    if show_run_details and backtest:
        print(f"  Market change: {describe_change(backtest_timeframe, market_change_pct, change_info)}")
    formula_exponent = -(2.98228 * market_change_pct - 5.4098)
    if formula_exponent > 709:
        run_vol_index = 0.0
    elif formula_exponent < -709:
        run_vol_index = 99.81711
    else:
        run_vol_index = 99.81711 / (1 + math.exp(formula_exponent))
    # Compatibility scaling in original code: question = vol_index * 8
    run_vol_index = run_vol_index * 8
    daily      = generate_daily(timestamps, run_vol_index, verbose=show_run_details)
    if backtest:
        backtest_cycles.append(
            {"cycle": run_number, "sim": daily.copy(), "start": sim_start, "end": sim_end}
        )

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
            volume=False,
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

    # Validate generated CSV row count matches timeframe expectation (skip 'ytd' which is auto-scaled).
    # ytd is date-dependent, so its expected row count is the number of timestamps generated.
    expected = len(timestamps) if backtest_timeframe == "ytd" else bars_for(backtest_timeframe, backtest_interval, today)
    if expected is not None:
        try:
            df_check = pd.read_csv(csv_file, parse_dates=[0])
            actual = len(df_check)
            if actual != expected:
                raise RuntimeError(
                    f"Generated CSV rows ({actual}) != expected for timeframe '{backtest_timeframe}' at {backtest_interval} candles ({expected}) - file: {csv_file}"
                )
        except Exception as e:
            # Raise any validation/parsing errors to fail fast and make the issue visible.
            raise

    if show_run_details:
        if render_source_charts:
            print(f"\nSaved chart to : {image_file}")
        print(f"Saved data to  : {csv_file}")

# ─── Analysis ─────────────────────────────────────────────────────────────────
analysis_result = analyze_looped_results(
    results_dir,
    analysis_dir,
    render_charts=render_analysis_charts,
    render_source_charts=render_source_charts,
    timeframe=backtest_timeframe,
    chart_label=f"{chart_symbol} ({interval_label(backtest_interval)})",
)

summary_path     = Path(analysis_dir) / "analysis_summary.csv"
analysis_summary = pd.read_csv(summary_path)
analysis_summary.insert(1, "instrument",  ticker_name)
analysis_summary.insert(2, "asset_type",  asset_type)
analysis_summary.insert(4, "sim_date",    today.isoformat())
analysis_summary.insert(5, "is_backtest", int(backtest))
analysis_summary.to_csv(summary_path, index=False)

print(
    f"\nAnalysis complete: {analysis_result['runs']} pairs, "
    f"{analysis_result['bars']} bars, {analysis_result['trades']} crossover trades."
)
# ─── Backtest: real price ghost halo + accuracy index (one plot) ──────────────
if backtest:
    run_backtest_comparison(
        backtest_cycles,
        symbol     = chart_symbol,
        timeframe  = backtest_timeframe,
        output_dir = analysis_dir,
        today      = today,
        bar        = interval_spec(backtest_timeframe, backtest_interval),
        interval   = backtest_interval,
    )

if backtest == "n":
# ─── ML option ────────────────────────────────────────────────────────────────
    if run_count < 100:
        ml_choice = input("\nRun ML model on these results? (y/n): ").strip().lower()
        run_ml    = ml_choice in ("y", "yes")
    
    if run_ml:
        from ml_model import run_ml_pipeline
        run_ml_pipeline(
            analysis_dir,
            instrument  = ticker_name,
            asset_type  = asset_type,
            timeframe   = backtest_timeframe,
        )
    
    df = daily