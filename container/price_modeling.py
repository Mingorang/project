"""Price simulation: forward runs and Rewind & Replay.

Mode 1  Forward simulation  - N random OHLC paths from today's price, analysed by
                              market_analysis.py and optionally fed to ml_model.py.
Mode 2  Rewind & Replay     - pick a timeframe, a candle interval and N cycles.  The
                              program goes back N x timeframe (after checking it is
                              possible), then simulates from there up to today.  Cycle 1
                              opens at the real price on the rewind date and every later
                              cycle opens at the previous cycle's simulated close, so the
                              plots are one continuous path.  Every cycle is scored against
                              the real market (backtest_compare.py) and the whole chain can
                              be repeated ("loops").  The six output options apply to both
                              modes (menu appears from 100 runs).

Dependencies: market_analysis.py (analysis), backtest_compare.py (real-price accuracy for
Rewind & Replay), ml_model.py (optional ML), price_change.py / price_open.py (Yahoo data).
"""
from __future__ import annotations

import math
import os
import random as rand
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplfinance as mpf
import pandas as pd

import price_open
from backtest_compare import comparable_cycles, run_rewind_comparison, YAHOO_INTERVALS
from market_analysis import analyze_looped_results
from price_change import _daily_closes, _yfinance_symbol, get_timeframe_change_details

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
# 24-hour simulated day would leave most candles without real data to compare against.
SESSION_START_HOUR = 9
SESSION_MINUTES    = 540

# Candle-count guide-rails shown in the interval menu.
MIN_CANDLES_ANALYSIS = 12    # analysis needs SMA 12 -> fewer candles cannot be analysed
MIN_CANDLES_MACD     = 35    # MACD 12-26 + 9 signal needs 35 candles to complete
MAX_CANDLES_COMFY    = 100   # beyond this the chart gets crowded

VALID_TIMEFRAMES = set(TIMEFRAME_CONFIG)

# ─── Market-change input to the volatility formula ────────────────────────────
# The % change used as a volatility scale spans the chosen timeframe
# (1d = last day, 5d, 1m, 3m, 1y, 5y, ytd = Jan 1 -> now).
#
# SCALE_PERIOD_CHANGE_TO_DAILY
#   Convert a multi-day move to a daily-equivalent magnitude so the scale remains
#   comparable across timeframes. Direction is not used as volatility.
#   False -> use the raw period move magnitude.
SCALE_PERIOD_CHANGE_TO_DAILY = True


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
    """Describe one candle interval in a form the generator and the accuracy scorer share.

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
    - Forward runs (price_modeling.py) use today as sim_start.
    - Rewind runs (rewind_replay.py) pass a historical sim_start for every cycle.

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


# ─── Volatility formula + market change ───────────────────────────────────────
def market_change(symbol: str, asset_type: str, timeframe: str, as_of: date | None = None,
                  span_days: int | None = None, today: date | None = None) -> tuple[float, dict]:
    """Return (value fed to the vol_index formula, details dict).

    as_of=None  -> measured up to today ("Jan 1 -> now" for ytd).
    as_of=date  -> point-in-time: the same-length period ending before `as_of`.
    Daily closes are downloaded once per symbol and cached by price_change.
    """
    details = get_timeframe_change_details(symbol, asset_type, timeframe,
                                           as_of=as_of, span_days=span_days, today=today)
    value = details["daily_equivalent_pct"] if SCALE_PERIOD_CHANGE_TO_DAILY else details["pct"]
    return value, details


def describe_change(tf: str, value: float, d: dict) -> str:
    text = (
        f"{tf.upper()} change {d['pct']:+.2f}%  "
        f"({d['base_date']} close {d['base_price']:.6g} -> {d['end_date']} close {d['end_price']:.6g}, "
        f"{d['trading_days']} trading day{'s' if d['trading_days'] != 1 else ''})"
    )
    if SCALE_PERIOD_CHANGE_TO_DAILY and d["trading_days"] > 1:
        text += f"  ->  daily-equivalent magnitude {abs(value):.4f}% used in the volatility formula"
    else:
        text += f"  ->  magnitude {abs(value):.4f}% used in the volatility formula"
    return text


def vol_index_from_change(market_change_pct: float) -> float:
    """Use the absolute daily-equivalent market move as a direction-neutral scale."""
    return abs(market_change_pct)


# ─── Bar generation ───────────────────────────────────────────────────────────
def generate_ohlc(timestamps: list[pd.Timestamp], vol_index: float, start_price: float,
                  allow_negative: bool, verbose: bool = False) -> pd.DataFrame:
    """Simulate one OHLC path over `timestamps`, opening at `start_price`.

    The last Close of the returned frame is the natural start_price of the next
    run, which is how rewind_replay.py chains cycles together.
    """
    rows = []
    prev_close = start_price
    for ts in timestamps:
        open_price = prev_close
        simulated_change_pct = rand.uniform(-vol_index, vol_index) * (6 / 15)
        if allow_negative:
            close_price = open_price * (1 + simulated_change_pct / 100)
        else:
            close_price = open_price * math.exp(simulated_change_pct / 100)

        wick_range = (3.5 / 15) * (prev_close / 100) * vol_index
        high = max(open_price, close_price) + abs(rand.normalvariate(0, wick_range / 2))
        low = min(open_price, close_price) - abs(rand.normalvariate(0, wick_range / 2))
        if not allow_negative:
            low = max(0.0, low)

        row = {"Open": open_price, "High": high, "Low": low, "Close": close_price}
        rows.append((ts, row))
        prev_close = close_price
        if verbose:
            print(f"{ts}: O={row['Open']} H={row['High']} L={row['Low']} C={row['Close']}")

    daily = pd.DataFrame([r for _, r in rows],
                         index=pd.DatetimeIndex([ts for ts, _ in rows], name="Date"))
    if verbose:
        print("\nCandlestick table:")
        print(daily.head(12).to_string())
    return daily


# ─── Shared prompts ───────────────────────────────────────────────────────────
def prompt_instrument() -> tuple[str, str]:
    asset_type = input(
        "Asset type [forex/futures/stock/crypto/other] (default forex): "
    ).strip().lower() or "forex"
    ticker_name = input("Instrument or pair to simulate (default GBP/USD): ").strip() or "GBP/USD"
    return asset_type, ticker_name


def default_yahoo_symbol(ticker_name: str, asset_type: str) -> str:
    return ticker_name.upper().replace("/", "") + "=X" if asset_type == "forex" else ticker_name


def prompt_timeframe(default: str = "5d", exclude: tuple[str, ...] = (), reason: str = "") -> str:
    """Show the timeframe menu and return a canonical timeframe key."""
    print("\nAvailable timeframes:")
    for tf, cfg in TIMEFRAME_CONFIG.items():
        if tf in exclude:
            print(f"  {tf:<4} — not available here ({reason})")
        else:
            print(f"  {tf:<4} — {cfg['description']}")
    allowed = [tf for tf in TIMEFRAME_CONFIG if tf not in exclude]
    while True:
        raw = input(f"Timeframe [{default}]: ").strip() or default
        timeframe = normalise_timeframe(raw)
        if timeframe in allowed:
            print(f"  Selected: {timeframe} — {TIMEFRAME_CONFIG[timeframe]['description']}")
            return timeframe
        print(f"  '{raw}' is not valid. Choose from: {', '.join(allowed)}")


def prompt_price_mode() -> bool:
    """Return allow_negative."""
    mode = input("Price behaviour: [1] allow negative futures prices or [2] floor prices at zero? ").strip().lower()
    if mode in ("1", "futures"):
        return True
    if mode in ("2", "floor"):
        return False
    sys.exit("Choose 1 (allow negative) or 2 (floor at zero).")


def prompt_whole_number(question: str, default: int | None = None, minimum: int = 1) -> int:
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{question}{suffix}: ").strip()
    if not raw and default is not None:
        return default
    try:
        value = int(raw)
    except ValueError:
        sys.exit(f"Enter a whole number >= {minimum}.")
    if value < minimum:
        sys.exit(f"Enter a whole number >= {minimum}.")
    return value


def choose_output_mode(total_runs: int) -> tuple[bool, bool, bool]:
    """The six output options. Returns (render_source_charts, render_analysis_charts, run_ml).

    The menu only appears from 100 runs upwards; below that, both chart sets are
    drawn and ML is offered at the end (see maybe_run_ml).
    """
    if total_runs < 100:
        return True, True, False
    print(f"\nChoose outputs for {total_runs:,} runs:")
    print("  Source PNGs: the original OHLC chart for each run.")
    print("  Analysis PNGs: the indicator and trade chart for each run.")
    print("  Per-run CSVs and combined analysis metrics are always saved.")
    print("  1) Source + analysis charts")
    print("  2) Source + analysis charts + ML (longest processing time)")
    print("  3) Analysis charts + ML (no source charts)")
    print("  4) Analysis charts only (no source charts or ML)")
    print("  5) ML only (no charts; required analysis metrics are still calculated)")
    print("  6) CSVs + metrics only (no charts or ML; fastest) [default]")
    choice = input("Choose 1-6 [6]: ").strip() or "6"
    modes = {"1": (True, True, False), "2": (True, True, True), "3": (False, True, True),
             "4": (False, True, False), "5": (False, False, True), "6": (False, False, False)}
    if choice not in modes:
        sys.exit("Choose one of the output modes 1-6.")
    return modes[choice]


def maybe_run_ml(analysis_dir: str | Path, total_runs: int, run_ml: bool,
                 instrument: str, asset_type: str, timeframe: str) -> None:
    if total_runs < 100:
        run_ml = input("\nRun ML model on these results? (y/n): ").strip().lower() in ("y", "yes")
    if run_ml:
        from ml_model import run_ml_pipeline
        run_ml_pipeline(analysis_dir, instrument=instrument, asset_type=asset_type, timeframe=timeframe)


# ─── Folders and per-run files ────────────────────────────────────────────────
def ensure_output_dirs(master_dir: str | Path) -> tuple[str, str]:
    """looped_results / looped_analysis must already exist (never created automatically)."""
    results_dir = os.path.join(master_dir, "looped_results")
    analysis_dir = os.path.join(master_dir, "looped_analysis")
    for folder, label in [(results_dir, "Output"), (analysis_dir, "Analysis output")]:
        if not os.path.isdir(folder):
            sys.exit(f"{label} folder does not exist: {folder}\nCreate the folder manually before generating data.")
    return results_dir, analysis_dir


def clear_generated_files(directory: str | Path, patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        for path in Path(directory).glob(pattern):
            if path.is_file():
                path.unlink()


BLACK_BACKGROUND_STYLE = mpf.make_mpf_style(
    base_mpf_style="charles", figcolor="black", facecolor="black", y_on_right=True,
    rc={"axes.labelcolor": "white", "axes.titlecolor": "white", "text.color": "white",
        "xtick.color": "white", "ytick.color": "white"},
)


def render_source_chart(daily: pd.DataFrame, image_file: str, title: str) -> None:
    lowest_low, highest_high = daily["Low"].min(), daily["High"].max()
    low_to_high = ("Low to high: N/A" if lowest_low <= 0
                   else f"Low to high: {abs((highest_high - lowest_low) / lowest_low * 100):.2f}%")
    start_price = float(daily["Open"].iloc[0])
    end_price = float(daily["Close"].iloc[-1])
    period_change = (
        f"{(end_price / start_price - 1) * 100:.2f}%"
        if start_price != 0 and math.isfinite(start_price) and math.isfinite(end_price)
        else "N/A"
    )
    figure, axes = mpf.plot(daily, type="candle", style=BLACK_BACKGROUND_STYLE, volume=False,
                            figsize=(12, 6), ylabel="Price", title=title, returnfig=True)
    axes[0].legend(handles=[Line2D([], [], linestyle="None", color="none")] * 2,
                   labels=[low_to_high, f"Period change: {period_change}"],
                   loc="upper left", handlelength=0, handletextpad=0)
    figure.savefig(image_file, facecolor=figure.get_facecolor())
    plt.close(figure)


def save_run(daily: pd.DataFrame, results_dir: str, run_number: int, expected_rows: int,
             title: str, render_source: bool) -> str:
    """Write data_N.csv (+ image_N.png) and verify the CSV has the expected row count."""
    csv_file = os.path.join(results_dir, f"data_{run_number}.csv")
    if render_source:
        render_source_chart(daily, os.path.join(results_dir, f"image_{run_number}.png"), title)
    daily.to_csv(csv_file)
    actual = len(pd.read_csv(csv_file))
    if actual != expected_rows:
        raise RuntimeError(f"Generated CSV rows ({actual}) != expected ({expected_rows}) - file: {csv_file}")
    return csv_file

# ─── Rewind & Replay: planning and feasibility ────────────────────────────────
DEFAULT_CYCLES = 40
# True  -> every cycle's volatility comes from the same-length real period that ENDED
#          just before that cycle began, so the simulation never sees what it is scored on.
# False -> every cycle uses the change measured up to today.
POINT_IN_TIME_CHANGE = True
NO_REWIND = ("ytd",)             # ytd has no fixed length, so "N x timeframe" is undefined


@dataclass
class Plan:
    timeframe: str
    interval: str
    spec: dict
    n_cycles: int
    cycle_len_bdays: int
    cycle_starts: list[date]
    timestamps: list[list[pd.Timestamp]]

    @property
    def start(self) -> date:
        return self.cycle_starts[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.timestamps[-1][-1]

    @property
    def bars_per_cycle(self) -> int:
        return len(self.timestamps[0])


def cycle_length_bdays(timeframe: str, spec: dict) -> int:
    """Business days one cycle occupies (week candles can overshoot the nominal span)."""
    span = TIMEFRAME_CONFIG[timeframe]["span_days"]
    if spec["kind"] == "week":
        return math.ceil(span / 5 / spec["weeks"]) * spec["weeks"] * 5
    return span


def make_plan(timeframe: str, interval: str, n_cycles: int, today: date) -> Plan:
    """Lay N back-to-back cycles so the last one ends on the last completed business day."""
    spec = interval_spec(timeframe, interval)
    length = cycle_length_bdays(timeframe, spec)
    total = n_cycles * length
    end_day = _prev_business_day(today - pd.Timedelta(days=1))
    first = pd.bdate_range(end=end_day, periods=total)[0]
    if spec["kind"] == "week":                      # weekly candles open on Mondays
        first -= pd.Timedelta(days=first.weekday())
    starts = [d.date() for d in pd.bdate_range(start=first, periods=total)[::length]]
    stamps = [build_timestamps(timeframe, s, None, interval) for s in starts]
    return Plan(timeframe, interval, spec, n_cycles, length, starts, stamps)


def ask_yes_no(question: str, default: bool = False) -> bool:
    raw = input(f"{question} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not raw else raw in ("y", "yes")


def check_feasibility(symbol: str, asset_type: str, timeframe: str, interval: str,
                      requested: int, today: date) -> Plan:
    """Check that N x timeframe fits the available history, report, and return the Plan to run."""
    spec = interval_spec(timeframe, interval)
    length = cycle_length_bdays(timeframe, spec)
    span = TIMEFRAME_CONFIG[timeframe]["span_days"]
    print(f"\nChecking whether {requested} x {timeframe} ({requested * length:,} trading days) is possible...")
    try:
        closes = _daily_closes(symbol, asset_type)
    except Exception as error:
        sys.exit(f"Could not download Yahoo Finance history for {symbol}: {error}")

    end_day = _prev_business_day(today - pd.Timedelta(days=1))
    available = len(pd.bdate_range(closes.index[0], end_day))
    # Room is needed for the cycles AND for one look-back period before the first cycle
    # (the volatility of cycle 1 is measured on the period that precedes it).
    n = min(requested, max(0, (available - span - 5) // length))
    plan = None
    while n >= 1:
        candidate = make_plan(timeframe, interval, n, today)
        try:
            market_change(symbol, asset_type, timeframe, as_of=candidate.start, today=today)
            plan = candidate
            break
        except ValueError:
            n -= 1
    if plan is None:
        sys.exit(f"Not even one {timeframe} cycle fits: {symbol} history starts {closes.index[0].date()}.")

    print(f"  History     : {symbol} daily closes since {closes.index[0].date()} "
          f"(~{available:,} trading days available)")
    if plan.n_cycles < requested:
        print(f"  NOT POSSIBLE as requested: {requested} cycles need more history than exists.")
        print(f"  Maximum     : {plan.n_cycles} cycle{'s' if plan.n_cycles != 1 else ''} of {timeframe}.")
        if not ask_yes_no(f"  Use {plan.n_cycles} cycle{'s' if plan.n_cycles != 1 else ''} instead?", default=True):
            sys.exit("Cancelled.")
    else:
        print("  Possible    : yes, enough history for every cycle.")

    flags = comparable_cycles(plan.cycle_starts, plan.spec, today)
    scorable = sum(flags)
    nominal = span
    print(f"\nRewind plan")
    print(f"  Cycle       : {timeframe} = {plan.cycle_len_bdays} trading day{'s' if plan.cycle_len_bdays != 1 else ''}"
          + (f" (nominal {nominal}; {interval_label(interval)} candles overshoot to whole candles)"
             if plan.cycle_len_bdays != nominal else ""))
    print(f"  Candles     : {plan.bars_per_cycle} x {interval_label(interval)} per cycle, "
          f"{plan.bars_per_cycle * plan.n_cycles:,} in the chain")
    print(f"  Rewind to   : {plan.start}   ({plan.n_cycles} x {plan.cycle_len_bdays} = "
          f"{plan.n_cycles * plan.cycle_len_bdays:,} trading days back)")
    print(f"  Replay until: {plan.end:%Y-%m-%d %H:%M}  (last completed trading day)")
    print(f"  Real prices : Yahoo can serve comparison data for {scorable} of {plan.n_cycles} cycles")
    if scorable < plan.n_cycles:
        first_ok = flags.index(True) + 1 if scorable else None
        why = ("Yahoo only keeps candles this small for a limited time "
               f"(5m/15m: ~{YAHOO_INTERVALS['5m'][1]} days, 1h: ~{YAHOO_INTERVALS['1h'][1]} days)")
        print(f"  WARNING     : {plan.n_cycles - scorable} older cycle(s) cannot be scored; {why}.")
        print("                They are still simulated and chained, just without a real-price comparison."
              + (f" Scoring starts at cycle {first_ok}." if first_ok else ""))
        print("                A larger candle interval (e.g. 1h or 1d) keeps more cycles comparable.")
        if not ask_yes_no("  Continue anyway?", default=True):
            sys.exit("Cancelled.")
    return plan


def cycle_volatilities(plan: Plan, symbol: str, asset_type: str, today: date) -> list[dict]:
    out = []
    for cycle, start in enumerate(plan.cycle_starts, start=1):
        try:
            value, details = market_change(symbol, asset_type, plan.timeframe, today=today,
                                           as_of=start if POINT_IN_TIME_CHANGE else None)
        except ValueError as error:
            print(f"  Warning: no point-in-time change before {start} ({error}); using the change up to today.")
            value, details = market_change(symbol, asset_type, plan.timeframe, today=today)
        out.append({"vol_index": vol_index_from_change(value), "value": value, "details": details})
    return out


# ─── Mode 1: forward simulation ───────────────────────────────────────────────
def run_forward() -> None:
    asset_type, ticker_name = prompt_instrument()

    price_option = input(f"Starting price: [1] enter manually or [2] use current {ticker_name}? ").strip()
    if price_option == "2":
        default_symbol = default_yahoo_symbol(ticker_name, asset_type)
        price_symbol = input(f"Yahoo Finance symbol for {ticker_name} [{default_symbol}]: ").strip() or default_symbol
        price_data = price_open.get_starting_price(price_symbol)
        if not isinstance(price_data, dict):
            sys.exit(f"Could not get {ticker_name} price: {price_data}")
        starting_price = float(price_data["current"])
        chart_symbol = _yfinance_symbol(price_symbol, asset_type)
    else:
        starting_price = float(input("Starting price: "))
        chart_symbol = _yfinance_symbol(ticker_name, asset_type)

    results_dir, analysis_dir = ensure_output_dirs(os.path.dirname(os.path.abspath(__file__)))

    timeframe = prompt_timeframe(default="5d")
    today = date.today()
    try:
        market_change_pct, change_info = market_change(chart_symbol, asset_type, timeframe, today=today)
        print(f"Yahoo {describe_change(timeframe, market_change_pct, change_info)}")
    except Exception as error:
        sys.exit(f"Could not get Yahoo Finance {timeframe.upper()} change for {ticker_name} ({chart_symbol}): {error}")

    allow_negative = prompt_price_mode()
    interval = choose_interval(timeframe)

    plot_mode = input("\nCreate [1] one plot or [M] many plots? ").strip().lower()
    if plot_mode in ("1", "one", "single"):
        run_count = 1
    elif plot_mode in ("m", "many"):
        run_count = prompt_whole_number("How many plots?")
    else:
        sys.exit("Choose 1 for one plot or M for many plots.")

    render_source, render_analysis, run_ml = choose_output_mode(run_count)
    clear_generated_files(results_dir, ("data_*.csv", "image_*.png"))
    clear_generated_files(analysis_dir, ("*.csv", "*.png"))

    show_run_details = run_count < 100
    progress_interval = max(1, run_count // 100)
    vol_index = vol_index_from_change(market_change_pct)
    expected_rows = len(build_timestamps(timeframe, today, None, interval))
    title = f"{ticker_name} — {timeframe.upper()} simulation ({interval_label(interval)} candles)"

    for run_number in range(1, run_count + 1):
        if show_run_details:
            print(f"\nGenerating run {run_number} of {run_count}...")
        elif run_number == 1 or run_number % progress_interval == 0 or run_number == run_count:
            print(f"Generating run {run_number:,} of {run_count:,}...")
        timestamps = build_timestamps(timeframe, today, None, interval)
        daily = generate_ohlc(timestamps, vol_index, starting_price, allow_negative, verbose=show_run_details)
        csv_file = save_run(daily, results_dir, run_number, expected_rows, title, render_source)
        if show_run_details:
            if render_source:
                print(f"\nSaved chart to : {os.path.join(results_dir, f'image_{run_number}.png')}")
            print(f"Saved data to  : {csv_file}")

    analysis_result = analyze_looped_results(
        results_dir, analysis_dir, render_charts=render_analysis, render_source_charts=render_source,
        timeframe=timeframe, instrument=ticker_name, asset_type=asset_type, sim_date=today.isoformat(),
        chart_label=f"{chart_symbol} ({interval_label(interval)})",
    )
    print(f"\nAnalysis complete: {analysis_result['runs']} pairs, "
          f"{analysis_result['bars']} bars, {analysis_result['trades']} crossover trades.")
    maybe_run_ml(analysis_dir, run_count, run_ml, ticker_name, asset_type, timeframe)
    globals()["df"] = daily


# ─── Mode 2: Rewind & Replay ──────────────────────────────────────────────────
def run_rewind() -> None:
    asset_type, ticker_name = prompt_instrument()
    default_symbol = default_yahoo_symbol(ticker_name, asset_type)
    price_symbol = input(f"Yahoo Finance symbol for {ticker_name} [{default_symbol}]: ").strip() or default_symbol
    chart_symbol = _yfinance_symbol(price_symbol, asset_type)
    results_dir, analysis_dir = ensure_output_dirs(os.path.dirname(os.path.abspath(__file__)))

    timeframe = prompt_timeframe(default="5d", exclude=NO_REWIND,
                                 reason="its length changes every day, so 'N x timeframe' is undefined")
    interval = choose_interval(timeframe)
    n_cycles = prompt_whole_number("Cycles N (the program goes back N x timeframe)", default=DEFAULT_CYCLES)
    today = date.today()

    plan = check_feasibility(chart_symbol, asset_type, timeframe, interval, n_cycles, today)
    n_cycles = plan.n_cycles

    vols = cycle_volatilities(plan, chart_symbol, asset_type, today)
    print("\nVolatility input (point-in-time: each cycle only sees the market BEFORE it starts)")
    for cycle in (range(1, n_cycles + 1) if n_cycles <= 6 else (1, n_cycles)):
        print(f"  Cycle {cycle}: {describe_change(timeframe, vols[cycle - 1]['value'], vols[cycle - 1]['details'])}")

    price_choice = input(f"\nStarting price at {plan.start}: [1] real {ticker_name} close before that date "
                         f"or [2] enter manually? [1]: ").strip() or "1"
    if price_choice == "2":
        start_price = float(input("Starting price: "))
    else:
        d = vols[0]["details"]
        start_price = float(d["end_price"])
        print(f"  Using the {d['end_date']} close: {start_price:.6g}")
    allow_negative = prompt_price_mode()

    loop_mode = input("\nRun the rewind [1] once or [M] many times (loops)? [1]: ").strip().lower()
    if loop_mode in ("1", "one", "once", ""):
        loops = 1
    elif loop_mode in ("m", "many"):
        loops = prompt_whole_number("How many loops?")
    else:
        sys.exit("Choose 1 for once or M for many loops.")
    total_runs = loops * n_cycles
    print(f"\n{loops} loop(s) x {n_cycles} cycles = {total_runs:,} runs")
    render_source, render_analysis, run_ml = choose_output_mode(total_runs)

    clear_generated_files(results_dir, ("data_*.csv", "image_*.png"))
    clear_generated_files(analysis_dir, ("*.csv", "*.png"))

    frames_by_loop: list[list[pd.DataFrame]] = []
    progress = max(1, total_runs // 100)
    for loop in range(1, loops + 1):
        price, frames = start_price, []
        for cycle in range(1, n_cycles + 1):
            run = (loop - 1) * n_cycles + cycle
            daily = generate_ohlc(plan.timestamps[cycle - 1], vols[cycle - 1]["vol_index"], price, allow_negative)
            title = (f"{ticker_name} — {timeframe.upper()} rewind, loop {loop}, cycle {cycle}/{n_cycles} "
                     f"({interval_label(interval)} candles)")
            save_run(daily, results_dir, run, plan.bars_per_cycle, title, render_source)
            end_price = float(daily["Close"].iloc[-1])
            if total_runs < 100 or run == 1 or run % progress == 0 or run == total_runs:
                print(f"Loop {loop}/{loops} cycle {cycle}/{n_cycles} (run {run:,}): "
                      f"{plan.cycle_starts[cycle - 1]}  {price:.6g} -> {end_price:.6g}")
            frames.append(daily)
            price = end_price
        frames_by_loop.append(frames)

    result = analyze_looped_results(
        results_dir, analysis_dir, render_charts=render_analysis, render_source_charts=render_source,
        timeframe=timeframe, instrument=ticker_name, asset_type=asset_type, sim_date=today.isoformat(),
        chart_label=f"{chart_symbol} ({interval_label(interval)})",
    )
    summary_path = Path(analysis_dir) / "analysis_summary.csv"
    summary = pd.read_csv(summary_path)
    summary.insert(1, "loop", (summary["run"] - 1) // n_cycles + 1)
    summary.insert(2, "cycle", (summary["run"] - 1) % n_cycles + 1)
    summary.to_csv(summary_path, index=False)
    print(f"\nAnalysis complete: {result['runs']} pairs, {result['bars']} bars, {result['trades']} crossover trades.")

    run_rewind_comparison(frames_by_loop, plan.cycle_starts, plan.timestamps, chart_symbol, timeframe,
                          interval_label(interval), plan.spec, analysis_dir, today)
    maybe_run_ml(analysis_dir, total_runs, run_ml, ticker_name, asset_type, timeframe)


def main() -> None:
    mode = input("Mode: [1] Forward simulation  [2] Rewind & Replay "
                 "(go back N x timeframe, replay up to today, score vs real prices) [1]: ").strip() or "1"
    if mode == "1":
        run_forward()
    elif mode == "2":
        run_rewind()
    else:
        sys.exit("Choose 1 or 2.")


if __name__ == "__main__":
    main()