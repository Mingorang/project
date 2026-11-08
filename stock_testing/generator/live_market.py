"""live_market.py - option 3 of price_modeling.py: current market data from Yahoo Finance.

For one instrument (any Yahoo symbol: SAP, ^GSPC, GBPUSD=X, BTC-USD, ES=F ...) it

  1. builds the market snapshot the daily scan artifact shows (last price, 1d/5d/1m/3m/ytd/1y change,
     20-day volatility and where it sits in its 5-year range, RSI, distance from the 200-day average,
     ATR, 1-day / 5-day bands, data-health checks) for the CURRENT price, plus context rows;
  2. downloads one REAL window per period (1d, 5d, 1m, 3m, 6m, ytd, 1y, 2y, 5y, 10y, max), each
     running from "that long ago" up to now. The candle size is chosen automatically per window so
     every chart gets about [live] candles_min..candles_max candles: the coarsest Yahoo interval that
     can reach the range is downloaded, then N consecutive bars are merged into one candle;
  3. simulates forward paths per period from the current price with the existing generator, the
     volatility calibrated on that period's real candles (not on the market-change formula);
  4. writes everything as data_N.csv / image_N.png so the usual analysis -> ML -> report pipeline
     runs on it, tagged with data_source / period / sample_weight: real rows weigh more than
     simulated ones, and the live ML folder is cleared before every training.

Real candles are start-labelled in exchange-local time (Yahoo's convention). Intraday candles are
merged within each session only, so no candle spans the overnight gap.
"""
from __future__ import annotations

import copy
import importlib
import math
import shutil
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import console as ui

HERE = Path(__file__).resolve().parent

# period key -> (yfinance period, calendar days it spans (None = open-ended), base intervals coarse -> fine)
PERIOD_SPECS: dict[str, tuple[str, int | None, tuple[str, ...]]] = {
    "1d":  ("1d",  4,    ("15m", "5m", "2m", "1m")),
    "5d":  ("5d",  9,    ("60m", "30m", "15m", "5m", "2m")),
    "1m":  ("1mo", 32,   ("1d", "60m", "30m", "15m")),
    "3m":  ("3mo", 93,   ("1d", "60m", "30m")),
    "6m":  ("6mo", 185,  ("1wk", "1d", "60m")),
    "ytd": ("ytd", None, ("1wk", "1d", "60m", "30m", "15m", "5m")),
    "1y":  ("1y",  367,  ("1wk", "1d", "60m")),
    "2y":  ("2y",  732,  ("1mo", "1wk", "1d")),
    "5y":  ("5y",  1828, ("1mo", "1wk", "1d")),
    "10y": ("10y", 3654, ("3mo", "1mo", "1wk", "1d")),
    "max": ("max", None, ("3mo", "1mo", "1wk", "1d")),
}
PERIOD_TEXT = {"1d": "1 day", "5d": "5 days", "1m": "1 month", "3m": "3 months", "6m": "6 months",
               "ytd": "year to date", "1y": "1 year", "2y": "2 years", "5y": "5 years", "10y": "10 years",
               "max": "all history"}
# Yahoo serves intraday bars only this many calendar days back.
YAHOO_INTRADAY_DAYS = {"1m": 7, "2m": 59, "5m": 59, "15m": 59, "30m": 59, "60m": 729, "90m": 59}
BASE_MINUTES = {"1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "90m": 90}
SAVE_STATES = {  # same menu as price_modeling options 1 and 2: (source PNGs, analysis PNGs, ML)
    1: (True, True, False, "source + analysis charts"),
    2: (True, True, True, "source + analysis charts + ML"),
    3: (False, True, True, "analysis charts + ML"),
    4: (False, True, False, "analysis charts only"),
    5: (False, False, True, "ML only"),
    6: (False, False, False, "CSVs + metrics only"),
}
SIGMA_PER_VOL = 0.4 / math.sqrt(3.0)   # per-bar % sigma of the plain generator per unit of vol_index
OHLC = ["Open", "High", "Low", "Close"]


class LiveDataError(RuntimeError):
    """Yahoo returned nothing usable for the instrument (wrong symbol, delisted, no network)."""


# ─── Real windows ─────────────────────────────────────────────────────────────
@dataclass
class Window:
    period: str
    frame: pd.DataFrame          # OHLCV, tz-naive exchange time, one row per candle
    base: str                    # Yahoo interval downloaded
    factor: int                  # base bars merged into one candle
    raw_bars: int
    notes: list[str] = field(default_factory=list)

    @property
    def intraday(self) -> bool:
        return self.base in BASE_MINUTES

    @property
    def n(self) -> int:
        return len(self.frame)

    @property
    def candle_minutes(self) -> int | None:
        return BASE_MINUTES[self.base] * self.factor if self.intraday else None

    @property
    def candle_label(self) -> str:
        return candle_label(self.base, self.factor)

    @property
    def span_text(self) -> str:
        start, end = self.frame.index[0], self.frame.index[-1]
        fmt = "%d %b %H:%M" if self.intraday else "%d %b %Y"
        return f"{start:{fmt}} → {end:{fmt}}"


def candle_label(base: str, factor: int) -> str:
    if base in BASE_MINUTES:
        minutes = BASE_MINUTES[base] * factor
        if minutes % 60 == 0:
            hours = minutes // 60
            return f"{hours} hour{'s' if hours != 1 else ''}"
        return f"{minutes} min"
    unit, count = {"1d": ("trading day", factor), "1wk": ("week", factor),
                   "1mo": ("month", factor), "3mo": ("month", 3 * factor)}[base]
    if unit == "month" and count % 12 == 0:
        unit, count = "year", count // 12
    elif unit == "month" and count == 3:
        return "1 quarter"
    return f"{count} {unit}{'s' if count != 1 else ''}"


def period_span_days(period: str, today: date) -> int | None:
    span = PERIOD_SPECS[period][1]
    if period == "ytd":
        return (today - date(today.year, 1, 1)).days + 1
    return span


def candidate_bases(period: str, today: date) -> list[str]:
    """Base intervals Yahoo can serve for this period (intraday ones only inside their look-back)."""
    span = period_span_days(period, today)
    out = []
    for base in PERIOD_SPECS[period][2]:
        limit = YAHOO_INTRADAY_DAYS.get(base)
        if limit is None or (span is not None and span <= limit):
            out.append(base)
    return out


def clean_frame(history: pd.DataFrame, allow_nonpositive: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Yahoo OHLCV -> tz-naive, sorted, de-duplicated, finite, internally consistent candles."""
    notes: list[str] = []
    if history is None or history.empty or not set(OHLC).issubset(history.columns):
        return pd.DataFrame(columns=OHLC + ["Volume"]), notes
    frame = history[OHLC + (["Volume"] if "Volume" in history.columns else [])].copy()
    if "Volume" not in frame:
        frame["Volume"] = 0.0
    frame = frame.apply(pd.to_numeric, errors="coerce")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_localize(None)                          # keep the exchange's wall-clock time
    frame.index = index
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    bad = ~np.isfinite(frame[OHLC].to_numpy(dtype=float)).all(axis=1)
    if not allow_nonpositive:
        bad |= (frame[OHLC] <= 0).any(axis=1).to_numpy()
    if bad.any():
        notes.append(f"{int(bad.sum())} unusable bar(s) dropped (missing or non-positive prices)")
        frame = frame[~bad]
    volume = frame["Volume"].to_numpy(dtype=float)
    frame["Volume"] = np.where(np.isfinite(volume) & (volume >= 0), volume, 0.0)
    high = frame[["High", "Open", "Close"]].max(axis=1)
    low = frame[["Low", "Open", "Close"]].min(axis=1)
    repaired = int(((high != frame["High"]) | (low != frame["Low"])).sum())
    if repaired:
        notes.append(f"{repaired} bar(s) with High/Low inside the body repaired (adjusted-price rounding)")
    frame["High"], frame["Low"] = high, low
    frame.index.name = "Date"
    return frame.astype(float), notes


def candles_after(frame: pd.DataFrame, factor: int, intraday: bool) -> int:
    if intraday:
        sizes = frame.groupby(frame.index.normalize()).size().to_numpy()
        return int(np.sum(np.ceil(sizes / factor)))
    return int(math.ceil(len(frame) / factor))


def aggregate(frame: pd.DataFrame, factor: int, intraday: bool) -> pd.DataFrame:
    """Merge `factor` consecutive bars into one candle (open first, high max, low min, close last).

    Intraday groups restart at every session open, so a candle never spans the overnight gap.
    Daily and longer groups are anchored at the newest bar, so the latest candle is a full one and
    the oldest may be partial. Each candle is labelled with the time of its first bar.
    """
    if factor <= 1 or frame.empty:
        return frame.copy()
    if intraday:
        day = frame.index.normalize()
        keys = [day, frame.groupby(day).cumcount().to_numpy() // factor]
    else:
        keys = [np.arange(len(frame))[::-1] // factor]          # bar i -> (n - 1 - i) // factor: newest group is 0
    grouped = frame.groupby(keys, sort=True)
    out = pd.DataFrame({
        "Open": grouped["Open"].first(), "High": grouped["High"].max(), "Low": grouped["Low"].min(),
        "Close": grouped["Close"].last(), "Volume": grouped["Volume"].sum(),
    })
    stamps = pd.Series(frame.index, index=frame.index).groupby(keys, sort=True).first()
    out.index = pd.DatetimeIndex(stamps.to_numpy(), name="Date")
    return out.sort_index()


def choose_factor(frame: pd.DataFrame, intraday: bool, lo: int, hi: int, target: int) -> tuple[int, int]:
    """(factor, candles): the merge factor whose candle count lands in [lo, hi] closest to target.

    Intraday, a factor that splits the typical session evenly wins over one that leaves a stub
    candle at the close (24 hourly FX bars -> 12-hour candles, never 23 + 1). Counts within 5 candles
    of each other are a tie and the finer candle (smaller factor) wins. Falls back to the count
    nearest to the range when no factor reaches it.
    """
    best: tuple[tuple[float, float, float, int], int, int] | None = None
    typical = 0
    if intraday:
        sizes = frame.groupby(frame.index.normalize()).size()
        max_factor = int(sizes.max())
        typical = int(sizes.mode().max())
    else:
        max_factor = len(frame)
    for factor in range(1, max(1, max_factor) + 1):
        count = candles_after(frame, factor, intraday)
        outside = 0 if lo <= count <= hi else (lo - count if count < lo else count - hi)
        rest = typical % factor if intraday else 0
        stub = round((factor - rest) / factor, 1) if rest else 0.0      # share of the last candle that is missing
        score = (outside, stub, abs(count - target) // 5, factor)
        if best is None or score < best[0]:
            best = (score, factor, count)
        if count < lo:          # counts only fall as the factor grows
            break
    assert best is not None
    return best[1], best[2]


Fetcher = Callable[[str, str, str], pd.DataFrame]


def yahoo_fetcher(symbol: str, period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf
    return yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)


def fetch_window(symbol: str, period: str, cfg: dict[str, Any], today: date, fetcher: Fetcher = yahoo_fetcher,
                 cache: dict | None = None, allow_nonpositive: bool = False) -> Window | None:
    """Download the real window for one period and merge its bars into ~[candles_min, candles_max] candles."""
    live = cfg["live"]
    lo, hi, target = int(live["candles_min"]), int(live["candles_max"]), int(live["candles_target"])
    hard_min = int(cfg["indicators"]["slow_sma"]) + 2
    cache = {} if cache is None else cache
    yf_period = PERIOD_SPECS[period][0]
    best: tuple[tuple[float, float], Window] | None = None
    for base in candidate_bases(period, today):
        key = (symbol, yf_period, base)
        if key not in cache:
            try:
                cache[key] = clean_frame(fetcher(symbol, yf_period, base), allow_nonpositive)
            except Exception as error:                            # one bad interval must not end the run
                cache[key] = (pd.DataFrame(columns=OHLC + ["Volume"]), [f"{base}: {type(error).__name__}"])
        frame, notes = cache[key]
        if len(frame) < 2:
            continue
        intraday = base in BASE_MINUTES
        factor, count = choose_factor(frame, intraday, lo, hi, target)
        window = Window(period, aggregate(frame, factor, intraday), base, factor, len(frame), list(notes))
        outside = 0 if lo <= count <= hi else (lo - count if count < lo else count - hi)
        score = (outside, abs(count - target))
        if best is None or score < best[0]:
            best = (score, window)
        if outside == 0:
            break
    if best is None:
        return None
    window = best[1]
    if window.n < hard_min and period == "1d":
        # Just after the open there are only a handful of bars today: show the last full session instead.
        window = _previous_session(symbol, cfg, fetcher, cache, allow_nonpositive) or window
    if window.n < hard_min:
        return None
    if not lo <= window.n <= hi:
        window.notes.append(f"{window.n} candles (outside {lo}-{hi}: not enough history at any Yahoo interval)")
    return window


def _previous_session(symbol: str, cfg: dict[str, Any], fetcher: Fetcher, cache: dict,
                      allow_nonpositive: bool) -> Window | None:
    live = cfg["live"]
    for base in ("2m", "1m"):
        key = (symbol, "5d", base)
        if key not in cache:
            try:
                cache[key] = clean_frame(fetcher(symbol, "5d", base), allow_nonpositive)
            except Exception:
                continue
        frame, notes = cache[key]
        days = sorted(set(frame.index.normalize()))
        if len(days) < 2:
            continue
        session = frame[frame.index.normalize() == days[-2]]
        factor, _ = choose_factor(session, True, int(live["candles_min"]), int(live["candles_max"]),
                                  int(live["candles_target"]))
        return Window("1d", aggregate(session, factor, True), base, factor, len(session),
                      notes + ["today's session has only just opened: showing the previous full session"])
    return None


# ─── Session hours and future timestamps (for the simulated paths) ────────────
def trading_session(symbol: str, windows: list[Window]) -> tuple[int, int, bool]:
    """(open_minute, close_minute, trades_weekends) of the instrument's regular session, local time."""
    open_min = close_min = None
    weekends = False
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        meta = ticker.history_metadata or {}
        if not meta:
            ticker.history(period="5d", interval="1d")
            meta = ticker.history_metadata or {}
        regular = (meta.get("currentTradingPeriod") or {}).get("regular") or {}
        start, end = regular.get("start"), regular.get("end")
        if start is not None and end is not None:
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            open_min = start.hour * 60 + start.minute
            close_min = end.hour * 60 + end.minute
            if close_min >= 1439 or close_min <= open_min:
                close_min = 1440
        weekends = str(meta.get("instrumentType", "")).upper() == "CRYPTOCURRENCY"
    except Exception:
        pass
    if open_min is None:                                          # infer from the intraday windows
        stamps = [w.frame.index for w in windows if w.intraday and w.n]
        if stamps:
            all_times = pd.DatetimeIndex(np.concatenate([s.to_numpy() for s in stamps]))
            minutes = all_times.hour * 60 + all_times.minute
            open_min, close_min = int(minutes.min()), int(min(1440, minutes.max() + 60))
            weekends = bool((all_times.dayofweek >= 5).any())
        else:
            open_min, close_min = 9 * 60 + 30, 16 * 60
    return int(open_min), int(close_min), bool(weekends)


def future_index(window: Window, n: int, session: tuple[int, int, bool]) -> pd.DatetimeIndex:
    """n candle start times after the window's last candle, on the same candle grid."""
    last = window.frame.index[-1]
    open_min, close_min, weekends = session
    step_day = pd.offsets.Day(1) if weekends else pd.offsets.BDay(1)
    if window.intraday:
        length = int(window.candle_minutes)
        slots = [m for m in range(open_min, close_min, length)] or [open_min]
        stamps: list[pd.Timestamp] = []
        day = last.normalize()
        last_minute = last.hour * 60 + last.minute
        stamps.extend(day + pd.Timedelta(minutes=m) for m in slots if m > last_minute)
        while len(stamps) < n:
            day = day + step_day
            stamps.extend(day + pd.Timedelta(minutes=m) for m in slots)
        return pd.DatetimeIndex(stamps[:n], name="Date")
    k = window.factor
    if window.base == "1d":
        offset = (pd.offsets.Day(k) if weekends else pd.offsets.BDay(k))
        return pd.DatetimeIndex([last + offset * i for i in range(1, n + 1)], name="Date")
    if window.base == "1wk":
        return pd.DatetimeIndex([last + pd.Timedelta(weeks=k * i) for i in range(1, n + 1)], name="Date")
    months = k * (3 if window.base == "3mo" else 1)
    return pd.DatetimeIndex([last + pd.DateOffset(months=months * i) for i in range(1, n + 1)], name="Date")


# ─── Forward simulation calibrated on the real window ─────────────────────────
def real_sigma_pct(frame: pd.DataFrame) -> float:
    """Per-candle close-to-close standard deviation of the real window, in %."""
    returns = 100.0 * frame["Close"].pct_change().dropna()
    returns = returns[np.isfinite(returns)]
    return float(returns.std(ddof=1)) if len(returns) > 2 else float("nan")


def calibrate_vol(sigma_pct: float, params: Any, cfg: dict[str, Any]) -> tuple[float, str]:
    """vol_index whose simulated per-candle sigma matches the real one (closed form + one MC refinement)."""
    import backtest_compare as bc
    from project_config import clamp_vol_index
    if not (np.isfinite(sigma_pct) and sigma_pct > 0):
        return clamp_vol_index(cfg["limits"]["vol_index_min"], cfg)[0], "flat real window: minimum vol used"
    vol, note = clamp_vol_index(sigma_pct / SIGMA_PER_VOL, cfg)
    expected = bc.expected_bar_sigma_pct(params.with_(vol_index=vol), False)
    if np.isfinite(expected) and expected > 0 and not note:
        vol, note = clamp_vol_index(vol * sigma_pct / expected, cfg)
    return vol, note


def simulate_forward(window: Window, runs: int, params: Any, cfg: dict[str, Any], seed: int,
                     session: tuple[int, int, bool], period_index: int) -> tuple[list[pd.DataFrame], float, str]:
    import backtest_compare as bc
    sigma = real_sigma_pct(window.frame)
    volume = window.frame["Volume"].to_numpy(dtype=float)
    median_volume = float(np.median(volume[volume > 0])) if (volume > 0).any() else 0.0
    sim_params = params.with_(profile=(), volume_on=median_volume > 0,
                              volume_base=median_volume if median_volume > 0 else params.volume_base)
    vol, note = calibrate_vol(sigma, sim_params, cfg)
    stamps = future_index(window, window.n, session)
    start = float(window.frame["Close"].iloc[-1])
    paths = []
    for j in range(runs):
        rng = np.random.default_rng(bc.cycle_seed(seed, 1000 * (period_index + 1) + j, 31))
        paths.append(bc.generate_ohlc(stamps, start, sim_params.with_(vol_index=vol), rng, False, False))
    return paths, vol, note


# ─── Snapshot: the daily scan, for this instrument ────────────────────────────
def _scan_modules():
    """routine/scan.py + scan_page.py (one source of the snapshot maths, shared with the daily artifact)."""
    routine = HERE / "routine"
    if str(routine) not in sys.path:
        sys.path.insert(0, str(routine))
    with warnings.catch_warnings():                 # scan.py silences warnings globally at import
        scan = importlib.import_module("scan")
        page = importlib.import_module("scan_page")
    return scan, page


def parse_context(entries: list[str]) -> dict[str, str]:
    out = {}
    for entry in entries:
        name, _, symbol = entry.partition("=")
        out[name.strip()] = symbol.strip()
    return out


def _period_change(close: pd.Series, offset: pd.DateOffset | None, trading_days: int | None = None,
                   start: pd.Timestamp | None = None) -> float:
    if trading_days is not None:
        return float(close.iloc[-1] / close.iloc[-(trading_days + 1)] - 1) * 100 if len(close) > trading_days else float("nan")
    target = start if start is not None else close.index[-1] - offset
    base = close.loc[:target]
    return float(close.iloc[-1] / base.iloc[-1] - 1) * 100 if len(base) else float("nan")


def build_snapshot(name: str, symbol: str, cfg: dict[str, Any], history_path: Path) -> dict[str, Any]:
    """The scan's regime / extremes / health rows for the instrument (+ context), plus extra returns."""
    import yfinance as yf
    scan, _ = _scan_modules()
    marks = {name: symbol}
    context = {label: sym for label, sym in parse_context(cfg["live"]["context_symbols"]).items()
               if sym.upper() != symbol.upper() and label != name}      # never shadow the instrument itself
    symbols = {**marks, **context}
    raw = yf.download(list(dict.fromkeys(symbols.values())), period=f"{int(cfg['live']['snapshot_years'])}y",
                      interval="1d", auto_adjust=True, progress=False, threads=True)
    if raw is None or raw.empty:
        raise LiveDataError(f"Yahoo returned no daily history for {symbol}")

    def column(field_name: str) -> pd.DataFrame:
        part = raw[field_name]
        part = part.to_frame(symbol) if isinstance(part, pd.Series) else part
        inverse = {v: k for k, v in symbols.items()}
        part = part.rename(columns=inverse)
        index = pd.DatetimeIndex(part.index)
        part.index = index.tz_localize(None) if index.tz is not None else index
        return part

    close, high, low = column("Close"), column("High"), column("Low")
    asof = close.index[-1]
    rows, issues, failed = [], [], []
    for label in symbols:
        series = close[label].dropna() if label in close else pd.Series(dtype=float)
        if series.size < scan.MIN_HISTORY_BARS or (series <= 0).any():
            failed.append(label)
            issues.append(dict(instrument=label, severity="critical", issue="download failed" if series.size < 30
                               else "not enough history" if series.size < scan.MIN_HISTORY_BARS else "impossible price",
                               detail=f"{series.size} daily bars (the snapshot needs {scan.MIN_HISTORY_BARS})"))
            continue
        row = scan.regime_row(label, close[label], high[label], low[label])
        row.update(
            last=float(series.iloc[-1]),        # the scan rounds to 4 decimals: SHIB (0.000005) would read 0.0
            r5d=round(_period_change(series, None, trading_days=5), 2),
            r3m=round(_period_change(series, pd.DateOffset(months=3)), 2),
            rytd=round(_period_change(series, None, start=pd.Timestamp(asof.year, 1, 1) - pd.Timedelta(days=1)), 2),
            r1y=round(_period_change(series, pd.DateOffset(years=1)), 2),
            symbol=symbols[label], role="instrument" if label in marks else "context",
        )
        rows.append(row)
        issues.extend(scan.health_rows(label, close[label], high[label], low[label], asof))
    if name in failed:
        main_bars = int(close[name].dropna().size) if name in close else 0
        if main_bars < 2:                                     # nothing at all: wrong symbol or delisted
            raise LiveDataError(f"{symbol}: Yahoo returned no daily prices")
        # A recent listing: no snapshot row (it needs MIN_HISTORY_BARS), but the windows still work.
    previous = None
    if history_path.exists():
        try:
            previous = pd.read_csv(history_path)
            previous = previous[previous["instrument"].isin(list(symbols))]
        except (OSError, ValueError, KeyError, pd.errors.ParserError):
            previous = None
    changes = scan.changes_vs_previous(rows, previous)
    return dict(asof=str(asof.date()), rows=rows, issues=issues, changes=changes, failed=failed,
                first_run=previous is None or previous.empty, marks=list(marks),
                context=[k for k in symbols if k not in marks])


def append_snapshot_history(snapshot: dict[str, Any], history_path: Path) -> None:
    if not snapshot["rows"]:
        return
    frame = pd.DataFrame(snapshot["rows"])
    frame.insert(0, "scan_date", date.today().isoformat())
    frame.to_csv(history_path, mode="a", header=not history_path.exists(), index=False)


def print_snapshot(snapshot: dict[str, Any], name: str, cfg: dict[str, Any]) -> None:
    rows = {r["instrument"]: r for r in snapshot["rows"]}
    if name not in rows:
        ui.section(f"Snapshot · {name}")
        ui.warn(f"{name} has too little daily history for the scan metrics "
                f"(needs [scan] min_history_bars); the real windows below still work.")
        for issue in snapshot["issues"]:
            ui.note(f"{issue['severity']} · {issue['instrument']}: {issue['issue']} - {issue['detail']}", "warn")
        return
    main = rows[name]
    rsi = main["rsi14"]
    ind = cfg["indicators"]
    years = int(cfg["live"]["snapshot_years"])
    rsi_word = ("overbought" if rsi >= ind["rsi_overbought"] else "oversold" if rsi <= ind["rsi_oversold"] else "neutral")
    pct = main["vol_pctile"]
    ui.section(f"Snapshot · {name} · daily bars as of {snapshot['asof']}")
    ui.kv([
        ("Last", f"{ui.fmt_price(main['last'])}   (bar of {main['last_bar']})"),
        ("Change", "   ".join(f"{label} {ui.fmt_pct(main[key])}" for label, key in
                              (("1D", "r1d"), ("5D", "r5d"), ("1M", "r1m"), ("3M", "r3m"), ("YTD", "rytd"), ("1Y", "r1y")))),
        ("Volatility", f"20-day {ui.fmt_num(main['vol20_ann'], 1, '%')} a year · calmer than {100 - pct}% of the last "
                       f"{years} years (percentile {pct})"),
        ("Momentum", f"RSI {ui.paint(f'{rsi:.1f}', 'down' if rsi_word == 'overbought' else 'up' if rsi_word == 'oversold' else None)}"
                     f" ({rsi_word}) · {ui.fmt_pct(main['vs_sma200'])} vs the 200-day average"),
        ("Range", f"ATR {ui.fmt_num(main['atr14_pct'], 2, '%')} · 1-day band {ui.fmt_num(main['band_1d_pct'], 2, '%', '±')} · "
                  f"5-day band {ui.fmt_num(main['band_5d_pct'], 2, '%', '±')} · today {ui.fmt_num(main['move_sigma'], 2)} sigma"),
    ])
    table_rows = []
    for row in snapshot["rows"]:
        tag = "" if row["role"] == "instrument" else ui.paint(" ·ctx", "muted")
        table_rows.append([row["instrument"] + tag, ui.fmt_price(row["last"]), ui.fmt_pct(row["r1d"]),
                           ui.fmt_pct(row["r5d"]), ui.fmt_pct(row["r1m"]), ui.fmt_num(row["vol20_ann"], 1, "%"),
                           str(row["vol_pctile"]), ui.fmt_num(row["rsi14"], 1), ui.fmt_pct(row["vs_sma200"]),
                           ui.fmt_num(row["band_1d_pct"], 2, "%", "±"), ui.fmt_num(row["band_5d_pct"], 2, "%", "±")])
    print()
    ui.table(table_rows, ["instrument", "last", "1d", "5d", "1m", "20d vol", "pctile", "rsi", "vs 200d",
                          "1d band", "5d band"], "lrrrrrrrrrr")
    if snapshot["changes"]:
        print()
        for change in snapshot["changes"]:
            ui.note(f"changed · {change['instrument']}: {change['change']} - {change['detail']}", "accent")
    for issue in snapshot["issues"]:
        style = "bad" if issue["severity"] == "critical" else "warn"
        ui.note(f"{issue['severity']} · {issue['instrument']}: {issue['issue']} - {issue['detail']}", style)


# ─── Charts ───────────────────────────────────────────────────────────────────
def save_source_chart(frame: pd.DataFrame, path: Path, title: str, simulated: bool) -> None:
    """The same black mplfinance chart the generator draws, with a REAL / SIM badge."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplfinance as mpf
    from matplotlib.lines import Line2D
    style = mpf.make_mpf_style(base_mpf_style="charles", figcolor="black", facecolor="black", y_on_right=True,
                               rc={"axes.labelcolor": "white", "axes.titlecolor": "white", "text.color": "white",
                                   "xtick.color": "white", "ytick.color": "white"})
    low, high = float(frame["Low"].min()), float(frame["High"].max())
    first, last = float(frame["Open"].iloc[0]), float(frame["Close"].iloc[-1])
    volume = frame["Volume"].gt(0).any() if "Volume" in frame else False
    with plt.rc_context():          # mplfinance's style must not leak into the analysis charts (chart_jobs.py)
        figure, axes = mpf.plot(frame, type="candle", style=style, volume=bool(volume), figsize=(12, 6), ylabel="Price",
                                title=title, returnfig=True, warn_too_much_data=10_000)
        axes[0].legend(handles=[Line2D([], [], linestyle="None")] * 2,
                       labels=[f"Low to high: {100 * (high - low) / low:.2f}%" if low > 0 else "Low to high: n/a",
                               f"Period change: {100 * (last / first - 1):+.2f}%"],
                       loc="upper left", handlelength=0, handletextpad=0)
        badge = "SIMULATED" if simulated else "REAL"
        figure.text(0.012, 0.975, badge, fontsize=11, weight="bold", va="top",
                    color="#ffc857" if simulated else "#7fd1ff",
                    bbox=dict(facecolor="black", edgecolor="#ffc857" if simulated else "#7fd1ff", boxstyle="round,pad=.3"))
        figure.savefig(path, facecolor=figure.get_facecolor())
        plt.close(figure)


def plot_overview(name: str, windows: list[Window], sims: dict[str, list[pd.DataFrame]], path: Path, asof: str) -> None:
    """One figure: every real window (white) and the fan of its simulated futures (amber), side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_panels = len(windows)
    cols = 3 if n_panels > 4 else max(1, n_panels)
    rows = math.ceil(n_panels / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.0 * rows + 1.1), squeeze=False, facecolor="black")
    for ax, window in zip(axes.ravel(), windows):
        ax.set_facecolor("black")
        ax.tick_params(colors="#cfd6df", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#1e61bd")
        ax.grid(True, color="#2a3038", linewidth=.35)
        close = window.frame["Close"].to_numpy(dtype=float)
        n = len(close)
        x_real = np.arange(n)
        ax.fill_between(x_real, window.frame["Low"].to_numpy(float), window.frame["High"].to_numpy(float),
                        color="#7fd1ff", alpha=.16, linewidth=0)
        ax.plot(x_real, close, color="#e8f6ff", linewidth=1.0)
        lines = [f"real {100 * (close[-1] / close[0] - 1):+.1f}%"]
        paths = sims.get(window.period) or []
        if paths:
            sim_close = np.vstack([np.concatenate([[close[-1]], p["Close"].to_numpy(float)]) for p in paths])
            x_sim = np.arange(n - 1, n - 1 + sim_close.shape[1])
            for row in sim_close:
                ax.plot(x_sim, row, color="#ffc857", linewidth=.5, alpha=.35)
            p10, p50, p90 = np.percentile(sim_close, [10, 50, 90], axis=0)
            ax.fill_between(x_sim, p10, p90, color="#ffc857", alpha=.14, linewidth=0)
            ax.plot(x_sim, p50, color="#ffc857", linewidth=1.3)
            end = sim_close[:, -1] / close[-1] - 1
            lines.append(f"sim median {100 * np.median(end):+.1f}%  ·  80% {100 * np.percentile(end, 10):+.1f}% "
                         f"… {100 * np.percentile(end, 90):+.1f}%")
        ax.axvline(n - 1, color="#58ddb4", linewidth=.8, linestyle="--")
        ax.set_title(f"{window.period}  ·  {window.n} × {window.candle_label}", color="white", fontsize=9, loc="left")
        ax.text(.01, .03, "\n".join(lines), transform=ax.transAxes, fontsize=6.8, color="#cfd6df",
                bbox=dict(facecolor="black", alpha=.6, edgecolor="none"))
        stamp = "%d %b %H:%M" if window.intraday else "%d %b %y"
        ticks = [0, n - 1] + ([n - 1 + len(paths[0])] if paths else [])
        labels = [f"{window.frame.index[0]:{stamp}}", "now"] + ([f"{paths[0].index[-1]:{stamp}}"] if paths else [])
        ax.set_xticks(ticks, labels)
        ax.yaxis.tick_right()
        shown = np.concatenate([window.frame["Low"].to_numpy(float)] + ([sim_close.ravel()] if paths else []))
        shown = shown[np.isfinite(shown) & (shown > 0)]
        if len(shown) and shown.max() / shown.min() > 4:      # decades of growth: a log axis keeps the past readable
            from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))      # 50, 100, 200, 500 ... not 4x10^2
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 10 else f"{v:.3g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.text(.99, .97, "log scale", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="#8a929c")
    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")
    fig.suptitle(f"{name} — real market windows up to {asof} (white) and simulated futures (amber, median + 80% band)",
                 color="white", fontsize=12, weight="bold")
    fig.text(.5, .005, "Simulated paths: price_modeling's generator, volatility calibrated on each window's real candles. "
             "They describe a range, not a forecast.", ha="center", color="#8a929c", fontsize=7.5)
    fig.tight_layout(rect=(0, .02, 1, .96))
    fig.savefig(path, dpi=110, facecolor="black")
    plt.close(fig)


# ─── ML weighting ─────────────────────────────────────────────────────────────
def row_weight(simulated: bool, n_real: int, n_simulated: int, live: dict[str, Any]) -> float:
    """ML sample weight of one row of the live training set (real market window or simulated path).

    Today: a fixed per-row ratio from config.toml [live] (real_weight 3 vs sim_weight 1). With 11 real
    windows and 99 simulated paths that gives each real row 3x the say of a simulated one, yet the real
    rows still hold only 33 / (33 + 99) = 25% of the total weight, and that share keeps falling as
    forward_runs_per_period grows.
    TODO(human): decide what "previous market data uses a larger weight than predicted data" should mean
    here and implement it (2-6 lines): keep the per-row ratio below, or give the real rows a fixed share
    of the TOTAL weight however many simulated paths exist. n_real / n_simulated are the row counts.
    Must return a positive finite number; real rows must end up weighing more than simulated rows.
    """
    return float(live["sim_weight"] if simulated else live["real_weight"])


# ─── Orchestration ────────────────────────────────────────────────────────────
@dataclass
class LiveContext:
    cfg: dict[str, Any]
    name: str                    # instrument as typed (SAP, GBP/USD ...)
    asset_type: str
    symbol: str                  # Yahoo symbol
    results_dir: Path
    analysis_dir: Path
    engine_seed: int
    base_params: Any             # backtest_compare.GenParams
    save_state: int
    clear_outputs: Callable[[], None]
    today: date = field(default_factory=date.today)
    fetcher: Fetcher = yahoo_fetcher


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index_label="Date", date_format="%Y-%m-%d %H:%M:%S")


def run_live_mode(ctx: LiveContext) -> dict[str, Any]:
    """Option 3 end to end. Returns a small summary (windows, runs, files)."""
    cfg, live = ctx.cfg, ctx.cfg["live"]
    src_png, ana_png, run_ml, state_text = SAVE_STATES[ctx.save_state]
    history_path = ctx.analysis_dir / "live_scan_history.csv"
    ui.banner("Live market data", f"{ctx.name} · {ctx.asset_type} · Yahoo symbol {ctx.symbol} · save state "
                                   f"{ctx.save_state} ({state_text})")

    # 1. snapshot -----------------------------------------------------------------
    try:
        snapshot = build_snapshot(ctx.name, ctx.symbol, cfg, history_path)
    except LiveDataError as error:
        raise SystemExit(f"\nNo usable market data: {error}. Check the symbol on finance.yahoo.com.")
    except Exception as error:
        raise SystemExit(f"\nYahoo Finance could not be reached ({type(error).__name__}: {error}).")
    print_snapshot(snapshot, ctx.name, cfg)

    # 2. real windows --------------------------------------------------------------
    ui.section(f"Real windows · candle size picked for {live['candles_min']}-{live['candles_max']} candles each")
    allow_nonpositive = ctx.asset_type == "futures"
    cache: dict = {}
    windows: list[Window] = []
    skipped: list[str] = []
    progress = ui.Progress(len(live["periods"]), "downloading")
    for i, period in enumerate(live["periods"], start=1):
        window = fetch_window(ctx.symbol, period, cfg, ctx.today, ctx.fetcher, cache, allow_nonpositive)
        progress.update(i, period)
        if window is None:
            skipped.append(period)
        else:
            windows.append(window)
    if not windows:
        raise SystemExit(f"\nYahoo returned no window with enough bars for {ctx.symbol}.")
    ui.table([[w.period, w.span_text, f"{w.raw_bars} × {w.base}" + (f"  → ×{w.factor}" if w.factor > 1 else ""),
               w.candle_label, str(w.n), "; ".join(w.notes)] for w in windows],
             ["period", "window (exchange time)", "downloaded", "candle", "candles", "notes"], "lllrrl")
    for period in skipped:
        ui.warn(f"{period}: skipped - not enough bars at any Yahoo interval")

    # 3. simulated futures -----------------------------------------------------------
    runs_per = int(live["forward_runs_per_period"])
    sims: dict[str, list[pd.DataFrame]] = {}
    sim_vol: dict[str, float] = {}
    if runs_per:
        session = trading_session(ctx.symbol, windows)
        latest = float(windows[0].frame["Close"].iloc[-1])           # the finest window ends at the latest trade
        ui.section(f"Simulated futures · {runs_per} path(s) per period from the latest price {ui.fmt_price(latest)}")
        rows = []
        for index, window in enumerate(windows):
            paths, vol, vol_note = simulate_forward(window, runs_per, ctx.base_params, cfg, ctx.engine_seed, session,
                                                    index)
            sims[window.period], sim_vol[window.period] = paths, vol
            rows.append([window.period, f"{real_sigma_pct(window.frame):.3f}%", f"{vol:.4g}", vol_note or "—"])
        ui.table(rows, ["period", "real sigma / candle", "vol_index", "note"], "lrrl")

    # 4. write runs (only now: a failed download above leaves the previous outputs untouched) -----------
    ctx.clear_outputs()
    meta_rows: list[dict[str, Any]] = []
    labels: dict[int, str] = {}
    chart_runs: set[int] = set()
    vol_by_run: dict[int, float] = {}
    jobs: list[tuple[int, Window, pd.DataFrame, bool]] = []
    run = 0
    for window in windows:
        run += 1
        jobs.append((run, window, window.frame, False))
    for window in windows:
        for path in sims.get(window.period, []):
            run += 1
            jobs.append((run, window, path, True))
    total = len(jobs)
    writer = ui.Progress(total, "writing runs")
    for done, (run_id, window, frame, simulated) in enumerate(jobs, start=1):
        _write_csv(frame[OHLC + (["Volume"] if "Volume" in frame else [])], ctx.results_dir / f"data_{run_id}.csv")
        kind = "SIM next" if simulated else "REAL"
        label = f"{ctx.symbol} {kind} {window.period} ({window.n} × {window.candle_label})"
        labels[run_id] = label
        draw = (not simulated) or bool(live["chart_simulated_runs"])
        if draw:
            chart_runs.add(run_id)
            if src_png:
                save_source_chart(frame, ctx.results_dir / f"image_{run_id}.png",
                                  f"{ctx.name} — {kind} {window.period} · {window.n} candles × {window.candle_label}",
                                  simulated)
        if simulated:
            vol_by_run[run_id] = sim_vol[window.period]
        meta_rows.append({
            "run": run_id, "data_source": "simulated" if simulated else "real", "period": window.period,
            "candle": window.candle_label, "yahoo_interval": window.base, "merge_factor": window.factor,
            "window_start": frame.index[0], "window_end": frame.index[-1],
            "sample_weight": row_weight(simulated, len(windows), total - len(windows), live),
        })
        writer.update(done, f"run {run_id}")

    # 5. analysis ---------------------------------------------------------------------
    from market_analysis import analyze_looped_results
    ui.section("Analysis")
    result = analyze_looped_results(ctx.results_dir, ctx.analysis_dir, render_charts=ana_png, render_source_charts=src_png,
                                    timeframe="live", chart_label=ctx.symbol, vol_by_run=vol_by_run,
                                    label_by_run=labels, chart_runs=chart_runs)
    summary_path = ctx.analysis_dir / "analysis_summary.csv"
    summary = pd.read_csv(summary_path)
    meta = pd.DataFrame(meta_rows)
    summary = summary.merge(meta, on="run", how="left")
    summary.insert(1, "instrument", ctx.name)
    summary.insert(2, "asset_type", ctx.asset_type)
    summary.insert(4, "sim_date", ctx.today.isoformat())
    summary.insert(5, "is_backtest", 0)
    summary["is_augmented"] = 0
    summary.to_csv(summary_path, index=False)
    ui.kv([("Runs analysed", f"{result['runs']} ({len(windows)} real, {result['runs'] - len(windows)} simulated)"),
           ("Candles", f"{result['bars']:,}"), ("Crossover trades", f"{result['trades']:,}")])

    window_table = pd.DataFrame([{"period": w.period, "candles": w.n, "candle": w.candle_label, "yahoo_interval": w.base,
                                  "merge_factor": w.factor, "start": w.frame.index[0], "end": w.frame.index[-1],
                                  "real_sigma_pct": real_sigma_pct(w.frame), "sim_vol_index": sim_vol.get(w.period),
                                  "notes": "; ".join(w.notes)} for w in windows])
    window_table.to_csv(ctx.analysis_dir / "live_windows.csv", index=False)
    pd.DataFrame(snapshot["rows"]).to_csv(ctx.analysis_dir / "live_snapshot.csv", index=False)
    plot_overview(ctx.name, windows, sims, ctx.analysis_dir / "live_overview.png", snapshot["asof"])
    _, scan_page = _scan_modules()
    page = scan_page.write_page(snapshot, ctx.analysis_dir / "live_scan.html", title=f"Live market snapshot · {ctx.name}",
                                footer_html=("Written by price_modeling.py option 3; one row per instrument per run is "
                                             "appended to <code>looped_analysis/live_scan_history.csv</code>, which the "
                                             "&ldquo;what changed&rdquo; section reads. Bands are volatility ranges, not "
                                             "forecasts. Nothing here is investment advice."))
    append_snapshot_history(snapshot, history_path)

    # 6. ML ----------------------------------------------------------------------------
    ml_dir = None
    if run_ml:
        ml_dir = train_live_model(ctx)
    return {"windows": windows, "sims": sims, "runs": total, "snapshot": snapshot, "page": page, "ml_dir": ml_dir}


def train_live_model(ctx: LiveContext) -> Path | None:
    """Clear the instrument's live ML folder, then train with real rows weighted above simulated rows."""
    import ml_model
    ml_root = (ctx.analysis_dir / "ml_results").resolve()
    ml_dir = (ml_root / ml_model._asset_output_name(ctx.asset_type, ctx.name) / "live").resolve()
    if ml_dir.name != "live" or ml_root not in ml_dir.parents:
        raise RuntimeError(f"refusing to clear unexpected folder {ml_dir}")
    if ml_dir.exists():
        shutil.rmtree(ml_dir)
    ui.section("ML · live folder cleared, real rows weighted "
               f"{ctx.cfg['live']['real_weight']:g} vs simulated {ctx.cfg['live']['sim_weight']:g}")
    cfg = copy.deepcopy(ctx.cfg)
    cfg["ml"]["use_sample_weight"] = True          # the weights are the point of this training
    cfg["ml"]["online_update"] = False
    try:
        ml_model.run_ml_pipeline(ctx.analysis_dir, keep_history=False, instrument=ctx.name, asset_type=ctx.asset_type,
                                 timeframe="live", config=cfg)
    except (ValueError, FileNotFoundError) as error:
        ui.warn(f"ML skipped: {error}")
        return None
    return ml_dir
