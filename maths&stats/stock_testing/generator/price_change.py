"""Market-data helpers used by the price model."""

import math
from datetime import date
from functools import lru_cache

import pandas as pd

try:
    # Reuse the yfinance import already used by the price-opening helper.
    from .price_open import yf
except ImportError:
    from price_open import yf


def _yfinance_symbol(symbol, asset_type=""):
    symbol = str(symbol).strip().upper()
    kind = str(asset_type).strip().lower()

    if kind in {"fx", "forex", "foreign exchange", "currency", "currencies"}:
        if symbol.endswith("=X"):
            return symbol
        pair = symbol.replace("/", "").replace("-", "").replace("_", "")
        return f"{pair}=X"

    if kind in {"future", "futures"} and "=" not in symbol:
        return f"{symbol}=F"

    if kind in {"crypto", "cryptocurrency", "cryptocurrencies"} and "/" in symbol:
        return symbol.replace("/", "-")

    return symbol


@lru_cache(maxsize=32)
def get_daily_percent_change(symbol, asset_type=""):
    """Return the latest daily close-to-close percentage change from Yahoo data."""
    yahoo_symbol = _yfinance_symbol(symbol, asset_type)
    history = yf.Ticker(yahoo_symbol).history(period="5d", interval="1d")
    if history.empty or "Close" not in history:
        raise ValueError(f"No daily price history returned for {yahoo_symbol}.")

    closes = history["Close"].dropna()
    if len(closes) < 2:
        raise ValueError(f"At least two daily closes are required for {yahoo_symbol}.")

    previous_close = float(closes.iloc[-2])
    latest_close = float(closes.iloc[-1])
    if previous_close == 0:
        raise ValueError(f"Previous close is zero for {yahoo_symbol}.")

    return ((latest_close / previous_close) - 1.0) * 100.0


# ─── Timeframe-aware % change ─────────────────────────────────────────────────
# get_daily_percent_change() above only ever looks at the last day. The helpers
# below measure the change over the span that matches the chosen timeframe:
#
#   1d   previous close            -> latest close   (identical to the function above)
#   5d   close 5 trading days ago  -> latest close
#   1m   close 1 calendar month ago-> latest close
#   3m   close 3 calendar months ago
#   1y   close 1 calendar year ago
#   5y   close 5 calendar years ago
#   ytd  close on/before Jan 1 of this year (markets are shut on Jan 1, so this
#        is the prior year's final close) -> latest close
#
# "close N ago" always means the last close ON OR BEFORE that calendar date, so
# weekends and holidays never break the lookup.
SUPPORTED_TIMEFRAMES = ("1d", "5d", "1m", "3m", "1y", "5y", "ytd")
_TRADING_DAY_SPANS = {"1d": 1, "5d": 5}
_CALENDAR_SPANS = {"1m": {"months": 1}, "3m": {"months": 3}, "1y": {"years": 1}, "5y": {"years": 5}}


@lru_cache(maxsize=32)
def _daily_closes(symbol, asset_type=""):
    """Full daily close history (one network call per symbol, then cached)."""
    yahoo_symbol = _yfinance_symbol(symbol, asset_type)
    history = yf.Ticker(yahoo_symbol).history(period="max", interval="1d")
    if history is None or history.empty or "Close" not in history:
        raise ValueError(f"No daily price history returned for {yahoo_symbol}.")

    closes = history["Close"].dropna()
    index = pd.DatetimeIndex(closes.index)
    if index.tz is not None:
        index = index.tz_localize(None)  # keep the exchange-local calendar date
    series = pd.Series(closes.to_numpy(dtype=float), index=index.normalize()).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    if series.empty:
        raise ValueError(f"No usable daily closes for {yahoo_symbol}.")
    return series


def timeframe_change_from_closes(closes, timeframe, today=None, as_of=None, span_days=None):
    """Compute the % change for `timeframe` from a daily close Series.

    closes     pd.Series of closes indexed by calendar date (ascending).
    today      the "now" date for forward measurements (default: date.today()).
    as_of      point-in-time mode, used by backtests: only closes strictly BEFORE
               this date are visible and the span is measured back from the day
               before it, so nothing from the cycle being tested leaks in.
    span_days  only for ytd + as_of: length of the lookback in calendar days
               (ytd has no fixed length, so the cycle's own length is used).

    Returns a dict with the raw % change ("pct"), the number of trading-day
    moves it covers, a daily-equivalent figure (pct / sqrt(trading days), equal
    to pct for 1d), and the base/end dates and prices for display.
    """
    tf = str(timeframe).strip().lower()
    if tf not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe {timeframe!r}. Use one of: {', '.join(SUPPORTED_TIMEFRAMES)}.")

    if as_of is None:
        reference = pd.Timestamp(today or date.today()).normalize()
        data = closes[closes.index <= reference]
    else:
        cutoff = pd.Timestamp(as_of).normalize()
        reference = cutoff - pd.Timedelta(days=1)
        data = closes[closes.index < cutoff]
    if data.empty:
        raise ValueError("No closes available before the reference date.")

    end_date, end_price = data.index[-1], float(data.iloc[-1])

    if tf in _TRADING_DAY_SPANS:
        moves = _TRADING_DAY_SPANS[tf]
        if len(data) < moves + 1:
            raise ValueError(f"At least {moves + 1} daily closes are required for {tf}.")
        base_date, base_price = data.index[-(moves + 1)], float(data.iloc[-(moves + 1)])
    else:
        if tf == "ytd":
            if as_of is None:
                target = pd.Timestamp(year=reference.year, month=1, day=1)
            else:
                if span_days is None:
                    raise ValueError("span_days is required for a point-in-time ytd lookback.")
                target = reference - pd.Timedelta(days=int(span_days))
        else:
            target = reference - pd.DateOffset(**_CALENDAR_SPANS[tf])
        window = data.loc[:target]
        if window.empty:
            raise ValueError(
                f"Not enough history for {tf}: data starts {data.index[0].date()}, "
                f"but the span needs {target.date()}."
            )
        base_date, base_price = window.index[-1], float(window.iloc[-1])

    if base_price == 0:
        raise ValueError(f"Base close on {base_date.date()} is zero.")

    trading_days = int(((data.index > base_date) & (data.index <= end_date)).sum())
    pct = (end_price / base_price - 1.0) * 100.0
    return {
        "timeframe": tf,
        "pct": pct,
        "trading_days": trading_days,
        "daily_equivalent_pct": pct / math.sqrt(max(trading_days, 1)),
        "base_date": base_date.date(),
        "base_price": base_price,
        "end_date": end_date.date(),
        "end_price": end_price,
    }


def get_timeframe_change_details(symbol, asset_type="", timeframe="1d", as_of=None, span_days=None, today=None):
    """Download (cached) Yahoo daily closes and return the details dict above."""
    closes = _daily_closes(str(symbol), str(asset_type))
    return timeframe_change_from_closes(closes, timeframe, today=today, as_of=as_of, span_days=span_days)


def get_timeframe_percent_change(symbol, asset_type="", timeframe="1d", as_of=None, span_days=None, today=None):
    """Raw % change over the span that matches `timeframe` (see module notes)."""
    return get_timeframe_change_details(symbol, asset_type, timeframe, as_of, span_days, today)["pct"]