"""Backtest comparison: real Yahoo price (ghost halo) vs. simulated paths.

For every backtest cycle this module
  1. pulls the REAL price history for the cycle's calendar window from Yahoo,
  2. aligns it to the simulated bars,
  3. scores how closely the simulation tracks reality on a -1 .. +1 scale, and
  4. draws ONE combined figure (backtest_plot.png): simulated candles in front,
     a translucent "ghost halo" of the real price behind them.

Accuracy index (per cycle, all inputs rebased to % change from the first
comparable bar so the simulated and real price levels do not matter)
------------------------------------------------------------------------------
Each component is mapped to [-1, +1] (+1 = identical, -1 = fully divergent)
and combined with the weights in INDEX_WEIGHTS:

  path       40%  RMSE between the simulated and real close paths, relative to
                  the larger of the two high-low ranges.
  direction  25%  share of bars whose candle direction (up/down) matches.
  net_move   20%  difference between the simulated and real net period move.
  range      15%  ratio of simulated to real high-low range (3x off = -1).

The overall index is the mean of the per-cycle indices that could be scored.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import time
import tomllib
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

# ─── Index configuration ──────────────────────────────────────────────────────
INDEX_WEIGHTS: dict[str, float] = {
    "path": 0.40,
    "direction": 0.25,
    "net_move": 0.20,
    "range": 0.15,
}
# RMSE / net-move error equal to this fraction of the combined range scores -1.
PATH_TOLERANCE = 0.5
# A cycle needs at least this share of its bars covered by real data to be scored.
MIN_COVERAGE = 0.5
MIN_BARS_ABSOLUTE = 8

# ─── Real-data configuration ──────────────────────────────────────────────────
NS_PER_MIN = 60 * 10**9
NS_PER_DAY = 86_400 * 10**9

# Yahoo interval -> (bar length in ns, max age in days Yahoo serves it; None = no limit).
YAHOO_INTERVALS: dict[str, tuple[int, int | None]] = {
    "5m":  (5 * NS_PER_MIN, 58),
    "15m": (15 * NS_PER_MIN, 58),
    "1h":  (60 * NS_PER_MIN, 725),
    "1d":  (NS_PER_DAY, None),
    "1wk": (7 * NS_PER_DAY, None),
}

# A "bar spec" describes the simulated candles (built by price_modeling.interval_spec):
#   {"kind": "intraday", "minutes": N}   start-labelled, covers N minutes of the session
#   {"kind": "bday", "days": N}          end-labelled at the 16:00 close of N business days
#                                        (days=None: ytd's ~50 sampled days)
#   {"kind": "week", "weeks": N}         start-labelled at the Monday 09:00 open, N weeks long
# Defaults reproduce the fixed spacing each timeframe used before intervals were selectable.
DEFAULT_BARS: dict[str, dict[str, Any]] = {
    "1d": {"kind": "intraday", "minutes": 10},
    "5d": {"kind": "intraday", "minutes": 60},
    "1m": {"kind": "intraday", "minutes": 240},
    "3m": {"kind": "bday", "days": 1},
    "1y": {"kind": "week", "weeks": 1},
    "5y": {"kind": "week", "weeks": 1},
    "ytd": {"kind": "bday", "days": None},
}


def bar_step_ns(bar: dict[str, Any]) -> int:
    """Nominal length of one simulated candle in nanoseconds."""
    if bar["kind"] == "intraday":
        return int(bar["minutes"]) * NS_PER_MIN
    if bar["kind"] == "week":
        return int(bar["weeks"]) * 7 * NS_PER_DAY
    return int(bar.get("days") or 1) * NS_PER_DAY


def yahoo_interval_order(step_ns: int) -> list[str]:
    """Yahoo intervals to try for a candle length, best fit first.

    1. the largest interval that divides the candle evenly (5m, 15m, 1h, 1d... aggregate
       cleanly into it), then 2. other intervals not longer than the candle (largest first),
    then 3. coarser ones as a last resort (they must still pass the coverage check).
    """
    ordered = sorted(YAHOO_INTERVALS, key=lambda name: YAHOO_INTERVALS[name][0])
    dividing = [n for n in ordered if YAHOO_INTERVALS[n][0] <= step_ns and step_ns % YAHOO_INTERVALS[n][0] == 0]
    shorter = [n for n in ordered if YAHOO_INTERVALS[n][0] <= step_ns and n not in dividing]
    coarser = [n for n in ordered if YAHOO_INTERVALS[n][0] > step_ns]
    return dividing[::-1] + shorter[::-1] + coarser


Fetcher = Callable[[str, str, pd.Timestamp, pd.Timestamp], "pd.DataFrame | None"]

# Floor mode (not futures): a bar, gap or wick that would reach 0 or below is set to this share of its
# reference price instead (bars that stay above 0 are left exactly as generated).
MIN_PRICE_FRACTION = 0.01

HALO_COLOR = "#7fd1ff"
UP_COLOR = "#21d6a2"
DOWN_COLOR = "#ff6b77"
SIM_LINE_COLOR = "#ffc857"


# ─── Fetching ─────────────────────────────────────────────────────────────────
def _fetch_history(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    """Download OHLC from Yahoo as a tz-naive frame (exchange wall-clock time)."""
    import yfinance as yf  # imported lazily so the module loads without yfinance

    history = yf.Ticker(symbol).history(
        start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval=interval
    )
    if history is None or history.empty:
        return None
    frame = history[["Open", "High", "Low", "Close"]].dropna().sort_index().astype(float)
    if frame.empty:
        return None
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_localize(None)  # keep exchange-local wall-clock time
    if interval in ("1d", "1wk"):
        # Daily/weekly bars are stamped 00:00; park them at midday so they fall
        # inside the matching simulated bar (09:00 weekly open / 16:00 daily close).
        index = index.normalize() + pd.Timedelta(hours=12)
    frame.index = index
    return frame


# ─── Alignment ────────────────────────────────────────────────────────────────
def _ns(values: Any) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[ns]").astype("int64")


def _first_day_of_first_candle(ts: pd.DatetimeIndex, bar: dict[str, Any]) -> pd.Timestamp:
    """First business day covered by the first end-labelled (bday) candle."""
    days = int(bar.get("days") or 1)
    return pd.bdate_range(end=ts[0].normalize(), periods=days)[0]


def align_real_to_sim(real: pd.DataFrame, timestamps: Any, bar: dict[str, Any]) -> pd.DataFrame:
    """Aggregate real bars into the simulated candle windows (NaN where no real bar).

    start-labelled candles (intraday, week) own [ts_i, ts_i+1); intraday ones are capped
    at one candle length so overnight data never leaks into the last candle of a day.
    end-labelled candles (bday) own (ts_i-1, ts_i]; the first one reaches back to the
    first business day of its group.
    """
    ts = pd.DatetimeIndex(timestamps)
    n = len(ts)
    ts_ns = _ns(ts)
    step = bar_step_ns(bar)
    t = _ns(real.index)

    if bar["kind"] == "bday":
        hi = ts_ns
        lo = np.append(_ns([_first_day_of_first_candle(ts, bar)])[0], ts_ns[:-1])
        bucket = np.searchsorted(hi, t, side="left")
        ok = bucket < n
        ok &= t > lo[np.clip(bucket, 0, n - 1)]
    else:
        lo = ts_ns
        nxt = np.append(ts_ns[1:], ts_ns[-1] + step)
        hi = np.minimum(nxt, ts_ns + step) if bar["kind"] == "intraday" else nxt
        bucket = np.searchsorted(lo, t, side="right") - 1
        ok = bucket >= 0
        ok &= t < hi[np.clip(bucket, 0, n - 1)]

    frame = real.loc[ok].copy()
    frame["bucket"] = bucket[ok]
    grouped = frame.groupby("bucket")
    aggregated = pd.DataFrame({
        "Open": grouped["Open"].first(),
        "High": grouped["High"].max(),
        "Low": grouped["Low"].min(),
        "Close": grouped["Close"].last(),
    }).reindex(range(n))
    aggregated.index = ts
    return aggregated


def fetch_real_for_cycle(
    symbol: str,
    bar: dict[str, Any],
    timestamps: Any,
    today: date,
    fetcher: Fetcher | None = None,
) -> tuple[pd.DataFrame | None, str, str]:
    """Return (aligned_real, interval_used, note). aligned_real is None if unavailable."""
    fetcher = fetcher or _fetch_history
    ts = pd.DatetimeIndex(timestamps)
    n = len(ts)
    needed = max(MIN_BARS_ABSOLUTE, math.ceil(MIN_COVERAGE * n))
    first = (_first_day_of_first_candle(ts, bar) if bar["kind"] == "bday" else ts[0].normalize())
    last = ts[-1].normalize()
    start = first - pd.Timedelta(days=1)
    end = min(last + pd.Timedelta(days=2), pd.Timestamp(today) + pd.Timedelta(days=1))
    age_days = (pd.Timestamp(today) - start).days  # age of the actual request, not the first bar

    tried: list[str] = []
    for interval in yahoo_interval_order(bar_step_ns(bar)):
        max_age = YAHOO_INTERVALS[interval][1]
        if max_age is not None and age_days > max_age:
            tried.append(f"{interval}: older than Yahoo serves")
            continue
        try:
            history = fetcher(symbol, interval, start, end)
        except Exception as error:  # network/Yahoo errors must not kill the whole run
            tried.append(f"{interval}: {type(error).__name__}")
            continue
        if history is None or history.empty:
            tried.append(f"{interval}: no data")
            continue
        aligned = align_real_to_sim(history, ts, bar)
        covered = int(aligned["Close"].notna().sum())
        if covered >= needed:
            return aligned, interval, ""
        tried.append(f"{interval}: only {covered}/{n} bars covered")
    return None, "", "; ".join(tried) if tried else "no interval available"


# ─── Scoring ──────────────────────────────────────────────────────────────────
def _tolerance_score(error: float, scale: float) -> float:
    """Map an error to [-1, 1]: 0 error -> +1, error >= PATH_TOLERANCE*scale -> -1."""
    return 1.0 - 2.0 * min(1.0, error / (PATH_TOLERANCE * scale))


OHLC_FIELDS = ["Open", "High", "Low", "Close"]
_O, _H, _L, _C = range(4)                        # column positions in an OHLC array


def ohlc_array(frame: "pd.DataFrame | np.ndarray") -> np.ndarray:
    """(n, 4) float array of Open, High, Low, Close (an array is passed through unchanged).

    The scorers below work on arrays: on 100-bar tables pandas column lookups cost ~20 ms per
    comparison while the arithmetic costs microseconds, which made the chance level and the
    volatility search slow at hundreds of cycles. Convert a frame once and reuse the array.
    """
    if isinstance(frame, np.ndarray):
        return frame
    return frame[OHLC_FIELDS].to_numpy(dtype=float)


def _wick_ratio(a: np.ndarray) -> float:
    high, low = a[:, _H], a[:, _L]
    body = np.abs(a[:, _C] - a[:, _O])
    body_mean = float(np.nanmean(body)) if len(body) else float("nan")
    if not np.isfinite(body_mean) or body_mean <= 1e-12:
        return float("nan")
    return float(np.nanmean((high - low) - body) / body_mean)


def wick_to_body_ratio(frame: "pd.DataFrame | np.ndarray") -> float:
    """Mean wick length / mean candle body (scale-free, so sim and real need no rebasing)."""
    return _wick_ratio(ohlc_array(frame))


def score_cycle(sim: "pd.DataFrame | np.ndarray", real: "pd.DataFrame | np.ndarray",
                exact_weights: dict[str, float] | None = None) -> dict[str, Any]:
    """Compare one simulated cycle with its aligned real data -> accuracy index (frames or ohlc_array()s)."""
    S, R = ohlc_array(sim), ohlc_array(real)
    valid = ~(np.isnan(R[:, _O]) | np.isnan(R[:, _C]))
    positions = np.flatnonzero(valid)
    needed = max(MIN_BARS_ABSOLUTE, math.ceil(MIN_COVERAGE * len(S)))
    if len(positions) < needed:
        return {"status": "insufficient real data", "valid_bars": int(len(positions))}

    base = int(positions[0])
    sim_base, real_base = float(S[base, _O]), float(R[base, _O])
    if sim_base <= 0 or real_base <= 0:
        return {"status": "non-positive base price", "valid_bars": int(len(positions))}

    s, r = S[positions], R[positions]

    def pct(a: np.ndarray, column: int, base_value: float) -> np.ndarray:
        return (a[:, column] / base_value - 1.0) * 100.0

    sim_close, real_close = pct(s, _C, sim_base), pct(r, _C, real_base)
    sim_range = pct(s, _H, sim_base).max() - pct(s, _L, sim_base).min()
    real_range = pct(r, _H, real_base).max() - pct(r, _L, real_base).min()
    scale = max(sim_range, real_range, 1e-9)

    rmse = float(np.sqrt(np.mean((sim_close - real_close) ** 2)))
    path = _tolerance_score(rmse, scale)

    sim_up = (s[:, _C] >= s[:, _O])
    real_up = (r[:, _C] >= r[:, _O])
    direction = 2.0 * float(np.mean(sim_up == real_up)) - 1.0

    net_move = _tolerance_score(abs(float(sim_close[-1] - real_close[-1])), scale)

    # [BARS] criterion: share of paired bars whose candle direction matches (idea 2).
    direction_match_pct = float(np.mean(sim_up == real_up))
    # [PRICE] criterion: 1 - |sim_close - real_close| / real_close (idea 3). The real close is
    # rebased to the simulation start so the two price levels are comparable (see module notes).
    real_close_last = float(r[-1, _C]) * (sim_base / real_base)
    sim_close_last = float(s[-1, _C])
    net_price_match = (1.0 - abs(sim_close_last - real_close_last) / abs(real_close_last)
                       if real_close_last != 0 else float("nan"))
    # Wick-to-body ratio (ideas 'wick-ratio self-correction' and EXACT delta term).
    sim_wick, real_wick = _wick_ratio(s), _wick_ratio(r)
    if np.isfinite(sim_wick) and np.isfinite(real_wick) and sim_wick > 0 and real_wick > 0:
        wick_score = 1.0 - 2.0 * min(1.0, abs(math.log(sim_wick / real_wick)) / math.log(3.0))
    else:
        wick_score = float("nan")

    if sim_range <= 0 or real_range <= 0:
        range_score = 1.0 if sim_range == real_range else -1.0
    else:
        range_score = 1.0 - 2.0 * min(1.0, abs(math.log(sim_range / real_range)) / math.log(3.0))

    components = {"path": path, "direction": direction, "net_move": net_move, "range": range_score}
    index = sum(INDEX_WEIGHTS[name] * value for name, value in components.items())
    ew = DEFAULT_CONFIG["exact"] if exact_weights is None else exact_weights
    net_clip = float(np.clip(net_price_match, -1.0, 1.0)) if np.isfinite(net_price_match) else -1.0
    exact_score = (ew["alpha"] * path + ew["beta"] * direction + ew["gamma"] * net_clip
                   + ew["delta"] * (wick_score if np.isfinite(wick_score) else 0.0))
    return {
        "status": "scored",
        "index": float(np.clip(index, -1.0, 1.0)),
        "direction_match_pct": direction_match_pct,
        "net_price_match": float(net_price_match),
        "net_price_error_pct": float(100.0 * abs(sim_close_last - real_close_last) / abs(real_close_last))
                               if real_close_last != 0 else float("nan"),
        "path_error": float(rmse / scale),
        "exact_score": float(exact_score),
        "wick_ratio_sim": float(sim_wick), "wick_ratio_real": float(real_wick),
        "score_wick": float(wick_score),
        "valid_bars": int(len(positions)),
        "base_pos": base,
        "rmse_pct": rmse,
        "sim_range_pct": float(sim_range),
        "real_range_pct": float(real_range),
        "sim_net_pct": float(sim_close[-1]),
        "real_net_pct": float(real_close[-1]),
        **{f"score_{name}": float(value) for name, value in components.items()},
    }


# ─── real_indicator (terminal only) ───────────────────────────────────────────
def real_indicator(sim: "pd.DataFrame | np.ndarray", real: "pd.DataFrame | np.ndarray") -> dict[str, Any]:
    """How close one simulated run is to the real price, as a -1 .. +1 index.

    TERMINAL ONLY: the result is printed by run_backtest_comparison() and is never
    written to the plot, the CSVs, or anything the ML step reads.

    Method
    ------
    1. Every bar's Open, High, Low and Close (simulated and real) becomes a log
       distance from its own first comparable bar's Open: ln(price / first_open).
       Log is symmetric for up/down moves, additive over time, and stays
       well-behaved on big 1y / 5y moves; rebasing removes the price-level gap
       (the simulation starts at today's price, the real data at a historical one).
    2. MSE = mean squared difference between simulated and real over every bar x
       (O, H, L, C).
    3. A raw MSE has no natural scale, so it is divided by the MSE two UNRELATED
       paths with the same spread and level would give:
           reference = var(sim) + var(real) + (mean(sim) - mean(real))^2
       1 - MSE/reference is Lin's concordance correlation: +1 for an exact match,
       0 for unrelated paths, negative for inverse ones.
    4. Mapped to the requested scale (+1 = 100% correlated, -1 = 0 correlation):
           real_indicator = 2 * max(0, concordance) - 1  =  1 - 2 * min(1, MSE/reference)
    """
    S, R = ohlc_array(sim), ohlc_array(real)
    present = ~np.isnan(R).any(axis=1)
    positions = np.flatnonzero(present)
    needed = max(MIN_BARS_ABSOLUTE, math.ceil(MIN_COVERAGE * len(S)))
    if len(positions) < needed:
        return {"status": "insufficient real data", "valid_bars": int(len(positions))}

    base = int(positions[0])
    sim_ohlc = S[positions]
    real_ohlc = R[positions]
    sim_base, real_base = float(S[base, _O]), float(R[base, _O])
    if min(sim_ohlc.min(), real_ohlc.min(), sim_base, real_base) <= 0:
        return {"status": "non-positive price", "valid_bars": int(len(positions))}

    x = np.log(sim_ohlc / sim_base).ravel()
    y = np.log(real_ohlc / real_base).ravel()
    mse = float(np.mean((x - y) ** 2))
    reference = float(x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    if reference <= 1e-18:                       # both paths completely flat
        concordance = 1.0 if mse <= 1e-18 else 0.0
    else:
        concordance = 1.0 - mse / reference
    index = 2.0 * max(0.0, concordance) - 1.0
    return {
        "status": "scored",
        "index": float(np.clip(index, -1.0, 1.0)),
        "concordance": float(concordance),
        "mse": mse,
        "rmse_pct": math.sqrt(mse) * 100.0,      # typical O/H/L/C log distance, in ~%
        "valid_bars": int(len(positions)),
    }


def chance_baseline(items: list[dict[str, Any]], max_others: int = 20, max_pairs: int = 2000) -> dict[str, float]:
    """Score every simulation against the real prices of OTHER cycles (same candle grid, unrelated path).

    A generator that knows nothing about the real path scores like this on average, so it is the
    chance level of both indices: (overall index - chance) is the skill, and an index far BELOW
    chance means the simulation is broken (e.g. prices collapsing to zero), not just uninformed.
    At most `max_pairs` pairs in total (identical up to 100 cycles): a mean of 2,000 pairs is already
    precise to about +-0.01, and 20 pairs per cycle cost ~2.5 minutes at 1,000 cycles.
    """
    usable = [it for it in items if it.get("real") is not None]
    per_item = max(1, min(max_others, max_pairs // max(1, len(usable))))
    sims = [ohlc_array(it["sim"]) for it in usable]      # convert once, not once per pair
    reals = [ohlc_array(it["real"]) for it in usable]
    acc, ind = [], []
    for i in range(len(usable)):
        others = [j for j in range(len(usable)) if j != i and len(reals[j]) == len(sims[i])]
        step = max(1, len(others) // per_item)
        for j in others[::step][:per_item]:
            score = score_cycle(sims[i], reals[j])
            if score.get("status") == "scored":
                acc.append(score["index"])
            indicator = real_indicator(sims[i], reals[j])
            if indicator.get("status") == "scored":
                ind.append(indicator["index"])
    return {"accuracy_index": float(np.mean(acc)) if acc else float("nan"),
            "real_indicator": float(np.mean(ind)) if ind else float("nan"), "pairs": len(acc)}


def _print_real_indicator(results: list[tuple[Any, dict[str, Any]]], total: int, chance: float = float("nan")) -> float:
    """Print the per-cycle and overall real_indicator; return the overall value."""
    verbose = total <= 100
    step = max(1, total // 20)
    print("\nreal_indicator  (terminal only: log-OHLC mean-squared distance to the real price; "
          "+1 = 100% correlated, -1 = 0 correlation)")
    values = []
    for position, (cycle_id, result) in enumerate(results, start=1):
        scored = result.get("status") == "scored"
        if scored:
            values.append(result["index"])
        if verbose or position == 1 or position % step == 0 or position == total:
            if scored:
                print(f"  Cycle {cycle_id}: real_indicator {result['index']:+.3f}   "
                      f"(log-RMSE {result['rmse_pct']:.3f}%, {result['valid_bars']} bars compared)")
            else:
                print(f"  Cycle {cycle_id}: real_indicator n/a — {result.get('status', 'no real data')}")
    overall = float(np.mean(values)) if values else float("nan")
    shown = f"{overall:+.3f}" if np.isfinite(overall) else "n/a"
    print(f"  Overall real_indicator: {shown}  ({len(values)} of {total} cycles)")
    if np.isfinite(chance) and np.isfinite(overall):
        print(f"  Chance level:           {chance:+.3f}  (the same simulations scored against OTHER cycles' real "
              f"prices)  ->  skill {overall - chance:+.3f}")
    return overall


# ─── Plotting ─────────────────────────────────────────────────────────────────
def _score_color(value: float) -> str:
    return "#58ddb4" if value > 0.25 else "#ffc857" if value > -0.25 else "#ff6b77"


def _tick_format(timeframe: str) -> str:
    return {"1d": "%H:%M", "1y": "%b %Y", "5y": "%b %Y"}.get(timeframe, "%d %b")


def _draw_panel(ax: Any, item: dict[str, Any], total: int, timeframe: str, compact: bool) -> None:
    sim: pd.DataFrame = item["sim"]
    real: pd.DataFrame | None = item.get("real")
    score: dict[str, Any] | None = item.get("score")
    n = len(sim)
    x = np.arange(n)

    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=6 if compact else 8)
    for spine in ax.spines.values():
        spine.set_color("#1e61bd")
    ax.grid(True, color="#3a414b", linewidth=.35)
    ax.yaxis.tick_right()
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")

    # Ghost halo of the real price (rebased so it starts where the simulation does).
    if real is not None and score is not None and score.get("status") == "scored":
        base = score["base_pos"]
        factor = float(sim["Open"].iloc[base]) / float(real["Open"].iloc[base])
        shown = real * factor
        high = np.ma.masked_invalid(shown["High"].to_numpy(dtype=float))
        low = np.ma.masked_invalid(shown["Low"].to_numpy(dtype=float))
        close = shown["Close"].to_numpy(dtype=float)
        ax.fill_between(x, low, high, where=~np.ma.getmaskarray(high), color=HALO_COLOR,
                        alpha=.13, linewidth=0, zorder=1)
        for width, alpha in ((9, .07), (5, .12), (2.4, .32)):
            ax.plot(x, close, color=HALO_COLOR, linewidth=width, alpha=alpha,
                    solid_capstyle="round", zorder=1.5)
        ax.plot(x, close, color="#e8f6ff", linewidth=.8, alpha=.85, zorder=1.6)

    # Simulated price in front.
    opens, highs = sim["Open"].to_numpy(float), sim["High"].to_numpy(float)
    lows, closes = sim["Low"].to_numpy(float), sim["Close"].to_numpy(float)
    if compact:
        ax.fill_between(x, lows, highs, color=SIM_LINE_COLOR, alpha=.18, linewidth=0, zorder=2)
        ax.plot(x, closes, color=SIM_LINE_COLOR, linewidth=1.0, zorder=3)
    else:
        for xpos in range(n):
            color = UP_COLOR if closes[xpos] >= opens[xpos] else DOWN_COLOR
            ax.vlines(xpos, lows[xpos], highs[xpos], color="#d7dde5", linewidth=.65, zorder=3)
            ax.add_patch(Rectangle((xpos - .3, min(opens[xpos], closes[xpos])), .6,
                                   abs(closes[xpos] - opens[xpos]), facecolor=color,
                                   edgecolor="#d7dde5", linewidth=.3, zorder=4))
    ax.set_xlim(-1, n)
    ax.autoscale(axis="y")

    count = min(5, n)
    ticks = np.unique(np.linspace(0, n - 1, count).round().astype(int))
    ax.set_xticks(ticks)
    ax.set_xticklabels([sim.index[i].strftime(_tick_format(timeframe)) for i in ticks],
                       rotation=0 if compact else 20, ha="center" if compact else "right")

    size = 7 if compact else 10
    window = pd.Timestamp(item["start"]).strftime("%d %b %Y")
    ax.set_title(f"Cycle {item['cycle']}/{total} · {window}", color="white", fontsize=size)
    if score is not None and score.get("status") == "scored":
        color = _score_color(score["index"])
        ax.text(.99, .97, f"{score['index']:+.2f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=size + 3, weight="bold", color=color, zorder=10,
                bbox=dict(facecolor="black", alpha=.75, edgecolor=color, boxstyle="round,pad=.25"))
        if not compact:
            ax.text(.01, .02,
                    f"path {score['score_path']:+.2f} · dir {score['score_direction']:+.2f} "
                    f"({100 * score['direction_match_pct']:.0f}% bars) · "
                    f"net {score['score_net_move']:+.2f} · range {score['score_range']:+.2f}  |  "
                    f"real {item['interval']}, {score['valid_bars']}/{n} bars",
                    transform=ax.transAxes, fontsize=6.5, color="#cfd6df", zorder=10,
                    bbox=dict(facecolor="black", alpha=.65, edgecolor="none"))
    else:
        reason = item.get("note") or (score or {}).get("status", "no real data")
        ax.text(.99, .97, "no real data", transform=ax.transAxes, ha="right", va="top",
                fontsize=size, color="#8a929c", zorder=10)
        if not compact:
            ax.text(.01, .02, reason, transform=ax.transAxes, fontsize=6.5, color="#8a929c", zorder=10)


def plot_backtest(
    items: list[dict[str, Any]],
    symbol: str,
    timeframe: str,
    overall: float,
    output_path: Path,
    interval: str | None = None,
) -> None:
    """Draw every cycle into ONE figure and save it as a single PNG."""
    total = len(items)
    compact = total > 12
    if total == 1:
        ncols, panel_w, panel_h, dpi = 1, 11.0, 5.2, 130
    elif not compact:
        ncols, panel_w, panel_h, dpi = 2, 6.4, 3.6, 120
    else:
        ncols, panel_w, panel_h, dpi = min(8, math.ceil(math.sqrt(total))), 3.5, 2.3, 90
    nrows = math.ceil(total / ncols)
    head, foot = 1.4, 0.45
    fig_w, fig_h = panel_w * ncols, panel_h * nrows + head + foot

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False, facecolor="black")
    flat = axes.ravel()
    for ax, item in zip(flat, items):
        _draw_panel(ax, item, total, timeframe, compact)
    for ax in flat[total:]:
        ax.axis("off")

    scored = [item["score"]["index"] for item in items
              if item.get("score") and item["score"].get("status") == "scored"]
    overall_text = f"{overall:+.3f}" if np.isfinite(overall) else "n/a"
    candle_text = f" · {interval} candles" if interval else ""
    fig.suptitle(f"{symbol} — {timeframe.upper()}{candle_text} backtest: simulated vs real price",
                 color="white", fontsize=15, weight="bold", y=1 - .1 / fig_h, va="top")
    fig.text(.5, 1 - .62 / fig_h,
             f"Overall accuracy index {overall_text}  ({len(scored)} of {total} cycles scored)   "
             f"·   +1 = identical   −1 = fully divergent",
             color=_score_color(overall) if np.isfinite(overall) else "#8a929c",
             ha="center", va="top", fontsize=10)
    fig.legend(
        handles=[
            Line2D([], [], color=HALO_COLOR, linewidth=6, alpha=.45),
            Line2D([], [], color=UP_COLOR, marker="s", linestyle="None"),
            Line2D([], [], color=DOWN_COLOR, marker="s", linestyle="None"),
        ] if not compact else [
            Line2D([], [], color=HALO_COLOR, linewidth=6, alpha=.45),
            Line2D([], [], color=SIM_LINE_COLOR, linewidth=1.2),
        ],
        labels=["Real price (ghost halo, rebased to simulation start)", "Simulated up", "Simulated down"]
        if not compact else ["Real price (ghost halo, rebased to simulation start)", "Simulated close"],
        loc="upper center", ncol=3, fontsize=8, facecolor="#111111", edgecolor="#aab2bd",
        labelcolor="white", framealpha=.9, bbox_to_anchor=(.5, 1 - .95 / fig_h),
    )
    weights = " / ".join(f"{name} {int(w * 100)}%" for name, w in INDEX_WEIGHTS.items())
    fig.text(.01, .12 / fig_h, f"Index weights: {weights}. Real data aligned to simulated bars; "
             f"both rebased to % change from the first comparable bar.",
             color="white", fontsize=7)
    fig.tight_layout(rect=(0, foot / fig_h, 1, 1 - head / fig_h))
    fig.savefig(output_path, dpi=dpi, facecolor="black")
    plt.close(fig)


# ─── Orchestration ────────────────────────────────────────────────────────────
def run_backtest_comparison(
    cycles: list[dict[str, Any]],
    symbol: str,
    timeframe: str,
    output_dir: str | Path,
    today: date | None = None,
    fetcher: Fetcher | None = None,
    bar: dict[str, Any] | None = None,
    interval: str | None = None,
    prefetched: dict[Any, tuple] | None = None,
    file_suffix: str = "",
) -> dict[str, Any]:
    """Score every backtest cycle against real prices and write ONE combined plot.

    prefetched: {cycle: (aligned_real, interval_used, note)} re-used instead of calling Yahoo again
    (the self-correction loop scores the same real data many times). file_suffix lets the baseline
    ("_null") and the self-corrected pass write side by side.


    cycles: list of {"cycle": int, "sim": OHLC DataFrame indexed by timestamp,
                     "start": date, "end": date | None}
    Writes (into output_dir, which must already exist):
        backtest_plot.png          the single combined figure
        backtest_accuracy.csv      per-cycle component scores + an OVERALL row
        backtest_real_prices.csv   aligned real OHLC next to simulated OHLC
    """
    bar = bar or DEFAULT_BARS[timeframe]
    output = Path(output_dir)
    if not output.is_dir():
        raise FileNotFoundError(f"Analysis output folder does not exist (not creating it): {output}")
    today = today or date.today()
    total = len(cycles)
    verbose = total <= 100
    step = max(1, total // 20)

    items: list[dict[str, Any]] = []
    price_rows: list[pd.DataFrame] = []
    indicator_results: list[tuple[Any, dict[str, Any]]] = []   # real_indicator: terminal only
    print(f"\nFetching real {symbol} prices for {total} backtest cycle(s)...")
    for position, cycle in enumerate(cycles, start=1):
        sim: pd.DataFrame = cycle["sim"]
        if prefetched is not None and cycle["cycle"] in prefetched:
            real, real_interval, note = prefetched[cycle["cycle"]]
        else:
            real, real_interval, note = fetch_real_for_cycle(symbol, bar, sim.index, today, fetcher)
        score = score_cycle(sim, real) if real is not None else None
        indicator_results.append(
            (cycle["cycle"], real_indicator(sim, real) if real is not None
             else {"status": "no real data at this candle size"})   # details are in the fetch lines above
        )
        item = {**cycle, "real": real, "interval": real_interval, "note": note, "score": score}
        items.append(item)

        if verbose or position == 1 or position % step == 0 or position == total:
            if score and score["status"] == "scored":
                print(f"  Cycle {cycle['cycle']}: index {score['index']:+.3f}  (real {real_interval}, "
                      f"{score['valid_bars']}/{len(sim)} bars)")
            else:
                why = note or (score or {}).get("status", "no real data")
                print(f"  Cycle {cycle['cycle']}: not scored — {why}")

        if real is not None:
            aligned = pd.DataFrame({
                "cycle": cycle["cycle"], "bar": np.arange(1, len(sim) + 1), "Date": sim.index,
                "sim_open": sim["Open"].to_numpy(), "sim_high": sim["High"].to_numpy(),
                "sim_low": sim["Low"].to_numpy(), "sim_close": sim["Close"].to_numpy(),
                "real_open": real["Open"].to_numpy(), "real_high": real["High"].to_numpy(),
                "real_low": real["Low"].to_numpy(), "real_close": real["Close"].to_numpy(),
                "real_interval": real_interval,
            })
            price_rows.append(aligned)

    chance = chance_baseline(items)
    real_indicator_overall = _print_real_indicator(indicator_results, total, chance["real_indicator"])

    scored_values = [item["score"]["index"] for item in items
                     if item["score"] and item["score"]["status"] == "scored"]
    overall = float(np.mean(scored_values)) if scored_values else float("nan")

    rows = []
    for item in items:
        score = item["score"] or {"status": item["note"] or "no real data"}
        start = pd.Timestamp(item["start"]).date()
        rows.append({
            "cycle": item["cycle"], "symbol": symbol, "timeframe": timeframe,
            "candle_interval": interval or "default", "window_start": start.isoformat(),
            "window_end": (pd.Timestamp(item["end"]).date().isoformat() if item.get("end")
                           else item["sim"].index[-1].date().isoformat()),
            "status": score.get("status"), "real_interval": item["interval"],
            "valid_bars": score.get("valid_bars"), "total_bars": len(item["sim"]),
            "accuracy_index": score.get("index"),
            "score_path": score.get("score_path"), "score_direction": score.get("score_direction"),
            "score_net_move": score.get("score_net_move"), "score_range": score.get("score_range"),
            "rmse_pct": score.get("rmse_pct"), "path_error": score.get("path_error"),
            "bar_direction_match_pct": (100.0 * score["direction_match_pct"]
                                        if score.get("direction_match_pct") is not None else None),
            "net_price_match": score.get("net_price_match"),
            "net_price_error_pct": score.get("net_price_error_pct"),
            "exact_score": score.get("exact_score"), "score_wick": score.get("score_wick"),
            "vol_index": item.get("vol_index"),
            "sim_net_pct": score.get("sim_net_pct"), "real_net_pct": score.get("real_net_pct"),
            "sim_range_pct": score.get("sim_range_pct"), "real_range_pct": score.get("real_range_pct"),
        })
    rows.append({
        "cycle": "OVERALL", "symbol": symbol, "timeframe": timeframe,
        "candle_interval": interval or "default",
        "status": f"{len(scored_values)} of {total} cycles scored", "accuracy_index": overall,
    })
    accuracy_path = output / f"backtest_accuracy{file_suffix}.csv"
    accuracy_frame = pd.DataFrame(rows)
    accuracy_frame.to_csv(accuracy_path, index=False)
    # Same table under the name used by the self-correction notes (adds nothing new; one place to look).
    accuracy_frame.to_csv(output / f"rewind_accuracy_summary{file_suffix}.csv", index=False)
    prices_path = output / f"backtest_real_prices{file_suffix}.csv"
    (pd.concat(price_rows, ignore_index=True) if price_rows else pd.DataFrame()).to_csv(prices_path, index=False)

    plot_path = output / f"backtest_plot{file_suffix}.png"
    plot_backtest(items, symbol, timeframe, overall, plot_path, interval)

    summary = (f"{overall:+.3f}" if np.isfinite(overall) else "n/a")
    print(f"\nBacktest accuracy index for {symbol}: {summary}  "
          f"({len(scored_values)} of {total} cycles scored)")
    if np.isfinite(chance["accuracy_index"]) and np.isfinite(overall):
        print(f"  chance level {chance['accuracy_index']:+.3f} (simulations scored against other cycles' real prices)"
              f"  ->  skill {overall - chance['accuracy_index']:+.3f}")
    print(f"Saved single backtest plot : {plot_path}")
    print(f"Saved accuracy table       : {accuracy_path}")
    direction_values = [item["score"]["direction_match_pct"] for item in items
                        if item["score"] and item["score"].get("status") == "scored"]
    return {"overall_index": overall, "scored": len(scored_values), "total": total,
            "overall_direction_match_pct": float(np.mean(direction_values)) if direction_values else float("nan"),
            "real_indicator": real_indicator_overall,   # in-memory only, never written to disk
            "chance": chance,                           # chance level of both indices (terminal + in-memory)
            "items": items,                             # sim + aligned real per cycle (re-used by the search)
            "plot": str(plot_path), "accuracy_csv": str(accuracy_path), "real_prices_csv": str(prices_path)}

# ══════════════════════════════════════════════════════════════════════════════
#  SELF-CORRECTION ENGINE
#  price_modeling -> CSV/PNG -> market_analysis -> backtest_compare (this file) -> search
#  -> adjusted generator parameters -> repeat until an equilibrium is found.
# ══════════════════════════════════════════════════════════════════════════════
from scipy import optimize as spo  # noqa: E402
from scipy import stats as sps     # noqa: E402

# Files written to the analysis folder that must SURVIVE the per-run clean-up.
PERSISTENT_FILES = (
    "vol_equilibrium.csv", "vol_history.csv", "vol_search_audit.csv",
    "regime_vol_multipliers.csv", "equilibrium_log.txt", "recal_state.json",
    "live_scan_history.csv",
)

# Every tunable (and its allowed range) lives in project_config.py; config.toml overrides any key.
# Re-exported here so existing callers (bc.load_config, bc.DEFAULT_CONFIG) keep working.
from project_config import (  # noqa: E402
    CONFIG_FILENAME, DEFAULT_CONFIG, ConfigError, _deep_merge, clamp_vol_index,
    load_config as _load_config,
)


def apply_scoring_config(cfg: dict[str, Any]) -> None:
    """Point the accuracy-index constants at config.toml [scoring] (module-level so every scorer sees them)."""
    global PATH_TOLERANCE, MIN_COVERAGE, MIN_BARS_ABSOLUTE
    scoring = cfg["scoring"]
    INDEX_WEIGHTS.update({"path": scoring["w_path"], "direction": scoring["w_direction"],
                          "net_move": scoring["w_net_move"], "range": scoring["w_range"]})
    PATH_TOLERANCE = float(scoring["path_tolerance"])
    MIN_COVERAGE = float(scoring["min_coverage"])
    MIN_BARS_ABSOLUTE = int(scoring["min_bars_absolute"])


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Validated config.toml (see project_config.load_config); also applies the [scoring] section."""
    cfg = _load_config(path)
    apply_scoring_config(cfg)
    return cfg


# ─── Generator parameters ─────────────────────────────────────────────────────
@dataclass
class GenParams:
    """Everything the bar generator can be tuned on. Defaults reproduce the original model."""
    vol_index: float = 3.6
    wick_factor: float = 3.5 / 15
    wick_bull: float = 1.0
    wick_bear: float = 1.0
    jump_prob: float = 0.0
    jump_size: float = 2.0            # mean of the Exponential jump multiplier (multiplier = 1 + Exp)
    ar1: float = 0.0
    gap_prob: float = 0.0
    gap_sigma_pct: float = 0.05
    dist: str = "uniform"             # uniform | skewnorm | empirical | gbm_quantile
    skew_alpha: float = 0.0
    quantile_level: float = 0.5
    streak_on: bool = False
    streak_c1: float = 0.5
    streak_decay: float = 1.0
    streak_every: int = 1
    profile: tuple = ()               # 24 hourly variance multipliers (empty = off)
    volume_on: bool = True
    volume_base: float = 1_000_000.0
    volume_beta: float = 1.0
    volume_noise: float = 0.35

    def with_(self, **changes: Any) -> "GenParams":
        return dataclasses.replace(self, **changes)


# Search bounds for the genes tuned by the GA / COBYLA (vol is searched in its own range).
SHAPE_BOUNDS: dict[str, tuple[float, float]] = {
    "wick_factor": (0.05, 0.60), "jump_prob": (0.0, 0.15), "ar1": (-0.45, 0.45),
    "gap_prob": (0.0, 0.50), "quantile_level": (0.10, 0.95), "streak_c1": (0.35, 0.95),
}


def default_intraday_profile(session_start: int = 9, session_minutes: int = 540) -> tuple:
    """U-shaped hourly variance multipliers: high at the open/close, low mid-session (mean 1)."""
    hours = [h for h in range(24) if session_start <= h < session_start + max(1, session_minutes // 60)]
    profile = np.ones(24)
    if len(hours) >= 2:
        mid = (hours[0] + hours[-1]) / 2.0
        half = max((hours[-1] - hours[0]) / 2.0, 0.5)
        shape = np.array([0.65 + 1.0 * ((h - mid) / half) ** 2 for h in hours])
        profile[hours] = shape / shape.mean()
    return tuple(float(x) for x in profile)


# ─── Idea 1: streak decay ─────────────────────────────────────────────────────
def streak_continue_table(c1: float, decay: float, every: int, max_len: int = 64) -> np.ndarray:
    """P(next bar goes the same way | the current run is n bars long), n = 1..max_len."""
    n = np.arange(1, max_len + 1)
    return np.clip(c1 * decay ** ((n - 1) // max(1, int(every))), 0.0, 1.0)


def streak_run_probability(c1: float, decay: float, every: int, length: int = 10) -> float:
    """P(`length` consecutive bars all go the same way); the first bar's direction is free."""
    table = streak_continue_table(c1, decay, every, max(length, 2))
    return float(np.prod(table[: length - 1]))


def calibrate_streak(target_p10: float = 0.032, decay_factor: float = 0.04,
                     max_c1: float = 0.95, length: int = 10) -> dict[str, float]:
    """Find (c1, every) so that P(10 same-direction bars) ~= target with decay = 1 - 0.04*e.

    Applying the decay on EVERY bar cannot reach 3.2 % (it would need c1 > 1: the best
    possible is ~1.6 %), so the smallest decay interval `every` that makes the target
    reachable with c1 <= max_c1 is used.
    """
    decay = 1.0 - decay_factor * math.e
    for every in range(1, 13):
        if streak_run_probability(max_c1, decay, every, length) < target_p10:
            continue
        c1 = float(spo.brentq(lambda c: streak_run_probability(c, decay, every, length) - target_p10,
                              0.30, max_c1))
        return {"c1": c1, "decay": decay, "every": every,
                "p10": streak_run_probability(c1, decay, every, length)}
    c1 = max_c1
    every = 12
    return {"c1": c1, "decay": decay, "every": every,
            "p10": streak_run_probability(c1, decay, every, length)}


def params_from_config(cfg: dict[str, Any], vol_index: float = 3.6, session_start: int = 9,
                       session_minutes: int = 540) -> GenParams:
    gen = cfg["generator"]
    params = GenParams(
        vol_index=vol_index, wick_factor=gen["wick_factor"], wick_bull=gen["wick_bull"],
        wick_bear=gen["wick_bear"], jump_prob=gen["jump_prob"], jump_size=gen["jump_size"],
        ar1=gen["ar1"], gap_prob=gen["gap_prob"], gap_sigma_pct=gen["gap_sigma_pct"],
        dist=gen["dist"], skew_alpha=gen["skew_alpha"], quantile_level=gen["quantile_level"],
        volume_on=bool(gen["volume_enabled"]), volume_base=gen["volume_base"],
        volume_beta=gen["volume_beta"], volume_noise=gen["volume_noise"],
    )
    if gen["profile_enabled"]:
        params = params.with_(profile=default_intraday_profile(session_start, session_minutes))
    if gen["streak_enabled"]:
        cal = calibrate_streak(gen["streak_target_p10"], gen["streak_decay_factor"], gen["streak_max_c1"])
        params = params.with_(streak_on=True, streak_c1=cal["c1"], streak_decay=cal["decay"],
                              streak_every=int(cal["every"]))
    return params


# ─── Bar generator (vectorised across paths) ──────────────────────────────────
def empirical_from_returns(returns_pct: Any) -> dict[str, np.ndarray] | None:
    """Pools for the `empirical` / `gbm_quantile` generators from real bar returns (in %)."""
    arr = np.asarray(returns_pct, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 30:
        return None
    std = float(arr.std(ddof=1))
    if std <= 0:
        return None
    z = (arr - arr.mean()) / std
    return {"z": z, "abs": np.abs(arr), "rms": float(np.sqrt(np.mean(arr ** 2)))}


def _magnitude_pools(params: GenParams, empirical: dict | None, rng: np.random.Generator):
    """(up_pool, down_pool, p_up): magnitudes scaled so rms(mag) = 1/sqrt(3) like U(0, 1)."""
    scale = 1.0 / math.sqrt(3.0)
    if params.dist == "skewnorm" and abs(params.skew_alpha) > 1e-9:
        z = sps.skewnorm.rvs(params.skew_alpha, size=20_000, random_state=rng)
    elif params.dist == "empirical" and empirical is not None:
        z = np.asarray(empirical["z"], dtype=float)
    else:
        pool = rng.random(4096)
        return pool, pool, 0.5
    z = (z - z.mean()) / (z.std() or 1.0)
    up, down = z[z > 0] * scale, -z[z < 0] * scale
    if len(up) < 8 or len(down) < 8:
        pool = rng.random(4096)
        return pool, pool, 0.5
    return up, down, float(np.mean(z > 0))


def generate_ohlc_many(
    timestamps: Any, start_price: float, params: GenParams, rng: np.random.Generator,
    n_paths: int = 1, allow_negative: bool = False, intraday: bool = False,
    empirical: dict | None = None,
) -> dict[str, np.ndarray]:
    """Simulate `n_paths` OHLC(V) paths at once; returns arrays of shape (n_paths, n_bars).

    With default parameters the maths is the original generator: change = U(-v, v) * 6/15 %,
    wicks ~ |N(0, (3.5/15) * price/100 * v / 2)|, open = previous close. The extras
    (streak decay, AR(1), jumps, skew / empirical / GBM-quantile shocks, bull/bear wicks,
    gaps, intraday profile, volume) are only active when their parameters are set.
    """
    ts = pd.DatetimeIndex(timestamps)
    n, m = len(ts), int(n_paths)
    v = float(params.vol_index)
    if not math.isfinite(v) or v <= 0:
        # Hard guard: every caller clamps into config.toml [limits]; anything else is a bug upstream.
        raise ValueError(f"vol_index must be a positive, finite number (got {params.vol_index!r})")
    if not math.isfinite(float(start_price)) or float(start_price) <= 0:
        raise ValueError(f"start price must be a positive, finite number (got {start_price!r})")
    mult = np.ones(n)
    if intraday and len(params.profile) == 24:
        mult = np.asarray(params.profile, dtype=float)[ts.hour.to_numpy()]
    up_pool, down_pool, p_up = _magnitude_pools(params, empirical, rng)
    gbm = params.dist == "gbm_quantile" and empirical is not None
    cont = streak_continue_table(params.streak_c1, params.streak_decay, params.streak_every)

    # Floor mode keeps every price ABOVE zero. Only values the old code clamped to 0 change: a bar that would
    # lose >= 100 % loses 99 % instead, and a gap open or wick low at or below 0 is set to 1 % of its reference.
    # Before, such a bar became exactly 0 and, because every later bar multiplies the previous close, the run
    # stayed at 0 for good ("falling to zero"). Paths that never touched 0 are byte-identical to the old engine.
    floor = MIN_PRICE_FRACTION if not allow_negative else None
    o = np.empty((m, n)); h = np.empty((m, n)); l = np.empty((m, n)); c = np.empty((m, n))
    vol_out = np.empty((m, n))
    prev_close = np.full(m, float(start_price))
    prev_change = np.zeros(m)
    sign_prev = rng.choice([-1.0, 1.0], size=m)
    run_len = np.ones(m, dtype=int)
    sigma_ref = 0.4 * v / math.sqrt(3.0)

    for t in range(n):
        open_price = prev_close.copy()
        if params.gap_prob > 0:
            gap = rng.random(m) < params.gap_prob
            if gap.any():
                open_price[gap] = open_price[gap] * (1.0 + rng.normal(0.0, params.gap_sigma_pct / 100.0, gap.sum()))
                if floor is not None:
                    open_price = np.where(open_price > 0, open_price, floor * prev_close)
        if params.streak_on:
            if t == 0:
                sign = sign_prev.copy()
            else:
                keep = rng.random(m) < cont[np.minimum(run_len - 1, len(cont) - 1)]
                sign = np.where(keep, sign_prev, -sign_prev)
                run_len = np.where(keep, run_len + 1, 1)
        else:
            sign = np.where(rng.random(m) < p_up, 1.0, -1.0)
        sign_prev = sign

        if gbm:
            q = np.clip(params.quantile_level + rng.uniform(-0.25, 0.25, m), 0.01, 0.99)
            sigma_i = np.quantile(empirical["abs"], q) / max(empirical["rms"], 1e-12) / math.sqrt(3.0)
            mag = np.abs(rng.standard_normal(m)) * sigma_i
        else:
            idx = rng.integers(0, 1 << 30, m)
            mag = np.where(sign > 0, up_pool[idx % len(up_pool)], down_pool[idx % len(down_pool)])
        shock_pct = sign * mag * v * 0.4 * mult[t]
        change_pct = shock_pct + params.ar1 * prev_change
        if params.jump_prob > 0:
            jump = rng.random(m) < params.jump_prob
            if jump.any():
                change_pct = np.where(jump, change_pct * (1.0 + rng.exponential(params.jump_size, m)), change_pct)
        if gbm:
            sig_frac = np.abs(shock_pct) / 100.0
            close = open_price * np.exp(change_pct / 100.0 - 0.5 * sig_frac ** 2)
        else:
            close = open_price * (1.0 + change_pct / 100.0)
            if floor is not None:
                wiped = close <= 0                                               # would lose >= 100 %
                if wiped.any():
                    close = np.where(wiped, floor * open_price, close)           # lose 99 % instead
                    change_pct = np.where(wiped, (floor - 1.0) * 100.0, change_pct)
        if not allow_negative:
            close = np.maximum(0.0, close)

        wick_range = params.wick_factor * (prev_close / 100.0) * v * mult[t]
        up_wick = np.abs(rng.normal(0.0, np.maximum(wick_range, 0.0) / 2.0 + 1e-300))
        dn_wick = np.abs(rng.normal(0.0, np.maximum(wick_range, 0.0) / 2.0 + 1e-300))
        wick_mult = np.where(close >= open_price, params.wick_bull, params.wick_bear)
        high = np.maximum(open_price, close) + up_wick * wick_mult
        low = np.minimum(open_price, close) - dn_wick * wick_mult
        if floor is not None:
            low = np.where(low > 0, low, floor * np.minimum(open_price, close))

        o[:, t], h[:, t], l[:, t], c[:, t] = open_price, high, low, close
        if params.volume_on:
            base_sigma = max(sigma_ref * mult[t], 1e-12)
            activity = 1.0 + params.volume_beta * np.abs(change_pct) / base_sigma
            vol_out[:, t] = (params.volume_base * mult[t] * activity
                             * rng.lognormal(0.0, params.volume_noise, m))
        prev_close = close
        prev_change = change_pct

    def _round(a: np.ndarray) -> np.ndarray:
        # 9 decimals, except where that would turn a floored (positive) price below 5e-10 into exactly 0.
        r = np.round(a, 9)
        return np.where((r <= 0) & (a > 0), a, r) if floor is not None else r

    out = {"Open": _round(o), "High": _round(h), "Low": _round(l), "Close": _round(c)}
    if params.volume_on:
        out["Volume"] = np.maximum(np.round(vol_out), 1.0)
    return out


def generate_ohlc(
    timestamps: Any, start_price: float, params: GenParams, rng: np.random.Generator,
    allow_negative: bool = False, intraday: bool = False, empirical: dict | None = None,
) -> pd.DataFrame:
    """One simulated path as a DataFrame indexed by `Date` (same layout as the old generator)."""
    arrays = generate_ohlc_many(timestamps, start_price, params, rng, 1, allow_negative, intraday, empirical)
    frame = pd.DataFrame({k: v[0] for k, v in arrays.items()}, index=pd.DatetimeIndex(timestamps, name="Date"))
    return frame


def expected_bar_sigma_pct(params: GenParams, intraday: bool = False, n_bars: int = 4000,
                           seed: int = 12345, empirical: dict | None = None) -> float:
    """Per-bar close-to-close std (%) the generator should produce with these parameters.

    Closed form 0.4*v/sqrt(3) for the plain model; a long Monte-Carlo run otherwise.
    """
    plain = (params.dist == "uniform" and params.jump_prob == 0 and params.ar1 == 0
             and params.gap_prob == 0 and not params.profile)
    if plain and params.vol_index < 50:
        return 0.4 * params.vol_index / math.sqrt(3.0)
    rng = np.random.default_rng(seed)
    stamps = pd.date_range("2020-01-06 09:00", periods=n_bars, freq="h" if intraday else "D")
    sim = generate_ohlc_many(stamps, 100.0, params.with_(volume_on=False), rng, 1, True, intraday, empirical)
    close = sim["Close"][0]
    open_ = sim["Open"][0]
    ret = 100.0 * (close / np.where(open_ == 0, np.nan, open_) - 1.0)
    ret = ret[np.isfinite(ret)]
    return float(ret.std(ddof=1)) if len(ret) > 2 else float("nan")


# ─── Extended comparison metrics ──────────────────────────────────────────────
@dataclass
class CycleContext:
    """One backtest cycle: its timestamps, the real aligned bars and per-cycle search state."""
    cycle: int
    start: Any
    timestamps: pd.DatetimeIndex
    real: pd.DataFrame | None
    intraday: bool = False
    start_price: float = 1.0
    regime: str = "unknown"
    base_vol: float = 3.6        # vol_index from the raw market-change formula (the "null" model)
    scale: float = 1.0           # decay * regime multiplier applied to the global vol
    age_index: int = 0           # 0 = oldest cycle (used by the decay exp(-lambda * t))
    learned_vol: float | None = None

    @property
    def scoreable(self) -> bool:
        if self.real is None:
            return False
        valid = int((self.real["Open"].notna() & self.real["Close"].notna()).sum())
        return valid >= max(MIN_BARS_ABSOLUTE, math.ceil(MIN_COVERAGE * len(self.timestamps)))


def _paired_rebased(sim: pd.DataFrame, real: pd.DataFrame):
    valid = (real["Open"].notna() & real["Close"].notna()).to_numpy()
    positions = np.flatnonzero(valid)
    if len(positions) < MIN_BARS_ABSOLUTE:
        return None
    base = int(positions[0])
    sim_base, real_base = float(sim["Open"].iloc[base]), float(real["Open"].iloc[base])
    if sim_base <= 0 or real_base <= 0:
        return None
    cols = ["Open", "High", "Low", "Close"]
    s = sim[cols].iloc[positions].astype(float)
    r = real[cols].iloc[positions].astype(float) * (sim_base / real_base)
    return positions, s, r


def _adjacent_returns(close: np.ndarray, positions: np.ndarray) -> np.ndarray:
    adjacent = np.diff(positions) == 1
    prev, cur = close[:-1][adjacent], close[1:][adjacent]
    ok = prev > 0
    return 100.0 * (cur[ok] / prev[ok] - 1.0)


def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) <= nlags + 1:
        return np.full(nlags, np.nan)
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if denom <= 0:
        return np.full(nlags, np.nan)
    return np.array([np.sum(x[k:] * x[:-k]) / denom for k in range(1, nlags + 1)])


def _run_lengths(up: np.ndarray) -> np.ndarray:
    if len(up) == 0:
        return np.array([], dtype=int)
    change = np.flatnonzero(np.diff(up.astype(int)) != 0)
    edges = np.concatenate(([-1], change, [len(up) - 1]))
    return np.diff(edges)


def kde_kl_divergence(real_ret: np.ndarray, sim_ret: np.ndarray, points: int = 256) -> float:
    """KL(real || sim) of two return samples through Gaussian kernel density estimates."""
    real_ret, sim_ret = np.asarray(real_ret, float), np.asarray(sim_ret, float)
    if len(real_ret) < 8 or len(sim_ret) < 8 or real_ret.std() <= 0 or sim_ret.std() <= 0:
        return float("nan")
    lo = min(real_ret.min(), sim_ret.min())
    hi = max(real_ret.max(), sim_ret.max())
    pad = 0.1 * (hi - lo + 1e-12)
    grid = np.linspace(lo - pad, hi + pad, points)
    p = sps.gaussian_kde(real_ret)(grid) + 1e-12
    q = sps.gaussian_kde(sim_ret)(grid) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _max_drawdown_pct(close: np.ndarray) -> float:
    close = np.asarray(close, float)
    if len(close) == 0 or (close <= 0).any():
        return float("nan")
    return float(100.0 * (1.0 - (close / np.maximum.accumulate(close)).min()))


def _crossover_win_rate(frame: pd.DataFrame) -> float:
    """Crossover win rate through market_analysis (imported lazily: it imports matplotlib)."""
    import market_analysis as ma
    data = frame.reset_index(drop=True).copy()
    if len(data) < ma.SLOW + 2:
        return float("nan")
    data["Date"] = pd.date_range("2000-01-03", periods=len(data), freq="D")
    data = ma._add_indicators(data, 0)
    trades = ma._trades_for_run(data, 0)
    if not trades:
        return float("nan")
    return float(np.mean([t["r_multiple"] > 0 for t in trades]))


def heavy_metrics(sim: pd.DataFrame, real: pd.DataFrame, cfg: dict[str, Any] | None = None) -> dict[str, float]:
    """KL, ACF, streak KS, kurtosis, drawdown, info ratio, body/range, gap and crossover metrics."""
    paired = _paired_rebased(sim, real)
    out: dict[str, float] = {}
    if paired is None:
        return out
    positions, s, r = paired
    sc, rc = s["Close"].to_numpy(), r["Close"].to_numpy()
    sim_ret, real_ret = _adjacent_returns(sc, positions), _adjacent_returns(rc, positions)
    k = min(len(sim_ret), len(real_ret))
    out["kl_divergence"] = kde_kl_divergence(real_ret, sim_ret)
    nlags = int(min(20, max(1, k // 3)))
    acf_s, acf_r = _acf(sim_ret, nlags), _acf(real_ret, nlags)
    out["acf_ssd"] = float(np.nansum((acf_s - acf_r) ** 2)) if k > nlags + 2 else float("nan")
    out["acf_lag1_sim"] = float(acf_s[0]) if len(acf_s) and np.isfinite(acf_s[0]) else float("nan")
    out["acf_lag1_real"] = float(acf_r[0]) if len(acf_r) and np.isfinite(acf_r[0]) else float("nan")
    sim_runs = _run_lengths((s["Close"].to_numpy() >= s["Open"].to_numpy()))
    real_runs = _run_lengths((r["Close"].to_numpy() >= r["Open"].to_numpy()))
    out["streak_ks_p"] = (float(sps.ks_2samp(sim_runs, real_runs).pvalue)
                          if len(sim_runs) >= 3 and len(real_runs) >= 3 else float("nan"))
    out["max_streak_sim"], out["max_streak_real"] = float(sim_runs.max(initial=0)), float(real_runs.max(initial=0))
    if k >= 8 and sim_ret.std() > 0 and real_ret.std() > 0:
        out["kurtosis_diff"] = float(abs(sps.kurtosis(sim_ret) - sps.kurtosis(real_ret)))
    else:
        out["kurtosis_diff"] = float("nan")
    if k >= 3:
        diff = sim_ret[:k] - real_ret[:k]
        out["info_ratio_abs"] = float(abs(diff.mean() / diff.std(ddof=1))) if diff.std(ddof=1) > 0 else 0.0
    else:
        out["info_ratio_abs"] = float("nan")
    out["drawdown_diff_pct"] = abs(_max_drawdown_pct(sc) - _max_drawdown_pct(rc))

    def body_range(frame: pd.DataFrame) -> float:
        rng_mean = float((frame["High"] - frame["Low"]).mean())
        return float((frame["Close"] - frame["Open"]).abs().mean() / rng_mean) if rng_mean > 0 else float("nan")

    out["body_range_sim"], out["body_range_real"] = body_range(s), body_range(r)
    out["body_range_diff"] = abs(out["body_range_sim"] - out["body_range_real"])

    def gap_freq(frame: pd.DataFrame) -> float:
        prev = frame["Close"].to_numpy()[:-1][np.diff(positions) == 1]
        nxt = frame["Open"].to_numpy()[1:][np.diff(positions) == 1]
        ok = prev > 0
        return float(np.mean(np.abs(nxt[ok] / prev[ok] - 1.0) > 1e-6)) if ok.any() else float("nan")

    out["gap_freq_sim"], out["gap_freq_real"] = gap_freq(s), gap_freq(r)
    wins_sim, wins_real = _crossover_win_rate(s), _crossover_win_rate(r)
    out["winrate_sim"], out["winrate_real"] = wins_sim, wins_real
    out["winrate_diff"] = abs(wins_sim - wins_real) if np.isfinite(wins_sim) and np.isfinite(wins_real) else float("nan")
    return out


def rolling_direction_match(sim: pd.DataFrame, real: pd.DataFrame, window: int = 10,
                            threshold: float = 0.40) -> list[dict[str, Any]]:
    """Direction match in rolling windows; windows below `threshold` are the 'weak segments'."""
    paired = _paired_rebased(sim, real)
    if paired is None:
        return []
    positions, s, r = paired
    same = (s["Close"].to_numpy() >= s["Open"].to_numpy()) == (r["Close"].to_numpy() >= r["Open"].to_numpy())
    weak, w = [], int(max(3, window))
    for start in range(0, max(1, len(same) - w + 1)):
        chunk = same[start:start + w]
        if len(chunk) < w:
            break
        match = float(chunk.mean())
        if match < threshold:
            weak.append({"first_bar": int(positions[start]) + 1, "last_bar": int(positions[start + w - 1]) + 1,
                         "direction_match_pct": match})
    merged: list[dict[str, Any]] = []
    for seg in weak:                                    # merge overlapping windows into one segment
        if merged and seg["first_bar"] <= merged[-1]["last_bar"] + 1:
            merged[-1]["last_bar"] = max(merged[-1]["last_bar"], seg["last_bar"])
            merged[-1]["direction_match_pct"] = min(merged[-1]["direction_match_pct"], seg["direction_match_pct"])
        else:
            merged.append(dict(seg))
    return merged


# ─── Composite objective F and the stopping criteria ──────────────────────────
HEAVY_PENALTY_KEYS = {
    "kl": "kl_divergence", "acf": "acf_ssd", "drawdown": "drawdown_diff_pct", "body_range": "body_range_diff",
    "winrate": "winrate_diff", "info_ratio": "info_ratio_abs", "streak_ks": "streak_ks_p",
}


def _needs_heavy(cfg: dict[str, Any]) -> bool:
    vs, ob = cfg["volatility_search"], cfg["objective"]
    heavy_criteria = {"streak_ks", "kl", "acf", "drawdown", "info_ratio"}
    return any(ob[k] > 0 for k in HEAVY_PENALTY_KEYS) or bool(heavy_criteria & set(vs["criteria"]))


def _penalties(sc: dict[str, float], cfg: dict[str, Any]) -> dict[str, float]:
    vs = cfg["volatility_search"]
    def frac(value: float, limit: float) -> float:
        return float(min(1.0, abs(value) / limit)) if np.isfinite(value) and limit > 0 else 0.0
    return {
        "kl": frac(sc.get("kl_divergence", np.nan), vs["kl_max"]),
        "acf": frac(sc.get("acf_ssd", np.nan), vs["acf_ssd_max"]),
        "drawdown": frac(sc.get("drawdown_diff_pct", np.nan), 4 * vs["drawdown_tolerance_pct"]),
        "body_range": frac(sc.get("body_range_diff", np.nan), 4 * vs["body_range_tolerance"]),
        "winrate": frac(sc.get("winrate_diff", np.nan), 2 * vs["winrate_tolerance"]),
        "info_ratio": frac(sc.get("info_ratio_abs", np.nan), vs["info_ratio_max"]),
        "streak_ks": (0.0 if not np.isfinite(sc.get("streak_ks_p", np.nan))
                      else float(1.0 - min(1.0, sc["streak_ks_p"] / max(vs["ks_p_min"], 1e-9)))),
        "wick": (0.0 if not np.isfinite(sc.get("score_wick", np.nan)) else float((1.0 - sc["score_wick"]) / 2.0)),
    }


def composite_F(sc: dict[str, float], cfg: dict[str, Any]) -> float:
    """F = w1*path + w2*direction + w3*net_price_match + w4*range  (minus optional soft penalties).

    The range term is an addition to the three-term F in the ideas file: path RMSE and net-move
    error against ONE realised path are minimised by a nearly flat simulation (the conditional
    mean), so without a variance anchor the search collapses vol_index to its lower bound.
    Set w_range = 0 to reproduce the pure three-term F.
    """
    vs, ob = cfg["volatility_search"], cfg["objective"]
    net = sc.get("net_price_match", np.nan)
    net = float(np.clip(net, -1.0, 1.0)) if np.isfinite(net) else -1.0
    value = (vs["w_path"] * sc["score_path"] + vs["w_direction"] * sc["score_direction"] + vs["w_net"] * net
             + vs.get("w_range", 0.0) * sc.get("score_range", 0.0))
    pen = _penalties(sc, cfg)
    for key, weight in ob.items():
        if weight > 0:
            value -= weight * pen.get(key, 0.0)
    return float(value)


def criteria_status(agg: dict[str, float], cfg: dict[str, Any]) -> dict[str, bool]:
    """Which of the selected criteria hold for an aggregate (mean over cycles of per-cycle medians)."""
    vs = cfg["volatility_search"]
    def ok(key: str, test: Callable[[float], bool]) -> bool:
        value = agg.get(key, np.nan)
        return bool(np.isfinite(value) and test(value))
    path_ok = ok("index", lambda x: x >= vs["target"])
    if vs["path_band"]:
        path_ok = path_ok and ok("path_error", lambda x: vs["path_error_lo"] <= x <= vs["path_error_hi"])
    table = {
        "path": path_ok,
        "bars": ok("direction_match_pct", lambda x: x >= vs["bars_min"]),
        "price": ok("net_price_match", lambda x: x >= vs["price_min"]),
        "exact": ok("exact_score", lambda x: x >= vs["exact_pass"]),
        "streak_ks": ok("streak_ks_p", lambda x: x > vs["ks_p_min"]),
        "kl": ok("kl_divergence", lambda x: x <= vs["kl_max"]),
        "acf": ok("acf_ssd", lambda x: x <= vs["acf_ssd_max"]),
        "drawdown": ok("drawdown_diff_pct", lambda x: x <= vs["drawdown_tolerance_pct"]),
        "info_ratio": ok("info_ratio_abs", lambda x: x <= vs["info_ratio_max"]),
    }
    return {name: table[name] for name in vs["criteria"] if name in table}


def cycle_seed(master_seed: int, cycle: int, salt: int = 0) -> int:
    """Deterministic per-(cycle, salt) seed so every evaluation is reproducible."""
    return int((int(master_seed) * 1_000_003 + int(cycle) * 9_176 + int(salt) * 101 + 17) % (2 ** 32))


NUMERIC_STAT_KEYS = (
    "index", "score_path", "score_direction", "score_net_move", "score_range", "score_wick",
    "direction_match_pct", "net_price_match", "net_price_error_pct", "path_error", "exact_score",
    "rmse_pct", "sim_range_pct", "real_range_pct", "sim_net_pct", "real_net_pct",
    "wick_ratio_sim", "wick_ratio_real",
    "kl_divergence", "acf_ssd", "acf_lag1_sim", "acf_lag1_real", "streak_ks_p", "kurtosis_diff",
    "info_ratio_abs", "drawdown_diff_pct", "body_range_sim", "body_range_real", "body_range_diff",
    "gap_freq_sim", "gap_freq_real", "winrate_sim", "winrate_real", "winrate_diff",
    "max_streak_sim", "max_streak_real", "F",
)


def evaluate_cycle(
    ctx: CycleContext, params: GenParams, cfg: dict[str, Any], n_paths: int, seed: int,
    heavy: bool | None = None, empirical: dict | None = None, allow_negative: bool = False,
) -> dict[str, float] | None:
    """Median metrics over `n_paths` simulated paths for one cycle (None if not scoreable)."""
    if not ctx.scoreable:
        return None
    rng = np.random.default_rng(seed)
    arrays = generate_ohlc_many(ctx.timestamps, ctx.start_price, params, rng, n_paths,
                                allow_negative, ctx.intraday, empirical)
    do_heavy = _needs_heavy(cfg) if heavy is None else heavy
    real_arr = ohlc_array(ctx.real)
    paths = np.stack([arrays[name] for name in OHLC_FIELDS], axis=2)    # (n_paths, bars, 4)
    rows: list[dict[str, float]] = []
    for k in range(n_paths):
        scored = score_cycle(paths[k], real_arr, cfg["exact"])
        if scored.get("status") != "scored":
            continue
        row = {key: float(scored[key]) for key in NUMERIC_STAT_KEYS if key in scored}
        if do_heavy:                                  # heavy metrics still take a DataFrame
            sim = pd.DataFrame({name: arrays[name][k] for name in OHLC_FIELDS}, index=ctx.timestamps)
            row.update(heavy_metrics(sim, ctx.real, cfg))
        row["F"] = composite_F(row, cfg)
        rows.append(row)
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    out = {key: float(np.nanmedian(frame[key])) if frame[key].notna().any() else float("nan")
           for key in frame.columns}
    out["n_paths"] = float(len(rows))
    out["F_std"] = float(frame["F"].std(ddof=1)) if len(rows) > 1 else 0.0
    return out


def aggregate_stats(per_cycle: list[dict[str, float]]) -> dict[str, float]:
    """Mean over cycles of the per-cycle medians."""
    if not per_cycle:
        return {}
    keys = set().union(*[set(d) for d in per_cycle])
    return {k: float(np.nanmean([d.get(k, np.nan) for d in per_cycle]))
            if any(np.isfinite(d.get(k, np.nan)) for d in per_cycle) else float("nan") for k in keys}


def bar_sigma_pct(real: pd.DataFrame | None) -> float:
    """Per-candle std (%) of real close-to-close returns between ADJACENT real candles of one window
    (an overnight or weekend jump between two covered candles that are not neighbours is left out)."""
    if real is None or "Close" not in real:
        return float("nan")
    valid = np.flatnonzero(real["Close"].notna().to_numpy())
    returns = _adjacent_returns(real["Close"].to_numpy(dtype=float)[valid], valid)
    return float(returns.std(ddof=1)) if len(returns) > 3 else float("nan")


def real_sigma_pct(contexts: list[CycleContext]) -> float:
    """Per-bar std (%) of the real close-to-close returns, averaged across scoreable cycles."""
    values = [bar_sigma_pct(ctx.real) for ctx in contexts if ctx.scoreable]
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def real_returns_pooled(contexts: list[CycleContext]) -> np.ndarray:
    pool = []
    for ctx in contexts:
        if ctx.scoreable:
            valid = np.flatnonzero(ctx.real["Close"].notna().to_numpy())
            pool.append(_adjacent_returns(ctx.real["Close"].to_numpy(dtype=float)[valid], valid))
    return np.concatenate(pool) if pool else np.array([])


def ensemble_vote(ctx: CycleContext, params: GenParams, k: int, seed: int,
                  empirical: dict | None = None, allow_negative: bool = False) -> dict[str, float]:
    """K generators (different seeds, same params): majority-vote direction per bar."""
    if not ctx.scoreable or k < 1:
        return {}
    rng = np.random.default_rng(seed)
    a = generate_ohlc_many(ctx.timestamps, ctx.start_price, params.with_(volume_on=False), rng, k,
                           allow_negative, ctx.intraday, empirical)
    up = a["Close"] >= a["Open"]                       # (k, n)
    valid = (ctx.real["Open"].notna() & ctx.real["Close"].notna()).to_numpy()
    real_up = (ctx.real["Close"].to_numpy() >= ctx.real["Open"].to_numpy())
    majority_up = up.mean(axis=0) >= 0.5
    per_path = [float(np.mean(up[i][valid] == real_up[valid])) for i in range(k)]
    net_up = (a["Close"][:, -1] >= a["Open"][:, 0])
    return {
        "ensemble_direction_match": float(np.mean(majority_up[valid] == real_up[valid])),
        "mean_path_direction_match": float(np.mean(per_path)),
        "ensemble_net_up": float(net_up.mean() >= 0.5),
        "ensemble_agreement": float(abs(net_up.mean() - 0.5) * 2),
    }


# ─── Persistent memory (vol_equilibrium.csv, history, audit, certificate, log) ─
def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class VolMemory:
    """Everything the search learns, persisted in the analysis folder across runs."""

    EQUILIBRIUM = "vol_equilibrium.csv"
    HISTORY = "vol_history.csv"
    AUDIT = "vol_search_audit.csv"
    REGIME = "regime_vol_multipliers.csv"
    LOG = "equilibrium_log.txt"
    CERT = "convergence_certificate.json"
    KEY = ["instrument", "asset_type", "timeframe", "interval", "scope", "window_start"]

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    def _read(self, name: str) -> pd.DataFrame:
        path = self.dir / name
        if path.exists() and path.stat().st_size > 0:
            try:
                return pd.read_csv(path)
            except (pd.errors.ParserError, pd.errors.EmptyDataError):
                return pd.DataFrame()
        return pd.DataFrame()

    # equilibrium table ------------------------------------------------------
    def equilibrium(self) -> pd.DataFrame:
        return self._read(self.EQUILIBRIUM)

    def upsert_equilibrium(self, rows: list[dict[str, Any]]) -> None:
        new = pd.DataFrame(rows)
        if new.empty:
            return
        old = self.equilibrium()
        if not old.empty:
            merged = old.merge(new[self.KEY].drop_duplicates().assign(_drop=1), on=self.KEY, how="left")
            old = old[merged["_drop"].isna().to_numpy()]
        pd.concat([old, new], ignore_index=True, sort=False).to_csv(self.dir / self.EQUILIBRIUM, index=False)

    def global_row(self, instrument: str, asset_type: str, timeframe: str, interval: str) -> dict[str, Any] | None:
        table = self.equilibrium()
        if table.empty:
            return None
        hit = table[(table["instrument"] == instrument) & (table["asset_type"] == asset_type)
                    & (table["timeframe"] == timeframe) & (table["interval"].astype(str) == str(interval))
                    & (table["scope"] == "global")]
        return None if hit.empty else hit.sort_values("updated").iloc[-1].to_dict()

    def nearest_cycle_row(self, instrument: str, asset_type: str, timeframe: str, interval: str,
                          window_start: Any) -> dict[str, Any] | None:
        """Per-cycle learned vol from the nearest stored window date (idea: per-cycle vol learning)."""
        table = self.equilibrium()
        if table.empty:
            return None
        hit = table[(table["instrument"] == instrument) & (table["asset_type"] == asset_type)
                    & (table["timeframe"] == timeframe) & (table["interval"].astype(str) == str(interval))
                    & (table["scope"] == "cycle")].copy()
        if hit.empty:
            return None
        dates = pd.to_datetime(hit["window_start"], errors="coerce")
        target = pd.Timestamp(window_start)
        hit = hit.assign(_gap=(dates - target).abs()).dropna(subset=["_gap"])
        return None if hit.empty else hit.sort_values("_gap").iloc[0].to_dict()

    def transfer_row(self, instrument: str, asset_type: str, timeframe: str, interval: str,
                     sigma_pct: float | None) -> dict[str, Any] | None:
        """Nearest OTHER instrument (same asset class, timeframe, interval; closest real sigma)."""
        table = self.equilibrium()
        if table.empty:
            return None
        hit = table[(table["instrument"] != instrument) & (table["asset_type"] == asset_type)
                    & (table["timeframe"] == timeframe) & (table["interval"].astype(str) == str(interval))
                    & (table["scope"] == "global")].copy()
        if "vol_ratio" not in hit.columns:
            return None
        hit = hit[np.isfinite(pd.to_numeric(hit["vol_ratio"], errors="coerce"))]
        if hit.empty:
            return None
        if sigma_pct and np.isfinite(sigma_pct) and "real_sigma_pct" in hit:
            sig = pd.to_numeric(hit["real_sigma_pct"], errors="coerce")
            hit = hit.assign(_dist=(np.log(sig.clip(lower=1e-9)) - math.log(max(sigma_pct, 1e-9))).abs())
            hit = hit.sort_values("_dist")
        return hit.iloc[0].to_dict()

    # history ----------------------------------------------------------------
    def history(self) -> pd.DataFrame:
        return self._read(self.HISTORY)

    def append_history(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        old = self.history()
        pd.concat([old, pd.DataFrame(rows)], ignore_index=True, sort=False).to_csv(self.dir / self.HISTORY, index=False)

    def history_seed(self, instrument: str, timeframe: str, interval: str, k: int = 5) -> float | None:
        """Median vol_index of the k most recent rows for this instrument/timeframe."""
        table = self.history()
        if table.empty:
            return None
        hit = table[(table["instrument"] == instrument) & (table["timeframe"] == timeframe)
                    & (table["interval"].astype(str) == str(interval))]
        if "kind" in hit.columns:
            hit = hit[hit["kind"] == "final"]
        if hit.empty:
            return None
        return float(np.median(hit.sort_values("date").tail(k)["vol_index"].astype(float)))

    # audit / log / certificate ----------------------------------------------
    def append_audit(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path = self.dir / self.AUDIT
        new = pd.DataFrame(rows)
        new.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)

    def append_log(self, line: str) -> None:
        with open(self.dir / self.LOG, "a", encoding="utf-8") as handle:
            handle.write(f"{_now()}  {line}\n")

    def write_certificate(self, certificate: dict[str, Any]) -> Path:
        path = self.dir / self.CERT
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(certificate, handle, indent=2, default=_json_default)
        return path

    # regime multipliers -----------------------------------------------------
    def regime_multipliers(self, instrument: str, timeframe: str) -> dict[str, float]:
        table = self._read(self.REGIME)
        if table.empty:
            return {}
        hit = table[(table["instrument"] == instrument) & (table["timeframe"] == timeframe)]
        return {str(r["regime"]): float(r["multiplier"]) for _, r in hit.iterrows()}

    def update_regime_multipliers(self, instrument: str, timeframe: str, observed: dict[str, tuple[float, int]],
                                  smoothing: float = 0.5) -> pd.DataFrame:
        """observed: regime -> (median ratio learned/base, n cycles). Exponentially smoothed."""
        table = self._read(self.REGIME)
        previous = self.regime_multipliers(instrument, timeframe)
        rows = []
        for regime, (ratio, n) in observed.items():
            shrink = n / (n + 3.0)
            target = 1.0 + (ratio - 1.0) * shrink
            old = previous.get(regime)
            value = target if old is None else (1 - smoothing) * old + smoothing * target
            rows.append({"instrument": instrument, "timeframe": timeframe, "regime": regime,
                         "multiplier": float(value), "cycles_seen": int(n), "updated": _now()})
        if not rows:
            return table
        new = pd.DataFrame(rows)
        if not table.empty:
            keep = ~table.set_index(["instrument", "timeframe", "regime"]).index.isin(
                new.set_index(["instrument", "timeframe", "regime"]).index)
            table = table[keep]
        out = pd.concat([table, new], ignore_index=True, sort=False)
        out.to_csv(self.dir / self.REGIME, index=False)
        return out


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return str(obj)


# ─── Early stopping (SPRT) ────────────────────────────────────────────────────
class Sprt:
    """Sequential probability ratio test on the per-step improvement of the best F.

    H0: the vol_index is already optimal (mean improvement 0). H1: improvements of size delta
    are still arriving. Evidence accumulates per step; crossing the lower bound accepts H0
    (stop early, alpha = 0.05), crossing the upper bound confirms H1 and restarts the evidence.
    """

    def __init__(self, alpha: float, beta: float, sigma: float, min_steps: int = 4):
        self.upper = math.log((1.0 - beta) / alpha)
        self.lower = math.log(beta / (1.0 - alpha))
        self.sigma = max(float(sigma), 1e-3)
        self.delta = self.sigma
        self.llr, self.steps, self.min_steps = 0.0, 0, min_steps

    def update(self, improvement: float) -> str:
        self.steps += 1
        self.llr += (self.delta / self.sigma ** 2) * (improvement - self.delta / 2.0)
        if self.llr >= self.upper:
            self.llr = 0.0
            return "improving"
        if self.llr <= self.lower and self.steps >= self.min_steps:
            return "accept_h0"
        return "continue"


@dataclass
class SearchResult:
    params: GenParams
    stats: dict[str, float]
    converged: bool
    reason: str
    iterations: int
    method: str
    criteria_met: bool = False
    trace: list[tuple[float, float]] = field(default_factory=list)
    extra: dict[str, float] = field(default_factory=dict)


class VolSearch:
    """Searches generator parameters against the real price for a set of cycles.

    Evaluations use common random numbers (the same seeds for every candidate), so F(v) is a
    deterministic curve that bisection / COBYLA / Bayesian optimisation can follow; fresh seeds
    are used for the Monte-Carlo calibration mode and for out-of-sample validation.
    """

    def __init__(self, contexts: list[CycleContext], cfg: dict[str, Any], master_seed: int, salt: int = 0,
                 label: str = "global", empirical: dict | None = None, allow_negative: bool = False,
                 use_scale: bool = True, fresh_seeds: bool = False, n_paths: int | None = None,
                 meta: dict[str, Any] | None = None):
        self.ctxs = [c for c in contexts if c.scoreable]
        self.cfg, self.vs = cfg, cfg["volatility_search"]
        self.master_seed, self.salt, self.label = master_seed, salt, label
        self.empirical, self.allow_negative = empirical, allow_negative
        self.use_scale, self.fresh = use_scale, fresh_seeds
        self.n_paths = int(n_paths or self.vs["mc_paths"])
        self.meta = meta or {}
        self.cache: dict[tuple, dict[str, float]] = {}
        self.audit: list[dict[str, Any]] = []
        self.evals = 0
        self._fresh_counter = 0

    # evaluation -------------------------------------------------------------
    @staticmethod
    def _key(params: GenParams) -> tuple:
        return (round(params.vol_index, 9), round(params.wick_factor, 9), round(params.jump_prob, 9),
                round(params.ar1, 9), round(params.gap_prob, 9), round(params.quantile_level, 9),
                round(params.streak_c1, 9), round(params.wick_bull, 9), round(params.wick_bear, 9),
                params.profile)

    def evaluate(self, params: GenParams, method: str, reason: str, step_delta: float | None = None) -> dict[str, float]:
        key = self._key(params)
        if not self.fresh and key in self.cache:
            return self.cache[key]
        self._fresh_counter += 1
        salt = self.salt + (self._fresh_counter * 7919 if self.fresh else 0)
        per = []
        limits = self.cfg["limits"]
        for ctx in self.ctxs:
            vol = params.vol_index * (ctx.scale if self.use_scale else 1.0)
            # A regime multiplier or a strong decay can push a cycle's vol outside [limits]; never score that.
            effective = params.with_(vol_index=min(max(vol, limits["vol_index_min"]), limits["vol_index_max"]))
            stats = evaluate_cycle(ctx, effective, self.cfg, self.n_paths,
                                   cycle_seed(self.master_seed, ctx.cycle, salt), None,
                                   self.empirical, self.allow_negative)
            if stats:
                per.append(stats)
        agg = aggregate_stats(per)
        if not agg:
            agg = {"F": -9.0, "index": -1.0, "score_path": -1.0, "score_direction": -1.0,
                   "net_price_match": -1.0, "direction_match_pct": 0.0, "F_std": 0.0}
        agg["F_se"] = float(np.nanmean([d.get("F_std", 0.0) for d in per]) / math.sqrt(self.n_paths)) if per else 0.0
        self.evals += 1
        self.audit.append({
            "run_time": _now(), "scope": self.label, "method": method, "iteration": self.evals,
            "vol_index": params.vol_index, "path_score": agg.get("score_path"),
            "direction_match": agg.get("direction_match_pct"), "net_move_match": agg.get("net_price_match"),
            "composite_F": agg.get("F"), "step_delta": step_delta, "reason_for_step": reason,
            **self.meta,
        })
        if not self.fresh:
            self.cache[key] = agg
        return agg

    def met(self, stats: dict[str, float]) -> bool:
        status = criteria_status(stats, self.cfg)
        return bool(status) and all(status.values())

    def _result(self, params: GenParams, stats: dict[str, float], converged: bool, reason: str,
                iterations: int, method: str, trace: list | None = None) -> SearchResult:
        return SearchResult(params, stats, converged or self.met(stats), reason, iterations, method,
                            self.met(stats), trace or [])

    # linear / step search with learning-rate scheduling ----------------------
    def step_search(self, params0: GenParams, v_lo: float, v_hi: float, max_iters: int | None = None,
                    method: str = "step") -> SearchResult:
        vs = self.vs
        max_iters = int(max_iters or vs["max_iters"])
        params = params0.with_(vol_index=float(np.clip(params0.vol_index, v_lo, v_hi)))
        best = self.evaluate(params, method, "start")
        trace = [(params.vol_index, best["F"])]
        if self.met(best):
            return self._result(params, best, True, "criteria already met at start", 0, method, trace)
        lr = 1.0
        sprt = Sprt(vs["sprt_alpha"], vs["sprt_beta"], best.get("F_se", 0.0))
        quiet = 0
        reason = "max iterations"
        for it in range(1, max_iters + 1):
            v = params.vol_index
            base_step = v * vs["step_pct"] / 100.0 if vs["step_mode"] == "pct" else vs["step_abs"]
            step = base_step * lr
            floor = v * vs["min_step_pct"] / 100.0 if vs["step_mode"] == "pct" else vs["step_abs"] * vs["min_step_pct"] / 100.0
            if step < floor:
                reason = "step below minimum"
                return self._result(params, best, True, reason, it - 1, method, trace)
            options = []
            for direction, name in ((+1.0, "up"), (-1.0, "down")):
                v_new = float(np.clip(v + direction * step, v_lo, v_hi))
                if abs(v_new - v) < 1e-12:
                    continue
                cand = params.with_(vol_index=v_new)
                stats = self.evaluate(cand, method, f"try {name} (step {step:.4g}, lr {lr:.3g})", None)
                options.append((stats["F"], cand, stats, name))
            if not options:
                reason = "no feasible neighbour"
                return self._result(params, best, True, reason, it, method, trace)
            f_new, cand, stats, name = max(options, key=lambda o: o[0])
            improvement = f_new - best["F"]
            if improvement > 1e-9:
                self.audit.append({"run_time": _now(), "scope": self.label, "method": method, "iteration": self.evals,
                                   "vol_index": cand.vol_index, "path_score": stats.get("score_path"),
                                   "direction_match": stats.get("direction_match_pct"),
                                   "net_move_match": stats.get("net_price_match"), "composite_F": stats["F"],
                                   "step_delta": cand.vol_index - v, "reason_for_step": f"accepted {name}", **self.meta})
                params, best = cand, stats
                trace.append((params.vol_index, best["F"]))
                quiet = quiet + 1 if improvement < vs["convergence_eps"] else 0
                if self.met(best):
                    return self._result(params, best, True, "all selected criteria met", it, method, trace)
            else:
                lr /= 2.0                                  # overshoot / oscillation -> halve the step
                self.audit.append({"run_time": _now(), "scope": self.label, "method": method, "iteration": self.evals,
                                   "vol_index": params.vol_index, "path_score": best.get("score_path"),
                                   "direction_match": best.get("direction_match_pct"),
                                   "net_move_match": best.get("net_price_match"), "composite_F": best["F"],
                                   "step_delta": 0.0, "reason_for_step": f"no gain: halve step to lr {lr:.3g}", **self.meta})
                quiet += 1
            if quiet >= vs["convergence_patience"]:
                return self._result(params, best, True, f"|dF| < {vs['convergence_eps']} for {quiet} steps", it, method, trace)
            if vs["sprt_enabled"]:
                verdict = sprt.update(max(improvement, 0.0))
                if verdict == "accept_h0":
                    return self._result(params, best, True, "SPRT: H0 (already optimal) not rejected", it, method, trace)
        return self._result(params, best, False, reason, max_iters, method, trace)

    # Monte-Carlo calibration: fresh seeds every evaluation -----------------------
    def mc_search(self, params0: GenParams, v_lo: float, v_hi: float) -> SearchResult:
        previous = self.fresh
        self.fresh = True
        try:
            result = self.step_search(params0, v_lo, v_hi, method="mc")
        finally:
            self.fresh = previous
        return result

    # grid searches -------------------------------------------------------------
    def grid_search(self, params0: GenParams, grid: np.ndarray, method: str = "grid") -> SearchResult:
        best_stats, best_params, trace = None, params0, []
        for v in grid:
            cand = params0.with_(vol_index=float(v))
            stats = self.evaluate(cand, method, "grid point")
            trace.append((float(v), stats["F"]))
            if best_stats is None or stats["F"] > best_stats["F"]:
                best_stats, best_params = stats, cand
        return self._result(best_params, best_stats or {}, True, "grid complete", len(grid), method, trace)

    def _capped_grid(self, lo: float, hi: float, step: float) -> np.ndarray:
        cap = int(self.vs["max_grid_points"])
        count = int(math.floor((hi - lo) / step)) + 1 if step > 0 else 2
        if count > cap:
            count = cap
        return np.unique(np.round(np.linspace(lo, hi, max(count, 2)), 10))

    def coarse_fine_search(self, params0: GenParams, v_lo: float, v_hi: float) -> SearchResult:
        """Iterative deepening: coarse grid (step 0.5) then a fine grid (step 0.05) around the winner."""
        coarse_step = self.vs["coarse_step_abs"]
        coarse = self.grid_search(params0, self._capped_grid(v_lo, v_hi, coarse_step), "coarse_grid")
        centre = coarse.params.vol_index
        span = max(coarse_step, (v_hi - v_lo) / max(self.vs["max_grid_points"], 1))
        lo, hi = max(v_lo, centre - span), min(v_hi, centre + span)
        fine = self.grid_search(params0, self._capped_grid(lo, hi, self.vs["fine_step_abs"]), "fine_grid")
        best = fine if fine.stats.get("F", -9) >= coarse.stats.get("F", -9) else coarse
        trace = coarse.trace + fine.trace
        return self._result(best.params, best.stats, True, "coarse then fine grid", len(trace), "coarse_fine", trace)

    # bisection with linear fallback ------------------------------------------------
    def bisect_search(self, params0: GenParams, v_lo: float, v_hi: float) -> SearchResult:
        """Bisection on log(sim range / real range) = 0, O(log N) evaluations.

        The range ratio rises monotonically with vol_index, so a root brackets the vol that matches
        the real high-low range. If the three bracket probes are not monotone, linear search is used.
        """
        def g(v: float, reason: str) -> tuple[float, dict[str, float]]:
            stats = self.evaluate(params0.with_(vol_index=v), "bisect", reason)
            sim_r, real_r = stats.get("sim_range_pct", np.nan), stats.get("real_range_pct", np.nan)
            ratio = sim_r / real_r if np.isfinite(sim_r) and np.isfinite(real_r) and real_r > 0 and sim_r > 0 else np.nan
            return (math.log(ratio) if np.isfinite(ratio) else float("nan")), stats

        lo, hi = float(v_lo), float(v_hi)
        g_lo, _ = g(lo, "bracket low")
        g_hi, _ = g(hi, "bracket high")
        mid = math.sqrt(lo * hi)
        g_mid, _ = g(mid, "bracket middle")
        monotone = (np.isfinite([g_lo, g_mid, g_hi]).all() and g_lo < g_mid < g_hi and g_lo < 0.0 < g_hi)
        if not monotone:
            self.audit.append({"run_time": _now(), "scope": self.label, "method": "bisect", "iteration": self.evals,
                               "vol_index": mid, "path_score": None, "direction_match": None, "net_move_match": None,
                               "composite_F": None, "step_delta": None,
                               "reason_for_step": "monotonicity assumption broken: linear-search fallback", **self.meta})
            grid = np.geomspace(max(lo, 1e-6), hi, 12)
            coarse = self.grid_search(params0, grid, "bisect_fallback_grid")
            refined = self.step_search(coarse.params, lo, hi, 15, "bisect_fallback_step")
            return self._result(refined.params, refined.stats, refined.converged, "fallback to linear search",
                                self.evals, "bisect(fallback)", coarse.trace + refined.trace)
        iterations = 0
        while iterations < int(self.vs["max_iters"]) and hi / lo - 1.0 > 0.005:
            iterations += 1
            mid = math.sqrt(lo * hi)
            g_mid, _ = g(mid, "bisect midpoint")
            if g_mid < 0:
                lo = mid           # sim range still too small -> raise the lower bracket
            else:
                hi = mid           # sim range too large -> lower the upper bracket
        v_star = math.sqrt(lo * hi)
        stats = self.evaluate(params0.with_(vol_index=v_star), "bisect", "range-matched root")
        polished = self.step_search(params0.with_(vol_index=v_star), v_lo, v_hi, 6, "bisect_polish")
        best = polished if polished.stats.get("F", -9) >= stats["F"] else self._result(
            params0.with_(vol_index=v_star), stats, True, "root", iterations, "bisect")
        return self._result(best.params, best.stats, True, f"bisection converged in {iterations} halvings",
                            iterations, "bisect", [(v_star, best.stats.get("F", float("nan")))])

    # COBYLA ---------------------------------------------------------------------------
    def cobyla_search(self, params0: GenParams, v_lo: float, v_hi: float) -> SearchResult:
        """min -F(v) subject to v_lo < v < v_hi with scipy COBYLA (derivative-free)."""
        def objective(x: np.ndarray) -> float:
            v = float(np.clip(x[0], v_lo, v_hi))
            return -self.evaluate(params0.with_(vol_index=v), "cobyla", "COBYLA evaluation")["F"]

        constraints = [{"type": "ineq", "fun": lambda x: x[0] - v_lo}, {"type": "ineq", "fun": lambda x: v_hi - x[0]}]
        outcome = spo.minimize(objective, x0=[float(np.clip(params0.vol_index, v_lo, v_hi))], method="COBYLA",
                               constraints=constraints,
                               options={"rhobeg": 0.2 * max(params0.vol_index, 1e-3), "maxiter": int(self.vs["cobyla_maxiter"]),
                                        "tol": 1e-4})
        v_star = float(np.clip(outcome.x[0], v_lo, v_hi))
        best = params0.with_(vol_index=v_star)
        stats = self.evaluate(best, "cobyla", "COBYLA optimum")
        return self._result(best, stats, bool(outcome.success), str(outcome.message), int(outcome.nfev), "cobyla")

    # Bayesian optimisation with a small Gaussian-process surrogate ---------------------
    def bayes_search(self, params0: GenParams, v_lo: float, v_hi: float) -> SearchResult:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        rng = np.random.default_rng(cycle_seed(self.master_seed, 0, self.salt + 31))
        n_init, n_total = int(self.vs["bayes_init"]), int(self.vs["bayes_evals"])
        grid = np.linspace(math.log(v_lo), math.log(v_hi), 300)
        xs = list(np.linspace(math.log(v_lo), math.log(v_hi), n_init))
        ys = []
        for x in xs:
            ys.append(self.evaluate(params0.with_(vol_index=math.exp(x)), "bayes", "initial design")["F"])
        while len(xs) < n_total:
            kernel = ConstantKernel(1.0) * Matern(length_scale=0.5, nu=2.5) + WhiteKernel(1e-4, noise_level_bounds=(1e-12, 1e1))
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2,
                                          random_state=int(rng.integers(0, 2 ** 31 - 1)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(np.array(xs).reshape(-1, 1), np.array(ys))
            mean, std = gp.predict(grid.reshape(-1, 1), return_std=True)
            best_y = max(ys)
            z = (mean - best_y) / np.maximum(std, 1e-12)
            ei = (mean - best_y) * sps.norm.cdf(z) + std * sps.norm.pdf(z)
            ei[[any(abs(g - x) < 1e-6 for x in xs) for g in grid]] = -1.0
            x_next = float(grid[int(np.argmax(ei))])
            xs.append(x_next)
            ys.append(self.evaluate(params0.with_(vol_index=math.exp(x_next)), "bayes",
                                    f"expected improvement {float(ei.max()):.4g}")["F"])
        best_x = xs[int(np.argmax(ys))]
        best = params0.with_(vol_index=math.exp(best_x))
        return self._result(best, self.evaluate(best, "bayes", "best observed"), True,
                            f"{len(xs)} evaluations", len(xs), "bayes", [(math.exp(x), y) for x, y in zip(xs, ys)])

    # genetic algorithm over [vol, wick, jump, ar1, ...] -------------------------------------
    def genes_for(self, params: GenParams) -> list[str]:
        genes = ["wick_factor", "jump_prob", "ar1", "gap_prob"]
        if params.dist == "gbm_quantile":
            genes.append("quantile_level")
        if params.streak_on:
            genes.append("streak_c1")
        return genes

    def ga_search(self, params0: GenParams, v_lo: float, v_hi: float) -> SearchResult:
        rng = np.random.default_rng(cycle_seed(self.master_seed, 0, self.salt + 53))
        genes = self.genes_for(params0)
        pop_size, generations = int(self.vs["ga_population"]), int(self.vs["ga_generations"])
        lo = np.array([math.log(v_lo)] + [SHAPE_BOUNDS[g][0] for g in genes])
        hi = np.array([math.log(v_hi)] + [SHAPE_BOUNDS[g][1] for g in genes])

        def decode(vec: np.ndarray) -> GenParams:
            vec = np.clip(vec, lo, hi)
            return params0.with_(vol_index=float(math.exp(vec[0])), **{g: float(vec[i + 1]) for i, g in enumerate(genes)})

        start = np.array([math.log(float(np.clip(params0.vol_index, v_lo, v_hi)))] + [float(getattr(params0, g)) for g in genes])
        population = [start] + [np.clip(start + rng.normal(0, 0.15, len(start)) * (hi - lo), lo, hi)
                                for _ in range(pop_size - 1)]
        fitness = [self.evaluate(decode(p), "ga", "initial population")["F"] for p in population]
        for gen in range(1, generations + 1):
            order = np.argsort(fitness)[::-1]
            elites = [population[i].copy() for i in order[:2]]
            children = []
            while len(children) < pop_size - len(elites):
                picks = []
                for _ in range(2):
                    cand = rng.choice(len(population), size=min(3, len(population)), replace=False)
                    picks.append(population[int(cand[np.argmax([fitness[i] for i in cand])])])
                blend = rng.uniform(-0.25, 1.25, len(start))
                child = blend * picks[0] + (1 - blend) * picks[1]
                child = np.clip(child + rng.normal(0, 0.08, len(start)) * (hi - lo), lo, hi)
                children.append(child)
            population = elites + children
            fitness = [self.evaluate(decode(p), "ga", f"generation {gen}")["F"] for p in population]
        best_vec = population[int(np.argmax(fitness))]
        best = decode(best_vec)
        return self._result(best, self.evaluate(best, "ga", "best individual"), True,
                            f"{generations} generations x {pop_size}", self.evals, "ga")

    # decay lambda grid -------------------------------------------------------------------------
    def decay_lambda_search(self, params0: GenParams) -> tuple[float, dict[str, float]]:
        """Vol decay over cycles: vol(t) = vol0 * exp(-lambda * t); lambda picked by grid search."""
        results = []
        saved = [c.scale for c in self.ctxs]
        try:
            for lam in self.vs["decay_lambda_grid"]:
                for c, old in zip(self.ctxs, saved):
                    c.scale = old * decay_factor(lam, c.age_index)
                self.cache.clear()
                stats = self.evaluate(params0, "decay_lambda", f"lambda={lam}")
                results.append((stats["F"], lam, stats))
        finally:
            for c, old in zip(self.ctxs, saved):
                c.scale = old
        best_f, best_lam, best_stats = max(results, key=lambda r: r[0])
        self.cache.clear()
        return float(best_lam), best_stats


# ─── Orchestration helpers ────────────────────────────────────────────────────
def decay_factor(lam: float, age: float) -> float:
    """exp(-lam * age), kept above 0. With lam up to 10 (allowed by the config), exp() underflowed to exactly 0.0
    for old cycles, the cycle's vol became 0 and the engine's vol guard stopped the whole self-correction."""
    return math.exp(-min(lam * age, 300.0))


def discover_vol_range(contexts: list[CycleContext], cfg: dict[str, Any]) -> tuple[float, float, float]:
    """v_min = 0.1 x sigma_real, v_max = 5 x sigma_real (sigma = per-bar std of real returns, %)."""
    vs = cfg["volatility_search"]
    sigma = real_sigma_pct(contexts)
    if np.isfinite(sigma) and sigma > 0:
        lo, hi = vs["range_lo_mult"] * sigma, vs["range_hi_mult"] * sigma
    else:
        base = float(np.median([c.base_vol for c in contexts])) if contexts else 3.6
        lo, hi = 0.2 * base, 5.0 * base
    limits = cfg["limits"]
    # Upper bound for lo never below vol_index_min: np.clip with a_min > a_max returns a_max, which put lo
    # under the configured minimum when vol_index_min > vol_index_max / 2.
    lo = float(min(max(lo, 1e-3, limits["vol_index_min"]), max(limits["vol_index_min"], limits["vol_index_max"] / 2.0)))
    hi = float(min(max(hi, lo * 2.0), limits["vol_index_max"]))
    return lo, hi, sigma


def detect_change_point(values: list[float]) -> int | None:
    """Index of a structural break in the sequence of learned vols (ruptures if installed, else CUSUM)."""
    x = np.log(np.clip(np.asarray(values, dtype=float), 1e-9, None))
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return None
    try:
        import ruptures as rpt                                   # optional
        algo = rpt.Pelt(model="rbf").fit(x.reshape(-1, 1))
        found = algo.predict(pen=3.0)
        return int(found[0]) if len(found) > 1 else None
    except Exception:
        pass
    z = (x - x.mean()) / (x.std() or 1.0)
    k, threshold = 0.5, 4.0
    pos = neg = 0.0
    for i, value in enumerate(z):
        pos, neg = max(0.0, pos + value - k), max(0.0, neg - value - k)
        if pos > threshold or neg > threshold:
            return int(i)
    return None


def learn_intraday_profile(contexts: list[CycleContext], params: GenParams, cfg: dict[str, Any], master_seed: int,
                           empirical: dict | None = None, allow_negative: bool = False,
                           passes: int = 3) -> tuple[tuple, pd.DataFrame]:
    """Tune the 24-bucket variance profile so per-hour mean |bar return| matches the real bars.

    A symmetric generator's direction match does not depend on scale, so the multiplier is driven
    by the real/sim variance ratio per hour; the per-hour direction match is recorded alongside.
    """
    intraday = [c for c in contexts if c.scoreable and c.intraday]
    if not intraday or len(params.profile) != 24:
        return params.profile, pd.DataFrame()
    profile = np.asarray(params.profile, dtype=float).copy()
    table = pd.DataFrame()
    for pass_no in range(passes):
        sim_abs = {h: [] for h in range(24)}; real_abs = {h: [] for h in range(24)}
        agree = {h: [] for h in range(24)}
        for ctx in intraday:
            rng = np.random.default_rng(cycle_seed(master_seed, ctx.cycle, 700 + pass_no))
            arrays = generate_ohlc_many(ctx.timestamps, ctx.start_price, params.with_(profile=tuple(profile), volume_on=False),
                                        rng, 4, allow_negative, True, empirical)
            sim_ret = 100.0 * np.abs(arrays["Close"] / np.where(arrays["Open"] == 0, np.nan, arrays["Open"]) - 1.0)
            sim_up = arrays["Close"] >= arrays["Open"]
            real = ctx.real
            valid = (real["Open"].notna() & real["Close"].notna()).to_numpy()
            real_ret = 100.0 * np.abs(real["Close"].to_numpy(float) / real["Open"].to_numpy(float) - 1.0)
            real_up = real["Close"].to_numpy(float) >= real["Open"].to_numpy(float)
            hours = ctx.timestamps.hour.to_numpy()
            for h in np.unique(hours):
                mask = (hours == h) & valid
                if mask.sum() == 0:
                    continue
                sim_abs[h].extend(np.nanmean(sim_ret[:, mask], axis=0).tolist())
                real_abs[h].extend(real_ret[mask].tolist())
                agree[h].extend((sim_up[:, mask] == real_up[mask]).mean(axis=0).tolist())
        rows = []
        for h in range(24):
            if len(real_abs[h]) >= 3 and len(sim_abs[h]) >= 3 and np.nanmean(sim_abs[h]) > 0:
                ratio = float(np.nanmean(real_abs[h]) / np.nanmean(sim_abs[h]))
                rows.append({"hour": h, "sim_mean_abs_pct": float(np.nanmean(sim_abs[h])),
                             "real_mean_abs_pct": float(np.nanmean(real_abs[h])), "ratio": ratio,
                             "direction_match": float(np.mean(agree[h])), "bars": len(real_abs[h]),
                             "multiplier_before": float(profile[h])})
                profile[h] *= float(np.clip(ratio, 0.25, 4.0)) ** 0.5     # damped update
        active = [r["hour"] for r in rows]
        if active:
            profile[active] *= len(active) / profile[active].sum()        # keep the session mean at 1
        for r in rows:
            r["multiplier_after"] = float(profile[r["hour"]])
        table = pd.DataFrame(rows)
    return tuple(float(x) for x in profile), table


def evaluate_null(contexts: list[CycleContext], base_params: GenParams, cfg: dict[str, Any], master_seed: int,
                  salt: int, n_paths: int, empirical: dict | None, allow_negative: bool,
                  heavy: bool | None = None) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
    """The 'null' run: every cycle uses its raw market-change vol_index (no optimisation)."""
    per, by_cycle = [], {}
    for ctx in contexts:
        stats = evaluate_cycle(ctx, base_params.with_(vol_index=ctx.base_vol), cfg, n_paths,
                               cycle_seed(master_seed, ctx.cycle, salt), heavy, empirical, allow_negative)
        if stats:
            per.append(stats)
            by_cycle[ctx.cycle] = stats
    return aggregate_stats(per), by_cycle


def format_equilibrium_line(vol: float, stats: dict[str, float], converged: bool, iterations: int) -> str:
    path = stats.get("score_path", float("nan"))
    direction = 100.0 * stats.get("direction_match_pct", float("nan"))
    price_err = stats.get("net_price_error_pct", float("nan"))
    return (f"[Equilibrium] vol={vol:.2f} | path={path:.2f} | dir={direction:.0f}% | price_err=±{price_err:.2f}% | "
            f"converged={'YES' if converged else 'NO'} after {iterations} iterations")


def accuracy_decayed(memory: VolMemory, meta: dict[str, Any], current_index: float, threshold: float) -> bool:
    """Automatic re-run trigger: is the new accuracy_index worse than the stored one by > threshold?"""
    row = memory.global_row(meta["instrument"], meta["asset_type"], meta["timeframe"], meta["interval"])
    if row is None or not np.isfinite(current_index):
        return False
    stored = pd.to_numeric(pd.Series([row.get("accuracy_index")]), errors="coerce").iloc[0]
    return bool(np.isfinite(stored) and (stored - current_index) > threshold)


def recalibration_due(memory: VolMemory, meta: dict[str, Any], max_age_days: float) -> bool:
    """--auto-recal: is the stored optimum older than N days (or missing)?"""
    row = memory.global_row(meta["instrument"], meta["asset_type"], meta["timeframe"], meta["interval"])
    if row is None:
        return True
    updated = pd.to_datetime(row.get("updated"), errors="coerce")
    return bool(pd.isna(updated) or (pd.Timestamp.now() - updated).days >= max_age_days)


def plan_augmentation(result: dict[str, Any], contexts: list[CycleContext], n_aug: int, master_seed: int) -> list[dict[str, Any]]:
    """Extra high-quality runs at the optimal vol (new seeds) to add to the ML set with a boosted weight."""
    cycles = [c for c in contexts if c.scoreable] or contexts
    plan = []
    for i in range(int(n_aug)):
        ctx = cycles[i % len(cycles)]
        vol = result["per_cycle_vol"].get(ctx.cycle, result["vol"] * ctx.scale)
        plan.append({"cycle": ctx.cycle, "vol_index": float(vol),
                     "seed": cycle_seed(master_seed, ctx.cycle, 9000 + i)})
    return plan


# ─── The self-correction run ──────────────────────────────────────────────────
def run_self_correction(
    contexts: list[CycleContext], base_params: GenParams, cfg: dict[str, Any], meta: dict[str, Any],
    out_dir: str | Path, master_seed: int, method: str | None = None, loops: int | None = None,
    empirical: dict | None = None, allow_negative: bool = False, lightweight: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """generate -> score against the real price -> learn -> repeat, until an equilibrium is found.

    Order of work: vol range discovery -> seed (history / transfer / formula) -> null baseline ->
    global search on the training cycles (consensus of independent searches) -> decay lambda ->
    shape genes (GA) -> intraday profile -> per-cycle learning (loops, running log-mean) ->
    validation on fresh seeds (train / out-of-sample) -> bootstrap CI, regimes, change-point ->
    persist (equilibrium, history, audit, certificate, log).
    """
    vs = cfg["volatility_search"]
    method = method or vs["method"]
    loops = int(loops or vs["loops"])
    memory = VolMemory(out_dir)
    ctxs = sorted([c for c in contexts if c.scoreable], key=lambda c: c.cycle)
    if len(ctxs) < 2:
        return {"status": "not enough scoreable cycles (need >= 2 with real data)", "scoreable": len(ctxs)}
    for i, c in enumerate(ctxs):
        c.age_index = i
    audit_rows: list[dict[str, Any]] = []
    meta = {k: meta[k] for k in ("instrument", "asset_type", "timeframe", "interval") if k in meta}
    interval = str(meta.get("interval", ""))
    n_val = int(vs["validation_paths"])

    # 1. train / out-of-sample split (oldest cycles train, newest cycles are held out)
    n_test = int(round(len(ctxs) * vs["holdout_frac"])) if len(ctxs) >= 4 else 0
    train, test = (ctxs[: len(ctxs) - n_test], ctxs[len(ctxs) - n_test:]) if n_test else (ctxs, [])

    # 2. vol range + seed
    v_lo, v_hi, sigma = discover_vol_range(ctxs, cfg)
    base_mean = float(np.mean([c.base_vol for c in ctxs]))
    seed_source, v0 = "market-change formula", base_mean
    if vs["use_history_seed"]:
        hist = memory.history_seed(meta["instrument"], meta["timeframe"], interval)
        if hist is not None:
            v0, seed_source = hist, "median of the 5 most recent stored runs"
    if seed_source == "market-change formula" and vs["use_transfer"]:
        near = memory.transfer_row(meta["instrument"], meta["asset_type"], meta["timeframe"], interval, sigma)
        if near is not None:
            # learned vol per unit of real per-bar sigma on the nearest instrument, applied to this instrument's sigma
            near_sigma = pd.to_numeric(pd.Series([near.get("real_sigma_pct")]), errors="coerce").iloc[0]
            if np.isfinite(near_sigma) and near_sigma > 0 and np.isfinite(sigma) and sigma > 0:
                v0 = sigma * float(near["learned_vol"]) / float(near_sigma)
            else:
                v0 = base_mean * float(near["vol_ratio"])
            seed_source = f"transfer from {near['instrument']}"
    v0 = float(np.clip(v0, v_lo, v_hi))
    progress(f"[Self-correct] real per-bar sigma {sigma:.4f}% -> vol range [{v_lo:.3f}, {v_hi:.3f}]; "
             f"start vol {v0:.3f} ({seed_source}); train {len(train)} / out-of-sample {len(test)} cycles")

    # 3. regime multipliers + nearest-date learned vol
    mult = memory.regime_multipliers(meta["instrument"], meta["timeframe"]) if vs["use_regimes"] else {}
    for c in ctxs:
        c.scale = float(mult.get(c.regime, 1.0))
        near_c = memory.nearest_cycle_row(meta["instrument"], meta["asset_type"], meta["timeframe"], interval, c.start)
        c.learned_vol = float(near_c["learned_vol"]) if near_c is not None and np.isfinite(near_c.get("learned_vol", np.nan)) else None
        c._loops_done = int(near_c["loops_done"]) if near_c is not None and "loops_done" in near_c and np.isfinite(near_c.get("loops_done", np.nan)) else 0

    # 4. null baseline (raw market-change vol, no optimisation), fresh validation seeds
    VAL = 5_000
    null_all, _ = evaluate_null(ctxs, base_params, cfg, master_seed, VAL, n_val, empirical, allow_negative, heavy=True)
    null_test, _ = evaluate_null(test, base_params, cfg, master_seed, VAL, n_val, empirical, allow_negative) if test else ({}, {})

    # 5. global search on the training cycles (subsampled for speed), consensus of N searches
    cap = int(vs["global_cycle_cap"])
    sample = train if len(train) <= cap else [train[i] for i in np.unique(np.linspace(0, len(train) - 1, cap).round().astype(int))]
    n_consensus = 1 if lightweight else max(1, int(vs["consensus_runs"]))
    found: list[SearchResult] = []
    iterations = 0
    base_scale = {c.cycle: c.scale for c in ctxs}                # regime multiplier before any decay
    for k in range(n_consensus):
        for c in ctxs:
            c.scale = base_scale[c.cycle]
        searcher = VolSearch(sample, cfg, master_seed, salt=100 * (k + 1), label="global", empirical=empirical,
                             allow_negative=allow_negative, meta={**meta, "consensus_run": k + 1})
        start = base_params.with_(vol_index=v0)
        if lightweight:
            result = searcher.coarse_fine_search(start, v_lo, v_hi) if vs["recal_iters"] <= 0 else \
                searcher.step_search(start, v_lo, v_hi, int(vs["recal_iters"]), "recal_step")
        elif method == "step":
            result = searcher.step_search(start, v_lo, v_hi)
        elif method == "bisect":
            result = searcher.bisect_search(start, v_lo, v_hi)
        elif method == "grid":
            result = searcher.grid_search(start, np.geomspace(v_lo, v_hi, min(int(vs["max_grid_points"]), 40)))
        elif method == "coarse_fine":
            result = searcher.coarse_fine_search(start, v_lo, v_hi)
        elif method == "cobyla":
            result = searcher.cobyla_search(start, v_lo, v_hi)
        elif method == "bayes":
            result = searcher.bayes_search(start, v_lo, v_hi)
        elif method == "ga":
            result = searcher.ga_search(start, v_lo, v_hi)
        elif method == "mc":
            result = searcher.mc_search(start, v_lo, v_hi)
        else:                                                    # auto: bracket -> refine -> decay -> shape -> polish
            coarse = searcher.coarse_fine_search(start, v_lo, v_hi)
            result = searcher.step_search(coarse.params, v_lo, v_hi, max(6, int(vs["max_iters"]) // 2), "step_refine")
            lam, _ = searcher.decay_lambda_search(result.params)
            if lam > 0:
                for c in sample:
                    c.scale = base_scale[c.cycle] * decay_factor(lam, c.age_index)
                searcher.cache.clear()
                result = searcher.step_search(result.params, v_lo, v_hi, 8, "step_after_decay")
            if vs["tune_shape"]:
                shape = searcher.ga_search(result.params, v_lo, v_hi)
                if shape.stats.get("F", -9) > result.stats.get("F", -9):
                    result = shape
                result = searcher.step_search(result.params, v_lo, v_hi, 8, "step_final_polish")
            result.reason = f"{result.reason}; decay lambda {lam}"
            result.extra["lambda"] = float(lam)
        iterations += searcher.evals
        audit_rows.extend(searcher.audit)
        found.append(result)
        if lightweight:
            break
    found.sort(key=lambda r: r.params.vol_index)
    chosen = found[len(found) // 2]                              # median optimal vol across the consensus runs
    global_params = chosen.params
    lam_used = float(chosen.extra.get("lambda", 0.0))
    for c in ctxs:                                               # one consistent decay for every cycle
        c.scale = base_scale[c.cycle] * decay_factor(lam_used, c.age_index)
    consensus_vols = [r.params.vol_index for r in found]
    progress(f"[Self-correct] global vol {global_params.vol_index:.4f} (consensus {', '.join(f'{v:.3f}' for v in consensus_vols)}) "
             f"after {iterations} evaluations; decay lambda {lam_used}")

    # 6. intraday variance profile
    profile_table = pd.DataFrame()
    if vs["profile_learning"] and not lightweight and any(c.intraday for c in ctxs):
        profile, profile_table = learn_intraday_profile(ctxs, global_params, cfg, master_seed, empirical, allow_negative)
        global_params = global_params.with_(profile=profile)

    # 7. per-cycle learning loops (each loop: fresh CRN seeds, running log-mean across loops/runs)
    per_cycle_vol: dict[int, float] = {}
    per_cycle_iters: dict[int, int] = {}
    if vs["per_cycle"]:
        for loop in range(1, loops + 1):
            for c in ctxs:
                searcher = VolSearch([c], cfg, master_seed, salt=loop * 17 + 3, label=f"cycle {c.cycle}", empirical=empirical,
                                     allow_negative=allow_negative, use_scale=False,
                                     meta={**meta, "loop": loop})
                init = c.learned_vol if c.learned_vol else global_params.vol_index * c.scale
                lo_c = max(v_lo * 0.25, cfg["limits"]["vol_index_min"])
                hi_c = min(v_hi * 2.0, cfg["limits"]["vol_index_max"])
                res = searcher.step_search(global_params.with_(vol_index=float(np.clip(init, lo_c, hi_c))), lo_c, hi_c,
                                           max_iters=min(int(vs["max_iters"]), 12 if not lightweight else 4), method="cycle_step")
                done = getattr(c, "_loops_done", 0)
                prev = c.learned_vol if c.learned_vol else res.params.vol_index
                c.learned_vol = float(math.exp((done * math.log(prev) + math.log(res.params.vol_index)) / (done + 1)))
                c._loops_done = done + 1
                per_cycle_iters[c.cycle] = per_cycle_iters.get(c.cycle, 0) + searcher.evals
                iterations += searcher.evals
                audit_rows.extend(searcher.audit)
            progress(f"[Self-correct] loop {loop}/{loops}: per-cycle vols updated "
                     f"(median {np.median([c.learned_vol for c in ctxs]):.3f})")
        per_cycle_vol = {c.cycle: float(c.learned_vol) for c in ctxs}

    # 8. validation on FRESH seeds: global (train + out-of-sample) and per-cycle (in-sample)
    def validate(cs: list[CycleContext], use_cycle_vol: bool, salt: int, heavy: bool = True) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
        per, by = [], {}
        for c in cs:
            vol = per_cycle_vol[c.cycle] if use_cycle_vol and c.cycle in per_cycle_vol else global_params.vol_index * c.scale
            st = evaluate_cycle(c, global_params.with_(vol_index=vol), cfg, n_val, cycle_seed(master_seed, c.cycle, salt),
                                heavy, empirical, allow_negative)
            if st:
                per.append(st)
                by[c.cycle] = st
        return aggregate_stats(per), by

    final_global, final_global_by = validate(ctxs, False, VAL)
    final_train, _ = validate(train, False, VAL)
    final_oos, _ = validate(test, False, VAL) if test else ({}, {})
    final_cycle, final_cycle_by = validate(ctxs, True, VAL) if per_cycle_vol else (final_global, final_global_by)
    criteria = criteria_status(final_global, cfg)
    criteria_all = {k: bool(v) for k, v in criteria.items()}
    converged = bool(all(r.converged for r in found)) or (bool(criteria_all) and all(criteria_all.values()))
    optimal_F = final_global["F"]
    null_F = null_all.get("F", float("nan"))
    improvement = (optimal_F - null_F) / abs(null_F) if np.isfinite(null_F) and abs(null_F) > 1e-9 else float("nan")
    oos_improvement = ((final_oos.get("F", np.nan) - null_test.get("F", np.nan)) / abs(null_test.get("F", np.nan))
                       if test and np.isfinite(null_test.get("F", np.nan)) and abs(null_test.get("F", np.nan)) > 1e-9 else float("nan"))

    # 9. bootstrap confidence interval on the optimal vol (resample cycles over a vol grid)
    ci = (float("nan"), float("nan"))
    if not lightweight and vs["bootstrap_resamples"] > 0:
        grid = np.geomspace(max(v_lo, global_params.vol_index / 3.0), min(v_hi, global_params.vol_index * 3.0), 15)
        curve = np.full((len(sample), len(grid)), np.nan)
        for j, c in enumerate(sample):
            for i, v in enumerate(grid):
                st = evaluate_cycle(c, global_params.with_(vol_index=float(v * c.scale)), cfg, 4,
                                    cycle_seed(master_seed, c.cycle, 333), False, empirical, allow_negative)
                if st:
                    curve[j, i] = st["F"]
        rng = np.random.default_rng(cycle_seed(master_seed, 0, 444))
        picks = []
        for _ in range(int(vs["bootstrap_resamples"])):
            idx = rng.integers(0, len(sample), len(sample))
            mean_curve = np.nanmean(curve[idx], axis=0)
            if np.isfinite(mean_curve).any():
                picks.append(float(grid[int(np.nanargmax(mean_curve))]))
        if picks:
            tail = (1.0 - vs["ci_level"]) / 2.0 * 100.0
            ci = (float(np.percentile(picks, tail)), float(np.percentile(picks, 100.0 - tail)))

    # 10. regime multipliers learned from the per-cycle results; change-point alert
    regime_table = pd.DataFrame()
    if per_cycle_vol and vs["use_regimes"]:
        ratios: dict[str, list[float]] = {}
        for c in ctxs:
            ratios.setdefault(c.regime, []).append(per_cycle_vol[c.cycle] / max(global_params.vol_index, 1e-12))
        observed = {r: (float(np.median(v)), len(v)) for r, v in ratios.items() if r != "unknown"}
        regime_table = memory.update_regime_multipliers(meta["instrument"], meta["timeframe"], observed)
    change_point = detect_change_point([per_cycle_vol[c.cycle] for c in ctxs]) if per_cycle_vol else None
    if change_point is not None:
        progress(f"[Self-correct] ALERT: structural break in the learned vol sequence near cycle "
                 f"{ctxs[min(change_point, len(ctxs) - 1)].cycle} (new volatility regime?)")

    # 11. weak segments for the final per-cycle path (rolling direction match)
    weak_rows = []
    for c in ctxs:
        vol = per_cycle_vol.get(c.cycle, global_params.vol_index * c.scale)
        sim = generate_ohlc(c.timestamps, c.start_price, global_params.with_(vol_index=vol, volume_on=False),
                            np.random.default_rng(cycle_seed(master_seed, c.cycle, VAL)), allow_negative, c.intraday, empirical)
        for seg in rolling_direction_match(sim, c.real, int(vs["rolling_window"]), vs["weak_segment_threshold"]):
            weak_rows.append({"cycle": c.cycle, **seg})

    # 12. persistence ---------------------------------------------------------------------------
    stamp = _now()
    vol_ratio = global_params.vol_index / base_mean if base_mean > 0 else float("nan")
    base_row = {**meta, "updated": stamp, "accuracy_index": final_global.get("index"),
                "direction_match_pct": final_global.get("direction_match_pct"),
                "net_price_match": final_global.get("net_price_match"), "composite_F": optimal_F,
                "exact_score": final_global.get("exact_score")}
    equilibrium_rows = [{**base_row, "scope": "global", "window_start": "GLOBAL", "cycle": 0,
                         "learned_vol": global_params.vol_index, "vol_ratio": vol_ratio, "real_sigma_pct": sigma,
                         "wick_factor": global_params.wick_factor, "jump_prob": global_params.jump_prob,
                         "ar1": global_params.ar1, "gap_prob": global_params.gap_prob,
                         "streak_c1": global_params.streak_c1, "decay_lambda": lam_used,
                         "oos_F": final_oos.get("F"), "null_F": null_F, "loops_done": loops,
                         "converged": int(converged)}]
    for c in ctxs:
        stc = final_cycle_by.get(c.cycle, {})
        equilibrium_rows.append({**meta, "updated": stamp, "scope": "cycle", "window_start": pd.Timestamp(c.start).date().isoformat(),
                                 "cycle": c.cycle, "learned_vol": c.learned_vol if c.learned_vol else global_params.vol_index,
                                 "vol_ratio": (c.learned_vol or global_params.vol_index) / max(c.base_vol, 1e-12),
                                 "regime": c.regime, "accuracy_index": stc.get("index"),
                                 "direction_match_pct": stc.get("direction_match_pct"),
                                 "net_price_match": stc.get("net_price_match"), "composite_F": stc.get("F"),
                                 "loops_done": getattr(c, "_loops_done", loops)})
    memory.upsert_equilibrium(equilibrium_rows)
    history_rows = [{"vol_index": float(a["vol_index"]), "accuracy_index": a.get("path_score"), "date": stamp,
                     "timeframe": meta["timeframe"], "instrument": meta["instrument"], "interval": interval,
                     "composite_F": a.get("composite_F"), "kind": "search"}
                    for a in audit_rows if a.get("scope") == "global" and a.get("vol_index") is not None]
    history_rows.append({"vol_index": global_params.vol_index, "accuracy_index": final_global.get("index"), "date": stamp,
                         "timeframe": meta["timeframe"], "instrument": meta["instrument"], "interval": interval,
                         "composite_F": optimal_F, "kind": "final"})
    memory.append_history(history_rows)
    memory.append_audit([{**a, "master_seed": master_seed} for a in audit_rows])

    line = format_equilibrium_line(global_params.vol_index, final_global, converged, iterations)
    memory.append_log(f"{meta['instrument']} {meta['timeframe']} {interval}  {line}")
    certificate = {
        "optimal_vol": global_params.vol_index, "accuracy_index": final_global.get("index"),
        "composite_F": optimal_F, "null_F": null_F, "improvement_over_null": improvement,
        "out_of_sample_F": final_oos.get("F"), "out_of_sample_null_F": null_test.get("F"),
        "out_of_sample_improvement": oos_improvement,
        "criteria_selected": list(vs["criteria"]), "criteria_met": criteria_all,
        "exact_met": bool(final_global.get("exact_score", -9) >= vs["exact_pass"]),
        "price_met": bool(final_global.get("net_price_match", -9) >= vs["price_min"]),
        "bars_met": bool(final_global.get("direction_match_pct", 0) >= vs["bars_min"]),
        "n_iterations": iterations, "random_seed": master_seed, "timestamp": stamp,
        "converged": converged, "method": method, "loops": loops, "seed_source": seed_source,
        "vol_range": [v_lo, v_hi], "real_sigma_pct": sigma, "vol_ci": list(ci), "ci_level": vs["ci_level"],
        "decay_lambda": lam_used, "params": dataclasses.asdict(global_params),
        "per_cycle_vol": per_cycle_vol, "cycles": [c.cycle for c in ctxs],
        "validation_salt": VAL, "validation_paths": n_val, "meta": meta,
        "cycle_seeds": {str(c.cycle): cycle_seed(master_seed, c.cycle, VAL) for c in ctxs},
    }
    cert_path = memory.write_certificate(certificate)
    if len(profile_table):
        profile_table.to_csv(Path(out_dir) / "intraday_variance_profile.csv", index=False)
    if weak_rows:
        pd.DataFrame(weak_rows).to_csv(Path(out_dir) / "weak_segments.csv", index=False)
    return {
        "status": "ok", "vol": global_params.vol_index, "params": global_params, "per_cycle_vol": per_cycle_vol,
        "final": final_global, "final_per_cycle": final_cycle, "final_train": final_train, "oos": final_oos,
        "null": null_all, "null_oos": null_test, "improvement_over_null": improvement,
        "oos_improvement": oos_improvement, "criteria": criteria_all, "converged": converged, "iterations": iterations,
        "line": line, "ci": ci, "change_point": change_point, "certificate": str(cert_path),
        "seed_source": seed_source, "vol_range": (v_lo, v_hi), "sigma": sigma, "consensus_vols": consensus_vols,
        "decay_lambda": lam_used, "weak_segments": len(weak_rows), "regime_multipliers": regime_table,
        "train_cycles": [c.cycle for c in train], "test_cycles": [c.cycle for c in test],
        "per_cycle_stats": final_cycle_by, "global_stats_by_cycle": final_global_by,
    }


def reproduce_from_certificate(cert_path: str | Path, contexts: list[CycleContext], cfg: dict[str, Any],
                               empirical: dict | None = None, allow_negative: bool = False) -> dict[str, float]:
    """Re-run the validation stored in a convergence certificate and report the difference."""
    with open(cert_path, "r", encoding="utf-8") as handle:
        cert = json.load(handle)
    params = GenParams(**{**dataclasses.asdict(GenParams()), **cert["params"]})
    params = params.with_(profile=tuple(params.profile))
    per = []
    for c in sorted([c for c in contexts if c.scoreable], key=lambda c: c.cycle):
        st = evaluate_cycle(c, params.with_(vol_index=params.vol_index * c.scale), cfg, int(cert["validation_paths"]),
                            int(cert["cycle_seeds"][str(c.cycle)]), True, empirical, allow_negative)
        if st:
            per.append(st)
    agg = aggregate_stats(per)
    return {"reproduced_F": agg.get("F", float("nan")), "certificate_F": cert["composite_F"],
            "abs_difference": abs(agg.get("F", float("nan")) - cert["composite_F"])}