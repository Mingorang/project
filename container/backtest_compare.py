"""Rewind & Replay accuracy engine: real Yahoo prices vs. simulated chains.

price_modeling.py (mode 2, "Rewind & Replay") simulates N back-to-back cycles from
the past up to today, each cycle opening at the previous cycle's close.  This module
then, for every cycle of every loop,
  1. pulls the REAL price history for the cycle's calendar window from Yahoo,
  2. aligns it to the simulated bars,
  3. scores how closely the simulation tracks reality on a -1 .. +1 scale, and
  4. writes the plots and tables (rewind_*.png / rewind_*.csv).

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
THE INDEX DEFINITION IS UNCHANGED from the previous backtest version.
"""
from __future__ import annotations

import math
from datetime import date
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
#   {"kind": "week", "weeks": N}         start-labelled at the Monday 09:00 open, N weeks long


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


def score_cycle(sim: pd.DataFrame, real: pd.DataFrame) -> dict[str, Any]:
    """Compare one simulated cycle with its aligned real data -> accuracy index."""
    valid = (real["Open"].notna() & real["Close"].notna()).to_numpy()
    positions = np.flatnonzero(valid)
    needed = max(MIN_BARS_ABSOLUTE, math.ceil(MIN_COVERAGE * len(sim)))
    if len(positions) < needed:
        return {"status": "insufficient real data", "valid_bars": int(len(positions))}

    base = int(positions[0])
    sim_base, real_base = float(sim["Open"].iloc[base]), float(real["Open"].iloc[base])
    if sim_base <= 0 or real_base <= 0:
        return {"status": "non-positive base price", "valid_bars": int(len(positions))}

    s, r = sim.iloc[positions], real.iloc[positions]

    def pct(frame: pd.DataFrame, column: str, base_value: float) -> np.ndarray:
        return (frame[column].to_numpy(dtype=float) / base_value - 1.0) * 100.0

    sim_close, real_close = pct(s, "Close", sim_base), pct(r, "Close", real_base)
    sim_range = pct(s, "High", sim_base).max() - pct(s, "Low", sim_base).min()
    real_range = pct(r, "High", real_base).max() - pct(r, "Low", real_base).min()
    scale = max(sim_range, real_range, 1e-9)

    rmse = float(np.sqrt(np.mean((sim_close - real_close) ** 2)))
    path = _tolerance_score(rmse, scale)

    sim_up = (s["Close"].to_numpy() >= s["Open"].to_numpy())
    real_up = (r["Close"].to_numpy() >= r["Open"].to_numpy())
    direction = 2.0 * float(np.mean(sim_up == real_up)) - 1.0

    net_move = _tolerance_score(abs(float(sim_close[-1] - real_close[-1])), scale)

    if sim_range <= 0 or real_range <= 0:
        range_score = 1.0 if sim_range == real_range else -1.0
    else:
        range_score = 1.0 - 2.0 * min(1.0, abs(math.log(sim_range / real_range)) / math.log(3.0))

    components = {"path": path, "direction": direction, "net_move": net_move, "range": range_score}
    index = sum(INDEX_WEIGHTS[name] * value for name, value in components.items())
    return {
        "status": "scored",
        "index": float(np.clip(index, -1.0, 1.0)),
        "valid_bars": int(len(positions)),
        "base_pos": base,
        "rmse_pct": rmse,
        "sim_range_pct": float(sim_range),
        "real_range_pct": float(real_range),
        "sim_net_pct": float(sim_close[-1]),
        "real_net_pct": float(real_close[-1]),
        **{f"score_{name}": float(value) for name, value in components.items()},
    }


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
                    f"path {score['score_path']:+.2f} · dir {score['score_direction']:+.2f} · "
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


def plot_panels(
    items: list[dict[str, Any]],
    symbol: str,
    timeframe: str,
    overall: float,
    output_path: Path,
    interval: str | None = None,
) -> None:
    """Draw every cycle of one loop into ONE figure and save it as a single PNG."""
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
    fig.suptitle(f"{symbol} — {timeframe.upper()}{candle_text} rewind replay: simulated vs real price",
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


# ─── Rewind & Replay: feasibility helper, chain plot, scoring, orchestration ──────
MAX_STITCHED_PLOTS = 10          # stitched chain PNGs written (first N loops)


def comparable_cycles(cycle_starts: list[date], bar: dict[str, Any], today: date) -> list[bool]:
    """Which cycles are young enough for Yahoo to serve a matching candle size.

    Yahoo only keeps 5m/15m candles for ~58 days and 1h for ~725 days; daily and weekly
    candles are unlimited.  (Coarser fallbacks are still tried when the data is fetched.)
    """
    step = bar_step_ns(bar)
    usable = [i for i in yahoo_interval_order(step) if YAHOO_INTERVALS[i][0] <= step]
    flags = []
    for start in cycle_starts:
        age = (pd.Timestamp(today) - (pd.Timestamp(start) - pd.Timedelta(days=1))).days
        flags.append(any(YAHOO_INTERVALS[i][1] is None or age <= YAHOO_INTERVALS[i][1] for i in usable))
    return flags


def plot_chain(items: list[dict[str, Any]], symbol: str, timeframe: str, interval: str,
               loop: int, overall: float, output_path: Path) -> None:
    """One continuous figure: the stitched simulated chain vs the real price over the same dates."""
    sim = pd.concat([item["sim"] for item in items])
    blank = lambda s: pd.DataFrame(np.nan, index=s.index, columns=["Open", "High", "Low", "Close"])
    real = pd.concat([item["real"] if item.get("real") is not None else blank(item["sim"]) for item in items])
    x = np.arange(len(sim))

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="black")
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#1e61bd")
    ax.grid(True, color="#3a414b", linewidth=.35)
    ax.yaxis.tick_right()
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")

    valid_real = real["Open"].dropna()
    if len(valid_real) and float(valid_real.iloc[0]) > 0:
        shown = real * (float(sim["Open"].iloc[0]) / float(valid_real.iloc[0]))
        high = np.ma.masked_invalid(shown["High"].to_numpy(dtype=float))
        low = np.ma.masked_invalid(shown["Low"].to_numpy(dtype=float))
        close = shown["Close"].to_numpy(dtype=float)
        ax.fill_between(x, low, high, where=~np.ma.getmaskarray(high), color=HALO_COLOR,
                        alpha=.13, linewidth=0, zorder=1)
        for width, alpha in ((9, .07), (5, .12), (2.4, .32)):
            ax.plot(x, close, color=HALO_COLOR, linewidth=width, alpha=alpha, solid_capstyle="round", zorder=1.5)
        ax.plot(x, close, color="#e8f6ff", linewidth=.8, alpha=.85, zorder=1.6)

    ax.fill_between(x, sim["Low"].to_numpy(float), sim["High"].to_numpy(float),
                    color=SIM_LINE_COLOR, alpha=.18, linewidth=0, zorder=2)
    ax.plot(x, sim["Close"].to_numpy(float), color=SIM_LINE_COLOR, linewidth=1.0, zorder=3)

    offset = 0
    for item in items:
        if offset:
            ax.axvline(offset - .5, color="#5b6573", linewidth=.6, linestyle=":", zorder=2)
        score = item.get("score")
        if score and score.get("status") == "scored" and len(items) <= 40:
            ax.text(offset + len(item["sim"]) / 2, .985, f"{score['index']:+.2f}",
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7,
                    color=_score_color(score["index"]))
        offset += len(item["sim"])

    ticks = np.unique(np.linspace(0, len(sim) - 1, min(10, len(sim))).round().astype(int))
    ax.set_xticks(ticks)
    ax.set_xticklabels([sim.index[i].strftime("%H:%M %d %b" if timeframe == "1d" else "%d %b %Y") for i in ticks],
                       rotation=20, ha="right")
    ax.set_xlim(-1, len(sim))
    ax.set_ylabel("Price", color="white")
    overall_text = f"{overall:+.3f}" if np.isfinite(overall) else "n/a"
    ax.set_title(f"{symbol} — rewind {timeframe.upper()} x {len(items)} cycles ({interval} candles), "
                 f"loop {loop}: continuous chain vs real price · accuracy {overall_text}", color="white", fontsize=11)
    ax.legend(handles=[Line2D([], [], color=SIM_LINE_COLOR, linewidth=1.4),
                       Line2D([], [], color=HALO_COLOR, linewidth=6, alpha=.45)],
              labels=["Simulated chain (each cycle opens at the previous close)",
                      "Real price (rebased once to the chain's start)"],
              loc="lower left", fontsize=8, facecolor="#111111", edgecolor="#aab2bd",
              labelcolor="white", framealpha=.9)
    fig.text(.01, .005, "Numbers on top = per-cycle accuracy index (each cycle is scored with its own rebasing).",
             fontsize=7, color="white")
    fig.tight_layout(rect=(0, .025, 1, 1))
    fig.savefig(output_path, dpi=120, facecolor="black")
    plt.close(fig)


def prefetch_real(symbol: str, bar: dict[str, Any], timestamps: list[Any], today: date,
                  fetcher: Fetcher | None = None) -> list[dict[str, Any]]:
    """Real prices do not depend on the random path, so each cycle is fetched once for all loops."""
    n = len(timestamps)
    step = max(1, n // 20)
    print(f"\nFetching real {symbol} prices for {n} cycle(s)...")
    out = []
    for cycle, ts in enumerate(timestamps, start=1):
        real, used, note = fetch_real_for_cycle(symbol, bar, ts, today, fetcher)
        out.append({"real": real, "interval": used, "note": note})
        if n <= 100 or cycle == 1 or cycle % step == 0 or cycle == n:
            print(f"  Cycle {cycle}/{n}: " + (f"real {used} candles found" if real is not None else f"no real data - {note}"))
    return out


def _accuracy_rows(items, cycle_starts, symbol, timeframe, interval, loop, overall, scored) -> list[dict[str, Any]]:
    n = len(items)
    rows = []
    for item in items:
        score = item["score"] or {"status": item["note"] or "no real data"}
        rows.append({
            "loop": loop, "cycle": item["cycle"], "run": (loop - 1) * n + item["cycle"],
            "symbol": symbol, "timeframe": timeframe, "candle_interval": interval,
            "window_start": pd.Timestamp(item["start"]).date().isoformat(),
            "window_end": item["sim"].index[-1].date().isoformat(),
            "status": score.get("status"), "real_interval": item["interval"],
            "valid_bars": score.get("valid_bars"), "total_bars": len(item["sim"]),
            "accuracy_index": score.get("index"),
            "score_path": score.get("score_path"), "score_direction": score.get("score_direction"),
            "score_net_move": score.get("score_net_move"), "score_range": score.get("score_range"),
            "rmse_pct": score.get("rmse_pct"),
            "sim_net_pct": score.get("sim_net_pct"), "real_net_pct": score.get("real_net_pct"),
            "sim_range_pct": score.get("sim_range_pct"), "real_range_pct": score.get("real_range_pct"),
        })
    rows.append({"loop": loop, "cycle": "OVERALL", "symbol": symbol, "timeframe": timeframe,
                 "candle_interval": interval, "status": f"{scored} of {n} cycles scored",
                 "accuracy_index": overall})
    return rows


def run_rewind_comparison(
    frames_by_loop: list[list[pd.DataFrame]],
    cycle_starts: list[date],
    timestamps: list[list[Any]],
    symbol: str,
    timeframe: str,
    interval: str,
    bar: dict[str, Any],
    output_dir: str | Path,
    today: date | None = None,
    fetcher: Fetcher | None = None,
) -> dict[str, Any]:
    """Score every cycle of every loop against real prices and write the rewind outputs.

    frames_by_loop[loop][cycle] is the simulated OHLC DataFrame (indexed by timestamp);
    cycle_starts / timestamps describe the (identical) cycle windows of every loop.
    Writes into output_dir (which must already exist):
        rewind_panels.png         every cycle of loop 1: simulated candles + real "ghost halo"
        rewind_chain_loop_L.png   the stitched continuous chain of loop L vs real price
        rewind_accuracy.csv       per-cycle component scores, per-loop and ALL overall rows
        rewind_loops_summary.csv  one row per loop
        rewind_real_prices.csv    the real OHLC aligned to the simulated candles
    """
    output = Path(output_dir)
    if not output.is_dir():
        raise FileNotFoundError(f"Analysis output folder does not exist (not creating it): {output}")
    today = today or date.today()
    loops, n = len(frames_by_loop), len(cycle_starts)
    real_info = prefetch_real(symbol, bar, timestamps, today, fetcher)

    all_rows, loop_rows, loop_values = [], [], []
    for loop, frames in enumerate(frames_by_loop, start=1):
        items = []
        for cycle, sim in enumerate(frames, start=1):
            info = real_info[cycle - 1]
            score = score_cycle(sim, info["real"]) if info["real"] is not None else None
            items.append({"loop": loop, "cycle": cycle, "sim": sim, "start": cycle_starts[cycle - 1], "end": None,
                          "real": info["real"], "interval": info["interval"], "note": info["note"], "score": score})
        values = [i["score"]["index"] for i in items if i["score"] and i["score"]["status"] == "scored"]
        overall = float(np.mean(values)) if values else float("nan")
        all_rows.extend(_accuracy_rows(items, cycle_starts, symbol, timeframe, interval, loop, overall, len(values)))
        loop_rows.append({"loop": loop, "symbol": symbol, "timeframe": timeframe, "candle_interval": interval,
                          "cycles": n, "cycles_scored": len(values), "accuracy_index": overall,
                          "best_cycle_index": max(values) if values else np.nan,
                          "worst_cycle_index": min(values) if values else np.nan})
        loop_values.append(overall)
        if loop == 1:
            plot_panels(items, symbol, timeframe, overall, output / "rewind_panels.png", interval)
        if loop <= MAX_STITCHED_PLOTS:
            plot_chain(items, symbol, timeframe, interval, loop, overall, output / f"rewind_chain_loop_{loop}.png")
        if loops <= 20 or loop == 1 or loop % max(1, loops // 20) == 0 or loop == loops:
            shown = f"{overall:+.3f}" if np.isfinite(overall) else "n/a"
            print(f"Loop {loop}/{loops}: accuracy index {shown}  ({len(values)} of {n} cycles scored)")

    finite = [v for v in loop_values if np.isfinite(v)]
    grand = float(np.mean(finite)) if finite else float("nan")
    spread = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
    all_rows.append({"loop": "ALL", "cycle": "OVERALL", "symbol": symbol, "timeframe": timeframe,
                     "candle_interval": interval, "status": f"{len(finite)} of {loops} loops scored",
                     "accuracy_index": grand})
    accuracy_path = output / "rewind_accuracy.csv"
    pd.DataFrame(all_rows).to_csv(accuracy_path, index=False)
    pd.DataFrame(loop_rows).to_csv(output / "rewind_loops_summary.csv", index=False)

    price_parts = []
    for cycle, (ts, info) in enumerate(zip(timestamps, real_info), start=1):
        if info["real"] is None:
            continue
        r = info["real"]
        price_parts.append(pd.DataFrame({
            "cycle": cycle, "bar": np.arange(1, len(ts) + 1), "Date": pd.DatetimeIndex(ts),
            "real_open": r["Open"].to_numpy(), "real_high": r["High"].to_numpy(),
            "real_low": r["Low"].to_numpy(), "real_close": r["Close"].to_numpy(), "real_interval": info["interval"]}))
    prices_path = output / "rewind_real_prices.csv"
    (pd.concat(price_parts, ignore_index=True) if price_parts else pd.DataFrame()).to_csv(prices_path, index=False)

    text = f"{grand:+.3f}" if np.isfinite(grand) else "n/a"
    sd = f" (std between loops {spread:.3f})" if np.isfinite(spread) else ""
    print(f"\nRewind accuracy index for {symbol}: {text}{sd}  [{len(finite)} of {loops} loops scored]")
    print(f"Saved panels      : {output / 'rewind_panels.png'}  (loop 1)")
    print(f"Saved chain plots : {output / 'rewind_chain_loop_N.png'}  (first {min(loops, MAX_STITCHED_PLOTS)} loop(s))")
    print(f"Saved tables      : {accuracy_path.name}, rewind_loops_summary.csv, {prices_path.name}")
    return {"overall_index": grand, "loops": loops, "cycles": n, "scored_loops": len(finite),
            "accuracy_csv": str(accuracy_path), "real_prices_csv": str(prices_path)}