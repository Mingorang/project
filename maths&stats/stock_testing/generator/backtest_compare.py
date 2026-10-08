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
) -> dict[str, Any]:
    """Score every backtest cycle against real prices and write ONE combined plot.

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
    print(f"\nFetching real {symbol} prices for {total} backtest cycle(s)...")
    for position, cycle in enumerate(cycles, start=1):
        sim: pd.DataFrame = cycle["sim"]
        real, real_interval, note = fetch_real_for_cycle(symbol, bar, sim.index, today, fetcher)
        score = score_cycle(sim, real) if real is not None else None
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
            "rmse_pct": score.get("rmse_pct"),
            "sim_net_pct": score.get("sim_net_pct"), "real_net_pct": score.get("real_net_pct"),
            "sim_range_pct": score.get("sim_range_pct"), "real_range_pct": score.get("real_range_pct"),
        })
    rows.append({
        "cycle": "OVERALL", "symbol": symbol, "timeframe": timeframe,
        "candle_interval": interval or "default",
        "status": f"{len(scored_values)} of {total} cycles scored", "accuracy_index": overall,
    })
    accuracy_path = output / "backtest_accuracy.csv"
    pd.DataFrame(rows).to_csv(accuracy_path, index=False)
    prices_path = output / "backtest_real_prices.csv"
    (pd.concat(price_rows, ignore_index=True) if price_rows else pd.DataFrame()).to_csv(prices_path, index=False)

    plot_path = output / "backtest_plot.png"
    plot_backtest(items, symbol, timeframe, overall, plot_path, interval)

    summary = (f"{overall:+.3f}" if np.isfinite(overall) else "n/a")
    print(f"\nBacktest accuracy index for {symbol}: {summary}  "
          f"({len(scored_values)} of {total} cycles scored)")
    print(f"Saved single backtest plot : {plot_path}")
    print(f"Saved accuracy table       : {accuracy_path}")
    return {"overall_index": overall, "scored": len(scored_values), "total": total,
            "plot": str(plot_path), "accuracy_csv": str(accuracy_path), "real_prices_csv": str(prices_path)}