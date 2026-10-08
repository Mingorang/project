"""Batch-analysis helpers for the looped OHLC CSV/PNG generator."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

FAST = 5
SLOW = 12
CONFIDENCE = 0.95
T_CRITICAL_DF11 = 2.200985160082949
TARGET_R = 2.0
Trade = dict[str, Any]


def _ema(values: pd.Series, span: int) -> pd.Series:
    return values.ewm(span=span, adjust=False, min_periods=span).mean()


def _wilder(values: pd.Series, period: int) -> pd.Series:
    return values.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _add_indicators(frame: pd.DataFrame, run_id: int) -> pd.DataFrame:
    frame = frame.copy()
    close = frame["Close"].astype(float)
    frame["run"] = run_id
    frame["bar"] = np.arange(1, len(frame) + 1)
    frame["fast_sma_5"] = close.rolling(FAST, min_periods=FAST).mean()
    frame["slow_sma_12"] = close.rolling(SLOW, min_periods=SLOW).mean()
    sd12 = close.rolling(SLOW, min_periods=SLOW).std(ddof=1)
    margin = T_CRITICAL_DF11 * sd12 / math.sqrt(SLOW)
    frame["ci95_lower_12"] = frame["slow_sma_12"] - margin
    frame["ci95_upper_12"] = frame["slow_sma_12"] + margin

    frame["bb_mid_20"] = close.rolling(20, min_periods=20).mean()
    sd20 = close.rolling(20, min_periods=20).std(ddof=1)
    frame["bb_upper_20_2sd"] = frame["bb_mid_20"] + 2 * sd20
    frame["bb_lower_20_2sd"] = frame["bb_mid_20"] - 2 * sd20
    band_width = frame["bb_upper_20_2sd"] - frame["bb_lower_20_2sd"]
    frame["bb_percent_b"] = (close - frame["bb_lower_20_2sd"]) / band_width.replace(0, np.nan)
    frame["bb_width_pct"] = 100 * band_width / frame["bb_mid_20"].abs().replace(0, np.nan)

    delta = close.diff()
    average_gain = _wilder(delta.clip(lower=0), 14)
    average_loss = _wilder(-delta.clip(upper=0), 14)
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    frame["rsi_14"] = 100 - 100 / (1 + relative_strength)
    frame.loc[(average_loss == 0) & (average_gain > 0), "rsi_14"] = 100
    frame.loc[(average_loss == 0) & (average_gain == 0), "rsi_14"] = 50

    frame["macd_12_26"] = _ema(close, 12) - _ema(close, 26)
    frame["macd_signal_9"] = _ema(frame["macd_12_26"], 9)
    frame["macd_histogram"] = frame["macd_12_26"] - frame["macd_signal_9"]

    previous_close = close.shift(1)
    true_range = pd.concat([
        frame["High"] - frame["Low"],
        (frame["High"] - previous_close).abs(),
        (frame["Low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    true_range.iloc[0] = frame["High"].iloc[0] - frame["Low"].iloc[0]
    frame["true_range"] = true_range
    frame["atr_14"] = _wilder(true_range, 14)
    frame["bar_change"] = close.diff()
    frame["bar_return_pct"] = np.where(previous_close > 0, 100 * (close / previous_close - 1), np.nan)
    frame["bar_return_pct"] = pd.Series(frame["bar_return_pct"], index=frame.index).replace([np.inf, -np.inf], np.nan)
    frame["direction"] = np.where(frame["Close"] >= frame["Open"], "Up", "Down")
    return frame


def _trade_from_signal(frame: pd.DataFrame, signal_index: int, side: str, run_id: int) -> Trade | None:
    entry_index = signal_index + 1
    if entry_index >= len(frame):
        return None
    opens: list[float] = frame["Open"].astype(float).tolist()
    highs: list[float] = frame["High"].astype(float).tolist()
    lows: list[float] = frame["Low"].astype(float).tolist()
    closes: list[float] = frame["Close"].astype(float).tolist()
    entry = float(opens[entry_index])
    stop = float(lows[signal_index] if side == "Long" else highs[signal_index])
    risk = entry - stop if side == "Long" else stop - entry
    if not np.isfinite(risk) or risk <= 0:
        return None
    target = entry + TARGET_R * risk if side == "Long" else entry - TARGET_R * risk
    exit_index = len(frame) - 1
    exit_price = float(closes[exit_index])
    reason = "time_exit"
    for index in range(entry_index, len(frame)):
        opening = float(opens[index])
        if (side == "Long" and opening <= stop) or (side == "Short" and opening >= stop):
            exit_index, exit_price, reason = index, opening, "stop_gap"
            break
        if (side == "Long" and opening >= target) or (side == "Short" and opening <= target):
            exit_index, exit_price, reason = index, opening, "target_gap"
            break
        stop_hit = lows[index] <= stop if side == "Long" else highs[index] >= stop
        target_hit = highs[index] >= target if side == "Long" else lows[index] <= target
        if stop_hit:
            exit_index, exit_price, reason = index, stop, "stop"
            break
        if target_hit:
            exit_index, exit_price, reason = index, target, "target"
            break
    pnl = exit_price - entry if side == "Long" else entry - exit_price
    return {
        "run": run_id, "side": side, "signal_index": signal_index + 1,
        "signal_time": frame.loc[signal_index, "Date"], "entry_index": entry_index + 1,
        "entry_time": frame.loc[entry_index, "Date"], "entry_price": entry,
        "stop_price": stop, "target_price": target, "risk_per_unit": risk,
        "exit_index": exit_index + 1, "exit_time": frame.loc[exit_index, "Date"],
        "exit_price": exit_price, "pnl_per_unit": pnl, "r_multiple": pnl / risk,
        "exit_reason": reason,
    }


def _trades_for_run(frame: pd.DataFrame, run_id: int) -> list[Trade]:
    difference = frame["fast_sma_5"] - frame["slow_sma_12"]
    trades = []
    for index in range(1, len(frame)):
        previous, current = difference.iloc[index - 1], difference.iloc[index]
        if not np.isfinite(previous) or not np.isfinite(current):
            continue
        side = "Long" if previous <= 0 < current else "Short" if previous >= 0 > current else None
        if side:
            trade = _trade_from_signal(frame, index, side, run_id)
            if trade:
                trades.append(trade)
    return trades


def _stat(values: pd.Series, function) -> float:
    values = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return float(function(values)) if len(values) else np.nan


def _best(trades: list[Trade], side: str) -> Trade | None:
    choices = [trade for trade in trades if trade["side"] == side]
    return max(choices, key=lambda trade: trade["r_multiple"], default=None)


def _metrics(frame: pd.DataFrame, trades: list[Trade], run_id: int, source_png: str) -> tuple[dict[str, Any], Trade | None, Trade | None]:
    close = frame["Close"]
    changes = frame["bar_change"].dropna()
    returns = frame["bar_return_pct"].dropna() / 100
    negative_returns = returns[returns < 0]
    drawdown_points = close.cummax() - close
    drawdown_pct = close / close.cummax() - 1 if (close > 0).all() else pd.Series(dtype=float)
    downside_deviation = math.sqrt(float(np.mean(np.square(negative_returns))) if len(negative_returns) else np.nan)
    valid_return_sd = returns.std(ddof=1) if len(returns) > 1 else np.nan
    sharpe = returns.mean() / valid_return_sd if np.isfinite(valid_return_sd) and valid_return_sd > 0 else np.nan
    sortino = returns.mean() / downside_deviation if np.isfinite(downside_deviation) and downside_deviation > 0 else np.nan
    price_range = frame["High"] - frame["Low"]
    body = (frame["Close"] - frame["Open"]).abs()
    gap = frame["Open"] - close.shift(1)
    difference = frame["fast_sma_5"] - frame["slow_sma_12"]
    best_long, best_short = _best(trades, "Long"), _best(trades, "Short")

    result = {
        "run": run_id, "source_csv": f"data_{run_id}.csv", "source_png": source_png,
        "bars": len(frame), "confidence_level": CONFIDENCE,
        "first_timestamp": frame.Date.iloc[0], "last_timestamp": frame.Date.iloc[-1],
        "first_open": float(frame.Open.iloc[0]), "last_close": float(close.iloc[-1]),
        "net_change_points": float(close.iloc[-1] - frame.Open.iloc[0]),
        "total_return_pct": 100 * (close.iloc[-1] / frame.Open.iloc[0] - 1) if frame.Open.iloc[0] > 0 else np.nan,
        "highest_high": float(frame.High.max()), "highest_high_time": frame.loc[frame.High.idxmax(), "Date"],
        "lowest_low": float(frame.Low.min()), "lowest_low_time": frame.loc[frame.Low.idxmin(), "Date"],
        "high_low_range_points": float(frame.High.max() - frame.Low.min()),
        "high_low_range_pct_of_low": 100 * (frame.High.max() - frame.Low.min()) / frame.Low.min() if frame.Low.min() > 0 else np.nan,
        "mean_close": float(close.mean()), "median_close": float(close.median()),
        "mean_bar_change_points": _stat(changes, np.mean), "median_bar_change_points": _stat(changes, np.median),
        "sd_bar_change_points": _stat(changes, lambda values: values.std(ddof=1)),
        "mean_valid_bar_return_pct": 100 * _stat(returns, np.mean),
        "median_valid_bar_return_pct": 100 * _stat(returns, np.median),
        "realized_vol_per_bar_pct": 100 * _stat(returns, lambda values: values.std(ddof=1)),
        "valid_return_observations": int(len(returns)), "sharpe_per_bar_no_rf": sharpe,
        "sortino_per_bar_no_rf": sortino, "max_drawdown_points": float(drawdown_points.max()),
        "max_drawdown_pct_positive_prices_only": 100 * float(drawdown_pct.min()) if len(drawdown_pct) else np.nan,
        "bullish_candles": int((frame.direction == "Up").sum()), "bearish_candles": int((frame.direction == "Down").sum()),
        "bullish_candle_pct": 100 * float((frame.direction == "Up").mean()),
        "bearish_candle_pct": 100 * float((frame.direction == "Down").mean()),
        "avg_candle_body_points": float(body.mean()), "avg_candle_range_points": float(price_range.mean()),
        "median_candle_range_points": float(price_range.median()), "avg_true_range_points": float(frame.true_range.mean()),
        "atr14_last": _stat(frame.atr_14.tail(1), np.mean), "rsi14_last": _stat(frame.rsi_14.tail(1), np.mean),
        "macd12_26_last": _stat(frame.macd_12_26.tail(1), np.mean),
        "macd_signal9_last": _stat(frame.macd_signal_9.tail(1), np.mean),
        "macd_histogram_last": _stat(frame.macd_histogram.tail(1), np.mean),
        "bollinger20_upper_last": _stat(frame.bb_upper_20_2sd.tail(1), np.mean),
        "bollinger20_lower_last": _stat(frame.bb_lower_20_2sd.tail(1), np.mean),
        "bollinger_percent_b_last": _stat(frame.bb_percent_b.tail(1), np.mean),
        "bollinger_width_pct_last": _stat(frame.bb_width_pct.tail(1), np.mean),
        "sma_bull_crossovers_5_12": int(((difference.shift(1) <= 0) & (difference > 0)).sum()),
        "sma_bear_crossovers_5_12": int(((difference.shift(1) >= 0) & (difference < 0)).sum()),
        "gap_bars": int((gap.abs() > 1e-9).sum()), "avg_abs_gap_points": _stat(gap.abs(), np.mean),
        "winning_crossover_trades": int(sum(trade["r_multiple"] > 0 for trade in trades)),
        "valid_crossover_trades": len(trades),
    }
    for side, prefix, trade in (("Long", "best_long", best_long), ("Short", "best_short", best_short)):
        for suffix, key in (("entry", "entry_time"), ("exit", "exit_time"), ("entry_price", "entry_price"),
                            ("stop", "stop_price"), ("target_2R", "target_price"), ("exit_price", "exit_price"),
                            ("pnl_per_unit", "pnl_per_unit"), ("r_multiple", "r_multiple"), ("exit_reason", "exit_reason")):
            result[f"{prefix}_{suffix}"] = trade[key] if trade else ("no valid trade" if suffix == "exit_reason" else pd.NaT if suffix in ("entry", "exit") else np.nan)
    return result, best_long, best_short


def _draw_chart(frame: pd.DataFrame, run_id: int, best_long: Trade | None, best_short: Trade | None, output: Path, guide_path: Path, instrument: str | None = None, chart_label: str | None = None) -> None:
    x = frame["bar"].astype(int).tolist()
    figure, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                                gridspec_kw={"height_ratios": [4.2, 1.2, 1.2]}, facecolor="black")
    price_ax, rsi_ax, macd_ax = axes
    for axis in axes:
        axis.set_facecolor("black")
        axis.tick_params(colors="white")
        for spine in axis.spines.values():
            spine.set_color("#1e61bd")

    price_ax.yaxis.tick_right()
    price_ax.yaxis.set_label_position("right")
    price_ax.fill_between(x, frame.ci95_lower_12.to_numpy(dtype=float), frame.ci95_upper_12.to_numpy(dtype=float),
                          color="#7aa6d8", alpha=.22, label="95% t interval: 12-close mean")
    price_ax.fill_between(x, frame.bb_lower_20_2sd.to_numpy(dtype=float), frame.bb_upper_20_2sd.to_numpy(dtype=float),
                          color="#aeb7c2", alpha=.13, label="Bollinger 20 +/- 2 SD")
    opens: list[float] = frame["Open"].astype(float).tolist()
    highs: list[float] = frame["High"].astype(float).tolist()
    lows: list[float] = frame["Low"].astype(float).tolist()
    closes: list[float] = frame["Close"].astype(float).tolist()
    for xpos, (open_price, high, low, close) in enumerate(zip(opens, highs, lows, closes), start=1):
        color = "#21d6a2" if close >= open_price else "#ff6b77"
        price_ax.vlines(xpos, low, high, color="#d7dde5", linewidth=.65, zorder=2)
        body_low = min(open_price, close)
        # The former 0.012 minimum made low-volatility candles dwarf the data and skew price-axis autoscaling.
        body_height = abs(close - open_price)
        price_ax.add_patch(Rectangle((xpos-.30, body_low), .60, body_height, facecolor=color,
                                     edgecolor="#d7dde5", linewidth=.3, zorder=3))
    price_ax.plot(x, frame.fast_sma_5, color="#66b6ff", linewidth=1.0, label="SMA 5")
    price_ax.plot(x, frame.slow_sma_12, color="#ffc857", linewidth=1.1, label="SMA 12")
    price_ax.plot(x, frame.bb_mid_20, color="#ccd4df", linewidth=.7, linestyle="--", label="BB mid 20")
    for side, trade, color in (("Long", best_long, "#21d6a2"), ("Short", best_short, "#ff6b77")):
        if trade:
            price_ax.scatter(trade["entry_index"], trade["entry_price"], marker="^" if side == "Long" else "v",
                             color=color, s=48, zorder=5)
            price_ax.annotate(f"{side[0]} in {trade['r_multiple']:+.2f}R", (trade["entry_index"], trade["entry_price"]),
                              xytext=(4, 8 if side == "Long" else -14), textcoords="offset points", fontsize=7, color=color, weight="bold")
            price_ax.scatter(trade["exit_index"], trade["exit_price"], marker="x", color=color, s=38, zorder=5)
            price_ax.annotate(f"{side[0]} out", (trade["exit_index"], trade["exit_price"]),
                              xytext=(4, -12), textcoords="offset points", fontsize=7, color=color)
    price_ax.set_ylabel("Share price", color="white")
    # Use provided instrument metadata if available; fall back to blank.
    title_instrument = ticker_name if 'ticker_name' in globals() else ''
    if 'instrument' in globals() and instrument:
        title_instrument = instrument
    if chart_label:
        # Yahoo Finance symbol supplied by the caller (e.g. GBPUSD=X) takes priority.
        title_instrument = chart_label
    if title_instrument:
        price_ax.set_title(f"Run {run_id}:  {title_instrument} OHLC + SMA / Bollinger / 95% interval", color="white")
    else:
        price_ax.set_title(f"Run {run_id}: OHLC + SMA / Bollinger / 95% interval", color="white")
    price_ax.legend(loc="upper left", fontsize=7, ncol=2, facecolor="#111111", edgecolor="#aab2bd",
                    labelcolor="white", framealpha=.9)
    price_ax.grid(True, color="#3a414b", linewidth=.4)

    rsi_ax.plot(x, frame.rsi_14, color="#c4a7ff", linewidth=.85)
    rsi_ax.axhline(70, color="#ff8585", linestyle="--", linewidth=.6)
    rsi_ax.axhline(30, color="#58ddb4", linestyle="--", linewidth=.6)
    rsi_ax.set_ylim(0, 100)
    rsi_ax.set_ylabel("RSI 14", color="white")
   #rsi_ax.set_xlabel("Study [4]: RSI guide in analysis_learning_resources.md", color="white")
    rsi_ax.grid(True, color="#3a414b", linewidth=.4)

    macd_ax.bar(x, frame.macd_histogram, color=np.where(frame.macd_histogram >= 0, "#16846b", "#d94c5c"), width=.65, alpha=.65)
    macd_ax.plot(x, frame.macd_12_26, color="#66b6ff", linewidth=.8, label="MACD 12-26")
    macd_ax.plot(x, frame.macd_signal_9, color="#ffc857", linewidth=.8, label="Signal 9")
    macd_ax.axhline(0, color="#777777", linewidth=.55)
    macd_ax.set_ylabel("MACD", color="white")
    #macd_ax.set_xlabel("Study [5]: MACD guide in analysis_learning_resources.md", color="white")
    macd_ax.legend(loc="upper left", fontsize=7, ncol=2, facecolor="#3D1A1A", edgecolor="#aab2bd",
                   labelcolor="white", framealpha=.9)
    macd_ax.grid(True, color="#3a414b", linewidth=.4)
    dates = pd.to_datetime(frame.Date).dt.date
    ticks = np.flatnonzero(dates.ne(dates.shift()).to_numpy())
    
    # sample ticks to target roughly N labels (change N to taste)
    target_labels = 10
    if len(ticks) > target_labels:
        step = max(1, len(ticks) // target_labels)
        ticks = ticks[::step]

    labels = [pd.Timestamp(dates.iloc[i]).strftime("%a %d %b") for i in ticks]
    if len(ticks) <= 1 and len(frame) > 1:
        # Single-day chart (1d timeframe): one date label is useless, so label the
        # clock times of ~8 evenly spaced candles instead.
        ticks = np.unique(np.linspace(0, len(frame) - 1, min(8, len(frame))).round().astype(int))
        labels = [pd.Timestamp(frame.Date.iloc[i]).strftime("%H:%M") for i in ticks]
    macd_ax.set_xticks(ticks + 1, labels, rotation=25, ha="right")
    figure.text(.01, .005, f"Study links [1]-[11]: {guide_path.name}. PNG labels are references; open the companion Markdown for clickable URLs. Best crossovers are hindsight only; no fees/slippage.", fontsize=7, color="white")
    figure.tight_layout(rect=(0, .035, 1, 1))
    figure.savefig(output, dpi=135, facecolor="black", bbox_inches="tight")
    plt.close(figure)


def _contact_sheet(
    items: list[tuple[int, Path]],
    output: Path,
    columns: int = 8,
    tile_width: int = 310,
    tile_height: int = 185,
    max_tiles: int = 256,
) -> None:
    for page_number, start in enumerate(range(0, len(items), max_tiles), start=1):
        page_items = items[start:start + max_tiles]
        rows = math.ceil(len(page_items) / columns)
        sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
        draw = ImageDraw.Draw(sheet)
        for position, (run_id, path) in enumerate(page_items):
            x, y = (position % columns) * tile_width, (position // columns) * tile_height
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((tile_width - 12, tile_height - 28), Image.Resampling.LANCZOS)
                sheet.paste(image, (x + (tile_width - image.width) // 2, y + 22))
            draw.text((x + 6, y + 4), f"Run {run_id}", fill="#111111")

        if len(items) > max_tiles:
            page_output = output.with_name(f"{output.stem}_page_{page_number:03}{output.suffix}")
        else:
            page_output = output
        sheet.save(page_output)


def _overall_metrics(summary: pd.DataFrame, bars: pd.DataFrame, trades: list[Trade]) -> pd.DataFrame:
    high_values: list[float] = bars["High"].astype(float).tolist()
    low_values: list[float] = bars["Low"].astype(float).tolist()
    run_values: list[int] = bars["run"].astype(int).tolist()
    date_values: list[Any] = bars["Date"].tolist()
    high_index = high_values.index(max(high_values))
    low_index = low_values.index(min(low_values))
    highest_volatility_values: list[float] = summary["realized_vol_per_bar_pct"].astype(float).tolist()
    summary_run_values: list[int] = summary["run"].astype(int).tolist()
    valid_volatility_indices = [index for index, value in enumerate(highest_volatility_values) if np.isfinite(value)]
    highest_volatility_index = max(valid_volatility_indices, key=highest_volatility_values.__getitem__, default=None)
    best_long = _best(trades, "Long")
    best_short = _best(trades, "Short")
    returns = summary.total_return_pct.dropna()
    values = [
        ("run_pairs", len(summary), "count"), ("ohlc_bars", len(bars), "count"),
        ("invalid_ohlc_rows", int(((bars.High < bars[["Open", "Close"]].max(axis=1)) | (bars.Low > bars[["Open", "Close"]].min(axis=1))).sum()), "count"),
        ("highest_high", high_values[high_index], f"run {run_values[high_index]} at {date_values[high_index]}"),
        ("lowest_low", low_values[low_index], f"run {run_values[low_index]} at {date_values[low_index]}"),
        ("average_run_return_pct", _stat(returns, np.mean), "percent, positive-price simple close/open"),
        ("median_run_return_pct", _stat(returns, np.median), "percent, positive-price simple close/open"),
        ("positive_return_runs", int((summary.total_return_pct > 0).sum()), "of runs"),
        ("negative_return_runs", int((summary.total_return_pct < 0).sum()), "of runs"),
        ("median_realized_volatility_per_bar_pct", _stat(summary.realized_vol_per_bar_pct, np.median), "percent, unannualized"),
        ("highest_volatility_run", summary_run_values[highest_volatility_index] if highest_volatility_index is not None else np.nan, "run id"),
        ("median_high_low_range_points", _stat(summary.high_low_range_points, np.median), "price points"),
        ("mean_bullish_candle_share_pct", _stat(summary.bullish_candle_pct, np.mean), "percent"),
        ("mean_bearish_candle_share_pct", _stat(summary.bearish_candle_pct, np.mean), "percent"),
        ("total_bullish_sma_crossovers", int(summary.sma_bull_crossovers_5_12.sum()), "count"),
        ("total_bearish_sma_crossovers", int(summary.sma_bear_crossovers_5_12.sum()), "count"),
        ("candidate_crossover_trades", len(trades), "count; next-bar open entries"),
        ("best_long_r_multiple", best_long["r_multiple"] if best_long else np.nan, f"run {best_long['run']}" if best_long else "none"),
        ("best_short_r_multiple", best_short["r_multiple"] if best_short else np.nan, f"run {best_short['run']}" if best_short else "none"),
        ("runs_with_positive_best_long", int((summary.best_long_r_multiple > 0).sum()), "runs"),
        ("runs_with_positive_best_short", int((summary.best_short_r_multiple > 0).sum()), "runs"),
        ("mean_atr14_last", _stat(summary.atr14_last, np.mean), "price points"),
        ("mean_rsi14_last", _stat(summary.rsi14_last, np.mean), "0-100 scale"),
        ("mean_macd_histogram_last", _stat(summary.macd_histogram_last, np.mean), "price points"),
        ("mean_bollinger_width_last_pct", _stat(summary.bollinger_width_pct_last, np.mean), "percent"),
        ("volume_indicators", "not available", "input CSVs contain no Volume column"),
    ]
    return pd.DataFrame(values, columns=["measure", "value", "context"])


def analyze_looped_results(
    source_dir: str | Path,
    output_dir: str | Path,
    render_charts: bool = True,
    render_source_charts: bool = True,
    timeframe: str | None = None,
    instrument: str | None = None,
    asset_type: str | None = None,
    sim_date: str | None = None,
    is_backtest: int | None = None,
    chart_label: str | None = None,
) -> dict:
    """Analyze CSV runs and optionally label outputs with their simulation timeframe.

    Optional metadata (instrument, asset_type, sim_date, is_backtest) can be
    provided by the caller and will be inserted into the analysis summary if
    supplied.
    """
    source = Path(source_dir)
    output = Path(output_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {source}")
    if not output.is_dir():
        raise FileNotFoundError(f"Analysis output folder does not exist (not creating it): {output}")
    guide_path = Path(__file__).with_name("analysis_learning_resources.md")
    csv_paths = sorted(source.glob("data_*.csv"), key=lambda path: int(path.stem.split("_")[-1]))
    png_ids = {int(path.stem.split("_")[-1]) for path in source.glob("image_*.png")}
    csv_ids = {int(path.stem.split("_")[-1]) for path in csv_paths}
    run_ids = sorted(csv_ids & png_ids) if render_source_charts else sorted(csv_ids)
    if not run_ids:
        expected = "matching data_N.csv and image_N.png pairs" if render_source_charts else "data_N.csv files"
        raise ValueError(f"No {expected} in {source}")
    if render_source_charts and csv_ids != png_ids:
        raise ValueError(f"Unmatched files: CSV-only={sorted(csv_ids-png_ids)}, PNG-only={sorted(png_ids-csv_ids)}")

    summaries, processed_frames, all_trades, analysis_images = [], [], [], []
    for run_id in run_ids:
        csv_path, png_path = source / f"data_{run_id}.csv", source / f"image_{run_id}.png"
        frame = pd.read_csv(csv_path)
        frame.columns = frame.columns.astype(str).str.strip()
        required = {"Date", "Open", "High", "Low", "Close"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{csv_path.name} lacks required columns: {sorted(required-set(frame.columns))}")
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame = frame.sort_values("Date").reset_index(drop=True)
        numeric = frame[["Open", "High", "Low", "Close"]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{csv_path.name} has non-finite OHLC values")
        if ((frame.High < frame[["Open", "Close"]].max(axis=1)) | (frame.Low > frame[["Open", "Close"]].min(axis=1))).any():
            raise ValueError(f"{csv_path.name} has invalid OHLC candle ordering")
        if len(frame) < SLOW:
            raise ValueError(f"{csv_path.name} has fewer than {SLOW} rows")
        frame = _add_indicators(frame, run_id)
        trades = _trades_for_run(frame, run_id)
        all_trades.extend(trades)
        source_png = png_path.name if png_path.exists() else ""
        metrics, long_trade, short_trade = _metrics(frame, trades, run_id, source_png)
        summaries.append(metrics)
        processed_frames.append(frame)
        if render_charts:
            chart_path = output / f"image_{run_id}_analysis.png"
            _draw_chart(frame, run_id, long_trade, short_trade, chart_path, guide_path, instrument=instrument, chart_label=chart_label)
            analysis_images.append((run_id, chart_path))

    summary = pd.DataFrame(summaries).sort_values("run")
    bars = pd.concat(processed_frames, ignore_index=True)
    # Insert optional metadata fields into the summary and bars if provided by caller.
    if timeframe is not None:
        summary.insert(1, "timeframe", timeframe)
        bars.insert(2, "timeframe", timeframe)
    if instrument is not None:
        # place instrument column near the front (after run/timeframe)
        insert_idx = 1 if "timeframe" not in summary.columns else 2
        summary.insert(insert_idx, "instrument", instrument)
        bars.insert(1, "instrument", instrument)
    if asset_type is not None:
        insert_idx = 2 if "timeframe" not in summary.columns else 3
        summary.insert(insert_idx, "asset_type", asset_type)
        bars.insert(2, "asset_type", asset_type)
    if sim_date is not None:
        # sim_date belongs in the summary metadata, not the bars listing
        # place after asset_type (if present) or after instrument
        try:
            idx = summary.columns.get_loc("asset_type") + 1
        except KeyError:
            try:
                idx = summary.columns.get_loc("instrument") + 1
            except KeyError:
                idx = 1
        summary.insert(idx, "sim_date", sim_date)
    if is_backtest is not None:
        try:
            idx = summary.columns.get_loc("sim_date") + 1
        except KeyError:
            try:
                idx = summary.columns.get_loc("asset_type") + 1
            except KeyError:
                idx = 1
        summary.insert(idx, "is_backtest", int(is_backtest))

    summary.to_csv(output / "analysis_summary.csv", index=False)
    bars.to_csv(output / "processed_bars.csv", index=False)
    trade_columns = ["run", "side", "signal_index", "signal_time", "entry_index", "entry_time", "entry_price", "stop_price", "target_price", "risk_per_unit", "exit_index", "exit_time", "exit_price", "pnl_per_unit", "r_multiple", "exit_reason"]
    pd.DataFrame(all_trades, columns=trade_columns).to_csv(output / "crossover_trades.csv", index=False)
    overall = _overall_metrics(summary, bars, all_trades)
    if timeframe is not None:
        overall.insert(0, "timeframe", timeframe)
    overall.to_csv(output / "overall_metrics.csv", index=False)
    if render_source_charts:
        _contact_sheet([(run_id, source / f"image_{run_id}.png") for run_id in run_ids], output / "all_source_charts_contact_sheet.png")
    if render_charts:
        _contact_sheet(analysis_images, output / "all_analysis_charts_contact_sheet.png")
    return {"runs": len(run_ids), "bars": len(bars), "trades": len(all_trades), "output_dir": str(output)}


def main() -> None:
    model_dir = Path(__file__).resolve().parent
    analysis_dir = model_dir / "looped_analysis"
    result = analyze_looped_results(model_dir / "looped_results", analysis_dir)
    print(f"Analyzed {result['runs']} pairs / {result['bars']} bars / {result['trades']} trades -> {result['output_dir']}")


if __name__ == "__main__":
    main()