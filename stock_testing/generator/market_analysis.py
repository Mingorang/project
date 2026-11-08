"""Batch-analysis helpers for the looped OHLC CSV/PNG generator."""
from __future__ import annotations

import io
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# Indicator settings: config.toml [indicators] is the single source (see configure()). The values
# below are only the import-time defaults; configure() replaces them before any analysis runs.
FAST = 4                 # fast SMA window (crossover trigger)
SLOW = 14                # slow SMA window = t-interval window = fewest rows a run may have
CONFIDENCE = 0.975       # two-sided confidence level of the t-interval around the slow-SMA mean
T_CRITICAL = 2.533       # t quantile for CONFIDENCE with SLOW - 1 degrees of freedom (derived, never typed)
TARGET_R = 1.4           # crossover take-profit in multiples of the initial risk
BB_WINDOW, BB_SD = 20, 2.0
RSI_PERIOD, RSI_HIGH, RSI_LOW = 14, 70.0, 30.0
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD = 14
Trade = dict[str, Any]


def t_critical(confidence: float, window: int) -> float:
    """Two-sided Student-t critical value for a `confidence` interval on the mean of `window` values."""
    # scipy.special.stdtrit is the routine stats.t.ppf calls (bit-identical on 3,493 (q, df) pairs); importing it
    # instead of scipy.stats saves ~1 s, and this runs on every `import market_analysis` (configure at load).
    from scipy.special import stdtrit
    return float(stdtrit(window - 1, 1.0 - (1.0 - confidence) / 2.0))


SETTINGS: dict[str, Any] = {}   # the [indicators] table last applied (sent to chart worker processes)


def configure(indicators: dict[str, Any]) -> None:
    """Apply config.toml [indicators] (already validated by project_config.load_config)."""
    global FAST, SLOW, CONFIDENCE, T_CRITICAL, TARGET_R, BB_WINDOW, BB_SD, SETTINGS
    global RSI_PERIOD, RSI_HIGH, RSI_LOW, MACD_FAST, MACD_SLOW, MACD_SIGNAL, ATR_PERIOD
    SETTINGS = dict(indicators)
    FAST, SLOW = int(indicators["fast_sma"]), int(indicators["slow_sma"])
    CONFIDENCE, TARGET_R = float(indicators["ci_level"]), float(indicators["target_r"])
    T_CRITICAL = t_critical(CONFIDENCE, SLOW)
    BB_WINDOW, BB_SD = int(indicators["bollinger_window"]), float(indicators["bollinger_sd"])
    RSI_PERIOD = int(indicators["rsi_period"])
    RSI_HIGH, RSI_LOW = float(indicators["rsi_overbought"]), float(indicators["rsi_oversold"])
    MACD_FAST, MACD_SLOW, MACD_SIGNAL = int(indicators["macd_fast"]), int(indicators["macd_slow"]), int(indicators["macd_signal"])
    ATR_PERIOD = int(indicators["atr_period"])


def _configure_from_file() -> None:
    try:
        from project_config import ConfigError, section
        configure(section("indicators"))
    except ConfigError as error:            # standalone use with a broken file: say so, keep the defaults
        print(f"[analysis] {error}\n[analysis] Using the built-in indicator defaults.")
        configure({"fast_sma": FAST, "slow_sma": SLOW, "ci_level": CONFIDENCE, "target_r": TARGET_R,
                   "bollinger_window": BB_WINDOW, "bollinger_sd": BB_SD, "rsi_period": RSI_PERIOD,
                   "rsi_overbought": RSI_HIGH, "rsi_oversold": RSI_LOW, "macd_fast": MACD_FAST,
                   "macd_slow": MACD_SLOW, "macd_signal": MACD_SIGNAL, "atr_period": ATR_PERIOD})


def _pct_label(level: float) -> str:
    return f"{100.0 * level:.4g}%"


_configure_from_file()


def _ema(values: pd.Series | pd.DataFrame, span: int) -> pd.Series | pd.DataFrame:
    return values.ewm(span=span, adjust=False, min_periods=span).mean()


def _wilder(values: pd.Series | pd.DataFrame, period: int) -> pd.Series | pd.DataFrame:
    return values.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


# Columns _add_indicators appends, in order (after "run" and "bar"); "direction" is last.
INDICATOR_COLUMNS = (
    "fast_sma_5", "slow_sma_12", "ci95_lower_12", "ci95_upper_12", "bb_mid_20", "bb_upper_20_2sd",
    "bb_lower_20_2sd", "bb_percent_b", "bb_width_pct", "rsi_14", "macd_12_26", "macd_signal_9",
    "macd_histogram", "true_range", "atr_14", "bar_change", "bar_return_pct",
)


def _indicators_wide_pandas(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> dict[str, np.ndarray]:
    """Every indicator for k runs of the same length at once: inputs are (bars, k) arrays, one column per run.

    pandas runs its rolling / ewm kernels column by column, so column j is bit-for-bit what the one-run
    calculation gives; doing all runs in one call removes ~40 pandas calls per run (the analysis of
    1,000 runs went from ~31 ms of indicators per run to well under 1 ms).
    """
    c = pd.DataFrame(close)
    out: dict[str, np.ndarray] = {}
    fast = c.rolling(FAST, min_periods=FAST).mean()
    slow = c.rolling(SLOW, min_periods=SLOW).mean()
    sd_slow = c.rolling(SLOW, min_periods=SLOW).std(ddof=1)
    margin = T_CRITICAL * sd_slow / math.sqrt(SLOW)
    out["fast_sma_5"], out["slow_sma_12"] = fast.to_numpy(), slow.to_numpy()
    out["ci95_lower_12"], out["ci95_upper_12"] = (slow - margin).to_numpy(), (slow + margin).to_numpy()

    mid = c.rolling(BB_WINDOW, min_periods=BB_WINDOW).mean()
    sd_bb = c.rolling(BB_WINDOW, min_periods=BB_WINDOW).std(ddof=1)
    upper, lower = mid + BB_SD * sd_bb, mid - BB_SD * sd_bb
    band_width = upper - lower
    out["bb_mid_20"], out["bb_upper_20_2sd"], out["bb_lower_20_2sd"] = mid.to_numpy(), upper.to_numpy(), lower.to_numpy()
    out["bb_percent_b"] = ((c - lower) / band_width.replace(0, np.nan)).to_numpy()
    out["bb_width_pct"] = (100 * band_width / mid.abs().replace(0, np.nan)).to_numpy()

    delta = c.diff()
    average_gain = _wilder(delta.clip(lower=0), RSI_PERIOD).to_numpy()
    average_loss = _wilder(-delta.clip(upper=0), RSI_PERIOD).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_strength = average_gain / np.where(average_loss == 0, np.nan, average_loss)
        rsi = 100 - 100 / (1 + relative_strength)
    rsi = np.where((average_loss == 0) & (average_gain > 0), 100.0, rsi)
    out["rsi_14"] = np.where((average_loss == 0) & (average_gain == 0), 50.0, rsi)

    macd = _ema(c, MACD_FAST) - _ema(c, MACD_SLOW)
    signal = _ema(macd, MACD_SIGNAL)
    out["macd_12_26"], out["macd_signal_9"] = macd.to_numpy(), signal.to_numpy()
    out["macd_histogram"] = (macd - signal).to_numpy()

    previous_close = np.vstack([np.full((1, close.shape[1]), np.nan), close[:-1]])
    with np.errstate(invalid="ignore"):
        # = pd.concat([...], axis=1).max(axis=1): fmax skips the NaN of the first bar, like skipna
        true_range = np.fmax(np.fmax(high - low, np.abs(high - previous_close)), np.abs(low - previous_close))
    true_range[0] = high[0] - low[0]
    out["true_range"] = true_range
    out["atr_14"] = _wilder(pd.DataFrame(true_range), ATR_PERIOD).to_numpy()
    out["bar_change"] = close - previous_close
    with np.errstate(divide="ignore", invalid="ignore"):
        bar_return = np.where(previous_close > 0, 100 * (close / previous_close - 1), np.nan)
    out["bar_return_pct"] = np.where(np.isinf(bar_return), np.nan, bar_return)
    return out



# pandas runs a rolling / ewm kernel once PER COLUMN (one Python round trip per run), which made the indicators
# ~1.7 ms per run. The kernels themselves take explicit window bounds, so the fast path lays all k runs end to
# end in one array and calls each kernel once with bounds that never cross a run: the same kernel doing the same
# additions in the same order for every window, so the same bits. pandas' kernels are private, so the fast path
# compares itself with the pandas path once per process and is used only if every value matches.
_KERNELS: Any = None            # None = not checked yet, False = use the pandas path, else the kernel functions


def _run_windows(k: int, n: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling-window bounds for k runs of n bars end to end (= pandas' FixedWindowIndexer within each run)."""
    end = np.arange(1, k * n + 1, dtype=np.int64)
    return np.maximum(end - window, (np.arange(k * n, dtype=np.int64) // n) * n), end


def _indicators_runs(kernels: Any, open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                     close: np.ndarray) -> dict[str, np.ndarray]:
    roll_mean, roll_var, ewm, zsqrt = kernels
    n, k = close.shape
    c, h, l = (np.ascontiguousarray(np.asarray(a, dtype=np.float64).T) for a in (close, high, low))   # (k, n)
    run_start = np.arange(k, dtype=np.int64) * n
    run_end = run_start + n
    bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def prep(values: np.ndarray) -> np.ndarray:                 # = BaseWindow._prep_values
        values = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
        infinite = np.isinf(values)
        return np.where(infinite, np.nan, values) if infinite.any() else values

    def window(w: int) -> tuple[np.ndarray, np.ndarray]:
        if w not in bounds:
            bounds[w] = _run_windows(k, n, w)
        return bounds[w]

    def mean(values: np.ndarray, w: int) -> np.ndarray:        # = .rolling(w, min_periods=w).mean()
        return roll_mean(prep(values), *window(w), w).reshape(k, n)

    def std(values: np.ndarray, w: int) -> np.ndarray:         # = .rolling(w, min_periods=w).std(ddof=1)
        return zsqrt(roll_var(prep(values), *window(w), w, ddof=1)).reshape(k, n)

    def ewm_mean(values: np.ndarray, **options: Any) -> np.ndarray:
        spec = pd.Series([0.0]).ewm(adjust=False, **options)   # pandas' own com / min_periods for these options
        return ewm(prep(values), run_start, run_end, spec.min_periods, com=spec._com, adjust=spec.adjust,
                   ignore_na=spec.ignore_na, deltas=None, normalize=True).reshape(k, n)

    out: dict[str, np.ndarray] = {}
    fast, slow, sd_slow = mean(c, FAST), mean(c, SLOW), std(c, SLOW)
    margin = T_CRITICAL * sd_slow / math.sqrt(SLOW)
    out["fast_sma_5"], out["slow_sma_12"] = fast, slow
    out["ci95_lower_12"], out["ci95_upper_12"] = slow - margin, slow + margin
    mid, sd_bb = mean(c, BB_WINDOW), std(c, BB_WINDOW)
    upper, lower = mid + BB_SD * sd_bb, mid - BB_SD * sd_bb
    band_width = upper - lower
    out["bb_mid_20"], out["bb_upper_20_2sd"], out["bb_lower_20_2sd"] = mid, upper, lower
    out["bb_percent_b"] = (c - lower) / np.where(band_width == 0, np.nan, band_width)
    absolute_mid = np.abs(mid)
    out["bb_width_pct"] = 100 * band_width / np.where(absolute_mid == 0, np.nan, absolute_mid)

    delta = np.full((k, n), np.nan)
    delta[:, 1:] = c[:, 1:] - c[:, :-1]                         # = DataFrame.diff()
    missing = np.isnan(delta)
    gain = np.where(missing | (delta >= 0), delta, 0.0)         # = .clip(lower=0), NaN kept
    loss = -np.where(missing | (delta <= 0), delta, 0.0)        # = -.clip(upper=0)
    average_gain = ewm_mean(gain, alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD)
    average_loss = ewm_mean(loss, alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_strength = average_gain / np.where(average_loss == 0, np.nan, average_loss)
        rsi = 100 - 100 / (1 + relative_strength)
    rsi = np.where((average_loss == 0) & (average_gain > 0), 100.0, rsi)
    out["rsi_14"] = np.where((average_loss == 0) & (average_gain == 0), 50.0, rsi)

    macd = ewm_mean(c, span=MACD_FAST, min_periods=MACD_FAST) - ewm_mean(c, span=MACD_SLOW, min_periods=MACD_SLOW)
    signal = ewm_mean(macd, span=MACD_SIGNAL, min_periods=MACD_SIGNAL)
    out["macd_12_26"], out["macd_signal_9"], out["macd_histogram"] = macd, signal, macd - signal

    previous_close = np.full((k, n), np.nan)
    previous_close[:, 1:] = c[:, :-1]
    with np.errstate(invalid="ignore"):
        true_range = np.fmax(np.fmax(h - l, np.abs(h - previous_close)), np.abs(l - previous_close))
    true_range[:, 0] = h[:, 0] - l[:, 0]
    out["true_range"] = true_range
    out["atr_14"] = ewm_mean(true_range, alpha=1 / ATR_PERIOD, min_periods=ATR_PERIOD)
    out["bar_change"] = c - previous_close
    with np.errstate(divide="ignore", invalid="ignore"):
        bar_return = np.where(previous_close > 0, 100 * (c / previous_close - 1), np.nan)
    out["bar_return_pct"] = np.where(np.isinf(bar_return), np.nan, bar_return)
    return {name: values.T for name, values in out.items()}     # (bars, k) like the pandas path


def _kernels() -> Any:
    """pandas' window kernels if the end-to-end path matches the pandas path exactly here, else False."""
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = False
        try:
            from pandas._libs.window.aggregations import ewm, roll_mean, roll_var
            from pandas.core.window.common import zsqrt
            kernels = (roll_mean, roll_var, ewm, zsqrt)
            rng = np.random.default_rng(12345)
            for n, k in ((max(SLOW, MACD_SLOW + MACD_SIGNAL) + 7, 5), (SLOW, 3), (1, 2)):
                close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, k)), axis=0))
                close[:, 1] = 50.0                                       # a flat run (zero variance, RSI 50)
                close[n // 2:, 0] *= 1e-9                                # a collapse to tiny prices
                wick = np.abs(rng.normal(0, 0.5, (2, n, k)))
                open_ = np.vstack([close[:1], close[:-1]])
                high, low = np.maximum(open_, close) + wick[0], np.minimum(open_, close) - wick[1]
                want = _indicators_wide_pandas(open_, high, low, close)
                got = _indicators_runs(kernels, open_, high, low, close)
                for name in INDICATOR_COLUMNS:
                    a, b = np.asarray(want[name], dtype=float), np.asarray(got[name], dtype=float)
                    if (a.shape != b.shape or not np.array_equal(a, b, equal_nan=True)
                            or not np.array_equal(np.signbit(a), np.signbit(b))):
                        return _KERNELS
            _KERNELS = kernels
        except Exception:                                                # internals moved: keep the pandas path
            _KERNELS = False
    return _KERNELS


def _indicators_wide(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> dict[str, np.ndarray]:
    """Every indicator for k runs of the same length at once: inputs are (bars, k) arrays, one column per run."""
    kernels = _kernels() if close.size else False
    if kernels:
        return _indicators_runs(kernels, open_, high, low, close)
    return _indicators_wide_pandas(open_, high, low, close)

def _ohlc_columns(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return tuple(frame[name].to_numpy(dtype=float) for name in ("Open", "High", "Low", "Close"))  # type: ignore[return-value]


def _add_indicators(frame: pd.DataFrame, run_id: int) -> pd.DataFrame:
    """One run's frame with the indicator columns added (same columns and values as always)."""
    frame = frame.copy()
    o, h, l, c = (column.reshape(-1, 1) for column in _ohlc_columns(frame))
    ind = _indicators_wide(o, h, l, c)
    frame["run"] = run_id
    frame["bar"] = np.arange(1, len(frame) + 1)
    for name in INDICATOR_COLUMNS:
        frame[name] = ind[name][:, 0]
    frame["direction"] = np.where(frame["Close"] >= frame["Open"], "Up", "Down")
    return frame


def _price_lists(frame: pd.DataFrame) -> tuple[list[float], list[float], list[float], list[float]]:
    return tuple(frame[column].astype(float).tolist() for column in ("Open", "High", "Low", "Close"))  # type: ignore[return-value]


def _trade_from_signal(frame: pd.DataFrame | None, signal_index: int, side: str, run_id: int,
                       prices: tuple[list[float], list[float], list[float], list[float]] | None = None,
                       dates: Any = None) -> Trade | None:
    # prices: the four columns as lists, built once per run (not once per signal); dates: the run's Date values
    opens, highs, lows, closes = prices if prices is not None else _price_lists(frame)
    dates = dates if dates is not None else frame["Date"]
    entry_index = signal_index + 1
    if entry_index >= len(opens):
        return None
    entry = float(opens[entry_index])
    stop = float(lows[signal_index] if side == "Long" else highs[signal_index])
    risk = entry - stop if side == "Long" else stop - entry
    if not np.isfinite(risk) or risk <= 0:
        return None
    target = entry + TARGET_R * risk if side == "Long" else entry - TARGET_R * risk
    exit_index = len(opens) - 1
    exit_price = float(closes[exit_index])
    reason = "time_exit"
    for index in range(entry_index, len(opens)):
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
    trade = {
        "run": run_id, "side": side, "signal_index": signal_index + 1,
        "signal_time": dates.iloc[signal_index] if hasattr(dates, "iloc") else dates[signal_index],
        "entry_index": entry_index + 1,
        "entry_time": dates.iloc[entry_index] if hasattr(dates, "iloc") else dates[entry_index],
        "entry_price": entry, "stop_price": stop, "target_price": target, "risk_per_unit": risk,
        "exit_index": exit_index + 1,
        "exit_time": dates.iloc[exit_index] if hasattr(dates, "iloc") else dates[exit_index],
        "exit_price": exit_price, "pnl_per_unit": pnl, "r_multiple": pnl / risk,
        "exit_reason": reason,
    }
    trade.update(slippage_fields(trade["r_multiple"], reason))
    return trade


# ─── Slippage decomposition (add-only columns) ────────────────────────────────
# A gap exit fills at the bar's open instead of the barrier. target_gap fills ABOVE the target
# (favourable, positive slippage); stop_gap fills BELOW the stop (unfavourable, negative slippage).
SLIPPAGE_TRADE_COLUMNS = ["expected_r", "slippage_r", "positive_slippage_r", "negative_slippage_r",
                          "slippage_flag", "time_exit_flag"]
SLIPPAGE_SUMMARY_COLUMNS = [
    "slippage_count", "positive_slippage_count", "negative_slippage_count",
    "slippage_share", "positive_slippage_share", "negative_slippage_share", "time_exit_share",
    "total_positive_slippage_r", "total_negative_slippage_r",
    "mean_positive_slippage_r", "mean_negative_slippage_r",
    "best_positive_slippage_r", "worst_negative_slippage_r",
]


def slippage_fields(r_multiple: float, reason: str) -> dict[str, float | int]:
    """Per-trade slippage in R. Barrier exits fill exactly at the barrier, so their slippage is 0 by definition
    (r_multiple differs from TARGET_R / -1 only by float rounding). time_exit has no barrier: expected_r is
    NaN and slippage_r is measured against 0, but it is NOT gap slippage, so the positive/negative parts are 0."""
    expected = TARGET_R if reason in ("target", "target_gap") else -1.0 if reason in ("stop", "stop_gap") else np.nan
    gap = reason in ("target_gap", "stop_gap")
    slip = r_multiple - expected if gap else 0.0 if reason in ("target", "stop") else r_multiple - 0.0
    return {"expected_r": expected, "slippage_r": slip,
            "positive_slippage_r": max(slip, 0.0) if reason == "target_gap" else 0.0,
            "negative_slippage_r": min(slip, 0.0) if reason == "stop_gap" else 0.0,
            "slippage_flag": int(gap), "time_exit_flag": int(reason == "time_exit")}


def slippage_summary(trades: list[Trade]) -> dict[str, float | int]:
    """Per-run slippage aggregates. Shares are NaN for a run without trades; means/extremes NaN without gaps."""
    n = len(trades)
    pos = [t["positive_slippage_r"] for t in trades if t["exit_reason"] == "target_gap"]
    neg = [t["negative_slippage_r"] for t in trades if t["exit_reason"] == "stop_gap"]
    n_time = sum(t["exit_reason"] == "time_exit" for t in trades)
    share = (lambda k: k / n) if n else (lambda k: np.nan)
    return {
        "slippage_count": len(pos) + len(neg), "positive_slippage_count": len(pos), "negative_slippage_count": len(neg),
        "slippage_share": share(len(pos) + len(neg)), "positive_slippage_share": share(len(pos)),
        "negative_slippage_share": share(len(neg)), "time_exit_share": share(n_time),
        "total_positive_slippage_r": float(sum(pos)), "total_negative_slippage_r": float(sum(neg)),
        "mean_positive_slippage_r": float(np.mean(pos)) if pos else np.nan,
        "mean_negative_slippage_r": float(np.mean(neg)) if neg else np.nan,
        "best_positive_slippage_r": float(max(pos)) if pos else np.nan,
        "worst_negative_slippage_r": float(min(neg)) if neg else np.nan,
    }


def with_slippage_columns(frame: pd.DataFrame, kind: str = "summary") -> pd.DataFrame:
    """Backward compatibility for CSVs written before the slippage columns existed: add each missing column as
    NaN with one warning per column. Never raises and never rewrites the file on disk."""
    import warnings
    columns = SLIPPAGE_SUMMARY_COLUMNS if kind == "summary" else SLIPPAGE_TRADE_COLUMNS
    frame = frame.copy()
    for column in columns:
        if column not in frame.columns:
            warnings.warn(f"legacy {kind} CSV has no '{column}' column; filled with NaN", stacklevel=2)
            frame[column] = np.nan
    return frame


def _schema_entry(name: str, series: pd.Series) -> dict[str, Any]:
    """One column of analysis_schema.json: dtype + discrete/continuous rule for the CSV viewer."""
    identifiers = {"run", "cycle"}
    timestamps = {"first_timestamp", "last_timestamp", "highest_high_time", "lowest_low_time", "sim_date",
                  "best_long_entry", "best_long_exit", "best_short_entry", "best_short_exit"}
    count_like = ("_count", "_flag", "bars", "_candles", "crossovers", "_trades", "observations", "gap_bars",
                  "is_backtest", "fast_window", "slow_window")
    if name in identifiers:
        kind, dtype = "identifier", "int"
    elif name in timestamps:
        kind, dtype = "timestamp", "datetime64"
    elif pd.api.types.is_bool_dtype(series):
        kind, dtype = "boolean", "bool"
    elif pd.api.types.is_integer_dtype(series) or (name.endswith(count_like) and pd.api.types.is_numeric_dtype(series)):
        kind, dtype = "count", "int"
    elif pd.api.types.is_numeric_dtype(series):
        kind, dtype = "measurement", "float"
    else:
        kind, dtype = "category", "string"
    unit = ("R" if name.endswith("_r") or "r_multiple" in name or name == "target_r" else
            "share" if name.endswith("_share") else "%" if "pct" in name else
            "count" if kind == "count" else "price" if name.endswith(("_points", "_price", "_close", "_open", "_high", "_low"))
            else "")
    numeric = pd.to_numeric(series, errors="coerce") if dtype in ("int", "float") else None
    entry = {"name": name, "dtype": dtype, "unit": unit,
             "is_discrete": kind in ("identifier", "count", "category", "boolean"),
             "is_continuous": kind == "measurement", "description": kind,
             "range": ([float(numeric.min()), float(numeric.max())] if numeric is not None and numeric.notna().any()
                       else None)}
    if kind == "identifier":
        entry["identifier"] = True
    return entry


def write_analysis_schema(summary: pd.DataFrame, path: Path) -> list[dict[str, Any]]:
    """analysis_schema.json next to analysis_summary.csv: one entry per column, in CSV order."""
    import json
    schema = [_schema_entry(str(name), summary[name]) for name in summary.columns]
    Path(path).write_text(json.dumps({"columns": schema}, indent=1, default=str), encoding="utf-8")
    return schema


def read_analysis_summary(path: Path) -> pd.DataFrame:
    """Read analysis_summary.csv with the dtypes from analysis_schema.json beside it (when present). A CSV cannot
    carry the type of an all-empty column (pandas reads it as float), so the schema is the source of truth."""
    import json
    path = Path(path)
    schema_path = path.with_name("analysis_schema.json")
    if not schema_path.exists():
        return with_slippage_columns(pd.read_csv(path), "summary")
    columns = json.loads(schema_path.read_text(encoding="utf-8"))["columns"]
    strings = {c["name"]: "string" for c in columns if c["dtype"] == "string"}
    frame = pd.read_csv(path, dtype=strings)
    for c in columns:
        column = frame.get(c["name"])
        if c["dtype"] == "int" and column is not None and column.isna().any():
            whole = pd.to_numeric(column, errors="coerce")
            if (whole.dropna() % 1 == 0).all():                       # nullable int instead of silent float;
                frame[c["name"]] = whole.astype("Int64")              # leave the column alone if it is not whole
    return with_slippage_columns(frame, "summary")


def _trades_from_arrays(difference: np.ndarray, prices: tuple[list[float], ...], dates: Any, run_id: int) -> list[Trade]:
    # The crossings are found with array operations (was a Python loop over every bar); only the few
    # signal bars go through the per-trade simulation, in the same order as before.
    difference = np.asarray(difference, dtype=float)
    previous, current = difference[:-1], difference[1:]
    with np.errstate(invalid="ignore"):
        finite = np.isfinite(previous) & np.isfinite(current)
        longs = finite & (previous <= 0) & (current > 0)
        shorts = finite & (previous >= 0) & (current < 0)
    trades = []
    for index in np.flatnonzero(longs | shorts) + 1:
        trade = _trade_from_signal(None, int(index), "Long" if longs[index - 1] else "Short", run_id, prices, dates)
        if trade:
            trades.append(trade)
    return trades


def _trades_for_run(frame: pd.DataFrame, run_id: int) -> list[Trade]:
    difference = (frame["fast_sma_5"] - frame["slow_sma_12"]).to_numpy(dtype=float)
    return _trades_from_arrays(difference, _price_lists(frame), frame["Date"], run_id)


def _finite(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _stat(values: Any, function) -> float:
    values = _finite(values)
    return float(function(values)) if len(values) else np.nan


def _sd(values: np.ndarray) -> float:
    """Sample standard deviation (ddof=1); NaN below two values (what pandas' .std() returns)."""
    n = len(values)
    if n <= 1:
        return np.nan
    # np.std(values, ddof=1) step by step (the same pairwise sums, divisions and square root, so the same
    # bits) without its wrapper, which was most of the cost of the ~6 calls per run.
    deviations = values - np.add.reduce(values) / n
    return float(math.sqrt(np.add.reduce(deviations * deviations) / (n - 1)))


def _best(trades: list[Trade], side: str) -> Trade | None:
    choices = [trade for trade in trades if trade["side"] == side]
    return max(choices, key=lambda trade: trade["r_multiple"], default=None)


VOLUME_METRICS = ("mean_volume", "volume_cv", "volume_abs_return_corr", "obv_last", "obv_slope_norm", "vwap_dev_pct")


def _volume_metrics_arrays(volume: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                           bar_return_pct: np.ndarray) -> dict[str, float]:
    if np.isnan(volume).any() or not (volume > 0).any():
        return {name: np.nan for name in VOLUME_METRICS}
    abs_return = np.abs(bar_return_pct)
    valid = ~np.isnan(abs_return)
    if valid.sum() > 3 and _sd(volume[valid]) > 0 and _sd(abs_return[valid]) > 0:
        corr = float(np.corrcoef(volume[valid], abs_return[valid])[0, 1])
    else:
        corr = np.nan
    change = np.diff(close, prepend=np.nan)
    obv = np.cumsum(np.sign(np.where(np.isnan(change), 0.0, change)) * volume)
    typical = (high + low + close) / 3.0
    vwap = float((typical * volume).sum() / volume.sum())
    mean_volume = volume.mean()
    return {
        "mean_volume": float(mean_volume),
        "volume_cv": float(_sd(volume) / mean_volume) if mean_volume > 0 else np.nan,
        "volume_abs_return_corr": corr,
        "obv_last": float(obv[-1]),
        "obv_slope_norm": float((obv[-1] - obv[0]) / volume.sum()),
        "vwap_dev_pct": float(100.0 * (close[-1] / vwap - 1.0)) if vwap > 0 else np.nan,
    }


def _volume_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Volume indicators (only when the CSV carries a Volume column from the volume simulation)."""
    if "Volume" not in frame.columns:
        return {}
    volume = pd.to_numeric(frame["Volume"], errors="coerce").to_numpy(dtype=float)
    return _volume_metrics_arrays(volume, frame["High"].to_numpy(dtype=float), frame["Low"].to_numpy(dtype=float),
                                  frame["Close"].to_numpy(dtype=float), frame["bar_return_pct"].to_numpy(dtype=float))


def _metrics_arrays(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, dates: Any, ind: dict[str, np.ndarray],
                    volume: np.ndarray | None, trades: list[Trade], run_id: int,
                    source_png: str) -> tuple[dict[str, Any], Trade | None, Trade | None]:
    """One run's summary row from plain arrays (same values as the pandas version, a fraction of the time).

    Without the optional `bottleneck` package pandas' mean / std / cumsum are numpy underneath, so the
    numbers are identical; the analysis self-check compares the output files byte for byte.
    """
    def at(i: int) -> Any:
        return dates.iloc[i] if hasattr(dates, "iloc") else dates[i]

    changes = _finite(ind["bar_change"])
    returns = ind["bar_return_pct"][~np.isnan(ind["bar_return_pct"])] / 100
    negative_returns = returns[returns < 0]
    peak = np.maximum.accumulate(c)
    drawdown_points = peak - c
    drawdown_pct = c / peak - 1 if (c > 0).all() else np.empty(0)
    downside_deviation = math.sqrt(float(np.mean(np.square(negative_returns))) if len(negative_returns) else np.nan)
    valid_return_sd = _sd(returns)
    mean_return = returns.mean() if len(returns) else np.nan
    sharpe = mean_return / valid_return_sd if np.isfinite(valid_return_sd) and valid_return_sd > 0 else np.nan
    sortino = mean_return / downside_deviation if np.isfinite(downside_deviation) and downside_deviation > 0 else np.nan
    price_range = h - l
    body = np.abs(c - o)
    gap = o - np.r_[np.nan, c[:-1]]
    abs_gap = np.abs(gap)
    difference = ind["fast_sma_5"] - ind["slow_sma_12"]
    previous_difference = np.r_[np.nan, difference[:-1]]
    up = c >= o
    best_long, best_short = _best(trades, "Long"), _best(trades, "Short")
    lowest_low, highest_high = l.min(), h.max()

    def last(name: str) -> float:
        # = _stat(ind[name][-1:], np.mean): the mean of one finite value is that value, else NaN
        value = float(ind[name][-1]) if len(ind[name]) else np.nan
        return value if math.isfinite(value) else np.nan

    result = {
        "run": run_id, "source_csv": f"data_{run_id}.csv", "source_png": source_png,
        "bars": len(c), "confidence_level": CONFIDENCE,
        "fast_window": FAST, "slow_window": SLOW, "target_r": TARGET_R,
        "first_timestamp": at(0), "last_timestamp": at(len(c) - 1),
        "first_open": float(o[0]), "last_close": float(c[-1]),
        "net_change_points": float(c[-1] - o[0]),
        "total_return_pct": 100 * (c[-1] / o[0] - 1) if o[0] > 0 else np.nan,
        "highest_high": float(highest_high), "highest_high_time": at(int(np.argmax(h))),
        "lowest_low": float(lowest_low), "lowest_low_time": at(int(np.argmin(l))),
        "high_low_range_points": float(highest_high - lowest_low),
        "high_low_range_pct_of_low": 100 * (highest_high - lowest_low) / lowest_low if lowest_low > 0 else np.nan,
        "mean_close": float(c.mean()), "median_close": float(np.median(c)),
        "mean_bar_change_points": _stat(changes, np.mean), "median_bar_change_points": _stat(changes, np.median),
        "sd_bar_change_points": _stat(changes, _sd),
        "mean_valid_bar_return_pct": 100 * _stat(returns, np.mean),
        "median_valid_bar_return_pct": 100 * _stat(returns, np.median),
        "realized_vol_per_bar_pct": 100 * _stat(returns, _sd),
        "valid_return_observations": int(len(returns)), "sharpe_per_bar_no_rf": sharpe,
        "sortino_per_bar_no_rf": sortino, "max_drawdown_points": float(drawdown_points.max()),
        "max_drawdown_pct_positive_prices_only": 100 * float(drawdown_pct.min()) if len(drawdown_pct) else np.nan,
        "bullish_candles": int(up.sum()), "bearish_candles": int((~up).sum()),
        "bullish_candle_pct": 100 * float(up.mean()),
        "bearish_candle_pct": 100 * float((~up).mean()),
        "avg_candle_body_points": float(body.mean()), "avg_candle_range_points": float(price_range.mean()),
        "median_candle_range_points": float(np.median(price_range)),
        "avg_true_range_points": float(np.nanmean(ind["true_range"])),
        "atr14_last": last("atr_14"), "rsi14_last": last("rsi_14"),
        "macd12_26_last": last("macd_12_26"),
        "macd_signal9_last": last("macd_signal_9"),
        "macd_histogram_last": last("macd_histogram"),
        "bollinger20_upper_last": last("bb_upper_20_2sd"),
        "bollinger20_lower_last": last("bb_lower_20_2sd"),
        "bollinger_percent_b_last": last("bb_percent_b"),
        "bollinger_width_pct_last": last("bb_width_pct"),
        "sma_bull_crossovers_5_12": int(((previous_difference <= 0) & (difference > 0)).sum()),
        "sma_bear_crossovers_5_12": int(((previous_difference >= 0) & (difference < 0)).sum()),
        "gap_bars": int((abs_gap > 1e-9).sum()), "avg_abs_gap_points": _stat(abs_gap, np.mean),
        "winning_crossover_trades": int(sum(trade["r_multiple"] > 0 for trade in trades)),
        "valid_crossover_trades": len(trades),
    }
    if volume is not None:
        result.update(_volume_metrics_arrays(volume, h, l, c, ind["bar_return_pct"]))
    result.update(slippage_summary(trades))
    for side, prefix, trade in (("Long", "best_long", best_long), ("Short", "best_short", best_short)):
        for suffix, key in (("entry", "entry_time"), ("exit", "exit_time"), ("entry_price", "entry_price"),
                            ("stop", "stop_price"), ("target_2R", "target_price"), ("exit_price", "exit_price"),
                            ("pnl_per_unit", "pnl_per_unit"), ("r_multiple", "r_multiple"), ("exit_reason", "exit_reason")):
            result[f"{prefix}_{suffix}"] = trade[key] if trade else ("no valid trade" if suffix == "exit_reason" else pd.NaT if suffix in ("entry", "exit") else np.nan)
    return result, best_long, best_short


def _metrics(frame: pd.DataFrame, trades: list[Trade], run_id: int, source_png: str) -> tuple[dict[str, Any], Trade | None, Trade | None]:
    o, h, l, c = _ohlc_columns(frame)
    ind = {name: frame[name].to_numpy(dtype=float) for name in INDICATOR_COLUMNS}
    volume = pd.to_numeric(frame["Volume"], errors="coerce").to_numpy(dtype=float) if "Volume" in frame.columns else None
    return _metrics_arrays(o, h, l, c, frame["Date"], ind, volume, trades, run_id, source_png)


def _time_ticks(dates: pd.Series, target: int = 10) -> tuple[np.ndarray, list[str]]:
    """Tick positions + labels whose format suits the span: clock time, weekday, month or year."""
    stamps = pd.to_datetime(dates).reset_index(drop=True)
    n = len(stamps)
    if n == 0:
        return np.array([], dtype=int), []
    days = stamps.dt.normalize()
    if days.nunique() <= 1:          # one session: label clock times of ~8 evenly spaced candles
        ticks = np.unique(np.linspace(0, n - 1, min(8, n)).round().astype(int))
        return ticks, [stamps.iloc[i].strftime("%H:%M") for i in ticks]
    span_days = (stamps.iloc[-1] - stamps.iloc[0]).total_seconds() / 86_400
    if span_days <= 120:
        fmt, key = "%a %d %b", days
    elif span_days <= 3 * 366:
        fmt, key = "%b %Y", stamps.dt.to_period("M")
    else:
        fmt, key = "%Y", stamps.dt.year
    ticks = np.flatnonzero(key.ne(key.shift()).to_numpy())     # first candle of each day / month / year
    if len(ticks) > target:
        ticks = ticks[:: math.ceil(len(ticks) / target)]
    return ticks, [stamps.iloc[i].strftime(fmt) for i in ticks]


def _draw_chart(frame: pd.DataFrame, run_id: int, best_long: Trade | None, best_short: Trade | None, output: Path, guide_path: Path, instrument: str | None = None, chart_label: str | None = None) -> None:
    import matplotlib  # lazy: only chart drawing needs matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    x = frame["bar"].astype(int).tolist()
    has_volume = "Volume" in frame.columns and pd.to_numeric(frame["Volume"], errors="coerce").notna().all()
    rows = 4 if has_volume else 3
    figure, axes = plt.subplots(rows, 1, figsize=(13, 9 if not has_volume else 10.5), sharex=True,
                                gridspec_kw={"height_ratios": [4.2, 1.2, 1.2] + ([1.0] if has_volume else [])},
                                facecolor="black")
    price_ax, rsi_ax, macd_ax = axes[0], axes[1], axes[2]
    bottom_ax = axes[-1]
    for axis in axes:
        axis.set_facecolor("black")
        axis.tick_params(colors="white")
        for spine in axis.spines.values():
            spine.set_color("#1e61bd")

    price_ax.yaxis.tick_right()
    price_ax.yaxis.set_label_position("right")
    price_ax.fill_between(x, frame.ci95_lower_12.to_numpy(dtype=float), frame.ci95_upper_12.to_numpy(dtype=float),
                          color="#7aa6d8", alpha=.22, label=f"{_pct_label(CONFIDENCE)} t interval: {SLOW}-close mean")
    price_ax.fill_between(x, frame.bb_lower_20_2sd.to_numpy(dtype=float), frame.bb_upper_20_2sd.to_numpy(dtype=float),
                          color="#aeb7c2", alpha=.13, label=f"Bollinger {BB_WINDOW} ± {BB_SD:g} SD")
    opens: list[float] = frame["Open"].astype(float).tolist()
    highs: list[float] = frame["High"].astype(float).tolist()
    lows: list[float] = frame["Low"].astype(float).tolist()
    closes: list[float] = frame["Close"].astype(float).tolist()
    # All wicks in one call and all bodies in one collection (was one line + one patch per candle):
    # the same pixels, with far fewer objects for matplotlib to lay out.
    o_, c_ = np.asarray(opens), np.asarray(closes)
    positions = np.arange(1, len(o_) + 1)
    price_ax.vlines(positions, lows, highs, color="#d7dde5", linewidth=.65, zorder=2)
    body_low = np.minimum(o_, c_)
    # The former 0.012 minimum made low-volatility candles dwarf the data and skew price-axis autoscaling.
    body_top = body_low + np.abs(c_ - o_)
    bodies = [((xpos - .30, low), (xpos + .30, low), (xpos + .30, top), (xpos - .30, top))
              for xpos, low, top in zip(positions, body_low, body_top)]
    price_ax.add_collection(PolyCollection(bodies, facecolors=np.where(c_ >= o_, "#21d6a2", "#ff6b77"),
                                           edgecolors="#d7dde5", linewidths=.3, zorder=3))
    price_ax.plot(x, frame.fast_sma_5, color="#66b6ff", linewidth=1.0, label=f"SMA {FAST}")
    price_ax.plot(x, frame.slow_sma_12, color="#ffc857", linewidth=1.1, label=f"SMA {SLOW}")
    price_ax.plot(x, frame.bb_mid_20, color="#ccd4df", linewidth=.7, linestyle="--", label=f"BB mid {BB_WINDOW}")
    for side, trade, color in (("Long", best_long, "#21d6a2"), ("Short", best_short, "#ff6b77")):
        if trade:
            price_ax.scatter(trade["entry_index"], trade["entry_price"], marker="^" if side == "Long" else "v",
                             color=color, s=48, zorder=5)
            price_ax.annotate(f"{side[0]} in {trade['r_multiple']:+.2f}R", (trade["entry_index"], trade["entry_price"]),
                              xytext=(4, 8 if side == "Long" else -14), textcoords="offset points", fontsize=7, color=color, weight="bold")
            price_ax.scatter(trade["exit_index"], trade["exit_price"], marker="x", color=color, s=38, zorder=5)
            price_ax.annotate(f"{side[0]} out", (trade["exit_index"], trade["exit_price"]),
                              xytext=(4, -12), textcoords="offset points", fontsize=7, color=color)
    price_ax.set_ylabel("Price", color="white")
    # The caller's label (Yahoo symbol + candle size, or a per-run REAL / SIM label) takes priority.
    title_instrument = chart_label or instrument or ""
    studies = f"SMA {FAST}/{SLOW} · Bollinger {BB_WINDOW} · {_pct_label(CONFIDENCE)} t-interval"
    price_ax.set_title(f"Run {run_id}:  {title_instrument}  —  {studies}" if title_instrument
                       else f"Run {run_id}:  {studies}", color="white")
    price_ax.legend(loc="upper left", fontsize=7, ncol=2, facecolor="#111111", edgecolor="#aab2bd",
                    labelcolor="white", framealpha=.9)
    price_ax.grid(True, color="#3a414b", linewidth=.4)

    rsi_ax.plot(x, frame.rsi_14, color="#c4a7ff", linewidth=.85)
    rsi_ax.axhline(RSI_HIGH, color="#ff8585", linestyle="--", linewidth=.6)
    rsi_ax.axhline(RSI_LOW, color="#58ddb4", linestyle="--", linewidth=.6)
    rsi_ax.set_ylim(0, 100)
    rsi_ax.set_ylabel(f"RSI {RSI_PERIOD}", color="white")
   #rsi_ax.set_xlabel("Study [4]: RSI guide in analysis_learning_resources.md", color="white")
    rsi_ax.grid(True, color="#3a414b", linewidth=.4)

    macd_ax.bar(x, frame.macd_histogram, color=np.where(frame.macd_histogram >= 0, "#16846b", "#d94c5c"), width=.65, alpha=.65)
    macd_ax.plot(x, frame.macd_12_26, color="#66b6ff", linewidth=.8, label=f"MACD {MACD_FAST}-{MACD_SLOW}")
    macd_ax.plot(x, frame.macd_signal_9, color="#ffc857", linewidth=.8, label=f"Signal {MACD_SIGNAL}")
    macd_ax.axhline(0, color="#777777", linewidth=.55)
    macd_ax.set_ylabel("MACD", color="white")
    #macd_ax.set_xlabel("Study [5]: MACD guide in analysis_learning_resources.md", color="white")
    macd_ax.legend(loc="upper left", fontsize=7, ncol=2, facecolor="#3D1A1A", edgecolor="#aab2bd",
                   labelcolor="white", framealpha=.9)
    macd_ax.grid(True, color="#3a414b", linewidth=.4)
    if has_volume:
        volume_ax = axes[3]
        volume_ax.bar(x, pd.to_numeric(frame["Volume"]).to_numpy(dtype=float),
                      color=np.where(frame["Close"] >= frame["Open"], "#16846b", "#d94c5c"), width=.65, alpha=.7)
        volume_ax.set_ylabel("Volume", color="white")
        volume_ax.grid(True, color="#3a414b", linewidth=.4)
    ticks, labels = _time_ticks(frame.Date)
    bottom_ax.set_xticks(ticks + 1, labels, rotation=25, ha="right")
    footer = "Best crossovers are hindsight only; no fees or slippage."
    if guide_path.exists():
        footer = (f"Study links [1]-[11]: {guide_path.name}. PNG labels are references; open the companion "
                  f"Markdown for clickable URLs. " + footer)
    figure.text(.01, .005, footer, fontsize=7, color="white")
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
    threads: int = 1,
) -> None:
    from PIL import Image, ImageDraw  # lazy
    box = (tile_width - 12, tile_height - 28)

    def tile(path: Path) -> Any:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail(box, Image.Resampling.LANCZOS)
            return image

    # Decoding + shrinking each PNG (~0.1 s) was the whole cost and ran on one core. Pillow releases the
    # GIL while it decodes and resamples, so threads make the tiles in parallel; pasting stays in order,
    # so the sheets are the same pixels. One page of tiles is held at a time.
    pool = ThreadPoolExecutor(threads) if threads > 1 and len(items) > 8 else None
    try:
        for page_number, start in enumerate(range(0, len(items), max_tiles), start=1):
            page_items = items[start:start + max_tiles]
            rows = math.ceil(len(page_items) / columns)
            sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
            draw = ImageDraw.Draw(sheet)
            tiles = pool.map(tile, [path for _, path in page_items]) if pool else map(tile, (p for _, p in page_items))
            for position, ((run_id, _), image) in enumerate(zip(page_items, tiles)):
                x, y = (position % columns) * tile_width, (position // columns) * tile_height
                sheet.paste(image, (x + (tile_width - image.width) // 2, y + 22))
                draw.text((x + 6, y + 4), f"Run {run_id}", fill="#111111")

            if len(items) > max_tiles:
                page_output = output.with_name(f"{output.stem}_page_{page_number:03}{output.suffix}")
            else:
                page_output = output
            sheet.save(page_output)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


def _volume_overall_row(summary: pd.DataFrame) -> tuple[str, Any, str]:
    if "mean_volume" in summary.columns and summary["mean_volume"].notna().any():
        return ("mean_volume_abs_return_corr", _stat(summary.volume_abs_return_corr, np.mean),
                "volume simulation: correlation of volume with |bar return|")
    return ("volume_indicators", "not available", "input CSVs contain no Volume column")


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
        _volume_overall_row(summary),
    ]
    return pd.DataFrame(values, columns=["measure", "value", "context"])


def realised_vol_feedback(
    summary: pd.DataFrame,
    vol_by_run: dict[int, float],
    expected_sigma_pct: Callable[[int, float], float],
    tolerance: float = 0.20,
) -> pd.DataFrame:
    """Compare realized_vol_per_bar_pct with the per-bar sigma the vol_index should have produced.

    expected_sigma_pct(run, vol_index) -> expected per-bar std in %. Runs diverging by more than
    `tolerance` (20 %) are flagged with a suggested vol_index = vol * expected / realised, which the
    caller re-generates when it is in auto-correct mode.
    """
    rows = []
    for run, vol in vol_by_run.items():
        hit = summary.loc[summary["run"] == run, "realized_vol_per_bar_pct"]
        if hit.empty or not np.isfinite(vol):
            continue
        realised = float(hit.iloc[0])
        expected = float(expected_sigma_pct(int(run), float(vol)))
        if not (np.isfinite(realised) and np.isfinite(expected) and expected > 0):
            continue
        divergence = realised / expected - 1.0
        rows.append({
            "run": int(run), "vol_index": float(vol), "expected_sigma_pct": expected,
            "realised_sigma_pct": realised, "divergence": divergence, "flagged": abs(divergence) > tolerance,
            "suggested_vol_index": float(vol * expected / realised) if realised > 0 else float(vol),
        })
    return pd.DataFrame(rows, columns=["run", "vol_index", "expected_sigma_pct", "realised_sigma_pct",
                                       "divergence", "flagged", "suggested_vol_index"])


def performance_settings() -> dict[str, int]:
    """config.toml [performance] (defaults if the file cannot be read)."""
    try:
        from project_config import section
        return dict(section("performance"))
    except Exception:
        return {"chart_workers": 0, "parallel_min_charts": 24}


def _draw_charts(jobs: list[tuple], performance: dict[str, int] | None, progress: Callable[[int, str], Any] | None) -> None:
    """Every analysis PNG; on several cores when there are enough of them (chart_jobs.py)."""
    import chart_jobs
    perf = performance or performance_settings()
    workers = chart_jobs.worker_count(len(jobs), perf["chart_workers"], perf["parallel_min_charts"])
    bar = progress(len(jobs), f"analysis charts ({workers} process{'es' if workers > 1 else ''})") if progress else None
    chart_jobs.run(chart_jobs.draw_analysis_chart, jobs, workers, bar.update if bar else None)


# ─── Reading many run CSVs fast ───────────────────────────────────────────────
# With 100,000 small run CSVs the analysis spent most of its time per FILE, not per bar: ~5 ms to open each
# file on Windows (antivirus scans every open), ~3 ms of read_csv set-up and ~2.5 ms guessing the date format.
# The fast path opens files on several threads, parses up to READ_BATCH files in one read_csv call and
# converts their dates in one to_datetime call. It is used only when that provably gives the same frames as
# reading file by file; anything unusual (other headers, quotes, blank lines, odd dates, a bad candle ...)
# goes through the original per-file loop, which also raises the original error messages.
READ_BATCH = 4096
READ_THREADS = 16
_ISO_DATE_FORMATS = {"%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"}
_EXACT_FLOAT = 2.0 ** 53          # above this an integer written in a CSV may not survive float64 the same way


class _Run:
    """One run's arrays (dates as a datetime array slice). `frame()` rebuilds the run's DataFrame only when it is needed (chart drawing)."""
    __slots__ = ("run_id", "ohlc", "dates", "volume", "_frame", "_source", "_rows")

    def __init__(self, run_id: int, ohlc: tuple[np.ndarray, ...], dates: list, volume: np.ndarray | None,
                 frame: pd.DataFrame | None = None, source: pd.DataFrame | None = None, rows: tuple[int, int] = (0, 0)):
        self.run_id, self.ohlc, self.dates, self.volume = run_id, ohlc, dates, volume
        self._frame, self._source, self._rows = frame, source, rows

    def frame(self) -> pd.DataFrame:
        if self._frame is not None:
            return self._frame
        start, stop = self._rows
        return self._source.iloc[start:stop].reset_index(drop=True)


def _read_bytes(paths: list[Path]) -> list[bytes]:
    def read(path: Path) -> bytes:
        with open(path, "rb") as handle:
            return handle.read()
    if len(paths) < 64:
        return [read(path) for path in paths]
    with ThreadPoolExecutor(READ_THREADS) as pool:     # opening is I/O-bound: threads overlap the waits
        return list(pool.map(read, paths))


def _parse_batch(blobs: list[bytes], expected: dict[str, Any]) -> tuple[pd.DataFrame, list[int]] | None:
    """One read_csv for many run files with the same header line, or None when that might differ from
    reading them one by one. expected carries the header and the date shape across batches."""
    header = expected.setdefault("header", blobs[0][:blobs[0].find(b"\n") + 1])
    if not header.endswith(b"\n") or b'"' in header:
        return None
    ending = b"\r\n" if header.endswith(b"\r\n") else b"\n"
    bodies, counts = [], []
    for blob in blobs:
        if not blob.startswith(header):
            return None
        body = blob[len(header):]
        # Each row must be one line: no quoted fields (they can hold line breaks), no blank lines, no bare \r.
        if (not body or b'"' in body or body.startswith((b"\n", b"\r")) or b"\n\n" in body or b"\n\r\n" in body
                or body.count(b"\r") != body.count(b"\r\n")):
            return None
        if not body.endswith(b"\n"):
            body += ending
        bodies.append(body)
        counts.append(body.count(b"\n"))
    try:
        frame = pd.read_csv(io.BytesIO(header + b"".join(bodies)))
    except Exception:
        return None
    if len(frame) != sum(counts):
        return None
    frame.columns = frame.columns.astype(str).str.strip()
    if not {"Date", "Open", "High", "Low", "Close"}.issubset(frame.columns) or frame.columns.duplicated().any():
        return None
    for name in frame.columns:
        if name == "Date":
            continue
        column = frame[name]
        # Only plain numbers: per-file int columns then concatenate to the same float64 / int64 values.
        if column.dtype not in (np.float64, np.int64):
            return None
        values = column.to_numpy(dtype=float)
        if (np.abs(values[np.isfinite(values)]) >= _EXACT_FLOAT).any():
            return None
    dates = frame["Date"]
    if not pd.api.types.is_string_dtype(dates) or dates.isna().any():
        return None
    # pandas guesses one date format per to_datetime call from the first value. Batching is the same as per
    # file when every file's first date has the same shape and that shape is an unambiguous ISO year-month-day.
    starts = np.cumsum([0] + counts[:-1])
    shapes = dates.iloc[starts].str.replace(r"\d", "0", regex=True).unique()
    if len(shapes) != 1 or expected.setdefault("shape", shapes[0]) != shapes[0]:
        return None
    if "format" not in expected:
        from pandas.tseries.api import guess_datetime_format
        expected["format"] = guess_datetime_format(str(dates.iloc[0]))
    if expected["format"] not in _ISO_DATE_FORMATS:
        return None
    try:
        frame["Date"] = pd.to_datetime(dates)
    except Exception:
        return None
    if frame["Date"].isna().any():
        return None
    return frame, counts


def _fast_read(paths: list[Path], run_ids: list[int]) -> tuple[list[_Run], pd.DataFrame] | None:
    """Every run as a _Run plus all rows stacked in run order (= processed_bars before the indicator
    columns), or None when the per-file loop must be used."""
    expected: dict[str, Any] = {}
    frames, counts = [], []
    for start in range(0, len(paths), READ_BATCH):
        parsed = _parse_batch(_read_bytes(paths[start:start + READ_BATCH]), expected)
        if parsed is None:
            return None
        frames.append(parsed[0])
        counts.extend(parsed[1])
    frame = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    bounds = np.cumsum([0] + counts)
    if min(counts) < SLOW:
        return None
    o, h, l, c = _ohlc_columns(frame)
    if not (np.isfinite(o).all() and np.isfinite(h).all() and np.isfinite(l).all() and np.isfinite(c).all()):
        return None
    if ((h < np.maximum(o, c)) | (l > np.minimum(o, c))).any():
        return None
    # A run whose dates go backwards is sorted exactly as the per-file loop does it.
    stamps = frame["Date"].to_numpy()
    backwards = np.flatnonzero(stamps[1:] < stamps[:-1]) + 1         # row earlier than the row before it
    backwards = backwards[~np.isin(backwards, bounds)]                 # ... ignoring the first row of each run
    unsorted = np.unique(np.searchsorted(bounds, backwards, side="right") - 1).tolist()
    if unsorted:
        order = np.arange(len(frame))
        for k in unsorted:
            piece = frame.iloc[bounds[k]:bounds[k + 1]].reset_index(drop=True)
            order[bounds[k]:bounds[k + 1]] = bounds[k] + piece.sort_values("Date").index.to_numpy()
        frame = frame.take(order).reset_index(drop=True)
        o, h, l, c = _ohlc_columns(frame)
    dates = frame["Date"].array          # sliced per run; Timestamps are made one run at a time (as before)
    volume = frame["Volume"].to_numpy(dtype=float) if "Volume" in frame.columns else None
    runs = [_Run(run_id, (o[a:b], h[a:b], l[a:b], c[a:b]), dates[a:b], None if volume is None else volume[a:b],
                 source=frame, rows=(a, b))
            for run_id, a, b in zip(run_ids, bounds[:-1].tolist(), bounds[1:].tolist())]
    return runs, frame


def _read_runs(source: Path, run_ids: list[int]) -> tuple[list[_Run], pd.DataFrame | None]:
    """Read and check every run CSV. The second value is all rows stacked when the fast path was used."""
    paths = [source / f"data_{run_id}.csv" for run_id in run_ids]
    fast = _fast_read(paths, run_ids)
    if fast is not None:
        return fast
    runs = []
    for run_id, csv_path in zip(run_ids, paths):
        frame = pd.read_csv(csv_path)
        frame.columns = frame.columns.astype(str).str.strip()
        required = {"Date", "Open", "High", "Low", "Close"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{csv_path.name} lacks required columns: {sorted(required-set(frame.columns))}")
        frame["Date"] = pd.to_datetime(frame["Date"])
        if not frame["Date"].is_monotonic_increasing:
            frame = frame.sort_values("Date").reset_index(drop=True)
        o, h, l, c = _ohlc_columns(frame)
        if not (np.isfinite(o).all() and np.isfinite(h).all() and np.isfinite(l).all() and np.isfinite(c).all()):
            raise ValueError(f"{csv_path.name} has non-finite OHLC values")
        if ((h < np.maximum(o, c)) | (l > np.minimum(o, c))).any():
            raise ValueError(f"{csv_path.name} has invalid OHLC candle ordering")
        if len(frame) < SLOW:
            raise ValueError(f"{csv_path.name} has fewer than {SLOW} rows")
        volume = pd.to_numeric(frame["Volume"], errors="coerce").to_numpy(dtype=float) if "Volume" in frame.columns else None
        runs.append(_Run(run_id, (o, h, l, c), frame["Date"].array, volume, frame=frame))
    return runs, None


# ─── Writing big CSVs on several cores ────────────────────────────────────────
PARALLEL_CSV_CELLS = 20_000_000    # below this (rows x columns, ~12,000 runs of bars) one process is as fast


def _write_csv(frame: pd.DataFrame, path: Path, workers: int = 1) -> None:
    """frame.to_csv(path, index=False), byte for byte, with big frames formatted in `workers` processes.

    pandas formats a CSV in chunks of 100,000 // columns rows and decides some formats per chunk (dates
    without a time of day), so every piece here is a whole number of those chunks: each piece then
    formats exactly as it would have inside one to_csv call.
    """
    cells = len(frame) * max(len(frame.columns), 1)
    if workers <= 1 or cells < PARALLEL_CSV_CELLS:
        frame.to_csv(path, index=False)
        return
    import chart_jobs
    chunk = max(1, 100_000 // (len(frame.columns) or 1))         # pandas.io.formats.csvs._DEFAULT_CHUNKSIZE_CELLS
    rows = chunk * max(1, -(-len(frame) // (workers * 4 * chunk)))
    jobs = [(frame.iloc[start:start + rows], start == 0) for start in range(0, len(frame), rows)]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        for text in chart_jobs.map_ordered(chart_jobs.csv_piece, jobs, workers):
            handle.write(text)


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
    vol_by_run: dict[int, float] | None = None,
    label_by_run: dict[int, str] | None = None,
    chart_runs: set[int] | None = None,
    performance: dict[str, int] | None = None,
    progress: Callable[[int, str], Any] | None = None,
) -> dict:
    """Analyze CSV runs and optionally label outputs with their simulation timeframe.

    Optional metadata (instrument, asset_type, sim_date, is_backtest) can be
    provided by the caller and will be inserted into the analysis summary if
    supplied. label_by_run overrides chart_label per run (e.g. "SAP REAL 3m");
    chart_runs limits the analysis PNGs and the source-PNG pairing check to those runs
    (None = every run, the original behaviour). Every CSV is always analysed.
    performance: the [performance] table for the analysis PNGs (None = read config.toml); progress:
    optional factory (total, label) -> object with update(done), e.g. console.Progress.
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
    charted = set(csv_ids) if chart_runs is None else set(chart_runs) & csv_ids
    if chart_runs is None:
        run_ids = sorted(csv_ids & png_ids) if render_source_charts else sorted(csv_ids)
    else:
        run_ids = sorted(csv_ids)
    if not run_ids:
        expected = "matching data_N.csv and image_N.png pairs" if render_source_charts else "data_N.csv files"
        raise ValueError(f"No {expected} in {source}")
    if render_source_charts and (charted - png_ids or png_ids - csv_ids):
        raise ValueError(f"Unmatched files: CSV-only={sorted(charted - png_ids)}, PNG-only={sorted(png_ids - csv_ids)}")

    # 1. Read and check every run (fast batched path when it is provably the same, else file by file).
    runs, stacked = _read_runs(source, run_ids)

    # 2. Indicators: every run of the same length in one batch (one column per run).
    indicators: list[dict[str, np.ndarray]] = [{} for _ in runs]
    by_length: dict[int, list[int]] = {}
    for position, run in enumerate(runs):
        by_length.setdefault(len(run.ohlc[3]), []).append(position)
    for positions in by_length.values():
        wide = _indicators_wide(*(np.column_stack([runs[p].ohlc[k] for p in positions]) for k in range(4)))
        per_run = {name: np.ascontiguousarray(values.T) for name, values in wide.items()}   # row j = run j
        for j, p in enumerate(positions):
            indicators[p] = {name: per_run[name][j] for name in INDICATOR_COLUMNS}

    # 3. Per run: crossover trades, the summary row, and (optionally) the analysis chart job.
    summaries, all_trades, analysis_images, chart_jobs_ = [], [], [], []
    for position, run in enumerate(runs):
        run_id, (o, h, l, c), dates, volume = run.run_id, run.ohlc, run.dates.tolist(), run.volume
        ind = indicators[position]
        trades = _trades_from_arrays(ind["fast_sma_5"] - ind["slow_sma_12"], (o.tolist(), h.tolist(), l.tolist(), c.tolist()),
                                     dates, run_id)
        all_trades.extend(trades)
        source_png = f"image_{run_id}.png" if run_id in png_ids else ""
        metrics, long_trade, short_trade = _metrics_arrays(o, h, l, c, dates, ind, volume, trades, run_id, source_png)
        summaries.append(metrics)
        if render_charts and run_id in charted:
            frame = run.frame()
            chart_frame = frame.copy()
            chart_frame["run"] = run_id
            chart_frame["bar"] = np.arange(1, len(frame) + 1)
            for name in INDICATOR_COLUMNS:
                chart_frame[name] = ind[name]
            chart_frame["direction"] = np.where(chart_frame["Close"] >= chart_frame["Open"], "Up", "Down")
            chart_path = output / f"image_{run_id}_analysis.png"
            label = (label_by_run or {}).get(run_id, chart_label)
            chart_jobs_.append((chart_frame, run_id, long_trade, short_trade, str(chart_path), str(guide_path),
                                instrument, label, SETTINGS))
            analysis_images.append((run_id, chart_path))
    if chart_jobs_:
        _draw_charts(chart_jobs_, performance, progress)

    summary = pd.DataFrame(summaries).sort_values("run")
    # processed_bars: every run stacked, same columns as one run's frame (CSV columns, run, bar, indicators, direction)
    lengths = [len(run.ohlc[3]) for run in runs]
    bars = stacked if stacked is not None else pd.concat([run.frame() for run in runs], ignore_index=True)
    bars["run"] = np.concatenate([np.full(n, run.run_id) for run, n in zip(runs, lengths)])
    bars["bar"] = np.concatenate([np.arange(1, n + 1) for n in lengths])
    for name in INDICATOR_COLUMNS:
        bars[name] = np.concatenate([ind[name] for ind in indicators])
    bars["direction"] = np.where(bars["Close"] >= bars["Open"], "Up", "Down")
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

    if vol_by_run:
        # The vol_index each run was generated with (feeds the feature x vol sensitivity map in ml_model).
        summary.insert(1, "vol_index", summary["run"].map(lambda r: vol_by_run.get(int(r), np.nan)))
    import chart_jobs
    perf = performance or performance_settings()
    csv_workers = chart_jobs.worker_count(10**9, perf["chart_workers"])
    _write_csv(summary, output / "analysis_summary.csv", csv_workers)
    _write_csv(bars, output / "processed_bars.csv", csv_workers)
    trade_columns = ["run", "side", "signal_index", "signal_time", "entry_index", "entry_time", "entry_price", "stop_price", "target_price", "risk_per_unit", "exit_index", "exit_time", "exit_price", "pnl_per_unit", "r_multiple", "exit_reason"] + SLIPPAGE_TRADE_COLUMNS
    write_analysis_schema(summary, output / "analysis_schema.json")
    _write_csv(pd.DataFrame(all_trades, columns=trade_columns), output / "crossover_trades.csv", csv_workers)
    overall = _overall_metrics(summary, bars, all_trades)
    if timeframe is not None:
        overall.insert(0, "timeframe", timeframe)
    overall.to_csv(output / "overall_metrics.csv", index=False)
    sheet_threads = max(1, min(perf["chart_workers"] or (os.cpu_count() or 2) - 1, os.cpu_count() or 1))
    if render_source_charts:
        sheet = [(run_id, source / f"image_{run_id}.png") for run_id in run_ids if run_id in charted]
        if sheet:
            _contact_sheet(sheet, output / "all_source_charts_contact_sheet.png", threads=sheet_threads)
    if render_charts and analysis_images:
        _contact_sheet(analysis_images, output / "all_analysis_charts_contact_sheet.png", threads=sheet_threads)
    return {"runs": len(run_ids), "bars": len(bars), "trades": len(all_trades), "output_dir": str(output)}


def main() -> None:
    model_dir = Path(__file__).resolve().parent
    analysis_dir = model_dir / "looped_analysis"
    result = analyze_looped_results(model_dir / "looped_results", analysis_dir)
    print(f"Analyzed {result['runs']} pairs / {result['bars']} bars / {result['trades']} trades -> {result['output_dir']}")


if __name__ == "__main__":
    main()