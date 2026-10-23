"""
ml_model.py — Random Forest classifier on price_modeling.py analysis output.

Reads   : <analysis_dir>/analysis_summary.csv
           (produced by market_analysis.py OR analyze_looped_results.R — same schema)
Writes  : <analysis_dir>/ml_results/<asset_type>_<instrument>/
               analysis_history.csv      accumulated input data when history is enabled
               ohlc_candle_index.csv         indexed simulation OHLC candle rows
               historical_stock_candle_index.csv indexed yfinance OHLC candle rows
               run_stock_matches.csv         closest historical stock for every run
               stock_similarity_ranking.csv  ranked stocks across the simulation runs
               stock_similarity_summary.txt overall best run and stock match
               model.pkl                 trained sklearn Pipeline (imputer → scaler → RF)
               predictions.csv           run, split, y_true, y_pred, prob_bullish, correct
               feature_importances.csv   feature, importance, rank
               feature_importances.png   horizontal bar chart (black background)
               summary.txt               accuracy, OOB, confusion matrix, top features
               accuracy_metrics.csv     numeric accuracy values tagged by asset

Entry points
    run_ml_pipeline(analysis_dir, instrument=..., asset_type=...)
    python ml_model.py [--analysis-dir PATH] [--instrument SYMBOL] [--asset-type TYPE]
                       [--keep-history | --clear-history]

Null-hypothesis expectation
    Prices are drawn from uniform noise → no indicator predicts return direction.
    Test accuracy should converge to ≈ 0.50 as n_runs → ∞.
    Feature importances reflect spurious correlation in random data — useful as a
    baseline when comparing against real market data later.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import json
import re
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ─── Feature configuration ────────────────────────────────────────────────────
#
# 30 predictors. Feature engineering is performed in _add_engineered_features()
# immediately after analysis_summary.csv is loaded.
#
# The model does NOT force feature values to sum to 1. Random Forest
# feature_importances_ are normalised by sklearn and sum to 1.
#
FEATURES: list[str] = [
    # 01–07: Volatility / range
    "realized_vol_per_bar_pct",
    "sd_bar_change_points",
    "high_low_range_points",
    "avg_candle_body_points",
    "avg_candle_range_points",
    "avg_true_range_points",
    "atr14_last",

    # 08–11: Core technical indicators from the analysis graph
    "rsi14_last",
    "macd_histogram_last",
    "bollinger_width_pct_last",
    "bollinger_percent_b_last",

    # 12–17: Directional path / crossover behaviour
    "bullish_candle_pct",
    "sma_bull_crossovers_5_12",
    "sma_bear_crossovers_5_12",
    "valid_crossover_trades",
    "winning_crossover_trades",
    "max_drawdown_pct_positive_prices_only",

    # 18–30: Engineered features derived from existing analysis columns
    "rsi_distance_from_50",
    "rsi_overbought_flag",
    "macd_histogram_abs",
    "macd_histogram_sign",
    "bollinger_width_squared",
    "bollinger_position_distance",
    "atr_to_avg_range",
    "volatility_to_range_ratio",
    "candle_direction_balance",
    "crossover_direction_balance",
    "crossover_win_rate",
    "drawdown_to_volatility",
    "candle_body_to_range_ratio",

    # 31-32: volume features (only present when the CSVs carry a Volume column; skipped otherwise)
    "volume_cv",
    "volume_abs_return_corr",
]

# Columns excluded to prevent data leakage.
# All of these encode mean return direction directly or are transforms of it.
LEAKAGE_COLS: list[str] = [
    "net_change_points",           # last_close − first_open  (numerator of total_return_pct)
    "mean_valid_bar_return_pct",   # mean(bar_return) ≈ total_return / n_bars
    "median_valid_bar_return_pct", # correlated with direction
    "mean_bar_change_points",      # mean(bar_change) = net_change / n_bars
    "median_bar_change_points",    # correlated with direction
    "sharpe_per_bar_no_rf",        # mean(return) / std(return) — numerator is mean return
    "sortino_per_bar_no_rf",       # mean(return) / downside_dev — same numerator
    "bullish_candles",             # raw count — 100 % linearly related to bullish_candle_pct
    "bearish_candles",             # 45 − bullish_candles
    "best_long_r_multiple",        # best Long trade R — inflated in bullish runs
    "best_short_r_multiple",       # best Short trade R — inflated in bearish runs
    "obv_last",                    # on-balance volume sums signed volume: encodes direction
    "obv_slope_norm",              # same, normalised
    "vwap_dev_pct",                # last close vs VWAP: correlated with the move
]

TARGET_COL  = "total_return_pct"   # y = (target > 0) → 1=bullish, 0=bearish
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
# test fraction = 1 − TRAIN_FRAC − VAL_FRAC = 0.15

# NaN threshold: drop a feature if > NAN_DROP_PCT of training rows are NaN
NAN_DROP_PCT = 0.80
HISTORY_FILENAME = "analysis_history.csv"
HISTORY_SOURCES_FILENAME = "analysis_history_sources.json"
BENCHMARK_SYMBOL_LIMIT = 250
BENCHMARK_CACHE_HOURS = 12
BENCHMARK_UNIVERSE_CACHE_DAYS = 7
INTRADAY_HISTORY_DAYS = 59
OHLC_COLUMNS = ["Open", "High", "Low", "Close"]
YFINANCE_INTERVALS = {
    "1d": "5m",   # Resample Yahoo 5-minute bars to the simulation candle count.
    "5d": "30m",
    "1m": "60m",
    "3m": "1d",
    "ytd": "1d",
    "1y": "1wk",
    "5y": "1wk",
}
FALLBACK_BENCHMARK_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "BRK-B",
    "AVGO", "TSLA", "JPM", "WMT", "LLY", "V", "MA", "XOM", "ORCL",
    "COST", "NFLX", "JNJ", "HD", "PG", "ABBV", "BAC", "KO", "CRM",
    "CVX", "CSCO", "AMD", "PEP",
]


def _benchmark_symbols(cache_dir: Path) -> tuple[list[str], str]:
    """Get a refreshed list of large US stocks from Yahoo's yfinance screener."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    symbols_path = cache_dir / "us_large_cap_symbols.json"
    cached_symbols: list[str] = []
    try:
        payload = json.loads(symbols_path.read_text(encoding="utf-8"))
        cached_symbols = [str(symbol) for symbol in payload.get("symbols", [])]
        age_seconds = pd.Timestamp.now().timestamp() - symbols_path.stat().st_mtime
        if cached_symbols and age_seconds < BENCHMARK_UNIVERSE_CACHE_DAYS * 86_400:
            return cached_symbols, "Yahoo Finance US large-cap screener (cached)"
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        pass

    try:
        import yfinance as yf
        from yfinance import EquityQuery

        query = EquityQuery("and", [
            EquityQuery("eq", ["region", "us"]),
            EquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
            EquityQuery("gte", ["intradaymarketcap", 5_000_000_000]),
        ])
        response = yf.screen(
            query,
            size=BENCHMARK_SYMBOL_LIMIT,
            sortField="intradaymarketcap",
            sortAsc=False,
        )
        quotes = response.get("quotes", []) if isinstance(response, dict) else []
        symbols = list(dict.fromkeys(
            str(quote["symbol"]).strip()
            for quote in quotes
            if isinstance(quote, dict) and quote.get("symbol")
        ))[:BENCHMARK_SYMBOL_LIMIT]
        if symbols:
            symbols_path.write_text(
                json.dumps({"symbols": symbols}, indent=2) + "\n",
                encoding="utf-8",
            )
            return symbols, "Yahoo Finance US large-cap screener"
    except Exception as exc:
        print(f"[ML] Yahoo stock screener unavailable: {exc}")

    if cached_symbols:
        return cached_symbols, "stale Yahoo Finance US large-cap screener cache"
    return FALLBACK_BENCHMARK_SYMBOLS.copy(), "built-in US large-cap fallback list"


def _fresh_file(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return False
    age_seconds = pd.Timestamp.now().timestamp() - path.stat().st_mtime
    return 0 <= age_seconds < max_age_hours * 3_600


def _download_stock_history(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
    cache_dir: Path,
) -> pd.DataFrame:
    """Batch-download adjusted OHLC histories with a short local cache."""
    import yfinance as yf

    cache_dir.mkdir(parents=True, exist_ok=True)
    symbol_hash = hashlib.sha256("\n".join(sorted(symbols)).encode()).hexdigest()[:12]
    cache_path = cache_dir / (
        f"ohlc_{interval}_{start:%Y%m%d}_{end:%Y%m%d}_{symbol_hash}.csv"
    )
    if _fresh_file(cache_path, BENCHMARK_CACHE_HOURS):
        cached = pd.read_csv(cache_path, parse_dates=["Date"])
        if not cached.empty:
            return cached

    records: list[pd.DataFrame] = []
    chunk_size = 40
    for offset in range(0, len(symbols), chunk_size):
        batch = symbols[offset:offset + chunk_size]
        try:
            downloaded = yf.download(
                tickers=batch,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,
                actions=False,
                group_by="ticker",
                threads=True,
                progress=False,
                ignore_tz=True,
                timeout=30,
            )
        except Exception as exc:
            print(f"[ML] Yahoo OHLC download failed for a ticker batch: {exc}")
            continue
        if downloaded is None or downloaded.empty:
            continue

        for symbol in batch:
            if isinstance(downloaded.columns, pd.MultiIndex):
                first_level = downloaded.columns.get_level_values(0).astype(str)
                second_level = downloaded.columns.get_level_values(1).astype(str)
                if symbol in first_level:
                    history = downloaded[symbol].copy()
                elif symbol in second_level:
                    history = downloaded.xs(symbol, axis=1, level=1).copy()
                else:
                    continue
            elif len(batch) == 1:
                history = downloaded.copy()
            else:
                continue

            renamed = {
                column: str(column[0] if isinstance(column, tuple) else column).title()
                for column in history.columns
            }
            history = history.rename(columns=renamed)
            if not set(OHLC_COLUMNS).issubset(history.columns):
                continue
            history = history[OHLC_COLUMNS].copy()
            dates = pd.to_datetime(history.index, errors="coerce")
            if getattr(dates, "tz", None) is not None:
                dates = dates.tz_convert("America/New_York").tz_localize(None)
            history.index = dates
            history.index.name = "Date"
            history = history.replace([np.inf, -np.inf], np.nan).dropna()
            if history.empty:
                continue
            one_symbol = history.reset_index()
            one_symbol["ticker"] = symbol
            records.append(one_symbol[["ticker", "Date", *OHLC_COLUMNS]])

    if not records:
        if cache_path.exists():
            try:
                stale_cache = pd.read_csv(cache_path, parse_dates=["Date"])
                if not stale_cache.empty:
                    print("[ML] Yahoo download failed; using the older local OHLC cache.")
                    return stale_cache
            except (OSError, pd.errors.ParserError, ValueError):
                pass
        raise ValueError("Yahoo Finance returned no usable OHLC histories.")

    history = pd.concat(records, ignore_index=True)
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    history = history.dropna(subset=["Date", *OHLC_COLUMNS])
    history.to_csv(cache_path, index=False)
    return history


def _compress_ohlc(frame: pd.DataFrame, target_bars: int) -> np.ndarray:
    """Aggregate contiguous OHLC candles into target_bars chronological bins."""
    values = (
        frame[OHLC_COLUMNS].to_numpy(dtype=float)
        if isinstance(frame, pd.DataFrame)
        else np.asarray(frame, dtype=float)
    )
    if target_bars < 1 or len(values) < target_bars:
        raise ValueError("Cannot aggregate OHLC history to the requested candle count.")
    if len(values) == target_bars:
        return values

    compressed = []
    for indices in np.array_split(np.arange(len(values)), target_bars):
        candles = values[indices]
        compressed.append([
            candles[0, 0],
            np.nanmax(candles[:, 1]),
            np.nanmin(candles[:, 2]),
            candles[-1, 3],
        ])
    return np.asarray(compressed, dtype=float)


def _normalised_ohlc_path(values: np.ndarray) -> np.ndarray:
    """Express each OHLC price as percent change from the first open."""
    base = float(values[0, 0])
    scale = abs(base)
    if not np.isfinite(scale) or scale < 1e-12:
        scale = float(np.nanmax(np.abs(values)))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return (values - base) * (100.0 / scale)


def _write_stock_similarity(
    analysis_dir: Path,
    ml_dir: Path,
    current_summary: pd.DataFrame,
    timeframe: str,
) -> None:
    """Index run candles and rank their closest historical US stock paths."""
    match_path = ml_dir / "run_stock_matches.csv"
    ranking_path = ml_dir / "stock_similarity_ranking.csv"
    summary_path = ml_dir / "stock_similarity_summary.txt"
    index_path = ml_dir / "ohlc_candle_index.csv"
    historical_index_path = ml_dir / "historical_stock_candle_index.csv"
    status_path = ml_dir / "stock_similarity_status.txt"
    cache_dir = analysis_dir / ".yfinance_cache"

    def save_status(message: str) -> None:
        status_path.write_text(message.strip() + "\n", encoding="utf-8")
        summary_path.write_text(
            "Historical stock path comparison\n"
            "Metric: normalized Open/High/Low/Close path RMSE; lower is closer.\n"
            "This is path similarity, not forecast accuracy.\n"
            f"Status: {message.strip()}\n",
            encoding="utf-8",
        )
        print(f"[ML] Historical stock comparison: {message}")

    try:
        if "run" not in current_summary.columns:
            raise ValueError("analysis_summary.csv has no run column.")
        bars_path = analysis_dir / "processed_bars.csv"
        if not bars_path.exists():
            raise FileNotFoundError("processed_bars.csv is required for candle-level matching.")
        bars = pd.read_csv(bars_path)
        required = {"run", "Date", *OHLC_COLUMNS}
        missing = sorted(required.difference(bars.columns))
        if missing:
            raise ValueError(f"processed_bars.csv lacks columns: {missing}")

        bars["run"] = pd.to_numeric(bars["run"], errors="coerce")
        bars["Date"] = pd.to_datetime(bars["Date"], errors="coerce")
        for column in OHLC_COLUMNS:
            bars[column] = pd.to_numeric(bars[column], errors="coerce")
        if "bar" not in bars.columns:
            bars = bars.sort_values(["run", "Date"])
            bars["bar"] = bars.groupby("run").cumcount() + 1
        else:
            bars["bar"] = pd.to_numeric(bars["bar"], errors="coerce")

        run_ids = pd.to_numeric(current_summary["run"], errors="coerce").dropna()
        run_ids = set(run_ids.astype(int).tolist())
        bars = bars.dropna(subset=["run", "Date", "bar", *OHLC_COLUMNS])
        bars["run"] = bars["run"].astype(int)
        bars["bar"] = bars["bar"].astype(int)
        bars = bars[bars["run"].isin(run_ids)].copy()
        if "timeframe" in bars.columns and timeframe != "unknown":
            bars = bars[bars["timeframe"].astype(str).str.lower() == timeframe.lower()]
        if bars.empty:
            raise ValueError("No candle rows matched the selected runs and timeframe.")
        bars = bars.sort_values(["run", "bar", "Date"]).reset_index(drop=True)

        candle_index = bars[["run", "bar", "Date", *OHLC_COLUMNS]].rename(
            columns={"bar": "candle_index", "Date": "timestamp"}
        )
        instrument_label = (
            str(current_summary["instrument"].dropna().iloc[0])
            if "instrument" in current_summary.columns
            and current_summary["instrument"].notna().any()
            else "unknown"
        )
        asset_label = (
            str(current_summary["asset_type"].dropna().iloc[0])
            if "asset_type" in current_summary.columns
            and current_summary["asset_type"].notna().any()
            else "unknown"
        )
        candle_index.insert(1, "instrument", instrument_label)
        candle_index.insert(2, "asset_type", asset_label)
        candle_index.insert(3, "timeframe", timeframe)
        first_open = candle_index.groupby("run")["Open"].transform("first")
        scale = first_open.abs().replace(0, np.nan)
        for column in OHLC_COLUMNS:
            candle_index[f"{column.lower()}_from_start_pct"] = (
                100.0 * (candle_index[column] - first_open) / scale
            )
        candle_index.to_csv(index_path, index=False)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        save_status(f"could not build the OHLC candle index: {exc}")
        return

    interval = YFINANCE_INTERVALS.get(timeframe.lower())
    if interval is None:
        save_status(f"timeframe '{timeframe}' has no yfinance interval mapping.")
        return

    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    run_ranges: dict[int, tuple[pd.Timestamp, pd.Timestamp, int]] = {}
    run_paths: dict[int, np.ndarray] = {}
    period_runs: dict[tuple[pd.Timestamp, pd.Timestamp, int], list[int]] = {}
    skipped: dict[int, str] = {}
    intraday = interval in {"5m", "30m", "60m"}
    intraday_cutoff = today - pd.Timedelta(days=INTRADAY_HISTORY_DAYS)
    for run_value, raw_frame in bars.groupby("run", sort=False):
        run = int(run_value)
        frame = raw_frame.sort_values(["bar", "Date"]).reset_index(drop=True)
        start = frame["Date"].min().normalize()
        end = frame["Date"].max().normalize()
        run_ranges[run] = (start, end, len(frame))
        if end >= today:
            skipped[run] = "period_not_fully_historical"
        elif intraday and start < intraday_cutoff:
            skipped[run] = "outside_yfinance_intraday_history_window"
        else:
            run_paths[run] = frame[OHLC_COLUMNS].to_numpy(dtype=float)
            period_runs.setdefault((start, end, len(frame)), []).append(run)

    match_rows: list[dict] = []
    for run, reason in skipped.items():
        start, end, candle_count = run_ranges[run]
        match_rows.append({
            "run": run, "closest_stock": "", "ohlc_rmse_pct": np.nan,
            "compared_candles": 0, "simulated_candles": candle_count,
            "coverage_pct": 0.0, "period_start": start, "period_end": end,
            "benchmark_stocks_checked": 0, "status": reason,
        })

    if not period_runs:
        pd.DataFrame(match_rows).to_csv(match_path, index=False)
        pd.DataFrame(columns=["ticker", "mean_ohlc_rmse_pct", "comparisons", "rank"]).to_csv(
            ranking_path, index=False
        )
        save_status(
            "all runs are outside available historical coverage; intraday yfinance data "
            "is limited to recent dates and future candles cannot be matched."
        )
        return

    try:
        symbols, universe_source = _benchmark_symbols(cache_dir)
        start = min(period[0] for period in period_runs)
        last_date = max(period[1] for period in period_runs)
        end = last_date + pd.Timedelta(days=8 if interval == "1wk" else 1)
        historical = _download_stock_history(
            symbols, start, end, interval, cache_dir
        )
    except Exception as exc:
        for run in run_paths:
            run_start, run_end, candle_count = run_ranges[run]
            match_rows.append({
                "run": run, "closest_stock": "", "ohlc_rmse_pct": np.nan,
                "compared_candles": 0, "simulated_candles": candle_count,
                "coverage_pct": 0.0, "period_start": run_start,
                "period_end": run_end, "benchmark_stocks_checked": 0,
                "status": f"yfinance_unavailable: {exc}",
            })
        pd.DataFrame(match_rows).sort_values("run").to_csv(match_path, index=False)
        pd.DataFrame(columns=["ticker", "mean_ohlc_rmse_pct", "comparisons", "rank"]).to_csv(
            ranking_path, index=False
        )
        save_status(f"Yahoo Finance history could not be loaded: {exc}")
        return

    historical["Date"] = pd.to_datetime(historical["Date"], errors="coerce")
    historical = historical.dropna(subset=["Date", *OHLC_COLUMNS])
    downloaded_symbol_count = int(historical["ticker"].nunique())
    historical_index = historical.sort_values(["ticker", "Date"]).reset_index(drop=True)
    historical_index.insert(1, "timeframe", timeframe)
    historical_index.insert(
        2,
        "candle_index",
        historical_index.groupby("ticker").cumcount() + 1,
    )
    historical_index = historical_index[
        ["ticker", "timeframe", "candle_index", "Date", *OHLC_COLUMNS]
    ]
    historical_index.to_csv(historical_index_path, index=False)
    historical_by_ticker = {
        str(ticker): group.sort_values("Date").reset_index(drop=True)
        for ticker, group in historical.groupby("ticker", sort=False)
    }
    ticker_stats: dict[str, dict[str, float]] = {}
    for (run_start, run_end, sim_count), period_run_ids in period_runs.items():
        history_end = run_end
        candidates: dict[str, pd.DataFrame] = {}
        minimum_history = int(np.ceil(sim_count * 0.70))
        for ticker, history in historical_by_ticker.items():
            dates = history["Date"].dt.normalize()
            if interval == "1wk":
                # Yahoo weekly bars can be labelled by the week start/end date;
                # compare their Monday week anchor with the simulator's Monday bars.
                dates = dates - pd.to_timedelta(dates.dt.weekday, unit="D")
            within_period = history[(dates >= run_start) & (dates <= history_end)]
            if len(within_period) >= minimum_history:
                candidates[ticker] = within_period.reset_index(drop=True)

        if not candidates:
            for run in period_run_ids:
                match_rows.append({
                    "run": run, "closest_stock": "", "ohlc_rmse_pct": np.nan,
                    "compared_candles": 0, "simulated_candles": sim_count,
                    "coverage_pct": 0.0, "period_start": run_start,
                    "period_end": run_end, "benchmark_stocks_checked": 0,
                    "status": "no_stock_history_for_period",
                })
            continue

        common_bars = min(sim_count, min(len(frame) for frame in candidates.values()))
        if common_bars < 5:
            for run in period_run_ids:
                match_rows.append({
                    "run": run, "closest_stock": "", "ohlc_rmse_pct": np.nan,
                    "compared_candles": common_bars, "simulated_candles": sim_count,
                    "coverage_pct": 100.0 * common_bars / max(sim_count, 1),
                    "period_start": run_start, "period_end": run_end,
                    "benchmark_stocks_checked": 0, "status": "too_few_common_candles",
                })
            continue

        candidate_tickers = list(candidates)
        reference_paths = []
        for ticker in candidate_tickers:
            history = candidates[ticker]
            stock_values = _compress_ohlc(history, common_bars)
            reference_paths.append(
                _normalised_ohlc_path(stock_values).reshape(-1)
            )
        reference_matrix = np.vstack(reference_paths)
        reference_mean_square = np.mean(np.square(reference_matrix), axis=1)
        feature_count = reference_matrix.shape[1]

        # Runs sharing the same historical window can reuse reference candles.
        # Bound the pairwise score matrix so large Monte Carlo batches stay manageable.
        batch_size = 512
        for offset in range(0, len(period_run_ids), batch_size):
            batch_runs = period_run_ids[offset:offset + batch_size]
            simulation_matrix = np.vstack([
                _normalised_ohlc_path(
                    _compress_ohlc(run_paths[run], common_bars)
                ).reshape(-1)
                for run in batch_runs
            ])
            simulation_mean_square = np.mean(np.square(simulation_matrix), axis=1)
            cross_mean = (simulation_matrix @ reference_matrix.T) / feature_count
            rmse_matrix = np.sqrt(np.maximum(
                simulation_mean_square[:, None]
                + reference_mean_square[None, :]
                - 2.0 * cross_mean,
                0.0,
            ))
            for stock_index, ticker in enumerate(candidate_tickers):
                stats = ticker_stats.setdefault(
                    ticker, {"rmse_sum": 0.0, "comparisons": 0.0}
                )
                stats["rmse_sum"] += float(rmse_matrix[:, stock_index].sum())
                stats["comparisons"] += len(batch_runs)

            for row_index, run in enumerate(batch_runs):
                closest_index = int(np.argmin(rmse_matrix[row_index]))
                closest_stock = candidate_tickers[closest_index]
                difference = simulation_matrix[row_index] - reference_matrix[closest_index]
                match_rows.append({
                    "run": run, "closest_stock": closest_stock,
                    "ohlc_rmse_pct": float(rmse_matrix[row_index, closest_index]),
                    "ohlc_mae_pct": float(np.mean(np.abs(difference))),
                    "compared_candles": common_bars, "simulated_candles": sim_count,
                    "coverage_pct": 100.0 * common_bars / max(sim_count, 1),
                    "period_start": run_start, "period_end": run_end,
                    "benchmark_stocks_checked": len(candidate_tickers),
                    "universe_source": universe_source, "status": "matched",
                })

    matches = pd.DataFrame(match_rows)
    if not matches.empty:
        matches["run_rank"] = np.nan
        matched = matches["status"] == "matched"
        matches.loc[matched, "run_rank"] = (
            matches.loc[matched, "ohlc_rmse_pct"].rank(method="min").astype(int)
        )
        matches = matches.sort_values(
            ["status", "ohlc_rmse_pct", "run"], ascending=[True, True, True]
        )
    matches.to_csv(match_path, index=False)

    ranking_rows = []
    for ticker, stats in ticker_stats.items():
        comparisons = int(stats["comparisons"])
        if comparisons:
            ranking_rows.append({
                "ticker": ticker,
                "mean_ohlc_rmse_pct": stats["rmse_sum"] / comparisons,
                "comparisons": comparisons,
            })
    ranking = pd.DataFrame(ranking_rows)
    if not ranking.empty:
        ranking = ranking.sort_values("mean_ohlc_rmse_pct").reset_index(drop=True)
        ranking["rank"] = np.arange(1, len(ranking) + 1)
    else:
        ranking = pd.DataFrame(columns=[
            "ticker", "mean_ohlc_rmse_pct", "comparisons", "rank"
        ])
    ranking.to_csv(ranking_path, index=False)

    matched_runs = matches[matches["status"] == "matched"] if not matches.empty else matches
    report_lines = [
        "Historical stock path comparison",
        f"Simulated instrument: {asset_label} {instrument_label} ({timeframe}).",
        f"Universe: {universe_source}; up to {BENCHMARK_SYMBOL_LIMIT} US large-cap stocks.",
        f"Historical candles available: {downloaded_symbol_count} of {len(symbols)} screened stocks.",
        "Historical source: Yahoo Finance via yfinance; OHLC auto-adjusted for corporate actions.",
        "Metric: RMSE across candle Open/High/Low/Close, normalized as percent change from each path's first Open.",
        "Lower RMSE is a closer price-path match; it is a similarity score, not forecast accuracy.",
        f"Candle index: {index_path}",
        f"Historical stock candle index: {historical_index_path}",
        f"Per-run matches: {match_path}",
        f"Stock ranking: {ranking_path}",
    ]
    if not matched_runs.empty:
        best_run = matched_runs.sort_values("ohlc_rmse_pct").iloc[0]
        report_lines.extend([
            f"Closest run: {int(best_run['run'])}",
            f"Closest stock for that run: {best_run['closest_stock']}",
            f"OHLC RMSE: {best_run['ohlc_rmse_pct']:.6f}% across {int(best_run['compared_candles'])} candles.",
        ])
    if not ranking.empty:
        best_stock = ranking.iloc[0]
        report_lines.append(
            f"Closest stock on average across runs: {best_stock['ticker']} "
            f"(mean RMSE {best_stock['mean_ohlc_rmse_pct']:.6f}%)."
        )
    summary_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    status_path.write_text(
        f"Matched {len(matched_runs)} of {len(run_ranges)} runs against "
        f"{downloaded_symbol_count} stocks with yfinance history.\n",
        encoding="utf-8",
    )
    print(f"[ML] Candle index saved -> {index_path}")
    print(f"[ML] Stock matches saved -> {match_path}")
    if not matched_runs.empty:
        best_run = matched_runs.sort_values("ohlc_rmse_pct").iloc[0]
        print(
            f"[ML] Closest historical match: run {int(best_run['run'])} vs "
            f"{best_run['closest_stock']} (OHLC RMSE {best_run['ohlc_rmse_pct']:.6f}%)."
        )


_LAST_BATCH_DUPLICATE = False


def _prepare_training_data(
    current_df: pd.DataFrame,
    summary_csv: Path,
    ml_dir: Path,
    keep_history: bool,
) -> pd.DataFrame:
    """Append the current analysis batch to a persistent history, or reset it."""
    history_path = ml_dir / HISTORY_FILENAME
    sources_path = ml_dir / HISTORY_SOURCES_FILENAME

    if not keep_history:
        history_path.unlink(missing_ok=True)
        sources_path.unlink(missing_ok=True)
        print("[ML] History cleared; training on this run only.")
        return current_df

    source_hash = hashlib.sha256(summary_csv.read_bytes()).hexdigest()
    try:
        processed_sources = json.loads(sources_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        processed_sources = []
    if not isinstance(processed_sources, list):
        processed_sources = []

    global _LAST_BATCH_DUPLICATE
    _LAST_BATCH_DUPLICATE = False
    if source_hash in processed_sources and history_path.exists():
        print("[ML] This analysis file is already in history; skipping duplicate rows.")
        _LAST_BATCH_DUPLICATE = True
        return pd.read_csv(history_path)

    batch = current_df.copy()
    if "run" in batch.columns:
        batch["source_run"] = batch["run"]

    if history_path.exists():
        history = pd.read_csv(history_path)
        next_run = (
            int(pd.to_numeric(history["run"], errors="coerce").max()) + 1
            if "run" in history.columns and history["run"].notna().any()
            else 1
        )
    else:
        history = pd.DataFrame()
        next_run = 1

    batch["run"] = np.arange(next_run, next_run + len(batch))
    combined = pd.concat([history, batch], ignore_index=True, sort=False)
    combined.to_csv(history_path, index=False)

    processed_sources.append(source_hash)
    sources_path.write_text(
        json.dumps(processed_sources, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[ML] Saved {len(batch)} new rows; history now has {len(combined)} rows.")
    return combined


def _resolve_asset_value(
    df: pd.DataFrame, column: str, override: str | None, default: str
) -> str:
    """Get one instrument label from an override or the analysis CSV."""
    if override and override.strip():
        return override.strip()

    if column in df.columns:
        values = df[column].dropna().astype(str).str.strip()
        values = values[values != ""].unique().tolist()
        if len(values) > 1:
            raise ValueError(
                f"Analysis summary contains multiple {column} values: {values}. "
                "Run the ML pipeline separately for each instrument."
            )
        if values:
            return values[0]
    return default


def _asset_output_name(asset_type: str, instrument: str) -> str:
    """Build a safe per-instrument results folder name."""
    name = re.sub(
        r"[^A-Za-z0-9._-]+", "_", f"{asset_type}_{instrument}"
    ).strip("._-").lower()
    return name or "unspecified_instrument"


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the 13 additional predictors used by the 30-feature model.

    These are deterministic transforms of existing analysis columns. They do
    not use total_return_pct or any other leakage column.
    """
    df = df.copy()

    def require(cols: list[str]) -> bool:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"[ML] Engineered features unavailable; missing: {missing}")
            return False
        return True

    # RSI: distance from neutral and an overbought-state indicator.
    if require(["rsi14_last"]):
        df["rsi_distance_from_50"] = df["rsi14_last"] - 50.0
        import market_analysis as _ma          # configured from config.toml [indicators]
        df["rsi_overbought_flag"] = (df["rsi14_last"] >= _ma.RSI_HIGH).astype(float)
    else:
        df["rsi_distance_from_50"] = np.nan
        df["rsi_overbought_flag"] = np.nan

    # MACD: separate magnitude from sign so the RF can split on either property.
    if require(["macd_histogram_last"]):
        df["macd_histogram_abs"] = df["macd_histogram_last"].abs()
        df["macd_histogram_sign"] = np.sign(df["macd_histogram_last"])
    else:
        df["macd_histogram_abs"] = np.nan
        df["macd_histogram_sign"] = np.nan

    # Bollinger Bands: non-linearity and distance from the middle of the band.
    if require(["bollinger_width_pct_last"]):
        df["bollinger_width_squared"] = df["bollinger_width_pct_last"] ** 2
    else:
        df["bollinger_width_squared"] = np.nan

    if require(["bollinger_percent_b_last"]):
        df["bollinger_position_distance"] = (
            df["bollinger_percent_b_last"] - 0.5
        ).abs()
    else:
        df["bollinger_position_distance"] = np.nan

    # Volatility ratios: scale ATR/volatility against the typical candle range.
    if require(["atr14_last", "avg_candle_range_points"]):
        denom = df["avg_candle_range_points"].replace(0, np.nan)
        df["atr_to_avg_range"] = df["atr14_last"] / denom
    else:
        df["atr_to_avg_range"] = np.nan

    if require(["realized_vol_per_bar_pct", "avg_candle_range_points"]):
        denom = df["avg_candle_range_points"].replace(0, np.nan)
        df["volatility_to_range_ratio"] = (
            df["realized_vol_per_bar_pct"] / denom
        )
    else:
        df["volatility_to_range_ratio"] = np.nan

    # Directional balance: bullish percentage centred on zero.
    if require(["bullish_candle_pct"]):
        df["candle_direction_balance"] = (
            df["bullish_candle_pct"] - (100.0 - df["bullish_candle_pct"])
        ) / 100.0
    else:
        df["candle_direction_balance"] = np.nan

    # Crossover balance: positive = more bullish crosses, negative = more bearish.
    if require(["sma_bull_crossovers_5_12", "sma_bear_crossovers_5_12"]):
        total_crosses = (
            df["sma_bull_crossovers_5_12"]
            + df["sma_bear_crossovers_5_12"]
        )
        df["crossover_direction_balance"] = (
            df["sma_bull_crossovers_5_12"]
            - df["sma_bear_crossovers_5_12"]
        ) / total_crosses.replace(0, np.nan)
    else:
        df["crossover_direction_balance"] = np.nan

    # Signal quality: proportion of crossover trades that won.
    if require(["winning_crossover_trades", "valid_crossover_trades"]):
        denom = df["valid_crossover_trades"].replace(0, np.nan)
        df["crossover_win_rate"] = (
            df["winning_crossover_trades"] / denom
        )
    else:
        df["crossover_win_rate"] = np.nan

    # Drawdown normalised by realised volatility.
    if require(["max_drawdown_pct_positive_prices_only", "realized_vol_per_bar_pct"]):
        denom = df["realized_vol_per_bar_pct"].abs().replace(0, np.nan)
        df["drawdown_to_volatility"] = (
            df["max_drawdown_pct_positive_prices_only"].abs() / denom
        )
    else:
        df["drawdown_to_volatility"] = np.nan

    # Candle conviction: body size relative to the average full candle range.
    if require(["avg_candle_body_points", "avg_candle_range_points"]):
        denom = df["avg_candle_range_points"].replace(0, np.nan)
        df["candle_body_to_range_ratio"] = (
            df["avg_candle_body_points"] / denom
        )
    else:
        df["candle_body_to_range_ratio"] = np.nan

    return df


# ─── Public entry point ───────────────────────────────────────────────────────

def run_ml_pipeline(
    analysis_dir: str | Path,
    keep_history: bool | None = None,
    instrument: str | None = None,
    asset_type: str | None = None,
    timeframe: str | None = None,
    label_mode: str | None = None,
    tune_rf: bool | None = None,
    compare: bool = True,
    online: bool | None = None,
    validation: dict | None = None,
    config: dict | None = None,
) -> None:
    """
    Full pipeline: load → prune → split → preprocess → train → evaluate → extras → save.

    Extras (config.toml [ml], rf_tuning.toml): purged-CV model comparison, optional RF tuning,
    accuracy-weighted samples, direction-match / ensemble labels, online forest update, conformal
    vol bounds, adversarial validation, SHAP / tree contributions, vol-sensitivity heatmap, HTML report.
    `validation` (from the backtest) may hold: real_frames, sim_frames (rebased OHLC DataFrames) and
    vol_points (DataFrame with vol_index, composite_F).

    Parameters
    ----------
    analysis_dir : path to the folder containing analysis_summary.csv.
                   Accepts either the Python output dir (looped_analysis/) or
                   the R output dir (looped_results_analysis_300/) — same CSV schema.
    keep_history : append each new analysis summary to the selected asset's history CSV.
                   False clears that archive and uses only the current summary.
    instrument : instrument or pair label, e.g. GBP/USD, ES=F, or AAPL.
    asset_type : market category, e.g. forex, futures, or stock.
    timeframe  : simulation timeframe key (1d / 5d / 1m / 3m / 1y / 5y / ytd).
                 When set, only rows with a matching 'timeframe' column value
                 are used for training. Prevents mixing indicator scales across
                 timeframes (daily bars vs. 10-min bars have very different ATR).

    The pipeline also compares each current run's indexed OHLC path with
    historical OHLC from up to 250 large-cap US stocks downloaded by yfinance.
    This reports path similarity; it does not measure future prediction accuracy.
    """
    analysis_dir = Path(analysis_dir)
    summary_csv  = analysis_dir / "analysis_summary.csv"

    if not summary_csv.exists():
        raise FileNotFoundError(
            f"analysis_summary.csv not found in {analysis_dir}.\n"
            "Run market_analysis.py or analyze_looped_results.R first."
        )

    df = pd.read_csv(summary_csv)
    df_all_timeframes = (
        df["timeframe"].unique().tolist() if "timeframe" in df.columns else []
    )
    instrument = _resolve_asset_value(
        df, "instrument", instrument, "Unspecified instrument"
    )
    asset_type = _resolve_asset_value(df, "asset_type", asset_type, "other").lower()
    df["instrument"] = instrument
    df["asset_type"] = asset_type

    timeframe = _resolve_asset_value(df, "timeframe", timeframe, "unknown").lower()

    # Filter to the requested timeframe when the column is present.
    # This matters when history is accumulated across multiple timeframe runs.
    if "timeframe" in df.columns and timeframe != "unknown":
        before = len(df)
        df = df[df["timeframe"].astype(str).str.lower() == timeframe].copy()
        if len(df) == 0:
            raise ValueError(
                f"No rows with timeframe='{timeframe}' in {summary_csv.name}.\n"
                f"Available timeframes: {df_all_timeframes}"
            )
        if len(df) < before:
            print(f"[ML] Filtered to timeframe='{timeframe}': {len(df)} / {before} rows kept.")

    # Each instrument+timeframe combination gets its own subfolder so models
    # trained on different bar scales never overwrite each other.
    ml_dir = analysis_dir / "ml_results" / _asset_output_name(asset_type, instrument) / timeframe
    ml_dir.mkdir(parents=True, exist_ok=True)

    # Compare the current candle batch before history accumulation so each run
    # is matched only to the OHLC rows produced by this invocation.
    _write_stock_similarity(analysis_dir, ml_dir, df.copy(), timeframe)

    if keep_history is None:
        choice = input(
            "[ML] Keep analysis data across Python runs? "
            "(y = accumulate, n = clear history) [n]: "
        ).strip().lower()
        keep_history = choice in {"y", "yes"}

    # ── Load and prepare data ─────────────────────────────────────────────────
    cfg = config if config is not None else _config()
    ml_cfg = cfg["ml"]
    train_frac = float(ml_cfg.get("train_frac", TRAIN_FRAC))      # config.toml [ml]
    val_frac = float(ml_cfg.get("val_frac", VAL_FRAC))
    nan_drop_pct = float(ml_cfg.get("nan_drop_pct", NAN_DROP_PCT))
    label_mode = label_mode or ml_cfg["label_mode"]
    tune_rf = ml_cfg["tune_rf"] if tune_rf is None else tune_rf
    online = ml_cfg["online_update"] if online is None else online
    n_batch = len(df)
    if label_mode == "forecast":
        # Leak-free setup: features from the early bars, label from the late bars. History accumulation
        # needs the bars of every earlier batch, so this mode always trains on the current batch.
        if keep_history:
            print("[ML] forecast mode trains on the current batch only (history accumulation is off).")
        keep_history = False
        df = build_forecast_dataset(analysis_dir, df, float(ml_cfg.get("forecast_split", 0.70)))
        n_batch = len(df)
    df = _prepare_training_data(df, summary_csv, ml_dir, keep_history)

    # Build the additional 13 features from columns already produced by the
    # analysis pipeline. This keeps ml_model.py self-contained and means the
    # model can be run against existing analysis_summary.csv files.
    df = _add_engineered_features(df)

    # Outlier run detection: drop runs far (% difference) from the batch median (config.toml [pruning]).
    n_before_prune = len(df)
    df, pruned = prune_outlier_runs(df, cfg["pruning"])
    pruned.to_csv(ml_dir / "pruned_runs.csv", index=False)
    if len(df) < n_before_prune:
        print(f"[ML] Pruned {n_before_prune - len(df)} outlier run(s) (> {cfg['pruning']['pct_difference']}% "
              f"from the median) -> pruned_runs.csv")
    df = df.reset_index(drop=True)

    n_runs = len(df)
    asset_label = f"{asset_type} {instrument}"
    print(f"\n[ML][{asset_label}] Loaded {n_runs} rows from {summary_csv.name}")

    if n_runs < 10:
        raise ValueError(
            f"Need at least 10 runs for a train/val/test split. Got {n_runs}.\n"
            "Re-run price_modeling.py with a larger run count."
        )
    if n_runs < 50:
        print(
            f"[ML] Warning: only {n_runs} runs. "
            "Results are not statistically meaningful until n_runs >= 200."
        )

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {summary_csv.name}.")

    # ── Feature validation ────────────────────────────────────────────────────
    available_features = _select_features(df, FEATURES, train_frac, nan_drop_pct)
    if not available_features:
        raise ValueError("No usable features remain after NaN filtering.")

    if label_mode != "return" and "net_price_error_pct" in df.columns and df["net_price_error_pct"].notna().any():
        # Only safe when the label is not the simulated return itself (the error depends on the sim close).
        available_features = available_features + ["net_price_error_pct"]
        print("[ML] Added feature net_price_error_pct (Rewind runs; safe because the label is not the return).")

    X = df[available_features]
    y, label_text = make_target(df, label_mode)
    weights = sample_weights(df, ml_cfg["use_sample_weight"])
    audit = leakage_audit(X, y)
    audit.to_csv(ml_dir / "leakage_audit.csv", index=False)
    leaky = audit[audit["flag"] == 1] if len(audit) else audit
    if len(leaky):
        print(f"[ML] LEAKAGE WARNING ({label_mode} label): {len(leaky)} feature(s) separate the label on their own "
              f"(single-feature AUC >= 0.80): "
              + ", ".join(f"{r.feature} ({r.separation:.2f})" for r in leaky.head(5).itertuples())
              + ".\n     Features and label are measured on the SAME finished path, so accuracy above 0.50 is mechanical, "
              "not forecasting.\n     Use label_mode = \"forecast\" (config.toml [ml] or --label forecast): features from the "
              "first 70% of the bars, label from the last 30%.")
    print(f"[ML][{asset_label}] Label: {label_text}; sample weights {describe_weights(df, weights)}")

    # ── Online learning: add trees from the NEW batch to the saved forest (no retrain) ────────
    if online and keep_history and _LAST_BATCH_DUPLICATE:
        print("[ML] Online update: this batch is already in the history, so there is nothing new to learn from.\n")
        return
    if online and keep_history and (ml_dir / "model.pkl").exists() and (ml_dir / "model_meta.json").exists():
        k = min(n_batch, len(df))
        new_X, new_y, new_w = X.iloc[-k:], y[-k:], weights[-k:]
        saved = joblib.load(ml_dir / "model.pkl")
        meta_saved = json.loads((ml_dir / "model_meta.json").read_text(encoding="utf-8"))
        preq = float(accuracy_score(new_y, saved.predict(new_X[meta_saved["features"]]))) \
            if all(f in new_X.columns for f in meta_saved["features"]) else float("nan")
        info = online_update_forest(ml_dir, X.iloc[-k:], new_y, new_w, ml_cfg["online_add_trees"])
        if info:
            note = (f"Online update: prequential accuracy of the existing forest on the {k} new rows = {preq:.4f} "
                    f"(measured BEFORE learning from them); trees {info['trees_before']} -> {info['trees_after']}.")
            (ml_dir / "online_update.txt").write_text(note + "\n", encoding="utf-8")
            print(f"[ML] {note}")
            print("[ML] Done (online mode: standard train/val/test reports are only written by a full retrain).\n")
            return

    # ── Block split ───────────────────────────────────────────────────────────
    # Block (not random) because a single rand.seed() before price_modeling.py's
    # run loop means consecutive runs share one RNG chain; shuffling would allow
    # RNG-correlated information to leak between train and test sets.
    train_end = int(n_runs * train_frac)
    val_end   = int(n_runs * (train_frac + val_frac))

    X_train, y_train = X.iloc[:train_end].copy(),       y[:train_end]
    X_val,   y_val   = X.iloc[train_end:val_end].copy(), y[train_end:val_end]
    X_test,  y_test  = X.iloc[val_end:].copy(),          y[val_end:]
    w_train = weights[:train_end]

    run_ids = df["run"].values if "run" in df.columns else np.arange(1, n_runs + 1)

    print(
        f"[ML][{asset_label}] Split (block):  train={len(X_train)}  |  "
        f"val={len(X_val)}  |  test={len(X_test)}"
    )
    print(
        f"[ML][{asset_label}] Class balance - bullish: {int(y.sum())} / {n_runs} "
        f"({100 * y.mean():.1f}%)  (expected about 50% on random data)"
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    # min_samples_leaf scales with n_train so the tree can still grow at small N.
    # At n_train=14 (20-run demo): min_samples_leaf=2
    # At n_train=3500 (5000-run real): min_samples_leaf=35
    n_train       = len(X_train)
    min_leaf      = max(2, n_train // 100)
    rf_params     = dict(
        n_estimators    = int(ml_cfg.get("n_estimators", 500)),
        max_features    = "sqrt",       # sqrt(n_features) per split — Breiman's recommendation
        min_samples_leaf= min_leaf,
        class_weight    = "balanced",   # guard against class imbalance at small N
        random_state    = 42,
        n_jobs          = -1,           # use all cores
        oob_score       = True,         # free val estimate from bootstrap
    )

    tuning_table = None
    if tune_rf and len(np.unique(y_train)) == 2:
        try:
            print(f"[ML][{asset_label}] Tuning RandomForest (rf_tuning.toml, purged CV)...")
            tuned, tuning_table = tune_random_forest(X_train, y_train, w_train, ml_cfg["tuning_config"], ml_dir, rf_params)
            rf_params = {**tuned, "oob_score": True, "random_state": 42, "n_jobs": -1}
            print(f"[ML][{asset_label}] Tuned: " + ", ".join(f"{k}={v}" for k, v in tuned.items()
                                                              if k not in ("n_jobs", "random_state", "oob_score")))
        except Exception as exc:
            print(f"[ML] RF tuning failed ({type(exc).__name__}: {exc}); using the default forest.")
    elif tune_rf:
        print("[ML] RF tuning skipped: the training block contains one class only.")

    # The untrained Pipeline — nothing learned yet until .fit() is called below.
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # handles NaN (e.g. futures mode drawdown)
        ("scale",  StandardScaler()),                    # zero-mean, unit-variance per feature
        ("rf",     RandomForestClassifier(**rf_params)), # ← the untrained model
    ])

    # ── Train ─────────────────────────────────────────────────────────────────
    print(
        f"[ML][{asset_label}] Training RandomForestClassifier "
        f"(n_estimators={rf_params['n_estimators']}, min_samples_leaf={rf_params['min_samples_leaf']})..."
    )
    pipeline.fit(X_train, y_train, rf__sample_weight=w_train)

    rf: RandomForestClassifier = pipeline.named_steps["rf"]
    print(f"[ML][{asset_label}] OOB accuracy (train-set estimate): {rf.oob_score_:.4f}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    acc_tr = accuracy_score(y_train, pipeline.predict(X_train))
    acc_va = accuracy_score(y_val,   pipeline.predict(X_val))
    acc_te = accuracy_score(y_test,  pipeline.predict(X_test))

    print(f"[ML][{asset_label}] Train accuracy : {acc_tr:.4f}")
    print(f"[ML][{asset_label}] Val   accuracy : {acc_va:.4f}")
    print(f"[ML][{asset_label}] Test  accuracy : {acc_te:.4f}")
    print(
        "[ML] Note: expected test accuracy about 0.50 on synthetic random data.\n"
        f"     Compare with real historical data for {asset_label} to assess market signal."
    )

    # ── Save model ────────────────────────────────────────────────────────────
    accuracy_path = ml_dir / "accuracy_metrics.csv"
    pd.DataFrame([{
        "asset_type": asset_type,
        "instrument": instrument,
        "timeframe": timeframe,
        "oob_accuracy": rf.oob_score_,
        "train_accuracy": acc_tr,
        "validation_accuracy": acc_va,
        "test_accuracy": acc_te,
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_rows": len(X_test),
        "label_mode": label_mode,
        "test_p_value_vs_chance": binomial_p_value(int((pipeline.predict(X_test) == y_test).sum()), len(y_test)),
        "min_detectable_accuracy": minimum_detectable_accuracy(len(y_test)),
        "tuned": int(tuning_table is not None),
    }]).to_csv(accuracy_path, index=False)
    print(f"[ML][{asset_label}] Accuracy metrics saved -> {accuracy_path}")

    model_path = ml_dir / "model.pkl"
    joblib.dump(pipeline, model_path)
    (ml_dir / "model_meta.json").write_text(json.dumps({
        "features": available_features, "label_mode": label_mode, "label": label_text,
        "trees": len(rf.estimators_), "trained_rows": len(X_train), "saved": str(pd.Timestamp.now()),
    }, indent=2), encoding="utf-8")
    print(f"[ML] Model saved          -> {model_path}")

    # ── Save predictions ──────────────────────────────────────────────────────
    all_X = pd.concat([X_train, X_val, X_test])
    all_y = np.concatenate([y_train, y_val, y_test])
    all_run_ids = np.concatenate([
        run_ids[:train_end],
        run_ids[train_end:val_end],
        run_ids[val_end:],
    ])
    all_preds = pipeline.predict(all_X)

    # A very small training split can contain only one class. In that case
    # predict_proba has one column, so select the bullish class by label and
    # use 0/1 probabilities if that class was absent during training.
    rf_classes = pipeline.named_steps["rf"].classes_
    bullish_class = np.flatnonzero(rf_classes == 1)
    if len(bullish_class):
        all_probs = pipeline.predict_proba(all_X)[:, bullish_class[0]]
    else:
        all_probs = np.zeros(len(all_X), dtype=float)

    splits = (
        ["train"] * len(X_train)
        + ["val"]   * len(X_val)
        + ["test"]  * len(X_test)
    )

    # Use directional indicators only, converted to a shared -1 to +1 scale.
    # Volatility, range, and drawdown features are excluded because their
    # positive values do not consistently mean bullish.
    bullish_score_parts: dict[str, pd.Series] = {}

    if "rsi14_last" in all_X.columns:
        bullish_score_parts["rsi"] = (
            (all_X["rsi14_last"] - 50.0) / 50.0
        ).clip(-1.0, 1.0)

    if "macd_histogram_sign" in all_X.columns:
        bullish_score_parts["macd"] = all_X["macd_histogram_sign"].clip(
            -1.0, 1.0
        )
    elif "macd_histogram_last" in all_X.columns:
        bullish_score_parts["macd"] = np.sign(all_X["macd_histogram_last"])

    if "bollinger_percent_b_last" in all_X.columns:
        bullish_score_parts["bollinger"] = (
            (all_X["bollinger_percent_b_last"] - 0.5) * 2.0
        ).clip(-1.0, 1.0)

    if "candle_direction_balance" in all_X.columns:
        bullish_score_parts["candles"] = all_X[
            "candle_direction_balance"
        ].clip(-1.0, 1.0)
    elif "bullish_candle_pct" in all_X.columns:
        bullish_score_parts["candles"] = (
            (all_X["bullish_candle_pct"] - 50.0) / 50.0
        ).clip(-1.0, 1.0)

    if "crossover_direction_balance" in all_X.columns:
        bullish_score_parts["crossovers"] = all_X[
            "crossover_direction_balance"
        ].clip(-1.0, 1.0)

    if not bullish_score_parts:
        raise ValueError(
            "No directional bullish indicators are available to calculate "
            "the bullish indicator average."
        )

    bullish_scores = pd.DataFrame(bullish_score_parts, index=all_X.index)
    bullish_indicator_score = bullish_scores.mean(axis=1)
    bullish_indicator_running_avg = bullish_indicator_score.expanding(
        min_periods=1
    ).mean()

    pred_df = pd.DataFrame({
        "run": all_run_ids,
        "asset_type": asset_type,
        "instrument": instrument,
        "timeframe": timeframe,
        "split": splits,
        "y_true": all_y,
        "y_pred": all_preds,
        "prob_bullish": all_probs.round(4),
        "correct": (all_y == all_preds).astype(int),
        "bullish_indicator_score": bullish_indicator_score.to_numpy(),
        "bullish_indicator_running_avg": (
            bullish_indicator_running_avg.to_numpy()
        ),
    })

    pred_path = ml_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"[ML] Predictions saved    -> {pred_path}")

    # ── Save feature importances ──────────────────────────────────────────────
    imp_df = (
        pd.DataFrame({
            "feature"   : available_features,
            "importance": rf.feature_importances_.round(6),
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp_df["rank"] = imp_df["importance"].rank(ascending=False).astype(int)

    imp_path = ml_dir / "feature_importances.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"[ML] Importances saved    -> {imp_path}")

    # ── Save summary text ─────────────────────────────────────────────────────
    cm_te = confusion_matrix(
        y_test, pipeline.predict(X_test), labels=[0, 1]
    )
    summary_path = ml_dir / "summary.txt"
    _write_summary(
        summary_path, analysis_dir, asset_type, instrument, timeframe, n_runs, available_features,
        train_end, val_end, X_train, X_val, X_test,
        rf, acc_tr, acc_va, acc_te, cm_te, imp_df,
    )
    print(f"[ML] Summary saved        -> {summary_path}")

    # ── Save importance chart ─────────────────────────────────────────────────
    chart_path = ml_dir / "feature_importances.png"
    _plot_importances(imp_df, chart_path)
    print(f"[ML] Importance chart     -> {chart_path}")

    # ── Extras: each is reported but cannot stop the core pipeline ──────────────────────────
    def optional(name: str, function):
        try:
            return function()
        except Exception as exc:
            print(f"[ML] {name} failed ({type(exc).__name__}: {exc})")
            return None

    validation = validation or {}
    if compare:
        table = optional("Model comparison", lambda: compare_models(
            X_train, y_train, w_train, X_val, y_val, X_test, y_test, rf_params, ml_cfg, ml_dir / "model_comparison.csv"))
        if table is not None and len(table):
            print(f"[ML] Model comparison (purged {ml_cfg['cv_folds']}-fold CV on the training block):")
            print(table[["model", "cv_accuracy_mean", "val_accuracy", "test_accuracy", "test_p_value_vs_chance"]]
                  .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if tuning_table is not None:
        baseline_cv = float("nan")
        splits = purged_kfold_splits(len(X_train), ml_cfg["cv_folds"], ml_cfg["embargo_pct"], y=y_train)
        base_scores = []
        for tr, te in splits:
            baseline = fit_pipeline(Pipeline([("impute", SimpleImputer(strategy="median")),
                                              ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=min_leaf,
                                                                            class_weight="balanced", random_state=42,
                                                                            n_jobs=-1))]),
                                    X_train.iloc[tr], y_train[tr], w_train[tr])
            base_scores.append(float(accuracy_score(y_train[te], baseline.predict(X_train.iloc[te]))))
        baseline_cv = float(np.mean(base_scores)) if base_scores else float("nan")
        optional("Tuning report", lambda: write_rf_tuning_report(
            ml_dir / "rf_tuning_report.md", tuning_table, rf_params, baseline_cv, n_runs, len(y_test), acc_te, label_text,
            "Excluded as leakage: " + ", ".join(LEAKAGE_COLS + META_LEAK_COLS[:6]) + "."))
        print(f"[ML] Tuning report        -> {ml_dir / 'rf_tuning_report.md'}")
    vol_values = df["vol_index"].to_numpy(dtype=float) if "vol_index" in df.columns else None
    explained = optional("Feature contributions", lambda: explain_vol_decisions(pipeline, X, vol_values, ml_dir))
    if explained:
        print(f"[ML] Contributions ({explained['method']}) -> feature_contributions.png/csv; "
              f"top: {', '.join(explained['top_features'][:3])}")
    heat = optional("Vol sensitivity heatmap", lambda: vol_sensitivity_heatmap(
        df, available_features, pd.Series(rf.feature_importances_, index=available_features), ml_dir))
    if heat:
        print(f"[ML] Vol-sensitive features: {', '.join(heat['most_vol_sensitive'])} -> vol_sensitivity_heatmap.png")
    points = validation.get("vol_points")
    if points is not None and len(points):
        threshold = float(np.percentile(points["composite_F"], 70))
        bounds = optional("Conformal vol bounds", lambda: conformal_vol_bounds(
            points["vol_index"].to_numpy(), (points["composite_F"] >= threshold).astype(int).to_numpy(),
            ml_cfg["conformal_alpha"], ml_dir / "conformal_vol_bounds.csv"))
        if bounds:
            (ml_dir / "conformal_vol_bounds.json").write_text(json.dumps(bounds, indent=2), encoding="utf-8")
            print(f"[ML] Conformal vol bounds (alpha {bounds['alpha']}): [{bounds['low']:.4g}, {bounds['high']:.4g}]")
    if validation.get("real_frames") and validation.get("sim_frames"):
        adv = optional("Adversarial validation", lambda: adversarial_validation(
            validation["real_frames"], validation["sim_frames"], ml_dir, ml_cfg["adversarial_auc_flag"]))
        if adv:
            print(f"[ML] {adv['verdict']}")
        else:
            print("[ML] Adversarial validation needs >= 8 windows of 30 bars on each side; not enough data this run.")
    report = optional("HTML report", lambda: write_html_report(analysis_dir, ml_dir))
    if report:
        print(f"[ML] HTML report          -> {report}")

    print(f"\n[ML] Data split for {n_runs} rows (ordered blocks):")
    print(f"  Training:   {len(X_train)} rows ({100 * len(X_train) / n_runs:.1f}%)")
    print(f"  Validation: {len(X_val)} rows ({100 * len(X_val) / n_runs:.1f}%)")
    print(f"  Test:       {len(X_test)} rows ({100 * len(X_test) / n_runs:.1f}%)")
    print("  Training fits the model; validation checks it during development.")
    print("  This script reports validation accuracy but does not tune on validation data.")
    print("  Test data stays held back for the final accuracy check.")
    print("\n[ML] Done.\n")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _select_features(
    df: pd.DataFrame, candidates: list[str], train_frac: float, nan_drop_pct: float = NAN_DROP_PCT
) -> list[str]:
    """
    Return the subset of candidates that are present in df and have fewer than
    nan_drop_pct NaN values in the training partition (config.toml [ml] nan_drop_pct).
    Prints a warning for every excluded column.
    """
    n_train = int(len(df) * train_frac)
    kept: list[str] = []
    for col in candidates:
        if col not in df.columns:
            print(f"[ML] Skipped (missing)  : {col}")
            continue
        nan_frac = df[col].iloc[:n_train].isna().mean()
        if nan_frac >= nan_drop_pct:
            print(
                f"[ML] Skipped ({nan_frac:.0%} NaN): {col}  "
                "(likely all-NaN in futures mode — run with floor prices to populate)"
            )
            continue
        kept.append(col)
    print(f"[ML] Using {len(kept)} / {len(candidates)} candidate features")
    return kept


def _write_summary(
    path: Path, analysis_dir: Path, asset_type: str, instrument: str,
    timeframe: str, n_runs: int, features: list[str],
    train_end: int, val_end: int,
    X_train, X_val, X_test,
    rf: RandomForestClassifier,
    acc_tr: float, acc_va: float, acc_te: float,
    cm_te: np.ndarray,
    imp_df: pd.DataFrame,
) -> None:
    lines = [
        "=== ML Pipeline Summary ===",
        f"Analysis dir      : {analysis_dir}",
        f"Asset type        : {asset_type}",
        f"Instrument        : {instrument}",
        f"Timeframe         : {timeframe}",
        f"Runs              : {n_runs}",
        f"Features used     : {len(features)}",
        f"Target            : {TARGET_COL} > 0  (1 = bullish, 0 = bearish)",
        "",
        "Split (block — preserves RNG-correlation structure across runs)",
        f"  Train  : rows 1 – {train_end}          ({len(X_train)} rows, {100 * len(X_train) / n_runs:.1f}%)",
        f"  Val    : rows {train_end + 1} – {val_end}  ({len(X_val)} rows, {100 * len(X_val) / n_runs:.1f}%)",
        f"  Test   : rows {val_end + 1} – {n_runs}  ({len(X_test)} rows, {100 * len(X_test) / n_runs:.1f}%)",
        "  Training fits the preprocessing and model parameters.",
        "  Validation is a held-out check; this script reports it but does not tune on it.",
        "  Test is held back for the final performance estimate, not model fitting.",
        "",
        "Accuracy",
        f"  OOB (train, bootstrap estimate) : {rf.oob_score_:.4f}",
        f"  Train                           : {acc_tr:.4f}",
        f"  Val                             : {acc_va:.4f}",
        f"  Test                            : {acc_te:.4f}",
        "",
        "Confusion matrix (test set)  [row = true, col = predicted]",
        f"                 Pred 0  Pred 1",
        f"  True 0 (bear)    {cm_te[0,0]}       {cm_te[0,1]}",
        f"  True 1 (bull)    {cm_te[1,0]}       {cm_te[1,1]}",
        "",
        "Interpretation",
        "  On synthetic random data test accuracy should converge to ~0.50.",
        "  Feature importances reflect spurious correlation in the null distribution.",
        f"  Compare against real historical {asset_type} data for {instrument} to assess signal.",
        "",
        "Excluded (leakage) columns:",
    ]
    for col in LEAKAGE_COLS:
        lines.append(f"  {col}")
    lines += [
        "",
        "Feature importances (all features, ranked):",
    ]
    for _, row in imp_df.iterrows():
        lines.append(f"  {int(row['rank']):2d}. {row['feature']:<45} {row['importance']:.6f}")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _plot_importances(imp_df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(imp_df))), facecolor="black")
    ax.set_facecolor("black")

    # Top-5 highlighted in teal, rest in muted grey-blue
    n = len(imp_df)
    colors = ["#00d4aa" if i < 5 else "#2a4a5a" for i in range(n)]

    # Plot reversed so the highest bar is at the top
    ax.barh(
        imp_df["feature"].iloc[::-1],
        imp_df["importance"].iloc[::-1],
        color=list(reversed(colors)),
        height=0.65,
        edgecolor="none",
    )

    for spine in ax.spines.values():
        spine.set_color("#2d3a44")
    ax.tick_params(colors="white", labelsize=9)
    ax.set_xlabel("Normalised Gini importance", color="#aab2bd", fontsize=10)
    ax.set_title(
        "Feature Importances — Random Forest (30-feature model)",
        color="white", fontsize=11, pad=12,
    )
    ax.xaxis.label.set_color("#aab2bd")
    for label in ax.get_yticklabels():
        label.set_color("white")

    fig.tight_layout()
    fig.savefig(output, dpi=130, facecolor="black", bbox_inches="tight")
    plt.close(fig)




# ═════════════════════════════════════════════════════════════════════════════
#  EXTENSIONS: cross-validation, model zoo, RF tuning, labels, weights, pruning
# ═════════════════════════════════════════════════════════════════════════════
import inspect  # noqa: E402
import tomllib  # noqa: E402
import warnings  # noqa: E402
from scipy import stats as _sps  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import FunctionTransformer  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def _config() -> dict:
    """config.toml through backtest_compare.load_config (defaults if the module is unavailable)."""
    try:
        from backtest_compare import load_config
        return load_config()
    except ValueError as exc:                                  # ConfigError: a value outside its allowed range
        raise SystemExit(f"[ML] {exc}") from None
    except Exception as exc:                                   # pragma: no cover - defensive
        print(f"[ML] Could not load config.toml ({exc}); using built-in ml defaults.")
        return {"ml": {"label_mode": "return", "cv_folds": 5, "embargo_pct": 0.02, "use_sample_weight": True,
                       "models": ["random_forest", "logistic"], "conformal_alpha": 0.10,
                       "adversarial_auc_flag": 0.55, "online_add_trees": 100, "online_update": False,
                       "tune_rf": False, "tuning_config": "rf_tuning.toml"},
                "pruning": {"enabled": True, "pct_difference": 150.0, "min_rows_kept": 10,
                            "columns": ["realized_vol_per_bar_pct"]},
                "volatility_search": {"target": 0.30}, "daily_tip": {}}


# Columns that describe HOW GOOD the simulation was (known only after scoring) - never predictors.
META_LEAK_COLS: list[str] = [
    "accuracy_index", "bar_direction_match_pct", "direction_match_pct", "exact_score", "net_price_match",
    "sample_weight", "is_augmented", "ensemble_label", "vol_index", "obv_last", "obv_slope_norm",
    "vwap_dev_pct", "composite_F", "run", "source_run",
]
OPTIONAL_FEATURES: list[str] = ["volume_cv", "volume_abs_return_corr"]


# ─── Outlier run detection (pruning), threshold in config.toml [pruning] ───────
def prune_outlier_runs(df: pd.DataFrame, pruning_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop runs whose % difference from the batch median exceeds pct_difference in any listed column.

    % difference = 100 * |x - median| / max(|median|, 1.4826 * MAD): the MAD floor stops columns
    whose median is ~0 from flagging everything. Never prunes below `min_rows_kept` rows.
    Live-market batches (price_modeling option 3) mix candle sizes from 2 minutes to a quarter, so
    there the median is taken within each `period`, and rows with data_source == "real" are never
    pruned: a real window is what the market did, not an outlier of the generator.
    """
    empty = pd.DataFrame(columns=["run", "column", "pct_difference", "value", "median"])
    if not pruning_cfg.get("enabled", False) or len(df) < 5:
        return df, empty
    limit = float(pruning_cfg["pct_difference"])
    reasons = []
    flagged = pd.Series(False, index=df.index)
    protected = (df["data_source"].astype(str) == "real") if "data_source" in df.columns \
        else pd.Series(False, index=df.index)
    if "period" in df.columns and df["period"].nunique(dropna=True) > 1:
        groups = [group.index for _, group in df.groupby("period", sort=False)]
    else:
        groups = [df.index]
    for column in pruning_cfg.get("columns", []):
        if column not in df.columns:
            continue
        for rows in groups:
            values = pd.to_numeric(df.loc[rows, column], errors="coerce")
            median = float(values.median())
            mad = float((values - median).abs().median()) * 1.4826
            scale = max(abs(median), mad, 1e-12)
            pct = 100.0 * (values - median).abs() / scale
            hits = (pct > limit).fillna(False) & ~protected.loc[rows]
            flagged.loc[rows] |= hits
            for idx in rows[hits.to_numpy()]:
                reasons.append({"run": df.at[idx, "run"] if "run" in df.columns else idx, "column": column,
                                "pct_difference": float(pct[idx]), "value": float(values[idx]), "median": median})
    kept = df[~flagged]
    if len(kept) < int(pruning_cfg.get("min_rows_kept", 10)):
        print(f"[ML] Pruning would leave {len(kept)} rows (< min_rows_kept); pruning skipped.")
        return df, empty
    return kept.copy(), pd.DataFrame(reasons, columns=empty.columns)


# ─── Purged k-fold with embargo (Lopez de Prado, ch. 7) ───────────────────────
def purged_kfold_splits(n: int, n_splits: int = 5, embargo_pct: float = 0.02, purge: int | None = None,
                        y: np.ndarray | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous test folds; rows within `purge` before and `embargo` after each fold leave the train set.

    Rows are ordered in time (runs / cycles), so removing the neighbours stops information that
    overlaps the test block (shared RNG chain, overlapping windows) leaking into training.
    """
    indices = np.arange(n)
    embargo = int(math.ceil(n * embargo_pct))
    purge = embargo if purge is None else int(purge)
    splits = []
    for test in np.array_split(indices, max(2, min(n_splits, n))):
        if len(test) == 0:
            continue
        lo, hi = test[0] - purge, test[-1] + embargo
        train = indices[(indices < lo) | (indices > hi)]
        if len(train) < 5:
            continue
        if y is not None and len(np.unique(y[train])) < 2:
            continue
        splits.append((train, test))
    return splits


def binomial_p_value(correct: int, total: int) -> float:
    """Two-sided p-value of `correct` hits out of `total` against pure chance (p = 0.5)."""
    return float(_sps.binomtest(int(correct), int(total), 0.5).pvalue) if total > 0 else float("nan")


def minimum_detectable_accuracy(n_test: int, z_alpha: float = 1.96, z_power: float = 0.84) -> float:
    """Smallest accuracy distinguishable from 0.5 with 80 % power at the 5 % level for n_test rows."""
    return float(0.5 + (z_alpha + z_power) * 0.5 / math.sqrt(max(n_test, 1)))


# ─── Leakage audit and the leak-free forecast dataset ─────────────────────────
def leakage_audit(X: pd.DataFrame, y: np.ndarray, threshold: float = 0.80) -> pd.DataFrame:
    """Single-feature AUC against the label: features that alone separate it are measuring the label itself.

    With label_mode 'return' the features and y = (total_return > 0) come from the SAME finished path, so
    several features (bullish_candle_pct, candle_direction_balance, rsi_distance_from_50, ...) leak the
    answer and 'accuracy well above 0.50' is mechanical, not forecasting.
    """
    rows = []
    if len(np.unique(y)) < 2:
        return pd.DataFrame(columns=["feature", "auc", "separation", "flag"])
    for column in X.columns:
        values = pd.to_numeric(X[column], errors="coerce")
        if values.nunique(dropna=True) < 2:
            continue
        auc = float(roc_auc_score(y, values.fillna(values.median())))
        separation = max(auc, 1.0 - auc)
        rows.append({"feature": column, "auc": auc, "separation": separation, "flag": int(separation >= threshold)})
    table = pd.DataFrame(rows)
    return table.sort_values("separation", ascending=False).reset_index(drop=True) if len(table) else table


def build_forecast_dataset(analysis_dir: Path, summary: pd.DataFrame, split: float = 0.70) -> pd.DataFrame:
    """One row per run: features from the first `split` of its bars, label = return over the remaining bars.

    Uses processed_bars.csv (written by the analysis step). Run-level columns that are not path features
    (vol_index, accuracy_index, sample_weight, ...) are carried over from analysis_summary.csv.
    """
    import market_analysis as ma
    bars_path = analysis_dir / "processed_bars.csv"
    if not bars_path.exists():
        raise FileNotFoundError("forecast mode needs processed_bars.csv (run the analysis step first).")
    bars = pd.read_csv(bars_path)
    rows = []
    for run, group in bars.groupby("run", sort=True):
        group = group.sort_values("bar") if "bar" in group.columns else group.sort_values("Date")
        group = group.reset_index(drop=True)
        cut = int(len(group) * split)
        if cut < 26 or len(group) - cut < 3:
            continue
        head = group.iloc[:cut][["Date", "Open", "High", "Low", "Close"]].copy().reset_index(drop=True)
        head["Date"] = pd.to_datetime(head["Date"])
        data = ma._add_indicators(head, int(run))
        metrics, _, _ = ma._metrics(data, ma._trades_for_run(data, int(run)), int(run), "")
        metrics["future_return_pct"] = float(100.0 * (group["Close"].iloc[-1] / group["Close"].iloc[cut - 1] - 1.0))
        metrics["forecast_bars"] = int(len(group) - cut)
        rows.append(metrics)
    if not rows:
        raise ValueError("forecast mode: no run has enough bars (needs >= 26 feature bars and 3 forecast bars).")
    dataset = pd.DataFrame(rows)
    carry = [c for c in summary.columns if c not in dataset.columns or c == "run"]
    return dataset.merge(summary[carry], on="run", how="left")


# ─── Targets and weights ──────────────────────────────────────────────────────
def make_target(df: pd.DataFrame, mode: str) -> tuple[np.ndarray, str]:
    """y for the chosen label mode; falls back to the return label when a column is missing."""
    if mode == "forecast":
        if "future_return_pct" in df.columns and df["future_return_pct"].notna().any():
            return (df["future_return_pct"] > 0).astype(int).to_numpy(), \
                "future_return_pct > 0 (last part of each run; features use only the earlier bars)"
        print("[ML] label_mode 'forecast' needs future_return_pct; using total_return_pct > 0.")
    elif mode == "direction_match":
        column = "bar_direction_match_pct"
        if column in df.columns and df[column].notna().any():
            return (pd.to_numeric(df[column], errors="coerce").fillna(50.0) > 50.0).astype(int).to_numpy(), \
                "direction_match_pct > 50 % ([BARS] criterion)"
        print("[ML] label_mode 'direction_match' needs bar_direction_match_pct (backtest runs); using total_return_pct > 0.")
    elif mode == "ensemble":
        if "ensemble_label" in df.columns and df["ensemble_label"].notna().any():
            return pd.to_numeric(df["ensemble_label"], errors="coerce").fillna(0).astype(int).to_numpy(), \
                "ensemble majority-vote direction"
        print("[ML] label_mode 'ensemble' needs ensemble_label (ensemble_k > 1); using total_return_pct > 0.")
    return (df[TARGET_COL] > 0).astype(int).to_numpy(), f"{TARGET_COL} > 0"


def describe_weights(df: pd.DataFrame, weights: np.ndarray) -> str:
    """One line on what the sample weights do (real vs simulated rows in live mode)."""
    if not (weights != 1).any():
        return "uniform"
    if "data_source" in df.columns:
        source = df["data_source"].astype(str).to_numpy()
        parts = [f"{name} x{weights[source == name].mean():g}" for name in ("real", "simulated") if (source == name).any()]
        share = float(weights[source == "real"].sum() / weights.sum()) if weights.sum() > 0 else float("nan")
        return f"on ({', '.join(parts)}; the {int((source == 'real').sum())} real rows carry {share:.0%} of the total weight)"
    return "on (accuracy_index x augmentation boost)"


def sample_weights(df: pd.DataFrame, enabled: bool) -> np.ndarray:
    """Accuracy-weighted sample importance: w = (accuracy_index + 1) / 2 in [0.05, 1], x augmentation boost."""
    weights = np.ones(len(df), dtype=float)
    if not enabled:
        return weights
    if "accuracy_index" in df.columns and df["accuracy_index"].notna().any():
        acc = pd.to_numeric(df["accuracy_index"], errors="coerce").fillna(-1.0).to_numpy()
        weights *= np.clip((acc + 1.0) / 2.0, 0.05, 1.0)
    if "sample_weight" in df.columns:
        weights *= pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).to_numpy()
    return weights


def _supports_weight(estimator) -> bool:
    try:
        return "sample_weight" in inspect.signature(estimator.fit).parameters
    except (TypeError, ValueError):
        return False


def fit_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, weights: np.ndarray | None) -> Pipeline:
    """Fit with sample weights when the final estimator supports them."""
    if weights is not None and _supports_weight(pipeline.steps[-1][1]):
        pipeline.fit(X, y, **{f"{pipeline.steps[-1][0]}__sample_weight": weights})
    else:
        pipeline.fit(X, y)
    return pipeline


# ─── Model zoo (#96) ──────────────────────────────────────────────────────────
def build_model_zoo(rf_params: dict, names: list[str], seed: int = 42) -> tuple[dict[str, Pipeline], list[str]]:
    """Pipelines for the requested models; optional libraries are skipped with a message."""
    zoo: dict[str, Pipeline] = {}
    skipped: list[str] = []

    def pipe(est, tanh: bool = False, scale: bool = True) -> Pipeline:
        steps = [("impute", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scale", StandardScaler()))
        if tanh:
            steps.append(("tanh", FunctionTransformer(np.tanh)))
        steps.append(("model", est))
        return Pipeline(steps)

    rf = {**rf_params, "oob_score": False}
    for name in names:
        if name == "random_forest":
            zoo[name] = pipe(RandomForestClassifier(**rf), scale=False)
        elif name == "extra_trees":
            zoo[name] = pipe(ExtraTreesClassifier(n_estimators=400, min_samples_leaf=rf.get("min_samples_leaf", 2),
                                                  class_weight="balanced", random_state=seed, n_jobs=-1), scale=False)
        elif name == "logistic":
            zoo[name] = pipe(LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))
        elif name == "logistic_tanh":
            zoo[name] = pipe(LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000), tanh=True)
        elif name == "mlp_tanh":
            zoo[name] = pipe(MLPClassifier(hidden_layer_sizes=(32, 16), activation="tanh", alpha=1e-2,
                                           max_iter=800, early_stopping=False, random_state=seed))
        elif name == "gradient_boosting":
            zoo[name] = pipe(GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                        subsample=0.8, random_state=seed), scale=False)
        elif name == "hist_gradient_boosting":
            zoo[name] = pipe(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4,
                                                            l2_regularization=1.0, random_state=seed), scale=False)
        elif name == "xgboost":
            try:
                from xgboost import XGBClassifier
                zoo[name] = pipe(XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                               colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1,
                                               random_state=seed), scale=False)
            except ImportError:
                skipped.append("xgboost (pip install xgboost)")
        elif name == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
                zoo[name] = pipe(LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8,
                                                colsample_bytree=0.8, verbose=-1, random_state=seed, n_jobs=-1),
                                 scale=False)
            except ImportError:
                skipped.append("lightgbm (pip install lightgbm)")
        else:
            skipped.append(f"{name} (unknown model name)")
    return zoo, skipped


def compare_models(X_train: pd.DataFrame, y_train: np.ndarray, w_train: np.ndarray, X_val: pd.DataFrame,
                   y_val: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray, rf_params: dict,
                   ml_cfg: dict, out_path: Path) -> pd.DataFrame:
    """Purged-CV accuracy on the training block + validation / test accuracy for every model."""
    zoo, skipped = build_model_zoo(rf_params, list(ml_cfg["models"]))
    for item in skipped:
        print(f"[ML] Model skipped: {item}")
    splits = purged_kfold_splits(len(X_train), ml_cfg["cv_folds"], ml_cfg["embargo_pct"], y=y_train)
    rows = []
    for name, model in zoo.items():
        scores = []
        for tr, te in splits:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = fit_pipeline(_clone(model), X_train.iloc[tr], y_train[tr], w_train[tr])
                    scores.append(float(accuracy_score(y_train[te], fitted.predict(X_train.iloc[te]))))
            except Exception as exc:
                print(f"[ML] {name}: CV fold failed ({type(exc).__name__}: {exc})")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = fit_pipeline(_clone(model), X_train, y_train, w_train)
                acc_val = float(accuracy_score(y_val, fitted.predict(X_val))) if len(y_val) else float("nan")
                predictions = fitted.predict(X_test) if len(y_test) else np.array([])
                acc_test = float(accuracy_score(y_test, predictions)) if len(y_test) else float("nan")
                hits = int((predictions == y_test).sum()) if len(y_test) else 0
        except Exception as exc:
            print(f"[ML] {name}: fit failed ({type(exc).__name__}: {exc})")
            continue
        rows.append({"model": name, "cv_folds_used": len(scores),
                     "cv_accuracy_mean": float(np.mean(scores)) if scores else float("nan"),
                     "cv_accuracy_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else float("nan"),
                     "val_accuracy": acc_val, "test_accuracy": acc_test, "test_rows": len(y_test),
                     "test_p_value_vs_chance": binomial_p_value(hits, len(y_test))})
    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values("cv_accuracy_mean", ascending=False).reset_index(drop=True)
    table.to_csv(out_path, index=False)
    return table


def _clone(model):
    from sklearn.base import clone
    return clone(model)


# ─── Random-forest hyper-parameter tuning (#95, rf_tuning.toml) ───────────────
RF_TUNING_DEFAULT = {
    "search": {"n_iter": 30, "cv_folds": 5, "embargo_pct": 0.02, "scoring": "accuracy",
               "random_state": 42, "n_jobs": 1},
    "params": {
        "n_estimators": [200, 500, 800], "max_depth": ["none", 4, 6, 8, 12],
        "min_samples_leaf": [1, 2, 5, 10, 20], "max_features": ["sqrt", "log2", 0.3, 0.5],
        "max_samples": [0.5, 0.7, 0.9, 1.0], "class_weight": ["balanced", "balanced_subsample", "none"],
        "criterion": ["gini", "entropy"], "ccp_alpha": [0.0, 0.001, 0.005],
    },
}


def load_rf_tuning(path: str | Path | None) -> dict:
    """rf_tuning.toml overlaid on the defaults ("none" -> None, max_samples 1.0 -> None)."""
    config = {"search": dict(RF_TUNING_DEFAULT["search"]),
              "params": {k: list(v) for k, v in RF_TUNING_DEFAULT["params"].items()}}
    candidate = Path(path) if path else Path(__file__).with_name("rf_tuning.toml")
    if not candidate.is_absolute() and not candidate.exists():
        candidate = Path(__file__).with_name(candidate.name)
    if candidate.exists():
        with open(candidate, "rb") as handle:
            user = tomllib.load(handle)
        config["search"].update(user.get("search", {}))
        for key, values in user.get("params", {}).items():
            config["params"][key] = list(values)
    cleaned: dict[str, list] = {}
    for key, values in config["params"].items():
        items = [None if (isinstance(v, str) and v.lower() == "none") else v for v in values]
        if key == "max_samples":
            items = [None if (v is not None and float(v) >= 1.0) else v for v in items]
        cleaned[key] = items
    config["params"] = cleaned
    return config


def tune_random_forest(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, tuning_path: str | Path | None,
                       ml_dir: Path, base_params: dict) -> tuple[dict, pd.DataFrame]:
    """Randomised search with purged CV; returns the tuned RF parameters and the full results table."""
    cfg = load_rf_tuning(tuning_path)
    search = cfg["search"]
    splits = purged_kfold_splits(len(X), int(search["cv_folds"]), float(search["embargo_pct"]), y=y)
    if len(splits) < 2:
        raise ValueError("Not enough rows for purged cross-validation during tuning.")
    grid_size = int(np.prod([len(v) for v in cfg["params"].values()]))
    n_iter = int(min(int(search["n_iter"]), grid_size))
    pipeline = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("rf", RandomForestClassifier(random_state=42, n_jobs=-1))])
    distributions = {f"rf__{k}": v for k, v in cfg["params"].items()}
    searcher = RandomizedSearchCV(pipeline, distributions, n_iter=n_iter, cv=splits, scoring=search["scoring"],
                                  random_state=int(search["random_state"]), n_jobs=int(search["n_jobs"]),
                                  refit=False, error_score="raise")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        searcher.fit(X, y, **{"rf__sample_weight": w})
    results = pd.DataFrame(searcher.cv_results_)
    keep = ["rank_test_score", "mean_test_score", "std_test_score"] + [c for c in results.columns if c.startswith("param_")]
    table = results[keep].sort_values("rank_test_score").reset_index(drop=True)
    table.columns = [c.replace("param_rf__", "") for c in table.columns]
    table.to_csv(ml_dir / "rf_tuning_results.csv", index=False)
    best_index = int(results["rank_test_score"].idxmin())
    best = {k.replace("rf__", ""): v for k, v in results.loc[best_index, "params"].items()}
    tuned = {**base_params, **best}
    return tuned, table


def write_rf_tuning_report(path: Path, table: pd.DataFrame, tuned: dict, baseline_cv: float, n_rows: int,
                           n_test: int, acc_test: float, label_text: str, leakage_note: str) -> None:
    """Extensive plain-text/Markdown report on how to make the ML work (diagnostics from this run)."""
    best_cv = float(table["mean_test_score"].iloc[0]) if len(table) else float("nan")
    mda = minimum_detectable_accuracy(max(n_test, 1))
    lines = [
        "# How to make the ML work - random forest tuning report", "",
        f"Target: {label_text}", f"Rows used: {n_rows}; held-out test rows: {n_test}", "",
        "## 1. What the tuning found",
        f"- Baseline forest, purged-CV accuracy: {baseline_cv:.4f}",
        f"- Best tuned forest, purged-CV accuracy: {best_cv:.4f} (test accuracy {acc_test:.4f})",
        f"- Tuned parameters: {json.dumps({k: (v if v is not None else 'none') for k, v in tuned.items() if k not in ('n_jobs', 'random_state', 'oob_score')})}",
        f"- Search size: {len(table)} configurations (rf_tuning.toml); full table in rf_tuning_results.csv", "",
        "## 2. Is a number above 0.50 meaningful?",
        (f"- With {n_test} test rows the smallest accuracy distinguishable from a coin flip (5 % level, 80 % power) is {mda:.3f}."
         if mda < 1.0 else
         f"- With only {n_test} test rows NO accuracy (even 100 %) can be told apart from a coin flip: more runs are needed first."),
        f"- Best CV accuracy minus 0.50 = {best_cv - 0.5:+.3f}. Gains smaller than the detectable edge are noise.",
        "- Tuning on CV accuracy selects the luckiest configuration; the held-out test number is the honest one.", "",
        "## 3. How to make it work (ordered by expected payoff)",
        "1. More runs. Edge detection scales with sqrt(n): quadrupling the rows halves the detectable edge. "
        "Use the 'CSVs + metrics only' output mode for 1,000+ runs.",
        "2. Use a label that carries signal. On pure random-walk data total_return_pct > 0 is a coin flip by construction "
        "and ~0.50 test accuracy is the CORRECT answer. direction_match_pct > 50 % (label_mode in config.toml) ties "
        "the classifier to real prices, but only backtest runs have it.",
        "3. Remove leakage before celebrating. " + leakage_note,
        "4. Cross-validate with purging + embargo (done here) - random shuffles inflate scores when runs share an RNG chain.",
        "5. Prefer regularised models when rows are scarce: raise min_samples_leaf / lower max_depth (see the best rows "
        "of the table), or compare against logistic / tanh models in model_comparison.csv.",
        "6. Weight by accuracy_index (use_sample_weight) so runs that already track the real price count more.",
        "7. Add real data: the adversarial AUC (adversarial_validation.txt) says whether simulated and real windows are "
        "distinguishable; ~0.50 means the simulator is statistically indistinguishable on these features.",
        "8. Stay out-of-sample: calibrate on the older cycles, judge on the newest (the self-correction oos_F in "
        "convergence_certificate.json).", "",
        "## 4. Top configurations",
    ]
    lines.append(table.head(10).to_string(index=False) if len(table) else "(no results)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



# ─── Online learning: incremental random forest (warm_start) ──────────────────
def online_update_forest(ml_dir: Path, X_new: pd.DataFrame, y_new: np.ndarray, w_new: np.ndarray | None,
                         add_trees: int = 100) -> dict | None:
    """Add trees fitted on a NEW batch to the saved forest; the existing trees are kept untouched."""
    model_path, meta_path = ml_dir / "model.pkl", ml_dir / "model_meta.json"
    if not model_path.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if len(np.unique(y_new)) < 2:
        print("[ML] Online update skipped: the new batch contains one class only.")
        return None
    pipeline = joblib.load(model_path)
    rf: RandomForestClassifier = pipeline.named_steps["rf"]
    missing = [f for f in meta["features"] if f not in X_new.columns]
    if missing:
        print(f"[ML] Online update skipped: new batch lacks features {missing}.")
        return None
    before = len(rf.estimators_)
    transformed = pipeline[:-1].transform(X_new[meta["features"]])
    rf.set_params(warm_start=True, n_estimators=before + int(add_trees), oob_score=False)
    rf.fit(transformed, y_new, sample_weight=w_new)
    joblib.dump(pipeline, model_path)
    meta["trees"] = len(rf.estimators_)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"trees_before": before, "trees_after": len(rf.estimators_), "new_rows": len(y_new)}


# ─── Conformal prediction on vol_index ────────────────────────────────────────
def conformal_vol_bounds(vol: np.ndarray, good: np.ndarray, alpha: float, out_path: Path,
                         seed: int = 7) -> dict | None:
    """Split-conformal p-values that a vol_index yields a 'good' simulation; returns bounds for the search.

    Features are log(vol) and log(vol)^2 (so a peak can be learned). p-values are valid only if rows are
    exchangeable; search points chosen adaptively are not, so treat the bounds as guidance, not a guarantee.
    """
    vol, good = np.asarray(vol, float), np.asarray(good, int)
    ok = np.isfinite(vol) & (vol > 0)
    vol, good = vol[ok], good[ok]
    if len(vol) < 30 or len(np.unique(good)) < 2 or len(np.unique(np.round(vol, 6))) < 5:
        return None
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(vol))
    n_train = int(0.5 * len(vol))
    train, cal = order[:n_train], order[n_train:]
    if len(np.unique(good[train])) < 2 or len(cal) < 10:
        return None
    feat = lambda v: np.column_stack([np.log(v), np.log(v) ** 2])                      # noqa: E731
    model = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(C=10.0, max_iter=2000))])
    model.fit(feat(vol[train]), good[train])
    classes = list(model.named_steps["lr"].classes_)
    p_cal = model.predict_proba(feat(vol[cal]))
    scores = 1.0 - p_cal[np.arange(len(cal)), [classes.index(g) for g in good[cal]]]
    grid = np.geomspace(vol.min(), vol.max(), 200)
    p_grid = model.predict_proba(feat(grid))
    rows = []
    for k, v in enumerate(grid):
        pv = {}
        for label in (1, 0):
            prob = p_grid[k, classes.index(label)] if label in classes else 0.0
            pv[label] = (1.0 + float(np.sum(scores >= 1.0 - prob))) / (len(scores) + 1.0)
        rows.append({"vol_index": float(v), "p_value_good": pv[1], "p_value_bad": pv[0]})
    table = pd.DataFrame(rows)
    table.to_csv(out_path, index=False)
    keep = table[table["p_value_good"] >= alpha]
    if keep.empty:
        return {"low": float("nan"), "high": float("nan"), "alpha": alpha, "calibration_rows": len(cal)}
    return {"low": float(keep["vol_index"].min()), "high": float(keep["vol_index"].max()),
            "alpha": alpha, "calibration_rows": int(len(cal))}


# ─── Adversarial validation (simulated vs real windows) ───────────────────────
def window_feature_rows(frames: list[pd.DataFrame], window: int = 30, stride: int = 15) -> pd.DataFrame:
    """Analysis features for rolling windows of OHLC frames (same indicators as analysis_summary)."""
    import market_analysis as ma
    rows = []
    for frame in frames:
        frame = frame[["Open", "High", "Low", "Close"]].dropna().reset_index(drop=True)
        for start in range(0, max(1, len(frame) - window + 1), stride):
            chunk = frame.iloc[start:start + window].copy().reset_index(drop=True)
            if len(chunk) < window:
                continue
            chunk["Date"] = pd.date_range("2000-01-03", periods=len(chunk), freq="D")
            data = ma._add_indicators(chunk, 0)
            metrics, _, _ = ma._metrics(data, ma._trades_for_run(data, 0), 0, "")
            rows.append(metrics)
    return _add_engineered_features(pd.DataFrame(rows)) if rows else pd.DataFrame()


def adversarial_validation(real_frames: list[pd.DataFrame], sim_frames: list[pd.DataFrame], ml_dir: Path,
                           auc_flag: float = 0.55, window: int = 30) -> dict | None:
    """Train an RF to tell simulated windows (0) from real windows (1); cross-validated AUC.

    AUC > flag: possible distribution shift / leakage. AUC ~ 0.50: indistinguishable on these features.
    Both sides are rebased to the same start price, so price LEVEL cannot give the answer away.
    """
    real_df = window_feature_rows(real_frames, window, window // 2)
    sim_df = window_feature_rows(sim_frames, window, window // 2)
    if len(real_df) < 8 or len(sim_df) < 8:
        return None
    features = [f for f in FEATURES if f in real_df.columns and f in sim_df.columns
                and f not in OPTIONAL_FEATURES]
    data = pd.concat([sim_df[features], real_df[features]], ignore_index=True)
    labels = np.r_[np.zeros(len(sim_df), dtype=int), np.ones(len(real_df), dtype=int)]
    data = data.loc[:, data.nunique(dropna=True) > 1]
    folds = int(min(5, np.bincount(labels).min()))
    if folds < 2 or data.shape[1] == 0:
        return None
    model = Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                    class_weight="balanced", random_state=42, n_jobs=-1))])
    from sklearn.model_selection import cross_val_predict
    proba = cross_val_predict(model, data, labels, cv=StratifiedKFold(folds, shuffle=True, random_state=42),
                              method="predict_proba")[:, 1]
    auc = float(roc_auc_score(labels, proba))
    model.fit(data, labels)
    importance = pd.Series(model.named_steps["rf"].feature_importances_, index=data.columns).sort_values(ascending=False)
    if auc > auc_flag:
        verdict = (f"AUC {auc:.3f} > {auc_flag}: simulated and real windows are distinguishable - possible "
                   f"distribution shift or leakage. Most telling features: {', '.join(importance.index[:3])}.")
    else:
        verdict = (f"AUC {auc:.3f} <= {auc_flag}: the classifier cannot reliably separate simulated from real "
                   "windows on these features (a self-validation milestone, not proof of a good forecast).")
    (ml_dir / "adversarial_validation.txt").write_text(
        f"Adversarial validation (RF, {folds}-fold CV)\nSimulated windows: {len(sim_df)}  Real windows: {len(real_df)}\n"
        f"{verdict}\n\nTop features:\n{importance.head(8).to_string()}\n", encoding="utf-8")
    return {"auc": auc, "verdict": verdict, "sim_windows": len(sim_df), "real_windows": len(real_df),
            "top_features": importance.index[:5].tolist()}


# ─── Explaining the model: SHAP when installed, else exact tree-path contributions ─
def tree_path_contributions(rf: RandomForestClassifier, X: np.ndarray, class_label: int = 1) -> tuple[np.ndarray, float]:
    """Per-row feature contributions to P(class) from the decision paths of every tree (Saabas method).

    contribution[i, f] = sum over path steps that split on f of the change in the node's class probability,
    averaged over trees; bias + contributions = the forest's predicted probability for row i.
    """
    classes = list(rf.classes_)
    if class_label not in classes:
        return np.zeros((len(X), X.shape[1])), 0.0
    column = classes.index(class_label)
    contributions = np.zeros((len(X), X.shape[1]))
    bias = 0.0
    for estimator in rf.estimators_:
        tree = estimator.tree_
        value = tree.value[:, 0, :]
        prob = value[:, column] / np.maximum(value.sum(axis=1), 1e-12)
        bias += float(prob[0])
        path = estimator.decision_path(X.astype(np.float32))
        for i in range(len(X)):
            nodes = path.indices[path.indptr[i]:path.indptr[i + 1]]
            for parent, child in zip(nodes[:-1], nodes[1:]):
                contributions[i, tree.feature[parent]] += prob[child] - prob[parent]
    n_trees = len(rf.estimators_)
    return contributions / n_trees, bias / n_trees


def explain_vol_decisions(pipeline: Pipeline, X: pd.DataFrame, vol_index: np.ndarray | None, ml_dir: Path,
                          max_rows: int = 300) -> dict:
    """Feature contributions per run (SHAP if available) with the vol_index used overlaid."""
    rf: RandomForestClassifier = pipeline.named_steps["rf"]
    if len(X) > max_rows:
        idx = np.unique(np.linspace(0, len(X) - 1, max_rows).round().astype(int))
        X = X.iloc[idx]
        vol_index = None if vol_index is None else np.asarray(vol_index)[idx]
    transformed = pipeline[:-1].transform(X)
    method = "tree-path contributions (Saabas, exact decomposition of the forest probability)"
    try:
        import shap                                                              # optional
        explainer = shap.TreeExplainer(rf)
        values = explainer.shap_values(transformed)
        values = values[1] if isinstance(values, list) else (values[:, :, 1] if np.ndim(values) == 3 else values)
        method = "SHAP TreeExplainer"
    except Exception:
        values, _ = tree_path_contributions(rf, transformed)
    frame = pd.DataFrame(values, columns=X.columns)
    if vol_index is not None:
        frame.insert(0, "vol_index", np.asarray(vol_index, float))
    frame.insert(0, "row", np.arange(len(frame)))
    frame.to_csv(ml_dir / "feature_contributions.csv", index=False)

    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=X.columns).sort_values(ascending=False)
    top = mean_abs.index[:8].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="black")
    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#2d3a44")
    axes[0].barh(top[::-1], mean_abs[top].to_numpy()[::-1], color="#00d4aa")
    axes[0].set_title(f"Mean |contribution| to P(bullish)\n{method}", color="white", fontsize=9)
    if vol_index is not None and np.isfinite(np.asarray(vol_index, float)).any():
        order = np.argsort(np.asarray(vol_index, float))
        for feature in top[:4]:
            axes[1].plot(np.asarray(vol_index, float)[order], frame[feature].to_numpy()[order], marker="o",
                         linewidth=.8, markersize=3, label=feature)
        axes[1].set_xlabel("vol_index used", color="white")
        axes[1].set_ylabel("contribution to P(bullish)", color="white")
        axes[1].set_title("Top feature contributions vs the vol_index of each run", color="white", fontsize=9)
        axes[1].legend(facecolor="#111111", labelcolor="white", fontsize=7)
    else:
        for feature in top[:4]:
            axes[1].plot(frame[feature].to_numpy(), linewidth=.8, label=feature)
        axes[1].set_xlabel("run (rows)", color="white")
        axes[1].set_title("Top feature contributions per run", color="white", fontsize=9)
        axes[1].legend(facecolor="#111111", labelcolor="white", fontsize=7)
    fig.tight_layout()
    fig.savefig(ml_dir / "feature_contributions.png", dpi=120, facecolor="black")
    plt.close(fig)
    return {"method": method, "top_features": top[:5], "rows": len(frame)}


def vol_sensitivity_heatmap(df: pd.DataFrame, features: list[str], importances: pd.Series, ml_dir: Path) -> dict | None:
    """Feature x vol heatmap: mean z-score of each indicator inside vol_index quantile bins."""
    if "vol_index" not in df.columns:
        return None
    vol = pd.to_numeric(df["vol_index"], errors="coerce")
    if vol.nunique(dropna=True) < 6:
        return None
    usable = [f for f in features if f in df.columns and pd.to_numeric(df[f], errors="coerce").std() > 0]
    if not usable:
        return None
    z = df[usable].apply(pd.to_numeric, errors="coerce")
    z = (z - z.mean()) / z.std(ddof=0)
    bins = pd.qcut(vol, q=min(5, vol.nunique()), duplicates="drop")
    heat = z.groupby(bins, observed=True).mean().T
    heat.columns = [f"{iv.left:.2f}-{iv.right:.2f}" for iv in heat.columns]
    spearman = {f: float(_sps.spearmanr(vol, pd.to_numeric(df[f], errors="coerce"), nan_policy="omit")[0]) for f in usable}
    weight = importances.reindex(usable).fillna(0.0)
    sensitivity = pd.DataFrame({"spearman_with_vol": pd.Series(spearman),
                                "importance": weight,
                                "importance_x_abs_spearman": weight * pd.Series(spearman).abs()})
    sensitivity = sensitivity.sort_values("importance_x_abs_spearman", ascending=False)
    sensitivity.to_csv(ml_dir / "feature_vol_sensitivity.csv")
    ordered = heat.loc[sensitivity.index[:20]]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(ordered))), facecolor="black")
    ax.set_facecolor("black")
    image = ax.imshow(ordered.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    ax.set_xticks(range(ordered.shape[1]), ordered.columns, rotation=30, ha="right", color="white", fontsize=8)
    ax.set_yticks(range(ordered.shape[0]), ordered.index, color="white", fontsize=8)
    ax.set_xlabel("vol_index bin", color="white")
    ax.set_title("Feature x vol sensitivity (mean z-score per vol bin; rows ranked by importance x |rho|)",
                 color="white", fontsize=9)
    fig.colorbar(image, ax=ax).ax.tick_params(colors="white")
    fig.tight_layout()
    fig.savefig(ml_dir / "vol_sensitivity_heatmap.png", dpi=120, facecolor="black")
    plt.close(fig)
    return {"most_vol_sensitive": sensitivity.index[:3].tolist()}


# ─── Improvement tips (#87) and sending ───────────────────────────────────────
def _read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def collect_findings(analysis_dir: Path) -> dict:
    """Facts from the latest outputs; used by the HTML report and the daily improvement tip."""
    facts: dict = {}
    cert_path = analysis_dir / "convergence_certificate.json"
    if cert_path.exists():
        try:
            facts["certificate"] = json.loads(cert_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    ml_root = analysis_dir / "ml_results"
    accuracy_files = sorted(ml_root.rglob("accuracy_metrics.csv"), key=lambda p: p.stat().st_mtime) if ml_root.exists() else []
    if accuracy_files:
        facts["accuracy"] = pd.read_csv(accuracy_files[-1]).iloc[0].to_dict()
        facts["ml_dir"] = accuracy_files[-1].parent
        ml_dir = accuracy_files[-1].parent
        comp = _read_csv_safe(ml_dir / "model_comparison.csv")
        if len(comp):
            facts["best_model"] = comp.iloc[0].to_dict()
        adv = ml_dir / "adversarial_validation.txt"
        if adv.exists():
            facts["adversarial_text"] = adv.read_text(encoding="utf-8")
        pruned = _read_csv_safe(ml_dir / "pruned_runs.csv")
        facts["pruned_rows"] = int(len(pruned))
    weak = _read_csv_safe(analysis_dir / "weak_segments.csv")
    facts["weak_segments"] = int(len(weak))
    acc = _read_csv_safe(analysis_dir / "backtest_accuracy.csv")
    if len(acc) and "bar_direction_match_pct" in acc:
        facts["mean_direction_match_pct"] = float(pd.to_numeric(acc["bar_direction_match_pct"], errors="coerce").mean())
    return facts


def daily_tip_text(analysis_dir: str | Path) -> str:
    """Rule-based improvement suggestions from the latest results (what to try next, and why)."""
    facts = collect_findings(Path(analysis_dir))
    tips: list[str] = []
    cert = facts.get("certificate")
    if cert:
        if not cert.get("converged"):
            tips.append("Self-correction did not converge: raise volatility_search.max_iters or loops in config.toml, "
                        "or switch method to 'bayes' (fewer evaluations).")
        oos = cert.get("out_of_sample_improvement")
        if oos is not None and np.isfinite(oos) and oos <= 0:
            tips.append(f"Out-of-sample improvement over the null is {oos:+.2f}: the calibrated vol does not generalise. "
                        "Use more cycles, raise consensus_runs, and trust oos_F rather than in-sample F.")
        if not cert.get("bars_met", True):
            tips.append("The [BARS] criterion (direction match) is not met. A symmetric generator cannot beat 50 % by "
                        "tuning volatility; add conditional structure (AR(1), streak parameters, regime multipliers) or "
                        "drop 'bars' from volatility_search.criteria.")
        ci = cert.get("vol_ci") or [None, None]
        if ci[0] is not None and ci[1] is not None and cert.get("optimal_vol"):
            width = (ci[1] - ci[0]) / cert["optimal_vol"]
            if width > 0.6:
                tips.append(f"The vol 90 % CI is wide ({width:.0%} of the optimum): add cycles or mc_paths to tighten it.")
    acc = facts.get("accuracy")
    if acc:
        test_acc = float(acc.get("test_accuracy", float("nan")))
        n_test = int(acc.get("test_rows", 0) or 0)
        if np.isfinite(test_acc) and n_test:
            mda = minimum_detectable_accuracy(n_test)
            if mda >= 1.0:
                tips.append(f"Only {n_test} test rows: no accuracy, even 100%, can be told apart from a coin flip at that size. "
                            "Generate many more runs (aim for 200+) before reading anything into the ML result.")
            elif test_acc < mda:
                tips.append(f"Test accuracy {test_acc:.3f} is below the detectable edge {mda:.3f} for {n_test} test rows: "
                            "generate more runs before reading anything into the ML result.")
        if float(acc.get("train_accuracy", 0)) - test_acc > 0.25:
            tips.append("Train accuracy far exceeds test accuracy (overfit): raise min_samples_leaf / use rf_tuning.toml.")
    leak = _read_csv_safe(facts["ml_dir"] / "leakage_audit.csv") if facts.get("ml_dir") else pd.DataFrame()
    if len(leak) and int(leak["flag"].sum()) > 0:
        tips.insert(0, f"{int(leak['flag'].sum())} features leak the label (e.g. {leak.iloc[0]['feature']}): set "
                       "[ml] label_mode = \"forecast\" so features come from earlier bars than the label, otherwise "
                       "high ML accuracy means nothing.")
    best = facts.get("best_model")
    if best and best.get("test_p_value_vs_chance", 1) > 0.05:
        tips.append(f"Best model '{best['model']}' is not significantly better than a coin flip (p = "
                    f"{best['test_p_value_vs_chance']:.2f}): try label_mode = 'direction_match' on backtest runs.")
    if facts.get("adversarial_text") and "possible" in facts["adversarial_text"]:
        tips.append("Adversarial validation separates simulated from real windows: compare wick_factor, ar1 and "
                    "jump_prob against the real series and let the GA tune them (volatility_search.tune_shape).")
    if facts.get("weak_segments", 0) > 0:
        tips.append(f"{facts['weak_segments']} weak direction-match segments were flagged (weak_segments.csv): inspect "
                    "those windows for regime changes or gaps.")
    if facts.get("pruned_rows", 0) > 0:
        tips.append(f"{facts['pruned_rows']} outlier rows were pruned; review pruning.pct_difference if that is a lot.")
    if not tips:
        tips.append("No problems detected. Next step: increase the number of backtest cycles or add a second instrument "
                    "so transfer learning can seed its volatility search.")
    header = f"Daily ML improvement tips - {pd.Timestamp.now():%Y-%m-%d}"
    return header + "\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(tips[:6], start=1))


def send_message(text: str, tip_cfg: dict, dry_run: bool = False) -> str:
    """Deliver the tip: print | ntfy (free push, https://ntfy.sh) | smtp (any mailbox)."""
    channel = str(tip_cfg.get("channel", "print")).lower()
    if dry_run or channel == "print":
        print(text)
        return "printed"
    if channel == "ntfy":
        import urllib.request
        topic = str(tip_cfg.get("ntfy_topic", "")).strip()
        if not topic:
            raise ValueError("daily_tip.ntfy_topic is empty.")
        request = urllib.request.Request(f"{str(tip_cfg.get('ntfy_server', 'https://ntfy.sh')).rstrip('/')}/{topic}",
                                         data=text.encode("utf-8"), headers={"Title": "ML improvement tips"})
        with urllib.request.urlopen(request, timeout=20) as response:
            return f"ntfy {response.status}"
    if channel == "smtp":
        import os
        import smtplib
        from email.message import EmailMessage
        password = os.environ.get(str(tip_cfg.get("smtp_password_env", "ML_TIP_SMTP_PASSWORD")), "")
        if not (tip_cfg.get("smtp_host") and tip_cfg.get("smtp_user") and tip_cfg.get("smtp_to") and password):
            raise ValueError("smtp needs smtp_host, smtp_user, smtp_to and the password environment variable.")
        message = EmailMessage()
        message["Subject"], message["From"], message["To"] = "ML improvement tips", tip_cfg["smtp_user"], tip_cfg["smtp_to"]
        message.set_content(text)
        with smtplib.SMTP(tip_cfg["smtp_host"], int(tip_cfg.get("smtp_port", 587)), timeout=30) as server:
            server.starttls()
            server.login(tip_cfg["smtp_user"], password)
            server.send_message(message)
        return "smtp sent"
    raise ValueError(f"Unknown daily_tip.channel '{channel}'.")


def seconds_until(clock: str, now: pd.Timestamp | None = None) -> float:
    """Seconds from `now` to the next HH:MM."""
    now = now or pd.Timestamp.now()
    hour, minute = (int(part) for part in clock.split(":"))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += pd.Timedelta(days=1)
    return float((target - now).total_seconds())


def run_daily_tip(analysis_dir: Path, send: bool = False, daemon: bool = False, dry_run: bool = False) -> None:
    """One tip now, or (daemon) one every day at daily_tip.time. Prefer cron / Task Scheduler for always-on use."""
    import time as _time
    cfg = _config()["daily_tip"]
    while True:
        text = daily_tip_text(analysis_dir)
        (analysis_dir / "daily_tip.txt").write_text(text + "\n", encoding="utf-8")
        if send or daemon:
            print(f"[tip] {send_message(text, cfg, dry_run=dry_run)}")
        else:
            print(text)
        if not daemon:
            return
        wait = seconds_until(str(cfg.get("time", "08:30")))
        print(f"[tip] next message in {wait / 3600:.1f} h (at {cfg.get('time', '08:30')}). Ctrl+C to stop.")
        _time.sleep(wait)


SCHEDULER_HELP = (
    "Linux/macOS cron (08:30 daily):   30 8 * * *  cd /path/to/model && python ml_model.py --daily-tip --send\n"
    "Windows Task Scheduler (08:30):   schtasks /Create /SC DAILY /ST 08:30 /TN MLTip /TR "
    "\"python C:\\path\\to\\model\\ml_model.py --daily-tip --send\""
)


# ─── HTML long summary report with keyword summary (#86) ──────────────────────
_STOPWORDS = set("""the and for that this with from are was were will not but have has had its into over than then them
their there which when what where while about after before between both each other some such only also more most very
can could should would may might must been being does did done use used using per via one two three all any out off
run runs row rows data file files column columns value values must""".split())


def _embed_png(path: Path) -> str:
    import base64
    if not path.exists():
        return ""
    return f'<img alt="{path.name}" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}">'


def _html_table(frame: pd.DataFrame, max_rows: int = 25) -> str:
    if frame is None or frame.empty:
        return "<p><em>none</em></p>"
    return frame.head(max_rows).to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}", classes="t")


def keyword_summary(text: str, top: int = 18) -> list[tuple[str, int]]:
    import re
    from collections import Counter
    words = [w for w in re.findall(r"[a-z][a-z_]{3,}", text.lower()) if w not in _STOPWORDS]
    return Counter(words).most_common(top)


def write_html_report(analysis_dir: str | Path, ml_dir: str | Path | None = None) -> Path:
    """One long, self-contained HTML report from whatever outputs exist (also usable without the ML step)."""
    import html
    analysis_dir = Path(analysis_dir)
    facts = collect_findings(analysis_dir)
    ml_dir = Path(ml_dir) if ml_dir else facts.get("ml_dir")
    sections: list[tuple[str, str]] = []
    text_blob: list[str] = []

    findings = []
    cert = facts.get("certificate")
    if cert:
        findings.append(f"Self-correction {'converged' if cert.get('converged') else 'did NOT converge'}: optimal vol "
                        f"{cert['optimal_vol']:.4g}, accuracy index {cert['accuracy_index']:.3f}, composite F "
                        f"{cert['composite_F']:.3f} vs null {cert['null_F']:.3f}.")
        if cert.get("out_of_sample_F") is not None:
            findings.append(f"Out-of-sample F {cert['out_of_sample_F']:.3f} vs out-of-sample null "
                            f"{cert['out_of_sample_null_F']:.3f} (newest cycles, not used to calibrate).")
        findings.append("Criteria met: " + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in cert["criteria_met"].items()) + ".")
    if "mean_direction_match_pct" in facts:
        findings.append(f"Mean bar direction match: {facts['mean_direction_match_pct']:.1f}% (50% = coin flip).")
    if facts.get("accuracy"):
        a = facts["accuracy"]
        findings.append(f"ML test accuracy {a['test_accuracy']:.3f} on {int(a['test_rows'])} rows "
                        f"(train {a['train_accuracy']:.3f}, OOB {a['oob_accuracy']:.3f}).")
    if facts.get("best_model"):
        b = facts["best_model"]
        findings.append(f"Best model by purged CV: {b['model']} (CV {b['cv_accuracy_mean']:.3f}, test {b['test_accuracy']:.3f}, "
                        f"p vs chance {b['test_p_value_vs_chance']:.3f}).")
    if facts.get("adversarial_text"):
        findings.append(facts["adversarial_text"].splitlines()[2])
    findings.append(f"Weak direction segments flagged: {facts.get('weak_segments', 0)}; pruned outlier runs: {facts.get('pruned_rows', 0)}.")
    sections.append(("Key findings", "<ul>" + "".join(f"<li>{html.escape(f)}</li>" for f in findings) + "</ul>"))
    text_blob.extend(findings)

    log = analysis_dir / "equilibrium_log.txt"
    if log.exists():
        lines = log.read_text(encoding="utf-8").splitlines()[-15:]
        sections.append(("Equilibrium log (latest)", "<pre>" + html.escape("\n".join(lines)) + "</pre>"))
        text_blob.extend(lines)
    for title, name in (("Per-cycle accuracy (backtest)", "backtest_accuracy.csv"),
                        ("Volatility equilibrium (learned)", "vol_equilibrium.csv"),
                        ("Regime volatility multipliers", "regime_vol_multipliers.csv"),
                        ("Weak direction-match segments", "weak_segments.csv"),
                        ("Intraday variance profile", "intraday_variance_profile.csv"),
                        ("Live market snapshot (current price)", "live_snapshot.csv"),
                        ("Live market windows (real candles per period)", "live_windows.csv")):
        frame = _read_csv_safe(analysis_dir / name)
        if len(frame):
            sections.append((title, _html_table(frame)))
    if cert:
        sections.append(("Convergence certificate", "<pre>" + html.escape(json.dumps(
            {k: v for k, v in cert.items() if k not in ("per_cycle_vol", "cycle_seeds", "params")}, indent=2, default=str)) + "</pre>"))
    for caption, name in (("Backtest: simulated vs real (before = null)", "backtest_plot_null.png"),
                          ("Backtest: simulated vs real (self-corrected)", "backtest_plot.png"),
                          ("Live market: real windows and simulated futures", "live_overview.png")):
        image = _embed_png(analysis_dir / name)
        if image:
            sections.append((caption, image))
    if ml_dir and Path(ml_dir).exists():
        ml_dir = Path(ml_dir)
        for title, name in (("Model comparison (purged CV)", "model_comparison.csv"),
                            ("Random-forest tuning (top)", "rf_tuning_results.csv"),
                            ("Feature importances", "feature_importances.csv"),
                            ("Outlier runs pruned", "pruned_runs.csv")):
            frame = _read_csv_safe(ml_dir / name)
            if len(frame):
                sections.append((title, _html_table(frame, 15)))
        for title, name in (("summary.txt", "summary.txt"), ("How to make the ML work", "rf_tuning_report.md"),
                            ("Adversarial validation", "adversarial_validation.txt"),
                            ("Stock path similarity", "stock_similarity_summary.txt")):
            path = ml_dir / name
            if path.exists():
                content = path.read_text(encoding="utf-8")
                sections.append((title, "<pre>" + html.escape(content) + "</pre>"))
                text_blob.append(content)
        for caption, name in (("Feature importances", "feature_importances.png"),
                              ("Feature contributions per run vs vol_index", "feature_contributions.png"),
                              ("Feature x vol sensitivity", "vol_sensitivity_heatmap.png")):
            image = _embed_png(ml_dir / name)
            if image:
                sections.append((caption, image))
    sections.append(("Today's improvement tips", "<pre>" + html.escape(daily_tip_text(analysis_dir)) + "</pre>"))

    keywords = keyword_summary(" ".join(text_blob))
    chips = " ".join(f'<span class="chip">{html.escape(w)} <b>{n}</b></span>' for w, n in keywords)
    body = f'<section><h2>Keyword summary</h2><p>{chips or "<em>no text yet</em>"}</p></section>' + "".join(
        f"<section><h2>{html.escape(t)}</h2>{c}</section>" for t, c in sections)
    page = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Simulation self-correction report</title><style>"
        "body{font-family:system-ui,sans-serif;background:#0b0e11;color:#dfe5ec;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        "h1,h2{color:#7fd1ff}section{margin:2rem 0;border-top:1px solid #243040;padding-top:1rem}"
        "pre{background:#111820;padding:1rem;overflow:auto;border-radius:6px;font-size:.8rem}"
        "table.t{border-collapse:collapse;font-size:.78rem}table.t td,table.t th{padding:.25rem .6rem;border-bottom:1px solid #243040}"
        ".chip{display:inline-block;background:#18222e;border-radius:999px;padding:.15rem .7rem;margin:.15rem}"
        "img{max-width:100%;border-radius:6px}</style></head><body>"
        f"<h1>Simulation self-correction report</h1><p>Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}.</p>{body}</body></html>"
    )
    out = analysis_dir / "summary_report.html"
    out.write_text(page, encoding="utf-8")
    return out


# ─── Standalone entry point ───────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ML pipeline on price_modeling.py analysis output.\n"
            "Works with both Python (looped_analysis/) and R "
            "(looped_results_analysis_300/) output directories — same CSV schema."
        )
    )
    parser.add_argument(
        "--analysis-dir",
        default=None,
        metavar="PATH",
        help=(
            "Directory containing analysis_summary.csv. "
            "Defaults to <script dir>/looped_analysis."
        ),
    )
    parser.add_argument(
        "--instrument",
        default=None,
        help="Instrument or pair name to attach to this model and its accuracy metrics.",
    )
    parser.add_argument(
        "--asset-type",
        default=None,
        help="Asset category such as forex, futures, stock, or crypto.",
    )
    parser.add_argument(
        "--timeframe",
        default=None,
        help=(
            "Simulation timeframe to filter on (1d / 5d / 1m / 3m / 1y / 5y / ytd). "
            "When set, only rows whose 'timeframe' column matches are used for training."
        ),
    )
    parser.add_argument("--label", default=None, choices=["return", "forecast", "direction_match", "ensemble"],
                        help="Target definition (default: config.toml [ml] label_mode).")
    parser.add_argument("--tune-rf", action="store_true", help="Tune the random forest with rf_tuning.toml + purged CV.")
    parser.add_argument("--no-compare", action="store_true", help="Skip the multi-model comparison.")
    parser.add_argument("--online", action="store_true",
                        help="With --keep-history: add trees from the new batch to the saved forest instead of retraining.")
    parser.add_argument("--daily-tip", action="store_true", help="Print today's improvement tips and exit.")
    parser.add_argument("--send", action="store_true", help="With --daily-tip: deliver via config.toml [daily_tip].")
    parser.add_argument("--daemon", action="store_true", help="With --daily-tip: keep running, one tip per day at daily_tip.time.")
    parser.add_argument("--dry-run", action="store_true", help="With --send/--daemon: print instead of delivering.")
    parser.add_argument("--html-report", action="store_true", help="Write summary_report.html from existing outputs and exit.")
    parser.add_argument("--scheduler-help", action="store_true", help="Show cron / Task Scheduler lines for the daily tip.")
    history_group = parser.add_mutually_exclusive_group()
    history_group.add_argument(
        "--keep-history",
        action="store_true",
        help="Accumulate analysis rows in the selected asset's history CSV.",
    )
    history_group.add_argument(
        "--clear-history",
        action="store_true",
        help="Delete accumulated analysis history and use only the current summary.",
    )
    args = parser.parse_args()

    if args.analysis_dir:
        analysis_dir = Path(args.analysis_dir)
    else:
        analysis_dir = Path(__file__).resolve().parent / "looped_analysis"

    if args.scheduler_help:
        print(SCHEDULER_HELP)
        return
    if args.daily_tip:
        run_daily_tip(analysis_dir, send=args.send, daemon=args.daemon, dry_run=args.dry_run)
        return
    if args.html_report:
        print(f"[ML] HTML report -> {write_html_report(analysis_dir)}")
        return

    keep_history = True if args.keep_history else False if args.clear_history else None
    try:
        run_ml_pipeline(
            analysis_dir,
            keep_history=keep_history,
            instrument=args.instrument,
            asset_type=args.asset_type,
            timeframe=args.timeframe,
            label_mode=args.label,
            tune_rf=True if args.tune_rf else None,
            compare=not args.no_compare,
            online=True if args.online else None,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ML] Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()