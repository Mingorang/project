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

    if source_hash in processed_sources and history_path.exists():
        print("[ML] This analysis file is already in history; skipping duplicate rows.")
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
        df["rsi_overbought_flag"] = (df["rsi14_last"] >= 70).astype(float)
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
) -> None:
    """
    Full pipeline: load → split → preprocess → train → evaluate → save.

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
    df = _prepare_training_data(df, summary_csv, ml_dir, keep_history)

    # Build the additional 13 features from columns already produced by the
    # analysis pipeline. This keeps ml_model.py self-contained and means the
    # model can be run against existing analysis_summary.csv files.
    df = _add_engineered_features(df)

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
    available_features = _select_features(df, FEATURES, TRAIN_FRAC)
    if not available_features:
        raise ValueError("No usable features remain after NaN filtering.")

    X = df[available_features]
    y = (df[TARGET_COL] > 0).astype(int).values

    # ── Block split ───────────────────────────────────────────────────────────
    # Block (not random) because a single rand.seed() before price_modeling.py's
    # run loop means consecutive runs share one RNG chain; shuffling would allow
    # RNG-correlated information to leak between train and test sets.
    train_end = int(n_runs * TRAIN_FRAC)
    val_end   = int(n_runs * (TRAIN_FRAC + VAL_FRAC))

    X_train, y_train = X.iloc[:train_end].copy(),       y[:train_end]
    X_val,   y_val   = X.iloc[train_end:val_end].copy(), y[train_end:val_end]
    X_test,  y_test  = X.iloc[val_end:].copy(),          y[val_end:]

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
        n_estimators    = 500,
        max_features    = "sqrt",       # sqrt(n_features) per split — Breiman's recommendation
        min_samples_leaf= min_leaf,
        class_weight    = "balanced",   # guard against class imbalance at small N
        random_state    = 42,
        n_jobs          = -1,           # use all cores
        oob_score       = True,         # free val estimate from bootstrap
    )

    # The untrained Pipeline — nothing learned yet until .fit() is called below.
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # handles NaN (e.g. futures mode drawdown)
        ("scale",  StandardScaler()),                    # zero-mean, unit-variance per feature
        ("rf",     RandomForestClassifier(**rf_params)), # ← the untrained model
    ])

    # ── Train ─────────────────────────────────────────────────────────────────
    print(
        f"[ML][{asset_label}] Training RandomForestClassifier "
        f"(n_estimators=500, min_samples_leaf={min_leaf})..."
    )
    pipeline.fit(X_train, y_train)

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
    }]).to_csv(accuracy_path, index=False)
    print(f"[ML][{asset_label}] Accuracy metrics saved -> {accuracy_path}")

    model_path = ml_dir / "model.pkl"
    joblib.dump(pipeline, model_path)
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
    df: pd.DataFrame, candidates: list[str], train_frac: float
) -> list[str]:
    """
    Return the subset of candidates that are present in df and have fewer than
    NAN_DROP_PCT NaN values in the training partition.
    Prints a warning for every excluded column.
    """
    n_train = int(len(df) * train_frac)
    kept: list[str] = []
    for col in candidates:
        if col not in df.columns:
            print(f"[ML] Skipped (missing)  : {col}")
            continue
        nan_frac = df[col].iloc[:n_train].isna().mean()
        if nan_frac >= NAN_DROP_PCT:
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

    keep_history = True if args.keep_history else False if args.clear_history else None
    try:
        run_ml_pipeline(
            analysis_dir,
            keep_history=keep_history,
            instrument=args.instrument,
            asset_type=args.asset_type,
            timeframe=args.timeframe,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ML] Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
