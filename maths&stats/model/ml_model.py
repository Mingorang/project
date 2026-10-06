"""
ml_model.py — Random Forest classifier on price_modeling.py analysis output.

Reads   : <analysis_dir>/analysis_summary.csv
           (produced by market_analysis.py OR analyze_looped_results.R — same schema)
Writes  : <analysis_dir>/ml_results/<asset_type>_<instrument>/
               analysis_history.csv      accumulated input data when history is enabled
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

    # Give every accumulated row a unique run ID, even though each simulation
    # invocation starts its own run numbering at 1.
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
    """
    analysis_dir = Path(analysis_dir)
    summary_csv  = analysis_dir / "analysis_summary.csv"

    if not summary_csv.exists():
        raise FileNotFoundError(
            f"analysis_summary.csv not found in {analysis_dir}.\n"
            "Run market_analysis.py or analyze_looped_results.R first."
        )

    df = pd.read_csv(summary_csv)
    instrument = _resolve_asset_value(
        df, "instrument", instrument, "Unspecified instrument"
    )
    asset_type = _resolve_asset_value(df, "asset_type", asset_type, "other").lower()
    df["instrument"] = instrument
    df["asset_type"] = asset_type

    # Keep each instrument's archive and trained model in its own folder.
    ml_dir = analysis_dir / "ml_results" / _asset_output_name(asset_type, instrument)
    ml_dir.mkdir(parents=True, exist_ok=True)

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
        summary_path, analysis_dir, asset_type, instrument, n_runs, available_features,
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
    n_runs: int, features: list[str],
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
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ML] Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
