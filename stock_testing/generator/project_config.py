"""project_config.py - the single source of truth for every tunable and the range it may take.

config.toml overrides any default below. load_config() merges the file over DEFAULT_CONFIG and then
validates the result against SCHEMA, so a value the code was never built for (a volatility of 1e-100,
a probability of 5, a window of 0 candles, a string where a number belongs ...) stops the program
with one readable list of problems instead of failing somewhere deep inside a run.

Used by price_modeling.py, backtest_compare.py (re-exports load_config), market_analysis.py,
ml_model.py, live_market.py and routine/scan.py. analyze_looped_results.R reads the [indicators]
section of the same file.
"""
from __future__ import annotations

import copy
import math
import re
import tomllib
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "config.toml"
CONFIG_PATH = Path(__file__).with_name(CONFIG_FILENAME)

MODEL_NAMES = ("random_forest", "extra_trees", "logistic", "logistic_tanh", "mlp_tanh",
               "gradient_boosting", "hist_gradient_boosting", "xgboost", "lightgbm")
LIVE_PERIODS = ("1d", "5d", "1m", "3m", "6m", "ytd", "1y", "2y", "5y", "10y", "max")
SEARCH_CRITERIA = ("path", "bars", "price", "exact", "streak_ks", "kl", "acf", "drawdown", "info_ratio")

# ─── Defaults (every key config.toml may set) ─────────────────────────────────
DEFAULT_CONFIG: dict[str, Any] = {
    "tutorial": {"mode": "off"},                                  # off | beginner | advanced
    "storage": {
        "auto_create_folders": True,
        "archive_after_run": False, "archive_dir": "",
        "delete_loose_after_archive": False, "keep_png": True,
    },
    "indicators": {                                               # every chart, analysis and ML feature
        "fast_sma": 4, "slow_sma": 14, "ci_level": 0.975, "target_r": 1.4,
        "bollinger_window": 20, "bollinger_sd": 2.0,
        "rsi_period": 14, "rsi_overbought": 70.0, "rsi_oversold": 30.0,
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "atr_period": 14,
    },
    "session": {                                                  # simulated trading session (price_modeling)
        "start_hour": 9, "minutes": 540,
        "scale_period_change_to_daily": True, "backtest_point_in_time_change": True,
        "max_candles_comfy": 100,
    },
    "volatility": {                                               # where each run's starting vol_index comes from
        "source": "realized",                                     # realized | formula
        "daily_lookback": 20,                                     # fallback: daily closes before the window
    },
    "limits": {                                                   # hard bounds for prompts and learned values
        "vol_index_min": 0.001, "vol_index_max": 200.0,
        "start_price_min": 1e-6, "start_price_max": 1e9,
        "max_runs": 200_000, "max_cycles": 500, "max_loops": 50,
    },
    "live": {                                                     # option 3: current market data
        "periods": list(LIVE_PERIODS),
        "candles_min": 80, "candles_max": 140, "candles_target": 110,
        "forward_runs_per_period": 9,
        "real_weight": 3.0, "sim_weight": 1.0,
        "chart_simulated_runs": False,
        "context_symbols": ["S&P 500=^GSPC", "VIX=^VIX", "DXY=DX-Y.NYB"],
        "snapshot_years": 5,
        "default_save_state": 1,
    },
    "performance": {                                              # big runs: charts on several CPU cores
        "chart_workers": 0,                                       # 0 = all cores but one; 1 = no worker processes
        "parallel_min_charts": 24,                                # fewer charts than this: draw them in-process
    },
    "scan": {                                                     # routine/scan.py and the live snapshot
        "vol_pctile_shift": 10, "big_move_sigma": 2.0, "stale_bars": 4, "flat_run": 4,
        "min_history_bars": 250,
    },
    "scoring": {                                                  # backtest accuracy index (backtest_compare)
        "w_path": 0.40, "w_direction": 0.25, "w_net_move": 0.20, "w_range": 0.15,
        "path_tolerance": 0.5, "min_coverage": 0.5, "min_bars_absolute": 8,
    },
    "pruning": {
        "enabled": True, "pct_difference": 150.0, "min_rows_kept": 10,
        "columns": ["realized_vol_per_bar_pct", "high_low_range_pct_of_low", "max_drawdown_pct_positive_prices_only"],
    },
    "generator": {
        "wick_factor": 3.5 / 15, "wick_bull": 1.0, "wick_bear": 1.0,
        "jump_prob": 0.0, "jump_size": 2.0, "ar1": 0.0,
        "gap_prob": 0.0, "gap_sigma_pct": 0.05,
        "dist": "uniform", "skew_alpha": 0.0, "quantile_level": 0.5,
        "profile_enabled": True, "volume_enabled": True,
        "volume_base": 1_000_000.0, "volume_beta": 1.0, "volume_noise": 0.35,
        # after a run of n same-direction bars the chance of one more is c1 * decay**((n-1)//every);
        # decay = 1 - 0.04*e; c1/every are calibrated so that P(10 bars the same way) ~ target_p10.
        "streak_enabled": True, "streak_decay_factor": 0.04, "streak_target_p10": 0.032,
        "streak_max_c1": 0.95,
    },
    "volatility_search": {
        "enabled": True, "method": "auto",
        "criteria": ["path", "bars", "price"],
        "target": 0.30, "step_abs": 0.05, "step_pct": 10.0, "step_mode": "pct",
        "max_iters": 40, "min_step_pct": 0.25, "mc_paths": 6, "validation_paths": 12,
        "convergence_eps": 0.002, "convergence_patience": 3,
        "coarse_step_abs": 0.5, "fine_step_abs": 0.05, "max_grid_points": 120,
        "range_lo_mult": 0.1, "range_hi_mult": 5.0,
        "w_path": 0.30, "w_direction": 0.15, "w_net": 0.20, "w_range": 0.35,
        "bars_min": 0.55, "price_min": 0.98, "exact_pass": 0.30,
        "path_error_lo": 0.02, "path_error_hi": 0.60, "path_band": False,
        "ks_p_min": 0.05, "kl_max": 0.50, "acf_ssd_max": 1.0, "info_ratio_max": 0.30,
        "drawdown_tolerance_pct": 5.0, "winrate_tolerance": 0.25, "body_range_tolerance": 0.15,
        "realised_vol_tolerance": 0.20, "accuracy_decay_threshold": 0.10,
        "recal_days": 30, "recal_iters": 5, "recal_every_runs": 100, "auto_recalibrate": False,
        "consensus_runs": 3, "bootstrap_resamples": 200, "ci_level": 0.90,
        "sprt_enabled": True, "sprt_alpha": 0.05, "sprt_beta": 0.05,
        "tune_shape": True, "ga_population": 12, "ga_generations": 5,
        "bayes_init": 5, "bayes_evals": 20, "cobyla_maxiter": 40,
        "holdout_frac": 0.25, "global_cycle_cap": 30, "loops": 1, "n_aug": 0, "aug_weight": 2.0, "ensemble_k": 1,
        "decay_lambda_grid": [0.0, 0.005, 0.01, 0.02, 0.05], "seed_range": 10_000_000,
        "per_cycle": True, "use_regimes": True, "use_history_seed": True, "use_transfer": True,
        "rolling_window": 10, "weak_segment_threshold": 0.40, "profile_learning": True,
    },
    "objective": {
        "kl": 0.0, "acf": 0.0, "drawdown": 0.0, "wick": 0.0,
        "body_range": 0.0, "winrate": 0.0, "info_ratio": 0.0, "streak_ks": 0.0,
    },
    "exact": {"alpha": 0.40, "beta": 0.25, "gamma": 0.20, "delta": 0.15},
    "ml": {
        "label_mode": "return",                                   # return | forecast | direction_match | ensemble
        "forecast_split": 0.70,
        "cv_folds": 5, "embargo_pct": 0.02, "use_sample_weight": True,
        "models": ["random_forest", "extra_trees", "logistic", "logistic_tanh", "mlp_tanh",
                   "hist_gradient_boosting", "xgboost", "lightgbm"],
        "conformal_alpha": 0.10, "adversarial_auc_flag": 0.55, "online_add_trees": 100, "online_update": False,
        "tune_rf": False, "tuning_config": "rf_tuning.toml",
        "train_frac": 0.70, "val_frac": 0.15, "nan_drop_pct": 0.80, "n_estimators": 500,
        "keep_history": "ask",                                    # ask | yes | no
    },
    "daily_tip": {
        "enabled": False, "time": "08:30", "channel": "print",    # print | ntfy | smtp
        "ntfy_topic": "", "ntfy_server": "https://ntfy.sh",
        "smtp_host": "", "smtp_port": 587, "smtp_user": "", "smtp_to": "",
        "smtp_password_env": "ML_TIP_SMTP_PASSWORD",
    },
}


# ─── Schema: what each key may hold ───────────────────────────────────────────
def _num(lo: float, hi: float, *, integer: bool = False, open_lo: bool = False, open_hi: bool = False) -> dict:
    return {"kind": "int" if integer else "float", "lo": lo, "hi": hi, "open_lo": open_lo, "open_hi": open_hi}


def F(lo: float, hi: float, **kw: bool) -> dict:
    return _num(lo, hi, **kw)


def I(lo: int, hi: int) -> dict:
    return _num(lo, hi, integer=True)


BOOL = {"kind": "bool"}


def CHOICE(*choices: str) -> dict:
    return {"kind": "choice", "choices": choices}


def TEXT(pattern: str | None = None, max_len: int = 300) -> dict:
    return {"kind": "str", "pattern": pattern, "max_len": max_len}


def TEXTS(allowed: tuple[str, ...] | None = None, pattern: str | None = None, min_items: int = 0,
          max_items: int = 64) -> dict:
    return {"kind": "list_str", "allowed": allowed, "pattern": pattern, "min_items": min_items, "max_items": max_items}


def NUMS(lo: float, hi: float, min_items: int = 1, max_items: int = 64) -> dict:
    return {"kind": "list_float", "lo": lo, "hi": hi, "min_items": min_items, "max_items": max_items}


_COLUMN = r"^[a-z0-9_]{1,64}$"
_SYMBOL_ENTRY = r"^[^=\n]{1,32}=[A-Za-z0-9^=.\-_]{1,24}$"      # "Display name=YAHOO_SYMBOL"

SCHEMA: dict[str, dict[str, dict]] = {
    "tutorial": {"mode": CHOICE("off", "beginner", "advanced")},
    "storage": {
        "auto_create_folders": BOOL, "archive_after_run": BOOL,
        "archive_dir": TEXT(r"^[^\x00-\x1f\"*?<>|]*$", 260),
        "delete_loose_after_archive": BOOL, "keep_png": BOOL,
    },
    "indicators": {
        "fast_sma": I(2, 200), "slow_sma": I(3, 400), "ci_level": F(0.5, 0.9999),
        "target_r": F(0.1, 20.0), "bollinger_window": I(2, 400), "bollinger_sd": F(0.1, 6.0),
        "rsi_period": I(2, 200), "rsi_overbought": F(50.0, 99.0, open_lo=True), "rsi_oversold": F(1.0, 50.0, open_hi=True),
        "macd_fast": I(2, 200), "macd_slow": I(3, 400), "macd_signal": I(2, 200), "atr_period": I(2, 200),
    },
    "session": {
        "start_hour": I(0, 23), "minutes": I(60, 1440),
        "scale_period_change_to_daily": BOOL, "backtest_point_in_time_change": BOOL,
        "max_candles_comfy": I(20, 5000),
    },
    "volatility": {"source": CHOICE("realized", "formula"), "daily_lookback": I(5, 260)},
    "limits": {
        "vol_index_min": F(1e-6, 1.0), "vol_index_max": F(1.0, 249.0),
        "start_price_min": F(1e-12, 1.0, open_lo=False), "start_price_max": F(1.0, 1e15),
        "max_runs": I(1, 10_000_000), "max_cycles": I(1, 100_000), "max_loops": I(1, 1000),
    },
    "performance": {"chart_workers": I(0, 61), "parallel_min_charts": I(2, 1_000_000)},
    "live": {
        "periods": TEXTS(LIVE_PERIODS, min_items=1, max_items=len(LIVE_PERIODS)),
        "candles_min": I(20, 2000), "candles_max": I(30, 5000), "candles_target": I(20, 5000),
        "forward_runs_per_period": I(0, 1000),
        "real_weight": F(0.0, 1000.0, open_lo=True), "sim_weight": F(0.0, 1000.0, open_lo=True),
        "chart_simulated_runs": BOOL,
        "context_symbols": TEXTS(pattern=_SYMBOL_ENTRY, max_items=12),
        "snapshot_years": I(2, 30),
        "default_save_state": I(1, 6),
    },
    "scan": {
        "vol_pctile_shift": I(1, 100), "big_move_sigma": F(0.1, 50.0), "stale_bars": I(1, 60),
        "flat_run": I(2, 500), "min_history_bars": I(30, 5000),
    },
    "scoring": {
        "w_path": F(0.0, 1.0), "w_direction": F(0.0, 1.0), "w_net_move": F(0.0, 1.0), "w_range": F(0.0, 1.0),
        "path_tolerance": F(0.0, 10.0, open_lo=True), "min_coverage": F(0.0, 1.0, open_lo=True),
        "min_bars_absolute": I(2, 10_000),
    },
    "pruning": {
        "enabled": BOOL, "pct_difference": F(1.0, 1e6), "min_rows_kept": I(2, 10_000_000),
        "columns": TEXTS(pattern=_COLUMN, max_items=100),
    },
    "generator": {
        "wick_factor": F(0.0, 5.0), "wick_bull": F(0.0, 10.0), "wick_bear": F(0.0, 10.0),
        "jump_prob": F(0.0, 1.0), "jump_size": F(0.0, 100.0), "ar1": F(-0.99, 0.99),
        "gap_prob": F(0.0, 1.0), "gap_sigma_pct": F(0.0, 50.0),
        "dist": CHOICE("uniform", "skewnorm", "empirical", "gbm_quantile"),
        "skew_alpha": F(-50.0, 50.0), "quantile_level": F(0.01, 0.99),
        "profile_enabled": BOOL, "volume_enabled": BOOL,
        "volume_base": F(1.0, 1e15), "volume_beta": F(0.0, 100.0), "volume_noise": F(0.0, 5.0),
        "streak_enabled": BOOL, "streak_decay_factor": F(0.0, 1 / math.e, open_hi=True),
        "streak_target_p10": F(0.0, 1.0, open_lo=True, open_hi=True), "streak_max_c1": F(0.30, 0.999, open_lo=True),
    },
    "volatility_search": {
        "enabled": BOOL,
        "method": CHOICE("auto", "step", "bisect", "grid", "coarse_fine", "cobyla", "bayes", "ga", "mc"),
        "criteria": TEXTS(SEARCH_CRITERIA, min_items=1),
        "target": F(-1.0, 1.0), "step_abs": F(0.0, 100.0, open_lo=True), "step_pct": F(0.0, 100.0, open_lo=True),
        "step_mode": CHOICE("pct", "abs"), "max_iters": I(1, 10_000), "min_step_pct": F(0.0, 100.0, open_lo=True),
        "mc_paths": I(1, 1000), "validation_paths": I(1, 10_000),
        "convergence_eps": F(0.0, 1.0, open_lo=True), "convergence_patience": I(1, 100),
        "coarse_step_abs": F(0.0, 100.0, open_lo=True), "fine_step_abs": F(0.0, 100.0, open_lo=True),
        "max_grid_points": I(3, 100_000),
        "range_lo_mult": F(0.0, 1000.0, open_lo=True), "range_hi_mult": F(0.0, 10_000.0, open_lo=True),
        "w_path": F(0.0, 10.0), "w_direction": F(0.0, 10.0), "w_net": F(0.0, 10.0), "w_range": F(0.0, 10.0),
        "bars_min": F(0.0, 1.0), "price_min": F(-10.0, 1.0), "exact_pass": F(-1.0, 1.0),
        "path_error_lo": F(0.0, 10.0), "path_error_hi": F(0.0, 10.0), "path_band": BOOL,
        "ks_p_min": F(0.0, 1.0), "kl_max": F(0.0, 100.0), "acf_ssd_max": F(0.0, 100.0),
        "info_ratio_max": F(0.0, 100.0), "drawdown_tolerance_pct": F(0.0, 100.0),
        "winrate_tolerance": F(0.0, 1.0), "body_range_tolerance": F(0.0, 1.0),
        "realised_vol_tolerance": F(0.0, 10.0, open_lo=True), "accuracy_decay_threshold": F(0.0, 2.0),
        "recal_days": F(0.0, 3650.0), "recal_iters": I(0, 1000), "recal_every_runs": I(1, 10_000_000),
        "auto_recalibrate": BOOL, "consensus_runs": I(1, 100), "bootstrap_resamples": I(0, 100_000),
        "ci_level": F(0.0, 1.0, open_lo=True, open_hi=True), "sprt_enabled": BOOL,
        "sprt_alpha": F(0.0, 0.5, open_lo=True), "sprt_beta": F(0.0, 0.5, open_lo=True),
        "tune_shape": BOOL, "ga_population": I(2, 1000), "ga_generations": I(1, 1000),
        "bayes_init": I(1, 1000), "bayes_evals": I(1, 10_000), "cobyla_maxiter": I(1, 10_000),
        "holdout_frac": F(0.0, 0.9), "global_cycle_cap": I(2, 100_000), "loops": I(1, 1000),
        "n_aug": I(0, 100_000), "aug_weight": F(0.0, 1000.0, open_lo=True), "ensemble_k": I(1, 1000),
        "decay_lambda_grid": NUMS(0.0, 10.0), "seed_range": I(1, 2**31 - 1),
        "per_cycle": BOOL, "use_regimes": BOOL, "use_history_seed": BOOL, "use_transfer": BOOL,
        "rolling_window": I(2, 10_000), "weak_segment_threshold": F(0.0, 1.0), "profile_learning": BOOL,
    },
    "objective": {key: F(0.0, 100.0) for key in DEFAULT_CONFIG["objective"]},
    "exact": {key: F(0.0, 10.0) for key in DEFAULT_CONFIG["exact"]},
    "ml": {
        "label_mode": CHOICE("return", "forecast", "direction_match", "ensemble"),
        "forecast_split": F(0.3, 0.95), "cv_folds": I(2, 50), "embargo_pct": F(0.0, 0.5),
        "use_sample_weight": BOOL, "models": TEXTS(MODEL_NAMES, min_items=1),
        "conformal_alpha": F(0.0, 0.5, open_lo=True), "adversarial_auc_flag": F(0.5, 1.0),
        "online_add_trees": I(1, 100_000), "online_update": BOOL, "tune_rf": BOOL,
        "tuning_config": TEXT(r"^[\w .\-/\\:]{1,260}$"),
        "train_frac": F(0.3, 0.9), "val_frac": F(0.0, 0.5), "nan_drop_pct": F(0.0, 1.0, open_lo=True),
        "n_estimators": I(10, 20_000), "keep_history": CHOICE("ask", "yes", "no"),
    },
    "daily_tip": {
        "enabled": BOOL, "time": TEXT(r"^([01]\d|2[0-3]):[0-5]\d$", 5), "channel": CHOICE("print", "ntfy", "smtp"),
        "ntfy_topic": TEXT(r"^[A-Za-z0-9_\-]{0,64}$", 64), "ntfy_server": TEXT(r"^https?://[^\s]{1,200}$", 210),
        "smtp_host": TEXT(r"^[A-Za-z0-9.\-]{0,253}$", 253), "smtp_port": I(1, 65_535),
        "smtp_user": TEXT(r"^[^\s]{0,254}$", 254), "smtp_to": TEXT(r"^[^\s]{0,254}$", 254),
        "smtp_password_env": TEXT(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$", 64),
    },
}


class ConfigError(ValueError):
    """config.toml holds a value the code was not built to handle."""

    def __init__(self, path: Path | str, problems: list[str]):
        self.path, self.problems = Path(path), list(problems)
        lines = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"{Path(path).name} has {len(self.problems)} invalid setting(s):\n{lines}")


def _describe(rule: dict) -> str:
    kind = rule["kind"]
    if kind in ("int", "float"):
        lo = "(" if rule["open_lo"] else "["
        hi = ")" if rule["open_hi"] else "]"
        word = "a whole number" if kind == "int" else "a number"
        return f"{word} in {lo}{rule['lo']:g}, {rule['hi']:g}{hi}"
    if kind == "bool":
        return "true or false"
    if kind == "choice":
        return "one of " + " | ".join(rule["choices"])
    if kind == "str":
        return "text" + (f" matching {rule['pattern']}" if rule.get("pattern") else "")
    if kind == "list_str":
        allowed = rule.get("allowed")
        return "a list of " + (("values from " + ", ".join(allowed)) if allowed else "text values")
    return f"a list of numbers in [{rule['lo']:g}, {rule['hi']:g}]"


def _check_number(value: Any, rule: dict) -> bool:
    # bool is a subclass of int in Python: `true` must never pass as 1.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if rule["kind"] == "int" and not isinstance(value, int):
        return False
    if not math.isfinite(float(value)):
        return False
    lo_ok = value > rule["lo"] if rule["open_lo"] else value >= rule["lo"]
    hi_ok = value < rule["hi"] if rule["open_hi"] else value <= rule["hi"]
    return lo_ok and hi_ok


def _check(value: Any, rule: dict) -> bool:
    kind = rule["kind"]
    if kind in ("int", "float"):
        return _check_number(value, rule)
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "choice":
        return isinstance(value, str) and value in rule["choices"]
    if kind == "str":
        return (isinstance(value, str) and len(value) <= rule["max_len"]
                and (rule["pattern"] is None or re.fullmatch(rule["pattern"], value) is not None))
    if kind == "list_str":
        if not isinstance(value, list) or not (rule["min_items"] <= len(value) <= rule["max_items"]):
            return False
        if len(set(map(str, value))) != len(value):
            return False
        for item in value:
            if not isinstance(item, str):
                return False
            if rule["allowed"] is not None and item not in rule["allowed"]:
                return False
            if rule["pattern"] is not None and re.fullmatch(rule["pattern"], item) is None:
                return False
        return True
    if kind == "list_float":
        return (isinstance(value, list) and rule["min_items"] <= len(value) <= rule["max_items"]
                and all(_check_number(v, {**rule, "kind": "float", "open_lo": False, "open_hi": False}) for v in value))
    return False


def _cross_checks(cfg: dict[str, Any], bad: set[tuple[str, str]]) -> list[str]:
    """Rules that involve more than one key. A rule is skipped when one of its keys is already invalid,
    so every problem is reported in one go without comparing a number to, say, a string."""
    out: list[str] = []

    def rule(keys: str, holds, message) -> None:
        pairs = [tuple(k.split(".")) for k in keys.split()]
        if any(pair in bad for pair in pairs):
            return
        if not holds():
            out.append(message())

    ind, live, lim, session = cfg["indicators"], cfg["live"], cfg["limits"], cfg["session"]
    vs, ml, score = cfg["volatility_search"], cfg["ml"], cfg["scoring"]
    rule("indicators.fast_sma indicators.slow_sma", lambda: ind["fast_sma"] < ind["slow_sma"],
         lambda: f"indicators.fast_sma ({ind['fast_sma']}) must be smaller than indicators.slow_sma ({ind['slow_sma']})")
    rule("indicators.macd_fast indicators.macd_slow", lambda: ind["macd_fast"] < ind["macd_slow"],
         lambda: f"indicators.macd_fast ({ind['macd_fast']}) must be smaller than indicators.macd_slow ({ind['macd_slow']})")
    rule("indicators.rsi_oversold indicators.rsi_overbought", lambda: ind["rsi_oversold"] < ind["rsi_overbought"],
         lambda: "indicators.rsi_oversold must be below indicators.rsi_overbought")
    rule("live.candles_min live.candles_target live.candles_max",
         lambda: live["candles_min"] <= live["candles_target"] <= live["candles_max"],
         lambda: f"live: candles_min ({live['candles_min']}) <= candles_target ({live['candles_target']}) "
                 f"<= candles_max ({live['candles_max']}) must hold")
    rule("live.candles_min indicators.slow_sma", lambda: live["candles_min"] >= ind["slow_sma"] + 2,
         lambda: f"live.candles_min ({live['candles_min']}) must be at least indicators.slow_sma + 2 "
                 f"({ind['slow_sma'] + 2}) so every chart can be analysed")
    rule("live.real_weight live.sim_weight", lambda: live["real_weight"] > live["sim_weight"],
         lambda: f"live.real_weight ({live['real_weight']:g}) must be larger than live.sim_weight "
                 f"({live['sim_weight']:g}): real market rows outweigh simulated ones")
    rule("limits.vol_index_min limits.vol_index_max", lambda: lim["vol_index_min"] < lim["vol_index_max"],
         lambda: "limits.vol_index_min must be below limits.vol_index_max")
    rule("limits.start_price_min limits.start_price_max", lambda: lim["start_price_min"] < lim["start_price_max"],
         lambda: "limits.start_price_min must be below limits.start_price_max")
    rule("session.start_hour session.minutes", lambda: session["start_hour"] * 60 + session["minutes"] <= 1440,
         lambda: f"session: start_hour {session['start_hour']} + {session['minutes']} minutes runs past midnight "
                 f"(use start_hour = 0 and minutes = 1440 for a 24-hour market)")
    rule("volatility_search.range_lo_mult volatility_search.range_hi_mult",
         lambda: vs["range_lo_mult"] < vs["range_hi_mult"],
         lambda: "volatility_search.range_lo_mult must be below range_hi_mult")
    rule("volatility_search.path_error_lo volatility_search.path_error_hi",
         lambda: vs["path_error_lo"] < vs["path_error_hi"],
         lambda: "volatility_search.path_error_lo must be below path_error_hi")
    rule("volatility_search.fine_step_abs volatility_search.coarse_step_abs",
         lambda: vs["fine_step_abs"] <= vs["coarse_step_abs"],
         lambda: "volatility_search.fine_step_abs must not exceed coarse_step_abs")
    rule("ml.train_frac ml.val_frac", lambda: ml["train_frac"] + ml["val_frac"] <= 0.95,
         lambda: f"ml.train_frac + ml.val_frac = {ml['train_frac'] + ml['val_frac']:.2f} leaves under 5% for the test split")
    rule("scoring.w_path scoring.w_direction scoring.w_net_move scoring.w_range",
         lambda: abs(score["w_path"] + score["w_direction"] + score["w_net_move"] + score["w_range"] - 1.0) <= 1e-6,
         lambda: "scoring weights w_path + w_direction + w_net_move + w_range must sum to 1 (they sum to "
                 f"{score['w_path'] + score['w_direction'] + score['w_net_move'] + score['w_range']:g})")
    return out


def validate_config(cfg: dict[str, Any], path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Raise ConfigError listing every value outside its allowed type/range; return cfg when clean."""
    problems: list[str] = []
    bad: set[tuple[str, str]] = set()
    for section, rules in SCHEMA.items():
        values = cfg.get(section)
        if not isinstance(values, dict):
            problems.append(f"[{section}] must be a table of settings")
            bad.update((section, key) for key in rules)
            continue
        for key, rule in rules.items():
            if key not in values:
                continue
            if not _check(values[key], rule):
                bad.add((section, key))
                problems.append(f"{section}.{key} = {values[key]!r} is not allowed: expected {_describe(rule)}")
    problems.extend(_cross_checks(cfg, bad))
    if problems:
        raise ConfigError(path, problems)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, *, quiet: bool = False) -> dict[str, Any]:
    """DEFAULT_CONFIG overlaid with config.toml (next to this file unless `path` is given), validated.

    Unknown sections/keys are reported rather than silently ignored, so a typo in the toml cannot
    quietly leave a setting at its default. Invalid values raise ConfigError.
    """
    config_path = Path(path) if path else CONFIG_PATH
    if path and not config_path.exists():
        raise ConfigError(config_path, [f"file not found: {config_path}"])
    if not config_path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    try:
        with open(config_path, "rb") as handle:
            user = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(config_path, [f"not valid TOML: {error}"]) from None
    for section, values in user.items():
        if section not in DEFAULT_CONFIG:
            if not quiet:
                print(f"[config] Unknown section [{section}] in {config_path.name} (ignored).")
        elif isinstance(values, dict):
            for key in values:
                if key not in DEFAULT_CONFIG[section] and not quiet:
                    print(f"[config] Unknown key {section}.{key} in {config_path.name} (ignored).")
    return validate_config(_deep_merge(DEFAULT_CONFIG, user), config_path)


def section(name: str, path: str | Path | None = None) -> dict[str, Any]:
    """One validated section, falling back to the defaults if the file cannot be read (light callers)."""
    try:
        return load_config(path, quiet=True)[name]
    except ConfigError:
        raise
    except OSError:
        return copy.deepcopy(DEFAULT_CONFIG[name])


def clamp_vol_index(value: float, cfg: dict[str, Any]) -> tuple[float, str]:
    """(vol_index inside [limits], note). Non-finite input raises: it can never be a volatility."""
    lim = cfg["limits"]
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"vol_index {value!r} is not a finite number")
    if v < lim["vol_index_min"]:
        return float(lim["vol_index_min"]), f"vol_index {v:.3g} raised to the minimum {lim['vol_index_min']:g}"
    if v > lim["vol_index_max"]:
        return float(lim["vol_index_max"]), f"vol_index {v:.3g} capped at the maximum {lim['vol_index_max']:g}"
    return v, ""
