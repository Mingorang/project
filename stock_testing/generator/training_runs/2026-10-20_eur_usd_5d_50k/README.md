# Training run, 2026-10-20: EUR/USD, 5d timeframe, 90 bars, 50,000 runs

Your own `price_modeling.py` produced the data (50,000 runs accumulated into history) and your own
`ml_model.py` trained both models, unmodified, with all 8 models from `config.toml` available.

| Run | Label | Rows kept | RF test | Best model test |
|---|---|---|---|---|
| `return_label/` — yours, trained in your terminal | `total_return_pct > 0` | 49,674 (326 pruned) | 0.924 | logistic 0.966 |
| `forecast_label/` — leak-free retrain | `future_return_pct > 0` | 49,808 (192 pruned) | 0.493 | hist_gradient_boosting 0.533 |

The forecast run was done in a scratch copy, so your `looped_analysis/ml_results/forex_eur_usd/5d/`
still holds the `return` model. Its large files (`ohlc_candle_index.csv` is 765 MB,
`analysis_history.csv` is 57 MB) were left there and not copied here.

## Why the `return` label reaches 92-97%: a price-level leak

Every one of the 50,000 runs starts at exactly 1.1237217. Any feature denominated in price **points**
is therefore proportional to the price level during the run, and with a fixed start price the level
*is* the return (`mean_close` vs `total_return_pct`: 0.864).

| Feature | vs price level | vs return | vs \|return\| |
|---|---|---|---|
| atr14_last | 0.876 | **0.894** | 0.324 |
| avg_true_range_points | 0.957 | 0.823 | 0.275 |
| avg_candle_range_points | 0.957 | 0.823 | 0.275 |
| sd_bar_change_points | 0.926 | 0.790 | 0.231 |
| avg_candle_body_points | 0.915 | 0.788 | 0.279 |
| realized_vol_per_bar_pct (control, already a %) | -0.018 | -0.032 | -0.103 |

`atr14_last` tracks the *signed* return at 0.894 but the *size* of the move at only 0.324. That is
backwards for a volatility measure and is the signature of the leak. Expressing ATR as a percent of
`mean_close` only drops it to 0.639, because that denominator carries the return too. These columns
are not in `LEAKAGE_COLS`, so the audit flags them (`separation` 0.83-0.94 on 8 features) and the
run continues anyway.

## Why the leak-free run is at chance, including the three "significant" models

`model_comparison.csv` reports `test_p_value_vs_chance`, which compares against 0.50. The classes
are not balanced: a multiplicative random walk has a negative median, so 52.9% of the test block
ends down. Measured against the majority-class baseline instead, nothing survives.

| Model | test | vs 0.50 | vs baseline 0.5294 | p vs baseline |
|---|---|---|---|---|
| hist_gradient_boosting | 0.533 | +0.033 | +0.003 | 0.301 |
| xgboost | 0.520 | +0.020 | -0.010 | 0.951 |
| lightgbm | 0.514 | +0.014 | -0.015 | 0.995 |
| mlp_tanh | 0.509 | +0.009 | -0.020 | 1.000 |
| extra_trees | 0.508 | +0.008 | -0.022 | 1.000 |
| logistic | 0.505 | +0.005 | -0.024 | 1.000 |
| logistic_tanh | 0.500 | -0.000 | -0.030 | 1.000 |
| random_forest | 0.494 | -0.006 | -0.035 | 1.000 |

With 7,472 test rows the smallest accuracy distinguishable from that baseline is 0.546. No model
reaches it. The models are leaning bearish and collecting the class imbalance, not forecasting.

### The imbalance is volatility drag, and it is not a usable signal

Median future return -0.710% against a predicted drag of -0.5 x sigma^2 x n = -0.634%. But it does
not vary with the volatility the model can see: bucketing all 50,000 runs by the volatility of the
visible first 62 bars gives a flat 47.5-48.6% bullish with no trend, and the rule "above-median
visible volatility -> predict down" scores 0.4990 (95% CI 0.4947-0.5034). The drag sets the base
rate; it gives no per-run edge.

## Reproduce

```powershell
python ml_model.py --label forecast --clear-history
```

Forecast mode rebuilds indicators per run from `processed_bars.csv` (1.6 GB here) at about 40 ms per
run, so 50,000 runs takes roughly 33 minutes plus training — this run took 57 minutes end to end.
