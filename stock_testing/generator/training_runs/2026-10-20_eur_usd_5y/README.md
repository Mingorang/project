# Training run, 2026-10-20: EUR/USD, 5y weekly candles (simulated)

Everything here was produced by your own `price_modeling.py`, `analyze_looped_results.R` and `ml_model.py`, unmodified, with `xgboost 3.4.1` and `lightgbm 4.7.0` installed so all 8 models in `config.toml` ran. The big batch was generated in a scratch copy of the code, so your `looped_results/` (the 50 CSV + PNG pairs) was not cleared.

## What was trained

| Run | Rows | Label | Folder |
|---|---|---|---|
| Your pipeline, retrained in place | 50 | `return` (your `config.toml`) | `looped_analysis/ml_results/forex_eur_usd/5y/` (updated, not copied here) |
| Big batch, your label | 2,000 (1,992 after pruning 8 outliers) | `total_return_pct > 0` | `return_label_2000runs/` |
| Big batch, leak-free label | 2,000 (1,993 after pruning 7) | `future_return_pct > 0` (features from the first 70% of bars, label from the last 30%) | `forecast_label_2000runs/` |
| R leg on your 50 CSV + PNG pairs | 50 | `future_return_pct > 0` | `r_leg_50_csv_png_pairs/` |

The 2,000-run batch is CSV + metrics only (output mode 6). Its run-level table is `analysis_summary_2000runs.csv`. `generator_answers_2000runs.txt` holds the answers fed to the prompts, one per line: asset type `forex`, instrument `EUR/USD`, starting price option `2` (current price), Yahoo symbol (Enter = default), timeframe `5y`, price behaviour `2` (floor at zero), backtesting `n`, candle interval (Enter = default, weekly), `m` (many plots), `2000`, output mode `6`. Seed `1020`.

## Results

Test split = last 15% of rows (299 rows). With 299 test rows the smallest accuracy distinguishable from a coin flip is about **0.58**.

| Model | `return` label: CV / test | `forecast` label: CV / test | `forecast` p vs chance |
|---|---|---|---|
| logistic | 0.936 / 0.950 | 0.520 / 0.488 | 0.729 |
| logistic_tanh | 0.930 / 0.940 | 0.518 / 0.512 | 0.729 |
| mlp_tanh | 0.929 / 0.953 | 0.496 / 0.482 | 0.563 |
| hist_gradient_boosting | 0.925 / 0.940 | 0.508 / 0.545 | 0.133 |
| xgboost | 0.925 / 0.936 | 0.505 / 0.495 | 0.908 |
| lightgbm | 0.923 / 0.933 | 0.501 / 0.495 | 0.908 |
| extra_trees | 0.922 / 0.920 | 0.505 / 0.438 | 0.037 |
| random_forest | 0.918 / 0.926 | 0.505 / 0.435 | 0.028 |

- **`return` label: 92–95% is mechanical.** `leakage_audit.csv` flags 10 features that each separate the label on their own (`bullish_candle_pct` and `candle_direction_balance` at AUC 0.95). The features and the label are measured on the same finished path.
- **`forecast` label: every model is at chance (CV 0.496–0.520, test 0.435–0.545).** That is the correct answer: the prices are random walks, so there is nothing to learn. The two test scores below 0.5 (p = 0.028 and 0.037) are what luck produces when 8 models are compared; they are not a real signal.
- **Overfitting is visible.** The random forest scores 0.918 on its own training rows but 0.500 out-of-bag.
- **The 50-row runs mean nothing.** Their test split is 8 rows (the smallest detectable accuracy is about 1.0), so a "100% test accuracy" or a "62.5%" there is noise. On those tiny sets `lightgbm` and `hist_gradient_boosting` give identical numbers, most likely because both default to at least 20 rows per leaf, so neither can split 35 training rows.

## About the PNGs

The PNGs are charts for you, not inputs to any model. `ml_model.py` never reads `image_N.png`; it only embeds its own charts in `summary_report.html`. `analyze_looped_results.R` reads the PNGs (package `png`) only to build the contact sheets (`pngs/`). Every number the models use comes from the CSVs. `market_analysis.py` raises an error on an unpaired CSV or PNG when source charts are on, and the R script analyses only the matched pairs (warning about the rest), which is why the pairs travel together. The R and Python analyses agree on `total_return_pct` to 5e-14.

One cosmetic issue in the R charts: for 260 weekly bars the price panel's y-axis is forced down to 0 (by `geom_hline(yintercept = 0)`), which flattens EUR/USD around 1.1, and the x-axis labels overlap. See `pngs/example_R_analysis_chart_image_1.png`.

## Reproduce

Run from the `generator` folder. `price_modeling.py` clears `looped_results/` and the CSV/PNG files in `looped_analysis/` before it starts, so move anything you want to keep first.

```powershell
python price_modeling.py --seed 1020 < training_runs\2026-10-20_eur_usd_5y\generator_answers_2000runs.txt
python ml_model.py --label return   --clear-history
python ml_model.py --label forecast --clear-history
```

R leg (the script will not create its output folder, and `Rscript` is not on your PATH, so use the full path):

```powershell
mkdir looped_results_analysis_300
& "C:\Users\Admin\Downloads\app downloads\r\R-4.6.1\bin\Rscript.exe" analyze_looped_results.R
python ml_model.py --analysis-dir looped_results_analysis_300 --instrument eur/usd --asset-type forex --timeframe 5y --label forecast --clear-history
```
