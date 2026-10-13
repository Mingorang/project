# Comparing Synthetic Market Data — expanded eight-week programme

**Scope:** 56 days, 28 two-day checkpoints. Each week contains four checkpoints? No: 56 days / 2 = 28; weeks cross checkpoint boundaries. The sequence below uses 28 consecutive two-day blocks and week labels based on starting day. Four extra research items are integrated as extension tasks.

**Core principle:** only features known at forecast origin may predict subsequent observed prices. Simulation similarity is not proof of predictive skill.

## Days 01–02 | Week 1: Define the prediction contract

Choose a symbol, candle interval, decision timestamp t and horizon h. Record whether the target is next-bar direction, h-bar log return, or a future path; do not mix them. Write the dataset schema with symbol, UTC timestamp, OHLCV, source, observed/synthetic flag, run_id and seed. Keep raw inputs immutable and assign a version to every transformed dataset. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Derive log return r_t=ln(C_t/C_(t-1)); explain why summing log returns gives a multi-bar return.

## Days 03–04 | Week 1: Audit the historical feed

Import real historical candles from a licensed or documented source into a raw directory, retaining source timestamps and metadata. Check duplicate times, missing candles, OHLC inequalities, nonpositive prices, suspicious volume and session gaps. Distinguish a genuine market closure from a failed download, and produce a machine-readable quality report rather than silently filling everything. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Use set operations to compare expected and observed timestamp sets; quantify missingness.

## Days 05–06 | Week 1: Align simulated and observed schemas

Adapt the existing price_modeling.py output to the same canonical candle columns as the real feed, preserving the existing PNG/CSV run pairs. Never identify a generated candle as observed. Save calibration cutoff and random seed for each run, and separate source files from derived features. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Show that H>=max(O,C,L) and L<=min(O,C,H) are necessary candle constraints.

## Days 07–08 | Week 1: Establish reproducible splits

Make chronological train, validation and untouched test periods, with a gap/embargo of at least the target horizon when labels overlap. Store split boundaries in config and confirm that simulated paths are calibrated only from the corresponding training history. Never random-shuffle future market observations into the training set. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Express interval overlap as [t,t+h] intersect [s,s+h]; find the minimum safe separation.

## Days 09–10 | Week 2: Measure historical stylised facts

Compute returns, realized volatility, lag autocorrelation, absolute-return autocorrelation, quantiles, skew and excess kurtosis by instrument and regime. Use R to plot return histograms and volatility clustering. Record sample sizes and do not assume price levels are stationary. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Calculate E[r], Var(r), skewness and kurtosis from a finite sample; interpret excess kurtosis.

## Days 11–12 | Week 2: Create a transparent null generator

Implement a seeded independent-return baseline calibrated on training data only. Generate OHLC bars with consistent highs and lows, and explicitly label the assumptions. Compare this null model against the existing random simulation before introducing more sophisticated generators. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Prove Var(sum of n independent returns)=n sigma^2; identify when this fails.

## Days 13–14 | Week 2: Bootstrap realistic blocks

Add moving-block bootstrap of historical returns to preserve short-range dependence. Fit the block length on training/validation data, not test observations. Rebuild synthetic candles and compare distributions of returns, ranges and volatility with the source history. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Derive the sample covariance and use it to motivate dependence across adjacent returns.

## Days 15–16 | Week 3: Test simulation fidelity

Build side-by-side diagnostics for mean, standard deviation, tail quantiles, autocorrelation and maximum drawdown on synthetic versus held-out real data. Run multiple seeds and report variation; matching a histogram alone does not establish realistic market dynamics. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Compute a standardized discrepancy |mu_s-mu_r|/sigma_r and discuss zero-denominator handling.

## Days 17–18 | Week 3: Implement causal technical indicators

Build lagged RSI14, MACD(12,26,9), SMA/EMA, ATR14 and Bollinger width from candles available at prediction time. Document warm-up NaNs and verify indicators do not consume the future label window. Match R and Python definitions with tolerance tests. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Derive EMA_t=alpha x_t+(1-alpha)EMA_(t-1) and solve its recurrence.

## Days 19–20 | Week 3: Add candle and return features

Create bullish/bearish candle percentages over trailing windows, wick/body ratios, lagged log returns, rolling realized volatility and normalized ranges. Avoid using whole-run outcome statistics when predicting an earlier point within the same run. Add a feature availability timestamp to each row. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Use binomial variance p(1-p)/n to assess instability of candle proportions.

## Days 21–22 | Week 3: Add volume and regime features

When real volume exists, compute relative volume, rolling volume z-scores, price-volume correlation and volatility regime labels based only on past observations. For FX feeds with tick volume or absent volume, explicitly mark the definition or missingness rather than pretending all volumes are comparable. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Derive z=(x-mu)/sigma and investigate sensitivity to outliers.

## Days 23–24 | Week 4: Create a feature registry

Maintain 25–40 candidate features with definition, unit, lookback, source, missingness, leakage risk and computational cost. Run constant-column, near-duplicate and pairwise-correlation checks. Prefer a small justified baseline to blindly adding every candidate. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Represent features as columns of X; calculate X^T X and explain multicollinearity.

## Days 25–26 | Week 4: Create horizon-aware labels

Construct forward log-return and directional labels at horizons 1, 5 and 10 bars, with a neutral threshold where useful. Drop rows without fully observed future outcomes, and never create labels for genuinely unknown future candles. Keep regression and classification targets separate. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Prove ln(C_(t+h)/C_t)=sum_(j=1)^h r_(t+j).

## Days 27–28 | Week 4: Fit elementary baselines

Compare majority-class, persistence, zero-return and simple logistic-regression baselines before using a Random Forest. Fit imputers and scalers on training data only, then apply unchanged transforms to validation/test. Record class balance and model seeds. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Derive sigmoid(z)=1/(1+e^-z) and the binary cross-entropy loss.

## Days 29–30 | Week 5: Train and diagnose the Random Forest

Refactor ml_model.py to train on causal snapshot features rather than completed-run end-state features. Tune tree depth, leaf size and number of trees on validation only. Inspect train-versus-validation performance and use permutation importance measured on held-out validation. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Explain entropy H(p)=-p ln p-(1-p)ln(1-p) and information gain.

## Days 31–32 | Week 5: Calibrate probabilities

Measure Brier score and reliability by predicted probability bins; use a separate chronological calibration segment if applying sigmoid/isotonic calibration. Report balanced accuracy, precision/recall and log loss rather than relying on a single accuracy threshold. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Differentiate binary cross-entropy with respect to the logit and obtain sigmoid(z)-y.

## Days 33–34 | Week 5: Implement rolling origins

For each historical cutoff t, fit only on observations available by t, generate synthetic forecasts, and score against the subsequent real window. Store cutoff, training range, forecast horizon, run seed and prediction. Make overlapping labels and refit cadence explicit. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Write expanding-window and fixed-window index ranges using inequalities.

## Days 35–36 | Week 5: Compare trajectories fairly

Compare synthetic and actual price paths on matched timestamps and common starting price. Calculate MAE/RMSE of log prices or returns, direction accuracy and horizon-specific errors; include a naive persistence forecast. A visually similar PNG is not a numerical validation. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Derive RMSE and show why squaring disproportionately penalizes large errors.

## Days 37–38 | Week 6: Design the 0–1 similarity index

Create an explicitly defined bounded composite from normalized price-path error, volatility discrepancy and directional performance; estimate normalization scales on validation data and fix weights before final test. Report components alongside the index so a high score cannot conceal failure in one dimension. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Study S=exp(-E/s), show 0<S<=1 for E>=0 and compute dS/dE.

## Days 39–40 | Week 6: Check uncertainty across seeds

For each origin, run many independently seeded paths and report median, prediction intervals, interval coverage and width. Evaluate probabilistic forecasts with appropriate proper scores rather than selecting the best-looking simulation after observing the truth. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Derive sample quantiles and a binomial standard error for empirical coverage.

## Days 41–42 | Week 6: Build leakage adversarial tests

Shift or corrupt future targets and confirm no feature changes; assert feature_time<=cutoff and label_end>cutoff. Audit post-run trade outcomes, last-close statistics and crossover win counts that may leak the target. Maintain a prohibited-feature registry. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Use a dependency graph to prove there is no directed path from future observations to X_t.

## Days 43–44 | Week 7: Run ablations and regime tests

Remove indicator families one at a time and compare performance by low/high volatility, symbol and horizon. Include null synthetic data and real data as distinct evaluation domains. Investigate whether apparent signal survives realistic transaction frictions where relevant. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Compare conditional expectations E[loss|regime] and overall weighted loss.

## Days 45–46 | Week 7: Evaluate distribution shift

Measure how feature distributions change across chronological folds using standardized mean differences and quantile comparisons. Flag unseen regimes, stale calibration and provider changes. Never assume a good result on one market transfers automatically to another. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Introduce covariance matrices and Mahalanobis distance with regularization.

## Days 47–48 | Week 7: Quantify statistical uncertainty

Use block-resampling or fold-level confidence intervals rather than naive independent-row bootstrap on serially correlated candles. Compare models with paired errors on identical origins and document multiple-comparison risk. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Compute paired differences d_i=L_Ai-L_Bi and a confidence interval; discuss dependence.

## Days 49–50 | Week 7: Create an R diagnostic pipeline + Build reproducible reports

Read canonical predictions and candle CSVs into R; validate schema and plot feature missingness, returns, RSI, MACD and residuals. Keep analytical R outputs separate from Python training artifacts and label every figure with symbol, period and source. Create fold-level metrics CSVs, seed-level forecast CSVs, comparison PNGs and a concise model card. Include train/validation/test boundaries, calibration window, data licence, features, targets and known limitations. Make all charts regenerable from saved tabular results. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Fit a simple linear model using matrix notation beta_hat=(X^T X)^-1 X^T y when invertible. Explain how a weighted mean of fold scores differs from an unweighted mean.

## Days 51–52 | Week 8: Automate tests and CLI + Perform an end-to-end dry run

Expose commands for ingest, generate, features, train, walk-forward and report. Add tests for time alignment, OHLC integrity, reproducibility, missing volume and embargo boundaries. Use small offline fixture data for CI; keep network access optional. Run the entire pipeline on a small public or permitted historical sample and a synthetic null dataset. Inspect outputs manually and reconcile R/Python indicator values. Fix schema drift, path errors and hidden defaults before scaling the number of runs. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Write invariants as logical predicates and construct boundary-case tests. Estimate algorithmic cost O(n p) for feature computation and discuss tree-training costs.

## Days 53–54 | Week 8: Freeze the experiment + Audit forecast skill

Lock feature definitions, weights, thresholds and hyperparameters before opening the untouched test period. Run walk-forward scoring without selecting seeds based on test outcomes. Record environment and dataset hashes for replication. Compare model performance to naive and random baselines across horizons and regimes, with confidence intervals. A test accuracy just above 0.5 is not automatically meaningful; account for class imbalance, dependence, costs and uncertainty. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Use a hypothesis-testing framework: null baseline versus incremental skill. Compute a confidence interval for paired loss improvement using fold-level differences.

## Days 55–56 | Week 8: Package reproducible deliverables + Plan the next research iteration

Write README, data dictionary, experiment manifest, model card and example commands. Include a provenance record for each simulation, input cutoff and evaluation window. Keep raw licensed data and secrets out of the distributed archive. Review failure cases and prioritize improvements by evidence: richer historical regimes, alternative generators, better calibration or a parsimonious new feature family. Separate scientific conclusions about forecast skill from engineering success in building the pipeline. **Deliverable:** commit the corresponding module or report, a reproducible small CSV/PNG where appropriate, and a test demonstrating the stated constraint. **Relevant mathematics:** Explain deterministic functions f(data,config,seed) and sources of nondeterminism. Formulate constrained optimization: minimize validation loss subject to feature count and runtime budgets.


## Mathematics crosswalk (use only when relevant)

FP1 themes: complex numbers (Fourier representation of cyclical components; do not imply cycles predict markets), roots/polynomials (characteristic roots of autoregressive recurrences), matrices and transformations (feature covariance, linear regression and PCA), series (EMA and accumulated returns), calculus (log returns, gradient of log loss, optimization), numerical methods (iterative fitting and numerical error), vectors (feature-space distance). Topics differ by awarding-body specification; the mathematics is tied to the model rather than inserted as unrelated exercises.

## Completion criteria

- No target leakage; chronological embargoed evaluation.
- Raw and synthetic data clearly distinguished.
- Reproducible seeds, metadata, and feature registry.
- Null, naive, and ML models compared on the same historical origins.
- Score in [0,1] reported with its constituent errors and uncertainty.
- R plots and Python numerical metrics regenerate from saved CSVs.
