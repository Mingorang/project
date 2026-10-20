"""
daily_ml_advisor.py — Automated Daily ML & Quant Improvement Advisor
Implements Roadmap Idea 87: "every day send me a message at a set time on how to improve code for ml etc"

Usage:
  python daily_ml_advisor.py --today         # Output today's ML improvement insight
  python daily_ml_advisor.py --day N         # View tip for specific day N (1-34)
  python daily_ml_advisor.py --send-now      # Log and deliver current tip to log file
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

TIPS = [
    ("Day 01", "Eliminate Data Leakage with Purged K-Fold CV",
     "Standard K-fold cross-validation leaks forward-looking temporal information. Always use Purged K-Fold with an embargo window after each test split (López de Prado, 2018)."),
    ("Day 02", "Feature Stationarity via Fractional Differentiation",
     "Integer differentiation (e.g., returns = pct_change) removes all memory from a series. Use fractional differentiation d in [0.3, 0.7] to achieve stationarity while preserving long-term memory."),
    ("Day 03", "Non-Linear Activation Tuning in Neural Networks",
     "When modeling return distributions with fat tails and bounded extremes, tanh(x) or LeakyReLU outperforms standard ReLU by avoiding dead neurons during negative market crash shocks."),
    ("Day 04", "Calibrate Probabilities with Isotonic Regression",
     "Raw Random Forest probabilities tend to push away from 0 and 1. Apply CalibratedClassifierCV(method='isotonic') to convert raw voting percentages into true empirical event frequencies."),
    ("Day 05", "Triple-Barrier Labeling over Fixed-Time Horizons",
     "Fixed-time labels (Close[t+k] > Close[t]) fail to reflect real market dynamics. Define three barriers: upper profit take, lower stop loss, and max holding period."),
    ("Day 06", "Outlier Pruning without Biasing Tails",
     "Never use arbitrary hard cuts on returns. Prune simulation runs where total return deviates by >25% from cohort median, but retain fat-tailed jump-diffusion components."),
    ("Day 07", "Hyperparameter Search: Ditch GridSearch for Bayesian Optimization",
     "GridSearch wastes 80% of computation on unpromising parameter regions. Use Optuna or RandomizedSearchCV with stratified scoring to explore complex loss surfaces efficiently."),
    ("Day 08", "Variance Inflation Factor (VIF) for Multicollinearity",
     "Technical indicators (RSI, Bollinger %B, MACD) are heavily collinear. Compute VIF and drop features with VIF > 10 to stabilize feature weights and tree splits."),
    ("Day 09", "SHAP Game-Theoretic Feature Importance",
     "Mean Decrease Impurity (MDI) in Random Forests over-weights high-cardinality features. Use TreeSHAP (shap library) to compute unbiased Shapley values per prediction."),
    ("Day 10", "Adversarial Validation to Detect Dataset Shift",
     "Train a classifier to distinguish simulated data from real Yahoo market data. If ROC-AUC > 0.55, your simulation has a discernible artificial signature; tune volatility to reach AUC ≈ 0.50."),
    ("Day 11", "Ensemble Bagging vs Boosting in Noisy Regimes",
     "In low signal-to-noise financial data, boosting algorithms (XGBoost, LightGBM) easily overfit to noise. Random Forest bagging is generally more robust unless strong regularization is applied."),
    ("Day 12", "Continuous Online Learning with Warm Starts",
     "Instead of retraining from scratch on every new batch of bars, use warm_start=True in sklearn or incremental stochastic gradient descent to update weights sequentially."),
    ("Day 13", "Bisection Search for Volatility Equilibrium",
     "Linear stepping can oscillate around optimal vol. Use bisection (binary search) bounded by [0.1 * sigma_real, 5.0 * sigma_real] to converge in O(log N) iterations."),
    ("Day 14", "Wick-to-Body Asymmetry Modeling",
     "Real market bear candles typically exhibit longer lower wicks due to liquidity-seeking spikes. Parameterize separate bull and bear wick ratios in the generator."),
    ("Day 15", "Vectorized SIMD NumPy Operations",
     "Replace Python for-loops with np.where and np.maximum. A 10,000-candle simulation should run in under 5 milliseconds on a single CPU core."),
    ("Day 16", "Synthetic Volume Coupling to Volatility",
     "Volume is not independent of price: high volume clusters near session open/close and increases return dispersion. Couple volume lognormal draws to your return shock multiplier."),
    ("Day 17", "GARCH(1,1) Volatility Clustering Integration",
     "Volatility is conditionally heteroskedastic. Incorporate sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2 into the synthetic path generator."),
    ("Day 18", "Defensive Invariant Assertions on All Pipeline Outputs",
     "Never allow High < Low or Close < 0 to propagate silently. Wrap generator functions in invariant decorators that raise immediately on corrupt data."),
    ("Day 19", "Automated HTML Reporting with KPI Scorecards",
     "Track model metrics systematically. Generate an executive HTML report at the end of each backtest loop recording exact, price, and bars match criteria."),
    ("Day 20", "Kelly Criterion for Position Sizing",
     "A 55% win rate model loses money if sized poorly. Size positions according to f* = (p * b - q) / b with a half-Kelly safety dampener."),
    ("Day 21", "Information Ratio as Simulation Fitness Metric",
     "Rather than raw RMSE, compute the Information Ratio of (sim_returns - real_returns). A truly calibrated simulation drives active tracking error towards 0."),
    ("Day 22", "Disk Space Hygiene: Columnar Parquet over CSV",
     "78,000 small CSV files fragment disk sectors and slow git. Store bulk simulation runs in snappy-compressed Apache Parquet or prune obsolete chart images."),
]


def get_daily_advice(day_idx: int | None = None) -> tuple[str, str, str]:
    if day_idx is None:
        day_idx = (datetime.date.today().toordinal()) % len(TIPS)
    else:
        day_idx = (day_idx - 1) % len(TIPS)
    return TIPS[day_idx]


def main():
    parser = argparse.ArgumentParser(description="Daily ML & Quant Improvement Advisor")
    parser.add_argument("--today", action="store_true", help="Display today's ML advice")
    parser.add_argument("--day", type=int, default=None, help="Display advice for specific day (1-22)")
    parser.add_argument("--send-now", action="store_true", help="Log today's advice to log file")
    args = parser.parse_args()

    day_label, title, body = get_daily_advice(args.day)

    print("\n" + "=" * 70)
    print(f"  DAILY QUANT & ML ADVISOR [{day_label}]")
    print(f"  Topic: {title}")
    print("=" * 70)
    print(f"\n{body}\n")
    print("=" * 70 + "\n")

    if args.send_now:
        log_file = Path(__file__).parent / "daily_advisor_log.txt"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now().isoformat()}] {day_label} - {title}\n{body}\n\n")
        print(f"[Advisor] Advice logged to {log_file}")


if __name__ == "__main__":
    main()
