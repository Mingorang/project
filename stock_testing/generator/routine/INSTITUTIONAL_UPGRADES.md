# Five upgrades to reach institutional standard (10 minutes each, done by you)

Written 2026-10-20 from a live run of the project. Every code block below was executed before it was put here, and the "you're done when" outputs are the real outputs. You do the editing; the point is that you understand each line you add.

## Where the model stands today

**Honest answer: the model has no measurable edge yet, and its headline accuracy is a measurement artefact.** Three facts, all reproducible with `python routine/daily_routine.py`:

| Finding | Evidence |
|---|---|
| The 85–97% test accuracy is mechanical | Same RF with the labels shuffled scores 50% (the machinery is fine). On the 100,000-run forex histories the *current* feature set still scores 89–94% under `label_mode = "return"`. One feature alone, `bullish_candle_pct`, separates the label with AUC 0.92. |
| Part of the stored data is corrupt | `stock_sap`: 70–81% of rows have a price <= 0 and 64–75% have no return at all. In the SAP 5d file, run 2 starts exactly where run 1 ended (232.93) and dips to a low of -145.5. TSLA and both forex sets are clean. |
| On real daily bars there is nothing to find yet | Expanding-window walk-forward, next-day direction, 5 years of Yahoo data: SAP 51.5%, TSLA 51.4%, GBP/USD 48.6%, EUR/GBP 49.4%, EUR/USD 51.6%. Pure luck lands within +-3.9 points of 50%, and always guessing the majority class scores 50.7–53.0%. |

### The market right now (Yahoo Finance, last bar 2026-10-20)

| | Last | 3m | 20d vol (ann.) | vol percentile (5y) | RSI14 | 1-day 1σ | 5-day 1σ |
|---|---|---|---|---|---|---|---|
| SAP | 227.27 | +52.8% | 14.1% | 3rd | 71.3 | ±0.89% | ±1.99% |
| TSLA | 373.98 | 0.0% | 32.8% | 3rd | 53.8 | ±2.07% | ±4.62% |
| GBP/USD | 1.3374 | 0.0% | 5.3% | 17th | 56.6 | ±0.34% | ±0.75% |
| EUR/USD | 1.1233 | -1.6% | 4.5% | 15th | 36.3 | ±0.28% | ±0.64% |
| EUR/GBP | 0.8397 | -1.6% | 2.6% | 7th | 19.9 | ±0.17% | ±0.37% |
| S&P 500 | 7,723.6 | +3.0% | 9.2% | 11th | 50.5 | ±0.58% | ±1.30% |

What this means for your model:

- **Volatility is near five-year lows almost everywhere** (3rd–17th percentile). Anything calibrated in a busier regime (your 5-year forex histories, the learned `vol_index`) is oversized for today. That is upgrade 5.
- **Two instruments sit at momentum extremes**: EUR/GBP is deeply oversold (RSI 19.9, -2.6% under its 200-day average) and SAP is overbought after +52.8% in three months (+19% above its 200-day average). Extremes are where a model trained on calm data is most likely to be wrong.
- **GBP/USD and EUR/USD are one bet, not two** (6-month return correlation 0.85); TSLA moves with the S&P 500 (0.66). Position sizing that treats them as independent doubles the risk (upgrade 4).
- **There is no directional call to make.** The honest suggestion for price movement is the volatility band in the last two columns, not "up" or "down". This is not investment advice.

I did not launch `price_modeling.py`: it is fully interactive, writes into `looped_results/`, and your history is already the 78,000-file problem described in `prompt_ideas.txt`. The audit reads the files it already produced, and re-runs your own `ml_model.py` code on them.

**Videos:** I searched for one per topic and found nothing I could vouch for (results were papers and articles, or low-quality clips), so I have not included any rather than send you to something unchecked. Each upgrade lists written references taken from the search results (titles only, not opened by me) and a search phrase if you prefer video.

## How to work through them

Order matters: 1 → 2 → 3 → 4 → 5. Save the new files in the **generator** folder (not `routine/`), using the exact names given, and tomorrow's morning brief will tick the box automatically.

---

## Upgrade 1 — Stop the model from "predicting" what it can already see

**Why it matters today.** Under `label_mode = "return"` the features (share of up-candles, drawdown, RSI) and the label ("did the run finish up?") are measured on the same finished path, so the model just reads the answer. Your code already knows: `leakage_audit()` flags it and `config.toml` documents it, but it only prints a warning and carries on, and the default is still `"return"`.

**The idea in 60 seconds.** A trader at bar 31 of 45 can only use bars 1–31. The label must be what happens *after* that. Your `forecast` mode does exactly this (features from the first 70% of bars, label from the last 30%). The missing piece is a guard that refuses to train when a feature alone separates the label.

```
for each simulated run:
    features = indicators(bars[1 .. 31])          # only what was knowable
    label    = close[45] > close[31]              # what happened next
audit every feature: AUC(feature, label)
if any AUC >= 0.80: STOP with an error             # do not print-and-continue
```

**Do it.**

1. *(1 min)* Run the demo to see the effect on pure random walks:
   ```
   python leak_demo.py
   ```
   Save this as `leak_demo.py` anywhere:
   ```python
   """Why 90% accuracy on random data is a red flag: the feature and the label see the same bars."""
   import numpy as np
   from sklearn.metrics import roc_auc_score

   rng = np.random.default_rng(7)
   RUNS, BARS = 4000, 45
   CUT = int(BARS * 0.7)                                   # bar 31: "now"
   steps = rng.normal(0, 1, (RUNS, BARS))                  # pure random walks: nothing is predictable

   up_all = (steps > 0).mean(axis=1) * 100                 # % of up-candles over the WHOLE run (what 'return' mode uses)
   up_seen = (steps[:, :CUT] > 0).mean(axis=1) * 100       # % of up-candles over the part we could really see

   label_return = steps.sum(axis=1) > 0                    # did the whole run finish up?
   label_forecast = steps[:, CUT:].sum(axis=1) > 0         # does the UNSEEN last 30% finish up?


   def separation(feature, label):
       auc = roc_auc_score(label, feature)                 # 0.5 = useless, 1.0 = reads the answer off the page
       return max(auc, 1 - auc)


   print(f"return label,   feature from the same bars : {separation(up_all, label_return):.3f}")
   print(f"forecast label, feature from earlier bars  : {separation(up_seen, label_forecast):.3f}")
   ```
2. *(1 min)* `config.toml`, line 132: change `label_mode = "return"` to `label_mode = "forecast"`.
3. *(4 min)* In `ml_model.py`, paste this function directly above `def build_forecast_dataset(`:
   ```python
   def leakage_guard(audit: pd.DataFrame, label_mode: str, limit: float = 0.80) -> None:
       """Stop the run instead of printing a warning nobody reads (audit comes from leakage_audit)."""
       bad = audit[audit["separation"] >= limit] if len(audit) else audit
       if len(bad):
           names = ", ".join(f"{r.feature} ({r.separation:.2f})" for r in bad.itertuples())
           raise ValueError(f"Leakage: {names}. label_mode='{label_mode}' measures features and label on the "
                            "same finished path. Set label_mode = \"forecast\" in config.toml.")
   ```
   Then find these two lines (search for `audit.to_csv`) and add the call under them:
   ```python
       audit.to_csv(ml_dir / "leakage_audit.csv", index=False)
       leakage_guard(audit, label_mode)          # <- new line
   ```
4. *(2 min)* Forecast mode needs `processed_bars.csv`, written by the analysis step, and `looped_analysis/` has none today. Generate a fresh batch with `price_modeling.py` (many plots, 300+ runs, output mode 6 = CSV + metrics only), then:
   ```
   python ml_model.py --label return      # must now STOP with a "Leakage:" error
   python ml_model.py --label forecast    # must run to the end
   ```

**You're done when.** `leak_demo.py` prints

```
return label,   feature from the same bars : 0.884
forecast label, feature from earlier bars  : 0.510
```

`--label return` stops with `Leakage: bullish_candle_pct (0.92), candle_direction_balance (0.92), ...`, and `--label forecast` reports a test accuracy near 0.50 (anything from 0.45 to 0.55 is honest on random data).

**Read.** [Purged cross-validation (Wikipedia)](https://en.wikipedia.org/wiki/Purged_cross-validation) · [On Leakage in Machine Learning Pipelines (arXiv 2311.04179)](https://arxiv.org/pdf/2311.04179) · [How to avoid machine learning pitfalls (arXiv 2108.02497)](https://arxiv.org/pdf/2108.02497). Video search phrase: *"data leakage look-ahead bias machine learning time series"*.

---

## Upgrade 2 — A data-integrity gate (bad rows never reach the model)

**Why it matters today.** 81% of the SAP 5d rows and 70% of the SAP 3m rows have impossible prices, and nothing stopped them reaching the model or the saved history. In the SAP 5d file, run 1 ends at 232.9326 and run 2 starts at exactly 232.9326, then falls to a low of -145.5: once a chained run crosses zero, every later run inherits a broken starting price. Institutions treat "garbage in" as a hard stop, not a statistic. Your own `daily_ml_advisor.py` lists this (Day 18) but it is not enforced.

**The idea in 60 seconds.** Before training, check the rows against rules that must always be true, count the casualties, and stop if too many fail. Fix the cause (the simulator) rather than quietly dropping most of your data.

```
for each row:
    require first_open, last_close, lowest_low, highest_high > 0
    require return is a number and |return| <= 500%
    require highest_high >= lowest_low
if more than 5% of rows fail: STOP and say so
else: continue with the survivors
```

**Do it.**

1. *(3 min)* In `ml_model.py`, paste this next to `prune_outlier_runs` (search for `def prune_outlier_runs`):
   ```python
   def validate_run_frame(df: pd.DataFrame, max_drop_pct: float = 5.0, max_abs_return_pct: float = 500.0) -> pd.DataFrame:
       """Drop impossible rows; refuse to train when too many are impossible."""
       df = df.copy()
       df.columns = df.columns.str.strip()                          # some saved histories have padded headers
       n0 = len(df)
       ok = pd.Series(True, index=df.index)
       for col in ("first_open", "last_close", "lowest_low", "highest_high"):
           if col in df.columns:
               ok &= pd.to_numeric(df[col], errors="coerce") > 0    # a price must be positive (and not NaN)
       if "total_return_pct" in df.columns:
           ret = pd.to_numeric(df["total_return_pct"], errors="coerce")
           ok &= ret.notna() & (ret.abs() <= max_abs_return_pct)    # no NaN, no +4,000,000,000 % runs
       if {"highest_high", "lowest_low"} <= set(df.columns):
           ok &= df["highest_high"] >= df["lowest_low"]             # High can never sit below Low
       clean = df[ok]
       dropped = 100.0 * (n0 - len(clean)) / max(n0, 1)
       print(f"[ML] data gate: kept {len(clean)}/{n0} rows ({dropped:.1f}% dropped)")
       if dropped > max_drop_pct:
           raise ValueError(f"Data gate: {dropped:.0f}% of rows have impossible prices. Fix the generator "
                            "(price behaviour [2] = floor at zero) instead of training on this.")
       return clean.reset_index(drop=True)
   ```
2. *(2 min)* In `run_ml_pipeline`, add one line directly above `df = _prepare_training_data(df, summary_csv, ml_dir, keep_history)`. This puts the gate *before* the history file is written, so bad rows can never be saved:
   ```python
       df = validate_run_frame(df)               # <- new line
       df = _prepare_training_data(df, summary_csv, ml_dir, keep_history)
   ```
3. *(2 min)* Test it on the stored SAP file from the generator folder:
   ```
   python -c "import pandas as pd, ml_model as m; m.validate_run_frame(pd.read_csv(r'looped_analysis/ml_results/stock_sap/5d/analysis_history.csv'))"
   ```
4. *(3 min)* Fix the cause. When `price_modeling.py` asks `Price behaviour: [1] allow negative futures prices or [2] floor prices at zero?`, answer **2** for stocks. Only oil-style futures can legitimately go negative.

**You're done when.** Step 3 prints `[ML] data gate: kept 377/2000 rows (81.2% dropped)` followed by `ValueError: Data gate: 81% of rows have impossible prices. Fix the generator ...`. Pointed at the TSLA file it prints `kept 3000/3000 rows (0.0% dropped)` and carries on.

**Read.** [How to avoid machine learning pitfalls (arXiv 2108.02497)](https://arxiv.org/pdf/2108.02497) for a general checklist of data and evaluation mistakes. Video search phrase: *"data validation pipeline machine learning pandera great expectations"*.

---

## Upgrade 3 — One honest accuracy number (`honest_eval.py`)

**Why it matters today.** You currently quote accuracies of 85–97%. On real bars the same kind of model scores 49–52%, inside the ±3.9-point luck band. A number only means something next to its baselines, with a confidence interval, on data the model has never seen and that comes *after* the training data in time.

**The idea in 60 seconds.** Train on the past, skip a few rows (the embargo) so no overlapping information leaks across the boundary, test on the next block, slide forward. Compare with "always guess the most common outcome". Only call it an edge if the *lower* end of the 95% interval beats both 50% and that baseline.

```
split rows in time order; first half is training only
for each later block:
    train on everything before it, minus an embargo gap
    predict the block; record hits and the majority-class baseline hits
accuracy = hits / n ; interval = Wilson 95% CI
verdict = EDGE only if CI.low > max(0.5, baseline)
```

**Do it.**

1. *(4 min)* Save this as `honest_eval.py` in the generator folder and read each comment as you go:
   ```python
   """honest_eval.py - the only accuracy number worth quoting: time-ordered, embargoed, compared with dumb baselines."""
   import numpy as np
   import pandas as pd
   from scipy import stats
   from sklearn.ensemble import RandomForestClassifier


   def honest_eval(X: pd.DataFrame, y: np.ndarray, n_folds: int = 5, embargo: int = 5, seed: int = 0) -> dict:
       """Rows must be in time order. Train only on the past, leave an embargo gap, test on the next block."""
       n = len(y)
       folds = np.array_split(np.arange(n // 2, n), n_folds)         # the first half is only ever training data
       hits, base_hits = [], []
       for test in folds:
           train = np.arange(0, test[0] - embargo)                    # expanding window + gap before the test block
           model = RandomForestClassifier(n_estimators=150, min_samples_leaf=20, random_state=seed, n_jobs=-1)
           model.fit(X.iloc[train], y[train])
           hits.extend((model.predict(X.iloc[test]) == y[test]).tolist())
           majority = round(y[train].mean())                          # baseline: always guess the most common class
           base_hits.extend((np.full(len(test), majority) == y[test]).tolist())
       k, total = int(np.sum(hits)), len(hits)
       ci = stats.binomtest(k, total, 0.5).proportion_ci(0.95, method="wilson")
       acc, base = k / total, float(np.mean(base_hits))
       verdict = "EDGE" if ci.low > max(0.5, base) else "NO EDGE (indistinguishable from chance/baseline)"
       return dict(accuracy=round(acc, 3), majority_baseline=round(base, 3),
                   ci95=(round(float(ci.low), 3), round(float(ci.high), 3)), n=total, verdict=verdict)


   if __name__ == "__main__":
       rng = np.random.default_rng(3)
       X = pd.DataFrame(rng.normal(size=(3000, 6)), columns=list("abcdef"))
       noise = (rng.random(3000) > 0.5).astype(int)                                   # nothing to find
       planted = ((X["a"] * 0.35 + rng.normal(size=3000)) > 0).astype(int).to_numpy()  # a weak real signal in column a
       print("pure noise    :", honest_eval(X, noise))
       print("planted signal:", honest_eval(X, planted))
   ```
2. *(1 min)* Run `python honest_eval.py`.
3. *(3 min)* Point it at your own data. For real bars, in a Python prompt from the generator folder:
   ```python
   import numpy as np, pandas as pd, yfinance as yf
   from honest_eval import honest_eval
   px = yf.download("GBPUSD=X", period="5y", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
   r = np.log(px).diff()
   X = pd.concat({f"lag{i}": r.shift(i) for i in range(5)}, axis=1)
   y = (r.shift(-1) > 0).astype(float)                   # direction of the NEXT day
   ok = X.notna().all(axis=1) & y.notna()
   print(honest_eval(X[ok].reset_index(drop=True), y[ok].to_numpy().astype(int)))
   ```
4. *(2 min)* Replace the accuracy line in your reports with this dictionary's `accuracy`, `ci95` and `verdict`.

**You're done when.** Step 2 prints

```
pure noise    : {'accuracy': 0.513, 'majority_baseline': 0.52, 'ci95': (0.488, 0.539), 'n': 1500, 'verdict': 'NO EDGE (indistinguishable from chance/baseline)'}
planted signal: {'accuracy': 0.585, 'majority_baseline': 0.518, 'ci95': (0.56, 0.61), 'n': 1500, 'verdict': 'EDGE'}
```

That shows the instrument is calibrated: it says no edge when there is none and finds a weak planted one. On GBP/USD real bars step 3 gave `accuracy 0.5, ci95 (0.462, 0.538), NO EDGE` today (the window rolls forward daily, so yours will differ slightly but should also say NO EDGE).

**Read.** [Walk-forward testing: the honest method](https://dev.to/shaktitiwari/walk-forward-testing-for-trading-models-the-honest-method-lopez-de-prado-165j) · [Purged cross-validation (Wikipedia)](https://en.wikipedia.org/wiki/Purged_cross-validation) · [How should you discount your backtest PnL? (arXiv 1902.01802)](https://arxiv.org/pdf/1902.01802) on why many tries need a stricter bar. Video search phrase: *"walk forward validation purged embargo trading backtest"*.

---

## Upgrade 4 — Costs and sizing (`pnl_report`): accuracy is not money

**Why it matters today.** Even a model that is right 52% of the time loses money if it trades often and pays the spread. With a no-skill signal, volatility-targeted and half-Kelly sized, the net Sharpe falls from -0.18 to -1.05 as costs go from 0 to 3 basis points. Volatility is also near five-year lows, so volatility targeting wants to *lever up* the FX pairs right now (10% target vs 5% realised): the leverage cap is a risk control, not a detail. Your `daily_ml_advisor.py` mentions Kelly sizing (Day 20); this makes it real.

**The idea in 60 seconds.** Turn the probability into a position (more conviction → bigger), shrink the position when the market is wild (volatility targeting), halve it for safety (half-Kelly), charge a cost every time the position changes, then report the Sharpe ratio, drawdown and turnover *after* costs.

```
conviction = 2*p_up - 1                         # -1 .. +1
size       = min(target_vol / realised_vol, max_leverage)
position   = conviction * size * 0.5
pnl[t+1]   = position[t] * return[t+1] - cost_bps * |position[t] - position[t-1]|
report net Sharpe, max drawdown, turnover
```

**Do it.**

1. *(4 min)* Save as `pnl_report.py` in the generator folder:
   ```python
   """pnl_report.py - turn probabilities into a net-of-cost, volatility-targeted P&L. Accuracy is not money."""
   import numpy as np
   import pandas as pd


   def pnl_report(p_up: pd.Series, ret: pd.Series, cost_bps: float = 1.0, target_vol: float = 0.10,
                  max_leverage: float = 2.0, kelly_fraction: float = 0.5) -> dict:
       """p_up[t] = P(ret[t+1] > 0) known at the close of t.  ret = simple daily returns."""
       vol = ret.rolling(20).std() * np.sqrt(252)                   # annualised vol, known at t
       size = (target_vol / vol).clip(upper=max_leverage)           # vol targeting: smaller when markets are wild
       pos = (2 * p_up - 1) * size * kelly_fraction                 # conviction (-1..+1) x size x half-Kelly
       gross = (pos * ret.shift(-1)).dropna()                       # the position held at t earns the return of t+1
       cost = pos.diff().abs().fillna(0) * cost_bps / 1e4           # pay the spread whenever the position changes
       net = (pos * ret.shift(-1) - cost).dropna()

       def sharpe(x: pd.Series) -> float:
           return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else float("nan")

       equity = (1 + net).cumprod()
       return dict(gross_sharpe=round(sharpe(gross), 2), net_sharpe=round(sharpe(net), 2),
                   ann_return_net_pct=round(float(net.mean() * 252 * 100), 2),
                   max_drawdown_pct=round(float((equity / equity.cummax() - 1).min() * 100), 1),
                   turnover_per_year=round(float(pos.diff().abs().sum() / len(pos) * 252), 1), days=len(net))


   if __name__ == "__main__":
       rng = np.random.default_rng(5)
       days = pd.bdate_range("2023-01-02", periods=750)
       ret = pd.Series(rng.normal(0.0, 0.006, len(days)), index=days)          # a market with no pattern in it
       coin_flip = pd.Series(rng.uniform(0.35, 0.65, len(days)), index=days)   # a "model" with no skill
       for bps in (0, 1, 3):
           print(f"cost {bps} bp:", pnl_report(coin_flip, ret, cost_bps=bps))
   ```
2. *(1 min)* Run `python pnl_report.py`.
3. *(3 min)* Change the assumption to your real cost. `cost_bps=1.0` is **my assumption** for a tight FX spread, not your broker's number; use their quoted spread (in basis points of price) and a higher figure for single stocks.
4. *(2 min)* Feed it a real probability series (`p_up` indexed by date) and the matching daily `ret`; compare net Sharpe at your cost with gross Sharpe.

**You're done when.** Step 2 prints

```
cost 0 bp: {'gross_sharpe': -0.18, 'net_sharpe': -0.18, ...}
cost 1 bp: {'gross_sharpe': -0.18, 'net_sharpe': -0.47, ...}
cost 3 bp: {'gross_sharpe': -0.18, 'net_sharpe': -1.05, ...}
```

and you can explain why the gross number never changes while the net number keeps falling. The pass mark for a real model is a *net* Sharpe that stays positive at your realistic cost.

**Read.** [The Science and Practice of Trend-Following Systems (arXiv 2607.19497)](https://arxiv.org/pdf/2607.19497) for volatility targeting as fractional Kelly. Video search phrase: *"volatility targeting position sizing half Kelly backtest transaction costs"*.

---

## Upgrade 5 — Know when the market has left the regime you trained in (`regime_check.py`)

**Why it matters today.** Six of six instruments are in the bottom fifth of their five-year volatility range, and two are at RSI extremes. A model silently trained in a different regime keeps producing confident numbers. Institutions keep a receipt of every training run (code version, config, seed, regime) and monitor for drift.

**The idea in 60 seconds.** When you train, write a "run card": git commit, a fingerprint of `config.toml`, the random seed, and where volatility sat in its 5-year history. On later days compare today's volatility percentile with the card. If it moved a lot, or crossed between low/mid/high, the model is stale.

```
on train:   card = {commit, hash(config.toml), seed, vol_percentile_today}; save as JSON
on a later day:
    now = vol_percentile_today
    if |now - card.pctile| > 25 or tier(now) != tier(card.pctile): print RETRAIN
```

**Do it.**

1. *(4 min)* Save as `regime_check.py` in the generator folder (it reads `config.toml` next to it):
   ```python
   """regime_check.py - write a run card when you train, warn when the market has left the regime you trained in.

     python regime_check.py GBPUSD=X          # first call: writes run_card_GBPUSD=X.json
     python regime_check.py GBPUSD=X          # later calls: compares today's volatility regime with the card
   """
   import hashlib
   import json
   import subprocess
   import sys
   from pathlib import Path

   import numpy as np
   import yfinance as yf

   HERE = Path(__file__).resolve().parent


   def vol_percentile(symbol: str) -> dict:
       """20-day annualised volatility and where it sits in its own 5-year history (0 = calmest, 100 = wildest)."""
       close = yf.download(symbol, period="5y", interval="1d", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
       vol = np.log(close).diff().rolling(20).std().dropna() * np.sqrt(252) * 100
       return dict(vol20_ann=round(float(vol.iloc[-1]), 2), pctile=int(round((vol < vol.iloc[-1]).mean() * 100)),
                   asof=str(close.index[-1].date()))


   def write_run_card(path: Path, symbol: str, label_mode: str, seed: int) -> dict:
       """The receipt for one training run: enough to reproduce it and to tell later whether it has gone stale."""
       config = (HERE / "config.toml").read_bytes()
       git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True)
       card = dict(symbol=symbol, label_mode=label_mode, seed=seed, git_commit=git.stdout.strip() or "unknown",
                   config_sha256=hashlib.sha256(config).hexdigest()[:16], **vol_percentile(symbol))
       path.write_text(json.dumps(card, indent=2), encoding="utf-8")
       return card


   def check_drift(path: Path, symbol: str, max_shift: int = 25) -> str:
       card = json.loads(path.read_text(encoding="utf-8"))
       now = vol_percentile(symbol)
       tier = lambda p: "low" if p < 33 else "mid" if p < 67 else "high"          # noqa: E731
       shift = abs(now["pctile"] - card["pctile"])
       stale = shift > max_shift or tier(now["pctile"]) != tier(card["pctile"])
       return (f"{symbol}: trained in {tier(card['pctile'])}-vol ({card['pctile']}th pct), now {tier(now['pctile'])}-vol "
               f"({now['pctile']}th pct) -> {'RETRAIN / RECALIBRATE' if stale else 'regime unchanged'}")


   if __name__ == "__main__":
       symbol = sys.argv[1] if len(sys.argv) > 1 else "GBPUSD=X"
       card_path = HERE / f"run_card_{symbol.replace('=', '_')}.json"
       if card_path.exists():
           print(check_drift(card_path, symbol))
       else:
           print("run card written:", write_run_card(card_path, symbol, label_mode="forecast", seed=42))
   ```
2. *(1 min)* `python regime_check.py GBPUSD=X` writes the card. Run it again: it compares.
3. *(3 min)* Edit the JSON file it wrote and change `"pctile"` to `80`, run again, and watch the verdict flip. This proves the alarm works.
4. *(2 min)* Call `write_run_card(...)` at the end of every training run, and add the check to whatever you run each morning.

**You're done when.** The first call prints a card like `{'symbol': 'GBPUSD=X', 'label_mode': 'forecast', 'seed': 42, 'git_commit': '...', 'vol20_ann': 5.32, 'pctile': 16, ...}`, the second prints `GBPUSD=X: trained in low-vol (16th pct), now low-vol (16th pct) -> regime unchanged`, and with `pctile` set to 80 it prints `trained in high-vol (80th pct), now low-vol (16th pct) -> RETRAIN / RECALIBRATE`.

**Read.** [Spurious Predictability in Financial Machine Learning (arXiv 2604.15531)](https://arxiv.org/pdf/2604.15531). Video search phrase: *"model drift monitoring regime change machine learning finance"*.

---

## After all five

The morning brief reads `1/5 … 5/5` from your code. When it shows 5/5, retrain with `label_mode = "forecast"`, read the `honest_eval` verdict, and only then look at the `pnl_report` net Sharpe. If the verdict is still `NO EDGE`, that is the correct answer for daily-bar direction on liquid assets, and the useful outputs are the volatility bands, the regime flag and the risk numbers.
