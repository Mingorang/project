# The Quant Build Manual

**A single reference for building a multi-variable financial machine-learning system — and for not being the ninth person out of ten who fails at it.**

| | |
|---|---|
| **Owner** | QMUL Financial Mathematics undergraduate, Mile End |
| **Covers** | 26 November 2026 → 1 January 2028 (57 weeks) |
| **Replaces** | `QMUL_FM_STUDY_PLAN_AND_AI_DEBUGGING_GUIDE.txt`, `ROADMAP_AND_SECTION_B_STUDY_GUIDE.md`, `VIBECODING_SAFEGUARDS_AND_SECURITY.md` |
| **Codebase** | `stock_testing/generator/` — `price_modeling.py`, `market_analysis.py`, `ml_model.py`, `backtest_compare.py`, `analyze_looped_results.R` |
| **Written** | 20 October 2026 |
| **Status of its numbers** | Every figure in Part I and Part III was measured on this machine on 20 October 2026, not estimated. Reproduction commands are given. |

---

## Contents

| Part | Title | What it is for |
|---|---|---|
| [I](#part-i--ground-truth-where-you-actually-are) | Ground truth: where you actually are | The measurements. Read first. |
| [II](#part-ii--the-people-who-tried-this-before-and-what-actually-happened-to-them) | The people who tried this before | Base rates, the literature, why attempts fail |
| [III](#part-iii--the-goal-restated-so-that-it-can-actually-be-reached) | The goal, restated so it can be reached | The detection floor and the n/p arithmetic |
| [IV](#part-iv--the-57-week-plan-26-november-2026--1-january-2028) | The 57-week plan | Six phases, each with a numeric gate |
| [V](#part-v--the-engineering-spine) | The engineering spine | Working with AI without drowning in unverified code |
| [VI](#part-vi--the-mathematics-and-when-you-need-it) | The mathematics | SDEs, volatility estimators, distances, metrics |
| [VII](#part-vii--not-fooling-yourself-the-validation-statistics) | Not fooling yourself | Purged CV, CPCV, deflated Sharpe, PBO |
| [VIII](#part-viii--security-and-operational-safety) | Security and operational safety | Packages, credentials, audits |
| [IX](#part-ix--the-debugging-playbook) | The debugging playbook | Triage order, symptom table |
| [X](#part-x--the-100-idea-roadmap-re-triaged-against-the-measurements) | The 100 ideas, re-triaged | Which are dead, repurposed, essential, new |
| [XI](#part-xi--qmul-modules-exams-viva-and-the-city) | QMUL and the City | Module mapping, formula sheet, viva, interviews |
| [XII](#part-xii--reference) | Reference | Reading list, glossary, checklists, templates |

---

## How to read this file

It is long on purpose, but it is not meant to be read front to back in one sitting. It has four distinct jobs, and you will use them at different times:

1. **Parts I–III decide what you build.** Read these once, properly, before you write another line of code. They contain the measurements that make most of the obvious plan impossible, and the reformulation that makes a version of it possible. If you only ever read twenty pages of this file, read these.
2. **Part IV is the calendar.** 57 weeks, six phases, each with a numeric gate you either pass or fail. Open it every Monday.
3. **Parts V–IX are the craft.** Engineering discipline, mathematics, validation statistics, security, debugging. Reference material — you dip in when the week calls for it.
4. **Parts X–XII are the lookup tables.** The 100-idea roadmap re-triaged against reality, the QMUL module mapping, the glossary, the reading list, the prompt templates.

A warning about the three files this replaces. They were good on engineering craft and genuinely good on mathematics, and almost all of that survives here. But their *plan* aimed at a target that cannot be hit, and they encouraged a framing — "make the simulator realistic, then the ML will find the signal" — that the measurements in Part I rule out. Where this file contradicts them, this file is right, and Part I shows the working.

### The one-page version

If you remember nothing else:

- Your current model's 92–96% accuracy is **label leakage**, not skill. Measured. The same model on shuffled labels scores 50%.
- **Daily direction on one instrument is unreachable.** Not hard — unreachable. The smallest signal your pipeline could detect needs roughly 20,000 to 80,000 training bars. Five years of daily data gives you 1,250. Measured.
- **You will not fix this with a better model.** The limit is the sample size relative to the signal, and no algorithm moves it.
- **Two things move it.** More rows (go cross-sectional: 500 instruments, not 1) and an easier target (predict the *size* of moves, which is measurable today, instead of their *direction*, which is not).
- Your stated goal — "hundreds if not thousands of variables" — is achievable, but only on a panel of roughly 500 instruments. With 1,250 rows and 1,000 features you are guaranteed to overfit. With 600,000 rows and 1,000 features you are doing what the published literature does.
- The people who have succeeded at this got **out-of-sample monthly R² well under 1%**. That is the honest benchmark. Anything you build that claims more is broken, and you should go looking for the bug rather than celebrating.

---

# Part I — Ground truth: where you actually are

Everything in this part was measured on 20 October 2026 against your own code and your own stored results. No claim here is an estimate or a rule of thumb. Where a number came from a script, the script is named.

## I.1 The headline accuracy is a measurement artefact

Your stored models report test accuracies between 85% and 97%. On data generated by a random-number generator, the correct answer is 50%.

| Stored result set | Rows | Reported test accuracy | Accuracy with labels shuffled | Gap |
|---|---|---|---|---|
| `forex_eur_gbp/5y` | 100,000 | 0.926 | 0.491 | **+0.409** |
| `forex_gbp_usd/5y` | 87,000 | 0.898 | 0.510 | **+0.388** |
| `stock_tsla/5d` | 3,000 | 0.904 | 0.516 | **+0.389** |
| `stock_sap/3m` | 4,000 | 0.915 | 0.674 | +0.218 |
| `stock_sap/5d` | 2,000 | 0.883 | 0.643 | +0.240 |

The shuffled-label column is the diagnostic. Train the identical forest on the identical features with the labels randomly permuted, and it scores 50%. That proves the machinery is sound — there is no bug in the training code. The 92% comes from somewhere else.

It comes from the label definition. In `config.toml`:

```toml
[ml]
label_mode = "return"
```

With that setting, the features and the label are both computed over the *same finished price path*. `bullish_candle_pct` — the share of up-candles in the run — separates the label `total_return_pct > 0` with an AUC of 0.92 on its own. Of course it does: a run that finished up mostly had up-candles. The model is not forecasting, it is reading the answer off the page.

Your code already knows. `leakage_audit()` in `ml_model.py` computes exactly this single-feature AUC and prints a warning naming the offenders. It then carries on and trains anyway. The warning has been printing for every run you have ever done.

### The clean experiment

On 4,000 pure random walks of 45 bars each, with the feature measured two different ways:

| Setup | Single-feature AUC |
|---|---|
| Feature over the whole path, label = whole path finished up | **0.884** |
| Feature over the first 70% of bars, label = last 30% finished up | **0.510** |

Same generator, same feature, same label family. The only difference is whether the feature is allowed to see the bars the label is computed from. That difference is worth 37 points of AUC, and all of it is fake.

Reproduce:

```bash
python leak_demo.py
```

### What the honest setting gives

Switching to `label_mode = "forecast"` — features from the first 70% of bars, label from the last 30% — on a fresh 2,000-run batch, with all eight configured models:

| Model | CV accuracy | Test accuracy | p-value vs chance |
|---|---|---|---|
| logistic | 0.520 | 0.488 | 0.729 |
| logistic_tanh | 0.518 | 0.512 | 0.729 |
| hist_gradient_boosting | 0.508 | 0.545 | 0.133 |
| xgboost | 0.505 | 0.495 | 0.908 |
| lightgbm | 0.501 | 0.495 | 0.908 |
| extra_trees | 0.505 | 0.438 | 0.037 |
| random_forest | 0.505 | 0.435 | 0.028 |
| mlp_tanh | 0.496 | 0.482 | 0.563 |

Every model at chance. This is the **correct** result — the prices are random walks, so there is nothing to learn. The two p-values below 0.05 are what you get when you compare eight models on the same test set; they are multiple-testing artefacts, not discoveries. Part VII explains how to stop being fooled by them.

The random forest scored 0.918 on its own training rows and 0.500 out-of-bag in the same run. That spread is the visible signature of a model memorising noise.

## I.2 Part of your stored data is physically impossible

| Result set | Rows | Rows with a price ≤ 0 | Rows with no computable return |
|---|---|---|---|
| `stock_sap/5d` | 2,000 | **81.1%** | 74.5% |
| `stock_sap/3m` | 4,000 | **70.1%** | 63.8% |
| `stock_tsla/5d` | 3,000 | 0.0% | 0.0% |
| `forex_gbp_usd/5y` | 87,000 | 0.0% | 0.0% |
| `forex_eur_gbp/5y` | 100,000 | 0.0% | 0.0% |

A share price cannot be negative. In the SAP 5d file, run 1 ends at 232.9326 and run 2 opens at exactly 232.9326, then falls to a low of −145.5. The runs are chained: each one starts where the last ended. Once a chained path crosses zero — which it can, because the simulator was answered with "allow negative futures prices" — every subsequent run inherits a broken starting price and the corruption propagates forward through the whole batch.

Four-fifths of that dataset is unusable, and nothing in the pipeline stopped it being written to disk, appended to the accumulated history, or fed to the model. The fix is two-sided: a gate that refuses to train on it (Part V.4), and answering **2 — floor prices at zero** at the generator prompt for anything that is not a genuine negative-capable future.

## I.3 On real market data there is nothing to find — yet

Expanding-window walk-forward, next-day direction, five years of real daily bars, five-bar embargo, retrain every 90 days:

| Instrument | Walk-forward accuracy | Majority-class baseline | Test rows | 95% luck band |
|---|---|---|---|---|
| EUR/USD | 0.516 | 0.530 | 640 | ±0.039 |
| SAP | 0.515 | 0.507 | 617 | ±0.039 |
| TSLA | 0.514 | 0.509 | 617 | ±0.039 |
| EUR/GBP | 0.494 | 0.511 | 640 | ±0.039 |
| GBP/USD | 0.486 | 0.509 | 640 | ±0.039 |

Every instrument sits inside the luck band. Three of the five fail to beat the dumbest possible baseline — always predicting whichever direction was more common in training.

This is not a failure of your code. It is the expected result, and Part II explains why it is the expected result for everyone.

## I.4 The market as of 20 October 2026

Context for any calibration you do in the first weeks of the plan.

| | Last | 3-month | 20d vol (annualised) | Vol percentile (5y) | RSI(14) | 1-day 1σ | 5-day 1σ |
|---|---|---|---|---|---|---|---|
| SAP | 227.27 | +52.8% | 14.1% | 3rd | 71.3 | ±0.89% | ±1.99% |
| TSLA | 373.98 | 0.0% | 32.8% | 3rd | 53.8 | ±2.07% | ±4.62% |
| GBP/USD | 1.3374 | 0.0% | 5.3% | 17th | 56.6 | ±0.34% | ±0.75% |
| EUR/USD | 1.1233 | −1.6% | 4.5% | 15th | 36.3 | ±0.28% | ±0.64% |
| EUR/GBP | 0.8397 | −1.6% | 2.6% | 7th | 19.9 | ±0.17% | ±0.37% |
| S&P 500 | 7,723.6 | +3.0% | 9.2% | 11th | 50.5 | ±0.58% | ±1.30% |

Three things matter for the build:

- **Volatility is near five-year lows across the board** — 3rd to 17th percentile. Anything you calibrate now is calibrated to an unusually quiet market, and will be mis-sized when volatility normalises. This is why Part IV builds regime detection in early rather than late.
- **Two instruments are at momentum extremes.** EUR/GBP at RSI 19.9 and 2.6% below its 200-day average; SAP at RSI 71.3 after +52.8% in three months. Models trained on calm data are least reliable exactly here.
- **GBP/USD and EUR/USD are nearly the same bet.** Six-month return correlation 0.85. TSLA moves with the S&P 500 at 0.66. Sizing five instruments as if they were five independent bets roughly doubles your true risk. Part VI.7 covers the covariance arithmetic.

## I.5 What this means for the file you are reading

The three documents this replaces laid out a 34-week plan whose centre of gravity was making the simulator more realistic — jump diffusion, stochastic volatility, wick asymmetry, intraday volume curves — on the theory that a more realistic simulator would let the ML find real signal.

That theory is testable, and Part III tests it. The short version: a perfect simulator does not help, because the problem is not that your synthetic data is unrealistic. The problem is that the quantity you are asking the model to predict is smaller than the smallest quantity your sample size can resolve. Improving the simulator does not change the sample size and does not change the signal.

The mathematics in those documents is still worth learning, and all of it is carried into Part VI. What changes is **what it is for**. Jump diffusion stops being a step toward predicting direction and becomes a tool for modelling tail risk and sizing positions, which is where it genuinely earns its place.

---

# Part II — The people who tried this before, and what actually happened to them

You asked for the rational logic: someone smarter has tried this, so what makes your attempt different? This part is the honest answer. It is not discouragement. It is the map of where the bodies are buried, so you can walk around them instead of into them.

## II.1 The thing everybody tries first

Essentially every quantitatively-minded person who encounters financial data has the same first idea in the same order:

1. Get price history.
2. Compute technical indicators — RSI, MACD, Bollinger, moving averages.
3. Label each bar by whether the next bar went up.
4. Train a classifier.
5. Observe high accuracy.
6. Conclude they have found something.

Step 5 is where it goes wrong, and it goes wrong the same way for almost everyone: a leak. Yours was `bullish_candle_pct` against a same-path label. Other classic forms:

| Leak | How it looks | Why it fools people |
|---|---|---|
| Same-path features and labels | Yours. Feature and target measured over identical bars. | Accuracy is spectacular and the code has no bug in it. |
| Normalising before splitting | `StandardScaler().fit(all_data)` then split. | Test-set means and standard deviations leak into training. Subtle, tiny, and it inflates everything. |
| Random k-fold on time series | `KFold(shuffle=True)` on ordered bars. | Most training folds sit *after* the test fold in time. The model learns the future. |
| Restated fundamentals | Using a company's reported earnings on the date they refer to, not the date they were published. | The number did not exist yet. Often weeks of look-ahead. |
| Survivorship | A stock universe containing only companies that still exist. | The failures were silently deleted. Returns look wonderful. |
| Overlapping labels | A 10-day forward return computed every day. | Consecutive rows share 9 of 10 days. Your effective sample size is a tenth of what you think. |

The last one is why the embargo exists in purged cross-validation, and why your `purged_kfold_splits()` function takes an `embargo_pct`.

**Base rate to hold in your head:** when an untrained practitioner reports above 60% directional accuracy on liquid daily data, the overwhelmingly likely explanation is a leak, not a discovery. The prior should be roughly "99 times out of 100 it is a leak." Your own project was one of those 99. That is not embarrassing; it is the default state, and you have now left it, which most people never do.

## II.2 What the serious attempts actually achieved

This is the part that matters for your goal, because it sets the realistic ceiling.

**Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning", *Review of Financial Studies* 33(5).** This is the canonical large-scale test of exactly your idea — many variables, many models, proper validation. They used roughly 94 firm characteristics across roughly 30,000 US stocks over about 60 years of monthly data. They compared linear models, penalised regressions, random forests, gradient-boosted trees and neural networks.

Their headline finding, stated honestly: machine learning did beat linear methods, neural networks did best, and the out-of-sample **monthly R² was well under 1%** — on the order of 0.4% for the best models. Verify the exact figure yourself when you read it; the point stands at any value in that neighbourhood.

Sit with that number. The best-resourced, most careful, peer-reviewed implementation of "throw hundreds of variables and modern ML at asset returns" explains well under one percent of the out-of-sample variation. And that was considered — correctly — a significant positive result, because across 30,000 stocks and 60 years, a 0.4% monthly R² compounds into an economically meaningful long-short portfolio.

The lesson is not "give up." The lesson is **calibrate your expectations by three orders of magnitude** from where most beginners set them, and build the measurement apparatus that can detect a 0.4% effect rather than one that can only see a 40% effect.

**Other load-bearing results you should know by name:**

| Work | What it established | What it means for you |
|---|---|---|
| Harvey, Liu & Zhu (2016), "...and the Cross-Section of Expected Returns" | Hundreds of published return predictors; after correcting for the number of things tested, a new factor needs roughly a t-statistic above 3.0, not the usual 2.0. | Your significance bar is higher than the textbook one, because you will test many ideas. |
| Bailey, Borwein, López de Prado & Zhu, "The Probability of Backtest Overfitting" | With enough configurations tried, an impressive in-sample Sharpe is the *expected* outcome of noise. They quantify the probability. | Count your trials. Part VII.5 shows how. |
| López de Prado (2018), *Advances in Financial Machine Learning* | Purged k-fold with embargo, triple-barrier labelling, meta-labelling, fractional differentiation, sample uniqueness. | The methodological spine of Parts VI–VII. Chapter 7 is the single most important chapter for you. |
| Arnott, Harvey & Markowitz (2019), "A Backtesting Protocol in the Era of Machine Learning" | A practical protocol: pre-register, limit trials, hold out genuinely, report costs. | The discipline checklist in Part VII.8 is built from this. |
| Fama & French factor models | A small number of simple, economically-motivated factors explain much of the cross-section of returns. | This is your **baseline**. Any 1,000-variable model must beat a 5-factor linear model after costs, or it has earned nothing. |

Treat all of these as things to verify rather than trust. Part XII.1 gives the full reading list with what to read each for.

## II.3 The real reason most attempts fail

It is not insufficient cleverness and it is rarely a bad algorithm. It is almost always one of five structural errors:

**1. They never measured whether the signal was above their detection floor.** They jumped to modelling without asking what the smallest effect they could reliably see was. Part III does this arithmetic for you, and the answer reframes the entire project.

**2. They had more parameters than observations.** With 1,000 features and 1,250 rows, a sufficiently flexible model fits the training set perfectly and has learned nothing. The failure is invisible in-sample.

**3. They tested many ideas on one dataset and reported the best one.** Try forty configurations on five years of EUR/USD and the best will look significant by construction. Without counting the trials, the number is meaningless.

**4. They measured accuracy instead of money.** A 52% model that trades daily and pays the spread loses. Measured in Part III.4: a no-skill signal's net Sharpe falls from −0.18 to −1.05 as costs go from 0 to 3 basis points. Accuracy and profit are different questions.

**5. They predicted the hardest possible target.** Next-bar direction on a liquid instrument is close to the theoretical maximum difficulty in all of finance. Volatility, range, relative cross-sectional ranking and regime are all measurably easier, and all of them are useful.

## II.4 So what makes your attempt different

Not intelligence, and not effort. Four specific structural advantages, each of which you either already have or can have by Christmas:

1. **You have already found your leak.** Most people never do. You now have a measured, reproducible demonstration of it and a guard that stops the run. You are starting from a true 50%, which is the only honest place to start from.

2. **You have a detection floor, as a number.** Part III. Almost nobody computes this. It converts "is this working?" from a feeling into an inequality you can evaluate.

3. **You have a simulator with ground truth.** Your `price_modeling.py` can generate data where *you* chose the answer. That makes it a calibration instrument: you can measure what your pipeline can and cannot see before you spend a year looking. Almost no beginner has this, and it is the single most valuable asset in your repository — just not for the reason it was built.

4. **You are measuring and writing down kill criteria in advance.** Part IV attaches a numeric gate to every phase. Failing a gate is not failure of the project; it is the project working. The people who fail are the ones who never wrote the gate down and so could never admit the answer was no.

The honest framing of your goal: you are not going to beat Citadel. You are going to build, understand and validate a system that does something real, measured honestly, with the apparatus to prove it. That is a first-class dissertation, a portfolio that stands out in a City interview, and a genuine skill. It is also the only version of this that succeeds.

## II.5 The failure modes specific to how you are working

You are building with AI assistance. That is a genuine accelerant and it carries three specific risks that do not appear in the textbooks:

- **The model will agree with you.** Ask "is 92% accuracy good?" and you will often get encouragement. Ask "what would make this 92% fake?" and you get the leak. Phrase prompts adversarially. Part V.2 gives the templates.
- **Generated code tends toward silent failure.** `try/except: pass`, `.fillna(0)`, dropped assertions. Each one converts a loud crash into quiet corruption. Part V.3 is the watchlist.
- **You can produce more code than you can verify.** This is the real danger. The discipline that keeps it in check is that nothing enters the repository without a test that would fail if it were wrong. Part V.5.

---

# Part III — The goal, restated so that it can actually be reached

Your stated goal:

> *"Building a machine learning model that takes all forms of data and uses it for testing, taking hundreds if not thousands of variables."*

That goal is achievable. The version of it you are currently pointed at is not. The difference is entirely about arithmetic, and this part does the arithmetic.

## III.1 The detection floor — the number that reframes everything

The question nobody asks before starting: **what is the smallest effect my setup could detect even if it were really there?**

I measured it on your pipeline. The method: inject a known lag-1 autocorrelation φ into simulated return series, train on N bars, test on a fixed held-out 40,000 bars, and record whether the recovered accuracy clears the significance threshold for that sample size. Because φ was chosen rather than discovered, the right answer is known exactly.

First, the rig is correct. At 100,000 training bars, the recovered accuracy matches the theoretical oracle:

| True φ | Theoretical maximum accuracy | Oracle `sign(φ·rₜ)` | Logistic model achieved |
|---|---|---|---|
| −0.05 | 0.5159 | 0.5178 | 0.4991 |
| −0.10 | 0.5318 | 0.5333 | **0.5357** |
| −0.20 | 0.5639 | 0.5657 | **0.5660** |
| −0.35 | 0.6136 | 0.6156 | **0.6093** |

Agreement to about 0.003. The measuring instrument works.

Now the result that matters — **training bars required before the model can find the signal at all**:

| Training bars | φ = −0.05 | φ = −0.10 | φ = −0.20 | φ = −0.35 |
|---|---|---|---|---|
| 250 | 0.5003 | 0.5005 | 0.5003 | 0.5235 |
| 640 | 0.5007 | 0.5030 | 0.5001 | 0.5036 |
| **1,250** *(your 5 years)* | 0.5000 | 0.5002 | 0.5036 | 0.5113 |
| 5,000 *(20 years)* | 0.5000 | 0.5047 | **0.5412** ✓ | **0.5813** ✓ |
| 20,000 *(79 years)* | 0.5059 | **0.5307** ✓ | **0.5622** ✓ | **0.6059** ✓ |
| 80,000 *(317 years)* | **0.5072** ✓ | **0.5321** ✓ | **0.5627** ✓ | **0.6081** ✓ |

✓ = the recovered edge exceeds what that many training bars can distinguish from chance.

**At 1,250 training bars, nothing is detectable. Not even φ = −0.35**, which is an enormous, utterly unrealistic degree of mean reversion.

Now measure what is actually in your instruments:

| Instrument | lag-1 autocorrelation (250 days) | lag-1 (5 years) |
|---|---|---|
| EUR/USD | −0.009 | +0.004 |
| EUR/GBP | −0.030 | −0.337 |
| SAP | +0.036 | +0.009 |
| GBP/USD | −0.046 | +0.001 |
| TSLA | −0.082 | −0.022 |

Every recent value is below 0.09 in absolute terms. Cross-referencing the table above: a signal that size requires somewhere between 20,000 and 80,000 training bars. In daily data that is **79 to 317 years**.

*(The EUR/GBP five-year figure of −0.337 is an artefact worth investigating rather than a tradable signal — a −0.34 autocorrelation over five years of daily data would be extraordinary, and almost certainly reflects a data issue such as a stale-quote or fixing convention in the vendor series. Check it before believing it. This is itself a useful lesson: when a number is too good, the first hypothesis is always "data problem".)*

### The conclusion, stated plainly

The signal you are hunting is roughly **two orders of magnitude smaller** than the smallest signal your current setup can resolve. This is not a modelling problem. No algorithm, no amount of feature engineering, no improvement to the simulator changes it. It is a sample-size problem, and only sample size or an easier target fixes it.

Reproduce: `python detection_limits.py` and the sample-size sweep in `sample_size_curve.csv`.

## III.2 The arithmetic of "thousands of variables"

Your goal explicitly wants hundreds to thousands of variables. Here is the constraint that governs it.

Call the number of features **p** and the number of independent observations **n**. The ratio n/p determines whether learning is possible:

| n/p | What happens |
|---|---|
| < 1 | A flexible model fits the training data perfectly and has learned nothing. Not detectable in-sample. |
| 1 – 10 | Severe overfitting. Heavy regularisation mandatory, and even then unstable. |
| 10 – 100 | Workable with regularisation and honest validation. The practical minimum for trees. |
| > 100 | Comfortable. Feature selection becomes a genuine exercise rather than a survival tactic. |

So **p = 1,000 features needs n ≥ 10,000 rows as an absolute floor, and 100,000+ to be comfortable.**

Now count the rows actually available to you:

| Design | Rows | Supports how many features? |
|---|---|---|
| 1 instrument, 5y daily | 1,250 | ~12 |
| 1 instrument, 10y daily | 2,500 | ~25 |
| 1 instrument, 5y hourly (FX, 24h) | ~43,800 | ~440 |
| 1 instrument, 5y 5-minute (equity) | ~98,000 | ~980 |
| **500 instruments, 5y daily** | **625,000** | **~6,250** |
| 500 instruments, 20y monthly | 120,000 | ~1,200 |
| 3,000 instruments, 20y monthly | 720,000 | ~7,200 |

There it is. **Your goal of thousands of variables is not unreasonable — it is simply incompatible with studying one instrument.** On a single daily series you can responsibly support about a dozen features. On a panel of 500 instruments you can support thousands, which is exactly what the published literature does: Gu, Kelly and Xiu used ~94 characteristics across ~30,000 stocks precisely because the cross-section is where the observations are.

This is the single most important structural decision in the whole 57 weeks, and it is a decision about **data shape**, not about models.

## III.3 From time series to panel: what actually changes

| | Now | Target |
|---|---|---|
| Unit of observation | one bar of one instrument | one instrument on one date |
| Rows | 1,250 | ~625,000 |
| Question | "Will EUR/USD go up tomorrow?" | "Which of these 500 instruments will outperform the others over the next month?" |
| Target | absolute direction | cross-sectional rank of forward return |
| Benchmark to beat | 50% coin flip | a 5-factor linear model, after costs |
| Supportable features | ~12 | thousands |
| What a win looks like | ~55% accuracy (unreachable) | monthly rank IC of 0.02–0.05, positive net of costs |

The cross-sectional question is genuinely easier, for a reason worth internalising. Asking "will this go up?" requires forecasting the market-wide component, which is close to a pure martingale. Asking "will this go up *more than that one*?" cancels the market-wide component — both instruments are exposed to it — and leaves the relative differences, where measurable structure actually lives. This is why essentially every successful systematic equity strategy is long-short rather than directional.

**Information Coefficient (IC)** becomes your core metric: the cross-sectional rank correlation between predicted and realised forward returns, computed per date and averaged. An IC of 0.03 is small, real, and genuinely useful. An IC of 0.30 means you have a leak.

## III.4 Accuracy is not money

Even a model with a genuine edge loses if it trades too often. Measured, with a deliberately skill-free signal, volatility-targeted and half-Kelly sized:

| Cost per trade | Gross Sharpe | Net Sharpe | Annual net return | Max drawdown |
|---|---|---|---|---|
| 0 bp | −0.18 | −0.18 | −0.17% | −1.6% |
| 1 bp | −0.18 | −0.47 | −0.44% | −1.9% |
| 3 bp | −0.18 | −1.05 | −0.99% | −3.2% |

The gross number never moves; the net number collapses. At 26.9 round-trips a year, three basis points costs you 0.87 Sharpe. Turnover is a first-class design parameter, not an afterthought — a monthly-rebalanced panel strategy is far more survivable than a daily-rebalanced one for exactly this reason.

Every result in this project is reported **net of costs** from Phase 1 onward. A gross Sharpe is a number you may look at while debugging; it is never a number you report or act on.

## III.5 The easier target: size instead of direction

Same bars, same features, same walk-forward with embargo, ten years of daily data. Out-of-sample R² against predicting the training mean:

| Instrument | R² predicting **size** (\|return\|) | R² predicting **direction** (signed return) |
|---|---|---|
| SAP | **+0.0312** | −0.0008 |
| TSLA | **+0.0116** | +0.0054 |
| EUR/GBP | **+0.0084** | +0.0005 |
| GBP/USD | **+0.0054** | −0.0000 |
| EUR/USD | **+0.0022** | +0.0002 |

Size is positive on all five. Direction is at zero and negative on two. This is the well-known asymmetry: volatility clusters and is therefore partially predictable, while the sign of returns is close to a martingale.

**Two honest caveats, both of which you must respect:**

1. **The baseline is soft.** R² is defined against predicting the training-set *mean*. For volatility that is an easy benchmark to beat, because volatility is persistent — simply reusing the trailing 20-day volatility already captures most of it. So +0.0312 means "beats guessing the average". It does **not** yet mean "beats trailing volatility", and only the second claim justifies building a model. That test is the first gate in Phase 1.

2. **Binarising destroys it.** Turning the magnitude into an above-or-below-median class and running the same classifier gave 50.5%–51.9% across all five instruments — no edge anywhere. The signal lives in the magnitude as a continuous quantity. Predict it as a number, with a regressor, scored by R². The moment you threshold it you throw the information away.

## III.6 What "all forms of data" actually requires

Your goal says all forms of data. Here is what each form costs you in discipline, because the cost is almost entirely about **point-in-time correctness** — knowing not just what a value was, but when you could first have known it.

| Data type | Example fields | Point-in-time hazard | Difficulty |
|---|---|---|---|
| Price / volume | OHLCV, spreads, turnover | Split and dividend adjustments are applied retroactively by vendors | Low |
| Derived technical | the ~30 you already compute, extended | None beyond the above — computed from price | Low |
| Cross-sectional | sector, size, liquidity rank, beta | Index membership is restated; delisted names vanish | Medium |
| Fundamental | earnings, book value, margins, accruals | **Severe.** Reported figures are restated; the publication date is not the period date | High |
| Macro | rates, inflation, PMI, yield curve | Releases are revised, sometimes substantially, months later | High |
| Market-implied | implied volatility, put/call skew, term structure | Needs options data; expensive but clean | Medium |
| Text / sentiment | filings, news, transcripts | Timestamp precision and survivorship of the archive | High |
| Alternative | satellite, card spend, web traffic | Short histories, licensing, usually unavailable to students | Very high |

The realistic plan for a 57-week student project: **master the top four rows, and only the top four.** That is comfortably hundreds of variables once you take the cross-product of base fields with transformations and horizons. Fundamentals are the first stretch goal and are where most of the point-in-time pain lives — do not touch them until Phase 2, and when you do, use a vendor that provides an explicit "as-reported" date.

The honest arithmetic on feature count: roughly 40 base series per instrument × roughly 8 transformations (level, change, z-score, rank, rolling mean, rolling std, momentum, acceleration) × roughly 4 horizons ≈ **1,280 features**. You reach "thousands of variables" through systematic transformation of a manageable base set, not by finding thousands of distinct data sources. That is also how the literature does it.

## III.7 The reformulated goal

**By 1 January 2028, build and honestly validate a cross-sectional machine-learning system that ingests a panel of roughly 500 instruments, constructs on the order of 1,000 point-in-time-correct features, predicts forward return rank and forward volatility, and reports performance net of costs with a confidence interval and an explicit count of the configurations tried.**

Compare against the original goal:

| | Original | Reformulated |
|---|---|---|
| Variables | hundreds to thousands | hundreds to thousands ✓ *(unchanged)* |
| All forms of data | yes | yes, four categories mastered properly ✓ |
| Used for testing | yes | yes, with a validation gauntlet ✓ |
| Data shape | one instrument | panel of ~500 — **changed** |
| Prediction | absolute direction | relative rank + volatility — **changed** |
| Success metric | accuracy | IC and net Sharpe with CI — **changed** |
| Reachable? | **no** | **yes** |

You keep everything you actually asked for. The two changes are the ones that move the target from inside the noise to outside it.

### Why this is still genuinely ambitious

Lest this read as lowering the bar: a working cross-sectional panel system with point-in-time-correct features, combinatorial purged validation, cost modelling and deflated significance testing is *more* technically demanding than the original plan, not less. It is what a junior quantitative researcher at a systematic fund actually does. The original plan was easier and impossible; this one is harder and possible.

---

# Part IV — The 57-week plan: 26 November 2026 → 1 January 2028

## IV.0 How this plan is built

Six phases. Every phase ends in a **gate**: a single numeric criterion you either pass or fail, written down before you start so you cannot move it afterwards. Every phase also has **kill criteria**: the conditions under which you stop that line of work and re-scope.

This is the mechanism that prevents failure. Not optimism — structure. The people who spend two years on something that never worked are the people who never wrote down, in advance, what "not working" would look like. If you fail a gate, the plan has succeeded at its job: it told you early, while you still have months to redirect.

**Time budget.** 1–3 hours per day, 7–21 hours per week, using the three-block protocol in Part V.1. The plan below marks reduced-load weeks around QMUL exams; respect them, because a missed exam costs more than a delayed phase.

**Calendar alignment.** Dates assume a start of Thursday 26 November 2026 and are indicative to within a few days. The QMUL notes assume your teaching year runs October 2026 – June 2027 with summer dissertation work; adjust if your year differs.

| Phase | Weeks | Dates | Theme | Load |
|---|---|---|---|---|
| **0** | 1–3 | 26 Nov – 16 Dec 2026 | Stop the bleeding | Normal |
| **1** | 4–12 | 17 Dec 2026 – 17 Feb 2027 | Data foundation and the decisive gate | Normal, light over holidays |
| **2** | 13–24 | 18 Feb – 12 May 2027 | The feature factory | Normal, light in late April |
| **3** | 25–34 | 13 May – 21 Jul 2027 | The model layer | **Reduced during exams** |
| **4** | 35–46 | 22 Jul – 13 Oct 2027 | The validation gauntlet | High — summer |
| **5** | 47–57 | 14 Oct 2027 – 1 Jan 2028 | Production, write-up, interviews | Normal |

---

## Phase 0 — Stop the bleeding
### Weeks 1–3 · 26 November – 16 December 2026

**Objective.** Make the existing pipeline incapable of lying to you. Nothing new is built; three things that are currently broken are fixed.

**Why first.** Every measurement you take from here forward depends on these. Building features on top of a pipeline that reports 92% on noise is building on sand.

| Week | Focus | Tasks |
|---|---|---|
| 1 | Leakage | Set `label_mode = "forecast"`. Add `leakage_guard()` above `build_forecast_dataset` and call it after `audit.to_csv(...)` so it raises instead of warns. Verify `--label return` now stops with an error naming `bullish_candle_pct`. |
| 2 | Data integrity | Add `validate_run_frame()` beside `prune_outlier_runs`. Call it *before* `_prepare_training_data` so bad rows never reach the saved history. Confirm it refuses the SAP 5d file at 81% and passes TSLA at 0%. Regenerate SAP answering **2 — floor at zero**. |
| 3 | Honest evaluation | Write `honest_eval.py`: time-ordered folds, 5-bar embargo, majority-class baseline, Wilson 95% interval, EDGE / NO EDGE verdict. Validate it returns NO EDGE on pure noise and EDGE on a planted signal. |

**Gate 0 — all four must hold:**

1. `python ml_model.py --label return` exits with a `Leakage:` error.
2. `python ml_model.py --label forecast` completes and reports test accuracy in [0.45, 0.55].
3. `validate_run_frame` rejects any batch with >5% impossible prices.
4. `honest_eval` returns NO EDGE on noise and EDGE on a planted signal.

**Kill criteria.** None — this phase is not optional and cannot fail. If it takes four weeks instead of three, take four.

**Deliverable.** A pipeline whose numbers you are willing to defend to a stranger.

**QMUL tie-in.** The leakage guard and the purged splits are the practical core of MTH793U / ECS784P. The `leakage_audit` AUC argument is a strong viva answer.

---

## Phase 1 — Data foundation and the decisive gate
### Weeks 4–12 · 17 December 2026 – 17 February 2027

**Objective.** Build the panel. Move from one instrument to roughly 500, with point-in-time discipline, and settle the single question that determines what the rest of the year looks like.

**Why now.** Part III showed that nothing works at n = 1,250. Everything downstream depends on having n in the hundreds of thousands. This phase is unglamorous and it is the phase that matters most.

| Week | Focus | Tasks |
|---|---|---|
| 4 | **The decisive test** | Re-run the magnitude R² test with trailing 20-day volatility as the competing prediction, not the training mean. This is Gate 1a and it decides the project's shape. |
| 5 | Universe | Define the instrument universe: ~500 liquid names with a documented inclusion rule. Record the rule, not just the list. Handle delisting explicitly. |
| 6–7 | Ingestion | Build a cached, resumable downloader. One Parquet file per instrument. Retry, rate-limit, log every failure. Never silently skip a symbol. |
| 8 | Survivorship | Audit the universe for survivorship bias. Can you obtain names that delisted during the window? If not, document the bias and quantify its likely direction. |
| 9 | Panel assembly | Build the long-format panel: one row per (instrument, date). Align calendars, handle holidays, forward-fill nothing silently. |
| 10 | Point-in-time | Build the as-of framework: every field carries the date it became knowable. Write the test that proves no field is readable before its knowable date. |
| 11 | Storage | Parquet with partitioning. Benchmark against the current CSV approach — expect >85% size reduction and a large query speedup. Your `processed_bars.csv` hit 1.6 GB at 50,000 runs; that approach does not scale. |
| 12 | Baselines | Implement the benchmarks every later model must beat: equal-weight, trailing-volatility forecast, momentum, 5-factor linear. |

**Gate 1a (week 4) — the fork in the road.** Does a model using volatility features beat trailing 20-day volatility out of sample, walk-forward, on at least 3 of 5 instruments?

- **Pass** → volatility forecasting is a live research line. Carry it through as a second prediction target alongside return rank.
- **Fail** → drop volatility *forecasting* as a research goal. Use trailing volatility directly for position sizing (it is already good enough) and put the full effort into cross-sectional return rank. **This is a success, not a failure** — it saves you months and it is the most likely outcome.

**Gate 1b (week 12) — all must hold:**

1. Panel contains ≥ 400 instruments × ≥ 1,000 dates ≥ **400,000 rows**.
2. Point-in-time test passes: no field readable before its knowable date.
3. Full panel rebuilds from scratch, unattended, in under 2 hours.
4. All four baselines produce a reported net-of-cost Sharpe with a confidence interval.

**Kill criteria.** If by week 10 you cannot assemble ≥ 200 instruments with clean history, stop and reduce scope to a single liquid sector or to an index-constituent set with a shorter window. A clean 200-instrument panel beats a dirty 500-instrument one.

**Deliverable.** A reproducible panel and a baseline table that every future model is measured against.

**QMUL tie-in.** MTH6139 (stationarity, alignment); the data engineering here is directly what ECS784P coursework rewards.

---

## Phase 2 — The feature factory
### Weeks 13–24 · 18 February – 12 May 2027

**Objective.** Go from ~30 features to ~1,000, systematically, without reintroducing leakage.

**Why now.** With the panel built, n/p finally permits it. Not before.

| Week | Focus | Tasks |
|---|---|---|
| 13 | Architecture | Design the feature registry: each feature declares its name, inputs, lookback, knowable-at offset, and unit. Features become data, not code. |
| 14 | Transformations | Implement the eight transformations: level, change, z-score, cross-sectional rank, rolling mean, rolling std, momentum, acceleration. Applied uniformly, this is where the count multiplies. |
| 15 | Price and volume | The base technical set, extended and verified. Port and prove your existing 30. |
| 16 | Volatility estimators | Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang. Part VI.4 has the formulas and their efficiency ratios. |
| 17 | Cross-sectional | Sector-relative, size-relative, beta-adjusted, liquidity-ranked. These only exist in a panel, and they are the ones that matter. |
| 18 | Stationarity | Fractional differentiation (López de Prado ch. 5): find the minimum d that passes ADF while retaining memory. Integer differencing throws away too much. |
| 19 | **Leakage sweep** | Run the single-feature audit across all ~1,000. Any feature separating the label alone is a bug. Quarantine and investigate every one. |
| 20 | Multicollinearity | VIF and correlation clustering. Technical indicators are heavily collinear; document the cluster structure rather than blindly dropping. |
| 21 | Stability | Feature stability across time and regime. A feature whose distribution shifts between folds will not generalise. |
| 22 | Macro | Yield curve, rates, inflation surprises — with their true release dates, not period dates. |
| 23 | Pipeline | Make feature computation incremental and cached. A full rebuild should be rare. |
| 24 | Audit | Full-panel feature audit report. Count, coverage, missingness, leakage status, stability, cluster membership. |

**Gate 2 — all must hold:**

1. ≥ 800 features computed across the full panel.
2. Zero features flagged by the leakage audit at the 0.80 threshold, or every flagged one individually explained and justified.
3. Feature computation is incremental: a one-day update takes < 5 minutes.
4. Every feature has a documented knowable-at offset.

**Kill criteria.** If the week-19 leakage sweep flags more than 10% of features and you cannot explain them, stop adding features and spend two full weeks on the cause. A thousand leaky features is worth less than thirty clean ones.

**Deliverable.** A versioned, audited feature store.

**QMUL tie-in.** Fractional differentiation and stationarity testing sit squarely in MTH6139. The multicollinearity work is standard ECS784P material.

---

## Phase 3 — The model layer
### Weeks 25–34 · 13 May – 21 July 2027 · *reduced load during exams*

**Objective.** Build models that can handle p ≈ 1,000, and find out whether they beat the Phase 1 baselines.

**Why now.** Models are the *easy* part and they come late deliberately. The ordering — data, features, then models — is the opposite of the instinctive one and it is the correct one.

| Week | Focus | Tasks |
|---|---|---|
| 25–26 | *Exam period* | **Reduced load.** Reading only: López de Prado chs. 4–7; Gu, Kelly & Xiu end to end. |
| 27 | Targets | Forward return rank (primary); forward volatility (if Gate 1a passed); triple-barrier labels (ch. 3). |
| 28 | Sample weights | Uniqueness weighting for overlapping labels (ch. 4). Overlapping forward returns badly inflate your effective sample size if untreated. |
| 29 | Linear | Ridge, Lasso, Elastic Net. With p ≈ 1,000 these are strong, fast and interpretable. Do not skip them because they are unglamorous — in the literature they are often competitive. |
| 30 | Trees | Gradient boosting (XGBoost, LightGBM — both now installed) with early stopping on a purged validation fold. |
| 31 | Neural | A small MLP. Modest expectations: Gu, Kelly & Xiu found NNs best, by a small margin, with vastly more data than you have. |
| 32 | Ensembling | Rank-average the model family. Ensembles beat individual models with remarkable consistency in this domain. |
| 33 | Interpretation | SHAP on the winner. Which feature families carry the signal? If the answer is incoherent, suspect a leak. |
| 34 | Selection | Pick one model family and freeze it. Record how many configurations you tried — you will need the count in Phase 4. |

**Gate 3 — all must hold:**

1. Best model achieves **mean monthly IC ≥ 0.02** out of sample.
2. IC is positive in **≥ 60% of individual months**.
3. It beats all four Phase 1 baselines on IC.
4. The configuration count is recorded honestly.

**Kill criteria.** If no model clears IC 0.02 by week 34, do **not** keep trying architectures — that is exactly the multiple-testing trap. Instead: go back to features (did Phase 2 build the wrong ones?) or lengthen the horizon (monthly is easier than weekly; quarterly easier still). Changing the model is the least likely fix and the most likely to produce a false positive.

**Deliverable.** A frozen model with an honest IC and a trial count.

**QMUL tie-in.** This is the dissertation's technical core. MTH773U / ECS784P in full.

---

## Phase 4 — The validation gauntlet
### Weeks 35–46 · 22 July – 13 October 2027

**Objective.** Try as hard as you can to prove your own result is fake. Whatever survives is real.

**Why now.** This is the phase that separates a student project from a professional one, and it is the phase beginners skip entirely.

| Week | Focus | Tasks |
|---|---|---|
| 35–36 | CPCV | Combinatorial purged cross-validation: many train/test paths, giving a *distribution* of performance rather than one number. |
| 37 | Deflated Sharpe | Correct the Sharpe ratio for the number of trials, skewness and kurtosis. Use the Phase 3 trial count. |
| 38 | PBO | Probability of backtest overfitting via combinatorially symmetric cross-validation. |
| 39 | Costs | Realistic spread, market impact, borrow costs for shorts. Your broker's actual numbers, not an assumption. |
| 40 | Capacity | At what size does your own trading move the price against you? Volume-participation limits. |
| 41 | Regime | Performance by volatility regime, by year, through 2020 and through 2022. Where does it break? |
| 42 | Ablation | Remove each feature family in turn. If performance is unchanged, those features were decoration. |
| 43 | Sensitivity | Perturb every hyperparameter. A result that survives only at one setting is noise. |
| 44 | Adversarial | Train a classifier to distinguish your training distribution from the live one. AUC near 0.5 is the goal. |
| 45 | Paper trading | Begin the locked, timestamped prediction log. It will not be conclusive by January — that is fine and expected. It is the start of a genuine track record. |
| 46 | Report | Write the honest validation report, including everything that failed. |

**Gate 4 — all must hold:**

1. **Deflated Sharpe > 0** after correcting for the trial count.
2. **PBO < 0.5.**
3. Net-of-cost Sharpe positive at your real cost assumption.
4. Positive IC in ≥ 2 of 3 volatility regimes.

**Kill criteria.** If deflated Sharpe ≤ 0, the Phase 3 result was multiple-testing noise. This is a **publishable, defensible finding** and an excellent dissertation — "I built the apparatus, applied it rigorously, and demonstrated that the apparent edge did not survive correction" is a genuinely strong result that most students never reach. Write it up as the finding it is.

**Deliverable.** A validation report that a sceptical professional could not easily dismantle.

**QMUL tie-in.** This *is* the dissertation. The viva bank in Part XI.3 maps directly onto it.

---

## Phase 5 — Production, write-up and the City
### Weeks 47–57 · 14 October 2027 – 1 January 2028

**Objective.** Turn the work into three artefacts: a running system, a dissertation, and a job.

| Week | Focus | Tasks |
|---|---|---|
| 47 | Automation | Daily pipeline: ingest, features, predict, log, report. Extend your existing `routine/daily_routine.py`. |
| 48 | Monitoring | Drift detection on features and predictions. Alert when the live distribution leaves the training distribution. |
| 49 | Reproducibility | Full environment lock, seed pinning, run cards. Anyone can rebuild any result. |
| 50 | Dashboard | Live IC, drawdown, feature drift, prediction distribution. |
| 51–53 | Dissertation | Write it. Methods, results, validation, limitations. The limitations section is where marks are won. |
| 54 | Code quality | Final pass: types, tests, documentation, security audit. |
| 55 | Portfolio | Public repository, README that leads with the honest result, reproducible notebooks. |
| 56 | Interviews | The Part XI.4 question bank. Practise explaining purged CV and deflated Sharpe out loud. |
| 57 | Close | Final report, retrospective, plan for 2028. |

**Gate 5 — all must hold:**

1. System runs unattended for 30 consecutive days.
2. Dissertation complete and submitted.
3. Public portfolio reproducible from a clean clone.
4. You can explain every component without notes.

**Deliverable.** A system, a dissertation, and the ability to defend both.

---

## IV.1 The three sentences that decide the year

Taped above the desk:

1. **"What is the smallest effect I could detect?"** — before every experiment.
2. **"How many things have I tried?"** — before every claim of significance.
3. **"What would make this result fake?"** — before every celebration.

## IV.2 Monthly review protocol

Last Sunday of each month, one hour, written into `Learn/progress_log.md`:

1. What gate am I working toward, and what is the number right now?
2. What did I measure this month that I did not know last month?
3. What did I try that failed? *(If nothing failed, you were not testing anything.)*
4. How many model configurations have I tried in total, cumulatively?
5. Am I still on the plan, or have I drifted into something more fun and less useful?

Question 5 is the one that catches most people. Drifting into simulator refinement because it is enjoyable and measurable, while the actual gate goes untouched for six weeks, is the single most likely way this plan quietly fails.

## IV.3 If you fall behind

You will. The plan has slack in it, and the order of sacrifice is fixed in advance so that you do not have to make the decision while stressed and behind:

1. **Never cut Phase 0 or Phase 4.** Honesty infrastructure at both ends is non-negotiable.
2. **Cut feature count before cutting panel size.** 400 features on 500 instruments beats 1,000 features on 100.
3. **Cut model variety before cutting validation.** One well-validated Ridge is worth more than five unvalidated neural networks.
4. **Lengthen the horizon rather than abandon the project.** Monthly prediction is materially easier than weekly, and quarterly easier still.

---

# Part V — The engineering spine

You are building with AI assistance. Done carelessly that produces more code than you can verify, which is worse than less code you understand. Done well it is a genuine multiplier. This part is the discipline that makes it the second thing.

## V.1 The three-block daily protocol

With 1–3 hours a day, intensity beats duration. Split every session:

| Block | Minutes | Activity |
|---|---|---|
| **A — Theory** | 20–30 | Read the equation. Derive it on paper. Understand the assumptions and when they break. No keyboard. |
| **B — Implementation** | 40–60 | Prompt with the contract template below. Review line by line. Never accept code you cannot explain aloud. |
| **C — Verification** | 20–30 | Write the test that would fail if B were wrong. Push the edge cases. |

Block C is the one people skip, and skipping it is what converts AI assistance from a multiplier into a liability. **If a session has to be cut short, cut Block B, not Block C.** Untested code is a liability on the balance sheet; no code is merely zero.

## V.2 Contract-driven prompting

Models default to permissive, agreeable, conversational code. Box them in with five tags:

| Tag | Purpose | Example |
|---|---|---|
| `[CONTRACT]` | Exact input and output types and shapes | `Accepts pd.DataFrame with DatetimeIndex; returns pd.Series[float] of identical length.` |
| `[INVARIANTS]` | Mathematical truths that must never be violated | `High >= max(Open, Close); Low <= min(Open, Close); price > 0.` |
| `[BANNED]` | Specific lazy shortcuts, named | `No .fillna(0), no bare except, no eval, no in-place mutation, no global state.` |
| `[FAIL-FAST]` | Raise rather than recover | `Raise ValueError naming both shapes if lengths differ. Never pad or truncate.` |
| `[TEST-FIRST]` | Tests before implementation | `Write the pytest asserting the invariants on 10,000 synthetic bars before the implementation.` |

Template:

```text
TASK: Implement the Yang-Zhang volatility estimator.

[CONTRACT]  yang_zhang(df: pd.DataFrame, window: int) -> pd.Series[float]
            df has columns Open, High, Low, Close and a DatetimeIndex.
            Output has the same index; the first (window-1) entries are NaN.

[INVARIANTS] Output >= 0 everywhere it is not NaN.
             On constant prices the estimator returns exactly 0.
             Result is within 20% of close-to-close vol on Gaussian data.

[BANNED]    No .fillna(). No bare except. No loops over rows — vectorise.
            No external packages beyond numpy and pandas.

[FAIL-FAST] Raise ValueError if window < 2 or any required column is missing.
            Raise ValueError if any price is <= 0.

[TEST-FIRST] First write a pytest covering: constant prices, a single bar,
             window longer than the frame, and a NaN in the middle.
```

**The adversarial follow-up.** After any result that pleases you, ask the model the opposite question. Not *"is this good?"* but:

> *"This reports 92% accuracy on data I generated with a random number generator. Give me the five most likely mechanical explanations, ranked by probability, and the exact test that would distinguish each."*

The first phrasing invites agreement. The second finds the leak. The difference between those two prompts is the difference between the project working and not.

## V.3 The anti-pattern watchlist

Review every generated diff against this list. Each item converts a loud failure into silent corruption:

| # | Pattern | Why it is dangerous |
|---|---|---|
| 1 | `try: ... except: pass` | Hides the crash instead of fixing it. Corruption continues downstream. |
| 2 | `.fillna(0)` on returns or prices | Zero is a *meaningful value* in finance. It is not "missing". |
| 3 | `if a == b:` on floats | Use `math.isclose(a, b, rel_tol=1e-9)`. |
| 4 | `df['y'] = df['Close'].shift(-1)` with no purge | Textbook look-ahead. |
| 5 | `StandardScaler().fit(X_all)` before the split | Test statistics leak into training. |
| 6 | `KFold(shuffle=True)` on time series | Trains on the future. |
| 7 | `for i in range(len(df))` | Usually 100× slower than vectorised; also more bug-prone. |
| 8 | Silently dropped assertions | The model removed a check to make the code "run". Always diff for deletions. |
| 9 | Misaligned Series arithmetic | Pandas aligns on index and fills NaN. Silent and destructive. |
| 10 | A changed function signature | Callers break, or worse, silently pass wrong positional arguments. |
| 11 | `random` without a seed | Irreproducible, therefore undebuggable. |
| 12 | Hardcoded paths or magic numbers | Belongs in `config.toml`. |

## V.4 Defensive decorators and the data gate

Enforce invariants at the boundary rather than hoping callers behave.

```python
import functools
import pandas as pd


def enforce_ohlc_invariants(func):
    """Reject physically impossible candles at the point they are produced."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        hi, lo = df["High"], df["Low"]
        body_hi = df[["Open", "Close"]].max(axis=1)
        body_lo = df[["Open", "Close"]].min(axis=1)
        assert (hi >= body_lo).all(), "High below the candle body"
        assert (lo <= body_hi).all(), "Low above the candle body"
        assert (hi >= lo).all(), "High below Low"
        assert not df.isna().any().any(), "NaN in generated output"
        if "Volume" in df:
            assert (df["Volume"] >= 0).all(), "Negative volume"
        return df
    return wrapper
```

The run-level gate, which is what would have caught the SAP corruption before it reached 2,000 rows of saved history:

```python
def validate_run_frame(df, max_drop_pct=5.0, max_abs_return_pct=500.0):
    """Drop impossible rows; refuse to train when too many are impossible."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    n0 = len(df)
    ok = pd.Series(True, index=df.index)
    for col in ("first_open", "last_close", "lowest_low", "highest_high"):
        if col in df.columns:
            ok &= pd.to_numeric(df[col], errors="coerce") > 0
    if "total_return_pct" in df.columns:
        ret = pd.to_numeric(df["total_return_pct"], errors="coerce")
        ok &= ret.notna() & (ret.abs() <= max_abs_return_pct)
    if {"highest_high", "lowest_low"} <= set(df.columns):
        ok &= df["highest_high"] >= df["lowest_low"]
    clean = df[ok]
    dropped = 100.0 * (n0 - len(clean)) / max(n0, 1)
    print(f"[gate] kept {len(clean)}/{n0} rows ({dropped:.1f}% dropped)")
    if dropped > max_drop_pct:
        raise ValueError(
            f"Data gate: {dropped:.0f}% of rows have impossible prices. "
            "Fix the generator rather than training on this."
        )
    return clean.reset_index(drop=True)
```

The design choice worth noting: it **raises** rather than silently returning a smaller frame. Dropping 81% of your data quietly is how you end up with a model trained on 377 rows while believing you have 2,000.

## V.5 Testing that is worth the time

Three tiers, in descending order of value per minute spent:

**Tier 1 — invariant tests.** Properties that must hold for all inputs. Highest value by a wide margin.

```python
import numpy as np
import pytest


@pytest.mark.parametrize("vol", [0.01, 0.1, 1.0, 10.0])
@pytest.mark.parametrize("start", [0.001, 1.0, 50_000.0])
def test_candles_are_physical(vol, start):
    df = generate_ohlc(n=500, vol_index=vol, start_price=start, allow_negative=False)
    assert (df["High"] >= df[["Open", "Close"]].max(axis=1)).all()
    assert (df["Low"] <= df[["Open", "Close"]].min(axis=1)).all()
    assert (df["Low"] > 0).all()
    assert np.isfinite(df.to_numpy()).all()
```

**Tier 2 — reproducibility tests.** The same seed gives bit-identical output. Without this you cannot debug anything intermittent.

```python
def test_seed_is_deterministic():
    a = generate_ohlc(n=1000, seed=42)
    b = generate_ohlc(n=1000, seed=42)
    pd.testing.assert_frame_equal(a, b)
```

**Tier 3 — statistical tests.** Generated distributions have the intended properties. Use a fixed seed and a generous tolerance, or you will write a flaky test that everyone learns to ignore.

```python
def test_jump_diffusion_has_fat_tails():
    r = merton_returns(n=100_000, lam=0.1, seed=7)
    assert scipy.stats.kurtosis(r) > 1.0, "jumps should produce excess kurtosis"
```

**The regression rule:** every bug you fix gets a test that would have caught it, written *before* the fix. Your leakage guard is one such test, promoted to a runtime check.

## V.6 Reproducibility

Non-negotiable, and cheap:

- **Seed everything.** `np.random.default_rng(seed)` over the global `np.random.seed()` — generator objects do not leak state between functions.
- **Pin the environment.** You have `requirements.txt` (floors) and `requirements-lock.txt` (exact). Regenerate the lock after every upgrade.
- **Write a run card.** Every training run emits JSON: git commit, config hash, seed, data range, row count, volatility regime percentile. Without it you cannot answer "what produced this number?" three months later.
- **Never edit results in place.** Append. A results file you have overwritten is a result you cannot reproduce.

## V.7 Presenting AI-assisted work honestly

Two failure modes, both fatal in a viva or interview:

- Claiming you hand-wrote everything. Two minutes of questioning exposes it, and it is the only unrecoverable answer.
- Saying "the AI wrote it". Signals you cannot defend the work.

The honest and genuinely strong framing:

> *"I used LLM tooling for implementation scaffolding. The system architecture, the mathematical specifications and the invariant contracts are mine. Every generated module sits inside automated boundary tests, static type checking and financial sanity assertions. The AI was treated as a junior developer whose code never merges without passing the test suite. The most important thing it did for me was help me find a data leak that had been inflating my accuracy from 50% to 92% — I can show you the measurement."*

That last sentence is worth more than any accuracy figure you could quote, because it demonstrates the one quality that is actually scarce: the willingness to find and report your own error.

---

# Part VI — The mathematics, and when you need it

Sequenced against the plan. Each block names the phase it is needed for; learning it earlier is wasted effort, learning it later means building on sand.

## VI.1 Discretising a stochastic differential equation — *Phase 1*

The continuous model you want to simulate:

$$dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$$

Three ways to step it forward, in increasing order of correctness:

**Euler–Maruyama** — the obvious one, and the source of your negative prices:

$$S_{t+\Delta t} = S_t + \mu S_t \Delta t + \sigma S_t \sqrt{\Delta t}\, Z, \qquad Z \sim \mathcal{N}(0,1)$$

A sufficiently negative $Z$ drives $S$ below zero. Strong convergence order 0.5.

**Milstein** — adds the second-order Itô correction:

$$S_{t+\Delta t} = S_t + \mu S_t \Delta t + \sigma S_t \sqrt{\Delta t}\,Z + \tfrac{1}{2}\sigma^2 S_t\left(\Delta t\, Z^2 - \Delta t\right)$$

Strong order 1.0. Still not positivity-preserving.

**Exact solution** — what you should actually use:

$$S_{t+\Delta t} = S_t \exp\!\left[\left(\mu - \tfrac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\,Z\right]$$

This is exact for GBM at any step size, and an exponential is **strictly positive**, so the SAP corruption in Part I.2 becomes structurally impossible. The $-\sigma^2/2$ term is the Itô correction: without it the expected price drifts upward spuriously, because $\mathbb{E}[e^X] = e^{\mu + \sigma^2/2}$ for $X \sim \mathcal{N}(\mu, \sigma^2)$.

**Why this matters beyond the bug.** Simulate in log space, always. Log-returns are additive, approximately stationary, symmetric in their treatment of gains and losses, and cannot produce a negative price. Essentially every error in Part I.2 follows from working in price space instead.

## VI.2 Jumps and mean reversion — *Phase 2*

**Merton jump-diffusion (1976).** Gaussian returns assign a 5-sigma daily move a probability of roughly once per 13,000 years; markets deliver them every few months. The fix is a compound Poisson jump term:

$$dS_t = (\mu - \lambda\kappa)S_t\,dt + \sigma S_t\,dW_t + (Y_t - 1)S_t\,dN_t$$

where $N_t$ is a Poisson process of intensity $\lambda$, $\ln Y_t \sim \mathcal{N}(\mu_J, \sigma_J^2)$, and $\kappa = \mathbb{E}[Y_t - 1] = e^{\mu_J + \sigma_J^2/2} - 1$ is the compensator that keeps the drift honest.

Discretised: draw $K \sim \text{Poisson}(\lambda \Delta t)$, and if $K > 0$ add $\sum_{i=1}^{K} \ln Y_i$ to the log-return.

Your `config.toml` already exposes `jump_prob` and `jump_size`. In the reformulated project their job is **tail risk and position sizing**, not direction prediction.

**Ornstein–Uhlenbeck.** The canonical mean-reverting process:

$$dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t$$

It has an exact discretisation, which you should use:

$$X_{t+\Delta t} = \mu + (X_t - \mu)e^{-\theta \Delta t} + \sigma\sqrt{\frac{1 - e^{-2\theta\Delta t}}{2\theta}}\,Z$$

Half-life of a shock: $t_{1/2} = \ln 2 / \theta$. This is the right model for spreads, for log-volatility, and for pairs — **not** for prices, which are not mean-reverting at daily frequency (Part III.1 measured exactly how not).

**Heston stochastic volatility (1993).** Volatility that is itself random:

$$dS_t = \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^{(1)}, \qquad dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^{(2)}$$

with $\text{corr}(dW^{(1)}, dW^{(2)}) = \rho$, typically around $-0.7$ for equities — the leverage effect, where falling prices coincide with rising volatility.

The **Feller condition** $2\kappa\theta > \xi^2$ keeps $v_t$ strictly positive. When it fails, use full-truncation: $v_{t+\Delta t} = \max(v_{t+\Delta t}, 0)$, and document that you did.

## VI.3 Wilder's smoothing is an EMA in disguise — *Phase 2*

A small derivation with a genuinely useful conclusion. Standard EMA:

$$\text{EMA}_t = \alpha X_t + (1-\alpha)\text{EMA}_{t-1}, \qquad \alpha = \frac{2}{N+1}$$

Wilder's recursive smoothing, used inside RSI and ATR:

$$W_t = \tfrac{1}{N_W} X_t + \left(1 - \tfrac{1}{N_W}\right)W_{t-1}, \qquad \alpha_W = \frac{1}{N_W}$$

Equate the smoothing weights:

$$\frac{2}{N+1} = \frac{1}{N_W} \implies N = 2N_W - 1$$

**A 14-period RSI has the memory of a 27-period EMA.** This is why RSI lags more than people expect, and it matters when you are choosing feature horizons: a "14-period" and a "27-period" feature that you believe are independent are very nearly the same number.

Implementation detail that bites: `pandas.ewm()` needs `adjust=False` to match the recursive definition, and `min_periods=N` to avoid emitting values during warm-up.

## VI.4 Volatility estimators — *Phase 2*

Close-to-close throws away the intraday range. Using High and Low recovers a great deal of information for free. Let $o = \ln(O/C_{-1})$, $u = \ln(H/O)$, $d = \ln(L/O)$, $c = \ln(C/O)$.

| Estimator | Formula (per bar) | Efficiency vs close-to-close | Handles drift? | Handles gaps? |
|---|---|---|---|---|
| Close-to-close | $r^2$, $r = \ln(C_t/C_{t-1})$ | 1× | — | yes |
| **Parkinson** (1980) | $\dfrac{1}{4\ln 2}\left[\ln(H/L)\right]^2$ | ~5× | no | no |
| **Garman–Klass** (1980) | $\tfrac{1}{2}\left[\ln(H/L)\right]^2 - (2\ln 2 - 1)\left[\ln(C/O)\right]^2$ | ~7.4× | no | no |
| **Rogers–Satchell** (1991) | $u(u-c) + d(d-c)$ | ~8× | **yes** | no |
| **Yang–Zhang** (2000) | $\sigma^2_{\text{overnight}} + k\,\sigma^2_{\text{open-close}} + (1-k)\,\sigma^2_{\text{RS}}$ | up to ~14× | **yes** | **yes** |

with $k = \dfrac{0.34}{1.34 + \frac{n+1}{n-1}}$.

"Efficiency 7×" means roughly the same statistical precision from 1 bar as close-to-close gets from 7. On a 1,250-row problem that is not a minor optimisation — it is equivalent to multiplying your sample size.

**Yang–Zhang is the default choice** for anything with overnight gaps, which is every equity. Use Rogers–Satchell for 24-hour instruments such as FX where the overnight term is meaningless.

## VI.5 Distances between distributions — *Phase 3*

When you need to ask "are these two distributions the same?":

**Kullback–Leibler divergence** — not a metric; asymmetric and unbounded:

$$D_{KL}(P \| Q) = \sum_x P(x)\ln\frac{P(x)}{Q(x)}$$

**Jensen–Shannon divergence** — the symmetric, bounded repair, and usually what you actually want:

$$\text{JSD}(P\|Q) = \tfrac{1}{2}D_{KL}(P\|M) + \tfrac{1}{2}D_{KL}(Q\|M), \qquad M = \tfrac{1}{2}(P+Q)$$

**Kolmogorov–Smirnov** — maximum gap between empirical CDFs; gives a p-value directly. Good for comparing streak-length distributions.

**Wasserstein (earth-mover)** — the cost of transporting one distribution onto the other. Better behaved than KL when supports differ, which they often do in the tails.

**Brier score** — for probability forecasts, mean squared error of the probability:

$$\text{BS} = \frac{1}{N}\sum_{i=1}^N (p_i - o_i)^2$$

A model that says 60% and is right 60% of the time is *calibrated*; one that says 95% and is right 60% of the time is not. Brier measures both discrimination and calibration, which is why it is the right score for a probabilistic trading signal and accuracy is not.

## VI.6 The metrics you will actually report — *Phase 4*

**Information Coefficient.** Per date, the Spearman rank correlation between predicted and realised forward returns; then averaged across dates.

$$\text{IC}_t = \text{corr}_{\text{Spearman}}\!\left(\hat{r}_{i,t},\, r_{i,t+1}\right)$$

Interpretation: 0.01–0.03 is weak but real and tradable at scale; 0.03–0.05 is good; above 0.10 on daily data means go looking for the leak.

**Information Ratio.** $\text{IR} = \overline{\text{IC}} / \sigma_{\text{IC}}$ — the consistency of the edge across periods, which matters more than its average size.

**Sharpe ratio.** $\sqrt{252}\cdot\mu_r/\sigma_r$ on daily returns. Always report it net of costs.

**Deflated Sharpe ratio.** Corrects for the number of configurations tried, plus skewness and kurtosis of the return series. With enough trials, a high Sharpe is the *expected* outcome of noise. Part VII.5 gives the procedure.

**Maximum drawdown and Calmar.** Drawdown is what actually ends careers and funds; a strategy with Sharpe 1.0 and a 40% drawdown is unusable in practice regardless of the Sharpe.

## VI.7 Correlation and the risk you did not price — *Phase 4*

Portfolio variance for weights $w$ and covariance $\Sigma$:

$$\sigma_p^2 = w^\top \Sigma\, w$$

For $n$ equally-weighted assets each with volatility $\sigma$ and uniform pairwise correlation $\rho$:

$$\sigma_p = \sigma\sqrt{\frac{1 + (n-1)\rho}{n}}$$

Put your own numbers in. With $n = 5$ and $\rho = 0$, portfolio volatility is $0.45\sigma$. With the measured $\rho = 0.85$ between GBP/USD and EUR/USD, those two are effectively **one position**, not two. Sizing them independently roughly doubles your true exposure to the dollar.

**Cholesky decomposition** generates correlated paths: factor $\Sigma = LL^\top$, draw independent $Z \sim \mathcal{N}(0, I)$, and set correlated shocks $= LZ$. Requires $\Sigma$ positive-definite; sample covariance matrices often are not when $n$ is large relative to the sample, which is what shrinkage estimators (Ledoit–Wolf) exist to fix.

---

# Part VII — Not fooling yourself: the validation statistics

This is the most important technical part of this file. Everything else is craft; this is what separates a result from a coincidence.

## VII.1 Why ordinary cross-validation destroys financial data

`KFold(shuffle=True)` is correct for independent observations and catastrophic for time series. Two separate failures:

**Look-ahead.** With 5 folds, when fold 3 is the test set, folds 4 and 5 are in training — and they lie *after* fold 3 in time. The model is trained on the future and tested on the past. Accuracy looks superb and means nothing.

**Overlapping labels.** Compute a 10-day forward return every day and consecutive rows share 9 of their 10 days. A row in training and a row in testing can be 90% the same observation. Your effective sample size is roughly a tenth of your nominal one, so every confidence interval you compute is roughly three times too narrow.

## VII.2 Purged k-fold with an embargo

The repair, from López de Prado chapter 7:

1. **Contiguous test folds**, in time order — never shuffled.
2. **Purge**: remove training rows whose *label window* overlaps the test fold.
3. **Embargo**: remove a further gap of rows immediately after the test fold, to absorb serial correlation that purging alone misses.

Your `purged_kfold_splits()` already implements this. The parameter that matters:

$$\text{embargo} = \text{label horizon} + k \times \text{feature lookback}, \qquad k \approx 2\text{–}3$$

Concretely: a 20-day forward label with 60-day features wants an embargo in the region of 20 + 2×60 = 140 bars. Your current `embargo_pct = 0.02` on 1,250 rows gives 25 bars, which is **too small** for any feature with a long lookback. Set it from the horizon, not as a fixed percentage.

## VII.3 Combinatorial purged cross-validation — *Phase 4*

Standard walk-forward gives you one backtest path, and therefore one number, and you cannot tell whether that number is typical or lucky.

CPCV splits the data into $N$ groups and tests on every combination of $k$ of them, purging and embargoing around each. With $N = 6$ and $k = 2$ you get $\binom{6}{2} = 15$ test combinations, which reconstruct multiple distinct backtest paths.

The output is a **distribution** of Sharpe ratios rather than a point estimate. You then report the median and the spread. A strategy whose Sharpe is 1.5 on one path and −0.3 on another is not a Sharpe-1.5 strategy; it is a coin flip you got lucky on once, and only CPCV shows you that.

## VII.4 The minimum detectable effect — compute this first, every time

Before any experiment, ask what you could see even if it were there. For a binary classification with $n$ test rows, at 5% significance and 80% power:

$$\text{MDA} = 0.5 + (z_{\alpha} + z_{\text{power}})\cdot\frac{0.5}{\sqrt{n}} = 0.5 + \frac{1.4}{\sqrt{n}}$$

| Test rows | Minimum detectable accuracy | Comment |
|---|---|---|
| 8 | 0.995 | your original 50-run batch — meaningless |
| 299 | 0.581 | 2,000-run batch |
| 640 | 0.555 | 5 years of real daily bars |
| 7,452 | 0.516 | 50,000-run batch |
| 50,000 | 0.506 | what a real panel gives you |

Inverted, to plan sample sizes:

$$n = \frac{(z_{\alpha} + z_{\text{power}})^2 \cdot 0.25}{\text{edge}^2}$$

| Target edge | Rows required | Days on one instrument |
|---|---|---|
| 60% | 196 | 0.8 years |
| 56% | 545 | 2.2 years |
| 54% | 1,225 | 4.9 years |
| 52% | 4,900 | **19.4 years** |

This single table is why Phase 1 builds a panel. Five hundred instruments turns 19.4 years into about a month of calendar time, because the rows come from the cross-section rather than from waiting.

## VII.5 Multiple testing and the deflated Sharpe ratio

**The problem.** Try 40 configurations on the same data and the best one looks significant *by construction*. The expected maximum of 40 draws from a zero-mean distribution is comfortably positive.

**Harvey, Liu & Zhu (2016)** concluded that, given the hundreds of factors already tested in the published literature, a newly proposed one needs roughly $|t| > 3.0$ rather than the conventional 2.0.

**The deflated Sharpe ratio** (Bailey & López de Prado) adjusts for exactly this. Given $N$ independent trials, the expected maximum Sharpe under the null is approximately

$$\mathbb{E}[\max SR] \approx \hat{\sigma}_{SR}\left[(1-\gamma)\,\Phi^{-1}\!\left(1 - \tfrac{1}{N}\right) + \gamma\,\Phi^{-1}\!\left(1 - \tfrac{1}{Ne}\right)\right]$$

with $\gamma \approx 0.5772$ (Euler–Mascheroni). The DSR then asks whether your observed Sharpe beats *that* benchmark, while also correcting for skewness $\gamma_3$ and kurtosis $\gamma_4$ of the return series:

$$\text{DSR} = \Phi\!\left(\frac{(SR - SR_0)\sqrt{T-1}}{\sqrt{1 - \gamma_3 SR + \frac{\gamma_4 - 1}{4}SR^2}}\right)$$

Verify the exact form against the paper before quoting it in a dissertation.

**The practical discipline, which matters more than the formula:**

- Keep a `trials.csv`. One row per configuration you *ever* evaluate. Append-only.
- Count honestly. Changing a hyperparameter is a trial. Changing a feature set is a trial. Re-running with a different seed and keeping the better one is a trial.
- Report the count alongside every result. "Sharpe 0.8 from 3 trials" and "Sharpe 0.8 from 300 trials" are completely different claims.

The single most common way a competent person fails at this: they try two hundred things, report the best, and genuinely do not think of it as cheating. Counting is the whole defence.

## VII.6 Probability of backtest overfitting

PBO, via combinatorially symmetric cross-validation: split the data into $S$ subsets, form all balanced train/test partitions, select the best configuration in-sample for each, and record its *out-of-sample rank*.

$$\text{PBO} = \Pr\left[\text{the in-sample best configuration ranks below median out of sample}\right]$$

PBO near 0 means in-sample selection carries genuine information. **PBO above 0.5 means your selection procedure is worse than random** — you are systematically picking the configurations that happen to be luckiest in-sample.

## VII.7 Adversarial validation

Train a classifier to tell training rows from live rows, using the same features.

- **AUC ≈ 0.50** — the two sets are statistically indistinguishable. Good.
- **AUC > 0.55** — they differ. Your model is extrapolating, and whatever you validated does not describe what it will meet in production.

`ml_model.py` already implements this (`adversarial_validation`, flag threshold 0.55). It is the correct early-warning system for regime drift, and it is also the honest test of whether your simulator produces realistic paths — on which the answer, so far, is that nobody has checked.

## VII.8 The pre-registration protocol

Adapted from Arnott, Harvey & Markowitz (2019). Before each experiment, write into `experiments/NNN.md` *before running anything*:

```markdown
# Experiment 014
Date: 2027-03-14
Hypothesis: cross-sectional momentum z-scores predict 20d forward return rank
Target: forward 20d return, cross-sectional rank
Features: momentum family only (47 features)
Model: Ridge, alpha in {0.1, 1, 10}
Validation: CPCV, N=6, k=2, embargo=140 bars
Success criterion: mean IC >= 0.02 with CI excluding zero
Trials used by this experiment: 3
Cumulative trials to date: 61
Prediction before running: IC ~0.015, probably fails
```

The last line is the one that trains your judgement. Writing down what you expect, before you see the answer, is the fastest way to find out whether you understand your own system. Within a few months you will be calibrated, and that calibration is worth more than any individual result.

---

# Part VIII — Security and operational safety

## VIII.1 Package hallucination

LLMs invent plausible package names. Attackers register the invented names on PyPI with malicious payloads — credential stealers, SSH key exfiltration, crypto miners. The attack is cheap because the model will keep recommending the same hallucinated name to thousands of people.

**The rule:** never install a package you have not verified. Check on PyPI:

- Does the project page exist and does it have a real repository link?
- Download counts in the millions, or at least high thousands?
- First release more than a year ago, with a regular release history?
- Does the repository have real issues, real contributors, real commit history?

Anything failing these is not installed. Your dependency set — numpy, pandas, scipy, scikit-learn, matplotlib, mplfinance, yfinance, joblib, pillow, python-dateutil, xgboost, lightgbm — is entirely mainstream, which is exactly how it should stay.

## VIII.2 Credentials

- Never paste a live API key, broker token or password into a chat window, a prompt, or a commit.
- `.env` for secrets, `.env` in `.gitignore`, always.
- Read through the environment and fail loudly when missing:

```python
import os

API_KEY = os.getenv("BROKER_API_KEY")
if not API_KEY:
    raise RuntimeError("BROKER_API_KEY is not set; refusing to continue.")
```

- Rotate anything that has ever been committed, even to a private repository, even if you removed it afterwards. Git history is forever and the removal commit is a signpost.

## VIII.3 The audit commands

Run before every submission and roughly monthly:

```bash
python -m pip install pip-audit && python -m pip-audit
python -m pip install bandit && python -m bandit -r stock_testing/ -ll
python -m pip install mypy && python -m mypy stock_testing/ --ignore-missing-imports
```

Note `python -m pip` throughout — on your machine `pip` is not on `PATH`, which is what broke your earlier install attempts.

## VIII.4 Operational safety

- **Never let a research script place a trade.** Research and execution are separate programs with separate credentials. A research script should not possess the ability to trade even if it tried.
- **Paper trade first, for longer than feels necessary.** Part III.4 shows how costs destroy a marginal edge; you will not see that in a backtest.
- **Rate-limit every data download.** Getting your IP blocked by a data provider in week 6 costs you weeks.
- **Back up.** The Learn folder this file replaced was **untracked by git** — three files, 341 KB, recoverable only because they were copied before deletion. Put `Learn/` under version control today.

---

# Part IX — The debugging playbook

## IX.1 Triage, in order

**1 — Lock the seed.** An intermittent bug is unfixable. Before anything else, make it reproducible.

```python
rng = np.random.default_rng(42)   # prefer a generator object over global state
```

**2 — Diff before you read.** `git diff` the file. The question is not "what does this code do?" but "what changed?" Look specifically for deleted assertions — the most dangerous AI edit is the one that removed a check to make an error go away.

**3 — Print shape and dtype at every boundary.** Most beginner failures in a pandas pipeline are shape or dtype problems:

```python
print(f"X {X.shape} {X.dtypes.unique()} | y {y.shape} {np.bincount(y)}")
```

**4 — Bisect the pipeline.** Not line by line — halve it. Does the bug exist at the feature stage? Before or after the join? Each bisection halves the search space; six of them narrow 1,000 lines to about 15.

**5 — Minimal reproducible example.** When asking for help, from a person or a model, never paste 500 lines. Paste the 15 that fail, the exact traceback, and the shapes:

```text
ValueError: operands could not be broadcast together with shapes (100,1) (100,)

def calculate_spread(high, low):
    return high - low

high: np.ndarray shape (100, 1)
low:  np.ndarray shape (100,)

Fix so the output is shape (100,). Do not use a loop. Explain which array
was the wrong shape at its source rather than patching at the point of error.
```

That last sentence matters. Fixing the symptom where it surfaced leaves the cause in place to resurface elsewhere.

**6 — Write the regression test, then fix.** The test comes first and must fail. A fix you cannot demonstrate is a fix you cannot trust.

## IX.2 Symptoms and their usual causes

| Symptom | First suspects |
|---|---|
| Accuracy far above 50% on random data | Label leakage; same-path features; shuffled CV |
| Training accuracy ≫ out-of-bag accuracy | Overfitting; n/p too small; no regularisation |
| Result changes between runs | Unseeded RNG; dict ordering; parallel non-determinism |
| NaN appearing mid-pipeline | Division by zero; log of a non-positive; misaligned join |
| Works on one instrument, fails on another | Hardcoded assumptions; different calendar; different scale |
| Slower than it should be | Row loop instead of vectorisation; repeated `.append`; no caching |
| Memory blows up | Full-file `read_csv` on a multi-GB file; groupby over 50,000 groups |
| A number that is "too good" | It is a bug. Every time. Treat it as a bug until proven otherwise. |

That last row is earned. Your 92% was a bug. The −0.337 five-year autocorrelation in Part III.1 is almost certainly a data bug. The 100% test accuracy on the 50-run batch was an 8-row test set. In this domain, a pleasant surprise is a defect report.

## IX.3 The memory failure you have already hit

On 20 October 2026 a 50,000-run leak-free training run died after the stock-similarity stage with a zero-byte log and no model. The probable cause is memory exhaustion while rebuilding indicators for 50,000 runs from a 1.6 GB `processed_bars.csv`, although the empty log means that is unconfirmed.

Three lessons, all of which Phase 1 addresses:

- **CSV does not scale.** Parquet with partitioning, read column-subsets, process in chunks.
- **Unbuffered logging for long runs.** `python -u`, or flush explicitly. A zero-byte log after an hour of compute is a total loss of information.
- **Checkpoint long jobs.** A run that dies at 95% and leaves nothing behind has cost you the entire runtime.

---

# Part X — The 100-idea roadmap, re-triaged against the measurements

The original roadmap contained 100 improvement ideas, about 60 of them aimed at "self-correcting accuracy via volatility tuning". Part I and Part III change the verdict on a large block of them. This is the honest re-triage.

**Read this before implementing any idea from the old roadmap.** Several of them are genuinely excellent engineering that serves no reachable purpose, which is the most expensive kind of work there is.

## X.1 The central reassessment

The old roadmap's thesis was a feedback loop:

> generate → analyse → score against real prices → adjust `vol_index` → repeat until the simulation matches reality → then the ML will find the signal

Each step works. The conclusion does not follow, for two measured reasons.

**First**, your own code already states the limiting result. From `price_modeling.py`:

> *"a symmetric random generator cannot beat 50% direction match by tuning volatility"*

That is correct and it is decisive. Volatility is a **scale** parameter. Direction agreement is a **sign** property. Tuning a scale parameter cannot systematically improve sign agreement, for the same reason that turning up the volume on a radio does not change the song. Roughly thirty ideas optimise a quantity that is mathematically incapable of moving the thing they were meant to move.

**Second**, even a perfect simulator would not help. Part III.1 measured the detection floor at 20,000–80,000 bars for signals of the size real markets contain. A perfect simulator does not add rows to your real dataset and does not enlarge the real signal. It changes neither term in the inequality.

## X.2 The triage

### DEAD — do not implement (≈ 28 ideas)

Everything whose objective is "tune generator parameters so simulated paths match a specific real path."

Includes: the volatility step-search; bisection over `vol_index`; the multi-metric equilibrium surface; per-cycle volatility learning; carry-over volatility history; realised-vol feedback; the volatility decay schedule; multi-run consensus volatility; population-based and Bayesian optimisation *of path match*; COBYLA on path match; iterative deepening grids; the exact-replica composite score; path-RMSE target bands; the automatic backtest re-run trigger; the convergence certificate; the volatility confidence interval; learning-rate scheduling for the volatility search; the GA over timeframe and interval; the null-hypothesis volatility benchmark.

**Why.** All of them optimise path agreement between a random path and one specific realised path. Two independent random walks with identical volatility have, in expectation, 50% direction agreement; the search converges on a number that is pure sampling noise and does not generalise to any other window.

**What to salvage.** The *optimisation machinery* is genuinely good and transferable. Bisection, COBYLA, Bayesian optimisation and genetic algorithms are exactly the right tools for hyperparameter search in Phase 3. Keep the code; point it at a different objective.

### REPURPOSED — implement, for a different reason (≈ 14 ideas)

| Idea | Original purpose | New purpose |
|---|---|---|
| Merton jump-diffusion | realistic paths → better prediction | tail risk, stress testing, position sizing (Phase 4) |
| GARCH / stochastic volatility | realistic paths | volatility forecasting baseline (Phase 1 Gate 1a) |
| Wick asymmetry, leverage effect | path realism | intraday range estimators as features (Phase 2) |
| Volume simulation, diurnal profile | path realism | liquidity and turnover features; cost modelling |
| Skew and empirical-quantile generators | path realism | fat-tailed scenario generation for stress tests |
| Intraday variance profile | path realism | time-of-day features for intraday extension |
| Correlation-aware multi-asset, Cholesky | multi-asset realism | **portfolio covariance for the panel — now central** |
| Regime change detection | adjust vol | regime-conditional validation (Phase 4, week 41) |
| Adversarial validation | prove sim looks real | **distribution-drift detection — now central** |
| Synthetic data augmentation | more training rows | calibration of the detection floor (Part III.1) |
| Conformal prediction | bound the vol search | genuine prediction intervals on forecasts |
| Monte Carlo averaging | stabilise path scores | bootstrap confidence intervals on all metrics |
| KL / ACF / streak-KS comparisons | match the simulator | feature-distribution drift monitoring |
| Ensemble of generators | voting on direction | model ensembling (Phase 3, week 32) |

### ESSENTIAL — implement as planned (≈ 32 ideas)

Unaffected by the reassessment, and most of them now matter more:

**Validation and leakage.** Purged k-fold with embargo; combinatorial purging; the leakage audit; direction-match as an ML label *(as a diagnostic, not a target)*; accuracy-weighted sample importance; sample uniqueness; probability calibration via isotonic regression; SHAP over impurity importance; feature-importance stability.

**Engineering.** TOML configuration; headless CLI; strict typing with mypy; the pytest invariant suite; deterministic seeding; structured logging; tqdm progress; data isolation hierarchies; run continuation; Pandera schema validation; multiprocessing; **Parquet storage**; MLflow experiment tracking; Docker; the HTML report; the daily advisor; disk hygiene.

**Analysis.** The 30-feature engineering pipeline *(now the seed of 1,000)*; non-linear transforms; VIF and multicollinearity; the model zoo; hyperparameter tuning; Kelly sizing; the information ratio; audit logging.

Parquet and multiprocessing move **up** in priority: a 500-instrument panel is roughly 500× the data volume, and your 1.6 GB CSV already caused a fatal memory failure.

### NEW — not in the original roadmap, and now load-bearing (12 ideas)

| # | Idea | Phase |
|---|---|---|
| 101 | Cross-sectional panel construction | 1 |
| 102 | Point-in-time as-of framework | 1 |
| 103 | Survivorship-bias audit | 1 |
| 104 | Information Coefficient as the primary metric | 1 |
| 105 | Feature registry with knowable-at offsets | 2 |
| 106 | Fractional differentiation for stationarity | 2 |
| 107 | Cross-sectional rank transformations | 2 |
| 108 | Deflated Sharpe ratio | 4 |
| 109 | Probability of backtest overfitting | 4 |
| 110 | Trial counting and pre-registration | all |
| 111 | Capacity and market-impact modelling | 4 |
| 112 | Locked, timestamped paper-trading log | 4–5 |

Ideas 101 and 110 are the two highest-value items in this entire file. Neither appeared in the original roadmap.

---

# Part XI — QMUL: modules, exams, viva and the City

## XI.1 Module mapping

| Module | Syllabus | Where this project meets it |
|---|---|---|
| **MTH6154 / MTH6155** Financial Mathematics I & II | Binomial models, GBM, Itô's lemma, risk-neutral measure, Black–Scholes | Part VI.1. The exact GBM discretisation and the $-\sigma^2/2$ Itô correction are directly examinable, and you have a bug report showing what happens without them. |
| **Stochastic calculus** (no QMUL module code verified; check the current module directory. The nearest confirmed modules are MTH6141 Random Processes and MTH6155 Financial Mathematics II) | Martingales, stopping times, SDEs, Girsanov, local/stochastic volatility | Part VI.2. Heston, the Feller condition, Ornstein–Uhlenbeck and its exact discretisation. |
| **MTH6139** Time Series | Stationarity, ARMA, ACF/PACF, ARCH/GARCH, spectral methods | Parts III.1 and VI.3–VI.4. Your AR(1) detection experiment *is* an applied ACF study with a power analysis attached. |
| **MTH793U / ECS784P** Machine Learning | Supervised classification, ensembles, regularisation, evaluation | Parts VII and the Phase 3 model layer. Purged CV and the leakage audit are well beyond the usual coursework level. |
| **MTH6100 / MTH700P** Research Project | Independent research, dissertation, viva | Phases 4–5 in full. |

## XI.2 Exam-ready formula sheet

**No-arbitrage, one period.** A market is arbitrage-free iff there exists a probability measure $\mathbb{Q}$ with $S_0 = e^{-rT}\mathbb{E}^{\mathbb{Q}}[S_T]$. For a binomial step with up factor $u$ and down factor $d$, the risk-neutral probability is

$$q = \frac{e^{r\Delta t} - d}{u - d}, \qquad \text{requiring } d < e^{r\Delta t} < u$$

**Itô's lemma.** For $f(t, S_t)$ with $dS = \mu S\,dt + \sigma S\,dW$:

$$df = \left(\frac{\partial f}{\partial t} + \mu S\frac{\partial f}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 f}{\partial S^2}\right)dt + \sigma S \frac{\partial f}{\partial S}\,dW$$

The second-order term survives because $(dW)^2 = dt$, which is the single fact that distinguishes stochastic from ordinary calculus.

**Black–Scholes call.**

$$C = S_0\Phi(d_1) - Ke^{-rT}\Phi(d_2), \qquad d_{1,2} = \frac{\ln(S_0/K) + (r \pm \sigma^2/2)T}{\sigma\sqrt{T}}$$

**Stationarity.** $\{X_t\}$ is weakly stationary if $\mathbb{E}[X_t] = \mu$, $\text{Var}(X_t) = \sigma^2$, and $\text{Cov}(X_t, X_{t+h})$ depends only on $h$. Prices are not stationary; log-returns approximately are. This is why every feature in Part II.2 is a transformation of returns rather than of prices.

**AR(1).** $X_t = \phi X_{t-1} + \varepsilon_t$ is stationary iff $|\phi| < 1$, with $\text{ACF}(h) = \phi^h$.

**GARCH(1,1).** $\sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2$, stationary iff $\alpha + \beta < 1$; persistence is $\alpha + \beta$, typically around 0.95–0.99 for daily equity data. The high persistence is precisely why volatility is forecastable and direction is not.

**Bias–variance.** $\mathbb{E}[(y - \hat{f})^2] = \text{Bias}^2 + \text{Var} + \sigma^2_{\text{irreducible}}$. In finance the irreducible term dominates overwhelmingly, which is why simple regularised models are competitive with elaborate ones.

## XI.3 Viva question bank

Prepare these aloud, not on paper.

1. *Your model reports 92% accuracy on random data. Explain.* — The flagship answer. Walk through the single-feature AUC, the shuffled-label control, and the forecast-mode fix. Lead with it; it demonstrates self-correction, which is the rarest quality in a student project.
2. *Why does ordinary k-fold fail on financial data?* — Look-ahead plus overlapping labels. Then purging and the embargo, with the sizing rule.
3. *Why does Euler–Maruyama produce negative prices, and the exponential form not?* — Additive versus multiplicative noise; $e^x > 0$ for all real $x$; your SAP corruption as the worked example.
4. *What is the Itô correction and where does it come from?* — $(dW)^2 = dt$; $\mathbb{E}[e^X] = e^{\mu + \sigma^2/2}$; without $-\sigma^2/2$ the drift is biased upward.
5. *You claim an edge. How do I know it is not luck?* — Minimum detectable effect, trial count, deflated Sharpe, PBO, CPCV distribution.
6. *Why predict volatility rather than direction?* — The measured R² table in Part III.5; GARCH persistence as the mechanism; and the caveat that trailing volatility is the real baseline.
7. *Why 500 instruments rather than one?* — The n/p arithmetic. This is the answer that shows you understand the problem structurally rather than procedurally.
8. *What is the Feller condition and what do you do when it fails?* — $2\kappa\theta > \xi^2$; full truncation; and say that you document it.
9. *Your Sharpe is 1.2. What is it net of costs?* — If you cannot answer instantly, the number is not real. Quote the measured cost table.
10. *What did you get wrong?* — Have three specific answers ready. The leak, the SAP negative prices, the 8-row test set. Candour here reads as competence, and evasion reads as inexperience.

## XI.4 City interview preparation

London desks — Jane Street, Citadel, Man AHL, Squarepoint, Marshall Wace, G-Research — filter hard on exactly the failures this file is about. What they actually test:

- **Do you respect non-stationarity?** Raw prices into a model is an immediate reject.
- **Are you paranoid about leakage?** The single most reliable signal of a real quant.
- **Can you size a bet?** Kelly, volatility targeting, the correlation arithmetic in Part VI.7.
- **Do you know what you do not know?** Confidence intervals, trial counts, honest limitations.

**The sentence that opens doors:** *"I had a model reporting 92% accuracy. I proved it was leakage, fixed it, and the honest number was 50%. Then I measured the smallest effect my setup could detect and found it was two orders of magnitude above anything in the data, so I redesigned the whole project around a cross-sectional panel."*

Almost no undergraduate can say that. It demonstrates measurement, scepticism and the willingness to discard your own work — which is the actual job.

**The claim that closes them:** *"I built an LSTM that predicts Bitcoin with 93% accuracy."* Every experienced practitioner hears "I have an undiagnosed leak."

---

# Part XII — Reference

## XII.1 Reading list, in the order you need it

**Before Phase 1**
- López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018) — **chapters 1–7 are the core of this project.** Chapter 7 is the single most important thing you will read all year.
- Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning", *Review of Financial Studies* 33(5) — the benchmark for what success looks like.

**During Phases 1–2**
- McKinney, *Python for Data Analysis* (3rd ed., 2022) — pandas and Parquet.
- Harvey, Liu & Zhu (2016), "...and the Cross-Section of Expected Returns" — multiple testing.
- Cont (2001), "Empirical properties of asset returns" — the stylised facts, concisely.

**During Phases 3–4**
- Géron, *Hands-On Machine Learning* (3rd ed., 2022) — model selection and tuning.
- Bailey, Borwein, López de Prado & Zhu, "The Probability of Backtest Overfitting".
- Arnott, Harvey & Markowitz (2019), "A Backtesting Protocol in the Era of Machine Learning".
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* — free PDF; regularisation chapters.

**Reference throughout**
- Viafore, *Robust Python* (2021) — typing and interfaces.
- Okken, *Python Testing with pytest* (2022).
- Lehalle & Laruelle, *Market Microstructure in Practice* (2013) — for the cost and capacity work.

**Free and online**
- statlearning.com (ISL) · hastie.su.domains/ElemStatLearn (ESL) · probml.github.io · r4ds.hadley.nz · scikit-learn.org/stable/user_guide.html

## XII.2 Glossary

**Adversarial validation** — training a classifier to distinguish two datasets; AUC ≈ 0.5 means they are indistinguishable.
**Alpha** — return not explained by known risk factors.
**Autocorrelation** — correlation of a series with its own lag. Measured here at −0.009 to −0.082 on daily returns.
**Backtest** — simulating a strategy on historical data. Optimistic by default, always.
**Base rate** — the unconditional frequency of an outcome; the thing your model must beat.
**Combinatorial purged CV (CPCV)** — purged cross-validation over many train/test combinations, yielding a distribution of results.
**Deflated Sharpe ratio** — Sharpe corrected for the number of trials, skewness and kurtosis.
**Detection floor** — the smallest effect a given sample size can distinguish from chance.
**Drawdown** — peak-to-trough decline. The number that ends careers.
**Embargo** — rows removed after a test fold to break serial correlation.
**Feller condition** — $2\kappa\theta > \xi^2$; keeps Heston variance positive.
**Fractional differentiation** — differencing by a non-integer order $d$ to achieve stationarity while retaining memory.
**Information Coefficient (IC)** — cross-sectional rank correlation between prediction and outcome. The primary metric from Phase 1 onward.
**Information Ratio** — mean IC divided by its standard deviation; consistency of edge.
**Itô correction** — the $-\sigma^2/2$ term; without it, simulated drift is biased upward.
**Kelly criterion** — optimal growth bet size $f^* = (pb - q)/b$. Use half-Kelly or less in practice.
**Leakage** — information in the features that would not have been available at prediction time.
**Look-ahead bias** — the commonest leakage: using data from after the decision point.
**Martingale** — a process whose best forecast of the future is its present value. Prices are close to this at daily frequency.
**Meta-labelling** — a second model that predicts whether the first model's signal should be acted on.
**Minimum detectable effect** — see detection floor. $0.5 + 1.4/\sqrt{n}$ for binary accuracy.
**Non-stationarity** — distributional properties changing over time. The defining difficulty of financial data.
**Overfitting** — fitting noise. Invisible in-sample by construction.
**PBO** — probability of backtest overfitting; above 0.5 means your selection is worse than random.
**Point-in-time** — data as it was knowable at the time, not as later restated.
**Purging** — removing training rows whose label window overlaps the test fold.
**Rank IC** — Spearman version of IC; robust to outliers and the usual choice.
**Regime** — a persistent market state (volatility level, trend). Models are regime-specific more often than people assume.
**Sharpe ratio** — excess return per unit volatility, annualised. Report net of costs or not at all.
**Shrinkage (Ledoit–Wolf)** — pulling a sample covariance matrix toward a structured target to make it well-conditioned.
**Slippage** — the gap between the price you expected and the price you got.
**Stationarity** — constant mean, variance and autocovariance structure over time.
**Survivorship bias** — a universe containing only the survivors. Inflates returns, sometimes dramatically.
**Triple-barrier labelling** — labelling by which of profit-take, stop-loss or time-limit is hit first.
**Turnover** — how much of the portfolio changes per period. The multiplier on your costs.
**Volatility clustering** — large moves follow large moves. Why volatility is forecastable.
**Walk-forward** — train on the past, test on the next block, slide forward.
**Yang–Zhang** — volatility estimator handling both drift and overnight gaps; up to ~14× the efficiency of close-to-close.

## XII.3 Checklists

### Before claiming any result

- [ ] Features and labels come from disjoint time windows.
- [ ] Preprocessing was fitted on training folds only.
- [ ] Validation is time-ordered with purge and embargo.
- [ ] The result beats the majority-class and the trailing-value baselines.
- [ ] A confidence interval is computed and the lower bound clears the baseline.
- [ ] The minimum detectable effect for this sample size is known.
- [ ] The cumulative trial count is recorded.
- [ ] Performance is reported net of realistic costs.
- [ ] The run is reproducible from the seed and run card.
- [ ] I have actively tried to prove the result is fake and failed.

### Before merging generated code

- [ ] I can explain every line aloud.
- [ ] No bare `except`, no `.fillna(0)`, no float `==`.
- [ ] No assertion was deleted — checked by diff, not by reading.
- [ ] Function signatures unchanged, or every caller updated.
- [ ] A test exists that fails without this change.
- [ ] No new dependency, or a verified mainstream one.
- [ ] No credential anywhere in the diff.

### Monthly

- [ ] `pip-audit`, `bandit`, `mypy` all run.
- [ ] `requirements-lock.txt` regenerated.
- [ ] `Learn/progress_log.md` updated with the five review questions.
- [ ] Trial count reconciled.
- [ ] Backups verified — **including `Learn/`, which was untracked until today.**

## XII.4 Prompt templates

**Implementation**

```text
TASK: <one sentence>
[CONTRACT]  <signature, types, shapes, index behaviour>
[INVARIANTS] <properties that must hold for all inputs>
[BANNED]    No .fillna(), no bare except, no row loops, no new dependencies.
[FAIL-FAST] Raise <Error> with specific values when <condition>.
[TEST-FIRST] Write the pytest covering <edge cases> before the implementation.
```

**Adversarial review** — use after every pleasing result

```text
This result pleases me, which makes me suspicious.
Context: <what you did>
Result: <the number>
Give the five most likely mechanical explanations that would produce this
WITHOUT any real predictive signal, ranked by probability. For each, give the
exact test that distinguishes it. Do not reassure me.
```

**Debugging**

```text
<exact traceback>
<the 15 relevant lines>
Shapes and dtypes: <...>
Expected: <...>. Got: <...>.
Identify which value was wrong at its SOURCE. Do not patch at the point
where the error surfaced.
```

## XII.5 Provenance

This file replaces three documents, which were backed up before deletion:

| Replaced | Lines | Fate |
|---|---|---|
| `QMUL_FM_STUDY_PLAN_AND_AI_DEBUGGING_GUIDE.txt` | 2,646 | Mathematics → Part VI; module mapping and viva → Part XI; 34-week plan → superseded by Part IV |
| `ROADMAP_AND_SECTION_B_STUDY_GUIDE.md` | 2,546 | 100 ideas → re-triaged in Part X; engineering guidance → Part V |
| `VIBECODING_SAFEGUARDS_AND_SECURITY.md` | 196 | → Parts V, VIII, IX, expanded |

**What changed substantively, and why:**

1. The 34-week plan became a 57-week plan aimed at a different, reachable target. *Part III.1 measured that the original target was below the detection floor.*
2. About 28 roadmap ideas were marked dead. *A scale parameter cannot move a sign property; your own code says so.*
3. Direction prediction was demoted and cross-sectional rank promoted. *The n/p arithmetic in Part III.2.*
4. Every phase gained a numeric gate and kill criteria. *The absence of these is the main mechanism by which projects like this consume years.*
5. Parts VII (validation statistics) and X.2's "NEW" block are almost entirely additions. *They are the parts that decide whether any result is real.*

**The standing instruction, if you read nothing else twice:** when a number pleases you, that is the moment to go looking for the bug. It was true of 92%, it was true of 100% on eight test rows, and it will be true of whatever pleases you in 2027.

---

*Last measured 20 October 2026. Every figure in Parts I and III is reproducible with the scripts named beside it. Simulated price paths are random by construction and contain no market signal. Nothing in this document is investment advice.*
