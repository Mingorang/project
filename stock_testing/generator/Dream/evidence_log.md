# Evidence Log

Things you have **proven you can do**, with the artefact that proves it.

Not a CV and not aspirations. A CV says "experienced in machine learning". This says
"measured a detection floor against a known-answer control, here is the script". The second
is what survives an interview.

Rule: nothing enters this file without a reproducible artefact. If you cannot point at the
file or the command, it is not evidence yet.

---

## Proven

### Found and proved a data leak in my own work
**Date:** 2026-10-20
**Claim:** Identified that a reported 85–97% test accuracy was mechanical, not predictive.
**Evidence:** Identical model on permuted labels scores 0.491 against 0.926. Single-feature
AUC of 0.92 for `bullish_candle_pct`. Clean 4,000-path control: 0.884 AUC with the feature on
the same path, 0.510 with the feature on earlier bars.
**Artefact:** `leak_demo.py`; `leakage_audit.csv` in each results directory.
**Why it matters:** This is the single most convincing thing on the list. Most people never
find their leak, and finding your own is the skill the job actually requires.

### Measured a detection floor against a known-answer control
**Date:** 2026-10-20
**Claim:** Quantified the smallest signal this pipeline can resolve at a given sample size.
**Evidence:** Injected autocorrelation recovered to within 0.003 of the theoretical oracle at
100,000 training bars. At 1,250 bars nothing is detectable, not even φ = −0.35. Real instruments
all sit below 0.09, implying 20,000 to 80,000 bars required.
**Artefact:** `detection_limits.py`, `sample_size_curve.csv`.
**Why it matters:** Almost nobody computes this. It converts "is it working?" into an inequality.

### Demonstrated that direction is unpredictable where magnitude is not
**Date:** 2026-10-20
**Claim:** On identical bars, features and validation, move size has positive out-of-sample R²
on all five instruments while direction has none.
**Evidence:** Size +0.0022 to +0.0312; direction −0.0008 to +0.0054. Plus the finding that
binarising the magnitude destroys it (0.031 R² to a 50.7% coin flip).
**Artefact:** `r2_comparison.csv`, `target_comparison.csv`.

### Built an honest walk-forward evaluator
**Date:** 2026-10-20
**Claim:** Time-ordered folds with embargo, majority baseline, Wilson interval, explicit verdict.
**Evidence:** Returns NO EDGE on pure noise (0.513, CI 0.488–0.539) and EDGE on a planted signal
(0.585, CI 0.560–0.610). Calibrated in both directions.
**Artefact:** `honest_eval.py`.

### Quantified cost sensitivity
**Date:** 2026-10-20
**Claim:** Showed that a no-skill signal's net Sharpe collapses with realistic costs.
**Evidence:** −0.18 gross; −0.47 at 1bp; −1.05 at 3bp, at 26.9 round trips per year.
**Artefact:** `pnl_report.py`.

### Shipped a working multi-model ML pipeline
**Date:** before 2026-10-20
**Claim:** Eight model families, purged k-fold, SHAP-style contributions, adversarial validation,
conformal bounds, HTML reporting, 2,000 and 50,000-run batches.
**Artefact:** `ml_model.py`; `training_runs/2026-10-20_eur_usd_5y/`.

### Built a polyglot pipeline that agrees across languages
**Date:** 2026-10-20
**Claim:** Python and R analyses of the same 50 CSV/PNG pairs agree on `total_return_pct` to 5e-14.
**Artefact:** `analyze_looped_results.R`, `market_analysis.py`, the comparison in the training-run README.

---

## In progress

| Claim | Blocked on | Phase |
|---|---|---|
| Built a 500-instrument point-in-time panel | universe + ingestion | 1 |
| ~1,000 audited features, zero leaky | feature registry | 2 |
| Honest IC with a confidence interval | model layer | 3 |
| Deflated Sharpe with a real trial count | CPCV + trials.csv | 4 |
| A year of locked timestamped predictions | start the log | 4 |
| Reproducible public repository | write-up | 5 |

---

## The three-sentence version

For the top of a README, or the first thing you say when asked about the project:

> I built a synthetic price simulator and an ML pipeline, then discovered my own 92% accuracy
> was a data leak — the features and the label were measured over the same bars. I rebuilt the
> validation honestly, measured the detection floor against a known-answer control, and showed
> that daily direction on a single instrument is unreachable with the data that exists: the
> signal is roughly two orders of magnitude below what 1,250 observations can resolve. The
> project is now a cross-sectional panel, where the observations come from the cross-section
> instead of from waiting 79 years.

That paragraph will get more respect from a practitioner than any accuracy figure, because
every sentence is falsifiable and every one has a script behind it.
