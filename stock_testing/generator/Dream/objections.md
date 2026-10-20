# Objections

The hard questions, and your answers. Written as you go, not reconstructed at the end.

Rule: when you make a non-obvious choice, write the question an examiner or interviewer would
ask, and your answer. If you cannot answer it in four sentences, you do not yet understand
your own decision.

Status key: **solid** (can defend cold) · **shaky** (needs work) · **open** (no answer yet)

---

## On the result

### "Your accuracy was 92% and now you say it is 50%. Which is it?"
**solid.** Fifty. The 92% came from `label_mode = "return"`, where features and label were both
computed over the same finished price path — `bullish_candle_pct` separated the label alone at
AUC 0.92. The proof it was mechanical: the identical model on randomly permuted labels scored
0.491. Switching to a forecast label, with features from the first 70% of bars and the label
from the last 30%, gave 0.435–0.545 across eight model families. Fifty is the correct answer on
random walks, and getting it is the evidence that the pipeline is now honest.

### "So your project failed?"
**solid.** The original target failed and I measured why, which is a result rather than a
failure. The detection floor says a signal of the size real instruments carry — lag-1
autocorrelation under 0.09 — needs 20,000 to 80,000 training bars. Daily data gives 1,250.
That is not a modelling problem and no algorithm fixes it. The project reformulated to a
cross-sectional panel, where the observations come from the cross-section instead of from
waiting 79 years.

### "Why should I believe your detection floor?"
**solid.** It is calibrated against a known answer. I injected a chosen autocorrelation into
simulated paths and measured recovery against a fixed 40,000-bar test set. At 100,000 training
bars the recovered accuracy matched the theoretical oracle to within 0.003 — 0.5357 against
0.5333 at φ = −0.10. The instrument measures correctly, so I trust what it says at smaller
sample sizes.

### "Isn't a 0.02 information coefficient meaningless?"
**solid.** It is small and it is the right order of magnitude. Gu, Kelly and Xiu (2020) used
roughly 94 characteristics across roughly 30,000 stocks over 60 years and reported out-of-sample
monthly R² well under 1%. Anyone claiming much more on daily liquid data has a leak. An IC of
0.02 becomes economically meaningful through breadth — hundreds of instruments — not through
magnitude.

---

## On method

### "Why purged cross-validation rather than ordinary k-fold?"
**solid.** Two reasons. Shuffled k-fold puts most training folds after the test fold in time, so
the model trains on the future. And overlapping forward-return labels mean consecutive rows share
most of their information, which makes confidence intervals roughly three times too narrow.
Purging removes training rows whose label window overlaps the test fold; the embargo removes a
further gap for serial correlation.

### "How did you size the embargo?"
**shaky — fix in Phase 1.** It should be the label horizon plus two to three times the feature
lookback. The default `embargo_pct = 0.02` gives 25 bars on 1,250 rows, which is too small for a
60-day feature. *To do: compute it from the horizon rather than as a fixed percentage.*

### "Why cross-sectional rather than time-series?"
**solid.** Arithmetic and economics. Arithmetic: 1,250 rows supports about 12 features, while
500 instruments over 5 years gives 625,000 rows and supports thousands — which is what the stated
goal requires. Economics: predicting absolute direction requires forecasting the market factor,
which is close to a martingale. Predicting relative rank cancels the market factor and leaves the
differences, where measurable structure lives.

### "Why predict magnitude rather than direction?"
**solid.** Measured on the same bars, same features, same walk-forward: out-of-sample R² for
move size was +0.0022 to +0.0312 across five instruments, all positive. For direction it was
−0.0008 to +0.0054, essentially zero and negative twice. Volatility clusters and is partly
predictable; the sign of returns is close to a martingale. One caveat I tested — binarising the
magnitude destroys it: SAP's 0.031 R² became a 50.7% coin flip as an above-or-below-median class.

### "Your magnitude R² beats the mean. Does it beat trailing volatility?"
**open — this is Gate 1a.** The honest answer today is that I do not know. R² is defined against
the training mean, which is a soft benchmark for a persistent quantity. The real competitor is
trailing 20-day volatility. If the features do not beat it, the correct conclusion is to use
trailing volatility directly for sizing and drop volatility forecasting as a research line.

---

## On engineering

### "How do you know you have no look-ahead now?"
**shaky — hardening through Phase 1.** Currently: a single-feature AUC audit that raises rather
than warns, a forecast-mode label that cuts each path, and purged folds with an embargo. Planned:
a knowable-at timestamp on every field, a test that corrupts all data after date T and asserts
features up to T are unchanged, and a shuffled-label canary in CI failing above 0.55.

### "Why did 81% of one dataset have negative prices?"
**solid.** The simulator was run in a mode allowing negative prices, which is legitimate for some
futures and wrong for an equity. Runs are chained — run N+1 opens where run N closed — so once a
path crossed zero every later run inherited a broken start. Two fixes: answer "floor at zero" for
equities, and a data gate that refuses to train when more than 5% of rows are impossible. The
underlying cause is working in price space; the exact GBM solution is an exponential and cannot
go negative.

### "You used AI to write this. What did you actually do?"
**solid.** I specified the architecture, the mathematics and the invariant contracts; generated
code sits inside boundary tests, type checking and sanity assertions. The most valuable thing it
did was help me find a leak that had inflated accuracy from 50% to 92% — and I can show you that
measurement and the guard that now stops it. I treat generated code as a junior developer's
output: never merged without passing tests.

---

## On the dream

### "Why should a fund hire you rather than an Imperial PhD?"
**shaky — strengthen through 2027.** Honestly, for a research seat they often will take the PhD.
What I offer is a finished, reproducible, honestly validated system built end to end — data,
features, validation, costs — with a trial log and a confidence interval, plus a year of locked
timestamped predictions. Most candidates at any level cannot show a trial count or a deflated
Sharpe. *To do: make this concrete by having the artefact finished.*

### "What would you do differently?"
**solid.** Build the panel first. I spent months improving a simulator when the binding constraint
was sample size. The correct order is data, then features, then models; I did it backwards and the
measurement that showed me is the detection floor.

### "What is the weakest part of your work?"
**solid.** Right now, that I have not yet demonstrated a real edge on real data — only that my
previous apparent edge was fake and that the current setup could not detect a real one. The
validation apparatus is ahead of the result. That ordering is deliberate but it means the headline
is still outstanding.

---

## Open items

- [ ] Gate 1a: does volatility forecasting beat trailing volatility? (week 4)
- [ ] Embargo sizing from label horizon rather than percentage
- [ ] Confirm the cause of the 50,000-run memory failure
- [ ] Investigate the EUR/GBP −0.337 five-year autocorrelation — almost certainly a data artefact
- [ ] Prepare the ten-minute project narrative and rehearse it aloud
