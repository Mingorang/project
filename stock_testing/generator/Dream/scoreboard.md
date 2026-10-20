# Scoreboard

One hour, last Sunday of every month. Append a new row; never edit an old one.
An edited scoreboard is a scoreboard that lies to you.

## The numbers that matter

| Metric | Start (2026-10-20) | Target by 2028-01-01 | Where it comes from |
|---|---|---|---|
| Panel rows | 1,250 | 600,000 | `len(panel)` |
| Instruments | 5 | ~500 | panel density report |
| Audited features | 30 | ~1,000 | feature audit report |
| Leaky features (AUC ≥ 0.80) | 10 | 0 | `leakage_audit.csv` |
| Data corruption (rows with price ≤ 0) | 81% worst set | < 1% | `validate_run_frame` |
| Mechanical accuracy gap | +0.41 | ~0.00 | `routine/daily_routine.py` |
| Honest out-of-sample IC | n/a | ≥ 0.02 | CPCV |
| Deflated Sharpe | n/a | > 0 | Phase 4 |
| Break-even cost (bp) | n/a | state it | cost sensitivity curve |
| Cumulative trials | unknown | logged | `trials.csv` |
| Ideas complete | 0 / 310 | — | `python ideas.py --stats` |
| Gates passed | 0 / 6 | 6 | manual Part IV |

## Monthly log

Copy this block each month.

```
### YYYY-MM

Phase: _          Gate in progress: _          On plan? yes / drifted

Numbers
  panel rows ______   instruments ______   features ______   leaky ______
  IC ______ (CI ______)   deflated Sharpe ______   cumulative trials ______
  ideas done ______ / 310

Measured this month that I did not know last month
  -

Failed this month  (if this is empty, I tested nothing real)
  -

Hours on-gate vs off-gate: ______ / ______

Could I explain this month's work to a sceptic?  yes / no

Next month's single most important thing
  -
```

---

### 2026-10 (baseline)

Phase: 0 not yet started. Gate in progress: none. On plan: yes, plan written.

**Numbers.** 1,250 rows, 5 instruments, 30 features, 10 leaky, IC n/a, trials unknown, 0/310 ideas.

**Measured this month that I did not know last month.**
- Stored 85–97% test accuracy is label leakage; shuffled labels score 50%.
- 81% of the SAP 5d rows have a price at or below zero; runs chain, so corruption propagates.
- Real-bar walk-forward direction accuracy is 48.6–51.6% across five instruments, inside the ±3.9pp luck band.
- The detection floor: at 1,250 training bars nothing is detectable, not even an autocorrelation of −0.35. Real instruments are all below 0.09.
- Magnitude is predictable where direction is not: R² +0.0022 to +0.0312 versus −0.0008 to +0.0054.
- Binarising the magnitude target destroys the signal (SAP 0.031 R² collapsed to a 50.7% coin flip).

**Failed this month.** The 50,000-run leak-free retrain died with a zero-byte log and no model, probably memory while rebuilding indicators from a 1.6 GB file. Unconfirmed.

**Could I explain this month's work to a sceptic?** Yes — every figure has a reproduction command.

**Next month's single most important thing.** Phase 0, week 1: promote the leakage audit from a warning to an exception and set `label_mode = "forecast"`.
