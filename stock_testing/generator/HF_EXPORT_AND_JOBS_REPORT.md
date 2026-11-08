# HF dataset export and training handoff — built and measured; job submission blocked on billing

**Date:** 2026-10-30 · **Repo:** `stock_testing/generator` · **HF repo:** `ay0k200/sim-archive` (private dataset)

**Status: the exporter, trainer and job package are written, the dataset is pushed, the round-trip
and leakage checks pass, and the weighted-vs-uniform measurement is done locally. The HF Jobs
submission is refused with HTTP 402 — an account billing state that only you can clear (STOP-6).**

**The headline number changed after a bug in my own trainer.** The first run reported R² +0.83. That
was the target predicting itself. Corrected, R² is **negative**. See §0.5.

This report is written in layers and the layers are kept, because two of my conclusions in it turned
out to be wrong. How each was caught is worth more than a tidy summary. §0 is where things stand;
§1b, §1, §2, §3 and §5 are the history that led there.

---

## 0. Final status

### 0.1 What shipped

| File | What it does |
|---|---|
| `export_hf_dataset.py` | reads the archived `analysis_summary` snapshots, computes `sample_weight`, writes three Parquet splits under `dataset/`, pushes to the private repo |
| `train_weighted.py` | loads the splits, fits `HistGradientBoostingRegressor` twice (weighted / uniform), reports the out-of-sample R² delta with a bootstrap 95% CI |
| `hf_job/train_weighted.py` | the same trainer plus `_upload_artifacts`, which pushes `metrics.json`, `model.pkl` and `training.log` to `runs/<job_id>/` when `HF_JOB_ID` is in the environment |
| `hf_job/requirements-hf.txt`, `hf_job/run.sh`, `hf_job/README.md` | job packaging |
| `submit_hf_job.py` | builds and runs the `hf jobs uv run` command; refuses GPU flavors outright |
| `requirements-hf.txt` | HF dependencies, kept out of `requirements.txt` per the negative constraint |

### 0.2 STOP-6 — the exact command and the exact error

The command, verbatim:

```bash
hf jobs uv run --flavor cpu-upgrade --env SIM_ARCHIVE_REPO=ay0k200/sim-archive --env BACKEND=sklearn --env MODEL_OUT=/results --env HF_DATASET_PREFIX=dataset --timeout 3600s --detach --with datasets --with scikit-learn --with joblib --format json hf_job/train_weighted.py
```

The error, verbatim:

```
Error: Client error '402 Payment Required' for url 'https://huggingface.co/api/jobs/ay0k200'
Pre-paid credit balance is insufficient - add more credits to your account to use Jobs.
```

Diagnosis, in the order it was established:

1. `hf auth whoami` returns `ay0k200` — auth is fine, and no re-login was attempted.
2. `hf jobs ps -a` returns "No results found" without error — the Jobs API is reachable and the
   token carries the right scope. The failure is not connectivity or permissions.
3. `hf jobs uv run --help` has no confirmation prompt and no `-y` flag — so exit 1 was not a
   suppressed interactive prompt.
4. `--dry-run` exits **0** and prints a valid job configuration — so the command itself is
   well-formed. The failure is server-side, at submission.
5. Re-running with full output capture surfaced the 402. **The error arrives after the uploader's
   tqdm progress bars**, which is why `submit_hf_job.py` hid it: it printed the first 900
   characters of the output, and those are entirely the configuration block. Fixed — it now prints
   the last 12 lines and names a 402 explicitly.

**This is not retryable by me.** Clearing it means adding pre-paid credits at
<https://huggingface.co/settings/billing>, which is a payment action and yours to take. Once the
balance is positive, `python submit_hf_job.py --repo ay0k200/sim-archive` runs unchanged.

A side note from the error output: HF reports *"Your script and Job artifacts will be saved in this
bucket: huggingface.co/buckets/ay0k200/jobs-artifacts"*. So artifacts would land in a bucket as
well as in `runs/<job_id>/`. The explicit upload is kept, because the bucket is not the dataset
repo and the data card points at the repo.

### 0.3 Verification

| | Result |
|---|---|
| V1 round-trip | **PASS** — 198 rows read back via `load_dataset` with `data_files={split: "dataset/<split>.parquet"}` (206 scored source rows minus 8 purged at the split boundaries) |
| V2 weights | **PASS** — every split carries non-null weights; train min 0.2079, max 0.6911, 123 distinct, 0 nulls; all within (0, 4.0] |
| V3 leakage | **PASS** — no name from `ml_model.LEAKAGE_COLS` among the exported features |
| V3b self-leakage | **FAIL then PASS** — the resolved target was also a feature on the first run; fixed, re-measured (§0.5) |
| V4 job artifacts | **BLOCKED** — no `job_id` exists, so `runs/<job_id>/` was never written. The upload code is in place and unexercised. |
| V5 delta + CI | **PASS, measured locally** — see §0.5 |

### 0.4 Weights, as exported

The `--scored-only` policy you chose drops rows with no `accuracy_index` rather than inventing a
value for them. That is what makes the splits usable:

| | rows | non-null weights |
|---|---|---|
| train | 123 | 123 |
| validation | 37 | 37 |
| test | 38 | 38 |

`export_hf_dataset.py --scored-only` reported *"kept 206 weighted rows, dropped 794 unscored"*.
Pushed to <https://huggingface.co/datasets/ay0k200/sim-archive/tree/main/dataset>.

### 0.5 The measurement, and the correction that changed it

**CORRECTION 2 (my own bug, caught after the first run).** The first training run reported:

```
R2 weighted +0.8320 | uniform +0.8452 | delta -0.0132 CI95 [-0.0349, +0.0029]
```

`metrics.json` listed the features as `vol_index, total_return_pct, realised_vol_per_bar_pct,
accuracy_index, bar_direction_match_pct` — and the target as `total_return_pct`. **The target was
in the feature matrix.** `_resolve_target` falls back from the all-null `total_return_pct_label` to
the bare `total_return_pct`, but `_feature_cols` only excluded the `*_label` names, so the fallback
reintroduced the target as a feature. The model was reading the answer. R² +0.83 on 123 rows and 5
features should have looked implausible to me immediately; it is the same shape of mistake as the
`label_mode = "return"` leakage that made the original ML accuracy read 85–97%.

Fixed in both trainer copies: once the target is resolved it is removed from the feature list and a
note is recorded in `metrics.json`. Re-run, honestly:

```
backend=sklearn target=total_return_pct features=4 train=123 eval=37 weight[min=0.2079 max=0.6911]
R2 weighted -0.4045 | uniform -0.3216 | delta -0.0829 CI95 [-0.2114, +0.0804] -> straddles zero
```

Two things to read off this:

1. **R² is negative on both fits.** The model does worse than always predicting the training mean.
   Four run-level summary statistics over 123 rows do not predict a run's total return. That is
   consistent with everything else measured in this project — the detection-floor work, the ~50%
   walk-forward accuracy — and it is the honest answer, not a failure of the pipeline.
2. **The weighting delta straddles zero** (−0.0829, CI95 [−0.2114, +0.0804]). The accuracy-derived
   weights carry no demonstrable information at this sample size. The interval is 0.29 wide on 37
   evaluation rows, so the correct reading is "not detectable here", not "proven absent".

The deliverable asked for the delta and its CI. Both exist and both are reported. What they say is
that the weighting scheme cannot be shown to help on 123 training rows.

### 0.6 What is still open

| | Needs |
|---|---|
| HF Jobs run, `job_id`, `runs/<job_id>/` artifacts | credits on the HF account, then one unchanged command |
| `total_return_pct_label` / `future_vol_pct_label` | Prompt D has never run, so both label columns are null and the trainer falls back to a bare column |
| `data_source` | still absent upstream, so `row_mult` is always 1.0 and the 3:1 real-vs-sim ratio stays inapplicable (§5) |
| sample size | 206 scored rows total. More backtest runs (`python loop_runner.py --iterations 50 --mode backtest --cycles 10`) would narrow the CI |

### 0.7 Reproduction

The whole chain, from export to measurement, in three commands:

```bash
python export_hf_dataset.py --analysis-dir looped_analysis --repo ay0k200/sim-archive --scored-only --replace
python train_weighted.py --repo ay0k200/sim-archive --backend sklearn --out model_out
python submit_hf_job.py --repo ay0k200/sim-archive
```

The third fails with the 402 above until the account has credits. The first two are what produced
every number in §0.3–§0.5.

---

## History: the two stops, in the order they were found

---

> ### CORRECTION (same session, after the stop was written)
>
> **The first version of this report was wrong.** It concluded `accuracy_index` was absent
> *everywhere*, based on the 24 Oct snapshots. The 27 Oct snapshots in `cloud_archive/local/`
> contain **500 rows each, all `is_backtest=1`, with `accuracy_index` AND `bar_direction_match_pct`
> present** — plus `backtest_accuracy.csv` and `rewind_accuracy_summary.csv` (501 rows, columns
> `accuracy_index`, `score_direction`, `bar_direction_match_pct`, `vol_index`).
>
> I caught this because `git status --untracked-files=all` listed `backtest_accuracy.csv.gz`, which
> cannot exist without a scored backtest. My own reflection had flagged exactly this risk
> ("I sampled 6 of 58 snapshots"), and the risk materialised.
>
> **Revised measurement across both 27 Oct snapshots — 1,000 rows:**
>
> | Quantity | Value |
> |---|---|
> | `accuracy_index` | min −0.5904, median −0.2636, max 0.4232, **206 distinct**, **794 NaN of 1,000** |
> | computed `sample_weight` | min 0.2048, median 0.3682, max 0.7116, **206 distinct** |
> | weights vary? | **YES** — so weighted ≠ uniform, and the C15 delta is measurable |
> | V2 gate | min > 0 PASS · max ≤ 4.0 PASS · sum > 0 PASS · **no NaN FAIL (794)** |
>
> **The real stop is therefore STOP-4 (NaN in `sample_weight`), not "inputs absent".** ~206 of 1,000
> rows carry a usable `accuracy_index`; the other 794 are unscored cycles where the column is NaN.
>
> **The spec does not define a NaN policy, and this needs your decision**, because every option
> changes what the dataset means:
> 1. **Drop NaN rows** → ~206 clean, genuinely weighted rows. Smallest and most honest.
> 2. **Floor NaN at 0.05** → keeps 1,000 rows, but asserts "unscored = worst possible", which is a
>    claim about data that was never measured. Arguably the fabrication the spec forbids.
> 3. **Two-column export** → keep all rows, emit `sample_weight` as null where unscored, and let the
>    trainer drop them. Defers the choice to training time.
>
> What does **not** change: `data_source` is still absent in every snapshot, so `row_mult` is always
> `1.0` and the 3:1 real-vs-sim ratio remains inapplicable. The reflection in §5 stands on that point.

## 1b. SECOND BLOCKER — the ordered split puts every weight in `test`

The null-weight policy was implemented and the export ran clean. V1 passes. But a structural
problem surfaced that **no NaN policy can fix**:

| split | rows | non-null weights | run range |
|---|---|---|---|
| train | 600 | **0** | 1–300 |
| validation | 180 | 6 | 311–400 |
| test | 180 | **180 (all of them)** | 411–500 |

Every scored row sits in **runs 398–500** — the tail of the run order. `accuracy_index` is written
only for cycles the self-correction actually scored, and those are the most recent windows of each
backtest run, so they cluster at the high end of `run`.

A 60/20/20 split **by run order** therefore hands all usable weights to `test`, which the spec
forbids touching until final evaluation. **Train has zero weighted rows, so weighted training is
impossible as specified.**

Three ways out, none of which I took unilaterally because each bends a spec rule:

1. **Stratify on weight availability** — keep time order *within* each split but ensure scored rows
   appear in all three proportionally. Conflicts with "60/20/20 by run order, NOT random".
2. **Export scored rows only** — 186 rows, split 60/20/20 → ~111/37/37. Every split then has
   weights. Coherent but small, and discards 774 unscored rows that are still valid features.
3. **Generate more backtest data** so scored rows span the whole run range, then re-export.
   `python loop_runner.py --iterations 50 --mode backtest --cycles 10`.

### Verification status

| | Result |
|---|---|
| V1 round-trip | **PASS** — 960 rows (1,000 source − 40 purged) via `load_dataset(..., data_files="dataset/*.parquet")` |
| V2 weights | **PASS on value, FAIL on placement** — 206 distinct, min 0.2048, max 0.7116, all ≤ 4.0, but 0 in train |
| V3 leakage | **PASS** — no `ml_model.LEAKAGE_COLS` name among the exported features |
| V4 job | not submitted — training cannot run with an unweighted train split |
| V5 delta + CI | not measurable yet, for the same reason |

### Two further bugs found and fixed while building

- `pd.Series(pd.NA, dtype="float64")` raises `TypeError` in pandas 3.0; absent numeric columns now
  use `np.nan`.
- `upload_large_folder` is **not** a top-level export in `huggingface_hub` 2.4.0 — it is an `HfApi`
  method. The spec's Section 2 snippet would have failed on import. Now called as
  `HfApi(token=...).upload_large_folder(...)`, and only in the large branch.
- The repo doubles as the 58-snapshot archive, so Parquet at the repo root made `load_dataset`
  auto-detect the archive (26 rows, no `train` split). The dataset now lives under `dataset/`.

## 1. The first blocking finding (superseded by the CORRECTION above)

The deliverable's headline metric is the weighted-vs-uniform R² delta with a 95% CI (C15, V5). On this dataset **that delta is exactly 0 by construction**, because every sample weight is identical.

### Weight formula inputs, measured

```
base      = clip((accuracy_index + 1) / 2, 0.05, 1.0)
row_mult  = 3.0 if data_source == "real" else 1.0
sample_weight = base * row_mult
```

| Input | Present? | Evidence |
|---|---|---|
| `accuracy_index` | **absent everywhere** | not in local CSV, not in any archived snapshot, **0 mentions in `analysis_schema.json`** |
| `data_source` | **absent everywhere** | not in local CSV, not in archived snapshots |
| existing `sample_weight` | present but constant | `min=1.0 max=1.0 nunique=1` on a 50-row snapshot |

Neither factor can be computed. `base` has no input; `row_mult` has no input and is therefore always `1.0`.

### Why `accuracy_index` is missing

It is written by `backtest_compare.py` when a run is scored against real prices. Sampling **6 of 58 archived snapshots: 1,050 rows, 0 with `is_backtest=1`.** Every run in this archive is a *forward simulation*, which is never scored, so the column is never produced.

### What the existing `sample_weight` actually is

`price_modeling.py:1721`

```python
table["sample_weight"] = np.where(table["is_augmented"] == 1, VOLATILITY_SEARCH["aug_weight"], 1.0)
```

It is an **augmentation flag**, unrelated to accuracy. With no augmented rows it collapses to constant `1.0`. Passing it to `.fit(X, y, sample_weight=w)` yields the identical estimator to `.fit(X, y)`.

**Consequence:** V2 would *pass* (min > 0, max ≤ 4.0, no NaN) while being meaningless, and C15/V5 would report `delta = 0.000` with a CI of `[0, 0]`. The pipeline would work perfectly and measure nothing.

---

## 2. Secondary findings

| # | Finding | Severity | Fix |
|---|---|---|---|
| S1 | `looped_analysis/analysis_summary.csv` **no longer exists locally** — archived to HF and deleted by the retention workflow between my first and second check. `processed_bars.csv` is now `.gz`. | Blocking as written | Spec says "Reads: looped_analysis/analysis_summary.csv". Source must become the 58 archived snapshots (or `--restore` first). |
| S2 | Spec says `realised_vol_per_bar_pct`; the column is `realized_vol_per_bar_pct` (American spelling). | Would silently null a real column | Rename in the schema. |
| S3 | `bar_direction_match_pct` absent (same backtest-only reason as `accuracy_index`). | Degrades schema | Emit null + record. |
| S4 | Both `*_label` columns absent — Prompt D has not run. | Expected; spec handles it | Emit null + record. |
| S5 | `datasets` not installed (now declared in `requirements-hf.txt`, not installed), so V1 (`load_dataset` round-trip) cannot run. | Blocks V1 | `python -m pip install datasets` (belongs in `requirements-hf.txt`, not `requirements.txt`). |
| S6 | Row scale: ~20 local, ~1,160 across the archive (58 × ~20–50). Spec sizes `cpu-upgrade` for "a few hundred thousand rows". | Over-provisioned | A 60/20/20 split of ~1,160 rows is fine, but the merge rule will trigger per-snapshot. |

---

## 3. The one upstream fix

`accuracy_index` requires **backtest** runs, not forward runs:

```bash
python loop_runner.py --iterations 50 --mode backtest --cycles 10
```

`[loop] mode = "backtest"` with `cycles` produces rows where `backtest_compare.py` scores each cycle against real Yahoo prices and writes `accuracy_index` (and `bar_direction_match_pct`). Once those exist, `base` varies per row, weights become genuinely non-uniform, and the weighted-vs-uniform comparison becomes a real measurement rather than an identity.

`data_source` is a separate gap — nothing in the pipeline currently writes it, so the 3:1 real/sim multiplier has no source column at all and would need one added upstream.

---

## 4. Checklist — final

| | Item | Status |
|---|---|---|
| C1–C6 | `export_hf_dataset.py` and its outputs | **CONFIRMED** — three Parquet splits under `dataset/`, pushed to the private repo |
| C7–C9 | `train_weighted.py` | **CONFIRMED** — runs, writes `model_out/metrics.json` and `model_out/model.pkl` |
| C10 | `submit_hf_job.py` | **CONFIRMED** — written, builds a valid command (`--dry-run` exits 0) |
| C11–C12 | `job_id`, `runs/<job_id>/` artifacts | **BLOCKED** — HTTP 402, no credits on the account (§0.2). Upload code written, never exercised. |
| C13 | Round-trip row count | **CONFIRMED** — 198 rows (206 scored − 8 purged) |
| C14 | No `LEAKAGE_COLS` in features | **CONFIRMED** — filtered in the exporter and again in the trainer |
| C14b | Target not among features | **CONFIRMED after a fix** — it was, on the first run. See §0.5. |
| C15 | Weighted-vs-uniform delta + CI | **CONFIRMED** — −0.0829, CI95 [−0.2114, +0.0804], straddles zero |
| C16 | `requirements.txt` untouched | **CONFIRMED after a fix** — I had added `huggingface_hub` to it earlier this session, which this spec forbids. Reverted with `git checkout --`, now byte-identical to HEAD; the dependency lives in the new `requirements-hf.txt`. |
| C17 | LICENSE / README.md byte-identical | **VACUOUS** — neither file exists, in this folder or at the repo root. `git diff --exit-code LICENSE README.md` exits nonzero because git errors on unknown paths, not because anything changed. The spec's required final command therefore cannot exit 0 on this repo. |
| C18 | Nothing committed | **CONFIRMED** — no `git commit` run |
| C19 | `stress_loop.py` / `abload.py` excluded | **CONFIRMED** — neither referenced |
| C20 | No GPU flavor | **CONFIRMED** — `cpu-upgrade` only; `submit_hf_job.py` raises on any GPU flavor prefix |
| C21 | Repo private | **CONFIRMED** — `private=True` on create; never flipped |
| C22 | Token never logged or written | **CONFIRMED** — read from env or the cached login, passed to `HfApi(token=...)` only; absent from the data card and from every file written |

Hard stop reached: **STOP-6**, on submission. Not a defect in the command — the account has no
pre-paid credit for Jobs. Everything upstream of submission is done and verified.
---

## 5. Reflection

### What this prompt should have asked

1. **"Does `accuracy_index` exist in the data, and does it vary?"** The entire weighting design rests on it, yet the spec asserts the formula without ever checking its input. One `grep accuracy_index analysis_schema.json` (result: 0) would have caught this before any effort was allocated.
2. **"Are these rows forward simulations or backtests?"** `accuracy_index` is only produced by scored backtests. The spec treats the dataset as homogeneous when `is_backtest` cleanly partitions it — and every row is on the unscored side.
3. **"Where does the source CSV actually live now?"** The spec names a local path, but the archive workflow it credits as "proven" is precisely what deletes that path. Those two statements contradict each other, and the contradiction was live during this session.
4. **"What is the minimum row count for the delta to be detectable?"** ~1,160 rows with a bootstrap CI over 1,000 resamples will produce a wide interval; the spec specifies the CI machinery but never a power threshold.

### Assumptions I made that could be wrong

1. **I sampled 6 of 58 snapshots, not all 58**, to check for backtest rows. If backtest runs exist in the other 52, `accuracy_index` may be present there and the conclusion weakens. Full verification costs ~52 more downloads; I stopped at 6 because `analysis_schema.json` documents no such column at all, which is stronger evidence than any sample.
2. **I assumed `sample_weight` being constant across one 50-row snapshot generalises.** It is derived from `is_augmented`, which is only set when `n_aug > 0` in `[volatility_search]`; the default is `n_aug = 0`. So the constancy is structural, not sampling luck — but I verified it on one snapshot, not all.
3. **I assumed the spec wants a meaningful delta rather than a pipeline that runs.** If the goal is to have the plumbing in place ready for weighted data later, then building it against uniform weights is legitimate and my stop is wrong. That is a judgement call I have flagged rather than made unilaterally.


### What the self-leak should have taught me earlier (added after CORRECTION 2)

The +0.83 R² survived a full verification pass because V3 asked the wrong question. It checked the
features against `ml_model.LEAKAGE_COLS` — a fixed list of known-bad names — and passed, because
`total_return_pct` is not on that list. It is not *inherently* a leakage column; it only became one
the moment `_resolve_target` chose it as the target.

The lesson is that a name-based leakage check cannot catch a leak created at runtime. Two cheap
guards would have:

1. **Assert the target is not in the feature list**, right after resolution. Now implemented.
2. **Treat an implausibly good score as a defect to investigate, not a result to report.** On 123
   rows with 4 features predicting a simulated run's return, anything above about R² 0.2 should
   have triggered a check of the feature list before the number went into a report. The project
   already had this exact lesson once, in the 85–97% ML accuracy that turned out to be
   `label_mode = "return"`. I did not apply it to my own output.

This is the second wrong conclusion in this document. The first came from sampling too few
snapshots; this one from trusting a passing check that was not testing what I thought it was.
### On the 3:1 real-vs-sim ratio — directly addressed

**It is not defensible on this dataset, and not because of the ratio.** The spec asks whether 3:1 holds once simulated rows dominate. The answer is sharper than that: **there are no real rows and no `data_source` column to identify them.** Sampled 1,050 rows, 0 backtests. The multiplier is inert — every row takes `1.0`.

Had real rows existed, a *fixed share of total weight* would be the better construction. A constant 3:1 per-row multiplier means the real block's influence falls as simulated rows accumulate: at 100 real and 1,000 simulated, real carries 300/1300 ≈ 23%; at 100 real and 100,000 simulated it carries 0.3%. Since this pipeline generates simulated rows by the thousand per campaign and real rows arrive one per instrument per day, the ratio silently decays toward irrelevance. Targeting, say, "real rows hold 25% of total weight" and solving for the multiplier each export keeps the intent stable as the mix shifts.

**Follow-up required:** real-vs-sim row counts cannot currently be computed at all, because no column distinguishes them. Adding `data_source` upstream is a prerequisite for any weighting scheme that treats them differently.

---

## 6. Deliverables

Superseded by §0.1, which lists what was actually produced. The "not written — blocked" entries
that stood here belonged to the first layer of this report, before the null-weight and scored-only
decisions unblocked the build.

Working tree left uncommitted, as required.

## 7. Reproduction

See §0.7 for the three commands that reproduce the whole chain. The command below is kept because
it reproduces the *original* blocking finding on a 24 Oct snapshot — useful only as a check that
the forward-simulation snapshots really do lack the scoring columns:

```bash
python -c "import pandas as pd,gzip;from huggingface_hub import hf_hub_download;p=hf_hub_download('ay0k200/sim-archive','20261024_162852_EUR-USD_5d_seed2026102/looped_analysis/analysis_summary.csv.gz',repo_type='dataset');d=pd.read_csv(gzip.open(p,'rt',errors='replace'));d.columns=[c.strip() for c in d.columns];print('accuracy_index' in d.columns, d['sample_weight'].nunique(), d['is_backtest'].sum())"
```

Expected: `False 1 0` — no `accuracy_index`, one distinct weight, zero backtest rows. The 27 Oct
snapshots behave differently, which is what CORRECTION 1 is about.
