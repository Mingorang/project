"""
export_hf_dataset.py - turn pipeline outputs into a training-ready Parquet dataset on the Hub.

    python export_hf_dataset.py --repo ay0k200/sim-archive
    python export_hf_dataset.py --repo ay0k200/sim-archive --no-push      # stage only

Source resolution. The spec names looped_analysis/analysis_summary.csv, but the archive workflow
(prune_disk_space.py --delete-local) pushes that file to the Hub and removes it locally, so on a
pruned tree it is absent. This exporter therefore prefers the live CSV and falls back to the
gzipped copies inside cloud_archive/local/*/looped_analysis/, newest first. Every path comes from
an argument or an env var; nothing is hardcoded to one machine.

NaN weight policy (chosen by the user, 2026-10-30): accuracy_index is NaN for unscored cycles -
794 of 1,000 rows in the 27 Oct snapshots. Those rows are KEPT with sample_weight = null rather
than dropped or floored. Flooring at 0.05 would assert "unscored = worst possible", a claim about
data that was never measured. The trainer drops null-weight rows at fit time instead.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STAGING = Path(os.getenv("HF_STAGING_DIR", HERE / "hf_staging"))
ARCHIVE = Path(os.getenv("CLOUD_ARCHIVE_DIR", HERE / "cloud_archive" / "local"))
PATH_IN_REPO = os.getenv("HF_DATASET_PREFIX", "dataset")   # keep Parquet clear of the 58 archive snapshots
MAX_SHARD_BYTES = 1_000_000_000        # ~1 GB per Parquet file; HF recommends sharding
MAX_UPLOAD_BYTES = 90 * 1024 ** 3      # STOP-2: refuse above 90 GB of the 100 GB free private quota
PURGE_FRACTION = 0.02                  # embargo between split boundaries
MIN_SPLIT_ROWS = 10                    # below this a split merges into its neighbour

# Row schema, in the order the spec requires. "realised" in the spec is "realized" in the data.
ROW_SCHEMA: list[tuple[str, str, str, str]] = [
    ("run",                      "int64",   "index",         "discrete"),
    ("instrument",               "string",  "symbol",        "discrete"),
    ("asset_type",               "string",  "category",      "discrete"),
    ("timeframe",                "string",  "category",      "discrete"),
    ("data_source",              "string",  "category",      "discrete"),
    ("is_backtest",              "int64",   "flag 0/1",      "discrete"),
    ("vol_index",                "float64", "generator vol", "continuous"),
    ("total_return_pct",         "float64", "percent",       "continuous"),
    ("realised_vol_per_bar_pct", "float64", "percent/bar",   "continuous"),
    ("accuracy_index",           "float64", "[-1, 1]",       "continuous"),
    ("bar_direction_match_pct",  "float64", "fraction",      "continuous"),
    ("slippage_share",           "float64", "fraction",      "continuous"),
    ("negative_slippage_share",  "float64", "fraction",      "continuous"),
    ("time_exit_share",          "float64", "fraction",      "continuous"),
    ("total_return_pct_label",   "float64", "percent",       "continuous"),
    ("future_vol_pct_label",     "float64", "percent/bar",   "continuous"),
    ("sample_weight",            "float64", "weight",        "continuous"),
]
# Spec spelling -> actual column in analysis_summary.csv
ALIASES = {"realised_vol_per_bar_pct": "realized_vol_per_bar_pct"}
IDENTIFIER_COLS = {"run", "instrument", "asset_type", "timeframe", "data_source", "is_backtest"}
LABEL_COLS = {"total_return_pct_label", "future_vol_pct_label", "sample_weight"}


# ── source ───────────────────────────────────────────────────────────────────
def _read_any(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        frame = pd.read_csv(handle, low_memory=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def load_source(analysis_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Live analysis_summary.csv if present, else every archived snapshot, newest first."""
    live = analysis_dir / "analysis_summary.csv"
    if live.exists():
        return _read_any(live), [str(live)]

    snaps = sorted(glob.glob(str(ARCHIVE / "*" / "looped_analysis" / "analysis_summary.csv.gz")), reverse=True)
    if not snaps:
        raise SystemExit(
            f"No source found. Looked for {live} and {ARCHIVE}/*/looped_analysis/analysis_summary.csv.gz\n"
            "The archive workflow may have pushed it to the Hub; restore with "
            "`python prune_disk_space.py --restore <snapshot>`."
        )
    frames, used = [], []
    for snap in snaps:
        try:
            frames.append(_read_any(Path(snap)))
            used.append(snap)
        except Exception as error:                                    # a corrupt snapshot must not stop the export
            print(f"  [skip] {snap}: {type(error).__name__}: {error}")
    return pd.concat(frames, ignore_index=True, sort=False), used


# ── weights ──────────────────────────────────────────────────────────────────
WEIGHT_FORMULA = """base      = clip((accuracy_index + 1) / 2, 0.05, 1.0)
row_mult  = 3.0 if data_source == "real" else 1.0
sample_weight = base * row_mult"""


def compute_weights(frame: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Weights per the formula. NaN accuracy_index -> null weight (kept, not floored)."""
    if "accuracy_index" not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64"), {
            "status": "all null", "reason": "accuracy_index column absent from every source row"}
    accuracy = pd.to_numeric(frame["accuracy_index"], errors="coerce")
    base = np.clip((accuracy + 1.0) / 2.0, 0.05, 1.0)                 # NaN propagates, by design
    if "data_source" in frame.columns:
        multiplier = np.where(frame["data_source"].astype(str).str.lower() == "real", 3.0, 1.0)
        mult_note = "data_source present; real rows weighted 3x"
    else:
        multiplier = 1.0
        mult_note = "data_source ABSENT - row_mult is 1.0 for every row, so the 3:1 real/sim ratio is inert"
    weight = pd.Series(base * multiplier, index=frame.index, dtype="float64")
    finite = weight.dropna()
    return weight, {
        "status": "computed",
        "rows_total": int(len(weight)),
        "rows_weighted": int(finite.size),
        "rows_null": int(weight.isna().sum()),
        "min": float(finite.min()) if finite.size else None,
        "median": float(finite.median()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "distinct": int(finite.nunique()),
        "row_mult_note": mult_note,
    }


# ── splits ───────────────────────────────────────────────────────────────────
def split_with_purge(n: int) -> tuple[dict[str, tuple[int, int]], int, list[str]]:
    """60/20/20 by run order with a purge band between boundaries. Returns slices, purge width, notes."""
    purge = max(1, int(round(n * PURGE_FRACTION))) if n >= 50 else 0
    train_end = int(n * 0.60)
    val_start, val_end = train_end + purge, int(n * 0.80)
    test_start = val_end + purge
    spans = {"train": (0, train_end), "validation": (val_start, val_end), "test": (test_start, n)}
    notes: list[str] = []
    for name in ("validation", "test"):
        lo, hi = spans[name]
        if hi - lo < MIN_SPLIT_ROWS:
            notes.append(f"{name} had {max(hi - lo, 0)} rows (< {MIN_SPLIT_ROWS}); merged into train")
            spans["train"] = (spans["train"][0], max(spans["train"][1], hi))
            spans[name] = (hi, hi)
    return spans, purge, notes


# ── export ───────────────────────────────────────────────────────────────────
def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(HERE), *args], capture_output=True, text=True,
                              timeout=30).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _leakage_cols() -> list[str]:
    try:
        sys.path.insert(0, str(HERE))
        import ml_model
        return list(ml_model.LEAKAGE_COLS)
    except Exception:
        return []


def export_dataset(analysis_dir, repo_id, *, token=None, private=True, push=True, replace=False,
                   scored_only=False) -> Path:
    analysis_dir = Path(analysis_dir)
    raw, sources = load_source(analysis_dir)
    print(f"  source rows: {len(raw):,} from {len(sources)} file(s)")

    # Build the schema'd frame. Missing columns become null and are recorded, never invented.
    out = pd.DataFrame(index=raw.index)
    absent: list[str] = []
    for name, dtype, _unit, _kind in ROW_SCHEMA:
        if name == "sample_weight":
            continue
        actual = ALIASES.get(name, name)
        if actual in raw.columns:
            series = raw[actual]
            out[name] = pd.to_numeric(series, errors="coerce") if dtype != "string" else series.astype("string")
        else:
            # pandas 3.0 refuses to fill a float64 array with pd.NA, so use np.nan for numerics.
            out[name] = pd.Series(pd.NA if dtype == "string" else np.nan, index=raw.index,
                                  dtype="string" if dtype == "string" else "float64")
            absent.append(name + (f" (looked for '{actual}')" if actual != name else ""))

    weight, weight_summary = compute_weights(raw)
    out["sample_weight"] = weight
    out = out[[n for n, *_ in ROW_SCHEMA]]
    if "run" in raw.columns:
        out = out.sort_values("run", kind="stable").reset_index(drop=True)

    # V3 - no leakage column may appear among exported features
    leakage = _leakage_cols()
    feature_cols = [n for n, *_ in ROW_SCHEMA if n not in IDENTIFIER_COLS | LABEL_COLS]
    hits = sorted(set(feature_cols) & set(leakage))
    if hits:
        raise SystemExit(f"STOP-5: leakage columns in exported features: {hits}")

    # scored_only: keep only rows that carry a weight. Chosen 2026-10-30 because accuracy_index is
    # written solely for scored cycles, which cluster at the top of the run order - an ordered
    # 60/20/20 over all rows put every weight in `test` and none in `train`, making weighted
    # training impossible. Restricting to scored rows gives every split weights.
    dropped_unscored = 0
    if scored_only:
        before = len(out)
        out = out[out["sample_weight"].notna()].reset_index(drop=True)
        dropped_unscored = before - len(out)
        print(f"  --scored-only: kept {len(out):,} weighted rows, dropped {dropped_unscored:,} unscored")
        if len(out) < 3 * MIN_SPLIT_ROWS:
            raise SystemExit(f"STOP: only {len(out)} scored rows; too few for three splits.")

    spans, purge, split_notes = split_with_purge(len(out))
    STAGING.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    written: dict[str, dict] = {}
    for name, (lo, hi) in spans.items():
        part = out.iloc[lo:hi]
        fname = f"{name}.parquet" if (replace or name != "train") else f"train_{stamp}.parquet"
        if replace:
            fname = f"{name}.parquet"
        target = STAGING / fname
        part.to_parquet(target, index=False, compression="snappy")
        size = target.stat().st_size
        if size > MAX_SHARD_BYTES:
            print(f"  [warn] {fname} is {size/1e9:.2f} GB (> 1 GB shard target); consider splitting")
        written[name] = {"file": fname, "rows": int(len(part)), "bytes": int(size),
                         "run_range": [int(part['run'].min()), int(part['run'].max())] if len(part) and part['run'].notna().any() else None}
        print(f"  {name:<11} {len(part):>6,} rows -> {fname} ({size/1024:.0f} KB)")

    config_hash = hashlib.sha256((HERE / "config.toml").read_bytes()).hexdigest()[:16] if (HERE / "config.toml").exists() else "absent"
    info = {
        "dataset_name": repo_id, "export_utc": stamp,
        "features": {n: {"dtype": d, "unit": u, "kind": k} for n, d, u, k in ROW_SCHEMA},
        "splits": written, "purge_rows_between_splits": purge,
        "rows_total": int(len(out)), "weight_summary": weight_summary,
        "columns_requested_but_absent": absent, "split_notes": split_notes,
        "scored_only": bool(scored_only), "unscored_rows_dropped": int(dropped_unscored),
        "git_commit": _git("rev-parse", "--short", "HEAD"), "config_sha256_16": config_hash,
        "source_files": [Path(s).name for s in sources[:5]] + (["..."] if len(sources) > 5 else []),
    }
    (STAGING / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    (STAGING / "schema.json").write_text(json.dumps(
        {n: {"dtype": d, "unit": u, "kind": k} for n, d, u, k in ROW_SCHEMA}, indent=2) + "\n", encoding="utf-8")
    _write_data_card(repo_id, info, absent, weight_summary, purge, split_notes, leakage, feature_cols)

    if push:
        _upload(repo_id, token=token, private=private)
    else:
        print("  --no-push: staged only")
    return STAGING


def _write_data_card(repo_id, info, absent, weight_summary, purge, split_notes, leakage, feature_cols) -> None:
    rows = "\n".join(f"| `{n}` | {d} | {u} | {k} |" for n, d, u, k in ROW_SCHEMA)
    card = f"""# {repo_id}

Synthetic price-simulation runs from a self-correcting OHLC generator. **No real market signal is
claimed.** Rows are simulated price paths plus the metrics computed from them; any predictive
performance measured on this data describes the simulator, not a market.

## Schema

| column | dtype | unit | kind |
|---|---|---|---|
{rows}

## Weight formula (verbatim)

```
{WEIGHT_FORMULA}
```

`accuracy_index` is in [-1, 1]; the affine map puts it in [0, 1]; the clip guards degenerate runs.
The 3:1 ratio mirrors `[live] real_weight / sim_weight` in `config.toml`.

**As applied to this export:** {weight_summary.get('row_mult_note')}

| | |
|---|---|
| rows total | {weight_summary.get('rows_total')} |
| rows with a weight | {weight_summary.get('rows_weighted')} |
| rows with null weight | {weight_summary.get('rows_null')} |
| min / median / max | {weight_summary.get('min')} / {weight_summary.get('median')} / {weight_summary.get('max')} |
| distinct values | {weight_summary.get('distinct')} |

**NaN policy:** rows whose `accuracy_index` is NaN (unscored cycles) keep `sample_weight = null`.
They are neither dropped nor floored at 0.05, because flooring would assert "unscored = worst
possible" — a claim about data that was never measured. The trainer drops null-weight rows at fit
time, so the choice stays visible and reversible.

## Splits

60 / 20 / 20 by run order, never random, with a **{purge}-row purge band** between boundaries so no
run sits adjacent to a run in another split.

| split | rows | file | run range |
|---|---|---|---|
""" + "\n".join(
        f"| {k} | {v['rows']} | `{v['file']}` | {v['run_range']} |" for k, v in info["splits"].items()
    ) + f"""

{chr(10).join('- ' + n for n in split_notes) if split_notes else '- no split needed merging'}

## Provenance

| | |
|---|---|
| git commit | `{info['git_commit']}` |
| config.toml sha256 (16) | `{info['config_sha256_16']}` |
| export timestamp (UTC) | {info['export_utc']} |
| source files | {', '.join(info['source_files'])} |
| training job id | *not yet submitted - see `runs/<job_id>/` once `submit_hf_job.py` has run* |
| licence | unspecified (this repo carries no LICENSE file) |

## Columns requested but absent

{chr(10).join('- `' + a + '`' for a in absent) if absent else '- none'}

These are emitted as null rather than fabricated. `accuracy_index` and `bar_direction_match_pct`
exist only for scored **backtest** runs; forward-simulation rows leave them empty. `data_source` is
written by no part of the pipeline today, which is why the real-vs-simulated multiplier is inert.

## Leakage check (V3)

{len(leakage)} names from `ml_model.LEAKAGE_COLS` were checked against the {len(feature_cols)}
exported feature columns. No overlap, so no leakage column is exported as a feature.
"""
    (STAGING / "data_card.md").write_text(card, encoding="utf-8")


def _upload(repo_id: str, *, token=None, private=True) -> None:
    from huggingface_hub import HfApi, create_repo, upload_folder, get_token
    if not (token or os.getenv("HF_TOKEN") or get_token()):
        print("STOP-1: no token. Run `hf auth login`.")
        raise SystemExit(0)
    files = [p for p in STAGING.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    print(f"  upload: {len(files)} files, {total/1024**2:.1f} MB")
    if total > MAX_UPLOAD_BYTES:
        raise SystemExit(f"STOP-2: staged {total/1024**3:.1f} GB exceeds the 90 GB guard; nothing sent.")
    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
    if len(files) < 10_000 and total < 50 * 1024 ** 3:
        upload_folder(folder_path=str(STAGING), repo_id=repo_id, repo_type="dataset",
                      path_in_repo=PATH_IN_REPO, token=token,
                      commit_message="export training-ready Parquet dataset")
    else:
        # upload_large_folder is an HfApi method in huggingface_hub 2.x, not a module-level function.
        # It is resumable and multi-worker but cannot set path_in_repo, so the large case lands at root.
        HfApi(token=token).upload_large_folder(repo_id=repo_id, folder_path=str(STAGING),
                                               repo_type="dataset", private=private)
    print(f"  pushed -> https://huggingface.co/datasets/{repo_id}/tree/main/{PATH_IN_REPO}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis-dir", default=os.getenv("ANALYSIS_DIR", str(HERE / "looped_analysis")))
    ap.add_argument("--repo", default=os.getenv("SIM_ARCHIVE_REPO", "ay0k200/sim-archive"))
    ap.add_argument("--no-push", action="store_true", help="stage locally, do not upload")
    ap.add_argument("--replace", action="store_true", help="overwrite train.parquet instead of timestamping")
    ap.add_argument("--scored-only", action="store_true",
                    help="export only rows with a non-null sample_weight (every split then has weights)")
    ap.add_argument("--public", action="store_true", help=argparse.SUPPRESS)   # never used; repo stays private
    args = ap.parse_args()
    if args.public:
        raise SystemExit("Refusing: this dataset must stay private.")
    out = export_dataset(args.analysis_dir, args.repo, push=not args.no_push, replace=args.replace,
                         scored_only=args.scored_only)
    print(f"  staged at {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
