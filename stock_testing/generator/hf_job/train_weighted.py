"""
train_weighted.py - train on the exported Parquet dataset with accuracy-derived sample weights.

    python train_weighted.py --repo ay0k200/sim-archive --backend sklearn --out model_out

Every path comes from an argument or an environment variable (SIM_ARCHIVE_REPO, MODEL_OUT,
HF_HOME). Nothing is hardcoded to one machine: the previous harness (abload.py) embedded
r"C:/Users/Admin/..." and could not run anywhere else, which is exactly why this runs on HF Jobs.

The point of the run is the comparison, not the model: fit once with sample_weight and once with
uniform weights, and report the out-of-sample R2 delta with a bootstrap 95% CI. If the weights
carry no information the delta should straddle zero.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO = os.getenv("SIM_ARCHIVE_REPO", "ay0k200/sim-archive")
OUT_DIR = Path(os.getenv("MODEL_OUT", "model_out"))
PREFIX = os.getenv("HF_DATASET_PREFIX", "dataset")

# Identifiers and targets are never features.
IDENTIFIER_COLS = {"run", "instrument", "asset_type", "timeframe", "data_source", "is_backtest"}
TARGET_COLS = {"total_return_pct_label", "future_vol_pct_label"}
WEIGHT_COL = "sample_weight"


def _splits(repo_id: str) -> dict:
    from datasets import load_dataset
    return load_dataset(repo_id, data_files={s: f"{PREFIX}/{s}.parquet"
                                             for s in ("train", "validation", "test")})


def _feature_cols(columns: list[str]) -> list[str]:
    """Numeric columns that are neither identifier, target nor weight. Leakage names are excluded
    defensively here too, so a future export cannot sneak one past the trainer."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import ml_model
        leakage = set(ml_model.LEAKAGE_COLS)
    except Exception:
        leakage = set()
    return [c for c in columns if c not in IDENTIFIER_COLS | TARGET_COLS | {WEIGHT_COL} | leakage]


def _matrix(split, cols: list[str]) -> np.ndarray:
    return np.column_stack([np.asarray(split[c], dtype="float64") for c in cols])


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _bootstrap_delta_ci(y: np.ndarray, pred_w: np.ndarray, pred_u: np.ndarray,
                        resamples: int = 1000, seed: int = 42) -> tuple[float, float]:
    """95% CI on (R2_weighted - R2_uniform) by resampling the evaluation rows, predictions fixed."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, n)
        deltas[i] = _r2(y[idx], pred_w[idx]) - _r2(y[idx], pred_u[idx])
    return float(np.nanpercentile(deltas, 2.5)), float(np.nanpercentile(deltas, 97.5))


def _resolve_target(split, requested: str | None) -> tuple[str, list[str]]:
    """Pick a target column that actually has values. Never invent one."""
    notes: list[str] = []
    candidates = [requested] if requested else ["total_return_pct_label", "total_return_pct"]
    for name in candidates:
        if name and name in split.column_names:
            values = np.asarray(split[name], dtype="float64")
            if np.isfinite(values).any():
                if name != candidates[0]:
                    notes.append(f"target fell back to '{name}' because '{candidates[0]}' is all-null "
                                 "(Prompt D has not written the label columns)")
                return name, notes
            notes.append(f"'{name}' present but entirely null")
    raise SystemExit(f"No usable target. Tried {candidates}. Pass --target explicitly.")


def train(repo_id=REPO, *, split="train", backend="sklearn", out_dir=OUT_DIR, seed=42,
          target=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = _splits(repo_id)
    tr, ev = ds[split], ds["validation"]            # test is NOT touched here, by design

    target_col, notes = _resolve_target(tr, target)
    cols = _feature_cols(tr.column_names)
    if target_col in cols:
        # _resolve_target may fall back to a bare column (e.g. total_return_pct) that
        # _feature_cols only excludes under its '_label' name. Left in, the model predicts
        # the target from itself and R2 is meaningless.
        cols = [c for c in cols if c != target_col]
        notes.append(f"removed the resolved target '{target_col}' from the feature set - "
                     "the fallback target is itself a feature column, so leaving it in "
                     "would let the model read the answer")
    keep = [c for c in cols if np.isfinite(np.asarray(tr[c], dtype="float64")).any()]
    if len(keep) < len(cols):
        notes.append(f"dropped {len(cols) - len(keep)} all-null feature column(s): "
                     f"{sorted(set(cols) - set(keep))}")

    X, y = _matrix(tr, keep), np.asarray(tr[target_col], dtype="float64")
    Xe, ye = _matrix(ev, keep), np.asarray(ev[target_col], dtype="float64")
    w = np.asarray(tr[WEIGHT_COL], dtype="float64") if WEIGHT_COL in tr.column_names else None

    if w is None or not np.isfinite(w).any():
        notes.append("sample_weight absent or all-null - falling back to uniform; no weights invented")
        w = np.ones(len(y))
    elif np.unique(w[np.isfinite(w)]).size == 1:
        notes.append("sample_weight is constant - weighted and uniform fits are identical by construction")

    finite = np.isfinite(y) & np.isfinite(X).all(axis=1) & np.isfinite(w)
    X, y, w = X[finite], y[finite], w[finite]
    fe = np.isfinite(ye) & np.isfinite(Xe).all(axis=1)
    Xe, ye = Xe[fe], ye[fe]
    print(f"  backend={backend} target={target_col} features={len(keep)} "
          f"train={len(y)} eval={len(ye)} weight[min={w.min():.4f} max={w.max():.4f}]")

    if backend == "sklearn":
        from sklearn.ensemble import HistGradientBoostingRegressor
        def fit(sample_weight):
            model = HistGradientBoostingRegressor(random_state=seed, max_iter=200)
            model.fit(X, y, sample_weight=sample_weight)
            return model
        model_w, model_u = fit(w), fit(np.ones(len(y)))
        pred_w, pred_u = model_w.predict(Xe), model_u.predict(Xe)
        import joblib
        joblib.dump(model_w, out_dir / "model.pkl")
    elif backend == "transformers":
        raise SystemExit("backend 'transformers' needs torch/transformers; install via "
                         "requirements-hf.txt and see WeightedTrainer in this file's docstring.")
    else:
        raise SystemExit(f"unknown backend {backend!r}")

    r2_w, r2_u = _r2(ye, pred_w), _r2(ye, pred_u)
    lo, hi = _bootstrap_delta_ci(ye, pred_w, pred_u, 1000, seed)
    metrics = {
        "repo_id": repo_id, "backend": backend, "target": target_col, "seed": seed,
        "features": keep, "n_train": int(len(y)), "n_eval": int(len(ye)),
        "R2_weighted": r2_w, "R2_uniform": r2_u, "delta": r2_w - r2_u,
        "delta_ci95": [lo, hi],
        "delta_excludes_zero": bool(lo > 0 or hi < 0),
        "weight_min": float(w.min()), "weight_max": float(w.max()),
        "weight_distinct": int(np.unique(w).size),
        "bootstrap_resamples": 1000, "notes": notes,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"  R2 weighted {r2_w:+.4f} | uniform {r2_u:+.4f} | delta {r2_w - r2_u:+.4f} "
          f"CI95 [{lo:+.4f}, {hi:+.4f}] -> {'significant' if metrics['delta_excludes_zero'] else 'straddles zero'}")
    for note in notes:
        print(f"  note: {note}")
    return metrics


# WeightedTrainer for backend="transformers" - kept verbatim per spec, imported lazily so this
# module has no torch dependency when the sklearn backend is used.
WEIGHTED_TRAINER_SRC = '''
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        weights = inputs.pop("sample_weight")
        outputs = model(**inputs)
        loss = F.mse_loss(outputs.logits.squeeze(), labels, reduction="none")
        loss = (loss * weights).mean()
        return (loss, outputs) if return_outputs else loss
'''


def _upload_artifacts(repo_id: str, out_dir: Path) -> str | None:
    """Push metrics.json / model.pkl / training.log into runs/<job_id>/ of the dataset repo.

    Only runs inside an HF Job, where the job id is in the environment. Locally there is no
    job id, so this is a no-op and the files just stay in out_dir. The token comes from the
    HF_TOKEN the job is started with; it is never printed and never written to a file.
    """
    job_id = os.getenv("HF_JOB_ID") or os.getenv("JOB_ID") or ""
    if not job_id:
        print("  no HF_JOB_ID in the environment - skipping artifact upload (local run)")
        return None
    from huggingface_hub import HfApi
    HfApi(token=os.getenv("HF_TOKEN")).upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=f"runs/{job_id}",
        commit_message=f"job {job_id}: training artifacts",
    )
    url = f"https://huggingface.co/datasets/{repo_id}/tree/main/runs/{job_id}"
    print(f"  artifacts uploaded to {url}")
    return job_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--split", default="train")
    ap.add_argument("--backend", default="sklearn", choices=["sklearn", "transformers"])
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--target", default=None, help="override the target column")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = Path(args.out)
    log = out_dir / "training.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = train(args.repo, split=args.split, backend=args.backend, out_dir=out_dir,
                    seed=args.seed, target=args.target)
    log.write_text(json.dumps(metrics, indent=2) + chr(10), encoding="utf-8")
    _upload_artifacts(args.repo, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
