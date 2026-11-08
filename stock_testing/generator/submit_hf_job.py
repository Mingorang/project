"""submit_hf_job.py - run train_weighted.py on HF Jobs, reading the same repo the data went to.

    python submit_hf_job.py --repo ay0k200/sim-archive --backend sklearn

`uv run` rather than `docker run`: this is pure Python with four wheels and no system libraries,
which is exactly what HF recommends uv for. `--flavor cpu-upgrade` (~8 vCPU / 32 GB) is ample for
HistGradientBoosting on a few hundred thousand rows; no GPU flavor is requested because nothing
here benefits from one and GPU flavors are quota-gated.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOB_DIR = Path(os.getenv("HF_JOB_DIR", HERE / "hf_job"))


def submit(repo_id, *, backend="sklearn", flavor="cpu-upgrade", timeout_s=3600) -> str:
    if flavor.startswith(("t4", "l4", "l40", "a10", "a100", "h200", "rtx")):
        raise SystemExit(f"Refusing GPU flavor {flavor!r}: cpu-upgrade only.")
    script = JOB_DIR / "train_weighted.py"
    if not script.exists():
        raise SystemExit(f"{script} missing - run the packaging step first.")
    cmd = ["hf", "jobs", "uv", "run", "--flavor", flavor,
           "--env", f"SIM_ARCHIVE_REPO={repo_id}", "--env", f"BACKEND={backend}",
           "--env", "MODEL_OUT=/results", "--env", "HF_DATASET_PREFIX=dataset",
           "--timeout", f"{int(timeout_s)}s", "--detach",
           "--with", "datasets", "--with", "scikit-learn", "--with", "joblib",
           "--format", "json", str(script)]
    print("  " + " ".join(cmd))
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out = (done.stdout or "") + (done.stderr or "")
    if done.returncode != 0:
        # The real error arrives AFTER the uploader's progress bars, so print the TAIL.
        # Printing the head showed only the job-configuration block and hid a 402.
        print(chr(10).join(out.strip().splitlines()[-12:]))
        if "402" in out or "credit balance is insufficient" in out:
            raise SystemExit(
                "STOP-6: HF Jobs refused the submission with 402 Payment Required - "
                "the account's pre-paid credit balance is insufficient. This is an account "
                "billing state, not a defect in the command: add credits at "
                "https://huggingface.co/settings/billing and run this script again unchanged.")
        raise SystemExit(f"STOP-6: submission failed (exit {done.returncode}). Command printed above; not retrying.")
    job_id = ""
    try:
        payload = json.loads(done.stdout)
        job_id = payload.get("id") or payload.get("job_id") or ""
    except json.JSONDecodeError:
        for tok in out.split():
            if len(tok) >= 12 and tok.strip("\"',").isalnum():
                job_id = tok.strip("\"',"); break
    if not job_id:
        print(out.strip()[:600])
        raise SystemExit("STOP-6: submitted but no job_id parsed; see output above.")
    print(f"  job_id: {job_id}")
    print(f"  artifacts will land at https://huggingface.co/datasets/{repo_id}/tree/main/runs/{job_id}")
    print(f"  follow with: hf jobs logs {job_id}")
    return job_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.getenv("SIM_ARCHIVE_REPO", "ay0k200/sim-archive"))
    ap.add_argument("--backend", default="sklearn", choices=["sklearn", "transformers"])
    ap.add_argument("--flavor", default="cpu-upgrade")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    submit(a.repo, backend=a.backend, flavor=a.flavor, timeout_s=a.timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
