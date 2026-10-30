"""
loop_runner.py - run price_modeling.py unattended, many times, for hours or days.

Built for a headless server / cloud environment (e.g. Apocrita HPC, Linux/Windows cloud VMs)
with finite disk quotas (e.g. 100 GB). It answers or bypasses interactive prompts, varies the
random seed per iteration, keeps the footprint minimal by calling prune_disk_space.py, writes a
persistent ledger of every iteration, and resumes where it left off after a crash, reboot, or timeout.

    python loop_runner.py --dry-run                 validate the answer sequence, run 1 iteration
    python loop_runner.py --iterations 500          run 500 iterations then stop
    python loop_runner.py --hours 48                run for 48 hours then stop
    python loop_runner.py --mode backtest           run backtest self-correction sweeps
    python loop_runner.py --resume                  continue an interrupted campaign
    python loop_runner.py --status                  what the last campaign did
    python loop_runner.py --cloud-backend onedrive  archive snapshots to cloud, keeping < 1GB locally

Why the answers and CLI flags matter
    price_modeling.py asks its questions through console.ask(). In headless or looped execution,
    loop_runner passes full CLI flags (--non-interactive, --runs/--cycles, --timeframe, --interval)
    and an exact stdin answer sequence so no prompt ever hangs or crashes with
    "Input ended before an answer to: ...".

Why the disk and speed matter (QMUL Free Cloud Stack)
    A days-long campaign is bounded by disk and chart rendering, not CPU.
    1. Per-run charts take ~2.4s per run (33 core-hours at 50k runs). By default, loop_runner uses
       output_mode = "6" (CSVs + metrics only - no charts, no ML), cutting loop time to 15-25 min (80x faster!).
    2. prune_disk_space.py is called with --stage-only --apply --delete-local --min-age-minutes 0
       after every iteration (or when reaching disk_soft_gb), immediately converting outputs to Parquet/gzip
       and freeing raw run CSVs and PNGs. This keeps disk usage strictly under server quotas (100 GB).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = HERE / "loop_state.json"
LEDGER = HERE / "loop_ledger.csv"
SIM = HERE / "price_modeling.py"
PRUNER = HERE / "prune_disk_space.py"
WATCHED = ("looped_results", "looped_analysis", "training_runs", "cloud_archive")

INPUT_EXHAUSTED = "Input ended before an answer to:"

# Transient upstream failures, observed in loop_ledger.csv at iteration 11: a Yahoo DNS blip makes
# price_modeling.py exit 1 three times in a row, which trips max_consecutive_failures and ends a
# multi-day campaign over a network hiccup. These are retried with backoff instead, and do not
# count toward the failure limit, which exists to catch real bugs.
NETWORK_MARKERS = (
    "Cookie/crumb fetch failed",
    "possibly delisted; no price data found",
    "No daily price history returned",
    "Could not get Yahoo Finance",
    "Max retries exceeded", "Connection aborted", "Read timed out",
    "Temporary failure in name resolution", "getaddrinfo failed", "DNSError",
)

DEFAULTS = {
    "mode": "forward",               # forward | backtest
    "asset_type": "forex",
    "instrument": "EUR/USD",
    "timeframe": "5d",
    "interval": "1h",
    "runs_per_iteration": 200,       # forward runs per iteration
    "cycles": 10,                    # backtest cycles per iteration
    "output_mode": "6",              # 6 = CSVs + metrics only (fastest, no charts, minimal disk)
    "ml_every": 0,                   # run an ML pass every N iterations (0 = never)
    "base_seed": 2026102,            # valid base seed (-10,000,000 .. 10,000,000)
    "iteration_timeout_s": 3600,
    "pause_between_s": 5,
    "max_consecutive_failures": 3,
    "max_transient_retries": 10,    # transient upstream errors retried this often before giving up
    "disk_soft_gb": 40.0,            # past this, archive and prune between iterations
    "disk_hard_gb": 80.0,            # past this, stop the campaign cleanly (under 100 GB server quota)
    "free_space_floor_gb": 10.0,     # stop if the volume free space falls below this
    "auto_prune_each_run": True,     # immediately reclaim space and convert to Parquet after each iteration
    "prune_command": ["--stage-only", "--apply", "--delete-local", "--min-age-minutes", "0"],
    "cloud_backend": "",             # "" (local spool) | "onedrive" | "hf" | "azure"
    # Destination for the chosen backend. load_loop_config() only accepts [loop] keys that exist
    # here, so a key missing from this dict is silently dropped from config.toml.
    "hf_repo": "",                   # backend hf: private DATASET repo id, e.g. mingorang/sim-archive
    "azure_container": "",           # backend azure: blob container name
    "archive_dest": "",              # backend folder: destination directory
    "rclone_remote": "",             # backend rclone: configured remote name
}


# ── config ───────────────────────────────────────────────────────────────────
def load_loop_config(path: Path | None = None) -> dict:
    """[loop] from config.toml merged over DEFAULTS. Unknown keys are reported, not ignored."""
    cfg = dict(DEFAULTS)
    toml_path = path or (HERE / "config.toml")
    if toml_path.exists():
        try:
            table = tomllib.loads(toml_path.read_text(encoding="utf-8")).get("loop", {})
        except tomllib.TOMLDecodeError as exc:
            raise SystemExit(f"config.toml is not valid TOML: {exc}")
        unknown = sorted(set(table) - set(DEFAULTS))
        if unknown:
            print(f"[loop] ignoring unknown [loop] keys: {', '.join(unknown)}")
        cfg.update({k: v for k, v in table.items() if k in DEFAULTS})
    return cfg


def build_answers(cfg: dict) -> str:
    """The answer stream for one iteration, matching price_modeling.py prompts."""
    if "answers" in cfg and isinstance(cfg["answers"], list):
        filled = [
            str(a).format(
                runs=cfg.get("runs_per_iteration", 200),
                cycles=cfg.get("cycles", 10),
                output_mode=cfg.get("output_mode", "6"),
                timeframe=cfg.get("timeframe", "5d"),
                interval=cfg.get("interval", "1h"),
            )
            for a in cfg["answers"]
        ]
        return "\n".join(filled) + "\n"

    mode = str(cfg.get("mode", "forward")).lower()
    if mode == "backtest":
        answers = [
            "2",                                     # [1] enter price manually or [2] use current price
            "",                                      # Yahoo symbol (blank = default)
            "2",                                     # [1] allow negative or [2] keep above zero
            str(cfg.get("timeframe", "5d")),         # backtest timeframe
            str(cfg.get("interval", "1h")),          # candle interval
            str(cfg.get("cycles", 10)),              # number of cycles
            "y",                                     # self-correct volatility?
            "auto",                                  # search method
            "1",                                     # learning loops
            "n",                                     # run ML?
            "n",                                     # keep ML history?
        ]
    else:
        # forward simulation
        answers = [
            "2",                                     # [1] enter price manually or [2] use current price
            "",                                      # Yahoo symbol (blank = default)
            "2",                                     # [1] allow negative or [2] keep above zero
            str(cfg.get("timeframe", "5d")),         # timeframe
            str(cfg.get("interval", "1h")),          # candle interval
            "m",                                     # one or many plots
            str(cfg.get("runs_per_iteration", 200)), # how many plots
            "y",                                     # use learned volatility if offered
            str(cfg.get("output_mode", "6")),        # save state 1-6
            "n",                                     # run ML?
            "n",                                     # keep ML history?
        ]
    return "\n".join(answers) + "\n"


# ── disk ─────────────────────────────────────────────────────────────────────
def footprint_gb(root: Path = HERE) -> float:
    """Bytes under the directories a campaign actually grows, in GB."""
    total = 0
    for name in WATCHED:
        directory = root / name
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                pass
    return total / 1024 ** 3


def free_gb(root: Path = HERE) -> float:
    return shutil.disk_usage(root).free / 1024 ** 3


def run_pruner(cfg: dict, log) -> bool:
    """Hand retention to prune_disk_space.py. True if it exited cleanly."""
    if not PRUNER.exists():
        log("prune_disk_space.py is missing - cannot reclaim space")
        return False
    backend = str(cfg.get("cloud_backend", "")).strip()
    if backend:
        command = [sys.executable, str(PRUNER), "--archive", "--backend", backend,
                   "--apply", "--delete-local", "--min-age-minutes", "0"]
        # Every backend needs its own destination argument or the pruner exits on the first call:
        # "--backend hf needs --hf-repo USER/NAME" (prune_disk_space.py line 1086). Without this the
        # campaign would prune-fail every iteration while still reporting the runs themselves as ok.
        needs = {
            "hf": ("--hf-repo", cfg.get("hf_repo", "")),
            "azure": ("--container", cfg.get("azure_container", "")),
            "folder": ("--dest", cfg.get("archive_dest", "")),
            "rclone": ("--remote", cfg.get("rclone_remote", "")),
        }.get(backend)
        if needs and str(needs[1]).strip():
            command += [needs[0], str(needs[1]).strip()]
        elif needs:
            log(f"cloud_backend={backend!r} needs {needs[0]} set in [loop] - "
                "staging to the local spool instead this iteration")
            command = [sys.executable, str(PRUNER), "--stage-only", "--apply",
                       "--delete-local", "--min-age-minutes", "0"]
    else:
        cmd_args = cfg.get("prune_command", ["--stage-only", "--apply", "--delete-local", "--min-age-minutes", "0"])
        # Ensure --apply is present so deletion and conversion actually happen
        if "--apply" not in cmd_args and "--dry-run" not in cmd_args:
            cmd_args = list(cmd_args) + ["--apply"]
        command = [sys.executable, str(PRUNER), *cmd_args]

    log(f"reclaiming space: {' '.join(command[1:])}")
    try:
        done = subprocess.run(command, cwd=HERE, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=1800)
    except subprocess.TimeoutExpired:
        log("pruner timed out after 30 min")
        return False
    # exit code 0 = ok, 2 = some outputs were younger than min-age, which is fine
    if done.returncode not in (0, 2):
        log(f"pruner exited {done.returncode}: {(done.stderr or done.stdout or '').strip()[:300]}")
    return done.returncode in (0, 2)


# ── state and ledger ─────────────────────────────────────────────────────────
def load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


LEDGER_FIELDS = ["iteration", "started_utc", "seconds", "exit_code", "status",
                 "seed", "runs", "footprint_gb_before", "footprint_gb_after", "detail"]


def append_ledger(row: dict) -> None:
    new = not LEDGER.exists()
    with LEDGER.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LEDGER_FIELDS})


# ── one iteration ────────────────────────────────────────────────────────────
def run_once(cfg: dict, seed: int, log) -> tuple[str, int, str]:
    """Drive one price_modeling.py run. Returns (status, exit_code, detail).

    status is 'ok', 'failed' (retryable) or 'fatal' (the answer sequence is wrong - stop).
    """
    mode = str(cfg.get("mode", "forward")).lower()
    command = [
        sys.executable, "-u", str(SIM),
        "--mode", mode,
        "--seed", str(seed),
        "--asset-type", str(cfg.get("asset_type", "forex")),
        "--instrument", str(cfg.get("instrument", "EUR/USD")),
        "--timeframe", str(cfg.get("timeframe", "5d")),
        "--interval", str(cfg.get("interval", "1h")),
        "--save-state", str(cfg.get("output_mode", "6")),
        "--non-interactive",
    ]
    if mode == "backtest":
        command += ["--cycles", str(cfg.get("cycles", 10))]
    else:
        command += ["--runs", str(cfg.get("runs_per_iteration", 200))]

    env = {**os.environ, "NO_COLOR": "1", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"}
    try:
        done = subprocess.run(command, cwd=HERE, input=build_answers(cfg), text=True,
                              capture_output=True, encoding="utf-8", errors="replace",
                              timeout=float(cfg["iteration_timeout_s"]), env=env)
    except subprocess.TimeoutExpired:
        return "failed", -1, f"timed out after {cfg['iteration_timeout_s']}s"

    output = (done.stdout or "") + (done.stderr or "")
    if INPUT_EXHAUSTED in output:
        line = next((ln.strip() for ln in output.splitlines() if INPUT_EXHAUSTED in ln), INPUT_EXHAUSTED)
        return "fatal", done.returncode, line[:300]
    if done.returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-3:])[:300]
        if any(marker in output for marker in NETWORK_MARKERS):
            return "transient", done.returncode, tail or "upstream data source unavailable"
        return "failed", done.returncode, tail or "no output"
    return "ok", 0, ""


def ensure_output_dirs() -> None:
    """Re-create the output folders before every iteration.

    With auto_prune_each_run the pruner runs between iterations and can remove these; the next run
    then dies on its first write. That is exactly the FileNotFoundError on looped_results/data_N.csv
    recorded at iterations 1 and 10 of loop_ledger.csv. Re-creating is cheap and idempotent.
    """
    for name in ("looped_results", "looped_analysis"):
        (HERE / name).mkdir(parents=True, exist_ok=True)


def run_ml_pass(cfg: dict, log) -> None:
    """Occasional ML pass. Separated because it writes the GB-scale candle index."""
    command = [sys.executable, "-u", str(HERE / "ml_model.py"), "--label", "forecast", "--clear-history"]
    log("ML pass (this one writes the large candle index)")
    try:
        done = subprocess.run(command, cwd=HERE, input="", text=True, capture_output=True,
                              encoding="utf-8", errors="replace", timeout=7200)
        log(f"ML pass exited {done.returncode}")
    except subprocess.TimeoutExpired:
        log("ML pass timed out after 2 h")


# ── campaign ─────────────────────────────────────────────────────────────────
class Stopper:
    """SIGINT/SIGTERM asks the campaign to finish the current iteration and exit cleanly."""

    def __init__(self) -> None:
        self.requested = False
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._handle)
            except (ValueError, OSError):
                pass

    def _handle(self, signum, frame) -> None:  # noqa: ARG002
        if self.requested:
            raise SystemExit("Second signal - exiting now.")
        self.requested = True
        print("\n[loop] stop requested - finishing the current iteration, then exiting.", flush=True)


def campaign(cfg: dict, iterations: int | None, hours: float | None, resume: bool) -> int:
    started = time.time()
    deadline = started + hours * 3600 if hours else None
    state = load_state() if resume else {}
    done_count = int(state.get("completed", 0))
    consecutive_failures = 0
    transient_retries = 0
    stopper = Stopper()

    def log(message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        print(f"[{stamp}] {message}", flush=True)

    mode = str(cfg.get("mode", "forward"))
    units = f"{cfg.get('cycles', 10)} cycles" if mode == "backtest" else f"{cfg.get('runs_per_iteration', 200)} runs"
    log(f"campaign start: mode={mode} {cfg['instrument']} "
        f"work/iteration={units} already_done={done_count}")
    log(f"budget: footprint soft {cfg['disk_soft_gb']} GB / hard {cfg['disk_hard_gb']} GB, "
        f"free-space floor {cfg['free_space_floor_gb']} GB, auto-prune={cfg.get('auto_prune_each_run', True)}")

    while True:
        if stopper.requested:
            log("stopping: signal")
            break
        if iterations is not None and done_count >= iterations:
            log(f"stopping: reached {iterations} iterations")
            break
        if deadline and time.time() >= deadline:
            log("stopping: time budget reached")
            break

        before = footprint_gb()
        free = free_gb()
        if free < float(cfg["free_space_floor_gb"]):
            log(f"stopping: only {free:.1f} GB free on the volume (under floor {cfg['free_space_floor_gb']} GB)")
            break
        if before >= float(cfg["disk_soft_gb"]):
            log(f"footprint {before:.1f} GB is over the soft limit ({cfg['disk_soft_gb']} GB)")
            run_pruner(cfg, log)
            before = footprint_gb()
            log(f"footprint after reclaiming: {before:.1f} GB")
        if before >= float(cfg["disk_hard_gb"]):
            log(f"stopping: footprint {before:.1f} GB is over the hard limit ({cfg['disk_hard_gb']} GB)")
            break

        iteration = done_count + 1
        # Seed stays within SEED_LIMIT 10_000_000
        seed = (int(cfg["base_seed"]) + iteration) % 10_000_000
        began = time.time()
        log(f"iteration {iteration} starting (seed {seed})")
        ensure_output_dirs()
        status, code, detail = run_once(cfg, seed, log)
        elapsed = time.time() - began
        after = footprint_gb()

        work_units = cfg.get("cycles" if mode == "backtest" else "runs_per_iteration", 200)
        append_ledger({
            "iteration": iteration,
            "started_utc": datetime.fromtimestamp(began, timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(elapsed, 1), "exit_code": code, "status": status, "seed": seed,
            "runs": work_units,
            "footprint_gb_before": round(before, 3), "footprint_gb_after": round(after, 3),
            "detail": detail,
        })

        if status == "fatal":
            log("FATAL - prompt answer issue detected:")
            log(f"        {detail}")
            log("        Verify loop configuration and CLI parameters.")
            save_state({**state, "completed": done_count, "last_status": "fatal", "last_detail": detail})
            return 2

        if status == "ok":
            consecutive_failures = 0
            transient_retries = 0
            done_count = iteration
            log(f"iteration {iteration} ok in {elapsed:.0f}s - footprint {after:.1f} GB "
                f"({after - before:+.2f} GB)")
            if int(cfg["ml_every"]) and done_count % int(cfg["ml_every"]) == 0:
                run_ml_pass(cfg, log)
            # Reclaim space immediately after iteration to prevent disk buildup on 100 GB servers
            if cfg.get("auto_prune_each_run", True):
                run_pruner(cfg, log)
                after = footprint_gb()
        elif status == "transient":
            transient_retries += 1
            cap = int(cfg.get("max_transient_retries", 10))
            wait = min(60 * transient_retries, 900)
            log(f"iteration {iteration} hit a transient upstream error "
                f"({transient_retries}/{cap}) - retrying in {wait}s: {detail.splitlines()[-1][:120]}")
            if transient_retries >= cap:
                log("stopping: the upstream data source stayed unreachable")
                save_state({**state, "completed": done_count, "last_status": "transient"})
                return 1
            time.sleep(wait)
            continue
        else:
            consecutive_failures += 1
            log(f"iteration {iteration} FAILED ({detail}) - {consecutive_failures} in a row")
            if consecutive_failures >= int(cfg["max_consecutive_failures"]):
                log("stopping: too many consecutive failures")
                save_state({**state, "completed": done_count, "last_status": "failed"})
                return 1
            time.sleep(min(60 * consecutive_failures, 300))

        save_state({
            "completed": done_count,
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "instrument": cfg["instrument"], "mode": cfg["mode"],
            "footprint_gb": round(after, 3), "last_status": status,
        })
        if cfg["pause_between_s"]:
            time.sleep(float(cfg["pause_between_s"]))

    total = timedelta(seconds=int(time.time() - started))
    log(f"campaign end: {done_count} iterations completed, elapsed {total}, "
        f"footprint {footprint_gb():.1f} GB")
    return 0


# ── entry ────────────────────────────────────────────────────────────────────
def show_status(cfg: dict) -> None:
    state = load_state()
    print(f"footprint now        : {footprint_gb():.2f} GB  (soft {cfg['disk_soft_gb']} GB, hard {cfg['disk_hard_gb']} GB)")
    print(f"free on this volume  : {free_gb():.1f} GB")
    if not state:
        print("no campaign state yet")
        return
    for key in ("completed", "updated_utc", "instrument", "mode", "last_status"):
        if key in state:
            print(f"{key:21}: {state[key]}")
    if LEDGER.exists():
        rows = list(csv.DictReader(LEDGER.open(encoding="utf-8")))
        ok = sum(1 for r in rows if r["status"] == "ok")
        print(f"ledger               : {len(rows)} iterations, {ok} ok, {len(rows) - ok} not ok")
        if rows:
            times = [float(r["seconds"]) for r in rows if r["status"] == "ok" and r["seconds"]]
            if times:
                print(f"median iteration     : {sorted(times)[len(times) // 2]:.0f}s")


def dry_run(cfg: dict) -> int:
    """Validate without committing to a campaign: show the command and the answers, run one iteration."""
    mode = str(cfg.get("mode", "forward")).lower()
    work = f"--cycles {cfg.get('cycles', 10)}" if mode == "backtest" else f"--runs {cfg.get('runs_per_iteration', 200)}"
    print("command per iteration:")
    print(f"  {sys.executable} -u price_modeling.py --mode {mode} --seed <n> "
          f"--asset-type {cfg.get('asset_type', 'forex')} --instrument {cfg.get('instrument', 'EUR/USD')} "
          f"--timeframe {cfg.get('timeframe', '5d')} --interval {cfg.get('interval', '1h')} "
          f"--save-state {cfg.get('output_mode', '6')} {work} --non-interactive")
    print("\nanswers sent on stdin (as non-interactive fallback):")
    for i, line in enumerate(build_answers(cfg).rstrip("\n").split("\n"), 1):
        print(f"  {i:>2}  {line!r}")
    print(f"\nfootprint {footprint_gb():.2f} GB, free {free_gb():.1f} GB")
    print(f"auto-prune: {cfg.get('auto_prune_each_run', True)} ({' '.join(cfg.get('prune_command', []))})")
    print("\nrunning ONE iteration to prove the sequence is complete ...")
    test_seed = int(cfg["base_seed"]) % 10_000_000
    status, code, detail = run_once(cfg, test_seed, print)
    if status == "fatal":
        print(f"\nFAILED - the answers ran out: {detail}")
        return 2
    if status == "failed":
        print(f"\nIteration failed (exit {code}): {detail}")
        return 1
    print("\nOK - the execution and answer sequence is complete. Safe to start a campaign.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=None, help="stop after this many successful iterations")
    parser.add_argument("--hours", type=float, default=None, help="stop after this many hours")
    parser.add_argument("--resume", action="store_true", help="continue from loop_state.json")
    parser.add_argument("--dry-run", action="store_true", help="validate the answer sequence with one iteration")
    parser.add_argument("--status", action="store_true", help="show campaign state and disk, run nothing")
    parser.add_argument("--config", default=None, help="path to config.toml")
    parser.add_argument("--mode", choices=["forward", "backtest"], default=None, help="override [loop] mode (forward|backtest)")
    parser.add_argument("--instrument", default=None, help="override [loop] instrument")
    parser.add_argument("--asset-type", default=None, help="override [loop] asset_type")
    parser.add_argument("--timeframe", default=None, help="override [loop] timeframe (1d, 5d, 1m, 3m, 1y, 5y)")
    parser.add_argument("--interval", default=None, help="override [loop] interval (10m, 1h, 1d, etc.)")
    parser.add_argument("--runs", type=int, default=None, help="override [loop] runs_per_iteration")
    parser.add_argument("--cycles", type=int, default=None, help="override [loop] cycles")
    parser.add_argument("--save-state", default=None, help="override [loop] output_mode (1-6)")
    parser.add_argument("--disk-hard-gb", type=float, default=None, help="override the hard footprint ceiling")
    parser.add_argument("--disk-soft-gb", type=float, default=None, help="override the soft footprint threshold")
    parser.add_argument("--cloud-backend", choices=["onedrive", "hf", "azure", "folder", "rclone"], default=None,
                        help="archive snapshots to cloud backend after pruning (onedrive|hf|azure)")
    parser.add_argument("--auto-prune", action="store_true", default=None, help="force auto-prune after each iteration")
    parser.add_argument("--no-auto-prune", action="store_true", help="disable auto-pruning after each iteration")
    parser.add_argument("--slurm-task", type=int, default=None, help="run as a single task in a Slurm array job")
    args = parser.parse_args()

    cfg = load_loop_config(Path(args.config) if args.config else None)
    if args.mode:
        cfg["mode"] = args.mode
    if args.instrument:
        cfg["instrument"] = args.instrument
    if args.asset_type:
        cfg["asset_type"] = args.asset_type
    if args.timeframe:
        cfg["timeframe"] = args.timeframe
    if args.interval:
        cfg["interval"] = args.interval
    if args.runs:
        cfg["runs_per_iteration"] = args.runs
    if args.cycles:
        cfg["cycles"] = args.cycles
    if args.save_state:
        cfg["output_mode"] = str(args.save_state)
    if args.disk_hard_gb:
        cfg["disk_hard_gb"] = args.disk_hard_gb
    if args.disk_soft_gb:
        cfg["disk_soft_gb"] = args.disk_soft_gb
    if args.cloud_backend:
        cfg["cloud_backend"] = args.cloud_backend
    if args.auto_prune:
        cfg["auto_prune_each_run"] = True
    if args.no_auto_prune:
        cfg["auto_prune_each_run"] = False

    if not SIM.exists():
        raise SystemExit(f"price_modeling.py not found beside this script ({SIM})")
    if args.status:
        show_status(cfg)
        return 0
    if args.dry_run:
        return dry_run(cfg)

    # Support Slurm array jobs (Apocrita / cloud environment)
    slurm_id = args.slurm_task or os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_id is not None:
        task_id = int(slurm_id)
        seed = (int(cfg["base_seed"]) + task_id) % 10_000_000
        print(f"[slurm-task {task_id}] Running seed {seed} ...", flush=True)
        status, code, detail = run_once(cfg, seed, print)
        if status == "ok" and cfg.get("auto_prune_each_run", True):
            run_pruner(cfg, print)
        return code

    if args.iterations is None and args.hours is None:
        raise SystemExit("Give --iterations or --hours (or --dry-run / --status / --slurm-task). "
                         "An unbounded campaign is never what you want on a shared server.")
    return campaign(cfg, args.iterations, args.hours, args.resume)


if __name__ == "__main__":
    sys.exit(main())
