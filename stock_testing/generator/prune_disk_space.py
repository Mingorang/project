"""
prune_disk_space.py — Local Disk Space Reclamation & File Pruner
Addresses the 78,000 files / 7.6GB disk space emergency.

Usage:
  python prune_disk_space.py --status        # Show current file count & space used
  python prune_disk_space.py --prune-png     # Remove chart PNGs (saves 80-90% of disk space)
  python prune_disk_space.py --compress-zip  # Archive old CSV runs into a compressed zip
  python prune_disk_space.py --clear-cache   # Remove __pycache__ and temporary yfinance caches
"""
from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path


def get_dir_stats(path: Path) -> tuple[int, int]:
    """Return (file_count, total_bytes) in path recursively."""
    if not path.exists():
        return 0, 0
    count = 0
    total_bytes = 0
    for root, _, files in os.walk(path):
        for f in files:
            count += 1
            try:
                total_bytes += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return count, total_bytes


def format_bytes(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def show_status(root_dir: Path) -> None:
    print(f"\n========================================================")
    print(f"DISK SPACE AUDIT: {root_dir}")
    print(f"========================================================")
    total_files = 0
    total_size = 0
    targets = [
        root_dir / "stock_testing" / "generator" / "looped_results",
        root_dir / "stock_testing" / "generator" / "looped_analysis",
        root_dir / "stock_testing" / "generator" / "reports",
        root_dir / "stock_testing" / "generator" / "Learn",
        root_dir / "stock_testing" / "generator" / "__pycache__",
    ]
    for target in targets:
        if target.exists():
            c, s = get_dir_stats(target)
            total_files += c
            total_size += s
            print(f"  {target.name:<25}: {c:>6,} files | {format_bytes(s):>10}")

    entire_c, entire_s = get_dir_stats(root_dir)
    print(f"--------------------------------------------------------")
    print(f"  TOTAL REPOSITORY USAGE   : {entire_c:>6,} files | {format_bytes(entire_s):>10}")
    print(f"========================================================\n")


def prune_pngs(root_dir: Path) -> None:
    results_dir = root_dir / "stock_testing" / "generator" / "looped_results"
    analysis_dir = root_dir / "stock_testing" / "generator" / "looped_analysis"
    deleted_count = 0
    freed_bytes = 0
    for folder in [results_dir, analysis_dir]:
        if folder.exists():
            for png in folder.glob("*.png"):
                try:
                    freed_bytes += png.stat().st_size
                    png.unlink()
                    deleted_count += 1
                except OSError:
                    pass
    print(f"[Prune] Removed {deleted_count:,} PNG images. Freed {format_bytes(freed_bytes)} of disk space!")


def compress_csvs(root_dir: Path) -> None:
    results_dir = root_dir / "stock_testing" / "generator" / "looped_results"
    if not results_dir.exists():
        print("[Compress] looped_results does not exist.")
        return
    csvs = list(results_dir.glob("data_*.csv"))
    if not csvs:
        print("[Compress] No CSV files found to archive.")
        return
    archive_path = results_dir / "archived_runs.zip"
    print(f"[Compress] Archiving {len(csvs):,} CSV runs into {archive_path.name}...")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as z:
        for csv in csvs:
            z.write(csv, arcname=csv.name)
            csv.unlink()
    print(f"[Compress] Finished compression! Created {format_bytes(archive_path.stat().st_size)} archive.")


def clear_caches(root_dir: Path) -> None:
    removed = 0
    for root, dirs, _ in os.walk(root_dir):
        for d in dirs:
            if d in ("__pycache__", ".pytest_cache", ".mypy_cache"):
                full_p = os.path.join(root, d)
                shutil.rmtree(full_p, ignore_errors=True)
                removed += 1
    print(f"[Cache] Removed {removed} cache directories.")


def main():
    parser = argparse.ArgumentParser(description="Prune and optimize disk space")
    parser.add_argument("--status", action="store_true", help="Display storage statistics")
    parser.add_argument("--prune-png", action="store_true", help="Delete generated PNG charts")
    parser.add_argument("--compress-zip", action="store_true", help="Compress CSVs into a zip archive")
    parser.add_argument("--clear-cache", action="store_true", help="Clear temporary Python caches")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent

    if args.prune_png:
        prune_pngs(repo_root)
    if args.compress_zip:
        compress_csvs(repo_root)
    if args.clear_cache:
        clear_caches(repo_root)

    show_status(repo_root)


if __name__ == "__main__":
    main()
