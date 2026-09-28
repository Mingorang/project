#!/usr/bin/env python3
"""
tidy_repo.py - sort loose files in a repo's root into neat folders.

Usage (run from inside your cloned repo, or pass the path):
    python tidy_repo.py            # dry run: only prints what WOULD move
    python tidy_repo.py --apply    # actually moves the files
    python tidy_repo.py path/to/repo --apply

- Only touches files sitting directly in the repo root. Existing folders are left alone.
- Uses `git mv` for tracked files so history is preserved (plain move otherwise).
- Keeps README, LICENSE, .gitignore, config files, etc. in the root.
- Never overwrites: name clashes get a numeric suffix.
- Edit CATEGORIES / KEEP_IN_ROOT below to suit your repo.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# folder name -> extensions that go in it
CATEGORIES = {
    "notebooks": {".ipynb"},
    "src": {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h",
            ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".r", ".m"},
    "scripts": {".sh", ".bat", ".ps1"},
    "data": {".csv", ".tsv", ".json", ".xml", ".xlsx", ".xls", ".parquet",
             ".db", ".sqlite", ".pkl", ".npy", ".npz", ".h5", ".txt"},
    "docs": {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".md", ".rtf", ".tex"},
    "images": {".png", ".jpg", ".jpeg", ".gif", ".svg", ".bmp", ".webp"},
    "web": {".html", ".css"},
}

# Files that should stay in the root (case-insensitive names)
KEEP_IN_ROOT = {
    "readme.md", "readme", "readme.txt", "license", "license.md", "license.txt",
    "requirements.txt", "environment.yml", "pyproject.toml", "setup.py",
    "setup.cfg", "package.json", "package-lock.json", "yarn.lock", "makefile",
    "dockerfile", "docker-compose.yml", "tsconfig.json", "cargo.toml",
    "cargo.lock", "go.mod", "go.sum", "pipfile", "pipfile.lock", "changelog.md",
    "contributing.md", "code_of_conduct.md",
}

EXT_TO_FOLDER = {ext: folder for folder, exts in CATEGORIES.items() for ext in exts}


def is_tracked(repo: Path, path: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--error-unmatch", path.name],
        capture_output=True,
    )
    return result.returncode == 0


def unique_target(dest_dir: Path, name: str) -> Path:
    target = dest_dir / name
    stem, suffix, n = target.stem, target.suffix, 1
    while target.exists():
        target = dest_dir / f"{stem}_{n}{suffix}"
        n += 1
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Sort loose repo files into folders.")
    parser.add_argument("repo", nargs="?", default=".", help="path to the repo (default: .)")
    parser.add_argument("--apply", action="store_true", help="actually move files")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists():
        sys.exit(f"{repo} doesn't look like a git repo (no .git folder).")

    moves = []
    for item in sorted(repo.iterdir()):
        if not item.is_file() or item.name.startswith("."):
            continue
        if item.name.lower() in KEEP_IN_ROOT:
            continue
        folder = EXT_TO_FOLDER.get(item.suffix.lower())
        if folder is None:
            folder = "misc"
        moves.append((item, folder))

    if not moves:
        print("Nothing to move. Root is already tidy.")
        return

    print(("MOVING" if args.apply else "DRY RUN (nothing will change)") + f" in {repo}\n")
    for item, folder in moves:
        dest_dir = repo / folder
        target = unique_target(dest_dir, item.name)
        print(f"  {item.name}  ->  {folder}/{target.name}")
        if args.apply:
            dest_dir.mkdir(exist_ok=True)
            if is_tracked(repo, item):
                subprocess.run(["git", "-C", str(repo), "mv", str(item), str(target)], check=True)
            else:
                shutil.move(str(item), str(target))

    if args.apply:
        print("\nDone. Next steps:")
        print("  1. Check that your code still finds its files (paths like 'data.csv' may need updating).")
        print("  2. git status   # review")
        print("  3. git add -A && git commit -m 'Organize repo into folders' && git push")
    else:
        print("\nLooks good? Re-run with --apply to do it.")


if __name__ == "__main__":
    main()
