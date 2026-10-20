"""
ideas.py - callable interface to the idea registry in prompt_ideas.txt.

The registry is plain text so it stays readable and diffable; this script makes
it searchable and trackable. Completion status lives in ideas_status.json so
that prompt_ideas.txt itself never changes once written.

    python ideas.py 201              show one idea in full
    python ideas.py 201 202 310      show several
    python ideas.py --list           one line per idea
    python ideas.py --search leak    search title, why and do
    python ideas.py --phase 1        everything for phase 1
    python ideas.py --tag costs      filter by tag
    python ideas.py --effort S       filter by effort
    python ideas.py --todo           undone only (combines with other filters)
    python ideas.py --next           the next undone idea, lowest phase first
    python ideas.py --next 5         the next five
    python ideas.py --random         one at random, for when you are stuck
    python ideas.py --done 201 202   mark complete
    python ideas.py --undone 201     mark incomplete again
    python ideas.py --stats          progress by phase and by tag
    python ideas.py --tags           every tag with a count

Filters combine: `python ideas.py --phase 2 --tag leakage --todo`
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "prompt_ideas.txt"
STATUS = HERE / "ideas_status.json"

HEADER_RE = re.compile(r"^###\s+IDEA-(\d+)\s*::\s*(.+?)\s*$")
META_RE = re.compile(
    r"^phase:\s*(\d+)\s*\|\s*effort:\s*([SML])\s*\|\s*tags:\s*(.+?)\s*$"
)
PHASE_NAMES = {
    0: "stop the bleeding",
    1: "data foundation",
    2: "feature factory",
    3: "model layer",
    4: "validation",
    5: "production",
}
EFFORT_NAMES = {"S": "under 2h", "M": "half a day", "L": "multi-day"}


@dataclass
class Idea:
    num: int
    title: str
    phase: int = 0
    effort: str = "M"
    tags: list[str] = field(default_factory=list)
    why: str = ""
    do: str = ""

    def haystack(self) -> str:
        return " ".join([self.title, self.why, self.do, " ".join(self.tags)]).lower()


def parse_registry(path: Path = REGISTRY) -> list[Idea]:
    """Read every '### IDEA-NNN :: Title' block. Unparseable blocks are skipped loudly."""
    if not path.exists():
        sys.exit(f"Registry not found: {path}")
    ideas: list[Idea] = []
    current: Idea | None = None
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.rstrip()
        header = HEADER_RE.match(line)
        if header:
            if current is not None:
                ideas.append(current)
            current = Idea(num=int(header.group(1)), title=header.group(2))
            continue
        if current is None:
            continue
        meta = META_RE.match(line)
        if meta:
            current.phase = int(meta.group(1))
            current.effort = meta.group(2)
            current.tags = [t.strip() for t in meta.group(3).split(",") if t.strip()]
        elif line.startswith("why:"):
            current.why = line[4:].strip()
        elif line.startswith("do:"):
            current.do = line[3:].strip()
    if current is not None:
        ideas.append(current)
    seen: dict[int, int] = {}
    for idea in ideas:
        seen[idea.num] = seen.get(idea.num, 0) + 1
    duplicates = sorted(n for n, c in seen.items() if c > 1)
    if duplicates:
        print(f"[warn] duplicate idea numbers: {duplicates}", file=sys.stderr)
    return sorted(ideas, key=lambda i: i.num)


def load_status() -> dict[str, str]:
    try:
        data = json.loads(STATUS.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_status(status: dict[str, str]) -> None:
    STATUS.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def is_done(idea: Idea, status: dict[str, str]) -> bool:
    return status.get(str(idea.num)) == "done"


def wrap(text: str, width: int = 88, indent: str = "     ") -> str:
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return f"\n{indent}".join(lines)


def show(idea: Idea, status: dict[str, str]) -> None:
    mark = "DONE" if is_done(idea, status) else "todo"
    print(f"\nIDEA-{idea.num}  [{mark}]  {idea.title}")
    print(
        f"  phase {idea.phase} ({PHASE_NAMES.get(idea.phase, '?')})"
        f"  |  effort {idea.effort} ({EFFORT_NAMES.get(idea.effort, '?')})"
        f"  |  tags: {', '.join(idea.tags)}"
    )
    if idea.why:
        print(f"  why: {wrap(idea.why)}")
    if idea.do:
        print(f"  do:  {wrap(idea.do)}")


def one_line(idea: Idea, status: dict[str, str]) -> str:
    mark = "x" if is_done(idea, status) else " "
    return f"[{mark}] {idea.num}  p{idea.phase} {idea.effort}  {idea.title}"


def apply_filters(ideas: list[Idea], args, status: dict[str, str]) -> list[Idea]:
    out = ideas
    if args.phase is not None:
        out = [i for i in out if i.phase == args.phase]
    if args.tag:
        want = args.tag.lower()
        out = [i for i in out if any(want in t.lower() for t in i.tags)]
    if args.effort:
        out = [i for i in out if i.effort == args.effort.upper()]
    if args.search:
        needle = args.search.lower()
        out = [i for i in out if needle in i.haystack()]
    if args.todo:
        out = [i for i in out if not is_done(i, status)]
    return out


def cmd_stats(ideas: list[Idea], status: dict[str, str]) -> None:
    done = sum(1 for i in ideas if is_done(i, status))
    total = len(ideas)
    pct = 100.0 * done / total if total else 0.0
    print(f"\n{done}/{total} ideas complete ({pct:.1f}%)\n")
    print("by phase:")
    for phase in sorted({i.phase for i in ideas}):
        group = [i for i in ideas if i.phase == phase]
        gdone = sum(1 for i in group if is_done(i, status))
        bar = "#" * int(20 * gdone / len(group)) if group else ""
        print(
            f"  phase {phase} {PHASE_NAMES.get(phase, '?'):<20} "
            f"{gdone:>3}/{len(group):<3} {bar}"
        )
    print("\nby effort:")
    for effort in ("S", "M", "L"):
        group = [i for i in ideas if i.effort == effort]
        if not group:
            continue
        gdone = sum(1 for i in group if is_done(i, status))
        print(f"  {effort} ({EFFORT_NAMES[effort]:<12}) {gdone:>3}/{len(group)}")
    counts: dict[str, list[int]] = {}
    for idea in ideas:
        for tag in idea.tags:
            bucket = counts.setdefault(tag, [0, 0])
            bucket[1] += 1
            if is_done(idea, status):
                bucket[0] += 1
    print("\ntop tags:")
    for tag, (gdone, gtotal) in sorted(counts.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f"  {tag:<18} {gdone:>3}/{gtotal}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Callable interface to the idea registry in prompt_ideas.txt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Filters combine, e.g. python ideas.py --phase 2 --tag leakage --todo",
    )
    parser.add_argument("nums", nargs="*", type=int, help="idea numbers to show")
    parser.add_argument("--list", action="store_true", help="one line per idea")
    parser.add_argument("--search", metavar="TEXT", help="search title, why and do")
    parser.add_argument("--phase", type=int, help="filter by phase 0-5")
    parser.add_argument("--tag", help="filter by tag (substring match)")
    parser.add_argument("--effort", choices=list("SMLsml"), help="filter by effort")
    parser.add_argument("--todo", action="store_true", help="undone only")
    parser.add_argument("--next", nargs="?", const=1, type=int, metavar="N",
                        help="next N undone ideas, lowest phase first")
    parser.add_argument("--random", action="store_true", help="one at random")
    parser.add_argument("--done", nargs="+", type=int, metavar="N", help="mark complete")
    parser.add_argument("--undone", nargs="+", type=int, metavar="N", help="mark incomplete")
    parser.add_argument("--stats", action="store_true", help="progress summary")
    parser.add_argument("--tags", action="store_true", help="list every tag with a count")
    args = parser.parse_args()

    ideas = parse_registry()
    by_num = {i.num: i for i in ideas}
    status = load_status()

    if args.done or args.undone:
        for num in args.done or []:
            if num not in by_num:
                print(f"[warn] no IDEA-{num}", file=sys.stderr)
                continue
            status[str(num)] = "done"
            print(f"IDEA-{num} marked done: {by_num[num].title}")
        for num in args.undone or []:
            status.pop(str(num), None)
            print(f"IDEA-{num} marked todo")
        save_status(status)
        return

    if args.stats:
        cmd_stats(ideas, status)
        return

    if args.tags:
        counts: dict[str, int] = {}
        for idea in ideas:
            for tag in idea.tags:
                counts[tag] = counts.get(tag, 0) + 1
        for tag, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"{count:>4}  {tag}")
        return

    if args.nums:
        for num in args.nums:
            if num in by_num:
                show(by_num[num], status)
            else:
                print(f"[warn] no IDEA-{num}", file=sys.stderr)
        return

    selected = apply_filters(ideas, args, status)

    if args.next is not None:
        undone = [i for i in selected if not is_done(i, status)]
        undone.sort(key=lambda i: (i.phase, {"S": 0, "M": 1, "L": 2}[i.effort], i.num))
        if not undone:
            print("Nothing left matching those filters.")
            return
        for idea in undone[: args.next]:
            show(idea, status)
        return

    if args.random:
        pool = [i for i in selected if not is_done(i, status)] or selected
        if not pool:
            print("Nothing matches those filters.")
            return
        show(random.choice(pool), status)
        return

    if not selected:
        print("Nothing matches those filters.")
        return

    if args.list or args.phase is not None or args.tag or args.effort or args.todo:
        for idea in selected:
            print(one_line(idea, status))
        done = sum(1 for i in selected if is_done(i, status))
        print(f"\n{len(selected)} ideas, {done} done")
        return

    if args.search:
        for idea in selected:
            show(idea, status)
        print(f"\n{len(selected)} match")
        return

    cmd_stats(ideas, status)
    print("\n(use --help for the full command list)")


if __name__ == "__main__":
    main()
