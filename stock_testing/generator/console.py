"""console.py - dependency-free terminal helpers: section headers, tables, colour, progress, safe prompts.

Output
    banner, section, kv, table, Progress, manifest. Colour is used only when stdout is a real
    terminal and NO_COLOR is not set, so piped logs and answer-file runs stay plain text.
Input
    ask, ask_choice, ask_number, ask_text, ask_yes_no. Every prompt validates its answer, re-asks a
    few times with the reason, then stops cleanly (no traceback). When the input ends (an answer file
    ran out, stdin closed) the default is used if there is one, otherwise the run stops with a message.
"""
from __future__ import annotations

import atexit
import logging
import math
import os
import re
import shutil
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

MAX_ATTEMPTS = 5
MAX_ANSWER_LENGTH = 200
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_CODES = {"bold": "1", "dim": "2", "up": "32", "down": "31", "warn": "33", "accent": "36",
          "title": "1;36", "muted": "90", "ok": "32", "bad": "31"}


# ─── Streams and colour ───────────────────────────────────────────────────────
def _prepare_streams() -> None:
    """Box-drawing characters must never crash a redirected stdout (Windows defaults to cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if "utf" not in (stream.encoding or "").lower():
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def _enable_windows_vt() -> bool:
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)                      # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(mode.value & 0x0004) or bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def _colour_wanted() -> bool:
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not getattr(sys.stdout, "isatty", lambda: False)():
        return False
    return _enable_windows_vt() if os.name == "nt" else True


_prepare_streams()
# matplotlib logs ~26 "findfont: Failed to find font weight ..." lines per run (mplfinance asks for
# weights DejaVu Sans does not ship); they are harmless and only bury the real output.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
COLOUR = _colour_wanted()
INTERACTIVE = bool(getattr(sys.stdout, "isatty", lambda: False)())


def paint(text: Any, style: str | None) -> str:
    text = str(text)
    if not COLOUR or not style or style not in _CODES:
        return text
    return f"\x1b[{_CODES[style]}m{text}\x1b[0m"


def visible_len(text: str) -> int:
    return len(_ANSI.sub("", text))


def width() -> int:
    return max(60, min(112, shutil.get_terminal_size((100, 24)).columns - 1))


# ─── Formatting ───────────────────────────────────────────────────────────────
def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def fmt_pct(value: Any, places: int = 2, signed: bool = True) -> str:
    if not is_number(value):
        return paint("n/a", "muted")
    text = f"{value:+.{places}f}%" if signed else f"{value:.{places}f}%"
    if not signed or abs(value) < 0.5 * 10 ** -places:
        return text
    return paint(text, "up" if value > 0 else "down")


def fmt_num(value: Any, places: int = 2, suffix: str = "", prefix: str = "") -> str:
    """'1,234.50%' for numbers; a muted 'n/a' (without the unit) for NaN / None."""
    return f"{prefix}{value:,.{places}f}{suffix}" if is_number(value) else paint("n/a", "muted")


def fmt_price(value: Any) -> str:
    if not is_number(value):
        return paint("n/a", "muted")
    magnitude = abs(float(value))
    if magnitude >= 1 or magnitude == 0:
        places = 2 if magnitude >= 100 else 3 if magnitude >= 10 else 4
    else:                                    # 0.000005234, not 0.000000: keep ~4 significant digits
        places = min(12, max(4, 3 - math.floor(math.log10(magnitude))))
    return f"{value:,.{places}f}"


def fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    return f"{minutes}m{sec:02d}s" if minutes < 60 else f"{minutes // 60}h{minutes % 60:02d}m"


def fmt_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.1f} {unit}"
        size /= 1024
    return f"{size:,.1f} GB"


# ─── Layout ───────────────────────────────────────────────────────────────────
def banner(title: str, subtitle: str = "") -> None:
    """Top-of-section heading: a heavy rule with the title, optional dim subtitle."""
    line = f"━━ {title.upper()} "
    print()
    print(paint(line + "━" * max(3, width() - len(line)), "title"))
    if subtitle:
        print(paint(f"   {subtitle}", "muted"))


def section(title: str) -> None:
    line = f"── {title} "
    print()
    print(paint(line + "─" * max(3, width() - len(line)), "accent"))


def note(text: str, style: str = "muted") -> None:
    print(paint(f"   {text}", style))


def warn(text: str) -> None:
    print(paint(f"   ! {text}", "warn"))


def kv(rows: Iterable[tuple[str, Any]], indent: int = 3) -> None:
    """Aligned 'label   value' block."""
    rows = [(str(k), "" if v is None else str(v)) for k, v in rows]
    if not rows:
        return
    pad = max(len(k) for k, _ in rows) + 2
    for key, value in rows:
        print(" " * indent + paint(key.ljust(pad), "muted") + value)


def table(rows: Sequence[Sequence[Any]], headers: Sequence[str], align: str | None = None,
          indent: int = 3, max_rows: int | None = None) -> None:
    """Minimal table: dim upper-case headers, one rule, aligned columns ('l' / 'r' per column).

    Cells may carry ANSI colour (fmt_pct ...); widths are measured on the visible text.
    """
    rows = [[("" if c is None else str(c)) for c in row] for row in rows]
    if max_rows is not None and len(rows) > max_rows:
        hidden = len(rows) - max_rows
        rows = rows[:max_rows] + [["…"] + [f"({hidden} more rows)"] + [""] * (len(headers) - 2)]
    align = align or "l" * len(headers)
    widths = [max([len(h)] + [visible_len(r[i]) for r in rows if i < len(r)]) for i, h in enumerate(headers)]

    def cell(text: str, i: int) -> str:
        gap = widths[i] - visible_len(text)
        return (" " * gap + text) if align[i] == "r" else (text + " " * gap)

    print(" " * indent + "  ".join(paint(cell(h.upper(), i), "muted") for i, h in enumerate(headers)))
    print(" " * indent + "  ".join(paint("─" * w, "muted") for w in widths))
    for row in rows:
        print(" " * indent + "  ".join(cell(row[i] if i < len(row) else "", i) for i in range(len(headers))).rstrip())


class _BarSafeStream:
    """Stands in for sys.stdout / sys.stderr while a Progress bar is on screen.

    Without it, a print() during the loop lands on the bar's line (the bar has no newline yet), freezes a
    copy of the bar there, and the live bar carries on one line lower: two bars. Here any other output first
    clears the bar line, and the bar is redrawn under each finished line, so it always stays the last line.
    """

    def __init__(self, stream: Any, bar: "Progress"):
        self._stream, self._bar = stream, bar

    def write(self, text: str) -> int:
        if text and self._bar._drawn:
            self._bar._clear()
        written = self._stream.write(text)
        if text.endswith("\n") and self._bar._active:
            self._stream.flush()
            self._bar._draw()
        return written

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name: str) -> Any:            # encoding, isatty, fileno, buffer ...
        return getattr(self._stream, name)


def _columns() -> int:
    """Usable characters on one terminal line (a line that wraps breaks the carriage-return redraw)."""
    return max(20, shutil.get_terminal_size((100, 24)).columns - 1)


class Progress:
    """One self-updating line in a terminal (kept below any other output); about ten plain lines in a log."""

    def __init__(self, total: int, label: str):
        self.total, self.label = max(1, int(total)), label
        self.start = time.perf_counter()
        self._last_step = -1
        self._line = ""
        self._drawn = self._active = False
        self._out: Any = None
        self._saved: tuple[Any, Any] | None = None

    # -- terminal plumbing ------------------------------------------------------------------------
    def _install(self) -> None:
        self._saved = (sys.stdout, sys.stderr)
        self._out = sys.stdout
        sys.stdout = _BarSafeStream(self._saved[0], self)
        sys.stderr = _BarSafeStream(self._saved[1], self)
        self._active = True
        _ACTIVE_BARS.append(self)

    def _uninstall(self) -> None:
        if self._saved is not None:
            if isinstance(sys.stdout, _BarSafeStream) and sys.stdout._bar is self:
                sys.stdout = self._saved[0]
            if isinstance(sys.stderr, _BarSafeStream) and sys.stderr._bar is self:
                sys.stderr = self._saved[1]
            self._saved = None
        self._active = False
        if self in _ACTIVE_BARS:
            _ACTIVE_BARS.remove(self)

    def _draw(self) -> None:
        self._out.write("\r" + self._line)
        self._out.flush()
        self._drawn = True

    def _clear(self) -> None:
        self._out.write("\r" + " " * _columns() + "\r")
        self._out.flush()
        self._drawn = False

    def _render(self, done: int, detail: str) -> str:
        frac = done / self.total
        elapsed = time.perf_counter() - self.start
        eta = elapsed / frac - elapsed if frac > 0 else 0.0
        cols = _columns()
        bar_width = max(8, min(26, cols - 50))
        filled = int(round(frac * bar_width))
        head = f"   {self.label} "
        count = f" {done:>{len(str(self.total))}}/{self.total} {100 * frac:5.1f}%  "
        eta_text = f"eta {fmt_seconds(eta)}"
        if len(head) + bar_width + len(count) + len(eta_text) > cols:      # very narrow terminal: count only
            plain = f"   {self.label} {done}/{self.total} {100 * frac:.0f}%"[:cols]
            return plain + " " * (cols - len(plain))
        room = cols - len(head) - bar_width - len(count) - len(eta_text) - 2      # what is left for the detail
        if len(detail) > room:
            detail = detail[:room - 1] + "…" if room > 5 else ""
        plain_len = len(head) + bar_width + len(count) + len(eta_text) + (2 + len(detail) if detail else 0)
        bar = paint("█" * filled, "accent") + paint("░" * (bar_width - filled), "muted")
        line = head + bar + count + paint(eta_text, "muted") + ("  " + detail if detail else "")
        return line + " " * max(0, cols - plain_len)   # pad to the full width: wipes a longer previous line

    def update(self, done: int, detail: str = "") -> None:
        done = min(max(0, int(done)), self.total)
        if INTERACTIVE:
            if not self._active and done < self.total:
                self._install()
            self._line = self._render(done, detail)
            if self._out is None:                         # finished on the very first update
                self._out = sys.stdout
            self._draw()
            if done >= self.total:
                self._out.write("\n")
                self._out.flush()
                self._drawn = False
                self._uninstall()
            return
        # Plain log: one line at the end for short loops, about every 20% for long ones.
        step = done * 5 // self.total
        if done == self.total or (self.total > 20 and step != self._last_step):
            self._last_step = step
            print(f"   {self.label} {done:,}/{self.total:,}" + (f"  {detail}" if detail else ""))

    def done(self, summary: str = "") -> None:
        self.update(self.total)
        if summary:
            note(f"{summary} in {fmt_seconds(time.perf_counter() - self.start)}")


_ACTIVE_BARS: list[Progress] = []


@atexit.register
def _release_bars() -> None:
    """A loop that stopped early (error, Ctrl+C) leaves its bar half drawn: end the line, restore the streams."""
    for bar in list(_ACTIVE_BARS):
        if bar._drawn and bar._out is not None:
            bar._out.write("\n")
            bar._out.flush()
            bar._drawn = False
        bar._uninstall()


def manifest(groups: dict[str, Sequence[Path]], root: Path | None = None) -> None:
    """'What was written where': one line per group with file count and size."""
    rows = []
    for label, paths in groups.items():
        files = [Path(p) for p in paths if Path(p).is_file()]
        if not files:
            continue
        size = sum(p.stat().st_size for p in files)
        shown = files[0] if len(files) == 1 else files[0].parent
        try:
            shown = shown.relative_to(root) if root else shown
        except ValueError:
            pass
        rows.append([label, f"{len(files):,}", fmt_bytes(size), str(shown)])
    if rows:
        table(rows, ["output", "files", "size", "where"], "lrrl")


# ─── Safe prompts ─────────────────────────────────────────────────────────────
def clean_answer(raw: str) -> str:
    """Strip BOMs (PowerShell answer files), control characters and surrounding space."""
    text = unicodedata.normalize("NFKC", raw.replace("﻿", ""))
    return "".join(ch for ch in text if ch.isprintable()).strip()


def _read(prompt: str) -> str | None:
    try:
        return input(prompt)
    except EOFError:
        return None
    except KeyboardInterrupt:
        print()
        raise SystemExit("Cancelled.")


def ask(prompt: str, parse: Callable[[str], Any], default: str | None = None) -> Any:
    """Ask until parse(answer) succeeds; parse raises ValueError("why") for a bad answer.

    An empty answer means the default (when there is one). After MAX_ATTEMPTS bad answers the run
    stops cleanly, so a garbled answer file cannot loop forever.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = _read(prompt)
        if raw is None:
            if default is None:
                raise SystemExit(f"\nInput ended before an answer to: {prompt.strip()}")
            print(f"{default}   (input ended: default used)")
            raw = default
        answer = clean_answer(raw)
        if len(answer) > MAX_ANSWER_LENGTH:
            print(paint(f"   ✗ answer too long ({len(answer)} characters)", "warn"))
            continue
        if not answer and default is not None:
            answer = default
        if not answer:
            print(paint("   ✗ an answer is required", "warn"))
            continue
        try:
            return parse(answer)
        except ValueError as error:
            left = MAX_ATTEMPTS - attempt
            print(paint(f"   ✗ {error}", "warn") + (paint(f"  ({left} {'try' if left == 1 else 'tries'} left)", "muted") if left else ""))
    raise SystemExit(f"No valid answer after {MAX_ATTEMPTS} tries - stopping. ({prompt.strip()})")


def ask_choice(prompt: str, options: dict[str, Any], default: str | None = None,
               aliases: dict[str, str] | None = None) -> Any:
    keys = {str(k).lower(): v for k, v in options.items()}
    alias = {str(k).lower(): str(v).lower() for k, v in (aliases or {}).items()}

    def parse(raw: str) -> Any:
        key = alias.get(raw.lower(), raw.lower())
        if key in keys:
            return keys[key]
        raise ValueError(f"'{raw}' is not an option - choose {' / '.join(options)}")
    return ask(prompt, parse, default)


def ask_yes_no(prompt: str, default: bool | None = None) -> bool:
    return ask_choice(prompt, {"y": True, "n": False}, None if default is None else ("y" if default else "n"),
                      aliases={"yes": "y", "no": "n"})


def parse_number(raw: str, lo: float, hi: float, integer: bool = False, what: str = "value") -> float | int:
    """A finite number inside [lo, hi]; '1,10' is refused (decimal comma) instead of read as 110."""
    if "," in raw:
        raise ValueError(f"'{raw}': use a dot for decimals and no thousands separators")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"'{raw}' is not a number") from None
    if not math.isfinite(value):
        raise ValueError(f"'{raw}' is not a finite number")
    if integer:
        if value != int(value):
            raise ValueError(f"'{raw}' must be a whole number")
        value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{what} {value:g} is outside the allowed range {lo:g} to {hi:g}")
    return value


def ask_number(prompt: str, lo: float, hi: float, integer: bool = False, default: float | None = None,
               what: str = "value") -> float | int:
    return ask(prompt, lambda raw: parse_number(raw, lo, hi, integer, what), None if default is None else f"{default:g}")


def ask_text(prompt: str, pattern: str, default: str | None = None, max_len: int = 40,
             what: str = "value", hint: str = "") -> str:
    def parse(raw: str) -> str:
        if not raw:
            raise ValueError(f"{what} cannot be empty")
        if len(raw) > max_len:
            raise ValueError(f"{what} is longer than {max_len} characters")
        if re.fullmatch(pattern, raw) is None:
            raise ValueError(f"'{raw}' is not a valid {what}" + (f" ({hint})" if hint else ""))
        return raw
    return ask(prompt, parse, default)
