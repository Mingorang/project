"""
scan.py - the daily Yahoo Finance scan behind the published artifact.

Three things, per the agreed scope (8 symbols: the five instruments plus S&P 500, VIX, DXY):

  1. regime and extremes  - where volatility sits in its own 5-year history, RSI, trend, bands
  2. what changed         - only what moved since the previous run, read from scan_history.csv
  3. data health          - stale quotes, missing bars, impossible prices, failed downloads

Writes reports/scan.html (the artifact page, replaced each run) and appends one row per
instrument to scan_history.csv (the trail the "what changed" section reads next time).
"""
from __future__ import annotations

import datetime as dt
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
HISTORY = HERE / "scan_history.csv"
CONTEXT = {"S&P 500": "^GSPC", "VIX": "^VIX", "DXY": "DX-Y.NYB"}


def _load_settings() -> dict:
    """[scan] thresholds and the RSI settings from ../config.toml; the built-in values if it cannot be read."""
    settings = {"vol_pctile_shift": 10, "big_move_sigma": 2.0, "stale_bars": 4, "flat_run": 4,
                "min_history_bars": 250, "rsi_period": 14, "rsi_overbought": 70.0, "rsi_oversold": 30.0}
    try:
        if str(HERE.parent) not in sys.path:
            sys.path.insert(0, str(HERE.parent))
        import project_config
        cfg = project_config.load_config(quiet=True)
        settings.update(cfg["scan"])
        settings.update({key: cfg["indicators"][key] for key in ("rsi_period", "rsi_overbought", "rsi_oversold")})
    except Exception as exc:  # the routine must keep running even with a broken config.toml
        print(f"[scan] config.toml not applied ({type(exc).__name__}: {exc}); built-in thresholds used")
    return settings


_SETTINGS = _load_settings()
# A change must clear one of these to be worth printing (config.toml [scan] / [indicators]).
VOL_PCTILE_SHIFT = int(_SETTINGS["vol_pctile_shift"])        # percentile points
RSI_OVERBOUGHT, RSI_OVERSOLD = float(_SETTINGS["rsi_overbought"]), float(_SETTINGS["rsi_oversold"])
RSI_PERIOD = int(_SETTINGS["rsi_period"])
BIG_MOVE_SIGMA = float(_SETTINGS["big_move_sigma"])          # today's move in units of its own 20-day sigma
STALE_BARS = int(_SETTINGS["stale_bars"])                    # business days without a new bar
FLAT_RUN = int(_SETTINGS["flat_run"])                        # identical consecutive closes
MIN_HISTORY_BARS = int(_SETTINGS["min_history_bars"])        # fewer usable daily bars = failed download


def _rsi(s: pd.Series, n: int | None = None) -> pd.Series:
    n = int(n or RSI_PERIOD)
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def _max_flat_run(s: pd.Series) -> int:
    """Longest run of identical consecutive closes - the signature of a stale feed."""
    best = run = 0
    for same in (s.diff() == 0).to_numpy():
        run = run + 1 if same else 0
        best = max(best, run)
    return int(best)


# -- 1. regime and extremes ---------------------------------------------------
def regime_row(name: str, close: pd.Series, high: pd.Series, low: pd.Series) -> dict:
    s = close.dropna()
    r = np.log(s).diff().dropna()
    daily = float(r.tail(20).std())
    vol_hist = r.rolling(20).std().dropna()
    # Align high/low with the closes first: in a multi-symbol download one missing bar used to put a NaN
    # into the last 14 true ranges and blank the ATR (e.g. CL=F next to the stock-market calendar).
    h, l = high.reindex(s.index), low.reindex(s.index)
    tr = pd.concat([h - l, (h - s.shift()).abs(), (l - s.shift()).abs()], axis=1).max(axis=1).dropna()
    sma200 = float(s.rolling(200).mean().iloc[-1])
    move_sigma = float(np.log(s.iloc[-1] / s.iloc[-2]) / daily) if daily > 0 else 0.0
    return dict(
        instrument=name,
        last=round(float(s.iloc[-1]), 4),
        r1d=round(float(s.iloc[-1] / s.iloc[-2] - 1) * 100, 2),
        r1m=round(float(s.iloc[-1] / s.iloc[-22] - 1) * 100, 2) if len(s) > 22 else float("nan"),
        move_sigma=round(move_sigma, 2),
        vol20_ann=round(daily * np.sqrt(252) * 100, 1),
        vol_pctile=int(round(float((vol_hist < daily).mean()) * 100)),
        rsi14=round(float(_rsi(s).iloc[-1]), 1),
        vs_sma200=round(float(s.iloc[-1] / sma200 - 1) * 100, 2) if np.isfinite(sma200) else float("nan"),
        atr14_pct=round(float(tr.rolling(14).mean().iloc[-1] / s.iloc[-1]) * 100, 2) if len(tr) >= 14 else float("nan"),
        band_1d_pct=round(daily * 100, 2),
        band_5d_pct=round(daily * np.sqrt(5) * 100, 2),
        last_bar=str(s.index[-1].date()),
    )


# -- 3. data health -----------------------------------------------------------
def health_rows(name: str, close: pd.Series, high: pd.Series, low: pd.Series,
                asof: pd.Timestamp) -> list[dict]:
    issues: list[dict] = []
    s = close.dropna()

    def add(severity: str, issue: str, detail: str) -> None:
        issues.append(dict(instrument=name, severity=severity, issue=issue, detail=detail))

    if s.empty:
        add("critical", "no data", "the download returned nothing")
        return issues

    age = int(np.busday_count(s.index[-1].date(), asof.date()))
    if age > STALE_BARS:
        add("critical", "stale series", f"last bar {s.index[-1].date()}, {age} business days old")

    recent = s.tail(60)
    missing = len(pd.bdate_range(recent.index[0], recent.index[-1]).difference(recent.index))
    if missing > 6:
        add("warning", "missing bars",
            f"{missing} business days absent across the last 60 bars (public holidays inflate this)")

    if (s <= 0).any():
        add("critical", "impossible price", f"{int((s <= 0).sum())} non-positive closes")
    if (high.dropna() < low.dropna()).any():
        add("critical", "High below Low", f"{int((high.dropna() < low.dropna()).sum())} bars")

    flat = _max_flat_run(s.tail(120))
    if flat >= FLAT_RUN:
        add("warning", "flat quote", f"{flat} identical consecutive closes within the last 120 bars")

    r = np.log(s).diff().dropna().tail(250)
    if len(r) > 30 and float(r.std()) > 0:
        z = (r - r.mean()).abs() / r.std()
        if (z > 10).any():
            add("warning", "extreme return",
                f"{int((z > 10).sum())} bar(s) beyond 10 sigma - check for a bad split or dividend adjustment")
    return issues


# -- 2. what changed since the previous run -----------------------------------
def changes_vs_previous(rows: list[dict], prev: pd.DataFrame | None) -> list[dict]:
    out: list[dict] = []

    def add(instrument: str, change: str, detail: str) -> None:
        out.append(dict(instrument=instrument, change=change, detail=detail))

    prev_by: dict[str, pd.Series] = {}
    if prev is not None and len(prev):
        block = prev[prev["scan_date"] == prev["scan_date"].iloc[-1]]
        prev_by = {row["instrument"]: row for _, row in block.iterrows()}

    for row in rows:
        name = row["instrument"]
        if abs(row["move_sigma"]) >= BIG_MOVE_SIGMA:
            add(name, "large move",
                f"{row['r1d']:+.2f}% today, {abs(row['move_sigma']):.1f}x its own daily sigma")

        previous = prev_by.get(name)
        if previous is None:
            continue

        shift = row["vol_pctile"] - int(previous["vol_pctile"])
        if abs(shift) >= VOL_PCTILE_SHIFT:
            add(name, "volatility regime",
                f"percentile {int(previous['vol_pctile'])} to {row['vol_pctile']} ({shift:+d} points)")

        before, now = float(previous["rsi14"]), row["rsi14"]
        for level, label in ((RSI_OVERBOUGHT, "overbought"), (RSI_OVERSOLD, "oversold")):
            if (before < level <= now) or (before > level >= now):
                add(name, f"RSI crossed {level:.0f}", f"{before:.1f} to {now:.1f}, the {label} boundary")

        was, is_now = float(previous["vs_sma200"]), row["vs_sma200"]
        if np.isfinite(was) and np.isfinite(is_now) and np.sign(was) != np.sign(is_now):
            add(name, "crossed the 200-day average",
                f"now {'above' if is_now > 0 else 'below'} it ({is_now:+.2f}%)")
    return out


# -- orchestration ------------------------------------------------------------
def run_scan(marks: dict[str, str]) -> dict:
    """Download, measure, compare with the previous run. Returns everything the page needs."""
    import yfinance as yf

    symbols = {**marks, **CONTEXT}
    raw = yf.download(list(symbols.values()), period="5y", interval="1d",
                      auto_adjust=True, progress=False)
    inverse = {v: k for k, v in symbols.items()}
    close = raw["Close"].rename(columns=inverse)
    high = raw["High"].rename(columns=inverse)
    low = raw["Low"].rename(columns=inverse)
    asof = close.index[-1]

    rows: list[dict] = []
    issues: list[dict] = []
    failed: list[str] = []
    for name in symbols:
        if name not in close or close[name].dropna().size < MIN_HISTORY_BARS:
            failed.append(name)
            issues.append(dict(instrument=name, severity="critical", issue="download failed",
                               detail=f"fewer than {MIN_HISTORY_BARS} usable bars returned"))
            continue
        rows.append(regime_row(name, close[name], high[name], low[name]))
        issues.extend(health_rows(name, close[name], high[name], low[name], asof))

    prev = pd.read_csv(HISTORY) if HISTORY.exists() else None
    first_run = prev is None or not len(prev)
    return dict(asof=str(asof.date()), rows=rows, issues=issues,
                changes=changes_vs_previous(rows, prev), failed=failed, first_run=first_run,
                marks=list(marks), context=list(CONTEXT))


def append_history(scan: dict) -> Path:
    """One row per instrument per run - the trail the next run reads to find changes."""
    frame = pd.DataFrame(scan["rows"])
    frame.insert(0, "scan_date", dt.date.today().isoformat())
    frame.to_csv(HISTORY, mode="a", header=not HISTORY.exists(), index=False)
    return HISTORY
