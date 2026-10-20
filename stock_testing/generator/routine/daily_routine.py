"""
daily_routine.py - the "morning routine" for the ML project.

Read-only and bounded: it never edits or deletes project files and writes only inside routine/.

Stages (each one is a plain function below):
  1. digest_inputs      what changed in the .toml / .txt / .md files since the last run
  2. audit_histories    is the stored simulation data sane, and is the model's accuracy real or mechanical?
  3. market_snapshot    live prices for every instrument found in ml_results/, vol regime, scenario bands
  4. honest_baseline    expanding-window walk-forward test on REAL daily bars (the number that matters)
  5. upgrade_progress   which of the 5 institutional upgrades exist in the code yet
  6. write_report       routine/reports/latest.html + latest.md, one row appended to routine_log.csv

Usage:
  python routine/daily_routine.py              # full run
  python routine/daily_routine.py --offline    # skip Yahoo Finance (audit + progress only)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import re
import sys
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
GEN = HERE.parent
RESULTS = GEN / "looped_analysis" / "ml_results"
OUT = HERE / "reports"
STATE = HERE / "routine_state.json"
LOG = HERE / "routine_log.csv"
MAX_AUDIT_ROWS = 20_000          # keep the audit fast on the 100k-row histories
CONTEXT = {"S&P500": "^GSPC", "VIX": "^VIX", "DXY": "DX-Y.NYB"}
TRACKED_INPUTS = ["config.toml", "rf_tuning.toml", "ideas.txt", "prompt_ideas.txt"]


# ─── 1. inputs ────────────────────────────────────────────────────────────────
def digest_inputs() -> dict:
    files = [GEN / n for n in TRACKED_INPUTS] + sorted((GEN / "Learn").glob("*.md")) + sorted((GEN / "Learn").glob("*.txt"))
    now = {str(p.relative_to(GEN)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}
    try:
        before = json.loads(STATE.read_text(encoding="utf-8")).get("hashes", {})
    except (FileNotFoundError, json.JSONDecodeError):
        before = {}
    changed = [k for k, v in now.items() if k in before and before[k] != v]
    new = [k for k in now if k not in before]
    cfg = tomllib.loads((GEN / "config.toml").read_text(encoding="utf-8")) if (GEN / "config.toml").exists() else {}
    ml = cfg.get("ml", {})
    return {"hashes": now, "changed": changed, "new": new if before else [], "first_run": not before,
            "label_mode": ml.get("label_mode", "?"), "models": ml.get("models", []),
            "tune_rf": ml.get("tune_rf", False), "pruning": cfg.get("pruning", {}).get("enabled", False)}


# ─── 2. data audit ────────────────────────────────────────────────────────────
def _load_ml_model():
    sys.path.insert(0, str(GEN))
    try:
        import ml_model
        return ml_model
    except Exception as exc:  # the audit still works without it
        print(f"[routine] ml_model import failed ({exc}); using a generic feature set")
        return None


def _read_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if df[col].dtype == object:
            as_num = pd.to_numeric(df[col], errors="coerce")
            if as_num.notna().mean() > 0.9:
                df[col] = as_num
    return df


def audit_histories() -> pd.DataFrame:
    ml = _load_ml_model()
    rows = []
    for path in sorted(RESULTS.glob("*/*/analysis_history.csv")):
        name = "/".join(path.parts[-3:-1])
        df = _read_history(path)
        n = len(df)
        price_cols = [c for c in ("first_open", "lowest_low", "last_close") if c in df]
        bad_price = float((df[price_cols] <= 0).any(axis=1).mean() * 100) if price_cols else float("nan")
        nan_ret = float(df["total_return_pct"].isna().mean() * 100)
        row = dict(set=name, rows=n, pct_nonpositive_price=round(bad_price, 1), pct_return_nan=round(nan_ret, 1))
        d = df.dropna(subset=["total_return_pct"]).tail(MAX_AUDIT_ROWS).reset_index(drop=True)
        y = (d["total_return_pct"] > 0).astype(int).to_numpy()
        if len(d) < 100 or len(np.unique(y)) < 2:
            row.update(usable_rows=len(d), verdict="too little usable data")
            rows.append(row)
            continue
        if ml is not None:
            d = ml._add_engineered_features(d)
            feats = [c for c in ml.FEATURES if c in d.columns]
        else:
            feats = [c for c in d.select_dtypes("number").columns if c not in {"total_return_pct", "run", "cycle", "loop"}]
        X = d[feats].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median()).fillna(0).clip(-1e12, 1e12)
        cut = int(len(d) * 0.7)
        rf = RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=-1).fit(X.iloc[:cut], y[:cut])
        acc = float(rf.score(X.iloc[cut:], y[cut:]))
        base = float(max(y[cut:].mean(), 1 - y[cut:].mean()))
        shuf = np.random.default_rng(1).permutation(y)
        rf2 = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1).fit(X.iloc[:cut], shuf[:cut])
        shuf_acc = float(rf2.score(X.iloc[cut:], shuf[cut:]))
        n_leaky, worst = "n/a", ""
        if ml is not None:
            audit = ml.leakage_audit(X, y)
            n_leaky = int(audit["flag"].sum())
            worst = ", ".join(f"{r.feature} ({r.separation:.2f})" for r in audit.head(3).itertuples())
        gap = acc - max(base, shuf_acc)
        verdict = "MECHANICAL - accuracy is leakage" if gap > 0.10 else "ok"
        row.update(usable_rows=len(d), return_label_acc=round(acc, 3), base_rate=round(base, 3),
                   shuffled_label_acc=round(shuf_acc, 3), mechanical_gap=round(gap, 3),
                   leaky_features=n_leaky, worst_features=worst, verdict=verdict)
        rows.append(row)
    return pd.DataFrame(rows)


# ─── 3 + 4. market snapshot and honest baseline ───────────────────────────────
def watchlist() -> dict[str, str]:
    """Instrument folders in ml_results -> Yahoo symbols (stock_sap -> SAP, forex_gbp_usd -> GBPUSD=X)."""
    found: dict[str, str] = {}
    for folder in sorted(p for p in RESULTS.glob("*") if p.is_dir()):
        kind, _, rest = folder.name.partition("_")
        if kind == "forex":
            found[rest.upper().replace("_", "/")] = rest.upper().replace("_", "") + "=X"
        elif kind == "stock":
            found[rest.upper()] = rest.upper()
    return found


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def _features(s: pd.Series) -> pd.DataFrame:
    """Everything here is known at the close of day t; the label is the direction of day t+1."""
    r = np.log(s).diff()
    X = pd.concat({f"lag{i}": r.shift(i) for i in range(5)}, axis=1)
    X["vol10"] = r.rolling(10).std()
    X["mom20"] = r.rolling(20).sum()
    X["rsi"] = _rsi(s)
    return X


def market_snapshot(marks: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    import yfinance as yf
    symbols = {**marks, **CONTEXT}
    raw = yf.download(list(symbols.values()), period="5y", interval="1d", auto_adjust=True, progress=False)
    inv = {v: k for k, v in symbols.items()}
    close = raw["Close"].rename(columns=inv)
    high, low = raw["High"].rename(columns=inv), raw["Low"].rename(columns=inv)
    asof = str(close.index[-1].date())
    snap, base_rows = [], []
    for name in symbols:
        if name not in close or close[name].dropna().size < 300:
            continue
        s = close[name].dropna()
        r = np.log(s).diff().dropna()
        daily = float(r.tail(20).std())
        vol_hist = r.rolling(20).std().dropna()
        tr = pd.concat([high[name] - low[name], (high[name] - s.shift()).abs(), (low[name] - s.shift()).abs()], axis=1).max(axis=1)
        atr_pct = float(tr.rolling(14).mean().iloc[-1] / s.iloc[-1] * 100)
        snap.append(dict(
            instrument=name, last=round(float(s.iloc[-1]), 4),
            r1d=round((s.iloc[-1] / s.iloc[-2] - 1) * 100, 2), r1m=round((s.iloc[-1] / s.iloc[-22] - 1) * 100, 2),
            r3m=round((s.iloc[-1] / s.iloc[-64] - 1) * 100, 2), vol20_ann=round(daily * np.sqrt(252) * 100, 1),
            vol_pctile=int(round((vol_hist < daily).mean() * 100)), rsi14=round(float(_rsi(s).iloc[-1]), 1),
            vs_sma200=round((s.iloc[-1] / s.rolling(200).mean().iloc[-1] - 1) * 100, 2),
            atr14_pct=round(atr_pct, 2), band_1d_pct=round(daily * 100, 2), band_5d_pct=round(daily * np.sqrt(5) * 100, 2)))
        if name in marks:
            X = _features(s)
            y = (np.log(s).diff().shift(-1) > 0).astype(float)
            ok = X.notna().all(axis=1)
            d = X[ok].copy()
            d["y"] = y[ok]
            live = d.iloc[[-1]].drop(columns="y")           # latest close -> predicts tomorrow
            d = d.iloc[:-1].dropna()
            start = int(len(d) * 0.5)
            hits = []
            for i in range(start, len(d), 90):
                m = RandomForestClassifier(n_estimators=100, min_samples_leaf=20, random_state=0, n_jobs=-1)
                m.fit(d.iloc[:i].drop(columns="y"), d["y"].iloc[:i])
                te = d.iloc[i:i + 90]
                hits.extend((m.predict(te.drop(columns="y")) == te["y"]).tolist())
            acc, n = float(np.mean(hits)), len(hits)
            base = float(max(d["y"].iloc[start:].mean(), 1 - d["y"].iloc[start:].mean()))
            band = 1.96 * np.sqrt(0.25 / n)
            final = RandomForestClassifier(n_estimators=100, min_samples_leaf=20, random_state=0, n_jobs=-1)
            final.fit(d.drop(columns="y"), d["y"])
            p_up = float(final.predict_proba(live)[0][1])
            edge = acc > 0.5 + band and acc > base
            base_rows.append(dict(instrument=name, walk_fwd_acc=round(acc, 3), majority_baseline=round(base, 3), n=n,
                                  chance_band=round(band, 3), p_up_raw=round(p_up, 3),
                                  lean=("UP" if p_up > 0.5 else "DOWN") if edge else "none (no edge over chance)"))
    return pd.DataFrame(snap), pd.DataFrame(base_rows), asof


# ─── 5. upgrade progress ──────────────────────────────────────────────────────
UPGRADES = [
    ("U1", "Kill label leakage", "config.toml [ml] label_mode = \"forecast\" and a hard-fail leakage guard"),
    ("U2", "Data-integrity gate", "def validate_run_frame() called before training"),
    ("U3", "Honest evaluation", "honest_eval.py - walk-forward vs chance/majority/shuffled baselines with a CI"),
    ("U4", "Costs and sizing", "def pnl_report() - net-of-cost Sharpe, vol-targeted size"),
    ("U5", "Regime drift + run card", "regime_check.py - vol-percentile drift alert and a JSON run card"),
]


def upgrade_progress(inputs: dict) -> list[dict]:
    code = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in GEN.glob("*.py"))
    has_hard_fail = bool(re.search(r"def\s+(assert_no_leakage|leakage_guard)\s*\(", code))
    checks = {
        "U1": inputs["label_mode"] == "forecast" and has_hard_fail,
        "U2": bool(re.search(r"def\s+validate_run_frame\s*\(", code)),
        "U3": (GEN / "honest_eval.py").exists(),
        "U4": bool(re.search(r"def\s+pnl_report\s*\(", code)),
        "U5": (GEN / "regime_check.py").exists(),
    }
    return [dict(id=i, title=t, done=checks[i], needs=need) for i, t, need in UPGRADES]


# ─── 6. report ────────────────────────────────────────────────────────────────
def _tip() -> tuple[str, str, str]:
    sys.path.insert(0, str(GEN))
    try:
        from daily_ml_advisor import get_daily_advice
        return get_daily_advice()
    except Exception:
        return ("", "", "")


def _storage() -> str:
    out = []
    for name in ("looped_results", "looped_analysis"):
        p = GEN / name
        files = [f for f in p.rglob("*") if f.is_file()] if p.exists() else []
        out.append(f"{name}: {len(files):,} files, {sum(f.stat().st_size for f in files) / 1e6:,.1f} MB")
    return " | ".join(out)


def _table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p class='dim'>no data</p>"
    return "<div style='overflow-x:auto'>" + df.to_html(index=False, classes="t", border=0, escape=True) + "</div>"


def write_report(inputs, audit, snap, honest, asof, progress, offline) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    done = sum(p["done"] for p in progress)
    worst_gap = float(audit["mechanical_gap"].max()) if "mechanical_gap" in audit and audit["mechanical_gap"].notna().any() else float("nan")
    worst_bad = float(audit["pct_nonpositive_price"].max()) if len(audit) else float("nan")
    best_edge = float((honest["walk_fwd_acc"] - honest["majority_baseline"]).max()) if len(honest) else float("nan")
    tip = _tip()
    changed = inputs["changed"] + inputs["new"]
    prog_html = "".join(
        f"<tr><td>{p['id']}</td><td>{'DONE' if p['done'] else 'TODO'}</td><td>{html.escape(p['title'])}</td>"
        f"<td>{html.escape(p['needs'])}</td></tr>" for p in progress)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ML Morning Routine {today}</title><style>
body{{background:#0b0e14;color:#e6edf3;font:15px/1.55 system-ui,Segoe UI,sans-serif;margin:0;padding:24px}}
.w{{max-width:1100px;margin:0 auto}} h1,h2{{color:#58a6ff;border-bottom:1px solid #21262d;padding-bottom:6px}}
.k{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:16px 0}}
.c{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;text-align:center}}
.c b{{display:block;font-size:26px;margin:6px 0}} .bad{{color:#f85149}} .ok{{color:#3fb950}} .warn{{color:#d29922}}
.t{{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0 24px}} .t th,.t td{{border:1px solid #30363d;padding:6px 10px;text-align:left}}
.t th{{background:#21262d}} .dim{{color:#8b949e}} .box{{background:#161b22;border-left:4px solid #58a6ff;padding:12px 16px;border-radius:0 8px 8px 0}}
</style></head><body><div class="w">
<h1>ML Morning Routine - {today}</h1>
<p class="dim">Market data as of {html.escape(asof)}{' (offline run - no live data)' if offline else ''}. Read-only run; scenario bands are volatility ranges, not forecasts.</p>
<div class="k">
<div class="c">Data corruption<b class="{'bad' if worst_bad > 5 else 'ok'}">{worst_bad:.0f}%</b><span class="dim">worst set: rows with price &le; 0</span></div>
<div class="c">Mechanical accuracy gap<b class="{'bad' if worst_gap > 0.1 else 'ok'}">{worst_gap:+.2f}</b><span class="dim">label-leak accuracy minus chance</span></div>
<div class="c">Honest edge (real bars)<b class="{'ok' if best_edge > 0.04 else 'warn'}">{best_edge:+.3f}</b><span class="dim">best walk-forward acc minus baseline</span></div>
<div class="c">Institutional upgrades<b class="{'ok' if done == 5 else 'warn'}">{done}/5</b><span class="dim">done in the code</span></div></div>
<div class="box"><b>Config in force:</b> label_mode = <code>{html.escape(str(inputs['label_mode']))}</code>, pruning = {inputs['pruning']}, tune_rf = {inputs['tune_rf']}, models = {html.escape(', '.join(inputs['models']))}.<br>
<b>Inputs changed since last run:</b> {html.escape(', '.join(changed)) if changed else ('first run - baseline stored' if inputs['first_run'] else 'none')}</div>
<h2>Market now and scenario bands</h2>{_table(snap)}
<h2>Honest walk-forward test on real daily bars</h2>{_table(honest)}
<p class="dim">chance_band = 95% range around 50% for pure luck. A lean is only shown when accuracy beats both the band and the majority-class baseline; ignore p_up_raw otherwise.</p>
<h2>Stored simulation data audit</h2>{_table(audit)}
<h2>Upgrade checklist</h2><table class="t"><tr><th>#</th><th>state</th><th>upgrade</th><th>detected when</th></tr>{prog_html}</table>
<h2>Tip of the day {html.escape(tip[0])}</h2><p><b>{html.escape(tip[1])}</b><br>{html.escape(tip[2])}</p>
<p class="dim">Storage: {html.escape(_storage())}</p></div></body></html>"""
    (OUT / "latest.html").write_text(page, encoding="utf-8")
    (OUT / f"{today}.html").write_text(page, encoding="utf-8")

    md = [f"# ML Morning Routine - {today}", f"Market data as of {asof}{' (offline)' if offline else ''}.", "",
          f"- Worst data corruption: {worst_bad:.0f}% of rows have a price <= 0",
          f"- Worst mechanical accuracy gap: {worst_gap:+.2f} (leakage, not skill)",
          f"- Best honest edge on real bars: {best_edge:+.3f}", f"- Upgrades done: {done}/5",
          f"- label_mode = {inputs['label_mode']}", f"- Inputs changed: {', '.join(changed) if changed else 'none'}", ""]
    for title, frame in (("Market now", snap), ("Honest walk-forward", honest), ("Data audit", audit)):
        md += [f"## {title}", frame.to_string(index=False) if len(frame) else "no data", ""]
    md += ["## Upgrades"] + [f"- [{'x' if p['done'] else ' '}] {p['id']} {p['title']}" for p in progress]
    (OUT / "latest.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    row = dict(date=today, asof=asof, worst_corruption_pct=worst_bad, worst_mechanical_gap=worst_gap, best_honest_edge=best_edge,
               upgrades_done=done, label_mode=inputs["label_mode"])
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    STATE.write_text(json.dumps({"date": today, "hashes": inputs["hashes"]}, indent=2), encoding="utf-8")
    return OUT / "latest.html"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="skip Yahoo Finance")
    args = ap.parse_args()

    inputs = digest_inputs()
    print("[1/6] inputs digested:", len(inputs["hashes"]), "files;", "changed:", inputs["changed"] + inputs["new"])
    audit = audit_histories()
    print("[2/6] audited", len(audit), "history files")
    snap = honest = pd.DataFrame()
    asof = "n/a"
    if not args.offline:
        try:
            snap, honest, asof = market_snapshot(watchlist())
            print("[3/6] market snapshot as of", asof)
        except Exception as exc:
            print(f"[3/6] market data unavailable ({type(exc).__name__}: {exc}); continuing offline")
            args.offline = True
    progress = upgrade_progress(inputs)
    print("[4/6] upgrades done:", sum(p["done"] for p in progress), "/ 5")
    path = write_report(inputs, audit, snap, honest, asof, progress, args.offline)
    print("[5/6] report ->", path)

    # Stage 6: the scan published as the artifact. The page is rewritten in full each run,
    # and one row per instrument is appended to scan_history.csv so the next run can say
    # what changed. A scan failure must never lose the local report written above.
    if args.offline:
        print("[6/6] scan skipped (offline)")
        return
    try:
        import scan as _scan
        import scan_page as _scan_page
        result = _scan.run_scan(watchlist())
        page = _scan_page.write_page(result)
        _scan.append_history(result)
        print(f"[6/6] scan page -> {page}")
        print(f"       as of {result['asof']}: {len(result['changes'])} change(s), "
              f"{len(result['issues'])} data issue(s) across {len(result['rows'])} series")
    except Exception as exc:
        print(f"[6/6] scan failed ({type(exc).__name__}: {exc}); the local report above is unaffected")


if __name__ == "__main__":
    main()
