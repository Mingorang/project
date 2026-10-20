"""
scan_page.py - renders the scan into reports/scan.html, the page published as the artifact.

The file is an artifact fragment: no doctype, html or body tags - the publisher wraps it.
Colours are tokens defined for dark (default), then redefined for light, so the page follows
whatever theme the reader's browser is set to.
"""
from __future__ import annotations

import datetime as dt
import html
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAGE = HERE / "reports" / "scan.html"

STYLE = """
<style>
:root{
  --bg:#0a0e17; --surface:#121825; --surface-2:#1a2232; --line:#273145;
  --fg:#e4e9f2; --fg-dim:#8e9ab0; --fg-faint:#64718a;
  --ok:#4dd4c0; --warn:#e8a33d; --crit:#e5604d; --cool:#5b9bd9;
  --font-display:"Archivo","Helvetica Neue",Arial,sans-serif;
  --font-body:"Source Sans 3",system-ui,-apple-system,sans-serif;
  --font-data:"JetBrains Mono",ui-monospace,Consolas,monospace;
  color-scheme:dark;
}
@media (prefers-color-scheme:light){
  :root:not([data-theme="dark"]){
    --bg:#f6f8fb; --surface:#ffffff; --surface-2:#eceff5; --line:#d2d9e4;
    --fg:#0e131d; --fg-dim:#56637a; --fg-faint:#7d8798;
    --ok:#0c8872; --warn:#9c6206; --crit:#bf3220; --cool:#2e6ca8;
    color-scheme:light;
  }
}
:root[data-theme="light"]{
  --bg:#f6f8fb; --surface:#ffffff; --surface-2:#eceff5; --line:#d2d9e4;
  --fg:#0e131d; --fg-dim:#56637a; --fg-faint:#7d8798;
  --ok:#0c8872; --warn:#9c6206; --crit:#bf3220; --cool:#2e6ca8;
  color-scheme:light;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font-family:var(--font-body);font-size:16px;line-height:1.55;margin:0}
.wrap{max-width:980px;margin:0 auto;
  padding-left:max(16px,env(safe-area-inset-left,0px));
  padding-right:max(16px,env(safe-area-inset-right,0px));
  padding-block:36px 56px}
h1,h2{font-family:var(--font-display);text-wrap:balance;margin:0}
h1{font-size:clamp(26px,5vw,40px);font-weight:800;letter-spacing:-.022em;line-height:1.1}
h2{font-size:clamp(18px,3vw,23px);font-weight:700;letter-spacing:-.01em}
p{margin:0}
.eyebrow{font-family:var(--font-data);font-size:11px;font-weight:600;letter-spacing:.14em;
  text-transform:uppercase;color:var(--fg-faint)}
header{display:flex;flex-direction:column;gap:9px;padding-bottom:24px;border-bottom:1px solid var(--line)}
.asof{font-family:var(--font-data);font-size:12.5px;color:var(--fg-faint);margin-top:14px}
section{margin-top:40px;display:flex;flex-direction:column;gap:16px}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:18px}
.quiet{color:var(--fg-dim);font-size:14.5px}
.tablebox{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:9px 11px;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-family:var(--font-data);font-size:10.5px;font-weight:600;letter-spacing:.08em;
  text-transform:uppercase;color:var(--fg-faint)}
td.num{font-family:var(--font-data);font-variant-numeric:tabular-nums;text-align:right}
tr:last-child td{border-bottom:none}
tbody tr:hover{background:var(--surface-2)}
.inst{font-weight:600}
.up{color:var(--ok)} .down{color:var(--crit)} .flat{color:var(--fg-dim)}
.meter{display:inline-block;width:52px;height:7px;border-radius:4px;background:var(--surface-2);
  vertical-align:middle;margin-right:7px;overflow:hidden}
.meter span{display:block;height:100%;border-radius:4px;background:var(--cool)}
.card{display:flex;gap:13px;align-items:flex-start;background:var(--surface);
  border:1px solid var(--line);border-left:3px solid var(--cool);border-radius:0 9px 9px 0;padding:13px 16px}
.card.crit{border-left-color:var(--crit)}
.card.warn{border-left-color:var(--warn)}
.card .who{font-family:var(--font-data);font-size:12px;font-weight:600;flex:none;width:86px;color:var(--fg)}
.card .what{font-size:14.5px;color:var(--fg)}
.card .detail{font-size:13.5px;color:var(--fg-dim)}
.pill{display:inline-block;font-family:var(--font-data);font-size:10px;font-weight:600;
  letter-spacing:.05em;text-transform:uppercase;padding:2px 8px;border-radius:999px;
  border:1px solid var(--line);color:var(--fg-dim);background:var(--surface-2)}
.pill.ok{color:var(--ok);border-color:color-mix(in srgb,var(--ok) 40%,transparent)}
.pill.warn{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 40%,transparent)}
.pill.crit{color:var(--crit);border-color:color-mix(in srgb,var(--crit) 40%,transparent)}
footer{margin-top:48px;padding-top:18px;border-top:1px solid var(--line);
  font-size:12.5px;color:var(--fg-faint)}
code{font-family:var(--font-data);font-size:.92em;background:var(--surface-2);
  padding:1px 5px;border-radius:4px}
@media (max-width:560px){ .card{flex-direction:column;gap:4px} .card .who{width:auto} }
</style>
"""

FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=Archivo:wght@600;800&family=Source+Sans+3:wght@400;600&'
         'family=JetBrains+Mono:wght@400;600&display=swap">')


def _tone(value: float) -> str:
    if not np.isfinite(value) or abs(value) < 0.005:
        return "flat"
    return "up" if value > 0 else "down"


def _num(value: float, suffix: str = "", places: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return '<span class="flat">n/a</span>'
    return f"{value:.{places}f}{suffix}"


def _signed(value: float, suffix: str = "%") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return '<span class="flat">n/a</span>'
    return f'<span class="{_tone(value)}">{value:+.2f}{suffix}</span>'


def _regime_table(rows: list[dict], marks: list[str]) -> str:
    head = ("<thead><tr><th>Instrument</th><th>Last</th><th>1 day</th><th>1 month</th>"
            "<th>Vol percentile</th><th>20d vol</th><th>RSI 14</th><th>vs 200d</th>"
            "<th>1d band</th><th>5d band</th></tr></thead>")
    body = []
    for r in rows:
        pct = r["vol_pctile"]
        rsi = r["rsi14"]
        rsi_cls = "down" if rsi >= 70 else "up" if rsi <= 30 else "flat"
        tag = "" if r["instrument"] in marks else ' <span class="pill">context</span>'
        body.append(
            f'<tr><td class="inst">{html.escape(r["instrument"])}{tag}</td>'
            f'<td class="num">{r["last"]:g}</td>'
            f'<td class="num">{_signed(r["r1d"])}</td>'
            f'<td class="num">{_signed(r["r1m"])}</td>'
            f'<td class="num"><span class="meter"><span style="width:{max(pct,2)}%"></span></span>{pct}</td>'
            f'<td class="num">{_num(r["vol20_ann"], "%", 1)}</td>'
            f'<td class="num"><span class="{rsi_cls}">{rsi:.1f}</span></td>'
            f'<td class="num">{_signed(r["vs_sma200"])}</td>'
            f'<td class="num">&plusmn;{_num(r["band_1d_pct"], "%")}</td>'
            f'<td class="num">&plusmn;{_num(r["band_5d_pct"], "%")}</td></tr>')
    return f'<div class="panel tablebox"><table>{head}<tbody>{"".join(body)}</tbody></table></div>'


def _cards(items: list[dict], who_key: str, what_key: str, severity=None) -> str:
    out = []
    for it in items:
        cls = f" {severity(it)}" if severity else ""
        out.append(
            f'<div class="card{cls}"><div class="who">{html.escape(str(it[who_key]))}</div>'
            f'<div><div class="what">{html.escape(str(it[what_key]))}</div>'
            f'<div class="detail">{html.escape(str(it["detail"]))}</div></div></div>')
    return "".join(out)


def write_page(scan: dict, path: Path = PAGE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, changes, issues = scan["rows"], scan["changes"], scan["issues"]
    crit = [i for i in issues if i["severity"] == "critical"]
    warn = [i for i in issues if i["severity"] == "warning"]

    if crit:
        verdict = f'{len(crit)} data problem{"s" if len(crit) != 1 else ""} need attention before you trust today’s numbers.'
    elif changes:
        verdict = f'{len(changes)} thing{"s" if len(changes) != 1 else ""} moved since the last scan.'
    else:
        verdict = "Nothing moved enough to report. Volatility regimes, RSI and trend are where they were."

    if scan["first_run"]:
        changed_block = ('<p class="quiet">First run, so there is nothing to compare against yet. '
                         'Tomorrow this section lists only what moved.</p>')
    elif changes:
        changed_block = _cards(changes, "instrument", "change")
    else:
        changed_block = ('<p class="quiet">Nothing crossed a threshold. A change is reported when the '
                         'volatility percentile shifts by 10 points or more, RSI crosses 30 or 70, price '
                         'crosses its 200-day average, or a daily move exceeds twice its own sigma.</p>')

    if issues:
        health_block = _cards(sorted(issues, key=lambda i: i["severity"] != "critical"),
                              "instrument", "issue",
                              severity=lambda i: "crit" if i["severity"] == "critical" else "warn")
    else:
        health_block = ('<p class="quiet">All eight series passed: fresh bars, no missing days beyond '
                        'holidays, no non-positive prices, no flat-quote runs, no returns beyond 10 sigma.</p>')

    status = ('<span class="pill crit">%d critical</span>' % len(crit) if crit else '<span class="pill ok">no critical issues</span>')
    if warn:
        status += ' <span class="pill warn">%d warning%s</span>' % (len(warn), "s" if len(warn) != 1 else "")

    page = f"""<title>Daily Market Scan</title>
<meta name="claude:viewport-fit-sides" content="cover">
{FONTS}
{STYLE}
<div class="wrap">
<header>
  <p class="eyebrow">{len(scan['marks'])} instruments &middot; {len(scan['context'])} context &middot; Yahoo Finance daily bars</p>
  <h1>Daily market scan</h1>
  <p class="quiet">{html.escape(verdict)}</p>
  <p class="asof">Market data as of {html.escape(scan['asof'])} &middot; page rebuilt {dt.datetime.now():%Y-%m-%d %H:%M} &middot; {status}</p>
</header>

<section>
  <h2>What changed</h2>
  {changed_block}
</section>

<section>
  <h2>Regime and extremes</h2>
  {_regime_table(rows, scan['marks'])}
  <p class="quiet">Vol percentile is where the current 20-day volatility sits within its own five-year
  range: 3 means calmer than 97&#37; of the last five years. The bands are one standard deviation of
  daily moves, scaled to one and five days &mdash; they describe how far price usually travels, not which
  way it will go.</p>
</section>

<section>
  <h2>Data health</h2>
  {health_block}
</section>

<footer>
  Replaced on every run, so this link always shows the latest scan; one row per instrument per run is
  appended to <code>routine/scan_history.csv</code>, which is what the &ldquo;what changed&rdquo; section reads.
  Rebuild by hand with <code>python routine/daily_routine.py</code>.
  Bands are volatility ranges, not forecasts. Nothing here is investment advice.
</footer>
</div>
"""
    path.write_text(page, encoding="utf-8")
    return path
