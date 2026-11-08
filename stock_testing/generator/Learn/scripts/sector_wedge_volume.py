"""Reproduce every "measured" figure and the chart in Learn/GEOMETRY_OF_A_MARKET_CRASH.md.

    python Learn/scripts/sector_wedge_volume.py                     # the doc's data: up to the 22 Oct 2026 close
    python Learn/scripts/sector_wedge_volume.py --end 2027-01-01 --out somewhere_else

Writes sector_wedge_volume_facts.json and sector_wedge_volume.png into --out (default: Learn/img, the files the doc
uses). With a later --end, pass another --out folder, or the chart and facts stop matching the doc's text.
Needs numpy, pandas, matplotlib and yfinance (already in requirements.txt). Yahoo re-issues its dividend-adjusted
history now and then, so re-runs can differ in the 5th significant digit (measured: at most 7e-5 relative, two runs
on 23 Oct 2026 at 01:59 and 11:10 UTC). The doc quotes 2 or 3 digits, which did not move, and the last 60-day
window's identities reproduced exactly.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

SECTORS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLB", "XLU"]   # the 9 original SPDR sectors (since 1998)
W = 60                                                                     # trading days per window (the essay's choice)
MIN_DAYS = 5                                                               # hysteresis: minimum days in a state (4.6)
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "img"


def log_volume(X: np.ndarray, demean: bool) -> float:
    """log ||u1 ^ ... ^ un|| for the unit-length columns of X (demeaned first for the correlation version)."""
    if demean:
        X = X - X.mean(axis=0)
    U = X / np.linalg.norm(X, axis=0)
    sign, logdet = np.linalg.slogdet(U.T @ U)          # det of the Gram matrix = squared volume
    return 0.5 * logdet if sign > 0 else -np.inf


def hysteresis(lv: pd.Series, enter: pd.Series, leave: pd.Series, min_days: int = MIN_DAYS) -> list[int]:
    """Section 4.6: enter below `enter`, leave above `leave`, stay at least `min_days` in a state."""
    state, days, out = 0, 0, []
    for value, e, l in zip(lv, enter, leave):
        if np.isfinite(e) and np.isfinite(l):          # no thresholds yet = the first-year warm-up
            if state == 0 and value < e and days >= min_days:
                state, days = 1, 0
            elif state == 1 and value > l and days >= min_days:
                state, days = 0, 0
            days += 1
        out.append(state)
    return out


def measure(end: str) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(7)
    prices = yf.download(SECTORS + ["SPY"], start="2005-01-01", end=end, auto_adjust=True, progress=False)["Close"].dropna()
    logret = np.log(prices / prices.shift(1)).dropna()
    sec = logret[SECTORS]

    values = sec.to_numpy()
    idx = sec.index[W:]
    cos_lv = np.array([log_volume(values[i - W:i], False) for i in range(W, len(values))])
    cor_lv = np.array([log_volume(values[i - W:i], True) for i in range(W, len(values))])
    series = pd.DataFrame({"log_volume_cos": cos_lv, "log_volume_corr": cor_lv}, index=idx)
    series["volume_corr"] = np.exp(series["log_volume_corr"])
    series["spy"] = prices["SPY"].reindex(idx)
    spy_vol = logret["SPY"].rolling(20).std().reindex(idx) * np.sqrt(252) * 100
    series["spy_vol20"] = spy_vol

    # Point-in-time regimes (expanding percentiles, at least 1 year of history, decided on data up to t-1)
    lv = series["log_volume_corr"]
    low_thr = lv.expanding(252).quantile(0.15).shift(1)
    geom_regime = (lv < low_thr).astype(int)
    hi_thr = spy_vol.expanding(252).quantile(0.85).shift(1)
    vol_regime = (spy_vol > hi_thr).astype(int)
    hyst = hysteresis(lv, low_thr, lv.expanding(252).quantile(0.30).shift(1))
    series["geom_regime"], series["geom_regime_hyst"], series["vol_regime"] = geom_regime, hyst, vol_regime
    years = (series.index[-1] - series.index[252]).days / 365.25

    def switches(s: pd.Series) -> float:
        return float(s.iloc[252:].diff().abs().sum() / years)

    # Monte-Carlo null: 9 INDEPENDENT return vectors of length 60 (no correlation at all)
    null_cos = np.array([log_volume(rng.standard_normal((W, len(SECTORS))), False) for _ in range(4000)])
    null_cor = np.array([log_volume(rng.standard_normal((W, len(SECTORS))), True) for _ in range(4000)])

    # Identity checks on the LAST window, including the last day's return (correlation version)
    X = values[-W:] - values[-W:].mean(axis=0)
    R = np.corrcoef(X, rowvar=False)
    det_R = float(np.linalg.det(R))
    eig = np.linalg.eigvalsh(R)
    seq = []                                       # sequential regression: 1 - R^2 of sector i on sectors 1..i-1
    for i in range(len(SECTORS)):
        y = X[:, i]
        if i == 0:
            seq.append(1.0)
            continue
        A = X[:, :i]
        resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        seq.append(float(resid @ resid / (y @ y)))
    vif = np.diag(np.linalg.inv(R))
    loo = {}
    for j, name in enumerate(SECTORS):           # volume ratio when sector j is removed = sqrt(1 - R^2_j|rest)
        keep = [k for k in range(len(SECTORS)) if k != j]
        loo[name] = float(np.sqrt(det_R / np.linalg.det(R[np.ix_(keep, keep)])))
    # Lagrange identity in R^60: the 2x2 Gram determinant vs the sum of the squared bivector coefficients
    a, b = X[:, 0], X[:, 1]
    iu60 = np.triu_indices(W, 1)
    lagrange_gram_det = float(np.linalg.det(np.array([[a @ a, a @ b], [a @ b, b @ b]])))
    lagrange_bivector_sum = float(np.sum((a[iu60[0]] * b[iu60[1]] - a[iu60[1]] * b[iu60[0]]) ** 2))
    tri = np.array([[1, -0.5, -0.5], [-0.5, 1, -0.5], [-0.5, -0.5, 1]])

    # GBP/USD vs EUR/USD (the manual's 0.85) as a 2-blade: the 6 months to the last close (22 Apr to 22 Oct 2026)
    fx_start = str((pd.Timestamp(end) - pd.DateOffset(months=6) - pd.Timedelta(days=1)).date())
    fx = yf.download(["GBPUSD=X", "EURUSD=X"], start=fx_start, end=end, auto_adjust=True, progress=False)["Close"].dropna()
    fx_r = np.log(fx / fx.shift(1)).dropna()
    rho_fx = float(fx_r.corr().iloc[0, 1])

    def at(date: str, column: str = "volume_corr") -> float:
        return float(series[column].loc[:date].iloc[-1])

    def window_min(start: str, stop: str) -> tuple[str, float]:
        part = series["volume_corr"].loc[start:stop]
        return str(part.idxmin().date()), float(part.min())

    last = str(series.index[-1].date())
    facts = {
        "asof": last, "window": W, "days": int(len(series)),
        "volume_today": at(last),
        "volume_pctile_today_full": float((series["volume_corr"] < at(last)).mean() * 100),
        "volume_median_full": float(series["volume_corr"].median()),
        "min_2008": window_min("2008-06-01", "2009-06-30"), "min_2020": window_min("2020-02-01", "2020-06-30"),
        "volume_2018_02_09": at("2018-02-09"), "spy_vol20_2018_02_09": at("2018-02-09", "spy_vol20"),
        "volume_2017_median": float(series["volume_corr"].loc["2017"].median()),
        "volume_2018_q1_min": float(series["volume_corr"].loc["2018-01-15":"2018-04-30"].min()),
        "geom_regime_2018_02_09": int(series["geom_regime"].loc[:"2018-02-09"].iloc[-1]),
        "vol_regime_2018_02_09": int(series["vol_regime"].loc[:"2018-02-09"].iloc[-1]),
        "geom_regime_2020_03_16": int(series["geom_regime"].loc[:"2020-03-16"].iloc[-1]),
        "vol_regime_2020_03_16": int(series["vol_regime"].loc[:"2020-03-16"].iloc[-1]),
        "switches_per_year": {"geometry_p15": switches(series["geom_regime"]),
                              "geometry_hysteresis": switches(series["geom_regime_hyst"]),
                              "spy_volatility_p85": switches(series["vol_regime"])},
        "cos_vs_corr_logvol_corr": float(np.corrcoef(series["log_volume_cos"], series["log_volume_corr"])[0, 1]),
        "null_volume_corr": {"median": float(np.exp(np.median(null_cor))), "p5": float(np.exp(np.percentile(null_cor, 5))),
                             "p95": float(np.exp(np.percentile(null_cor, 95)))},
        "null_volume_cos_median": float(np.exp(np.median(null_cos))),
        "identities_today": {
            "det_R": det_R, "product_eigenvalues": float(np.prod(eig)),
            "product_sequential_1_minus_R2": float(np.prod(seq)),
            "largest_eigenvalue_share_pct": float(eig.max() / eig.sum() * 100),
            "effective_number_of_bets": float(eig.sum() ** 2 / np.sum(eig ** 2)),
            "leave_one_out_ratio": loo,
            "one_over_sqrt_vif": {name: float(1 / np.sqrt(v)) for name, v in zip(SECTORS, vif)},
            "lagrange_gram_det": lagrange_gram_det, "lagrange_bivector_sum": lagrange_bivector_sum,
            "lagrange_pairs": int(len(iu60[0])),
            "det_three_vectors_at_120_degrees": float(np.linalg.det(tri)),
            "correlations_today_min_max": [float(R[np.triu_indices(9, 1)].min()), float(R[np.triu_indices(9, 1)].max())],
        },
        "fx_rho_6m": rho_fx, "fx_area_2_blade": float(np.sqrt(1 - rho_fx ** 2)),
    }

    # ---- the other figures quoted in the doc ----
    vol = series["volume_corr"]
    year_min = {str(y): [str(g.idxmin().date()), float(g.min())] for y, g in vol.groupby(vol.index.year)}
    v_identity = float(np.sqrt(det_R))                    # the window ENDING on the last day (the identity block)
    hyst_s = series["geom_regime_hyst"]
    entries_2018 = hyst_s.loc["2018"].diff().fillna(0)

    def window_corr(end_date: str, cols=SECTORS) -> np.ndarray:
        """Correlation matrix of the 60 days BEFORE end_date (the series convention: usable at the open)."""
        pos = sec.index.get_indexer([pd.Timestamp(end_date)])[0]
        Xw = sec[cols].to_numpy()[pos - W:pos]
        return np.corrcoef(Xw - Xw.mean(axis=0), rowvar=False)

    def mode_share(end_date: str) -> float:
        e = np.linalg.eigvalsh(window_corr(end_date))
        return float(e.max() / e.sum() * 100)

    def mode_bets(end_date: str) -> float:
        e = np.linalg.eigvalsh(window_corr(end_date))
        return float(e.sum() ** 2 / np.sum(e ** 2))

    # average pairwise correlation per window vs log volume (the video's 23:40 comparison)
    iu = np.triu_indices(9, 1)
    avg_rho = np.array([np.corrcoef(values[i - W:i], rowvar=False)[iu].mean() for i in range(W, len(values))])

    # the video's 4-asset A/D example (all rho 0.2, then rho_AD -> 0.99)
    def ad_case(rho_ad: float) -> dict:
        R4 = np.full((4, 4), 0.2)
        np.fill_diagonal(R4, 1.0)
        R4[0, 3] = R4[3, 0] = rho_ad
        return {"mean_rho": float(R4[np.triu_indices(4, 1)].mean()), "det": float(np.linalg.det(R4)),
                "volume": float(np.sqrt(max(np.linalg.det(R4), 0.0)))}

    # the video's 3-sector visual: materials, financials, industrials
    three = ["XLB", "XLF", "XLI"]
    v3 = np.array([np.exp(log_volume(sec[three].to_numpy()[i - W:i], True)) for i in range(W, len(values))])
    v3 = pd.Series(v3, index=idx)
    null3 = np.array([np.exp(log_volume(rng.standard_normal((W, 3)), True)) for _ in range(4000)])
    facts["extra"] = {
        "year_min": year_min,
        "2010": year_min["2010"], "2011": year_min["2011"], "2015": year_min["2015"], "2022": year_min["2022"],
        "2025": year_min["2025"],
        "2018_jan15_jun30_min": [str(vol.loc["2018-01-15":"2018-06-30"].idxmin().date()),
                                 float(vol.loc["2018-01-15":"2018-06-30"].min())],
        "threshold_on_2018_02_09": float(np.exp(low_thr.loc[:"2018-02-09"].iloc[-1])),
        "volume_2018_01_26": at("2018-01-26"),
        "first_hyst_entry_2018": str(entries_2018[entries_2018 > 0].index[0].date()) if (entries_2018 > 0).any() else None,
        "regime_share_pct_all_days": float(hyst_s.mean() * 100),
        "regime_share_pct_after_warmup": float(hyst_s.iloc[252:].mean() * 100),
        "volume_range": [float(vol.min()), float(vol.max())],
        "orders_of_magnitude": float(np.log10(vol.max() / vol.min())),
        "volume_identity_window": v_identity,
        "volume_identity_window_pctile": float((vol < v_identity).mean() * 100),
        "market_mode_share_pct": {d: mode_share(d) for d in ("2020-03-19", "2011-09-23", "2009-01-23")},
        "effective_bets": {d: mode_bets(d) for d in ("2020-03-19", "2011-09-23", "2009-01-23")},
        "market_mode_share_pct_identity_window": float(eig.max() / eig.sum() * 100),
        "corr_avg_rho_vs_log_volume": float(np.corrcoef(avg_rho, series["log_volume_corr"])[0, 1]),
        "ad_example": {"all_0.2": ad_case(0.2), "rho_ad_0.99": ad_case(0.99)},
        "three_sector_XLB_XLF_XLI": {"min": [str(v3.idxmin().date()), float(v3.min())],
                                     "2020_03_19": float(v3.loc[:"2020-03-19"].iloc[-1]),
                                     "median": float(v3.median()), "today": float(v3.iloc[-1]),
                                     "null_median": float(np.median(null3)),
                                     "null_p5_p95": [float(np.percentile(null3, 5)), float(np.percentile(null3, 95))]},
        "tri_120_eigenvalues": [float(x) for x in np.linalg.eigvalsh(tri)],
        "last_close_spy": float(prices["SPY"].iloc[-1]), "last_date": str(prices.index[-1].date()),
    }
    series["volume_3_sector"] = v3
    return series, facts


def draw(s: pd.DataFrame, facts: dict, path: Path) -> None:
    """The doc's chart (reference palette, light mode, validated colours)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    BLUE, ORANGE, BAND = "#2a78d6", "#eb6834", "#ecebe7"
    null = facts["null_volume_corr"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#d3d2cd",
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, (top, ax) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True, gridspec_kw={"height_ratios": [1, 2.3]},
                                  facecolor=SURFACE)
    for a in (top, ax):
        a.set_facecolor(SURFACE)
        a.grid(True, axis="y", color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.tick_params(length=0)

    top.plot(s.index, s["spy"], color=ORANGE, linewidth=1.4, solid_capstyle="round")
    top.set_yscale("log")
    top.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    top.yaxis.set_minor_formatter(NullFormatter())
    top.set_title("S&P 500 (SPY), log scale", loc="left", color=INK2, fontsize=9)

    vol = s["volume_corr"]
    ax.axhspan(null["p5"], null["p95"], color=BAND, zorder=0)
    ax.text(s.index[40], null["p95"] * 1.08, "where 9 independent sectors would sit (Monte Carlo, 60-day windows)",
            color=INK2, fontsize=8, va="bottom")
    regime = s["geom_regime_hyst"].astype(bool)
    ax.fill_between(s.index, 1e-6, 1.2, where=regime, color=BLUE, alpha=0.10, linewidth=0, step="mid", zorder=0)
    ax.plot(s.index, vol, color=BLUE, linewidth=1.4, solid_capstyle="round", zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1.2)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("volume  ||v1 ∧ … ∧ v9||  (log scale)")

    def note(date: str, value: float, text: str, dx: int, dy: int) -> None:
        ax.annotate(text, (pd.Timestamp(date), value), xytext=(dx, dy), textcoords="offset points", color=INK,
                    fontsize=8, arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.7), zorder=5)

    d08, v08 = facts["min_2008"]
    d20, v20 = facts["min_2020"]
    d11, v11 = facts["extra"]["2011"]
    note(d08, v08, f"2008-09 crisis low {v08:.4f}", 16, -14)
    note(d11, v11, f"Sep 2011 trough {v11:.1e}\nUS downgrade, euro crisis", 26, 4)
    note(d20, v20, f"Mar 2020 trough {v20:.1e}", 22, 2)
    note("2018-02-09", facts["volume_2018_02_09"],
         "Feb 2018 volatility spike: volume fell 22x\nbut stayed above the collapse line (not flagged)", -250, 52)
    today = s.index[-1]
    ax.scatter([today], [vol.iloc[-1]], s=36, color=BLUE, edgecolor=SURFACE, linewidth=1.6, zorder=6)
    note(str(today.date()), vol.iloc[-1], f"today {vol.iloc[-1]:.2f}\n({facts['volume_pctile_today_full']:.1f}th pct)",
         -110, 4)
    share = facts["extra"]["regime_share_pct_after_warmup"]
    last = pd.Timestamp(facts["extra"]["last_date"])
    fig.text(0.07, 0.012, "Shaded: collapse regime, point-in-time (enters below the trailing 15th percentile, leaves "
             f"above the 30th): {share:.0f}% of days after the first-year warm-up.\n"
             f"Data: Yahoo Finance, 2005 to the {last.day} {last:%b %Y} close.", color=INK2, fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Sector market volume: the wedge product of the 9 SPDR sector return vectors", x=0.07, ha="left",
                 color=INK, fontsize=12, fontweight="bold")
    fig.text(0.07, 0.905, "60-day windows of daily log returns, each sector demeaned and scaled to length 1.\n"
             "Lower = the sectors move as one; 1 = exactly uncorrelated (orthogonal) windows.", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.89))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--end", default="2026-10-23", help="exclusive end date (default: the doc's data, to the 22 Oct 2026 close)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="folder for the facts JSON and the chart")
    parser.add_argument("--series-csv", type=Path, help="also save the daily series (about 0.7 MB)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    series, facts = measure(args.end)
    (args.out / "sector_wedge_volume_facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    draw(series, facts, args.out / "sector_wedge_volume.png")
    if args.series_csv:
        series.to_csv(args.series_csv)
    ident = facts["identities_today"]
    print(f"{facts['asof']}: volume {facts['volume_today']:.4f} ({facts['volume_pctile_today_full']:.1f}th pct), "
          f"det R {ident['det_R']:.11f}, Lagrange {ident['lagrange_gram_det']:.10e} = {ident['lagrange_bivector_sum']:.10e}")
    print(f"wrote {args.out / 'sector_wedge_volume_facts.json'} and {args.out / 'sector_wedge_volume.png'}")


if __name__ == "__main__":
    main()
