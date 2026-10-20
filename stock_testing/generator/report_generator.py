"""
report_generator.py — Standalone Executive HTML Summary Report Generator
Implements Roadmap Idea 86: Long summary report with keyword summary, KPI cards,
calibration history, and multi-model comparison.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any
import pandas as pd


def generate_html_report(
    analysis_dir: str | Path,
    instrument: str = "GBP/USD",
    timeframe: str = "5d",
    asset_type: str = "forex",
    equilibrium_info: dict[str, Any] | None = None,
    ml_info: dict[str, Any] | None = None,
    pruned_runs: list[int] | None = None,
) -> Path:
    """Generate a self-contained dark-themed HTML executive summary report."""
    analysis_dir = Path(analysis_dir)
    reports_dir = analysis_dir.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Read accuracy CSV if available
    acc_csv = analysis_dir / "rewind_accuracy.csv"
    acc_table_html = "<p class='dim'>No rewind accuracy data available.</p>"
    overall_acc = 0.0
    dir_match = 0.0
    net_match = 0.0
    if acc_csv.exists():
        try:
            df_acc = pd.read_csv(acc_csv)
            scored = df_acc[df_acc["status"] == "scored"]
            if not scored.empty:
                overall_acc = float(scored["index"].mean())
                dir_match = float(scored["score_direction"].mean())
                dir_match = max(0.0, min(100.0, (dir_match + 1.0) / 2.0 * 100.0))
                net_match = float(scored["score_net_move"].mean())
                net_match = max(0.0, min(100.0, (net_match + 1.0) / 2.0 * 100.0))
                acc_table_html = df_acc.head(15).to_html(classes="report-table", index=False)
        except Exception:
            pass

    # Read Equilibrium CSV if available
    eq_csv = analysis_dir / "vol_equilibrium.csv"
    eq_table_html = "<p class='dim'>No equilibrium calibration log found.</p>"
    if eq_csv.exists():
        try:
            df_eq = pd.read_csv(eq_csv)
            eq_table_html = df_eq.tail(10).to_html(classes="report-table", index=False)
        except Exception:
            pass

    # Read ML Model Comparison CSV if available
    ml_comp_csv = analysis_dir / "model_comparison.csv"
    ml_table_html = "<p class='dim'>No model comparison executed yet.</p>"
    if ml_comp_csv.exists():
        try:
            df_ml = pd.read_csv(ml_comp_csv)
            ml_table_html = df_ml.to_html(classes="report-table", index=False)
        except Exception:
            pass

    # Keyword status assessment
    status_exact = "PASS" if overall_acc >= 0.35 else "CONVERGING" if overall_acc >= 0.15 else "CALIBRATING"
    status_price = "PASS" if net_match >= 70.0 else "SUB-OPTIMAL"
    status_bars = "PASS" if dir_match >= 55.0 else "SUB-OPTIMAL"
    status_eq = "CONVERGED" if (equilibrium_info and equilibrium_info.get("converged")) else "ACTIVE"

    pruned_count = len(pruned_runs) if pruned_runs else 0
    pruned_msg = f"{pruned_count} Outlier Runs Pruned" if pruned_count > 0 else "0 Outlier Runs (Data Within Tolerance)"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Quantitative Price Modeling & ML Executive Summary — {instrument}</title>
<style>
    body {{
        background-color: #0b0e14;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        line-height: 1.6;
        margin: 0;
        padding: 30px;
    }}
    .container {{
        max-width: 1200px;
        margin: 0 auto;
    }}
    h1, h2, h3 {{
        color: #58a6ff;
        font-weight: 600;
        border-bottom: 1px solid #21262d;
        padding-bottom: 8px;
    }}
    .subtitle {{
        color: #8b949e;
        font-size: 14px;
        margin-top: -10px;
        margin-bottom: 25px;
    }}
    .badge {{
        display: inline-block;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 700;
        border-radius: 6px;
        margin-right: 8px;
    }}
    .badge-pass {{ background-color: #238636; color: #ffffff; }}
    .badge-warn {{ background-color: #9e6a03; color: #ffffff; }}
    .badge-info {{ background-color: #1f6feb; color: #ffffff; }}
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-bottom: 30px;
    }}
    .kpi-card {{
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }}
    .kpi-title {{
        font-size: 13px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .kpi-value {{
        font-size: 28px;
        font-weight: 700;
        color: #3fb950;
        margin: 8px 0;
    }}
    .kpi-value.blue {{ color: #58a6ff; }}
    .kpi-value.yellow {{ color: #d29922; }}
    .keyword-summary {{
        background-color: #161b22;
        border-left: 4px solid #58a6ff;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 30px;
    }}
    .report-table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
        margin-bottom: 30px;
        font-size: 13px;
    }}
    .report-table th, .report-table td {{
        border: 1px solid #30363d;
        padding: 8px 12px;
        text-align: left;
    }}
    .report-table th {{
        background-color: #21262d;
        color: #c9d1d9;
    }}
    .report-table tr:nth-child(even) {{
        background-color: #161b22;
    }}
    .dim {{ color: #8b949e; font-style: italic; }}
    footer {{
        margin-top: 50px;
        border-top: 1px solid #21262d;
        padding-top: 15px;
        color: #8b949e;
        font-size: 12px;
        text-align: center;
    }}
</style>
</head>
<body>
<div class="container">
    <h1>Quantitative Simulation & Machine Learning Executive Report</h1>
    <div class="subtitle">Generated on {timestamp_str} | Instrument: <strong>{instrument}</strong> | Asset Type: <strong>{asset_type}</strong> | Timeframe: <strong>{timeframe}</strong></div>

    <div class="keyword-summary">
        <h3>[EXECUTIVE KEYWORD SUMMARY]</h3>
        <p>
            <span class="badge badge-info">[EXACT] {status_exact}</span>
            <span class="badge badge-pass">[PRICE MATCH] {status_price}</span>
            <span class="badge badge-pass">[BARS MATCH] {status_bars}</span>
            <span class="badge badge-info">[EQUILIBRIUM] {status_eq}</span>
            <span class="badge badge-warn">[PRUNING] {pruned_msg}</span>
        </p>
        <p>
            <strong>Core Learning State:</strong> The simulation engine continuously compares synthetic candlestick chains
            against point-in-time real market returns fetched from Yahoo Finance. Volatility, wick ratio, and jump-diffusion
            parameters are dynamically updated per run to find an equilibrium surface minimising path RMSE and maximizing
            directional sign agreement.
        </p>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Accuracy Index</div>
            <div class="kpi-value">{overall_acc:+.3f}</div>
            <small style="color: #8b949e;">Scale: -1.0 to +1.0</small>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Direction Match</div>
            <div class="kpi-value blue">{dir_match:.1f}%</div>
            <small style="color: #8b949e;">[BARS] Criterion Target &ge;50%</small>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Net Price Match</div>
            <div class="kpi-value blue">{net_match:.1f}%</div>
            <small style="color: #8b949e;">[PRICE] Net Move Accuracy</small>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Outlier Runs Pruned</div>
            <div class="kpi-value yellow">{pruned_count}</div>
            <small style="color: #8b949e;">Filtered via config.toml</small>
        </div>
    </div>

    <h2>1. Self-Correcting Volatility Equilibrium Surface (Roadmap Ideas 1–4, 60)</h2>
    <p>Historical progression of the self-correction loop recording optimal volatility parameters discovered:</p>
    {eq_table_html}

    <h2>2. Rewind & Replay Accuracy Breakdown vs Real Yahoo Data</h2>
    <p>Per-cycle comparison breakdown against real market price movements:</p>
    {acc_table_html}

    <h2>3. Machine Learning Model Benchmark & Comparison (Roadmap Ideas 94–96)</h2>
    <p>Multi-model evaluation utilizing Purged K-Fold Cross Validation with non-linear feature architectures (Random Forest, Logistic Regression, MLP Neural Net with tanh, and HistGradientBoosting):</p>
    {ml_table_html}

    <footer>
        Self-Correcting Quantitative Trading System | QMUL Financial Mathematics Pipeline | Automated Audit Artifact
    </footer>
</div>
</body>
</html>
"""

    out_file1 = reports_dir / "executive_summary.html"
    out_file2 = analysis_dir / "executive_summary.html"
    out_file1.write_text(html_content, encoding="utf-8")
    out_file2.write_text(html_content, encoding="utf-8")
    print(f"[Report] Generated Executive HTML Report -> {out_file1}")
    return out_file1
