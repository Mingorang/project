import random as rand
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplfinance as mpf
import os
from pathlib import Path
from market_analysis import analyze_looped_results
import price_open

#Very cool, should learn R for data analysis to do something with this data
# Added input to choose for real-time price 
#Curb the random model for high prices, as an option for forex, relates to the manipulation comment.
# need to add time module to also add real time  and price manipulation as a random event, modelled to real market manipulation
# change this for forex online data for some starting price of 2 currencies, (EUR/USD) and calculate each one seperately, and plot the difference, so the plot will show the Euro dvided by the USD for a more realistic simulation

#Placing forex at volativity of 0.4 to 1 is reasonable over 5 days
#Oil is reasonabl at 35 over 5 days
asset_type = input(
    "Asset type [forex/futures/stock/crypto/other] (default forex): "
).strip().lower() or "forex"
ticker_name = input(
    "Instrument or pair to simulate (default GBP/USD): "
).strip() or "GBP/USD"

price_option = input(
    f"Starting price: [1] enter manually or [2] use current {ticker_name}? "
).strip()
if price_option == "2":
    if asset_type == "forex":
        default_price_symbol = ticker_name.upper().replace("/", "") + "=X"
    else:
        default_price_symbol = ticker_name
    price_symbol = input(
        f"Yahoo Finance symbol for {ticker_name} [{default_price_symbol}]: "
    ).strip() or default_price_symbol
    price_data = price_open.get_starting_price(price_symbol)
    if isinstance(price_data, dict):
        starting_price = float(price_data["current"])
    else:
        print(f"Could not get {ticker_name} price: {price_data}")
        exit()
else:
    starting_price = float(input("Starting price: "))

# Simulate OHLC stock data for a candle chart like the example image.
master_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(master_dir, "looped_results")
if not os.path.isdir(results_dir):
    print(f"Output folder does not exist: {results_dir}")
    print("Create the folder manually before generating data.")
    exit()
analysis_dir = os.path.join(master_dir, "looped_analysis")
if not os.path.isdir(analysis_dir):
    print(f"Analysis output folder does not exist: {analysis_dir}")
    print("Create the folder manually before generating data.")
    exit()


def clear_generated_files(directory, patterns):
    for pattern in patterns:
        for path in Path(directory).glob(pattern):
            if path.is_file():
                path.unlink()

rand.seed(rand.randint(-1000000,1000000))
start_day = pd.Timestamp("2026-09-14")
#Scales from 1 to 100, .05% --> 50+%, log scale necessary
question = float(input("Volativity of market: "))
vol_index = question/8
if vol_index > 100 or vol_index < 0:
    print("Choose a volatility index from 1 to 100.")
    exit()

price_mode = input(
    "Price behavior: [1] allow negative futures prices or [2] floor prices at zero? "
).strip().lower()
if price_mode in ("1", "futures"):
    allow_negative = True
elif price_mode in ("2", "floor"):
    allow_negative = False
else:
    print("Choose 1 for negative-capable futures or 2 to enforce a zero floor.")
    exit()

plot_mode = input("Create [1] one plot or [M] many plots? ").strip().lower()
if plot_mode in ("1", "one", "single"):
    run_count = 1
elif plot_mode in ("m", "many"):
    try:
        run_count = int(input("How many plots? "))
    except ValueError:
        print("Enter a whole number greater than zero.")
        exit()
    if run_count < 1:
        print("Enter a whole number greater than zero.")
        exit()
else:
    print("Choose 1 for one plot or M for many plots.")
    exit()

render_analysis_charts = True
render_source_charts = True
run_ml = False
if run_count >= 100:
    print(f"\nChoose outputs for {run_count:,} runs:")
    print("  Source PNGs: the original OHLC chart for each run.")
    print("  Analysis PNGs: the indicator and trade chart for each run.")
    print("  Per-run CSVs and combined analysis metrics are always saved.")
    print("  1) Source + analysis charts")
    print("  2) Source + analysis charts + ML (longest processing time)")
    print("  3) Analysis charts + ML (no source charts)")
    print("  4) Analysis charts only (no source charts or ML)")
    print("  5) ML only (no charts; required analysis metrics are still calculated)")
    print("  6) CSVs + metrics only (no charts or ML; fastest) [default]")
    output_choice = input("Choose 1-6 [6]: ").strip() or "6"
    output_modes = {
        "1": (True, True, False),
        "2": (True, True, True),
        "3": (False, True, True),
        "4": (False, True, False),
        "5": (False, False, True),
        "6": (False, False, False),
    }
    if output_choice not in output_modes:
        print("Choose one of the output modes 1-6.")
        exit()
    render_source_charts, render_analysis_charts, run_ml = output_modes[output_choice]

clear_generated_files(results_dir, ("data_*.csv", "image_*.png"))
clear_generated_files(analysis_dir, ("*.csv", "*.png"))

all_timestamps = []
for offset in range(5):
    day = start_day + pd.Timedelta(days=offset)
    for hour in range(9, 18):
        all_timestamps.append(pd.Timestamp(day.date()) + pd.Timedelta(hours=hour))

#This function need a lot of work for prices near 0 and high starting prices ones with small variance
show_run_details = run_count < 100


def generate_daily(verbose: bool = show_run_details) -> pd.DataFrame:
    rows = []
    prev_close = starting_price
    for ts in all_timestamps:
        open_price = prev_close
        close_price = open_price + ((6/15)*(prev_close/100)*(rand.uniform(-vol_index,vol_index)))
        if not allow_negative:
            close_price = max(0.0, close_price)

        wick_range = ((3.5/15) * (prev_close / 100) * vol_index)
        # Signed normal samples could invert the wick and fail analyze_looped_results' OHLC ordering check.
        high = max(open_price, close_price) + abs(rand.normalvariate(0, wick_range/2))
        low = min(open_price, close_price) - abs(rand.normalvariate(0, wick_range/2))
        if not allow_negative:
            low = max(0.0, low)

        row = {
            "Open": round(open_price, 9),
            "High": round(high, 9),
            "Low": round(low, 9),
            "Close": round(close_price, 9),
        }
        rows.append((ts, row))
        prev_close = close_price

        if verbose:
            print(f"{ts}: O={row['Open']} H={row['High']} L={row['Low']} C={row['Close']}")

    daily = pd.DataFrame(
        [row for _, row in rows],
        index=pd.DatetimeIndex([ts for ts, _ in rows], name="Date")
    )
    if verbose:
        print("\nCandlestick table:")
        print(daily.head(12).to_string())
    return daily

black_background_style = mpf.make_mpf_style(
    base_mpf_style="charles",
    figcolor="black",
    facecolor="black",
    y_on_right=True,
    rc={
        "axes.labelcolor": "white",
        "axes.titlecolor": "white",
        "text.color": "white",
        "xtick.color": "white",
        "ytick.color": "white",
    },
)

daily = pd.DataFrame()
progress_interval = max(1, run_count // 100)
for run_number in range(1, run_count + 1):
    if show_run_details:
        print(f"\nGenerating plot {run_number} of {run_count}...")
    elif run_number == 1 or run_number % progress_interval == 0 or run_number == run_count:
        print(f"Generating run {run_number:,} of {run_count:,}...")
    daily = generate_daily(verbose=show_run_details)
    csv_file = os.path.join(results_dir, f"data_{run_number}.csv")
    image_file = os.path.join(results_dir, f"image_{run_number}.png")

    lowest_low = daily["Low"].min()
    open_01 = daily["Open"].iloc[0]
    close_01 = daily["Close"].iloc[-1]
    highest_high = daily["High"].max()
    if lowest_low <= 0:
        low_to_high_label = "Low to high: N/A"
    else:
        percent_change = abs((highest_high - lowest_low) / lowest_low * 100)
        low_to_high_label = f"Low to high: {percent_change:.2f}%"
    market_change = (close_01 - open_01) / open_01 * 100

    if render_source_charts:
        figure, axes = mpf.plot(
            daily,
            type="candle",
            style=black_background_style,
            volume=False,
            figsize=(12, 6),
            ylabel="Price",
            title=f"Stock Price of {ticker_name} (9am-5pm, 5 Days)",
            returnfig=True,
        )

        axes[0].legend(
            handles=[
                Line2D([], [], linestyle="None", color="none"),
                Line2D([], [], linestyle="None", color="none"),
            ],
            labels=[
                low_to_high_label,
                f"Monday open to Friday close: {market_change:.2f}%",
            ],
            loc="upper left",
            handlelength=0,
            handletextpad=0,
        )
        figure.savefig(image_file, facecolor=figure.get_facecolor())
        plt.close(figure)
    daily.to_csv(csv_file)

    if show_run_details:
        print(f"\nSaved chart to: {image_file}")
        print(f"Saved data to: {csv_file}")

analysis_result = analyze_looped_results(
    results_dir,
    analysis_dir,
    render_charts=render_analysis_charts,
    render_source_charts=render_source_charts,
)

# Attach the selected instrument to every analyzed run so ML results can be
# identified and kept separate when the instrument changes.
summary_path = Path(analysis_dir) / "analysis_summary.csv"
analysis_summary = pd.read_csv(summary_path)
analysis_summary.insert(1, "instrument", ticker_name)
analysis_summary.insert(2, "asset_type", asset_type)
analysis_summary.to_csv(summary_path, index=False)

print(
    f"\nAnalysis complete: {analysis_result['runs']} pairs, "
    f"{analysis_result['bars']} bars, {analysis_result['trades']} crossover trades."
)
if run_count < 100:
    ml_choice = input("\nRun ML model on these results? (y/n): ").strip().lower()
    run_ml = ml_choice in ("y", "yes")

if run_ml:
    from ml_model import run_ml_pipeline
    run_ml_pipeline(
        analysis_dir,
        instrument=ticker_name,
        asset_type=asset_type,
    )

df = daily


#def desmos_points(column_name):
#    points = ", ".join(
#        f"({index}, {value:.2f})"
#        for index, value in enumerate(df[column_name])
#    )
#    return f"[{points}]"
#
#
#data_given = input("What data do you want? ").strip().lower()
#print()
#if data_given == "open":
#    print(desmos_points("Open"))
#elif data_given == "high":
#    print(desmos_points("High"))
#elif data_given == "low":
#    print(desmos_points("Low"))
#elif data_given == "close":
#    print(desmos_points("Close"))
#else:
#    print(f"Unknown data type: {data_given}. Choose open, high, low, or close.")
