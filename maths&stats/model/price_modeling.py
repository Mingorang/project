import random as rand
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplfinance as mpf
import os
from market_analysis import analyze_looped_results

#Very cool, should learn R for data analysis to do something with this data

#Either build an AI bot that has access to the .csv file for the precise numerical data of OHLC
    #and the .png for visual needs such as ratio of bullish/bearish candles in a timeframe and other measures to make an index of market conditions.
#Add a for loop and a much larger csv file so the AI bot/script has access to data over a larger timeframe
#If necessary change conditions as it is currently random and unaffected by market manipulation of large firms.


# Simulate OHLC stock data for a candle chart like the example image.
master_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(master_dir, "looped_results")
if not os.path.isdir(results_dir):
    print(f"Output folder does not exist: {results_dir}")
    print("Create the folder manually before generating data.")
    exit()
analysis_dir = os.path.join(master_dir, "looped_results_analysis_300")
if not os.path.isdir(analysis_dir):
    print(f"Analysis output folder does not exist: {analysis_dir}")
    print("Create the folder manually before generating data.")
    exit()

rand.seed(rand.randint(-1000000,1000000))
start_day = pd.Timestamp("2026-09-14")
#Scales from 1 to 100, .05% --> 50+%, log scale necessary
question = int(input("Volativity of market: "))
vol_index = question
if vol_index > 100 or vol_index < 1:
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

all_timestamps = []
for offset in range(5):
    day = start_day + pd.Timedelta(days=offset)
    for hour in range(9, 18):
        all_timestamps.append(pd.Timestamp(day.date()) + pd.Timedelta(hours=hour))


def generate_daily():
    rows = []
    prev_close = 100
    for ts in all_timestamps:
        open_price = prev_close
        close_price = open_price + (6/15)*(rand.uniform(-vol_index,vol_index))
        if not allow_negative:
            close_price = max(0.0, close_price)

        wick_range = max(vol_index / 40, 0.025)
        high = max(open_price, close_price) + rand.uniform(0, wick_range)
        low = min(open_price, close_price) - rand.uniform(0, wick_range)
        if not allow_negative:
            low = max(0.0, low)

        row = {
            "Open": round(open_price, 4),
            "High": round(high, 4),
            "Low": round(low, 4),
            "Close": round(close_price, 4),
        }
        rows.append((ts, row))
        prev_close = close_price

        print(f"{ts}: O={row['Open']} H={row['High']} L={row['Low']} C={row['Close']}")

    daily = pd.DataFrame(
        [row for _, row in rows],
        index=pd.DatetimeIndex([ts for ts, _ in rows], name="Date")
    )
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
for run_number in range(1, run_count + 1):
    print(f"\nGenerating plot {run_number} of {run_count}...")
    daily = generate_daily()
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

    figure, axes = mpf.plot(
        daily,
        type="candle",
        style=black_background_style,
        volume=False,
        figsize=(12, 6),
        ylabel="Price",
        title="Stock Price (9am-5pm, 5 Days)",
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

    print(f"\nSaved chart to: {image_file}")
    print(f"Saved data to: {csv_file}")

analysis_result = analyze_looped_results(results_dir, analysis_dir)
print(
    f"\nAnalysis complete: {analysis_result['runs']} pairs, "
    f"{analysis_result['bars']} bars, {analysis_result['trades']} crossover trades."
)

df = daily


def desmos_points(column_name):
    points = ", ".join(
        f"({index}, {value:.2f})"
        for index, value in enumerate(df[column_name])
    )
    return f"[{points}]"


data_given = input("What data do you want? ").strip().lower()
print()
if data_given == "open":
    print(desmos_points("Open"))
elif data_given == "high":
    print(desmos_points("High"))
elif data_given == "low":
    print(desmos_points("Low"))
elif data_given == "close":
    print(desmos_points("Close"))
else:
    print(f"Unknown data type: {data_given}. Choose open, high, low, or close.")
