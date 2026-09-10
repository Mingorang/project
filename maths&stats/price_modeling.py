import random as rand
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import mplfinance as mpf
import os

#Very cool, should learn R for data analysis to do something with this data

#Either build an AI bot that has access to the .csv file for the precise numerical data of OHLC
    #and the .png for visual needs such as ratio of bullish/bearish candles in a timeframe and other measures to make an index of market conditions.
#Add a for loop and a much larger csv file so the AI bot/script has access to data over a larger timeframe
#If necessary change conditions as it is currently random and unaffected by market manipulation of large firms.
# Simulate OHLC stock data for a candle chart like the example image.
master_dir = os.path.dirname(os.path.abspath(__file__))
folder_name = os.path.join(master_dir, "processor")
os.makedirs(folder_name, exist_ok=True)
filename = os.path.join(folder_name, "generated_data.csv")
chart_file = os.path.join(folder_name, "stock_chart.png")
seed=rand.randint(-1000000,1000000)
rand.seed(seed)
start_day = pd.Timestamp("2026-09-14")
all_timestamps = []
for offset in range(5):
    day = start_day + pd.Timedelta(days=offset)
    for hour in range(9, 18):
        all_timestamps.append(pd.Timestamp(day.date()) + pd.Timedelta(hours=hour))

rows = []
max_1
max_1
max_1
max_1
prev_close = 100.0
#Looking at gpbusd the ratio between (O-C)/(H-L) ranges from (2-3)% to 100%, so close price should reflect this
#try making % change over 5 days between (5-23)% in both ways
for idx, ts in enumerate(all_timestamps):
    k = rand.randint(1,70)/4
    open_price = prev_close
    close_price = open_price + rand.uniform(-k, k)
    high = max(open_price, close_price) + rand.gauss(mu=0.3, sigma=0.1*rand.randint(10,23) )
    low = min(open_price, close_price) -  rand.gauss(mu=0.3, sigma=0.1*rand.randint(10,23) )
        
        
        
    
    row = {
        "Open": round(open_price, 2),
        "High": round(high, 2),
        "Low": round(low, 2),
        "Close": round(close_price, 2),
    }
    rows.append((ts, row))
    prev_close = close_price

    print(f"{ts}: O={row['Open']} H={row['High']} L={row['Low']} C={row['Close']}")

daily = pd.DataFrame(
    [row for _, row in rows],
    index=pd.DatetimeIndex([ts for ts, _ in rows], name="Date")
)

daily.to_csv(filename)

print("\nCandlestick table:")
print(daily.head(12).to_string())

mpf.plot(
    daily,
    type="candle",
    style="charles",
    volume=False,
    figsize=(12, 6),
    ylabel="Price",
    title="Stock Price (9am-5pm, 5 Days)",
    savefig=chart_file,
)

print(f"\nSaved chart to: {chart_file}")

df = pd.read_csv(filename)


def desmos_points(column_name):
    points = ", ".join(
        f"({index}, {value:.2f})"
        for index, value in enumerate(df[column_name])
    )
    return f"[{points}]"

print(f"Seed is {seed}")
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